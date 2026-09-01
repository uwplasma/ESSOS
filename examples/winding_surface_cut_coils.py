"""Cut filament coils from the current potential saved by winding_surface_opt_2.py."""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
from contourpy import LineType, contour_generator
from netCDF4 import Dataset

from essos.surfaces import SurfaceRZFourier


MU0 = 4 * np.pi * 1e-7
COILS_PER_HALF_PERIOD = 10
RESOLUTION = 96
RESULT_FILE = "winding_surface_opt.npz"
OUTPUT_FILE = "winding_surface_coils"


def evaluate_surface(surface, theta, phi):
    angle = (np.asarray(surface.xm)[:, None] * np.ravel(theta)
             - np.asarray(surface.xn)[:, None] * np.ravel(phi))
    radius = np.asarray(surface.rc) @ np.cos(angle)
    z = np.asarray(surface.zs) @ np.sin(angle)
    xyz = np.column_stack((radius * np.cos(np.ravel(phi)),
                           radius * np.sin(np.ravel(phi)), z))
    return xyz.reshape(theta.shape + (3,))


def cut_coils(surface, potential):
    """Take equally spaced contours over one field period and replicate them."""
    theta0 = np.linspace(0, 2 * np.pi, potential.shape[0], endpoint=False)
    phi = np.linspace(0, 2 * np.pi / surface.nfp,
                      potential.shape[1], endpoint=False)
    levels = (np.arange(2 * COILS_PER_HALF_PERIOD) + 0.5) / (
        2 * COILS_PER_HALF_PERIOD)
    period = 2 * np.pi / surface.nfp
    phi3 = np.concatenate((phi - period, phi, phi + period))
    for shift in range(potential.shape[0]):
        theta = theta0[(-shift) % len(theta0)] + theta0
        shifted = np.roll(potential, shift, axis=0)
        potential3 = np.concatenate((shifted - 1, shifted, shifted + 1), axis=1)
        generator = contour_generator(x=phi3, y=theta, z=potential3,
                                      line_type=LineType.Separate)
        contours = [generator.lines(float(level)) for level in levels]
        if all(len(contour) == 1 for contour in contours):
            curves = []
            for contour in contours:
                for field_period in range(surface.nfp):
                    phi_curve = contour[0][:, 0] + field_period * period
                    theta_curve = contour[0][:, 1]
                    curve = evaluate_surface(surface, theta_curve, phi_curve)
                    curves.append(np.vstack((curve, curve[0])))
            return curves, levels
    raise RuntimeError("Could not find closed current-potential contours")


def filament_field(points, curves, current):
    """Magnetic field from straight segments joining the contour points."""
    field = np.zeros_like(points)
    for start in range(0, len(points), 512):
        evaluation_points = points[start:start + 512]
        for curve in curves:
            r1 = evaluation_points[:, None, :] - curve[None, :-1, :]
            r2 = evaluation_points[:, None, :] - curve[None, 1:, :]
            norm1 = np.linalg.norm(r1, axis=2)
            norm2 = np.linalg.norm(r2, axis=2)
            denominator = norm1 * norm2 + np.sum(r1 * r2, axis=2)
            field[start:start + 512] += MU0 * current / (4 * np.pi) * np.sum(
                np.cross(r1, r2, axis=2) / denominator[:, :, None]
                * (1 / norm1 + 1 / norm2)[:, :, None], axis=1)
    return field


def vmec_mod_b(wout, theta, phi):
    with Dataset(wout) as dataset:
        bmnc = 1.5 * dataset.variables["bmnc"][-1] - 0.5 * dataset.variables["bmnc"][-2]
        bmns = (1.5 * dataset.variables["bmns"][-1]
                - 0.5 * dataset.variables["bmns"][-2]
                if "bmns" in dataset.variables else np.zeros_like(bmnc))
        xm = dataset.variables["xm_nyq"][:]
        xn = dataset.variables["xn_nyq"][:]
    angle = xm[:, None] * np.ravel(theta) - xn[:, None] * np.ravel(phi)
    return np.sum(bmnc[:, None] * np.cos(angle)
                  + bmns[:, None] * np.sin(angle), axis=0)


result = np.load(RESULT_FILE)
wout = str(result["vmec_input"])
with Dataset(wout) as dataset:
    nfp = int(dataset.variables["nfp"][0])
plasma = SurfaceRZFourier.from_wout_file(
    wout, s=1, ntheta=RESOLUTION, nphi=RESOLUTION * nfp,
    close=False, range_torus="full torus")
winding = SurfaceRZFourier.from_wout_file(
    wout, s=1, ntheta=RESOLUTION, nphi=RESOLUTION * nfp,
    close=False, range_torus="full torus")
winding.dofs = result["dofs"]

theta, phi = np.meshgrid(
    np.linspace(0, 2 * np.pi, RESOLUTION, endpoint=False),
    np.linspace(0, 2 * np.pi / nfp, RESOLUTION, endpoint=False), indexing="ij")
angle = result["potential_xm"][:, None] * theta.ravel() \
        - result["potential_xn"][:, None] * phi.ravel()
net_current = float(result["net_poloidal_current"])
potential = (result["coefficients"] @ np.sin(angle)).reshape(theta.shape)
potential += net_current * phi / (2 * np.pi)
potential = potential / net_current * nfp
curves, levels = cut_coils(winding, potential)
current = net_current / len(curves)

points = np.asarray(plasma.gamma).reshape(-1, 3)
normals = np.asarray(plasma.unitnormal).reshape(-1, 3)
field = filament_field(points, curves, current)
bnormal = np.sum(field * normals, axis=1)
mod_b = vmec_mod_b(wout, plasma.theta2d, plasma.phi2d)
area_weights = (np.asarray(plasma.area_element).ravel()
                * (2 * np.pi / plasma.ntheta) * (2 * np.pi / plasma.nphi))
print(f"coils: {len(curves)}, current per coil: {current:.6e} A")
print(f"filament f_B: {np.sum(area_weights * bnormal ** 2):.6e} T^2 m^2")
print(f"max |B.n|/B: {np.max(np.abs(bnormal) / np.abs(mod_b)):.6e}")
np.savez(f"{OUTPUT_FILE}.npz", current_per_coil_A=current,
         **{f"coil_{i}": curve for i, curve in enumerate(curves)})

figure = plt.figure(figsize=(12, 5), constrained_layout=True)
axis = figure.add_subplot(121)
colors = axis.contourf(phi[0] * nfp, theta[:, 0], potential, 40, cmap="viridis")
axis.contour(phi[0] * nfp, theta[:, 0], potential,
             levels=levels, colors="black", linewidths=0.8)
axis.set(xlabel=r"$N_{fp}\phi$", ylabel=r"$\theta$",
         title="Current potential; black contours are coils")
figure.colorbar(colors, ax=axis, label=r"$N_{fp}\Phi/I_{pol}$")

axis = figure.add_subplot(122, projection="3d")
theta3d, phi3d = np.meshgrid(
    np.linspace(0, 2 * np.pi, RESOLUTION + 1),
    np.linspace(0, 2 * np.pi, RESOLUTION * nfp + 1), indexing="ij")
xyz = evaluate_surface(winding, theta3d, phi3d)
potential3d = np.pad(np.tile(potential, (1, nfp)), ((0, 1), (0, 1)), mode="wrap")
axis.plot_surface(*np.moveaxis(xyz, 2, 0),
                  facecolors=plt.cm.viridis(np.mod(potential3d * len(levels), 1)),
                  shade=False, alpha=0.45, linewidth=0)
for curve in curves:
    axis.plot(*curve.T, color="black", linewidth=0.8)
axis.set_box_aspect(np.ptp(xyz.reshape(-1, 3), axis=0))
axis.set_axis_off()
axis.set_title("Filament coils on the winding surface")
figure.savefig(f"{OUTPUT_FILE}.png", dpi=200)
plt.close(figure)
print(f"saved {OUTPUT_FILE}.npz and {OUTPUT_FILE}.png")
