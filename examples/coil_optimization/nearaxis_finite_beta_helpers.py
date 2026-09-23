"""Diagnostics, VMEX free-boundary benchmarks and figures for
``optimize_coils_and_nearaxis_finite_beta.py``.

The physics lives in the libraries. pyQSC_JAX supplies the near-axis equilibrium,
the plasma field/gradient/Hessian and the VMEC boundary, ESSOS the coil field,
and VMEX the free-boundary equilibrium. This module only connects them.

All field-derivative arrays are sample-first and output-first, as returned by
``jax.jacfwd``: ``B[n, i]``, ``G[n, i, j] = dB_i/dx_j`` and
``H[n, i, j, k] = d2B_i/dx_j dx_k``.
"""
import dataclasses
import json
import traceback
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import vmex as vj
from jax import jacfwd, jit, vmap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from vmex.core.freeboundary import _external_field_from_input
from vmex.core.mgrid import read_mgrid, tabulate_cartesian_field, write_mgrid
from vmex.core.plotting import surface_rz

from essos.coils import Coils
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from pyqsc_jax.plasma import plasma_hessian_on_axis
from pyqsc_jax.vmec import to_vmec, uniform_cylindrical_surface

INK, MUTED = "#1b1f24", "#6b7280"
COLORS = {"near": INK, "direct": "#c2410c", "mgrid": "#0f6fae", "initial": "#9aa3ad", "optimized": "#0f6fae"}
STYLE = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
         "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
         "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#d7dbe0", "grid.linewidth": 0.6,
         "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "mathtext.fontset": "cm"}


# ----------------------------- fields and targets -----------------------------
def coil_targets(solution, radius):
    """External (coil) field, gradient and Hessian on the axis: total minus plasma."""
    data = plasma_hessian_on_axis(solution, formal_radius=radius)
    return dict(points=np.asarray(solution.geometry.position_cartesian),
                B=np.asarray(data.field.external_field), G=np.asarray(data.field.external_gradient),
                H=np.asarray(data.external_hessian), B_plasma=np.asarray(data.field.field.field),
                G_plasma=np.asarray(data.field.gradient), H_plasma=np.asarray(data.hessian),
                B_total=np.asarray(solution.B_axis), G_total=np.asarray(solution.grad_B_axis),
                gradient_asymmetry=float(data.field.maximum_external_asymmetry),
                hessian_asymmetry=float(data.maximum_external_symmetry_error))


def axis_weights(solution):
    """Arclength quadrature weights on the uniform-phi axis grid, normalized to one."""
    weights = np.asarray(solution.geometry.d_l_d_phi)
    return weights / weights.sum()


def refine_coils(field, n_segments):
    """The same Fourier coils with a finer Biot-Savart quadrature."""
    curves = field.coils.curves.copy()
    curves.n_segments = n_segments
    return BiotSavart(Coils(curves=curves, currents=field.coils.dofs_currents_raw,
                            currents_scale=field.coils.currents_scale))


def evaluate_field(field, xyz, chunk=512):
    """Direct Biot-Savart field at points of any leading shape, in bounded memory."""
    points = np.asarray(xyz).reshape(-1, 3)
    evaluate = jit(vmap(field.B))
    blocks = []
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        padded = np.pad(block, ((0, chunk - len(block)), (0, 0)), mode="edge")
        blocks.append(np.asarray(evaluate(jnp.asarray(padded)))[:len(block)])
    return np.concatenate(blocks).reshape(np.shape(xyz))


def axis_match(solution, targets, field):
    """Arclength-weighted RMS mismatch between the coils and the external-field target.

    Every quantity is returned in SI units and, under ``*_over_B0`` keys, in the objective's
    normalization: field / B0, gradient R0 / B0 and Hessian R0^2 / B0, with B0 the on-axis
    field and R0 the axis major radius at phi = 0.
    """
    weights, points = axis_weights(solution), jnp.asarray(targets["points"])
    dB = evaluate_field(field, targets["points"]) - targets["B"]
    dG = np.asarray(jit(vmap(field.dB_by_dX))(points)) - targets["G"]
    dH = np.asarray(jit(vmap(jacfwd(jacfwd(field.B))))(points)) - targets["H"]
    rms = lambda value, axes: float(np.sqrt(np.sum(weights * np.sum(value**2, axis=axes))))
    match = dict(field_rms_T=rms(dB, 1), gradient_rms_T_per_m=rms(dG, (1, 2)),
                 hessian_rms_T_per_m2=rms(dH, (1, 2, 3)), target_hessian_rms_T_per_m2=rms(targets["H"], (1, 2, 3)),
                 plasma_hessian_rms_T_per_m2=rms(targets["H_plasma"], (1, 2, 3)),
                 plasma_field_rms_T=rms(targets["B_plasma"], 1),
                 plasma_gradient_rms_T_per_m=rms(targets["G_plasma"], (1, 2)),
                 total_gradient_rms_T_per_m=rms(targets["G_total"], (1, 2)))
    B0, R0 = float(solution.inputs.B0), float(solution.R0[0])
    relative = {"_T": ("_over_B0", 1.0), "_T_per_m": ("_R0_over_B0", R0), "_T_per_m2": ("_R0sq_over_B0", R0**2)}
    for key, value in list(match.items()):
        suffix = next(s for s in ("_T_per_m2", "_T_per_m", "_T") if key.endswith(s))
        name, scale = relative[suffix]
        match[key[:-len(suffix)] + name] = value * scale / B0
    return dict(match, B0_T=B0, R0_m=R0)


# --------------------------------- surfaces ----------------------------------
def flux_surface(solution, radius, ntheta=64):
    """Near-axis flux surface on a uniform (theta, cylindrical phi) grid of one field period."""
    R, Z, _, residual = uniform_cylindrical_surface(solution, radius, ntheta=ntheta)
    R, Z, phi = np.asarray(R), np.asarray(Z), np.asarray(solution.phi)
    if float(residual) > 1e-9:
        raise ValueError(f"Surface r={radius:.4g} m folds in the toroidal angle (residual {float(residual):.2e}).")
    nfp = int(solution.inputs.axis.nfp)
    spectral = lambda f, axis, period: np.real(np.fft.ifft(
        1j * np.fft.fftfreq(f.shape[axis], period / (2 * np.pi * f.shape[axis])).reshape(
            [-1 if k == axis else 1 for k in range(2)]) * np.fft.fft(f, axis=axis), axis=axis))
    Rt, Zt = spectral(R, 0, 2 * np.pi), spectral(Z, 0, 2 * np.pi)
    Rp, Zp = spectral(R, 1, 2 * np.pi / nfp), spectral(Z, 1, 2 * np.pi / nfp)
    c, s = np.cos(phi), np.sin(phi)
    xyz = np.stack((R * c, R * s, Z), -1)
    d_theta = np.stack((Rt * c, Rt * s, Zt), -1)
    d_phi = np.stack((Rp * c - R * s, Rp * s + R * c, Zp), -1)
    normal = np.cross(d_theta, d_phi)
    area = np.linalg.norm(normal, axis=-1)
    normal /= area[..., None]
    outward = np.sign(np.sum(normal * (xyz - np.asarray(solution.geometry.position_cartesian)), -1))
    if not (np.all(outward > 0) or np.all(outward < 0)):  # A necessary condition only, not an intersection test.
        raise ValueError(f"Surface r={radius:.4g} m has normals that do not all point away from the axis.")
    return dict(R=R, Z=Z, phi=phi, xyz=xyz, normal=normal * outward[..., None], area=area, nfp=nfp, radius=radius)


def normal_field_error(solution, targets, field, radius, ntheta=64):
    """Coil-field mismatch normal to a near-axis surface, using the target's Taylor continuation.

    The external target is continued from each axis sample to second order in the
    displacement. It is a diagnostic of the coils, not a resolved off-axis plasma field.
    The error is (B_coils - B_target) . n / |B_target|, relative to the local field strength.
    """
    surface = flux_surface(solution, radius, ntheta)
    step = surface["xyz"] - targets["points"][None]
    linear = targets["B"][None] + np.einsum("pij,tpj->tpi", targets["G"], step)
    quadratic = linear + 0.5 * np.einsum("pijk,tpj,tpk->tpi", targets["H"], step, step)
    strength = np.linalg.norm(quadratic, axis=-1)  # Local field, so the error is relative everywhere.
    error = np.sum((evaluate_field(field, surface["xyz"]) - quadratic) * surface["normal"], -1) / strength
    hessian_term = np.sum((quadratic - linear) * surface["normal"], -1) / strength
    rms = lambda value: float(np.sqrt(np.sum(surface["area"] * value**2) / np.sum(surface["area"])))
    surface["error"] = error
    return surface, dict(radius_m=float(radius), normal_error_rms=rms(error),
                         normal_error_max=float(np.max(np.abs(error))), hessian_term_rms=rms(hessian_term))


def contour_distance(first, second):
    """Symmetric RMS and maximum distance between two closed (R, Z) polylines."""
    distances = []
    for points, curve in ((first, second), (second, first)):
        edge = np.roll(curve, -1, axis=0) - curve
        delta = points[:, None, :] - curve[None, :, :]
        fraction = np.clip(np.sum(delta * edge, -1) / np.maximum(np.sum(edge * edge, -1), 1e-300), 0, 1)
        distances.append(np.sqrt(np.min(np.sum((delta - fraction[..., None] * edge)**2, -1), axis=1)))
    distances = np.concatenate(distances)
    return float(np.sqrt(np.mean(distances**2))), float(distances.max())


def trace_poincare(field, solution, radius, levels, maxtime=600.0, n_steps=12000, tolerance=1e-9):
    """Poincare sections of the coil field lines started on the near-axis flux surfaces.

    Only meaningful without plasma, where the coil field is the total field. Then it is exact,
    unlike a free-boundary solve, which is marginal with neither pressure nor current.
    Returns the sections at phi = 0 and half a field period, and the RMS distance of the
    punctures from the near-axis contour of the same flux label, relative to its flux radius.
    """
    nfp, phi_grid = int(solution.inputs.axis.nfp), np.asarray(solution.phi)
    surfaces = [flux_surface(solution, radius * np.sqrt(s), 256) for s in levels]
    start = np.array([[surface["R"][0, 0], 0.0, surface["Z"][0, 0]] for surface in surfaces])
    tracing = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=jnp.asarray(start),
                      maxtime=maxtime, times_to_trace=n_steps, atol=tolerance, rtol=tolerance)
    sections, rows = {}, []
    for fraction in (0.0, 0.5):
        k = int(np.argmin(np.abs(phi_grid - fraction * 2 * np.pi / nfp)))
        sections[fraction] = dict(index=k, lines=[])
        for line, surface, s_level in zip(np.asarray(tracing.trajectories_xyz), surfaces, levels):
            x, y, z = line[:, :3].T
            phase = (np.unwrap(np.arctan2(y, x)) - phi_grid[k]) * nfp / (2 * np.pi)  # Periods since the plane.
            crossing = np.flatnonzero(np.floor(phase[1:]) != np.floor(phase[:-1]))
            weight = (np.ceil(np.minimum(phase[crossing], phase[crossing + 1])) - phase[crossing]) / (
                phase[crossing + 1] - phase[crossing])
            R = np.hypot(x, y)
            points = np.stack((R[crossing] + weight * (R[crossing + 1] - R[crossing]),
                               z[crossing] + weight * (z[crossing + 1] - z[crossing])), -1)
            curve = np.stack((surface["R"][:, k], surface["Z"][:, k]), -1)
            edge = np.roll(curve, -1, axis=0) - curve
            delta = points[:, None, :] - curve[None, :, :]
            along = np.clip(np.sum(delta * edge, -1) / np.sum(edge * edge, -1), 0, 1)
            distance = np.sqrt(np.min(np.sum((delta - along[..., None] * edge)**2, -1), axis=1))
            sections[fraction]["lines"].append(points)
            rows.append(dict(s=float(s_level), plane=fraction, punctures=int(len(points)),
                             rms_over_flux_radius=float(np.sqrt(np.mean(distance**2)) / surface["radius"])))
    return sections, rows


def poloidal_orientation(R, Z, R_axis, Z_axis):
    """+1 if the poloidal angle turns counterclockwise in the (R, Z) plane, -1 otherwise.

    The laboratory rotation of a field line per toroidal turn, taken in +phi (cylindrical), is
    this sign times a transform defined with that poloidal angle and a toroidal angle along +phi.
    """
    turn = np.sum(np.diff(np.unwrap(np.arctan2(np.r_[Z, Z[:1]] - Z_axis, np.r_[R, R[:1]] - R_axis))))
    return int(np.sign(turn))


def near_axis_lab_iota(solution, radius):
    """Near-axis transform as a laboratory rotation: counterclockwise in (R, Z) per +phi turn."""
    surface = flux_surface(solution, radius, 64)
    sign = poloidal_orientation(surface["R"][:, 0], surface["Z"][:, 0], float(solution.R0[0]), float(solution.Z0[0]))
    return sign * float(solution.iota), sign  # The Boozer angle increases with phi, as dvarphi/dphi > 0.


def traced_lab_iota(field, solution, radius, maxtime=300.0, n_steps=6000, tolerance=1e-10):
    """Laboratory rotation of one traced coil-field line about the near-axis magnetic axis.

    Only a vacuum test: then the coil field is the total field. The winding angle about the
    axis, sampled at the line's own toroidal angle, is divided by the toroidal angle travelled.
    """
    surface = flux_surface(solution, radius, 64)
    start = jnp.asarray([[surface["R"][0, 0], 0.0, surface["Z"][0, 0]]])
    tracing = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=start,
                      maxtime=maxtime, times_to_trace=n_steps, atol=tolerance, rtol=tolerance)
    x, y, z = np.asarray(tracing.trajectories_xyz)[0, :, :3].T
    phi = np.unwrap(np.arctan2(y, x))
    nfp, grid = int(solution.inputs.axis.nfp), np.asarray(solution.phi)
    period = 2 * np.pi / nfp
    R_axis = np.interp(np.mod(phi, period), grid, np.asarray(solution.R0), period=period)
    Z_axis = np.interp(np.mod(phi, period), grid, np.asarray(solution.Z0), period=period)
    winding = np.unwrap(np.arctan2(z - Z_axis, np.hypot(x, y) - R_axis))
    return float((winding[-1] - winding[0]) / (phi[-1] - phi[0])), float(abs(phi[-1] - phi[0]) / (2 * np.pi))


def toroidal_flux(field, solution, radius, plane=0, nrho=24, ntheta=128):
    """Toroidal flux of a field through a near-axis cross-section in the plane phi = phi[plane].

    An independent check of the surface labels: with the coil field as the total field (vacuum),
    a surface of label r should enclose pi B0 r^2. Gauss-Legendre in the fraction rho of the way
    from the axis to the contour, trapezoidal in the contour parameter (star-shaped contours).
    """
    surface = flux_surface(solution, radius, ntheta)
    Rc, Zc, phi = surface["R"][:, plane], surface["Z"][:, plane], float(surface["phi"][plane])
    Ra, Za = float(solution.R0[plane]), float(solution.Z0[plane])
    k = 1j * np.fft.fftfreq(ntheta, 1 / ntheta)
    dRc, dZc = np.real(np.fft.ifft(k * np.fft.fft(Rc))), np.real(np.fft.ifft(k * np.fft.fft(Zc)))
    rho, weight = np.polynomial.legendre.leggauss(nrho)
    rho, weight = (rho + 1) / 2, weight / 2
    R, Z = Ra + rho[:, None] * (Rc - Ra), Za + rho[:, None] * (Zc - Za)
    jacobian = rho[:, None] * ((Rc - Ra) * dZc - (Zc - Za) * dRc)
    B = evaluate_field(field, np.stack((R * np.cos(phi), R * np.sin(phi), Z), -1))
    B_phi = -B[..., 0] * np.sin(phi) + B[..., 1] * np.cos(phi)
    return float(abs(np.sum(weight[:, None] * B_phi * jacobian) * 2 * np.pi / ntheta))


# ------------------------------ VMEX benchmark -------------------------------
def flux_indices(wout, levels):
    s = np.asarray(wout.phi) / np.asarray(wout.phi)[-1]
    return [(int(i), float(s[i])) for i in sorted({int(np.argmin(np.abs(s - level))) for level in levels}) if i > 0]


def compare_to_near_axis(wout, solution, radius, levels, ntheta=256):
    """Distance between VMEX and near-axis surfaces of equal toroidal flux, on every axis plane.

    Two errors are reported separately because they have different causes. The axis offset
    is where the coils put the magnetic axis. The shape error measures each near-axis surface
    after moving it onto VMEX's own axis, so it tests the expansion itself. Left combined, a
    small rigid offset dominates the inner surfaces, whose flux radius is small.
    """
    phi, theta = np.asarray(solution.phi), np.arange(ntheta) * 2 * np.pi / ntheta
    RA, ZA = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=phi)
    dR, dZ = RA[0] - np.asarray(solution.R0), ZA[0] - np.asarray(solution.Z0)
    rows = []
    for index, s in flux_indices(wout, levels):
        r = radius * np.sqrt(s)
        near = flux_surface(solution, r, ntheta)
        RV, ZV = surface_rz(wout, s_index=index, theta=theta, phi=phi)
        vmex = [np.stack((RV[:, k], ZV[:, k]), -1) for k in range(phi.size)]
        raw = [contour_distance(np.stack((near["R"][:, k], near["Z"][:, k]), -1), vmex[k]) for k in range(phi.size)]
        moved = [contour_distance(np.stack((near["R"][:, k] + dR[k], near["Z"][:, k] + dZ[k]), -1), vmex[k])[0]
                 for k in range(phi.size)]
        rms = float(np.sqrt(np.mean([p[0]**2 for p in raw])))
        rows.append(dict(s=s, flux_radius_m=float(r), rms_m=rms, max_m=float(max(p[1] for p in raw)),
                         rms_over_flux_radius=rms / r,
                         shape_rms_over_flux_radius=float(np.sqrt(np.mean(np.square(moved)))) / r))
    shift = np.hypot(dR, dZ)
    return dict(surfaces=rows, axis_shift_rms_m=float(np.sqrt(np.mean(shift**2))),
                axis_shift_max_m=float(shift.max()), axis_shift_over_benchmark_radius=float(shift.max() / radius),
                benchmark_radius_m=float(radius))


def compare_equilibria(first, second, radius, levels, nfp, ntheta=256, nplanes=8):
    """Distance between two VMEX equilibria on shared flux labels (e.g. mgrid against direct)."""
    theta, phi = np.arange(ntheta) * 2 * np.pi / ntheta, np.arange(nplanes) * 2 * np.pi / nfp / nplanes
    rows = []
    for (i, s), (j, _) in zip(flux_indices(first, levels), flux_indices(second, levels)):
        RA, ZA = surface_rz(first, s_index=i, theta=theta, phi=phi)
        RB, ZB = surface_rz(second, s_index=j, theta=theta, phi=phi)
        rms = [contour_distance(np.stack((RA[:, k], ZA[:, k]), -1), np.stack((RB[:, k], ZB[:, k]), -1))[0]
               for k in range(nplanes)]
        rows.append(dict(s=s, rms_m=float(np.sqrt(np.mean(np.square(rms)))),
                         rms_over_flux_radius=float(np.sqrt(np.mean(np.square(rms))) / (radius * np.sqrt(s)))))
    return rows


class GuardedMgridField(vj.MgridField):
    """An mgrid field that is NaN outside its (R, Z) table rather than clamped to its edge.

    VMEX clamps R and Z to the table, so a boundary that left the grid would silently see the
    edge field. Every NESTOR evaluation goes through ``b_cyl``, so a NaN there fails the solve.
    """

    def b_cyl(self, r, phi, z):
        values = super().b_cyl(r, phi, z)
        r, _, z = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
        inside = (r >= self.rmin) & (r <= self.rmax) & (z >= self.zmin) & (z <= self.zmax)
        return tuple(jnp.where(inside, value, jnp.nan) for value in values)


jax.tree_util.register_dataclass(GuardedMgridField, data_fields=["br", "bp", "bz", "extcur"],
                                 meta_fields=["rmin", "rmax", "zmin", "zmax", "nfp"])


def _cartesian(field_cyl, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.stack((field_cyl[0] * c - field_cyl[1] * s, field_cyl[0] * s + field_cyl[1] * c, field_cyl[2]), -1)


def write_coil_mgrid(field, surface, radius, shape, nfp, path, margin=None):
    """MAKEGRID NetCDF file of the coils bracketing the plasma, checked by reading it back.

    One coil group, scaled mode, unit current: with ``EXTCUR = 1`` the file reproduces the
    physical coil currents exactly once. Returns the file path and the direct, in-memory and
    reloaded fields compared on the plasma boundary.
    """
    margin = max(0.08 * float(surface["R"].mean()), 4 * radius) if margin is None else margin
    bounds = dict(rmin=float(surface["R"].min()) - margin, rmax=float(surface["R"].max()) + margin,
                  zmin=float(surface["Z"].min()) - margin, zmax=float(surface["Z"].max()) + margin)
    start = perf_counter()
    data = tabulate_cartesian_field(field, ir=shape[0], jz=shape[1], kp=shape[2], nfp=nfp, label="essos_coils",
                                    **bounds)
    write_mgrid(path, data)
    reloaded = read_mgrid(path)
    table_difference = max(float(np.max(np.abs(getattr(reloaded, k) - getattr(data, k)))) for k in ("br", "bp", "bz"))
    R, Z, phi = surface["R"], surface["Z"], np.broadcast_to(surface["phi"], surface["R"].shape)
    direct = evaluate_field(field, surface["xyz"])
    sample = lambda grid: _cartesian([np.asarray(v) for v in grid.b_cyl(jnp.asarray(R), jnp.asarray(phi),
                                                                         jnp.asarray(Z))], phi)
    in_memory = sample(vj.MgridField.from_mgrid_data(data, extcur=[1.0]))
    from_file = sample(vj.MgridField.from_file(path, extcur=[1.0]))
    error = np.linalg.norm(from_file - direct, axis=-1)
    return Path(path), dict(
        file=Path(path).name, bounds=bounds, margin_m=margin, shape_R_Z_phi=list(shape), seconds=perf_counter() - start,
        mgrid_mode=reloaded.mgrid_mode, raw_coil_cur=list(reloaded.raw_coil_cur), extcur=[1.0],
        write_read_table_difference_T=table_difference,
        reloaded_minus_in_memory_max_T=float(np.max(np.abs(from_file - in_memory))),
        interpolation_error_rms_T=float(np.sqrt(np.mean(error**2))), interpolation_error_max_T=float(error.max()),
        interpolation_error_max_over_B=float(np.max(error / np.linalg.norm(direct, axis=-1))))


def grid_margin(wout, bounds, ntheta=128, nphi=64):
    """Smallest distance [m] from the final plasma boundary to the edge of the mgrid table."""
    phi = np.arange(nphi) * 2 * np.pi / int(wout.nfp) / nphi
    R, Z = surface_rz(wout, s_index=int(wout.ns) - 1, theta=np.arange(ntheta) * 2 * np.pi / ntheta, phi=phi)
    return float(min(R.min() - bounds["rmin"], bounds["rmax"] - R.max(), Z.min() - bounds["zmin"],
                     bounds["zmax"] - Z.max()))


def interface_check(wout, ntheta=64, nphi=64):
    """Plasma-side against vacuum-side field on the free boundary.

    VMEX enforces B.n = 0 on the vacuum side (NESTOR) and total-pressure balance
    |B_in|^2 + 2 mu0 p = |B_out|^2. It does not impose continuity of the tangential field, so a
    jump there is a surface current mu0 K = n x (B_out - B_in) that the no-sheet-current source
    model excludes. The plasma side is extrapolated from the last two half-mesh surfaces.
    """
    if getattr(wout, "bsubumnc_sur", None) is None:
        return None
    theta = np.arange(ntheta) * 2 * np.pi / ntheta
    phi = np.arange(nphi) * 2 * np.pi / int(wout.nfp) / nphi
    xm, xn = np.asarray(wout.xm_nyq), np.asarray(wout.xn_nyq)
    cosine = np.cos(xm[:, None, None] * theta[None, :, None] - xn[:, None, None] * phi[None, None, :])
    evaluate = lambda c: np.einsum("m,mtp->tp", np.asarray(c), cosine)
    edge = lambda name: 1.5 * np.asarray(getattr(wout, name))[-1] - 0.5 * np.asarray(getattr(wout, name))[-2]
    Bu_in, Bv_in, B_in = evaluate(edge("bsubumnc")), evaluate(edge("bsubvmnc")), evaluate(edge("bmnc"))
    Bu_out, Bv_out = evaluate(wout.bsubumnc_sur), evaluate(wout.bsubvmnc_sur)
    B2_out = Bu_out * evaluate(wout.bsupumnc_sur) + Bv_out * evaluate(wout.bsupvmnc_sur)
    # Tangential metric of the boundary, to turn the covariant jumps into a physical field.
    x, xg = np.asarray(wout.xm), np.asarray(wout.xn)
    angle = x[:, None, None] * theta[None, :, None] - xg[:, None, None] * phi[None, None, :]
    rmnc, zmns = np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]
    R = np.einsum("m,mtp->tp", rmnc, np.cos(angle))
    Ru = np.einsum("m,mtp->tp", -rmnc * x, np.sin(angle))
    Rv = np.einsum("m,mtp->tp", rmnc * xg, np.sin(angle))
    Zu = np.einsum("m,mtp->tp", zmns * x, np.cos(angle))
    Zv = np.einsum("m,mtp->tp", -zmns * xg, np.cos(angle))
    guu, guv, gvv = Ru**2 + Zu**2, Ru * Rv + Zu * Zv, R**2 + Rv**2 + Zv**2
    det = guu * gvv - guv**2
    du, dv = Bu_out - Bu_in, Bv_out - Bv_in
    jump = np.sqrt((gvv * du**2 - 2 * guv * du * dv + guu * dv**2) / det)
    p_edge = float(np.asarray(wout.presf)[-1])
    balance = (B_in**2 + 2 * 4e-7 * np.pi * p_edge - B2_out) / B2_out
    return dict(tangential_jump_max_over_B=float(np.max(jump / np.sqrt(B2_out))),
                tangential_jump_rms_over_B=float(np.sqrt(np.mean((jump / np.sqrt(B2_out))**2))),
                pressure_balance_max_rel=float(np.max(np.abs(balance))),
                pressure_balance_rms_rel=float(np.sqrt(np.mean(balance**2))),
                note="plasma side extrapolated from the last two half-mesh surfaces")


def solve_free_boundary(solution, external_field, radius, directory, name, settings, mgrid=None):
    """VMEX free-boundary equilibrium in fixed coils, seeded by the near-axis boundary.

    The near-axis surface is only the initial guess: with ``LFREEB = T`` VMEX moves it.
    ``external_field`` is an ESSOS ``BiotSavart`` (direct evaluation). With ``mgrid``, a
    ``(path, report)`` pair from :func:`write_coil_mgrid`, the field is read from that MAKEGRID
    file through the deck's ``MGRID_FILE`` and ``EXTCUR`` instead, and made NaN off the grid.
    The deck actually solved is saved as ``input.<name>.runtime``.
    Returns ``(wout or None, report)``; only ``report["converged"]`` equilibria are accepted.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(route=name, radius_m=float(radius), converged=False)
    start = perf_counter()
    try:
        export = to_vmec(solution, directory / f"input.{name}", r=radius, mpol=settings["mpol"],
                         ntor=settings["ntor"], ntheta=settings["ntheta_boundary"],
                         parameters=dict(ns_array=settings["ns"], ftol_array=settings["ftol"],
                                         niter_array=settings["niter"], delt=settings["delt"]))
        report.update(phiedge_Wb=export.phiedge, pressure_axis_Pa=export.pressure_axis, curtor_A=export.curtor,
                      boundary_fit_error_m=float(max(export.boundary.maximum_R_reconstruction_error,
                                                     export.boundary.maximum_Z_reconstruction_error)))
        # A deck without MGRID_FILE is demoted to fixed boundary on reading, so name the field source.
        source = dict(mgrid_file=mgrid[0].name, extcur=np.array([1.0])) if mgrid else dict(
            mgrid_file=f"essos_coils({name})")  # Direct route: a label, the coils are in coils_*.json.
        inp = dataclasses.replace(vj.VmecInput.from_file(export.path), lfreeb=True, nzeta=settings["nzeta"], **source)
        inp.to_indata(directory / f"input.{name}.runtime")
        report.update(runtime=dict(file=f"input.{name}.runtime", lfreeb=bool(inp.lfreeb), mgrid_file=inp.mgrid_file,
                                   mpol=int(inp.mpol), ntor=int(inp.ntor), nzeta=int(inp.nzeta), ncurr=int(inp.ncurr),
                                   highest_poloidal_mode_max_m=float(max(np.max(np.abs(np.asarray(inp.rbc)[:, -1])),
                                                                         np.max(np.abs(np.asarray(inp.zbs)[:, -1]))))))
        if mgrid:  # The solver's own file loader and EXTCUR scaling, then the off-grid guard.
            loaded = _external_field_from_input(inp, mgrid[0])
            external_field = GuardedMgridField(br=loaded.br, bp=loaded.bp, bz=loaded.bz, extcur=loaded.extcur,
                                               rmin=loaded.rmin, rmax=loaded.rmax, zmin=loaded.zmin,
                                               zmax=loaded.zmax, nfp=loaded.nfp)
        with (directory / f"vmex_{name}.log").open("w") as log:
            def emit(*values, **kwargs):
                print(*values, **kwargs)
                print(*values, **{**kwargs, "file": log, "flush": True})
            result = vj.solve_free_boundary_multigrid(inp, external_field=external_field, verbose=True,
                                                      emit=emit, raise_on_max_iterations=False)
        wout = vj.wout_from_state(inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                                  fsql=float(result.fsql), niter=int(result.iterations),
                                  converged=bool(result.converged), vacuum_output=result.vacuum)
        vj.write_wout(directory / f"wout_{name}.nc", wout)
        report.update(seconds=perf_counter() - start, iterations=int(result.iterations),
                      fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
                      vacuum_active=result.vacuum is not None, betatotal=float(wout.betatotal),
                      iota_axis=float(np.asarray(wout.iotaf)[0]), iota_edge=float(np.asarray(wout.iotaf)[-1]),
                      iota_near_axis=float(solution.iota), aspect=float(wout.aspect))
        # Signed transforms converted to one laboratory convention (see poloidal_orientation).
        middle = int(wout.ns) // 2
        theta = np.arange(64) * 2 * np.pi / 64
        RM, ZM = surface_rz(wout, s_index=middle, theta=theta, phi=np.zeros(1))
        RA, ZA = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=np.zeros(1))
        vmex_sign = poloidal_orientation(RM[:, 0], ZM[:, 0], float(RA[0, 0]), float(ZA[0, 0]))
        near_lab, near_sign = near_axis_lab_iota(solution, radius)
        report.update(signed=dict(vmex_iota_axis_raw=report["iota_axis"], vmex_theta_orientation=vmex_sign,
                                  vmex_iota_axis_lab=vmex_sign * report["iota_axis"],
                                  near_axis_iota_raw=float(solution.iota), near_axis_theta_orientation=near_sign,
                                  near_axis_iota_lab=near_lab))
        report["interface"] = interface_check(wout)
        report["converged"] = bool(result.converged and result.vacuum is not None)
        if mgrid:
            report["grid_margin_m"] = grid_margin(wout, mgrid[1]["bounds"])
            if report["grid_margin_m"] <= 0:
                report["converged"] = False
                report["error"] = "The final boundary leaves the mgrid table."
        if report["converged"]:
            report["near_axis"] = compare_to_near_axis(wout, solution, radius, settings["flux_levels"])
            return wout, report
        report.setdefault("error", "VMEX did not reach FTOL with the vacuum region active.")
    except Exception as error:  # A failed case must not stop the remaining benchmarks.
        report.update(error=f"{type(error).__name__}: {error}", seconds=perf_counter() - start)
        (directory / f"failure_{name}.txt").write_text(traceback.format_exc())
    print(f"  [{name}] NOT ACCEPTED: {report['error']}")
    return None, report


# ---------------------------------- figures ----------------------------------
def _torus(array, nfp):
    """Replicate one field period of [theta, phi, xyz] samples around the torus and close both seams."""
    turns = []
    for k in range(nfp):
        c, s = np.cos(2 * np.pi * k / nfp), np.sin(2 * np.pi * k / nfp)
        turns.append(np.stack((array[..., 0] * c - array[..., 1] * s, array[..., 0] * s + array[..., 1] * c,
                               array[..., 2]), -1))
    full = np.concatenate(turns, axis=1)
    full = np.concatenate((full, full[:, :1]), axis=1)
    return np.concatenate((full, full[:1]), axis=0)


def plot_optimization(history, matches, path, title):
    """Least-squares cost history, and the axis mismatch before and after against the plasma terms."""
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.9), layout="constrained", width_ratios=[1.15, 1])
        if len(history):
            evaluation = np.arange(1, len(history) + 1)
            ax.semilogy(evaluation, history, ".", color=MUTED, ms=3, alpha=0.6, label="every trial step")
            ax.semilogy(evaluation, np.minimum.accumulate(history), color=COLORS["optimized"], lw=1.8,
                        label="running minimum")
            ax.legend(loc="upper right")
        ax.set(xlabel="function evaluation", ylabel=r"cost  $\frac{1}{2}\sum r^2$", title="Optimization history")
        ax.grid(True, which="both", alpha=0.6)
        # The normalizations of the objective, so the bars are the residuals the optimizer saw.
        labels = ["field\n/ $B_0$", "gradient\n$R_0$ / $B_0$", "Hessian\n$R_0^2$ / $B_0$"]
        keys = [("field_rms_over_B0", "plasma_field_rms_over_B0"),
                ("gradient_rms_R0_over_B0", "plasma_gradient_rms_R0_over_B0"),
                ("hessian_rms_R0sq_over_B0", "plasma_hessian_rms_R0sq_over_B0")]
        position = np.arange(3)
        for offset, name in ((-0.27, "initial"), (0.0, "optimized")):
            bx.bar(position + offset, [matches[name][k[0]] for k in keys], 0.25, color=COLORS[name],
                   label=f"{name} coils $-$ target")
        bx.bar(position + 0.27, [matches["optimized"][k[1]] for k in keys], 0.25, color=COLORS["direct"],
               label="plasma contribution")
        bx.set_yscale("log")
        bx.set_xticks(position, labels)
        bx.set(ylabel="axis RMS, relative", title="Coil match on the magnetic axis")
        bx.grid(True, axis="y", which="both", alpha=0.6)
        fig.legend(*bx.get_legend_handles_labels(), loc="outside lower center", ncols=3)
        fig.suptitle(title, fontsize=12)
        fig.savefig(path)
    return fig


def plot_axis_profiles(solution, targets, field, path, title):
    """Frenet components along the axis of the plasma field, and of the coil mismatch, relative to B0."""
    frame = np.stack([np.asarray(getattr(solution.geometry, n + "_cartesian")) for n in ("tangent", "normal", "binormal")], 1)
    phi = np.asarray(solution.phi) * int(solution.inputs.axis.nfp) / (2 * np.pi)
    B0 = float(solution.inputs.B0)
    plasma = np.einsum("nai,ni->na", frame, targets["B_plasma"]) / B0
    mismatch = np.einsum("nai,ni->na", frame, evaluate_field(field, targets["points"]) - targets["B"]) / B0
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained", sharex=True)
        for k, (name, color) in enumerate(zip(("tangent", "normal", "binormal"), ("#0f6fae", "#c2410c", "#15803d"))):
            axes[0].plot(phi, plasma[:, k], color=color, lw=1.8, label=name)
            axes[1].plot(phi, mismatch[:, k], color=color, lw=1.8, label=name)
        axes[0].set(title=r"Plasma field on the axis, $\mathbf{B}_p$",
                    ylabel=r"$\mathbf{B}_p\cdot\hat{\mathbf{e}}\,/\,B_0$")
        axes[1].set(title=r"Optimized coils minus target, $\mathbf{B}_{coils}-(\mathbf{B}_{tot}-\mathbf{B}_p)$",
                    ylabel=r"$\delta\mathbf{B}\cdot\hat{\mathbf{e}}\,/\,B_0$")
        for ax in axes:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
            ax.set_xlabel(r"toroidal angle  $\phi\,n_{fp}/2\pi$")
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.grid(True, alpha=0.6)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
        fig.suptitle(title, fontsize=12)
        fig.savefig(path)
    return fig


def _equal_3d(axes, clouds):
    """Equal-aspect matplotlib 3D axes. Unused here since the 3D figure moved to pyvista; kept for callers."""
    points = np.concatenate([np.asarray(c).reshape(-1, 3) for c in clouds])
    centre, half = (points.min(0) + points.max(0)) / 2, 0.52 * np.ptp(points, axis=0).max()
    for ax in axes:
        ax.set(xlim=centre[0] + half * np.array([-1, 1]), ylim=centre[1] + half * np.array([-1, 1]),
               zlim=centre[2] + half * np.array([-1, 1]), xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("z [m]", rotation=90)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=32, azim=38)
        ax.tick_params(labelsize=7, pad=-2)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
            pane._axinfo["grid"].update(color="#e3e6ea", linewidth=0.4)


# The boundary mismatch is shown as its magnitude on a logarithmic scale with viridis. A signed
# diverging map is white (or, for the dark-centred ones, black) wherever the mismatch is small,
# which after optimization is almost the whole surface, so the structure is lost. The magnitude on
# a log scale resolves the small background and the localized peaks at once; viridis is perceptually
# uniform, readable in greyscale and for colour-blind readers, and nowhere near white.
NORMAL_ERROR_CMAP = "viridis"
NORMAL_ERROR_DECADES = 3.0            # Colour range of each panel: its maximum and three decades below
COIL_COLOR = "#2f343b"


def surface_inflation(surface, axis):
    """Factor by which a thin near-axis surface is drawn inflated about its axis to be visible next to the coils."""
    major = np.mean(np.hypot(np.asarray(axis)[..., 0], np.asarray(axis)[..., 1]))
    return max(1.0, 0.12 * major / surface["radius"])


def normal_error_norm(surface, decades=NORMAL_ERROR_DECADES):
    """Logarithmic colour scale of |B.n|/|B| in percent, from the surface maximum down ``decades``."""
    top = 100 * float(np.max(np.abs(surface["error"])))
    return LogNorm(top * 10.0**-decades, top, clip=True)


def render_coils_and_surface(surface, field, axis=None, norm=None, cmap=NORMAL_ERROR_CMAP, inflate=None,
                             extent=None, window_size=(1600, 1400), elevation=32.0, azimuth=38.0):
    """Depth-correct off-screen pyvista render of coils and a boundary coloured by |B.n|/|B|.

    ``surface`` is the dict of :func:`normal_field_error` (one field period of ``xyz`` and
    ``error``), ``field`` an ESSOS ``BiotSavart`` or ``Coils``, and ``axis`` the Cartesian axis
    of the same period (``solution.geometry.position_cartesian``). The surface is drawn inflated by
    ``inflate`` about the axis (default :func:`surface_inflation`, 1 without an axis), but it is
    coloured by the error on the physical surface. ``norm`` maps |B.n|/|B| in percent to colour
    (default :func:`normal_error_norm`). Renders with the same ``extent`` (half-width of the scene
    in metres, default from the coils) and angles share one camera, so panels compare directly.
    Returns an RGB image on a white background, to place in a matplotlib figure.
    """
    try:
        import pyvista as pv
    except ImportError as error:
        raise ImportError("The 3D coil figure needs pyvista: pip install pyvista") from error
    coils = getattr(field, "coils", field)
    nfp, norm = int(surface["nfp"]), norm or normal_error_norm(surface)
    xyz = _torus(surface["xyz"], nfp)
    if axis is not None:
        axis = _torus(np.asarray(axis)[None], nfp)[0]
        inflate = surface_inflation(surface, axis) if inflate is None else inflate
        xyz = axis[None] + inflate * (xyz - axis[None])
    error = 100 * np.abs(np.pad(np.tile(surface["error"], (1, nfp)), ((0, 1), (0, 1)), mode="wrap"))
    colours = (255 * plt.get_cmap(cmap)(norm(error))[..., :3]).astype(np.uint8)
    # Fortran order: the (theta, phi) grid becomes VTK's (i, j) with i fastest.
    mesh = pv.StructuredGrid(*(np.asfortranarray(xyz[..., k]) for k in range(3)))
    mesh.point_data["colour"] = colours.reshape(-1, 3, order="F")
    gamma = np.asarray(coils.gamma)
    extent = 0.55 * np.ptp(gamma.reshape(-1, 3), axis=0).max() if extent is None else extent
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size), lighting="light_kit")
    try:
        plotter.set_background("white")
        plotter.add_mesh(mesh, scalars="colour", rgb=True, smooth_shading=True, ambient=0.25, diffuse=0.8,
                         specular=0.1)
        for curve in gamma:
            tube = pv.Spline(np.vstack((curve, curve[:1])), 4 * len(curve)).tube(radius=0.009 * extent, n_sides=24)
            plotter.add_mesh(tube, color=COIL_COLOR, smooth_shading=True, ambient=0.2, diffuse=0.7,
                             specular=0.45, specular_power=30)
        elevation, azimuth = np.radians(elevation), np.radians(azimuth)
        direction = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                              np.sin(elevation)])
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.position = tuple(8 * extent * direction)
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.camera.view_angle = 2 * np.degrees(np.arctan(1.1 / 8))
        plotter.enable_anti_aliasing("ssaa")
        return plotter.screenshot(return_img=True)
    finally:
        plotter.close()


def plot_coils_and_normal_error(states, surfaces, path, title):
    """Coils and the normal-field mismatch on the plasma boundary, before and after.

    The 3D scenes are rendered by pyvista, which occludes correctly, and laid out with
    matplotlib so that the titles and colour bars match the other figures.
    """
    names = list(states)
    extent = max(0.55 * np.ptp(np.asarray(states[n]["field"].coils.gamma).reshape(-1, 3), axis=0).max() for n in names)
    axes_xyz = {n: np.asarray(states[n]["solution"].geometry.position_cartesian) for n in names}
    inflate = min(surface_inflation(surfaces[n], _torus(axes_xyz[n][None], surfaces[n]["nfp"])[0]) for n in names)
    norms = {n: normal_error_norm(surfaces[n]) for n in names}
    images = {n: render_coils_and_surface(surfaces[n], states[n]["field"], axis=axes_xyz[n], norm=norms[n],
                                          inflate=inflate, extent=extent) for n in names}
    # One crop for every panel, so the common camera also gives a common scale.
    filled = np.any([np.any(image < 250, axis=-1) for image in images.values()], axis=0)
    rows, columns = np.flatnonzero(filled.any(1)), np.flatnonzero(filled.any(0))
    pad = 12
    crop = (slice(max(rows[0] - pad, 0), rows[-1] + pad), slice(max(columns[0] - pad, 0), columns[-1] + pad))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(names), figsize=(6.6, 3.55), layout="constrained", squeeze=False)
        for ax, name in zip(axes[0], names):
            normal = states[name]["normal"]
            ax.imshow(images[name][crop], interpolation="lanczos")
            ax.set_axis_off()
            ax.set_title(f"{name.capitalize()} coils\nmax {100 * normal['normal_error_max']:.3g} %,  "
                         f"RMS {100 * normal['normal_error_rms']:.3g} %", fontsize=9)
            bar = fig.colorbar(ScalarMappable(norm=norms[name], cmap=NORMAL_ERROR_CMAP), ax=ax, shrink=0.8,
                               orientation="horizontal", aspect=24, pad=0.02)
            bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            bar.ax.tick_params(labelsize=8)
            bar.outline.set_linewidth(0.5)
            bar.set_label(r"$|\mathbf{B}\cdot\hat{\mathbf{n}}|\,/\,|\mathbf{B}|$  [%]", fontsize=9)
        note = f" (surface drawn {inflate:.0f}x inflated about the axis)" if inflate > 1 else ""
        fig.suptitle(f"{title}\nBoundary at a = {surfaces[names[-1]]['radius']:.3g} m{note}", fontsize=10)
        fig.savefig(path, dpi=300)
    return fig


def plot_cross_sections(states, equilibria, radius, levels, path, title, fractions=(0.0, 0.25, 0.5, 0.75)):
    """Equal-flux contours of the near-axis model and the VMEX free-boundary equilibria."""
    theta = np.arange(257) * 2 * np.pi / 256
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(states), len(fractions), figsize=(3.3 * len(fractions), 3.5 * len(states)),
                                 layout="constrained", squeeze=False)
        for row, (name, state) in enumerate(states.items()):
            solution = state["solution"]
            phi_grid, period = np.asarray(solution.phi), 2 * np.pi / int(solution.inputs.axis.nfp)
            routes = {k: w for k, w in equilibria[name].items() if w is not None}
            reference = next(iter(routes.values()), None)
            used = flux_indices(reference, levels) if reference is not None else [(None, s) for s in levels]
            for column, fraction in enumerate(fractions):
                ax, k = axes[row, column], int(np.argmin(np.abs(phi_grid - fraction * period)))
                for _, s in used:
                    near = flux_surface(solution, radius * np.sqrt(s), 256)
                    ax.plot(np.r_[near["R"][:, k], near["R"][0, k]], np.r_[near["Z"][:, k], near["Z"][0, k]],
                            color=COLORS["near"], lw=1.6)
                for route, (wout, style) in {r: (w, s) for (r, w), s in zip(routes.items(), ("--", ":"))}.items():
                    for index, _ in flux_indices(wout, levels):
                        RV, ZV = surface_rz(wout, s_index=index, theta=theta, phi=phi_grid[k:k + 1])
                        ax.plot(RV[:, 0], ZV[:, 0], style, color=COLORS[route], lw=1.5)
                    RA, ZA = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=phi_grid[k:k + 1])
                    ax.plot(RA[0, 0], ZA[0, 0], "x", color=COLORS[route], ms=6, mew=1.6)
                ax.plot(float(solution.R0[k]), float(solution.Z0[k]), "+", color=COLORS["near"], ms=8, mew=1.6)
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(True, alpha=0.6)
                ax.set_title(rf"$\phi$ = {fraction:g} period" if len(states) == 1 else rf"{name}:  $\phi$ = {fraction:g} period")
                ax.set_xlabel("R [m]")
                if column == 0:
                    ax.set_ylabel("Z [m]")
        handles = [plt.Line2D([], [], color=COLORS["near"], lw=1.6, label="near-axis expansion")]
        handles += [plt.Line2D([], [], color=COLORS[r], ls=s, lw=1.5, label=f"VMEX free boundary, {r} coil field")
                    for r, s in zip(("direct", "mgrid"), ("--", ":")) if any(equilibria[n].get(r) is not None for n in states)]
        fig.legend(handles=handles, loc="outside lower center", ncols=3)
        fig.suptitle(f"{title}\nSurfaces of equal toroidal flux, s = " + ", ".join(f"{s:g}" for _, s in used)
                     + f"  (a = {radius:.3g} m); markers are the magnetic axes", fontsize=12)
        fig.savefig(path)
    return fig


def plot_poincare(solution, sections, radius, levels, path, title):
    """Field-line punctures of the coil field over the near-axis flux surfaces."""
    shades = plt.get_cmap("plasma")(np.linspace(0.1, 0.8, len(levels)))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(sections), figsize=(5.2 * len(sections), 4.6), layout="constrained")
        for ax, (fraction, section) in zip(np.atleast_1d(axes), sections.items()):
            k = section["index"]
            for s_level, points, shade in zip(levels, section["lines"], shades):
                near = flux_surface(solution, radius * np.sqrt(s_level), 256)
                ax.plot(np.r_[near["R"][:, k], near["R"][0, k]], np.r_[near["Z"][:, k], near["Z"][0, k]],
                        color=COLORS["near"], lw=1.4)
                ax.scatter(points[:, 0], points[:, 1], s=5, color=shade, linewidths=0, zorder=3)
            ax.plot(float(solution.R0[k]), float(solution.Z0[k]), "+", color=COLORS["near"], ms=8, mew=1.6)
            # A field line that leaves the confined region would otherwise set the scale of the panel.
            half = 1.45 * max(np.ptp(near["R"][:, k]), np.ptp(near["Z"][:, k])) / 2
            ax.set(xlim=near["R"][:, k].mean() + half * np.array([-1, 1]), ylim=near["Z"][:, k].mean() + half * np.array([-1, 1]))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.6)
            ax.set(xlabel="R [m]", ylabel="Z [m]", title=rf"$\phi$ = {fraction:g} period")
        most = max(len(points) for section in sections.values() for points in section["lines"])
        escaped = [min(len(section["lines"][j]) for section in sections.values()) < most / 2 for j in range(len(levels))]
        handles = [plt.Line2D([], [], color=COLORS["near"], lw=1.4, label="near-axis expansion")]
        handles += [plt.Line2D([], [], ls="", marker="o", ms=4, color=c,
                               label=f"coil field line, s = {s:g}" + (" (not confined)" if lost else ""))
                    for s, c, lost in zip(levels, shades, escaped)]
        fig.legend(handles=handles, loc="outside lower center", ncols=3)
        fig.suptitle(f"{title}\nPoincare section of the coil field, exact in vacuum", fontsize=12)
        fig.savefig(path)
    return fig


def plot_benchmark_summary(summary, path, title):
    """Surface and axis agreement between VMEX and the near-axis model, for every solved case."""
    rows = [(f"{kind}\n{name}\n{route}", report["near_axis"]) for kind, cases in summary.items()
            for name, routes in cases.items() for route, report in routes.items()
            if isinstance(report, dict) and report.get("converged")]
    if not rows:
        return None
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained", width_ratios=[1.6, 1])
        shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, max(len(r[1]["surfaces"]) for r in rows)))
        width = 0.8 / len(shades)
        for j, shade in enumerate(shades):
            values = [r[1]["surfaces"][j]["shape_rms_over_flux_radius"] if j < len(r[1]["surfaces"]) else np.nan
                      for r in rows]
            ax.bar(np.arange(len(rows)) + (j - (len(shades) - 1) / 2) * width, values, width, color=shade,
                   label=f"s = {rows[0][1]['surfaces'][j]['s']:.3g}")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.set(ylabel=r"RMS shape error / flux radius $a\sqrt{s}$",
               title="Surface shape, with each surface placed on VMEX's axis")
        ax.grid(True, axis="y", which="both", alpha=0.6)
        low, high = ax.get_ylim()
        ax.set_ylim(low, high * 2.2)  # Headroom on the log axis so the legend clears every bar.
        ax.legend(ncols=4, loc="upper center", handlelength=1.2, columnspacing=1.0)
        shifts = [r[1]["axis_shift_over_benchmark_radius"] for r in rows]
        bx.bar(np.arange(len(rows)), shifts, 0.6, color=COLORS["direct"])
        # One magnitude, so a linear axis from zero; a log axis fitted to near-equal bars exaggerates them.
        bx.set_ylim(0, 1.25 * max(shifts))
        for x, value in enumerate(shifts):
            bx.annotate(f"{100 * value:.2f} %", (x, value), ha="center", va="bottom", xytext=(0, 3),
                        textcoords="offset points", fontsize=8)
        bx.set_xticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=8)
        bx.set(ylabel=r"max axis displacement / $a_b$", title="Magnetic-axis agreement")
        bx.grid(True, axis="y", which="both", alpha=0.6)
        fig.suptitle(title, fontsize=12)
        fig.savefig(path)
    return fig


def save_json(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.generic, np.ndarray, jnp.ndarray)):
            value = np.asarray(value).tolist()
        return None if isinstance(value, float) and not np.isfinite(value) else value
    Path(path).write_text(json.dumps(clean(data), indent=2) + "\n")
