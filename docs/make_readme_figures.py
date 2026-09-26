"""Regenerate the two README figures.

    python docs/make_readme_figures.py            # both
    python docs/make_readme_figures.py orbits     # just one: orbits or gradients

orbits:    alpha-particle guiding centers in the bundled reactor-scale QA
           equilibrium, through the magnetic axis and out to the LCFS
           (about 30 s).
gradients: a coil fit to a VMEC boundary, and the gradient of a guiding-center
           orbit objective with respect to every coil degree of freedom,
           checked against central finite differences (about 2 min).

Figures are written as palette PNGs next to this file.
"""

import os
import subprocess
import sys
from pathlib import Path

# The orbits are sharded over four CPU devices; the coil fit runs on one.
_DEVICES = {"orbits": 4}.get(sys.argv[1] if len(sys.argv) > 1 else "", 1)
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={_DEVICES}")

import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from essos.coils import Coils, CreateEquallySpacedCurves, Curves
from essos.constants import ONE_EV, PROTON_MASS
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart, Vmec

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "examples" / "input_files"
COILS_JSON = INPUTS / "ESSOS_biot_savart_LandremanPaulQA.json"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
INSIDE, OUTSIDE, COIL = "#1f5fa8", "#c8452c", "0.55"


def save(fig, name):
    """Save a figure as an optimized 64-colour palette PNG."""
    path = HERE / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    Image.open(path).convert("RGB").quantize(64, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"wrote {path.name}: {path.stat().st_size / 1024:.0f} kB")


def orbits():
    vmec = Vmec(str(INPUTS / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"), ntheta=32, nphi=64)
    # The bundled QA coils scaled to this reactor-scale equilibrium, for context.
    qa = Coils.from_json(str(COILS_JSON))
    coils = Coils(Curves(qa.dofs_curves * 10.127, qa.curves.n_segments, qa.nfp, qa.stellsym),
                  qa.dofs_currents_raw * -19.62 * 10.127)
    n = 48
    keys = jax.random.split(jax.random.key(1), 4)
    seeds = jnp.stack([jax.random.uniform(keys[0], (n,), minval=0.5, maxval=0.99),
                       jax.random.uniform(keys[1], (n,), maxval=2 * jnp.pi),
                       jax.random.uniform(keys[2], (n,), maxval=2 * jnp.pi)], axis=1)
    pitch = jax.random.uniform(keys[3], (n,), minval=-1, maxval=1)
    # Four seeds, found by a search, whose orbits pass within sqrt(s) < 0.02 of the axis.
    near_axis = jnp.array([[0.0370719, 3.8699471, 3.482848, 0.1616618], [0.0098948, 0.721827, 5.8708337, -0.490672],
                           [0.0142754, 3.6181706, 1.1491422, 0.3173895], [0.007962, 2.3559379, 4.3473652, 0.8969839]])
    seeds = jnp.concatenate([seeds, near_axis[:, :3]])
    pitch = jnp.concatenate([pitch, near_axis[:, 3]])
    particles = Particles(initial_xyz=seeds, initial_vparallel_over_v=pitch)  # 3.5 MeV alpha particles
    tracing = Tracing(field=vmec, model="GuidingCenterAdaptative", particles=particles, maxtime=1e-4,
                      timestep=1e-9, times_to_trace=2000, atol=1e-8, rtol=1e-8)
    lost_times = np.asarray(tracing.lost_times)
    lost = lost_times >= 0
    print(f"lost at the LCFS: {int(lost.sum())} of {len(lost)}")

    traj = np.asarray(tracing.trajectories)
    finite = np.isfinite(traj[..., 0]) & (traj[..., 0] < 1)
    s = np.where(finite, traj[..., 0], np.nan)
    xyz = np.asarray(jax.vmap(jax.vmap(vmec.to_xyz))(jnp.asarray(np.where(finite[..., None], traj[..., :3], 0.5))))
    xyz = np.where(finite[..., None], xyz, np.nan)
    t = np.asarray(tracing.times) * 1e6

    # Through the axis, one confined orbit near the edge, and the two latest
    # losses. The 3-D view shows the confined orbits for 40 us.
    order = np.flatnonzero(lost)[np.argsort(lost_times[lost])]
    edge = int(np.nanargmax(np.where(lost, np.nan, np.nanmax(s, axis=1))))
    show = [n, edge] + list(order[-2:])
    print(f"closest approach to the axis: sqrt(s) = {np.sqrt(np.nanmin(s[n, 100:])):.1e}")

    fig = plt.figure(figsize=(9.0, 3.6))
    ax = fig.add_axes([0.0, 0.0, 0.48, 1.0], projection="3d")
    for curve in np.asarray(coils.gamma):
        ax.plot(*np.vstack([curve, curve[:1]]).T, color=COIL, lw=0.7)
    lcfs = np.asarray(vmec.surface.gamma)
    ax.plot_surface(*np.moveaxis(lcfs, -1, 0), color=INSIDE, alpha=0.10, lw=0)
    for i in show:
        ax.plot(*xyz[i, (t <= 40) | lost[i]].T, color=OUTSIDE if lost[i] else INSIDE, lw=0.9)
        if lost[i]:
            last = np.flatnonzero(finite[i])[-1]
            ax.scatter(*xyz[i, last], color="k", s=10, depthshade=False)
    ax.set_box_aspect((1, 1, 0.35), zoom=1.45)
    ax.view_init(38, 25)
    ax.set_axis_off()

    ax = fig.add_axes([0.56, 0.15, 0.42, 0.78])
    ax.axhline(1, color="k", lw=0.8)
    ax.text(t[-1], 1, "LCFS ", ha="right", va="bottom")
    for i in show:
        ax.plot(t, np.sqrt(s[i]), color=OUTSIDE if lost[i] else INSIDE, lw=0.9)
        if lost[i]:
            ax.plot(lost_times[i] * 1e6, 1, "o", color="k", ms=3)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"time [$\mu$s]")
    ax.set_ylabel(r"$\sqrt{s}$")
    save(fig, "readme_orbits.png")


def gradients():
    from scipy.optimize import least_squares

    from essos.losses import custom_loss
    from essos.surfaces import BdotN_over_B, SurfaceRZFourier

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2))

    # (a) Fit coils to a VMEC boundary: normal field, length and curvature.
    surface = SurfaceRZFourier.from_wout_file(str(INPUTS / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"),
                                              s=1, ntheta=30, nphi=30, range_torus="half period")
    coils = Coils(curves=CreateEquallySpacedCurves(3, 3, 10.0, 5.6, n_segments=45, nfp=2, stellsym=True),
                  currents=[1.0] * 3)
    L = (custom_loss(lambda field, surface: jnp.sum(jnp.abs(BdotN_over_B(surface, field))), "field", surface=surface)
         + custom_loss(lambda field: jnp.mean(jnp.maximum(0, field.coils.length - 32.0)), "field")
         + custom_loss(lambda field: jnp.mean(jnp.maximum(0, field.coils.curvature - 0.1)), "field"))
    L.dependencies = {"field": BiotSavart(coils)}
    history = []

    def objective(x):
        value = L(x)
        history.append(float(value))
        return value

    least_squares(objective, L.starting_dofs, L.grad, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=400)
    ax1.semilogy(np.minimum.accumulate(history), color=INSIDE)
    ax1.set_xlabel("objective evaluations")
    ax1.set_ylabel("objective")
    ax1.set_title("coil fit to a QA boundary", fontsize=9)
    print(f"coil fit: {history[0]:.3g} -> {min(history):.3g} in {len(history)} evaluations")

    # (b) d/dx of the mean distance from the axis of 8 guiding centers after
    # 10 us, along a random direction in coil space: AD against finite differences.
    field = BiotSavart(Coils.from_json(str(COILS_JSON)))
    R0 = jnp.linspace(1.23, 1.28, 8)
    particles = Particles(initial_xyz=jnp.array([R0, 0 * R0, 0 * R0]).T, initial_vparallel_over_v=jnp.linspace(-0.9, 0.9, 8),
                          mass=PROTON_MASS, energy=5e3 * ONE_EV)

    def final_radius(field):
        tracing = Tracing(field=field, model="GuidingCenterAdaptative", particles=particles, maxtime=1e-5,
                          timestep=1e-9, times_to_trace=2, atol=1e-11, rtol=1e-11)
        xyz = tracing.trajectories[:, -1, :3]
        return jnp.mean(jnp.hypot(jnp.hypot(xyz[:, 0], xyz[:, 1]) - field.r_axis, xyz[:, 2] - field.z_axis))

    x, unravel = jax.flatten_util.ravel_pytree(field)
    f = lambda dofs: final_radius(unravel(dofs))
    value, gradient = jax.value_and_grad(f)(x)
    v = jax.random.normal(jax.random.key(0), x.shape)
    v = v / jnp.linalg.norm(v) * jnp.linalg.norm(x)
    exact = float(gradient @ v)
    steps = np.logspace(-10, -2, 17)
    error = [abs(float((f(x + h * v) - f(x - h * v)) / (2 * h)) / exact - 1) for h in steps]
    ax2.loglog(steps, error, "o-", color=INSIDE, ms=3)
    ax2.loglog(steps[10:], error[-1] * (steps[10:] / steps[-1])**2, color="k", lw=0.7, ls="--")
    ax2.text(steps[13], error[-1] * (steps[13] / steps[-1])**2 / 6, r"$h^2$")
    ax2.set_xlabel("finite-difference step $h$ (relative)")
    ax2.set_ylabel("|FD / AD - 1|")
    ax2.set_title(f"gradient through orbits, {x.size} coil DOFs", fontsize=9)
    print(f"orbit gradient: best |FD/AD - 1| = {min(error):.1e}")
    fig.tight_layout()
    save(fig, "readme_gradients.png")


PANELS = {"orbits": orbits, "gradients": gradients}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        PANELS[sys.argv[1]]()
    else:  # one panel per process, so that the jitted kernels do not share a cache
        for name in PANELS:
            subprocess.run([sys.executable, __file__, name], check=True)
