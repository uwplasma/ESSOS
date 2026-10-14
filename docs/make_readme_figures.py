"""Regenerate the README figures.

Each panel is the shortest useful form of one workflow, and the README quotes
these same calls, so the pictures and the snippets cannot drift apart.

    python docs/make_readme_figures.py            # every panel
    python docs/make_readme_figures.py fieldlines # just one

"""

import os
from pathlib import Path


import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, ONE_EV
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "examples" / "input_files"
COILS_JSON = INPUTS / "ESSOS_biot_savart_LandremanPaulQA.json"
DPI = 110


def coil_optimization() -> None:
    """Fit coils to a VMEC boundary: normal field, length and curvature."""
    from scipy.optimize import least_squares

    from essos.losses import custom_loss
    from essos.surfaces import BdotN_over_B, SurfaceRZFourier

    surface = SurfaceRZFourier.from_wout_file(
        str(INPUTS / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"),
        s=1, ntheta=30, nphi=30, range_torus="half period")
    init_coils = Coils(curves=CreateEquallySpacedCurves(3, 3, 10.0, 5.6, n_segments=45,
                                                        nfp=2, stellsym=True),
                       currents=[1.0] * 3)
    init_field = BiotSavart(init_coils)

    L = (custom_loss(lambda field, surface: jnp.sum(jnp.abs(BdotN_over_B(surface, field))),
                     "field", surface=surface)
         + custom_loss(lambda field: jnp.mean(jnp.maximum(0, field.coils.length - 32.0)), "field")
         + custom_loss(lambda field: jnp.mean(jnp.maximum(0, field.coils.curvature - 0.1)), "field"))
    L.dependencies = {"field": init_field}

    before = float(L(L.starting_dofs))
    res = least_squares(L, L.starting_dofs, L.grad, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=400)
    after = float(L(res.x))
    opt_coils = L.dofs_to_pytree(res.x)["field"].coils

    fig = plt.figure(figsize=(7.4, 3.8))
    for k, (title, case) in enumerate((("initial", init_coils), ("optimized", opt_coils))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        case.plot(ax=ax, show=False)
        surface.plot(ax=ax, show=False)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"coils fitted to a QA boundary: loss {before:.3g} $\\rightarrow$ {after:.3g}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "readme_coil_optimization.png", dpi=DPI)
    plt.close(fig)
    print(f"coil optimization: loss {before:.4g} -> {after:.4g}")


def field_line_tracing() -> None:
    """Poincare section of the vacuum field of a coil set."""
    field = BiotSavart(Coils.from_json(str(COILS_JSON)))
    R0 = jnp.linspace(1.21, 1.40, 8)
    zeros = jnp.zeros_like(R0)
    seeds = jnp.array([R0, zeros, zeros]).T

    tracing = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=seeds,
                      maxtime=8000, times_to_trace=40000, atol=1e-8, rtol=1e-8)

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    tracing.poincare_plot(ax=ax, show=False, shifts=[0.0])
    ax.set_title("Poincare section, coil field", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "readme_fieldlines.png", dpi=DPI)
    plt.close(fig)
    print("field lines: wrote Poincare section")


def particle_tracing() -> None:
    """Guiding-centre alpha orbits in the same coil field."""
    field = BiotSavart(Coils.from_json(str(COILS_JSON)))
    R0 = jnp.linspace(1.23, 1.27, 4)
    zeros = jnp.zeros_like(R0)
    particles = Particles(initial_xyz=jnp.array([R0, zeros, zeros]).T,
                          mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                          energy=4000 * ONE_EV)

    tracing = Tracing(field=field, model="GuidingCenterAdaptative", particles=particles,
                      maxtime=1e-4, times_to_trace=800, atol=1e-7, rtol=1e-7)

    fig = plt.figure(figsize=(5.0, 4.0))
    ax = fig.add_subplot(projection="3d")
    tracing.plot(ax=ax, show=False)
    ax.set_title("guiding-centre alpha orbits, 4 keV", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "readme_particles.png", dpi=DPI)
    plt.close(fig)
    print("particles: wrote guiding-centre orbits")


def wall_tracing() -> None:
    """Guiding centres from a VMEX equilibrium through the LCFS to a wall (needs vmex)."""
    import dataclasses

    import jax
    import numpy as np
    import vmex
    from essos.constants import ELEMENTARY_CHARGE, PROTON_MASS
    from essos.surfaces import SurfaceRZFourier

    coils = Coils.from_json(str(COILS_JSON))
    grid = vmex.MgridField.from_coils(coils, rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6,
                                      ir=96, jz=96, kp=32)
    inp = vmex.VmecInput.from_file(str(INPUTS / "input.LandremanPaul2021_QA_reactorScale_lowres"))
    inp = dataclasses.replace(inp, rbc=inp.rbc / 10.127, zbs=inp.zbs / 10.127, ns_array=[16],
                              niter_array=[4000], ftol_array=[1e-10], lfreeb=True,
                              mgrid_file="essos_coils(direct)", nzeta=16, nvacskip=6, phiedge=-0.025)
    res = vmex.solve_free_boundary_multigrid(inp, external_field=grid, raise_on_max_iterations=False)
    wout = vmex.wout_from_state(inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
                                fsql=float(res.fsql), niter=int(res.iterations),
                                converged=bool(res.converged), vacuum_output=res.vacuum)

    gap = WALL_GAP
    vmec = vmex.essos_vmec_field(wout)
    outside = vmex.VmecExtender.from_wout(wout, external_field=dataclasses.replace(grid, order=3),
                                          plasma="vacuum")
    n = WALL_N
    ks, kt, kp, kv = jax.random.split(jax.random.key(WALL_SEED), 4)
    seeds = jnp.stack([jax.random.uniform(ks, (n,), minval=0.7, maxval=0.95),
                       jax.random.uniform(kt, (n,), maxval=2 * jnp.pi),
                       jax.random.uniform(kp, (n,), maxval=2 * jnp.pi)], axis=1)
    particles = Particles(initial_xyz=seeds, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
                          energy=WALL_ENERGY_EV * ONE_EV,
                          initial_vparallel_over_v=jax.random.uniform(kv, (n,), minval=-1, maxval=1))
    tracing = Tracing(field=vmec, model="GuidingCenterAdaptative", particles=particles,
                      maxtime=4e-5, timestep=1e-9, times_to_trace=2000, atol=1e-8, rtol=1e-8,
                      exterior_field=outside, wall=gap)

    crossed = np.isfinite(np.asarray(tracing.lcfs_times))
    struck = np.asarray(tracing.wall_hits)
    fates = {"stay inside": ~crossed,
             "cross and return": crossed & ~struck & (np.asarray(tracing.returns) > 0),
             "strike the wall": struck}
    print("wall:", {k: int(v.sum()) for k, v in fates.items()}, "of", n)
    colors = {"stay inside": "#3b6fb6", "cross and return": "#e08a1e", "strike the wall": "#c0392b"}

    lcfs = SurfaceRZFourier.from_vmec(vmec, ntheta=48, nphi=96)
    wall = SurfaceRZFourier.from_vmec(vmec, ntheta=48, nphi=96, offset=gap)
    xyz = np.asarray(tracing.trajectories_xyz)
    hits = np.asarray(tracing.wall_positions)

    fig = plt.figure(figsize=(10.0, 4.6))
    grid_spec = fig.add_gridspec(1, 2, width_ratios=(1.5, 1.0), wspace=0.02)
    ax = fig.add_subplot(grid_spec[0], projection="3d")
    g = np.asarray(lcfs.gamma)
    ax.plot_surface(g[..., 0], g[..., 1], g[..., 2], color="0.75", alpha=0.25, linewidth=0)
    for label, mask in fates.items():
        for i in np.flatnonzero(mask)[:WALL_SHOW]:
            ax.plot(*xyz[i].T, lw=1.0, color=colors[label])
    ax.scatter(*hits[struck].T, color="k", marker="x", s=22, depthshade=False, label="wall strikes")
    lim = np.abs(g[..., :2]).max()
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-0.5 * lim, 0.5 * lim))
    ax.set_box_aspect((1, 1, 0.5), zoom=1.45)
    ax.set_axis_off()
    ax.view_init(elev=38, azim=-60)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = fig.add_subplot(grid_spec[1])
    for surface, style, label in ((lcfs, "-", "LCFS"), (wall, "--", f"wall, {gap * 100:.0f} cm out")):
        c = np.asarray(surface.gamma)[0]
        c = np.vstack([c, c[:1]])
        ax.plot(np.hypot(c[:, 0], c[:, 1]), c[:, 2], "k" + style, lw=1, label=label)
    period = 2 * np.pi / vmec.nfp
    phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    near = np.abs(np.mod(phi + period / 2, period) - period / 2) < 0.03
    R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
    for label, mask in fates.items():
        sel = near & mask[:, None]
        ax.scatter(R[sel], Z[sel], s=1.5, color=colors[label], label=f"{label} ({mask.sum()})")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.set_title(f"{n} protons, {WALL_ENERGY_EV / 1e3:.0f} keV, 40 $\\mu$s; points near $\\phi = 0$",
                 fontsize=9)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=5, frameon=False)
    fig.subplots_adjust(left=0.0, right=0.84, top=0.92, bottom=0.11)
    fig.savefig(HERE / "readme_wall.png", dpi=DPI)
    plt.close(fig)
    print("wall: wrote tracing to the wall")


WALL_GAP, WALL_N, WALL_SEED, WALL_ENERGY_EV, WALL_SHOW = 0.03, 64, 0, 3e3, 5


PANELS = {"coils": coil_optimization,
          "fieldlines": field_line_tracing,
          "particles": particle_tracing,
          "wall": wall_tracing}

if __name__ == "__main__":
    import subprocess
    import sys

    if len(sys.argv) > 1:
        # One panel per process: ESSOS caches jitted field kernels on the Coils
        # pytree, and two tracings of different models in one process trip the
        # cache's metadata equality check.
        PANELS[sys.argv[1]]()
    else:
        for name in PANELS:
            subprocess.run([sys.executable, __file__, name], check=True)
