"""Regenerate the README figures.

Each panel is the shortest useful form of one workflow, and the README quotes
these same calls, so the pictures and the snippets cannot drift apart.

    python docs/make_readme_figures.py            # every panel
    python docs/make_readme_figures.py fieldlines # just one

"""

import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

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


PANELS = {"coils": coil_optimization,
          "fieldlines": field_line_tracing,
          "particles": particle_tracing}

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
