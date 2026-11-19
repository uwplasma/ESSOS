"""Importing ESSOS leaves JAX configuration alone; float64 is the caller's choice."""

import os
import subprocess
import sys

import pytest

MODULES = ["essos", "essos.augmented_lagrangian", "essos.background_species", "essos.boozer",
           "essos.coil_perturbation", "essos.coils", "essos.constants", "essos.dynamics",
           "essos.electric_field", "essos.fields", "essos.losses", "essos.mgrid",
           "essos.objective_functions", "essos.optimization", "essos.plot", "essos.surfaces",
           "essos.util"]
CLEAN = {k: v for k, v in os.environ.items() if not k.startswith(("JAX_", "XLA_FLAGS", "ESSOS_"))}


def run(code):
    out = subprocess.run([sys.executable, "-c", code], env=CLEAN, capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.parametrize("module", MODULES)
def test_import_leaves_every_jax_flag_unchanged(module):
    changed = run(f"""
import jax, importlib
before = dict(jax.config.values)
importlib.import_module({module!r})
print(sorted(k for k, v in jax.config.values.items() if before.get(k, object()) != v))
""")
    assert changed.split()[-1] == "[]", changed


def test_entry_points_name_the_float64_setting():
    out = run("""
import numpy as np
from essos.boozer import BoozerField
from essos.coils import CreateEquallySpacedCurves
from essos.dynamics import Tracing
from essos.fields import Vmec
s = np.linspace(0.1, 0.9, 5)
for build in (lambda: CreateEquallySpacedCurves(2, 1, 1.0, 0.3), lambda: Vmec("wout.nc"), lambda: Tracing(),
              lambda: BoozerField.from_booz(s, np.ones((1, 5)), [0], [0], s, s, s, 1.0, 1)):
    try:
        build()
    except RuntimeError as error:
        print('jax.config.update("jax_enable_x64", True)' in str(error))
""")
    assert out.split() == ["True"] * 4, out


def test_configure_jax_enables_float64_and_cpu_devices():
    out = run("""
import essos, jax
essos.configure_jax(cpu_devices=3)
print(jax.config.jax_enable_x64, len(jax.devices("cpu")))
""")
    assert out.split() == ["True", "3"], out


def test_boozer_trace_runs_in_float32():
    """Four alphas over 1e-4 s in float32 keep the energy error below 1e-5.

    On 2026-11-19, with float64 switched off after import and the field cast to float32, the
    errors were 1.1e-6 to 2.9e-6; the bound leaves room for a different field and starting points.
    """
    out = run("""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from essos.boozer import BoozerField, trace_boozer
from essos.constants import ALPHA_PARTICLE_CHARGE as Q, ALPHA_PARTICLE_MASS as M, FUSION_ALPHA_PARTICLE_ENERGY
s = np.linspace(0.005, 0.995, 50)
bmnc = np.stack([np.full_like(s, 5.0), -1.25 * np.sqrt(s)])
field = BoozerField.from_booz(s, bmnc, [0, 1], [0, 0], np.full_like(s, 0.4), np.full_like(s, 40.0),
                              np.zeros_like(s), 40.0, 1, angle_grid=None)
jax.config.update("jax_enable_x64", False)
field = jax.tree.map(lambda x: jnp.asarray(x, jnp.float32) if jnp.issubdtype(jnp.result_type(x), jnp.floating) else x, field)
out = trace_boozer(field, [0.3] * 4, [0.0, 1.5, 3.0, 4.5], [0.0] * 4, [-0.6, -0.2, 0.2, 0.6],
                   speed=float(np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / M)), mass=M, charge=Q, tmax=1e-4,
                   n_save=5, devices=jax.devices()[:1], progress=False)
print(out.states.dtype, int(out.lost.sum()), float(np.max(out.energy_error)))
""")
    dtype, lost, error = out.split()[-3:]
    assert dtype == "float32" and lost == "0" and float(error) < 1e-5, out
