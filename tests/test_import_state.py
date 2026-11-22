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
# Dependencies set their own warning filters and environment variables when imported; they are imported
# before the snapshot so that the test sees what ESSOS itself changes.
DEPENDENCIES = ["numpy", "scipy.interpolate", "scipy.ndimage", "scipy.optimize", "matplotlib.pyplot", "jaxtyping",
                "equinox", "diffrax", "optimistix", "optax", "jaxkd", "netCDF4", "h5py"]
CLEAN = {k: v for k, v in os.environ.items() if not k.startswith(("JAX_", "XLA_FLAGS", "ESSOS_"))}


def run(code):
    out = subprocess.run([sys.executable, "-c", code], env=CLEAN, capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.parametrize("module", MODULES)
def test_import_leaves_jax_environment_and_warnings_unchanged(module):
    """Import changes no JAX flag, environment variable or warning filter, and starts no JAX backend."""
    changed = run(f"""
import importlib, os, warnings, jax
for name in {DEPENDENCIES!r}:
    try:
        importlib.import_module(name)
    except ImportError:
        pass
from jax._src import xla_bridge
started = bool(xla_bridge._backends)  # diffrax 0.7.0 (Python 3.10) starts the CPU backend on import
config, environ, filters = dict(jax.config.values), dict(os.environ), list(warnings.filters)
importlib.import_module({module!r})
changed = sorted(k for k, v in jax.config.values.items() if config.get(k, object()) != v)
changed += sorted(set(os.environ.items()) ^ set(environ.items())) + (["warnings"] if warnings.filters != filters else [])
if not started:
    try:  # refused once a backend has started
        jax.config.update("jax_num_cpu_devices", 2)
    except RuntimeError:
        changed.append("backend started")
print(changed)
""")
    assert changed.strip().splitlines()[-1] == "[]", changed


def test_entry_points_name_the_float64_setting():
    out = run("""
import numpy as np
from essos.background_species import nu_D_ab, nu_s_ab
from essos.boozer import BoozerField, trace_boozer
from essos.coils import CreateEquallySpacedCurves
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart_from_gamma, Vmec
from essos.surfaces import SurfaceRZFourier
s = np.linspace(0.1, 0.9, 5)
for build in (lambda: CreateEquallySpacedCurves(2, 1, 1.0, 0.3), lambda: Vmec("wout.nc"), lambda: Tracing(),
              lambda: BoozerField.from_booz(s, np.ones((1, 5)), [0], [0], s, s, s, 1.0, 1),
              lambda: BiotSavart_from_gamma(np.zeros((1, 8, 3))), lambda: SurfaceRZFourier(s, s, 1, 1, 1),
              lambda: Particles(np.zeros((1, 3))), lambda: nu_D_ab(1.0, 1.0, 0, 1.0, s, None),
              lambda: nu_s_ab(1.0, 1.0, 0, 1.0, s, None),
              lambda: trace_boozer(None, [.3], [0.], [0.], [0.], speed=1., mass=1., charge=1., tmax=1., species="e")):
    try:
        build()
        print("no-error")
    except RuntimeError as error:
        print('jax.config.update("jax_enable_x64", True)' in str(error))
""")
    assert out.split() == ["True"] * 10, out


def test_configure_jax_enables_float64_and_cpu_devices():
    out = run("""
import essos, jax
essos.configure_jax(cpu_devices=3)
print(jax.config.jax_enable_x64, len(jax.devices("cpu")))
""")
    assert out.split() == ["True", "3"], out


DIFFRAX_STARTS_BACKEND = sys.version_info < (3, 11)  # resolves diffrax 0.7.0, whose import starts JAX


@pytest.mark.parametrize("module", ["essos.surfaces", "essos.fields", "essos.dynamics"])
def test_configure_jax_after_import_sets_the_device_count(module):
    """After importing essos.dynamics on diffrax 0.7.0 the count can no longer be set, and configure_jax says so."""
    out = run(f"""
import importlib, essos, jax
importlib.import_module({module!r})
try:
    essos.configure_jax(cpu_devices=3)
    print(len(jax.devices("cpu")))
except RuntimeError as error:
    print("refused" if "before the first JAX array" in str(error) else error)
""")
    expected = "refused" if DIFFRAX_STARTS_BACKEND and module == "essos.dynamics" else "3"
    assert out.split()[-1] == expected, out


def test_configure_jax_refuses_a_device_count_it_cannot_set():
    out = run("""
import essos, jax
jax.devices()
try:
    essos.configure_jax(cpu_devices=3)
except RuntimeError as error:
    print("jax_num_cpu_devices" in str(error))
""")
    assert out.split() == ["True"], out


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


@pytest.mark.parametrize("jaxlib_version, explicit, cached", [
    ("0.6.2", False, False), ("0.9.2", False, False), ("0.10.0rc1", False, True), ("0.11.1", False, True),
    ("0.6.2", True, True), ("unknown", False, False)])
def test_configure_jax_uses_the_compilation_cache_only_where_jaxlib_can_reload_it(tmp_path, jaxlib_version, explicit, cached):
    """jaxlib < 0.10 overflows a thread stack reloading a large cached XLA:CPU executable on a warm rerun."""
    out = run(f"""
import os, jaxlib.version, essos, jax
jaxlib.version.__version__ = {jaxlib_version!r}
os.environ["ESSOS_XLA_CACHE"] = {str(tmp_path)!r}
essos.configure_jax(cache_dir={str(tmp_path)!r} if {explicit} else None)
print(jax.config.jax_compilation_cache_dir == {str(tmp_path)!r})
""")
    assert out.split() == [str(cached)], out
