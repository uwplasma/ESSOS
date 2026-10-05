"""A repeated trace must reuse the compiled solve instead of recompiling it."""
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coils import Coils
from essos.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart

COILS_JSON = (Path(__file__).resolve().parents[1] / "examples" / "input_files"
              / "ESSOS_biot_savart_LandremanPaulQA.json")


@pytest.fixture(scope="module")
def field():
    return BiotSavart(Coils.from_json(str(COILS_JSON)))


def _protons(n=4):
    xyz = jnp.stack([jnp.linspace(1.20, 1.26, n), jnp.zeros(n), jnp.zeros(n)], axis=1)
    return Particles(initial_xyz=xyz, initial_vparallel_over_v=jnp.linspace(-0.5, 0.5, n),
                     charge=ELEMENTARY_CHARGE, mass=PROTON_MASS, energy=5000 * ONE_EV)


def _trace(field, maxtime=1e-6, particles=None):
    return Tracing(field=field, model="GuidingCenterAdaptative",
                   particles=_protons() if particles is None else particles,
                   maxtime=maxtime, timestep=1e-8, times_to_trace=10, rtol=1e-8, atol=1e-8)


def _compilations(caplog, run):
    """Run ``run`` and return its result with every jit it compiled."""
    caplog.clear()
    with caplog.at_level(logging.WARNING), jax.log_compiles(True):
        result = run()
    return result, [record.getMessage() for record in caplog.records
                    if record.getMessage().startswith("Compiling ")]


def test_identical_tracing_reuses_the_compiled_solve(field, caplog):
    first = _trace(field)
    # A new Tracing with new but identical Particles, and a new default
    # electric field, must hit the cache of the first.
    second, compiled = _compilations(caplog, lambda: _trace(field))
    assert compiled == []
    np.testing.assert_array_equal(np.asarray(first.trajectories), np.asarray(second.trajectories))


def test_changing_maxtime_or_coil_values_does_not_recompile(field, caplog):
    _trace(field)
    shifted = jax.tree_util.tree_map(lambda leaf: leaf * 1.001, field)
    _, compiled = _compilations(
        caplog, lambda: (_trace(field, maxtime=2e-6), _trace(shifted)))
    # Scaling a field value may compile a one-op eager kernel; the solve
    # itself must not be recompiled.
    assert not [message for message in compiled if "jit(_trace_batch)" in message]


def test_gradient_through_tracing_matches_finite_differences(field):
    particles = _protons(2)

    def final_radius(scale):
        scaled = jax.tree_util.tree_map(lambda leaf: leaf * scale, field)
        xyz = _trace(scaled, particles=particles).trajectories[:, -1, :3]
        return jnp.sum(jnp.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2))

    gradient = jax.grad(final_radius)(1.0)
    step = 1e-4
    finite_difference = (final_radius(1.0 + step) - final_radius(1.0 - step)) / (2 * step)
    assert jnp.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-3)


def test_vmec_exterior_tracing_reuses_the_compiled_solve(caplog):
    """The exterior field is wrapped anew by every Tracing and must still hit the cache."""
    from essos.fields import Vmec

    wout = COILS_JSON.parent / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
    vmec = Vmec(str(wout), ntheta=8, nphi=8)

    def toroidal(xyz):
        R2 = xyz[:, 0]**2 + xyz[:, 1]**2
        return 60.0 * jnp.stack([-xyz[:, 1] / R2, xyz[:, 0] / R2, 0 * R2], 1)

    def wall(x):
        return vmec.boundary_distance(x) + 0.05

    def trace():
        particles = Particles(initial_xyz=jnp.array([[0.9732, 2.2941, 5.0876], [0.9821, 0.6628, 3.5216]]),
                              initial_vparallel_over_v=jnp.array([-0.9872, 0.5453]))
        return Tracing(field=vmec, model="GuidingCenterAdaptative", particles=particles, maxtime=1e-7,
                       timestep=1e-9, times_to_trace=5, exterior_field=toroidal, wall=wall)

    first = trace()
    second, compiled = _compilations(caplog, trace)
    assert not [message for message in compiled if "jit(_trace_batch)" in message]
    np.testing.assert_array_equal(first.trajectories, second.trajectories)
