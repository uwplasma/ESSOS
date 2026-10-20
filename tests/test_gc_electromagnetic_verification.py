"""Independent conservation laws and an exact inductive trajectory."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.benchmarks.alfven_gc_verification import (
    MASS, CHARGE, SPEED, PHASES, UniformInduction, differential_checks,
    equilibrium, phase_space_density, spectrum, trace_induction, trace_wave,
)
from essos.boozer import guiding_center_rhs


@pytest.mark.parametrize("phase", PHASES)
def test_two_harmonic_wave_preserves_energy_momentum_and_phase_space(phase):
    field, wave = equilibrium(), spectrum(phase)
    points = jnp.array([[.45, .27, -.3, .7*SPEED], [.3, -.4, .7, -.65*SPEED]])
    times = jnp.array([1.3e-6, 4.7e-6])
    checks = np.asarray(jax.vmap(lambda y, t: differential_checks(
        field, wave, y, .5*SPEED**2/(2*5.), t))(points, times))
    assert np.max(abs(checks[:, 0])) < 1e-12
    assert np.max(abs(checks[:, 1])) < 1e-12
    # These laws must detect an omitted inductive force, not merely return zero.
    assert np.max(abs(checks[:, 2])) > 1e-6
    assert np.max(abs(checks[:, 3])) > 1e-10


def test_volume_density_has_a_nonzero_time_derivative():
    field, wave = equilibrium(), spectrum(np.pi/2)
    y = jnp.array([.45, .27, -.3, .7*SPEED])
    source = jax.grad(lambda t: phase_space_density(field, wave, y, t))(1.3e-6)
    assert abs(float(source)) > 1e-3
    # In u,w the volume density has no polar radial factor: it stays nonzero
    # as the axis is approached in the smooth, m>=3 perturbation fixture.
    densities = [float(phase_space_density(field, wave, y.at[:2].set(jnp.array([r, 0.])), 0.))
                 for r in (.01, .001)]
    assert min(densities) > 1.
    assert densities[0] == pytest.approx(densities[1], rel=.01)


def test_uniform_inductive_force_has_the_exact_sign_and_normalization():
    field, wave = equilibrium(uniform=True), UniformInduction()
    y = jnp.array([.45, .27, -.3, .7*SPEED])
    t = 1.3e-6
    dy = guiding_center_rhs(field, y, 1e12, MASS, CHARGE, t=t, perturbation=wave)
    expected = -CHARGE*5./MASS*wave.amplitude*wave.omega*np.cos(wave.omega*t+wave.phase)
    np.testing.assert_allclose(dy[:2], 0., atol=1e-12)
    assert float(dy[2]) == pytest.approx(5./40.*y[3], rel=1e-13)
    assert float(dy[3]) == pytest.approx(expected, rel=1e-13)


def test_adaptive_inductive_trajectory_matches_exact_velocity_and_position():
    trace, exact_v, exact_zeta = trace_induction(1e-10, tmax=1.5e-5, n_save=17)
    assert np.max(abs(trace.states[0, :, 3]-exact_v))/SPEED < 1e-8
    assert np.max(abs(trace.states[0, :, 2]-exact_zeta)) < 1e-7
    np.testing.assert_allclose(trace.states[0, :, 0], .3, atol=1e-13)
    np.testing.assert_allclose(trace.states[0, :, 1], .2, atol=1e-13)


def test_adaptive_two_harmonic_trajectory_exchanges_energy_at_fixed_wave_invariant():
    _, H, _, residual = trace_wave(equilibrium(), spectrum(np.pi/2), 1e-10,
                                  tmax=1.5e-5, n_save=17)
    assert np.max(abs(residual)) < 1e-8
    assert np.max(abs((H-H[:, :1])/H[:, :1])) > 1e-5
