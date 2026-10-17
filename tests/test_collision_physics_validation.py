"""Exact controls for the physical validation's independent observables."""
import jax.numpy as jnp
import numpy as np
import pytest

from examples.benchmarks.validate_collision_physics import constant_log_coefficients, equilibrium_reference
from examples.benchmarks.nonrelativistic_collision_reference import first_passage_reference, maxwell_weight
from examples.benchmarks.spatial_collision_reference import stopped_spatial_moments


def test_constant_log_control_has_maxwell_detailed_balance_and_exact_equilibrium_diffusion():
    ref = equilibrium_reference(0.)
    assert ref["speed_mean"] == pytest.approx(2*np.sqrt(2/np.pi), rel=1e-10)
    assert ref["speed2"] == pytest.approx(3., rel=1e-10)
    from essos.background_species import chandrasekhar
    expected = 1/(3*np.sqrt(np.pi)*float(chandrasekhar(jnp.asarray(1/np.sqrt(2.)))))
    assert ref["spatial_diffusivity"] == pytest.approx(expected, rel=1e-10)
    for v in (.1, 1., 2., 5.):
        c = constant_log_coefficients(jnp.asarray(v))
        assert float(c.friction/c.parallel) == pytest.approx(-v, rel=1e-12)
        derivative = (float(constant_log_coefficients(jnp.asarray(v+1e-5)).parallel)
                     - float(constant_log_coefficients(jnp.asarray(v-1e-5)).parallel))/(2e-5)
        assert float(c.parallel_derivative) == pytest.approx(derivative, rel=2e-7)


def test_spatial_second_reward_moment_reproduces_exact_residence_time_moment():
    class ConstantDiffusion:
        sigma = nu = 1.
        def diffusion(self, r):
            return np.full_like(r, .25)
        def coefficients(self, r):
            return dict(parallel=np.full_like(r, .25), perpendicular=np.full_like(r, .25))
    # D_X=.25 is constant, Q=4*.25*min(tau,H)=min(tau,H).
    case = ConstantDiffusion()
    times = np.array([.1, .5, 2.])
    ref = first_passage_reference(case.diffusion, maxwell_weight, boundary=2., start=np.sqrt(10.), nodes=384)
    result = stopped_spatial_moments(case, times, nodes=384)
    np.testing.assert_allclose(result["mean"], [ref.restricted_moments(t)[0] for t in times], atol=2e-8)
    np.testing.assert_allclose(result["second_moment"], [ref.restricted_moments(t)[1] for t in times], atol=2e-8)
