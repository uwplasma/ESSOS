"""Analytic OU and Maxwell controls for transient Coulomb references."""
import jax.numpy as jnp
import numpy as np
import pytest

from examples.benchmarks.thermal_relaxation_reference import RelaxationReference, PhysicalRelaxationReference, coefficients
from examples.benchmarks.validate_collision_physics import constant_log_coefficients


def test_independent_incomplete_gamma_coefficients_match_native_constant_log_control():
    for r in (.071, .2, 1., 2., np.sqrt(10.), 7.):
        dp, dt, nu = coefficients(r)
        native = constant_log_coefficients(jnp.asarray(r))
        assert dp == pytest.approx(float(native.parallel), rel=3e-12)
        assert dt == pytest.approx(float(native.perpendicular), rel=3e-12)
        assert nu == pytest.approx(2*dt/r**2)


def test_legendre_generator_matches_exact_cartesian_ornstein_uhlenbeck_relaxation():
    def isotropic(r):
        return np.ones_like(r), np.ones_like(r), 2/r**2
    ref = RelaxationReference(nodes=768, law=isotropic)
    times = np.array([.1, .5, 1., 3.])
    values = ref.moments(times)
    # dV=-V dt+sqrt(2)dW: exact Cartesian transition is Gaussian.
    np.testing.assert_allclose(values["speed2"], 3+7*np.exp(-2*times), atol=9e-5, rtol=0)
    np.testing.assert_allclose(values["parallel_mean"], ref.start*ref.pitch*np.exp(-times), atol=3e-5, rtol=0)
    np.testing.assert_allclose(values["parallel_variance"], 1-np.exp(-2*times), atol=5e-5, rtol=0)
    np.testing.assert_allclose(values["spatial_integral"], 4*times, atol=1e-10, rtol=0)


def test_reflecting_generator_conserves_probability_and_relaxes_to_maxwell():
    ref = RelaxationReference(nodes=384)
    np.testing.assert_allclose(ref.expectation([0., 1., 20.], 1.), 1., atol=3e-11)
    x = np.linspace(0., 8., 201)
    cdf = ref.speed_cdf(20.)(x)
    assert cdf[0] == 0 and cdf[-1] == 1
    assert np.min(np.diff(cdf)) >= 0
    values = ref.moments([200.])
    equilibrium = ref.equilibrium()
    for name in equilibrium:
        assert values[name][0] == pytest.approx(equilibrium[name], abs=2e-4)


def test_monoenergetic_initial_moments_are_exact_and_distinct_from_equilibrium():
    ref = RelaxationReference(nodes=64)
    values = ref.moments([0.])
    assert values["speed_variance"][0] == 0
    assert values["pitch_variance"][0] == 0
    assert values["speed2"][0] == pytest.approx(10.)
    assert values["parallel_mean"][0] == pytest.approx(np.sqrt(10.)*.6)
    assert values["spatial_integral"][0] == 0


def test_production_physical_coefficients_match_independent_formulas_over_the_thermal_range():
    from essos.background_species import BackgroundSpecies
    from essos.collisions import collision_coefficients
    from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
    from examples.benchmarks.nonrelativistic_collision_reference import ProtonBackground

    physical = ProtonBackground()
    species = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([physical.density]), jnp.array([physical.temperature_ev]))
    for r in (.071, .2, .5, 1., np.sqrt(3.)*(1-1e-7), np.sqrt(3.)*(1+1e-7), 3., 6.):
        v = physical.sigma*r
        independent = physical.coefficients(v)
        actual = collision_coefficients(species, PROTON_MASS, ELEMENTARY_CHARGE,
            jnp.asarray(v), jnp.zeros(3))
        for name in ("parallel", "perpendicular", "friction", "parallel_derivative"):
            assert float(getattr(actual, name)) == pytest.approx(float(independent[name]), rel=3e-9)


def test_physical_reference_resolves_the_coulomb_log_interface_and_its_stationary_density():
    ref = PhysicalRelaxationReference(nodes=1024)
    values = ref.moments([200.])
    exact = ref.equilibrium()
    np.testing.assert_allclose(ref.expectation([0., 1., 20.], 1.), 1., atol=3e-10)
    for name, expected in exact.items():
        assert values[name][0] == pytest.approx(expected, abs=2e-4)


def test_impact_coefficients_match_independent_algebra_and_relax_to_exact_maxwell():
    from essos.background_species import BackgroundSpecies
    from essos.collisions import collision_coefficients
    from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
    from examples.benchmarks.nonrelativistic_collision_reference import ProtonBackground, proton_first_passage, quadrature_mean
    from examples.benchmarks.thermal_relaxation_reference import impact_coefficients

    physical = ProtonBackground(coulomb_log_model="impact")
    species = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([physical.density]), jnp.array([physical.temperature_ev]), coulomb_log_model="impact")
    for r in (.071, .2, 1., np.sqrt(3.)*(1-1e-7), np.sqrt(3.)*(1+1e-7), 3., 6.):
        independent = physical.coefficients(physical.sigma*r)
        actual = collision_coefficients(species, PROTON_MASS, ELEMENTARY_CHARGE,
            jnp.asarray(physical.sigma*r), jnp.zeros(3))
        for name in ("parallel", "perpendicular", "friction", "parallel_derivative"):
            assert float(getattr(actual, name)) == pytest.approx(float(independent[name]), rel=3e-9)
    ref = RelaxationReference(nodes=512, law=impact_coefficients)
    values = ref.moments([200.])
    for name, expected in ref.equilibrium().items():
        assert values[name][0] == pytest.approx(expected, abs=3e-5)
    np.testing.assert_allclose(ref.expectation([0., 1., 20.], 1.), 1., atol=3e-10)
    # The smooth model permits passage across the former E/T=1.5 switch.
    passage = proton_first_passage(physical, threshold_energy_over_T=.5)
    assert passage.mean == pytest.approx(quadrature_mean(physical, passage.boundary, passage.start), rel=8e-5)
