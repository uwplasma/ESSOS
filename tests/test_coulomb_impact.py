"""Physical cutoff, regularity, and detailed-balance checks for the opt-in log."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.background_species import (
    BackgroundSpecies, coulomb_logarithm, d_nu_D_ab, d_nu_par_ab, nu_D_ab, nu_par_ab,
)
from essos.collisions import collision_coefficients
from essos.constants import ELEMENTARY_CHARGE as E, PROTON_MASS as M, ELECTRON_MASS, EPSILON_0, HBAR
from essos.coulomb import debye_inverse_squared, impact_parameters, impact_coulomb_logarithm


def bath(*, model="impact", density=1e20, temperature=1000., screening_species=None):
    return BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([density]), jnp.array([temperature]), coulomb_log_model=model,
        screening_species=screening_species)


def test_model_selection_is_explicit_and_default_remains_nrl():
    default = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([1e20]), jnp.array([1000.]))
    assert default.coulomb_log_model == "nrl"
    for v in (1e4, 1e6):
        assert coulomb_logarithm(M, E, 0, v, jnp.zeros(3), default) == coulomb_logarithm(M, E, 0, v, jnp.zeros(3), bath(model="nrl"))
    with pytest.raises(ValueError, match="coulomb_log_model"):
        bath(model="unknown")
    with pytest.raises(ValueError, match="screening_species"):
        bath(model="nrl", screening_species=bath())


@pytest.mark.parametrize("ma,qa,mb,qb", [(M, E, M, E), (4*M, 2*E, 12*M, 6*E), (ELECTRON_MASS, -E, M, E)])
def test_cutoffs_have_correct_charge_product_reduced_mass_and_asymptotes(ma, qa, mb, qb):
    # At zero background temperature the specified effective speed is the
    # actual relative speed. Classical b90 gives a Rutherford angle of pi/2.
    for speed in (1e3, 1e5, 1e7):
        reduced_mass = 1/(1/ma+1/mb)
        b90 = abs(qa*qb)/(4*np.pi*EPSILON_0*reduced_mass*speed**2)
        quantum = HBAR/(2*reduced_mass*speed)
        bmin, bmax = impact_parameters(ma, qa, mb, qb, speed, 0., 1e-4)
        assert bmax == 1e-4
        assert float(bmin)**2 == pytest.approx(b90*b90+quantum*quantum, rel=2e-14)
        swapped, _ = impact_parameters(mb, qb, ma, qa, speed, 0., 1e-4)
        assert float(swapped) == pytest.approx(float(bmin), rel=2e-14)
        if speed == 1e3:
            assert float(bmin) == pytest.approx(b90, rel=1e-7)


def test_static_screening_and_separate_electron_screeners_follow_local_profiles():
    grid = jnp.array([0., 1.])
    electron = BackgroundSpecies(1, jnp.array([ELECTRON_MASS/M]), jnp.array([-1.]),
        jnp.array([[1e20, 2e20]]), jnp.array([[1000., 2000.]]), radial_grid=grid)
    species = bath(screening_species=electron)
    for s in (0., .3, 1.):
        point = jnp.array([s, 0., 0.])
        n, temperature = 1e20*(1+s), 1000.*(1+s)
        expected_length = np.sqrt(EPSILON_0*temperature*E/(n*E**2))
        bmin, _ = impact_parameters(M, E, M, E, 2e5, 1000., expected_length)
        assert float(coulomb_logarithm(M, E, 0, 2e5, point, species)) == pytest.approx(
            np.log(expected_length/float(bmin)), rel=2e-14)
    # Equal-temperature electron and ion screening halves lambda_D squared.
    inverse = debye_inverse_squared(jnp.array([1e20, 1e20]), jnp.array([1000., 1000.]), jnp.array([-E, E]))
    assert float(inverse) == pytest.approx(2e20*E/(EPSILON_0*1000.), rel=2e-14)
    # A missing zero-temperature component contributes nothing.
    assert float(debye_inverse_squared(jnp.array([1e20, 0.]), jnp.array([1000., 0.]), jnp.array([E, -E]))) == pytest.approx(float(inverse)/2)


def test_rms_speed_makes_origin_regular_and_removes_old_branch_jump():
    species = bath()
    logarithm = lambda v: coulomb_logarithm(M, E, 0, v, jnp.zeros(3), species)
    assert np.isfinite(float(logarithm(0.)))
    assert float(jax.grad(logarithm)(0.)) == 0.
    assert np.isfinite(float(jax.grad(jax.grad(logarithm))(0.)))
    switch = np.sqrt(3*1000.*E/M)
    crossover = np.sqrt((2*E**2/(4*np.pi*EPSILON_0*HBAR))**2-3*1000.*E/M)
    for v in (1e3, switch, crossover, 1e7):
        squared = v*v+3*1000.*E/M
        reduced_mass = M/2
        classical = E**2/(4*np.pi*EPSILON_0*reduced_mass*squared)
        quantum = HBAR/(2*reduced_mass*np.sqrt(squared))
        exact_slope = v/squared*(2*classical**2+quantum**2)/(classical**2+quantum**2)
        assert float(jax.grad(logarithm)(v)) == pytest.approx(exact_slope, rel=2e-13)
        h = v*1e-4
        numerical = float((logarithm(v+h)-logarithm(v-h))/(2*h))
        assert numerical == pytest.approx(exact_slope, rel=2e-7)
        # No jump in the log or either of its first two speed derivatives.
        for order, fn in enumerate((logarithm, jax.grad(logarithm), jax.grad(jax.grad(logarithm)))):
            a, b = float(fn(v*(1-1e-8))), float(fn(v*(1+1e-8)))
            # Scale derivatives by v**order; the curvature crosses zero and
            # cannot be checked with a relative difference against itself.
            assert abs(a-b)*v**order <= 1e-6*max(1., abs(a)*v**order, abs(b)*v**order)


@pytest.mark.parametrize("mass,charge", [(M, E), (4*M, 2*E), (ELECTRON_MASS, -E)])
def test_velocity_derivatives_and_maxwellian_zero_flux_for_each_test_mass(mass, charge):
    species, point = bath(), jnp.zeros(3)
    sigma = np.sqrt(1000.*E/mass)
    for r in (.02, .2, 1., np.sqrt(3.), 3., 10.):
        v = sigma*r
        for fn, derivative in ((nu_D_ab, d_nu_D_ab), (nu_par_ab, d_nu_par_ab)):
            rate = lambda speed: fn(mass, charge, 0, speed, point, species)
            assert float(derivative(mass, charge, 0, v, point, species)) == pytest.approx(float(jax.grad(rate)(v)), rel=2e-8)
        c = collision_coefficients(species, mass, charge, jnp.asarray(v), point, full_orbit=True)
        dp, dt, ddp, friction = map(float, c[:4])
        # The Maxwell radial density has log derivative 2/v-m_a*v/T_b.
        # Both speed and Cartesian Ito drifts must give zero probability flux.
        assert friction == pytest.approx(-dp*v/sigma**2, rel=3e-14)
        speed_drift = friction+ddp+2*dp/v
        assert speed_drift == pytest.approx(ddp+dp*(2/v-v/sigma**2), rel=3e-13)
        particle_drift = friction+ddp+2*(dp-dt)/v
        assert particle_drift == pytest.approx(ddp+2*(dp-dt)/v-dp*v/sigma**2, rel=3e-13)
        assert np.all(np.isfinite(np.asarray(c)))


def test_vacuum_is_zero_and_invalid_weak_coupling_is_not_silently_clipped():
    species = bath(density=0.)
    log = lambda v: coulomb_logarithm(M, E, 0, v, jnp.zeros(3), species)
    assert float(log(1e5)) == 0.
    assert float(jax.grad(log)(1e5)) == 0.
    assert np.all(np.asarray(collision_coefficients(species, M, E, jnp.asarray(1e5), jnp.zeros(3))) == 0)
    assert np.isnan(float(coulomb_logarithm(M, E, 0, 1e2, jnp.zeros(3), bath(density=1e40))))


def test_cutoff_model_keeps_parameter_derivatives_available():
    def logarithm(n, temperature):
        return impact_coulomb_logarithm(M, E, M, E, 3e5, temperature,
            jnp.array([n]), jnp.array([temperature]), jnp.array([E]))
    dn, dt = jax.grad(logarithm, argnums=(0, 1))(1e20, 1000.)
    assert float(dn) == pytest.approx(-.5/1e20, rel=2e-14)
    h = .01
    assert float(dt) == pytest.approx(float((logarithm(1e20, 1000.+h)-logarithm(1e20, 1000.-h))/(2*h)), rel=1e-8)
