"""Small independent checks for the nonrelativistic statistics reference."""
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import invgauss

from essos.background_species import BackgroundSpecies
from essos.collisions import collision_coefficients
from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q
from examples.benchmarks.nonrelativistic_collision_reference import (
    ProtonBackground, first_passage_reference, proton_first_passage, quadrature_mean,
    passage_statistics)


@pytest.mark.parametrize("r", [.05, .5, 1., 2., np.sqrt(10), 5.])
def test_numpy_proton_coefficients_match_independent_jax_implementation(r):
    physical = ProtonBackground()
    species = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
                                jnp.array([1e20]), jnp.array([1000.]))
    expected = physical.coefficients(physical.sigma*r)
    actual = collision_coefficients(species, M, Q, jnp.array(physical.sigma*r), jnp.zeros(3))
    for name in ("parallel", "perpendicular", "parallel_derivative", "friction"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=3e-10)


def test_backward_reference_matches_constant_drift_inverse_gaussian():
    # dX=-dt+sqrt(.5)dW, from 1 to 0: E[tau]=1, Var[tau]=.5.
    # The remote reflecting boundary at 8 has negligible effect here.
    reference = first_passage_reference(lambda x: np.full_like(x, .25),
        lambda x: np.exp(-4*x), boundary=0., start=1., upper=8., nodes=1024)
    assert reference.mean == pytest.approx(1., rel=5e-5)
    assert reference.variance == pytest.approx(.5, rel=3e-4)
    times = np.array([.2, .5, 1., 2., 4.])
    np.testing.assert_allclose(reference.cdf(times), invgauss.cdf(times, .5, scale=2.), atol=8e-5)
    raw = reference.restricted_moments(30.)
    np.testing.assert_allclose(raw[:2], [reference.mean, reference.mean**2+reference.variance], rtol=2e-8)


def test_physical_backward_reference_converges_and_matches_natural_boundary_quadrature():
    physical = ProtonBackground()
    coarse = proton_first_passage(physical, nodes=384)
    fine = proton_first_passage(physical, nodes=768)
    extended = proton_first_passage(physical, upper=10., nodes=1024)
    independent = quadrature_mean(physical, fine.boundary, fine.start)
    assert fine.mean == pytest.approx(independent, rel=3e-5)
    assert fine.mean == pytest.approx(coarse.mean, rel=2e-4)
    assert fine.variance == pytest.approx(extended.variance, rel=2e-4)
    times = np.linspace(0., 5*fine.mean, 41)
    np.testing.assert_allclose(fine.cdf(times), coarse.cdf(times), atol=2e-4)
    np.testing.assert_allclose(fine.cdf(times), extended.cdf(times), atol=1e-4)
    assert np.all(np.diff(fine.cdf(times)) >= -1e-12)
    assert fine.cdf(0.) == pytest.approx(0., abs=2e-12)


def test_first_passage_diagnostics_keep_censored_paths_and_reject_wrong_time_scale():
    reference = first_passage_reference(lambda x: np.full_like(x, .25),
        lambda x: np.exp(-4*x), boundary=0., start=1., upper=8., nodes=512)
    # Synthetic diagnostic samples only; these are NOT particle integrations.
    exact = invgauss.ppf((np.arange(32)+.5)/32, .5, scale=2.)
    horizon = 1.2
    hit = exact <= horizon
    observed = np.minimum(exact, horizon)
    correct = passage_statistics(observed, hit, reference, horizon)
    assert correct["passed"] and correct["censored"] > 0
    import json
    json.dumps(correct)  # Reports must contain native Python scalars.
    wrong = passage_statistics(observed*.01, np.ones(32, dtype=bool), reference, horizon)
    assert not wrong["passed"] and not wrong["cdf"]["passed"]


def test_physical_reference_requires_an_explicit_positive_thermal_band():
    with pytest.raises(ValueError, match="threshold"):
        proton_first_passage(ProtonBackground(), threshold_energy_over_T=0.)


def test_stopped_reward_reference_reproduces_restricted_first_passage_mean():
    from examples.benchmarks.nonrelativistic_collision_reference import radial_generator
    from examples.benchmarks.spatial_collision_reference import reward_spectrum, StoppedSpatialReference
    diffusion = lambda x: np.full_like(x, .25)
    weight = lambda x: np.exp(-4*x)
    ref = first_passage_reference(diffusion, weight, boundary=0., start=1., upper=8., nodes=384)
    grid, mass, _, _, diagonal, off, evaluate = radial_generator(diffusion, weight,
        boundary=0., start=1., upper=8., nodes=384)
    rates, weights = reward_spectrum(diagonal, off, mass, evaluate, np.ones_like(grid))
    reward = StoppedSpatialReference((rates, rates), (weights, weights*0), .6)
    times = np.array([.01, .2, 1., 3.])
    expected = [ref.restricted_moments(t)[0] for t in times]
    np.testing.assert_allclose(reward.msd(times)/4, expected, atol=1e-10)
    assert reward.msd(0.) == 0.


def test_stopped_physical_spatial_reference_converges_in_mesh_and_upper_boundary():
    from examples.benchmarks.spatial_collision_reference import stopped_spatial_reference
    physical = ProtonBackground()
    times = np.array([0., .01, .5, 2., 10.])
    coarse = stopped_spatial_reference(physical, nodes=384).msd(times)
    fine = stopped_spatial_reference(physical, nodes=768).msd(times)
    upper = stopped_spatial_reference(physical, nodes=1024, upper=10.).msd(times)
    np.testing.assert_allclose(fine, coarse, rtol=3e-4, atol=1e-10)
    np.testing.assert_allclose(fine, upper, rtol=3e-4, atol=1e-10)
    assert fine[0] == 0. and np.all(np.diff(fine) > 0)
    # d/dt MSD at t=0 is exactly four times the initial D_X.
    c = physical.coefficients(np.sqrt(10)*physical.sigma)
    local = (c["parallel"]*(1-.6**2)+c["perpendicular"]*(1+.6**2))/(2*physical.sigma**2*physical.nu)
    slope = stopped_spatial_reference(physical).msd(1e-5)/1e-5
    assert slope == pytest.approx(4*local, rel=3e-5)


@pytest.mark.parametrize("pitch", [-1., 0., .6, 1.])
def test_spatial_map_covariance_matches_paper_for_rotated_field_and_B_scaling(pitch):
    import jax
    from essos.collisions import CollisionCoefficients, guiding_center_spatial_increment
    c = CollisionCoefficients(.8, .3, 0., 0., 0., 0.)
    b = jnp.array([1., 2., 3.])/np.sqrt(14)
    omega = 7.
    matrix = np.asarray(jax.vmap(lambda w: guiding_center_spatial_increment(pitch, w, c, b, omega))(jnp.eye(3))).T
    dx = (.8*(1-pitch*pitch)+.3*(1+pitch*pitch))/(2*omega**2)
    np.testing.assert_allclose(matrix@matrix.T, 2*dx*(np.eye(3)-np.outer(b, b)), atol=1e-17)
    np.testing.assert_allclose(b@matrix, np.zeros(3), atol=2*np.finfo(float).eps*np.linalg.norm(matrix))
    doubled = np.asarray(jax.vmap(lambda w: guiding_center_spatial_increment(pitch, w, c, b, 2*omega))(jnp.eye(3))).T
    np.testing.assert_allclose(doubled, matrix/2, atol=0., rtol=0.)


@pytest.mark.parametrize("controller", ["conservative", "paper", "candidate"])
def test_stopped_spatial_benchmark_smoke(controller):
    import jax
    from examples.benchmarks.validate_collisional_slowing_down import native_solver
    from essos.split import SplitStatus
    physical = ProtonBackground()
    solve = native_solver("gc", controller, True, physical, 1e-4, 2., spatial=True,
                          times=np.array([0., 1e-4]))
    result = solve(jax.random.split(jax.random.key(17), 2), .003, 1e-4, 1e-4)
    assert np.all(result.status == SplitStatus.SUCCESS)
    assert np.all(np.isfinite(result.y)) and np.all(result.y[:, 5] > 0)
    np.testing.assert_array_equal(result.y[:, 2], np.zeros(2))
    np.testing.assert_allclose(result.y, result.ys[:, -1], atol=0., rtol=0.)
    np.testing.assert_allclose(result.t, 1e-4, rtol=2e-15)


def test_spatial_noise_diagnostic_uses_pathwise_variance_not_a_constant_scale():
    from examples.benchmarks.analyze_collisional_convergence import conditional_spatial_statistics
    # Synthetic quantiles only: changing physical variances must cancel in
    # the conditional ratio. A factor-of-two kick error must be detected.
    n = 64
    exact_ratios = -np.log1p(-(np.arange(n)+.5)/n)
    integral = np.geomspace(.01, 100., n)
    correct = conditional_spatial_statistics(integral*exact_ratios, integral)
    wrong = conditional_spatial_statistics(4*integral*exact_ratios, integral)
    assert correct["ks_distance"] == pytest.approx(.5/n, abs=2e-15)
    assert wrong["ks_distance"] > wrong["marginal_99_dkw_limit"]
    with pytest.raises(ValueError, match="positive"):
        conditional_spatial_statistics(exact_ratios, np.zeros(n))


@pytest.mark.parametrize("pitch", [-1., -.5, .6, 1.])
def test_nonrelativistic_spatial_tensor_follows_direct_gyroaverage_of_momentum_noise(pitch):
    import jax
    from essos.collisions import CollisionCoefficients, guiding_center_spatial_increment
    # Independent derivation: X=x+(v cross b)/Omega, so dX is the
    # perpendicular cross-product of momentum noise divided by m*Omega.
    b = np.array([1., 2., 3.])/np.sqrt(14)
    e1 = np.cross(b, [1., 0., 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b, e1)
    cross_b = np.stack([np.cross(e, b) for e in np.eye(3)], axis=1)
    physical = ProtonBackground()
    coeff = physical.coefficients(np.sqrt(10)*physical.sigma)
    dp, dt = coeff["parallel"], coeff["perpendicular"]
    omega = Q/M
    jacobian = cross_b/(M*omega)
    tensors = []
    for phi in 2*np.pi*np.arange(16)/16:
        vhat = pitch*b+np.sqrt(1-pitch*pitch)*(np.cos(phi)*e1+np.sin(phi)*e2)
        momentum_diffusion = M**2*(dt*np.eye(3)+(dp-dt)*np.outer(vhat, vhat))
        tensors.append(jacobian@momentum_diffusion@jacobian.T)
    expected = np.mean(tensors, axis=0)
    c = CollisionCoefficients(dp, dt, 0., 0., 0., 0.)
    amplitude = np.asarray(jax.vmap(lambda dw: guiding_center_spatial_increment(
        pitch, dw, c, jnp.asarray(b), omega))(jnp.eye(3))).T
    np.testing.assert_allclose(amplitude@amplitude.T/2, expected, rtol=2e-14, atol=2e-17)


