import unittest
import jax
import jax.numpy as jnp
import numpy as np
from essos.coils import Curves

from essos.coil_perturbation import (
    #ldl_decomposition,
    matrix_sqrt_via_spectral,
    GaussianSampler,
    PerturbationSample,
    perturb_curves_systematic,
    perturb_curves_statistic,
)

def dummy_apply_symmetries_to_gammas(gamma, nfp, stellsym):
    # Just return the input for testing
    return gamma

# Patch apply_symmetries_to_gammas in the tested module
import essos.coil_perturbation
essos.coil_perturbation.apply_symmetries_to_gammas = dummy_apply_symmetries_to_gammas

class TestCoilPerturbation(unittest.TestCase):
    @staticmethod
    def make_curves(n_base_curves=2, n_segments=5, nfp=1, stellsym=True, order=1):
        dofs = jnp.zeros((n_base_curves, 3, 2 * order + 1))
        return Curves(dofs, n_segments=n_segments, nfp=nfp, stellsym=stellsym)

    #def test_ldl_decomposition(self):
    #    A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
    #    L, D = ldl_decomposition(A)
    #    # Check shapes
    #    self.assertEqual(L.shape, (2, 2))
    #    self.assertEqual(D.shape, (2,))
    #    # Check that A ≈ L @ jnp.diag(D) @ L.T
    #    A_recon = L @ jnp.diag(D) @ L.T
    #    np.testing.assert_allclose(A, A_recon, atol=1e-6)

    def test_matrix_sqrt_via_spectral(self):
        A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
        sqrt_A = matrix_sqrt_via_spectral(A)
        # sqrt_A @ sqrt_A ≈ A
        A_recon = sqrt_A @ sqrt_A
        np.testing.assert_allclose(A, A_recon, atol=1e-6)

    def test_gaussian_sampler_covariances_and_draw(self):
        points = jnp.linspace(0, 1, 5)
        sampler0 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=0)
        sampler1 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=1)
        sampler2 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=2)
        # Covariance matrices
        cov0 = sampler0.get_covariance_matrix()
        cov1 = sampler1.get_covariance_matrix()
        cov2 = sampler2.get_covariance_matrix()
        self.assertEqual(cov0.shape[0], 5)
        self.assertEqual(cov1.shape[0], 10)
        self.assertEqual(cov2.shape[0], 15)
        # Draw samples
        key = jax.random.PRNGKey(0)
        sample0 = sampler0.draw_sample(key)
        sample1 = sampler1.draw_sample(key)
        sample2 = sampler2.draw_sample(key)
        self.assertEqual(sample0.shape, (1, 5, 3))
        self.assertEqual(sample1.shape, (2, 5, 3))
        self.assertEqual(sample2.shape, (3, 5, 3))

    def test_gaussian_sampler_kernels(self):
        points = jnp.linspace(0, 1, 3)
        sampler = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=2)
        # Test kernel and derivatives
        val = sampler.kernel_periodicity(0.1, 0.2)
        dval = sampler.d_kernel_periodicity_dx(0.1, 0.2)
        ddval = sampler.d_kernel_periodicity_dxdx(0.1, 0.2)
        dddval = sampler.d_kernel_periodicity_dxdxdx(0.1, 0.2)
        ddddval = sampler.d_kernel_periodicity_dxdxdxdx(0.1, 0.2)
        self.assertIsInstance(val, jnp.ndarray)
        self.assertIsInstance(dval, jnp.ndarray)
        self.assertIsInstance(ddval, jnp.ndarray)
        self.assertIsInstance(dddval, jnp.ndarray)
        self.assertIsInstance(ddddval, jnp.ndarray)

    def test_perturbation_sample(self):
        points = jnp.linspace(0, 1, 5)
        sampler = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=1)
        key = jax.random.PRNGKey(0)
        ps = PerturbationSample(sampler, key)
        # get_sample for deriv=0 and deriv=1
        s0 = ps.get_sample(0)
        s1 = ps.get_sample(1)
        self.assertEqual(s0.shape, (5, 3))
        self.assertEqual(s1.shape, (5, 3))
        # resample
        ps.resample()
        # get_sample with too high deriv should raise
        with self.assertRaises(ValueError):
            ps.get_sample(2)

    def test_perturbation_sample_accepts_array_sample(self):
        sampler = GaussianSampler(jnp.linspace(0, 1, 5), sigma=1.0, length_scale=0.5, n_derivs=1)
        sample = jnp.ones((2, 5, 3))
        ps = PerturbationSample(sampler, jax.random.PRNGKey(0), sample=sample)
        self.assertTrue(jnp.array_equal(ps.get_sample(1), sample[1]))

    def test_perturb_curves_systematic(self):
        points = jnp.linspace(0, 1, 5)
        sampler0 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=0)
        sampler1 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=1)
        sampler2 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=2)
        key = jax.random.PRNGKey(0)
        for sampler in [sampler0, sampler1, sampler2]:
            curves = self.make_curves(n_base_curves=2, n_segments=5, nfp=1, stellsym=True)
            curves = perturb_curves_systematic(curves, sampler, key)
            self.assertIsInstance(curves, Curves)
            self.assertEqual(curves.gamma.shape, (4, 5, 3))

    def test_perturb_curves_statistic(self):
        points = jnp.linspace(0, 1, 5)
        sampler0 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=0)
        sampler1 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=1)
        sampler2 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=2)
        key = jax.random.PRNGKey(0)
        for sampler in [sampler0, sampler1, sampler2]:
            curves = self.make_curves(n_base_curves=2, n_segments=5, nfp=1, stellsym=True)
            curves = perturb_curves_statistic(curves, sampler, key)
            self.assertIsInstance(curves, Curves)
            self.assertEqual(curves.gamma.shape, (4, 5, 3))

if __name__ == "__main__":
    unittest.main()


def test_gaussian_sampler_derivatives_are_derivatives_of_the_sample():
    n = 32
    sampler = GaussianSampler(jnp.linspace(0, 1, n, endpoint=False), sigma=0.01, length_scale=0.2, n_derivs=2)
    f, df, ddf = np.asarray(sampler.draw_sample(jax.random.PRNGKey(1)))
    k = 2j * np.pi * np.fft.fftfreq(n, 1 / n)[:, None]
    scale = np.abs(df).max()
    np.testing.assert_allclose(np.fft.ifft(k * np.fft.fft(f, axis=0), axis=0).real, df, atol=1e-3 * scale)
    np.testing.assert_allclose(np.fft.ifft(k**2 * np.fft.fft(f, axis=0), axis=0).real, ddf, atol=1e-3 * np.abs(ddf).max())


def test_zero_systematic_perturbation_keeps_discretized_coils_and_default_key():
    from essos.coils import DiscretizedCoils
    from essos.coil_perturbation import perturb_curves
    t = jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False)
    coils = DiscretizedCoils(jnp.stack([jnp.stack([3 + .5 * jnp.cos(t), 0 * t, .5 * jnp.sin(t)], axis=1)]),
                             jnp.array([1e5]), nfp=2, stellsym=True)
    sampler = GaussianSampler(jnp.linspace(0, 1, 32, endpoint=False), sigma=0., length_scale=0.2)
    np.testing.assert_allclose(perturb_curves(coils, sampler).gamma, coils.gamma, atol=1e-12)


def test_gaussian_process_has_the_requested_standard_deviation_and_periodic_smoothness():
    """The periodic squared-exponential kernel has Var X(s) = sigma^2 sum_i exp(-i^2 / 2 l^2) = sigma^2 for small
    l (Rasmussen & Williams, ch. 4); neighbouring points are strongly correlated, distant ones are not."""
    from essos.coil_perturbation import _draw_curve_perturbation
    n, sigma = 48, 0.02
    sampler = GaussianSampler(jnp.linspace(0, 1, n, endpoint=False), sigma=sigma, length_scale=0.1)
    draws = np.asarray(_draw_curve_perturbation(sampler, jax.random.PRNGKey(3), 64))  # (64, n, 3)
    np.testing.assert_allclose(draws.std(), sigma, rtol=0.1)
    corr = lambda lag: np.mean(draws * np.roll(draws, lag, axis=1)) / draws.var()
    assert corr(1) > 0.9 > 0.5 > abs(corr(n // 2))  # exp(-(1/48)^2 / 2 / 0.1^2) ~ 0.98 vs ~ e^-12.5
    # the sample object exposes the process and its derivatives, and refuses derivatives that were not drawn
    sample = PerturbationSample(sampler, key=jax.random.PRNGKey(0))
    assert sample.get_sample(0).shape == (n, 3)
    with np.testing.assert_raises(ValueError):
        sample.get_sample(1)
    resampled = sample._sample
    sample.resample()
    np.testing.assert_allclose(sample._sample, resampled)  # same key, same draw


def test_coil_perturbations_keep_currents_and_symmetry_or_break_it_by_type():
    from essos.coils import Coils, CreateEquallySpacedCurves, DiscretizedCoils
    from essos.coil_perturbation import perturb_curves
    n, sigma = 32, 0.01
    sampler = GaussianSampler(jnp.linspace(0, 1, n, endpoint=False), sigma=sigma, length_scale=0.3)
    coils = Coils(CreateEquallySpacedCurves(2, 3, 3.0, 0.5, n, 2, True), jnp.array([1e5, 2e5]))
    key = jax.random.PRNGKey(7)

    # systematic: only the base coils are perturbed, so the expanded set keeps its nfp / stellarator symmetry
    sys_coils = perturb_curves(coils, sampler, key, "systematic")
    np.testing.assert_allclose(sys_coils.currents, coils.currents)
    base = sys_coils.gamma[:2]
    expected = Coils(Curves(sys_coils.curves._dofs, n, 2, True), sys_coils.dofs_currents_raw).gamma
    np.testing.assert_allclose(sys_coils.gamma, expected, atol=1e-12)
    shift = np.asarray(sys_coils.gamma - coils.gamma)
    assert 0 < np.abs(shift[:2]).max() < 6 * sigma and np.abs(base - coils.gamma[:2]).max() > 0
    # rotation by pi about z maps coil 4 (k = 1, no flip) onto the base-coil perturbation
    rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    np.testing.assert_allclose(shift[4:6] @ rot.T, shift[:2], atol=1e-10)

    # statistical: every expanded coil gets its own draw and the symmetry structure is dropped
    stat = perturb_curves(coils, sampler, key, "statistical")
    assert stat.nfp == 1 and stat.stellsym is False and len(stat) == len(coils)
    np.testing.assert_allclose(stat.currents, coils.currents)
    d = np.asarray(stat.gamma - coils.gamma)
    assert np.abs(d[4:6] @ rot.T - d[:2]).max() > sigma
    dc = DiscretizedCoils(coils.gamma[:2], jnp.array([1e5, 2e5]), nfp=1, stellsym=False)
    dstat = perturb_curves(dc, sampler, key, "statistical")
    assert dstat.gamma.shape == dc.gamma.shape and np.abs(np.asarray(dstat.gamma - dc.gamma)).max() > 0
    curves_stat = perturb_curves(coils.curves, sampler, key, "statistical")
    assert curves_stat.gamma.shape == coils.curves.gamma.shape

    with np.testing.assert_raises(ValueError):
        perturb_curves(coils, sampler, key, "bogus")
    with np.testing.assert_raises(TypeError):
        perturb_curves(jnp.zeros((2, n, 3)), sampler, key, "systematic")
