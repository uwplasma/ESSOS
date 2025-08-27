import unittest
import jax
import jax.numpy as jnp
import numpy as np

from essos.coil_perturbation import (
    ldl_decomposition,
    matrix_sqrt_via_spectral,
    GaussianSampler,
    PerturbationSample,
    perturb_curves_systematic,
    perturb_curves_statistic,
)

# Dummy Curves and apply_symmetries_to_gammas for testing
class DummyCurves:
    def __init__(self, n_base_curves=2, nfp=1, stellsym=True, n_points=5, n_derivs=2):
        self.n_base_curves = n_base_curves
        self.nfp = nfp
        self.stellsym = stellsym
        self.gamma = jnp.zeros((n_base_curves, n_points, 3))
        self.gamma_dash = jnp.zeros((n_base_curves, n_points, 3))
        self.gamma_dashdash = jnp.zeros((n_base_curves, n_points, 3))

def dummy_apply_symmetries_to_gammas(gamma, nfp, stellsym):
    # Just return the input for testing
    return gamma

# Patch apply_symmetries_to_gammas in the tested module
import essos.coil_perturbation
essos.coil_perturbation.apply_symmetries_to_gammas = dummy_apply_symmetries_to_gammas

class TestCoilPerturbation(unittest.TestCase):
    def test_ldl_decomposition(self):
        A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
        L, D = ldl_decomposition(A)
        # Check shapes
        self.assertEqual(L.shape, (2, 2))
        self.assertEqual(D.shape, (2,))
        # Check that A ≈ L @ jnp.diag(D) @ L.T
        A_recon = L @ jnp.diag(D) @ L.T
        np.testing.assert_allclose(A, A_recon, atol=1e-6)

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

    def test_perturb_curves_systematic(self):
        points = jnp.linspace(0, 1, 5)
        sampler0 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=0)
        sampler1 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=1)
        sampler2 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=2)
        key = jax.random.PRNGKey(0)
        for sampler in [sampler0, sampler1, sampler2]:
            curves = DummyCurves(n_base_curves=2, nfp=1, stellsym=True, n_points=5)
            perturb_curves_systematic(curves, sampler, key)
            # Just check that gamma arrays are still the right shape
            self.assertEqual(curves.gamma.shape, (2, 5, 3))

    def test_perturb_curves_statistic(self):
        points = jnp.linspace(0, 1, 5)
        sampler0 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=0)
        sampler1 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=1)
        sampler2 = GaussianSampler(points, sigma=1.0, length_scale=0.5, n_derivs=2)
        key = jax.random.PRNGKey(0)
        for sampler in [sampler0, sampler1, sampler2]:
            curves = DummyCurves(n_base_curves=2, nfp=1, stellsym=True, n_points=5)
            perturb_curves_statistic(curves, sampler, key)
            self.assertEqual(curves.gamma.shape, (2, 5, 3))

if __name__ == "__main__":
    unittest.main()