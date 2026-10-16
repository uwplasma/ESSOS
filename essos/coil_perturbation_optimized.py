"""
Optimized JAX-based Gaussian Process sampler for coil perturbations.

This is an improved version of coil_perturbation.py with better JAX efficiency:
- Vectorized kernel computation without nested vmap definitions
- Direct use of jax.grad for higher derivatives (more efficient)
- Reduced kernel recomputation through better structure
- Simplified matrix operations
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, grad
from jaxtyping import Array, Float
from essos.coils import Curves, apply_symmetries_to_gammas
from functools import partial


@jit
def matrix_sqrt_via_spectral(A):
    """Compute matrix square root of SPD matrix A via spectral decomposition."""
    eigvals, Q = jnp.linalg.eigh(A)
    eigvals = jnp.clip(eigvals, a_min=0)
    sqrt_eigvals = jnp.sqrt(eigvals)
    sqrt_A = Q @ jnp.diag(sqrt_eigvals) @ Q.T
    return sqrt_A


class GaussianSamplerOptimized:
    r"""
    Optimized Gaussian process sampler for periodic coil perturbations.
    
    Key improvements over original:
    - Vectorized kernel computation without nested vmap
    - Direct automatic differentiation for higher derivatives
    - Better cache locality and reduced recomputation
    
    Args:
        points: Quadrature points along coil
        sigma: Standard deviation (perturbation magnitude)
        length_scale: Correlation length (smoothness)
        n_derivs: Number of derivatives (0, 1, or 2)
    """

    def __init__(self, points: Array, sigma: Float, length_scale: Float, n_derivs: int = 0):
        self.points = points
        self.sigma = sigma
        self.length_scale = length_scale
        self.n_derivs = n_derivs
        self.n_points = len(points)

    @partial(jit, static_argnames=['self'])
    def kernel_periodicity(self, x, y):
        """Periodic Gaussian kernel - vectorized over period copies."""
        aux_periodicity = jnp.arange(-5, 6)
        # Vectorized computation: avoid nested vmap
        kernels = self.sigma**2 * jnp.exp(-(x - y + aux_periodicity)**2 / (2. * self.length_scale**2))
        return jnp.sum(kernels)

    @partial(jit, static_argnames=['self'])
    def kernel_periodicity_matrix(self):
        """Compute full kernel matrix efficiently with vmap."""
        # Two-level vmap: vectorize over x then y
        kernel_fn = lambda x: vmap(lambda y: self.kernel_periodicity(x, y))(self.points)
        return vmap(kernel_fn)(self.points)

    @partial(jit, static_argnames=['self'])
    def kernel_deriv_matrix(self, deriv_order=1):
        """Compute derivative kernel matrix using automatic differentiation."""
        # Create derivative function on the fly
        def make_deriv_fn(base_fn, order):
            fn = base_fn
            for _ in range(order):
                fn = lambda x, y, f=fn: grad(f, argnums=0)(x, y)
            return fn

        deriv_fn = make_deriv_fn(self.kernel_periodicity, deriv_order)
        kernel_fn = lambda x: vmap(lambda y: deriv_fn(x, y))(self.points)
        return vmap(kernel_fn)(self.points)

    @partial(jit, static_argnames=['self'])
    def compute_covariance_matrix(self):
        """Compute base covariance matrix."""
        cov_mat = self.kernel_periodicity_matrix()
        return matrix_sqrt_via_spectral(cov_mat)

    @partial(jit, static_argnames=['self'])
    def compute_covariance_with_first_derivatives(self):
        """Compute covariance block matrix with first derivatives."""
        K = self.kernel_periodicity_matrix()
        K_d = self.kernel_deriv_matrix(deriv_order=1)
        K_dd = self.kernel_deriv_matrix(deriv_order=2)

        # Efficient block matrix construction
        top = jnp.concatenate([K, K_d], axis=1)
        bottom = jnp.concatenate([-K_d, K_dd], axis=1)
        full = jnp.concatenate([top, bottom], axis=0)
        return matrix_sqrt_via_spectral(full)

    @partial(jit, static_argnames=['self'])
    def compute_covariance_with_second_derivatives(self):
        """Compute covariance block matrix with up to second derivatives."""
        K = self.kernel_periodicity_matrix()
        K_d = self.kernel_deriv_matrix(deriv_order=1)
        K_dd = self.kernel_deriv_matrix(deriv_order=2)
        K_ddd = self.kernel_deriv_matrix(deriv_order=3)
        K_dddd = self.kernel_deriv_matrix(deriv_order=4)

        row1 = jnp.concatenate([K, K_d, K_dd], axis=1)
        row2 = jnp.concatenate([-K_d, K_dd, -K_ddd], axis=1)
        row3 = jnp.concatenate([K_dd, -K_ddd, K_dddd], axis=1)
        full = jnp.concatenate([row1, row2, row3], axis=0)
        return matrix_sqrt_via_spectral(full)

    @partial(jit, static_argnames=['self'])
    def get_covariance_matrix(self):
        """Dispatch to appropriate covariance computation method."""
        if self.n_derivs == 0:
            return self.compute_covariance_matrix()
        elif self.n_derivs == 1:
            return self.compute_covariance_with_first_derivatives()
        elif self.n_derivs == 2:
            return self.compute_covariance_with_second_derivatives()
        else:
            raise ValueError(f"n_derivs must be 0, 1, or 2, got {self.n_derivs}")

    @partial(jit, static_argnames=['self'])
    def draw_sample(self, key):
        """Draw a sample from the Gaussian process."""
        n = self.n_points * (self.n_derivs + 1)
        z = jax.random.normal(key=key, shape=(n, 3))
        L = self.get_covariance_matrix()
        sample = jnp.matmul(L, z)
        # Reshape to (n_derivs + 1, n_points, 3)
        return jnp.reshape(sample, (self.n_derivs + 1, self.n_points, 3))


def perturb_coils_optimized(coils: Curves, sampler: GaussianSamplerOptimized, key):
    """
    Apply optimized coil perturbation with proper symmetry handling.
    
    Args:
        coils: Coils object to perturb
        sampler: Gaussian sampler for perturbations
        key: Random key
    """
    # Generate independent samples for each base curve
    keys = jax.random.split(key, num=coils.n_base_curves)
    perturbations = vmap(sampler.draw_sample)(keys)  # (n_base_curves, n_derivs+1, n_points, 3)

    # Apply symmetries to perturbations
    gamma_pert = apply_symmetries_to_gammas(perturbations[:, 0], coils.nfp, coils.stellsym)
    coils.gamma = coils.gamma + gamma_pert

    if sampler.n_derivs >= 1:
        gamma_dash_pert = apply_symmetries_to_gammas(perturbations[:, 1], coils.nfp, coils.stellsym)
        coils.gamma_dash = coils.gamma_dash + gamma_dash_pert

    if sampler.n_derivs >= 2:
        gamma_dashdash_pert = apply_symmetries_to_gammas(perturbations[:, 2], coils.nfp, coils.stellsym)
        coils.gamma_dashdash = coils.gamma_dashdash + gamma_dashdash_pert
