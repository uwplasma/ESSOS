import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
from essos.coils import Curves, Coils, DiscretizedCoils, fit_dofs_from_coils
from functools import partial


#def ldl_decomposition(A):
#    """
#    Performs LDLᵀ decomposition on a symmetric positive-definite matrix A.
#    A = L D Lᵀ where:
#      - L is lower triangular with unit diagonal
#      - D is diagonal
#
#    Args:
#        A: (n, n) symmetric matrix
#
#    Returns:
#        L: (n, n) lower-triangular matrix with unit diagonal
#        D: (n,) diagonal elements of D
#    """
#    n = A.shape[0]
#    L = jnp.eye(n)
#    D = jnp.zeros(n)

#    def body_fun(k, val):
#        L, D = val

        # Compute D[k]
#        D_k = A[k, k] - jnp.sum((L[k, :k] ** 2) * D[:k])
#        D = D.at[k].set(D_k)

#        def inner_body(i, L):
#            L_ik = (A[i, k] - jnp.sum(L[i, :k] * L[k, :k] * D[:k])) / D_k
#            return L.at[i, k].set(L_ik)

        # Update column k of L below diagonal
#        L = jax.lax.fori_loop(k + 1, n, inner_body, L)

#        return (L, D)

#    L, D = jax.lax.fori_loop(0, n, body_fun, (L, D))

#    return L, D


@jit
def matrix_sqrt_via_spectral(A):
    """Compute matrix square root of SPD matrix A via spectral decomposition."""
    eigvals, Q = jnp.linalg.eigh(A)  # A = Q Λ Q^T

    # Ensure numerical stability (clip small negatives to 0)
    eigvals = jnp.clip(eigvals, min=0)

    sqrt_eigvals = jnp.sqrt(eigvals)
    sqrt_A = Q @ jnp.diag(sqrt_eigvals) @ Q.T

    return sqrt_A

#This is based on SIMSOPT's GaussianSampler, but with some modifications to make it work with JAX.
#Note: I am not sure this should be kept as a class, but it is for now to keep the interface similar to SIMSOPT.
class GaussianSampler():
    r"""
    Generate a periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
    The process has standard deviation ``sigma`` a correlation length scale ``length_scale``.
    Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
    functions.
    Also has the ability to sample the derivatives of the function.

    We consider the kernel

    .. math::

        \kappa(d) = \sigma^2 \exp(-d^2/l^2)

    and then consider a Gaussian process with covariance

    .. math::

        Cov(X(s), X(t)) = \sum_{i=-\infty}^\infty \sigma^2 \exp(-(s-t+i)^2/l^2)

    the sum is used to make the kernel periodic and in practice the infinite sum is truncated.

    Args:
        points: the quadrature points along which the perturbation should be computed. 
        sigma: standard deviation of the underlying gaussian process
            (measure for the magnitude of the perturbation).
        length_scale: length scale of the underlying gaussian process
                      (measure for the smoothness of the perturbation).
        n_derivs: number of derivatives to calculate, right now maximum is up to 2
    """

    points: Array
    sigma: Float
    length_scale: Float
    n_derivs: int

    def __init__(self,points: Array, sigma: Float, length_scale: Float, n_derivs: int = 0):
        self.points=points
        self.sigma=sigma
        self.length_scale=length_scale
        self.n_derivs=n_derivs


    @partial(jit, static_argnames=['self'])
    def kernel_periodicity(self,x, y):
        aux_periodicity=jnp.arange(-5, 6)
        def kernel(x, y,i):
            return self.sigma**2*jnp.exp(-(x-y+i)**2/(2.*self.length_scale**2))

        return jnp.sum(jax.vmap(kernel,in_axes=(None,None,0))(x,y,aux_periodicity))
    
    @partial(jit, static_argnames=['self'])
    def d_kernel_periodicity_dx(self,x, y):
        return jax.grad(self.kernel_periodicity, argnums=0)(x, y)

    @partial(jit, static_argnames=['self'])
    def d_kernel_periodicity_dxdx(self,x, y):
        return jax.grad(self.d_kernel_periodicity_dx, argnums=0)(x, y)     

    @partial(jit, static_argnames=['self'])
    def d_kernel_periodicity_dxdxdx(self,x, y):
        return jax.grad(self.d_kernel_periodicity_dxdx, argnums=0)(x, y)   

    @partial(jit, static_argnames=['self'])
    def d_kernel_periodicity_dxdxdxdx(self,x, y):
        return jax.grad(self.d_kernel_periodicity_dxdxdx, argnums=0)(x, y)     
 

    @partial(jit, static_argnames=['self'])
    def compute_covariance_matrix(self):
        final_mat= jax.vmap(jax.vmap(self.kernel_periodicity,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points)
        return matrix_sqrt_via_spectral(final_mat)


    @partial(jit, static_argnames=['self'])
    def compute_covariance_matrix_and_first_derivatives(self):
        cov_mat= jax.vmap(jax.vmap(self.kernel_periodicity,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points)
        dcov_mat_dx= jax.vmap(jax.vmap(self.d_kernel_periodicity_dx,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points)
        dcov_mat_dxdx= jax.vmap(jax.vmap(self.d_kernel_periodicity_dxdx,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points)
        final_mat = jnp.concatenate((jnp.concatenate((cov_mat, dcov_mat_dx),axis=0),jnp.concatenate((-dcov_mat_dx,dcov_mat_dxdx),axis=0 )), axis=1)
        return matrix_sqrt_via_spectral(final_mat)

    @partial(jit, static_argnames=['self'])
    def compute_covariance_matrix_and_second_derivatives(self):
        cov_mat= jax.vmap(jax.vmap(self.kernel_periodicity,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points)
        dcov_mat_dx= jax.vmap(jax.vmap(self.d_kernel_periodicity_dx,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points)
        dcov_mat_dxdx= jax.vmap(jax.vmap(self.d_kernel_periodicity_dxdx,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points)
        dcov_mat_dxdxdx= jax.vmap(jax.vmap(self.d_kernel_periodicity_dxdxdx,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points) 
        dcov_mat_dxdxdxdx= jax.vmap(jax.vmap(self.d_kernel_periodicity_dxdxdxdx,in_axes=(None,0)),in_axes=(0,None))(self.points, self.points)                
        final_mat= jnp.concatenate((jnp.concatenate((cov_mat, dcov_mat_dx,dcov_mat_dxdx),axis=0),
            jnp.concatenate((-dcov_mat_dx,dcov_mat_dxdx,-dcov_mat_dxdxdx),axis=0),
            jnp.concatenate((dcov_mat_dxdx,-dcov_mat_dxdxdx,dcov_mat_dxdxdxdx),axis=0 )), axis=1)  
        return matrix_sqrt_via_spectral(final_mat)

    @partial(jit, static_argnames=['self'])
    def get_covariance_matrix(self):
        if self.n_derivs ==0:
            return self.compute_covariance_matrix()
        elif self.n_derivs ==1:
            return self.compute_covariance_matrix_and_first_derivatives()
        elif self.n_derivs ==2:
            return self.compute_covariance_matrix_and_second_derivatives()


    @partial(jit, static_argnames=['self'])
    def draw_sample(self, key=0):

        n = len(self.points)
        z = jax.random.normal(key=key,shape=(len(self.points)*(self.n_derivs+1), 3))
        L=self.get_covariance_matrix()
        curve_and_derivs = jnp.matmul(L,z)
        if self.n_derivs ==0:            
            return jnp.reshape(jnp.matmul(L,z),(1,len(self.points),3))
        elif self.n_derivs ==1:            
            return jnp.reshape(jnp.matmul(L,z),(2,len(self.points),3))
        elif self.n_derivs ==2:            
            return jnp.reshape(jnp.matmul(L,z),(3,len(self.points),3))



class PerturbationSample():
    def __init__(self, sampler, key=0, sample=None):
        self.sampler = sampler
        self.key = key   # If not None, most likely fail with serialization
        if sample:
            self._sample = sample
        else:
            self.resample()

    def resample(self):
        self._sample = self.sampler.draw_sample(self.key)

    def get_sample(self, deriv):
        """
        Get the perturbation (if ``deriv=0``) or its ``deriv``-th derivative.
        """
        assert isinstance(deriv, int)
        if deriv >= len(self._sample):
            raise ValueError("""The sample on has {len(self._sample)-1} derivatives.
        Adjust the `n_derivs` parameter of the sampler to access higher derivatives.""")
        return self._sample[deriv]



#Util functions for perturbing curves and coils.
def _draw_curve_perturbation(sampler: GaussianSampler, key, n_curves: int):
    new_seeds = jax.random.split(key, num=n_curves)
    perturbation = jax.vmap(sampler.draw_sample, in_axes=(0))(new_seeds)
    return perturbation[:, 0, :, :]


def _make_curves_like(curves: Curves, dofs_new, nfp: int, stellsym: bool):
    return Curves(
        dofs_new,
        curves.n_segments,
        nfp=nfp,
        stellsym=stellsym,
        scaling_type=curves.scaling_type,
        scaling_factor=curves.scaling_factor,
        scale_fixed=curves.scale_fixed,
    )



#Perturb coisl fucntion
def perturb_curves(curves, sampler:GaussianSampler, key=None, perturbation_type="systematic"):
    """
    Apply a perturbation to curves or coils and return the perturbed object.

    Args:
        curves: Curves, Coils, or DiscretizedCoils to be perturbed.
        sampler: the gaussian sampler used to get the perturbations
        key: the seed which will be split to generate random but reproducible perturbations
        perturbation_type: "systematic" to perturb only unique/base coils and preserve symmetry,
            or "statistical"/"statistic" to perturb every expanded coil independently.

    Returns:
        A new perturbed object of the same family as the input.
    """
    if perturbation_type == "systematic":
        if isinstance(curves, DiscretizedCoils):
            perturbation = _draw_curve_perturbation(sampler, key, curves.n_base_curves)
            return DiscretizedCoils(
                curves.dofs_gamma + perturbation,
                currents=curves.dofs_currents_raw,
                nfp=curves.nfp,
                stellsym=curves.stellsym,
            )

        if isinstance(curves, Coils):
            perturbation = _draw_curve_perturbation(sampler, key, curves.curves.n_base_curves)
            base_curves = _make_curves_like(curves.curves, curves.dofs_curves, nfp=1, stellsym=False)
            perturbed_base_gamma = base_curves.gamma + perturbation
            dofs_new, _ = fit_dofs_from_coils(perturbed_base_gamma, curves.order, curves.n_segments, assume_uniform=True)
            new_curves = _make_curves_like(curves.curves, dofs_new, nfp=curves.nfp, stellsym=curves.stellsym)
            return Coils(curves=new_curves, currents=curves.dofs_currents_raw)

        if isinstance(curves, Curves):
            perturbation = _draw_curve_perturbation(sampler, key, curves.n_base_curves)
            base_curves = _make_curves_like(curves, curves.dofs, nfp=1, stellsym=False)
            perturbed_base_gamma = base_curves.gamma + perturbation
            dofs_new, _ = fit_dofs_from_coils(perturbed_base_gamma, curves.order, curves.n_segments, assume_uniform=True)
            return _make_curves_like(curves, dofs_new, nfp=curves.nfp, stellsym=curves.stellsym)

    elif perturbation_type in {"statistical"}:
        perturbation = _draw_curve_perturbation(sampler, key, curves.gamma.shape[0])
        gamma_perturbed = curves.gamma + perturbation

        if isinstance(curves, DiscretizedCoils):
            return DiscretizedCoils(gamma_perturbed, currents=curves.currents, nfp=1, stellsym=False)

        if isinstance(curves, Coils):
            dofs_new, _ = fit_dofs_from_coils(gamma_perturbed, curves.order, curves.n_segments, assume_uniform=True)
            new_curves = _make_curves_like(curves.curves, dofs_new, nfp=1, stellsym=False)
            return Coils(curves=new_curves, currents=curves.currents)

        if isinstance(curves, Curves):
            dofs_new, _ = fit_dofs_from_coils(gamma_perturbed, curves.order, curves.n_segments, assume_uniform=True)
            return _make_curves_like(curves, dofs_new, nfp=1, stellsym=False)

    else:
        raise ValueError(
            f"Unsupported perturbation_type {perturbation_type}. "
            "Expected 'systematic' or 'statistical'."
        )

    raise TypeError(f"Unsupported type {type(curves)}. Expected Curves, Coils, or DiscretizedCoils.")


def perturb_curves_systematic(curves, sampler:GaussianSampler, key=None):
    return perturb_curves(curves, sampler, key=key, perturbation_type="systematic")


def perturb_curves_statistic(curves, sampler:GaussianSampler, key=None):
    return perturb_curves(curves, sampler, key=key, perturbation_type="statistical")
