"""Fixed-Fourier Boozer adapter for the shared independent native GPU solver."""
import jax.numpy as jnp
from .rk_gpu import prepare_gpu_solver
from .rk_gpu_utils import _padded_spline, _KernelField, _kernel_rhs, _select_one


def prepare_orbit_solver(field, mass, charge, speed, times, dt0, rtol, atol, max_steps, *,
                         method="dopri8", adaptive=True, interpret=False):
    if field.table is not None or field.sine_coef is not None:
        raise ValueError("The CUDA adapter requires a fixed cosine Fourier Boozer field")
    radial, b = _padded_spline(field.r_knots, field.b_coef)
    flux, profiles = _padded_spline(field.s_knots, field.profile_coef)
    modes = b.shape[1]
    arrays = (radial, b, flux, profiles, jnp.pad(field.xm, (0,modes-len(field.xm))),
              jnp.pad(field.xn, (0,modes-len(field.xn))))
    scalar = lambda y,i: _select_one(y,i,0).reshape(())
    def factory(refs, point, moment):
        model = _KernelField(refs, float(field.psi0))
        evaluate = _kernel_rhs(model, point, moment, mass, charge)
        def estimate(endpoint, error, speed):
            a,b,c,d = (scalar(error,i) for i in range(4))
            return jnp.maximum(jnp.sqrt(a*a+b*b),jnp.maximum(jnp.abs(c),jnp.abs(d)/speed))
        return evaluate, estimate, lambda y: 1.-scalar(y,0)**2-scalar(y,1)**2
    return prepare_gpu_solver(arrays,factory,speed,times,dt0,rtol,atol,max_steps,
                              method=method,adaptive=adaptive,interpret=interpret)
