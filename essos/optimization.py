import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad
from functools import partial
from essos.coils import Curves, Coils
from scipy.optimize import least_squares, minimize
from essos.surfaces import SurfaceRZFourier

def optimize_loss_function(func, initial_dofs, coils, tolerance_optimization=1e-4, maximum_function_evaluations=30, method='L-BFGS-B', **kwargs):
    """Minimize a coil loss with ``scipy.optimize.least_squares``.

    Gradients are finite differences (``diff_step=1e-4``). ``method`` is only
    used by the commented ``minimize`` alternative below and has no effect.

    Args:
        func: Loss called as ``func(dofs, dofs_curves=..., currents_scale=...,
            nfp=..., n_segments=..., stellsym=..., **kwargs)``.
        initial_dofs: Starting vector: coil curve dofs, then currents, then
            optionally surface dofs.
        coils: Initial ``Coils``; supplies the symmetry and scaling arguments.
        tolerance_optimization: Used for both ``ftol`` and ``gtol``.
        maximum_function_evaluations: ``max_nfev`` of the solver.
        **kwargs: Passed to ``func``. ``surface_all`` selects whether the
            optimized surface is also returned.

    Returns:
        The optimized ``Coils``, followed by the optimized surface when it
        was optimized. ``None`` if the result could
        not be unpacked; the error is printed.
    """
    len_dofs_curves = len(jnp.ravel(coils.dofs_curves))
    nfp = coils.nfp
    stellsym = coils.stellsym
    n_segments = coils.n_segments
    dofs_curves_shape = coils.dofs_curves.shape
    currents_scale = coils.currents_scale
    
    loss_partial = partial(func, dofs_curves=coils.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym, **kwargs)
    
    ## Without JAX gradients, using finite differences
    result = least_squares(loss_partial, x0=initial_dofs, verbose=2, diff_step=1e-4,
                            ftol=tolerance_optimization, gtol=tolerance_optimization,
                            xtol=1e-14, max_nfev=maximum_function_evaluations)
    
    ## With JAX gradients
    ##jac_loss_partial = jit(grad(loss_partial))
    # result = least_squares(loss_partial, x0=initial_dofs, verbose=2, jac=jac_loss_partial,
    #                        ftol=tolerance_optimization, gtol=tolerance_optimization,
    #                        xtol=1e-14, max_nfev=maximum_function_evaluations)
    ##result = minimize(loss_partial, x0=initial_dofs, jac=jac_loss_partial, method=method,
    ##                  tol=tolerance_optimization, options={'maxiter': maximum_function_evaluations, 'disp': True, 'gtol': 1e-14, 'ftol': 1e-14})
    
    dofs_curves = jnp.reshape(result.x[:len_dofs_curves], (dofs_curves_shape))
    try:
        if len(initial_dofs) == len(coils.x):
            dofs_currents = result.x[len_dofs_curves:]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents*coils.currents_scale)
            return new_coils
        elif 'surface_all' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['surface_all'].x):
            surface_all = kwargs['surface_all']
            dofs_currents = result.x[len_dofs_curves:-len(surface_all.x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta,mpol=surface_all.mpol,ntor=surface_all.ntor)
            new_surface.dofs = result.x[-len(surface_all.x):]
            return new_coils, new_surface
    except Exception as e:
        jax.debug.print("Error: {}", e)
        return None
