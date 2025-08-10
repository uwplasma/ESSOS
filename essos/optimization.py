import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad
from functools import partial
from essos.coils import Curves, Coils
from scipy.optimize import least_squares, minimize
from essos.fields import near_axis
from essos.surfaces import SurfaceRZFourier

def new_nearaxis_from_x_and_old_nearaxis(new_field_nearaxis_x, field_nearaxis):
    len_rc = len(field_nearaxis.rc)
    len_zs = len(field_nearaxis.zs)
    # # keeping the first rc and zs the same
    # new_field_nearaxis_rc = jnp.concatenate((jnp.array([field_nearaxis.rc[0]]),new_field_nearaxis_x[:len_rc][1:]))
    # new_field_nearaxis_zs = jnp.concatenate((jnp.array([field_nearaxis.zs[0]]),new_field_nearaxis_x[len_rc:len_rc+len_zs][1:]))
    new_field_nearaxis_rc = new_field_nearaxis_x[:len_rc]
    new_field_nearaxis_zs = new_field_nearaxis_x[len_rc:len_rc+len_zs]
    new_field_nearaxis_etabar = new_field_nearaxis_x[-1]
    
    new_field_nearaxis = near_axis(rc=new_field_nearaxis_rc, zs=new_field_nearaxis_zs, etabar=new_field_nearaxis_etabar,
                                    B0=field_nearaxis.B0, sigma0=field_nearaxis.sigma0, I2=field_nearaxis.I2,
                                    nphi=field_nearaxis.nphi, spsi=field_nearaxis.spsi, sG=field_nearaxis.sG, nfp=field_nearaxis.nfp)
    return new_field_nearaxis

def optimize_loss_function(func, initial_dofs, coils, tolerance_optimization=1e-4, maximum_function_evaluations=30, method='L-BFGS-B', **kwargs):
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
        elif 'field_nearaxis' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['field_nearaxis'].x):
            dofs_currents = result.x[len_dofs_curves:-len(kwargs['field_nearaxis'].x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(result.x[-len(kwargs['field_nearaxis'].x):], kwargs['field_nearaxis'])
            return new_coils, new_field_nearaxis
        elif 'surface_all' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['surface_all'].x):
            surface_all = kwargs['surface_all']
            dofs_currents = result.x[len_dofs_curves:-len(surface_all.x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
            new_surface.dofs = result.x[-len(surface_all.x):]
            return new_coils, new_surface
        elif 'surface_all' in kwargs and 'field_nearaxis' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['surface_all'].x) + len(kwargs['field_nearaxis'].x):
            surface_all = kwargs['surface_all']
            field_nearaxis = kwargs['field_nearaxis']
            dofs_currents = result.x[len_dofs_curves:-len(surface_all.x)-len(field_nearaxis.x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
            new_surface.dofs = result.x[-len(surface_all.x)-len(field_nearaxis.x):-len(field_nearaxis.x)]
            new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(result.x[-len(field_nearaxis.x):], field_nearaxis)
            return new_coils, new_surface, new_field_nearaxis
    except Exception as e:
        jax.debug.print("Error: {}", e)
        return None
