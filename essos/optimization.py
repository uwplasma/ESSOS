import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad
from functools import partial
from essos.coils import Curves, Coils
from scipy.optimize import least_squares, minimize
from essos.objective_functions import (loss_optimize_coils_for_particle_confinement,
                                       loss_BdotN, loss_coils_for_nearaxis, loss_coils_and_nearaxis,
                                       new_nearaxis_from_x_and_old_nearaxis)

def optimize_loss_function(func, initial_dofs, coils, tolerance_optimization=1e-4, maximum_function_evaluations=30, **kwargs):
    len_dofs_curves = len(jnp.ravel(coils.dofs_curves))
    nfp = coils.nfp
    stellsym = coils.stellsym
    n_segments = coils.n_segments
    dofs_curves_shape = coils.dofs_curves.shape
    currents_scale = coils.currents_scale
    
    loss_partial = partial(func, dofs_curves=coils.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym, **kwargs)
    
    ## Without JAX gradients, using finite differences
    # result = least_squares(loss_partial, x0=initial_dofs, verbose=2, diff_step=1e-2,
    #                        ftol=tolerance_optimization, gtol=tolerance_optimization,
    #                        xtol=1e-14, max_nfev=maximum_function_evaluations)
    
    ## With JAX gradients
    jac_loss_partial = jit(grad(loss_partial))
    # result = least_squares(loss_partial, x0=initial_dofs, verbose=2, jac=jac_loss_partial,
    #                        ftol=tolerance_optimization, gtol=tolerance_optimization,
    #                        xtol=1e-14, max_nfev=maximum_function_evaluations)
    result = minimize(loss_partial, x0=initial_dofs, jac=jac_loss_partial, method='L-BFGS-B',
                      tol=tolerance_optimization, options={'maxiter': maximum_function_evaluations, 'disp': True, 'gtol': 1e-14, 'ftol': 1e-14})
    
    dofs_curves = jnp.reshape(result.x[:len_dofs_curves], (dofs_curves_shape))
    try:
        if len(initial_dofs) == len(coils.x):
            dofs_currents = result.x[len_dofs_curves:]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents*coils.currents_scale)
            return new_coils
        elif len(initial_dofs) == len(coils.x)+len(kwargs['field_nearaxis'].x):
            dofs_currents = result.x[len_dofs_curves:-len(kwargs['field_nearaxis'].x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents*coils.currents_scale)
            new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(result.x[-len(kwargs['field_nearaxis'].x):], kwargs['field_nearaxis'])
            return new_coils, new_field_nearaxis
    except Exception as e:
        jax.debug.print("Error: {}", e)
        return None

def optimize_coils_for_particle_confinement(coils, particles, target_B_on_axis=5.7, max_coil_length=22, model='GuidingCenter',
                                            maxtime=5e-6, num_steps=500, trace_tolerance=1e-5, tolerance_optimization=1e-4,
                                            maximum_function_evaluations=30, max_coil_curvature=0.1):
    return optimize_loss_function(loss_optimize_coils_for_particle_confinement, initial_dofs=coils.x, coils=coils,
                           tolerance_optimization=tolerance_optimization, particles=particles,
                           maximum_function_evaluations=maximum_function_evaluations, max_coil_curvature=max_coil_curvature,
                           target_B_on_axis=target_B_on_axis, max_coil_length=max_coil_length, model=model,
                           maxtime=maxtime, num_steps=num_steps, trace_tolerance=trace_tolerance)

def optimize_coils_for_vmec_surface(vmec, coils, tolerance_optimization=1e-10,
                                    maximum_function_evaluations=30,
                                    max_coil_length=42, max_coil_curvature=0.1):
    return optimize_loss_function(loss_BdotN, initial_dofs=coils.x, coils=coils, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, vmec=vmec,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)
    
def optimize_coils_for_nearaxis(field_nearaxis, coils, tolerance_optimization=1e-10,
                                    maximum_function_evaluations=30,
                                    max_coil_length=42, max_coil_curvature=0.1):
    return optimize_loss_function(loss_coils_for_nearaxis, initial_dofs=coils.x, coils=coils, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, field_nearaxis=field_nearaxis,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)

def optimize_coils_and_nearaxis(field_nearaxis, coils, tolerance_optimization=1e-10,
                                    maximum_function_evaluations=30,
                                    max_coil_length=42, max_coil_curvature=0.1):
    initial_dofs = jnp.concatenate((coils.x, field_nearaxis.x))
    return optimize_loss_function(loss_coils_and_nearaxis, initial_dofs=initial_dofs, coils=coils, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, field_nearaxis=field_nearaxis,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)
