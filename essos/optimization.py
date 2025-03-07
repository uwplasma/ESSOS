import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, grad
from functools import partial
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from essos.surfaces import BdotN_over_B, BdotN
from essos.coils import Curves, Coils
from scipy.optimize import least_squares

def loss_particle_drift(field, particles, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5):
    tracing = Tracing(field=field, model='GuidingCenter', particles=particles,
                    maxtime=maxtime, timesteps=num_steps, tol_step_size=trace_tolerance)
    trajectories = tracing.trajectories
    
    R_axis = jnp.mean(jnp.sqrt(vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(field.coils.dofs_curves)))
    
    # radial_drift=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(trajectories[:,:,0])+jnp.square(trajectories[:,:,1]))-R_axis)
    #                      +jnp.square(trajectories[:,:,2]))
    # radial_drift=jnp.sum(jnp.diff(radial_drift,axis=1),axis=1)/num_steps
    
    # angular_drift = jnp.arctan2(trajectories[:, :, 2], jnp.sqrt(trajectories[:, :, 0]**2+trajectories[:, :, 1]**2)-R_axis)
    # angular_drift=(jnp.sum(jnp.diff(angular_drift,axis=1),axis=1))/num_steps
    
    # return jnp.ravel(2./jnp.pi*jnp.absolute(jnp.arctan(radial_drift/angular_drift)))
    return jnp.ravel(jnp.abs(jnp.sqrt(jnp.square(trajectories[:,:,0])+jnp.square(trajectories[:,:,1]))-R_axis))+jnp.ravel(jnp.abs(trajectories[:,:,2]))
    ## return jnp.ravel(jnp.abs(radial_drift)) # This returns NaN gradients

@partial(jit, static_argnums=(0))
def loss_coil_length(field):
    return jnp.ravel(field.coils.length)

@partial(jit, static_argnums=(0))
def loss_coil_curvature(field):
    return jnp.mean(field.coils.curvature, axis=1)

@partial(jit, static_argnums=(0, 1))
def loss_normB_axis(field, npoints=15):
    R_axis = jnp.mean(jnp.sqrt(vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(field.coils.dofs_curves)))
    phi_array = jnp.linspace(0, 2 * jnp.pi, npoints)
    B_axis = vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return B_axis

@partial(jit, static_argnums=(1, 3, 5, 6, 7, 8, 9, 10, 11))
def loss_optimize_coils_for_particle_confinement(x, particles, dofs_curves, nfp, scale_current=1e5,
                                                 n_segments=60, stellsym=True, target_B_on_axis=5.7, maxtime=1e-5,
                                                 max_coil_length=22, num_steps=300, trace_tolerance=1e-6):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], dofs_curves.shape)
    dofs_currents = x[len_dofs_curves_ravelled:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*scale_current)
    field = BiotSavart(coils)
    
    particles_drift = loss_particle_drift(field, particles, maxtime, num_steps, trace_tolerance)
    coil_length = loss_coil_length(field)
    normB_axis = loss_normB_axis(field)

    loss = jnp.concatenate((jnp.abs(normB_axis-target_B_on_axis), jnp.abs(coil_length-max_coil_length), jnp.abs(particles_drift)))
    return jnp.sum(loss)

@partial(jit, static_argnums=(1, 3, 4, 5, 6))
def loss_BdotN(x, vmec, dofs_curves, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], (dofs_curves.shape))
    dofs_currents = x[len_dofs_curves_ravelled:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents)
    field = BiotSavart(coils)
    
    bdotn_over_b = BdotN_over_B(vmec.surface, field)
    coil_length = loss_coil_length(field)
    coil_curvature = loss_coil_curvature(field)
    
    bdotn_over_b_loss = jnp.sum(jnp.abs(bdotn_over_b))
    coil_length_loss    = jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))
    coil_curvature_loss = jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))
    
    # from jax.debug import print as jprint
    # jprint("#######################")
    # jprint("{}",jnp.sum(jnp.abs(bdotn_over_b)))
    # jprint("{}",coil_length)
    # jprint("{}",coil_curvature)
    # jprint("{}", target_coil_curvature)
    # jprint("{}",3e0*jnp.sum(bdotn_over_b)+jnp.sum(jnp.abs(coil_length-target_coil_length))+jnp.sum(jnp.abs(coil_curvature-target_coil_curvature)))

    return bdotn_over_b_loss+coil_length_loss+coil_curvature_loss

def optimize_loss_function(func, coils, tolerance_optimization=1e-4, maximum_function_evaluations=30, **kwargs):
    dofs = coils.x
    len_dofs_curves = len(jnp.ravel(coils.dofs_curves))
    nfp = coils.nfp
    stellsym = coils.stellsym
    n_segments = coils.n_segments
    dofs_curves_shape = coils.dofs_curves.shape
    
    loss_partial = partial(func, dofs_curves=coils.dofs_curves, nfp=nfp, n_segments=n_segments, stellsym=stellsym, **kwargs)
    
    ## Without JAX gradients, using finite differences
    # result = least_squares(loss_partial, x0=dofs, verbose=2, diff_step=1e-2,
    #                        ftol=tolerance_optimization, gtol=tolerance_optimization,
    #                        xtol=1e-14, max_nfev=maximum_function_evaluations)
    
    ## With JAX gradients
    jac_loss_partial = jit(grad(loss_partial))
    # result = least_squares(loss_partial, x0=dofs, verbose=2, jac=jac_loss_partial,
    #                        ftol=tolerance_optimization, gtol=tolerance_optimization,
    #                        xtol=1e-14, max_nfev=maximum_function_evaluations)
    from scipy.optimize import minimize
    result = minimize(loss_partial, x0=dofs, jac=jac_loss_partial, method='L-BFGS-B',
                      tol=tolerance_optimization, options={'maxiter': maximum_function_evaluations, 'disp': True})
    
    dofs_curves = jnp.reshape(result.x[:len_dofs_curves], (dofs_curves_shape))
    dofs_currents = result.x[len_dofs_curves:]
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    new_coils = Coils(curves=curves, currents=dofs_currents*coils.currents_scale)
    return new_coils

def optimize_coils_for_particle_confinement(coils, particles, target_B_on_axis=5.7, max_coil_length=22,
                                            maxtime=5e-6, num_steps=500, trace_tolerance=1e-5, tolerance_optimization=1e-4,
                                            maximum_function_evaluations=30):
    return optimize_loss_function(loss_optimize_coils_for_particle_confinement, coils,
                           tolerance_optimization=tolerance_optimization, particles=particles, scale_current=coils.currents_scale,
                           maximum_function_evaluations=maximum_function_evaluations,
                           target_B_on_axis=target_B_on_axis, max_coil_length=max_coil_length,
                           maxtime=maxtime, num_steps=num_steps, trace_tolerance=trace_tolerance)

def optimize_coils_for_vmec_surface(vmec, coils, tolerance_optimization=1e-10,
                                    maximum_function_evaluations=30,
                                    max_coil_length=42, max_coil_curvature=0.1):
    return optimize_loss_function(loss_BdotN, coils, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, vmec=vmec,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)
                                    