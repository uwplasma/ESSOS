import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import fori_loop
from functools import partial
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from essos.surfaces import BdotN_over_B, BdotN
from essos.coils import Curves, Coils
from essos.optimization import new_nearaxis_from_x_and_old_nearaxis

@partial(jit, static_argnums=(1, 2, 4, 5, 6, 7, 8))
def loss_coils_for_nearaxis(x, field_nearaxis, dofs_curves_shape, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    dofs_curves_size = dofs_curves_shape[0]*dofs_curves_shape[1]*dofs_curves_shape[2]
    dofs_curves = jnp.reshape(x[:dofs_curves_size], (dofs_curves_shape))
    dofs_currents = x[dofs_curves_size:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    
    Raxis = field_nearaxis.R0
    Zaxis = field_nearaxis.Z0
    phi = field_nearaxis.phi
    Xaxis = Raxis*jnp.cos(phi)
    Yaxis = Raxis*jnp.sin(phi)
    points = jnp.array([Xaxis, Yaxis, Zaxis])
    B_nearaxis = field_nearaxis.B_axis.T
    B_coils = vmap(field.B)(points.T)
    
    gradB_nearaxis = field_nearaxis.grad_B_axis.T
    gradB_coils = vmap(field.dB_by_dX)(points.T)
    
    B_difference_loss = jnp.sum(jnp.abs(jnp.array(B_coils)-jnp.array(B_nearaxis)))
    gradB_difference_loss = jnp.sum(jnp.abs(jnp.array(gradB_coils)-jnp.array(gradB_nearaxis)))
    coil_length_loss = 1e3*jnp.max(loss_coil_length(field, max_coil_length))
    coil_curvature_loss = 1e3*jnp.max(loss_coil_curvature(field, max_coil_curvature))
    
    
    return B_difference_loss+gradB_difference_loss+coil_length_loss+coil_curvature_loss

# @partial(jit, static_argnums=(0, 1))
def difference_B_gradB_onaxis(nearaxis_field, coils_field):
    Raxis = nearaxis_field.R0
    Zaxis = nearaxis_field.Z0
    phi = nearaxis_field.phi
    Xaxis = Raxis*jnp.cos(phi)
    Yaxis = Raxis*jnp.sin(phi)
    points = jnp.array([Xaxis, Yaxis, Zaxis])
    B_nearaxis = nearaxis_field.B_axis.T
    B_coils = vmap(coils_field.B)(points.T)
    
    gradB_nearaxis = nearaxis_field.grad_B_axis.T
    gradB_coils = vmap(coils_field.dB_by_dX)(points.T)
    
    return jnp.array(B_coils)-jnp.array(B_nearaxis), jnp.array(gradB_coils)-jnp.array(gradB_nearaxis)
    
@partial(jit, static_argnums=(1, 2, 4, 5, 6, 7, 8))
def loss_coils_and_nearaxis(x, field_nearaxis, dofs_curves_shape, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    dofs_curves_size = dofs_curves_shape[0]*dofs_curves_shape[1]*dofs_curves_shape[2]
    dofs_curves = jnp.reshape(x[:dofs_curves_size], (dofs_curves_shape))
    len_dofs_nearaxis = len(field_nearaxis.x)
    dofs_currents = x[dofs_curves_size:-len_dofs_nearaxis]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(x[-len_dofs_nearaxis:], field_nearaxis)
    
    elongation = new_field_nearaxis.elongation
    iota = new_field_nearaxis.iota
    
    B_difference, gradB_difference = difference_B_gradB_onaxis(new_field_nearaxis, field)
    B_difference_loss = 3*jnp.sum(jnp.abs(B_difference))
    gradB_difference_loss = jnp.sum(jnp.abs(gradB_difference))
    
    coil_length_loss = 1e3*jnp.max(loss_coil_length(field, max_coil_length))
    coil_curvature_loss = 1e3*jnp.max(loss_coil_curvature(field, max_coil_curvature))
    elongation_loss = jnp.sum(jnp.abs(elongation))
    iota_loss = 30/jnp.abs(iota)
    
    return B_difference_loss+gradB_difference_loss+coil_length_loss+coil_curvature_loss+elongation_loss+iota_loss

def loss_particle_drift(field, particles, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenter'):
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timesteps=num_steps, tol_step_size=trace_tolerance)
    trajectories = tracing.trajectories

    R_axis = jnp.mean(jnp.sqrt(vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(field.coils.dofs_curves)))
    radial_factor = jnp.sqrt(trajectories[:, :, 0]**2 + trajectories[:,:,1]**2)-R_axis
    vertical_factor = trajectories[:,:,2]
    
    radial_drift = radial_factor**2 + vertical_factor**2
    # radial_drift = jnp.sqrt(radial_drift)
    radial_drift = jnp.mean(jnp.diff(radial_drift, axis=1), axis=1)
    
    angular_drift = jnp.arctan2(vertical_factor, radial_factor+1e-10)
    angular_drift = jnp.mean(jnp.diff(angular_drift, axis=1), axis=1)
    
    return jnp.concatenate((jnp.max(radial_drift)*jnp.ravel(2./jnp.pi*jnp.abs(jnp.arctan(radial_drift/(angular_drift+1e-10)))), jnp.ravel(jnp.abs(radial_drift)), jnp.ravel(jnp.abs(vertical_factor))))
    # return jnp.concatenate((jnp.max(radial_drift)*jnp.ravel(2./jnp.pi*jnp.abs(jnp.arctan(radial_drift/(angular_drift+1e-10)))), jnp.ravel(jnp.abs(radial_drift))))
    # return jnp.concatenate((jnp.ravel(jnp.abs(angular_drift)), jnp.ravel(jnp.abs(radial_drift))))
    # return jnp.concatenate((jnp.ravel(jnp.abs(vertical_factor)),))

@partial(jit, static_argnames=['max_coil_length'])
def loss_coil_length(coils, max_coil_length):
    return jnp.square((coils.length-max_coil_length)/max_coil_length)

@partial(jit, static_argnames=['max_coil_curvature'])
def loss_coil_curvature(coils, max_coil_curvature):
    pointwise_curvature_loss = jnp.square(jnp.maximum(coils.curvature-max_coil_curvature, 0))
    return jnp.mean(pointwise_curvature_loss, axis=1)

@partial(jit, static_argnames=['min_separation'])
def loss_coil_separation(coils, min_separation):
    i_vals, j_vals = jnp.triu_indices(len(coils), k=1)

    def pair_loss(i, j):
        gamma_i = coils.gamma[i]
        gamma_j = coils.gamma[j]
        dists = jnp.linalg.norm(gamma_i[:, None, :] - gamma_j[None, :, :], axis=2)
        penalty = jnp.maximum(0, min_separation - dists)
        return jnp.mean(jnp.square(penalty))

    losses = jax.vmap(pair_loss)(i_vals, j_vals)
    return jnp.sum(losses)

# @partial(jit, static_argnames=['target_B_on_axis', 'npoints'])
def loss_normB_axis(field, target_B_on_axis, npoints=15):
    R_axis = jnp.mean(jnp.sqrt(vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(field.coils.dofs_curves)))
    phi_array = jnp.linspace(0, 2 * jnp.pi, npoints)
    B_axis = vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return jnp.square(B_axis-target_B_on_axis)

@partial(jit, static_argnums=(1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
def loss_optimize_coils_for_particle_confinement(x, particles, dofs_curves_shape, currents_scale, nfp, max_coil_curvature=0.5,
                                                 n_segments=60, stellsym=True, target_B_on_axis=5.7, maxtime=1e-5,
                                                 max_coil_length=22, num_steps=30, trace_tolerance=1e-5, model='GuidingCenter',
                                                 coil_length_loss_factor=1, coil_curvature_loss_factor=1):
    
    dofs_curves_size = dofs_curves_shape[0]*dofs_curves_shape[1]*dofs_curves_shape[2]
    dofs_curves = jnp.reshape(x[:dofs_curves_size], (dofs_curves_shape))
    dofs_currents = x[dofs_curves_size:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    
    particles_drift_loss = jnp.sum(loss_particle_drift(field, particles, maxtime, num_steps, trace_tolerance, model=model))
    normB_axis_loss = jnp.sum(loss_normB_axis(field, target_B_on_axis))
    coil_length_loss = coil_length_loss_factor * jnp.sum(loss_coil_length(coils, max_coil_length))
    coil_curvature_loss = coil_curvature_loss_factor * jnp.sum(loss_coil_curvature(coils, max_coil_curvature))
    coils_separation_loss = jnp.sum(loss_coil_separation(coils, 0.5))

    return normB_axis_loss + coil_length_loss + particles_drift_loss + coil_curvature_loss + coils_separation_loss

@partial(jit, static_argnums=(1, 2, 4, 5, 6, 7, 8))
def loss_BdotN(x, vmec, dofs_curves_shape, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1,
               coil_length_loss_factor=1, coil_curvature_loss_factor=1):
    
    dofs_curves_size = dofs_curves_shape[0]*dofs_curves_shape[1]*dofs_curves_shape[2]
    dofs_curves = jnp.reshape(x[:dofs_curves_size], (dofs_curves_shape))
    dofs_currents = x[dofs_curves_size:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    
    coil_length_loss = coil_length_loss_factor * jnp.sum(loss_coil_length(coils, max_coil_length))
    coil_curvature_loss = coil_curvature_loss_factor * jnp.sum(loss_coil_curvature(coils, max_coil_curvature))
    bdotn_over_b_loss = jnp.sum(jnp.square(BdotN_over_B(vmec.surface, field)))
    
    return bdotn_over_b_loss+coil_length_loss+coil_curvature_loss