import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from essos.dynamics import Tracing
from essos.fields import BiotSavart, near_axis
from essos.surfaces import BdotN_over_B, BdotN
from essos.coils import Curves, Coils

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

@partial(jit, static_argnums=(1, 4, 5, 6, 7, 8))
def loss_coils_for_nearaxis(x, field_nearaxis, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], (dofs_curves.shape))
    dofs_currents = x[len_dofs_curves_ravelled:]
    
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
    
    coil_length = loss_coil_length(field)
    coil_curvature = loss_coil_curvature(field)
    
    B_difference_loss = jnp.sum(jnp.abs(jnp.array(B_coils)-jnp.array(B_nearaxis)))
    gradB_difference_loss = jnp.sum(jnp.abs(jnp.array(gradB_coils)-jnp.array(gradB_nearaxis)))
    coil_length_loss    = 1e3*jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))
    coil_curvature_loss = 1e3*jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))
    
    return B_difference_loss+gradB_difference_loss+coil_length_loss+coil_curvature_loss

@partial(jit, static_argnums=(0, 1))
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
    
@partial(jit, static_argnums=(1, 4, 5, 6, 7, 8))
def loss_coils_and_nearaxis(x, field_nearaxis, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], (dofs_curves.shape))
    len_dofs_nearaxis = len(field_nearaxis.x)
    dofs_currents = x[len_dofs_curves_ravelled:-len_dofs_nearaxis]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(x[-len_dofs_nearaxis:], field_nearaxis)
    
    coil_length = loss_coil_length(field)
    coil_curvature = loss_coil_curvature(field)
    
    elongation = new_field_nearaxis.elongation
    iota = new_field_nearaxis.iota
    
    B_difference, gradB_difference = difference_B_gradB_onaxis(new_field_nearaxis, field)
    B_difference_loss = 3*jnp.sum(jnp.abs(B_difference))
    gradB_difference_loss = jnp.sum(jnp.abs(gradB_difference))
    
    coil_length_loss    = 1e3*jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))
    coil_curvature_loss = 1e3*jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))
    elongation_loss = jnp.sum(jnp.abs(elongation))
    iota_loss = 30/jnp.abs(iota)
    
    return B_difference_loss+gradB_difference_loss+coil_length_loss+coil_curvature_loss+elongation_loss+iota_loss

def loss_particle_drift(field, particles, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenter'):
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timesteps=num_steps, tol_step_size=trace_tolerance)
    trajectories = tracing.trajectories
    R_axis = jnp.mean(jnp.sqrt(vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(field.coils.dofs_curves)))
    radial_factor = jnp.sqrt(jnp.square(trajectories[:,:,0])+jnp.square(trajectories[:,:,1]))-R_axis
    vertical_factor = trajectories[:,:,2]
    radial_drift=jnp.square(radial_factor)+jnp.square(vertical_factor)
    radial_drift=jnp.sum(jnp.diff(radial_drift,axis=1),axis=1)/num_steps
    angular_drift = jnp.arctan2(trajectories[:, :, 2]+1e-10, jnp.sqrt(trajectories[:, :, 0]**2+trajectories[:, :, 1]**2)-R_axis)
    angular_drift=(jnp.sum(jnp.diff(angular_drift,axis=1),axis=1))/num_steps
    return jnp.concatenate((jnp.max(radial_drift)*jnp.ravel(2./jnp.pi*jnp.abs(jnp.arctan(radial_drift/(angular_drift+1e-10)))), jnp.ravel(jnp.abs(radial_drift)), jnp.ravel(jnp.abs(vertical_factor))))

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

@partial(jit, static_argnums=(1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
def loss_optimize_coils_for_particle_confinement(x, particles, dofs_curves, currents_scale, nfp, max_coil_curvature=0.5,
                                                 n_segments=60, stellsym=True, target_B_on_axis=5.7, maxtime=1e-5,
                                                 max_coil_length=22, num_steps=30, trace_tolerance=1e-5, model='GuidingCenter'):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], dofs_curves.shape)
    dofs_currents = x[len_dofs_curves_ravelled:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    
    particles_drift_loss = loss_particle_drift(field, particles, maxtime, num_steps, trace_tolerance, model=model)
    normB_axis = loss_normB_axis(field)
    normB_axis_loss = jnp.abs(normB_axis-target_B_on_axis)
    coil_length = loss_coil_length(field)
    coil_length_loss = jnp.array([jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))])
    coil_curvature = loss_coil_curvature(field)
    coil_curvature_loss = jnp.array([jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))])

    loss = jnp.concatenate((normB_axis_loss, coil_length_loss, particles_drift_loss, coil_curvature_loss))
    return jnp.sum(loss)

@partial(jit, static_argnums=(1, 4, 5, 6, 7))
def loss_BdotN(x, vmec, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], (dofs_curves.shape))
    dofs_currents = x[len_dofs_curves_ravelled:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    
    bdotn_over_b = BdotN_over_B(vmec.surface, field)
    coil_length = loss_coil_length(field)
    coil_curvature = loss_coil_curvature(field)
    
    bdotn_over_b_loss = jnp.sum(jnp.abs(bdotn_over_b))
    coil_length_loss    = jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))
    coil_curvature_loss = jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))
    
    return bdotn_over_b_loss+coil_length_loss+coil_curvature_loss