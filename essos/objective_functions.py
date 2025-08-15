import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from essos.surfaces import BdotN_over_B, BdotN
from essos.coils import Curves, Coils
from essos.optimization import new_nearaxis_from_x_and_old_nearaxis
import optax


def field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments=60, stellsym=True):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], dofs_curves.shape)
    dofs_currents = x[len_dofs_curves_ravelled:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    return field



@partial(jit, static_argnums=(1, 4, 5, 6, 7, 8))
def loss_coils_for_nearaxis(x, field_nearaxis, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    field=field_from_dofs(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp,n_segments=n_segments, stellsym=stellsym)               

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
    
    coil_length = loss_coil_length(x,dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_length=max_coil_length)
    coil_curvature = loss_coil_curvature(x,dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_curvature=max_coil_curvature)
    
    
    B_difference_loss = jnp.sum(jnp.abs(jnp.array(B_coils)-jnp.array(B_nearaxis)))
    gradB_difference_loss = jnp.sum(jnp.abs(jnp.array(gradB_coils)-jnp.array(gradB_nearaxis)))
    coil_length_loss    = 1e3*jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))
    coil_curvature_loss = 1e3*jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))
    
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
    
@partial(jit, static_argnums=(1, 4, 5, 6, 7, 8))
def loss_coils_and_nearaxis(x, field_nearaxis, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    #len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    len_dofs_nearaxis = len(field_nearaxis.x)          
    field=field_from_dofs(x[:-len_dofs_nearaxis],dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp,n_segments=n_segments, stellsym=stellsym)               
    new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(x[-len_dofs_nearaxis:], field_nearaxis)
    
    coil_length = loss_coil_length(x[:-len_dofs_nearaxis],dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_length=max_coil_length)
    coil_curvature = loss_coil_curvature(x[:-len_dofs_nearaxis],dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_curvature=max_coil_curvature)
    
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


def loss_particle_radial_drift(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    #Ideally here one would differentiate in time through diffrax !TODO
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,0])+jnp.square(xyz[:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,2]-Z_axis+1.e-12))
    v_r_cross=jnp.diff(r_cross,axis=1)#/tracing.times_to_trace*tracing.maxtime     
    return jnp.ravel((jnp.sum(jnp.square(jnp.average(v_r_cross,axis=1)))))


def loss_particle_alpha_drift(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',target=-1000.,boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    #def theta(x,R_axis=R_axis,Z_axis=Z_axis):
    #    return jnp.arctan2(x[2]-Z_axis+1.e-12, jnp.sqrt(x[0]**2+x[1]**2)-R_axis+1.e-12)
    #def phi(x):
    #    return jnp.arctan2(x[1], x[0])
    ##AbsB = vmap(self.field.AbsB)(xyz)
    ##B_contravariant=vmap(self.field.B_contravariant,in_axes=(0))(xyz)
    #Ideally here one would differentiate in time through diffrax !TODO                    
    #grad_theta=vmap(jax.jacfwd(theta,argnums=0),in_axes=0)(xyz)
    #grad_phi=vmap(jax.jacfwd(phi,argnums=0),in_axes=0)(xyz)
    #v_theta=jnp.tensordot(v_xyz,grad_theta,axes=(1,1))
    #v_alpha=v_theta-jnp.tensordot(B_contravariant,grad_theta,axes=(1,1))/jnp.tensordot(B_contravariant,grad_phi,axes=(1,1))*jnp.tensordot(v_xyz,grad_phi,axes=(1,1))                
    theta=jnp.arctan2(xyz[:,2]-Z_axis+1.e-12, jnp.sqrt(xyz[:,0]**2+xyz[:,1]**2)-R_axis+1.e-12)
    v_theta=jnp.diff(theta,axis=1)#/tracing.times_to_trace*tracing.maxtime                               
    return jnp.sum(jnp.square(jnp.average(v_theta,axis=1)))  

def loss_particle_gamma_c(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym) 
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    #def theta(x,R_axis=R_axis,Z_axis=Z_axis):
    #    return jnp.arctan2(x[2]-Z_axis+1.e-12, jnp.sqrt(x[0]**2+x[1]**2)-R_axis+1.e-12)
    #def phi(x):
    #    return jnp.arctan2(x[1], x[0])
    ##AbsB = vmap(self.field.AbsB)(xyz)
    ##B_contravariant=vmap(self.field.B_contravariant,in_axes=(0))(xyz)
    #Ideally here one would differentiate in time through diffrax !TODO                    
    #grad_theta=vmap(jax.jacfwd(theta,argnums=0),in_axes=0)(xyz)
    #grad_phi=vmap(jax.jacfwd(phi,argnums=0),in_axes=0)(xyz)
    #v_theta=jnp.tensordot(v_xyz,grad_theta,axes=(1,1))
    #v_alpha=v_theta-jnp.tensordot(B_contravariant,grad_theta,axes=(1,1))/jnp.tensordot(B_contravariant,grad_phi,axes=(1,1))*jnp.tensordot(v_xyz,grad_phi,axes=(1,1)) 
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,0])+jnp.square(xyz[:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,2]-Z_axis+1.e-12))
    v_r_cross=jnp.diff(r_cross,axis=1)#/tracing.times_to_trace*tracing.maxtime                        
    theta=jnp.arctan2(xyz[:,2]-Z_axis+1.e-12, jnp.sqrt(xyz[:,0]**2+xyz[:,1]**2)-R_axis+1.e-12)
    v_theta=jnp.diff(theta,axis=1)#/tracing.times_to_trace*tracing.maxtime                          
    #return jnp.sum(jnp.square((2./jnp.pi*jnp.absolute(jnp.arctan2(jnp.average(v_r_cross,axis=1),jnp.average(v_theta,axis=1))))))
    return jnp.max(2./jnp.pi*jnp.absolute(jnp.arctan2(jnp.average(v_r_cross,axis=1),jnp.average(v_theta,axis=1))))
    
def loss_particle_r_cross_final_new(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True,maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    return jnp.linlag.norm((jnp.average(r_cross,axis=1)))

def loss_particle_r_cross_max(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True,maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    return jnp.ravel(jnp.max(r_cross,axis=1))

def loss_lost_fraction(field, particles, maxtime=1e-5, num_steps=100, trace_tolerance=1e-5, model='GuidingCenterAdaptative',timestep=1.e-8,boundary=None):
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    lost_fraction = tracing.loss_fraction
    return lost_fraction

# @partial(jit, static_argnums=(0, 1))
def normB_axis(field, npoints=15,target_B_on_axis=5.7):
    R_axis=field.r_axis
    phi_array = jnp.linspace(0, 2 * jnp.pi, npoints)
    B_axis = vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return B_axis


# @partial(jit, static_argnums=(0))
#def loss_coil_length(field,max_coil_length=31):
#    coil_length=jnp.ravel(field.coils.length)
#    return jnp.array([jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))])

# @partial(jit, static_argnums=(0))
#def loss_coil_curvature(field,max_coil_curvature=0.4):
#    coil_curvature=jnp.mean(field.coils.curvature, axis=1)
#    return jnp.array([jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))])

# @partial(jit, static_argnums=(0))
def loss_coil_length(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,max_coil_length=31):
    field=field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments,stellsym)    
    coil_length=jnp.ravel(field.coils.length)
    return jnp.ravel(jnp.array([jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))]))

# @partial(jit, static_argnums=(0))
def loss_coil_curvature(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,max_coil_curvature=0.4):
    field=field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments,stellsym)
    coil_curvature=jnp.mean(field.coils.curvature, axis=1)
    return jnp.ravel(jnp.array([jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))]))

# @partial(jit, static_argnums=(0, 1))
def loss_normB_axis(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True, npoints=15,target_B_on_axis=5.7):
    field=field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments,stellsym)
    R_axis=field.r_axis
    phi_array = jnp.linspace(0, 2 * jnp.pi, npoints)
    B_axis = vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return jnp.ravel(jnp.absolute(B_axis-target_B_on_axis))

# @partial(jit, static_argnums=(0, 1))
def loss_normB_axis_average(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True, npoints=15,target_B_on_axis=5.7):
    field=field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments,stellsym)
    R_axis=field.r_axis
    phi_array = jnp.linspace(0, 2 * jnp.pi, npoints)
    B_axis = vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return jnp.absolute(jnp.average(B_axis)-target_B_on_axis)

@partial(jit, static_argnums=(1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14))
def loss_optimize_coils_for_particle_confinement(x, particles, dofs_curves, currents_scale, nfp, max_coil_curvature=0.5,
                                                 n_segments=60, stellsym=True, target_B_on_axis=5.7, maxtime=1e-5,
                                                 max_coil_length=22, num_steps=30, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp,n_segments=n_segments, stellsym=stellsym)

    particles_drift_loss = loss_particle_radial_drift(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp,n_segments=n_segments, stellsym=stellsym, particles=particles, maxtime=maxtime, num_steps=num_steps, trace_tolerance=trace_tolerance, model=model,boundary=boundary)
    normB_axis_loss = loss_normB_axis(x,dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,npoints=15,target_B_on_axis=target_B_on_axis)
    coil_length_loss = loss_coil_length(x,dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_length=max_coil_length)
    coil_curvature_loss = loss_coil_curvature(x,dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_curvature=max_coil_curvature)

    loss = jnp.concatenate((normB_axis_loss, coil_length_loss, coil_curvature_loss,particles_drift_loss))
    return jnp.sum(loss)


@partial(jit, static_argnums=(1, 4, 5, 6))
def loss_bdotn_over_b(x, vmec, dofs_curves, currents_scale, nfp, n_segments=60, stellsym=True):
    dofs_len = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:dofs_len], dofs_curves.shape)
    dofs_currents = x[dofs_len:]
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents * currents_scale)
    field = BiotSavart(coils)
    return jnp.sum(jnp.abs(BdotN_over_B(vmec.surface, field)))


@partial(jit, static_argnums=(1, 4, 5, 6, 7))
def loss_BdotN(x, vmec, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.1):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    
    bdotn_over_b = BdotN_over_B(vmec.surface, field)
    coil_length = loss_coil_length(x,dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_length=max_coil_length)
    coil_curvature = loss_coil_curvature(x,dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_curvature=max_coil_curvature)
    
    
    bdotn_over_b_loss = jnp.sum(jnp.abs(bdotn_over_b))
    coil_length_loss    = jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))
    coil_curvature_loss = jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))
    
    return bdotn_over_b_loss+coil_length_loss+coil_curvature_loss