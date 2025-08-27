import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from essos.dynamics import Tracing
from essos.fields import BiotSavart,BiotSavart_from_gamma
from essos.surfaces import BdotN_over_B, BdotN
from essos.coils import Curves, Coils,compute_curvature
from essos.optimization import new_nearaxis_from_x_and_old_nearaxis
from essos.constants import mu_0
from essos.coil_perturbation import perturb_curves_systematic, perturb_curves_statistic



def pertubred_field_from_dofs(x,key,sampler,dofs_curves,currents_scale,nfp,n_segments=60, stellsym=True):
    coils = perturbed_coils_from_dofs(x,key,sampler,dofs_curves,currents_scale,nfp=nfp,n_segments=n_segments, stellsym=stellsym)
    field = BiotSavart(coils)
    return field

def perturbed_coils_from_dofs(x,key,sampler,dofs_curves,currents_scale,nfp,n_segments=60, stellsym=True):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], dofs_curves.shape)
    dofs_currents = x[len_dofs_curves_ravelled:]
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    #Split once the key/seed given for one pertubred stellarator
    split_keys = jax.random.split(jax.random.key(key), 2)
    #Internally the following functions will then further split the two keys avoiding repeating keys
    perturb_curves_systematic(coils, sampler, key=split_keys[0])
    perturb_curves_statistic(coils, sampler, key=split_keys[1])
    return coils

def field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments=60, stellsym=True):
    coils = coils_from_dofs(x,dofs_curves,currents_scale,nfp=nfp,n_segments=n_segments, stellsym=stellsym)
    field = BiotSavart(coils)
    return field

def coils_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments=60, stellsym=True):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], dofs_curves.shape)
    dofs_currents = x[len_dofs_curves_ravelled:]
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    return coils

def curves_from_dofs(x,dofs_curves,nfp,n_segments=60, stellsym=True):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], dofs_curves.shape)
    dofs_currents = x[len_dofs_curves_ravelled:]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    return curves



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
    
def loss_particle_r_cross_final(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True,maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    return jnp.linalg.norm((jnp.average(r_cross,axis=1)))

def loss_particle_r_cross_max_constraint(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True,target_r=0.4,maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    #particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    return jnp.maximum(r_cross-target_r,0.0)


def loss_Br(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True,maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    #particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    fac_xy=jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))
    r_cross=jnp.sqrt(jnp.square(fac_xy-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    dr_cross_dx=(fac_xy-R_axis+1.e-12)*xyz[:,:,0]/(r_cross*fac_xy+1.e-12)
    dr_cross_dy=(fac_xy-R_axis+1.e-12)*xyz[:,:,1]/(r_cross*fac_xy+1.e-12)
    dr_cross_dz=(xyz[:,:,2]-Z_axis+1.e-12)/(r_cross+1.e-12)    
    B_particle=jax.vmap(jax.vmap(field.B_covariant,in_axes=0),in_axes=0)(xyz)
    B_r=jnp.multiply(B_particle[:,:,0],dr_cross_dx)+jnp.multiply(B_particle[:,:,1],dr_cross_dy)+jnp.multiply(B_particle[:,:,2],dr_cross_dz)
    return jnp.sum(jnp.abs(B_r))


def loss_iota(x,particles,dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True,target_iota=0.5,maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    #particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=1.e-8,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=tracing.field.r_axis
    Z_axis=tracing.field.z_axis
    #theta=jnp.arctan2(xyz[:,:,2]-Z_axis+1.e-12, jnp.sqrt(xyz[:,:,0]**2+xyz[:,:,1]**2)-R_axis+1.e-12)    
    fac_xy=jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))
    dtheta_dx=-(xyz[:,:,2]-Z_axis+1.e-12)*xyz[:,:,0]/(jnp.square(fac_xy-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12)+1.e-12)
    dtheta_dy=-(xyz[:,:,2]-Z_axis+1.e-12)*xyz[:,:,1]/(jnp.square(fac_xy-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12)+1.e-12)    
    dtheta_dz=(fac_xy-R_axis+1.e-12)/(jnp.square(fac_xy-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12)+1.e-12)      
    dphi_dx=-(xyz[:,:,1])/(fac_xy**2+1.e-12)
    dphi_dy=xyz[:,:,0]/(fac_xy**2+1.e-12)    
    B_particle=jax.vmap(jax.vmap(tracing.field.B_covariant,in_axes=0),in_axes=0)(xyz)
    B_theta=jnp.multiply(B_particle[:,:,0],dtheta_dx)+jnp.multiply(B_particle[:,:,1],dtheta_dy)+jnp.multiply(B_particle[:,:,2],dtheta_dz)
    B_phi=jnp.multiply(B_particle[:,:,0],dphi_dx)+jnp.multiply(B_particle[:,:,1],dphi_dy)
    return jnp.sum(jnp.maximum(target_iota-B_theta/B_phi,0.0))

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
#    coil_length=jnp.ravel(field.coils_length)
#    return jnp.array([jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))])

# @partial(jit, static_argnums=(0))
#def loss_coil_curvature(field,max_coil_curvature=0.4):
#    coil_curvature=jnp.mean(field.coils_curvature, axis=1)
#    return jnp.array([jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))])

# @partial(jit, static_argnums=(0))
def loss_coil_length(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,max_coil_length=31):
    field=field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments,stellsym)    
    coil_length=jnp.ravel(field.coils_length)
    return jnp.ravel(jnp.array([jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))]))

# @partial(jit, static_argnums=(0))
def loss_coil_curvature(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,max_coil_curvature=0.4):
    field=field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments,stellsym)
    coil_curvature=jnp.mean(field.coils_curvature, axis=1)
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
    return jnp.array([jnp.absolute(jnp.average(B_axis)-target_B_on_axis)])



# @partial(jit, static_argnums=(0))
def loss_coil_curvature_new(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,max_coil_curvature=0.4):
    field=field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments,stellsym)
    coil_curvature=jnp.mean(field.coils_curvature, axis=1)
    return jnp.maximum(coil_curvature-max_coil_curvature,0.0)

# @partial(jit, static_argnums=(0))
def loss_coil_length_new(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,max_coil_length=31):
    field=field_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments,stellsym)    
    coil_length=jnp.ravel(field.coils_length)
    return jnp.maximum(coil_length-max_coil_length,0.0)







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

@partial(jit, static_argnums=(1, 4, 5, 6))
def loss_BdotN_only(x, vmec, dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    
    bdotn_over_b = BdotN_over_B(vmec.surface, field)

    bdotn_over_b_loss = jnp.sum(jnp.abs(bdotn_over_b))

    return bdotn_over_b_loss

@partial(jit, static_argnums=(1, 4, 5, 6,7))
def loss_BdotN_only_constraint(x, vmec, dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True,target_tol=1.e-6):
    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
    
    bdotn_over_b = BdotN_over_B(vmec.surface, field)

    bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0)))
    #bdotn_over_b_loss = jnp.sqrt(0.5*jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0))
    return bdotn_over_b_loss


@partial(jit, static_argnums=(1,2,3, 6, 7, 8))
def loss_BdotN_only_stochastic(x,sampler,N_samples, vmec, dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True):
    keys= jnp.arange(N_samples)  
    def perturbed_bdotn_over_b(x,key,sampler,dofs_curves, currents_scale, nfp, n_segments, stellsym):
        perturbed_field = pertubred_field_from_dofs(x,key,sampler, dofs_curves, currents_scale, nfp, n_segments, stellsym)
        bdotn_over_b = BdotN_over_B(vmec.surface, perturbed_field)
        return jnp.sum(jnp.abs(bdotn_over_b))
    #Average over the N_samples
    expected_loss=jnp.average(jax.vmap(perturbed_bdotn_over_b, in_axes=(None,0,None,None,None,None,None,None))(x, keys,sampler, dofs_curves, currents_scale, nfp, n_segments, stellsym),axis=0)
    return expected_loss


@partial(jit, static_argnums=(1,2,3, 6, 7, 8,9))
def loss_BdotN_only_constraint_stochastic(x,sampler,N_samples, vmec, dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True,target_tol=1.e-6):
    keys= jnp.arange(N_samples)  
    def perturbed_bdotn_over_b(x,key,sampler,dofs_curves, currents_scale, nfp, n_segments, stellsym):
        perturbed_field = pertubred_field_from_dofs(x,key,sampler, dofs_curves, currents_scale, nfp, n_segments, stellsym)
        bdotn_over_b = BdotN_over_B(vmec.surface, perturbed_field)
        return jnp.square(bdotn_over_b)
    #Average over the N_samples
    expected_loss=jnp.average(jax.vmap(perturbed_bdotn_over_b, in_axes=(None,0,None,None,None,None,None,None))(x, keys,sampler, dofs_curves, currents_scale, nfp, n_segments, stellsym),axis=0)

    constrained_expected_loss = jnp.sqrt(jnp.sum(jnp.maximum(expected_loss-target_tol,0.0)))
    #bdotn_over_b_loss = jnp.sqrt(0.5*jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0))
    return constrained_expected_loss



#This is thr quickest way to get coil-surface distance (but I guess not the most efficient way for large sizes). 
# In that case we would do the candidates method from simsopt entirely
def loss_cs_distance(x,surface,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,min_distance_cs=1.3):
    coils=coils_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)    
    result=jnp.sum(jax.vmap(cs_distance_pure,in_axes=(0,0,None,None,None))(coils.gamma,coils.gamma_dash,surface.gamma,surface.unitnormal,min_distance_cs))
    return result

#Same as above but for individual constraints (useful in case one wants to target the several pairs individually)
def loss_cs_distance_array(x,surface,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,min_distance_cs=1.3):
    coils=coils_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)    
    result=jax.vmap(cs_distance_pure,in_axes=(0,0,None,None,None))(coils.gamma,coils.gamma_dash,surface.gamma,surface.unitnormal,min_distance_cs)
    return result.flatten()

#This is thr quickest way to get coil-coil distance (but I guess not the most efficient way for large sizes). 
# In that case we would do the candidates method from simsopt entirely
def loss_cc_distance(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,min_distance_cc=0.7,downsample=1):
    coils=coils_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)    
    result=jnp.sum(jnp.triu(jax.vmap(jax.vmap(cc_distance_pure,in_axes=(0,0,None,None,None,None)),in_axes=(None,None,0,0,None,None))(coils.gamma,coils.gamma_dash,coils.gamma,coils.gamma_dash,min_distance_cc,downsample),k=1))
    return result

#Same as above but for individual constraints (useful in case one wants to target the several pairs individually)
def loss_cc_distance_array(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,min_distance_cc=0.7,downsample=1):
    coils=coils_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)    
    result=jnp.triu(jax.vmap(jax.vmap(cc_distance_pure,in_axes=(0,0,None,None,None,None)),in_axes=(None,None,0,0,None,None))(coils.gamma,coils.gamma_dash,coils.gamma,coils.gamma_dash,min_distance_cc,downsample),k=1)
    return result[result != 0.0].flatten()



#One curve to curve distance (
#reused from Simsopt, no changes were necessary)
def cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance, downsample=1):
    """
    Compute the curve-curve distance penalty between two curves.

    Args:
        gamma1 (array-like): Points along the first curve.
        l1 (array-like): Tangent vectors along the first curve.
        gamma2 (array-like): Points along the second curve.
        l2 (array-like): Tangent vectors along the second curve.
        minimum_distance (float): The minimum allowed distance between curves.
        downsample (int, default=1): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

    Returns:
        float: The curve-curve distance penalty value.
    """
    gamma1 = gamma1[::downsample, :]
    gamma2 = gamma2[::downsample, :]
    l1 = l1[::downsample, :]
    l2 = l2[::downsample, :]
    dists = jnp.sqrt(jnp.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
    alen = jnp.linalg.norm(l1, axis=1)[:, None] * jnp.linalg.norm(l2, axis=1)[None, :]
    return jnp.sum(alen * jnp.maximum(minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])



#One coil to surface distance (reused from Simsopt, no changes were necessary)
def cs_distance_pure(gammac, lc, gammas, ns, minimum_distance):
    """
    Compute the curve-surface distance penalty between a curve and a surface.

    Args:
        gammac (array-like): Points along the curve.
        lc (array-like): Tangent vectors along the curve.
        gammas (array-like): Points on the surface.
        ns (array-like): Surface normal vectors.
        minimum_distance (float): The minimum allowed distance between curve and surface.

    Returns:
        float: The curve-surface distance penalty value.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None] \
        * jnp.linalg.norm(ns, axis=1)[None, :]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**2)



#This is thr quickest way to get coil-coil distance (but I guess not the most efficient way for large sizes). 
# In that case we would do the candidates method from simsopt entirely
def loss_linking_mnumber(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,downsample=1):
    coils=coils_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)    
    #Since the quadpoints are the same for every curve then we can calculate the increment is constant for every curve 
    # (needs change if quadpoints are allowed to be different)
    dphi=coils.quadpoints[1]-coils.quadpoints[0]
    result=jnp.sum(jnp.triu(jax.vmap(jax.vmap(linking_number_pure,in_axes=(0,0,None,None,None)),
                                        in_axes=(None,None,0,0,None))(coils.gamma[:,0:-1:downsample,:],
                                                                    coils.gamma_dash[:,0:-1:downsample,:],
                                                                    coils.gamma[:,0:-1:downsample,:],
                                                                    coils.gamma_dash[:,0:-1:downsample,:],
                                                                    dphi),k=1))
    return result


#This is thr quickest way to get coil-coil distance (but I guess not the most efficient way for large sizes). 
# In that case we would do the candidates method from simsopt entirely
def loss_linking_mnumber_constarint(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,downsample=1):
    coils=coils_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)    
    #Since the quadpoints are the same for every curve then we can calculate the increment is constant for every curve 
    # (needs change if quadpoints are allowed to be different)
    dphi=coils.quadpoints[1]-coils.quadpoints[0]
    result=jnp.triu(jax.vmap(jax.vmap(linking_number_pure,in_axes=(0,0,None,None,None)),
                                        in_axes=(None,None,0,0,None))(coils.gamma[:,0:-1:downsample,:],
                                                                    coils.gamma_dash[:,0:-1:downsample,:],
                                                                    coils.gamma[:,0:-1:downsample,:],
                                                                    coils.gamma_dash[:,0:-1:downsample,:],
                                                                    dphi)+1.e-18,k=1)  
    #The 1.e-18 above is just to get all the correct values in the following mask
    return result[result != 0.0].flatten()

def linking_number_pure(gamma1, lc1, gamma2, lc2,dphi):
    linking_number_ij=jnp.sum(jnp.abs(jax.vmap(integrand_linking_number, in_axes=(0, 0, 0, 0,None,None))(gamma1, lc1, gamma2, lc2,dphi,dphi)/ (4*jnp.pi)))
    return linking_number_ij

def integrand_linking_number(r1,dr1,r2,dr2,dphi1,dphi2):
    """
    Compute the integrand for the linking number between two curves.

    Args:
        r1 (array-like): Points along the first curve.
        dr1 (array-like): Tangent vectors along the first curve.
        r2 (array-like): Points along the second curve.
        dr2 (array-like): Tangent vectors along the second curve.
        dphi1 (array-like): increments of quadpoints 1  
        dphi2 (array-like): increments of quadpoints 2               

    Returns:
        float: The integrand value for the linking number.
    """
    return jnp.dot((r1-r2), jnp.cross(dr1, dr2)) / jnp.linalg.norm(r1-r2)**3*dphi1*dphi2



#Loss function penalizing force on coils using Landremann-Hurwitz method
def loss_lorentz_force_coils(x,dofs_curves,currents_scale,nfp,n_segments=60,stellsym=True,p=1,threshold=0.5e+6):
    coils=coils_from_dofs(x,dofs_curves,currents_scale,nfp,n_segments, stellsym) 
    curves_indeces=jnp.arange(coils.gamma.shape[0])
    #We want to calculate tangeng cross [B_self + B_mutual] for each coil
    #B_self is the self-field of the coil, B_mutual is the field from the other coils
    force_penalty=jax.vmap(lp_force_pure,in_axes=(0,None,None,None,None,None,None,None))(curves_indeces,coils.gamma,
                                                                                 coils.gamma_dash,coils.gamma_dashdash,coils.currents,coils.quadpoints,p, threshold)
    return force_penalty






def lp_force_pure(index,gamma, gamma_dash,gamma_dashdash,currents,quadpoints,p, threshold):
    """Pure function for minimizing the Lorentz force on a coil.
    """
    regularization = regularization_circ(1./jnp.average(compute_curvature( gamma_dash.at[index].get(), gamma_dashdash.at[index].get())))
    B_mutual=jax.vmap(BiotSavart_from_gamma(jnp.roll(gamma, -index, axis=0)[1:],
                                 jnp.roll(gamma_dash, -index, axis=0)[1:],
                                 jnp.roll(gamma_dashdash, -index, axis=0)[1:],
                                 jnp.roll(currents, -index, axis=0)[1:]).B,in_axes=0)(gamma[index])
    B_self = B_regularized_pure(gamma.at[index].get(),gamma_dash.at[index].get(), gamma_dashdash.at[index].get(), quadpoints, currents[index], regularization)
    gammadash_norm = jnp.linalg.norm(gamma_dash.at[index].get(), axis=1)[:, None]
    tangent = gamma_dash.at[index].get() / gammadash_norm
    force = jnp.cross(currents.at[index].get() * tangent, B_self + B_mutual)
    force_norm = jnp.linalg.norm(force, axis=1)[:, None]
    return (jnp.sum(jnp.maximum(force_norm - threshold, 0)**p * gammadash_norm))*(1./p)



def B_regularized_singularity_term(rc_prime, rc_prime_prime, regularization):
    """The term in the regularized Biot-Savart law in which the near-singularity
    has been integrated analytically.

    regularization corresponds to delta * a * b for rectangular x-section, or to
    a²/√e for circular x-section.

    A prefactor of μ₀ I / (4π) is not included.

    The derivatives rc_prime, rc_prime_prime refer to an angle that goes up to
    2π, not up to 1.
    """
    norm_rc_prime = jnp.linalg.norm(rc_prime, axis=1)
    return jnp.cross(rc_prime, rc_prime_prime) * (0.5 * (-2 + jnp.log(64 * norm_rc_prime * norm_rc_prime / regularization)) / (norm_rc_prime**3))[:, None]


def B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization):
    # The factors of 2π in the next few lines come from the fact that simsopt
    # uses a curve parameter that goes up to 1 rather than 2π.
    phi = quadpoints * 2 * jnp.pi
    rc = gamma
    rc_prime = gammadash / 2 / jnp.pi
    rc_prime_prime = gammadashdash / 4 / jnp.pi**2
    n_quad = phi.shape[0]
    dphi = 2 * jnp.pi / n_quad
    analytic_term = B_regularized_singularity_term(rc_prime, rc_prime_prime, regularization)
    dr = rc[:, None] - rc[None, :]
    first_term = jnp.cross(rc_prime[None, :], dr) / ((jnp.sum(dr * dr, axis=2) + regularization) ** 1.5)[:, :, None]
    cos_fac = 2 - 2 * jnp.cos(phi[None, :] - phi[:, None])
    denominator2 = cos_fac * jnp.sum(rc_prime * rc_prime, axis=1)[:, None] + regularization
    factor2 = 0.5 * cos_fac / denominator2**1.5
    second_term = jnp.cross(rc_prime_prime, rc_prime)[:, None, :] * factor2[:, :, None]
    integral_term = dphi * jnp.sum(first_term + second_term, 1)
    return current * mu_0 / (4 * jnp.pi) * (analytic_term + integral_term)



def regularization_circ(a):
    """Regularization for a circular conductor"""
    return a**2 / jnp.sqrt(jnp.e)


def regularization_rect(a, b):
    """Regularization for a rectangular conductor"""
    return a * b * rectangular_xsection_delta(a, b)

def rectangular_xsection_k(a, b):
    """Auxiliary function for field in rectangular conductor"""
    return (4 * b) / (3 * a) * jnp.arctan(a/b) + (4*a)/(3*b)*jnp.arctan(b/a)+ (b**2)/(6*a**2)*jnp.log(b/a) + (a**2)/(6*b**2)*jnp.log(a/b) -  (a**4 - 6*a**2*b**2 + b**4)/(6*a**2*b**2)*jnp.log(a/b+b/a)


def rectangular_xsection_delta(a, b):
    """Auxiliary function for field in rectangular conductor"""
    return jnp.exp(-25/6 + rectangular_xsection_k(a, b))


#def loss_BdotN_only_with_perturbation(x, vmec, dofs_curves, currents_scale, nfp,n_segments=60, stellsym=True, N_stells=10):
#    """
#    Compute the loss function for BdotN with a perturbation applied to the BdotN value.):
#    field=field_from_dofs(x,dofs_curves, currents_scale, nfp,n_segments, stellsym)
#    
#    bdotn_over_b = BdotN_over_B(vmec.surface, field)
#    
#    # Apply perturbation to the BdotN value
#    bdotn_over_b += perturbation
#    
#    bdotn_over_b_loss = jnp.sum(jnp.abs(bdotn_over_b))

#    return bdotn_over_b_loss