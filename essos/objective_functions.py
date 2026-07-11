import jax

# from build.lib.essos import coils
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from essos.dynamics import Tracing
from essos.fields import BiotSavart,BiotSavart_from_gamma
from essos.surfaces import BdotN_over_B
from essos.coils import Curves, Coils
from essos.constants import mu_0
from essos.coil_perturbation import perturb_curves





########################## NEAR-AXIS FIELD LOSSES ##########################
def near_axis_field_quantities(field_nearaxis):
    Raxis = field_nearaxis.R0
    Zaxis = field_nearaxis.Z0
    phi = field_nearaxis.phi
    Xaxis = Raxis*jnp.cos(phi)
    Yaxis = Raxis*jnp.sin(phi)
    points = jnp.array([Xaxis, Yaxis, Zaxis])
    B_nearaxis = field_nearaxis.B_axis.T
    gradB_nearaxis = field_nearaxis.grad_B_axis.T
    return points, B_nearaxis, gradB_nearaxis



def loss_B_difference_coils_near_axis(field, field_nearaxis):
    points, B_nearaxis, _ = near_axis_field_quantities(field_nearaxis)
    B_coils = vmap(field.B)(points.T)
    B_difference_loss = jnp.sum(jnp.abs(jnp.array(B_coils)-jnp.array(B_nearaxis)))
    return B_difference_loss

def loss_gradB_difference_coils_near_axis(field, field_nearaxis):
    points, _, gradB_nearaxis = near_axis_field_quantities(field_nearaxis)
    gradB_coils = vmap(field.dB_by_dX)(points.T)
    gradB_difference_loss = jnp.sum(jnp.abs(jnp.array(gradB_coils)-jnp.array(gradB_nearaxis)))
    return gradB_difference_loss

def loss_iota_near_axis(field_nearaxis,iota_target=0.41):
    return jnp.abs((field_nearaxis.iota - iota_target))

def loss_r0_near_axis(field_nearaxis, r0_target=1.0):
    return jnp.abs((field_nearaxis.R0[0] - r0_target))


##############################Particle confinement losses ##############################
def loss_particle_radial_drift_fullorbit(field, particles, timestep=1.e-8, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=field.r_axis
    Z_axis=field.z_axis
    #Ideally here one would differentiate in time through diffrax !TODO
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    v_r_cross=jnp.diff(r_cross,axis=1)#/tracing.times_to_trace*tracing.maxtime     
    return (jnp.sum(jnp.square(jnp.average(v_r_cross,axis=1))))

def loss_particle_radial_drift(field, particles, timestep=1.e-8, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=field.r_axis
    Z_axis=field.z_axis
    #Ideally here one would differentiate in time through diffrax !TODO
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    v_r_cross=jnp.diff(r_cross,axis=1)#/tracing.times_to_trace*tracing.maxtime     
    return (jnp.sum(jnp.square(jnp.average(v_r_cross,axis=1))))


def loss_particle_alpha_drift(field, particles, timestep=1.e-8, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=field.r_axis
    Z_axis=field.z_axis
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
    theta=jnp.arctan2(xyz[:,:,2]-Z_axis+1.e-12, jnp.sqrt(xyz[:,:,0]**2+xyz[:,:,1]**2)-R_axis+1.e-12)
    v_theta=jnp.diff(theta,axis=1)#/tracing.times_to_trace*tracing.maxtime                               
    return jnp.sum(jnp.square(jnp.average(v_theta,axis=1)))  

def loss_particle_gammac(field, particles, timestep=1.e-8, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=field.r_axis
    Z_axis=field.z_axis
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
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    v_r_cross=jnp.diff(r_cross,axis=1)#/tracing.times_to_trace*tracing.maxtime                        
    theta=jnp.arctan2(xyz[:,:,2]-Z_axis+1.e-12, jnp.sqrt(xyz[:,:,0]**2+xyz[:,:,1]**2)-R_axis+1.e-12)
    v_theta=jnp.diff(theta,axis=1)#/tracing.times_to_trace*tracing.maxtime                          
    #return jnp.sum(jnp.square((2./jnp.pi*jnp.absolute(jnp.arctan2(jnp.average(v_r_cross,axis=1),jnp.average(v_theta,axis=1))))))
    return jnp.max(2./jnp.pi*jnp.absolute(jnp.arctan2(jnp.average(v_r_cross,axis=1),jnp.average(v_theta,axis=1))))
    
def loss_particle_rcross_final(field, particles, timestep=1.e-8, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=field.r_axis
    Z_axis=field.z_axis
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,:,0])+jnp.square(xyz[:,:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,:,2]-Z_axis+1.e-12))
    return jnp.linalg.norm((jnp.average(r_cross,axis=1)))



def loss_particle_Br(field, particles, timestep=1.e-8, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
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

def loss_particle_iota(field, particles, timestep=1.e-8, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None,target_iota=0.41):
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
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








###################  B ON SURAFCE LOSSES ##########################
@partial(jit, static_argnames=['npoints'])
def normB_axis(field, npoints=15):
    R_axis=field.r_axis
    phi_array = jnp.linspace(0, 2 * jnp.pi, npoints)
    B_axis = vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return B_axis

@partial(jit, static_argnames=['npoints', 'target_B'])
def loss_normB_axis_average(field,npoints=15, target_B=5.7):
    B_axis = normB_axis(field, npoints)
    return jnp.abs(jnp.average(B_axis)-target_B)


def loss_BdotN(field,surface):
    return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))

@partial(jit, static_argnames=['target_tol'])
def loss_BdotN_constraint(field,surface,target_tol=1.e-6):
    bdotn_over_b = BdotN_over_B(surface, field)
    bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0)))
    return bdotn_over_b_loss


###########################  B ON SURAFCE LOSSES FOR STOCHASTIC OPTIMIZATION ##########################
def copy_coils_from_field(field):
    return field.coils.copy()

@partial(jit, static_argnames=['sampler'])
def perturbed_field_from_field(field, key, sampler):
    coils = copy_coils_from_field(field)
    base_key = jax.random.key(key)
    split_keys = jax.random.split(base_key, 2)
    coils = perturb_curves_systematic(coils, sampler, key=split_keys[0])
    coils = perturb_curves_statistic(coils, sampler, key=split_keys[1])
    return BiotSavart(coils)


@partial(jit, static_argnames=['sampler'])
def loss_bdotn_stochastic(field, surface, sampler, keys):
    def perturbed_loss(key):
        perturbed_field = perturbed_field_from_field(field, key, sampler)
        bdotn_over_b = BdotN_over_B(surface, perturbed_field)
        return jnp.sum(jnp.abs(bdotn_over_b))

    return jnp.mean(jax.vmap(perturbed_loss)(keys))


@partial(jit, static_argnames=['sampler', 'target_tol'])
def constraint_bdotn_stochastic(field, surface, sampler, keys, target_tol=1.0e-6):
    def perturbed_square(key):
        perturbed_field = perturbed_field_from_field(field, key, sampler)
        return jnp.square(BdotN_over_B(surface, perturbed_field))

    expected_square = jnp.mean(jax.vmap(perturbed_square)(keys), axis=0)
    return jnp.sqrt(jnp.sum(jnp.maximum(expected_square - target_tol, 0.0)))




######################### COIL GEOMETRY LOSSES #################################

@partial(jit, static_argnames=['max_coil_length'])
def loss_coil_length(coils, max_coil_length=0):
    return jnp.square(coils.length/max_coil_length - 1)

@partial(jit, static_argnames=['max_coil_curvature'])
def loss_coil_curvature(coils, max_coil_curvature=0):
    pointwise_curvature_loss = jnp.square(jnp.maximum(coils.curvature-max_coil_curvature, 0))
    return jnp.mean(pointwise_curvature_loss*jnp.linalg.norm(coils.gamma_dash, axis=-1), axis=1)

def compute_candidates(coils, min_separation):
    centers = coils.curves.curves[:, :, 0]
    a_n = coils.curves.curves[:, :, 2 : 2*coils.order+1 : 2]
    b_n = coils.curves.curves[:, :, 1 : 2*coils.order : 2]
    radii = jnp.sum(jnp.linalg.norm(a_n, axis=1)+jnp.linalg.norm(b_n, axis=1), axis=1)

    i_vals, j_vals = jnp.triu_indices(len(coils), k=1)
    centers_dists = jnp.linalg.norm(centers[i_vals] - centers[j_vals], axis=1)
    mask = centers_dists <= min_separation + radii[i_vals] + radii[j_vals]

    return i_vals[mask], j_vals[mask]


# Blockwise, memory-efficient coil separation loss
@partial(jit, static_argnames=["min_separation", "block_size"])
def loss_coil_separation(coils, min_separation, candidates=None, block_size=None):
    """
    Memory-efficient coil separation loss using blockwise vmap.
    Args:
        coils: Coils object
        min_separation: Minimum allowed separation
        candidates: Optional tuple of (i, j) coil index arrays
        block_size: Block size for memory efficiency. If None, uses full vmap (no chunking)
    Returns:
        Scalar loss (sum over all coil pairs)
    """
    if candidates is None:
        candidates = jnp.triu_indices(len(coils), k=1)

    def pair_loss(i, j):
        gamma_i = coils.gamma[i]
        gamma_dash_i = jnp.linalg.norm(coils.gamma_dash[i], axis=-1)
        gamma_j = coils.gamma[j]
        gamma_dash_j = jnp.linalg.norm(coils.gamma_dash[j], axis=-1)
        n_points = gamma_i.shape[0]

        # If block_size is None, use full vmap (no chunking)
        use_block_size = min(n_points, n_points if block_size is None else block_size)
        n_blocks = (n_points + use_block_size - 1) // use_block_size
        padded_points = n_blocks * use_block_size
        pad_width = padded_points - n_points

        gamma_j_blocks = jnp.pad(gamma_j, ((0, pad_width), (0, 0))).reshape(n_blocks, use_block_size, 3)
        gamma_dash_j_blocks = jnp.pad(gamma_dash_j, (0, pad_width)).reshape(n_blocks, use_block_size)
        valid_blocks = (jnp.arange(padded_points) < n_points).reshape(n_blocks, use_block_size)

        def block_sum(block_gamma_j, block_gamma_dash_j, block_valid):
            dists_block = jnp.linalg.norm(gamma_i[:, None, :] - block_gamma_j[None, :, :], axis=2)
            penalty_block = jnp.maximum(0, min_separation - dists_block)
            weighted_penalty = (
                jnp.square(penalty_block)
                * gamma_dash_i[:, None]
                * block_gamma_dash_j[None, :]
                * block_valid[None, :]
            )
            return jnp.sum(weighted_penalty)

        total = jnp.sum(jax.vmap(block_sum)(gamma_j_blocks, gamma_dash_j_blocks, valid_blocks))
        norm = gamma_i.shape[0] * gamma_j.shape[0]
        return total / norm

    losses = jax.vmap(pair_loss)(*candidates)
    return jnp.sum(losses)

# Blockwise, memory-efficient coil-surface distance loss
@partial(jit, static_argnames=["min_distance", "block_size"])
def loss_coil_surface_distance(coils, surface, min_distance, block_size=None):
    """
    Memory-efficient coil-surface distance loss using blockwise vmap and symmetry reduction.
    Args:
        coils: Coils object
        surface: Surface object (with gamma, unitnormal)
        min_distance: Minimum allowed coil-surface distance
        block_size: Block size for memory efficiency. If None, uses full vmap (no chunking)
        nfp: Number of field periods
        stellsym: Whether stellarator symmetry is present
    Returns:
        Scalar loss (sum over all relevant coil-surface pairs)
    """
    n_coils = coils.gamma.shape[0]
    n_points_coil = coils.gamma.shape[1]
    surface_points = surface.gamma.reshape(-1, 3)
    n_points_surface = surface_points.shape[0]

    # Only check unique coils for symmetry
    if coils.stellsym:
        n_unique_coils = n_coils // (2 * coils.nfp)
    else:
        n_unique_coils = n_coils // coils.nfp
    n_unique_coils = max(1, n_unique_coils)
    unique_coil_indices = jnp.arange(n_unique_coils)

    def single_coil_loss(idx):
        gamma_i = coils.gamma[idx]
        gamma_dash_i = coils.gamma_dash[idx]
        gamma_dash_norm = jnp.linalg.norm(gamma_dash_i, axis=1)
        n_points = gamma_i.shape[0]

        # If block_size is None, use full vmap (no chunking)
        use_block_size = min(n_points_surface, n_points_surface if block_size is None else block_size)
        n_blocks = (n_points_surface + use_block_size - 1) // use_block_size
        padded_points = n_blocks * use_block_size
        pad_width = padded_points - n_points_surface

        surface_point_blocks = jnp.pad(surface_points, ((0, pad_width), (0, 0))).reshape(n_blocks, use_block_size, 3)
        valid_blocks = (jnp.arange(padded_points) < n_points_surface).reshape(n_blocks, use_block_size)

        def block_sum(block_surface_points, block_valid):
            dists_block = jnp.linalg.norm(gamma_i[:, None, :] - block_surface_points[None, :, :], axis=2)
            penalty_block = jnp.maximum(0, min_distance - dists_block)
            weighted_penalty = jnp.square(penalty_block) * gamma_dash_norm[:, None] * block_valid[None, :]
            return jnp.sum(weighted_penalty)

        total = jnp.sum(jax.vmap(block_sum)(surface_point_blocks, valid_blocks))
        norm = gamma_i.shape[0] * n_points_surface
        return total / norm

    losses = jax.vmap(single_coil_loss)(unique_coil_indices)
    return jnp.sum(losses)


# Blockwise vmap linking number loss (memory efficient, fully differentiable)
@partial(jit, static_argnames=["block_size"])
def loss_linkingnumber(coils, candidates=None, block_size=None):
    if candidates is None:
        candidates = jnp.triu_indices(len(coils), k=1)
    dphi = coils.curves.quadpoints[1] - coils.curves.quadpoints[0]

    def pair_linking(i, j):
        gamma_i = coils.gamma[i]
        gamma_dash_i = coils.gamma_dash[i]
        gamma_j = coils.gamma[j]
        gamma_dash_j = coils.gamma_dash[j]
        n_points = gamma_j.shape[0]

        # If block_size is None, use full vmap (no chunking)
        use_block_size = min(n_points, n_points if block_size is None else block_size)
        n_blocks = (n_points + use_block_size - 1) // use_block_size
        padded_points = n_blocks * use_block_size
        pad_width = padded_points - n_points

        gamma_j_blocks = jnp.pad(gamma_j, ((0, pad_width), (0, 0))).reshape(n_blocks, use_block_size, 3)
        gamma_dash_j_blocks = jnp.pad(gamma_dash_j, ((0, pad_width), (0, 0))).reshape(n_blocks, use_block_size, 3)
        valid_blocks = (jnp.arange(padded_points) < n_points).reshape(n_blocks, use_block_size)

        def block_sum(block_gamma_j, block_gamma_dash_j, block_valid):
            def integrand(r2, dr2):
                diff = gamma_i - r2
                cross = jnp.cross(gamma_dash_i, dr2)
                norm = jnp.linalg.norm(diff, axis=1)
                return jnp.sum(diff * cross, axis=1) / (norm**3 + 1e-12)

            block_vals = jax.vmap(integrand, in_axes=(0, 0))(block_gamma_j, block_gamma_dash_j)
            return jnp.sum(block_vals * block_valid[:, None])

        total = jnp.sum(jax.vmap(block_sum)(gamma_j_blocks, gamma_dash_j_blocks, valid_blocks))
        linking = total * (dphi ** 2) / (4 * jnp.pi)
        return jnp.abs(linking)

    losses = jax.vmap(pair_linking)(*candidates)
    return jnp.sum(losses)




#  Lorentz force loss: accepts Coils object, keyword args, JAX-friendly
@partial(jit, static_argnames=["p", "threshold", "block_size"])
def loss_lorentz_force_coils(coils, p=1, threshold=0.5e6, block_size=None):
    """
    Loss function penalizing Lorentz force on coils using Landreman-Hurwitz method.
    Args:
        coils: Coils object (with gamma, gamma_dash, gamma_dashdash, currents, quadpoints)
        p: Power for penalty (default 1)
        threshold: Force threshold (default 0.5e6)
        block_size: Block size for memory efficiency. If None, uses full vmap (no chunking)
    Returns:
        Scalar loss (sum over all coils)
    """
    n_coils = coils.gamma.shape[0]
    indices = jnp.arange(n_coils)
    other_indices = jnp.array([
        [j for j in range(n_coils) if j != i]
        for i in range(n_coils)
    ], dtype=jnp.int32)

    def single_coil_loss(idx):
        gamma_i = coils.gamma[idx]
        gamma_dash_i = coils.gamma_dash[idx]
        gamma_dashdash_i = coils.gamma_dashdash[idx]
        current_i = coils.currents[idx]
        quadpoints = coils.curves.quadpoints
        curvature = Curves.compute_curvature(gamma_dash_i, gamma_dashdash_i)
        regularization = regularization_circ(1. / jnp.mean(curvature))
        other_idx = other_indices[idx]
        gamma_others = coils.gamma[other_idx]
        gamma_dash_others = coils.gamma_dash[other_idx]
        gamma_dashdash_others = coils.gamma_dashdash[other_idx]
        currents_others = coils.currents[other_idx]
        biot_savart = BiotSavart_from_gamma(gamma_others, gamma_dash_others, gamma_dashdash_others, currents_others)
        n_points = gamma_i.shape[0]
        use_block_size = min(n_points, n_points if block_size is None else block_size)
        n_blocks = (n_points + use_block_size - 1) // use_block_size
        padded_points = n_blocks * use_block_size
        pad_width = padded_points - n_points

        gamma_i_blocks = jnp.pad(gamma_i, ((0, pad_width), (0, 0))).reshape(n_blocks, use_block_size, 3)
        valid_blocks = (jnp.arange(padded_points) < n_points).reshape(n_blocks, use_block_size)

        def block_field(block_gamma_i, block_valid):
            block_B = jax.vmap(biot_savart.B)(block_gamma_i)
            return block_B * block_valid[:, None]

        block_B_mutual = jax.vmap(block_field)(gamma_i_blocks, valid_blocks).reshape(padded_points, 3)[:n_points]
        block_gammadash_norm = jnp.linalg.norm(gamma_dash_i, axis=1)
        block_tangent = gamma_dash_i / block_gammadash_norm[:, None]
        block_B_self = B_regularized_pure(
            gamma_i, gamma_dash_i, gamma_dashdash_i,
            quadpoints, current_i, regularization
        )
        block_force = jnp.cross(current_i * block_tangent, block_B_self + block_B_mutual)
        block_force_norm = jnp.linalg.norm(block_force, axis=1)
        total_penalty = jnp.sum(jnp.maximum(block_force_norm - threshold, 0) ** p * block_gammadash_norm)
        return total_penalty * (1. / p)

    penalties = jax.vmap(single_coil_loss)(indices)
    return jnp.sum(penalties)






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



#######################  SURFACE GEOMETRY LOSSES ##########################
