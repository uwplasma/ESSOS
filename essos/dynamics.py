from pyexpat import model
import os
import jax
jax.config.update("jax_enable_x64", True)
# Every Tracing compiles a new solve, since maxtime and the save times are constants of it: about 4 s on a CPU
# and 20 s on a GPU. XLA's persistent cache, keyed by the compiled program, serves a repeated trace (an
# optimization loop, a rerun) from disk instead. ESSOS_XLA_CACHE names the directory; an empty value disables it.
_XLA_CACHE = os.environ.get("ESSOS_XLA_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "essos", "xla"))
if _XLA_CACHE and not jax.config.jax_compilation_cache_dir:
    jax.config.update("jax_compilation_cache_dir", _XLA_CACHE)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import jit, vmap, tree_util, random, lax, device_put
from functools import partial
from time import perf_counter
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, Event, TqdmProgressMeter, NoProgressMeter
from diffrax import ControlTerm,UnsafeBrownianPath,MultiTerm,ItoMilstein,ClipStepSizeController #For collisions we need this to solve stochastic differential equation
import diffrax
import optimistix as optx
from essos.coils import Coils
from essos.fields import BiotSavart, ExternalField, MagneticField, Vmec
from essos.surfaces import SurfaceClassifier
from essos.electric_field import Electric_field_flux, Electric_field_zero
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ELEMENTARY_CHARGE,SPEED_OF_LIGHT
from essos.plot import fix_matplotlib_3d
from essos.background_species import nu_s_ab,nu_D_ab,nu_par_ab, d_nu_par_ab,d_nu_D_ab



def gc_to_fullorbit(field, initial_xyz, initial_vparallel, total_speed, mass, charge, phase_angle_full_orbit=0):
    """
    Computes full orbit positions for given guiding center positions,
    parallel speeds, and total velocities using JAX for efficiency.

    The full-orbit start satisfies ``x - b x v / Omega = X`` with the signed
    gyrofrequency ``Omega = charge |B| / mass``, so the guiding center of the
    returned state is the requested point for either sign of the charge.
    """
    def compute_orbit_params(xyz, vpar):
        Bs = field.B_contravariant(xyz)
        AbsBs = jnp.linalg.norm(Bs)
        eB = Bs / AbsBs
        p1 = eB
        # Reference axis for the perpendicular basis: z, unless B is nearly
        # parallel to it (cross(b, z) would vanish and give NaN).
        p2 = jnp.where(jnp.abs(eB[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]))
        p3 = -jnp.cross(p1, p2)
        p3 /= jnp.linalg.norm(p3)
        q1 = p1
        q2 = p2 - jnp.dot(q1, p2) * q1
        q2 /= jnp.linalg.norm(q2)
        q3 = p3 - jnp.dot(q1, p3) * q1 - jnp.dot(q2, p3) * q2
        q3 /= jnp.linalg.norm(q3)
        speed_perp = jnp.sqrt(total_speed**2 - vpar**2)
        vperp = -speed_perp * jnp.cos(phase_angle_full_orbit) * q2 + speed_perp * jnp.sin(phase_angle_full_orbit) * q3
        gyrofrequency = charge * AbsBs / mass
        xyz_full = xyz + jnp.cross(eB, vperp) / gyrofrequency
        v_init = vpar * q1 + vperp
        return xyz_full, v_init
    xyz_inits_full, v_inits = vmap(compute_orbit_params)(initial_xyz, initial_vparallel)
    return xyz_inits_full, v_inits

class Particles():
    def __init__(self, initial_xyz=None, initial_vparallel_over_v=None, charge=ALPHA_PARTICLE_CHARGE,
                 mass=ALPHA_PARTICLE_MASS, energy=FUSION_ALPHA_PARTICLE_ENERGY, min_vparallel_over_v=-1,
                 max_vparallel_over_v=1, field=None, initial_vxvyvz=None, initial_xyz_fullorbit=None, phase_angle_full_orbit = 0):
        self.charge = charge
        self.mass = mass
        self.energy = energy
        self.initial_xyz = jnp.array(initial_xyz)
        self.nparticles = len(initial_xyz)
        self.initial_xyz_fullorbit = initial_xyz_fullorbit
        self.initial_vxvyvz = initial_vxvyvz
        self.phase_angle_full_orbit = phase_angle_full_orbit
        self.particle_index=jnp.arange(self.nparticles)
        
        key=jax.random.key(42)
        #self.random_keys=jax.random.split(key,32)[20:22]#self.nparticles)
        self.random_keys=jax.random.split(key,self.nparticles)        
        
        if initial_vparallel_over_v is not None:
            self.initial_vparallel_over_v = jnp.array(initial_vparallel_over_v)
        else:
            self.initial_vparallel_over_v = random.uniform(random.PRNGKey(42), (self.nparticles,), minval=min_vparallel_over_v, maxval=max_vparallel_over_v)
        
        self.total_speed = jnp.sqrt(2*self.energy/self.mass)
        
        self.initial_vparallel = self.total_speed*self.initial_vparallel_over_v
        self.initial_vperpendicular = jnp.sqrt(self.total_speed**2 - self.initial_vparallel**2)
        
        if field is not None and initial_xyz_fullorbit is None:
            self.to_full_orbit(field)
        
    def to_full_orbit(self, field):
        self.initial_xyz_fullorbit, self.initial_vxvyvz = gc_to_fullorbit(field=field, initial_xyz=self.initial_xyz, initial_vparallel=self.initial_vparallel,
                                                                            total_speed=self.total_speed, mass=self.mass, charge=self.charge,
                                                                            phase_angle_full_orbit=self.phase_angle_full_orbit)

    def join(self, other, field=None):
        assert isinstance(other, Particles), "Cannot join with non-Particles object"
        assert self.charge == other.charge, "Cannot join particles with different charges"
        assert self.mass == other.mass, "Cannot join particles with different masses"
        assert self.energy == other.energy, "Cannot join particles with different energies"

        charge = self.charge
        mass = self.mass
        energy = self.energy
        initial_xyz = jnp.concatenate((self.initial_xyz, other.initial_xyz), axis=0)
        initial_vparallel_over_v = jnp.concatenate((self.initial_vparallel_over_v, other.initial_vparallel_over_v), axis=0)

        return Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=initial_vparallel_over_v, charge=charge, mass=mass, energy=energy, field=field)


    
    @classmethod
    def InitializeParticlesAroundSurfaceAxis(cls, surface, n_particles, 
                                            distance_from_axis=0.0,
                                            charge=ALPHA_PARTICLE_CHARGE,
                                            mass=ALPHA_PARTICLE_MASS, 
                                            energy=FUSION_ALPHA_PARTICLE_ENERGY,
                                            min_vparallel_over_v=-1,
                                            max_vparallel_over_v=1,
                                            field=None,
                                            random_seed=42,
                                            n_arc_samples=1000,
                                            boundary_surface=None,
                                            distance_mode='absolute',
                                            boundary_bisection_steps=32):
        """Initialize particles randomly distributed around/along a magnetic axis extracted from a surface.
        
        Args:
            surface: SurfaceRZFourier object to extract axis from
            n_particles: Number of particles to initialize
            distance_from_axis: Perpendicular distance (in Frenet frame) from the axis 
                               (0.0 for particles on axis, >0 for particles around axis).
                               If distance_mode='fraction_to_boundary', this is interpreted
                               as a fraction in [0, 1] of the local axis-to-boundary distance.
            charge: Particle charge (default: alpha particle charge)
            mass: Particle mass (default: alpha particle mass)
            energy: Particle kinetic energy
            min_vparallel_over_v: Minimum parallel velocity fraction
            max_vparallel_over_v: Maximum parallel velocity fraction
            field: Magnetic field object (for converting to full orbit if needed)
            random_seed: Seed for random number generation
            n_arc_samples: Number of samples for arc-length parametrization
            boundary_surface: Optional surface used as geometric boundary when
                             distance_mode='fraction_to_boundary'.
            distance_mode: 'absolute' or 'fraction_to_boundary'.
            boundary_bisection_steps: Number of bisection iterations used to
                                     find axis-to-boundary distance along each
                                     particle direction.
            
        Returns:
            Particles object with initial positions distributed around the axis
        """
        if distance_mode not in ('absolute', 'fraction_to_boundary'):
            raise ValueError("distance_mode must be 'absolute' or 'fraction_to_boundary'.")

        if distance_mode == 'fraction_to_boundary':
            if boundary_surface is None:
                raise ValueError("boundary_surface is required when distance_mode='fraction_to_boundary'.")
            if distance_from_axis < 0.0 or distance_from_axis > 1.0:
                raise ValueError("distance_from_axis must be in [0, 1] when distance_mode='fraction_to_boundary'.")

            from essos.surfaces import signed_distance_from_surface_jax

            # Global bound used to cap the ray search for boundary intersection.
            boundary_points = boundary_surface.gamma.reshape((-1, 3))
            boundary_extent = float(jnp.max(jnp.linalg.norm(boundary_points, axis=1)))
            boundary_search_cap = max(1.0, 4.0 * boundary_extent)

            def signed_distance_boundary(xyz):
                return float(jnp.squeeze(signed_distance_from_surface_jax(xyz, boundary_surface)))

            def axis_to_boundary_distance(axis_pos, direction):
                # Find t such that axis_pos + t * direction lies on boundary (signed distance ~ 0).
                # Assumes axis point is inside boundary and direction points outward in the local plane.
                t_low = 0.0
                t_high = 0.2
                s_high = signed_distance_boundary(axis_pos + t_high * direction)
                while s_high > 0.0 and t_high < boundary_search_cap:
                    t_low = t_high
                    t_high *= 2.0
                    s_high = signed_distance_boundary(axis_pos + t_high * direction)

                # If no crossing was found, return the current bound as a safe fallback.
                if s_high > 0.0:
                    return t_high

                for _ in range(boundary_bisection_steps):
                    t_mid = 0.5 * (t_low + t_high)
                    s_mid = signed_distance_boundary(axis_pos + t_mid * direction)
                    if s_mid > 0.0:
                        t_low = t_mid
                    else:
                        t_high = t_mid
                return t_high

        # Extract m=0 modes (magnetic axis) from surface
        m0_mask = surface.xm == 0
        rc_axis = surface.rc[m0_mask]
        zs_axis = surface.zs[m0_mask]
        xn_axis = surface.xn[m0_mask]
        
        # Helper function: compute axis curve at given phi
        def compute_axis_point(phi):
            """Compute axis position at toroidal angle phi"""
            angles = xn_axis * phi
            R_val = jnp.sum(rc_axis * jnp.cos(angles))
            Z = -jnp.sum(zs_axis * jnp.sin(angles))
            x = R_val * jnp.cos(phi)
            y = R_val * jnp.sin(phi)
            return jnp.array([x, y, Z])
        
        # Compute arc-length parametrization along the axis
        phi_arc = jnp.linspace(0, 2 * jnp.pi, n_arc_samples, endpoint=True)
        axis_arc_pts = jnp.array([compute_axis_point(p) for p in phi_arc])
        
        # Compute arc-length
        deltas = jnp.linalg.norm(jnp.diff(axis_arc_pts, axis=0), axis=1)
        cumulative_arc = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(deltas)])
        total_arc = cumulative_arc[-1]
        
        # Generate random arc-length positions
        key = jax.random.key(random_seed)
        key_arcs, key_thetas, key_vparallel = jax.random.split(key, 3)
        
        random_arcs = jax.random.uniform(key_arcs, (n_particles,)) * total_arc
        random_thetas = jax.random.uniform(key_thetas, (n_particles,)) * 2 * jnp.pi  # Poloidal angle
        
        # Map arc-length positions back to phi coordinates
        particle_phis = jnp.interp(random_arcs, cumulative_arc, phi_arc)
        
        # Compute axis positions and Frenet frames at particle locations
        def compute_particle_position(phi, theta, distance):
            """Compute particle position on/around axis using Frenet frame"""
            # Axis point at this phi
            axis_pos = compute_axis_point(phi)
            
            # Compute Frenet frame (tangent, normal, binormal)
            # Tangent: derivative along phi (using finite differences)
            eps = 1e-8
            axis_plus = compute_axis_point(phi + eps)
            axis_minus = compute_axis_point(phi - eps)
            tangent = (axis_plus - axis_minus) / (2 * eps)
            tangent = tangent / jnp.maximum(jnp.linalg.norm(tangent), 1e-12)

            # Build a robust orthonormal frame perpendicular to tangent.
            # This avoids degeneracy when axis-only Fourier data has zero poloidal derivative.
            ref = jnp.array([0.0, 0.0, 1.0])
            use_x = jnp.abs(jnp.dot(ref, tangent)) > 0.9
            ref = jnp.where(use_x, jnp.array([1.0, 0.0, 0.0]), ref)

            dot_rt = jnp.dot(ref, tangent)
            normal = ref - dot_rt * tangent
            normal = normal / jnp.maximum(jnp.linalg.norm(normal), 1e-12)
            
            # Binormal: tangent × normal
            binormal = jnp.cross(tangent, normal)
            binormal = binormal / jnp.maximum(jnp.linalg.norm(binormal), 1e-12)

            direction = jnp.cos(theta) * normal + jnp.sin(theta) * binormal
            direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-12)

            if distance_mode == 'fraction_to_boundary':
                max_distance = axis_to_boundary_distance(axis_pos, direction)
                actual_distance = distance * max_distance
            else:
                actual_distance = distance
            
            # Position: axis + distance * direction in local normal-binormal plane
            position = axis_pos + actual_distance * direction
            
            return position
        
        # Compute all particle positions
        initial_xyz = jnp.array([compute_particle_position(phi, theta, distance_from_axis) 
                                 for phi, theta in zip(particle_phis, random_thetas)])
        
        # Generate random parallel velocity fractions
        initial_vparallel_over_v = jax.random.uniform(key_vparallel, (n_particles,), 
                                                       minval=min_vparallel_over_v, 
                                                       maxval=max_vparallel_over_v)
        
        # Create and return Particles object
        return cls(initial_xyz=initial_xyz, 
                  initial_vparallel_over_v=initial_vparallel_over_v,
                  charge=charge, 
                  mass=mass, 
                  energy=energy,
                  field=field)



@partial(jit, static_argnums=(2))
def GuidingCenterCollisionsDiffusionMu(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z, vpar,mu = initial_condition
    field, particles,_,species,_ = args
    vpar=SPEED_OF_LIGHT*vpar
    mu=SPEED_OF_LIGHT**2*particles.mass*mu    
    q = particles.charge
    m = particles.mass
    points = jnp.array([x, y, z])
    #I_bb_tensor=jnp.identity(3)-jnp.diag(jnp.multiply(B_contravariant,B_contravariant))/AbsB**2
    I_bb_tensor=jnp.identity(3)-jnp.diag(jnp.multiply(field.B_contravariant(points),jnp.reshape(field.B_contravariant(points),(3,1))))/field.AbsB(points)**2
    v=jnp.sqrt(2./m*(0.5*m*vpar**2+mu*field.AbsB(points)))
    xi=vpar/v
    p=m*v
    indeces_species=species.species_indeces
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    Diffusion_par=p**2*nu_par/2.
    Diffusion_perp=p**2*nu_D/2. 
    Diffusion_x=0.0#((Diffusion_par-Diffusion_perp)*(1.-xi**2)/2.+Diffusion_perp)/(m*omega_mod)**2
    Yvv=(Diffusion_par*xi**2+Diffusion_perp*(1.-xi**2))/p**2
    Yvmu=2.*xi*(1.-xi**2)*(Diffusion_par-Diffusion_perp)/p**2
    Ymumu=4.*(1.-xi**2)*(Diffusion_par*(1.-xi**2)+Diffusion_perp*xi**2)/p**2 
    lambda_p=0.5*(Yvv+Ymumu+jnp.sqrt((Yvv-Ymumu)**2+4.*Yvmu**2))
    lambda_m=0.5*(Yvv+Ymumu-jnp.sqrt((Yvv-Ymumu)**2+4.*Yvmu**2))
    Q1=jnp.reshape(jnp.array([1, Yvmu/(lambda_p-Ymumu)])/jnp.sqrt(1.+(Yvmu/(lambda_p-Ymumu))**2),(2,1))
    Q2=jnp.reshape(jnp.array([ Yvmu/(lambda_m-Yvv),1])/jnp.sqrt(1.+(Yvmu/(lambda_m-Yvv))**2),(2,1)) 
    mat1=jnp.diag(jnp.array([v,0.5*m*v**2/field.AbsB(points)]))
    mat2=jnp.append(Q1,Q2,axis=1)
    mat3=jnp.diag(jnp.array([jnp.sqrt(2.*lambda_p),jnp.sqrt(2.*lambda_m)]))
    sigma=jnp.select(condlist=[jnp.abs(xi)<1,jnp.abs(xi)==1],choicelist=[jnp.matmul(mat1,jnp.matmul(mat2,mat3)),jnp.diag(jnp.array([jnp.sqrt(2.*Diffusion_par)/m,0.]))])
    dxdt = jnp.sqrt(2.*Diffusion_x)*I_bb_tensor
    sigma=sigma.at[0,:].set(sigma.at[0,:].get()/SPEED_OF_LIGHT)
    sigma=sigma.at[1,:].set(sigma.at[1,:].get()/(SPEED_OF_LIGHT**2*particles.mass) )   
    #Off diagonals between position an dvelocity are zero at zeroth order
    Dxv=jnp.zeros((2,3))
    Dvx=jnp.zeros((3,2))
    return jnp.append(jnp.append(dxdt,Dxv,axis=0),jnp.append(Dvx,sigma,axis=0),axis=1)


@partial(jit, static_argnums=(2))
def GuidingCenterCollisionsDriftMuStratonovich(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z,vpar,mu = initial_condition
    field, particles,electric_field,species,tag_gc = args
    #jax.debug.print("vpar  {x}", x=vpar)
    #jax.debug.print("mu {x}", x=mu)  
    vpar=SPEED_OF_LIGHT*vpar
    mu=SPEED_OF_LIGHT**2*particles.mass*mu
    m = particles.mass
    q=particles.charge
    points = jnp.array([x, y, z]) 
    v=jnp.sqrt(2./m*(0.5*m*vpar**2+mu*field.AbsB(points)))
    p=m*v
    xi=vpar/v
    #xi=jnp.select(condlist=[jnp.abs(xi)<=1,jnp.abs(xi)>1],choicelist=[jnp.sign(xi)*(2.-jnp.abs(xi)),xi])
    #vpar=xi*v
    Bstar=field.B_contravariant(points)+vpar*m/q*field.curl_b(points)#+m/q*flow.curl_U0(points)
    Ustar=vpar*field.B_contravariant(points)/field.AbsB(points)#+flow.U0(points) 
    F_gc=mu*field.dAbsB_by_dX(points)+m*vpar**2*field.kappa(points)-q*electric_field.E_covariant(points)#+vpar*flow.coriolis(points)+flow.centrifugal(points)        
    indeces_species=species.species_indeces
    nu_s=jnp.sum(jax.vmap(nu_s_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    dnu_par_dv=jnp.sum(jax.vmap(d_nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    dnu_D_dv=jnp.sum(jax.vmap(d_nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)      
    Diffusion_par=p**2*nu_par/2.
    Diffusion_perp=p**2*nu_D/2.
    d_Diffusion_par_dp=p*nu_par+p**2*dnu_par_dv/(2.*m)
    d_Diffusion_perp_dp=p*nu_D+p**2*dnu_D_dv/(2.*m)    
    Yvv=(Diffusion_par*xi**2+Diffusion_perp*(1.-xi**2))/p**2
    Yvmu=2.*xi*(1.-xi**2)*(Diffusion_par-Diffusion_perp)/p**2
    Ymumu=4.*(1.-xi**2)*(Diffusion_par*(1.-xi**2)+Diffusion_perp*xi**2)/p**2 
    #Dmuv=2.*mu*vpar/p**2*(Diffusion_par-Diffusion_perp)
    #Dmumu=2.*mu/(m*field.AbsB(points))*((1-xi**2)(Diffusion_par-Diffusion_perp)+Diffusion_perp)
    #Dvv=Diffusion_perp/m**2*(1.-xi**2)+Diffusion_par/m**2*xi**2

    d_Dmuv_dvpar=2.*mu/p**2*((Diffusion_par-Diffusion_perp)+xi**2*p*(d_Diffusion_par_dp-d_Diffusion_perp_dp)-2.*xi**2*(Diffusion_par-Diffusion_perp))
    d_Dmuv_dmu=2.*vpar/p**2*((Diffusion_par-Diffusion_perp)+(1.-xi**2)*p/2.*(d_Diffusion_par_dp-d_Diffusion_perp_dp)-(1.-xi**2)*(Diffusion_par-Diffusion_perp))
    d_Dmumu_dvpar=2.*mu*vpar/(m*v**2*field.AbsB(points))*(p*d_Diffusion_perp_dp+(1.-xi**2)*p*(d_Diffusion_par_dp-d_Diffusion_perp_dp)-2.*(1.-xi**2)*(Diffusion_par-Diffusion_perp))
    d_Dmumu_dmu=2.*Diffusion_perp/(m*field.AbsB(points))+2.*mu/p**2*(4.*(Diffusion_par-Diffusion_perp)
                                                                        +(1.-xi**2)*p*(d_Diffusion_par_dp-d_Diffusion_perp_dp)
                                                                        -2.*(1.-xi**2)*(Diffusion_par-Diffusion_perp)
                                                                        +p*d_Diffusion_perp_dp)
    d_Dvv_dvpar=2.*vpar/p**2*(p/2.*d_Diffusion_par_dp-(1.-xi**2)*p/2.*(d_Diffusion_par_dp-d_Diffusion_perp_dp)+(1.-xi**2)*(Diffusion_par-Diffusion_perp))
    d_Dvv_dmu=2.*field.AbsB(points)/m/p**2*(p/2*d_Diffusion_par_dp-(Diffusion_par-Diffusion_perp)
                                            -(1.-xi**2)*p/2*(d_Diffusion_par_dp-d_Diffusion_perp_dp)+(1.-xi**2)*(Diffusion_par-Diffusion_perp))



    d_Yvmu_dmu=-3.*field.AbsB(points)/(m*v**2)*Yvmu+2.*field.AbsB(points)/(m*v**3)*d_Dmuv_dmu
    d_Yvmu_dvpar=-3./v*xi*Yvmu+2.*field.AbsB(points)/(m*v**3)*d_Dmuv_dvpar
    d_Ymumu_dmu=-4.*field.AbsB(points)/(m*v**2)*Ymumu+4.*field.AbsB(points)**2/(m**2*v**4)*d_Dmumu_dmu
    d_Ymumu_dvpar=-4./v*xi*Ymumu+4.*field.AbsB(points)**2/(m**2*v**4)*d_Dmumu_dvpar
    d_Yvv_dmu=-2.*field.AbsB(points)/(m*v**2)*Yvv+d_Dvv_dmu/v**2
    d_Yvv_dvpar=-2./v*xi*Yvv+d_Dvv_dvpar/v**2

    lambda_p=0.5*(Yvv+Ymumu+jnp.sqrt((Yvv-Ymumu)**2+4.*Yvmu**2))
    lambda_m=0.5*(Yvv+Ymumu-jnp.sqrt((Yvv-Ymumu)**2+4.*Yvmu**2))

    d_lambda_p_dvpar=0.5*(d_Yvv_dvpar+d_Ymumu_dvpar+((Yvv-Ymumu)*(d_Yvv_dvpar-d_Ymumu_dvpar)+4.*Yvmu*d_Yvmu_dvpar)/jnp.sqrt((Yvv-Ymumu)**2+4.*Yvmu**2))
    d_lambda_p_dmu=0.5*(d_Yvv_dmu+d_Ymumu_dmu+((Yvv-Ymumu)*(d_Yvv_dmu-d_Ymumu_dmu)+4.*Yvmu*d_Yvmu_dmu)/jnp.sqrt((Yvv-Ymumu)**2+4.*Yvmu**2))
    d_lambda_m_dvpar=0.5*(d_Yvv_dvpar+d_Ymumu_dvpar-((Yvv-Ymumu)*(d_Yvv_dvpar-d_Ymumu_dvpar)+4.*Yvmu*d_Yvmu_dvpar)/jnp.sqrt((Yvv-Ymumu)**2+4.*Yvmu**2))
    d_lambda_m_dmu=0.5*(d_Yvv_dmu+d_Ymumu_dmu-((Yvv-Ymumu)*(d_Yvv_dmu-d_Ymumu_dmu)+4.*Yvmu*d_Yvmu_dmu)/jnp.sqrt((Yvv-Ymumu)**2+4.*Yvmu**2))

    Q1=jnp.reshape(jnp.array([1, Yvmu/(lambda_p-Ymumu)])/jnp.sqrt(1.+(Yvmu/(lambda_p-Ymumu))**2),(2,1))
    Q2=jnp.reshape(jnp.array([ Yvmu/(lambda_m-Yvv),1])/jnp.sqrt(1.+(Yvmu/(lambda_m-Yvv))**2),(2,1))

    d_Q11_dvpar=-Q1.at[1].get()*Q1.at[0].get()**2*(d_Yvmu_dvpar*(lambda_p-Ymumu)-Yvmu*(d_lambda_p_dvpar-d_Ymumu_dvpar))/(lambda_p-Ymumu)**2 
    d_Q11_dmu=-Q1.at[1].get()*Q1.at[0].get()**2*(d_Yvmu_dmu*(lambda_p-Ymumu)-Yvmu*(d_lambda_p_dmu-d_Ymumu_dmu))/(lambda_p-Ymumu)**2 
    d_Q21_dvpar=Q1.at[0].get()*(d_Yvmu_dvpar*(lambda_p-Ymumu)-Yvmu*(d_lambda_p_dvpar-d_Ymumu_dvpar))/(lambda_p-Ymumu)**2+d_Q11_dvpar*(Yvmu/(lambda_p-Ymumu))
    d_Q21_dmu=Q1.at[0].get()*(d_Yvmu_dmu*(lambda_p-Ymumu)-Yvmu*(d_lambda_p_dmu-d_Ymumu_dmu))/(lambda_p-Ymumu)**2+d_Q11_dmu*(Yvmu/(lambda_p-Ymumu)) 
    d_Q22_dvpar=-Q2.at[0].get()*Q2.at[1].get()**2*(d_Yvmu_dvpar*(lambda_m-Yvv)-Yvmu*(d_lambda_m_dvpar-d_Yvv_dvpar))/(lambda_m-Yvv)**2 
    d_Q22_dmu=-Q2.at[0].get()*Q2.at[1].get()**2*(d_Yvmu_dmu*(lambda_m-Yvv)-Yvmu*(d_lambda_m_dmu-d_Yvv_dmu))/(lambda_m-Yvv)**2 
    d_Q12_dvpar=Q2.at[1].get()*(d_Yvmu_dvpar*(lambda_m-Yvv)-Yvmu*(d_lambda_m_dvpar-d_Yvv_dvpar))/(lambda_m-Yvv)**2+d_Q22_dvpar*(Yvmu/(lambda_m-Yvv))
    d_Q12_dmu=Q2.at[1].get()*(d_Yvmu_dmu*(lambda_m-Yvv)-Yvmu*(d_lambda_m_dmu-d_Yvv_dmu))/(lambda_m-Yvv)**2+d_Q22_dmu*(Yvmu/(lambda_m-Yvv)) 

    #d_Q11_dvpar=-1./(1.+(Yvmu/(lambda_p-Ymumu))**2)**(1.5)*(Yvmu/(lambda_p-Ymumu))*(d_Yvmu_dvpar*(lambda_p-Ymumu)-Yvmu*(d_lambda_p_dvpar-d_Ymumu_dvpar))/(lambda_p-Ymumu)**2 
    #d_Q11_dmu=-1./(1.+(Yvmu/(lambda_p-Ymumu))**2)**(1.5)*(Yvmu/(lambda_p-Ymumu))*(d_Yvmu_dmu*(lambda_p-Ymumu)-Yvmu*(d_lambda_p_dmu-d_Ymumu_dmu))/(lambda_p-Ymumu)**2   

    #d_Q22_dvpar=-1./(1.+(Yvmu/(lambda_m-Yvv))**2)**(1.5)*(Yvmu/(lambda_m-Yvv))*(d_Yvmu_dvpar*(lambda_m-Yvv)-Yvmu*(d_lambda_m_dvpar-d_Yvv_dvpar))/(lambda_m-Yvv)**2 
    #d_Q22_dmu=-1./(1.+(Yvmu/(lambda_m-Yvv))**2)**(1.5)*(Yvmu/(lambda_m-Yvv))*(d_Yvmu_dmu*(lambda_m-Yvv)-Yvmu*(d_lambda_m_dmu-d_Yvv_dmu))/(lambda_m-Yvv)**2  

    #d_Q21_dvpar=-d_Q11_dvpar*(lambda_p-Ymumu)/Yvmu
    #d_Q21_dmu=-d_Q11_dmu*(lambda_p-Ymumu)/Yvmu

    #d_Q12_dvpar=-d_Q22_dvpar*(lambda_m-Yvv)/Yvmu 
    #d_Q12_dmu=-d_Q22_dmu*(lambda_m-Yvv)/Yvmu 
    sigma11=v*Q1.at[0].get()*jnp.sqrt(2.*lambda_p)
    sigma21=0.5*v**2*m/field.AbsB(points)*Q1.at[1].get()*jnp.sqrt(2.*lambda_p)
    sigma12=v*Q2.at[0].get()*jnp.sqrt(2.*lambda_m)
    sigma22=0.5*v**2*m/field.AbsB(points)*Q2.at[1].get()*jnp.sqrt(2.*lambda_m) 

    d_sigma11_dvpar=xi*Q1.at[0].get()*jnp.sqrt(2.*lambda_p)+v*d_Q11_dvpar*jnp.sqrt(2.*lambda_p)+v*Q1.at[0].get()*jnp.sqrt(2.)*d_lambda_p_dvpar/(2.*jnp.sqrt(lambda_p))  
    d_sigma11_dmu=field.AbsB(points)/(m*v)*Q1.at[0].get()*jnp.sqrt(2.*lambda_p)+v*d_Q11_dmu*jnp.sqrt(2.*lambda_p)+v*Q1.at[0].get()*jnp.sqrt(2.)*d_lambda_p_dmu/(2.*jnp.sqrt(lambda_p))      
    d_sigma12_dvpar=xi*Q2.at[0].get()*jnp.sqrt(2.*lambda_m)+v*d_Q12_dvpar*jnp.sqrt(2.*lambda_m)+v*Q2.at[0].get()*jnp.sqrt(2.)*d_lambda_m_dvpar/(2.*jnp.sqrt(lambda_m))    
    d_sigma12_dmu=field.AbsB(points)/(m*v)*Q2.at[0].get()*jnp.sqrt(2.*lambda_m)+v*d_Q12_dmu*jnp.sqrt(2.*lambda_m)+v*Q2.at[0].get()*jnp.sqrt(2.)*d_lambda_m_dmu/(2.*jnp.sqrt(lambda_m))      
    d_sigma21_dvpar=m*v/field.AbsB(points)*xi*Q1.at[1].get()*jnp.sqrt(2.*lambda_p)+0.5*m*v**2/field.AbsB(points)*d_Q21_dvpar*jnp.sqrt(2.*lambda_p)+0.5*m*v**2/field.AbsB(points)*Q1.at[1].get()*jnp.sqrt(2.)*d_lambda_p_dvpar/(2.*jnp.sqrt(lambda_p))    
    d_sigma21_dmu=Q1.at[1].get()*jnp.sqrt(2.*lambda_p)+0.5*m*v**2/field.AbsB(points)*d_Q21_dmu*jnp.sqrt(2.*lambda_p)+0.5*m*v**2/field.AbsB(points)*Q1.at[1].get()*jnp.sqrt(2.)*d_lambda_p_dmu/(2.*jnp.sqrt(lambda_p))      
    d_sigma22_dvpar=m*v/field.AbsB(points)*xi*Q2.at[1].get()*jnp.sqrt(2.*lambda_m)+0.5*m*v**2/field.AbsB(points)*d_Q22_dvpar*jnp.sqrt(2.*lambda_m)+0.5*m*v**2/field.AbsB(points)*Q2.at[1].get()*jnp.sqrt(2.)*d_lambda_m_dvpar/(2.*jnp.sqrt(lambda_m))    
    d_sigma22_dmu=Q2.at[1].get()*jnp.sqrt(2.*lambda_m)+0.5*m*v**2/field.AbsB(points)*d_Q22_dmu*jnp.sqrt(2.*lambda_m)+0.5*m*v**2/field.AbsB(points)*Q2.at[1].get()*jnp.sqrt(2.)*d_lambda_m_dmu/(2.*jnp.sqrt(lambda_m))        

    Avpar_corr=jnp.select(condlist=[jnp.abs(xi)<1,jnp.abs(xi)==1],choicelist=[-0.5*(sigma11*d_sigma11_dvpar+sigma12*d_sigma12_dvpar+sigma21*d_sigma11_dmu+sigma22*d_sigma12_dmu),-0.5*vpar/p**2*(p*d_Diffusion_par_dp)])
    Amu_corr=jnp.select(condlist=[jnp.abs(xi)<1,jnp.abs(xi)==1],choicelist=[-0.5*(sigma11*d_sigma21_dvpar+sigma12*d_sigma22_dvpar+sigma21*d_sigma21_dmu+sigma22*d_sigma22_dmu),-0.5*(d_Dmumu_dmu+d_Dmuv_dvpar)])  

    Avpar=-nu_s*vpar+d_Dvv_dvpar+d_Dmuv_dmu+Avpar_corr
    Amu=-nu_s*2.*mu+d_Dmumu_dmu+d_Dmuv_dvpar+Amu_corr
    dxdt =  tag_gc*(Ustar + jnp.cross(field.B_covariant(points), F_gc)/jnp.dot(field.B_covariant(points),Bstar)/q/field.sqrtg(points))
    dvpardt = (-jnp.dot(Bstar,F_gc)/jnp.dot(field.B_covariant(points),Bstar)*field.AbsB(points)/m*tag_gc+Avpar)/SPEED_OF_LIGHT

    dmudt = Amu/(SPEED_OF_LIGHT**2*particles.mass)  
    return jnp.append(dxdt,jnp.append(dvpardt,dmudt))



@partial(jit, static_argnums=(2))
def GuidingCenterCollisionsDriftMuIto(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z,vpar,mu = initial_condition
    field, particles,electric_field,species,tag_gc = args 
    vpar=SPEED_OF_LIGHT*vpar
    mu=SPEED_OF_LIGHT**2*particles.mass*mu
    m = particles.mass
    q=particles.charge
    points = jnp.array([x, y, z]) 
    v=jnp.sqrt(2./m*(0.5*m*vpar**2+mu*field.AbsB(points)))
    p=m*v
    xi=vpar/v

    Bstar=field.B_contravariant(points)+vpar*m/q*field.curl_b(points)#+m/q*flow.curl_U0(points)
    Ustar=vpar*field.B_contravariant(points)/field.AbsB(points)#+flow.U0(points) 
    F_gc=mu*field.dAbsB_by_dX(points)+m*vpar**2*field.kappa(points)-q*electric_field.E_covariant(points)#+vpar*flow.coriolis(points)+flow.centrifugal(points)        
    indeces_species=species.species_indeces
    nu_s=jnp.sum(jax.vmap(nu_s_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    dnu_par_dv=jnp.sum(jax.vmap(d_nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    dnu_D_dv=jnp.sum(jax.vmap(d_nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)      
    Diffusion_par=p**2*nu_par/2.
    Diffusion_perp=p**2*nu_D/2.
    d_Diffusion_par_dp=p*nu_par+p**2*dnu_par_dv/(2.*m)
    d_Diffusion_perp_dp=p*nu_D+p**2*dnu_D_dv/(2.*m)    

    d_Dmuv_dvpar=2.*mu/p**2*((Diffusion_par-Diffusion_perp)+xi**2*p*(d_Diffusion_par_dp-d_Diffusion_perp_dp)-2.*xi**2*(Diffusion_par-Diffusion_perp))
    d_Dmuv_dmu=2.*vpar/p**2*((Diffusion_par-Diffusion_perp)+(1.-xi**2)*p/2.*(d_Diffusion_par_dp-d_Diffusion_perp_dp)-(1.-xi**2)*(Diffusion_par-Diffusion_perp))
    d_Dmumu_dmu=2.*Diffusion_perp/(m*field.AbsB(points))+2.*mu/p**2*(4.*(Diffusion_par-Diffusion_perp)
                                                                        +(1.-xi**2)*p*(d_Diffusion_par_dp-d_Diffusion_perp_dp)
                                                                        -2.*(1.-xi**2)*(Diffusion_par-Diffusion_perp)
                                                                        +p*d_Diffusion_perp_dp)
    d_Dvv_dvpar=2.*vpar/p**2*(p/2.*d_Diffusion_par_dp-(1.-xi**2)*p/2.*(d_Diffusion_par_dp-d_Diffusion_perp_dp)+(1.-xi**2)*(Diffusion_par-Diffusion_perp))

    Avpar=-nu_s*vpar+d_Dvv_dvpar+d_Dmuv_dmu
    Amu=-nu_s*2.*mu+d_Dmumu_dmu+d_Dmuv_dvpar
    dxdt =  tag_gc*(Ustar + jnp.cross(field.B_covariant(points), F_gc)/jnp.dot(field.B_covariant(points),Bstar)/q/field.sqrtg(points))
    dvpardt = (-jnp.dot(Bstar,F_gc)/jnp.dot(field.B_covariant(points),Bstar)*field.AbsB(points)/m*tag_gc+Avpar)/SPEED_OF_LIGHT

    dmudt = Amu/(SPEED_OF_LIGHT**2*particles.mass) 
    return jnp.append(dxdt,jnp.append(dvpardt,dmudt))

@partial(jit, static_argnums=(2))
def GuidingCenterCollisionsDiffusion(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z, v,xi = initial_condition
    field, particles,electric_field,species,tag_gc = args
    q = particles.charge
    m = particles.mass

    points = jnp.array([x, y, z])
    I_bb_tensor=jnp.identity(3)-jnp.diag(jnp.multiply(field.B_contravariant(points),jnp.reshape(field.B_contravariant(points),(3,1))))/field.AbsB(points)**2
    p=m*v
    indeces_species=species.species_indeces
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    Diffusion_par=p**2/2.*nu_par
    Diffusion_perp=p**2/2.*nu_D 
    Diffusion_x=0.0#((Diffusion_par-Diffusion_perp)*(1.-xi**2)/2.+Diffusion_perp)/(m*omega_mod)**2
    dxdt = jnp.sqrt(2.*Diffusion_x)*I_bb_tensor
    dvdt=jnp.sqrt(2.*Diffusion_par)/m   #equation format was in p=m*v so we divide by m)
    dxidt=jnp.sqrt((1.-xi**2)*2.*Diffusion_perp/p**2)
    #jnp.select(condlist=[jnp.abs(xi)<1,jnp.abs(xi)==1],choicelist=[jnp.sqrt((1.-xi**2)*2.*Diffusion_perp/p**2),0.])
    #Off diagonals between position an dvelocity are zero at zeroth order
    Dxv=jnp.zeros((2,3))
    Dvx=jnp.zeros((3,2))
    return jnp.append(jnp.append(dxdt,Dxv,axis=0),jnp.append(Dvx,jnp.diag(jnp.append(dvdt,dxidt)),axis=0),axis=1)

@partial(jit, static_argnums=(2))
def GuidingCenterCollisionsDrift(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z, v,xi = initial_condition
    field, particles,electric_field,species,tag_gc = args
    q = particles.charge
    m = particles.mass

    vpar=xi*v

    points = jnp.array([x, y, z])
    mu = (m*v**2/2 - m*vpar**2/2)/field.AbsB(points)
    p=m*v
    Bstar=field.B_contravariant(points)+vpar*m/q*field.curl_b(points)#+m/q*flow.curl_U0(points)
    Ustar=vpar*field.B_contravariant(points)/field.AbsB(points)#+flow.U0(points) 
    F_gc=mu*field.dAbsB_by_dX(points)+m*vpar**2*field.kappa(points)-q*electric_field.E_covariant(points)#+vpar*flow.coriolis(points)+flow.centrifugal(points)    
    indeces_species=species.species_indeces
    nu_s=jnp.sum(jax.vmap(nu_s_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    dnu_par=jnp.sum(jax.vmap(d_nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    Diffusion_par=p**2/2.*nu_par
    Diffusion_perp=p**2/2.*nu_D 
    d_Diffusion_par_dp=p*nu_par+p**2/2.*dnu_par/m
    dxdt =  tag_gc*(Ustar + jnp.cross(field.B_covariant(points), F_gc)/jnp.dot(field.B_covariant(points),Bstar)/q/field.sqrtg(points))

    dvdt=(-nu_s*p+2.*Diffusion_par/p+d_Diffusion_par_dp*0.5)/m  #equation format was in p=m*v so we divide by m)
    dxidt = -jnp.dot(Bstar,F_gc)/jnp.dot(field.B_covariant(points),Bstar)*field.AbsB(points)/m/v*tag_gc-xi*2.*Diffusion_perp/p**2*0.5

    return jnp.append(dxdt,jnp.append(dvdt,dxidt))




def _gc_quantities(field, points):
    """Guiding-center field quantities, fused when the field provides them.

    Fields that do not override ``MagneticField.gc_quantities`` (for example
    :class:`Vmec`, or a field that is not a pytree) use their individual
    methods; the choice is made at trace time.
    """
    fused = getattr(type(field), "gc_quantities", None)
    if fused is not None and fused is not MagneticField.gc_quantities:
        return field.gc_quantities(points)
    return (field.B_covariant(points), field.B_contravariant(points), field.AbsB(points),
            field.dAbsB_by_dX(points), field.curl_b(points), field.kappa(points),
            field.sqrtg(points))


def _guiding_center_velocity(field, electric_field, q, m, points, vpar, mu=None, energy=None):
    """Guiding-center dx/dt and dv_par/dt, at magnetic moment ``mu`` or, if None, at ``energy``."""
    # One evaluation of every field quantity (fused for Biot-Savart fields).
    B_cov, B_contra, AbsB, dAbsB, curl_b, kappa, sqrtg = _gc_quantities(field, points)
    mu = (energy - m*vpar**2/2)/AbsB if mu is None else mu
    Bstar=B_contra+vpar*m/q*curl_b#+m/q*flow.curl_U0(points)
    Ustar=vpar*B_contra/AbsB#+flow.U0(points)
    F_gc=mu*dAbsB+m*vpar**2*kappa-q*electric_field.E_covariant(points)#+vpar*flow.coriolis(points)+flow.centrifugal(points)
    dxdt =  Ustar + jnp.cross(B_cov, F_gc)/jnp.dot(B_cov,Bstar)/q/sqrtg
    dvdt = -jnp.dot(Bstar,F_gc)/jnp.dot(B_cov,Bstar)*AbsB/m
    return dxdt, dvdt


@partial(jit, static_argnums=(2))
def GuidingCenter(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z, vpar = initial_condition
    field, particles,electric_field = args
    dxdt, dvdt = _guiding_center_velocity(field, electric_field, particles.charge, particles.mass,
                                          jnp.array([x, y, z]), vpar, energy=particles.energy)
    return jnp.append(dxdt,dvdt)


@partial(jit, static_argnums=(2))
def GuidingCenterMu(t, initial_condition, args) -> jnp.ndarray:
    """Collisionless guiding center with state (x, y, z, v_par, mu); mu is constant."""
    field, particles, electric_field = args
    dxdt, dvdt = _guiding_center_velocity(field, electric_field, particles.charge, particles.mass,
                                          initial_condition[:3], initial_condition[3], mu=initial_condition[4])
    return jnp.concatenate([dxdt, jnp.array([dvdt, 0.0])])
    # def zero_derivatives(_):
    #     return jnp.zeros(4, dtype=float)
    # return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)


@partial(jit, static_argnums=(2))
def LorentzCollisionsDiffusion(t,
            initial_condition,
            args) -> jnp.ndarray:
    x, y, z, vx, vy, vz = initial_condition
    field, particles,species = args
    q = particles.charge
    m = particles.mass
    #E = m/2*v**2 
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    v_vector=jnp.array([vx, vy, vz])
    v=jnp.sqrt(vx**2+vy**2+vz**2)
    p=m*v
    P_v=jnp.outer(v_vector,v_vector)/v**2  # projector on the velocity direction
    indeces_species=species.species_indeces
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    Diffusion_par=p**2/2.*nu_par
    Diffusion_perp=p**2/2.*nu_D 
    Dpar=jnp.sqrt(2.*Diffusion_par)#*0.0000
    Dperp=jnp.sqrt(2.*Diffusion_perp)#*0.0000
    dxdt = jnp.zeros((3,3))
    dvdt=Dpar/m*P_v+Dperp/m*(jnp.identity(3)-P_v)
    #Off diagonals between position an dvelocity are zero at zeroth order
    Dxv=jnp.zeros((3,3))
    Dvx=jnp.zeros((3,3))
    return jnp.append(jnp.append(dxdt,Dxv,axis=0),jnp.append(Dvx,dvdt,axis=0),axis=1)

@partial(jit, static_argnums=(2))
def LorentzCollisionsDrift(t,
            initial_condition,
            args) -> jnp.ndarray:
    x, y, z, vx, vy, vz = initial_condition
    field, particles,species = args
    q = particles.charge
    m = particles.mass
    v=jnp.sqrt(vx**2+vy**2+vz**2)
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(points)
    indeces_species=species.species_indeces
    nu_s=jnp.sum(jax.vmap(nu_s_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    dxdt = jnp.array([vx, vy, vz])
    dvdt =  q / m * jnp.cross(dxdt, B_contravariant)-nu_s*dxdt#*0.00000
    return jnp.append(dxdt, dvdt)
    # def zero_derivatives(_):
    #     return jnp.zeros(6, dtype=float)
    # return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)





@partial(jit, static_argnums=(2))
def Lorentz(t,
            initial_condition,
            args) -> jnp.ndarray:
    x, y, z, vx, vy, vz = initial_condition
    field, particles = args
    q = particles.charge
    m = particles.mass
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(points)
    dxdt = jnp.array([vx, vy, vz])
    dvdt = q / m * jnp.cross(dxdt, B_contravariant)
    return jnp.append(dxdt, dvdt)
    # def zero_derivatives(_):
    #     return jnp.zeros(6, dtype=float)
    # return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@partial(jit, static_argnums=(2))
def FieldLine(t,
              initial_condition,
              field) -> jnp.ndarray:
    x, y, z = initial_condition
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def compute_derivatives(_):
    position = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(position)
    dxdt = B_contravariant
    return dxdt
    # def zero_derivatives(_):
    #     return jnp.zeros(3, dtype=float)
    # return lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)


@partial(jit, static_argnums=(2))
def FieldLineArclength(t, initial_condition, field) -> jnp.ndarray:
    """Trace the same field line with physical arclength as the parameter."""
    del t
    B = field.B_contravariant(initial_condition)
    return B / jnp.maximum(jnp.linalg.norm(B), jnp.finfo(B.dtype).tiny)


@partial(jit, static_argnums=(2))
def FieldLineToroidal(t, initial_condition, field) -> jnp.ndarray:
    """Trace a flux-coordinate field with toroidal angle as the parameter."""
    del t
    B = field.B_contravariant(initial_condition)
    return B / B[2]


def _fill_stopped_trajectories(trajectories, criteria, args):
    """Hold each trajectory at its last point inside all level sets."""

    def fill_trajectory(trajectory):
        def remains_inside(state):
            values = [criterion(0.0, state, args) for criterion in criteria]
            return jnp.all(jnp.stack(values) > 0.0) & jnp.isfinite(state).all()

        def fill_state(carry, current):
            previous, active = carry
            active = active & remains_inside(current)
            state = jnp.where(active, current, previous)
            return (state, active), state

        initial_active = remains_inside(trajectory[0])
        _, tail = lax.scan(fill_state, (trajectory[0], initial_active), trajectory[1:])
        return jnp.vstack((trajectory[0], tail))

    return vmap(fill_trajectory)(trajectories)


_VMEC_GUIDING_CENTER_MODELS = frozenset(
    {
        "GuidingCenter",
        "GuidingCenterAdaptative",
        "GuidingCenterCollisions",
        "GuidingCenterCollisionsMuIto",
        "GuidingCenterCollisionsMuFixed",
        "GuidingCenterCollisionsMuAdaptative",
    }
)

_GUIDING_CENTER_COLLISION_MODELS = frozenset(
    {
        "GuidingCenterCollisions",
        "GuidingCenterCollisionsMuIto",
        "GuidingCenterCollisionsMuFixed",
        "GuidingCenterCollisionsMuAdaptative",
    }
)


# Extent in s of the region around the magnetic axis where _axis_regular
# advances the poloidal angle as a rotation of (u, w).
_AXIS_REGION = 1e-2


_AXIS_SEED = 1e-12  # smallest s of a seed: the VMEC Jacobian vanishes on the axis itself


def _to_axis_regular(y):
    """Map (s, theta, ...) to (sqrt(s) cos theta, sqrt(s) sin theta, ..., 0).

    A seed exactly on the axis (s = 0) is moved to s = _AXIS_SEED, where the
    guiding-center velocity is finite; orbits that later pass the axis never
    land on s = 0 exactly.
    """
    r = jnp.sqrt(jnp.maximum(y[0], _AXIS_SEED))
    return jnp.concatenate([jnp.array([r * jnp.cos(y[1]), r * jnp.sin(y[1])]), y[2:], jnp.zeros(1)])


def _from_axis_regular(y):
    """Map (u, w, ..., Theta) back to (s, theta, ...), with theta in [0, 2 pi)."""
    s = y[0]**2 + y[1]**2
    theta = jnp.mod(jnp.arctan2(y[1], y[0]) + y[-1], 2 * jnp.pi)
    return jnp.concatenate([jnp.array([s, jnp.where(jnp.isfinite(s), theta, s)]), y[2:-1]])


def _axis_regular(vector_field):
    """Express a VMEC guiding-center vector field in a chart that is regular on the axis.

    The state is (u, w, ..., Theta) with (u, w) = sqrt(s) (cos a, sin a) and
    theta = a + Theta. (s, theta) is singular on the magnetic axis and (u, w)
    is not, so orbits cross it instead of stopping there. Of the poloidal
    rotation dtheta/dt, the fraction c = s / (s + _AXIS_REGION) goes to Theta
    and the rest rotates (u, w):
    du/dt = u/(2s) ds/dt - (1 - c) w dtheta/dt, dw/dt = w/(2s) ds/dt + (1 - c) u dtheta/dt,
    dTheta/dt = c dtheta/dt, all regular on the axis. Away from it, a
    fixed-step solver then advances theta as an angle, not by rotating (u, w)
    with a truncation error that spirals orbits outward. Drift vectors and
    diffusion matrices transform alike; the map involves position only and
    the position has no noise, so Ito and Stratonovich forms need no extra
    drift.
    """
    def wrapped(t, y, args):
        u, w = y[0], y[1]
        s = u**2 + w**2
        half = jnp.where(s > 0, 0.5 / jnp.where(s > 0, s, 1.0), 0.0)
        c = s / (s + _AXIS_REGION)
        jacobian = jnp.array([[u * half, -(1 - c) * w], [w * half, (1 - c) * u]])
        f = vector_field(t, _from_axis_regular(y), args)
        return jnp.concatenate([jnp.tensordot(jacobian, f[:2], axes=1), f[2:], c * f[1:2]])
    return wrapped


# Outcome of each VMEC guiding center, ``Tracing.status``.
VMEC_STATUS = {
    0: "inside the LCFS at maxtime",
    1: "stopped at the LCFS (no exterior field)",
    2: "outside the LCFS at maxtime",
    3: "struck the wall",
    4: "failed: non-finite state or solver failure",
    5: "stopped after max_returns re-entries",
}

# Event times are refined to |dt| < 1e-8 t + 1e-11 s. Every event function
# starts on one side of zero, so its first sign change is the crossing sought.
_EVENT_ROOT_FINDER = optx.Newton(rtol=1e-8, atol=1e-11)


def _with_failure_flag(vector_field, drift=True):
    """Append a failure flag to the state of a vector field.

    An adaptive controller rejects every non-finite step until max_steps.
    Instead the drift returns zero with the flag's rate set to one, so the
    step is accepted and _failed_event stops the solve. The flag has no noise.
    """
    def wrapped(t, y, args):
        f = vector_field(t, y[:-1], args)
        bad = ~jnp.isfinite(f).all()
        f = jnp.where(bad, 0.0, f)
        if drift:
            return jnp.append(f, bad.astype(f.dtype))
        return jnp.concatenate([f, jnp.zeros((1,) + f.shape[1:], f.dtype)])
    return wrapped


def _failed_event(t, y, args, **kwargs):
    return (y[-1] > 0) | ~jnp.isfinite(y).all()


def _keep_energy(vpar, mu, E, B, m):
    """v_par and mu at energy E in a field of strength B, keeping the sign of v_par and, if possible, mu.

    The fields on the two sides of the LCFS differ slightly; where mu B > E the
    orbit mirrors there, with v_par = 0 and mu = E / B.
    """
    mu = jnp.minimum(mu, E / B)
    return jnp.sign(vpar) * jnp.sqrt(jnp.maximum(2 * (E - mu * B) / m, 0.0)), mu


# Drift, diffusion, solver and noise dimension of the stochastic models.
_SDE_MODELS = {
    'GuidingCenterCollisions': (GuidingCenterCollisionsDrift, GuidingCenterCollisionsDiffusion,
                                diffrax.StratonovichMilstein, 5),
    'GuidingCenterCollisionsMuFixed': (GuidingCenterCollisionsDriftMuStratonovich, GuidingCenterCollisionsDiffusionMu,
                                       diffrax.StratonovichMilstein, 5),
    'GuidingCenterCollisionsMuIto': (GuidingCenterCollisionsDriftMuIto, GuidingCenterCollisionsDiffusionMu,
                                     diffrax.ItoMilstein, 5),
    'GuidingCenterCollisionsMuAdaptative': (GuidingCenterCollisionsDriftMuStratonovich,
                                            GuidingCenterCollisionsDiffusionMu, diffrax.SPaRK, 5),
    'FullOrbitCollisions': (LorentzCollisionsDrift, LorentzCollisionsDiffusion, diffrax.SPaRK, 6),
}
_ADAPTIVE_MODELS = ('GuidingCenterAdaptative', 'FullOrbitAdaptative', 'FullOrbitCollisions',
                    'FieldLineAdaptative', 'FieldLineArclength', 'FieldLineToroidal')


class LevelsetStoppingCriterion:
    """Stop tracing when a signed-distance level set is crossed.

    ``classifier`` must be positive inside its reference surface. A positive
    ``maximum_distance`` permits tracing that far outside the surface before
    stopping.
    """

    def __init__(self, classifier, maximum_distance=0.0):
        if maximum_distance < 0.0:
            raise ValueError("maximum_distance must be non-negative")
        if not hasattr(classifier, "evaluate_xyz"):
            raise TypeError("classifier must provide evaluate_xyz(xyz)")
        self.classifier = classifier
        self.maximum_distance = float(maximum_distance)

    def __call__(self, t, y, args, **kwargs):
        del t, args, kwargs
        return self.classifier.evaluate_xyz(y[:3]) + self.maximum_distance


def _place_on_devices(x, target):
    """Place concrete arrays on ``target``; constrain tracers without a host copy."""
    if isinstance(x, jax.core.Tracer):
        if isinstance(target, NamedSharding):
            return lax.with_sharding_constraint(x, target)
        return x
    x = jax.device_get(x)
    if not jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        x = np.asarray(x)
    return device_put(x, target)



## !!!!  Here species and tag_gc were added  (E. Neto collisions modifications)
## species is a class for collision frquencies + possible temperature + density profiles in file species_background.py
## tag_gc is a tag to turn off 0, or on 1 the GC part of the equations for testing collision statistics independently of GC phsyics
## !!!!  Here particle_key was added to compute_trajectories (E. Neto collisions modifications)
## This is important for correct sampling of Brownian motion
class Tracing():
    def __init__(self, trajectories_input=None, initial_conditions=None, times_to_trace=None,
                 field=None, electric_field=None,model=None, maxtime: float = 1e-7, timestep: int = 1.e-8,
                 rtol= 1.e-7, atol = 1e-7, particles=None, condition=None,species=None,tag_gc=1.,boundary=None,rejected_steps=None,
                 solver=None, stopping_criteria=None, progress=False, devices=None,
                 max_steps=1_000_000, exterior_field=None, wall=None, max_returns=16, reentry_depth=None,
                 particle_batch_size=None):

        if condition is not None and stopping_criteria is not None:
            raise ValueError("Pass condition or stopping_criteria, not both")
        if stopping_criteria is not None:
            if callable(stopping_criteria):
                stopping_criteria = (stopping_criteria,)
            else:
                stopping_criteria = tuple(stopping_criteria)
            if not stopping_criteria or not all(callable(item) for item in stopping_criteria):
                raise ValueError("stopping_criteria must contain callable criteria")
            condition = stopping_criteria[0] if len(stopping_criteria) == 1 else stopping_criteria
        self.stopping_criteria = stopping_criteria
        self.devices = tuple(jax.devices() if devices is None else devices)
        if not self.devices:
            raise ValueError("devices must contain at least one JAX device")
        
        if electric_field==None:
            self.electric_field = Electric_field_zero()
        else:
            self.electric_field=electric_field

        if isinstance(field, Coils):
            self.field = BiotSavart(field)
        else:
            self.field = field

        # Both branches used to set 100, so a caller-supplied value was
        # silently discarded.
        self.rejected_steps = 100 if rejected_steps is None else rejected_steps

        self.model = model
        self.initial_conditions = initial_conditions
        self.times_to_trace = times_to_trace
        self.maxtime = maxtime
        self.timestep = timestep
        self.rtol = rtol
        self.atol = atol
        self._trajectories = trajectories_input
        self.particles = particles
        self.species=species
        self.tag_gc=tag_gc
        # VMEC guiding centers are traced in a chart that is regular on the
        # magnetic axis; see _axis_regular.
        self._axis_regular = isinstance(field, Vmec) and model in _VMEC_GUIDING_CENTER_MODELS
        # Without a user condition, VMEC guiding centers stop at the exact LCFS
        # crossing, report non-finite steps, and continue outside in
        # exterior_field up to the wall when one is given; see _vmec_orbit.
        self._vmec_default = self._axis_regular and condition is None
        if (exterior_field is not None or wall is not None) and not self._vmec_default:
            raise ValueError("exterior_field and wall need a VMEC field, a guiding-center model and no condition")
        if wall is not None and exterior_field is None:
            raise ValueError("a wall needs an exterior_field to trace the orbits outside the LCFS")
        if isinstance(exterior_field, Coils):
            exterior_field = BiotSavart(exterior_field)
        elif exterior_field is not None and not hasattr(exterior_field, "curl_b"):
            exterior_field = ExternalField(exterior_field)
        self.exterior_field, self.max_returns = exterior_field, int(max_returns)
        self.wall = getattr(wall, "evaluate_xyz", wall)
        # An orbit outside is back inside once it is reentry_depth [m] inside
        # the LCFS (default 1e-3 minor radii), so an orbit skimming the surface,
        # where the two fields differ slightly, does not bounce between them.
        if reentry_depth is None and self._vmec_default:
            reentry_depth = 1e-3 * float(jnp.abs(jnp.asarray(field.Aminor_p)))
        self.reentry_depth = reentry_depth
        # Diffrax's ceiling was effectively unbounded, so a trace that could not
        # finish ran until the process was killed rather than returning.
        self.max_steps = max_steps
        self.progress = bool(progress)
        # With particle_batch_size, particles are traced in batches of that
        # size and progress counts completed particles instead of Diffrax
        # steps. Under a JAX transformation the batches are bypassed.
        if particle_batch_size is not None and (
                isinstance(particle_batch_size, bool) or not isinstance(particle_batch_size, (int, np.integer))
                or particle_batch_size <= 0):
            raise ValueError("particle_batch_size must be a positive integer or None")
        self.particle_batch_size = None if particle_batch_size is None else int(particle_batch_size)
        self.progress_meter = (TqdmProgressMeter() if self.progress and self.particle_batch_size is None
                               else NoProgressMeter())
        # Diffrax solver to use for the adaptive integrators. If left as None,
        # each integrator falls back to its previous default (Dopri8), so
        # existing call sites are unaffected. Selecting the solver here (rather
        # than hard-coding it) lets the integrator-comparison examples sweep
        # several solvers. The fallback is a plain Python branch on this
        # attribute, so it is resolved at trace time and does not affect
        # differentiability of the traced trajectories.
        self.solver = solver
        if condition is None:
            self.condition = lambda t, y, args, **kwargs: False
            if isinstance(field, Vmec):
                if model in ('FieldLine', 'FieldLineAdaptative', 'FieldLineArclength', 'FieldLineToroidal'):
                    def condition_Vmec(t, y, args, **kwargs):
                        s, _, _ = y
                        return s-1	 
                    self.condition = condition_Vmec
            elif (isinstance(field, Coils) or isinstance(self.field, BiotSavart)) and isinstance(boundary,SurfaceClassifier):
                if model in _GUIDING_CENTER_COLLISION_MODELS:
                    def condition_BioSavart(t, y, args, **kwargs):
                        xx, yy, zz, _,_ = y
                        return boundary.evaluate_xyz(jnp.array([xx,yy,zz]))#<0.                      
                else:
                    def condition_BioSavart(t, y, args, **kwargs):                      
                        xx, yy, zz, _ = y
                        return boundary.evaluate_xyz(jnp.array([xx,yy,zz]))#<0.        
                self.condition = condition_BioSavart                
        elif self._axis_regular:
            self.condition = tree_util.tree_map(
                lambda c: lambda t, y, args, **kwargs: c(t, _from_axis_regular(y), args, **kwargs),
                condition)
        else:
            self.condition = condition
        if model == 'GuidingCenter' or model=='GuidingCenterAdaptative':
            self.ODE_term = ODETerm(self._vector_field(GuidingCenter))
            self.args = (self.field, self.particles,self.electric_field)
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz, self.particles.initial_vparallel[:, None]], axis=1)
        elif model == 'GuidingCenterCollisions':
            # Brownian motion
            #t0=0.0
            #t1=self.maxtime
            #tol=self.maxtime / self.timesteps*0.5
            #print('tol: ', tol)
            #bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,), key=jax.random.key(0), levy_area=diffrax.SpaceTimeTimeLevyArea)            
            #self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDrift),ControlTerm(GuidingCenterCollisionsDiffusion, bm))
            self.args = (self.field, self.particles,self.electric_field,self.species,self.tag_gc)
            total_speed_temp=self.particles.total_speed*jnp.ones(self.particles.nparticles)
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz,total_speed_temp[:, None], self.particles.initial_vparallel_over_v[:, None]], axis=1)
        elif model == 'GuidingCenterCollisionsMuIto' or model == 'GuidingCenterCollisionsMuFixed' or model == 'GuidingCenterCollisionsMuAdaptative':
            # Brownian motion
            #t0=0.0
            #t1=self.maxtime
            #tol=self.maxtime / self.timesteps*0.5   
            #print('tol: ', tol)
            #bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,), key=jax.random.key(0), levy_area=diffrax.SpaceTimeTimeLevyArea)
            #self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMu),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))
            self.args = (self.field, self.particles,self.electric_field,self.species,self.tag_gc)
            #x,y,z=self.particles.initial_xyz[]
            B_particle=jax.vmap(self.field.AbsB,in_axes=0)(self.particles.initial_xyz)
            mu=self.particles.initial_vperpendicular**2*self.particles.mass*0.5/B_particle/(SPEED_OF_LIGHT**2*self.particles.mass)          
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz,self.particles.initial_vparallel[:, None]/SPEED_OF_LIGHT,mu[:, None]],axis=1)        
        elif model == 'FullOrbit' or model == 'FullOrbit_Boris' or model == 'FullOrbitAdaptative':
            self.ODE_term = ODETerm(Lorentz)
            self.args = (self.field, self.particles)
            if self.particles.initial_xyz_fullorbit is None:
                raise ValueError("Initial full orbit positions require field input to Particles")
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz_fullorbit, self.particles.initial_vxvyvz], axis=1)
            if field is None:
                raise ValueError("Field parameter is required for FullOrbit model")
        elif model == 'FullOrbitCollisions':
            self.args = (self.field, self.particles, self.species)
            if self.particles.initial_xyz_fullorbit is None:
                raise ValueError("Initial full orbit positions require field input to Particles")
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz_fullorbit, self.particles.initial_vxvyvz], axis=1)
            if field is None:
                raise ValueError("Field parameter is required for FullOrbit model")
        elif model in ('FieldLine', 'FieldLineAdaptative', 'FieldLineArclength', 'FieldLineToroidal'):
            field_line_rhs = {
                'FieldLineArclength': FieldLineArclength,
                'FieldLineToroidal': FieldLineToroidal,
            }.get(model, FieldLine)
            self.ODE_term = ODETerm(field_line_rhs)
            self.args = self.field
        
        if self.times_to_trace is None:
            self.times = jnp.linspace(0, self.maxtime, 100,endpoint=True)
        else:
            self.times = jnp.linspace(0, self.maxtime, self.times_to_trace,endpoint=True)

            
        if self._vmec_default:
            return self._set_vmec_outputs(self.trace())
        trace_result = self.trace()
        if self.stopping_criteria is not None:
            trajectories, self.event_mask = trace_result
            event_leaves = tree_util.tree_leaves(self.event_mask)
            self.boundary_hits = jnp.any(jnp.stack(event_leaves), axis=0)
            self.axis_hits = jnp.zeros_like(self.boundary_hits)
            self._trajectories = _fill_stopped_trajectories(
                trajectories, self.stopping_criteria, self.args
            )
        else:
            self._trajectories = trace_result
            self.event_mask = None
            self.axis_hits = jnp.zeros(len(self._trajectories), dtype=bool)
            self.boundary_hits = jnp.zeros(len(self._trajectories), dtype=bool)
        self.total_particles_unresolved = jnp.sum(self.axis_hits)
        
        trajectory_points = self.trajectories[:, :, :3]
        if hasattr(self.field, "toroidal_angle_batch"):
            self.toroidal_angles = self.field.toroidal_angle_batch(
                trajectory_points.reshape((-1, 3))).reshape(trajectory_points.shape[:2])
        else:
            self.toroidal_angles = None
        if hasattr(self.field, "to_xyz_batch"):
            flat_points = trajectory_points.reshape((-1, 3))
            self.trajectories_xyz = self.field.to_xyz_batch(flat_points).reshape(
                trajectory_points.shape)
        else:
            self.trajectories_xyz = vmap(
                lambda xyz: vmap(lambda point: self.field.to_xyz(point))(xyz)
            )(trajectory_points)
        
        if isinstance(field, Vmec):
            if self.model in _GUIDING_CENTER_COLLISION_MODELS:
                self.loss_fractions, self.total_particles_lost, self.lost_times,self.lost_energies,self.lost_positions = self.loss_fraction_collisions()                    
            else:                
                self.loss_fractions, self.total_particles_lost, self.lost_times = self.loss_fraction()
        elif (isinstance(field, Coils) or isinstance(self.field, BiotSavart)) and isinstance(boundary,SurfaceClassifier):
            if self.model in _GUIDING_CENTER_COLLISION_MODELS:
                self.loss_fractions, self.total_particles_lost, self.lost_times,self.lost_energies,self.lost_positions = self.loss_fraction_BioSavart_collisions(boundary)                    
            else:                
                self.loss_fractions, self.total_particles_lost, self.lost_times = self.loss_fraction_BioSavart(boundary)

    def trace(self):
        @jit
        def compute_trajectory(initial_condition, particle_key) -> jnp.ndarray:
            if self._vmec_default:
                return self._vmec_orbit(initial_condition, particle_key)
            if self._axis_regular:
                initial_condition = _to_axis_regular(initial_condition)
            if self.model == 'FullOrbit_Boris':
                # Integrate the whole [0, maxtime] span: an inner scan of
                # Boris pushes between consecutive save times, with dt
                # adjusted (<= timestep) so the saves land on self.times.
                n_saves = len(self.times) - 1
                per_save = max(1, int(np.ceil(float(self.maxtime) / (n_saves * float(self.timestep)) - 1e-9)))
                dt = self.maxtime / (n_saves * per_save)
                charge_over_mass = self.particles.charge / self.particles.mass
                criteria = self.stopping_criteria

                def push(state, _):
                    x = state[:3]
                    v = state[3:]
                    t = charge_over_mass * self.field.B_contravariant(x) * 0.5 * dt
                    s = 2. * t / (1. + jnp.dot(t, t))
                    vprime = v + jnp.cross(v, t)
                    v = v + jnp.cross(vprime, s)
                    x = x + v * dt
                    return jnp.concatenate((x, v)), None

                def save_interval(carry, _):
                    state, alive, hits = carry
                    advanced, _ = lax.scan(push, state, None, length=per_save)
                    if criteria is None:
                        return (advanced, alive, hits), advanced
                    # A particle leaving any level set (value <= 0) is held at
                    # its last saved point inside, as the adaptive paths do.
                    outside = jnp.stack([c(0.0, advanced, self.args) <= 0.0 for c in criteria])
                    outside = outside | ~jnp.isfinite(advanced).all()
                    stopped = alive & jnp.any(outside)
                    hits = hits | (alive & outside)
                    state = jnp.where(alive & ~stopped, advanced, state)
                    return (state, alive & ~stopped, hits), state

                n_criteria = 0 if criteria is None else len(criteria)
                carry = (initial_condition, jnp.asarray(True), jnp.zeros((n_criteria,), bool))
                (_, _, hits), trajectory = lax.scan(save_interval, carry, None, length=n_saves)
                trajectory = jnp.vstack([initial_condition, trajectory])
                if criteria is not None:
                    event_mask = hits[0] if n_criteria == 1 else tuple(hits[i] for i in range(n_criteria))
                    return trajectory, event_mask
                return trajectory
            solution = self._solve(*self._terms(particle_key), 0.0, initial_condition, self.args, Event(self.condition),
                                   throw=self.model in ('GuidingCenter', 'FullOrbit', 'FieldLine'))
            trajectory = solution.ys[0]
            if self._axis_regular:
                trajectory = vmap(_from_axis_regular)(trajectory)
            if self.stopping_criteria is not None:
                return trajectory, solution.event_mask
            return trajectory

        y0, keys = self.initial_conditions, self.particles.random_keys if self.particles else None
        n = len(y0)
        traced = any(isinstance(x, jax.core.Tracer) for x in tree_util.tree_leaves((y0, keys)))
        batch = n if self.particle_batch_size is None or traced else min(self.particle_batch_size, n)
        # Sharding pays off for plain traces. A differentiated trace stays on one device: XLA 0.6 aborts compiling
        # a gradient through a while loop sharded over several CPU devices.
        count = 1 if traced else min(len(self.devices), batch)
        count = -(-batch // -(-batch // count))  # as many devices as the per-device share needs: 37 on 36 -> 19 x 2
        batch = -(-batch // count) * count  # padded below so every device traces equally many particles
        if count > 1:
            place = NamedSharding(Mesh(np.asarray(self.devices[:count], dtype=object), ("dev",)), PartitionSpec("dev"))
            solve = jit(vmap(compute_trajectory), in_shardings=place, out_shardings=place)
        else:
            place, solve = self.devices[0], jit(vmap(compute_trajectory))

        def run(y0, keys):
            with jax.default_device(self.devices[0]):
                return solve(_place_on_devices(y0, place), None if keys is None else _place_on_devices(keys, place))

        if batch == n:
            return run(y0, keys)
        # Every batch has `batch` particles, the last one padded by repeating
        # its final particle, so the solve compiles once. Batches are gathered
        # on the host, so the result combines with arrays on any device.
        from tqdm.auto import tqdm
        results = []
        with tqdm(total=n, desc="Tracing particles", unit="particle", disable=not self.progress) as bar:
            for start in range(0, n, batch):
                index = np.minimum(np.arange(start, start + batch), n - 1)
                done = min(batch, n - start)
                result = run(y0[index], None if keys is None else keys[index])
                results.append(tree_util.tree_map(lambda x: np.asarray(x)[:done], result))
                bar.update(done)
        return tree_util.tree_map(lambda *x: jnp.asarray(np.concatenate(x)), *results)

    def _terms(self, key, flag=False):
        """Diffrax terms, solver and step-size controller of self.model (with _with_failure_flag if ``flag``)."""
        flagged = (lambda f, drift=True: _with_failure_flag(f, drift)) if flag else (lambda f, drift=True: f)
        dt0, model = self.timestep, self.model
        if model in _SDE_MODELS:
            drift, diffusion, solver, dim = _SDE_MODELS[model]
            # With flag (segments), a vmapped solve of an orbit that starts at maxtime evaluates the path
            # one step past it.
            bm = diffrax.VirtualBrownianTree(0.0, self.maxtime + flag * dt0, tol=dt0 * 0.5, shape=(dim,), key=key,
                                             levy_area=diffrax.SpaceTimeTimeLevyArea)
            terms = MultiTerm(ODETerm(flagged(self._vector_field(drift))),
                              ControlTerm(flagged(self._vector_field(diffusion), drift=False), bm))
            solver = solver()
        else:
            terms = ODETerm(flagged(self.ODE_term.vector_field))
            solver = self.solver if self.solver is not None else diffrax.Dopri8()
        controller = diffrax.ConstantStepSize()
        if model == 'GuidingCenterCollisionsMuAdaptative':
            controller = ClipStepSizeController(
                controller=PIDController(pcoeff=0.1, icoeff=0.3, dcoeff=0.0, rtol=self.rtol, atol=self.atol,
                                         dtmin=dt0, dtmax=1.e-4, force_dtmin=True),
                step_ts=self.times, store_rejected_steps=self.rejected_steps)
        elif model in _ADAPTIVE_MODELS:
            controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol,
                                       dtmin=dt0 if model == 'FullOrbitCollisions' else None)
        return terms, solver, controller

    def _solve(self, terms, solver, controller, t0, y0, args, event, throw=False):
        """diffeqsolve from t0 to maxtime, saving at self.times (clipped to t0) and at the end."""
        import warnings
        warnings.simplefilter("ignore", category=FutureWarning)  # see https://github.com/patrick-kidger/diffrax/issues/445
        return diffeqsolve(terms, solver, t0=t0, t1=self.maxtime, dt0=self.timestep, y0=y0, args=args,
                           saveat=SaveAt(subs=[diffrax.SubSaveAt(ts=jnp.clip(self.times, t0, self.maxtime)),
                                               diffrax.SubSaveAt(t1=True)]),
                           stepsize_controller=controller, event=event, throw=throw, max_steps=self.max_steps,
                           progress_meter=self.progress_meter)

    # -- VMEC guiding centers: exact LCFS crossing, failures, exterior continuation --

    def _vmec_orbit(self, y0, key):
        """Trace one VMEC guiding center segment by segment: inside, outside, and back inside.

        Inside the LCFS the model runs in the axis-regular chart until maxtime,
        the root-found LCFS crossing, or a non-finite state. Without
        exterior_field a crossing ends the orbit. With it, the orbit continues
        as a collisionless guiding center (x, y, z, v_par, mu) with its energy
        and magnetic moment, until it strikes the wall, reaches maxtime or
        re-enters the LCFS; it then continues inside, up to max_returns times.
        Solves of inactive orbits start at maxtime and take no steps.
        """
        T, times, ext = self.maxtime, self.times, self.exterior_field
        inside = (*self._terms(key, flag=True), self.args,
                  Event((lambda t, y, args, **kwargs: 1.0 - y[0]**2 - y[1]**2, _failed_event),
                        root_finder=_EVENT_ROOT_FINDER))
        if ext is not None:
            wall = self.wall if self.wall is not None else (lambda x: jnp.ones(()))
            outside = (ODETerm(_with_failure_flag(GuidingCenterMu)), self.solver if self.solver is not None else
                       diffrax.Dopri8(), PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol),
                       (ext, self.particles, Electric_field_zero()),
                       Event((lambda t, y, args, **kwargs: wall(y[:3]),
                              lambda t, y, args, **kwargs: self.field.boundary_distance(y[:3]) - self.reentry_depth,
                              _failed_event), root_finder=_EVENT_ROOT_FINDER))

        def segment(o, run, spec, y, code, to_flux=lambda y: y):
            """Solve one segment of the orbits in ``run`` and record its saves."""
            solution = self._solve(*spec[:3], jnp.where(run, o["t"], T), jnp.append(y, 0.0), *spec[3:])
            saves, end = vmap(to_flux)(solution.ys[0][:, :-1]), to_flux(solution.ys[1][-1, :-1])
            failed = run & (solution.event_mask[-1] | ~((solution.result == diffrax.RESULTS.successful)
                            | (solution.result == diffrax.RESULTS.event_occurred)
                            | (solution.result == diffrax.RESULTS.nonlinear_max_steps_reached)))
            t_end = jnp.where(run, solution.ts[1][-1], o["t"])
            window = run & (times >= o["t"]) & (times <= t_end) & jnp.isfinite(saves).all(1)
            xyz = vmap(self.field.to_xyz)(saves[:, :3]) if code == 0 else saves[:, :3]
            o = dict(o, xyz=jnp.where(window[:, None], xyz, o["xyz"]), region=jnp.where(window, code, o["region"]))
            if code == 0:
                o["traj"] = jnp.where(window[:, None], saves, o["traj"])
            return o, t_end, end, [run & ~failed & hit for hit in solution.event_mask[:-1]], failed

        def step(o):
            # Inside; an orbit that starts outside the LCFS crosses at once.
            run = o["active"] & o["inside"]
            out = o["y"][0] >= 1.0
            o, t_end, end, (crossed,), failed = segment(o, run & ~out, inside, _to_axis_regular(o["y"]), 0,
                                                       _from_axis_regular)
            crossed, end = crossed | (run & out), jnp.where(out, o["y"], end)
            x, E = self._vmec_to_exterior(end)
            first = crossed & jnp.isinf(o["lcfs_time"])
            o.update(t=t_end, lcfs_time=jnp.where(first, t_end, o["lcfs_time"]),
                     lcfs_state=jnp.where(first, end, o["lcfs_state"]), lcfs_energy=jnp.where(first, E, o["lcfs_energy"]),
                     status=jnp.select([failed, run & ~crossed, crossed & (ext is None)], [4, 0, 1], o["status"]),
                     active=o["active"] & ~(failed | (run & ~crossed) | (crossed & (ext is None))),
                     inside=o["inside"] & ~crossed, x=jnp.where(crossed, x, o["x"]))
            if ext is None:
                return o
            # Outside, to the wall, maxtime, or back inside.
            run = o["active"] & ~o["inside"]
            o, t_end, end, (struck, returned), failed = segment(o, run, outside, o["x"], 1)
            returned &= ~struck
            back = returned & (o["returns"] < self.max_returns)
            o.update(t=t_end, returns=o["returns"] + back, inside=o["inside"] | back, active=o["active"] & ~(run & ~back),
                     status=jnp.select([failed, struck, returned & ~back, run & ~back], [4, 3, 5, 2], o["status"]),
                     x=jnp.where(run, end, o["x"]), y=jnp.where(back, self._vmec_from_exterior(end), o["y"]))
            return o

        nt, d = len(times), y0.shape[0]
        o = dict(t=jnp.zeros(()), y=y0, x=jnp.zeros(5), inside=jnp.array(True), active=jnp.array(True),
                 status=jnp.zeros((), jnp.int8), returns=jnp.zeros((), int), lcfs_time=jnp.full((), jnp.inf),
                 lcfs_state=jnp.full(d, jnp.nan), lcfs_energy=jnp.full((), jnp.nan), traj=jnp.full((nt, d), jnp.inf),
                 xyz=jnp.full((nt, 3), jnp.nan), region=jnp.full(nt, -1, dtype=jnp.int8))
        o = lax.while_loop(lambda o: o["active"], step, o)
        if ext is not None:  # energy at the end outside, the strike for a wall hit
            o["x_energy"] = 0.5 * self.particles.mass * o["x"][3]**2 + o["x"][4] * ext.AbsB(o["x"][:3])
        return o

    def _vmec_to_exterior(self, y):
        """Model state on the LCFS -> (x, y, z, v_par, mu) and energy, keeping E and mu."""
        point, m, c = y[:3], self.particles.mass, SPEED_OF_LIGHT
        B = self.field.AbsB(point)
        if self.model in ('GuidingCenter', 'GuidingCenterAdaptative'):
            vpar, E = y[3], self.particles.energy
            mu = (E - 0.5 * m * vpar**2) / B
        elif self.model == 'GuidingCenterCollisions':
            vpar, E = y[3] * y[4], 0.5 * m * y[3]**2
            mu = 0.5 * m * y[3]**2 * (1 - y[4]**2) / B
        else:
            vpar, mu = y[3] * c, y[4] * c**2 * m
            E = 0.5 * m * vpar**2 + mu * B
        x = self.field.to_xyz(point)
        if self.exterior_field is not None:
            vpar, mu = _keep_energy(vpar, mu, E, self.exterior_field.AbsB(x), m)
        return jnp.concatenate([x, jnp.array([vpar, mu])]), E

    def _vmec_from_exterior(self, y):
        """(x, y, z, v_par, mu) just inside the LCFS -> model state in flux coordinates."""
        m, c = self.particles.mass, SPEED_OF_LIGHT
        E = 0.5 * m * y[3]**2 + y[4] * self.exterior_field.AbsB(y[:3])
        point = self.field.flux_coordinates(y[:3])[0]
        vpar, mu = _keep_energy(y[3], y[4], E, self.field.AbsB(point), m)
        if self.model in ('GuidingCenter', 'GuidingCenterAdaptative'):
            return jnp.append(point, vpar)
        if self.model == 'GuidingCenterCollisions':
            v = jnp.sqrt(2 * E / m)
            return jnp.concatenate([point, jnp.array([v, vpar / v])])
        return jnp.concatenate([point, jnp.array([vpar / c, mu / (c**2 * m)])])

    def _set_vmec_outputs(self, o):
        """Per-orbit results of _vmec_orbit, and losses at the exact wall (with an exterior field) or LCFS times."""
        o = {k: np.asarray(v) for k, v in o.items()}
        self.status, self.returns, self.region = o["status"], o["returns"], o["region"]
        self.failed, self.wall_hits = self.status == 4, self.status == 3
        self.failure_times = np.where(self.failed, o["t"], np.inf)
        self.lcfs_times, self.lcfs_states, self.lcfs_energies = o["lcfs_time"], o["lcfs_state"], o["lcfs_energy"]
        self.lcfs_positions = np.asarray(vmap(self.field.to_xyz)(jnp.asarray(o["lcfs_state"][:, :3])))
        self.wall_times = np.where(self.wall_hits, o["t"], np.inf)
        self.wall_positions = np.where(self.wall_hits[:, None], o["x"][:, :3], np.nan)
        self.wall_energies = np.where(self.wall_hits, o.get("x_energy", np.nan), np.nan)
        self._trajectories, self.trajectories_xyz = jnp.asarray(o["traj"]), jnp.asarray(o["xyz"])
        self.boundary_hits = self.event_mask = jnp.asarray(np.isfinite(self.lcfs_times))
        self.axis_hits = jnp.zeros(len(o["t"]), dtype=bool)
        self.total_particles_unresolved, self.toroidal_angles = jnp.sum(self.axis_hits), None
        wall = self.exterior_field is not None
        lost = self.wall_times if wall else self.lcfs_times
        self.loss_fractions = jnp.asarray(np.mean(lost[:, None] <= np.asarray(self.times)[None, :], axis=0))
        self.total_particles_lost = jnp.sum(jnp.isfinite(lost))
        self.lost_times = jnp.asarray(np.where(np.isfinite(lost), lost, -1))
        if self.model in _GUIDING_CENTER_COLLISION_MODELS:
            hit = np.isfinite(lost)[:, None]
            self.lost_energies = jnp.asarray(np.where(hit[:, 0], self.wall_energies if wall else self.lcfs_energies, 0.0))
            self.lost_positions = jnp.asarray(np.where(hit, self.wall_positions if wall else self.lcfs_states[:, :3], 0.0))

    def _vector_field(self, vector_field):
        return _axis_regular(vector_field) if self._axis_regular else vector_field

    @property
    def trajectories(self):
        return self._trajectories
    
    @trajectories.setter
    def trajectories(self, value):
        self._trajectories = value
    
    def energy(self):
        assert 'GuidingCenter' in self.model or 'FullOrbit' in self.model or 'FullOrbit_Boris' in self.model, "Energy calculation is only available for GuidingCenter and FullOrbit models"
        mass = self.particles.mass

        if self.model == 'GuidingCenter' or self.model == 'GuidingCenterAdaptative':
            initial_xyz = self.initial_conditions[:, :3]
            initial_vparallel = self.initial_conditions[:, 3]
            initial_B = vmap(self.field.AbsB)(initial_xyz)
            mu_array = (self.particles.energy - 0.5 * mass * jnp.square(initial_vparallel)) / initial_B
            def compute_energy(trajectory, mu):
                xyz = trajectory[:, :3]
                vpar = trajectory[:, 3]
                AbsB = vmap(self.field.AbsB)(xyz)                
                return 0.5 * mass * jnp.square(vpar) + mu * AbsB
            energy = vmap(compute_energy)(self.trajectories, mu_array)
        elif self.model == 'GuidingCenterCollisionsMuIto' or self.model == 'GuidingCenterCollisionsMuFixed' or self.model == 'GuidingCenterCollisionsMuAdaptative':
            def compute_energy(trajectory):
                xyz = trajectory[:, :3]                
                vpar = trajectory[:, 3]*SPEED_OF_LIGHT
                mu = trajectory[:, 4]*self.particles.mass*SPEED_OF_LIGHT**2
                AbsB = vmap(self.field.AbsB)(xyz)
                return self.particles.mass * vpar**2 / 2 + mu*AbsB
            energy = vmap(compute_energy)(self.trajectories)            
        elif self.model == 'GuidingCenterCollisions':
            def compute_energy(trajectory):
                return 0.5 * mass * trajectory[:, 3]**2
            energy = vmap(compute_energy)(self.trajectories)

        else:  # the full-orbit models: FullOrbit, FullOrbit_Boris, FullOrbitAdaptative, FullOrbitCollisions
            def compute_energy(trajectory):
                vxvyvz = trajectory[:, 3:]
                v_squared = jnp.sum(jnp.square(vxvyvz), axis=1)
                return 0.5 * mass * v_squared
            energy = vmap(compute_energy)(self.trajectories)

        return energy
    
    
    def v_perp(self):
        assert 'GuidingCenter' in self.model or 'FullOrbit' in self.model or 'FullOrbit_Boris' in self.model, "Energy calculation is only available for GuidingCenter and FullOrbit models"
        mass = self.particles.mass

        if self.model == 'GuidingCenter' or self.model == 'GuidingCenterAdaptative':
            initial_xyz = self.initial_conditions[:, :3]
            initial_vparallel = self.initial_conditions[:, 3]
            initial_B = vmap(self.field.AbsB)(initial_xyz)
            mu_array = (self.particles.energy - 0.5 * mass * jnp.square(initial_vparallel)) / initial_B
            def compute_vperp(trajectory, mu):
                xyz = trajectory[:, :3]
                AbsB = vmap(self.field.AbsB)(xyz)                
                return jnp.sqrt(mu * AbsB/mass*2.)
            v_perp = vmap(compute_vperp)(self.trajectories, mu_array)

        elif  self.model == 'GuidingCenterCollisionsMuIto' or self.model == 'GuidingCenterCollisionsMuFixed' or self.model == 'GuidingCenterCollisionsMuAdaptative':
            def compute_vperp(trajectory):
                xyz = trajectory[:, :3]
                mu = trajectory[:, 4]*self.particles.mass*SPEED_OF_LIGHT**2
                AbsB = vmap(self.field.AbsB)(xyz)
                return jnp.sqrt(mu*AbsB/self.particles.mass*2.)
            v_perp = vmap(compute_vperp)(self.trajectories)           
        elif self.model == 'GuidingCenterCollisions':
            def compute_vperp(trajectory):
                return trajectory[:, 3]*jnp.sqrt(jnp.maximum(1-trajectory[:, 4]**2, 0.))
            v_perp = vmap(compute_vperp)(self.trajectories)

        else:  # the full-orbit models: FullOrbit, FullOrbit_Boris, FullOrbitAdaptative, FullOrbitCollisions
            def compute_vperp(trajectory):
                xyz = trajectory[:, :3]
                vxvyvz = trajectory[:, 3:]
                B = vmap(self.field.B)(xyz)
                vperp_squared = jnp.sum(jnp.square(vxvyvz), axis=1) - jnp.square(jnp.sum(vxvyvz * B, axis=1) / jnp.linalg.norm(B, axis=1))
                return jnp.sqrt(jnp.maximum(vperp_squared, 0.0))
            v_perp = vmap(compute_vperp)(self.trajectories)

        return v_perp

    def to_vtk(self, filename):
        try: import numpy as np
        except ImportError: raise ImportError("The 'numpy' library is required. Please install it using 'pip install numpy'.")
        try: from pyevtk.hl import polyLinesToVTK
        except ImportError: raise ImportError("The 'pyevtk' library is required. Please install it using 'pip install pyevtk'.")
        x = np.concatenate([xyz[:, 0] for xyz in self.trajectories_xyz])
        y = np.concatenate([xyz[:, 1] for xyz in self.trajectories_xyz])
        z = np.concatenate([xyz[:, 2] for xyz in self.trajectories_xyz])
        ppl = np.asarray([xyz.shape[0] for xyz in self.trajectories_xyz])
        data = np.array(jnp.concatenate([i*jnp.ones((self.trajectories[i].shape[0], )) for i in range(len(self.trajectories))]))
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})
    
    def plot(self, ax=None, show=True, axis_equal=True, n_trajectories_plot=5, **kwargs):
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        trajectories_xyz = jnp.array(self.trajectories_xyz)
        n_trajectories_plot = jnp.min(jnp.array([n_trajectories_plot, trajectories_xyz.shape[0]]))
        for i in random.choice(random.PRNGKey(0), trajectories_xyz.shape[0], (n_trajectories_plot,), replace=False):
            ax.plot(trajectories_xyz[i, :, 0], trajectories_xyz[i, :, 1], trajectories_xyz[i, :, 2], **kwargs)
        ax.grid(False)
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
            
            
    @partial(jit, static_argnums=(0,1))
    def loss_fraction_BioSavart(self, boundary):
        """Memory-efficient boundary loss fraction evaluation.
        
        Uses flattened single vmap instead of nested double vmap to reduce
        memory usage by ~80% while maintaining accuracy.
        
        Args:
            boundary: SurfaceClassifier for boundary evaluation
            
        Returns:
            loss_fractions: Cumulative loss fraction over time
            total_particles_lost: Total number of particles lost
            lost_times: Time of loss for each particle
        """
        trajectories_xyz = self.trajectories[:, :, :3]
        nparticles, ntimesteps = trajectories_xyz.shape[:2]
        
        # MEMORY OPTIMIZATION: Flatten to single vmap instead of nested double vmap
        # (nparticles, ntimesteps, 3) -> (nparticles*ntimesteps, 3)
        trajectories_flat = trajectories_xyz.reshape(-1, 3)
        
        # Single vmap: evaluates all points at once
        distances_flat = vmap(boundary.evaluate_xyz)(trajectories_flat)
        
        # Reshape back: (nparticles*ntimesteps,) -> (nparticles, ntimesteps)
        distances = distances_flat.reshape(nparticles, ntimesteps)
        
        # Lost mask: True where boundary distance < 0 (outside boundary)
        lost_mask = distances < 0
        
        # Find first crossing for each particle
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        
        # Compute cumulative loss
        safe_lost_indices = jnp.where(lost_indices != -1, lost_indices, len(self.times))
        loss_counts = jnp.bincount(safe_lost_indices, length=len(self.times) + 1)[:-1]
        loss_fractions = jnp.cumsum(loss_counts) / len(self.trajectories)
        total_particles_lost = loss_fractions[-1] * len(self.trajectories)
        
        return loss_fractions, total_particles_lost, lost_times

    def loss_fraction(self,r_max=1.0):
        """Cumulative loss fraction of a flux-coordinate trace.

        A particle is lost at the first saved time with ``s >= r_max``, or
        with a non-finite state, which is what the LCFS event leaves after it
        stops a trace. The default ``r_max`` is that LCFS.
        """
        trajectories_r = self.trajectories[:,:, 0]
        lost_mask = trajectories_r >= r_max
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        safe_lost_indices = jnp.where(lost_indices != -1, lost_indices, len(self.times))
        loss_counts = jnp.bincount(safe_lost_indices, length=len(self.times) + 1)[:-1]
        loss_fractions = jnp.cumsum(loss_counts) / len(self.trajectories)
        total_particles_lost = loss_fractions[-1] * len(self.trajectories)
        return loss_fractions, total_particles_lost, lost_times



    @partial(jit, static_argnums=(0,1))
    def loss_fraction_BioSavart_collisions(self, boundary):
        """Memory-efficient boundary loss fraction for collision models.
        
        Optimized version using flattened vmap.
        """
        trajectories_xyz = self.trajectories[:, :, :3]
        nparticles, ntimesteps = trajectories_xyz.shape[:2]
        
        # Flatten to single vmap for memory efficiency
        trajectories_flat = trajectories_xyz.reshape(-1, 3)
        distances_flat = vmap(boundary.evaluate_xyz)(trajectories_flat)
        distances = distances_flat.reshape(nparticles, ntimesteps)
        
        lost_mask = distances < 0
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        
        # OPTIMIZATION: Replace indexed vmap with vectorized masking (10-15x faster)
        has_lost = lost_indices != -1
        # Gather energy at loss time for particles that lost - use clip to keep indices valid
        safe_indices = jnp.clip(lost_indices, 0, ntimesteps - 1)
        particle_indices = jnp.arange(nparticles)
        lost_energies = jnp.where(has_lost, self.energy()[particle_indices, safe_indices], 0.)
        
        # Gather positions at loss time for particles that lost
        lost_positions = jnp.where(
            has_lost[:, None], 
            trajectories_xyz[particle_indices, safe_indices], 
            0.
        )                          
        safe_lost_indices = jnp.where(lost_indices != -1, lost_indices, len(self.times))
        loss_counts = jnp.bincount(safe_lost_indices, length=len(self.times) + 1)[:-1]
        loss_fractions = jnp.cumsum(loss_counts) / len(self.trajectories)
        total_particles_lost = loss_fractions[-1] * len(self.trajectories)
        return loss_fractions, total_particles_lost, lost_times,lost_energies,lost_positions

    @partial(jit, static_argnums=(0))
    def loss_fraction_collisions(self,r_max=1.0):
        """As :meth:`loss_fraction`, with the energy and position of each lost
        particle at its last finite saved state."""
        trajectories_rtz = self.trajectories[:,:, :3]
        lost_mask = trajectories_rtz[:,:,0] >= r_max
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        has_lost = lost_indices != -1
        finite = jnp.isfinite(self.trajectories).all(axis=2)
        last_finite = lax.cummax(jnp.where(finite, jnp.arange(len(self.times)), 0), axis=1)
        particle_indices = jnp.arange(self.particles.nparticles)
        safe_indices = last_finite[particle_indices, jnp.clip(lost_indices, 0, len(self.times) - 1)]
        lost_energies = jnp.where(has_lost, self.energy()[particle_indices, safe_indices], 0.)
        lost_positions = jnp.where(
            has_lost[:, None],
            trajectories_rtz[particle_indices, safe_indices],
            0.
        )
        safe_lost_indices = jnp.where(lost_indices != -1, lost_indices, len(self.times))
        loss_counts = jnp.bincount(safe_lost_indices, length=len(self.times) + 1)[:-1]
        loss_fractions = jnp.cumsum(loss_counts) / len(self.trajectories)
        total_particles_lost = loss_fractions[-1] * len(self.trajectories)
        return loss_fractions, total_particles_lost, lost_times,lost_energies,lost_positions


    
    def poincare_plot(self, shifts = [jnp.pi/2], orientation = 'toroidal', length = 1, ax=None, show=True, color=None, **kwargs):
        """
        Plot Poincare sections from Cartesian trajectories.
        Args:
            shifts (list, optional): Apply a linear shift to dependent data. Default is [pi/2].
            orientation (str, optional): 
                'toroidal' - find time values when toroidal angle = shift [0, 2pi].
                'z' - find time values where z coordinate = shift. Default is 'toroidal'.
            length (float, optional): A way to shorten data. 1 - plot full length, 0.1 - plot 1/10 of data length. Default is 1.
            ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib axis to plot on. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            color: ``"time"``, one Matplotlib color, or one color per trajectory.
            **kwargs: Additional keyword arguments for plotting.
        Toroidal crossings are found from the unwrapped Cartesian azimuth, so
        the branch cut at ``phi=0`` does not create or discard intersections.
        """
        kwargs.setdefault('s', 0.5)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        shifts = np.asarray(shifts, dtype=float)
        trajectories = np.asarray(self.trajectories_xyz)
        times = np.asarray(self.times)
        plotting_data = []
        for shift in shifts:
            sections = []
            native_angles = getattr(self, "toroidal_angles", None)
            for trace_index, trace in enumerate(trajectories):
                x, y, z = trace[:, :3].T
                if orientation == 'toroidal':
                    phase = (np.asarray(native_angles[trace_index]) if native_angles is not None
                             else np.unwrap(np.arctan2(y, x)))
                    delta = np.diff(phase)
                    turns = np.floor((phase - shift) / (2.0 * np.pi))
                    indices = np.flatnonzero(np.diff(turns) != 0)
                    levels = shift + 2.0 * np.pi * np.where(
                        delta[indices] > 0.0, turns[indices] + 1.0, turns[indices])
                    fraction = (levels - phase[indices]) / delta[indices]
                    first, second = np.hypot(x, y), z
                elif orientation == 'z':
                    values = z - shift
                    indices = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
                    denominator = values[indices] - values[indices + 1]
                    valid = denominator != 0.0; indices = indices[valid]
                    fraction = values[indices] / denominator[valid]
                    first, second = x, y
                else:
                    raise ValueError("orientation must be 'toroidal' or 'z'")
                fraction = np.clip(fraction, 0.0, 1.0)
                section_time = times[indices] + fraction * np.diff(times)[indices]
                first_section = first[indices] + fraction * np.diff(first)[indices]
                second_section = second[indices] + fraction * np.diff(second)[indices]
                count = int(len(indices) * length)
                sections.append((first_section[:count], second_section[:count], section_time[:count]))

            colors = plt.cm.ocean(np.linspace(0, 0.8, len(sections)))
            color_is_time = isinstance(color, str) and color == "time"
            per_trajectory_color = (color is not None and not color_is_time
                                    and not is_color_like(color) and len(color) == len(sections))
            for i, (X_plot, Y_plot, T_plot) in enumerate(sections):
                plotting_data.append((X_plot, Y_plot, T_plot))
                if color_is_time:
                    ax.scatter(X_plot, Y_plot, c=T_plot, **kwargs)
                else:
                    if color is None: c=[colors[i]]
                    elif per_trajectory_color: c=color[i]
                    else: c=color
                    ax.scatter(X_plot, Y_plot, c=c, **kwargs)
                    
        if orientation == 'toroidal':
            plt.xlabel('R',fontsize = 18)
            plt.ylabel('Z',fontsize = 18)
            # plt.title(r'$\phi$ = {:.2f} $\pi$'.format(shift/jnp.pi),fontsize = 20)
        elif orientation == 'z':
            plt.xlabel('X',fontsize = 18)
            plt.xlabel('Y',fontsize = 18)
            # plt.title('Z = {:.2f}'.format(shift),fontsize = 20)
        plt.axis('equal')
        plt.grid()
        plt.tight_layout()
        if show:
            plt.show()
        
        return plotting_data
    
    def _tree_flatten(self):
        children = (self.trajectories, self.initial_conditions, self.times)  # arrays / dynamic values
        aux_data = {'field': self.field, 'electric_field': self.electric_field, 'model': self.model, 'maxtime': self.maxtime, 'timestep': self.timestep,
                    'rtol': self.rtol, 'atol': self.atol, 'particles': self.particles, 'condition': self.condition, 'tag_gc': self.tag_gc,
                    'solver': self.solver, 'stopping_criteria': self.stopping_criteria,
                    'progress': self.progress, 'devices': self.devices, 'exterior_field': self.exterior_field,
                    'wall': self.wall, 'max_returns': self.max_returns, 'reentry_depth': self.reentry_depth,
                    'particle_batch_size': self.particle_batch_size}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Tracing,
                               Tracing._tree_flatten,
                               Tracing._tree_unflatten)


def trace_field_lines(
    field,
    initial_conditions,
    *,
    toroidal_turns=None,
    length=None,
    samples=1000,
    tolerance=1.0e-7,
    stopping_criteria=None,
    progress=True,
    label="field lines",
    devices=None,
):
    """Trace field lines by toroidal angle or physical arclength.

    Specify exactly one of ``toroidal_turns`` or ``length``. Toroidal tracing
    is intended for fields represented in flux coordinates; Cartesian coil
    fields use arclength, so multiplying the magnetic field does not change
    the traced distance. ``samples`` includes both endpoints. The returned
    :class:`Tracing` object provides trajectories, event flags, plotting, and
    Poincare sections.

    Args:
        field: ESSOS-compatible magnetic field.
        initial_conditions: One seed per row, in the field's coordinates.
        toroidal_turns: Number of full toroidal turns to follow.
        length: Physical arclength to follow for a Cartesian field.
        samples: Number of saved points along each line.
        tolerance: Relative and absolute adaptive-integration tolerance.
        stopping_criteria: Optional event callable or sequence of callables.
        progress: Show Diffrax's terminal progress bar.
        label: Text printed before compilation and after completion; set to
            ``None`` to suppress these two messages.
        devices: Optional explicit sequence of JAX devices.
    """
    if (toroidal_turns is None) == (length is None):
        raise ValueError("specify exactly one of toroidal_turns or length")
    if int(samples) < 2:
        raise ValueError("samples must be at least 2")
    if toroidal_turns is not None and float(toroidal_turns) <= 0.0:
        raise ValueError("toroidal_turns must be positive")
    if length is not None and float(length) <= 0.0:
        raise ValueError("length must be positive")

    extent = (2.0 * jnp.pi * float(toroidal_turns)
              if toroidal_turns is not None else float(length))
    model = "FieldLineToroidal" if toroidal_turns is not None else "FieldLineArclength"
    if label is not None:
        print(f"Tracing {label} (the first call compiles ESSOS)...", flush=True)
    started = perf_counter()
    result = Tracing(
        field=field,
        model=model,
        initial_conditions=initial_conditions,
        maxtime=extent,
        timestep=extent / (int(samples) - 1),
        times_to_trace=int(samples),
        atol=float(tolerance),
        rtol=float(tolerance),
        stopping_criteria=stopping_criteria,
        progress=bool(progress),
        devices=devices,
    )
    jax.block_until_ready(result.trajectories_xyz)
    if label is not None:
        message = f"{label} ready in {perf_counter() - started:.1f} s"
        if stopping_criteria is not None:
            hits = int(jnp.sum(result.boundary_hits))
            message += f"; {hits}/{len(initial_conditions)} lines reached a stopping event"
        print(message, flush=True)
    return result


def connection_length(field, initial_conditions, wall, *, max_length,
                      tolerance=1.0e-8, max_steps=100000):
    """Connection length and wall strike points of field lines.

    Each seed is followed along ``+B`` and ``-B`` by physical arclength until
    it crosses the wall or reaches ``max_length``. The crossing is located by
    Diffrax event root finding, so the strike point is exact up to
    ``tolerance`` rather than limited by a sampling interval. The result is
    differentiable with respect to the seeds and field parameters.

    Args:
        field: ESSOS-compatible Cartesian magnetic field.
        initial_conditions: Seeds of shape ``(n, 3)``. Seeds on or outside the
            wall return zero length and ``hit`` true.
        wall: Object with ``evaluate_xyz(xyz)`` (e.g. :class:`SurfaceClassifier`)
            or a callable ``wall(xyz)``, positive inside the wall and zero on it.
        max_length: Cap on the length followed in each direction.
        tolerance: Relative and absolute integration and root-finding tolerance.
        max_steps: Maximum adaptive steps per direction; a line that exhausts
            them returns ``nan`` length and ``hit`` false.

    Returns:
        Dict with ``lengths`` ``(n, 2)`` (forward, backward), ``connection_length``
        ``(n,)`` (their sum), ``strike_points`` ``(n, 2, 3)`` (end points; the
        wall hit when ``hit`` is true) and ``hit`` ``(n, 2)`` booleans.
    """
    if float(max_length) <= 0.0:
        raise ValueError("max_length must be positive")
    distance = wall.evaluate_xyz if hasattr(wall, "evaluate_xyz") else wall
    controller = PIDController(rtol=tolerance, atol=tolerance)
    event = Event(lambda t, y, args, **kwargs: distance(y),
                  root_finder=optx.Newton(rtol=tolerance, atol=tolerance))

    def vector_field(t, y, sign):
        B = field.B_contravariant(y)
        return sign * B / jnp.maximum(jnp.linalg.norm(B), jnp.finfo(B.dtype).tiny)

    def trace_one(seed, sign):
        solution = diffeqsolve(
            ODETerm(vector_field), diffrax.Dopri8(), t0=0.0, t1=float(max_length),
            dt0=float(max_length) / 1000, y0=seed, args=sign,
            saveat=SaveAt(t1=True), stepsize_controller=controller,
            event=event, max_steps=int(max_steps), throw=False)
        hit = solution.event_mask
        failed = (solution.result != diffrax.RESULTS.successful) & ~hit
        outside = distance(seed) <= 0.0
        length = jnp.where(outside, 0.0, jnp.where(failed, jnp.nan, solution.ts[-1]))
        return length, jnp.where(outside, seed, solution.ys[-1]), hit | outside

    signs = jnp.array([1.0, -1.0])
    trace = jit(vmap(vmap(trace_one, in_axes=(None, 0)), in_axes=(0, None)))
    lengths, points, hit = trace(jnp.asarray(initial_conditions, dtype=float), signs)
    return {"lengths": lengths, "connection_length": jnp.sum(lengths, axis=1),
            "strike_points": points, "hit": hit}
