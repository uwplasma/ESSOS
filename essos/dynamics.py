from pyexpat import model
import jax
jax.config.update("jax_enable_x64", True)
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
from essos.fields import BiotSavart, Vmec, ExternalField
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
    d_Diffusion_perp_dp=p*nu_par+p**2*dnu_D_dv/(2.*m)    
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
    d_Diffusion_perp_dp=p*nu_par+p**2*dnu_D_dv/(2.*m)    

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




def _guiding_center_velocity(field, electric_field, q, m, points, vpar, mu):
    """Guiding-center dx/dt and dv_par/dt at fixed magnetic moment ``mu``."""
    Bstar=field.B_contravariant(points)+vpar*m/q*field.curl_b(points)#+m/q*flow.curl_U0(points)
    Ustar=vpar*field.B_contravariant(points)/field.AbsB(points)#+flow.U0(points) 
    F_gc=mu*field.dAbsB_by_dX(points)+m*vpar**2*field.kappa(points)-q*electric_field.E_covariant(points)#+vpar*flow.coriolis(points)+flow.centrifugal(points)
    dxdt =  Ustar + jnp.cross(field.B_covariant(points), F_gc)/jnp.dot(field.B_covariant(points),Bstar)/q/field.sqrtg(points)
    dvdt = -jnp.dot(Bstar,F_gc)/jnp.dot(field.B_covariant(points),Bstar)*field.AbsB(points)/m    
    return dxdt, dvdt


@partial(jit, static_argnums=(2))
def GuidingCenter(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z, vpar = initial_condition
    field, particles,electric_field = args
    q = particles.charge
    m = particles.mass
    E = particles.energy
    points = jnp.array([x, y, z])
    mu = (E - m*vpar**2/2)/field.AbsB(points)
    dxdt, dvdt = _guiding_center_velocity(field, electric_field, q, m, points, vpar, mu)
    return jnp.append(dxdt,dvdt)


@partial(jit, static_argnums=(2))
def GuidingCenterMu(t, initial_condition, args) -> jnp.ndarray:
    """Collisionless guiding center with state (x, y, z, v_par, mu); mu is constant."""
    field, particles, electric_field = args
    dxdt, dvdt = _guiding_center_velocity(field, electric_field, particles.charge, particles.mass,
                                          initial_condition[:3], initial_condition[3], initial_condition[4])
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
    I_vv_tensor=jnp.identity(3)-jnp.diag(jnp.multiply(v_vector,jnp.reshape(v_vector,(3,1))))/v**2
    indeces_species=species.species_indeces
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    Diffusion_par=p**2/2.*nu_par
    Diffusion_perp=p**2/2.*nu_D 
    Dpar=jnp.sqrt(2.*Diffusion_par)#*0.0000
    Dperp=jnp.sqrt(2.*Diffusion_perp)#*0.0000
    dxdt = jnp.zeros((3,3))
    dvdt=Dpar/m*jnp.identity(3)-Dperp/m*I_vv_tensor
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


def _to_axis_regular(y):
    """Map (s, theta, ...) to (sqrt(s) cos theta, sqrt(s) sin theta, ..., 0)."""
    r = jnp.sqrt(y[0])
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


# Event times are refined to |dt| < 1e-8 t + 1e-11 s, and the event function
# (1 - s, or a distance in metres) to 1e-11.
_EVENT_ROOT_FINDER = optx.Newton(rtol=1e-8, atol=1e-11)


def _with_failure_flag(vector_field, drift=True):
    """Append a failure flag to the state of a vector field.

    A non-finite right-hand side makes an adaptive controller reject every
    step, and the solve runs to max_steps without the state ever turning
    non-finite. Instead, the drift returns zero with the flag's rate set to
    one, the step is accepted, and the flag event stops the solve there as a
    failure. The flag has no noise.
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
    del t, args, kwargs
    return (y[-1] > 0) | ~jnp.isfinite(y).all()


def _solve_succeeded(solution):
    """A solve that reached t1 or stopped on an event; an event whose root find
    did not reach the tolerance still stops at the end of its step."""
    result = solution.result
    return ((result == diffrax.RESULTS.successful) | (result == diffrax.RESULTS.event_occurred)
            | (result == diffrax.RESULTS.nonlinear_max_steps_reached))


def _keep_energy(vpar, mu, E, B, m):
    """v_par and mu in a field of strength B with energy E, the sign of v_par and, if possible, mu.

    The fields on the two sides of the LCFS differ slightly; where mu B > E the
    orbit mirrors there, with v_par = 0 and mu = E / B.
    """
    mu = jnp.minimum(mu, E / B)
    return jnp.sign(vpar) * jnp.sqrt(jnp.maximum(2 * (E - mu * B) / m, 0.0)), mu


def _wall_distance(wall):
    """Signed distance function of a wall: a classifier with ``evaluate_xyz`` or a callable."""
    if wall is None:
        return None
    if hasattr(wall, "evaluate_xyz"):
        return wall.evaluate_xyz
    if callable(wall):
        return wall
    raise TypeError("wall must provide evaluate_xyz(xyz) or be a callable of xyz")


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
                 max_steps=1_000_000, exterior_field=None, wall=None, max_returns=16, reentry_depth=None):

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
        # exterior_field up to the wall when one is given; see _trace_vmec.
        self._vmec_default = self._axis_regular and condition is None
        if (exterior_field is not None or wall is not None) and not self._vmec_default:
            raise ValueError("exterior_field and wall need a VMEC field, a guiding-center model and no condition")
        if wall is not None and exterior_field is None:
            raise ValueError("a wall needs an exterior_field to trace the orbits outside the LCFS")
        if isinstance(exterior_field, Coils):
            exterior_field = BiotSavart(exterior_field)
        elif exterior_field is not None and not hasattr(exterior_field, "curl_b"):
            exterior_field = ExternalField(exterior_field)
        self.exterior_field = exterior_field
        self.wall = wall
        self.max_returns = int(max_returns)
        # An orbit outside counts as back inside once it is reentry_depth [m]
        # inside the LCFS (default 1e-3 minor radii), so an orbit skimming the
        # surface, where the two fields differ slightly, does not bounce
        # between them.
        if reentry_depth is None and self._vmec_default:
            reentry_depth = 1e-3 * float(jnp.abs(jnp.asarray(field.Aminor_p)))
        self.reentry_depth = reentry_depth
        # Diffrax's ceiling was effectively unbounded, so a trace that could not
        # finish ran until the process was killed rather than returning.
        self.max_steps = max_steps
        self.progress = bool(progress)
        self.progress_meter = TqdmProgressMeter() if self.progress else NoProgressMeter()
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
            B_particle=jax.vmap(field.AbsB,in_axes=0)(particles.initial_xyz)
            mu=self.particles.initial_vperpendicular**2*self.particles.mass*0.5/B_particle/(SPEED_OF_LIGHT**2*particles.mass)          
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
            self.args = (self.field, self.particles,self.species,self.tag_gc)
            print(self.args)
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
            self._trace_vmec()
            self.loss_fractions, self.total_particles_lost, self.lost_times = self._vmec_losses()
            if self.model in _GUIDING_CENTER_COLLISION_MODELS:
                self.lost_energies, self.lost_positions = self._vmec_lost_states()
            return
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
            if self._axis_regular:
                initial_condition = _to_axis_regular(initial_condition)
            # initial_condition = initial_condition[0]
            if self.model == 'FullOrbit_Boris':
                dt=self.timestep#self.maxtime / self.timesteps
                def update_state(state, _):
                    # def update_fn(state):
                    x = state[:3]
                    v = state[3:]
                    t = self.particles.charge / self.particles.mass *  self.field.B_contravariant(x) * 0.5 * dt
                    s = 2. * t / (1. + jnp.dot(t,t))
                    vprime = v + jnp.cross(v, t)
                    v += jnp.cross(vprime, s)
                    x += v * dt
                    new_state = jnp.concatenate((x, v))
                    return new_state, new_state
                    # def no_update_fn(state):
                    #     x, v = state
                    #     return (x, v), jnp.concatenate((x, v))
                    # condition = (jnp.sqrt(x1**2 + x2**2) > 50) | (jnp.abs(x3) > 20)
                    # return lax.cond(condition, no_update_fn, update_fn, state)
                    # return update_fn(state)
                _, trajectory = lax.scan(update_state, initial_condition, jnp.arange(len(self.times)-1))
                trajectory = jnp.vstack([initial_condition, trajectory])
            elif self.model == 'GuidingCenterCollisions':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.timestep#self.maxtime / self.timesteps
                tol=dt0*0.5
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,), key=particle_key, levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(self._vector_field(GuidingCenterCollisionsDrift)),ControlTerm(self._vector_field(GuidingCenterCollisionsDiffusion), bm))
                solution = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=dt0,
                    y0=initial_condition,
                    #solver=diffrax.SlowRK(),
                    solver=diffrax.StratonovichMilstein(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    #stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.tol_step_size, atol=self.tol_step_size),
                    max_steps=self.max_steps,
                    event = Event(self.condition),
                    progress_meter=self.progress_meter,
                )
                trajectory = solution.ys
            elif self.model == 'GuidingCenterCollisionsMuAdaptative':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.timestep#self.maxtime / self.timesteps
                tol=dt0*0.5
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,),key=particle_key,levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(self._vector_field(GuidingCenterCollisionsDriftMuStratonovich)),ControlTerm(self._vector_field(GuidingCenterCollisionsDiffusionMu), bm))                
                solution = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=dt0,
                    y0=initial_condition,
                    solver=diffrax.SPaRK(),
                    #solver=diffrax.HalfSolver(diffrax.GeneralShARK()),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    stepsize_controller=ClipStepSizeController(controller=PIDController(pcoeff=0.1, icoeff=0.3, dcoeff=0.0, rtol=self.rtol, atol=self.atol,dtmin=dt0,dtmax=1.e-4,force_dtmin=True),step_ts=self.times,store_rejected_steps=self.rejected_steps),
                    max_steps=self.max_steps,
                    event = Event(self.condition),
                    progress_meter=self.progress_meter,
                )
                trajectory = solution.ys
            elif self.model == 'GuidingCenterCollisionsMuFixed':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.timestep#self.maxtime / self.timesteps
                tol=dt0*0.5
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,),key=particle_key,levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(self._vector_field(GuidingCenterCollisionsDriftMuStratonovich)),ControlTerm(self._vector_field(GuidingCenterCollisionsDiffusionMu), bm))                
                solution = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=dt0,
                    y0=initial_condition,
                    solver=diffrax.StratonovichMilstein(),                    
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    max_steps=self.max_steps,
                    event = Event(self.condition),
                    progress_meter=self.progress_meter,
                )
                trajectory = solution.ys
            elif self.model == 'GuidingCenterCollisionsMuIto':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.timestep#self.maxtime / self.timesteps
                tol=dt0*0.5
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,),key=particle_key,levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(self._vector_field(GuidingCenterCollisionsDriftMuIto)),ControlTerm(self._vector_field(GuidingCenterCollisionsDiffusionMu), bm))                
                solution = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=dt0,
                    y0=initial_condition,
                    solver=diffrax.ItoMilstein(),                    
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    max_steps=self.max_steps,
                    event = Event(self.condition),
                    progress_meter=self.progress_meter,
                )
                trajectory = solution.ys
            elif self.model == 'FullOrbitCollisions':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.timestep#self.maxtime / self.timesteps
                tol=dt0*0.5
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(6,), key=particle_key, levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(LorentzCollisionsDrift),ControlTerm(LorentzCollisionsDiffusion,bm))
                trajectory = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=dt0,
                    y0=initial_condition,
                    solver=diffrax.SPaRK(),
                    #solver=diffrax.ItoMilstein(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),                   
                    stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.tol_step_size, atol=self.tol_step_size,dtmin=dt0),
                    max_steps=self.max_steps,
                    event = Event(self.condition),
                    progress_meter=self.progress_meter,
                ).ys          
            elif self.model == 'GuidingCenterAdaptative' :  
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                solution = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.timestep,#self.maxtime / self.timesteps,
                    y0=initial_condition,
                    solver=(self.solver if self.solver is not None else diffrax.Dopri8()),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    progress_meter=self.progress_meter,
                    stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol),
                    max_steps=self.max_steps,
                    event = Event(self.condition)
                )
                trajectory = solution.ys
            elif self.model == 'FullOrbitAdaptative' :
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning)
                solution = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.timestep,
                    y0=initial_condition,
                    solver=(self.solver if self.solver is not None else diffrax.Dopri8()),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    progress_meter=self.progress_meter,
                    stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol),
                    max_steps=self.max_steps,
                    event = Event(self.condition)
                )
                trajectory = solution.ys
            elif self.model in ('FieldLineAdaptative', 'FieldLineArclength', 'FieldLineToroidal'):
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                solution = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.timestep,#self.maxtime / self.timesteps,
                    y0=initial_condition,
                    solver=(self.solver if self.solver is not None else diffrax.Dopri8()),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    progress_meter=self.progress_meter,
                    stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol),
                    max_steps=self.max_steps,
                    event = Event(self.condition)
                )
                trajectory = solution.ys
            #Fixed guiding center
            else:
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                solution = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.timestep,#self.maxtime / self.timesteps,
                    y0=initial_condition,
                    solver=(self.solver if self.solver is not None else diffrax.Dopri8()),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=True,
                    # adjoint=DirectAdjoint(),
                    progress_meter=self.progress_meter,
                    max_steps=self.max_steps,
                    event = Event(self.condition)
                )
                trajectory = solution.ys
            if self._axis_regular:
                trajectory = vmap(_from_axis_regular)(trajectory)
            if self.stopping_criteria is not None:
                return trajectory, solution.event_mask
            return trajectory
        
        devices = self.devices
        device_count = min(len(devices), len(self.initial_conditions))
        while device_count > 1 and len(self.initial_conditions) % device_count:
            device_count -= 1
        if device_count > 1:
            mesh = Mesh(np.asarray(devices[:device_count], dtype=object), ("dev",))
            sharding = NamedSharding(mesh, PartitionSpec("dev", None))
            sharding_index = NamedSharding(mesh, PartitionSpec("dev"))
        else:
            sharding = sharding_index = None

        output_sharding = sharding
        if self.stopping_criteria is not None:
            event_sharding = sharding_index
            if len(self.stopping_criteria) > 1:
                event_sharding = tuple(sharding_index for _ in self.stopping_criteria)
            output_sharding = (sharding, event_sharding)
        if sharding is not None:
            initial_conditions = device_put(
                np.asarray(jax.device_get(self.initial_conditions)), sharding)
            random_keys = self.particles.random_keys if self.particles else None
            if random_keys is not None:
                random_keys = device_put(jax.device_get(random_keys), sharding_index)
            return jit(vmap(compute_trajectory,in_axes=(0,0)), in_shardings=(sharding,sharding_index), out_shardings=output_sharding)(
                        initial_conditions, random_keys)
        else:
            device = devices[0]
            initial_conditions = device_put(
                np.asarray(jax.device_get(self.initial_conditions)), device)
            random_keys = self.particles.random_keys if self.particles else None
            if random_keys is not None:
                random_keys = device_put(jax.device_get(random_keys), device)
            with jax.default_device(device):
                return jit(vmap(compute_trajectory,in_axes=(0,0)))(
                    initial_conditions, random_keys)
        #x=jax.device_put(self.initial_conditions, sharding)
        #y=jax.device_put(self.particles.random_keys, sharding_index)        
        #sharded_fun = jax.jit(jax.shard_map(jax.vmap(compute_trajectory,in_axes=(0,0)), mesh=mesh, in_specs=(spec,spec_index), out_specs=spec))
        #return sharded_fun(x, y).block_until_ready()    

    # -- VMEC guiding centers: exact LCFS crossing, failures, exterior continuation --

    def _vmec_inside_solve(self, t0, y0, key):
        """Solve a VMEC guiding-center model from t0 to maxtime in the axis-regular chart.

        Stops at the exact LCFS crossing (root-found event) or on a non-finite
        state. Returns the saves at self.times, the end time and state in flux
        coordinates, the two event flags and whether the solver succeeded.
        """
        T, dt0, model = self.maxtime, self.timestep, self.model
        controller = diffrax.ConstantStepSize()
        if model in ('GuidingCenter', 'GuidingCenterAdaptative'):
            terms = ODETerm(_with_failure_flag(self._vector_field(GuidingCenter)))
            solver = self.solver if self.solver is not None else diffrax.Dopri8()
            if model == 'GuidingCenterAdaptative':
                controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol)
        else:
            drift, diffusion, solver = {
                'GuidingCenterCollisions': (GuidingCenterCollisionsDrift, GuidingCenterCollisionsDiffusion,
                                            diffrax.StratonovichMilstein()),
                'GuidingCenterCollisionsMuFixed': (GuidingCenterCollisionsDriftMuStratonovich,
                                                   GuidingCenterCollisionsDiffusionMu, diffrax.StratonovichMilstein()),
                'GuidingCenterCollisionsMuIto': (GuidingCenterCollisionsDriftMuIto, GuidingCenterCollisionsDiffusionMu,
                                                 diffrax.ItoMilstein()),
                'GuidingCenterCollisionsMuAdaptative': (GuidingCenterCollisionsDriftMuStratonovich,
                                                        GuidingCenterCollisionsDiffusionMu, diffrax.SPaRK()),
            }[model]
            bm = diffrax.VirtualBrownianTree(0.0, T, tol=dt0 * 0.5, shape=(5,), key=key,
                                             levy_area=diffrax.SpaceTimeTimeLevyArea)
            terms = MultiTerm(ODETerm(_with_failure_flag(self._vector_field(drift))),
                              ControlTerm(_with_failure_flag(self._vector_field(diffusion), drift=False), bm))
            if model == 'GuidingCenterCollisionsMuAdaptative':
                controller = ClipStepSizeController(
                    controller=PIDController(pcoeff=0.1, icoeff=0.3, dcoeff=0.0, rtol=self.rtol, atol=self.atol,
                                             dtmin=dt0, dtmax=1.e-4, force_dtmin=True),
                    step_ts=self.times, store_rejected_steps=self.rejected_steps)
        event = Event((lambda t, y, args, **kwargs: 1.0 - y[0]**2 - y[1]**2, _failed_event),
                      root_finder=_EVENT_ROOT_FINDER, direction=(False, None))
        solution = diffeqsolve(
            terms, solver, t0=t0, t1=T, dt0=dt0, y0=jnp.append(_to_axis_regular(y0), 0.0), args=self.args,
            saveat=SaveAt(subs=[diffrax.SubSaveAt(ts=jnp.clip(self.times, t0, T)), diffrax.SubSaveAt(t1=True)]),
            stepsize_controller=controller, event=event, throw=False, max_steps=self.max_steps,
            progress_meter=self.progress_meter)
        saves, end = solution.ys
        crossed, non_finite = solution.event_mask
        ok = _solve_succeeded(solution)
        return (vmap(_from_axis_regular)(saves[:, :-1]), solution.ts[1][-1], _from_axis_regular(end[-1, :-1]),
                crossed, non_finite, ok)

    def _vmec_exterior_solve(self, t0, y0):
        """Collisionless guiding center (x, y, z, v_par, mu) in exterior_field from t0 to maxtime.

        Stops, at the root-found crossing, on the wall or on re-entering the
        LCFS (Vmec.boundary_distance rising through reentry_depth), or on a
        non-finite state.
        """
        wall = _wall_distance(self.wall)
        reentry_depth = self.reentry_depth
        event = Event((lambda t, y, args, **kwargs: wall(y[:3]) if wall is not None else jnp.ones(()),
                       lambda t, y, args, **kwargs: self.field.boundary_distance(y[:3]) - reentry_depth,
                       _failed_event),
                      root_finder=_EVENT_ROOT_FINDER, direction=(False, True, None))
        solution = diffeqsolve(
            ODETerm(_with_failure_flag(GuidingCenterMu)), self.solver if self.solver is not None else diffrax.Dopri8(),
            t0=t0, t1=self.maxtime, dt0=self.timestep, y0=jnp.append(y0, 0.0),
            args=(self.exterior_field, self.particles, Electric_field_zero()),
            saveat=SaveAt(subs=[diffrax.SubSaveAt(ts=jnp.clip(self.times, t0, self.maxtime)),
                                diffrax.SubSaveAt(t1=True)]),
            stepsize_controller=PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol),
            event=event, throw=False, max_steps=self.max_steps, progress_meter=self.progress_meter)
        saves, end = solution.ys
        struck, returned, non_finite = solution.event_mask
        ok = _solve_succeeded(solution)
        return saves[:, :-1], solution.ts[1][-1], end[-1, :-1], struck, returned, non_finite, ok

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

    def _map_particles(self, fn, *arrays):
        """jit(vmap(fn)) over the particles, sharded over self.devices as in trace().

        The compiled function is kept, so later segments reuse it.
        """
        n = len(arrays[0])
        count = min(len(self.devices), n)
        while count > 1 and n % count:
            count -= 1
        cache = self.__dict__.setdefault("_compiled", {})
        key = (fn.__name__, count)
        if key not in cache:
            if count > 1:
                mesh = Mesh(np.asarray(self.devices[:count], dtype=object), ("dev",))
                sharding = NamedSharding(mesh, PartitionSpec("dev"))
                cache[key] = (jit(vmap(fn), in_shardings=sharding, out_shardings=sharding), sharding)
            else:
                cache[key] = (jit(vmap(fn)), None)
        compiled, sharding = cache[key]
        if sharding is not None:
            return compiled(*[device_put(a, sharding) for a in arrays])
        with jax.default_device(self.devices[0]):
            return compiled(*arrays)

    def _trace_vmec(self):
        """Trace VMEC guiding centers segment by segment: inside, outside, and back inside.

        Inside the LCFS the model runs in the axis-regular chart until it
        reaches maxtime, crosses the LCFS (found by root finding), or produces
        a non-finite state. Without exterior_field a crossing ends the orbit.
        With it, the orbit continues as a collisionless guiding center in
        Cartesian coordinates, with its energy and magnetic moment, until it
        strikes the wall, reaches maxtime or re-enters the LCFS; a re-entered
        orbit is mapped back to flux coordinates and continues inside, up to
        max_returns times.
        """
        T, times = self.maxtime, np.asarray(self.times)
        y_in = jnp.asarray(self.initial_conditions)
        n, nt = y_in.shape[0], len(times)
        keys = self.particles.random_keys
        trajectories = np.full((n, nt, y_in.shape[1]), np.inf)
        trajectories_xyz = np.full((n, nt, 3), np.nan)
        region = np.full((n, nt), -1, dtype=np.int8)
        t = np.zeros(n)
        inside, active = np.ones(n, bool), np.ones(n, bool)
        status = np.zeros(n, dtype=np.int8)
        lcfs_time, wall_time, failure_time = (np.full(n, np.inf) for _ in range(3))
        lcfs_state = np.full(y_in.shape, np.nan)
        lcfs_xyz, wall_xyz = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        lcfs_energy, wall_energy = np.full(n, np.nan), np.full(n, np.nan)
        returns = np.zeros(n, dtype=int)
        y_out = jnp.zeros((n, 5))
        inside_solve = self._vmec_inside_solve
        to_xyz = jit(vmap(vmap(self.field.to_xyz)))
        to_exterior = jit(vmap(self._vmec_to_exterior))
        from_exterior = jit(vmap(self._vmec_from_exterior))

        def record(run, t0, t_end, saves_xyz, saves=None, code=0):
            window = run[:, None] & (times >= t0[:, None]) & (times <= t_end[:, None])
            window &= np.isfinite(saves_xyz if saves is None else saves).all(axis=2)
            trajectories_xyz[window] = saves_xyz[window]
            region[window] = code
            if saves is not None:
                trajectories[window] = saves[window]

        started_outside = np.asarray(y_in[:, 0]) >= 1.0
        for segment in range(2 * self.max_returns + 2):
            run = active & inside
            if run.any():
                t0 = np.where(run, t, T)
                saves, t_end, y_end, crossed, non_finite, ok = (
                    np.asarray(a) for a in self._map_particles(inside_solve, jnp.asarray(t0), y_in, keys))
                t_end = np.where(run & (segment == 0) & started_outside, 0.0, t_end)
                crossed = crossed | (segment == 0) & started_outside
                y_end = np.where((run & (segment == 0) & started_outside)[:, None], np.asarray(y_in), y_end)
                record(run, t0, t_end, np.asarray(to_xyz(jnp.asarray(np.where(np.isfinite(saves), saves, 0.0)[..., :3]))),
                       saves, code=0)
                failed = run & (non_finite | ~ok)
                crossed = run & crossed & ~failed
                first = crossed & np.isinf(lcfs_time)
                exit_state, exit_energy = (np.asarray(a) for a in to_exterior(jnp.asarray(np.where(crossed[:, None], y_end, y_in))))
                lcfs_time[first], lcfs_state[first] = t_end[first], y_end[first]
                lcfs_xyz[first], lcfs_energy[first] = exit_state[first, :3], exit_energy[first]
                failure_time[failed] = t_end[failed]
                status[failed] = 4
                status[run & ~crossed & ~failed] = 0
                active &= ~(failed | (run & ~crossed))
                if self.exterior_field is None:
                    status[crossed] = 1
                    active &= ~crossed
                else:
                    inside &= ~crossed
                    y_out = jnp.where(jnp.asarray(crossed)[:, None], jnp.asarray(exit_state), y_out)
                t = np.where(run, t_end, t)
            run = active & ~inside
            if run.any():
                t0 = np.where(run, t, T)
                saves, t_end, y_end, struck, returned, non_finite, ok = (
                    np.asarray(a) for a in self._map_particles(self._vmec_exterior_solve, jnp.asarray(t0), y_out))
                record(run, t0, t_end, saves[..., :3], code=1)
                failed = run & (non_finite | ~ok)
                struck, returned = run & struck & ~failed, run & returned & ~failed & ~struck
                wall_time[struck], wall_xyz[struck] = t_end[struck], y_end[struck, :3]
                wall_energy[struck] = (0.5 * self.particles.mass * y_end[struck, 3]**2
                                       + y_end[struck, 4] * np.asarray(vmap(self.exterior_field.AbsB)(jnp.asarray(y_end[struck, :3]))))
                failure_time[failed] = t_end[failed]
                status[failed], status[struck] = 4, 3
                status[run & ~(failed | struck | returned)] = 2
                over = returned & (returns >= self.max_returns)
                status[over] = 5
                back = returned & ~over
                returns[back] += 1
                active &= ~(run & ~back)
                inside |= back
                if back.any():
                    y_in = jnp.where(jnp.asarray(back)[:, None],
                                     from_exterior(jnp.asarray(np.where(back[:, None], y_end, np.asarray(y_out)))), y_in)
                t = np.where(run, t_end, t)
            if not active.any():
                break

        self._trajectories = jnp.asarray(trajectories)
        self.trajectories_xyz = jnp.asarray(trajectories_xyz)
        self.region, self.status = region, status
        self.lcfs_times, self.lcfs_states, self.lcfs_positions, self.lcfs_energies = (
            lcfs_time, lcfs_state, lcfs_xyz, lcfs_energy)
        self.wall_hits, self.wall_times, self.wall_positions, self.wall_energies = (
            status == 3, wall_time, wall_xyz, wall_energy)
        self.returns, self.failed, self.failure_times = returns, status == 4, failure_time
        self.boundary_hits = jnp.asarray(np.isfinite(lcfs_time))
        self.event_mask = self.boundary_hits
        self.axis_hits = jnp.zeros(n, dtype=bool)
        self.total_particles_unresolved = jnp.sum(self.axis_hits)
        self.toroidal_angles = None

    def _vmec_loss_times(self):
        return self.wall_times if self.exterior_field is not None else self.lcfs_times

    def _vmec_losses(self):
        """Loss fractions from the exact loss times: the wall with an exterior field, else the LCFS."""
        lost_time = self._vmec_loss_times()
        loss_fractions = jnp.asarray(np.mean(lost_time[:, None] <= np.asarray(self.times)[None, :], axis=0))
        return loss_fractions, jnp.sum(jnp.isfinite(lost_time)), jnp.asarray(np.where(np.isfinite(lost_time), lost_time, -1))

    def _vmec_lost_states(self):
        """Energy of each lost particle at the loss, and where: xyz on the wall, else (s, theta, phi) on the LCFS."""
        if self.exterior_field is not None:
            return (jnp.asarray(np.where(self.wall_hits, self.wall_energies, 0.0)),
                    jnp.asarray(np.where(self.wall_hits[:, None], self.wall_positions, 0.0)))
        lost = np.isfinite(self.lcfs_times)
        return (jnp.asarray(np.where(lost, self.lcfs_energies, 0.0)),
                jnp.asarray(np.where(lost[:, None], self.lcfs_states[:, :3], 0.0)))

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

        elif self.model == 'FullOrbit' or self.model == 'FullOrbit_Boris' or self.model == 'FullOrbitAdaptative':
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
                vpar=trajectory[:, 3]*trajectory[:, 4]
                v=trajectory[:, 4]*SPEED_OF_LIGHT
                return jnp.sqrt(v**2-vpar**2)
            v_perp = vmap(compute_vperp)(self.trajectories)

        elif self.model == 'FullOrbit' or self.model == 'FullOrbit_Boris' or self.model == 'FullOrbitAdaptative':
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
                    'progress': self.progress, 'devices': self.devices}  # static values
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
