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
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, Event, TqdmProgressMeter, NoProgressMeter
from diffrax import ControlTerm,UnsafeBrownianPath,MultiTerm,ItoMilstein,ClipStepSizeController #For collisions we need this to solve stochastic differential equation
import diffrax
from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.surfaces import SurfaceClassifier
from essos.electric_field import Electric_field_flux, Electric_field_zero
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ELEMENTARY_CHARGE,SPEED_OF_LIGHT
from essos.plot import fix_matplotlib_3d
from essos.background_species import nu_s_ab,nu_D_ab,nu_par_ab, d_nu_par_ab,d_nu_D_ab



# If multiple devices are available, set up sharding for parallelization. Otherwise, set sharding to None.
if len(jax.devices()) > 1:
    mesh = Mesh(jax.devices(), ("dev",))
    spec = PartitionSpec("dev", None)
    spec_index = PartitionSpec("dev")
    sharding = NamedSharding(mesh, spec)
    sharding_index = NamedSharding(mesh, spec_index)
else:
    mesh = None
    sharding = None
    sharding_index = None



def gc_to_fullorbit(field, initial_xyz, initial_vparallel, total_speed, mass, charge, phase_angle_full_orbit=0):
    """
    Computes full orbit positions for given guiding center positions,
    parallel speeds, and total velocities using JAX for efficiency.
    """
    def compute_orbit_params(xyz, vpar):
        Bs = field.B_contravariant(xyz)
        AbsBs = jnp.linalg.norm(Bs)
        eB = Bs / AbsBs
        p1 = eB
        p2 = jnp.array([0, 0, 1])
        p3 = -jnp.cross(p1, p2)
        p3 /= jnp.linalg.norm(p3)
        q1 = p1
        q2 = p2 - jnp.dot(q1, p2) * q1
        q2 /= jnp.linalg.norm(q2)
        q3 = p3 - jnp.dot(q1, p3) * q1 - jnp.dot(q2, p3) * q2
        q3 /= jnp.linalg.norm(q3)
        speed_perp = jnp.sqrt(total_speed**2 - vpar**2)
        rg = mass * speed_perp / (jnp.abs(charge) * AbsBs)
        xyz_full = xyz + rg * (jnp.sin(phase_angle_full_orbit) * q2 + jnp.cos(phase_angle_full_orbit) * q3)
        vperp = -speed_perp * jnp.cos(phase_angle_full_orbit) * q2 + speed_perp * jnp.sin(phase_angle_full_orbit) * q3
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
        self.phase_angle_full_orbit = 0
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
    Bstar=field.B_contravariant(points)+vpar*m/q*field.curl_b(points)#+m/q*flow.curl_U0(points)
    Ustar=vpar*field.B_contravariant(points)/field.AbsB(points)#+flow.U0(points) 
    F_gc=mu*field.dAbsB_by_dX(points)+m*vpar**2*field.kappa(points)-q*electric_field.E_covariant(points)#+vpar*flow.coriolis(points)+flow.centrifugal(points)
    dxdt =  Ustar + jnp.cross(field.B_covariant(points), F_gc)/jnp.dot(field.B_covariant(points),Bstar)/q/field.sqrtg(points)
    dvdt = -jnp.dot(Bstar,F_gc)/jnp.dot(field.B_covariant(points),Bstar)*field.AbsB(points)/m    

    return jnp.append(dxdt,dvdt)
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


@jit
def _fill_terminated_trajectories(trajectories, axis_threshold=None):
    """Replace post-event or below-axis saves with the last valid state."""

    def fill_trajectory(trajectory):
        def fill_state(previous, current):
            is_valid = jnp.isfinite(current).all()
            if axis_threshold is not None:
                is_valid = is_valid & (current[0] > axis_threshold)
            state = jnp.where(is_valid, current, previous)
            return state, state

        _, tail = lax.scan(fill_state, trajectory[0], trajectory[1:])
        return jnp.vstack((trajectory[0], tail))

    return vmap(fill_trajectory)(trajectories)


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


def _vmec_radial_events(axis_threshold):
    """Return axis and LCFS events for VMEC guiding-center coordinates."""

    def reached_axis(t, y, args, **kwargs):
        del t, args, kwargs
        return y[0] <= axis_threshold

    def reached_boundary(t, y, args, **kwargs):
        del t, args, kwargs
        return y[0] >= 1.0

    return reached_axis, reached_boundary


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
                 solver=None, axis_threshold=1.e-6, stopping_criteria=None, progress=False):

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
        
        if electric_field==None:
            self.electric_field = Electric_field_zero()
        else:
            self.electric_field=electric_field

        if isinstance(field, Coils):
            self.field = BiotSavart(field)
        else:
            self.field = field

        if rejected_steps==None:
            self.rejected_steps=100
        else:
            self.rejected_steps=100

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
        if not 0.0 < axis_threshold < 1.0:
            raise ValueError("axis_threshold must be strictly between 0 and 1")
        self.axis_threshold = axis_threshold
        self._has_vmec_axis_event = False
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
                if model in _VMEC_GUIDING_CENTER_MODELS:
                    self.condition = _vmec_radial_events(self.axis_threshold)
                    self._has_vmec_axis_event = True
                elif model in ('FieldLine', 'FieldLineAdaptative', 'FieldLineArclength', 'FieldLineToroidal'):
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
        else:
            self.condition = condition
        if model == 'GuidingCenter' or model=='GuidingCenterAdaptative':
            self.ODE_term = ODETerm(GuidingCenter)
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

            
        trace_result = self.trace()
        if self._has_vmec_axis_event:
            trajectories, self.event_mask = trace_result
            self.axis_hits, self.boundary_hits = self.event_mask
            filled = _fill_terminated_trajectories(
                trajectories, self.axis_threshold
            )
            self._trajectories = jnp.where(
                self.axis_hits[:, None, None], filled, trajectories
            )
        elif self.stopping_criteria is not None:
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
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDrift),ControlTerm(GuidingCenterCollisionsDiffusion, bm))
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
                    max_steps=10000000000,
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
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMuStratonovich),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))                
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
                    max_steps=10000000000,
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
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMuStratonovich),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))                
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
                    max_steps=10000000000,
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
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMuIto),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))                
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
                    max_steps=10000000000,
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
                    max_steps=10000000000,
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
                    max_steps=10000000000,
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
                    max_steps=10000000000,
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
                    max_steps=10000000000,
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
                    max_steps=10000000000,
                    event = Event(self.condition)
                )
                trajectory = solution.ys
            if self._has_vmec_axis_event or self.stopping_criteria is not None:
                return trajectory, solution.event_mask
            return trajectory
        
        output_sharding = sharding
        if self._has_vmec_axis_event:
            output_sharding = (sharding, (sharding_index, sharding_index))
        elif self.stopping_criteria is not None:
            event_sharding = sharding_index
            if len(self.stopping_criteria) > 1:
                event_sharding = tuple(sharding_index for _ in self.stopping_criteria)
            output_sharding = (sharding, event_sharding)
        if sharding is not None:
            return jit(vmap(compute_trajectory,in_axes=(0,0)), in_shardings=(sharding,sharding_index), out_shardings=output_sharding)(
                        device_put(self.initial_conditions, sharding), device_put(self.particles.random_keys if self.particles else None, sharding_index))
        else:
            return jit(vmap(compute_trajectory,in_axes=(0,0)))(self.initial_conditions, self.particles.random_keys if self.particles else None)
        #x=jax.device_put(self.initial_conditions, sharding)
        #y=jax.device_put(self.particles.random_keys, sharding_index)        
        #sharded_fun = jax.jit(jax.shard_map(jax.vmap(compute_trajectory,in_axes=(0,0)), mesh=mesh, in_specs=(spec,spec_index), out_specs=spec))
        #return sharded_fun(x, y).block_until_ready()    

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

        elif self.model in ('FieldLine', 'FieldLineAdaptative', 'FieldLineArclength', 'FieldLineToroidal'):
            energy = jnp.ones((len(self.initial_conditions), self.times_to_trace))
            
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

        elif self.model in ('FieldLine', 'FieldLineAdaptative', 'FieldLineArclength', 'FieldLineToroidal'):
            v_perp = jnp.ones((len(self.initial_conditions), self.times_to_trace))
            
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

    def loss_fraction(self,r_max=0.99):
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
    def loss_fraction_collisions(self,r_max=0.99):
        trajectories_rtz = self.trajectories[:,:, :3]
        lost_mask = trajectories_rtz[:,:,0] >= r_max
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        has_lost = lost_indices != -1
        safe_indices = jnp.clip(lost_indices, 0, len(self.times) - 1)
        particle_indices = jnp.arange(self.particles.nparticles)
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
                    'progress': self.progress}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Tracing,
                               Tracing._tree_flatten,
                               Tracing._tree_unflatten)
