import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import jit, vmap, tree_util, random, lax, device_put
from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, Event
from diffrax import ControlTerm,UnsafeBrownianPath,MultiTerm,ItoMilstein #For collisions we need this to solve stochastic differential equation
import diffrax
from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.plot import fix_matplotlib_3d
from essos.util import roots
from essos.background_species import nu_s_ab,nu_D_ab,nu_par_ab, d_nu_par_ab

mesh = Mesh(jax.devices(), ("dev",))
spec=PartitionSpec("dev", None)
spec_index=PartitionSpec("dev")
sharding = NamedSharding(mesh, spec)
sharding_index = NamedSharding(mesh, spec_index)

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
                 max_vparallel_over_v=1, field=None, initial_vxvyvz=None, initial_xyz_fullorbit=None):
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






@partial(jit, static_argnums=(2))
def GuidingCenterCollisionsDiffusionMu(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z, vpar,mu = initial_condition
    field, particles,species,tag_gc = args
    q = particles.charge
    m = particles.mass
    #E = m/2*v**2   
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(points)
    AbsB = field.AbsB(points)
    #I_bb_tensor=jnp.identity(3)-jnp.diag(jnp.multiply(B_contravariant,B_contravariant))/AbsB**2
    I_bb_tensor=jnp.identity(3)-jnp.diag(jnp.multiply(B_contravariant,jnp.reshape(B_contravariant,(3,1))))/AbsB**2
    AbsB_par=AbsB #should take into account B_par modification, but it does not matter for vacuum fields, so let's keep this for now
    omega_mod = q*AbsB_par/m    
    v=jnp.sqrt(2./m*(0.5*m*vpar**2+mu*AbsB))
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
    mat1=jnp.diag(jnp.array([v,0.5*m*v**2/AbsB]))
    mat2=jnp.append(Q1,Q2,axis=1)
    mat3=jnp.diag(jnp.array([jnp.sqrt(2.*lambda_p),jnp.sqrt(2.*lambda_m)]))
    sigma=jnp.select(condlist=[jnp.abs(xi)<1,jnp.abs(xi)==1],choicelist=[jnp.matmul(mat1,jnp.matmul(mat2,mat3)),jnp.diag(jnp.array([jnp.sqrt(2.*Diffusion_par)/m,0.]))])
    dxdt = jnp.sqrt(2.*Diffusion_x)*I_bb_tensor
    #Off diagonals between position an dvelocity are zero at zeroth order
    Dxv=jnp.zeros((2,3))
    Dvx=jnp.zeros((3,2))
    return jnp.append(jnp.append(dxdt,Dxv,axis=0),jnp.append(Dvx,sigma,axis=0),axis=1)


@partial(jit, static_argnums=(2))
def GuidingCenterCollisionsDriftMu(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z,vpar,mu = initial_condition
    field, particles,species,tag_gc = args
    q = particles.charge
    m = particles.mass
    #E = m/2*v**2
    points = jnp.array([x, y, z])
    B_covariant = field.B_covariant(points)
    B_contravariant = field.B_contravariant(points)
    AbsB =field.AbsB(points)
    gradB = field.dAbsB_by_dX(points)
    AbsB_par=AbsB #should take into account B_par modification, but it does not matter for vacuum fields, so let's keep this for now
    omega_mod = q*AbsB_par/m    
    v=jnp.sqrt(2./m*(0.5*m*vpar**2+mu*AbsB))
    xi=vpar/v
    p=m*v
    indeces_species=species.species_indeces
    nu_s=jnp.sum(jax.vmap(nu_s_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    dnu_par_dv=jnp.sum(jax.vmap(d_nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    Diffusion_par=p**2*nu_par/2.
    Diffusion_perp=p**2*nu_D/2.
    d_Diffusion_par_dp=p*nu_par+p**2*dnu_par_dv/(2.*m)
    Avpar=-nu_s*vpar+vpar/p**2*(2.*(Diffusion_par-Diffusion_perp)+p*d_Diffusion_par_dp)
    Amu=-nu_s*2.*mu+2.*mu/p**2*(3.*(Diffusion_par-Diffusion_perp)+p*d_Diffusion_par_dp)+2.*Diffusion_perp/(m*AbsB)
    dxdt = tag_gc*(vpar*B_contravariant/AbsB + (vpar**2/omega_mod+mu/q)*jnp.cross(B_covariant, gradB)/AbsB/AbsB)
    dvpardt=  Avpar-mu/m*jnp.dot(B_contravariant,gradB)/AbsB*tag_gc
    dmudt = Amu
    return jnp.append(dxdt,jnp.append(dvpardt,dmudt))


@partial(jit, static_argnums=(2))
def GuidingCenterCollisionsDiffusion(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z, v,xi = initial_condition
    field, particles,species,tag_gc = args
    q = particles.charge
    m = particles.mass
    #E = m/2*v**2
    vpar=xi*v    
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(points)
    AbsB = field.AbsB(points)
    #I_bb_tensor=jnp.identity(3)-jnp.diag(jnp.multiply(B_contravariant,B_contravariant))/AbsB**2
    I_bb_tensor=jnp.identity(3)-jnp.diag(jnp.multiply(B_contravariant,jnp.reshape(B_contravariant,(3,1))))/AbsB**2
    AbsB_par=AbsB #should take into account B_par modification, but it does not matter for vacuum fields, so let's keep this for now
    omega_mod = q*AbsB_par/m    
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
    field, particles,species,tag_gc = args
    q = particles.charge
    m = particles.mass
    #E = m/2*v**2
    vpar=xi*v
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_covariant = field.B_covariant(points)
    B_contravariant = field.B_contravariant(points)
    AbsB = field.AbsB(points)
    gradB = field.dAbsB_by_dX(points)
    mu = (m*v**2/2 - m*vpar**2/2)/AbsB
    omega = q*AbsB/m
    p=m*v
    indeces_species=species.species_indeces
    nu_s=jnp.sum(jax.vmap(nu_s_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_D=jnp.sum(jax.vmap(nu_D_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    nu_par=jnp.sum(jax.vmap(nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    dnu_par=jnp.sum(jax.vmap(d_nu_par_ab,in_axes=(None,None,0,None,None,None))(m, q,indeces_species,v, points,species),axis=0)
    Diffusion_par=p**2/2.*nu_par
    Diffusion_perp=p**2/2.*nu_D 
    d_Diffusion_par_dp=p*nu_par+p**2/2.*dnu_par/m
    dxdt = tag_gc*(vpar*B_contravariant/AbsB + (vpar**2/omega+mu/q)*jnp.cross(B_covariant, gradB)/AbsB/AbsB)
    dvdt=(-nu_s*p+2.*Diffusion_par/p+d_Diffusion_par_dp*0.5)/m  #equation format was in p=m*v so we divide by m)
    dxidt =-xi*2.*Diffusion_perp/p**2*0.5-mu/m*jnp.dot(B_contravariant,gradB)/AbsB/v*tag_gc
    return jnp.append(dxdt,jnp.append(dvdt,dxidt))




@partial(jit, static_argnums=(2))
def GuidingCenter(t,
                  initial_condition,
                  args) -> jnp.ndarray:
    x, y, z, vpar = initial_condition
    field, particles = args
    q = particles.charge
    m = particles.mass
    E = particles.energy
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_covariant = field.B_covariant(points)
    B_contravariant = field.B_contravariant(points)
    AbsB = field.AbsB(points)
    gradB = field.dAbsB_by_dX(points)
    mu = (E - m*vpar**2/2)/AbsB
    omega = q*AbsB/m
    dxdt = vpar*B_contravariant/AbsB + (vpar**2/omega+mu/q)*jnp.cross(B_covariant, gradB)/AbsB/AbsB
    dvdt = -mu/m*jnp.dot(B_contravariant,gradB)/AbsB
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



## !!!!  Here species and tag_gc were added  (E. Neto collisions modifications)
## species is a class for collision frquencies + possible temperature + density profiles in file species_background.py
## tag_gc is a tag to turn off 0, or on 1 the GC part of the equations for testing collision statistics independently of GC phsyics
## !!!!  Here particle_key was added to compute_trajectories (E. Neto collisions modifications)
## This is important for correct sampling of Brownian motion
class Tracing():
    def __init__(self, trajectories_input=None, initial_conditions=None, times=None,
                 field=None, model=None, maxtime: float = 1e-7, timesteps: int = 500,
                 tol_step_size = 1e-7, particles=None, condition=None,species=None,tag_gc=1.):
        
        if isinstance(field, Coils):
            self.field = BiotSavart(field)
        else:
            self.field = field
        self.model = model
        self.initial_conditions = initial_conditions
        self.times = times
        self.maxtime = maxtime
        self.timesteps = timesteps
        self.tol_step_size = tol_step_size
        self._trajectories = trajectories_input
        self.particles = particles
        self.species=species
        self.tag_gc=tag_gc
        if condition is None:
            self.condition = lambda t, y, args, **kwargs: False
            if isinstance(field, Vmec):
                if model == 'GuidingCenterCollisionsMu':
                    def condition_Vmec(t, y, args, **kwargs):
                        s, _, _, _ ,_= y
                        return s-1
                else:
                    def condition_Vmec(t, y, args, **kwargs):
                        s, _, _, _ = y
                        return s-1	        
                self.condition = condition_Vmec                
        if model == 'GuidingCenter':
            self.ODE_term = ODETerm(GuidingCenter)
            self.args = (self.field, self.particles)
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz, self.particles.initial_vparallel[:, None]], axis=1)
        elif model == 'GuidingCenterCollisions':
            # Brownian motion
            #t0=0.0
            #t1=self.maxtime
            #tol=self.maxtime / self.timesteps*0.5
            #print('tol: ', tol)
            #bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,), key=jax.random.key(0), levy_area=diffrax.SpaceTimeTimeLevyArea)            
            #self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDrift),ControlTerm(GuidingCenterCollisionsDiffusion, bm))
            self.args = (self.field, self.particles,self.species,self.tag_gc)
            total_speed_temp=self.particles.total_speed*jnp.ones(self.particles.nparticles)
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz,total_speed_temp[:, None], self.particles.initial_vparallel_over_v[:, None]], axis=1)
        elif model == 'GuidingCenterCollisionsMu':
            # Brownian motion
            #t0=0.0
            #t1=self.maxtime
            #tol=self.maxtime / self.timesteps*0.5   
            #print('tol: ', tol)
            #bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,), key=jax.random.key(0), levy_area=diffrax.SpaceTimeTimeLevyArea)
            #self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMu),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))
            self.args = (self.field, self.particles,self.species,self.tag_gc)
            #x,y,z=self.particles.initial_xyz[]
            B_particle=jax.vmap(field.AbsB,in_axes=0)(particles.initial_xyz)
            mu=self.particles.initial_vperpendicular**2*self.particles.mass*0.5/B_particle
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz,self.particles.initial_vparallel[:, None],mu[:, None]],axis=1)        
        elif model == 'FullOrbit' or model == 'FullOrbit_Boris':
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
        elif model == 'FieldLine':
            self.ODE_term = ODETerm(FieldLine)
            self.args = self.field
            
        if self.times is None:
            self.times = jnp.linspace(0, self.maxtime, self.timesteps)
        else:
            self.maxtime = jnp.max(self.times)
            self.timesteps = len(self.times)
            
        self._trajectories = self.trace()
        
        if self.particles is not None:
            self.energy = jnp.zeros((self.particles.nparticles, self.timesteps))
            
        if model == 'GuidingCenter':
            @jit
            def compute_energy_gc(trajectory):
                xyz = trajectory[:, :3]
                vpar = trajectory[:, 3]
                AbsB = vmap(self.field.AbsB)(xyz)
                mu = (self.particles.energy - self.particles.mass * vpar[0]**2 / 2) / AbsB[0]
                return self.particles.mass * vpar**2 / 2 + mu * AbsB
            self.energy = vmap(compute_energy_gc)(self._trajectories)
        elif model == 'GuidingCenterCollisions':
            @jit
            def compute_energy_gc(trajectory):
                return 0.5*self.particles.mass* trajectory[:, 3]**2
            self.energy = vmap(compute_energy_gc)(self._trajectories)
        elif model == 'GuidingCenterCollisionsMu':
            @jit
            def compute_energy_gc(trajectory):
                xyz = trajectory[:, :3]                
                vpar = trajectory[:, 3]
                mu = trajectory[:, 4]
                AbsB = vmap(self.field.AbsB)(xyz)
                return self.particles.mass * vpar**2 / 2 + mu*AbsB
            self.energy = vmap(compute_energy_gc)(self._trajectories)
            @jit
            def compute_vperp_gc(trajectory):
                xyz = trajectory[:, :3]                
                mu = trajectory[:, 4]
                AbsB = vmap(self.field.AbsB)(xyz)
                return jnp.sqrt(2.*mu*AbsB/self.particles.mass)
            self.vperp_final = vmap(compute_vperp_gc)(self._trajectories)     
        elif model == 'FullOrbit' or model == 'FullOrbit_Boris' or model == 'FullOrbitCollisions':
            @jit
            def compute_energy_fo(trajectory):
                vxvyvz = trajectory[:, 3:]
                return self.particles.mass / 2 * (vxvyvz[:, 0]**2 + vxvyvz[:, 1]**2 + vxvyvz[:, 2]**2)
            self.energy = vmap(compute_energy_fo)(self._trajectories)
        elif model == 'FieldLine':
            self.energy = jnp.ones((len(initial_conditions), self.timesteps))
        


        self.trajectories_xyz = vmap(lambda xyz: vmap(lambda point: self.field.to_xyz(point[:3]))(xyz))(self.trajectories)
        
        if isinstance(field, Vmec):
            self.loss_fractions, self.total_particles_lost, self.lost_times = self.loss_fraction()
        else:
            self.loss_fractions = None
            self.total_particles_lost = None
            self.loss_times = None

    @partial(jit, static_argnums=(0))
    def trace(self):
        @jit
        def compute_trajectory(initial_condition, particle_key) -> jnp.ndarray:
            # initial_condition = initial_condition[0]
            if self.model == 'FullOrbit_Boris':
                dt=self.maxtime / self.timesteps
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
                dt0=self.maxtime / self.timesteps
                tol=dt0*0.5
                #print('tol: ', tol)
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,), key=particle_key, levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDrift),ControlTerm(GuidingCenterCollisionsDiffusion, bm))
                #print(self.model)
                #print(self.ODE_term)
                trajectory = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.maxtime / self.timesteps,
                    y0=initial_condition,
                    #solver=diffrax.SlowRK(),
                    solver=diffrax.StratonovichMilstein(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    #stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.tol_step_size, atol=self.tol_step_size),
                    max_steps=10000000000,
                    event = Event(self.condition)
                ).ys
            elif self.model == 'GuidingCenterCollisionsMu':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.maxtime / self.timesteps
                tol=dt0*0.5
                #print('tol: ', tol)
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,),key=particle_key,levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMu),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))                
                #print(self.model)
                #print(self.ODE_term)
                trajectory = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.maxtime / self.timesteps,
                    y0=initial_condition,
                    #solver=diffrax.SPaRK(),
                    solver=diffrax.ItoMilstein(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    #stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.tol_step_size, atol=self.tol_step_size,dtmin=dt0),
                    max_steps=10000000000,
                    event = Event(self.condition)
                ).ys        
            elif self.model == 'FullOrbitCollisions':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.maxtime / self.timesteps
                tol=dt0*0.5
                #print('tol: ', tol)
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(6,), key=particle_key, levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(LorentzCollisionsDrift),ControlTerm(LorentzCollisionsDiffusion,bm))
                #print(self.model)
                #print(self.ODE_term)
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
                    #stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.tol_step_size, atol=self.tol_step_size,dtmin=dt0),
                    max_steps=10000000000,
                    event = Event(self.condition)
                ).ys                            
            else:
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                trajectory = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.maxtime / self.timesteps,
                    y0=initial_condition,
                    solver=diffrax.Tsit5(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.tol_step_size, atol=self.tol_step_size),
                    max_steps=10000000000,
                    event = Event(self.condition)
                ).ys
            return trajectory
        
        return jit(vmap(compute_trajectory,in_axes=(0,0)), in_shardings=(sharding,sharding_index), out_shardings=sharding)(
            device_put(self.initial_conditions, sharding), device_put(self.particles.random_keys if self.particles else None, sharding_index))
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
    
    def _tree_flatten(self):
        children = (self.trajectories,)  # arrays / dynamic values
        aux_data = {'field': self.field, 'model': self.model}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
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
            ax.plot(trajectories_xyz[i, :, 0], trajectories_xyz[i, :, 1], trajectories_xyz[i, :, 2], linewidth=0.5, **kwargs)
        ax.grid(False)
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
            
    @partial(jit, static_argnums=(0))
    def loss_fraction(self, r_max=0.99):
        trajectories_r = jnp.array([traj[:, 0] for traj in self.trajectories])
        lost_mask = trajectories_r >= r_max
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        safe_lost_indices = jnp.where(lost_indices != -1, lost_indices, len(self.times))
        loss_counts = jnp.bincount(safe_lost_indices, length=len(self.times) + 1)[:-1]
        loss_fractions = jnp.cumsum(loss_counts) / len(self.trajectories)
        total_particles_lost = loss_fractions[-1] * len(self.trajectories)
        return loss_fractions, total_particles_lost, lost_times

    def poincare_plot(self, shifts = [jnp.pi/2], orientation = 'toroidal', length = 1, ax=None, show=True, color=None, **kwargs):
        """
        Plot Poincare plots using scipy to find the roots of an interpolation. Can take particle trace or field lines.
        Args:
            shifts (list, optional): Apply a linear shift to dependent data. Default is [0].
            orientation (str, optional): 
                'toroidal' - find time values when toroidal angle = shift [0, 2pi].
                'z' - find time values where z coordinate = shift. Default is 'toroidal'.
            length (float, optional): A way to shorten data. 1 - plot full length, 0.1 - plot 1/10 of data length. Default is 1.
            ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib axis to plot on. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            color: Can be time, None or a color to plot Poincaré points
            **kwargs: Additional keyword arguments for plotting.
        Notes:
            - If the data seem ill-behaved, there may not be enough steps in the trace for a good interpolation.
            - This will break if there are any NaNs.
            - Issues with toroidal interpolation: jnp.arctan2(Y, X) % (2 * jnp.pi) causes distortion in interpolation near phi = 0.
            - Maybe determine a lower limit on resolution needed per toroidal turn for "good" results.
        To-Do:
            - Format colorbars.
        """
        kwargs.setdefault('s', 0.5)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        shifts = jnp.array(shifts)
        plotting_data = []
        # from essos.util import roots_scipy
        for shift in shifts:
            @jit
            def compute_trajectory_toroidal(trace):
                X,Y,Z = trace[:,:3].T
                R = jnp.sqrt(X**2 + Y**2)
                phi = jnp.arctan2(Y,X)
                phi = jnp.where(shift==0, phi, jnp.abs(phi))
                T_slice = roots(self.times, phi, shift = shift)
                T_slice = jnp.where(shift==0, jnp.concatenate((T_slice[1::2],T_slice[1::2])), T_slice)
                # T_slice = roots_scipy(self.times, phi, shift = shift)
                R_slice = jnp.interp(T_slice, self.times, R)
                Z_slice = jnp.interp(T_slice, self.times, Z)
                return R_slice, Z_slice, T_slice
            @jit
            def compute_trajectory_z(trace):
                X,Y,Z = trace[:,:3].T
                T_slice = roots(self.times, Z, shift = shift)
                # T_slice = roots_scipy(self.times, Z, shift = shift)
                X_slice = jnp.interp(T_slice, self.times, X)
                Y_slice = jnp.interp(T_slice, self.times, Y)
                return X_slice, Y_slice, T_slice
            if orientation == 'toroidal':
                # X_slice, Y_slice, T_slice = vmap(compute_trajectory_toroidal)(self.trajectories)
                X_slice, Y_slice, T_slice = jit(vmap(compute_trajectory_toroidal), in_shardings=sharding, out_shardings=sharding)(
                    device_put(self.trajectories, sharding))
            elif orientation == 'z':
                # X_slice, Y_slice, T_slice = vmap(compute_trajectory_z)(self.trajectories)
                X_slice, Y_slice, T_slice = jit(vmap(compute_trajectory_z), in_shardings=sharding, out_shardings=sharding)(
                    device_put(self.trajectories, sharding))
            @partial(jax.vmap, in_axes=(0, 0, 0))
            def process_trajectory(X_i, Y_i, T_i):
                mask = (T_i[1:] != T_i[:-1])
                valid_idx = jnp.nonzero(mask, size=T_i.size - 1)[0] + 1
                return X_i[valid_idx], Y_i[valid_idx], T_i[valid_idx]
            X_s, Y_s, T_s = process_trajectory(X_slice, Y_slice, T_slice)
            length_ = (vmap(len)(X_s) * length).astype(int)
            colors = plt.cm.ocean(jnp.linspace(0, 0.8, len(X_s)))
            for i in range(len(X_s)):
                X_plot, Y_plot = X_s[i][:length_[i]], Y_s[i][:length_[i]]
                T_plot = T_s[i][:length_[i]]
                plotting_data.append((X_plot, Y_plot, T_plot))
                if color == 'time':
                    hits = ax.scatter(X_plot, Y_plot, c=T_s[i][:length_[i]], **kwargs)
                else:
                    if color is None: c=[colors[i]]
                    else: c=color
                    hits = ax.scatter(X_plot, Y_plot, c=c, **kwargs)
                    
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
        
tree_util.register_pytree_node(Tracing,
                               Tracing._tree_flatten,
                               Tracing._tree_unflatten)
