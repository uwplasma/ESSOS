import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import jit, vmap, tree_util, random, lax, device_put
from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, Event, TqdmProgressMeter
from diffrax import ControlTerm,UnsafeBrownianPath,MultiTerm,ItoMilstein,ClipStepSizeController #For collisions we need this to solve stochastic differential equation
import diffrax
from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.surfaces import SurfaceClassifier
from essos.electric_field import Electric_field_flux, Electric_field_zero
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ELEMENTARY_CHARGE,SPEED_OF_LIGHT
from essos.plot import fix_matplotlib_3d
from essos.util import roots
from essos.background_species import nu_s_ab,nu_D_ab,nu_par_ab, d_nu_par_ab,d_nu_D_ab

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



## !!!!  Here species and tag_gc were added  (E. Neto collisions modifications)
## species is a class for collision frquencies + possible temperature + density profiles in file species_background.py
## tag_gc is a tag to turn off 0, or on 1 the GC part of the equations for testing collision statistics independently of GC phsyics
## !!!!  Here particle_key was added to compute_trajectories (E. Neto collisions modifications)
## This is important for correct sampling of Brownian motion
class Tracing():
    def __init__(self, trajectories_input=None, initial_conditions=None, times_to_trace=None,
                 field=None, electric_field=None,model=None, maxtime: float = 1e-7, timestep: int = 1.e-8,
                 rtol= 1.e-7, atol = 1e-7, particles=None, condition=None,species=None,tag_gc=1.,boundary=None,rejected_steps=None):
        
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
        if condition is None:
            self.condition = lambda t, y, args, **kwargs: False
            if isinstance(field, Vmec):
                if model == 'GuidingCenterCollisionsMuIto' or model == 'GuidingCenterCollisionsMuFixed' or model == 'GuidingCenterCollisionsMuAdaptative'  or model=='GuidingCenterCollisions':
                    def condition_Vmec(t, y, args, **kwargs):
                        s, _, _, _ ,_= y
                        return s-1
                elif model == 'FieldLine' or model== 'FieldLineAdaptative':
                    def condition_Vmec(t, y, args, **kwargs):
                        s, _, _ = y
                        return s-1	 
                else:
                    def condition_Vmec(t, y, args, **kwargs):
                        s, _, _, _ = y
                        return s-1	        
                self.condition = condition_Vmec
            elif isinstance(field,BiotSavart) and isinstance(boundary,SurfaceClassifier):
                if model == 'GuidingCenterCollisionsMuIto' or model == 'GuidingCenterCollisionsMuFixed' or model == 'GuidingCenterCollisionsMuAdaptative' or model=='GuidingCenterCollisions':
                    def condition_BioSavart(t, y, args, **kwargs):
                        xx, yy, zz, _,_ = y
                        return boundary.evaluate_xyz(jnp.array([xx,yy,zz]))#<0.                      
                else:
                    def condition_BioSavart(t, y, args, **kwargs):                      
                        xx, yy, zz, _ = y
                        return boundary.evaluate_xyz(jnp.array([xx,yy,zz]))#<0.        
                self.condition = condition_BioSavart                
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
        elif model == 'FieldLine' or model== 'FieldLineAdaptative':
            self.ODE_term = ODETerm(FieldLine)
            self.args = self.field
        
        if self.times_to_trace is None:
            self.times = jnp.linspace(0, self.maxtime, 100,endpoint=True)
        else:
            self.times = jnp.linspace(0, self.maxtime, self.times_to_trace,endpoint=True)

            
        self._trajectories = self.trace()
        
        if self.particles is not None:
            self.energy = jnp.zeros((self.particles.nparticles, self.times_to_trace))
            
        if model == 'GuidingCenter' or  model == 'GuidingCenterAdaptative' :
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
        elif model == 'GuidingCenterCollisionsMuIto' or model == 'GuidingCenterCollisionsMuFixed' or model == 'GuidingCenterCollisionsMuAdaptative' :
            @jit
            def compute_energy_gc(trajectory):
                xyz = trajectory[:, :3]                
                vpar = trajectory[:, 3]*SPEED_OF_LIGHT
                mu = trajectory[:, 4]*self.particles.mass*SPEED_OF_LIGHT**2
                AbsB = vmap(self.field.AbsB)(xyz)
                return self.particles.mass * vpar**2 / 2 + mu*AbsB
            self.energy = vmap(compute_energy_gc)(self._trajectories)
            @jit
            def compute_vperp_gc(trajectory):
                xyz = trajectory[:, :3]                
                mu = trajectory[:, 4]*self.particles.mass*SPEED_OF_LIGHT**2
                AbsB = vmap(self.field.AbsB)(xyz)
                return jnp.sqrt(2.*mu*AbsB/self.particles.mass)
            self.vperp_final = vmap(compute_vperp_gc)(self._trajectories)     
        elif model == 'FullOrbit' or model == 'FullOrbit_Boris' or model == 'FullOrbitCollisions':
            @jit
            def compute_energy_fo(trajectory):
                vxvyvz = trajectory[:, 3:]
                return self.particles.mass / 2 * (vxvyvz[:, 0]**2 + vxvyvz[:, 1]**2 + vxvyvz[:, 2]**2)
            self.energy = vmap(compute_energy_fo)(self._trajectories)
        elif model == 'FieldLine' or model== 'FieldLineAdaptative':
            self.energy = jnp.ones((len(initial_conditions), self.times_to_trace))
        


        self.trajectories_xyz = vmap(lambda xyz: vmap(lambda point: self.field.to_xyz(point[:3]))(xyz))(self.trajectories)
        
        if isinstance(field, Vmec):
            if self.model == 'GuidingCenterCollisions' or model == 'GuidingCenterCollisionsMuIto' or self.model == 'GuidingCenterCollisionsMuFixed' or self.model == 'GuidingCenterCollisionsAdaptative':
                self.loss_fractions, self.total_particles_lost, self.lost_times,self.lost_energies,self.lost_positions = self.loss_fraction_collisions()                    
            else:                
                self.loss_fractions, self.total_particles_lost, self.lost_times = self.loss_fraction()
        elif (isinstance(field, Coils) or isinstance(self.field, BiotSavart)) and isinstance(boundary,SurfaceClassifier):
            if self.model == 'GuidingCenterCollisions' or model == 'GuidingCenterCollisionsMuIto' or self.model == 'GuidingCenterCollisionsMuFixed' or self.model == 'GuidingCenterCollisionsAdaptative':
                self.loss_fractions, self.total_particles_lost, self.lost_times,self.lost_energies,self.lost_positions = self.loss_fraction_BioSavart_collisions(boundary)                    
            else:                
                self.loss_fractions, self.total_particles_lost, self.lost_times = self.loss_fraction_BioSavart(boundary)
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
                trajectory = diffeqsolve(
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
                    progress_meter=TqdmProgressMeter(),                    
                ).ys
            elif self.model == 'GuidingCenterCollisionsMuAdaptative':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.timestep#self.maxtime / self.timesteps
                tol=dt0*0.5
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,),key=particle_key,levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMuStratonovich),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))                
                trajectory = diffeqsolve(
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
                    progress_meter=TqdmProgressMeter(),
                ).ys     
            elif self.model == 'GuidingCenterCollisionsMuFixed':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.timestep#self.maxtime / self.timesteps
                tol=dt0*0.5
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,),key=particle_key,levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMuStratonovich),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))                
                trajectory = diffeqsolve(
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
                    progress_meter=TqdmProgressMeter(),
                ).ys       
            elif self.model == 'GuidingCenterCollisionsMuIto':
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                t0=0.0
                t1=self.maxtime
                dt0=self.timestep#self.maxtime / self.timesteps
                tol=dt0*0.5
                bm = diffrax.VirtualBrownianTree(t0, t1, tol=tol, shape=(5,),key=particle_key,levy_area=diffrax.SpaceTimeTimeLevyArea)            
                self.ODE_term = MultiTerm(ODETerm(GuidingCenterCollisionsDriftMuIto),ControlTerm(GuidingCenterCollisionsDiffusionMu, bm))                
                trajectory = diffeqsolve(
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
                    progress_meter=TqdmProgressMeter(),
                ).ys                                       
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
                    progress_meter=TqdmProgressMeter()                   
                ).ys          
            elif self.model == 'GuidingCenterAdaptative' :  
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                trajectory = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.timestep,#self.maxtime / self.timesteps,
                    y0=initial_condition,
                    solver=diffrax.Tsit5(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    progress_meter=TqdmProgressMeter(),
                    stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol),
                    max_steps=10000000000,
                    event = Event(self.condition)
                ).ys
            elif self.model == 'FieldLineAdaptative' :  
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                trajectory = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.timestep,#self.maxtime / self.timesteps,
                    y0=initial_condition,
                    solver=diffrax.Tsit5(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    progress_meter=TqdmProgressMeter(),
                    stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.rtol, atol=self.atol),
                    max_steps=10000000000,
                    event = Event(self.condition)
                ).ys                
            #Fixed guiding center
            else:
                import warnings
                warnings.simplefilter("ignore", category=FutureWarning) # see https://github.com/patrick-kidger/diffrax/issues/445 for explanation
                trajectory = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=self.timestep,#self.maxtime / self.timesteps,
                    y0=initial_condition,
                    solver=diffrax.Tsit5(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=False,
                    # adjoint=DirectAdjoint(),
                    progress_meter=TqdmProgressMeter(),
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
            
    @partial(jit, static_argnums=(0,1))
    def loss_fraction_BioSavart(self,boundary):
        trajectories_xyz = self.trajectories[:,:, :3]
        lost_mask = jnp.transpose(vmap(vmap(boundary.evaluate_xyz,in_axes=(0)),in_axes=(1))(trajectories_xyz)) <0
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        safe_lost_indices = jnp.where(lost_indices != -1, lost_indices, len(self.times))
        loss_counts = jnp.bincount(safe_lost_indices, length=len(self.times) + 1)[:-1]
        loss_fractions = jnp.cumsum(loss_counts) / len(self.trajectories)
        total_particles_lost = loss_fractions[-1] * len(self.trajectories)
        return loss_fractions, total_particles_lost, lost_times

    @partial(jit, static_argnums=(0))
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
    def loss_fraction_BioSavart_collisions(self,boundary):
        trajectories_xyz = self.trajectories[:,:, :3]
        lost_mask = jnp.transpose(vmap(vmap(boundary.evaluate_xyz,in_axes=(0)),in_axes=(1))(trajectories_xyz)) <0
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        lost_energies=vmap(lambda x: jnp.where(lost_indices[x-1] != -1, self.energy[x-1,lost_indices[x-1]-1], 0.))(jnp.arange(self.particles.nparticles))
        lost_positions=vmap(lambda x: jnp.where(lost_indices[x-1] != -1, trajectories_xyz[x-1,lost_indices[x-1]-1,:], 0.))(jnp.arange(self.particles.nparticles))                          
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
        lost_energies=vmap(lambda x: jnp.where(lost_indices[x-1] != -1, self.energy[x-1,lost_indices[x-1]-1], 0.))(jnp.arange(self.particles.nparticles))
        lost_positions=vmap(lambda x: jnp.where(lost_indices[x-1] != -1, trajectories_rtz[x-1,lost_indices[x-1]-1,:], 0.))(jnp.arange(self.particles.nparticles))            
        safe_lost_indices = jnp.where(lost_indices != -1, lost_indices, len(self.times))
        loss_counts = jnp.bincount(safe_lost_indices, length=len(self.times) + 1)[:-1]
        loss_fractions = jnp.cumsum(loss_counts) / len(self.trajectories)
        total_particles_lost = loss_fractions[-1] * len(self.trajectories)
        return loss_fractions, total_particles_lost, lost_times,lost_energies,lost_positions
    
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
