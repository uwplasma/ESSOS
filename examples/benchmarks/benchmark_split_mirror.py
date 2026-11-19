"""Prescribed mirror/profile fields and an exact collisionless axial GC orbit."""
import jax.numpy as jnp
from essos.constants import PROTON_MASS as M


class MagneticMirror:
    """B/B0=(-a*x*z,-a*y*z,1+a*z*z), a=strength/L**2; div B=0."""
    def __init__(self,magnitude,length,strength=1.):
        self.magnitude,self.length,self.strength=magnitude,length,strength
    def B_contravariant(self,point):
        x,y,z=point;a=self.strength/self.length**2
        return self.magnitude*jnp.array([-a*x*z,-a*y*z,1+a*z*z])
    B_covariant=B_contravariant
    def AbsB(self,point):return jnp.linalg.norm(self.B_contravariant(point))
    def sqrtg(self,point):return 1.
    def gc_quantities(self,point):
        x,y,z=point;a=self.strength/self.length**2
        B=self.B_contravariant(point);value=jnp.linalg.norm(B);b=B/value
        jac=self.magnitude*jnp.array([[-a*z,0.,-a*x],[0.,-a*z,-a*y],[0.,0.,2*a*z]])
        gradient=jac.T@b
        db=(jac-b[:,None]*gradient[None,:])/value
        curl=jnp.array([db[2,1]-db[1,2],db[0,2]-db[2,0],db[1,0]-db[0,1]])
        return B,B,value,gradient,curl,db@b,1.
    def dAbsB_by_dX(self,point):return self.gc_quantities(point)[3]
    def curl_b(self,point):return self.gc_quantities(point)[4]
    def kappa(self,point):return self.gc_quantities(point)[5]

def exact_mirror_orbit(y,h,field):
    """Exact collisionless on-axis GC flow, including the mu=0 limit."""
    w=jnp.sqrt(2*y[4]*field.magnitude/(M*field.length**2));phase=w*h
    return y.at[2].set(y[2]*jnp.cos(phase)+y[3]*h*jnp.sinc(phase/jnp.pi)).at[3].set(
        y[3]*jnp.cos(phase)-y[2]*w*jnp.sin(phase))


class TransitField:
    """Uniform B along x so native radial profiles vary along the orbit."""
    def __init__(self,magnitude):self.magnitude=magnitude
    def AbsB(self,x):return self.magnitude
    def gc_quantities(self,x):
        b=jnp.array([self.magnitude,0.,0.]);z=jnp.zeros(3)
        return b,b,self.magnitude,z,z,z,1.

def background(physical,*,varying=False):
    from essos.background_species import BackgroundSpecies
    return BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),
        jnp.array([[.2e20,1.8e20]]) if varying else jnp.array([1e20]),
        jnp.array([[500.,1500.]]) if varying else jnp.array([1000.]),
        radial_grid=jnp.array([-.1,.1])*physical.sigma/physical.nu if varying else None,
        coulomb_log_model='impact')
