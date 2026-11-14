import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from jax import jit, jacfwd, grad, vmap, tree_util, lax


class ElectricFieldFlux():
    """Radial electric field read from an HDF5 file and mapped to VMEC ``s``.

    The file needs datasets ``rho`` (normalized minor radius) and ``Er``.
    ``E_covariant`` returns ``(E_s, 0, 0)``, with ``E_s = Er / (2 rho) * a``
    and ``a`` the VMEC ``Aminor_p``; ``E_s`` is set to zero on the axis.
    """
    def __init__(self, Er_filename,vmec):
        self.Er_filename = Er_filename
        from h5py import File
        self.h5 = File(self.Er_filename,'r')
        self.rho=jnp.array(self.h5['rho'][()])
        self.Er=jnp.array(self.h5['Er'][()])
        self.E_theta=0.0*self.Er
        self.E_phi=0.0*self.Er
        self.Aminor_p=vmec.Aminor_p
        self.Es=self.Er*0.5/self.rho*self.Aminor_p
        #Setting E_s(0) = 0, because the transformation is singular
        self.Es=self.Es.at[0].set(0.0)
        
    @partial(jit, static_argnames=['self'])
    def E_covariant(self, points):
        s, theta, phi = points
        #rho**2 here comes from going from E_r to E_s used by VMEC, variable drds is not stored to save memory
        Er_interp=jnp.interp(s, self.rho**2, self.Es, left='extrapolate')

        return jnp.array([Er_interp,0.0,0.0])
    
class ElectricFieldZero():
    """No electric field; ``E_covariant`` returns zeros."""
    def __init__(self):
        self.Er_filename = None

    # Stateless: every instance equals every other, so jit calls with it as a static argument share one compilation.
    def __eq__(self, other):
        return type(other) is ElectricFieldZero

    def __hash__(self):
        return hash(ElectricFieldZero)
        
    @partial(jit, static_argnames=['self'])
    def E_covariant(self, points):
        return jnp.array([0.0,0.0,0.0])    
    
