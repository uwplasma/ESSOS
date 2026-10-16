from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from jax import tree_util, jit, vmap, devices, device_put
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental.pjit import pjit
from essos.plot import fix_matplotlib_3d
import jaxkd

# mesh = Mesh(devices(), ("dev",))
# sharding = NamedSharding(mesh, PartitionSpec("dev"))


# Disable sharding for B_on_surface to avoid issues with frozen DOFs
# When DOFs are frozen, array sizes may not divide evenly across devices
USE_SHARDING = False  # Set to True to enable multi-device parallelization

if USE_SHARDING:
    _devices = devices()
    if len(_devices) > 1:
        mesh = Mesh(_devices, ("dev",))
        sharding = NamedSharding(mesh, PartitionSpec("dev"))
    else:
        mesh = None
        sharding = None
else:
    mesh = None
    sharding = None


@jit
def toroidal_flux(surface, field, idx=0) -> jnp.ndarray:
    curve = surface.gamma[idx]    
    dl = jnp.roll(curve, -1, axis=0) - curve
    A_vals = vmap(field.A)(curve)
    Adl = jnp.sum(A_vals * dl, axis=1) 
    tf = jnp.sum(Adl)
    #curve = surface.gamma[idx]    
    #dl = surface.gammadash_theta[idx]
    #A_vals = vmap(field.A)(curve)
    #Adl = jnp.sum(A_vals * dl, axis=1)/surface.ntheta 
    #tf = jnp.sum(Adl)    
    return tf

@jit
def poloidal_flux(surface, field, idx=0) -> jnp.ndarray:
    curve = surface.gamma[:,idx,:]    
    dl = jnp.roll(curve, -1, axis=0) - curve
    A_vals = vmap(field.A)(curve)
    Adl = jnp.sum(A_vals * dl, axis=1) 
    tf = jnp.sum(Adl)
    #curve = surface.gamma[:,idx,:]    
    #dl = surface.gammadash_phi[:,idx,:]
    #A_vals = vmap(field.A)(curve)
    #Adl = jnp.sum(A_vals * dl, axis=1)/surface.nphi 
    #tf = jnp.sum(Adl)    
    return tf

# Use pjit with sharding only when sharding is enabled
# Otherwise use regular jit to avoid pytree caching issues with variable structure
# (GFrame surfaces have variable pytree: 2 children without axis DOFs, 4 with axis DOFs)
if USE_SHARDING and sharding is not None:
    @partial(pjit, in_shardings=(sharding, None), out_shardings=sharding)
    def B_on_surface(surface, field):
        ntheta = surface.ntheta
        nphi = surface.nphi
        gamma = surface.gamma
        gamma_reshaped = gamma.reshape(nphi * ntheta, 3)

        # Map field.B over all positions
        B_on_surface_vals = vmap(field.B)(gamma_reshaped)

        return B_on_surface_vals.reshape(nphi, ntheta, 3)
else:
    @jit
    def B_on_surface(surface, field):
        ntheta = surface.ntheta
        nphi = surface.nphi
        gamma = surface.gamma
        gamma_reshaped = gamma.reshape(nphi * ntheta, 3)

        # Map field.B over all positions
        B_on_surface_vals = vmap(field.B)(gamma_reshaped)

        return B_on_surface_vals.reshape(nphi, ntheta, 3)
    

@jit
def BdotN(surface, field):
    B_surface = B_on_surface(surface, field)
    B_dot_n = jnp.sum(B_surface * surface.unitnormal, axis=2)
    return B_dot_n

@jit
def BdotN_over_B(surface, field, **kwargs):
    return BdotN(surface, field) / jnp.linalg.norm(B_on_surface(surface, field), axis=2)

@jit
def _squared_flux_local(surface, field):
    return 0.5 * jnp.mean(BdotN(surface, field)**2 / jnp.sum(B_on_surface(surface, field)**2, axis=2)
                          * surface.area_element)

@jit
def _squared_flux_global(surface, field):
    return 0.5 * jnp.mean(BdotN(surface, field)**2 * surface.area_element)

@jit
def _squared_flux_normalized(surface, field):
    return 0.5 * jnp.mean(BdotN(surface, field)**2 * surface.area_element) / \
                 jnp.mean(jnp.sum(B_on_surface(surface, field)**2, axis=2) * surface.area_element)

def SquaredFlux(surface, field, definition='local'):
    if definition == 'local':
        return _squared_flux_local(surface, field)
    elif definition == 'quadratic flux':
        return _squared_flux_global(surface, field)
    elif definition == 'normalized':
        return _squared_flux_normalized(surface, field)
    else:
        raise ValueError(f"Unknown definition: {definition}")

def nested_lists_to_array(ll):
    """
    Convert a ragged list of lists to a 2D jnp array.  Any entries
    that are None are replaced by 0. This routine is useful for
    parsing fortran namelists that include 2D arrays using f90nml.

    Args:
        ll: A list of lists to convert.
    """
    mdim = len(ll)
    ndim = max(len(x) for x in ll)
    arr = jnp.zeros((mdim, ndim))
    for jm, l in enumerate(ll):
        arr = arr.at[jm, :len(l)].set(jnp.array([x if x is not None else 0 for x in l]))
    return arr

    

class SurfaceRZFourier:
    def __init__(self, rc, zs, nfp, mpol, ntor, ntheta=30, nphi=30, close=True, range_torus='full torus',
                 scaling_type=2, scaling_factor=0, scale_fixed=1.0):
        """ rc, zs: dynamic arrays 
            nfp, mpol, ntor: static 
            scale_fixed: fixed multiplier applied to all modes (default=1.0, use >1.0 for equilibration)"""
        
        assert isinstance(nfp, int) and nfp > 0, "nfp must be a positive integer."
        assert isinstance(mpol, int) and mpol >= 0, "mpol must be a non-negative integer."
        assert isinstance(ntor, int) and ntor >= 0, "ntor must be a non-negative integer."
        assert isinstance(ntheta, int) and ntheta > 0, "ntheta must be a positive integer."
        assert isinstance(nphi, int) and nphi > 0, "nphi must be a positive integer."
        assert isinstance(close, bool), "close must be a boolean."
        assert range_torus in ['full torus', 'half period','full period'], f"Unknown range_torus: {range_torus}. Choose 'full torus' or 'half period'."

        self._rc = rc
        self._zs = zs
        self._nfp = nfp
        self._mpol = mpol
        self._ntor = ntor

        self._gamma = None
        self._gammadash_theta = None
        self._gammadash_phi = None
        self._gammadashdash_theta = None
        self._gammadashdash_phi = None
        self._gammadashdash_theta_phi = None
        self._normal = None
        self._unitnormal = None
        self._area_element = None
        self._xm = None
        self._xn = None

        self._ntheta = ntheta
        self._nphi = nphi
        self._close = close
        self._range_torus = range_torus
        
        self._quadpoints_theta = None
        self._quadpoints_phi = None
        self._theta2d = None
        self._phi2d = None
        self._angles = None

        self._scaling_type = scaling_type # 1 for L-1 norm, 2 for L-2 norm, jnp.inf for L-infinity norm
        self._scaling_factor = scaling_factor
        self._scale_fixed = scale_fixed
        self._scaling = None


    @classmethod
    def from_input_file(cls, file, ntheta=30, nphi=30, close=True, range_torus='full torus',scaling_type=2,scaling_factor=0.0,scale_fixed=1.0):
        from f90nml import Parser
        all_namelists = Parser().read(file)
        nml = all_namelists['indata']        
        #nml = Parser().read(file)['indata']

        nfp = nml["nfp"] if "nfp" in nml else 1
        mpol = nml['mpol']            
        ntor = nml['ntor']
        
        rc = jnp.ravel(nested_lists_to_array(nml['rbc']))[2:]
        zs = jnp.ravel(nested_lists_to_array(nml['zbs']))[2:]

        surface = cls(rc, zs, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus,scaling_factor=scaling_factor,scaling_type=scaling_type,scale_fixed=scale_fixed)
        return surface
    
    @classmethod
    def from_vmec(cls, vmec, s=1, ntheta=30, nphi=30, close=True, range_torus='full torus',scaling_type=2,scaling_factor=0.0,scale_fixed=1.0):
        nfp = vmec.nfp
        mpol = vmec.mpol
        ntor = vmec.ntor

        s_full_grid = vmec.s_full_grid
        rc = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(vmec.rmnc)
        zs = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(vmec.zmns)

        surface = cls(rc, zs, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus,scaling_factor=scaling_factor,scaling_type=scaling_type,scale_fixed=scale_fixed)
        surface._xm = vmec.xm
        surface._xn = vmec.xn

        return surface

    @classmethod
    def from_wout_file(cls, file, s=1, ntheta=30, nphi=30, close=True, range_torus='full torus',scaling_type=2,scaling_factor=0.0,scale_fixed=1.0):
        from netCDF4 import Dataset
        nc = Dataset(file)

        nfp = int(nc.variables["nfp"][0])
        xm = jnp.array(nc.variables["xm"][:])
        xn = jnp.array(nc.variables["xn"][:])
        mpol = int(jnp.max(xm))
        ntor = int(jnp.max(jnp.abs(xn)) / nfp)
        
        ns = nc.variables["ns"][0]
        s_full_grid = jnp.linspace(0, 1, ns)
        rc = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(jnp.array(nc.variables["rmnc"][:]))
        zs = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(jnp.array(nc.variables["zmns"][:]))

        surface = cls(rc, zs, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus,scaling_factor=scaling_factor,scaling_type=scaling_type,scale_fixed=scale_fixed)
        surface._xm = xm
        surface._xn = xn

        return surface

    # reset_cache method
    def reset_cache(self):
        self._gamma = None
        self._gammadash_theta = None
        self._gammadash_phi = None
        self._normal = None
        self._unitnormal = None
        self._area_element = None
        self._xm = None
        self._xn = None
        self._angles = None
    
    # reset_mesh method
    def reset_mesh(self):
        self._quadpoints_theta = None
        self._quadpoints_phi = None
        self._theta2d = None
        self._phi2d = None
        self._angles = None

    # rc property and setter
    @property
    def rc(self):
        return self._rc
    
    @rc.setter
    def rc(self, new_rc):
        self._rc = new_rc
        self.reset_cache()

    # zs property and setter
    @property
    def zs(self):
        return self._zs

    @zs.setter
    def zs(self, new_zs):
        self._zs = new_zs
        self.reset_cache()

    # nfp property
    @property
    def nfp(self):
        return self._nfp
    
    # mpol property
    @property
    def mpol(self):
        return self._mpol
    
    # ntor property
    @property
    def ntor(self):
        return self._ntor

    # xm property
    @property
    def xm(self):
        if self._xm is None:
            self._xm = jnp.repeat(jnp.arange(self.mpol + 1), 2 * self.ntor + 1)[self.ntor:]
        return self._xm

    # xn property
    @property
    def xn(self):
        if self._xn is None:
            self._xn = self.nfp * jnp.tile(jnp.arange(-self.ntor, self.ntor + 1), self.mpol + 1)[self.ntor:]
        return self._xn

    # _ntheta property and setter
    @property
    def ntheta(self):
        return self._ntheta

    @ntheta.setter
    def ntheta(self, new_ntheta):
        self._ntheta = new_ntheta
        self.reset_mesh()

    # n_phi property and setter
    @property
    def nphi(self):
        return self._nphi

    @nphi.setter
    def nphi(self, new_nphi):
        self._nphi = new_nphi
        self.reset_mesh()

    # close property and setter
    @property
    def close(self):
        return self._close

    @close.setter
    def close(self, new_close):
        self._close = new_close
        self.reset_mesh()

    # range_torus property and setter
    @property
    def range_torus(self):
        return self._range_torus

    @range_torus.setter
    def range_torus(self, new_range):
        self._range_torus = new_range
        self.reset_mesh()

    # _compute_meshgrid method
    @jit
    def _compute_meshgrid(self):
        if self.range_torus == "full torus":
            div, end_val = 1., 1.
        elif self.range_torus == "half period":
            div, end_val = self.nfp, 0.5
        elif self.range_torus == "full period":
            div, end_val = self.nfp, 1.0            
        quadpoints_theta = jnp.linspace(0, 2 * jnp.pi, num=self.ntheta, endpoint=self._close)
        quadpoints_phi   = jnp.linspace(0, 2 * jnp.pi * end_val / div, num=self.nphi, endpoint=self._close)
        theta2d, phi2d = jnp.meshgrid(quadpoints_theta, quadpoints_phi)
        return quadpoints_theta, quadpoints_phi, theta2d, phi2d

    # theta2d property
    @property
    def theta2d(self):
        if self._theta2d is None:
            self._quadpoints_theta, self._quadpoints_phi, self._theta2d, self._phi2d = self._compute_meshgrid()
        return self._theta2d

    # phi2d property
    @property
    def phi2d(self):
        if self._phi2d is None:
            self._quadpoints_theta, self._quadpoints_phi, self._theta2d, self._phi2d = self._compute_meshgrid()
        return self._phi2d

    # angles property
    @property
    def angles(self):
        if self._angles is None:
            self._angles =  jnp.einsum('i,jk->ijk', self.xm, self.theta2d) - jnp.einsum('i,jk->ijk', self.xn, self.phi2d)
        return self._angles
    
    # scaling_type property and setter
    @property
    def scaling_type(self):
        return self._scaling_type
    
    @scaling_type.setter
    def scaling_type(self, new_type):
        self._scaling_type = new_type
        self._scaling = None

    # scaling_factor property and setter
    @property
    def scaling_factor(self):
        return self._scaling_factor
    
    @scaling_factor.setter
    def scaling_factor(self, new_factor):
        self._scaling_factor = new_factor
        self._scaling = None

    # scale_fixed property and setter
    @property
    def scale_fixed(self):
        return self._scale_fixed
    
    @scale_fixed.setter
    def scale_fixed(self, new_scale):
        self._scale_fixed = new_scale
        self._scaling = None

    # scaling property
    @property
    def scaling(self):
        if self._scaling is None:
            mode_scaling = jnp.exp(self.scaling_factor * jnp.linalg.norm(jnp.vstack([self.xm, self.xn]), ord=self.scaling_type, axis=0))
            self._scaling = mode_scaling * self.scale_fixed
        return self._scaling
    
    # dofs property and setter
    @property
    def dofs(self):
        return jnp.hstack([self.rc * self.scaling, self.zs * self.scaling])
    
    @dofs.setter
    def dofs(self, new_dofs):
        self._rc = new_dofs[:self.rc.size] / self.scaling
        self._zs = new_dofs[self.rc.size:] / self.scaling
        self.reset_cache()
        
    # _compute_gamma method
    @jit
    def _compute_gamma(self):
        angles = self.angles
        sin_angles = jnp.sin(angles)
        cos_angles = jnp.cos(angles)
        phi2d = self.phi2d
        sin_phi2d = jnp.sin(phi2d)
        cos_phi2d = jnp.cos(phi2d)
        rc = self.rc; zs = self.zs; xm = self.xm; xn = self.xn
        #rmnc_interp = self.rc; zmns_interp = self.zs; xm = self.xm; xn = self.xn
        
        R = jnp.einsum('i,ijk->jk', rc, cos_angles)
        Z = jnp.einsum('i,ijk->jk', zs, sin_angles)
        X = R * cos_phi2d
        Y = R * sin_phi2d
        gamma = jnp.stack([X, Y, Z], axis=-1)

        dR_dtheta = -jnp.einsum('i,ijk->jk', xm * rc, sin_angles)
        dZ_dtheta = jnp.einsum('i,ijk->jk', xm * zs, cos_angles)
        dX_dtheta = dR_dtheta * cos_phi2d
        dY_dtheta = dR_dtheta * sin_phi2d
        gammadash_theta = 2.*jnp.pi*jnp.stack([dX_dtheta, dY_dtheta, dZ_dtheta], axis=-1)
        dR_d2theta = -jnp.einsum('i,ijk->jk', xm**2 * rc, cos_angles)
        dZ_d2theta = -jnp.einsum('i,ijk->jk', xm**2 * zs, sin_angles)
        dX_d2theta = dR_d2theta * cos_phi2d
        dY_d2theta = dR_d2theta * sin_phi2d
        gammadashdash_theta = (2.*jnp.pi)**2*jnp.stack([dX_d2theta, dY_d2theta, dZ_d2theta], axis=-1)
        dR_dthetadphi = jnp.einsum('i,ijk->jk', xn*xm * rc, cos_angles)
        dZ_dthetadphi = jnp.einsum('i,ijk->jk', xn*xm * zs, sin_angles)
        dX_dthetadphi = dR_dthetadphi * cos_phi2d - dR_dtheta * sin_phi2d
        dY_dthetadphi = dR_dthetadphi * sin_phi2d + dR_dtheta * cos_phi2d
        gammadashdash_theta_phi = (2.*jnp.pi)**2*jnp.stack([dX_dthetadphi, dY_dthetadphi, dZ_dthetadphi], axis=-1)


        dR_dphi = jnp.einsum('i,ijk->jk', xn*rc, sin_angles)
        dZ_dphi = -jnp.einsum('i,ijk->jk', xn*zs, cos_angles)
        dX_dphi = dR_dphi * cos_phi2d - R * sin_phi2d
        dY_dphi = dR_dphi * sin_phi2d + R * cos_phi2d
        gammadash_phi = 2.*jnp.pi*jnp.stack([dX_dphi, dY_dphi, dZ_dphi], axis=-1)
        dR_d2phi = jnp.einsum('i,ijk->jk', xn**2*rc, cos_angles)
        dZ_d2phi = -jnp.einsum('i,ijk->jk', xn**2*zs, sin_angles)
        dX_d2phi = dR_d2phi * cos_phi2d - 2 * dR_dphi * sin_phi2d - R * cos_phi2d
        dY_d2phi = dR_d2phi * sin_phi2d + 2 * dR_dphi * cos_phi2d - R * sin_phi2d
        gammadashdash_phi = (2.*jnp.pi)**2*jnp.stack([dX_d2phi, dY_d2phi, dZ_d2phi], axis=-1)  


        # r_coordinate = jnp.einsum('i,ijk->jk', rmnc_interp, cos_angles)
        # z_coordinate = jnp.einsum('i,ijk->jk', zmns_interp, sin_angles)
        # gamma = jnp.transpose(jnp.array([r_coordinate * jnp.cos(phi2d), r_coordinate * jnp.sin(phi2d), z_coordinate]), (1, 2, 0))

        # dX_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, rmnc_interp) * jnp.cos(phi2d)
        # dY_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, rmnc_interp) * jnp.sin(phi2d)
        # dZ_dtheta = jnp.einsum('i,ijk,i->jk',  self.xm, cos_angles, zmns_interp)
        # gammadash_theta = 2*jnp.pi*jnp.transpose(jnp.array([dX_dtheta, dY_dtheta, dZ_dtheta]), (1, 2, 0))

        # dX_dphi = jnp.einsum('i,ijk,i->jk',  self.xn, sin_angles, rmnc_interp) * jnp.cos(phi2d) - r_coordinate * jnp.sin(phi2d)
        # dY_dphi = jnp.einsum('i,ijk,i->jk',  self.xn, sin_angles, rmnc_interp) * jnp.sin(phi2d) + r_coordinate * jnp.cos(phi2d)
        # dZ_dphi = jnp.einsum('i,ijk,i->jk', -self.xn, cos_angles, zmns_interp)
        # gammadash_phi = 2*jnp.pi*jnp.transpose(jnp.array([dX_dphi, dY_dphi, dZ_dphi]), (1, 2, 0))        
        
        return gamma, gammadash_theta, gammadash_phi, gammadashdash_theta, gammadashdash_phi, gammadashdash_theta_phi
    
    # gamma, gammadash_theta, gammadash_phi, gammadashdash_theta properties
    @property
    def gamma(self):
        if self._gamma is None:
            self._gamma, self._gammadash_theta, self._gammadash_phi, self._gammadashdash_theta, self._gammadashdash_phi, self._gammadashdash_theta_phi = self._compute_gamma()
        return self._gamma
    
    @property
    def gammadash_theta(self):
        if self._gammadash_theta is None:
            self._gamma, self._gammadash_theta, self._gammadash_phi, self._gammadashdash_theta, self._gammadashdash_phi, self._gammadashdash_theta_phi = self._compute_gamma()
        return self._gammadash_theta
    
    @property
    def gammadash_phi(self):
        if self._gammadash_phi is None:
            self._gamma, self._gammadash_theta, self._gammadash_phi, self._gammadashdash_theta, self._gammadashdash_phi, self._gammadashdash_theta_phi = self._compute_gamma()
        return self._gammadash_phi

    @property
    def gammadashdash_theta(self):
        if self._gammadashdash_theta is None:
            self._gamma, self._gammadash_theta, self._gammadash_phi, self._gammadashdash_theta, self._gammadashdash_phi, self._gammadashdash_theta_phi = self._compute_gamma()
        return self._gammadashdash_theta

    @property
    def gammadashdash_phi(self):
        if self._gammadashdash_phi is None:
            self._gamma, self._gammadash_theta, self._gammadash_phi, self._gammadashdash_theta, self._gammadashdash_phi, self._gammadashdash_theta_phi = self._compute_gamma()
        return self._gammadashdash_phi        

    @property
    def gammadashdash_theta_phi(self):
        if self._gammadashdash_theta_phi is None:
            self._gamma, self._gammadash_theta, self._gammadash_phi, self._gammadashdash_theta, self._gammadashdash_phi, self._gammadashdash_theta_phi = self._compute_gamma()
        return self._gammadashdash_theta_phi


        

    # _compute_properties method
    @jit
    def _compute_properties(self):
        normal = jnp.cross(self.gammadash_phi, self.gammadash_theta, axis=2)
        unitnormal = normal / jnp.linalg.norm(normal, axis=2, keepdims=True)
        area_element = jnp.linalg.norm(normal, axis=2)
        return normal, unitnormal, area_element
    
    # normal, unitnormal, area_element properties
    @property
    def normal(self):
        if self._normal is None:
            self._normal, self._unitnormal, self._area_element = self._compute_properties()
        return self._normal
    
    @property
    def unitnormal(self):
        if self._unitnormal is None:
            self._normal, self._unitnormal, self._area_element = self._compute_properties()
        return self._unitnormal
    
    @property
    def area_element(self):
        if self._area_element is None:
            self._normal, self._unitnormal, self._area_element = self._compute_properties()
        return self._area_element

    # TODO: remove x property. This is a placeholder for compatibility with the examples that need to be updated.
    # x property and setter 
    @property
    def x(self):
        return self.dofs

    @x.setter
    def x(self, new_dofs):
        self.dofs = new_dofs

    @property
    def volume(self):

        xyz = self.gamma  # shape: (nphi, ntheta, 3)
        n = self.normal    # shape: (nphi, ntheta, 3)

        integrand = jnp.sum(xyz * n, axis=2)  # dot(x, n), shape: (nphi, ntheta)
        volume = jnp.sum(integrand) / (3.0*self.nphi*self.ntheta)
        return volume



    @property
    def area(self):
        #n = self.normal  # (nphi, ntheta, 3)
        #norm_n = jnp.linalg.norm(n, axis=2)  # shape: (nphi, ntheta)
        #avg_area = jnp.mean(norm_n)
        #return avg_area
        n = self.normal  # shape: (nphi, ntheta, 3)
        norm_n = jnp.linalg.norm(n, axis=2)  

        dphi = 2 * jnp.pi / self.nphi
        dtheta = 2 * jnp.pi / self.ntheta

        area = jnp.sum(norm_n) * dphi * dtheta
        return area

    def change_resolution(self, mpol=None, ntor=None, ntheta=None, nphi=None):
        """
        Change the number of m and n modes (mpol and ntor) and optionally the grid resolution.
        
        Parameters
        ----------
        mpol : int, optional
            New maximum poloidal mode number. If None, keeps current value.
        ntor : int, optional
            New maximum toroidal mode number. If None, keeps current value.
        ntheta : int, optional
            New number of theta grid points. If None, keeps current value.
        nphi : int, optional
            New number of phi grid points. If None, keeps current value.
            
        Notes
        -----
        - New Fourier coefficients (for modes not in the old representation) are initialized to zero.
        - Old coefficients outside the new range are discarded.
        - Existing modes are preserved when possible.
        - The dofs property automatically updates to reflect the new mode structure.
        """
        # Store old values
        rc_old = self._rc
        zs_old = self._zs
        mpol_old = self._mpol
        ntor_old = self._ntor
        xm_old = self.xm
        xn_old = self.xn
        
        # Set new values (use old values if not specified)
        mpol_new = mpol if mpol is not None else mpol_old
        ntor_new = ntor if ntor is not None else ntor_old
        
        # Validate inputs
        assert isinstance(mpol_new, int) and mpol_new >= 0, "mpol must be a non-negative integer."
        assert isinstance(ntor_new, int) and ntor_new >= 0, "ntor must be a non-negative integer."
        
        # Calculate new array size
        size_new = (mpol_new + 1) * (2 * ntor_new + 1) - ntor_new
        
        # Initialize new coefficient arrays with zeros
        rc_new = jnp.zeros(size_new)
        zs_new = jnp.zeros(size_new)
        
        # Calculate new xm and xn arrays
        xm_new = jnp.repeat(jnp.arange(mpol_new + 1), 2 * ntor_new + 1)[ntor_new:]
        xn_new = self._nfp * jnp.tile(jnp.arange(-ntor_new, ntor_new + 1), mpol_new + 1)[ntor_new:]
        
        # Vectorized mode matching using JAX operations
        # Create a 2D comparison matrix: (new_modes, old_modes)
        m_match = xm_new[:, None] == xm_old[None, :]  # Shape: (size_new, size_old)
        n_match = xn_new[:, None] == xn_old[None, :]  # Shape: (size_new, size_old)
        mode_match = m_match & n_match  # True where both m and n match
        
        # For each new mode, find if there's a matching old mode
        # jnp.any(mode_match, axis=1) tells us which new modes have a match
        # jnp.argmax(mode_match, axis=1) gives us the index of the first match
        has_match = jnp.any(mode_match, axis=1)
        old_indices = jnp.argmax(mode_match, axis=1)
        
        # Copy values from old arrays where matches exist
        rc_new = jnp.where(has_match, rc_old[old_indices], rc_new)
        zs_new = jnp.where(has_match, zs_old[old_indices], zs_new)
        
        # Update the object's attributes
        self._mpol = mpol_new
        self._ntor = ntor_new
        self._rc = rc_new
        self._zs = zs_new
        
        # Update grid resolution if requested
        if ntheta is not None:
            assert isinstance(ntheta, int) and ntheta > 0, "ntheta must be a positive integer."
            self._ntheta = ntheta
        if nphi is not None:
            assert isinstance(nphi, int) and nphi > 0, "nphi must be a positive integer."
            self._nphi = nphi
        
        # Reset all cached values (including scaling since it depends on xm/xn)
        self.reset_cache()
        self._scaling = None  # Reset scaling since it depends on mpol/ntor
        if ntheta is not None or nphi is not None:
            self.reset_mesh()

    # def change_resolution(self, mpol: int, ntor: int, ntheta=None, nphi=None,close=True):
    #     """
    #     Change the values of `mpol` and `ntor`.
    #     New Fourier coefficients are zero by default.
    #     Old coefficients outside the new range are discarded.
    #     """
    #     rc_old, zs_old = self.rc, self.zs
    #     mpol_old, ntor_old = self.mpol, self.ntor
    #     if ntheta is not None:
    #         self.ntheta = ntheta
    #     else:
    #         ntheta = self.ntheta

    #     if nphi is not None:
    #         self.nphi = nphi
    #     else:
    #         nphi = self.nphi

    #     #rc_new = jnp.zeros((mpol, 2 * ntor + 1))
    #     #zs_new = jnp.zeros((mpol, 2 * ntor + 1))
    #     rc_new = jnp.zeros(((mpol+1)*( 2 * ntor + 1)-ntor))
    #     zs_new = jnp.zeros(((mpol+1)*( 2 * ntor + 1)-ntor))
    #     m_keep = min(mpol_old, mpol)
    #     n_keep = min(ntor_old, ntor)

    #     xm_old=self.xm
    #     xn_old=self.xn
    #     self.xm =  jnp.repeat(jnp.arange(mpol+1), 2*ntor+1)[ntor:]
    #     self.xn = self.nfp*jnp.tile(jnp.arange(-ntor, ntor + 1), mpol+1)[ntor:]
    #     # Copy overlapping region
    #     for l in range(len(self.xm)):
    #         if self.xm[l]<=m_keep and jnp.abs(self.xn[l]/self.nfp)<=n_keep:
    #             index=self.xm[l]*(ntor_old*2+1)-self.xn[l]//self.nfp
    #             rc_new=rc_new.at[l].set(self.rc[index])
    #             zs_new=zs_new.at[l].set(self.zs[index])


    #     # Update attributes
    #     self.mpol, self.ntor = mpol, ntor
    #     self.rc, self.zs = rc_new, zs_new

    #     self.rmnc_interp = self.rc
    #     self.zmns_interp = self.zs

    #     # Update degrees of freedom
    #     self.num_dofs_rc = len(jnp.ravel(self.rc))
    #     self.num_dofs_zs = len(jnp.ravel(self.zs))
    #     self._dofs = jnp.concatenate((self.rescaling_function(jnp.ravel(self.rc)), self.rescaling_function(jnp.ravel(self.zs))))

    #     # Recompute angles and geometry
    #     if self.range_torus == 'full torus': div = 1
    #     else: div = self.nfp
    #     if self.range_torus == 'half period': end_val = 0.5
    #     else: end_val = 1.0        
    #     self.quadpoints_theta = jnp.linspace(0, 2 * jnp.pi, num=ntheta, endpoint=True if close else False)
    #     self.quadpoints_phi   = jnp.linspace(0, 2 * jnp.pi * end_val / div, num=nphi, endpoint=True if close else False)
    #     self.theta_2d, self.phi_2d = jnp.meshgrid(self.quadpoints_theta, self.quadpoints_phi)

    #     self.angles = (jnp.einsum('i,jk->ijk', self.xm, self.theta_2d)- jnp.einsum('i,jk->ijk', self.xn, self.phi_2d))
    #     (self._gamma, self._gammadash_theta, self._gammadash_phi,
    #     self._normal, self._unitnormal) = self._set_gamma(self.rmnc_interp, self.zmns_interp)


    #     # Recompute AbsB if available
    #     if hasattr(self, 'bmnc'):
    #         self._AbsB = self._set_AbsB()

    #     return self

    def plot(self, ax=None, show=True, close=False, axis_equal=True, **kwargs):
        if close: raise NotImplementedError("Call close=True when instantiating the VMEC/SurfaceRZFourier object.")
        
        kwargs.setdefault('alpha', 0.6)

        import matplotlib.pyplot as plt 
        from matplotlib import cm
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        
        boundary = self.gamma
        
        if hasattr(self, 'bmnc'):
            Bmag = self.AbsB
            B_rescaled = (Bmag - Bmag.min()) / (Bmag.max() - Bmag.min())
            ax.plot_surface(boundary[:, :, 0], boundary[:, :, 1], boundary[:, :, 2], facecolors=cm.jet(B_rescaled), linewidth=0, antialiased=True, **kwargs)
        else:
            ax.plot_surface(boundary[:, :, 0], boundary[:, :, 1], boundary[:, :, 2], linewidth=0, antialiased=True, **kwargs)
        # ax.set_axis_off()
        ax.grid(False)

        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
    
    def to_vtk(self, filename, extra_data=None, field=None):
        try: import numpy as np
        except ImportError: raise ImportError("The 'numpy' library is required. Please install it using 'pip install numpy'.")
        try: from pyevtk.hl import gridToVTK
        except ImportError: raise ImportError("The 'pyevtk' library is required. Please install it using 'pip install pyevtk'.")
        boundary = np.array(self.gamma)
        if hasattr(self, 'bmnc'):
            Bmag = np.array(self.AbsB)
            Bmag = Bmag.reshape((1, self.nphi, self.ntheta)).copy()
        x = boundary[:, :, 0].reshape((1, self.nphi, self.ntheta)).copy()
        y = boundary[:, :, 1].reshape((1, self.nphi, self.ntheta)).copy()
        z = boundary[:, :, 2].reshape((1, self.nphi, self.ntheta)).copy()
        pointData = {}
        if field is not None:
            B_dot_n_over_B = np.array(BdotN_over_B(self, field)).reshape((1,self. nphi, self.ntheta)).copy()
            pointData["B_dot_n_over_B"] = B_dot_n_over_B
            B_BiotSavart = np.array(vmap(lambda surf: vmap(lambda x: field.AbsB(x))(surf))(boundary)).reshape((1, self.nphi, self.ntheta)).copy()
            pointData["B_BiotSavart"] = B_BiotSavart
        if hasattr(self, 'bmnc'):
            pointData["B_VMEC"]=Bmag
        if extra_data is not None:
            pointData = {**pointData, **extra_data}
        gridToVTK(str(filename), x, y, z, pointData=pointData)

    def to_vmec(self, filename):
        """
        Generates a fortran namelist file containing the RBC/RBS/ZBC/ZBS
        coefficients, in the form used in VMEC and SPEC input
        files. The result will be returned as a string. For saving a
        file, see the ``write_nml()`` function.
        """
        nml = ''
        nml += '&INDATA\n'
        nml += 'LASYM = .FALSE.\n'
        nml += f'NFP = {self.nfp}\n'

        # Copy overlapping region
        for l in range(len(self.xm)):
            rc = self.rc[l]
            zs = self.zs[l]
            nml += f"RBC({self.xn[l]/self.nfp:4d},{self.xm[l]:4d}) ={rc:23.15e},    ZBS({self.xn[l]/self.nfp:4d},{self.xm[l]:4d}) ={zs:23.15e}\n"
        nml += '/\n'
        
        with open(filename, 'w') as f:
            f.write(nml)
            
    def mean_cross_sectional_area(self):
        xyz = self.gamma
        x2y2 = xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2
        dgamma1 = self.gammadash_phi
        dgamma2 = self.gammadash_theta
        J = jnp.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
        J = J.at[:, :, 0, 0].set((xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0]) / x2y2)
        J = J.at[:, :, 0, 1].set((xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0]) / x2y2)
        J = J.at[:, :, 1, 0].set(0)
        J = J.at[:, :, 1, 1].set(1)
        detJ = jnp.linalg.det(J)
        Jinv = jnp.linalg.inv(J)
        dZ_dtheta = dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
        mean_cross_sectional_area = jnp.abs(jnp.mean(jnp.sqrt(x2y2) * dZ_dtheta * detJ))/(2 * jnp.pi)
        return mean_cross_sectional_area
    
    def _tree_flatten(self):
        children = (self._rc, self._zs)  # arrays / dynamic values
        aux_data = {"nfp": self._nfp,
                    "mpol": self._mpol,
                    "ntor": self._ntor,
                    "ntheta": self._ntheta,
                    "nphi": self._nphi,
                    "close": self._close,
                    "range_torus": self._range_torus,
                    "scaling_type": self._scaling_type,
                    "scaling_factor": self._scaling_factor}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(SurfaceRZFourier,
                               SurfaceRZFourier._tree_flatten,
                               SurfaceRZFourier._tree_unflatten)


class SurfaceGFrameFourier:
    """
    Surface representation using a generalized Frenet frame (G-frame) approach.
    
    The surface is parametrized as:
        r(θ, φ) = r_axis(φ) + X1(θ, φ) * n1(φ) + X2(θ, φ) * n2(φ)
    
    where:
    - r_axis(φ) is the central axis curve
    - n1(φ), n2(φ) are orthonormal vectors perpendicular to the axis (Frenet frame)
    - X1, X2 are Fourier expanded in the poloidal angle θ
    
    This approach is inspired by GVEC (Galerkin Variational Equilibrium Code).
    """
    
    def __init__(self, x1_coeffs, x2_coeffs, nfp, mpol, ntor, ntheta=30, nphi=30, 
                 close=True, range_torus='full torus', axis_major_radius=1.0, 
                 axis_shape='circle', axis_pitch=0.0, axis_twist_rate=0.0,
                 axis_function=None,
                 enable_axis_dofs=False, axis_rc_coeffs=None, axis_zs_coeffs=None,
                 axis_ntor=0,
                 scaling_type=2, scaling_factor=0, scale_fixed=1.0):
        """
        Initialize a surface using G-frame parametrization.
        
        Parameters
        ----------
        x1_coeffs : jnp.ndarray
            Fourier coefficients for X1 (perpendicular coordinate 1)
        x2_coeffs : jnp.ndarray
            Fourier coefficients for X2 (perpendicular coordinate 2)
        nfp : int
            Number of field periods
        mpol : int
            Maximum poloidal mode number
        ntor : int
            Maximum toroidal mode number
        ntheta : int, optional
            Number of poloidal grid points (default: 30)
        nphi : int, optional
            Number of toroidal grid points (default: 30)
        close : bool, optional
            Whether the surface is topologically closed (default: True)
        range_torus : str, optional
            'full torus', 'half period', or 'full period'
        axis_major_radius : float, optional
            Major radius of the axis curve (default: 1.0)
        axis_shape : str, optional
            Shape of the axis: 'circle', 'ellipse', 'helical', 'fourier', or 'custom' (default: 'circle')
        axis_pitch : float, optional
            Pitch of helical axis (vertical rise per 2π toroidal turn) (default: 0.0)
        axis_twist_rate : float, optional
            Twist rate of the Frenet frame along the axis (radians per toroidal turn) (default: 0.0)
        axis_function : callable, optional
            Custom axis function r(phi) -> [x, y, z]. Only used if axis_shape == 'custom'.
            Must be JAX-compatible (use jnp operations). Function signature: (phi,) -> (3,)
            Example: lambda phi: jnp.array([R*cos(phi), R*sin(phi) + A*sin(2*phi), B*sin(phi)])
        enable_axis_dofs : bool, optional
            Enable axis degrees of freedom using Fourier expansion (default: False)
        axis_rc_coeffs : jnp.ndarray, optional
            Fourier coefficients for the axis R*cos(phi) component (only if enable_axis_dofs=True)
        axis_zs_coeffs : jnp.ndarray, optional
            Fourier coefficients for the axis Z*sin(phi) component (only if enable_axis_dofs=True)
        axis_ntor : int, optional
            Maximum toroidal mode number for axis (default: 0)
        scaling_type : int, optional
            Type of mode scaling (1: L1, 2: L2, inf: L∞) (default: 2)
        scaling_factor : float, optional
            Scaling factor for modes (default: 0)
        scale_fixed : float, optional
            Fixed scaling multiplier (default: 1.0)
        """
        
        assert isinstance(nfp, int) and nfp > 0, "nfp must be a positive integer."
        assert isinstance(mpol, int) and mpol >= 0, "mpol must be a non-negative integer."
        assert isinstance(ntor, int) and ntor >= 0, "ntor must be a non-negative integer."
        assert isinstance(ntheta, int) and ntheta > 0, "ntheta must be a positive integer."
        assert isinstance(nphi, int) and nphi > 0, "nphi must be a positive integer."
        assert range_torus in ['full torus', 'half period', 'full period']
        
        # Auto-detect axis_shape='custom' if axis_function is provided
        if axis_function is not None and axis_shape != 'custom':
            axis_shape = 'custom'
        # Auto-detect axis_shape='fourier' if axis DOFs are enabled
        if enable_axis_dofs:
            axis_shape = 'fourier'
            n_axis_modes = 2 * axis_ntor + 1
            
            # Helper function to check if a value is a valid numeric array-like
            def is_valid_numeric(val):
                if val is None:
                    return False
                # Check if it has dtype and if it's not object (sentinel pattern)
                if hasattr(val, 'dtype'):
                    dt = getattr(val, 'dtype', None)
                    if dt is not None:
                        return str(dt) != 'object'
                # If no dtype, it's not array-like
                return False
            
            if not is_valid_numeric(axis_rc_coeffs):
                axis_rc_coeffs = jnp.zeros(n_axis_modes)
                if n_axis_modes > 0:
                    axis_rc_coeffs = axis_rc_coeffs.at[axis_ntor].set(axis_major_radius)
            else:
                axis_rc_coeffs = jnp.asarray(axis_rc_coeffs)
                if axis_rc_coeffs.size != n_axis_modes:
                    if axis_rc_coeffs.size < n_axis_modes:
                        pad_width = n_axis_modes - axis_rc_coeffs.size
                        axis_rc_coeffs = jnp.pad(axis_rc_coeffs, (0, pad_width))
                    else:
                        axis_rc_coeffs = axis_rc_coeffs[:n_axis_modes]
            
            if not is_valid_numeric(axis_zs_coeffs):
                axis_zs_coeffs = jnp.zeros(n_axis_modes)
            else:
                axis_zs_coeffs = jnp.asarray(axis_zs_coeffs)
                if axis_zs_coeffs.size != n_axis_modes:
                    if axis_zs_coeffs.size < n_axis_modes:
                        pad_width = n_axis_modes - axis_zs_coeffs.size
                        axis_zs_coeffs = jnp.pad(axis_zs_coeffs, (0, pad_width))
                    else:
                        axis_zs_coeffs = axis_zs_coeffs[:n_axis_modes]
        
        self._x1_coeffs = x1_coeffs
        self._x2_coeffs = x2_coeffs
        self._nfp = nfp
        self._mpol = mpol
        self._ntor = ntor
        
        self._gamma = None
        self._gammadash_theta = None
        self._gammadash_phi = None
        self._gammadashdash_theta = None
        self._gammadashdash_phi = None
        self._gammadashdash_theta_phi = None
        self._normal = None
        self._unitnormal = None
        self._area_element = None
        self._xm = None
        self._xn = None
        
        self._ntheta = ntheta
        self._nphi = nphi
        self._close = close
        self._range_torus = range_torus
        
        self._quadpoints_theta = None
        self._quadpoints_phi = None
        self._theta2d = None
        self._phi2d = None
        self._angles = None
        
        self._axis_major_radius = axis_major_radius
        self._axis_shape = axis_shape
        self._axis_pitch = axis_pitch
        self._axis_twist_rate = axis_twist_rate
        self._axis_function = axis_function
        self._axis_curve = None
        self._axis_curve_dash = None
        self._axis_curve_dashdash = None
        self._frenet_n1 = None
        self._frenet_n2 = None
        self._frenet_n1_dash = None
        self._frenet_n2_dash = None

        # Axis DOFs
        self._enable_axis_dofs = enable_axis_dofs
        # Always ensure axis coefficients are proper JAX arrays, even when axis DOFs are disabled
        # This prevents JAX type errors during pytree reconstruction
        if not enable_axis_dofs:
            n_axis_modes = 2 * axis_ntor + 1
            if axis_rc_coeffs is None:
                axis_rc_coeffs = jnp.zeros(n_axis_modes)
            else:
                axis_rc_coeffs = jnp.asarray(axis_rc_coeffs, dtype=jnp.float64)
            if axis_zs_coeffs is None:
                axis_zs_coeffs = jnp.zeros(n_axis_modes)
            else:
                axis_zs_coeffs = jnp.asarray(axis_zs_coeffs, dtype=jnp.float64)
        self._axis_rc_coeffs = axis_rc_coeffs
        self._axis_zs_coeffs = axis_zs_coeffs
        self._axis_ntor = axis_ntor
        self._axis_xn = None
        
        self._scaling_type = scaling_type
        self._scaling_factor = scaling_factor
        self._scale_fixed = scale_fixed
        self._scaling = None
    
    def reset_cache(self):
        """Reset cached computed properties."""
        self._gamma = None
        self._gammadash_theta = None
        self._gammadash_phi = None
        self._normal = None
        self._unitnormal = None
        self._area_element = None
        self._xm = None
        self._xn = None
        self._angles = None
        self._axis_curve = None
        self._axis_curve_dash = None
        self._axis_curve_dashdash = None
        self._frenet_n1 = None
        self._frenet_n2 = None
        self._frenet_n1_dash = None
        self._frenet_n2_dash = None
        self._axis_xn = None
    
    def reset_mesh(self):
        """Reset mesh-related cached properties."""
        self._quadpoints_theta = None
        self._quadpoints_phi = None
        self._theta2d = None
        self._phi2d = None
        self._angles = None
    
    # Properties for X1 and X2 coefficients
    @property
    def x1_coeffs(self):
        return self._x1_coeffs
    
    @x1_coeffs.setter
    def x1_coeffs(self, new_x1):
        self._x1_coeffs = new_x1
        self.reset_cache()
    
    @property
    def x2_coeffs(self):
        return self._x2_coeffs
    
    @x2_coeffs.setter
    def x2_coeffs(self, new_x2):
        self._x2_coeffs = new_x2
        self.reset_cache()
    
    # Static geometry properties
    @property
    def nfp(self):
        return self._nfp
    
    @property
    def mpol(self):
        return self._mpol
    
    @property
    def ntor(self):
        return self._ntor
    
    @property
    def ntheta(self):
        return self._ntheta
    
    @ntheta.setter
    def ntheta(self, new_ntheta):
        self._ntheta = new_ntheta
        self.reset_mesh()
    
    @property
    def nphi(self):
        return self._nphi
    
    @nphi.setter
    def nphi(self, new_nphi):
        self._nphi = new_nphi
        self.reset_mesh()
    
    @property
    def xm(self):
        if self._xm is None:
            self._xm = jnp.repeat(jnp.arange(self.mpol + 1), 2 * self.ntor + 1)[self.ntor:]
        return self._xm
    
    @property
    def xn(self):
        if self._xn is None:
            self._xn = self.nfp * jnp.tile(jnp.arange(-self.ntor, self.ntor + 1), self.mpol + 1)[self.ntor:]
        return self._xn
    
    @property
    def scaling(self):
        if self._scaling is None:
            mode_scaling = jnp.exp(self.scaling_factor * jnp.linalg.norm(jnp.vstack([self.xm, self.xn]), ord=self._scaling_type, axis=0))
            self._scaling = mode_scaling * self._scale_fixed
        return self._scaling
    
    def _get_axis_scaling(self):
        """Compute scaling for axis modes (mode-dependent, consistent with surface scaling)."""
        if not self._enable_axis_dofs:
            return jnp.ones(0)
        
        n_axis_modes = 2 * self.axis_ntor + 1
        # Axis mode numbers: n = nfp * [-axis_ntor, ..., 0, ..., +axis_ntor]
        axis_n = self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
        
        # Apply same scaling formula as surface modes
        # Axis is 1D (only toroidal modes), so norm is just |n|
        mode_scaling = jnp.exp(self.scaling_factor * jnp.linalg.norm(axis_n[jnp.newaxis, :], ord=self._scaling_type, axis=0))
        return mode_scaling * self._scale_fixed
    
    @property
    def dofs(self):
        """Degrees of freedom: combined X1, X2, and optional axis coefficients with scaling."""
        if self._enable_axis_dofs:
            # Ensure axis coefficients are not None
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_scaling = self._get_axis_scaling()
            return jnp.hstack([
                self.x1_coeffs * self.scaling,
                self.x2_coeffs * self.scaling,
                axis_rc * axis_scaling,
                axis_zs * axis_scaling,
            ])
        return jnp.hstack([self.x1_coeffs * self.scaling, self.x2_coeffs * self.scaling])
    
    @dofs.setter
    def dofs(self, new_dofs):
        """Set degrees of freedom."""
        n1 = len(self.x1_coeffs)
        n2 = len(self.x2_coeffs)
        self._x1_coeffs = new_dofs[:n1] / self.scaling
        self._x2_coeffs = new_dofs[n1:n1 + n2] / self.scaling
        if self._enable_axis_dofs:
            axis_scaling = self._get_axis_scaling()
            n_axis = 2 * self.axis_ntor + 1
            self._axis_rc_coeffs = new_dofs[n1 + n2:n1 + n2 + n_axis] / axis_scaling
            self._axis_zs_coeffs = new_dofs[n1 + n2 + n_axis:n1 + n2 + 2*n_axis] / axis_scaling
        self.reset_cache()
    
    @property
    def axis_function(self):
        """Custom axis function for axis_shape='custom'."""
        return self._axis_function
    
    @axis_function.setter
    def axis_function(self, func):
        """Set custom axis function. Must be JAX-compatible."""
        self._axis_function = func
        self.reset_cache()

    @property
    def enable_axis_dofs(self):
        """Whether axis DOFs are enabled."""
        return self._enable_axis_dofs

    @property
    def axis_rc_coeffs(self):
        """Axis R*cos(phi) Fourier coefficients."""
        return self._axis_rc_coeffs

    @axis_rc_coeffs.setter
    def axis_rc_coeffs(self, new_rc):
        if not self._enable_axis_dofs:
            raise ValueError("Cannot set axis_rc_coeffs when enable_axis_dofs=False")
        self._axis_rc_coeffs = new_rc
        self.reset_cache()

    @property
    def axis_zs_coeffs(self):
        """Axis Z*sin(phi) Fourier coefficients."""
        return self._axis_zs_coeffs

    @axis_zs_coeffs.setter
    def axis_zs_coeffs(self, new_zs):
        if not self._enable_axis_dofs:
            raise ValueError("Cannot set axis_zs_coeffs when enable_axis_dofs=False")
        self._axis_zs_coeffs = new_zs
        self.reset_cache()

    @property
    def axis_ntor(self):
        """Maximum toroidal mode number for axis."""
        return self._axis_ntor

    @property
    def axis_xn(self):
        """Toroidal mode numbers for axis (m=0 only)."""
        # Compute if not cached and axis DOFs are enabled or axis_shape is 'fourier'
        if self._axis_xn is None and (self._enable_axis_dofs or self._axis_shape == 'fourier' or self._axis_rc_coeffs is not None):
            self._axis_xn = self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
        return self._axis_xn
    
    # Mesh and angle properties
    @jit
    def _compute_meshgrid(self):
        if self.range_torus == "full torus":
            div, end_val = 1., 1.
        elif self.range_torus == "half period":
            div, end_val = self.nfp, 0.5
        elif self.range_torus == "full period":
            div, end_val = self.nfp, 1.0
        
        quadpoints_theta = jnp.linspace(0, 2 * jnp.pi, num=self.ntheta, endpoint=self._close)
        quadpoints_phi = jnp.linspace(0, 2 * jnp.pi * end_val / div, num=self.nphi, endpoint=self._close)
        theta2d, phi2d = jnp.meshgrid(quadpoints_theta, quadpoints_phi)
        return quadpoints_theta, quadpoints_phi, theta2d, phi2d
    
    @property
    def theta2d(self):
        if self._theta2d is None:
            self._quadpoints_theta, self._quadpoints_phi, self._theta2d, self._phi2d = self._compute_meshgrid()
        return self._theta2d
    
    @property
    def phi2d(self):
        if self._phi2d is None:
            self._quadpoints_theta, self._quadpoints_phi, self._theta2d, self._phi2d = self._compute_meshgrid()
        return self._phi2d
    
    @property
    def angles(self):
        """Angle combinations for Fourier expansion."""
        if self._angles is None:
            self._angles = jnp.einsum('i,jk->ijk', self.xm, self.theta2d) - jnp.einsum('i,jk->ijk', self.xn, self.phi2d)
        return self._angles
    
    @property
    def range_torus(self):
        return self._range_torus
    
    @range_torus.setter
    def range_torus(self, new_range):
        self._range_torus = new_range
        self.reset_mesh()
    
    # scaling_type property and setter
    @property
    def scaling_type(self):
        return self._scaling_type

    @scaling_type.setter
    def scaling_type(self, new_type):
        self._scaling_type = new_type
        self._scaling = None

    # scaling_factor property and setter
    @property
    def scaling_factor(self):
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, new_factor):
        self._scaling_factor = new_factor
        self._scaling = None

    # Axis curve computation
    def _compute_axis_curve(self, phi):
        """
        Compute the central axis curve.
        
        Supports:
        - 'circle': Circular axis at major radius R0
        - 'ellipse': Elliptical axis (with aspect ratio parameter)
        - 'helical': Helical axis with pitch and twist rate
        - 'fourier': Fourier-expanded axis with optimizable coefficients
        - 'custom': User-defined axis function
        """
        if self._axis_shape == 'fourier':
            phi_arr = jnp.atleast_1d(phi)
            # Use private attributes directly to avoid issues inside vmap
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_xn_vals = self.axis_xn if self.axis_xn is not None else self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
            
            angles_axis = -axis_xn_vals[:, None] * phi_arr[None, :]
            r_phi = jnp.sum(axis_rc[:, None] * jnp.cos(angles_axis), axis=0)
            z_phi = jnp.sum(axis_zs[:, None] * jnp.sin(angles_axis), axis=0)
            x_axis = r_phi * jnp.cos(phi_arr)
            y_axis = r_phi * jnp.sin(phi_arr)
            z_axis = z_phi
            return jnp.squeeze(jnp.stack([x_axis, y_axis, z_axis], axis=-1))
        elif self._axis_shape == 'circle':
            # Circular axis at major radius R0
            x_axis = self._axis_major_radius * jnp.cos(phi)
            y_axis = self._axis_major_radius * jnp.sin(phi)
            z_axis = jnp.zeros_like(phi)
        elif self._axis_shape == 'ellipse':
            # Elliptical axis (would need additional parameters)
            aspect_ratio = 1.0  # Could be parameterized
            x_axis = self._axis_major_radius * jnp.cos(phi)
            y_axis = self._axis_major_radius * aspect_ratio * jnp.sin(phi)
            z_axis = jnp.zeros_like(phi)
        elif self._axis_shape == 'helical':
            # Helical axis: rises as pitch * phi / (2*pi), maintains circular cross-section
            x_axis = self._axis_major_radius * jnp.cos(phi)
            y_axis = self._axis_major_radius * jnp.sin(phi)
            z_axis = self._axis_pitch * phi / (2 * jnp.pi)
        elif self._axis_shape == 'custom':
            # Custom axis function provided by user
            if self._axis_function is None:
                raise ValueError("axis_shape='custom' requires axis_function to be provided")
            # Call custom function (must be JAX-compatible)
            axis_result = self._axis_function(phi)
            return axis_result
        else:
            # Default to circle
            x_axis = self._axis_major_radius * jnp.cos(phi)
            y_axis = self._axis_major_radius * jnp.sin(phi)
            z_axis = jnp.zeros_like(phi)
        
        return jnp.stack([x_axis, y_axis, z_axis], axis=-1)
    
    def _compute_frenet_frame(self, phi):
        """
        Compute the generalized Frenet frame at each point on the axis.
        
        Returns the two orthonormal vectors n1 and n2 perpendicular to the axis.
        Includes twist_rate to allow frame rotation along the axis.
        """
        # For circular and other simple axes, compute derivatives analytically to avoid
        # issues with nested grad operations inside vmap+pjit
        if self._axis_shape == 'circle':
            # For circular axis: axis(phi) = [R*cos(phi), R*sin(phi), 0]
            # Derivatives: daxis/dphi = [-R*sin(phi), R*cos(phi), 0]
            cos_phi = jnp.cos(phi)
            sin_phi = jnp.sin(phi)
            
            tangent = jnp.array([-sin_phi, cos_phi, 0.0])
            tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
            tangent_norm = jnp.maximum(tangent_norm, 1e-14)
            tangent = tangent / tangent_norm
            
            # n1 points radially outward from z-axis
            axis = self._compute_axis_curve(phi)
            r_xy = jnp.linalg.norm(axis[..., :2], axis=-1, keepdims=True)
            r_xy = jnp.maximum(r_xy, 1e-12)
            n1 = jnp.concatenate([axis[..., :2] / r_xy, jnp.zeros_like(axis[..., :1])], axis=-1)
            
            # n2 = n1 × T
            n2 = jnp.cross(n1, tangent, axis=-1)
            n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)
            n2_norm = jnp.maximum(n2_norm, 1e-14)
            n2 = n2 / n2_norm
        elif self._axis_shape == 'fourier':
            # For Fourier axis, compute analytical derivatives to avoid nested grad/vmap
            # This prevents memory issues when _compute_gamma is JIT-compiled with vmap
            phi_arr = jnp.atleast_1d(phi)
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_xn_vals = self.axis_xn if self.axis_xn is not None else self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
            
            angles_axis = -axis_xn_vals[:, None] * phi_arr[None, :]
            
            # Position: r = Σ rc_i * cos(angles), z = Σ zs_i * sin(angles)
            r_phi = jnp.sum(axis_rc[:, None] * jnp.cos(angles_axis), axis=0)
            z_phi = jnp.sum(axis_zs[:, None] * jnp.sin(angles_axis), axis=0)
            
            # Derivatives: dr/dφ = Σ rc_i * xn_i * sin(angles), dz/dφ = -Σ zs_i * xn_i * cos(angles)
            dr_dphi = jnp.sum(axis_rc[:, None] * axis_xn_vals[:, None] * jnp.sin(angles_axis), axis=0)
            dz_dphi = -jnp.sum(axis_zs[:, None] * axis_xn_vals[:, None] * jnp.cos(angles_axis), axis=0)
            
            # Convert to Cartesian derivatives
            cos_phi = jnp.cos(phi_arr)
            sin_phi = jnp.sin(phi_arr)
            
            dx_dphi = dr_dphi * cos_phi - r_phi * sin_phi
            dy_dphi = dr_dphi * sin_phi + r_phi * cos_phi
            # dz_dphi already computed
            
            tangent = jnp.squeeze(jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=-1))
            tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
            tangent_norm = jnp.maximum(tangent_norm, 1e-14)
            tangent = tangent / tangent_norm
            
            # n1 points radially outward from z-axis
            axis = self._compute_axis_curve(phi)
            r_xy = jnp.linalg.norm(axis[..., :2], axis=-1, keepdims=True)
            r_xy = jnp.maximum(r_xy, 1e-12)
            n1 = jnp.concatenate([axis[..., :2] / r_xy, jnp.zeros_like(axis[..., :1])], axis=-1)
            
            # n2 = n1 × T
            n2 = jnp.cross(n1, tangent, axis=-1)
            n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)
            n2_norm = jnp.maximum(n2_norm, 1e-14)
            n2 = n2 / n2_norm
        else:
            # For other axis types, use automatic differentiation
            from jax import grad as jax_grad
            
            def axis_x(p):
                return self._compute_axis_curve(p)[0]
            def axis_y(p):
                return self._compute_axis_curve(p)[1]
            def axis_z(p):
                return self._compute_axis_curve(p)[2]
            
            dx_dphi = jax_grad(axis_x)(phi)
            dy_dphi = jax_grad(axis_y)(phi)
            dz_dphi = jax_grad(axis_z)(phi)
            
            tangent = jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=-1)
            tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
            tangent_norm = jnp.maximum(tangent_norm, 1e-14)  # Prevent division by zero
            tangent = tangent / tangent_norm
            
            # First normal: radial direction (for a circular axis at z=0)
            # n1 points radially outward from the z-axis
            axis = self._compute_axis_curve(phi)
            r_xy = jnp.linalg.norm(axis[..., :2], axis=-1, keepdims=True)
            # Handle the case where r_xy is very small (near z-axis)
            r_xy = jnp.maximum(r_xy, 1e-12)
            n1 = jnp.concatenate([axis[..., :2] / r_xy, jnp.zeros_like(axis[..., :1])], axis=-1)
            
            # Second normal: perpendicular to both n1 and tangent (via cross product)
            # Use n1 × T to get correct orientation (right-handed system with n1 radial, T toroidal)
            n2 = jnp.cross(n1, tangent, axis=-1)
            n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)
            n2_norm = jnp.maximum(n2_norm, 1e-14)  # Prevent division by zero
            n2 = n2 / n2_norm
        
        # Apply twist rate: rotate n1 and n2 in their plane
        if self._axis_twist_rate != 0.0:
            twist_angle = self._axis_twist_rate * phi
            cos_twist = jnp.cos(twist_angle)
            sin_twist = jnp.sin(twist_angle)
            
            # Rotate (n1, n2) by twist_angle in their plane
            n1_twisted = cos_twist[..., None] * n1 + sin_twist[..., None] * n2
            n2_twisted = -sin_twist[..., None] * n1 + cos_twist[..., None] * n2
            
            n1 = n1_twisted
            n2 = n2_twisted
        
        return n1, n2
    
    @jax.jit
    def _compute_gamma(self):
        """
        Compute the surface position in Cartesian coordinates and its derivatives.
        
        r(θ, φ) = r_axis(φ) + X1(θ, φ) * n1(φ) + X2(θ, φ) * n2(φ)
        
        where: X1 = sum_i x1_i * cos(m_i*θ - n_i*φ)
               X2 = sum_i x2_i * sin(m_i*θ - n_i*φ)
        """
        # Get angles for Fourier expansion
        angles = self.angles  # Shape: (nmodes, nphi, ntheta)
        sin_angles = jnp.sin(angles)
        cos_angles = jnp.cos(angles)
        
        # Fourier expansion of X1 and X2
        X1 = jnp.einsum('i,ijk->jk', self.x1_coeffs, cos_angles)  # (nphi, ntheta)
        X2 = jnp.einsum('i,ijk->jk', self.x2_coeffs, sin_angles)  # (nphi, ntheta)
        
        # Get axis curve and Frenet frame
        # Note: phi2d shape is (nphi, ntheta), we need to evaluate at each phi value
        phi2d = self.phi2d
        
        # For standard axes, compute directly without vmap (to avoid nested jit+vmap)
        phi_flat = phi2d.flatten()
        
        # Compute axis curve directly - vectorized for Fourier/circle/ellipse/helical
        if self._axis_shape == 'fourier':
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_xn_vals = self.axis_xn if self.axis_xn is not None else self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
            
            angles_axis = -axis_xn_vals[:, None] * phi_flat[None, :]
            r_phi = jnp.sum(axis_rc[:, None] * jnp.cos(angles_axis), axis=0)
            z_phi = jnp.sum(axis_zs[:, None] * jnp.sin(angles_axis), axis=0)
            x_axis = r_phi * jnp.cos(phi_flat)
            y_axis = r_phi * jnp.sin(phi_flat)
            z_axis = z_phi
            axis_curve = jnp.stack([x_axis, y_axis, z_axis], axis=-1).reshape(self.nphi, self.ntheta, 3)
        elif self._axis_shape == 'circle':
            x_axis = self._axis_major_radius * jnp.cos(phi_flat)
            y_axis = self._axis_major_radius * jnp.sin(phi_flat)
            z_axis = jnp.zeros_like(phi_flat)
            axis_curve = jnp.stack([x_axis, y_axis, z_axis], axis=-1).reshape(self.nphi, self.ntheta, 3)
        elif self._axis_shape == 'ellipse':
            aspect_ratio = 1.0
            x_axis = self._axis_major_radius * jnp.cos(phi_flat)
            y_axis = self._axis_major_radius * aspect_ratio * jnp.sin(phi_flat)
            z_axis = jnp.zeros_like(phi_flat)
            axis_curve = jnp.stack([x_axis, y_axis, z_axis], axis=-1).reshape(self.nphi, self.ntheta, 3)
        elif self._axis_shape == 'helical':
            x_axis = self._axis_major_radius * jnp.cos(phi_flat)
            y_axis = self._axis_major_radius * jnp.sin(phi_flat)
            z_axis = self._axis_pitch * phi_flat / (2 * jnp.pi)
            axis_curve = jnp.stack([x_axis, y_axis, z_axis], axis=-1).reshape(self.nphi, self.ntheta, 3)
        elif self._axis_shape == 'custom':
            # For custom axes, compute directly
            axis_result = self._axis_function(phi_flat)
            if axis_result.ndim == 2 and axis_result.shape[1] == 3:
                axis_curve = axis_result
            else:
                axis_curve = axis_result.T
            axis_curve = axis_curve.reshape(self.nphi, self.ntheta, 3)
        else:
            # Should not reach here
            axis_curve = jnp.zeros((self.nphi, self.ntheta, 3))
        
        # Compute Frenet frame directly without vmap (eliminates nested jit+vmap issue)
        # All vectors are computed for all points at once using vectorized operations
        if self._axis_shape == 'circle':
            # Circular axis: compute n1 and n2 directly
            cos_phi = jnp.cos(phi_flat)
            sin_phi = jnp.sin(phi_flat)
            
            # tangent = [-sin(phi), cos(phi), 0] (from d/dphi of circular axis)
            tangent_x = -sin_phi
            tangent_y = cos_phi
            tangent_z = jnp.zeros_like(phi_flat)
            
            # n1 points radially outward: [cos(phi), sin(phi), 0]
            n1_x = cos_phi
            n1_y = sin_phi
            n1_z = jnp.zeros_like(phi_flat)
            
            # n2 = n1 × tangent (right-handed frame)
            n2_x = n1_y * tangent_z - n1_z * tangent_y  # = sin(phi) * 0 - 0 * cos(phi) = 0
            n2_y = n1_z * tangent_x - n1_x * tangent_z  # = 0 * (-sin(phi)) - cos(phi) * 0 = 0
            n2_z = n1_x * tangent_y - n1_y * tangent_x  # = cos(phi) * cos(phi) - sin(phi) * (-sin(phi)) = 1
            
            n1_vec = jnp.stack([n1_x, n1_y, n1_z], axis=-1)
            n2_vec = jnp.stack([n2_x, n2_y, n2_z], axis=-1)
            
            # Apply twist if needed
            if self._axis_twist_rate != 0.0:
                twist_angles = self._axis_twist_rate * phi_flat
                c = jnp.cos(twist_angles)
                s = jnp.sin(twist_angles)
                n1_x_new = c * n1_x + s * n2_x
                n1_y_new = c * n1_y + s * n2_y
                n1_z_new = c * n1_z + s * n2_z
                n2_x_new = -s * n1_x + c * n2_x
                n2_y_new = -s * n1_y + c * n2_y
                n2_z_new = -s * n1_z + c * n2_z
                n1_vec = jnp.stack([n1_x_new, n1_y_new, n1_z_new], axis=-1)
                n2_vec = jnp.stack([n2_x_new, n2_y_new, n2_z_new], axis=-1)
            
            n1 = n1_vec.reshape(self.nphi, self.ntheta, 3)
            n2 = n2_vec.reshape(self.nphi, self.ntheta, 3)
            
        elif self._axis_shape == 'fourier':
            # Fourier axis: compute n1 and n2 directly using analytical derivatives
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_xn_vals = self.axis_xn if self.axis_xn is not None else self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
            
            angles_axis = -axis_xn_vals[:, None] * phi_flat[None, :]
            r_phi = jnp.sum(axis_rc[:, None] * jnp.cos(angles_axis), axis=0)
            dr_dphi = jnp.sum(axis_rc[:, None] * axis_xn_vals[:, None] * jnp.sin(angles_axis), axis=0)
            dz_dphi = -jnp.sum(axis_zs[:, None] * axis_xn_vals[:, None] * jnp.cos(angles_axis), axis=0)
            
            cos_phi = jnp.cos(phi_flat)
            sin_phi = jnp.sin(phi_flat)
            
            # tangent derivatives in Cartesian
            dx_dphi = dr_dphi * cos_phi - r_phi * sin_phi
            dy_dphi = dr_dphi * sin_phi + r_phi * cos_phi
            
            # Normalize tangent with smooth regularization (no jnp.maximum discontinuity)
            tangent_norm_sq = dx_dphi**2 + dy_dphi**2 + dz_dphi**2
            tangent_norm = jnp.sqrt(tangent_norm_sq + 1e-28)  # Smooth regularization
            tangent_x = dx_dphi / tangent_norm
            tangent_y = dy_dphi / tangent_norm
            tangent_z = dz_dphi / tangent_norm
            
            # n1 points radially outward from z-axis, smooth regularization
            r_xy = r_phi  # Already computed as distance from z-axis
            r_xy_safe = jnp.sqrt(r_xy**2 + 1e-28)  # Smooth regularization instead of jnp.maximum
            n1_x = (r_phi * cos_phi) / r_xy_safe
            n1_y = (r_phi * sin_phi) / r_xy_safe
            n1_z = jnp.zeros_like(phi_flat)
            
            # n2 = n1 × tangent (right-handed frame)
            n2_x = n1_y * tangent_z - n1_z * tangent_y
            n2_y = n1_z * tangent_x - n1_x * tangent_z  
            n2_z = n1_x * tangent_y - n1_y * tangent_x
            
            # Normalize n2 with smooth regularization (no jnp.maximum discontinuity)
            n2_norm_sq = n2_x**2 + n2_y**2 + n2_z**2
            n2_norm = jnp.sqrt(n2_norm_sq + 1e-28)  # Smooth regularization
            n2_x = n2_x / n2_norm
            n2_y = n2_y / n2_norm
            n2_z = n2_z / n2_norm
            
            n1_vec = jnp.stack([n1_x, n1_y, n1_z], axis=-1)
            n2_vec = jnp.stack([n2_x, n2_y, n2_z], axis=-1)
            
            # Apply twist if needed
            if self._axis_twist_rate != 0.0:
                twist_angles = self._axis_twist_rate * phi_flat
                c = jnp.cos(twist_angles)
                s = jnp.sin(twist_angles)
                n1_x_new = c * n1_x + s * n2_x
                n1_y_new = c * n1_y + s * n2_y
                n1_z_new = c * n1_z + s * n2_z
                n2_x_new = -s * n1_x + c * n2_x
                n2_y_new = -s * n1_y + c * n2_y
                n2_z_new = -s * n1_z + c * n2_z
                n1_vec = jnp.stack([n1_x_new, n1_y_new, n1_z_new], axis=-1)
                n2_vec = jnp.stack([n2_x_new, n2_y_new, n2_z_new], axis=-1)
            
            n1 = n1_vec.reshape(self.nphi, self.ntheta, 3)
            n2 = n2_vec.reshape(self.nphi, self.ntheta, 3)
        else:
            # For other axis types (ellipse, helical, custom), compute using finite differences
            # to avoid grad() inside vmap inside jit
            eps = 1e-6
            axis_plus = self._compute_axis_curve_vectorized(phi_flat + eps)
            axis_minus = self._compute_axis_curve_vectorized(phi_flat - eps)
            d_axis = (axis_plus - axis_minus) / (2 * eps)
            
            tangent = d_axis
            tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
            tangent_norm = jnp.maximum(tangent_norm, 1e-14)
            tangent = tangent / tangent_norm
            
            # n1: perpendicular to z-axis for standard axes, or gram-schmidt from z-direction
            axis_pts = axis_curve.reshape(self.nphi * self.ntheta, 3)
            ref_z = jnp.array([0.0, 0.0, 1.0])
            dot_tz = jnp.sum(tangent * ref_z[None, :], axis=-1, keepdims=True)
            n1 = ref_z[None, :] - dot_tz * tangent
            n1_norm = jnp.linalg.norm(n1, axis=-1, keepdims=True)
            n1_norm = jnp.maximum(n1_norm, 1e-14)
            n1 = n1 / n1_norm
            
            # n2 = n1 × tangent (right-handed frame)
            n2 = jnp.cross(n1, tangent, axis=-1)
            n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)
            n2_norm = jnp.maximum(n2_norm, 1e-14)
            n2 = n2 / n2_norm
            
            # Apply twist if needed
            if self._axis_twist_rate != 0.0:
                twist_angles = self._axis_twist_rate * phi_flat
                c = jnp.cos(twist_angles)[:, None]
                s = jnp.sin(twist_angles)[:, None]
                n1_new = c * n1 + s * n2
                n2_new = -s * n1 + c * n2
                n1 = n1_new
                n2 = n2_new
            
            n1 = n1.reshape(self.nphi, self.ntheta, 3)
            n2 = n2.reshape(self.nphi, self.ntheta, 3)
            
        # Compute derivative of axis curve with respect to phi
        # For Fourier axis, use analytical derivatives (already computed above)
        if self._axis_shape == 'fourier':
            # Analytical derivatives already stored, just reshape
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_xn_vals = self.axis_xn if self.axis_xn is not None else self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
            
            angles_axis = -axis_xn_vals[:, None] * phi_flat[None, :]
            r_phi = jnp.sum(axis_rc[:, None] * jnp.cos(angles_axis), axis=0)
            dr_dphi = jnp.sum(axis_rc[:, None] * jnp.sin(angles_axis) * axis_xn_vals[:, None], axis=0)
            dz_dphi = jnp.sum(-axis_zs[:, None] * jnp.cos(angles_axis) * axis_xn_vals[:, None], axis=0)
            
            cos_phi = jnp.cos(phi_flat)
            sin_phi = jnp.sin(phi_flat)
            
            dx_dphi = dr_dphi * cos_phi - r_phi * sin_phi
            dy_dphi = dr_dphi * sin_phi + r_phi * cos_phi
            
            d_axis_flat = jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=-1)
            d_axis_curve = d_axis_flat.reshape(self.nphi, self.ntheta, 3)
        elif self._axis_shape == 'circle':
            # Analytical derivatives for circle
            cos_phi = jnp.cos(phi_flat)
            sin_phi = jnp.sin(phi_flat)
            
            dx_dphi = -self._axis_major_radius * sin_phi
            dy_dphi = self._axis_major_radius * cos_phi
            dz_dphi = jnp.zeros_like(phi_flat)
            
            d_axis_flat = jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=-1)
            d_axis_curve = d_axis_flat.reshape(self.nphi, self.ntheta, 3)
        elif self._axis_shape == 'ellipse':
            # Analytical derivatives for ellipse
            aspect_ratio = 1.0
            cos_phi = jnp.cos(phi_flat)
            sin_phi = jnp.sin(phi_flat)
            
            dx_dphi = -self._axis_major_radius * sin_phi
            dy_dphi = self._axis_major_radius * aspect_ratio * cos_phi
            dz_dphi = jnp.zeros_like(phi_flat)
            
            d_axis_flat = jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=-1)
            d_axis_curve = d_axis_flat.reshape(self.nphi, self.ntheta, 3)
        elif self._axis_shape == 'helical':
            # Analytical derivatives for helical
            cos_phi = jnp.cos(phi_flat)
            sin_phi = jnp.sin(phi_flat)
            
            dx_dphi = -self._axis_major_radius * sin_phi
            dy_dphi = self._axis_major_radius * cos_phi
            dz_dphi = (self._axis_pitch / (2 * jnp.pi)) * jnp.ones_like(phi_flat)
            
            d_axis_flat = jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=-1)
            d_axis_curve = d_axis_flat.reshape(self.nphi, self.ntheta, 3)
        else:
            # For custom axes, use finite differences
            eps = 1e-6
            axis_plus = self._compute_axis_curve_custom(phi_flat + eps)
            axis_minus = self._compute_axis_curve_custom(phi_flat - eps)
            d_axis_flat = (axis_plus - axis_minus) / (2 * eps)
            d_axis_curve = d_axis_flat.reshape(self.nphi, self.ntheta, 3)
        
        # Expand X1 and X2 to have a third dimension for vector operations
        X1_expanded = X1[..., None]  # (nphi, ntheta, 1)
        X2_expanded = X2[..., None]  # (nphi, ntheta, 1)
        
        # Compute surface position
        gamma = axis_curve + X1_expanded * n1 + X2_expanded * n2
        
        # Compute derivatives for surface normal, etc.
        # dX1 and dX2 with respect to theta
        dX1_dtheta = jnp.einsum('i,ijk->jk', -self.xm * self.x1_coeffs, sin_angles)
        dX2_dtheta = jnp.einsum('i,ijk->jk', self.xm * self.x2_coeffs, cos_angles)
        
        # dX1 and dX2 with respect to phi (using chain rule on Fourier expansion)
        # dX1/dphi = sum_i x1_i * d/dphi[cos(m_i*θ - n_i*φ)]
        #          = sum_i x1_i * n_i * sin(m_i*θ - n_i*φ)
        # dX2/dphi = sum_i x2_i * d/dphi[sin(m_i*θ - n_i*φ)]
        #          = -sum_i x2_i * n_i * cos(m_i*θ - n_i*φ)
        dX1_dphi = jnp.einsum('i,ijk->jk', self.xn * self.x1_coeffs, sin_angles)
        dX2_dphi = jnp.einsum('i,ijk->jk', -self.xn * self.x2_coeffs, cos_angles)
        
        # Compute Frenet frame derivatives analytically for Fourier axis
        # For other axis types, use finite differences
        if self._axis_shape == 'fourier':
            # Analytical derivatives for Fourier axis - no finite differences needed
            # IMPORTANT: Axis only depends on phi, not theta! Only compute for unique phi values
            phi_unique = phi2d[:, 0]  # Shape (nphi,) - axis derivatives same for all theta
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_xn_vals = self.axis_xn if self.axis_xn is not None else self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
            
            angles_axis = -axis_xn_vals[:, None] * phi_unique[None, :]
            cos_angles_axis = jnp.cos(angles_axis)
            sin_angles_axis = jnp.sin(angles_axis)
            
            # Position - compute only for unique phi values
            cos_phi = jnp.cos(phi_unique)
            sin_phi = jnp.sin(phi_unique)
            r_phi = jnp.sum(axis_rc[:, None] * cos_angles_axis, axis=0)  # Shape (nphi,)
            x = r_phi * cos_phi
            y = r_phi * sin_phi
            z = jnp.sum(axis_zs[:, None] * sin_angles_axis, axis=0)
            # Smooth regularization instead of jnp.maximum to avoid NaN gradients
            r = jnp.sqrt(x**2 + y**2 + 1e-24)
            
            # First derivatives
            dr_dphi_r = jnp.sum(axis_rc[:, None] * sin_angles_axis * axis_xn_vals[:, None], axis=0)
            
            dx_dphi = dr_dphi_r * cos_phi - r_phi * sin_phi
            dy_dphi = dr_dphi_r * sin_phi + r_phi * cos_phi
            dz_dphi = jnp.sum(-axis_zs[:, None] * cos_angles_axis * axis_xn_vals[:, None], axis=0)
            
            # dr/dφ = (x * dx/dφ + y * dy/dφ) / r
            dr_dphi = (x * dx_dphi + y * dy_dphi) / r
            
            # n1 = [x/r, y/r, 0]
            # dn1/dφ = [(dx/dφ * r - x * dr/dφ) / r², (dy/dφ * r - y * dr/dφ) / r², 0]
            r_safe_sq = r**2 + 1e-24
            r_safe_four = r_safe_sq**2
            dn1_x_dphi = (dx_dphi * r - x * dr_dphi) / r_safe_sq
            dn1_y_dphi = (dy_dphi * r - y * dr_dphi) / r_safe_sq
            dn1_dphi_1d = jnp.stack([dn1_x_dphi, dn1_y_dphi, jnp.zeros_like(x)], axis=-1)  # Shape (nphi, 3)
            # Broadcast to full grid: same for all theta values
            dn1_dphi = dn1_dphi_1d[:, None, :].repeat(self.ntheta, axis=1)  # Shape (nphi, ntheta, 3)
            
            # Second derivatives for second Frenet derivative
            d2r_dphi2_r = jnp.sum(-axis_rc[:, None] * cos_angles_axis * (axis_xn_vals[:, None]**2), axis=0)
            d2z_dphi2 = jnp.sum(axis_zs[:, None] * sin_angles_axis * (axis_xn_vals[:, None]**2), axis=0)
            
            d2x_dphi2 = d2r_dphi2_r * cos_phi - 2 * dr_dphi_r * sin_phi - r_phi * cos_phi
            d2y_dphi2 = d2r_dphi2_r * sin_phi + 2 * dr_dphi_r * cos_phi - r_phi * sin_phi
            
            # d²r/dφ²
            d2r_dphi2 = (x * d2x_dphi2 + y * d2y_dphi2 + dx_dphi**2 + dy_dphi**2) / r - (dr_dphi**2) / r
            
            # d²n1/dφ² using quotient rule
            numerator_x = dx_dphi * r - x * dr_dphi
            d_numerator_x_dphi = d2x_dphi2 * r + dx_dphi * dr_dphi - dr_dphi * dr_dphi - x * d2r_dphi2
            d2n1_x_dphi2 = (d_numerator_x_dphi * r_safe_sq - numerator_x * 2 * r * dr_dphi) / r_safe_four
            
            numerator_y = dy_dphi * r - y * dr_dphi
            d_numerator_y_dphi = d2y_dphi2 * r + dy_dphi * dr_dphi - dr_dphi * dr_dphi - y * d2r_dphi2
            d2n1_y_dphi2 = (d_numerator_y_dphi * r_safe_sq - numerator_y * 2 * r * dr_dphi) / r_safe_four
            
            d2n1_dphi2_1d = jnp.stack([d2n1_x_dphi2, d2n1_y_dphi2, jnp.zeros_like(x)], axis=-1)  # Shape (nphi, 3)
            # Broadcast to full grid
            d2n1_dphi2 = d2n1_dphi2_1d[:, None, :].repeat(self.ntheta, axis=1)  # Shape (nphi, ntheta, 3)
            
            # For n2 = n1 × T (right-handed), compute derivatives using cross product rule
            # T = [dx/dφ, dy/dφ, dz/dφ] / |dT|
            dT_mag = jnp.sqrt(dx_dphi**2 + dy_dphi**2 + dz_dphi**2 + 1e-28)
            T_x = dx_dphi / dT_mag
            T_y = dy_dphi / dT_mag
            T_z = dz_dphi / dT_mag
            
            # n1 components (1D along phi)
            n1_x = x / r
            n1_y = y / r
            n1_z = jnp.zeros_like(x)
            
            # dn1/dphi already computed via quotient rule above (dn1_x_dphi, dn1_y_dphi)
            dn1_z_dphi = jnp.zeros_like(phi_unique)
            
            # Compute dT/dφ using quotient rule
            d2T_mag = (dx_dphi * d2x_dphi2 + dy_dphi * d2y_dphi2 + dz_dphi * d2z_dphi2) / dT_mag
            
            dT_x_dphi = (d2x_dphi2 * dT_mag - dx_dphi * d2T_mag) / (dT_mag**2)
            dT_y_dphi = (d2y_dphi2 * dT_mag - dy_dphi * d2T_mag) / (dT_mag**2)
            dT_z_dphi = (d2z_dphi2 * dT_mag - dz_dphi * d2T_mag) / (dT_mag**2)
            
            # dn2/dφ = dn1/dφ × T + n1 × dT/dφ
            # First term: dn1/dφ × T
            dn2_x_dphi_term1 = dn1_y_dphi * T_z - dn1_z_dphi * T_y
            dn2_y_dphi_term1 = dn1_z_dphi * T_x - dn1_x_dphi * T_z
            dn2_z_dphi_term1 = dn1_x_dphi * T_y - dn1_y_dphi * T_x
            
            # Second term: n1 × dT/dφ
            dn2_x_dphi_term2 = n1_y * dT_z_dphi - n1_z * dT_y_dphi
            dn2_y_dphi_term2 = n1_z * dT_x_dphi - n1_x * dT_z_dphi
            dn2_z_dphi_term2 = n1_x * dT_y_dphi - n1_y * dT_x_dphi
            
            # Sum both terms
            dn2_x_dphi = dn2_x_dphi_term1 + dn2_x_dphi_term2
            dn2_y_dphi = dn2_y_dphi_term1 + dn2_y_dphi_term2
            dn2_z_dphi = dn2_z_dphi_term1 + dn2_z_dphi_term2
            
            dn2_dphi_1d = jnp.stack([dn2_x_dphi, dn2_y_dphi, dn2_z_dphi], axis=-1)  # Shape (nphi, 3)
            # Broadcast to full grid
            dn2_dphi = dn2_dphi_1d[:, None, :].repeat(self.ntheta, axis=1)  # Shape (nphi, ntheta, 3)
            
        # Compute derivatives of Frenet frame (dn1/dphi, dn2/dphi) analytically where possible
        if self._axis_shape == 'fourier':
            # Analytical derivatives for Fourier axis - already computed in the n1/n2 section above
            # Just extract them - they were computed inline, so compute d2n1/dphi2 and d2n2/dphi2 analytically too
            # For now, use simplified approach - can be enhanced if needed
            
            # Axis only depends on phi, not theta! Compute for unique phi values only
            phi_unique = phi2d[:, 0]  # Shape (nphi,) instead of (nphi*ntheta,)
            
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_xn_vals = self.axis_xn if self.axis_xn is not None else self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
            
            # Derivatives of position for unique phi values only
            angles_axis = -axis_xn_vals[:, None] * phi_unique[None, :]  # Shape (ntor, nphi)
            r_phi = jnp.sum(axis_rc[:, None] * jnp.cos(angles_axis), axis=0)  # Shape (nphi,)
            z_phi = jnp.sum(axis_zs[:, None] * jnp.sin(angles_axis), axis=0)  # Shape (nphi,)
            
            dr_dphi = jnp.sum(axis_rc[:, None] * jnp.sin(angles_axis) * axis_xn_vals[:, None], axis=0)  # Shape (nphi,)
            dz_dphi = jnp.sum(-axis_zs[:, None] * jnp.cos(angles_axis) * axis_xn_vals[:, None], axis=0)  # Shape (nphi,)
            
            d2r_dphi2 = jnp.sum(-axis_rc[:, None] * jnp.cos(angles_axis) * (axis_xn_vals[:, None]**2), axis=0)  # Shape (nphi,)
            d2z_dphi2 = jnp.sum(axis_zs[:, None] * jnp.sin(angles_axis) * (axis_xn_vals[:, None]**2), axis=0)  # Shape (nphi,)
            
            cos_phi = jnp.cos(phi_unique)  # Shape (nphi,)
            sin_phi = jnp.sin(phi_unique)  # Shape (nphi,)
            
            # Compute n1 and n2 derivatives analytically
            # n1 = [x/r, y/r, 0] where x = r_phi*cos(phi), y = r_phi*sin(phi), r = sqrt(x² + y²)
            # For Fourier axis, r_phi depends on phi, so we must use the quotient rule
            
            # First compute x, y, r with smooth regularization
            x = r_phi * cos_phi
            y = r_phi * sin_phi
            r_sq = x**2 + y**2
            r = jnp.sqrt(r_sq + 1e-24)  # Smooth regularization instead of jnp.maximum
            
            # dx/dphi = dr_phi/dphi * cos(phi) - r_phi * sin(phi)
            # dy/dphi = dr_phi/dphi * sin(phi) + r_phi * cos(phi)
            dx_dphi = dr_dphi * cos_phi - r_phi * sin_phi
            dy_dphi = dr_dphi * sin_phi + r_phi * cos_phi
            
            # dr/dphi = (x * dx/dphi + y * dy/dphi) / r
            dr_dphi_r = (x * dx_dphi + y * dy_dphi) / r
            
            # dn1/dphi using quotient rule: d(x/r)/dphi = (dx/dphi * r - x * dr/dphi) / r²
            r_safe_sq = r**2 + 1e-24
            r_safe_four = r_safe_sq**2
            dn1_x_dphi = (dx_dphi * r - x * dr_dphi_r) / r_safe_sq  # Shape (nphi,)
            dn1_y_dphi = (dy_dphi * r - y * dr_dphi_r) / r_safe_sq  # Shape (nphi,)
            dn1_z_dphi = jnp.zeros_like(phi_unique)  # Shape (nphi,)
            dn1_dphi_1d = jnp.stack([dn1_x_dphi, dn1_y_dphi, dn1_z_dphi], axis=-1)  # Shape (nphi, 3)
            # Broadcast to full grid (same for all theta)
            dn1_dphi = dn1_dphi_1d[:, None, :].repeat(self.ntheta, axis=1)  # Shape (nphi, ntheta, 3)
            
            # For n2, use finite differences (it's more complex analytically)
            eps = 1e-6
            dn2_dphi_fd = (n2.reshape(self.nphi * self.ntheta, 3)[self.ntheta:, :] - 
                          n2.reshape(self.nphi * self.ntheta, 3)[:-self.ntheta, :]) / (2 * eps)
            dn2_dphi = jnp.zeros_like(n2)
            
            # Second derivatives
            # d²x/d²φ = d²r_phi/dφ² * cos(φ) - 2 * dr_phi/dφ * sin(φ) - r_phi * cos(φ)
            # d²y/dφ² = d²r_phi/dφ² * sin(φ) + 2 * dr_phi/dφ * cos(φ) - r_phi * sin(φ)
            d2x_dphi2 = d2r_dphi2 * cos_phi - 2 * dr_dphi * sin_phi - r_phi * cos_phi
            d2y_dphi2 = d2r_dphi2 * sin_phi + 2 * dr_dphi * cos_phi - r_phi * sin_phi
            
            # d²r/dφ² = (x * d²x/dφ² + y * d²y/dφ² + (dx/dφ)² + (dy/dφ)²) / r - (dr/dφ)² / r
            d2r_dphi2_r = (x * d2x_dphi2 + y * d2y_dphi2 + dx_dphi**2 + dy_dphi**2) / r - (dr_dphi_r**2) / r
            
            # d²n1/dφ² using quotient rule on (dx/dφ * r - x * dr/dφ) / r²
            numerator_x = dx_dphi * r - x * dr_dphi_r
            d_numerator_x_dphi = d2x_dphi2 * r + dx_dphi * dr_dphi_r - dr_dphi_r * dr_dphi_r - x * d2r_dphi2_r
            d2n1_x_dphi2 = (d_numerator_x_dphi * r_safe_sq - numerator_x * 2 * r * dr_dphi_r) / r_safe_four
            
            numerator_y = dy_dphi * r - y * dr_dphi_r
            d_numerator_y_dphi = d2y_dphi2 * r + dy_dphi * dr_dphi_r - dr_dphi_r * dr_dphi_r - y * d2r_dphi2_r
            d2n1_y_dphi2 = (d_numerator_y_dphi * r_safe_sq - numerator_y * 2 * r * dr_dphi_r) / r_safe_four
            
            d2n1_z_dphi2 = jnp.zeros_like(phi_unique)  # Shape (nphi,)
            d2n1_dphi2_1d = jnp.stack([d2n1_x_dphi2, d2n1_y_dphi2, d2n1_z_dphi2], axis=-1)  # Shape (nphi, 3)
            # Broadcast to full grid (same for all theta)
            d2n1_dphi2 = d2n1_dphi2_1d[:, None, :].repeat(self.ntheta, axis=1)  # Shape (nphi, ntheta, 3)
            d2n2_dphi2 = jnp.zeros_like(n2)
            
        elif self._axis_shape == 'circle':
            # Analytical derivatives for circle
            phi_flat = phi2d.flatten()
            cos_phi = jnp.cos(phi_flat)
            sin_phi = jnp.sin(phi_flat)
            
            # n1 = [cos(phi), sin(phi), 0]
            # dn1/dphi = [-sin(phi), cos(phi), 0]
            dn1_x_dphi = -sin_phi
            dn1_y_dphi = cos_phi
            dn1_z_dphi = jnp.zeros_like(phi_flat)
            dn1_dphi_flat = jnp.stack([dn1_x_dphi, dn1_y_dphi, dn1_z_dphi], axis=-1)
            dn1_dphi = dn1_dphi_flat.reshape(self.nphi, self.ntheta, 3)
            
            # n2 = [0, 0, -1] (from n1 × tangent for circle)
            # dn2/dphi = 0
            dn2_dphi = jnp.zeros_like(n2)
            
            # Second derivatives
            d2n1_x_dphi2 = -cos_phi
            d2n1_y_dphi2 = -sin_phi
            d2n1_z_dphi2 = jnp.zeros_like(phi_flat)
            d2n1_dphi2_flat = jnp.stack([d2n1_x_dphi2, d2n1_y_dphi2, d2n1_z_dphi2], axis=-1)
            d2n1_dphi2 = d2n1_dphi2_flat.reshape(self.nphi, self.ntheta, 3)
            d2n2_dphi2 = jnp.zeros_like(n2)
            
        else:
            # For other axes, use finite differences without vmap
            # We compute derivatives by finite differences on the already-computed n1, n2
            eps = 1e-6
            
            # Reshape to (nphi*ntheta, 3) for easier finite differencing
            n1_flat = n1.reshape(self.nphi * self.ntheta, 3)
            n2_flat = n2.reshape(self.nphi * self.ntheta, 3)
            
            # Compute dn1/dphi using finite differences between adjacent phi slices
            # This assumes nphi > 1
            dn1_dphi_flat = jnp.roll(n1_flat, -self.ntheta, axis=0) - jnp.roll(n1_flat, self.ntheta, axis=0)
            dn1_dphi_flat = dn1_dphi_flat / (2 * eps)
            dn1_dphi = dn1_dphi_flat.reshape(self.nphi, self.ntheta, 3)
            
            # Compute dn2/dphi similarly
            dn2_dphi_flat = jnp.roll(n2_flat, -self.ntheta, axis=0) - jnp.roll(n2_flat, self.ntheta, axis=0)
            dn2_dphi_flat = dn2_dphi_flat / (2 * eps)
            dn2_dphi = dn2_dphi_flat.reshape(self.nphi, self.ntheta, 3)
            
            # Second derivatives using second-order central difference
            d2n1_dphi2_flat = jnp.roll(n1_flat, -2*self.ntheta, axis=0) - 2*n1_flat + jnp.roll(n1_flat, 2*self.ntheta, axis=0)
            d2n1_dphi2_flat = d2n1_dphi2_flat / ((eps)**2)
            d2n1_dphi2 = d2n1_dphi2_flat.reshape(self.nphi, self.ntheta, 3)
            
            d2n2_dphi2_flat = jnp.roll(n2_flat, -2*self.ntheta, axis=0) - 2*n2_flat + jnp.roll(n2_flat, 2*self.ntheta, axis=0)
            d2n2_dphi2_flat = d2n2_dphi2_flat / ((eps)**2)
            d2n2_dphi2 = d2n2_dphi2_flat.reshape(self.nphi, self.ntheta, 3)
        
        # Second derivative of axis curve with respect to phi
        if self._axis_shape == 'fourier':
            # Analytical second derivative for Fourier axis
            phi_flat = phi2d.flatten()
            axis_rc = self._axis_rc_coeffs if self._axis_rc_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_zs = self._axis_zs_coeffs if self._axis_zs_coeffs is not None else jnp.zeros(2 * self.axis_ntor + 1)
            axis_xn_vals = self.axis_xn if self.axis_xn is not None else self.nfp * jnp.arange(-self.axis_ntor, self.axis_ntor + 1)
            
            angles_axis = -axis_xn_vals[:, None] * phi_flat[None, :]
            r_phi = jnp.sum(axis_rc[:, None] * jnp.cos(angles_axis), axis=0)
            z_phi = jnp.sum(axis_zs[:, None] * jnp.sin(angles_axis), axis=0)
            
            # First derivatives
            dr_dphi = jnp.sum(axis_rc[:, None] * jnp.sin(angles_axis) * axis_xn_vals[:, None], axis=0)
            dz_dphi = jnp.sum(-axis_zs[:, None] * jnp.cos(angles_axis) * axis_xn_vals[:, None], axis=0)
            
            # Second derivatives: d²r/dφ² and d²z/dφ²
            d2r_dphi2 = jnp.sum(-axis_rc[:, None] * jnp.cos(angles_axis) * (axis_xn_vals[:, None]**2), axis=0)
            d2z_dphi2 = jnp.sum(axis_zs[:, None] * jnp.sin(angles_axis) * (axis_xn_vals[:, None]**2), axis=0)
            
            cos_phi = jnp.cos(phi_flat)
            sin_phi = jnp.sin(phi_flat)
            
            # d²x/dφ² = d²r/dφ² * cos(φ) - 2*(dr/dφ)*sin(φ) - r*cos(φ)
            # d²y/dφ² = d²r/dφ² * sin(φ) + 2*(dr/dφ)*cos(φ) - r*sin(φ)
            d2x_dphi2 = d2r_dphi2 * cos_phi - 2 * dr_dphi * sin_phi - r_phi * cos_phi
            d2y_dphi2 = d2r_dphi2 * sin_phi + 2 * dr_dphi * cos_phi - r_phi * sin_phi
            
            d2_axis_flat = jnp.stack([d2x_dphi2, d2y_dphi2, d2z_dphi2], axis=-1)
            d2_axis_dphi2 = d2_axis_flat.reshape(self.nphi, self.ntheta, 3)
        else:
            # Finite differences for other axis types
            eps_axis = 1e-5
                # For custom axes, use finite differences on already-computed d_axis_curve
            d_axis_flat = d_axis_curve.reshape(self.nphi * self.ntheta, 3)
            d2_axis_dphi2_flat = (jnp.roll(d_axis_flat, -self.ntheta, axis=0) - 
            jnp.roll(d_axis_flat, self.ntheta, axis=0)) / (2 * eps_axis)
            d2_axis_dphi2 = d2_axis_dphi2_flat.reshape(self.nphi, self.ntheta, 3)
        
        # Tangent vectors
        # d/dθ [r(θ, φ)] = dX1/dθ * n1 + X1 * dn1/dθ + dX2/dθ * n2 + X2 * dn2/dθ
        # For now, assume dn1/dθ ≈ 0 and dn2/dθ ≈ 0 (they only depend on φ)
        gammadash_theta = jnp.einsum('jk,jki->jki', dX1_dtheta, n1) + \
                         jnp.einsum('jk,jki->jki', dX2_dtheta, n2)
        
        # d/dφ [r(θ, φ)] = d_axis/dφ + dX1/dφ * n1 + X1 * dn1/dφ + dX2/dφ * n2 + X2 * dn2/dφ
        gammadash_phi = d_axis_curve + \
                       jnp.einsum('jk,jki->jki', dX1_dphi, n1) + \
                       jnp.einsum('jk,jki->jki', X1_expanded[..., 0], dn1_dphi) + \
                       jnp.einsum('jk,jki->jki', dX2_dphi, n2) + \
                       jnp.einsum('jk,jki->jki', X2_expanded[..., 0], dn2_dphi)
        
        # Scale by proper Jacobian factors (2π factor for [0,1] -> [0,2π] parameterization)
        # Both theta and phi need 2π scaling to match RZ surface convention
        gammadash_theta = (2 * jnp.pi) * gammadash_theta
        gammadash_phi = (2 * jnp.pi) * gammadash_phi
        
        # d²X1 and d²X2 with respect to theta
        d2X1_dtheta2 = jnp.einsum('i,ijk->jk', -self.xm**2 * self.x1_coeffs, cos_angles)
        d2X2_dtheta2 = jnp.einsum('i,ijk->jk', -self.xm**2 * self.x2_coeffs, sin_angles)
        
        # Second derivative with respect to theta
        # d²/dθ² [r(θ, φ)] = d²X1/dθ² * n1 + d²X2/dθ² * n2
        gammadashdash_theta = jnp.einsum('jk,jki->jki', d2X1_dtheta2, n1) + \
                             jnp.einsum('jk,jki->jki', d2X2_dtheta2, n2)
        gammadashdash_theta = ((2 * jnp.pi)**2) * gammadashdash_theta
        
        # d²X1 and d²X2 with respect to phi
        # ∂²X1/∂φ² = Σᵢ nᵢ² * x1ᵢ * cos(mᵢθ - nᵢφ)
        # ∂²X2/∂φ² = Σᵢ nᵢ² * x2ᵢ * sin(mᵢθ - nᵢφ)
        d2X1_dphi2 = jnp.einsum('i,ijk->jk', self.xn**2 * self.x1_coeffs, cos_angles)
        d2X2_dphi2 = jnp.einsum('i,ijk->jk', self.xn**2 * self.x2_coeffs, sin_angles)
        
        # Second derivative with respect to phi
        # d²/dφ² [r(θ, φ)] = d²_axis/dφ² + d²X1/dφ² * n1 + 2*(dX1/dφ * dn1/dφ) + X1 * d²n1/dφ² 
        #                     + d²X2/dφ² * n2 + 2*(dX2/dφ * dn2/dφ) + X2 * d²n2/dφ²
        gammadashdash_phi = d2_axis_dphi2 + \
                           jnp.einsum('jk,jki->jki', d2X1_dphi2, n1) + \
                           2 * jnp.einsum('jk,jki->jki', dX1_dphi, dn1_dphi) + \
                           jnp.einsum('jk,jki->jki', X1_expanded[..., 0], d2n1_dphi2) + \
                           jnp.einsum('jk,jki->jki', d2X2_dphi2, n2) + \
                           2 * jnp.einsum('jk,jki->jki', dX2_dphi, dn2_dphi) + \
                           jnp.einsum('jk,jki->jki', X2_expanded[..., 0], d2n2_dphi2)
        
        # d²X1 and d²X2 with respect to both theta and phi
        # ∂²X1/(∂θ∂φ) = Σᵢ mᵢ*nᵢ * x1ᵢ * sin(mᵢθ - nᵢφ)
        # ∂²X2/(∂θ∂φ) = Σᵢ mᵢ*nᵢ * x2ᵢ * cos(mᵢθ - nᵢφ)
        d2X1_dthetadphi = jnp.einsum('i,ijk->jk', self.xm * self.xn * self.x1_coeffs, sin_angles)
        d2X2_dthetadphi = jnp.einsum('i,ijk->jk', self.xm * self.xn * self.x2_coeffs, cos_angles)
        
        # Mixed derivative with respect to theta and phi
        # d²/dθdφ [r(θ, φ)] = d²X1/(dθdφ) * n1 + (dX1/dθ * dn1/dφ) + d²X2/(dθdφ) * n2 + (dX2/dθ * dn2/dφ)
        gammadashdash_theta_phi = jnp.einsum('jk,jki->jki', d2X1_dthetadphi, n1) + \
                                 jnp.einsum('jk,jki->jki', dX1_dtheta, dn1_dphi) + \
                                 jnp.einsum('jk,jki->jki', d2X2_dthetadphi, n2) + \
                                 jnp.einsum('jk,jki->jki', dX2_dtheta, dn2_dphi)
        gammadashdash_theta_phi = ((2 * jnp.pi)**2) * gammadashdash_theta_phi
        
        # Return surface and derivatives
        return gamma, gammadash_theta, gammadash_phi, gammadashdash_theta, gammadashdash_phi, gammadashdash_theta_phi
    
    @property
    def gamma(self):
        """Surface position in Cartesian coordinates."""
        if self._gamma is None:
            result = self._compute_gamma()
            self._gamma = result[0]
            self._gammadash_theta = result[1]
            self._gammadash_phi = result[2]
            self._gammadashdash_theta = result[3]
            self._gammadashdash_phi = result[4]
            self._gammadashdash_theta_phi = result[5]
        return self._gamma
    
    @property
    def gammadash_theta(self):
        """Derivative of surface position with respect to poloidal angle."""
        if self._gammadash_theta is None:
            result = self._compute_gamma()
            self._gamma = result[0]
            self._gammadash_theta = result[1]
            self._gammadash_phi = result[2]
            self._gammadashdash_theta = result[3]
            self._gammadashdash_phi = result[4]
            self._gammadashdash_theta_phi = result[5]
        return self._gammadash_theta
    
    @property
    def gammadash_phi(self):
        """Derivative of surface position with respect to toroidal angle."""
        if self._gammadash_phi is None:
            result = self._compute_gamma()
            self._gamma = result[0]
            self._gammadash_theta = result[1]
            self._gammadash_phi = result[2]
            self._gammadashdash_theta = result[3]
            self._gammadashdash_phi = result[4]
            self._gammadashdash_theta_phi = result[5]
        return self._gammadash_phi
    
    @property
    def gammadashdash_theta(self):
        """Second derivative of surface position with respect to poloidal angle."""
        if self._gammadashdash_theta is None:
            result = self._compute_gamma()
            self._gamma = result[0]
            self._gammadash_theta = result[1]
            self._gammadash_phi = result[2]
            self._gammadashdash_theta = result[3]
            self._gammadashdash_phi = result[4]
            self._gammadashdash_theta_phi = result[5]
        return self._gammadashdash_theta
    
    @property
    def gammadashdash_phi(self):
        """Second derivative of surface position with respect to toroidal angle."""
        if self._gammadashdash_phi is None:
            result = self._compute_gamma()
            self._gamma = result[0]
            self._gammadash_theta = result[1]
            self._gammadash_phi = result[2]
            self._gammadashdash_theta = result[3]
            self._gammadashdash_phi = result[4]
            self._gammadashdash_theta_phi = result[5]
        return self._gammadashdash_phi
    
    @property
    def gammadashdash_theta_phi(self):
        """Mixed second derivative of surface position."""
        if self._gammadashdash_theta_phi is None:
            result = self._compute_gamma()
            self._gamma = result[0]
            self._gammadash_theta = result[1]
            self._gammadash_phi = result[2]
            self._gammadashdash_theta = result[3]
            self._gammadashdash_phi = result[4]
            self._gammadashdash_theta_phi = result[5]
        return self._gammadashdash_theta_phi
    
    # Surface properties
    @jit
    def _compute_properties(self):
        """Compute normal vector and area element with smooth regularization."""
        # Use standard cross product order: phi × theta gives outward normal
        normal = jnp.cross(self.gammadash_phi, self.gammadash_theta, axis=2)
        # Compute area element directly from normal magnitude (matches RZFourier approach)
        area_element = jnp.linalg.norm(normal, axis=2)
        # Smooth regularization for normalized normal: use smooth division
        normal_norm_sq = jnp.sum(normal**2, axis=2, keepdims=True)
        normal_norm = jnp.sqrt(normal_norm_sq + 1e-28)  # Smooth regularization
        unitnormal = normal / normal_norm
        return normal, unitnormal, area_element
    
    @property
    def normal(self):
        if self._normal is None:
            self._normal, self._unitnormal, self._area_element = self._compute_properties()
        return self._normal
    
    @property
    def unitnormal(self):
        if self._unitnormal is None:
            self._normal, self._unitnormal, self._area_element = self._compute_properties()
        return self._unitnormal
    
    @property
    def area_element(self):
        if self._area_element is None:
            self._normal, self._unitnormal, self._area_element = self._compute_properties()
        return self._area_element
    
    @property
    def area(self):
        """Total surface area."""
        n = self.normal
        norm_n = jnp.linalg.norm(n, axis=2)
        dphi = 2 * jnp.pi / self.nphi
        dtheta = 2 * jnp.pi / self.ntheta
        area = jnp.sum(norm_n) * dphi * dtheta
        return area
    
    @property
    def volume(self):
        """Volume enclosed by the surface."""
        xyz = self.gamma
        n = self.normal
        integrand = jnp.sum(xyz * n, axis=2)
        volume = jnp.sum(integrand) / (3.0 * self.nphi * self.ntheta)
        return volume
    
    # Utility properties
    @property
    def x(self):
        """Alias for dofs."""
        return self.dofs
    
    @x.setter
    def x(self, new_dofs):
        self.dofs = new_dofs
    
    def plot(self, ax=None, show=True, axis_equal=True, **kwargs):
        """Plot the surface in 3D."""
        if ax is None or ax.name != "3d":
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        
        kwargs.setdefault('alpha', 0.6)
        boundary = self.gamma
        ax.plot_surface(boundary[:, :, 0], boundary[:, :, 1], boundary[:, :, 2], 
                       linewidth=0, antialiased=True, **kwargs)
        ax.grid(False)
        
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    
    def to_vtk(self, filename, extra_data=None, field=None):
        """Export surface to VTK format for visualization in ParaView."""
        try: 
            import numpy as np
        except ImportError: 
            raise ImportError("The 'numpy' library is required. Please install it using 'pip install numpy'.")
        try: 
            from pyevtk.hl import gridToVTK
        except ImportError: 
            raise ImportError("The 'pyevtk' library is required. Please install it using 'pip install pyevtk'.")
        
        boundary = np.array(self.gamma)
        x = boundary[:, :, 0].reshape((1, self.nphi, self.ntheta)).copy()
        y = boundary[:, :, 1].reshape((1, self.nphi, self.ntheta)).copy()
        z = boundary[:, :, 2].reshape((1, self.nphi, self.ntheta)).copy()
        
        pointData = {}
        if field is not None:
            B_dot_n_over_B = np.array(BdotN_over_B(self, field)).reshape((1, self.nphi, self.ntheta)).copy()
            pointData["B_dot_n_over_B"] = B_dot_n_over_B
            B_BiotSavart = np.array(vmap(lambda surf: vmap(lambda x: field.AbsB(x))(surf))(boundary)).reshape((1, self.nphi, self.ntheta)).copy()
            pointData["B_BiotSavart"] = B_BiotSavart
        if extra_data is not None:
            pointData = {**pointData, **extra_data}
        
        gridToVTK(str(filename), x, y, z, pointData=pointData)
    
    def _tree_flatten(self):
        # Variable pytree structure based on enable_axis_dofs
        # This keeps DOF sizes correct for optimization
        if self._enable_axis_dofs and self._axis_rc_coeffs is not None and self._axis_zs_coeffs is not None:
            children = (self._x1_coeffs, self._x2_coeffs, self._axis_rc_coeffs, self._axis_zs_coeffs)
        else:
            children = (self._x1_coeffs, self._x2_coeffs)
        aux_data = {
            "nfp": self._nfp,
            "mpol": self._mpol,
            "ntor": self._ntor,
            "ntheta": self._ntheta,
            "nphi": self._nphi,
            "close": self._close,
            "range_torus": self._range_torus,
            "axis_major_radius": self._axis_major_radius,
            "axis_shape": self._axis_shape,
            "scaling_type": self._scaling_type,
            "scaling_factor": self._scaling_factor,
            "scale_fixed": self._scale_fixed,
            "axis_ntor": self._axis_ntor,
        }
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # IMPORTANT: do not mutate aux_data.
        # JAX may reuse the cached unravel closure, and mutating this dict
        # (via pop) can corrupt subsequent unflatten calls.
        aux = dict(aux_data)
        axis_ntor = aux.pop("axis_ntor", 0)

        # Determine enable_axis_dofs based on the number of children
        # If we have 4 children, axis DOFs were flattened, so enable_axis_dofs must be True
        # If we have 2 children, axis DOFs were not flattened, so disable them
        if len(children) == 4:
            x1_coeffs, x2_coeffs, axis_rc_coeffs, axis_zs_coeffs = children
            enable_axis_dofs = True
        else:
            x1_coeffs, x2_coeffs = children
            # When axis DOFs are disabled, initialize axis coefficients properly
            # to avoid JAX type errors during array conversion
            n_axis_modes = 2 * axis_ntor + 1
            axis_rc_coeffs = jnp.zeros(n_axis_modes)
            axis_zs_coeffs = jnp.zeros(n_axis_modes)
            enable_axis_dofs = False

        return cls(
            x1_coeffs=x1_coeffs,
            x2_coeffs=x2_coeffs,
            enable_axis_dofs=enable_axis_dofs,
            axis_rc_coeffs=axis_rc_coeffs,
            axis_zs_coeffs=axis_zs_coeffs,
            axis_ntor=axis_ntor,
            **aux
        )

tree_util.register_pytree_node(SurfaceGFrameFourier,
                               SurfaceGFrameFourier._tree_flatten,
                               SurfaceGFrameFourier._tree_unflatten)


#This class is based on simsopt classifier but translated to fit jax    
class SurfaceClassifier():
    """
    Takes in a toroidal surface and constructs an interpolant of the signed distance function
    :math:`f:R^3\to R` that is positive inside the volume contained by the surface,
    (approximately) zero on the surface, and negative outisde the volume contained by the surface.
    """

    def __init__(self, surface,h=0.05):
        """
        Args:
            surface: the surface to contruct the distance from.
            h: grid resolution of the interpolant
        """
        gammas = surface.gamma
        r = jnp.linalg.norm(gammas[:, :, :2], axis=2)
        z = gammas[:, :, 2]
        rmin = max(jnp.min(r) - 0.1, 0.)
        rmax = jnp.max(r) + 0.1
        zmin = jnp.min(z) - 0.1
        zmax = jnp.max(z) + 0.1

        self.zrange = (zmin, zmax)
        self.rrange = (rmin, rmax)

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*jnp.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)

        def fbatch(rs, phis, zs):
            xyz = jnp.zeros(( 3))
            xyz=xyz.at[0].set( rs * jnp.cos(phis))
            xyz=xyz.at[1].set(rs * jnp.sin(phis))
            xyz=xyz.at[2].set(zs)
            return signed_distance_from_surface_jax(xyz, surface)   
            #return signed_distance_from_surface_extras(xyz, surface) ####memory bounded

        #rule = sopp.UniformInterpolationRule(p) 
        #self.dist = RegularGridInterpolator((jnp.linspace(rmin,rmax,nr),
        #            jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz)),
        #            vmap(vmap(vmap(fbatch,in_axes=(0,None,None)),in_axes=(None,0,None)),in_axes=(None,None,0))(jnp.linspace(rmin,rmax,nr),
        #            jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz)))
        #self.r_list=jnp.linspace(16.9,17.1,nr)
        #self.phi_list=jnp.linspace(0., 0.01, nphi)
        #self.z_list=jnp.linspace(-0.1, 0.1, nz)
        #self.test= vmap(vmap(vmap(fbatch,in_axes=(0,None,None)),in_axes=(None,0,None)),in_axes=(None,None,0))(self.r_list,
        #            self.phi_list, self.z_list)
        #self.r_list=jnp.linspace(rmin,rmax,nr)
        #self.phi_list=jnp.linspace(0., 2*jnp.pi, nphi)
        #self.z_list=jnp.linspace(zmin, zmax, nz)
        #self.test= vmap(vmap(vmap(fbatch,in_axes=(None,None,0)),in_axes=(None,0,None)),in_axes=(0,None,None))(jnp.linspace(rmin,rmax,nr),
        #            jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz))
        #self.dist = RegularGridInterpolator((self.r_list,self.phi_list, self.z_list),
        #            vmap(vmap(vmap(fbatch,in_axes=(None,None,0)),in_axes=(None,0,None)),in_axes=(0,None,None))(self.r_list,self.phi_list, self.z_list),fill_value=-1.)        
        self.dist = RegularGridInterpolator((jnp.linspace(rmin,rmax,nr),
                    jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz)),
                    vmap(vmap(vmap(fbatch,in_axes=(None,None,0)),in_axes=(None,0,None)),in_axes=(0,None,None))(jnp.linspace(rmin,rmax,nr),
                    jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz)),fill_value=-1.)
        #self.dist.interpolate_batch(fbatch)    

    @partial(jit, static_argnames=['self'])
    def evaluate_xyz(self, xyz):
        rphiz = jnp.zeros_like(xyz)
        rphiz=rphiz.at[0].set(jnp.linalg.norm(xyz[:2]))
        rphiz=rphiz.at[1].set(jnp.mod(jnp.arctan2(xyz[1], xyz[0]), 2*jnp.pi))
        rphiz=rphiz.at[2].set(xyz.at[2].get())
        # initialize to -1 since the regular grid interpolant will just keep
        # that value when evaluated outside of bounds
        d=self.dist(rphiz)[0][0]
        return d

    @partial(jit, static_argnames=['self'])
    def evaluate_rphiz(self, rphiz):
        # initialize to -1 since the regular grid interpolant will just keep
        # that value when evaluated outside of bounds
        d=self.dist(rphiz)[0][0]
        return d
    

partial(jit, static_argnames=['surface'])
def signed_distance_from_surface_jax(xyz, surface):
    """
    Compute the signed distances from points ``xyz`` to a surface.  The sign is
    positive for points inside the volume surrounded by the surface.
    """
    gammas = surface.gamma.reshape((-1, 3))
    #from scipy.spatial import KDTree ##better for cpu?
    tree = jaxkd.build_tree(gammas)
    mins, _ = jaxkd.query_neighbors(tree, xyz, k=1)    
    n = surface.unitnormal.reshape((-1, 3))
    nmins = n[mins]
    gammamins = gammas[mins]
    # Now that we have found the closest node, we approximate the surface with
    # a plane through that node with the appropriate normal and then compute
    # the distance from the point to that plane
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    mindist = jnp.sum((xyz-gammamins) * nmins, axis=1)
    a_point_in_the_surface = jnp.mean(surface.gamma[0, :, :], axis=0)
    sign_of_interiorpoint = jnp.sign(jnp.sum((a_point_in_the_surface-gammas[0, :])*n[0, :]))
    signed_dists = mindist * sign_of_interiorpoint
    return signed_dists

#@partial(jit, static_argnames=['surface'])
def signed_distance_from_surface_extras(xyz, surface):
    """
    Compute the signed distances from points ``xyz`` to a surface.  The sign is
    positive for points inside the volume surrounded by the surface.
    """
    gammas = surface.gamma.reshape((-1, 3))
    mins, _ = jaxkd.extras.query_neighbors_pairwise(gammas, xyz, k=1)    
    n = surface.unitnormal.reshape((-1, 3))
    nmins = n[mins]
    gammamins = gammas[mins]
    # Now that we have found the closest node, we approximate the surface with
    # a plane through that node with the appropriate normal and then compute
    # the distance from the point to that plane
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    mindist = jnp.sum((xyz-gammamins) * nmins, axis=1)
    a_point_in_the_surface = jnp.mean(surface.gamma[0, :, :], axis=0)
    sign_of_interiorpoint = jnp.sign(jnp.sum((a_point_in_the_surface-gammas[0, :])*n[0, :]))
    signed_dists = mindist * sign_of_interiorpoint
    return signed_dists



def plot_scalar_on_flux_surface(surface, scalar_map):
    '''
        surface: the surface object in which to plot the scalar_map
        scalar_map: a scalar_map as function of theta and phi
    ''' 



