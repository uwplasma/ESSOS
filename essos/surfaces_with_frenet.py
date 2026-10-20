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

# @jit
@partial(pjit, in_shardings=(sharding, None), out_shardings=sharding)
def B_on_surface(surface, field):
    ntheta = surface.ntheta
    nphi = surface.nphi
    gamma = surface.gamma
    gamma_reshaped = gamma.reshape(nphi * ntheta, 3)

    # Map field.B over all positions
    B_on_surface = vmap(field.B)(gamma_reshaped)

    return B_on_surface.reshape(nphi, ntheta, 3)
    

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
        print(angles.shape)
        sin_angles = jnp.sin(angles)
        cos_angles = jnp.cos(angles)
        phi2d = self.phi2d
        sin_phi2d = jnp.sin(phi2d)
        cos_phi2d = jnp.cos(phi2d)
        rc = self.rc; zs = self.zs; xm = self.xm; xn = self.xn
        #rmnc_interp = self.rc; zmns_interp = self.zs; xm = self.xm; xn = self.xn
        
        print(rc.shape, cos_angles.shape)
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
            Shape of the axis: 'circle', 'ellipse', 'helical', or 'custom' (default: 'circle')
        axis_pitch : float, optional
            Pitch of helical axis (vertical rise per 2π toroidal turn) (default: 0.0)
        axis_twist_rate : float, optional
            Twist rate of the Frenet frame along the axis (radians per toroidal turn) (default: 0.0)
        axis_function : callable, optional
            Custom axis function r(phi) -> [x, y, z]. Only used if axis_shape == 'custom'.
            Must be JAX-compatible (use jnp operations). Function signature: (phi,) -> (3,)
            Example: lambda phi: jnp.array([R*cos(phi), R*sin(phi) + A*sin(2*phi), B*sin(phi)])
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
    
    @property
    def dofs(self):
        """Degrees of freedom: combined X1 and X2 coefficients with scaling."""
        return jnp.hstack([self.x1_coeffs * self.scaling, self.x2_coeffs * self.scaling])
    
    @dofs.setter
    def dofs(self, new_dofs):
        """Set degrees of freedom."""
        n1 = len(self.x1_coeffs)
        self._x1_coeffs = new_dofs[:n1] / self.scaling
        self._x2_coeffs = new_dofs[n1:] / self.scaling
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
        - 'custom': User-defined axis function
        """
        if self._axis_shape == 'circle':
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
        # Compute the tangent vector (derivative of axis curve)
        # Use a more robust epsilon for numerical differentiation
        eps = 1e-6  # Slightly larger than machine epsilon for better numerical stability
        axis_plus = self._compute_axis_curve(phi + eps)
        axis_minus = self._compute_axis_curve(phi - eps)
        tangent = (axis_plus - axis_minus) / (2 * eps)
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
        
        # Second normal: perpendicular to both tangent and n1 (via cross product)
        n2 = jnp.cross(tangent, n1, axis=-1)
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
        
        # Handle custom axes specially to avoid vmap closure issues
        if self._axis_shape == 'custom':
            # For custom axes, call the function directly without vmap
            axis_result = self._axis_function(phi2d.flatten())
            if axis_result.ndim == 2 and axis_result.shape[1] == 3:
                axis_curve = axis_result
            else:
                axis_curve = axis_result.T
            axis_curve = axis_curve.reshape(self.nphi, self.ntheta, 3)
            
            # For Frenet frame with custom axes, compute using robust method
            phi_flat = phi2d.flatten()
            n_pts = phi_flat.shape[0]
            
            # Compute tangent vector using JAX automatic differentiation
            from jax import jacfwd
            tangent_func = jacfwd(self._axis_function)
            tangent = vmap(tangent_func)(phi_flat)  # Shape: (n_pts, 3)
            tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
            tangent_normalized = tangent / jnp.maximum(tangent_norm, 1e-14)
            
            # Compute initial normal n1 using a robust approach
            axis = self._axis_function(phi_flat)
            
            # Reference vectors to try
            ref_z = jnp.zeros((n_pts, 3))
            ref_z = ref_z.at[:, 2].set(1.0)  # [0, 0, 1]
            
            # Compute n1 as perpendicular to tangent using Gram-Schmidt
            dot_product = jnp.sum(ref_z * tangent_normalized, axis=-1, keepdims=True)
            n1 = ref_z - dot_product * tangent_normalized
            n1_norm = jnp.linalg.norm(n1, axis=-1, keepdims=True)
            
            # If n1 is too small, use a different reference vector [1, 0, 0]
            is_small = n1_norm[:, 0] < 1e-6
            ref_x = jnp.zeros((n_pts, 3))
            ref_x = ref_x.at[:, 0].set(1.0)
            dot_product_x = jnp.sum(ref_x * tangent_normalized, axis=-1, keepdims=True)
            n1_alt = ref_x - dot_product_x * tangent_normalized
            n1_alt_norm = jnp.linalg.norm(n1_alt, axis=-1, keepdims=True)
            n1 = jnp.where(is_small[:, None], n1_alt, n1)
            n1_norm = jnp.where(is_small[:, None], n1_alt_norm, n1_norm)
            
            n1_normalized = n1 / jnp.maximum(n1_norm, 1e-14)
            
            # Compute n2 = T × n1
            n2 = jnp.cross(tangent_normalized, n1_normalized)
            n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)
            n2_normalized = n2 / jnp.maximum(n2_norm, 1e-14)
            
            # Apply twist rate if specified
            if self._axis_twist_rate != 0.0:
                twist_angles = self._axis_twist_rate * phi_flat
                c = jnp.cos(twist_angles)
                s = jnp.sin(twist_angles)
                n1_twisted = c[:, None] * n1_normalized + s[:, None] * n2_normalized
                n2_twisted = -s[:, None] * n1_normalized + c[:, None] * n2_normalized
                n1_normalized = n1_twisted
                n2_normalized = n2_twisted
            
            n1 = n1_normalized.reshape(self.nphi, self.ntheta, 3)
            n2 = n2_normalized.reshape(self.nphi, self.ntheta, 3)
            
            # Compute derivatives of axis curve and Frenet frame for custom axes
            # d/dphi of axis curve (using jacfwd)
            axis_flat = self._axis_function(phi_flat)
            d_axis_func = jacfwd(self._axis_function)
            d_axis_flat = vmap(d_axis_func)(phi_flat)
            d_axis_curve = d_axis_flat.reshape(self.nphi, self.ntheta, 3)
        else:
            # For standard axes, use vmap
            axis_curve = vmap(self._compute_axis_curve)(phi2d.flatten())
            axis_curve = axis_curve.reshape(self.nphi, self.ntheta, 3)
            
            n1, n2 = vmap(self._compute_frenet_frame)(phi2d.flatten())
            n1 = n1.reshape(self.nphi, self.ntheta, 3)
            n2 = n2.reshape(self.nphi, self.ntheta, 3)
            
            # Compute derivative of axis curve using finite differences
            # (more efficient than vmap + jacfwd for this case)
            eps = 1e-8
            phi_flat = phi2d.flatten()
            axis_plus = vmap(self._compute_axis_curve)(phi_flat + eps)
            axis_minus = vmap(self._compute_axis_curve)(phi_flat - eps)
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
        
        # Derivatives of Frenet frame with respect to phi (using finite differences)
        eps = 1e-8
        phi2d_plus = phi2d + eps
        phi2d_minus = phi2d - eps
        
        n1_plus, n2_plus = vmap(self._compute_frenet_frame)(phi2d_plus.flatten())
        n1_plus = n1_plus.reshape(self.nphi, self.ntheta, 3)
        n2_plus = n2_plus.reshape(self.nphi, self.ntheta, 3)
        
        n1_minus, n2_minus = vmap(self._compute_frenet_frame)(phi2d_minus.flatten())
        n1_minus = n1_minus.reshape(self.nphi, self.ntheta, 3)
        n2_minus = n2_minus.reshape(self.nphi, self.ntheta, 3)
        
        dn1_dphi = (n1_plus - n1_minus) / (2 * eps)
        dn2_dphi = (n2_plus - n2_minus) / (2 * eps)
        
        # Second derivatives of Frenet frame with respect to phi
        eps2 = 1e-7
        phi2d_plus2 = phi2d + eps2
        phi2d_minus2 = phi2d - eps2
        
        n1_plus2, n2_plus2 = vmap(self._compute_frenet_frame)(phi2d_plus2.flatten())
        n1_plus2 = n1_plus2.reshape(self.nphi, self.ntheta, 3)
        n2_plus2 = n2_plus2.reshape(self.nphi, self.ntheta, 3)
        
        n1_minus2, n2_minus2 = vmap(self._compute_frenet_frame)(phi2d_minus2.flatten())
        n1_minus2 = n1_minus2.reshape(self.nphi, self.ntheta, 3)
        n2_minus2 = n2_minus2.reshape(self.nphi, self.ntheta, 3)
        
        dn1_dphi_plus = (n1_plus2 - n1_minus2) / (2 * eps2)
        dn2_dphi_plus = (n2_plus2 - n2_minus2) / (2 * eps2)
        
        d2n1_dphi2 = (dn1_dphi_plus - dn1_dphi) / eps2
        d2n2_dphi2 = (dn2_dphi_plus - dn2_dphi) / eps2
        
        # Second derivative of axis curve with respect to phi
        eps_axis = 1e-7
        axis_curve_plus2 = vmap(self._compute_axis_curve)(phi2d + eps_axis)
        axis_curve_plus2 = axis_curve_plus2.reshape(self.nphi, self.ntheta, 3)
        axis_curve_minus2 = vmap(self._compute_axis_curve)(phi2d - eps_axis)
        axis_curve_minus2 = axis_curve_minus2.reshape(self.nphi, self.ntheta, 3)
        
        d2_axis_dphi2 = (axis_curve_plus2 - 2*axis_curve + axis_curve_minus2) / (eps_axis**2)
        
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
        
        # Scale by proper Jacobian factors
        # Tangent in theta (normalize by 2π for proper scaling)
        gammadash_theta = (2 * jnp.pi) * gammadash_theta
        
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
        """Compute normal vector and area element."""
        normal = jnp.cross(self.gammadash_phi, self.gammadash_theta, axis=2)
        unitnormal = normal / jnp.linalg.norm(normal, axis=2, keepdims=True)
        area_element = jnp.linalg.norm(normal, axis=2)
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
            "scale_fixed": self._scale_fixed
        }
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

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



