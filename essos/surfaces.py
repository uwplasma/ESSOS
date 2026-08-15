from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from jax import tree_util, jit, vmap, devices, device_put
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from essos.plot import fix_matplotlib_3d
import jaxkd

mesh = Mesh(devices(), ("dev",))
sharding = NamedSharding(mesh, PartitionSpec("dev"))


def _cacheable(*values):
    """Return false for values created while an outer JAX transform is tracing."""
    return not any(isinstance(leaf, jax.core.Tracer)
                   for leaf in tree_util.tree_leaves(values))


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
@partial(jit, in_shardings=(sharding, None), out_shardings=sharding)
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


def surfacerzfourier_from_boundary(rbc, zbs, nfp, ntheta=30, nphi=30,
                                   close=False, range_torus="full torus"):
    """Create a differentiable surface from VMEC ``rbc`` and ``zbs`` arrays.

    VMEC stores arrays as ``[n + ntor, m]`` and omits negative-``n`` modes
    when ``m=0``. ESSOS stores the same independent coefficients as flat mode
    vectors; this function performs only that ordering conversion.
    """
    rbc, zbs = jnp.asarray(rbc), jnp.asarray(zbs)
    if rbc.ndim != 2 or rbc.shape != zbs.shape or rbc.shape[0] % 2 != 1:
        raise ValueError("rbc and zbs must have equal shape (2*ntor+1, mpol+1)")
    ntor, mpol = (rbc.shape[0] - 1) // 2, rbc.shape[1] - 1
    rc = jnp.concatenate((rbc[ntor:, 0], rbc[:, 1:].T.ravel()))
    zs = jnp.concatenate((zbs[ntor:, 0], zbs[:, 1:].T.ravel()))
    return SurfaceRZFourier(rc, zs, int(nfp), mpol, ntor, ntheta=ntheta, nphi=nphi,
                            close=close, range_torus=range_torus)

    

class SurfaceRZFourier:
    def __init__(self, rc, zs, nfp, mpol, ntor, ntheta=30, nphi=30, close=True, range_torus='full torus',
                 scaling_type=2, scaling_factor=0):
        """Initialize a Fourier surface.

        Args:
            rc: cosine Fourier coefficients for R.
            zs: sine Fourier coefficients for Z.
            nfp: number of field periods.
            mpol: maximum poloidal mode number.
            ntor: maximum toroidal mode number.
            ntheta: number of theta grid points.
            nphi: number of phi grid points.
            close: whether the surface mesh includes the endpoint.
            range_torus: either ``'full torus'`` or ``'half period'``.
            scaling_type: norm used in the mode scaling. Accepted values are
                ``'L1'`` or ``1``, ``'L2'`` or ``2``, and ``'Linfty'`` or ``-1``.
            scaling_factor: exponential weight used in the scaling
                ``exp(scaling_factor * ||(xm, xn)||)``.

        Note:
            The optimized dofs are stored as ``[rc * scaling, zs * scaling]``,
            with the scaling computed mode-by-mode from ``xm`` and ``xn``.
        """

        assert isinstance(nfp, int) and nfp > 0, "nfp must be a positive integer."
        assert isinstance(mpol, int) and mpol >= 0, "mpol must be a non-negative integer."
        assert isinstance(ntor, int) and ntor >= 0, "ntor must be a non-negative integer."
        assert isinstance(ntheta, int) and ntheta > 0, "ntheta must be a positive integer."
        assert isinstance(nphi, int) and nphi > 0, "nphi must be a positive integer."
        assert isinstance(close, bool), "close must be a boolean."
        assert range_torus in ['full torus', 'half period'], f"Unknown range_torus: {range_torus}. Choose 'full torus' or 'half period'."
        self._initialize_state(
            rc,
            zs,
            nfp,
            mpol,
            ntor,
            ntheta,
            nphi,
            close,
            range_torus,
            self._normalize_scaling_type(scaling_type),
            scaling_factor,
        )

    def _initialize_state(self, rc, zs, nfp, mpol, ntor, ntheta, nphi, close, range_torus, scaling_type, scaling_factor):
        self._rc = rc
        self._zs = zs
        self._nfp = nfp
        self._mpol = mpol
        self._ntor = ntor

        self._gamma = None
        self._gammadash_theta = None
        self._gammadash_phi = None
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
        self._scaling_type = scaling_type
        self._scaling_factor = scaling_factor
        self._scaling = None

    @staticmethod
    def _normalize_scaling_type(scaling_type):
        """Map public scaling_type inputs to norm orders used internally."""
        if scaling_type == "L1" or scaling_type == 1:
            return 1
        if scaling_type == "L2" or scaling_type == 2:
            return 2
        if scaling_type == "Linfty" or scaling_type == -1 or scaling_type == jnp.inf:
            return jnp.inf
        raise ValueError(
            f"Unknown scaling_type: {scaling_type}. "
            "Expected 'L1', 1, 'L2', 2, 'Linfty', -1, or jnp.inf."
        )

    @staticmethod
    def _compute_scaling(xm, xn, scaling_type, scaling_factor):
        return jnp.exp(scaling_factor * jnp.linalg.norm(jnp.vstack([xm, xn]), ord=scaling_type, axis=0))


    @classmethod
    def from_input_file(cls, file, ntheta=30, nphi=30, close=True, range_torus='full torus'):
        from f90nml import Parser
        nml = Parser().read(file)['indata']

        nfp = nml["nfp"] if "nfp" in nml else 1
        mpol = nml['mpol']            
        ntor = nml['ntor']
        
        rc = jnp.ravel(nested_lists_to_array(nml['rbc']))[2:]
        zs = jnp.ravel(nested_lists_to_array(nml['zbs']))[2:]

        surface = cls(rc, zs, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus)
        return surface
    
    @classmethod
    def from_vmec(cls, vmec, s=1, ntheta=30, nphi=30, close=True, range_torus='full torus'):
        nfp = vmec.nfp
        mpol = vmec.mpol
        ntor = vmec.ntor

        s_full_grid = vmec.s_full_grid
        rc = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(vmec.rmnc)
        zs = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(vmec.zmns)

        surface = cls(rc, zs, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus)
        surface._xm = vmec.xm
        surface._xn = vmec.xn

        return surface

    @classmethod
    def from_wout_file(cls, file, s=1, ntheta=30, nphi=30, close=True, range_torus='full torus'):
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

        surface = cls(rc, zs, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus)
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
            value = jnp.repeat(jnp.arange(self.mpol + 1), 2 * self.ntor + 1)[self.ntor:]
            if _cacheable(value):
                self._xm = value
            return value
        return self._xm

    # xn property
    @property
    def xn(self):
        if self._xn is None:
            value = self.nfp * jnp.tile(
                jnp.arange(-self.ntor, self.ntor + 1), self.mpol + 1)[self.ntor:]
            if _cacheable(value):
                self._xn = value
            return value
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
        quadpoints_theta = jnp.linspace(0, 2 * jnp.pi, num=self.ntheta, endpoint=self.close)
        quadpoints_phi   = jnp.linspace(0, 2 * jnp.pi * end_val / div, num=self.nphi, endpoint=self.close)
        theta2d, phi2d = jnp.meshgrid(quadpoints_theta, quadpoints_phi)
        return quadpoints_theta, quadpoints_phi, theta2d, phi2d

    # theta2d property
    @property
    def theta2d(self):
        if self._theta2d is None:
            values = self._compute_meshgrid()
            if _cacheable(*values):
                self._quadpoints_theta, self._quadpoints_phi, self._theta2d, self._phi2d = values
            return values[2]
        return self._theta2d

    # phi2d property
    @property
    def phi2d(self):
        if self._phi2d is None:
            values = self._compute_meshgrid()
            if _cacheable(*values):
                self._quadpoints_theta, self._quadpoints_phi, self._theta2d, self._phi2d = values
            return values[3]
        return self._phi2d

    # angles property
    @property
    def angles(self):
        if self._angles is None:
            value = (jnp.einsum('i,jk->ijk', self.xm, self.theta2d)
                     - jnp.einsum('i,jk->ijk', self.xn, self.phi2d))
            if _cacheable(value):
                self._angles = value
            return value
        return self._angles
    
    # scaling_type property and setter
    @property
    def scaling_type(self):
        return self._scaling_type
    
    @scaling_type.setter
    def scaling_type(self, new_type):
        self._scaling_type = self._normalize_scaling_type(new_type)
        self._scaling = None

    # scaling_factor property and setter
    @property
    def scaling_factor(self):
        return self._scaling_factor
    
    @scaling_factor.setter
    def scaling_factor(self, new_factor):
        self._scaling_factor = new_factor
        self._scaling = None

    # scaling property
    @property
    def scaling(self):
        """Mode-by-mode scaling ``exp(scaling_factor * ||(xm, xn)||)``."""
        if self._scaling is None:
            scaling = self._compute_scaling(self.xm, self.xn, self.scaling_type, self.scaling_factor)
            if not isinstance(scaling, jax.core.Tracer):
                self._scaling = scaling
            return scaling
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

        R = jnp.einsum('i,ijk->jk', rc, cos_angles)
        Z = jnp.einsum('i,ijk->jk', zs, sin_angles)
        X = R * cos_phi2d
        Y = R * sin_phi2d
        gamma = jnp.stack([X, Y, Z], axis=-1)

        dR_dtheta = -jnp.einsum('i,ijk->jk', xm * rc, sin_angles)
        dZ_dtheta = jnp.einsum('i,ijk->jk', xm * zs, cos_angles)
        dX_dtheta = dR_dtheta * cos_phi2d
        dY_dtheta = dR_dtheta * sin_phi2d
        gammadash_theta = jnp.stack([dX_dtheta, dY_dtheta, dZ_dtheta], axis=-1)

        dR_dphi = jnp.einsum('i,ijk->jk', xn*rc, sin_angles)
        dZ_dphi = -jnp.einsum('i,ijk->jk', xn*zs, cos_angles)
        dX_dphi = dR_dphi * cos_phi2d - R * sin_phi2d
        dY_dphi = dR_dphi * sin_phi2d + R * cos_phi2d
        gammadash_phi = jnp.stack([dX_dphi, dY_dphi, dZ_dphi], axis=-1)
        
        return gamma, gammadash_theta, gammadash_phi
    
    # gamma, gammadash_theta, gammadash_phi properties
    @property
    def gamma(self):
        if self._gamma is None:
            values = self._compute_gamma()
            if _cacheable(*values):
                self._gamma, self._gammadash_theta, self._gammadash_phi = values
            return values[0]
        return self._gamma
    
    @property
    def gammadash_theta(self):
        if self._gammadash_theta is None:
            values = self._compute_gamma()
            if _cacheable(*values):
                self._gamma, self._gammadash_theta, self._gammadash_phi = values
            return values[1]
        return self._gammadash_theta
    
    @property
    def gammadash_phi(self):
        if self._gammadash_phi is None:
            values = self._compute_gamma()
            if _cacheable(*values):
                self._gamma, self._gammadash_theta, self._gammadash_phi = values
            return values[2]
        return self._gammadash_phi

    # _compute_properties method
    @jit
    def _compute_properties(self):
        normal = jnp.cross(self.gammadash_theta, self.gammadash_phi, axis=2)
        unitnormal = normal / jnp.linalg.norm(normal, axis=2, keepdims=True)
        area_element = jnp.linalg.norm(normal, axis=2)
        return normal, unitnormal, area_element
    
    # normal, unitnormal, area_element properties
    @property
    def normal(self):
        if self._normal is None:
            values = self._compute_properties()
            if _cacheable(*values):
                self._normal, self._unitnormal, self._area_element = values
            return values[0]
        return self._normal
    
    @property
    def unitnormal(self):
        if self._unitnormal is None:
            values = self._compute_properties()
            if _cacheable(*values):
                self._normal, self._unitnormal, self._area_element = values
            return values[1]
        return self._unitnormal
    
    @property
    def area_element(self):
        if self._area_element is None:
            values = self._compute_properties()
            if _cacheable(*values):
                self._normal, self._unitnormal, self._area_element = values
            return values[2]
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
        volume = jnp.mean(integrand) / 3.0
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
            nml += f"RBC({self.xn[l]:4d},{self.xm[l]:4d}) ={rc:23.15e},    ZBS({self.xn[l]:4d},{self.xm[l]:4d}) ={zs:23.15e}\n"
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
        if hasattr(self._rc, "shape") and hasattr(self._zs, "shape"):
            children = (self.rc * self.scaling, self.zs * self.scaling)  # arrays / dynamic values
        else:
            children = (self._rc, self._zs)
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
        rc_scaled, zs_scaled = children

        if hasattr(rc_scaled, "shape") and hasattr(zs_scaled, "shape"):
            mpol = aux_data["mpol"]
            ntor = aux_data["ntor"]
            nfp = aux_data["nfp"]
            scaling_type = cls._normalize_scaling_type(aux_data["scaling_type"])
            scaling_factor = aux_data["scaling_factor"]

            xm = jnp.repeat(jnp.arange(mpol + 1), 2 * ntor + 1)[ntor:]
            xn = nfp * jnp.tile(jnp.arange(-ntor, ntor + 1), mpol + 1)[ntor:]
            scaling = cls._compute_scaling(xm, xn, scaling_type, scaling_factor)

            rc = rc_scaled / scaling
            zs = zs_scaled / scaling
        else:
            rc = rc_scaled
            zs = zs_scaled

        obj = object.__new__(cls)
        obj._initialize_state(
            rc,
            zs,
            aux_data["nfp"],
            aux_data["mpol"],
            aux_data["ntor"],
            aux_data["ntheta"],
            aux_data["nphi"],
            aux_data["close"],
            aux_data["range_torus"],
            aux_data["scaling_type"],
            aux_data["scaling_factor"],
        )
        return obj

tree_util.register_pytree_node(SurfaceRZFourier,
                               SurfaceRZFourier._tree_flatten,
                               SurfaceRZFourier._tree_unflatten)

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
