from functools import partial
import numpy as np
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

@jit
def B_on_surface(surface, field):
    ntheta = surface.ntheta
    nphi = surface.nphi
    gamma = surface.gamma
    gamma_reshaped = gamma.reshape(nphi * ntheta, 3)
    # Spread the points over the devices when they divide evenly (sharding the
    # surface pytree itself failed whenever its leaves did not).
    if len(gamma_reshaped) % sharding.num_devices == 0:
        gamma_reshaped = jax.lax.with_sharding_constraint(gamma_reshaped, sharding)

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
                                   close=False, range_torus="full torus", *, rbs=None, zbc=None, _cls=None):
    """Create a differentiable surface from VMEC ``rbc`` and ``zbs`` arrays.

    VMEC stores arrays as ``[n + ntor, m]`` and omits negative-``n`` modes
    when ``m=0``. ESSOS stores the same independent coefficients as flat mode
    vectors; this function performs only that ordering conversion.
    """
    rbc, zbs = jnp.asarray(rbc), jnp.asarray(zbs)
    if rbc.ndim != 2 or rbc.shape != zbs.shape or rbc.shape[0] % 2 != 1:
        raise ValueError("rbc and zbs must have equal shape (2*ntor+1, mpol+1)")
    ntor, mpol = (rbc.shape[0] - 1) // 2, rbc.shape[1] - 1
    def pack(table):
        table = jnp.asarray(table)
        if table.shape != rbc.shape:
            raise ValueError("Fourier partner arrays must match rbc.shape")
        return jnp.concatenate((table[ntor:, 0], table[:, 1:].T.ravel()))
    partners = {name: pack(table) for name, table in (('rs', rbs), ('zc', zbc)) if table is not None}
    cls = SurfaceRZFourier if _cls is None else _cls
    return cls(pack(rbc), pack(zbs), int(nfp), mpol, ntor, ntheta=ntheta, nphi=nphi,
                            close=close, range_torus=range_torus, **partners)

    

class SurfaceRZFourier:
    def __init__(self, rc, zs, nfp, mpol, ntor, ntheta=30, nphi=30, close=True, range_torus='full torus',
                 scaling_type=2, scaling_factor=0, *, rs=None, zc=None):
        """Initialize a Fourier surface.

        Args:
            rc: cosine Fourier coefficients for R.
            zs: sine Fourier coefficients for Z.
            rs, zc: optional sine R and cosine Z coefficients for asymmetric surfaces.
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
            Dofs contain scaled ``rc, zs``, followed by any active ``rs, zc`` arrays.
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
            scaling_factor, rs, zc,
        )

    def _initialize_state(self, rc, zs, nfp, mpol, ntor, ntheta, nphi, close, range_torus, scaling_type, scaling_factor, rs=None, zc=None):
        self._rc = rc
        self._zs = zs
        for name, table in (('rs', rs), ('zc', zc)):
            if table is not None and hasattr(table, 'shape') and table.shape != rc.shape:
                raise ValueError(f"{name} must match rc.shape")
            setattr(self, '_' + name, table)
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
        self._mode_numbers = None

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
        """Read indexed boundary coefficients; truncate outside MPOL/NTOR as VMEC does."""
        from f90nml import Parser
        nml = Parser().read(file)['indata']

        nfp, mpol, ntor = int(nml.get('nfp', 1)), int(nml.get('mpol', 6)) - 1, int(nml.get('ntor', 0))
        if mpol < 0 or ntor < 0:
            raise ValueError('Require MPOL >= 1 and NTOR >= 0')
        def coefficients(name):
            matrix = np.zeros((mpol + 1, 2 * ntor + 1))
            if name not in nml:
                return jnp.asarray(matrix.T)
            n0, m0 = nml.start_index[name]
            for i, row in enumerate(nml[name]):
                for j, value in enumerate(row or []):
                    m, n = m0 + i, n0 + j
                    if value is not None and 0 <= m <= mpol and abs(n) <= ntor:
                        sign = -1 if m == 0 and n < 0 and name in ('rbs', 'zbs') else 1
                        n = abs(n) if m == 0 else n
                        matrix[m, n + ntor] += sign * value
            return jnp.asarray(matrix.T)
        partners = {name: coefficients(name) for name in ('rbs', 'zbc')
                    if nml.get('lasym', False) and name in nml}
        return surfacerzfourier_from_boundary(coefficients('rbc'), coefficients('zbs'), nfp,
                                             ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus, _cls=cls, **partners)
    
    @classmethod
    def from_vmec(cls, vmec, s=1, ntheta=30, nphi=30, close=True, range_torus='full torus'):
        from essos.fields import _radial_interp
        nfp = vmec.nfp
        mpol = vmec.mpol
        ntor = vmec.ntor

        s_full_grid = vmec.s_full_grid
        rc = _radial_interp(s, s_full_grid, vmec.rmnc, vmec.xm)
        zs = _radial_interp(s, s_full_grid, vmec.zmns, vmec.xm)

        partners = {target: _radial_interp(s, s_full_grid, table, vmec.xm)
                    for target, table in (('rs', getattr(vmec, 'rmns', None)), ('zc', getattr(vmec, 'zmnc', None)))
                    if table is not None}
        surface = cls(rc, zs, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus, **partners)
        surface._mode_numbers = tuple(tuple(np.asarray(a).astype(int)) for a in (vmec.xm, vmec.xn))
        surface._xm = vmec.xm
        surface._xn = vmec.xn

        return surface

    @classmethod
    def from_wout_file(cls, file, s=1, ntheta=30, nphi=30, close=True, range_torus='full torus'):
        from essos.fields import _radial_interp
        from netCDF4 import Dataset
        with Dataset(file) as nc:
            nfp = int(nc.variables["nfp"][0])
            xm = jnp.array(nc.variables["xm"][:])
            xn = jnp.array(nc.variables["xn"][:])
            mpol = int(jnp.max(xm))
            ntor = int(jnp.max(jnp.abs(xn)) / nfp)

            ns = int(nc.variables["ns"][0])
            if ns < 3:
                raise ValueError("Require ns >= 3")
            s_full_grid = jnp.linspace(0, 1, ns)
            rc = _radial_interp(s, s_full_grid, jnp.array(nc.variables["rmnc"][:]), xm)
            zs = _radial_interp(s, s_full_grid, jnp.array(nc.variables["zmns"][:]), xm)
        
            lasym = any(bool(nc.variables[name][:].item())
                        for name in ('lasym__logical__', 'lasym') if name in nc.variables)
            if lasym and any(name not in nc.variables for name in ('rmns', 'zmnc')):
                raise ValueError("Asymmetric wout is missing geometry partner tables")
            partners = {target: _radial_interp(s, s_full_grid, jnp.array(nc.variables[name][:]), xm)
                        for target, name in (('rs', 'rmns'), ('zc', 'zmnc')) if name in nc.variables and np.any(nc.variables[name][:])}
            surface = cls(rc, zs, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus, **partners)
            surface._mode_numbers = tuple(tuple(np.asarray(a).astype(int)) for a in (xm, xn))
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

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, value):
        if value is not None and value.shape != self.rc.shape:
            raise ValueError("rs must match rc.shape")
        self._rs = value
        self.reset_cache()

    @property
    def zc(self):
        return self._zc

    @zc.setter
    def zc(self, value):
        if value is not None and value.shape != self.rc.shape:
            raise ValueError("zc must match rc.shape")
        self._zc = value
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
            value = (jnp.asarray(self._mode_numbers[0]) if self._mode_numbers is not None
                     else jnp.repeat(jnp.arange(self.mpol + 1), 2 * self.ntor + 1)[self.ntor:])
            if _cacheable(value):
                self._xm = value
            return value
        return self._xm

    # xn property
    @property
    def xn(self):
        if self._xn is None:
            value = (jnp.asarray(self._mode_numbers[1]) if self._mode_numbers is not None
                     else self.nfp * jnp.tile(jnp.arange(-self.ntor, self.ntor + 1), self.mpol + 1)[self.ntor:])
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
        return jnp.hstack([table * self.scaling for table in (self.rc, self.zs, self.rs, self.zc) if table is not None])
    
    @dofs.setter
    def dofs(self, new_dofs):
        names = [name for name in ('rc', 'zs', 'rs', 'zc') if getattr(self, name) is not None]
        if new_dofs.size != len(names) * self.rc.size:
            raise ValueError("dofs must contain every active Fourier coefficient family")
        for name, table in zip(names, jnp.split(new_dofs, len(names))):
            setattr(self, '_' + name, table / self.scaling)
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
        dR_dtheta = -jnp.einsum('i,ijk->jk', xm * rc, sin_angles)
        dZ_dtheta = jnp.einsum('i,ijk->jk', xm * zs, cos_angles)
        dR_dphi = jnp.einsum('i,ijk->jk', xn * rc, sin_angles)
        dZ_dphi = -jnp.einsum('i,ijk->jk', xn * zs, cos_angles)
        if self.rs is not None:
            R += jnp.einsum('i,ijk->jk', self.rs, sin_angles)
            dR_dtheta += jnp.einsum('i,ijk->jk', xm * self.rs, cos_angles)
            dR_dphi -= jnp.einsum('i,ijk->jk', xn * self.rs, cos_angles)
        if self.zc is not None:
            Z += jnp.einsum('i,ijk->jk', self.zc, cos_angles)
            dZ_dtheta -= jnp.einsum('i,ijk->jk', xm * self.zc, sin_angles)
            dZ_dphi += jnp.einsum('i,ijk->jk', xn * self.zc, sin_angles)
        gamma = jnp.stack([R * cos_phi2d, R * sin_phi2d, Z], axis=-1)
        gammadash_theta = jnp.stack([dR_dtheta * cos_phi2d, dR_dtheta * sin_phi2d, dZ_dtheta], axis=-1)

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
        """Write a VMEC boundary namelist, including any asymmetric partners."""
        families = {'RBC': self.rc, 'ZBS': self.zs, 'RBS': self.rs, 'ZBC': self.zc}
        asym = self.rs is not None or self.zc is not None
        lines = ['&INDATA', f'LASYM = .{str(asym).upper()}.', f'NFP = {self.nfp}',
                 f'MPOL = {self.mpol + 1}', f'NTOR = {self.ntor}']
        for m, xn, i in zip(self.xm, self.xn, range(len(self.xm))):
            if int(xn) % self.nfp:
                raise ValueError('Toroidal mode numbers must be multiples of nfp')
            n = int(xn) // self.nfp
            lines.append(', '.join(f'{name}({n},{int(m)}) = {float(table[i]):.15e}'
                                   for name, table in families.items() if table is not None))
        with open(filename, 'w') as f:
            f.write('\n'.join(lines) + '\n/\n')
            
    def mean_cross_sectional_area(self):
        """Mean over the toroidal grid of the area of the constant-phi cross sections.

        The cross-section area is ``A = int R dZ/dtheta dtheta`` over
        ``theta in [0, 2 pi)``, i.e. ``2 pi`` times the theta-average of the
        integrand. With ``close=True`` the duplicated end points are dropped so
        the average remains a periodic rectangle rule.
        """
        xyz = self.gamma
        dgamma1 = self.gammadash_phi
        dgamma2 = self.gammadash_theta
        if self.close:
            xyz, dgamma1, dgamma2 = (a[:-1, :-1] for a in (xyz, dgamma1, dgamma2))
        x2y2 = xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2
        J = jnp.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
        J = J.at[:, :, 0, 0].set((xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0]) / x2y2)
        J = J.at[:, :, 0, 1].set((xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0]) / x2y2)
        J = J.at[:, :, 1, 0].set(0)
        J = J.at[:, :, 1, 1].set(1)
        detJ = jnp.linalg.det(J)
        Jinv = jnp.linalg.inv(J)
        dZ_dtheta = dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
        mean_cross_sectional_area = 2 * jnp.pi * jnp.abs(jnp.mean(jnp.sqrt(x2y2) * dZ_dtheta * detJ))
        return mean_cross_sectional_area
    
    def _tree_flatten(self):
        tables = (self._rc, self._zs, self._rs, self._zc)
        children = tuple(table * self.scaling if hasattr(table, "shape") else table for table in tables)
        aux_data = {"nfp": self._nfp,
                    "mpol": self._mpol,
                    "ntor": self._ntor,
                    "ntheta": self._ntheta,
                    "nphi": self._nphi,
                    "close": self._close,
                    "range_torus": self._range_torus,
                    "scaling_type": self._scaling_type,
                    "scaling_factor": self._scaling_factor,
                    "mode_numbers": self._mode_numbers}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        rc_scaled, zs_scaled = children[:2]

        if hasattr(rc_scaled, "shape") and hasattr(zs_scaled, "shape"):
            mpol = aux_data["mpol"]
            ntor = aux_data["ntor"]
            nfp = aux_data["nfp"]
            scaling_type = cls._normalize_scaling_type(aux_data["scaling_type"])
            scaling_factor = aux_data["scaling_factor"]

            modes = aux_data["mode_numbers"]
            xm = jnp.repeat(jnp.arange(mpol + 1), 2 * ntor + 1)[ntor:] if modes is None else jnp.asarray(modes[0])
            xn = nfp * jnp.tile(jnp.arange(-ntor, ntor + 1), mpol + 1)[ntor:] if modes is None else jnp.asarray(modes[1])
            scaling = cls._compute_scaling(xm, xn, scaling_type, scaling_factor)

            rc, zs, rs, zc = (table / scaling if table is not None else None for table in children)
        else:
            rc, zs, rs, zc = children

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
            aux_data["scaling_factor"], rs, zc,
        )
        obj._mode_numbers = aux_data["mode_numbers"]
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

    def __init__(self, surface, h=0.05, padding=0.1):
        """
        Args:
            surface: the surface to contruct the distance from.
            h: grid resolution of the interpolant
            padding: distance represented outside the surface
        """
        if padding <= 0.0:
            raise ValueError("padding must be positive")
        gammas = surface.gamma
        r = jnp.linalg.norm(gammas[:, :, :2], axis=2)
        z = gammas[:, :, 2]
        rmin = max(jnp.min(r) - padding, 0.)
        rmax = jnp.max(r) + padding
        zmin = jnp.min(z) - padding
        zmax = jnp.max(z) + padding

        self.zrange = (zmin, zmax)
        self.rrange = (rmin, rmax)

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*jnp.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)

        gammas_flat = surface.gamma.reshape((-1, 3))
        normals_flat = surface.unitnormal.reshape((-1, 3))
        tree = jaxkd.build_tree(gammas_flat)
        interior = jnp.mean(surface.gamma[0, :, :], axis=0)
        sign_of_interior = jnp.sign(jnp.sum(
            (interior - gammas_flat[0]) * normals_flat[0]))

        def fbatch(rs, phis, zs):
            xyz = jnp.zeros(( 3))
            xyz=xyz.at[0].set( rs * jnp.cos(phis))
            xyz=xyz.at[1].set(rs * jnp.sin(phis))
            xyz=xyz.at[2].set(zs)
            nearest, _ = jaxkd.query_neighbors(tree, xyz, k=1)
            distance = jnp.sum(
                (xyz - gammas_flat[nearest]) * normals_flat[nearest], axis=1)
            return distance * sign_of_interior
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
