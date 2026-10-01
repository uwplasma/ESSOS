import jax
jax.config.update("jax_enable_x64", True)
from jax import vmap
from essos.coils import Curves
import jax.numpy as jnp
from functools import partial
from jax import jit, jacfwd, grad, vmap, tree_util, lax
from essos.surfaces import SurfaceRZFourier, BdotN_over_B, SurfaceClassifier
from essos.plot import fix_matplotlib_3d
from essos.util import newton

class MagneticField():
    def __init__(self):
        pass

    @jit
    def sqrtg(self, points):
        raise NotImplementedError("sqrtg method not implemented")

    @jit
    def B(self, points):
        raise NotImplementedError("B method not implemented")

    @jit
    def B_covariant(self, points):
        return self.B(points)

    @jit
    def B_contravariant(self, points):
        return self.B(points)
    
    @jit
    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points))
    
    @jit
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    @jit
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @jit
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)
    
    @jit
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] - grad_B_cov[1][2],
                          grad_B_cov[0][2] - grad_B_cov[2][0],
                          grad_B_cov[1][0] - grad_B_cov[0][1]])/self.sqrtg(points)

    @jit
    def curl_b(self, points):
        return self.curl_B(points) / self.AbsB(points) + jnp.cross(self.B_covariant(points), jnp.array(self.dAbsB_by_dX(points))) / self.AbsB(points)**2 / self.sqrtg(points)
    
    @jit
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points), self.curl_b(points)) * self.sqrtg(points) / self.AbsB(points)
    
    @jit
    def to_xyz(self, points):
        raise NotImplementedError("to_xyz method not implemented")

class BiotSavart(MagneticField):
    def __init__(self, coils):
        self.coils = coils
        self._r_axis = None
        self._z_axis = None
    
    @property
    def dofs(self):
        return self.coils.dofs
    
    @dofs.setter
    def dofs(self, new_dofs):
        self.coils.dofs = new_dofs

    @jit
    def sqrtg(self, points):
        return 1.
    
    @jit
    def B(self, points):
        dif_R = (jnp.array(points) - self.coils.gamma).T
        dB = jnp.cross(self.coils.gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0) / jnp.linalg.norm(dif_R, axis=0)**3
        dB_sum = jnp.einsum("i,bai", self.coils.currents*1e-7, dB, optimize="greedy")
        return jnp.mean(dB_sum, axis=0)

    @jit
    def b_cyl(self, R, phi, Z):
        """Return ``(B_R, B_phi, B_Z)`` on broadcast cylindrical arrays.

        This field-provider interface lets VMEC/NESTOR evaluate ESSOS coils
        directly on a changing plasma boundary without writing an mgrid file.
        It uses the same traceable Biot--Savart graph as :meth:`B`, so coil
        shape and current derivatives are retained.
        """
        R, phi, Z = jnp.broadcast_arrays(R, phi, Z)
        xyz = jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1)
        B = vmap(self.B)(xyz.reshape((-1, 3))).reshape(xyz.shape)
        br = B[..., 0] * jnp.cos(phi) + B[..., 1] * jnp.sin(phi)
        bp = -B[..., 0] * jnp.sin(phi) + B[..., 1] * jnp.cos(phi)
        return br, bp, B[..., 2]

    @property
    def r_axis(self):
        if self._r_axis is None:
            self._r_axis = jnp.mean(jnp.sqrt(vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(self.coils.dofs_curves)))
        return self._r_axis

    @property
    def z_axis(self):
        if self._z_axis is None:
            self._z_axis = jnp.mean(vmap(lambda dofs: dofs[2, 0])(self.coils.dofs_curves))
        return self._z_axis    

    @jit
    def to_xyz(self, points):
        return points
    
    def _tree_flatten(self):
        children = (self.coils,)
        aux_data = {}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(BiotSavart,
                               BiotSavart._tree_flatten,
                               BiotSavart._tree_unflatten)
    
@jit
def d_dtheta_fft(f_theta):
    ntheta = f_theta.shape[-1]
    k = jnp.fft.fftfreq(ntheta, d=1.0/ntheta)     # integer modes
    Fk = jnp.fft.fft(f_theta, axis=-1)
    dF = (1j * k) * Fk
    return jnp.fft.ifft(dF, axis=-1).real * (2*jnp.pi)

@jit
def d2_dtheta2_fft(f_theta):
    ntheta = f_theta.shape[-1]
    k = jnp.fft.fftfreq(ntheta, d=1.0/ntheta)     # integer modes
    Fk = jnp.fft.fft(f_theta, axis=-1)
    d2F = -(k**2) * Fk
    return jnp.fft.ifft(d2F, axis=-1).real * (2*jnp.pi)**2

@jit
def gamma_dash_from_gamma(gamma):
    return jnp.stack([
        d_dtheta_fft(gamma[..., 0]),
        d_dtheta_fft(gamma[..., 1]),
        d_dtheta_fft(gamma[..., 2]),
    ], axis=-1)

@jit
def gamma_dashdash_from_gamma(gamma):
    return jnp.stack([
        d2_dtheta2_fft(gamma[..., 0]),
        d2_dtheta2_fft(gamma[..., 1]),
        d2_dtheta2_fft(gamma[..., 2]),
    ], axis=-1)

class BiotSavart_from_gamma(MagneticField):
    def __init__(self, gamma, gamma_dash=None, gamma_dashdash=None, currents=None):
        self.currents = currents
        self.gamma = gamma
        self._gamma_dash = gamma_dash
        self._gamma_dashdash = gamma_dashdash

        self.coils_length = None
        self.coils_curvature = None
        self.r_axis = None
        self.z_axis = None

    @property
    def gamma_dash(self):
        if self._gamma_dash is None:
            self._gamma_dash = gamma_dash_from_gamma(self.gamma)
        return self._gamma_dash

    @property
    def gamma_dashdash(self):
        if self._gamma_dashdash is None:
            self._gamma_dashdash = gamma_dashdash_from_gamma(self.gamma)
        return self._gamma_dashdash

    @property
    def coils_length(self):
        if self.coils_length is None:
            self.coils_length = jnp.array([jnp.mean(jnp.linalg.norm(d1gamma, axis=1)) for d1gamma in self.gamma_dash])
        return self.coils_length

    @property
    def coils_curvature(self):
        if self._coils_curvature is None:
            self._coils_curvature = vmap(Curves.compute_curvature)(self.gamma_dash, self.gamma_dashdash)
        return self._coils_curvature
    
    @property
    def r_axis(self):
        if self._r_axis is None:
            self._r_axis = jnp.average(jnp.linalg.norm(jnp.average(self.gamma, axis=1)[:, 0:2], axis=1))
        return self._r_axis
    
    @property
    def z_axis(self):
        if self._z_axis is None:
            self._z_axis = jnp.average(jnp.average(self.gamma, axis=1)[:, 2])
        return self._z_axis
    
    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        return 1.
    
    @partial(jit, static_argnames=['self'])
    def B(self, points):
        dif_R = (jnp.array(points) - self.gamma).T
        dB = jnp.cross(self.gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0) / jnp.linalg.norm(dif_R, axis=0)**3
        dB_sum = jnp.einsum("i,bai", self.currents*1e-7, dB, optimize="greedy")
        return jnp.mean(dB_sum, axis=0)
    
    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        return points

class Vmec():
    def __init__(self, wout_filename, ntheta=50, nphi=50, close=True, range_torus='full torus'):
        self.wout_filename = wout_filename
        from netCDF4 import Dataset
        self.nc = Dataset(self.wout_filename)
        self.nfp = int(self.nc.variables["nfp"][0])
        self.bmnc = jnp.array(self.nc.variables["bmnc"][:])
        self.xm = jnp.array(self.nc.variables["xm"][:])
        self.xn = jnp.array(self.nc.variables["xn"][:])
        self.rmnc = jnp.array(self.nc.variables["rmnc"][:])
        self.zmns = jnp.array(self.nc.variables["zmns"][:])
        self.bsubsmns = jnp.array(self.nc.variables["bsubsmns"][:])
        self.bsubumnc = jnp.array(self.nc.variables["bsubumnc"][:])
        self.bsubvmnc = jnp.array(self.nc.variables["bsubvmnc"][:])
        self.bsupumnc = jnp.array(self.nc.variables["bsupumnc"][:])
        self.bsupvmnc = jnp.array(self.nc.variables["bsupvmnc"][:])
        self.gmnc = jnp.array(self.nc.variables["gmnc"][:])
        self.xm_nyq = jnp.array(self.nc.variables["xm_nyq"][:])
        self.xn_nyq = jnp.array(self.nc.variables["xn_nyq"][:])
        self.len_xm_nyq = len(self.xm_nyq)
        self.ns = self.nc.variables["ns"][0]
        self.s_full_grid = jnp.linspace(0, 1, self.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds
        self.r_axis = self.rmnc[0, 0]
        self.z_axis=self.zmns[0,0]
        self.mpol = int(jnp.max(self.xm))
        self.ntor = int(jnp.max(jnp.abs(self.xn)) / self.nfp)
        self.range_torus = range_torus
        self._surface = SurfaceRZFourier.from_vmec(self, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus)
        self.Aminor_p = jnp.array(self.nc.variables["Aminor_p"][:])
        #self._classifier=SurfaceClassifier(self._surface,p=1,h=0.05)
        
    @property
    def surface(self):
        return self._surface
        
    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        s, theta, phi = points
        bsubsmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.bsubsmns)
        bsubumnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsubumnc[1:])
        bsubvmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsubvmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        sinangle_nyq = jnp.sin(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sub_s = jnp.dot(bsubsmns_interp, sinangle_nyq)
        B_sub_theta = jnp.dot(bsubumnc_interp, cosangle_nyq)
        B_sub_phi = jnp.dot(bsubvmnc_interp, cosangle_nyq)
        return jnp.array([B_sub_s, B_sub_theta, B_sub_phi])
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        s, theta, phi = points
        bsupumnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsupumnc[1:])
        bsupvmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsupvmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sup_theta = jnp.dot(bsupumnc_interp, cosangle_nyq)
        B_sup_phi = jnp.dot(bsupvmnc_interp, cosangle_nyq)
        return jnp.array([0*B_sup_theta, B_sup_theta, B_sup_phi])
 
    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        s, theta, phi = points
        gmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.gmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        sqrt_g_vmec = jnp.dot(gmnc_interp, cosangle_nyq)
        return sqrt_g_vmec



    @partial(jit, static_argnames=['self'])
    def B(self, points):
        s, theta, phi = points
        gmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.gmnc[1:])
        rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
        zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
        d_rmnc_d_s_interp = vmap(lambda row: grad(lambda s: jnp.interp(s, self.s_full_grid, row))(s), in_axes=1)(self.rmnc)
        d_zmns_d_s_interp = vmap(lambda row: grad(lambda s: jnp.interp(s, self.s_full_grid, row))(s), in_axes=1)(self.zmns)
        
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sub_s, B_sub_theta, B_sub_phi = self.B_covariant(points)
        sqrt_g_vmec = jnp.dot(gmnc_interp, cosangle_nyq)
        
        cosangle  = jnp.cos(self.xm * theta - self.xn * phi)
        sinangle  = jnp.sin(self.xm * theta - self.xn * phi)
        msinangle = self.xm * sinangle
        nsinangle = self.xn * sinangle
        mcosangle = self.xm * cosangle
        ncosangle = self.xn * cosangle
        
        sinphi = jnp.sin(phi)
        cosphi = jnp.cos(phi)
        
        R = jnp.dot(rmnc_interp, cosangle)
        d_R_d_theta = jnp.dot(rmnc_interp, -msinangle)
        d_R_d_phi   = jnp.dot(rmnc_interp, nsinangle)
        d_R_d_s     = jnp.dot(d_rmnc_d_s_interp, cosangle)
        
        d_X_d_theta = d_R_d_theta * cosphi
        d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
        d_X_d_s = d_R_d_s * cosphi

        d_Y_d_theta = d_R_d_theta * sinphi
        d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
        d_Y_d_s = d_R_d_s * sinphi
        
        d_Z_d_s = jnp.dot(d_zmns_d_s_interp, sinangle)
        d_Z_d_theta = jnp.dot(zmns_interp, mcosangle)
        d_Z_d_phi = jnp.dot(zmns_interp, -ncosangle)

        grad_s_X = (d_Y_d_theta * d_Z_d_phi - d_Z_d_theta * d_Y_d_phi) / sqrt_g_vmec
        grad_s_Y = (d_Z_d_theta * d_X_d_phi - d_X_d_theta * d_Z_d_phi) / sqrt_g_vmec
        grad_s_Z = (d_X_d_theta * d_Y_d_phi - d_Y_d_theta * d_X_d_phi) / sqrt_g_vmec

        grad_theta_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
        grad_theta_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
        grad_theta_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

        grad_phi_X = (d_Y_d_s * d_Z_d_theta - d_Z_d_s * d_Y_d_theta) / sqrt_g_vmec
        grad_phi_Y = (d_Z_d_s * d_X_d_theta - d_X_d_s * d_Z_d_theta) / sqrt_g_vmec
        grad_phi_Z = (d_X_d_s * d_Y_d_theta - d_Y_d_s * d_X_d_theta) / sqrt_g_vmec
        
        return jnp.array([B_sub_s * grad_s_X + B_sub_theta * grad_theta_X + B_sub_phi * grad_phi_X,
                          B_sub_s * grad_s_Y + B_sub_theta * grad_theta_Y + B_sub_phi * grad_phi_Y,
                          B_sub_s * grad_s_Z + B_sub_theta * grad_theta_Z + B_sub_phi * grad_phi_Z])
        
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        s, theta, phi = points
        bmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bmnc[1:, :])
        cos_values = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        return jnp.dot(bmnc_interp, cos_values)
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)


    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @partial(jit, static_argnames=['self'])
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)    
 
    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] -grad_B_cov[1][2],
                          grad_B_cov[0][2] -grad_B_cov[2][0],
                          grad_B_cov[1][0] -grad_B_cov[0][1]])/self.sqrtg(points)
    
    
    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return self.curl_B(points)/self.AbsB(points)+jnp.cross(self.B_covariant(points),jnp.array(self.dAbsB_by_dX(points)))/self.AbsB(points)**2/self.sqrtg(points)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points),self.curl_b(points))*self.sqrtg(points)/self.AbsB(points)

    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        s, theta, phi = points
        rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
        zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
        cosangle = jnp.cos(self.xm * theta - self.xn * phi)
        sinangle = jnp.sin(self.xm * theta - self.xn * phi)
        R = jnp.dot(rmnc_interp, cosangle)
        Z = jnp.dot(zmns_interp, sinangle)
        X = R * jnp.cos(phi)
        Y = R * jnp.sin(phi)
        return jnp.array([X, Y, Z])

class near_axis:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "The 'near_axis' class has been migrated to the standalone 'pyQSC_JAX' repository. "
            "Please run 'pip install git+https://github.com/uwplasma/pyQSC_JAX.git' "
            "and import it via 'from pyqsc_jax.near_axis import near_axis'."
        )


class CombinedField(MagneticField):
    """Sum of several magnetic fields, traced as one.

    The usual case is a coil field plus a plasma contribution: ``B`` and
    ``B_contravariant`` add over the fields, while the geometry helpers
    ``sqrtg`` and ``to_xyz`` come from the first field, which is the one that
    carries the coordinate system.
    """

    def __init__(self, *fields):
        if len(fields) < 1:
            raise ValueError("CombinedField needs at least one field")
        self.fields = fields

    @jit
    def B(self, points):
        return sum(field.B(points) for field in self.fields)

    @jit
    def B_contravariant(self, points):
        return sum(field.B_contravariant(points) for field in self.fields)

    @jit
    def sqrtg(self, points):
        return self.fields[0].sqrtg(points)

    @jit
    def to_xyz(self, points):
        return self.fields[0].to_xyz(points)

    def _tree_flatten(self):
        return (self.fields,), {}

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[0], **aux_data)


tree_util.register_pytree_node(CombinedField,
                               CombinedField._tree_flatten,
                               CombinedField._tree_unflatten)



class DipoleField_old:
    """
    Magnetic field from a collection of magnetic dipoles.

    Supports optional precomputation of the interaction matrix G for fast
    optimization. When surf_pts and surf_n are provided at construction,
    G is computed once in __init__ and stored as self.G. The optimizer
    then uses the fast matrix-vector multiply:

        Bn_total = self.G @ pho + Bn_fixed

    rather than recomputing dipole geometry every step. This follows the
    same pattern as DESC's ObjectiveFunction.build() — expensive geometry
    is precomputed once, and compute() (or in our case the Adam step) is
    just fast arithmetic.

    Parameters
    ----------
    dipole_positions : jnp.ndarray, shape (N, 3)
        Magnet center positions [m].
    dipole_moments : jnp.ndarray, shape (N, 3)
        Dipole moment vectors [A·m²].
    pho_values : jnp.ndarray, shape (N,)
        Magnet strengths in [-1, 1].
    stellsym : bool, optional
        Apply stellarator symmetry (default False).
    nfp : int, optional
        Number of field periods (default 1).
    coordinate_flag : str, optional
        'cartesian' or 'cylindrical' (default 'cartesian').
    R0 : float, optional
        Major radius for cylindrical coordinates (default 1.0).
    scale_factor : float, optional
        Global scale factor applied to dipole_moments (default 1.0).
    surf_pts : jnp.ndarray, shape (M, 3), optional
        Surface quadrature points. If provided with surf_n, G is precomputed.
    surf_n : jnp.ndarray, shape (M, 3), optional
        Surface outward unit normals. Required with surf_pts to build G.
    """
    def __init__(self, dipole_positions, dipole_moments, pho_values,
                 stellsym=False, nfp=1, coordinate_flag='cartesian',
                 R0=1.0, scale_factor=1.0,
                 surf_pts=None, surf_n=None):
        self.mu0_over_4pi = 1e-7
        self.R0 = R0
        self.pho_values = pho_values
        self.scale_factor = scale_factor
        scaled_moments = dipole_moments * scale_factor
        self.dipole_positions, self.dipole_moments = self._apply_symmetries(
            dipole_positions, scaled_moments, stellsym, nfp, coordinate_flag)
        self.n_dipoles = self.dipole_positions.shape[0]
        self._last_field = None
        self._last_eval_points = None
        self._compute_field = jit(vmap(
            lambda x: jnp.sum(vmap(
                lambda pos, mom: self._compute_single_dipole_field(x, pos, mom),
                in_axes=(0, 0))(self.dipole_positions, self.dipole_moments), axis=0),
            in_axes=0))

        # Precompute interaction matrix G if surface points are provided.
        #
        # G[i, j] = Bn contribution of magnet j at surface point i at pho=1.
        #
        # During optimization only pho changes — magnet positions and orientations
        # are fixed. So we compute G once here (~8s for 99k magnets) and each
        # Adam step is just: Bn_total = G @ pho + Bn_fixed  (~0.15s).
        #
        # Using DipoleField.B() directly in the loop recomputes all distances
        # and angles every step, making it ~1000x slower.
        if surf_pts is not None and surf_n is not None:
            from essos.optimization import compute_G_parallel
            self.G = compute_G_parallel(self, surf_pts, surf_n)
        else:
            self.G = None

    @staticmethod
    @jit
    def _compute_single_dipole_field(x_eval, pos, mom):
        """Magnetic field from a single dipole at x_eval (Biot-Savart)."""
        mu0_over_4pi = 1e-7
        r_vec = x_eval - pos
        r_mag = jnp.linalg.norm(r_vec) + 1e-12
        r_hat = r_vec / r_mag
        B = (3 * jnp.dot(mom, r_hat) / r_mag**3 * r_hat - mom / r_mag**3) * mu0_over_4pi
        return B

    @partial(jit, static_argnames=['self'])
    def compute_interaction_matrix(self, surf_pts, surf_n):
        """
        Build the interaction matrix G, shape (n_surf_pts, n_dipoles).

        G[i, j] = normal component of B from magnet j at surface point i,
        evaluated at unit pho (full magnetization).

        Called automatically in __init__ when surf_pts and surf_n are provided.
        Can also be called manually if surface points change after construction.
        """
        positions = self.dipole_positions
        moments = self.dipole_moments

        def calc_matrix_column(mag_idx):
            pos_j = positions[mag_idx]
            mom_j = moments[mag_idx]
            B_vectors = vmap(lambda x: self._compute_single_dipole_field(x, pos_j, mom_j))(surf_pts)
            Bn_column = jnp.sum(B_vectors * surf_n, axis=1)
            return Bn_column

        magnet_indices = jnp.arange(self.n_dipoles)
        G_T = vmap(calc_matrix_column)(magnet_indices)
        return G_T.T

    @partial(jit, static_argnames=['self'])
    def B(self, eval_points, chunk_size=512):
        """Magnetic field at eval_points (with caching).

        BUG FIX: a single point has shape (3,), where shape[0]==3 was
        being misread as "3 points" by the caching/shape-comparison
        logic below (which assumes eval_points is always a batch of
        shape (n_points, 3)). This both computed the WRONG field (the
        3 components got treated as 3 separate 1D points, producing a
        (3,3) output instead of a (3,) field vector) and could return
        a STALE cached result whenever two different calls happened to
        share the same shape[0]. Single points are now explicitly
        reshaped to (1,3), computed, and squeezed back to (3,)."""
        is_single_point = eval_points.ndim == 1 and eval_points.shape[0] == 3
        query_points = eval_points.reshape(1, 3) if is_single_point else eval_points

        # PERFORMANCE NOTE: caching was removed here. The previous
        # cache-validity check used jnp.array_equal(...) inside a
        # Python if-statement, which requires a concrete (non-traced)
        # boolean -- this works in eager calls but raises
        # TracerBoolConversionError the moment this method is called
        # under vmap/jit tracing (e.g. essos.dynamics's internal
        # energy-conservation diagnostics call field.AbsB via vmap).
        # Since _compute_field itself is now fast (lax.scan-based,
        # verified ~2.5s for the full 64x64/99k-magnet MUSE case),
        # simply recomputing every call is both correct under tracing
        # and not a meaningful performance regression.
        result = self._compute_field(query_points)
        return result[0] if is_single_point else result
    
    @partial(jit, static_argnames=['self'])
    def B_covariant(self, eval_points):
        return self.B(eval_points)
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, eval_points):
        return self.B(eval_points)
    
    @partial(jit, static_argnames=['self'])
    def AbsB(self, eval_points):
        return jnp.linalg.norm(self.B(eval_points), axis=-1)
    
    def dAbsB_by_dX(self, eval_points, eps=1e-6):
        """Gradient of |B| via finite differences."""
        is_single_point = len(eval_points.shape) == 1 and eval_points.shape[0] == 3
        if is_single_point:
            eval_points = eval_points.reshape(1, 3)
        elif len(eval_points.shape) != 2 or eval_points.shape[1] != 3:
            raise ValueError(f"eval_points must be shape (n,3) or (3,), got {eval_points.shape}")
        n_points = len(eval_points)
        grad_B = jnp.zeros((n_points, 3))
        for i in range(3):
            delta = jnp.zeros((n_points, 3)).at[:, i].set(eps)
            grad_B = grad_B.at[:, i].set((self.AbsB(eval_points + delta) - self.AbsB(eval_points - delta)) / (2 * eps))
        if is_single_point:
            grad_B = grad_B.squeeze(0)
        return grad_B
    
    def update_dipole_pho(self, dipole_idx, new_pho, eval_points):
        """Incrementally update one dipole's pho and recompute the field."""
        if self._last_field is None or len(eval_points) != self._last_eval_points.shape[0]:
            raise ValueError("Call B with the same eval_points before updating a dipole.")
        old_moment = self.dipole_moments[dipole_idx]
        old_magnitude = jnp.linalg.norm(old_moment)
        if new_pho == 0:
            new_moment = jnp.array([0.0, 0.0, 0.0])
        else:
            old_pho = self.pho_values[dipole_idx]
            scale_factor = new_pho / old_pho if old_pho != 0 else new_pho
            new_magnitude = old_magnitude * scale_factor
            new_moment = old_moment * (new_magnitude / old_magnitude) if new_magnitude != 0 else jnp.array([0.0, 0.0, 0.0])
        new_contribs = vmap(lambda x: self._compute_single_dipole_field(x, self.dipole_positions[dipole_idx], new_moment))(eval_points)
        old_contribs = vmap(lambda x: self._compute_single_dipole_field(x, self.dipole_positions[dipole_idx], old_moment))(eval_points)
        updated_field = self._last_field - old_contribs + new_contribs
        self._last_field = updated_field
        self.pho_values = self.pho_values.at[dipole_idx].set(new_pho)
        self.dipole_moments = self.dipole_moments.at[dipole_idx].set(new_moment)
        return updated_field
    
    def _apply_symmetries(self, positions, moments, stellsym=False, nfp=1, coordinate_flag='cartesian'):
        """Apply stellarator symmetries to positions and moments."""
        step = 1
        pos = positions[::step]
        mom = moments[::step]
        if coordinate_flag == 'cylindrical':
            phi_dipole = jnp.arctan2(pos[:, 1], pos[:, 0])
            mom = jnp.stack([mom[:, 0] * jnp.cos(phi_dipole) - mom[:, 1] * jnp.sin(phi_dipole),
                             mom[:, 0] * jnp.sin(phi_dipole) + mom[:, 1] * jnp.cos(phi_dipole),
                             mom[:, 2]], axis=1)
        all_pos, all_mom = [], []
        stell_list = [1.0] if not stellsym else [1.0, -1.0]
        for stell in stell_list:
            pos_stell = pos * jnp.array([1.0, stell, stell])
            mom_stell = mom * jnp.array([stell, 1.0, 1.0]) if stellsym else mom
            for i in range(nfp):
                angle = 2 * jnp.pi * i / nfp
                R = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0.0],
                               [jnp.sin(angle), jnp.cos(angle), 0.0],
                               [0.0, 0.0, 1.0]])
                all_pos.append(pos_stell @ R.T)
                all_mom.append(mom_stell @ R.T)
        return jnp.concatenate(all_pos, axis=0), jnp.concatenate(all_mom, axis=0)

class DipoleField:
    """
    Magnetic field from a collection of magnetic dipoles.

    Supports optional precomputation of the interaction matrix G for fast
    optimization. When surf_pts and surf_n are provided at construction,
    G is computed once in __init__ and stored as self.G. The optimizer
    then uses the fast matrix-vector multiply:

        Bn_total = self.G @ pho + Bn_fixed

    rather than recomputing dipole geometry every step. This follows the
    same pattern as DESC's ObjectiveFunction.build() — expensive geometry
    is precomputed once, and compute() (or in our case the Adam step) is
    just fast arithmetic.

    Parameters
    ----------
    dipole_positions : jnp.ndarray, shape (N, 3)
        Magnet center positions [m].
    dipole_moments : jnp.ndarray, shape (N, 3)
        Dipole moment vectors [A·m²].
    pho_values : jnp.ndarray, shape (N,)
        Magnet strengths in [-1, 1].
    stellsym : bool, optional
        Apply stellarator symmetry (default False).
    nfp : int, optional
        Number of field periods (default 1).
    coordinate_flag : str, optional
        'cartesian' or 'cylindrical' (default 'cartesian').
    R0 : float, optional
        Major radius for cylindrical coordinates (default 1.0).
    scale_factor : float, optional
        Global scale factor applied to dipole_moments (default 1.0).
    surf_pts : jnp.ndarray, shape (M, 3), optional
        Surface quadrature points. If provided with surf_n, G is precomputed.
    surf_n : jnp.ndarray, shape (M, 3), optional
        Surface outward unit normals. Required with surf_pts to build G.
    """
    def __init__(self, dipole_positions, dipole_moments, pho_values,
                 stellsym=False, nfp=1, coordinate_flag='cartesian',
                 R0=1.0, scale_factor=1.0,
                 surf_pts=None, surf_n=None):
        self.mu0_over_4pi = 1e-7
        self.nfp = nfp
        self.R0 = R0
        self.pho_values = pho_values
        self.scale_factor = scale_factor
        scaled_moments = dipole_moments * scale_factor

        self.dipole_positions = dipole_positions
        self.dipole_moments = scaled_moments
        
        self.dipole_positions_full, self.dipole_moments_full = self._apply_symmetries(
            dipole_positions, scaled_moments, stellsym, coordinate_flag)
        self.n_dipoles = self.dipole_positions.shape[0]
        self._last_field = None
        self._last_eval_points = None
        # PERFORMANCE FIX: the original implementation double-vmapped 
        DIPOLE_CHUNK_SIZE = 5000

        def _field_kernel(eval_pts, dip_pos, dip_mom):
            """Vectorized B at eval_pts from a batch of dipoles (dip_pos,
            dip_mom), broadcasting over both -- no per-pair function calls."""
            P = eval_pts[:, None, :]           # (n_pts, 1, 3)
            Pos = dip_pos[None, :, :]          # (1, n_chunk, 3)
            Mom = dip_mom[None, :, :]          # (1, n_chunk, 3)
            R = P - Pos                        # (n_pts, n_chunk, 3)
            R_mag = jnp.linalg.norm(R, axis=-1, keepdims=True) + 1e-12
            dot_mr = jnp.sum(Mom * R, axis=-1, keepdims=True)
            term1 = 3.0 * dot_mr * R / (R_mag ** 5)
            term2 = Mom / (R_mag ** 3)
            B_per_dipole = (term1 - term2) * 1e-7   # mu0_over_4pi
            return jnp.sum(B_per_dipole, axis=1)     # (n_pts, 3)

        def _compute_field_chunked(eval_pts):
            n_dip_full = self.dipole_positions_full.shape[0]
            n_chunks = (n_dip_full + DIPOLE_CHUNK_SIZE - 1) // DIPOLE_CHUNK_SIZE
            pad = n_chunks * DIPOLE_CHUNK_SIZE - n_dip_full
            if pad > 0:
                pos_padded = jnp.concatenate([
                    self.dipole_positions_full,
                    jnp.zeros((pad, 3), self.dipole_positions_full.dtype)], axis=0)
                mom_padded = jnp.concatenate([
                    self.dipole_moments_full,
                    jnp.zeros((pad, 3), self.dipole_moments_full.dtype)], axis=0)
            else:
                pos_padded = self.dipole_positions_full
                mom_padded = self.dipole_moments_full
            pos_chunks = pos_padded.reshape(n_chunks, DIPOLE_CHUNK_SIZE, 3)
            mom_chunks = mom_padded.reshape(n_chunks, DIPOLE_CHUNK_SIZE, 3)

            def scan_body(carry, chunk):
                dip_pos_c, dip_mom_c = chunk
                contribution = _field_kernel(eval_pts, dip_pos_c, dip_mom_c)
                return carry + contribution, None

            n_pts = eval_pts.shape[0]
            init = jnp.zeros((n_pts, 3), eval_pts.dtype)
            total, _ = lax.scan(scan_body, init, (pos_chunks, mom_chunks))
            return total

        self._compute_field = jit(_compute_field_chunked)

        # During optimization only pho changes. magnet positions and orientations
        # are fixed. So we compute G once here (~8s for 99k magnets) and each
        # Adam step is just: Bn_total = G @ pho + Bn_fixed  (~0.15s).
        # Using DipoleField.B() directly in the loop recomputes all distances
        # and angles every step, making it ~1000x slower.
        if surf_pts is not None and surf_n is not None:
            from essos.optimization import compute_G_parallel
            self.G = compute_G_parallel(self, surf_pts, surf_n)
        else:
            self.G = None

    @staticmethod
    @jit
    def _compute_single_dipole_field(x_eval, pos, mom):
        """Magnetic field from a single dipole at x_eval (Biot-Savart)."""
        mu0_over_4pi = 1e-7
        r_vec = x_eval - pos
        r_mag = jnp.linalg.norm(r_vec) + 1e-12
        r_hat = r_vec / r_mag
        B = (3 * jnp.dot(mom, r_hat) / r_mag**3 * r_hat - mom / r_mag**3) * mu0_over_4pi
        return B

    def compute_interaction_matrix(self,surf_pts, surf_n, stellsym=True):
        """
        Build G (n_surf, n_mag) summing contributions from all symmetric copies.
        Uses pmap for fast parallel computation.
        """
        nfp = self.nfp 
        positions = self.dipole_positions
        moments = self.dipole_moments
        

        def _bn_one_copy_pmap(surf_pts, surf_n, mag_pos, mag_mom):
            """Bn at surf_pts from magnets. Uses pmap for parallelism."""
            n_devices = jax.device_count()
            n_points  = len(surf_pts)
            remainder = n_points % n_devices
            if remainder != 0:
                pad = n_devices - remainder
                surf_pts = jnp.concatenate([surf_pts, jnp.zeros((pad, 3), surf_pts.dtype)])
                surf_n   = jnp.concatenate([surf_n,   jnp.zeros((pad, 3), surf_n.dtype)])
        
            batch = len(surf_pts) // n_devices
            pts_s = surf_pts.reshape(n_devices, batch, 3)
            n_s   = surf_n.reshape(n_devices, batch, 3)
        
            m_pos = jnp.array(mag_pos)
            m_mom = jnp.array(mag_mom)
        
            def kernel(pts, norms):
                P      = jnp.expand_dims(pts,   1)
                M_pos  = jnp.expand_dims(m_pos, 0)
                M_vec  = jnp.expand_dims(m_mom, 0)
                N      = jnp.expand_dims(norms, 1)
                R      = P - M_pos
                R_mag  = jnp.linalg.norm(R, axis=2, keepdims=True)
                dot_mr = jnp.sum(M_vec * R, axis=2, keepdims=True)
                dot_rn = jnp.sum(R * N,     axis=2, keepdims=True)
                dot_mn = jnp.sum(M_vec * N, axis=2, keepdims=True)
                term1  = 3.0 * dot_mr * dot_rn / (R_mag**5 + 1e-30)
                term2  = -dot_mn / (R_mag**3 + 1e-30)
                return jnp.squeeze((term1 + term2) * 1e-7, axis=2)
        
            pts_d = jax.device_put_sharded(list(pts_s), jax.local_devices())
            n_d   = jax.device_put_sharded(list(n_s),   jax.local_devices())
            G_s   = jax.pmap(kernel)(pts_d, n_d)
            G_s.block_until_ready()
            G_full = G_s.reshape(-1, len(m_pos))
            if remainder != 0:
                G_full = G_full[:n_points]
            return G_full
    
        n_surf = len(surf_pts)
        n_mag  = len(positions)
        G = jnp.zeros((n_surf, n_mag), dtype=jnp.float32)
    
        stell_list = [1.0, -1.0] if stellsym else [1.0]
        for stell in stell_list:
            pos_s = positions * jnp.array([1.0, stell, stell])
            mom_s = moments * jnp.array([stell, 1.0, 1.0]) if stellsym else moments
            for i in range(nfp):
                angle = 2 * jnp.pi * i / nfp
                c, s  = jnp.cos(angle), jnp.sin(angle)
                R_mat = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0., 0., 1.0]])
                pos_r = pos_s @ R_mat.T
                mom_r = mom_s @ R_mat.T
                G = G + _bn_one_copy_pmap(surf_pts, surf_n, pos_r, mom_r)
    
        return G



    @partial(jit, static_argnames=['self'])
    def B(self, eval_points, chunk_size=512):
        """Magnetic field at eval_points (with caching).

        BUG FIX: a single point has shape (3,), where shape[0]==3 was
        being misread as "3 points" by the caching/shape-comparison
        logic below (which assumes eval_points is always a batch of
        shape (n_points, 3)). This both computed the WRONG field (the
        3 components got treated as 3 separate 1D points, producing a
        (3,3) output instead of a (3,) field vector) and could return
        a STALE cached result whenever two different calls happened to
        share the same shape[0]. Single points are now explicitly
        reshaped to (1,3), computed, and squeezed back to (3,)."""
        is_single_point = eval_points.ndim == 1 and eval_points.shape[0] == 3
        query_points = eval_points.reshape(1, 3) if is_single_point else eval_points

        # PERFORMANCE NOTE: caching was removed here. The previous
        # cache-validity check used jnp.array_equal(...) inside a
        # Python if-statement, which requires a concrete (non-traced)
        # boolean -- this works in eager calls but raises
        # TracerBoolConversionError the moment this method is called
        # under vmap/jit tracing (e.g. essos.dynamics's internal
        # energy-conservation diagnostics call field.AbsB via vmap).
        # Since _compute_field itself is now fast (lax.scan-based,
        # verified ~2.5s for the full 64x64/99k-magnet MUSE case),
        # simply recomputing every call is both correct under tracing
        # and not a meaningful performance regression.
        result = self._compute_field(query_points)
        return result[0] if is_single_point else result
    
    @partial(jit, static_argnames=['self'])
    def B_covariant(self, eval_points):
        return self.B(eval_points)
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, eval_points):
        return self.B(eval_points)
    
    @partial(jit, static_argnames=['self'])
    def AbsB(self, eval_points):
        return jnp.linalg.norm(self.B(eval_points), axis=-1)
    
    def dAbsB_by_dX(self, eval_points, eps=1e-6):
        """Gradient of |B| via finite differences."""
        is_single_point = len(eval_points.shape) == 1 and eval_points.shape[0] == 3
        if is_single_point:
            eval_points = eval_points.reshape(1, 3)
        elif len(eval_points.shape) != 2 or eval_points.shape[1] != 3:
            raise ValueError(f"eval_points must be shape (n,3) or (3,), got {eval_points.shape}")
        n_points = len(eval_points)
        grad_B = jnp.zeros((n_points, 3))
        for i in range(3):
            delta = jnp.zeros((n_points, 3)).at[:, i].set(eps)
            grad_B = grad_B.at[:, i].set((self.AbsB(eval_points + delta) - self.AbsB(eval_points - delta)) / (2 * eps))
        if is_single_point:
            grad_B = grad_B.squeeze(0)
        return grad_B
    
    def update_dipole_pho(self, dipole_idx, new_pho, eval_points):
        """Incrementally update one dipole's pho and recompute the field."""
        if self._last_field is None or len(eval_points) != self._last_eval_points.shape[0]:
            raise ValueError("Call B with the same eval_points before updating a dipole.")
        old_moment = self.dipole_moments[dipole_idx]
        old_magnitude = jnp.linalg.norm(old_moment)
        if new_pho == 0:
            new_moment = jnp.array([0.0, 0.0, 0.0])
        else:
            old_pho = self.pho_values[dipole_idx]
            scale_factor = new_pho / old_pho if old_pho != 0 else new_pho
            new_magnitude = old_magnitude * scale_factor
            new_moment = old_moment * (new_magnitude / old_magnitude) if new_magnitude != 0 else jnp.array([0.0, 0.0, 0.0])
        new_contribs = vmap(lambda x: self._compute_single_dipole_field(x, self.dipole_positions[dipole_idx], new_moment))(eval_points)
        old_contribs = vmap(lambda x: self._compute_single_dipole_field(x, self.dipole_positions[dipole_idx], old_moment))(eval_points)
        updated_field = self._last_field - old_contribs + new_contribs
        self._last_field = updated_field
        self.pho_values = self.pho_values.at[dipole_idx].set(new_pho)
        self.dipole_moments = self.dipole_moments.at[dipole_idx].set(new_moment)
        return updated_field
    
    def _apply_symmetries(self, positions, moments, stellsym=False, coordinate_flag='cartesian'):
        """Apply stellarator symmetries to positions and moments."""
        step = 1
        pos = positions[::step]
        mom = moments[::step]
        if coordinate_flag == 'cylindrical':
            phi_dipole = jnp.arctan2(pos[:, 1], pos[:, 0])
            mom = jnp.stack([mom[:, 0] * jnp.cos(phi_dipole) - mom[:, 1] * jnp.sin(phi_dipole),
                             mom[:, 0] * jnp.sin(phi_dipole) + mom[:, 1] * jnp.cos(phi_dipole),
                             mom[:, 2]], axis=1)
        all_pos, all_mom = [], []
        stell_list = [1.0] if not stellsym else [1.0, -1.0]
        for stell in stell_list:
            pos_stell = pos * jnp.array([1.0, stell, stell])
            mom_stell = mom * jnp.array([stell, 1.0, 1.0]) if stellsym else mom
            for i in range(self.nfp):
                angle = 2 * jnp.pi * i / self.nfp
                R = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0.0],
                               [jnp.sin(angle), jnp.cos(angle), 0.0],
                               [0.0, 0.0, 1.0]])
                all_pos.append(pos_stell @ R.T)
                all_mom.append(mom_stell @ R.T)
        return jnp.concatenate(all_pos, axis=0), jnp.concatenate(all_mom, axis=0)
