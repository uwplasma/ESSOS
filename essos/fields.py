import jax
jax.config.update("jax_enable_x64", True)
from jax import vmap
from essos.coils import Curves
import jax.numpy as jnp
import numpy as np
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

def _radial_interp(s, grid, table, xm, covariant_s=False, half_grid=False, axis_m1=None):
    """Interpolate every Fourier mode of a wout table at ``s``.

    ``table`` is on the full grid, or on the half grid with VMEC's unused
    first row (``half_grid=True``); both grids are uniform. Near the magnetic
    axis the modes of a regular scalar vanish as ``s**(m/2)``, and those of
    B_s (``covariant_s=True``) one power of ``sqrt(s)`` lower. Each mode is
    therefore divided by ``s**p``, interpolated linearly and multiplied back,
    with ``p = min(m, 2 + m % 2) / 2`` (less 1 for B_s, at least -1/2; 0 for
    m = 0). For m > 0 the axis row of a full-grid table is replaced by the
    extrapolation of the next two rows, or, for m = 1, by ``axis_m1`` when it
    is given. Interpolating the modes themselves leaves the m > 0 terms
    finite on the axis, where |B| then depends on theta.
    """
    m = np.asarray(xm).astype(int)
    k = np.minimum(m, 2 + m % 2)  # 2 p
    if covariant_s:
        k = np.where(m > 0, np.maximum(k - 2, -1), 0)
    with jax.ensure_compile_time_eval():  # folded at trace time for a concrete table
        if half_grid:
            table = table[1:]
        scaled = table / jnp.where(grid > 0, grid, 1.0)[:, None]**(k / 2)
        if not half_grid:
            scaled = scaled.at[0].set(jnp.where(m > 0, 2 * scaled[1] - scaled[2], scaled[0]))
            if axis_m1 is not None:
                scaled = scaled.at[0].set(jnp.where(m == 1, axis_m1, scaled[0]))
    ds = grid[1] - grid[0]
    i = jnp.clip(jnp.floor((s - grid[0]) / ds).astype(int), 0, len(grid) - 2)
    t = jnp.where(s > grid[-1], 1.0, (s - grid[i]) / ds)
    q = jnp.sqrt(jnp.maximum(s, jnp.finfo(jnp.result_type(s, float)).tiny))
    powers = jnp.stack([1 / q, jnp.ones_like(q), q, q * q, q * q * q])  # q**(2 p) for 2 p = -1..3
    return (powers @ (k == np.arange(-1, 4)[:, None])) * ((1 - t) * scaled[i] + t * scaled[i + 1])

class Vmec():
    """VMEC equilibrium from a wout file.

    ``mode_tolerance`` drops a Fourier mode when, in every table of its set,
    its largest amplitude over the radial grid is below that fraction of the
    table's largest: R and Z for the geometry modes, and |B|, sqrt(g) and the
    B components for the Nyquist modes. Evaluation cost scales with the
    number of modes kept.
    """
    def __init__(self, wout_filename, ntheta=50, nphi=50, close=True, range_torus='full torus', mode_tolerance=0.0):
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
        if mode_tolerance > 0:
            self._drop_small_modes(mode_tolerance)
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
        
    def _drop_small_modes(self, tolerance):
        for tables, numbers in ((('rmnc', 'zmns'), ('xm', 'xn')),
                                (('bmnc', 'gmnc', 'bsubsmns', 'bsubumnc', 'bsubvmnc', 'bsupumnc', 'bsupvmnc'),
                                 ('xm_nyq', 'xn_nyq'))):
            amplitude = [np.abs(np.asarray(getattr(self, name))).max(axis=0) for name in tables]
            keep = np.any([a > tolerance * a.max() for a in amplitude], axis=0)
            for name in tables + numbers:
                setattr(self, name, getattr(self, name)[..., keep])

    @property
    def surface(self):
        return self._surface

    def _bsubs_axis_m1(self):
        """Axis limit of sqrt(s) B_s for the m = 1 modes, from B_theta.

        Near the axis the leading m = 1 parts of B_s and B_theta are the
        gradient of sqrt(s) Psi(theta, phi), so sqrt(s) B_s tends to
        B_theta / (2 sqrt(s)) and their contributions to the toroidal current
        cancel. VMEC's B_s next to the axis misses that limit (by about 10% on
        an HSX wout), and extrapolating it gave curl B a toroidal component
        that grew as 1/sqrt(s) on the axis.
        """
        with jax.ensure_compile_time_eval():
            b_theta = self.bsubumnc[1:3] / jnp.sqrt(self.s_half_grid[:2])[:, None]
            return (1.5 * b_theta[0] - 0.5 * b_theta[1]) / 2
        
    # Nyquist tables: (on the half grid, _radial_interp options, cosine series).
    _NYQUIST = {'bmnc': (True, {}, True), 'gmnc': (True, {}, True),
                'bsubsmns': (False, {'covariant_s': True}, False),
                'bsubumnc': (True, {}, True), 'bsubvmnc': (True, {}, True),
                'bsupumnc': (True, {}, True), 'bsupvmnc': (True, {}, True)}

    @partial(jit, static_argnames=['self'])
    def _nyquist_series(self, points):
        """Each Nyquist table's Fourier sum at ``points`` and its (s, theta, phi) gradient.

        One set of angles, cosines, sines and radial weights serves every
        table and the gradients are analytic, so |B|, sqrt(g), the B components
        and their derivatives cost one evaluation between them when traced
        together, and curl b and the curvature need no automatic differentiation.
        """
        s, theta, phi = points
        angle = self.xm_nyq * theta - self.xn_nyq * phi
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        series = {}
        for name, (half_grid, options, is_cos) in self._NYQUIST.items():
            grid = self.s_half_grid if half_grid else self.s_full_grid
            if name == 'bsubsmns':
                options = dict(options, axis_m1=self._bsubs_axis_m1())
            f, df = jax.jvp(lambda s: _radial_interp(s, grid, getattr(self, name), self.xm_nyq,
                                                     half_grid=half_grid, **options), (s,), (jnp.ones_like(s),))
            if is_cos:
                series[name] = (f @ cos, jnp.array([df @ cos, -(self.xm_nyq * f) @ sin, (self.xn_nyq * f) @ sin]))
            else:
                series[name] = (f @ sin, jnp.array([df @ sin, (self.xm_nyq * f) @ cos, -(self.xn_nyq * f) @ cos]))
        return series

    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        series = self._nyquist_series(points)
        return jnp.array([series[name][0] for name in ('bsubsmns', 'bsubumnc', 'bsubvmnc')])

    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        series = self._nyquist_series(points)
        B_sup_theta, B_sup_phi = series['bsupumnc'][0], series['bsupvmnc'][0]
        return jnp.array([0*B_sup_theta, B_sup_theta, B_sup_phi])

    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        return self._nyquist_series(points)['gmnc'][0]

    @partial(jit, static_argnames=['self'])
    def B(self, points):
        s, theta, phi = points
        gmnc_interp = _radial_interp(s, self.s_half_grid, self.gmnc, self.xm_nyq, half_grid=True)
        rmnc_interp = _radial_interp(s, self.s_full_grid, self.rmnc, self.xm)
        zmns_interp = _radial_interp(s, self.s_full_grid, self.zmns, self.xm)
        d_rmnc_d_s_interp = jacfwd(_radial_interp)(s, self.s_full_grid, self.rmnc, self.xm)
        d_zmns_d_s_interp = jacfwd(_radial_interp)(s, self.s_full_grid, self.zmns, self.xm)
        
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
        return self._nyquist_series(points)['bmnc'][0]
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)


    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return self._nyquist_series(points)['bmnc'][1]
    
    @partial(jit, static_argnames=['self'])
    def grad_B_covariant(self, points):
        series = self._nyquist_series(points)
        return jnp.stack([series[name][1] for name in ('bsubsmns', 'bsubumnc', 'bsubvmnc')])
 
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
        rmnc_interp = _radial_interp(s, self.s_full_grid, self.rmnc, self.xm)
        zmns_interp = _radial_interp(s, self.s_full_grid, self.zmns, self.xm)
        cosangle = jnp.cos(self.xm * theta - self.xn * phi)
        sinangle = jnp.sin(self.xm * theta - self.xn * phi)
        R = jnp.dot(rmnc_interp, cosangle)
        Z = jnp.dot(zmns_interp, sinangle)
        X = R * jnp.cos(phi)
        Y = R * jnp.sin(phi)
        return jnp.array([X, Y, Z])

    def _boundary_rz(self, theta, phi):
        """R, Z of the LCFS and their first two theta derivatives, on a theta array."""
        angle = self.xm * theta[..., None] - self.xn * phi
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        r, z, m = self.rmnc[-1], self.zmns[-1], self.xm
        return (cos @ r, sin @ z, -sin @ (m * r), cos @ (m * z), -cos @ (m * m * r), -sin @ (m * m * z))

    @partial(jit, static_argnames=['self'])
    def boundary_distance(self, xyz):
        """Signed distance [m] from a Cartesian point to the LCFS, in its phi = const plane.

        Positive inside. The nearest point of the LCFS cross-section is found
        on 64 poloidal nodes and refined by Newton iterations in theta, so the
        distance is smooth and exact to rounding near the surface: its zero
        is the LCFS of :meth:`to_xyz` at s = 1.
        """
        R, Z, phi = jnp.hypot(xyz[0], xyz[1]), xyz[2], jnp.arctan2(xyz[1], xyz[0])
        grid = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        Rb, Zb = self._boundary_rz(grid, phi)[:2]
        theta = grid[jnp.argmin((R - Rb)**2 + (Z - Zb)**2)]

        def newton(theta, _):
            Rb, Zb, dR, dZ, d2R, d2Z = self._boundary_rz(theta, phi)
            slope = -(R - Rb) * dR - (Z - Zb) * dZ
            curvature = dR**2 + dZ**2 - (R - Rb) * d2R - (Z - Zb) * d2Z
            return theta - slope / jnp.where(curvature > 0, curvature, dR**2 + dZ**2), None

        theta, _ = lax.scan(newton, theta, None, length=4)
        Rb, Zb, dR, dZ = self._boundary_rz(theta, phi)[:4]
        # VMEC's theta runs either way round; the sign of the enclosed area fixes the outward normal.
        with jax.ensure_compile_time_eval():
            Rc, _, _, dZc = self._boundary_rz(grid, 0.0)[:4]
            orientation = jnp.sign(jnp.sum(Rc * dZc))
        outward = orientation * ((R - Rb) * dZ - (Z - Zb) * dR)
        return -jnp.sign(outward) * jnp.hypot(R - Rb, Z - Zb)

    @partial(jit, static_argnames=['self'])
    def flux_coordinates(self, xyz):
        """Invert :meth:`to_xyz`: a Cartesian point to (s, theta, phi), and the residual [m].

        Newton iterations in (sqrt(s) cos theta, sqrt(s) sin theta), which is
        regular on the axis, from the nearest of 12 x 32 nodes of the
        cross-section at the point's phi. Points outside the LCFS return
        s > 1 only as far as the extrapolated geometry allows; check the
        residual.
        """
        R, Z = jnp.hypot(xyz[0], xyz[1]), xyz[2]
        phi = jnp.mod(jnp.arctan2(xyz[1], xyz[0]), 2 * jnp.pi)
        target = jnp.array([R, Z])

        def rz(x):
            p = self.to_xyz(jnp.array([x[0]**2 + x[1]**2, jnp.arctan2(x[1], x[0]), phi]))
            return jnp.array([jnp.hypot(p[0], p[1]), p[2]])

        rho, theta = [a.ravel() for a in jnp.meshgrid(jnp.linspace(0.08, 1.0, 12),
                                                     jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False))]
        seeds = jnp.stack([rho * jnp.cos(theta), rho * jnp.sin(theta)], 1)
        x = seeds[jnp.argmin(jnp.sum((vmap(rz)(seeds) - target)**2, 1))]

        def newton(x, _):
            dx = jnp.linalg.solve(jacfwd(rz)(x), rz(x) - target)
            return x - dx * jnp.minimum(1.0, 0.1 / (jnp.linalg.norm(dx) + 1e-300)), None

        x, _ = lax.scan(newton, x, None, length=40)
        s = x[0]**2 + x[1]**2
        return jnp.array([s, jnp.mod(jnp.arctan2(x[1], x[0]), 2 * jnp.pi), phi]), jnp.linalg.norm(rz(x) - target)

class near_axis:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "The 'near_axis' class has been migrated to the standalone 'pyQSC_JAX' repository. "
            "Please run 'pip install git+https://github.com/uwplasma/pyQSC_JAX.git' "
            "and import it via 'from pyqsc_jax.near_axis import near_axis'."
        )


class ExternalField(MagneticField):
    """A Cartesian field from a batched source, for tracing one point at a time.

    ``source`` is an object with ``b_cyl(R, phi, Z) -> (B_R, B_phi, B_Z)``
    (a VMEX ``MgridField``), an object with a batched ``B(points)`` for
    points of shape ``(n, 3)`` (a VMEX ``VmecExtender``), or a callable
    ``xyz (n, 3) -> B (n, 3)``, in metres and tesla. It must be traceable by
    JAX; the derivatives the guiding-center equations need come from
    automatic differentiation of it.
    """

    def __init__(self, source):
        self.source = source

    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        return 1.

    @partial(jit, static_argnames=['self'])
    def B(self, points):
        if hasattr(self.source, "b_cyl"):
            R, phi = jnp.hypot(points[0], points[1]), jnp.arctan2(points[1], points[0])
            BR, Bphi, BZ = (jnp.ravel(b)[0] for b in self.source.b_cyl(R[None], phi[None], points[2][None]))
            return jnp.array([BR * jnp.cos(phi) - Bphi * jnp.sin(phi), BR * jnp.sin(phi) + Bphi * jnp.cos(phi), BZ])
        batched = self.source.B if hasattr(self.source, "B") else self.source
        return batched(points[None])[0]

    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        return self.B(points)

    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        return self.B(points)

    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points))

    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)

    @partial(jit, static_argnames=['self'])
    def grad_B_covariant(self, points):
        return jacfwd(self.B)(points)

    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        g = self.grad_B_covariant(points)
        return jnp.array([g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]])

    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return (self.curl_B(points) / self.AbsB(points)
                + jnp.cross(self.B(points), self.dAbsB_by_dX(points)) / self.AbsB(points)**2)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B(points), self.curl_b(points)) / self.AbsB(points)

    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        return points


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

