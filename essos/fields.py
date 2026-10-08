import os
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
    def gc_quantities(self, points):
        """Field quantities of the guiding-center equations at one point.

        Returns ``(B_covariant, B_contravariant, |B|, grad|B|, curl b, kappa,
        sqrtg)``. This generic version calls the individual methods; fields
        that can form all of them from one evaluation of B and its gradient
        (see :class:`BiotSavart`) override it.
        """
        return (
            self.B_covariant(points),
            self.B_contravariant(points),
            self.AbsB(points),
            self.dAbsB_by_dX(points),
            self.curl_b(points),
            self.kappa(points),
            self.sqrtg(points),
        )

    @jit
    def to_xyz(self, points):
        raise NotImplementedError("to_xyz method not implemented")

def _gc_from_jacobian(field, jacobian):
    """Guiding-center quantities from B and dB/dX in Cartesian coordinates (sqrtg = 1)."""
    magnitude = jnp.linalg.norm(field)
    grad_magnitude = jacobian.T @ (field / magnitude)
    curl_field = jnp.array([jacobian[2, 1] - jacobian[1, 2],
                            jacobian[0, 2] - jacobian[2, 0],
                            jacobian[1, 0] - jacobian[0, 1]])
    curl_unit = curl_field / magnitude + jnp.cross(field, grad_magnitude) / magnitude**2
    curvature = -jnp.cross(field, curl_unit) / magnitude
    return field, field, magnitude, grad_magnitude, curl_unit, curvature, 1.0


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
    def gc_quantities(self, points):
        """Guiding-center field quantities from one pass over the coils.

        The separate methods each rebuild B and its Jacobian (``kappa`` even
        recomputes ``curl_b``). Here a single forward-mode Jacobian yields B
        and dB/dX together, and in Cartesian coordinates (sqrtg = 1)
        grad|B| = (dB/dX)^T b, curl b = curl B/|B| + B x grad|B|/|B|^2 and
        kappa = -B x curl b / |B| follow algebraically.
        """
        jacobian, field = jacfwd(lambda x: (self.B(x), self.B(x)), has_aux=True)(jnp.asarray(points))
        return _gc_from_jacobian(field, jacobian)

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

        # Lazily computed caches behind the read-only properties below.
        self._coils_length = None
        self._coils_curvature = None
        self._r_axis = None
        self._z_axis = None

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
        if self._coils_length is None:
            self._coils_length = jnp.array([jnp.mean(jnp.linalg.norm(d1gamma, axis=1)) for d1gamma in self.gamma_dash])
        return self._coils_length

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
    therefore divided by ``s**p``, interpolated and multiplied back,
    with ``p = min(m, 2 + m % 2) / 2`` (less 1 for B_s, at least -1/2; 0 for
    m = 0). For m > 0 the axis row of a full-grid table is replaced by the
    extrapolation of the next two rows, or, for m = 1, by ``axis_m1`` when it
    is given. Interpolating the modes themselves leaves the m > 0 terms
    finite on the axis, where |B| then depends on theta.

    The interpolation is cubic Hermite with centred-difference slopes (linear
    in the first interval): it keeps the grid values and its s derivative is
    continuous. With linear
    interpolation grad|B| jumps at every grid point, and a guiding center
    whose radial drift changes sign there chatters across it while the
    adaptive step shrinks to nothing; under vmap the whole batch then runs to
    max_steps.
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
        # Slopes per grid interval. Hermite interpolation is C1 for any nodal slopes; the first interval
        # keeps the linear form that the axis rows above are extrapolated with.
        slope = jnp.gradient(scaled, axis=0)
        slope = slope.at[:2].set(scaled[1] - scaled[0])
    ds = grid[1] - grid[0]
    i = jnp.clip(jnp.floor((s - grid[0]) / ds).astype(int), 0, len(grid) - 2)
    t = jnp.where(s > grid[-1], 1.0, (s - grid[i]) / ds)
    q = jnp.sqrt(jnp.maximum(s, jnp.finfo(jnp.result_type(s, float)).tiny))
    powers = jnp.stack([1 / q, jnp.ones_like(q), q, q * q, q * q * q])  # q**(2 p) for 2 p = -1..3
    h = jnp.stack([(1 + 2 * t) * (1 - t)**2, t * (1 - t)**2, t * t * (3 - 2 * t), t * t * (t - 1)])
    return (powers @ (k == np.arange(-1, 4)[:, None])) * (h[0] * scaled[i] + h[1] * slope[i]
                                                          + h[2] * scaled[i + 1] + h[3] * slope[i + 1])

VMEC_WOUT_ARRAYS = ('bmnc', 'xm', 'xn', 'rmnc', 'zmns', 'bsubsmns', 'bsubumnc', 'bsubvmnc',
                    'bsupumnc', 'bsupvmnc', 'gmnc', 'xm_nyq', 'xn_nyq', 'Aminor_p')
VMEC_WOUT_PARTNERS = {'rmnc': 'rmns', 'zmns': 'zmnc', 'bmnc': 'bmns', 'gmnc': 'gmns',
                      'bsubsmns': 'bsubsmnc', 'bsubumnc': 'bsubumns', 'bsubvmnc': 'bsubvmns',
                      'bsupumnc': 'bsupumns', 'bsupvmnc': 'bsupvmns'}

class Vmec():
    """VMEC equilibrium, including asymmetric Fourier partners, from a wout file, an
    in-memory wout object with the wout variable names (e.g. a VMEX ``WoutData``;
    no file is written and gradients flow) or live arrays (:meth:`from_arrays`).

    ``mode_tolerance`` drops a Fourier mode when, in every table of its set,
    its largest amplitude over the radial grid is below that fraction of the
    table's largest: R and Z for the geometry modes, and |B|, sqrt(g) and the
    B components for the Nyquist modes. Evaluation cost scales with the
    number of modes kept.
    """
    def __init__(self, wout_filename, ntheta=50, nphi=50, close=True, range_torus='full torus', mode_tolerance=0.0):
        if not isinstance(wout_filename, (str, os.PathLike)):  # an in-memory wout, e.g. a VMEX WoutData
            w, kwargs = wout_filename, dict(ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus,
                                            mode_tolerance=mode_tolerance)
            if bool(np.asarray(getattr(w, 'lasym', False))):
                kwargs.update({name: getattr(w, name) for name in VMEC_WOUT_PARTNERS.values()
                               if getattr(w, name, None) is not None})
            self.__dict__.update(Vmec.from_arrays(w.nfp, w.ns, *(getattr(w, name) for name in VMEC_WOUT_ARRAYS),
                                                  **kwargs).__dict__)
            return
        self.wout_filename = wout_filename
        from netCDF4 import Dataset
        self.nc = Dataset(self.wout_filename)
        try:
            variables = self.nc.variables
            lasym = any(bool(np.asarray(variables[name][:]).item())
                        for name in ('lasym__logical__', 'lasym') if name in variables)
            if lasym and any(name not in variables for name in VMEC_WOUT_PARTNERS.values()):
                raise ValueError("Asymmetric wout is missing Fourier partner tables")
            partners = {name: jnp.array(variables[name][:]) for name in VMEC_WOUT_PARTNERS.values()
                        if name in variables and np.any(variables[name][:])}
            self._set_state(nfp=int(self.nc.variables["nfp"][0]), ns=int(self.nc.variables["ns"][0]),
                            ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus,
                            mode_tolerance=mode_tolerance,
                            **{name: jnp.array(self.nc.variables[name][:]) for name in VMEC_WOUT_ARRAYS}, **partners)
        except BaseException:
            self.nc.close()
            raise

    @classmethod
    def from_arrays(cls, nfp, ns, bmnc, xm, xn, rmnc, zmns, bsubsmns, bsubumnc, bsubvmnc,
                    bsupumnc, bsupvmnc, gmnc, xm_nyq, xn_nyq, Aminor_p,
                    ntheta=50, nphi=50, close=True, range_torus='full torus', mode_tolerance=0.0, **partners):
        """Build a differentiable VMEC field from in-memory wout arrays.

        Metadata and mode numbers must be concrete. Positive ``mode_tolerance``
        also requires concrete coefficient tables to select modes by amplitude.
        Optional sine/cosine partner tables use their wout names (e.g. ``bmns``).
        """
        self = cls.__new__(cls)
        self.wout_filename = None
        self.nc = None
        self._set_state(nfp=int(nfp), ns=int(ns), bmnc=bmnc, xm=xm, xn=xn, rmnc=rmnc, zmns=zmns,
                        bsubsmns=bsubsmns, bsubumnc=bsubumnc, bsubvmnc=bsubvmnc,
                        bsupumnc=bsupumnc, bsupvmnc=bsupvmnc, gmnc=gmnc, xm_nyq=xm_nyq,
                        xn_nyq=xn_nyq, Aminor_p=Aminor_p, ntheta=ntheta, nphi=nphi,
                        close=close, range_torus=range_torus, mode_tolerance=mode_tolerance, **partners)
        return self

    def _set_state(self, nfp, ns, bmnc, xm, xn, rmnc, zmns, bsubsmns, bsubumnc, bsubvmnc,
                   bsupumnc, bsupvmnc, gmnc, xm_nyq, xn_nyq, Aminor_p,
                   ntheta, nphi, close, range_torus, mode_tolerance=0.0, **partners):
        if ns < 3 or not 0 <= mode_tolerance < 1:
            raise ValueError("Require ns >= 3 and 0 <= mode_tolerance < 1")
        unknown = partners.keys() - VMEC_WOUT_PARTNERS.values()
        if unknown:
            raise TypeError(f"Unknown Fourier partner tables: {sorted(unknown)}")
        self.nfp = nfp
        self.bmnc = bmnc
        self.xm = xm
        self.xn = xn
        self.rmnc = rmnc
        self.zmns = zmns
        self.bsubsmns = bsubsmns
        self.bsubumnc = bsubumnc
        self.bsubvmnc = bsubvmnc
        self.bsupumnc = bsupumnc
        self.bsupvmnc = bsupvmnc
        self.gmnc = gmnc
        self.xm_nyq = xm_nyq
        self.xn_nyq = xn_nyq
        for name, partner in VMEC_WOUT_PARTNERS.items():
            table = partners.get(partner)
            if table is not None:
                table = jnp.asarray(table)
                if table.shape != getattr(self, name).shape:
                    raise ValueError(f"{partner} must match {name}.shape")
            setattr(self, partner, table)
        if mode_tolerance > 0:
            self._drop_small_modes(mode_tolerance)
        self.len_xm_nyq = len(self.xm_nyq)
        self.ns = ns
        self.s_full_grid = jnp.linspace(0, 1, self.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds
        self.r_axis = self.rmnc[0, 0]
        self.z_axis = self.zmns[0, 0] if self.zmnc is None else self.zmnc[0, 0]
        with jax.ensure_compile_time_eval():
            self.mpol = int(jnp.max(self.xm))
            self.ntor = int(jnp.max(jnp.abs(self.xn)) / self.nfp)
        self.range_torus = range_torus
        self._surface = SurfaceRZFourier.from_vmec(self, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus)
        self.Aminor_p = Aminor_p
        #self._classifier=SurfaceClassifier(self._surface,p=1,h=0.05)

    def _drop_small_modes(self, tolerance):
        for tables, numbers in ((('rmnc', 'zmns'), ('xm', 'xn')),
                                (('bmnc', 'gmnc', 'bsubsmns', 'bsubumnc', 'bsubvmnc', 'bsupumnc', 'bsupvmnc'),
                                 ('xm_nyq', 'xn_nyq'))):
            amplitude = [np.abs(np.asarray(getattr(self, name))) if getattr(self, VMEC_WOUT_PARTNERS[name]) is None
                         else np.hypot(np.asarray(getattr(self, name)), np.asarray(getattr(self, VMEC_WOUT_PARTNERS[name])))
                         for name in tables]
            amplitude = [a.max(axis=0) for a in amplitude]
            tables += tuple(VMEC_WOUT_PARTNERS[name] for name in tables
                            if getattr(self, VMEC_WOUT_PARTNERS[name]) is not None)
            keep = np.any([a > tolerance * a.max() for a in amplitude], axis=0)
            for name in tables + numbers:
                setattr(self, name, getattr(self, name)[..., keep])

    @property
    def surface(self):
        return self._surface

    def _bsubs_axis_m1(self, sine=False):
        """Axis limit of sqrt(s) B_s for the m = 1 modes, from B_theta.

        Near the axis the leading m = 1 parts of B_s and B_theta are the
        gradient of sqrt(s) Psi(theta, phi), so sqrt(s) B_s tends to
        B_theta / (2 sqrt(s)) and their contributions to the toroidal current
        cancel. VMEC's B_s next to the axis misses that limit (by about 10% on
        an HSX wout), and extrapolating it gave curl B a toroidal component
        that grew as 1/sqrt(s) on the axis.
        """
        table = self.bsubumns if sine else self.bsubumnc
        if table is None:
            return None
        b_theta = table[1:3] / jnp.sqrt(self.s_half_grid[:2])[:, None]
        return (1.5 * b_theta[0] - 0.5 * b_theta[1]) * (-0.5 if sine else 0.5)
        
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
        return self._series(points, self._NYQUIST, self.xm_nyq, self.xn_nyq)

    def _series(self, points, tables, xm, xn):
        s, theta, phi = points
        angle = xm * theta - xn * phi
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        series = {}
        for name, (half_grid, options, is_cos) in tables.items():
            grid = self.s_half_grid if half_grid else self.s_full_grid
            total = None
            for partner, cosine in ((name, is_cos), (VMEC_WOUT_PARTNERS[name], not is_cos)):
                table = getattr(self, partner)
                if table is None:
                    continue
                radial_options = options
                if name == 'bsubsmns':
                    radial_options = dict(options, axis_m1=self._bsubs_axis_m1(partner != name))
                f, df = jax.jvp(lambda s: _radial_interp(s, grid, table, xm,
                                                        half_grid=half_grid, **radial_options),
                               (s,), (jnp.ones_like(s),))
                if cosine:
                    value = (f @ cos, jnp.array([df @ cos, -(xm * f) @ sin, (xn * f) @ sin]))
                else:
                    value = (f @ sin, jnp.array([df @ sin, (xm * f) @ cos, -(xn * f) @ cos]))
                total = value if total is None else tuple(a + b for a, b in zip(total, value))
            series[name] = total
        return series

    def _geometry_series(self, points):
        return self._series(points, {'rmnc': (False, {}, True), 'zmns': (False, {}, False)}, self.xm, self.xn)

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
        """Cartesian B = B^theta e_theta + B^phi e_phi, so that B.grad s = 0 exactly."""
        geometry = self._geometry_series(points)
        R, dR = geometry['rmnc']
        _, dZ = geometry['zmns']
        phi = points[2]
        sin, cos = jnp.sin(phi), jnp.cos(phi)
        basis = jnp.stack([cos * dR, sin * dR, dZ])
        basis = basis.at[:, 2].add(jnp.array([-R * sin, R * cos, 0]))
        return basis[:, 1:] @ self.B_contravariant(points)[1:]

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
        geometry = self._geometry_series(points)
        R, Z = geometry['rmnc'][0], geometry['zmns'][0]
        return jnp.array([R * jnp.cos(points[2]), R * jnp.sin(points[2]), Z])

    def _boundary_rz(self, theta, phi):
        """R, Z of the LCFS and their first two theta derivatives, on a theta array."""
        angle = self.xm * theta[..., None] - self.xn * phi
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        m, zero = self.xm, jnp.zeros_like(self.xm, dtype=float)
        rc, zs = self.rmnc[-1], self.zmns[-1]
        rs = zero if self.rmns is None else self.rmns[-1]
        zc = zero if self.zmnc is None else self.zmnc[-1]

        def series(c, s):  # sum c cos + s sin, and its first two theta derivatives
            return cos @ c + sin @ s, cos @ (m * s) - sin @ (m * c), -(cos @ (m * m * c) + sin @ (m * m * s))

        (R, dR, d2R), (Z, dZ, d2Z) = series(rc, rs), series(zc, zs)
        return R, Z, dR, dZ, d2R, d2Z

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
            x = x - dx * jnp.minimum(1.0, 0.1 / (jnp.linalg.norm(dx) + 1e-300))
            # Stay in s <= 1: beyond it the extrapolated map can fold back over the plasma.
            return x / jnp.maximum(1.0, jnp.linalg.norm(x)), None

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


@tree_util.register_static
class ExternalField(MagneticField):
    """A Cartesian field from a batched source, traced one point at a time.

    ``source`` has ``b_cyl(R, phi, Z) -> (B_R, B_phi, B_Z)`` (a VMEX
    ``MgridField``), a batched ``B(points)`` for points of shape ``(n, 3)`` (a
    VMEX ``VmecExtender``), or is a callable ``xyz (n, 3) -> B (n, 3)``, in
    metres and tesla, traceable by JAX. The derivatives come from automatic
    differentiation (:class:`MagneticField`).
    """

    def __init__(self, source):
        self.source = source

    @jit
    def sqrtg(self, points):
        return 1.

    @jit
    def to_xyz(self, points):
        return points

    @jit
    def B(self, points):
        if hasattr(self.source, "b_cyl"):
            R, phi = jnp.hypot(points[0], points[1]), jnp.arctan2(points[1], points[0])
            BR, Bphi, BZ = (jnp.ravel(b)[0] for b in self.source.b_cyl(R[None], phi[None], points[2][None]))
            return jnp.array([BR * jnp.cos(phi) - Bphi * jnp.sin(phi), BR * jnp.sin(phi) + Bphi * jnp.cos(phi), BZ])
        return getattr(self.source, "B", self.source)(points[None])[0]


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


def _bspline_prefilter(n, periodic):
    """Matrix mapping n node values to cubic B-spline coefficients (periodic, or not-a-knot with 2 ghosts)."""
    if periodic:
        A = (4 * np.eye(n) + np.roll(np.eye(n), 1, 1) + np.roll(np.eye(n), -1, 1)) / 6
        return np.linalg.inv(A)
    A = np.zeros((n + 2, n + 2))
    for i in range(n):
        A[i, i:i + 3] = (1 / 6, 4 / 6, 1 / 6)
    A[n, :5] = A[n + 1, -5:] = (-1, 4, -6, 4, -1)  # third derivative continuous at the 2nd and penultimate knots
    return np.linalg.inv(A)[:, :n]


def _bspline_weights(u, n, periodic):
    """Indices, weights and derivative weights (per unit u) of the 4 cubic B-splines at grid coordinate u."""
    i = jnp.floor(u) if periodic else jnp.clip(jnp.floor(u), 0, n - 2)
    t = u - i
    i = i.astype(int) + jnp.arange(4)
    w = jnp.array([(1 - t)**3, 3 * t**3 - 6 * t**2 + 4, -3 * t**3 + 3 * t**2 + 3 * t + 1, t**3]) / 6
    dw = jnp.array([-(1 - t)**2, 3 * t**2 - 4 * t, -3 * t**2 + 2 * t + 1, t**2]) / 2
    return (i - 1) % n if periodic else i, jnp.stack((w, dw))


class InterpolatedField(MagneticField):
    """Any Cartesian field tabulated on a cylindrical grid and evaluated by tricubic B-splines.

    The source ``field`` (BiotSavart, CombinedField, MGrid, ExternalField, a
    dipole field, ...) is sampled once with ``field.B`` on ``nphi`` toroidal
    planes of one field period (``nfp``), ``nr`` radii in ``R=(rmin, rmax)``
    and ``nz`` heights in ``Z=(zmin, zmax)``; with ``stellsym`` only half a
    period is evaluated (Z must then be symmetric), in batches of ``chunk_size``
    points (by default sized to the source). The cylindrical components
    are interpolated by a C2 tricubic spline (periodic in phi, not-a-knot in R
    and Z) that is exact at the nodes, so B converges as h^4 and dB/dX as h^3.
    ``B``, ``dB_by_dX`` and the fused ``gc_quantities`` come from one 4x4x4
    stencil; points outside the R, Z box are extrapolated from the edge cells.
    """

    def __init__(self, field=None, R=(1.0, 2.0), Z=(-0.5, 0.5), nr=32, nz=32, nphi=32, nfp=1,
                 stellsym=False, chunk_size=None, table=None):
        (self.rmin, self.rmax), (self.zmin, self.zmax), self.nfp = map(float, R), map(float, Z), int(nfp)
        if field is not None and not isinstance(field, MagneticField):
            field = ExternalField(field)  # a batched source, e.g. a VMEX VmecExtender or MgridField
        if table is None:
            if stellsym and (nphi % 2 or not np.isclose(self.zmin, -self.zmax)):
                raise ValueError("stellsym needs an even nphi and Z = (-zmax, zmax)")
            nph = nphi // 2 + 1 if stellsym else nphi
            phi, z, r = np.meshgrid(np.arange(nph) * 2 * np.pi / (nfp * nphi), np.linspace(*Z, nz),
                                    np.linspace(*R, nr), indexing="ij")
            xyz = jnp.stack((r * np.cos(phi), r * np.sin(phi), z), -1).reshape(-1, 3)
            if chunk_size is None:  # bound points x source size (e.g. coil segments, dipoles) to ~2^24
                chunk_size = max(1, 2**24 // max(1, sum(np.size(x) for x in tree_util.tree_leaves(field))))
            B = lax.map(field.B, xyz, batch_size=min(chunk_size, len(xyz))).reshape(phi.shape + (3,))
            c, s = jnp.cos(phi), jnp.sin(phi)
            table = jnp.stack((c * B[..., 0] + s * B[..., 1], c * B[..., 1] - s * B[..., 0], B[..., 2]), -1)
            if stellsym:  # B_R(R, -phi, -Z) = -B_R(R, phi, Z); B_phi, B_Z even
                table = jnp.concatenate((table, table[1:nphi // 2, ::-1][::-1] * jnp.array([-1., 1., 1.])))
        self.table = jnp.asarray(table)  # (nphi, nz, nr, 3) node values of (B_R, B_phi, B_Z), mgrid layout
        nphi, nz, nr, _ = self.table.shape
        self.coefficients = jnp.einsum("pi,zj,rk,ijkc->pzrc", _bspline_prefilter(nphi, True),
                                       _bspline_prefilter(nz, False), _bspline_prefilter(nr, False), self.table)

    @classmethod
    def around(cls, field, surface, margin=0.05, n=48, **kwargs):
        """Tabulate ``field`` on ``n`` x ``n`` x ``2n`` nodes in the R, Z box of ``surface`` (e.g. a wall) plus ``margin`` [m]."""
        g = np.asarray(surface.gamma)
        R, z = np.hypot(g[..., 0], g[..., 1]), np.abs(g[..., 2]).max() + margin
        return cls(field, R=(R.min() - margin, R.max() + margin), Z=(-z, z),
                   **{"nr": n, "nz": n, "nphi": 2 * n, "nfp": surface.nfp, **kwargs})

    @classmethod
    def load(cls, filename):
        """Read a table written by :meth:`save` (``.npz``) or a VMEC mgrid file (``.nc``)."""
        if str(filename).endswith(".nc"):
            from essos.mgrid import MGrid
            g = MGrid.from_file(filename)
            return cls(R=(g.rmin, g.rmax), Z=(g.zmin, g.zmax), nfp=g.nfp, table=g.bvec)
        d = np.load(filename)
        return cls(R=d["R"], Z=d["Z"], nfp=int(d["nfp"]), table=d["table"])

    def save(self, filename):
        """Write the node table to ``.npz``, or to a VMEC mgrid ``.nc`` file."""
        if str(filename).endswith(".nc"):
            from essos.mgrid import MGrid
            nphi, nz, nr, _ = self.table.shape
            g = MGrid(nr=nr, nz=nz, nphi=nphi, nfp=self.nfp, rmin=self.rmin, rmax=self.rmax, zmin=self.zmin, zmax=self.zmax)
            g.add_field_cylindrical(*np.moveaxis(np.asarray(self.table), -1, 0))
            return g.write(filename)
        np.savez(filename, table=self.table, R=(self.rmin, self.rmax), Z=(self.zmin, self.zmax), nfp=self.nfp)

    @jit
    def sqrtg(self, points):
        return 1.

    @jit
    def to_xyz(self, points):
        return points

    @jit
    def B_and_dB_by_dX(self, points):
        """Cartesian B and its Jacobian dB_i/dx_j at one point, from one spline stencil."""
        x, y, z = points
        R, phi = jnp.hypot(x, y), jnp.arctan2(y, x)
        nphi, nz2, nr2, _ = self.coefficients.shape
        hp, hz, hr = 2 * jnp.pi / (self.nfp * nphi), (self.zmax - self.zmin) / (nz2 - 3), (self.rmax - self.rmin) / (nr2 - 3)
        ip, wp = _bspline_weights(phi / hp, nphi, True)
        iz, wz = _bspline_weights((z - self.zmin) / hz, nz2 - 2, False)
        ir, wr = _bspline_weights((R - self.rmin) / hr, nr2 - 2, False)
        C = self.coefficients[ip[:, None, None], iz[None, :, None], ir[None, None, :]]  # (4, 4, 4, 3)
        Cr = jnp.einsum("abrc,nr->nabc", C, wr)
        Cz = jnp.einsum("nabc,mb->nmac", Cr, wz)
        V = jnp.einsum("nmac,la->lmnc", Cz, wp)  # V[l, m, n] = d^l/dphi d^m/dZ d^n/dR of (B_R, B_phi, B_Z)
        b, dR, dphi, dZ = V[0, 0, 0], V[0, 0, 1] / hr, V[1, 0, 0] / hp, V[0, 1, 0] / hz
        c, s = jnp.cos(phi), jnp.sin(phi)
        rot = jnp.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        drot = jnp.array([[-s, -c, 0.], [c, -s, 0.], [0., 0., 0.]])
        dB_dcyl = jnp.stack((rot @ dR, rot @ dphi + drot @ b, rot @ dZ), -1)  # d B_xyz / d(R, phi, Z)
        dcyl_dx = jnp.array([[c, s, 0.], [-s / R, c / R, 0.], [0., 0., 1.]])
        return rot @ b, dB_dcyl @ dcyl_dx

    @jit
    def B(self, points):
        return self.B_and_dB_by_dX(points)[0]

    @jit
    def dB_by_dX(self, points):
        return self.B_and_dB_by_dX(points)[1]

    @jit
    def gc_quantities(self, points):
        return _gc_from_jacobian(*self.B_and_dB_by_dX(points))

    def _tree_flatten(self):
        return (self.table, self.coefficients), (self.rmin, self.rmax, self.zmin, self.zmax, self.nfp)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.table, obj.coefficients = children
        obj.rmin, obj.rmax, obj.zmin, obj.zmax, obj.nfp = aux_data
        return obj


tree_util.register_pytree_node(InterpolatedField, InterpolatedField._tree_flatten, InterpolatedField._tree_unflatten)
