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
    """Base class of the magnetic fields.

    A field is evaluated one point at a time in its own coordinates (vmap
    over points). The defaults describe Cartesian coordinates: ``to_xyz`` is
    the identity, ``sqrtg = 1`` and ``B`` is the Cartesian vector, so a
    Cartesian field only defines ``B``. Fields in other coordinates override
    ``to_xyz``, ``sqrtg``, ``B_covariant`` and ``B_contravariant`` and keep
    ``B`` Cartesian; derivatives and guiding-center quantities follow here.

    Fields combine as vectors: ``f + g``, ``f - g`` and ``2.0 * f`` are
    :class:`CombinedField`. :meth:`B_xyz` evaluates any field at a Cartesian
    point and :meth:`compare` measures two fields at the same physical points.
    """

    @jit
    def sqrtg(self, points):
        return 1.

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
        return points

    @jit
    def B_xyz(self, xyz):
        """Cartesian B at the Cartesian point ``xyz``."""
        return self.B(xyz)

    @jit
    def compare(self, other, points):
        """Relative difference ``|B - B_other| / |B|`` at ``points`` (n, 3) in this field's coordinates.

        ``other`` is evaluated with :meth:`B_xyz` at the same physical points,
        so the two fields may use different coordinates.
        """
        def difference(point):
            B = self.B(point)
            return jnp.linalg.norm(B - other.B_xyz(self.to_xyz(point))) / jnp.linalg.norm(B)
        return vmap(difference)(points)

    def __add__(self, other):
        return CombinedField(self, other)

    def __radd__(self, other):  # sum() starts from 0
        return self if isinstance(other, (int, float)) and other == 0 else CombinedField(other, self)

    def __mul__(self, scale):
        return CombinedField(self, weights=(scale,))

    __rmul__ = __mul__

    def __neg__(self):
        return -1. * self

    def __sub__(self, other):
        return self + (-other)

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
    def gc_quantities(self, points):
        """Guiding-center field quantities from one pass over the coils.

        The separate methods each rebuild B and its Jacobian (``kappa`` even
        recomputes ``curl_b``). Here a single forward-mode Jacobian yields B
        and dB/dX together, and in Cartesian coordinates (sqrtg = 1)
        grad|B| = (dB/dX)^T b, curl b = curl B/|B| + B x grad|B|/|B|^2 and
        kappa = -B x curl b / |B| follow algebraically.
        """
        points = jnp.asarray(points)
        jacobian, field = jacfwd(lambda x: (self.B(x), self.B(x)), has_aux=True)(points)
        magnitude = jnp.linalg.norm(field)
        grad_magnitude = jacobian.T @ (field / magnitude)
        curl_field = jnp.array([
            jacobian[2, 1] - jacobian[1, 2],
            jacobian[0, 2] - jacobian[2, 0],
            jacobian[1, 0] - jacobian[0, 1],
        ])
        curl_unit = curl_field / magnitude + jnp.cross(field, grad_magnitude) / magnitude**2
        curvature = -jnp.cross(field, curl_unit) / magnitude
        return field, field, magnitude, grad_magnitude, curl_unit, curvature, 1.0

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

VMEC_WOUT_ARRAYS = ('bmnc', 'xm', 'xn', 'rmnc', 'zmns', 'bsubsmns', 'bsubumnc', 'bsubvmnc',
                    'bsupumnc', 'bsupvmnc', 'gmnc', 'xm_nyq', 'xn_nyq', 'Aminor_p')
VMEC_WOUT_PARTNERS = {'rmnc': 'rmns', 'zmns': 'zmnc', 'bmnc': 'bmns', 'gmnc': 'gmns',
                      'bsubsmns': 'bsubsmnc', 'bsubumnc': 'bsubumns', 'bsubvmnc': 'bsubvmns',
                      'bsupumnc': 'bsupumns', 'bsupvmnc': 'bsupvmns'}

class ToroidalField(MagneticField):
    """A field in the toroidal coordinates (s, theta, phi) of a bounded plasma.

    s in [0, 1] labels the surfaces, zero on the magnetic axis and one on the
    boundary, theta is a poloidal angle and phi the cylindrical toroidal
    angle, both in radians. Subclasses give ``to_xyz``, ``sqrtg``, ``B``,
    ``B_covariant``, ``B_contravariant`` and the minor radius ``Aminor_p``.
    This class inverts ``to_xyz`` and measures the distance to the boundary,
    which the tracer uses to stop or continue orbits there.
    """

    @jit
    def B_xyz(self, xyz):
        return self.B(self.flux_coordinates(xyz)[0])

    def _boundary_rz(self, theta, phi):
        """R, Z of the boundary and their first two theta derivatives, on a theta array."""
        def rz(t):
            p = self.to_xyz(jnp.array([1., t, phi]))
            return jnp.array([jnp.hypot(p[0], p[1]), p[2]])
        def d(f):
            return lambda t: jax.jvp(f, (t,), (jnp.ones_like(t),))[1]
        values = vmap(lambda t: jnp.stack([rz(t), d(rz)(t), d(d(rz))(t)]))(jnp.ravel(theta))
        return tuple(values[:, i, j].reshape(jnp.shape(theta)) for i in range(3) for j in range(2))

    def _minor_radius(self):
        """Radius of the circle with the mean area of the boundary cross-sections of one field period."""
        theta = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        R, Z, dR, dZ = vmap(lambda phi: jnp.stack(self._boundary_rz(theta, phi)[:4]), out_axes=1)(
            jnp.linspace(0, 2 * jnp.pi / self.nfp, 8, endpoint=False))
        return jnp.sqrt(jnp.abs(jnp.mean(R * dZ - Z * dR)))

    @jit
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

    @jit
    def flux_coordinates(self, xyz):
        """Invert :meth:`to_xyz`: a Cartesian point to (s, theta, phi), and the residual [m].

        Newton iterations in (sqrt(s) cos theta, sqrt(s) sin theta), which is
        regular on the axis. They start from the point's angle about the axis
        (measured from theta = 0, in the sense theta turns) and its distance
        from the axis relative to the boundary's in that direction. Points
        outside the LCFS return s > 1 only as far as the extrapolated geometry
        allows; check the residual.
        """
        R, Z = jnp.hypot(xyz[0], xyz[1]), xyz[2]
        phi = jnp.mod(jnp.arctan2(xyz[1], xyz[0]), 2 * jnp.pi)
        target = jnp.array([R, Z])

        def rz(x):
            p = self.to_xyz(jnp.array([x[0]**2 + x[1]**2, jnp.arctan2(x[1], x[0]), phi]))
            return jnp.array([jnp.hypot(p[0], p[1]), p[2]])

        def angle(v):
            return jnp.arctan2(v[1], v[0])
        axis, start, quarter = rz(jnp.zeros(2)), rz(jnp.array([0.5, 0.])), rz(jnp.array([0., 0.5]))
        sense = jnp.sign(jnp.sin(angle(quarter - axis) - angle(start - axis)))
        theta = sense * (angle(target - axis) - angle(start - axis))
        direction = jnp.array([jnp.cos(theta), jnp.sin(theta)])
        x = direction * jnp.clip(jnp.linalg.norm(target - axis) / jnp.linalg.norm(rz(direction) - axis), 0.0, 1.0)

        def newton(x, _):
            dx = jnp.linalg.solve(jacfwd(rz)(x), rz(x) - target)
            x = x - dx * jnp.minimum(1.0, 0.3 / (jnp.linalg.norm(dx) + 1e-300))
            # Stay in s <= 1: beyond it the extrapolated map can fold back over the plasma.
            return x / jnp.maximum(1.0, jnp.linalg.norm(x)), None

        x, _ = lax.scan(newton, x, None, length=12)
        s = x[0]**2 + x[1]**2
        return jnp.array([s, jnp.mod(jnp.arctan2(x[1], x[0]), 2 * jnp.pi), phi]), jnp.linalg.norm(rz(x) - target)

@tree_util.register_static
class Vmec(ToroidalField):
    """VMEC equilibrium, including asymmetric Fourier partners, from wout or live arrays.

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
        geometry = self._geometry_series(points)
        R, dR = geometry['rmnc']
        _, dZ = geometry['zmns']
        phi = points[2]
        sin, cos = jnp.sin(phi), jnp.cos(phi)
        basis = jnp.stack([cos * dR, sin * dR, dZ])
        basis = basis.at[:, 2].add(jnp.array([-R * sin, R * cos, 0]))
        reciprocal = jnp.stack([jnp.cross(basis[:, 1], basis[:, 2]),
                                jnp.cross(basis[:, 2], basis[:, 0]),
                                jnp.cross(basis[:, 0], basis[:, 1])]) / self.sqrtg(points)
        return self.B_covariant(points) @ reciprocal

    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        return self._nyquist_series(points)['bmnc'][0]
    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return self._nyquist_series(points)['bmnc'][1]
    
    @partial(jit, static_argnames=['self'])
    def grad_B_covariant(self, points):
        series = self._nyquist_series(points)
        return jnp.stack([series[name][1] for name in ('bsubsmns', 'bsubumnc', 'bsubvmnc')])
 
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

def _hermite(x, values):
    """Cubic Hermite interpolation of ``values`` given on a uniform grid of [0, 1], with centred-difference slopes."""
    n = len(values) - 1
    slope = jnp.gradient(values)
    i = jnp.clip(jnp.floor(x * n).astype(int), 0, n - 1)
    t = x * n - i
    return ((1 + 2 * t) * (1 - t)**2 * values[i] + t * (1 - t)**2 * slope[i]
            + t * t * (3 - 2 * t) * values[i + 1] + t * t * (t - 1) * slope[i + 1])


def _fourier_zernike(c, modes, rho, theta, zeta, nfp):
    """sum_k c_k R_k(rho) F(m_k theta) F(n_k nfp zeta) of a DESC Fourier-Zernike series with modes (l, m, n).

    R_k = (-1)^j rho^|m| P_j^(|m|, 0)(1 - 2 rho^2), j = (l - |m|) / 2, with the
    Jacobi polynomial from its three-term recurrence (stable at high l), and
    F(k x) = cos(k x) for k >= 0, sin(|k| x) for k < 0, as in DESC.
    """
    l, m, n = (np.asarray(modes)[:, i] for i in range(3))
    a, j = np.abs(m), (l - np.abs(m)) // 2
    x = 1 - 2 * rho**2
    P_prev, P = 0. * a, 1. + 0. * a
    radial = jnp.where(j == 0, P, 0.)
    for k in range(int(j.max(initial=0))):
        if k == 0:
            P_prev, P = P, (a + 1) + (a + 2) * (x - 1) / 2
        else:
            b = 2 * k + a
            P_prev, P = P, ((b + 1) * (b * (b + 2) * x + a**2) * P - 2 * k * (k + a) * (b + 2) * P_prev) / (
                2 * (k + 1) * (k + a + 1) * b)
        radial = jnp.where(j == k + 1, P, radial)
    angle = lambda k, x: jnp.where(k >= 0, jnp.cos(np.abs(k) * x), jnp.sin(np.abs(k) * x))
    return jnp.sum(c * (-1.)**j * rho**a * radial * angle(m, theta) * angle(n, nfp * zeta))


@tree_util.register_static
class DescField(ToroidalField):
    """A DESC equilibrium (https://github.com/PlasmaControl/DESC) in the coordinates of :class:`ToroidalField`.

    DESC writes R, Z and the stream function lambda as Fourier-Zernike
    series in (rho, theta, zeta), with zeta the cylindrical angle. Here
    (s, theta, phi) = (rho^2, theta, zeta). With the toroidal flux ``Psi`` and
    sqrt(g) the Jacobian of ``to_xyz`` in (s, theta, phi),

        B^s = 0,  B^theta = Psi (iota - d_phi lambda) / (2 pi sqrt(g)),
        B^phi = Psi (1 + d_theta lambda) / (2 pi sqrt(g)).

    The series are evaluated as DESC writes them, so the field is exact on
    the axis. ``iota`` holds the rotational transform on a uniform grid in rho
    from 0 to 1, interpolated by cubic Hermite polynomials (continuous
    derivative, as guiding centers need). :meth:`from_desc` builds the field from a DESC
    equilibrium or file, or from the ``.npz`` that :meth:`save` writes, which
    needs no DESC install.
    """

    def __init__(self, R_lmn, Z_lmn, L_lmn, R_modes, Z_modes, L_modes, Psi, nfp, iota):
        self.coefficients = tuple(jnp.asarray(c) for c in (R_lmn, Z_lmn, L_lmn))
        self.modes = tuple(np.asarray(m, dtype=int) for m in (R_modes, Z_modes, L_modes))
        self.Psi, self.nfp, self.iota = float(Psi), int(nfp), jnp.asarray(iota)
        self.Aminor_p = self._minor_radius()

    @classmethod
    def from_desc(cls, source):
        """The field of a DESC ``Equilibrium``, of a DESC output file (both need DESC), or of an ``.npz`` from :meth:`save`."""
        if isinstance(source, str) and source.endswith(".npz"):
            data = np.load(source)
            return cls(*(data[k] for k in ("R_lmn", "Z_lmn", "L_lmn", "R_modes", "Z_modes", "L_modes", "Psi", "nfp", "iota")))
        if isinstance(source, str):
            import desc.io
            source = desc.io.load(source)
            source = source[-1] if hasattr(source, "__len__") else source  # the last of a solve sequence
        from desc.grid import LinearGrid
        grid = LinearGrid(rho=np.linspace(0, 1, 1025), M=source.M_grid, N=source.N_grid, NFP=source.NFP)
        iota = grid.compress(source.compute("iota", grid=grid)["iota"])
        return cls(source.R_lmn, source.Z_lmn, source.L_lmn, source.R_basis.modes, source.Z_basis.modes,
                   source.L_basis.modes, source.Psi, source.NFP, iota)

    def save(self, path):
        """Write the field to an ``.npz`` that :meth:`from_desc` reads without DESC."""
        np.savez(path, **dict(zip(("R_lmn", "Z_lmn", "L_lmn"), self.coefficients)),
                 **dict(zip(("R_modes", "Z_modes", "L_modes"), self.modes)), Psi=self.Psi, nfp=self.nfp, iota=self.iota)

    def _series(self, i, points):
        s, theta, phi = points
        return _fourier_zernike(self.coefficients[i], self.modes[i], jnp.sqrt(s), theta, phi, self.nfp)

    @jit
    def to_xyz(self, points):
        R, Z = self._series(0, points), self._series(1, points)
        return jnp.array([R * jnp.cos(points[2]), R * jnp.sin(points[2]), Z])

    def _frame(self, points):
        e = jacfwd(self.to_xyz)(points)
        sqrtg = jnp.linalg.det(e)
        dlambda = jax.grad(lambda p: self._series(2, p))(points)
        iota = _hermite(jnp.sqrt(points[0]), self.iota)
        B_con = self.Psi / (2 * jnp.pi * sqrtg) * jnp.array([0., iota - dlambda[2], 1 + dlambda[1]])
        return e, sqrtg, B_con

    @jit
    def sqrtg(self, points):
        return self._frame(points)[1]

    @jit
    def B_contravariant(self, points):
        return self._frame(points)[2]

    @jit
    def B(self, points):
        e, _, B_con = self._frame(points)
        return e @ B_con

    @jit
    def B_covariant(self, points):
        e, _, B_con = self._frame(points)
        return e.T @ (e @ B_con)


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
    def B(self, points):
        if hasattr(self.source, "b_cyl"):
            R, phi = jnp.hypot(points[0], points[1]), jnp.arctan2(points[1], points[0])
            BR, Bphi, BZ = (jnp.ravel(b)[0] for b in self.source.b_cyl(R[None], phi[None], points[2][None]))
            return jnp.array([BR * jnp.cos(phi) - Bphi * jnp.sin(phi), BR * jnp.sin(phi) + Bphi * jnp.cos(phi), BZ])
        return getattr(self.source, "B", self.source)(points[None])[0]


class CombinedField(MagneticField):
    """Sum of several magnetic fields, traced as one.

    ``sum_i weights[i] * fields[i]``, usually built as ``f + g``, ``f - g`` or
    ``a * f``. The components add over the fields in the coordinates of the
    first, which supplies ``sqrtg`` and ``to_xyz``, so the fields must share
    coordinates (a coil field plus a Cartesian plasma contribution, say).
    :meth:`B_xyz` adds Cartesian vectors and is valid for any fields.
    """

    def __init__(self, *fields, weights=None):
        if len(fields) < 1:
            raise ValueError("CombinedField needs at least one field")
        # Nested sums are flattened, so f + g + h is one sum of three fields.
        weights = (1.,) * len(fields) if weights is None else tuple(weights)
        self.fields, self.weights = (), ()
        for field, weight in zip(fields, weights, strict=True):
            parts = (field.fields, field.weights) if isinstance(field, CombinedField) else ((field,), (1.,))
            self.fields += parts[0]
            self.weights += tuple(float(weight * w) for w in parts[1])  # static: one compile per weight set

    def __getattr__(self, name):
        # Coordinate helpers (flux_coordinates, Aminor_p, nfp, ...) come from the first field.
        if name == "fields":
            raise AttributeError(name)
        return getattr(self.fields[0], name)

    def _sum(self, method, *args):
        return sum(w * getattr(field, method)(*args) for field, w in zip(self.fields, self.weights))

    @jit
    def B(self, points):
        return self._sum("B", points)

    @jit
    def B_covariant(self, points):
        return self._sum("B_covariant", points)

    @jit
    def B_contravariant(self, points):
        return self._sum("B_contravariant", points)

    @jit
    def B_xyz(self, xyz):
        return self._sum("B_xyz", xyz)

    @jit
    def sqrtg(self, points):
        return self.fields[0].sqrtg(points)

    @jit
    def to_xyz(self, points):
        return self.fields[0].to_xyz(points)

    def _tree_flatten(self):
        return (self.fields,), self.weights

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        self = cls.__new__(cls)
        self.fields, self.weights = children[0], aux_data
        return self


tree_util.register_pytree_node(CombinedField,
                               CombinedField._tree_flatten,
                               CombinedField._tree_unflatten)


def is_toroidal(field):
    """Whether ``field`` works in the (s, theta, phi) of :class:`ToroidalField`, alone or as the first of a sum."""
    return isinstance(getattr(field, "fields", (field,))[0], ToroidalField)
