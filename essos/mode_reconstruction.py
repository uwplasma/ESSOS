"""Differentiable Fourier potentials and field diagnostics for computed modes.

This implements the existing reduced action delta A=alpha*B0. The equilibrium
magnitude remains in the Hamiltonian. It does not infer an MHD polarization
from an arbitrary scalar eigenvector.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from essos.boozer import _evaluate
from essos.perturbations import BoozerPerturbation


class SpectralAlfvenMode(BoozerPerturbation):
    """Ideal wave with complex radial coefficients in the regular Boozer chart.

    Phi=Re sum[g_mn(s) (u+sign(m)*i*w)**abs(m)
               exp(i*(omega*t-n*zeta+phase))] * amplitude.
    g is in volts; it is the coefficient AFTER removing s**(abs(m)/2).
    Thus complex phase and negative m are retained without atan2 on the axis.
    alpha is derived harmonic by harmonic from the ideal parallel-E relation.

    Annular data require an explicit positive taper_width in s. A quintic
    window on each end makes the potentials and their first two derivatives
    zero outside the tabulated interval. This is a declared model extension,
    not additional eigenmode information from the upstream code.
    """

    knots: jax.Array
    coefficients: jax.Array
    harmonics: tuple = eqx.field(static=True)
    taper_width: float | None = eqx.field(static=True)
    omega: jax.Array
    amplitude: jax.Array
    phase: jax.Array

    def __init__(self, s, harmonics, coefficients, omega, *, amplitude=1., phase=0., taper_width=None):
        from scipy.interpolate import CubicSpline

        s, modes = np.asarray(s, float), np.asarray(harmonics)
        values = np.asarray(coefficients, complex)
        if (s.ndim != 1 or s.size < 4 or not np.isfinite(s).all()
                or np.any(np.diff(s) <= 0) or s[0] < 0 or s[-1] > 1):
            raise ValueError("s needs at least four finite increasing samples in [0,1]")
        if (modes.ndim != 2 or modes.shape[1] != 2 or not len(modes)
                or not np.isfinite(modes).all() or np.any(modes != np.round(modes))):
            raise ValueError("harmonics must contain integer (m,n) pairs, with full-torus n")
        labels = tuple((int(m), int(n)) for m, n in modes)
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate harmonic")
        if values.shape != (len(s), len(modes)) or not np.isfinite(values).all():
            raise ValueError("coefficients must be finite with shape (surface,harmonic)")
        for name, value in (("omega", omega), ("amplitude", amplitude), ("phase", phase)):
            if not isinstance(value, jax.core.Tracer) and (np.ndim(value) or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite scalar")
        if not isinstance(omega, jax.core.Tracer) and omega == 0:
            raise ValueError("omega must be nonzero")
        if taper_width is not None and (not np.isfinite(taper_width) or not 0 < taper_width <= (s[-1]-s[0])/2):
            raise ValueError("taper_width must lie in (0, half the radial span]")
        if taper_width is None and (s[0] != 0 or s[-1] != 1):
            raise ValueError("Annular data require an explicit taper_width for zero extension")
        spline = CubicSpline(s, values, axis=0)
        self.knots = jnp.asarray(s)
        self.coefficients = jnp.asarray(np.moveaxis(spline.c, 0, -1))
        self.harmonics = labels
        self.taper_width = None if taper_width is None else float(taper_width)
        self.omega, self.amplitude, self.phase = map(jnp.asarray, (omega, amplitude, phase))

    def potentials(self, field, xyz, t):
        u, w, zeta = xyz
        s = u*u+w*w
        position = s if self.taper_width is None else jnp.clip(s, self.knots[0], self.knots[-1])
        coefficients = _evaluate(self.knots, self.coefficients, position)[0]
        window = 1.
        if self.taper_width is not None:
            def smooth(x):
                x = jnp.clip(x, 0., 1.)
                return x*x*x*(10.+x*(-15.+6.*x))
            window = smooth((s-self.knots[0])/self.taper_width)*smooth((self.knots[-1]-s)/self.taper_width)
        (iota, G, I), _ = field.profiles(s)
        alpha, phi = jnp.zeros(()), jnp.zeros(())
        for index, (m, n) in enumerate(self.harmonics):
            harmonic = (u+(1j if m >= 0 else -1j)*w)**abs(m)
            value = jnp.real(coefficients[index]*harmonic*jnp.exp(1j*(self.omega*t-n*zeta+self.phase)))
            phi += value
            alpha += (n-m*iota)*value/(self.omega*(G+iota*I))
        return self.amplitude*window*jnp.stack([alpha, phi])


def perturbed_fields(field, perturbation, point, t):
    """Return (delta B contravariant [T/coordinate], delta E covariant).

    Uses the native geometry, including its radial covariant B component.
    This diagnostic does not change the GC RHS or add delta B twice.
    """
    (alpha, _), d = perturbation.derivatives(field, point, t)
    Bcov, jacobian = field.B_covariant(point), field.sqrtg(point)
    delta_B = jnp.cross(d[0, :3], Bcov)/jacobian+alpha*field.curl_B(point)
    delta_E = -d[1, :3]-d[0, 3]*Bcov
    return delta_B, delta_E


def physical_components(field, point, contravariant_B, covariant_E):
    """Convert native field components to Cartesian physical vectors."""
    basis = jax.jacfwd(field.to_xyz)(point)
    e1, e2, e3 = basis.T
    determinant = jnp.dot(e1, jnp.cross(e2, e3))
    gradients = jnp.stack([jnp.cross(e2, e3), jnp.cross(e3, e1), jnp.cross(e1, e2)])/determinant
    return basis@contravariant_B, covariant_E@gradients


def radial_field_rms(field, perturbation, s, omega, *, angular_points=16):
    """Physical-area and wave-phase RMS of delta B.normal/|B0| at fixed s.

    Two quadrature phases are exact in time for ONE fixed-frequency mode.
    Angular quadrature resolution must be checked for production use.
    """
    if not 0 < s < 1 or angular_points < 4 or not np.isfinite(omega) or omega == 0:
        raise ValueError("Need 0<s<1, angular_points>=4 and a finite nonzero omega")
    angles = jnp.linspace(0, 2*jnp.pi, angular_points, endpoint=False)
    theta, phi = jnp.meshgrid(angles, angles, indexing="ij")
    points = jnp.stack([jnp.full(theta.size, s), theta.ravel(), phi.ravel()], axis=1)

    def evaluate(point):
        basis = jax.jacfwd(field.to_xyz)(point)
        normal = jnp.cross(basis[:, 1], basis[:, 2])
        area = jnp.linalg.norm(normal)
        grad_s_norm = area/jnp.abs(jnp.linalg.det(basis))
        b0 = perturbed_fields(field, perturbation, point, 0.)[0][0]
        b1 = perturbed_fields(field, perturbation, point, jnp.pi/(2*omega))[0][0]
        return .5*(b0*b0+b1*b1)/(grad_s_norm*field.AbsB(point))**2, area

    values, areas = jax.jit(jax.vmap(evaluate))(points)
    return float(jnp.sqrt(jnp.sum(values*areas)/jnp.sum(areas)))
