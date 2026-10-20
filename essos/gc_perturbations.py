"""Prescribed alpha/Phi potentials for the full-geometry GC equations.

Positions and derivatives use the background field's native coordinates:
(s, theta_VMEC, phi) for Vmec, or Cartesian coordinates for Cartesian fields.
The reduced MHD action adds q*alpha*B0 to the GC one-form and q*Phi to H.
The static background magnitude, direction and full covariant components
are retained, including VMEC's radial covariant magnetic field.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from essos.boozer import _evaluate, _spline
from essos.perturbations import AnalyticalAlfvenMode, BoozerPerturbation


class GuidingCenterPerturbation(eqx.Module):
    """Return (alpha [m], Phi [V]) in the background's native coordinates."""

    def potentials(self, field, point, t):
        raise NotImplementedError

    def axis_stop_threshold(self):
        return None

    def derivatives(self, field, point, t):
        """Potentials and their Jacobian in (x1,x2,x3,t), shape (2,4)."""
        def evaluate(z):
            value = self.potentials(field, z[:3], z[3])
            return value, value

        derivative, value = jax.jacfwd(evaluate, has_aux=True)(jnp.append(point, t))
        return value, derivative


class AnalyticalVmecAlfvenMode(GuidingCenterPerturbation):
    """Ideal analytical wave in VMEC angles, not a Boozer-angle eigenmode.

    Phi = amplitude*s**(m/2)*envelope(s)*sin(m*theta_VMEC-n*phi+omega*t+phase).
    The coefficient alpha=(n*B^phi-m*B^theta)*Phi/(omega*B0**2) enforces
    B0 . delta E=0 for a static VMEC field with B^s=0. Unlike its Boozer
    counterpart, this coefficient can depend on the angles. A mapped Boozer
    potential is required to prescribe the identical Boozer-coordinate wave.
    Parameters and units follow AnalyticalAlfvenMode; only smooth envelopes
    are supported by this convenience class.
    """

    mode: AnalyticalAlfvenMode

    def __init__(self, m, n, omega, amplitude, *, phase=0., radial_center=.5, radial_width=None):
        self.mode = AnalyticalAlfvenMode(m, n, omega, amplitude, phase=phase,
                                         radial_center=radial_center, radial_width=radial_width)

    def potentials(self, field, point, t):
        s, theta, phi = point
        mode = self.mode
        harmonic = s**(mode.m/2)*jnp.sin(mode.m*theta-mode.n*phi+mode.omega*t+mode.phase)
        envelope = (1. if mode.radial_width is None else
                    jnp.exp(-((s-mode.radial_center)/mode.radial_width)**2))
        potential = mode.amplitude*envelope*harmonic
        B = field.AbsB(point)
        Bcontra = field.B_contravariant(point)
        alpha = (mode.n*Bcontra[2]-mode.m*Bcontra[1])*potential/(mode.omega*B**2)
        return jnp.stack([alpha, potential])


class VmecBoozerMap(eqx.Module):
    """VMEC -> Boozer angles using the same w and lambda as booz_xform.

    nu=(w-I*lambda)/(G+iota*I), theta_B=theta+lambda+iota*nu,
    zeta_B=phi+nu. The normalized toroidal flux s is unchanged. Fourier
    coefficients are interpolated in sqrt(s), with regular axis powers
    removed before interpolation. No angle inversion is needed during GC steps.
    """

    r_knots: jax.Array
    lambda_coef: jax.Array
    w_coef: jax.Array
    xm: jax.Array
    xn: jax.Array
    xm_nyq: jax.Array
    xn_nyq: jax.Array
    s_knots: jax.Array
    profile_coef: jax.Array

    @classmethod
    def from_booz_xform(cls, booz):
        """Use VMEC input tables of a stellarator-symmetric Booz_xform."""
        if booz.asym:
            raise ValueError("VmecBoozerMap currently requires stellarator symmetry")
        s, m, n = map(np.asarray, (booz.s_in, booz.xm_nyq, booz.xn_nyq))
        u, v = map(np.asarray, (booz.bsubumnc, booz.bsubvmnc))
        w = np.where(m[:, None] != 0, u/np.where(m[:, None] != 0, m[:, None], 1),
                     np.where(n[:, None] != 0, -v/np.where(n[:, None] != 0, n[:, None], 1), 0))
        xm, xn = map(np.asarray, (booz.xm, booz.xn))
        r = np.sqrt(s)

        def spline(table, modes):
            powers = np.minimum(modes, 2+modes % 2)
            return _spline(r, (np.asarray(table)/r[None, :]**powers[:, None]).T)

        knots, lam = spline(booz.lmns, xm)
        _, wc = spline(w, m)
        zero = (m == 0) & (n == 0)
        profiles = np.stack([np.asarray(booz.iota), v[zero][0], u[zero][0]], axis=1)
        sk, pc = _spline(s, profiles)
        return cls(knots, lam, wc, jnp.asarray(xm), jnp.asarray(xn),
                   jnp.asarray(m), jnp.asarray(n), sk, pc)

    def __call__(self, point):
        s, theta, phi = point
        r = jnp.sqrt(s)

        def series(coef, m, n):
            values, _ = _evaluate(self.r_knots, coef, r)
            return jnp.sum(values*r**jnp.minimum(m, 2+m % 2)*jnp.sin(m*theta-n*phi))

        lam = series(self.lambda_coef, self.xm, self.xn)
        w = series(self.w_coef, self.xm_nyq, self.xn_nyq)
        (iota, G, I), _ = _evaluate(self.s_knots, self.profile_coef, s)
        nu = (w-I*lam)/(G+iota*I)
        return jnp.array([s, theta+lam+iota*nu, phi+nu])

    def inverse(self, point):
        """Boozer -> VMEC birth coordinates; verify the residual before tracing."""
        def step(_, x):
            residual = self(x)[1:]-point[1:]
            jac = jax.jacfwd(self)(x)[1:, 1:]
            a, b, c, d = jac.ravel()
            correction = jnp.array([d*residual[0]-b*residual[1],
                                    a*residual[1]-c*residual[0]])/(a*d-b*c)
            return x.at[1:].add(-correction)

        return jax.lax.fori_loop(0, 12, step, jnp.asarray(point))


class MappedBoozerPerturbation(GuidingCenterPerturbation):
    """Evaluate an existing Boozer wave through an explicit coordinate map.

    coordinate_map(point) must return (s,theta_B,zeta_B) and be JAX
    differentiable with respect to point. Spatial derivatives include its
    full chain rule. VMEC angles must not be silently identified with Boozer
    angles. Computing a VMEC-to-Boozer map is separate from this adapter.
    Singular potentials transfer their inner cutoff only through VmecBoozerMap,
    which preserves s. Arbitrary maps must use smooth potentials.
    """

    background: eqx.Module
    perturbation: BoozerPerturbation
    coordinate_map: Callable

    def __init__(self, background, perturbation, coordinate_map):
        if not isinstance(perturbation, BoozerPerturbation):
            raise TypeError("perturbation must be a BoozerPerturbation")
        if perturbation.axis_stop_threshold() is not None and not isinstance(coordinate_map, VmecBoozerMap):
            raise ValueError("Mapped potentials need a smooth mode or an explicitly handled inner boundary")
        if not callable(coordinate_map):
            raise TypeError("coordinate_map must be callable")
        self.background, self.perturbation = background, perturbation
        self.coordinate_map = coordinate_map if isinstance(coordinate_map, eqx.Module) else eqx.Partial(coordinate_map)

    def axis_stop_threshold(self):
        return self.perturbation.axis_stop_threshold()

    def potentials(self, field, point, t):
        s, theta, zeta = self.coordinate_map(point)
        xyz = jnp.array([jnp.sqrt(s)*jnp.cos(theta), jnp.sqrt(s)*jnp.sin(theta), zeta])
        return self.perturbation.potentials(self.background, xyz, t)
