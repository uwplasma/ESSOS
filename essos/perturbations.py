"""Prescribed shear-Alfven potentials in Boozer coordinates.

The reduced model of Paul, Mynick & Bhattacharjee, JPP 89, 905890515
(2023), uses delta A = alpha B0 and leaves |B0| in the Hamiltonian.
These are imposed waves, without growth, saturation or particle feedback.
Frequencies are angular frequencies [rad/s]; n is a full-torus integer,
including any field-period factor. Boozer flux signs follow essos.boozer.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class BoozerPerturbation(eqx.Module):
    """Interface for a future computed-mode adapter, including FAX.

    ``potentials(field, xyz, t)`` returns ``(alpha [m], Phi [V])``;
    xyz is the regular chart ``(sqrt(s) cos(theta), sqrt(s) sin(theta), zeta)``.
    The equilibrium covariant field is ``I dtheta + G dzeta`` (K=0).
    """

    def potentials(self, field, xyz, t):
        raise NotImplementedError

    def axis_stop_threshold(self):
        """Optional inner boundary in normalized toroidal flux; None disables it."""
        return None

    def derivatives(self, field, xyz, t):
        """Potentials and their Jacobian in (u, w, zeta, time), shape (2, 4)."""
        def evaluate(z):
            value = self.potentials(field, z[:3], z[3])
            return value, value

        derivative, value = jax.jacfwd(evaluate, has_aux=True)(jnp.append(xyz, t))
        return value, derivative


class AnalyticalAlfvenMode(BoozerPerturbation):
    """Phi = amplitude * envelope(s) * sin(m theta - n zeta + omega t + phase).

    ``amplitude`` is in volts. By default multiply the harmonic by s**(m/2)
    for a smooth axis; amplitude is then its coefficient, not its peak value.
    Optional ``radial_center``/``radial_width`` specify a Gaussian in s.
    ``regularize_axis=False`` gives the paper's radially uniform harmonic
    (without a Gaussian); it is singular on axis for m>0 and must only be
    used away from it. The tracer stops at ``axis_stop_s`` (default 1e-8)
    for this singular option with nonzero amplitude. Stops have their own
    times, separate from LCFS exits and failures. None disables the stop.

    alpha = (n - m iota) Phi / [omega (G + iota I)] enforces
    B0 . (-dt(alpha B0) - grad(Phi)) = 0 in the linear ideal model.
    """

    m: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    omega: jax.Array
    amplitude: jax.Array
    phase: jax.Array
    regularize_axis: bool = eqx.field(static=True)
    radial_center: jax.Array
    radial_width: float | None
    axis_stop_s: float | None = eqx.field(static=True)

    def __init__(self, m, n, omega, amplitude, *, phase=0., regularize_axis=True,
                 radial_center=0.5, radial_width=None, axis_stop_s=1e-8):
        for name, value in (("m", m), ("n", n)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if m < 0:
            raise ValueError("m must be nonnegative")
        for name, value in (("omega", omega), ("amplitude", amplitude),
                            ("phase", phase), ("radial_center", radial_center)):
            if isinstance(value, jax.core.Tracer):
                if value.ndim:
                    raise ValueError(f"{name} must be a scalar")
                continue
            if np.ndim(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
        if not isinstance(omega, jax.core.Tracer) and omega == 0:
            raise ValueError("omega must be nonzero for the ideal Alfven relation")
        if radial_width is not None and (np.ndim(radial_width) or not np.isfinite(radial_width)
                                         or radial_width <= 0):
            raise ValueError("radial_width must be positive and finite")
        if not isinstance(regularize_axis, (bool, np.bool_)):
            raise ValueError("regularize_axis must be boolean")
        if axis_stop_s is not None and (np.ndim(axis_stop_s) or not np.isfinite(axis_stop_s)
                                       or not 0 < axis_stop_s < 1):
            raise ValueError("axis_stop_s must be None or a finite scalar strictly between 0 and 1")
        self.m, self.n = int(m), int(n)
        self.omega, self.amplitude, self.phase = map(jnp.asarray, (omega, amplitude, phase))
        self.regularize_axis = bool(regularize_axis)
        self.radial_center = jnp.asarray(radial_center)
        self.radial_width = None if radial_width is None else float(radial_width)
        self.axis_stop_s = None if axis_stop_s is None else float(axis_stop_s)

    def axis_stop_threshold(self):
        if self.regularize_axis or self.m == 0 or self.axis_stop_s is None:
            return None
        return jnp.where(self.amplitude != 0, self.axis_stop_s, 0.)

    def potentials(self, field, xyz, t):
        u, w, zeta = xyz
        s = u*u + w*w
        # Cartesian polynomials avoid atan2/sqrt derivatives on the axis.
        harmonic = jnp.imag((u + 1j*w)**self.m
                            * jnp.exp(1j*(self.omega*t - self.n*zeta + self.phase)))
        if not self.regularize_axis and self.m:
            harmonic = jnp.where(s > 0, harmonic / jnp.where(s > 0, s, 1.)**(self.m/2),
                                  jnp.where(self.amplitude == 0, 0., jnp.nan))
        envelope = 1. if self.radial_width is None else jnp.exp(-((s-self.radial_center)/self.radial_width)**2)
        phi = self.amplitude * envelope * harmonic
        (iota, G, I), _ = field.profiles(s)
        alpha = (self.n - self.m*iota) * phi / (self.omega*(G+iota*I))
        return jnp.stack([alpha, phi])


class AlfvenSpectrum(BoozerPerturbation):
    """Superposition of prescribed modes, retaining each frequency and phase."""

    modes: tuple[BoozerPerturbation, ...]

    def __init__(self, *modes):
        if not modes or not all(isinstance(mode, BoozerPerturbation) for mode in modes):
            raise ValueError("AlfvenSpectrum requires one or more BoozerPerturbation objects")
        self.modes = tuple(modes)

    def potentials(self, field, xyz, t):
        return sum(mode.potentials(field, xyz, t) for mode in self.modes)

    def axis_stop_threshold(self):
        thresholds = [mode.axis_stop_threshold() for mode in self.modes]
        thresholds = [value for value in thresholds if value is not None]
        return jnp.max(jnp.stack(thresholds)) if thresholds else None


def poloidal_flux(field, s):
    """Integral_0^s iota(s') psi0 ds', using the equilibrium's exact splines."""
    knots = field.s_knots
    c = field.profile_coef[:, 0, :]

    def primitive(coef, h):
        return h*(coef[..., 3] + h*(coef[..., 2]/2 + h*(coef[..., 1]/3 + h*coef[..., 0]/4)))

    prefix = jnp.concatenate([jnp.zeros(1), jnp.cumsum(primitive(c, jnp.diff(knots)))])
    index = jnp.clip(jnp.searchsorted(knots, s, side="right")-1, 0, knots.size-2)
    return field.psi0 * (prefix[index] + primitive(c[index], s-knots[index]))


def wave_frame_invariant(field, mode, y, mu, mass, charge, t, *, helicity=0):
    """(n-Nm) H - omega P_zeta in exact QS, for ONE fixed-frequency mode.

    y=(u,w,zeta,v_parallel), mu=v_perp**2/(2 B0), N=helicity, and
    chi=theta-N*zeta. This is a numerical invariant only for an exactly QS
    equilibrium and a collisionless single-mode trajectory. H includes q Phi;
    P_zeta includes q alpha (G+N I). General multi-mode fields have no such
    invariant. It also becomes degenerate when n-Nm=0.
    """
    s = y[0]**2 + y[1]**2
    B = field.modB(s, jnp.arctan2(y[1], y[0]), y[2])
    (_, G, I), _ = field.profiles(s)
    alpha, phi = mode.potentials(field, y[:3], t)
    energy = mass*(0.5*y[3]**2 + mu*B) + charge*phi
    momentum = (G+helicity*I)*(mass*y[3]/B+charge*alpha)
    momentum += charge*(helicity*field.psi0*s-poloidal_flux(field, s))
    return (mode.n-helicity*mode.m)*energy-mode.omega*momentum
