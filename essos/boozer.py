"""Fast guiding-centre tracing in Boozer coordinates, with optional collisions.

:class:`BoozerField` holds a Boozer spectrum of ``|B|`` (``cos(m theta - n zeta)``
modes from ``booz_xform`` / ``booz_xform_jax``) with cubic splines in
``r = sqrt(s)`` and the profiles ``iota``, ``G`` and ``I`` with cubic splines in
``s``.  For ``m >= 1`` the spline holds ``b_mn / r``, so ``|B|`` is regular at
the magnetic axis.

:func:`trace_boozer` integrates the guiding-centre equations of White (2014)
in the ``K = 0`` Boozer form used by SIMSOPT (``GuidingCenterNoKBoozerRHS``),
in the chart ``(u, w) = sqrt(s) (cos theta, sin theta)``, which is regular on
the axis, with fixed-step RK4 under ``vmap``.  A particle is lost when it
reaches ``s = 1``.  With ``species`` (an
:class:`essos.background_species.BackgroundSpecies` whose profiles are given on
``s``) a Monte Carlo collision operator (pitch-angle scattering, slowing down
and energy diffusion, Ito Euler-Maruyama, Boozer & Kuo-Petravic, J. Comput.
Phys. 1981) acts after every orbit step; a particle whose energy falls below
``thermal_cutoff`` times the local temperature of species 0 is thermalised and
stops, counted as confined.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from essos.background_species import nu_D_ab, nu_par_ab, d_nu_par_ab, nu_s_ab, JOULE_PER_EV

jax.config.update("jax_enable_x64", True)


def _spline(x, y):
    from scipy.interpolate import CubicSpline

    sp = CubicSpline(np.asarray(x, float), np.asarray(y, float), axis=0)
    return jnp.asarray(sp.x), jnp.asarray(np.moveaxis(sp.c, 0, -1))  # (nint, ..., 4)


def _evaluate(knots, coef, x):
    """Value and derivative of a piecewise cubic (end pieces extrapolate)."""
    i = jnp.clip(jnp.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
    c = coef[i]
    d = x - knots[i]
    d = jnp.reshape(d, d.shape + (1,) * (c.ndim - 1 - d.ndim))
    value = ((c[..., 0] * d + c[..., 1]) * d + c[..., 2]) * d + c[..., 3]
    slope = (3 * c[..., 0] * d + 2 * c[..., 1]) * d + c[..., 2]
    return value, slope


class BoozerField(eqx.Module):
    """``|B|`` spectrum and ``iota, G, I`` profiles in Boozer coordinates."""

    r_knots: jax.Array
    b_coef: jax.Array  # (nint, modes, 4)
    s_knots: jax.Array
    profile_coef: jax.Array  # (nint, 3, 4): iota, G, I
    xm: jax.Array
    xn: jax.Array
    psi0: float
    nfp: int = eqx.field(static=True)
    # (max m, min n / nfp, max n / nfp): the phases are then built from powers
    # of exp(i theta) and exp(-i nfp zeta) instead of one cos/sin per mode.
    harmonics: tuple | None = eqx.field(static=True, default=None)

    @classmethod
    def from_booz(cls, s, bmnc, xm, xn, iota, G, I, psi0, nfp, mode_tolerance=1e-6):
        """Build from half-mesh Boozer tables.

        ``bmnc`` is ``(modes, len(s))`` as written by ``booz_xform``; ``xn``
        includes the ``nfp`` factor; ``psi0`` is the toroidal flux at the
        boundary over ``2 pi``.  Modes whose amplitude never exceeds
        ``mode_tolerance`` times the largest amplitude are dropped.
        """
        s = np.asarray(s, float)
        bmnc = np.asarray(bmnc, float)
        xm = np.asarray(xm, int)
        amplitude = np.abs(bmnc).max(axis=1)
        keep = amplitude > mode_tolerance * amplitude.max()
        bmnc, xm, xn = bmnc[keep], xm[keep], np.asarray(xn, int)[keep]
        r = np.sqrt(s)
        scaled = np.where(xm[:, None] > 0, bmnc / r, bmnc)
        r_knots, b_coef = _spline(r, scaled.T)
        profiles = np.stack([iota, G, np.asarray(I, float)], axis=1)
        # Axis row: iota and G extrapolated linearly, I(0) = 0.
        axis = profiles[0] - s[0] * (profiles[1] - profiles[0]) / (s[1] - s[0])
        axis[2] = 0.0
        s_prof = np.concatenate([[0.0], s])
        profiles = np.vstack([axis, profiles])
        s_knots, profile_coef = _spline(s_prof, profiles)
        n = xn // int(nfp)
        harmonics = (int(xm.max()), int(n.min()), int(n.max()))
        return cls(r_knots, b_coef, s_knots, profile_coef, jnp.asarray(xm), jnp.asarray(xn),
                   float(psi0), int(nfp), harmonics)

    def _phases(self, theta, zeta):
        """``cos`` and ``sin`` of ``m theta - n zeta`` for every mode."""
        if self.harmonics is None:
            phase = self.xm * theta - self.xn * zeta
            return jnp.cos(phase), jnp.sin(phase)
        m_max, n_min, n_max = self.harmonics

        def table(x, k0, k1):  # cos and sin of k x for k = k0 .. k1, by angle addition
            c1, s1 = jnp.cos(x), jnp.sin(x)
            c, s = [jnp.cos(k0 * x)], [jnp.sin(k0 * x)]
            for _ in range(k1 - k0):
                c, s = c + [c[-1] * c1 - s[-1] * s1], s + [s[-1] * c1 + c[-1] * s1]
            return jnp.stack(c), jnp.stack(s)

        # one-hot selections as small matrix products, which vectorize better than gathers
        pick_m = jax.nn.one_hot(self.xm, m_max + 1)
        pick_n = jax.nn.one_hot(self.xn // self.nfp - n_min, n_max - n_min + 1)
        cm, sm = (x @ pick_m.T for x in table(theta, 0, m_max))
        cn, sn = (x @ pick_n.T for x in table(self.nfp * zeta, n_min, n_max))
        return cm * cn + sm * sn, sm * cn - cm * sn

    @classmethod
    def from_booz_xform(cls, booz, psi0, mode_tolerance=1e-6):
        """Build from a run ``Booz_xform`` object (every surface computed)."""
        return cls.from_booz(booz.s_b, booz.bmnc_b, booz.xm_b, booz.xn_b, booz.iota,
                             booz.Boozer_G, booz.Boozer_I, psi0, int(booz.nfp), mode_tolerance)

    def profiles(self, s):
        """``(iota, G, I)`` and their ``s`` derivatives."""
        return _evaluate(self.s_knots, self.profile_coef, s)

    def modB_derivatives(self, r, theta, zeta):
        """``|B|``, ``d|B|/dr``, ``(d|B|/dtheta)/r`` and ``d|B|/dzeta``."""
        a, da = _evaluate(self.r_knots, self.b_coef, r)
        c, s = self._phases(theta, zeta)
        has_m = self.xm > 0
        f = jnp.where(has_m, r * a, a)
        df = jnp.where(has_m, a + r * da, da)
        return (jnp.sum(f * c), jnp.sum(df * c), -jnp.sum(self.xm * a * s),
                jnp.sum(self.xn * f * s))

    def modB(self, s, theta, zeta):
        return self.modB_derivatives(jnp.sqrt(s), theta, zeta)[0]


def _chart(y):
    u, w = y[0], y[1]
    s = u * u + w * w
    r = jnp.sqrt(s)
    safe = jnp.where(r > 0, r, 1.0)
    cos_t, sin_t = jnp.where(r > 0, u / safe, 1.0), jnp.where(r > 0, w / safe, 0.0)
    return s, r, jnp.arctan2(w, u), cos_t, sin_t, safe


def guiding_center_rhs(field, y, mu, mass, charge):
    """Time derivative of ``(u, w, zeta, v_par)``; ``mu = v_perp^2 / (2 |B|)``."""
    s, r, theta, cos_t, sin_t, safe = _chart(y)
    zeta, vpar = y[2], y[3]
    B, dB_dr, dB_dtheta_r, dB_dzeta = field.modB_derivatives(r, theta, zeta)
    (iota, G, I), (_, dG_ds, dI_ds) = field.profiles(s)
    psi0 = field.psi0
    I_r = jnp.where(r > 0, I / safe, 0.0)
    fak1 = mass * (vpar * vpar / B + mu)
    C = -charge * iota + mass * vpar * dG_ds / (psi0 * B)
    F = charge + mass * vpar * dI_ds / (psi0 * B)
    D_iota = F * G - C * I
    # ds/dt / (2 r), r dtheta/dt and dzeta/dt, all regular on the axis.
    sdot_2r = (I_r * dB_dzeta - G * dB_dtheta_r) * fak1 / (2 * D_iota * psi0)
    r_thetadot = (G * dB_dr / (2 * psi0) * fak1 - r * C * vpar * B) / D_iota
    zetadot = (F * vpar * B - dB_dr * I_r / (2 * psi0) * fak1) / D_iota
    vpardot = (C * r * dB_dtheta_r - F * dB_dzeta) * mu * B / D_iota
    return jnp.array([cos_t * sdot_2r - sin_t * r_thetadot,
                      sin_t * sdot_2r + cos_t * r_thetadot, zetadot, vpardot])


def _collision_rates(species, mass, charge, v, point):
    idx = species.species_indeces

    def total(nu):
        return jnp.sum(jax.vmap(nu, in_axes=(None, None, 0, None, None, None))(
            mass, charge, idx, v, point, species))

    return total(nu_D_ab), total(nu_s_ab), total(nu_par_ab), total(d_nu_par_ab)


def collision_kick(species, mass, charge, v, pitch, point, dt, noise):
    """One Ito Euler-Maruyama step of the Lorentz and energy-scattering operator.

    ``dv = (-nu_s v + v^-2 d(v^2 D)/dv) dt + sqrt(2 D dt) xi_1`` with
    ``D = v^2 nu_par / 2``, and ``dlambda = -lambda nu_D dt +
    sqrt((1 - lambda^2) nu_D dt) xi_2``.
    """
    nu_D, nu_s, nu_par, dnu_par = _collision_rates(species, mass, charge, v, point)
    D = 0.5 * v * v * nu_par
    dD = v * nu_par + 0.5 * v * v * dnu_par
    v_new = jnp.abs(v + (-nu_s * v + 2 * D / v + dD) * dt + jnp.sqrt(2 * D * dt) * noise[0])
    lam = pitch - pitch * nu_D * dt + jnp.sqrt(jnp.maximum(1 - pitch**2, 0.0) * nu_D * dt) * noise[1]
    lam = jnp.where(jnp.abs(lam) > 1, jnp.sign(lam) * (2 - jnp.abs(lam)), lam)
    return v_new, jnp.clip(lam, -1.0, 1.0)


@dataclasses.dataclass(frozen=True)
class BoozerTrace:
    """Result of :func:`trace_boozer` (NumPy arrays).

    ``states`` is ``(particles, n_save, 5)``: ``s, theta, zeta, v_par`` and the
    speed ``v``, held at the loss or thermalisation point afterwards.
    ``loss_times`` is ``-1`` for particles that were not lost;
    ``thermalized_times`` likewise.
    """

    times: np.ndarray
    states: np.ndarray
    loss_times: np.ndarray
    thermalized_times: np.ndarray
    energy_error: np.ndarray  # max |E/E0 - 1| per particle, at the saved times when collisionless

    @property
    def lost(self):
        return self.loss_times >= 0

    def loss_fractions(self):
        """Cumulative lost fraction at each saved time."""
        lt = self.loss_times[self.lost]
        return np.array([(lt <= t).sum() for t in self.times]) / self.loss_times.size


def trace_boozer(field, s, theta, zeta, pitch, *, speed, mass, charge, tmax, timestep,
                 n_save=100, species=None, seed=0, thermal_cutoff=1.5, devices=None):
    """Trace guiding centres from Boozer ``(s, theta, zeta)`` with pitch ``v_par/v``.

    The step is shortened so that a whole number of steps fits between the
    ``n_save`` saved times (``t = 0`` included).  Particles are sharded over
    ``devices`` (default: every local device).
    """
    s, theta, zeta, pitch = (jnp.atleast_1d(jnp.asarray(a, float)) for a in (s, theta, zeta, pitch))
    n = s.size
    n_int = max(int(n_save) - 1, 1)
    n_sub = max(1, int(np.ceil(float(tmax) / n_int / float(timestep) - 1e-9)))
    dt = float(tmax) / (n_int * n_sub)
    speed = jnp.full(n, float(speed))
    r = jnp.sqrt(s)
    B0 = jax.vmap(field.modB)(s, theta, zeta)
    y0 = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), zeta, pitch * speed], axis=1)
    mu0 = speed**2 * (1 - pitch**2) / (2 * B0)
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), n)

    def rhs(y, mu):
        return guiding_center_rhs(field, y, mu, mass, charge)

    def one(y0, mu0, key):
        def step(carry, k):
            y, mu, t, alive, t_loss, t_therm, err, e0 = carry
            k1 = rhs(y, mu)
            k2 = rhs(y + 0.5 * dt * k1, mu)
            k3 = rhs(y + 0.5 * dt * k2, mu)
            k4 = rhs(y + dt * k3, mu)
            y1 = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            s1, r1, th1, *_ = _chart(y1)
            thermal = jnp.asarray(False)
            if species is not None:
                B1 = field.modB_derivatives(r1, th1, y1[2])[0]
                e_orbit = 0.5 * y1[3] ** 2 + mu * B1
                err = jnp.where(alive, jnp.maximum(err, jnp.abs(e_orbit / e0 - 1)), err)
                v = jnp.sqrt(2 * e_orbit)
                point = jnp.array([s1, th1, y1[2]])
                v, lam = collision_kick(species, mass, charge, v, y1[3] / v, point, dt,
                                        jax.random.normal(jax.random.fold_in(key, k), (2,)))
                y1 = y1.at[3].set(lam * v)
                mu1 = v * v * (1 - lam * lam) / (2 * B1)
                t_bulk = species.get_temperature(0, point) * JOULE_PER_EV
                thermal = 0.5 * mass * v * v < thermal_cutoff * t_bulk
                e0_new = 0.5 * v * v
            else:
                mu1, e0_new = mu, e0
            lost = alive & (s1 >= 1.0)
            therm = alive & ~lost & thermal
            t_loss = jnp.where(lost, t + dt, t_loss)
            t_therm = jnp.where(therm, t + dt, t_therm)
            keep = alive & ~therm & (jnp.isfinite(y1).all())
            y = jnp.where(keep | lost, y1, y)
            mu = jnp.where(keep, mu1, mu)
            e0 = jnp.where(keep, e0_new, e0)
            return (y, mu, t + dt, keep & ~lost, t_loss, t_therm, err, e0), None

        def interval(carry, i):
            carry, _ = jax.lax.scan(step, carry, i * n_sub + jnp.arange(n_sub))
            y, mu = carry[0], carry[1]
            s1, r1, th1, *_ = _chart(y)
            B1 = field.modB_derivatives(r1, th1, y[2])[0]
            v = jnp.sqrt(y[3] ** 2 + 2 * mu * B1)
            if species is None:  # collisionless: the energy is checked at saved times
                err = jnp.where(carry[3] | (carry[4] >= 0),
                                jnp.maximum(carry[6], jnp.abs(0.5 * v * v / carry[7] - 1)), carry[6])
                carry = carry[:6] + (err,) + carry[7:]
            return carry, jnp.array([s1, th1, y[2], y[3], v])

        e0 = 0.5 * y0[3] ** 2 + mu0 * field.modB_derivatives(*(_chart(y0)[1:3]), y0[2])[0]
        carry = (y0, mu0, 0.0, jnp.asarray(True), -1.0, -1.0, 0.0, e0)
        carry, saved = jax.lax.scan(interval, carry, jnp.arange(n_int))
        first = jnp.array([_chart(y0)[0], _chart(y0)[2], y0[2], y0[3], jnp.sqrt(2 * e0)])
        return jnp.vstack([first, saved]), carry[4], carry[5], carry[6]

    devices = jax.devices() if devices is None else devices
    ndev = max(1, min(len(devices), n))
    pad = (-n) % ndev
    args = [jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)]) for a in (y0, mu0, keys)]
    run = jax.jit(jax.vmap(one))
    if ndev > 1:
        sharding = NamedSharding(Mesh(np.asarray(devices[:ndev]), ("p",)), PartitionSpec("p"))
        args = [jax.device_put(a, sharding) for a in args]
    states, t_loss, t_therm, err = (np.asarray(x)[:n] for x in run(*args))
    return BoozerTrace(np.linspace(0.0, float(tmax), n_int + 1), states, t_loss, t_therm, err)
