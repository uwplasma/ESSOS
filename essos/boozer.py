"""Fixed-step K=0 Boozer guiding-centre tracing (White, 2014).

Interpolate the |B| spectrum in sqrt(s), with m>=1 coefficients divided by
sqrt(s) for axis regularity; interpolate iota, G and I in s. RK4 or Dopri8 runs in
sqrt(s)*(cos(theta), sin(theta)). LCFS crossings are lost; nonfinite steps
are failed. Optional Euler-Maruyama collisions apply pitch scattering,
slowing down and energy diffusion (Boozer & Kuo-Petravic, 1981).
"""

from __future__ import annotations

import dataclasses
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from essos.background_species import nu_D_ab, nu_par_ab, d_nu_par_ab, nu_s_ab, JOULE_PER_EV

jax.config.update("jax_enable_x64", True)


def psi0_from_vmec(phi_edge):
    """Boozer guiding-centre psi0 from VMEC's edge toroidal flux [Wb]."""
    return -float(phi_edge) / (2 * np.pi)


def _spline(x, y):
    from scipy.interpolate import CubicSpline

    sp = CubicSpline(np.asarray(x, float), np.asarray(y, float), axis=0)
    return jnp.asarray(sp.x), jnp.asarray(np.moveaxis(sp.c, 0, -1))  # (nint, ..., 4)


def _evaluate(knots, coef, x):
    """Value and derivative of a piecewise cubic (end pieces extrapolate)."""
    i = jnp.clip(jnp.searchsorted(knots, x, side="right", method="compare_all") - 1, 0, knots.size - 2)
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

    @classmethod
    def from_booz(cls, s, bmnc, xm, xn, iota, G, I, psi0, nfp, mode_tolerance=1e-6):
        """Build from half-mesh Boozer tables.

        ``bmnc`` is ``(modes, len(s))`` as written by ``booz_xform``; ``xn``
        includes the ``nfp`` factor; ``psi0`` uses the Boozer convention
        (``-VMEC phi[-1]/(2 pi)``).  Modes whose amplitude never exceeds
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
        return cls(r_knots, b_coef, s_knots, profile_coef, jnp.asarray(xm), jnp.asarray(xn),
                   float(psi0), int(nfp))

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
        phase = self.xm * theta - self.xn * zeta
        c, s = jnp.cos(phase), jnp.sin(phase)
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
    ``thermalized_times`` and ``failed_times`` likewise. A failed orbit
    is frozen at its last finite state and is not counted as confined.
    """

    times: np.ndarray
    states: np.ndarray
    loss_times: np.ndarray
    thermalized_times: np.ndarray
    energy_error: np.ndarray  # max |E/E0 - 1| per particle (collisionless drift only)
    failed_times: np.ndarray | None = None

    @property
    def lost(self):
        return self.loss_times >= 0

    @property
    def failed(self):
        return (np.zeros_like(self.loss_times, dtype=bool) if self.failed_times is None
                else self.failed_times >= 0)

    def loss_fractions(self, max_energy_error=1e-3):
        """Cumulative losses; ``None`` disables only the finite energy-drift limit."""
        if (self.failed.any() or not np.isfinite(self.energy_error).all()
                or not np.isfinite(self.states).all()):
            raise RuntimeError("Loss fraction is undefined when particle trajectories fail; inspect failed_times")
        if max_energy_error is not None:
            if (np.ndim(max_energy_error) != 0 or not np.isfinite(max_energy_error)
                    or max_energy_error < 0):
                raise ValueError("max_energy_error must be nonnegative and finite, or None")
            error = np.max(self.energy_error, initial=0.0)
            if error > max_energy_error:
                raise RuntimeError(f"Relative energy drift {error:.3g} exceeds {max_energy_error:.3g}; reduce timestep")
        lt = self.loss_times[self.lost]
        return np.searchsorted(np.sort(lt), self.times, side="right") / self.loss_times.size


def _rk_step(rhs, y, dt, method):
    if method == "rk4":
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    from diffrax import Dopri8

    tableau = Dopri8.tableau
    stages = [rhs(y)]
    for weights in tableau.a_lower[:np.flatnonzero(tableau.b_sol)[-1]]:
        stages.append(rhs(y + dt * sum(a * k for a, k in zip(weights, stages) if a)))
    return y + dt * sum(b * k for b, k in zip(tableau.b_sol, stages) if b)


@jax.jit
def _start(field, y0, mu0):
    def one(y, mu):
        s, r, theta, *_ = _chart(y)
        e0 = 0.5 * y[3] ** 2 + mu * field.modB_derivatives(r, theta, y[2])[0]
        first = jnp.array([s, theta, y[2], y[3], jnp.sqrt(2 * e0)])
        return (y, mu, 0.0, jnp.asarray(True), -1.0, -1.0, -1.0, 0.0, e0), first

    return jax.vmap(one)(y0, mu0)

@partial(jax.jit, static_argnames=("n_sub", "count", "species", "method"))
def _advance(field, dt, n_sub, mass, charge, species, thermal_cutoff,
             carry, keys, first_interval, count, method="rk4"):
    def one(carry, key):
        def step(carry, k):
            y, mu, t, alive, t_loss, t_therm, t_fail, err, e0 = carry
            rhs = lambda state: guiding_center_rhs(field, state, mu, mass, charge)
            y1 = _rk_step(rhs, y, dt, method)
            s1, r1, th1, *_ = _chart(y1)
            B1 = field.modB_derivatives(r1, th1, y1[2])[0]
            e_orbit = 0.5 * y1[3] ** 2 + mu * B1
            err = jnp.where(alive, jnp.maximum(err, jnp.abs(e_orbit / e0 - 1)), err)
            thermal = jnp.asarray(False)
            if species is not None:
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
            finite = (jnp.isfinite(y1).all() & jnp.isfinite(mu1) &
                      jnp.isfinite(e_orbit) & jnp.isfinite(e0_new) &
                      jnp.isfinite(B1) & (B1 > 0))
            lost = alive & finite & (s1 >= 1.0)
            failed = alive & ~lost & ~finite
            t_fail = jnp.where(failed, t + dt, t_fail)
            err = jnp.where(alive & ~finite, jnp.inf, err)
            therm = alive & ~lost & finite & thermal
            t_loss = jnp.where(lost, t + dt, t_loss)
            t_therm = jnp.where(therm, t + dt, t_therm)
            keep = alive & ~therm & finite
            y = jnp.where(keep | lost, y1, y)
            mu = jnp.where(keep, mu1, mu)
            e0 = jnp.where(keep, e0_new, e0)
            return (y, mu, t + dt, keep & ~lost, t_loss, t_therm, t_fail, err, e0), None

        def interval(carry, i):
            carry, _ = jax.lax.scan(step, carry, i * n_sub + jnp.arange(n_sub))
            y, mu = carry[0], carry[1]
            s1, r1, th1, *_ = _chart(y)
            B1 = field.modB_derivatives(r1, th1, y[2])[0]
            v = jnp.sqrt(y[3] ** 2 + 2 * mu * B1)
            return carry, jnp.array([s1, th1, y[2], y[3], v])

        return jax.lax.scan(interval, carry, first_interval + jnp.arange(count))

    return jax.vmap(one)(carry, keys)


def trace_boozer(field, s, theta, zeta, pitch, *, speed, mass, charge, tmax, timestep,
                 n_save=100, species=None, seed=0, thermal_cutoff=1.5, devices=None,
                 progress=None, compact=False, method="rk4"):
    """Trace guiding centres from Boozer ``(s, theta, zeta)`` with pitch ``v_par/v``.

    The step is shortened so that a whole number of steps fits between the
    ``n_save`` saved times (``t = 0`` included).  Particles are sharded over
    ``devices`` (default: every local device).

    ``progress``, if given, is called as ``progress(done, total)`` in saved
    intervals: the horizon then runs as up to ten host-side chunks of the same
    compiled program, with the whole state carried between them, so the orbits
    are those of an unchunked trace.
    ``compact`` can omit stopped particles after the first saved interval on
    one device; it adds a host synchronization and another compiled batch size.
    ``method="dopri8"`` uses a fixed, non-symplectic eighth-order Dormand–Prince step;
    the default is ``"rk4"``. Refine timestep and modes to check loss labels.
    """
    if method not in ("rk4", "dopri8"):
        raise ValueError("method must be 'rk4' or 'dopri8'")
    inputs = tuple(np.atleast_1d(np.asarray(a, float)) for a in (s, theta, zeta, pitch))
    n = inputs[0].size
    if n < 1 or any(a.ndim != 1 or a.size != n or not np.isfinite(a).all() for a in inputs):
        raise ValueError("Boozer births must be nonempty, one-dimensional, equally sized and finite")
    if np.any((inputs[0] < 0) | (inputs[0] >= 1)) or np.any(np.abs(inputs[3]) > 1):
        raise ValueError("Boozer births require 0 <= s < 1 and |pitch| <= 1")
    if not (np.isfinite(tmax) and tmax > 0 and np.isfinite(timestep) and timestep > 0
            and np.ndim(n_save) == 0 and np.isfinite(n_save)
            and n_save >= 2 and n_save == int(n_save)):
        raise ValueError("tmax and timestep must be positive and finite; n_save >= 2")
    if (not all(np.ndim(x) == 0 and np.isfinite(x) for x in (speed, mass, charge))
            or speed <= 0 or mass <= 0 or charge == 0):
        raise ValueError("speed and mass must be positive; charge must be nonzero and finite")
    s, theta, zeta, pitch = map(jnp.asarray, inputs)
    n_int = max(int(n_save) - 1, 1)
    n_sub = max(1, int(np.ceil(float(tmax) / n_int / float(timestep) - 1e-9)))
    dt = float(tmax) / (n_int * n_sub)
    speed = jnp.full(n, float(speed))
    r = jnp.sqrt(s)
    B0 = jax.vmap(field.modB)(s, theta, zeta)
    B0_host = np.asarray(B0)
    if not np.all(np.isfinite(B0_host) & (B0_host > 0)):
        raise ValueError("Boozer birth |B| must be positive and finite")
    y0 = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), zeta, pitch * speed], axis=1)
    mu0 = speed**2 * (1 - pitch**2) / (2 * B0)
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), n)

    devices = jax.devices() if devices is None else devices
    ndev = max(1, min(len(devices), n))
    pad = (-n) % ndev
    args = [jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)]) for a in (y0, mu0, keys)]
    if ndev > 1:
        sharding = NamedSharding(Mesh(np.asarray(devices[:ndev]), ("p",)), PartitionSpec("p"))
        args = [jax.device_put(a, sharding) for a in args]
    y0, mu0, keys = args
    carry, first = _start(field, y0, mu0)
    chunk = n_int if progress is None else -(-n_int // 10)
    if compact and n_int > 1 and ndev == 1:
        initial = carry
        carry, part = _advance(field, dt, n_sub, mass, charge, species, thermal_cutoff,
                               carry, keys, 0, 1, method)
        active = np.flatnonzero(np.asarray(carry[3])[:n])
        padded = 1 << (active.size - 1).bit_length() if active.size else 0
        if padded <= n // 2:
            selected = active
            if active.size:
                indices = np.pad(active, (0, padded - active.size), mode="edge")
                running = tuple(x[indices] for x in carry)
                running_keys = keys[indices]
            states = np.empty((n, n_int + 1, 5), dtype=np.asarray(part).dtype)
            states[:, 0] = np.asarray(first)[:n]
            states[:, 1:] = np.asarray(part)[:n]
            boundaries = (list(range(chunk, n_int, chunk)) + [n_int]
                          if progress is not None else [n_int])
            first_interval = 1
            for stop in boundaries:
                if stop > first_interval and selected.size:
                    running, next_part = _advance(field, dt, n_sub, mass, charge, species,
                                                  thermal_cutoff, running, running_keys,
                                                  first_interval, stop - first_interval, method)
                    states[selected, first_interval + 1:stop + 1] = np.asarray(next_part)[:selected.size]
                if progress is not None:
                    progress(stop, n_int)
                first_interval = stop
            status = [np.asarray(carry[i])[:n].copy() for i in (4, 5, 6, 7)]
            if selected.size:
                for result, i in zip(status, (4, 5, 6, 7)):
                    result[selected] = np.asarray(running[i])[:selected.size]
            t_loss, t_therm, t_fail, err = status
            return BoozerTrace(np.linspace(0.0, float(tmax), n_int + 1), states,
                               t_loss, t_therm, err, t_fail)
        carry = initial
    saved = []
    for first_interval in range(0, n_int, chunk):
        count = min(chunk, n_int - first_interval)
        carry, part = _advance(field, dt, n_sub, mass, charge, species, thermal_cutoff,
                               carry, keys, first_interval, count, method)
        saved.append(part)
        if progress is not None:
            jax.block_until_ready(part)
            progress(first_interval + count, n_int)
    states = np.concatenate([np.asarray(first)[:, None]] + [np.asarray(p) for p in saved], axis=1)[:n]
    t_loss, t_therm, t_fail, err = (np.asarray(x)[:n] for x in (carry[4], carry[5], carry[6], carry[7]))
    return BoozerTrace(np.linspace(0.0, float(tmax), n_int + 1), states, t_loss, t_therm, err, t_fail)
