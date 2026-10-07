"""Prototype: Boozer guiding-centre tracing as one Pallas GPU kernel.

Each program instance owns a block of particles and runs every step inside
the kernel, with the ``|B|`` spectrum and the profile splines loaded once.
Same equations and chart as :mod:`essos.boozer`; fixed-step RK4, collisionless.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


def _pow2(n):
    return 1 << max(int(n) - 1, 1).bit_length()


def pack(field, dtype=jnp.float64):
    """Pad the field to power-of-two tiles: modes, radial and profile knots."""
    nint, modes = np.asarray(field.b_coef).shape[:2]
    mp, kp = _pow2(modes), _pow2(nint + 1)
    coef = np.zeros((4, kp, mp))
    coef[:, :nint, :modes] = np.moveaxis(np.asarray(field.b_coef), -1, 0)
    knots = np.full(kp, np.inf)
    knots[:nint + 1] = np.asarray(field.r_knots)
    xm = np.zeros(mp); xm[:modes] = np.asarray(field.xm)
    xn = np.zeros(mp); xn[:modes] = np.asarray(field.xn)
    pint = np.asarray(field.profile_coef).shape[0]
    sp = _pow2(pint + 1)
    pcoef = np.zeros((4, sp, 4))  # iota, G, I and a zero pad row
    pcoef[:3, :pint] = np.moveaxis(np.asarray(field.profile_coef), 1, 0)
    sknots = np.full(sp, np.inf)
    sknots[:pint + 1] = np.asarray(field.s_knots)
    cast = lambda a: jnp.asarray(a, dtype)
    return dict(coef=cast(coef), knots=cast(knots), xm=cast(xm), xn=cast(xn), nint=nint,
                pcoef=cast(pcoef), sknots=cast(sknots), pint=pint, psi0=float(field.psi0))


def _interval(knots, x, n):
    """Index of the cubic piece holding ``x`` (end pieces extrapolate), per lane."""
    count = jnp.sum((x[:, None] >= knots[None, :]).astype(jnp.int32), axis=1)
    return jnp.clip(count - 1, 0, n - 1)


def trace_rk4(field, y0, mu, *, mass, charge, dt, n_steps, block=32, dtype=jnp.float64):
    """Final states, loss times and max energy errors after ``n_steps`` RK4 steps."""
    p = pack(field, dtype)
    nint, pint, psi0 = p["nint"], p["pint"], p["psi0"]
    n = y0.shape[0]
    npad = -(-n // block) * block
    y0 = jnp.pad(jnp.asarray(y0, dtype), ((0, npad - n), (0, 0)), mode="edge")
    mu = jnp.pad(jnp.asarray(mu, dtype), (0, npad - n), mode="edge")
    mp, kp, sp = p["xm"].size, p["knots"].size, p["sknots"].size

    def kernel(u_ref, w_ref, z_ref, v_ref, mu_ref, coef_ref, knots_ref, xm_ref, xn_ref, pcoef_ref, sknots_ref,
               uo_ref, wo_ref, zo_ref, vo_ref, tl_ref, err_ref):
        knots = knots_ref[...]
        xm, xn = xm_ref[...], xn_ref[...]
        sknots = sknots_ref[...]
        mu = mu_ref[...]
        modes = jnp.arange(mp)
        has_m = xm[None, :] > 0

        def field_at(r, theta, zeta):
            i = _interval(knots, r, nint)
            d = (r - knots[i])[:, None]
            c = [coef_ref[k, i[:, None], modes[None, :]] for k in range(4)]
            a = ((c[0] * d + c[1]) * d + c[2]) * d + c[3]
            da = (3 * c[0] * d + 2 * c[1]) * d + c[2]
            phase = xm[None, :] * theta[:, None] - xn[None, :] * zeta[:, None]
            cs, sn = jnp.cos(phase), jnp.sin(phase)
            f = jnp.where(has_m, r[:, None] * a, a)
            df = jnp.where(has_m, a + r[:, None] * da, da)
            return (jnp.sum(f * cs, 1), jnp.sum(df * cs, 1), -jnp.sum(xm[None, :] * a * sn, 1),
                    jnp.sum(xn[None, :] * f * sn, 1))

        def profiles(s):
            j = _interval(sknots, s, pint)
            d = s - sknots[j]
            out = []
            for q in range(3):
                c = [pcoef_ref[q, j, k] for k in range(4)]
                out.append((((c[0] * d + c[1]) * d + c[2]) * d + c[3],
                            (3 * c[0] * d + 2 * c[1]) * d + c[2]))
            return out

        def chart(u, w):
            s = u * u + w * w
            r = jnp.sqrt(s)
            safe = jnp.where(r > 0, r, 1.0)
            return s, r, jnp.arctan2(w, u), jnp.where(r > 0, u / safe, 1.0), jnp.where(r > 0, w / safe, 0.0), safe

        def rhs(u, w, zeta, vpar):
            s, r, theta, ct, st, safe = chart(u, w)
            B, dBr, dBt, dBz = field_at(r, theta, zeta)
            (iota, _), (G, dG), (I, dI) = profiles(s)
            Ir = jnp.where(r > 0, I / safe, 0.0)
            fak1 = mass * (vpar * vpar / B + mu)
            C = -charge * iota + mass * vpar * dG / (psi0 * B)
            F = charge + mass * vpar * dI / (psi0 * B)
            D = F * G - C * I
            sdot = (Ir * dBz - G * dBt) * fak1 / (2 * D * psi0)
            rth = (G * dBr / (2 * psi0) * fak1 - r * C * vpar * B) / D
            zdot = (F * vpar * B - dBr * Ir / (2 * psi0) * fak1) / D
            vdot = (C * r * dBt - F * dBz) * mu * B / D
            return ct * sdot - st * rth, st * sdot + ct * rth, zdot, vdot

        def energy(u, w, zeta, vpar):
            _, r, theta, *_ = chart(u, w)
            return 0.5 * vpar * vpar + mu * field_at(r, theta, zeta)[0]

        state = (u_ref[...], w_ref[...], z_ref[...], v_ref[...])
        e0 = energy(*state)
        alive = jnp.ones_like(mu, dtype=jnp.bool_)

        def step(k, carry):
            state, alive, t_loss, err = carry
            k1 = rhs(*state)
            k2 = rhs(*(x + 0.5 * dt * g for x, g in zip(state, k1)))
            k3 = rhs(*(x + 0.5 * dt * g for x, g in zip(state, k2)))
            k4 = rhs(*(x + dt * g for x, g in zip(state, k3)))
            new = tuple(x + dt / 6 * (a + 2 * b + 2 * c + e) for x, a, b, c, e in zip(state, k1, k2, k3, k4))
            s1 = new[0] * new[0] + new[1] * new[1]
            err = jnp.where(alive, jnp.maximum(err, jnp.abs(energy(*new) / e0 - 1)), err)
            lost = alive & (s1 >= 1.0)
            t_loss = jnp.where(lost, (k + 1) * dt, t_loss)
            state = tuple(jnp.where(alive, a, b) for a, b in zip(new, state))
            return state, alive & ~lost, t_loss, err

        state, alive, t_loss, err = jax.lax.fori_loop(
            0, n_steps, step, (state, alive, jnp.full_like(mu, -1.0), jnp.zeros_like(mu)))
        uo_ref[...], wo_ref[...], zo_ref[...], vo_ref[...] = state
        tl_ref[...] = t_loss
        err_ref[...] = err

    whole = lambda shape: pl.BlockSpec(shape, lambda g: (0,) * len(shape))
    call = pl.pallas_call(
        kernel, grid=(npad // block,),
        in_specs=[pl.BlockSpec((block,), lambda g: (g,))] * 5 + [
                  whole((4, kp, mp)), whole((kp,)), whole((mp,)), whole((mp,)), whole((4, sp, 4)), whole((sp,))],
        out_specs=[pl.BlockSpec((block,), lambda g: (g,))] * 6,
        out_shape=[jax.ShapeDtypeStruct((npad,), dtype)] * 6)
    *yo, tl, err = jax.jit(call)(*(y0[:, k] for k in range(4)), mu, p["coef"], p["knots"], p["xm"], p["xn"],
                                 p["pcoef"], p["sknots"])
    return jnp.stack(yo, axis=1)[:n], tl[:n], err[:n]
