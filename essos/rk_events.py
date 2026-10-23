"""Terminal event replay for forward RK solves.

The driver stops on the first accepted endpoint crossing. Only that final step
is replayed to locate the event with the ordinary RK interpolant. This avoids
carrying a root-search loop through every ordinary adaptive trial under vmap.
"""

import jax.numpy as jnp
from jax import lax


def finish_event_root(rhs, args, state, *, trial_step, nodes, rows, weights,
                      functions, events, interpolate, ts, record_steps):
    """Refine the terminal crossing; return state and root validity.

    Unreached saves retain the driver's infinity convention. Callers must
    retry with inline root searches if a root evaluation is nonfinite: the
    ordinary driver can reject that crossing and retry a smaller step.
    Initial events and failures do not replay any steps.
    """
    from .rk import RKStatus, _first_hit

    def locate(carry):
        _, state, _ = carry
        t, y, h = state["last_t"], state["last_y"], state["last_h"]
        trial = trial_step(rhs, t, y, h, args, state["last_f"], nodes, rows, weights)
        values = events(t, y)
        fractions, validity = [], []
        for function, old, hit in zip(functions, values, state["last_triggered"]):
            if old.dtype == jnp.bool_:
                fraction, valid = jnp.ones_like(t), jnp.asarray(True)
            else:
                def bisect(c):
                    lo, hi, iteration, valid = c
                    mid = (lo + hi) / 2
                    value = function(t + mid * h, interpolate(y, h, trial, mid), args)
                    same = (value > 0) == (old > 0)
                    return (jnp.where(same, mid, lo), jnp.where(same, hi, mid),
                            iteration + 1, valid & jnp.isfinite(value))

                lo, hi, _, valid = lax.while_loop(
                    lambda c: hit & c[3] & (c[2] < 52), bisect,
                    (jnp.zeros_like(t), jnp.ones_like(t), jnp.asarray(0), jnp.asarray(True)))
                fraction = (lo + hi) / 2
            fractions.append(jnp.where(hit, fraction, jnp.inf))
            validity.append(valid)
        fraction = jnp.min(jnp.asarray(fractions))
        valid = jnp.all(jnp.asarray(validity))
        stop_t = t + fraction * h
        stop_y = interpolate(y, h, trial, fraction)
        hits = _first_hit(tuple(hit & (value == fraction)
                               for hit, value in zip(state["last_triggered"], fractions)))
        state = dict(state, t=stop_t, y=stop_y, hits=hits)
        if len(ts):
            retain = state["saved_ts"] <= stop_t
            state = dict(state,
                ys=jnp.where(retain[:, None], jnp.where(
                    (state["saved_ts"] == stop_t)[:, None], stop_y, state["ys"]), jnp.inf),
                saved_ts=jnp.where(retain, state["saved_ts"], jnp.inf),
                index=jnp.sum(retain).astype(state["index"].dtype))
        if record_steps:
            state = dict(state, history=state["history"].at[state["accepted"]].set(stop_t))
        return jnp.asarray(False), state, valid

    crossing = (state["status"] == RKStatus.EVENT) & (state["accepted"] > 0)
    _, state, valid = lax.while_loop(lambda carry: carry[0], locate,
                                    (crossing, state, jnp.asarray(True)))
    return state, valid
