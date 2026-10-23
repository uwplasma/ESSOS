"""Terminal event refinement for forward RK solves.

The driver stops on the first acceptable endpoint crossing. Its RK stages
locate the event with the ordinary interpolant, outside the trial loop. An
invalid root rejects the pending trial and resumes that same adaptive loop.
"""

import jax.numpy as jnp
from jax import lax


def finish_event_root(state, *, args, functions, events, interpolate, ts,
                      record_steps, adaptive, h0, dtmin, dtmax, controller):
    """Refine the terminal crossing; return state and root validity.

    Crossings have not written any saves yet. A valid root writes saves only
    through the loss time. A nonfinite root rolls back the pending acceptance
    and its PI history, retaining prior saves and the original retry proposal.
    Callers resume their existing trial loop if validity is false and work
    can continue. No RHS replay or separately compiled fallback is needed.
    """
    from .rk import RKStatus, _first_hit, _RKTrial

    def locate(carry):
        _, state, _ = carry
        t, y, h = state["last_t"], state["last_y"], state["last_h"]
        trial = _RKTrial(state["y"], jnp.zeros_like(y), state["last_increments"], state["f"])
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
        hits = _first_hit(tuple(hit & (value == fraction) & valid
                               for hit, value in zip(state["last_triggered"], fractions)))
        next_h = jnp.clip(.2*h if adaptive else h0, dtmin, dtmax)
        stuck = (not adaptive) | (h <= dtmin) | (t + next_h <= t)
        status = jnp.where(valid, RKStatus.EVENT,
                           jnp.where(stuck, RKStatus.NONFINITE, RKStatus.SUCCESS))
        old_count = state["accepted"]
        state = dict(state, t=jnp.where(valid, stop_t, t), y=jnp.where(valid, stop_y, y),
            f=jnp.where(valid, state["f"], state["last_f"]),
            h=jnp.where(valid, state["h"], next_h), hits=hits, status=status,
            accepted=old_count-(~valid).astype(old_count.dtype),
            rejected=state["rejected"]+(~valid).astype(state["rejected"].dtype),
            values=tuple(jnp.where(valid, new, old) for new, old in zip(state["values"], values)))
        if adaptive and controller == "pi":
            state = dict(state, previous_error=jnp.where(valid,
                state["previous_error"], state["last_previous_error"]))
        if record_steps:
            state = dict(state, history=state["history"].at[old_count].set(
                jnp.where(valid, stop_t, jnp.inf)))
        if len(ts):
            def save(state):
                index = state["index"]
                value = interpolate(y, h, trial, (ts[index]-t)/h)
                return dict(state, ys=state["ys"].at[index].set(value),
                    saved_ts=state["saved_ts"].at[index].set(ts[index]), index=index+1)
            state = lax.while_loop(lambda s: valid & (s["index"] < len(ts)) &
                (ts[jnp.minimum(s["index"], len(ts)-1)] <= stop_t), save, state)
        return jnp.asarray(False), state, valid

    crossing = (state["status"] == RKStatus.EVENT) & (state["accepted"] > 0)
    _, state, valid = lax.while_loop(lambda carry: carry[0], locate,
                                    (crossing, state, jnp.asarray(True)))
    return state, valid
