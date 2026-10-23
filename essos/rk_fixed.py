"""Fixed Dopri5/8 execution for a validated, aligned, uniform save grid.

This forward helper shares rk.py's tableau, FSAL stages and continuous
extension. It checks endpoints each step, but locates a terminal crossing
after stepping finishes. Only that last step is replayed for the root search;
there is no trajectory-sized stage tape. Unaligned saves use solve_rk instead.
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from typing import NamedTuple

from .rk import (RKSolution, RKStats, RKStatus, _prepare_rhs, _select_trial_step,
                 _component_trial, _C, _A, _ERROR, _rk45_dense, _C8, _A8, _ERROR8, _DENSE8)


def aligned_substeps(horizon, timestep, saves, max_steps):
    """Return a static subdivision only when it preserves the requested step."""
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        return None
    ratio = horizon / ((saves - 1) * timestep)
    if not np.isfinite(ratio) or ratio < 1 - 8 * np.finfo(float).eps or ratio > max_steps:
        return None
    count = int(round(ratio))
    if count * (saves - 1) > max_steps:
        return None
    return count if abs(ratio - count) <= 8 * np.finfo(float).eps * max(ratio, 1.) else None


def _dense(y0, y1, increments, fraction):
    # The standard polynomial, using an elementwise reduction at the one
    # terminal root rather than introducing a batched matrix product.
    coefficients = jnp.asarray(_DENSE8, y0.dtype)
    weights = jax.vmap(jnp.polyval, in_axes=(0, None))(coefficients, fraction) * fraction
    value = y0 + jnp.sum(weights[:, None] * increments, axis=0)
    return jnp.where(fraction == 0, y0, jnp.where(fraction == 1, y1, value))


class _CheckedEndpoint(NamedTuple):
    y: tuple
    f1: tuple
    finite: jax.Array


def _component_step(rhs, t, y, h, args, f0, nodes=_C8, rows=_A8, error_weights=_ERROR8):
    """The RK recurrence on scalar components, with no validation stage bank.

    The RHS still receives its ordinary state vector. Keep components separate
    between evaluations so the GPU can fuse stage arithmetic and finite checks.
    Only root location needs a stacked bank; it replays the terminal step below.
    The error estimate and every increment retain their finite checks.
    """
    trial = _component_trial(rhs, t, y, h, args, f0, nodes, rows, error_weights)
    finite = jnp.asarray(True)
    for vector in (trial.y, trial.error, trial.f1, *trial.increments):
        for value in vector:
            finite = finite & jnp.isfinite(value)
    return _CheckedEndpoint(trial.y, trial.f1, finite)


def solve_fixed_grid(rhs, y0, args=None, *, ts, dt0, substeps, event,
                     stage_execution="loop", method="dopri8"):
    """Validated uniform grid; one scalar level-set event, forward only.

    The caller checks alignment and the work limit before selecting this path.
    Every accepted endpoint and all RK stages retain the normal finite checks.
    Losses stop each particle immediately; subsequent fixed-loop iterations
    leave its state and counters unchanged. Saved values after termination are
    infinity-padded, just as in solve_rk. Bisection failures roll back the step.
    """
    if method == "dopri5":
        nodes, rows, error_weights, dense = _C, _A, _ERROR, _rk45_dense
    elif method == "dopri8":
        nodes, rows, error_weights, dense = _C8, _A8, _ERROR8, _dense
    else:
        raise ValueError("method must be 'dopri5' or 'dopri8'")
    rhs = _prepare_rhs(rhs, 0., y0, args, coalesce_views=True)
    trial_step = _select_trial_step(stage_execution)
    components = stage_execution == "unrolled" and y0.ndim == 1
    pack = tuple if components else lambda value: value
    unpack = jnp.stack if components else lambda value: value
    t1 = ts[-1]
    t0, zero = jnp.zeros((), y0.dtype), jnp.asarray(0)
    value0 = event(t0, y0, args)
    hit0 = value0 == 0
    initial_status = jnp.where(jnp.isfinite(y0).all() & jnp.isfinite(value0),
        jnp.where(hit0, RKStatus.EVENT, RKStatus.SUCCESS), RKStatus.NONFINITE)
    f0 = pack(rhs(t0, y0, args))
    state = dict(y=pack(y0), f=f0, t=t0, value=value0, status=initial_status,
        steps=zero, accepted=zero, rejected=zero, old_y=pack(y0), old_t=t0,
        h=jnp.asarray(dt0, y0.dtype), old_f=f0)

    def interval(carry, i):
        def step(j, s):
            active = s["status"] == RKStatus.SUCCESS
            end = jnp.minimum((i * substeps + j + 1) * dt0, t1)
            end = jnp.where(t1 - end <= 8 * jnp.finfo(y0.dtype).eps *
                jnp.maximum(jnp.abs(t1), jnp.abs(s["t"])), t1, end)
            h = end - s["t"]
            if components:
                trial = _component_step(rhs, s["t"], s["y"], h, args, s["f"],
                    nodes, rows, error_weights)
                finite = trial.finite
            else:
                trial = trial_step(rhs, s["t"], s["y"], h, args, s["f"], nodes, rows, error_weights)
                finite = (jnp.isfinite(trial.y).all() & jnp.isfinite(trial.error).all() &
                    jnp.isfinite(trial.increments).all() & jnp.isfinite(trial.f1).all())
            value = event(end, unpack(trial.y), args)
            finite = finite & jnp.isfinite(value)
            accept = active & finite & (end > s["t"])
            hit = accept & (((value > 0) != (s["value"] > 0)) | (value == 0))
            status = jnp.where(~finite, RKStatus.NONFINITE,
                jnp.where(end <= s["t"], RKStatus.MIN_STEP,
                    jnp.where(hit, RKStatus.EVENT, RKStatus.SUCCESS)))

            def choose(new, old):
                return jax.tree_util.tree_map(lambda n, o: jnp.where(accept, n, o), new, old)

            return dict(y=choose(trial.y, s["y"]), f=choose(trial.f1, s["f"]),
                t=choose(end, s["t"]), value=choose(value, s["value"]),
                status=jnp.where(active, status, s["status"]), steps=s["steps"] + active,
                accepted=s["accepted"] + accept, rejected=s["rejected"] + (active & ~accept),
                old_y=choose(s["y"], s["old_y"]), old_t=choose(s["t"], s["old_t"]),
                h=choose(h, s["h"]), old_f=choose(s["f"], s["old_f"]))

        def run(local):
            return jnp.asarray(False), lax.fori_loop(0, substeps, step, local[1])

        # This guard skips an entire save interval once all lanes have stopped.
        _, carry = lax.while_loop(
            lambda local: local[0] & (local[1]["status"] == RKStatus.SUCCESS),
            run, (jnp.asarray(True), carry))
        return carry, unpack(carry["y"])

    state, ys = lax.scan(interval, state, jnp.arange(len(ts) - 1))
    crossing = (state["status"] == RKStatus.EVENT) & (state["accepted"] > 0)

    def locate(local):
        old_y = unpack(state["old_y"])
        trial = trial_step(rhs, state["old_t"], old_y, state["h"], args,
            unpack(state["old_f"]), nodes, rows, error_weights)
        old_value = event(state["old_t"], old_y, args)

        def bisect(c):
            lo, hi, iteration, valid = c
            mid = (lo + hi) / 2
            y = dense(old_y, trial.y, trial.increments, mid)
            value = event(state["old_t"] + mid * state["h"], y, args)
            same = (value > 0) == (old_value > 0)
            return (jnp.where(same, mid, lo), jnp.where(same, hi, mid),
                    iteration + 1, valid & jnp.isfinite(value))

        lo, hi, _, valid = lax.while_loop(lambda c: crossing & c[3] & (c[2] < 52), bisect,
            (t0, jnp.ones_like(t0), zero, jnp.asarray(True)))
        fraction = (lo + hi) / 2
        terminal = dense(old_y, trial.y, trial.increments, fraction)
        return jnp.asarray(False), terminal, state["old_t"] + fraction * state["h"], valid

    _, terminal, end, valid = lax.while_loop(lambda c: c[0], locate,
        (crossing, unpack(state["y"]), state["t"], jnp.asarray(True)))
    invalid_root = crossing & ~valid
    terminal = jnp.where(crossing & valid, terminal, unpack(state["y"]))
    end = jnp.where(crossing & valid, end, state["t"])
    terminal = jnp.where(invalid_root, unpack(state["old_y"]), terminal)
    end = jnp.where(invalid_root, state["old_t"], end)
    status = jnp.where(invalid_root, RKStatus.NONFINITE, state["status"])
    ys = jnp.concatenate((y0[None], ys))
    reached = (ts <= end) & ~hit0
    ys = jnp.where((crossing & (ts == end))[:, None], terminal, ys)
    ys = jnp.where(reached[:, None], ys, jnp.inf)
    return RKSolution(jnp.where(reached, ts, jnp.inf), ys, end, terminal, status,
        status == RKStatus.EVENT, RKStats(state["steps"], state["accepted"] - invalid_root,
        state["rejected"] + invalid_root), jnp.empty((0,), y0.dtype), dt0, jnp.zeros_like(y0))
