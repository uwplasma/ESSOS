"""Forward orbit/collision splitting, independent of Diffrax's SDE machinery.

``solve_split`` composes an adaptive native RK orbit solve with a separate
collision map. ``trace_split`` supplies the existing ESSOS drift and background
coefficients for Cartesian, VMEC and Boozer fields. Accepted-step reverse AD
is opt-in, with its checkpointed replay isolated in :mod:`essos.split_ad`.
"""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util

from .collisions import (CollisionStep, brownian_memory, brownian_propose, brownian_accept,
                         brownian_sample, brownian_consume, paper_collision_step_size,
                         collision_coefficients, guiding_center_collision_step, particle_collision_step,
                         guiding_center_spatial_increment)
from .rk import RKStatus, solve_rk, _concrete_check
from .collision_error import CollisionErrorModel
from .collision_controller import candidate_collision_step_size


class SplitStatus(IntEnum):
    SUCCESS = 0
    EVENT = 1
    MAX_STEPS = 2
    MIN_STEP = 3
    NONFINITE = 4
    INVALID_INPUT = 5
    BROWNIAN_CAPACITY = 6


class SplitStats(NamedTuple):
    steps: jax.Array
    accepted: jax.Array
    rejected: jax.Array
    orbit_steps: jax.Array
    orbit_rejected: jax.Array
    peak_pending: jax.Array
    max_accepted_error: jax.Array


class SplitSolution(NamedTuple):
    ts: jax.Array
    ys: jax.Array
    t: jax.Array
    y: jax.Array
    status: jax.Array
    stats: SplitStats
    step_ts: jax.Array
    event_tangent: jax.Array


class _SplitCheckpointTape(NamedTuple):
    times: jax.Array
    increments: jax.Array
    kicks: jax.Array
    offsets: jax.Array
    orbit_times: jax.Array
    fractions: jax.Array
    states: jax.Array
    count: jax.Array
    orbit_count: jax.Array
    initial_saves: jax.Array
    saved_count: jax.Array
    status: jax.Array


def solve_split(rhs, collision, y0, args=None, *, t0=0., t1, dt0, ts, key,
                noise_dim, adaptive=True, dtmin=0., dtmax=jnp.inf,
                method="dopri8", orbit_dt0=None, orbit_adaptive=True, reuse_orbit_step=True,
                collision_controller="conservative", collision_wiener_norm=None,
                rtol=1e-9, atol=1e-9, error_norm=None, controller="i",
                pcoeff=.4, icoeff=.3, event=None, max_steps=100_000,
                max_orbit_steps=10_000, max_pending=64, record_steps=False,
                step_limit=None, event_dtmax=None, adjoint="none",
                adjoint_checkpoint_steps=16, adjoint_orbit_checkpoint_steps=16,
                adjoint_max_accepted_steps=None, adjoint_max_orbit_steps=None,
                _record_ad=False, record_event_tangent=False):
    """Lie split: orbit(t, t+h), then collisions over h at the orbit endpoint.

    ``collision(t, y, h, dW, args)`` returns ``CollisionStep`` with normalized
    drift/diffusion estimates. Rejected trials repeat from the last accepted
    state and retain all Brownian samples. The conservative controller uses
    factors 0.9*min(E_drift**(-1/2), E_diff**(-2/3)), capped at 1.5.
    ``collision_controller="paper"`` selects section 5.3's Wiener look-ahead
    (Eqs. 78--82), retaining every sampled candidate including failed probes.
    ``collision_controller="candidate"`` predicts both errors using frozen
    local coefficients, then probes/backtracks on retained Wiener candidates.
    It requires ``CollisionStep.error_model`` and still checks actual errors
    for acceptance. The native collision kernels supply this metadata without
    extra runtime coefficient evaluations.
    ``collision_wiener_norm`` selects the Wiener components that enter the
    collision error estimate; by default it is the Euclidean norm of all noise.

    ``dt0``/``dtmax`` control collision/splitting intervals; ``orbit_dt0`` and
    rtol/atol control the independent RK solve within each interval. Collision
    estimates do not bound operator-splitting error: refine dtmax to check it.
    Saves are exact split boundaries, so changing ts can change the trajectory.
    No ODE dense interpolation is applied to a stochastic trajectory.
    ``reuse_orbit_step`` carries the next adaptive RK step-size guess across
    kicks and retries. Every solve recomputes its initial derivative; no FSAL
    derivative or PI error history is carried across a collision.

    Events are scalar booleans or positive-inside level sets (or their PyTrees).
    Orbit crossings are root-found before a kick. A kick crossing stops at its
    endpoint; Brownian crossings are not root-found by ODE interpolation.
    Optional ``step_limit(t, y, args)`` supplies a positive interval ceiling
    from the current accepted state. Optional ``event_dtmax`` retries a detected
    kick crossing until its interval is this small, retaining Brownian samples.
    The two controls support boundary-resolution diagnostics; they do not detect
    every crossing followed by a return, or certify an exact first-passage time.
    Neither is enabled by default. Step limits also subdivide fixed grids while
    retaining their original grid anchors. They must not depend on fresh noise.
    Stopped/failed outputs hold the last finite state; always inspect status.
    Work and Brownian memory are bounded. No inaccurate step is forced through
    dtmin or a memory limit.

    ``adjoint="accepted"`` differentiates accepted orbit and collision maps.
    Accepted outer/inner times, Wiener increments, controllers and event
    fractions are frozen; physical collision coefficients remain active.
    Segment checkpoints bound state storage. ``adjoint_max_accepted_steps``
    bounds outer-map storage; ``adjoint_max_orbit_steps`` bounds the total
    inner accepted schedule across committed maps, not a per-interval limit.
    Both default to max_steps. A full tape leaves the forward result unchanged
    and raises during reverse AD. The existing forward path is the default.
    ``record_event_tangent`` propagates the inner RK event's interpolant time
    derivative for the separate first-loss objective. Kick-event tangents are
    NaN: an instantaneous spatial kick has no continuous crossing-time rule.
    """
    for name, value in (("max_steps", max_steps), ("max_orbit_steps", max_orbit_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if adjoint not in ("none", "accepted"):
        raise ValueError("adjoint must be 'none' or 'accepted'")
    for name, value in (("adjoint_checkpoint_steps", adjoint_checkpoint_steps),
                        ("adjoint_orbit_checkpoint_steps", adjoint_orbit_checkpoint_steps),
                        ("adjoint_max_accepted_steps", adjoint_max_accepted_steps),
                        ("adjoint_max_orbit_steps", adjoint_max_orbit_steps)):
        if value is None and name.startswith("adjoint_max_"):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer" + (" or None" if name.startswith("adjoint_max_") else ""))
    if adjoint == "accepted":
        from .split_ad import solve_accepted_split
        return solve_accepted_split(rhs, collision, y0, args, t0=t0, t1=t1, dt0=dt0,
            ts=ts, key=key, noise_dim=noise_dim, adaptive=adaptive, dtmin=dtmin, dtmax=dtmax,
            method=method, orbit_dt0=orbit_dt0, orbit_adaptive=orbit_adaptive,
            reuse_orbit_step=reuse_orbit_step, collision_controller=collision_controller,
            collision_wiener_norm=collision_wiener_norm, rtol=rtol, atol=atol,
            error_norm=error_norm, controller=controller, pcoeff=pcoeff, icoeff=icoeff,
            event=event, max_steps=max_steps, max_orbit_steps=max_orbit_steps,
            max_pending=max_pending, record_steps=record_steps, step_limit=step_limit,
            event_dtmax=event_dtmax, adjoint_checkpoint_steps=adjoint_checkpoint_steps,
            adjoint_orbit_checkpoint_steps=adjoint_orbit_checkpoint_steps,
            adjoint_max_accepted_steps=adjoint_max_accepted_steps,
            adjoint_max_orbit_steps=adjoint_max_orbit_steps,
            record_event_tangent=record_event_tangent)
    if collision_controller not in ("conservative", "paper", "candidate"):
        raise ValueError("collision_controller must be 'conservative', 'paper' or 'candidate'")
    candidate = adaptive and collision_controller == "candidate"
    lookahead = adaptive and collision_controller in ("paper", "candidate")
    y0 = jnp.asarray(y0, dtype=jnp.result_type(y0, float))
    ts = jnp.asarray(ts, dtype=y0.dtype)
    if y0.ndim != 1 or y0.size == 0 or not jnp.issubdtype(y0.dtype, jnp.floating):
        raise ValueError("y0 must be a nonempty real vector")
    if ts.ndim != 1:
        raise ValueError("ts must be one-dimensional")
    orbit_dt0 = dt0 if orbit_dt0 is None else orbit_dt0
    for name, value in (("t0", t0), ("t1", t1), ("dt0", dt0), ("dtmin", dtmin),
                        ("dtmax", dtmax), ("orbit_dt0", orbit_dt0)):
        if jnp.ndim(value) != 0:
            raise ValueError(f"{name} must be scalar")
    t0, t1, dt0, dtmin, dtmax, orbit_dt0 = (
        jnp.asarray(x, dtype=y0.dtype) for x in (t0, t1, dt0, dtmin, dtmax, orbit_dt0))
    if event_dtmax is not None:
        if event is None or jnp.ndim(event_dtmax) != 0:
            raise ValueError("event_dtmax requires an event and a scalar interval")
        event_dtmax = jnp.asarray(event_dtmax, dtype=y0.dtype)
    valid = (jnp.isfinite(t0) & jnp.isfinite(t1) & (t1 >= t0) & jnp.isfinite(dt0) & (dt0 > 0)
        & jnp.isfinite(dtmin) & (dtmin >= 0) & (dt0 >= dtmin) & (dtmax > 0) & (dtmax >= dtmin)
        & jnp.isfinite(orbit_dt0) & (orbit_dt0 > 0)
        & jnp.all(jnp.isfinite(ts) & (ts >= t0) & (ts <= t1)) & jnp.all(jnp.diff(ts) >= 0))
    if event_dtmax is not None:
        valid &= jnp.isfinite(event_dtmax) & (event_dtmax > 0) & (event_dtmax >= dtmin)
    _concrete_check(valid, np.all, "invalid split times or step limits")
    # Trace the callback's output schema, without executing a second kick, so
    # the skipped-kick branch can return matching metadata on orbit events.
    empty_model = None
    if candidate:
        schema = jax.eval_shape(collision, t0, y0, dt0, jnp.zeros((noise_dim,), y0.dtype), args)
        if not isinstance(schema.error_model, CollisionErrorModel):
            raise ValueError("candidate controller requires CollisionStep.error_model")
        if any(x.shape != () for x in tree_util.tree_leaves(schema.error_model)):
            raise ValueError("collision error model coefficients must be scalars")
        empty_model = tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), schema.error_model)
        empty_model = empty_model._replace(tolerance=jnp.ones_like(empty_model.tolerance))
    functions = tree_util.tree_leaves(event)

    def values(t, y):
        out = tuple(jnp.asarray(f(t, y, args)) for f in functions)
        if any(x.shape != () or (x.dtype != jnp.bool_ and not jnp.issubdtype(x.dtype, jnp.floating)) for x in out):
            raise ValueError("events must return scalar booleans or floating level sets")
        return out

    def stopped(v):
        return jnp.any(jnp.asarray([x if x.dtype == jnp.bool_ else x <= 0 for x in v], dtype=bool))

    def finite_values(v):
        return jnp.all(jnp.asarray([jnp.isfinite(x) for x in v], dtype=bool))

    v0 = values(t0, y0)
    initial_status = jnp.where(~valid, SplitStatus.INVALID_INPUT,
        jnp.where(~jnp.isfinite(y0).all() | ~finite_values(v0), SplitStatus.NONFINITE,
                  jnp.where(stopped(v0), SplitStatus.EVENT, SplitStatus.SUCCESS)))
    memory = brownian_memory(key, noise_dim, capacity=max_pending, dtype=y0.dtype, t0=t0)
    s = dict(t=t0, y=y0, h=jnp.minimum(dt0, dtmax), orbit_h=orbit_dt0, status=initial_status,
        memory=memory, index=jnp.array(0), ys=jnp.broadcast_to(y0, (len(ts), y0.size)),
        steps=jnp.array(0), accepted=jnp.array(0), rejected=jnp.array(0),
        orbit_steps=jnp.array(0), orbit_rejected=jnp.array(0), max_error=jnp.zeros((), y0.dtype),
        history=jnp.full((max_steps+1 if record_steps else 0,), jnp.inf, dtype=y0.dtype))
    if record_steps:
        s["history"] = s["history"].at[0].set(t0)
    if _record_ad:
        outer_capacity = min(max_steps, adjoint_max_accepted_steps or max_steps)
        inner_capacity = adjoint_max_orbit_steps or max_steps
        checkpoints = (outer_capacity + adjoint_checkpoint_steps - 1) // adjoint_checkpoint_steps
        s.update(ad_times=jnp.zeros((outer_capacity, 2), y0.dtype),
            ad_dw=jnp.zeros((outer_capacity, noise_dim), y0.dtype),
            ad_kicks=jnp.zeros((outer_capacity,), bool),
            ad_offsets=jnp.zeros((outer_capacity+1,), jnp.int32),
            ad_orbit_times=jnp.zeros((inner_capacity, 2), y0.dtype),
            ad_fractions=jnp.ones((outer_capacity,), y0.dtype),
            ad_states=jnp.zeros((checkpoints, y0.size), y0.dtype),
            ad_count=jnp.array(0), ad_orbit_count=jnp.array(0))
    if record_event_tangent:
        s["event_tangent"] = jnp.zeros_like(y0)
    if not adaptive:
        s["grid_index"] = jnp.array(0)
    if event_dtmax is not None:
        s["event_h"] = jnp.asarray(jnp.inf, dtype=y0.dtype)

    def save(s):
        if not len(ts):
            return s
        def one(s):
            return dict(s, ys=s["ys"].at[s["index"]].set(s["y"]), index=s["index"]+1)
        return lax.while_loop(lambda s: (s["index"] < len(ts)) &
            (ts[jnp.minimum(s["index"], len(ts)-1)] <= s["t"]), one, s)

    s = save(s)
    initial_saves = s["index"]

    def advance_with_limit(s, limit):
        barrier = t1 if not len(ts) else jnp.where(s["index"] < len(ts),
            ts[jnp.minimum(s["index"], len(ts)-1)], t1)
        if step_limit is not None:
            barrier = jnp.minimum(barrier, s["t"]+limit)
        if event_dtmax is not None:
            barrier = jnp.minimum(barrier, s["t"]+s["event_h"])
        if adaptive:
            wanted_end = jnp.minimum(s["t"]+s["h"], barrier)
        else:
            # Anchor fixed times to the initial grid, rather than repeatedly
            # adding h. Saves inside a step add a split boundary without
            # shifting later grid points. Coincident saves/grid points can
            # differ by one ulp; choose the exact save before drawing noise.
            grid_end = t0+(s["grid_index"]+1)*jnp.minimum(dt0, dtmax)
            close = jnp.abs(grid_end-barrier) <= 8*jnp.finfo(y0.dtype).eps*jnp.maximum(
                jnp.abs(grid_end), jnp.abs(barrier))
            grid_end = jnp.where(close, barrier, grid_end)
            wanted_end = jnp.minimum(grid_end, barrier)
        if lookahead:
            m, dw, overflow = brownian_sample(s["memory"], wanted_end)
            end, h = wanted_end, wanted_end-s["t"]
        else:
            m, h, dw, overflow = brownian_propose(s["memory"], wanted_end-s["t"], t_end=wanted_end)
            end = m.ends[m.size-1]
        usable = ~overflow & (h > 0) & (end > s["t"])

        def attempt(_):
            orbit_out = solve_rk(rhs, s["y"], args, t0=s["t"], t1=end,
                dt0=jnp.minimum(s["orbit_h"] if reuse_orbit_step and orbit_adaptive else orbit_dt0, h),
                ts=jnp.empty((0,), y0.dtype),
                method=method, adaptive=orbit_adaptive, rtol=rtol, atol=atol,
                error_norm=error_norm, controller=controller, pcoeff=pcoeff, icoeff=icoeff,
                event=event, root_find=True, max_steps=max_orbit_steps,
                _record_schedule=_record_ad, record_event_tangent=record_event_tangent)
            if _record_ad:
                orbit, schedule = orbit_out
            else:
                orbit = orbit_out

            def kick(_):
                result = collision(end, orbit.y, h, dw, args)
                if result.y.shape != y0.shape:
                    raise ValueError("collision must preserve the state shape")
                if jnp.ndim(result.drift_error) or jnp.ndim(result.diffusion_error):
                    raise ValueError("collision errors must be scalars")
                return result if candidate else result._replace(error_model=None)
            trial = lax.cond(orbit.status == RKStatus.SUCCESS, kick,
                             lambda _: CollisionStep(orbit.y, jnp.zeros((), y0.dtype),
                                                      jnp.zeros((), y0.dtype), empty_model), None)
            v = values(end, trial.y)
            finite = (jnp.isfinite(trial.y).all() & finite_values(v)
                & jnp.isfinite(trial.drift_error) & (trial.drift_error >= 0)
                & jnp.isfinite(trial.diffusion_error) & (trial.diffusion_error >= 0))
            error = jnp.maximum(trial.drift_error, trial.diffusion_error)
            accepted = (orbit.status == RKStatus.SUCCESS) & finite & ((error <= 1) | (not adaptive))
            refine_event = jnp.array(False)
            if event_dtmax is not None:
                refine_event = accepted & stopped(v) & (h > event_dtmax*(1+8*jnp.finfo(y0.dtype).eps))
                accepted &= ~refine_event
            rejected = (orbit.status == RKStatus.SUCCESS) & finite & ~accepted
            orbit_event = orbit.status == RKStatus.EVENT
            status = jnp.where(orbit.status != RKStatus.SUCCESS, orbit.status,
                jnp.where(~finite, SplitStatus.NONFINITE,
                    jnp.where(accepted & stopped(v), SplitStatus.EVENT,
                        jnp.where(rejected & (h <= dtmin), SplitStatus.MIN_STEP, SplitStatus.SUCCESS))))
            if adaptive and not lookahead:
                factor = .9*jnp.minimum(jnp.maximum(trial.drift_error, 1e-30)**(-.5),
                                        jnp.maximum(trial.diffusion_error, 1e-30)**(-2/3))
                factor = jnp.clip(factor, .1, jnp.where(accepted, 1.5, .9))
                next_h = jnp.clip(h*factor, dtmin, dtmax)
            else:
                # Look-ahead supplies its own proposal below. The placeholder
                # is used only if this solve terminates before another trial.
                next_h = jnp.minimum(dt0, dtmax)
            state = dict(s, t=jnp.where(orbit_event, orbit.t, jnp.where(accepted, end, s["t"])),
                y=jnp.where(orbit_event, orbit.y, jnp.where(accepted, trial.y, s["y"])),
                h=next_h, status=status, steps=s["steps"]+1,
                accepted=s["accepted"]+accepted.astype(s["accepted"].dtype),
                rejected=s["rejected"]+rejected.astype(s["rejected"].dtype),
                orbit_steps=s["orbit_steps"]+orbit.stats.steps,
                orbit_rejected=s["orbit_rejected"]+orbit.stats.rejected,
                memory=lax.cond(accepted,
                    (lambda m: brownian_consume(m, end)) if lookahead else brownian_accept, lambda m: m, m),
                max_error=jnp.maximum(s["max_error"], jnp.where(accepted, error, 0.)))
            if record_event_tangent:
                state["event_tangent"] = jnp.where(orbit_event, orbit.event_tangent,
                    jnp.where(accepted & stopped(v), jnp.full_like(y0, jnp.nan), s["event_tangent"]))
            if event_dtmax is not None:
                state["event_h"] = jnp.where(refine_event, jnp.maximum(dtmin, .5*h), jnp.inf)
                state["h"] = jnp.where(refine_event, state["event_h"], state["h"])
            if reuse_orbit_step and orbit_adaptive:
                state["orbit_h"] = orbit.next_dt
            if not adaptive:
                state["grid_index"] = s["grid_index"]+(accepted & (end == grid_end)).astype(s["grid_index"].dtype)
            if record_steps:
                state["history"] = lax.cond(accepted,
                    lambda history: history.at[state["accepted"]].set(end), lambda history: history,
                    state["history"])
            state = save(state)
            if _record_ad:
                committed = accepted | orbit_event
                i, k = s["ad_count"], s["ad_orbit_count"]
                def record(state):
                    def checkpoint(states):
                        return states.at[i // adjoint_checkpoint_steps].set(s["y"])
                    states = lax.cond(i % adjoint_checkpoint_steps == 0,
                        checkpoint, lambda states: states, state["ad_states"])
                    orbit_times = lax.fori_loop(0, schedule.count,
                        lambda j, rows: rows.at[k+j].set(schedule.times[j]), state["ad_orbit_times"])
                    return dict(state,
                        ad_times=state["ad_times"].at[i].set(jnp.stack([s["t"], state["t"]])),
                        ad_dw=state["ad_dw"].at[i].set(dw),
                        ad_kicks=state["ad_kicks"].at[i].set(accepted),
                        ad_offsets=state["ad_offsets"].at[i+1].set((k+schedule.count).astype(state["ad_offsets"].dtype)),
                        ad_orbit_times=orbit_times,
                        ad_fractions=state["ad_fractions"].at[i].set(schedule.fraction),
                        ad_states=states)
                can_record = committed & (i < outer_capacity) & (k+schedule.count <= inner_capacity)
                state = lax.cond(can_record, record, lambda state: state, state)
                state = dict(state, ad_count=i+committed.astype(i.dtype),
                    ad_orbit_count=k+jnp.where(committed, schedule.count, 0))
            if lookahead:
                def choose(state):
                    barrier = t1 if not len(ts) else jnp.where(state["index"] < len(ts),
                        ts[jnp.minimum(state["index"], len(ts)-1)], t1)
                    if candidate:
                        memory, next_h, overflow = candidate_collision_step_size(
                            state["memory"], h, trial.error_model, accepted,
                            dtmin=dtmin, dtmax=dtmax, barrier=barrier)
                    else:
                        memory, next_h, overflow = paper_collision_step_size(
                            state["memory"], h, dw, trial.drift_error, trial.diffusion_error,
                            accepted, dtmin=dtmin, dtmax=dtmax, barrier=barrier,
                            wiener_norm=collision_wiener_norm)
                    return dict(state, memory=memory, h=next_h,
                        status=jnp.where(overflow, SplitStatus.BROWNIAN_CAPACITY, state["status"]))
                state = lax.cond((state["status"] == SplitStatus.SUCCESS) & (state["t"] < t1) & ~refine_event,
                                 choose, lambda state: state, state)
            return state

        return lax.cond(usable, attempt,
            lambda _: dict(s, memory=m, status=jnp.where(overflow, SplitStatus.BROWNIAN_CAPACITY,
                                                        SplitStatus.MIN_STEP)), None)

    def advance(s):
        if step_limit is None:
            return advance_with_limit(s, None)
        limit = jnp.asarray(step_limit(s["t"], s["y"], args), dtype=y0.dtype)
        if limit.ndim != 0:
            raise ValueError("step_limit must return a scalar interval")
        return lax.cond((limit > 0) & ~jnp.isnan(limit),
            lambda _: advance_with_limit(s, limit),
            lambda _: dict(s, status=jnp.asarray(SplitStatus.INVALID_INPUT)), None)

    s = lax.while_loop(lambda s: (s["status"] == SplitStatus.SUCCESS) & (s["t"] < t1)
                       & (s["steps"] < max_steps), advance, s)
    status = jnp.where((s["status"] == SplitStatus.SUCCESS) & (s["t"] < t1),
                        SplitStatus.MAX_STEPS, s["status"])
    ys = jnp.where((jnp.arange(len(ts)) >= s["index"])[:, None], s["y"], s["ys"])
    solution = SplitSolution(ts, ys, s["t"], s["y"], status,
        SplitStats(s["steps"], s["accepted"], s["rejected"], s["orbit_steps"], s["orbit_rejected"],
                    s["memory"].peak, s["max_error"]), s["history"],
        s["event_tangent"] if record_event_tangent else jnp.zeros_like(y0))
    if _record_ad:
        return solution, _SplitCheckpointTape(s["ad_times"], s["ad_dw"], s["ad_kicks"],
            s["ad_offsets"], s["ad_orbit_times"], s["ad_fractions"], s["ad_states"],
            s["ad_count"], s["ad_orbit_count"], initial_saves, s["index"], status)
    return solution


def trace_split(field, particles, species, *, model="GuidingCenter", tmax, timestep,
                times=None, n_save=100, integrator="essos_dopri8", collision_tolerance=1e-3,
                adaptive=True, collision_dtmax=None, orbit_dt0=None, orbit_adaptive=True,
                reuse_orbit_step=True, collision_controller="conservative",
                rtol=1e-9, atol=1e-9, controller="i", pcoeff=.4, icoeff=.3,
                speed_floor=None, spatial_diffusion=False, orbit=True, electric_field=None,
                event=None, max_steps=100_000, max_orbit_steps=10_000, max_pending=64,
                record_steps=False, adjoint="none", adjoint_checkpoint_steps=16,
                adjoint_orbit_checkpoint_steps=16, adjoint_max_accepted_steps=None,
                adjoint_max_orbit_steps=None, _record_ad=False, _replay_tapes=None,
                record_event_tangent=False):
    """Batched forward split tracing using ESSOS field and particle objects.

    Returns a ``SplitSolution`` with shared ts and particle-leading arrays. GC states are
    ``(position[3], v_parallel, mu)`` with **physical** mu=m*v_perp^2/(2 B)
    [J/T], unlike the legacy normalized collisional state. FullOrbit states
    are ``(xyz, vx, vy, vz)``. VMEC/Boozer positions are (s, theta, zeta).
    VMEC uses its regular axis chart; Boozer uses (sqrt(s)cos(theta),
    sqrt(s)sin(theta)) internally. Both stop at the LCFS. Exterior continuation
    is not part of this entry point. ``event`` receives public physical states.

    ``spatial_diffusion=True`` adds paper Eq. (69) for Cartesian GC fields;
    coordinate-field diffusion needs a separate coordinate transformation and
    is rejected here. By default the kick acts on velocity only, like ESSOS's
    existing mu collision path. ``orbit=False`` runs collision-only statistics.
    ``speed_floor`` defaults to 5% of the smallest active background thermal
    speed for the test mass. The GC reflection wall affects near-zero speeds;
    refine/check that parameter when studying the thermal population.

    Independent per-particle keys come from ``particles.random_keys``. The
    field classes and existing Tracing/trace_boozer entry points are unchanged.
    ``adjoint="accepted"`` enables checkpointed pathwise reverse AD for Cartesian
    GC/Lorentz. VMEC/Boozer reverse AD is deferred. See solve_split for capacities.

    Private Cartesian audit hooks leave the default path unchanged:
    ``_record_ad=True`` returns ``(solution, tapes)`` from the production solve;
    ``_replay_tapes=tapes`` returns ``(saved_states, terminal_states)`` at new
    physical parameters with the recorded grids/noise and no adaptation.
    Both require ``adjoint="none"`` and are used for independent JVP/FD checks.
    """
    from .boozer import BoozerField, guiding_center_rhs
    from .coils import Coils
    from .fields import BiotSavart, Vmec
    from .dynamics import _guiding_center_velocity, _axis_regular, _to_axis_regular, _from_axis_regular
    from .electric_field import Electric_field_zero
    from .constants import ELEMENTARY_CHARGE
    from .orbit_error import OrbitErrorNorm

    if integrator not in ("essos_rk45", "essos_dopri8"):
        raise ValueError("split integrator must be 'essos_rk45' or 'essos_dopri8'")
    if model not in ("GuidingCenter", "FullOrbit"):
        raise ValueError("split model must be 'GuidingCenter' or 'FullOrbit'")
    _concrete_check(collision_tolerance, lambda x: np.ndim(x) == 0 and np.isfinite(x) and x > 0,
                    "collision_tolerance must be positive and finite")
    if species is None:
        raise ValueError("split collisions require BackgroundSpecies")
    if species.number_species < 1:
        raise ValueError("BackgroundSpecies must contain at least one species")
    field = BiotSavart(field) if isinstance(field, Coils) else field
    electric_field = Electric_field_zero() if electric_field is None else electric_field
    boozer, vmec = isinstance(field, BoozerField), isinstance(field, Vmec)
    if record_event_tangent and (boozer or vmec):
        raise ValueError("first-loss event tangents currently support Cartesian GC/Lorentz fields only")
    if adjoint == "accepted" and (boozer or vmec):
        raise ValueError("accepted split AD currently supports Cartesian GC/Lorentz fields only")
    if (_record_ad or _replay_tapes is not None) and (boozer or vmec or adjoint != "none"):
        raise ValueError("split trace audit hooks require Cartesian fields and adjoint='none'")
    if _record_ad and _replay_tapes is not None:
        raise ValueError("record and replay are separate split trace audit operations")
    gc = model == "GuidingCenter"
    if not gc and (boozer or vmec):
        raise ValueError("FullOrbit requires a Cartesian field")
    if spatial_diffusion and (not gc or boozer or vmec):
        raise ValueError("spatial_diffusion requires a Cartesian guiding-center field")
    if boozer and not isinstance(electric_field, Electric_field_zero):
        raise ValueError("Boozer split tracing does not yet support electric fields")
    mass, charge = particles.mass, particles.charge
    _concrete_check(mass, lambda x: np.isfinite(x) and x > 0, "mass must be positive and finite")
    _concrete_check(charge, lambda x: np.isfinite(x) and x != 0, "charge must be finite and nonzero")
    if times is None:
        if isinstance(n_save, bool) or not isinstance(n_save, int) or n_save < 2:
            raise ValueError("n_save must be an integer >= 2")
        times = jnp.linspace(0., tmax, n_save)
    times = jnp.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")

    def magnitude(point):
        return field.modB(*point) if boozer else field.AbsB(point)

    def public(y):
        if vmec:
            return _from_axis_regular(y)
        if boozer:
            return y.at[0].set(y[0]**2+y[1]**2).at[1].set(jnp.arctan2(y[1], y[0]))
        return y

    def gc_drift(t, y, args):
        dx, dv = _guiding_center_velocity(field, electric_field, charge, mass, y[:3], y[3], mu=y[4])
        return jnp.concatenate([dx, jnp.array([dv, 0.])])

    def rhs(t, y, args):
        if not orbit:
            return jnp.zeros_like(y)
        if boozer:
            # The Boozer RHS uses mu/m, whereas this API stores physical mu.
            return jnp.append(guiding_center_rhs(field, y[:4], y[4]/mass, mass, charge), 0.)
        if vmec:
            return _axis_regular(gc_drift)(t, y, args)
        if gc:
            return gc_drift(t, y, args)
        return jnp.concatenate([y[3:], charge/mass*(electric_field.E_covariant(y[:3])
                                                   + jnp.cross(y[3:], field.B_contravariant(y[:3])))])

    if gc:
        points = jnp.asarray(particles.initial_xyz)
        b0 = jax.vmap(magnitude)(points)
        _concrete_check(b0, lambda x: np.all(np.isfinite(x) & (x > 0)), "birth |B| must be positive and finite")
        _concrete_check(particles.initial_vparallel_over_v,
                        lambda x: np.all(np.isfinite(x) & (np.abs(x) <= 1)), "birth pitch must lie in [-1, 1]")
        y0 = jnp.concatenate([points, particles.initial_vparallel[:, None],
                              (mass*particles.initial_vperpendicular**2/(2*b0))[:, None]], axis=1)
        if vmec:
            y0 = jax.vmap(_to_axis_regular)(y0)
        elif boozer:
            _concrete_check(points[:, 0], lambda x: np.all(np.isfinite(x) & (x >= 0)), "birth s must be nonnegative")
            r = jnp.sqrt(points[:, 0])
            y0 = y0.at[:, 0].set(r*jnp.cos(points[:, 1])).at[:, 1].set(r*jnp.sin(points[:, 1]))
    else:
        if particles.initial_xyz_fullorbit is None or particles.initial_vxvyvz is None:
            raise ValueError("FullOrbit requires initial full-orbit positions and velocities")
        y0 = jnp.concatenate([jnp.asarray(particles.initial_xyz_fullorbit),
                               jnp.asarray(particles.initial_vxvyvz)], axis=1)

    def one(initial, key, replay_tape=None):
        birth = public(initial)
        birth_speed = jnp.sqrt(birth[3]**2+2*birth[4]*magnitude(birth[:3])/mass) if gc else jnp.linalg.norm(birth[3:])
        if speed_floor is None:
            thermal = jnp.stack([jnp.where(species.get_density(i, birth[:3]) > 0,
                jnp.sqrt(2*ELEMENTARY_CHARGE*species.get_temperature(i, birth[:3])/mass), jnp.inf)
                for i in range(species.number_species)])
            floor = .05*jnp.min(thermal)
            floor = jnp.where(jnp.isfinite(floor), floor, birth_speed*1e-12)
        else:
            floor = jnp.asarray(speed_floor, dtype=initial.dtype)

        def kick(t, y, h, dw, args):
            state = public(y)
            point = state[:3]
            B = magnitude(point) if gc else jnp.ones((), y.dtype)
            speed = jnp.sqrt(state[3]**2 + 2*state[4]*B/mass) if gc else jnp.linalg.norm(state[3:])
            c = collision_coefficients(species, mass, charge, speed, point, full_orbit=not gc)
            if gc:
                step = guiding_center_collision_step(speed, state[3]/speed, h, dw[-2:], c,
                    tolerance=collision_tolerance, speed_floor=floor,
                    return_error_model=adaptive and collision_controller == "candidate")
                v, pitch = step.y
                if adjoint == "accepted":
                    from .collision_ad import speed_pitch_linearization
                    def velocity_map(v, xi):
                        coefficients = collision_coefficients(species, mass, charge, v, point)
                        return guiding_center_collision_step(v, xi, h, dw[-2:], coefficients,
                            tolerance=collision_tolerance, speed_floor=floor).y
                    v, pitch = speed_pitch_linearization(velocity_map, speed, state[3]/speed)
                if spatial_diffusion:
                    b = field.B_contravariant(point)/B
                    point = point + guiding_center_spatial_increment(
                        state[3]/speed, dw[:3], c, b, charge*B/mass)
                    y = y.at[:3].set(point)
                    B = magnitude(point)
                result = y.at[3].set(v*pitch).at[4].set(mass*v*v*(1-pitch*pitch)/(2*B))
            else:
                step = particle_collision_step(state[3:], h, dw, c, tolerance=collision_tolerance,
                    return_error_model=adaptive and collision_controller == "candidate")
                result = y.at[3:].set(step.y)
            # A nonpositive field is an explicit failure even if its algebraic
            # appearances in mu and speed would otherwise cancel.
            valid_c = (jnp.isfinite(jnp.asarray(c)).all() & (c.parallel >= 0) & (c.perpendicular >= 0))
            result = jnp.where(jnp.isfinite(B) & (B > 0) & (speed > 0) & valid_c,
                               result, jnp.full_like(result, jnp.nan))
            return step._replace(y=result)

        def position(y):
            point = public(y)[:3]
            return field.to_xyz(point) if vmec else point

        def collision_wiener_norm(dw):
            # Eq. (62) controls only velocity-parallel noise. GC controls
            # speed/pitch; spatial noise has no error limit in paper Sec. 5.2.
            return jnp.linalg.norm(dw[-2:]) if gc else jnp.abs(dw[2])

        norm = OrbitErrorNorm(rtol=rtol, atol=atol, speed_scale=birth_speed,
                             velocity_components=1 if gc else 3, position_map=position if vmec else None)
        if boozer:
            def norm(a, b, error):
                return jnp.maximum(jnp.linalg.norm(error[:2]),
                    jnp.maximum(jnp.abs(error[2]), jnp.abs(error[3])/birth_speed))/(rtol+atol)
        events = []
        if vmec or boozer:
            events.append(lambda t, y, args: 1-y[0]**2-y[1]**2)
        if event is not None:
            events.extend([lambda t, y, args, f=f: f(t, public(y), args) for f in tree_util.tree_leaves(event)])
        if replay_tape is not None:
            # Diagnostic oracle: reconstruct the same physical kernels at new
            # parameters, using the exact committed production noise/schedule.
            # No Brownian proposal or adaptive solve is executed here.
            from .split_ad import replay_accepted_split
            return replay_accepted_split(rhs, kick, initial, None, replay_tape, times,
                method="dopri5" if integrator == "essos_rk45" else "dopri8")
        solution = solve_split(rhs, kick, initial, t1=tmax, dt0=timestep, ts=times, key=key,
            noise_dim=(5 if spatial_diffusion else 2) if gc else 3,
            dtmax=timestep if collision_dtmax is None else collision_dtmax,
            adaptive=adaptive, orbit_dt0=orbit_dt0, orbit_adaptive=orbit_adaptive,
            reuse_orbit_step=reuse_orbit_step,
            collision_controller=collision_controller, collision_wiener_norm=collision_wiener_norm,
            method="dopri5" if integrator == "essos_rk45" else "dopri8",
            rtol=rtol, atol=atol, controller=controller, pcoeff=pcoeff, icoeff=icoeff,
            error_norm=norm, event=tuple(events) or None, max_steps=max_steps,
            max_orbit_steps=max_orbit_steps, max_pending=max_pending, record_steps=record_steps,
            adjoint=adjoint, adjoint_checkpoint_steps=adjoint_checkpoint_steps,
            adjoint_orbit_checkpoint_steps=adjoint_orbit_checkpoint_steps,
            adjoint_max_accepted_steps=adjoint_max_accepted_steps,
            adjoint_max_orbit_steps=adjoint_max_orbit_steps, _record_ad=_record_ad,
            record_event_tangent=record_event_tangent)
        if _record_ad:
            solution, tape = solution
            return solution._replace(ys=jax.vmap(public)(solution.ys), y=public(solution.y)), tape
        return solution._replace(ys=jax.vmap(public)(solution.ys), y=public(solution.y))

    if speed_floor is not None:
        _concrete_check(speed_floor, lambda x: np.ndim(x) == 0 and np.isfinite(x) and x > 0,
                        "speed_floor must be positive and finite")
    if _replay_tapes is not None:
        return jax.jit(jax.vmap(one))(y0, particles.random_keys, _replay_tapes)
    if _record_ad:
        solution, tapes = jax.jit(jax.vmap(one))(y0, particles.random_keys)
        return solution._replace(ts=times), tapes
    return jax.jit(jax.vmap(one))(y0, particles.random_keys)._replace(ts=times)
