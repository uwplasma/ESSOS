"""Discrete reverse AD through accepted native RK steps.

The adaptive forward driver owns its decisions. It records accepted step sizes
and segment-start state/FSAL checkpoints. Backward reconstruction follows those
fixed times, retaining one segment's states and small RK stage vectors. Local
VJPs include the RK recurrence, FSAL and saved interpolation. Rejected trials,
controllers, error estimates and event-root searches are not differentiated.

All dense-output modes differentiate their actual save polynomial. Event
fractions are frozen, so this is not a derivative of a moving first-hit time.
RHS setup is rebuilt once during backward execution. Small results of large
RHS reductions are retained within the current replay segment; field-sized
arrays are rematerialized. Parameter cotangents are accumulated through either
one step or a checkpoint segment before transposing setup. Fixed input data is
excluded by JAX activity flags; physical parameter derivatives are retained.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from jax.custom_derivatives import SymbolicZero, custom_vjp_primal_tree_values

from .rk import (solve_rk, _trial_step, _C, _A, _ERROR, _C8, _A8, _ERROR8,
    _rk45_dense, _dopri8_dense, _dopri8_output_coefficients,
    _higher_order_output_coefficients, _polynomial_dense, _partition_rhs, _RKTrial, RKStatus)
from .rk_ad_rhs import prepare_rhs_linearization, prepare_reverse_rhs


def _zeros(tree):
    def zero(x):
        x = jnp.asarray(x)
        dtype = x.dtype if jnp.issubdtype(x.dtype, jnp.inexact) else jax.dtypes.float0
        return jnp.zeros(x.shape, dtype=dtype)
    return tree_util.tree_map(zero, tree)


def _add(left, right):
    return tree_util.tree_map(lambda a, b: a if a.dtype == jax.dtypes.float0 else a+b, left, right)


class _ParameterActivity(eqx.Module):
    """Static JAX activity flags, distinct from the frozen time schedule."""
    flags: tuple = eqx.field(static=True)


def _convert_all(function, *examples):
    """Expose every graph constant, including outer jit tracers, as an input."""
    graph, shape = jax.make_jaxpr(function, return_shape=True)(*examples)
    structure = tree_util.tree_structure(shape)
    def evaluate(arguments, constants):
        flat = jax.core.eval_jaxpr(graph.jaxpr, constants, *tree_util.tree_leaves(arguments))
        return tree_util.tree_unflatten(structure, flat)
    return evaluate, graph.consts


def _samples(rhs, y, t, h, trial, parameters, fractions, owned, method, mode):
    dense = _rk45_dense if method == "dopri5" else _dopri8_dense
    if mode == "higher_order":
        needed = jnp.any(owned & (fractions > 0) & (fractions < 1))
        degree = 5 if method == "dopri5" else 7
        coefficients = lax.cond(needed,
            lambda _: _higher_order_output_coefficients(rhs, t, y, h, trial.increments, parameters, method),
            lambda _: jnp.zeros((degree,)+y.shape, y.dtype), None)
        return jax.vmap(lambda s: _polynomial_dense(y, trial.y, coefficients, s))(fractions)
    if mode == "optimized" and method == "dopri8":
        coefficients = _dopri8_output_coefficients(trial.increments)
        return jax.vmap(lambda s: _polynomial_dense(y, trial.y, coefficients, s))(fractions)
    return jax.vmap(lambda s: dense(y, trial.y, trial.increments, s))(fractions)


def _step_pullback(rhs, y, t, h, endpoint, increments, parameters,
                   fraction, sample_fractions, owned, event_save, by, bf, saved_bar, method, mode,
                   initial_fsal=False, cached_pullback=None):
    """Transpose the RK recurrence from reconstructed primal stage values.

    Segment reconstruction supplies the primal stage values. Linear RK/dense
    algebra is transposed separately; each RHS VJP retains the physical
    derivatives at the original stage time/state. Stock prepared Biot--Savart
    calls can use the exact native field linearization. Only one RHS transpose's temporary arrays are
    live at a time, rather than every stage's field intermediates. The reverse
    recurrence includes all stage dependencies; stages are not frozen in AD.
    """
    nodes, rows, _ = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    dense = _rk45_dense if method == "dopri5" else _dopri8_dense
    def outputs(y, endpoint, increments, parameters):
        terminal = dense(y, endpoint, increments, fraction)
        trial = _RKTrial(endpoint, jnp.zeros_like(y), increments, jnp.zeros_like(y))
        samples = _samples(rhs, y, t, h, trial, parameters,
                            sample_fractions, owned, method, mode)
        if mode == "higher_order":
            samples = jnp.where(event_save[:, None], terminal, samples)
        return terminal, samples
    _, output_vjp = jax.vjp(outputs, y, endpoint, increments, parameters)
    dy, dy1, dk, dense_parameters = output_vjp((by, jnp.where(owned[:, None], saved_bar, 0.)))
    dp = dense_parameters

    dy = dy+dy1
    bank = increments[:-1]
    dk = dk.at[:-1].add(jnp.asarray(rows[-1], y.dtype)[:, None]*dy1)
    coefficients = np.zeros((len(increments), len(bank)))
    for i, row in enumerate(rows, start=1):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, y.dtype)
    times = jnp.asarray((0.,)+nodes, y.dtype)

    def stage(reverse_index, carry):
        dy, dk, dp = carry
        i = len(increments)-1-reverse_index
        weights = coefficients[i]
        point = y+jnp.tensordot(weights, bank, axes=1)
        point = jnp.where(i == len(increments)-1, endpoint, jnp.where(i == 0, y, point))
        # The endpoint FSAL derivative is unscaled; its RK increment also
        # contributes to dense output. Stage zero is visited only on the first
        # accepted step, where the initial FSAL value is an actual RHS call.
        seed = h*dk[i]+jnp.where(i == len(increments)-1, bf, 0.)
        if cached_pullback is None:
            _, rhs_vjp = jax.vjp(lambda state,p: rhs(t+times[i]*h,state,p), point, parameters)
            dpoint, dparams = rhs_vjp(seed)
        else:
            dpoint, dparams = cached_pullback(i, seed)
        return dy+dpoint, dk.at[:-1].add(weights[:, None]*dpoint), _add(dp, dparams)
    bound = len(increments)-1+jnp.asarray(initial_fsal, jnp.int32)
    dy, dk, dp = lax.fori_loop(jnp.asarray(0, jnp.int32), bound, stage, (dy, dk, dp))
    return dy, jnp.where(initial_fsal, 0., h*dk[0]), dp


def _trial_with_residuals(linearize, t, y, h, f0, constants, initial_cache):
    """Replay the unchanged RK stages and keep compact RHS residuals.

    Used only in backward reconstruction; no embedded error is needed here.
    Stage zero's cache belongs to the initial trajectory RHS. It is consumed
    only when ``initial_fsal`` is true in the first accepted step's pullback.
    """
    nodes, rows, error_weights = constants
    n = len(error_weights)
    coefficients = np.zeros((n-2, n-1))
    for i, row in enumerate(rows[:-1]):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, y.dtype)
    times = jnp.asarray(nodes[:-1], y.dtype)
    bank = jnp.concatenate([(h*f0)[None], jnp.zeros((n-2,)+y.shape, y.dtype)])
    cache = tree_util.tree_map(lambda r: jnp.zeros((n,)+r.shape, r.dtype).at[0].set(r), initial_cache)

    def stage(i, values):
        bank, cache = values
        point = y+jnp.tensordot(coefficients[i-1], bank, axes=1)
        f, residuals = linearize(t+times[i-1]*h, point)
        return lax.dynamic_update_index_in_dim(bank, h*f, i, axis=0), tree_util.tree_map(
            lambda buffer, r: lax.dynamic_update_index_in_dim(buffer, r, i, axis=0), cache, residuals)

    bank, cache = lax.fori_loop(1, n-1, stage, (bank, cache))
    endpoint = y+jnp.tensordot(jnp.asarray(rows[-1], y.dtype), bank, axes=1)
    final, residuals = linearize(t+h, endpoint)
    cache = tree_util.tree_map(lambda buffer, r: buffer.at[-1].set(r), cache, residuals)
    trial = _RKTrial(endpoint, jnp.zeros_like(y), jnp.concatenate([bank, (h*final)[None]]), final)
    return trial, cache


def replay_accepted(rhs, y0, args, tape, ts, *, method="dopri5", dense_output="standard", stage_execution="loop"):
    """Recompute a recorded accepted grid at new parameters, without adaptation.

    Returns (saved_states, terminal_state). This also supplies a direct frozen-
    schedule finite-difference/JVP oracle. Saved checkpoints in ``tape`` are
    deliberately not used as states: the new numerical trajectory propagates.
    """
    from .rk import _select_trial_step
    trial_step = _select_trial_step(stage_execution)
    tape = tree_util.tree_map(lax.stop_gradient, tape)
    ts = lax.stop_gradient(jnp.asarray(ts))
    dense = _rk45_dense if method == "dopri5" else _dopri8_dense
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    indices = jnp.arange(len(ts))
    initial = (jnp.array(0), y0, rhs(tape.times[0, 0], y0, args),
        jnp.where((indices < tape.initial_saves)[:, None], y0, jnp.full((len(ts), y0.size), jnp.inf)))
    def advance(carry):
        i, y, f, ys = carry
        t, h = tape.times[i]
        trial = trial_step(rhs, t, y, h, args, f, *constants)
        fraction = tape.fractions[i]
        terminal = dense(y, trial.y, trial.increments, fraction)
        owned = (indices >= tape.save_start[i]) & (indices < tape.save_end[i])
        fractions = jnp.where(owned, (ts-t)/h, 0.)
        samples = _samples(rhs, y, t, h, trial, args, fractions, owned, method, dense_output)
        if dense_output == "higher_order":
            samples = jnp.where(((fraction < 1) & (ts == t+fraction*h))[:, None], terminal, samples)
        return i+1, terminal, trial.f1, jnp.where(owned[:, None], samples, ys)
    _, final, _, ys = lax.while_loop(lambda carry: carry[0] < tape.count, advance, initial)
    return ys, final


def solve_accepted(rhs, y0, args=None, **options):
    """Return the original forward result with an accepted-step reverse rule.

    Closure conversion exposes RHS field/coil parameters as explicit VJP inputs.
    A second conversion exposes forward-only controls (including event/norm
    closures) separately: their cotangents are zero under this AD contract.
    """
    if options.get("event") is not None:
        # The converted driver uses prepared RHS parameters internally; event
        # callbacks must still receive their original public args object.
        options["event"] = tree_util.tree_map(
            lambda function:lambda t,y,unused:function(t,y,args), options["event"])
    y0 = jnp.asarray(y0, dtype=jnp.result_type(y0, float))
    t0 = lax.stop_gradient(jnp.asarray(options.get("t0", 0.), dtype=y0.dtype))
    ts = lax.stop_gradient(jnp.asarray(options.get("ts", ()), dtype=y0.dtype))
    checkpoint_steps = int(options.pop("adjoint_checkpoint_steps", 16))
    parameter_accumulation = options.pop("adjoint_parameter_accumulation", "segment")
    if parameter_accumulation not in ("step", "segment"):
        raise ValueError("adjoint_parameter_accumulation must be 'step' or 'segment'")
    # Bind arbitrary ESSOS args before conversion. Particles intentionally is
    # an ordinary object, and need not become a new field/particle PyTree just
    # to cross the custom VJP boundary. Its actually used traced attributes,
    # and those of the field, are exposed by the converted numerical graph.
    setup, stage = _partition_rhs(rhs, t0, y0, args)
    raw_parameters = (setup.consts, stage.consts)

    def prepare(raw):
        return jax.core.eval_jaxpr(setup.jaxpr, raw[0]), raw[1]

    def evaluate(t, y, parameters):
        prepared, stage_constants = parameters
        return jax.core.eval_jaxpr(stage.jaxpr, stage_constants, *prepared, t, y)[0]

    def forward(initial, parameters):
        return solve_rk(evaluate, initial, parameters, **options, _record_ad=True,
                        _ad_checkpoint_steps=checkpoint_steps, _ad_compact=True)

    converted_forward, driver_constants = _convert_all(forward, y0, prepare(raw_parameters))
    controls = (driver_constants, t0, ts)
    constants = (_C, _A, _ERROR) if options.get("method", "dopri5") == "dopri5" else (_C8, _A8, _ERROR8)
    mode = options.get("dense_output", "standard")
    method = options.get("method", "dopri5")

    @jax.custom_vjp
    def integrate(initial, parameters, controls):
        return converted_forward((initial, prepare(parameters)), controls[0])[0]

    def fwd(initial, parameters, controls):
        activity = _ParameterActivity(tuple(leaf.perturbed for leaf in tree_util.tree_leaves(parameters)))
        initial, raw, controls = custom_vjp_primal_tree_values((initial, parameters, controls))
        parameters = prepare(raw)
        out, tape = converted_forward((initial, parameters), controls[0])
        return out, (initial, raw, controls, tape, activity)

    def bwd(residual, bar):
        initial, raw, controls, tape, activity = residual
        tape = eqx.error_if(tape, tape.count > tape.step_sizes.shape[0],
            "accepted-step AD capacity exceeded; increase adjoint_max_accepted_steps")
        raw_leaves, raw_structure = tree_util.tree_flatten(raw)
        active = tuple(value for value, flag in zip(raw_leaves, activity.flags) if flag)
        backward_prepare, backward_rhs = prepare_reverse_rhs(prepare, evaluate, controls[1], initial, raw)
        def prepare_active(values):
            iterator = iter(values)
            full = tree_util.tree_unflatten(raw_structure,
                [next(iterator) if flag else value for value, flag in zip(raw_leaves, activity.flags)])
            return backward_prepare(full)
        parameters, setup_pullback = jax.vjp(prepare_active, active)
        _, t0, ts = controls
        linearization = prepare_rhs_linearization(backward_rhs, t0, initial, parameters)
        if linearization is None:
            initial_cache = ()
        else:
            linearize, rhs_pullback, initial_cache = linearization
        sample_indices = jnp.arange(len(ts))
        terminal_bar = jnp.zeros_like(initial) if isinstance(bar.y, SymbolicZero) else bar.y
        saved_bar = jnp.zeros((len(ts), initial.size), initial.dtype) if isinstance(bar.ys, SymbolicZero) else bar.ys

        def replay_trial(t, y, h, f):
            if linearization is None:
                return _trial_step(backward_rhs, t, y, h, parameters, f, *constants), ()
            return _trial_with_residuals(linearize, t, y, h, f, constants, initial_cache)

        def reverse_step(carry, y, t, endpoint, increments, cache):
            i, by, bf, bp = carry
            i = i-1
            h = tape.step_sizes[i]
            fraction = jnp.where(i == tape.count-1, tape.fraction, 1.)
            end_time = jnp.where(i == tape.count-1, tape.final_time, t+h)
            owned = (ts > t) & (ts <= end_time)
            sample_fractions = jnp.where(owned, (ts-t)/h, 0.)
            event_save = (fraction < 1) & (ts == t+fraction*h)
            def cached_pullback(index, seed):
                return rhs_pullback(tree_util.tree_map(lambda buffer: buffer[index], cache), seed)
            dy, df, dp = _step_pullback(backward_rhs, y, t, h, endpoint, increments, parameters,
                fraction, sample_fractions, owned, event_save, by, bf, saved_bar, method, mode,
                initial_fsal=(i == 0), cached_pullback=None if linearization is None else cached_pullback)
            contribution = setup_pullback(dp)[0] if parameter_accumulation == "step" else dp
            return i, dy, df, _add(bp, contribution)

        def reverse_segment(carry):
            end = carry[0]
            segment = jnp.maximum((end-1) // checkpoint_steps, 0)
            start = segment*checkpoint_steps
            y, f = tape.states[segment], tape.derivatives[segment]
            t = tape.times[segment]
            if checkpoint_steps == 1:
                trial, cache = replay_trial(t, y, tape.step_sizes[start], f)
                return reverse_step(carry, y, t, trial.y, trial.increments, cache)

            # Reconstruct the segment once. Cache RK stage vectors and compact
            # RHS reductions; field-sized arrays are rematerialized.
            # Rejections and adaptation are absent from this RK reverse replay.
            states = jnp.zeros((checkpoint_steps, initial.size), initial.dtype)
            endpoints = jnp.zeros_like(states)
            increments = jnp.zeros((checkpoint_steps, len(constants[2]), initial.size), initial.dtype)
            times = jnp.zeros((checkpoint_steps,), initial.dtype)
            cache = tree_util.tree_map(lambda r: jnp.zeros(
                (checkpoint_steps, len(constants[2]))+r.shape, r.dtype), initial_cache)
            def reconstruct(local):
                i, t, y, f, states, endpoints, increments, times, cache = local
                slot = i-start
                states = states.at[slot].set(y)
                times = times.at[slot].set(t)
                h = tape.step_sizes[i]
                trial, residuals = replay_trial(t, y, h, f)
                cache = tree_util.tree_map(lambda buffer, r: buffer.at[slot].set(r), cache, residuals)
                endpoints = endpoints.at[slot].set(trial.y)
                increments = increments.at[slot].set(trial.increments)
                # An event can only terminate the last accepted step. Its
                # full endpoint is harmless here: no later slot consumes it.
                return i+1, t+h, trial.y, trial.f1, states, endpoints, increments, times, cache
            _, _, _, _, states, endpoints, increments, times, cache = lax.while_loop(
                lambda local: local[0] < end, reconstruct,
                (start, t, y, f, states, endpoints, increments, times, cache))
            def local_reverse(local):
                index = local[0]-1-start
                return reverse_step(local, states[index], times[index], endpoints[index], increments[index],
                                    tree_util.tree_map(lambda buffer: buffer[index], cache))
            return lax.while_loop(lambda local: local[0] > start, local_reverse, carry)

        def reverse(carry):
            if parameter_accumulation == "step":
                return reverse_segment(carry)
            # Preparation's pullback is linear in these cotangents. Sum them
            # locally before applying it; the outer carry still stores only
            # active raw-parameter bars, never a full-trajectory field tape.
            i, by, bf, bp = carry
            i, dy, df, dp = reverse_segment((i, by, bf, _zeros(parameters)))
            return i, dy, df, _add(bp, setup_pullback(dp)[0])

        _, by, bf, bp = lax.while_loop(lambda carry: carry[0] > 0, reverse,
            (tape.count, terminal_bar, jnp.zeros_like(initial), _zeros(active)))
        by = by+jnp.sum(jnp.where((sample_indices < tape.initial_saves)[:, None], saved_bar, 0.), axis=0)
        iterator = iter(bp)
        raw_bar = tree_util.tree_unflatten(raw_structure,
            [next(iterator) if flag else _zeros(value) for value, flag in zip(raw_leaves, activity.flags)])
        derivatives = (by, raw_bar, _zeros(controls))
        return eqx.error_if(derivatives, tape.status > RKStatus.EVENT,
            "accepted-step AD requires a successful/event solve; inspect status and max_steps")

    integrate.defvjp(fwd, bwd, symbolic_zeros=True)
    return integrate(y0, raw_parameters, controls)
