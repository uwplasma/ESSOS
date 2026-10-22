"""Checkpointed pathwise reverse AD through accepted orbit/collision maps.

The production forward solve chooses all times and draws all noise. Replay
uses only its accepted outer maps, inner RK schedules and Wiener increments.
Every collision coefficient, magnetic coordinate transform and physical kick
retains its physical derivatives. Sparse inner state/FSAL checkpoints come from
the accepted forward trials. Reverse rebuilds one inner segment at a time,
sharing the collisionless native field transposes, compact stage caches and
segment parameter accumulation. No outer orbit reconstruction is needed.
Controllers, rejected trials, error estimates
and event-time searches have no derivative path. No Brownian samples are drawn
in replay. The inner schedule is flat, rather than a product of step limits.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, tree_util
from jax.interpreters import partial_eval
from jax.custom_derivatives import SymbolicZero, custom_vjp_primal_tree_values

from .rk import (_trial_step, _C, _A, _ERROR, _C8, _A8, _ERROR8,
                 _rk45_dense, _dopri8_dense, _partition_graph)
from .rk_ad import (_zeros, _add, _convert_all, _ParameterActivity,
                    _step_pullback, _trial_with_residuals)
from .rk_ad_rhs import prepare_reverse_rhs, prepare_rhs_linearization
from .split import solve_split, SplitStatus


def _checked_tape(tape, *, compact=False):
    if compact:
        # Replay depends on count, so a checked count gates every reverse map
        # without routing the large checkpoint banks through error branches.
        count = eqx.error_if(tape.count, tape.status > SplitStatus.EVENT,
            "accepted split AD requires a successful/event solve; inspect status and work limits")
        count = eqx.error_if(count, count > tape.times.shape[0],
            "accepted split AD capacity exceeded; increase adjoint_max_accepted_steps")
        count = eqx.error_if(count, tape.orbit_count > tape.orbit_times.shape[0],
            "accepted orbit AD capacity exceeded; increase adjoint_max_orbit_steps")
        return tape._replace(count=count)
    tape = eqx.error_if(tape, tape.status > SplitStatus.EVENT,
        "accepted split AD requires a successful/event solve; inspect status and work limits")
    tape = eqx.error_if(tape, tape.count > tape.times.shape[0],
        "accepted split AD capacity exceeded; increase adjoint_max_accepted_steps")
    return eqx.error_if(tape, tape.orbit_count > tape.orbit_times.shape[0],
        "accepted orbit AD capacity exceeded; increase adjoint_max_orbit_steps")


def _partition(function, *examples):
    """Expose closure parameters and hoist parameter-only kernel preparation."""
    graph, shape = jax.make_jaxpr(function, return_shape=True)(*examples)
    structure = tree_util.tree_structure(shape)
    setup, stage = _partition_graph(graph, [True] * len(examples))

    def prepare(raw):
        return jax.core.eval_jaxpr(setup.jaxpr, raw[0]), raw[1]

    def evaluate(arguments, parameters):
        residuals, constants = parameters
        out = jax.core.eval_jaxpr(stage.jaxpr, constants, *residuals, *arguments)
        return tree_util.tree_unflatten(structure, out)

    return (setup.consts, stage.consts), prepare, evaluate


def _state_projection(prepare, evaluate, raw, *examples):
    """Prune collision error-only inputs before transposing parameter setup.

    The forward driver still evaluates the complete production CollisionStep.
    Reverse replay uses its state result only. Selecting live prepared inputs
    here avoids differentiating error-only preparation, including expressions
    whose derivatives can be singular even when the physical kick is regular.
    """
    parameters = prepare(raw)
    leaves = tree_util.tree_leaves(parameters)
    graph = jax.make_jaxpr(lambda arguments,p:evaluate(arguments,p).y)(examples,parameters)
    nargs = len(tree_util.tree_leaves(examples))
    state_graph, used = partial_eval.dce_jaxpr(graph.jaxpr,[True],
        instantiate=[True]*nargs+[False]*len(leaves))
    live = used[nargs:]

    def prepare_state(values):
        return tuple(value for value,flag in zip(tree_util.tree_leaves(prepare(values)),live) if flag)

    def state(arguments,values):
        return jax.core.eval_jaxpr(state_graph,graph.consts,*tree_util.tree_leaves(arguments),*values)[0]

    return prepare_state,state


def _orbit_replay(rhs, initial, parameters, tape, index, method):
    """Forward-only oracle; propagate new physical inputs on the frozen grid."""
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    dense = _rk45_dense if method == "dopri5" else _dopri8_dense
    start, end = tape.offsets[index], tape.offsets[index+1]
    f = rhs(tape.orbit_times[start, 0], initial, parameters)

    def step(local):
        i, y, f = local
        t, h = tape.orbit_times[i]
        trial = _trial_step(rhs, t, y, h, parameters, f, *constants)
        fraction = jnp.where(i == end-1, tape.fractions[index], 1.)
        terminal = dense(y, trial.y, trial.increments, fraction)
        return i+1, terminal, trial.f1

    _, final, _ = lax.while_loop(
        lambda local: local[0] < end, step, (start, initial, f))
    return final


def replay_accepted_split(rhs, collision, initial, args, tape, ts, *, method="dopri8"):
    """Forward-only frozen-schedule/noise oracle for finite-difference checks.

    New parameters propagate through every orbit and collision map. Recorded
    checkpoint states are deliberately ignored. No adaptation or noise drawing
    occurs. Returns (saved_states, terminal_state); ordinary JVPs are supported.
    """
    tape = _checked_tape(tree_util.tree_map(lax.stop_gradient, tape))
    ts = lax.stop_gradient(jnp.asarray(ts))
    rhs_bound = lambda t, y, unused: rhs(t, y, args)

    def advance(local):
        index, y, saved = local
        y = _orbit_replay(rhs_bound, y, None, tape, index, method)
        t, end = tape.times[index]
        y = lax.cond(tape.kicks[index],
            lambda y: collision(end, y, end-t, tape.increments[index], args).y,
            lambda y: y, y)
        owned = (ts > t) & (ts <= end)
        return index+1, y, jnp.where(owned[:, None], y, saved)

    _, final, saved = lax.while_loop(lambda local: local[0] < tape.count, advance,
        (jnp.array(0), initial, jnp.broadcast_to(initial, (len(ts), initial.size))))
    saved = jnp.where((jnp.arange(len(ts)) >= tape.saved_count)[:, None], final, saved)
    return saved, final


def solve_accepted_split(rhs, collision, y0, args=None, **options):
    """Attach a discrete reverse rule to the unchanged production split solve."""
    # Kernels receive prepared numerical parameters internally. Control
    # callbacks retain the user's original args ABI, while their numerical
    # decisions are frozen by the outer custom VJP.
    if options.get("event") is not None:
        options["event"] = tree_util.tree_map(
            lambda function: lambda t,y,unused,**kw: function(t,y,args,**kw), options["event"])
    if options.get("step_limit") is not None:
        step_limit = options["step_limit"]
        options["step_limit"] = lambda t,y,unused: step_limit(t,y,args)
    y0 = jnp.asarray(y0, dtype=jnp.result_type(y0, float))
    t0 = lax.stop_gradient(jnp.asarray(options.get("t0", 0.), y0.dtype))
    ts = lax.stop_gradient(jnp.asarray(options["ts"], y0.dtype))
    h0 = jnp.asarray(options["dt0"], y0.dtype)
    noise = jnp.zeros((options["noise_dim"],), y0.dtype)
    raw_rhs, prepare_rhs, evaluate_rhs = _partition(lambda t, y: rhs(t, y, args), t0, y0)
    raw_collision, prepare_collision, evaluate_collision = _partition(
        lambda t, y, h, dw: collision(t, y, h, dw, args), t0, y0, h0, noise)
    prepare_collision_state, collision_state = _state_projection(
        prepare_collision,evaluate_collision,raw_collision,t0,y0,h0,noise)

    def prepare(raw):
        return prepare_rhs(raw[0]), prepare_collision(raw[1])

    def orbit_rhs(t, y, parameters):
        return evaluate_rhs((t, y), parameters)

    # Components omitted from a stored endpoint must be copied unchanged by
    # the collision map. This metadata selects compact full-orbit storage;
    # generic maps and GC retain their previous layout.
    collision_indices = options.get("_collision_state_indices")
    forward_options = dict(options)
    if collision_indices is not None:
        # _partition already hoisted all parameter-only RHS preparation.
        forward_options["_orbit_rhs_prepared"] = True

    def forward(initial, parameters):
        return solve_split(lambda t, y, p: orbit_rhs(t, y, p[0]),
            lambda t, y, h, dw, p: evaluate_collision((t, y, h, dw), p[1]),
            initial, parameters, **forward_options, _record_ad=True)

    raw = (raw_rhs, raw_collision)
    converted_forward, driver_constants = _convert_all(forward, y0, prepare(raw))
    controls = (driver_constants, ts)
    outer_segment = int(options.get("adjoint_checkpoint_steps", 16))
    inner_segment = int(options.get("adjoint_orbit_checkpoint_steps", 16))
    workspace = min(options.get("max_orbit_steps", 10_000),
                    options.get("adjoint_max_orbit_steps") or options.get("max_steps", 100_000))
    method = options.get("method", "dopri8")
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)

    @jax.custom_vjp
    def integrate(initial, raw, controls):
        return converted_forward((initial, prepare(raw)), controls[0])[0]

    def fwd(initial, raw, controls):
        activity = _ParameterActivity(tuple(leaf.perturbed for leaf in tree_util.tree_leaves(raw)))
        initial, raw, controls = custom_vjp_primal_tree_values((initial, raw, controls))
        out, tape = converted_forward((initial, prepare(raw)), controls[0])
        return out, (initial, raw, controls, tape, activity,
                     out.y if collision_indices is not None else None)

    def bwd(residual, bar):
        initial, raw, controls, tape, activity, final = residual
        tape = _checked_tape(tape, compact=collision_indices is not None)
        raw_leaves, raw_structure = tree_util.tree_flatten(raw)
        active = tuple(value for value, flag in zip(raw_leaves, activity.flags) if flag)

        def active_raw(values):
            iterator = iter(values)
            return tree_util.tree_unflatten(raw_structure,
                [next(iterator) if flag else value for value, flag in zip(raw_leaves, activity.flags)])

        backward_prepare, backward_rhs = prepare_reverse_rhs(
            prepare_rhs, orbit_rhs, t0, initial, raw[0])
        rp, rhs_setup_pullback = jax.vjp(
            lambda values: backward_prepare(active_raw(values)[0]), active)
        linearization = prepare_rhs_linearization(backward_rhs, t0, initial, rp)
        if linearization is not None:
            linearize, rhs_pullback, initial_cache = linearization
        else:
            initial_cache = ()
        csp, collision_setup_pullback = jax.vjp(
            lambda values:prepare_collision_state(active_raw(values)[1]),active)
        ts = controls[1]
        saved_bar = jnp.zeros((len(ts), initial.size), initial.dtype) if isinstance(bar.ys, SymbolicZero) else bar.ys
        terminal_bar = jnp.zeros_like(initial) if isinstance(bar.y, SymbolicZero) else bar.y
        terminal_bar += jnp.sum(jnp.where((jnp.arange(len(ts)) >= tape.saved_count)[:, None], saved_bar, 0.), axis=0)

        def reverse_map(carry, initial):
            index, by, bp, post = carry
            index -= 1
            offset = tape.checkpoint_offsets[index]
            # The next map's initial state is this map's post-collision state.
            # Restore only changed components to obtain the pre-collision state.
            orbit_end = (tape.orbit_endpoints[index] if collision_indices is None else
                post.at[jnp.asarray(collision_indices)].set(tape.orbit_endpoints[index]))
            count = (workspace+inner_segment-1)//inner_segment
            rows = offset+jnp.arange(count)
            # Padded rows are never read by the bounded inner reverse loop.
            if collision_indices is None:
                checkpoints = (tape.orbit_states[rows],tape.orbit_derivatives[rows])
            else:
                # One first derivative per preceding interval is omitted.
                derivative_rows = offset-index+jnp.arange(max(count-1,0))
                checkpoints = (tape.orbit_states[rows],tape.orbit_derivatives[derivative_rows])
            t, end = tape.times[index]
            by += jnp.sum(jnp.where(((ts > t) & (ts <= end))[:, None], saved_bar, 0.), axis=0)

            def kick_reverse(values):
                by, bp = values
                _, pullback = jax.vjp(lambda y, values: collision_state(
                    (end, y, end-t, tape.increments[index]),values),orbit_end,csp)
                dy, dp = pullback(by)
                return dy, (bp[0],_add(bp[1],dp))

            by, bp = lax.cond(tape.kicks[index], kick_reverse, lambda values: values, (by, bp))
            orbit_start, orbit_stop = tape.offsets[index], tape.offsets[index+1]
            states, derivatives = checkpoints
            # Each collision starts a new orbit: its initial FSAL cache must
            # use this map's state, not the trajectory's original birth state.
            if collision_indices is None:
                map_initial_cache = (linearize(tape.orbit_times[orbit_start, 0], initial)[1]
                                     if linearization is not None else ())
            elif linearization is not None:
                map_initial_value, map_initial_cache = linearize(tape.orbit_times[orbit_start, 0], initial)
            else:
                map_initial_value = backward_rhs(tape.orbit_times[orbit_start, 0], initial, rp)
                map_initial_cache = ()

            def reverse_orbit_segment(local):
                stop, by, bf, bp = local
                segment = jnp.maximum((stop-orbit_start-1) // inner_segment, 0)
                start = orbit_start + segment*inner_segment
                y = states[segment]
                if collision_indices is None:
                    f = derivatives[segment]
                else:
                    f = (jnp.where(segment == 0, map_initial_value, derivatives[jnp.maximum(segment-1,0)])
                         if derivatives.shape[0] else map_initial_value)
                state_bank = jnp.zeros((inner_segment, initial.size), initial.dtype)
                endpoint_bank = jnp.zeros_like(state_bank)
                increment_bank = jnp.zeros((inner_segment, len(constants[2]), initial.size), initial.dtype)
                cache_bank = tree_util.tree_map(lambda value: jnp.zeros(
                    (inner_segment, len(constants[2]))+value.shape, value.dtype), initial_cache)

                def reconstruct(values):
                    i, y, f, state_bank, endpoint_bank, increment_bank, cache_bank = values
                    slot = i-start
                    t, h = tape.orbit_times[i]
                    if linearization is None:
                        trial = _trial_step(backward_rhs, t, y, h, rp, f, *constants)
                        residuals = ()
                    else:
                        trial, residuals = _trial_with_residuals(
                            linearize, t, y, h, f, constants, map_initial_cache)
                    cache_bank = tree_util.tree_map(
                        lambda buffer,value: buffer.at[slot].set(value), cache_bank, residuals)
                    return (i+1, trial.y, trial.f1, state_bank.at[slot].set(y),
                        endpoint_bank.at[slot].set(trial.y), increment_bank.at[slot].set(trial.increments),
                        cache_bank)

                _, _, _, state_bank, endpoint_bank, increment_bank, cache_bank = lax.while_loop(
                    lambda values: values[0] < stop, reconstruct,
                    (start, y, f, state_bank, endpoint_bank, increment_bank, cache_bank))

                def reverse_step(values):
                    i, by, bf, dp = values
                    i -= 1
                    slot = i-start
                    t, h = tape.orbit_times[i]
                    fraction = jnp.where(i == orbit_stop-1, tape.fractions[index], 1.)
                    def cached_pullback(stage, seed):
                        return rhs_pullback(tree_util.tree_map(
                            lambda buffer: buffer[slot, stage], cache_bank), seed)
                    dy, df, contribution = _step_pullback(backward_rhs, state_bank[slot], t, h,
                        endpoint_bank[slot], increment_bank[slot], rp, fraction,
                        jnp.empty((0,), initial.dtype), jnp.empty((0,), bool), jnp.empty((0,), bool),
                        by, bf, jnp.empty((0, initial.size), initial.dtype), method, "standard",
                        initial_fsal=(i == orbit_start),
                        cached_pullback=cached_pullback if linearization is not None else None)
                    return i, dy, df, _add(dp, contribution)

                # Share the orbit-only segment policy: transpose parameter
                # preparation once after summing the segment's prepared bars.
                i, by, bf, dp = lax.while_loop(lambda values: values[0] > start, reverse_step,
                    (stop, by, bf, _zeros(rp)))
                return i, by, bf, (_add(bp[0],rhs_setup_pullback(dp)[0]),bp[1])

            _, by, _, bp = lax.while_loop(lambda values: values[0] > orbit_start,
                reverse_orbit_segment, (orbit_stop, by, jnp.zeros_like(initial), bp))
            return index, by, bp, initial if collision_indices is not None else None

        def reverse(carry):
            stop = carry[0]
            segment = jnp.maximum((stop-1) // outer_segment, 0)
            start = segment*outer_segment
            def reverse_recorded(values):
                index = values[0]-1
                initial = (jnp.where(tape.offsets[index+1]>tape.offsets[index],
                    tape.orbit_states[tape.checkpoint_offsets[index]],tape.orbit_endpoints[index])
                    if collision_indices is None else tape.orbit_states[tape.checkpoint_offsets[index]])
                return reverse_map(values,initial)
            return lax.while_loop(lambda values: values[0]>start,reverse_recorded,carry)

        _, by, bp, _ = lax.while_loop(lambda values: values[0] > 0, reverse,
            (tape.count, terminal_bar, (_zeros(active),_zeros(csp)), final))
        has_kick = jnp.any((jnp.arange(tape.kicks.size) < tape.count) & tape.kicks)
        collision_bar = lax.cond(has_kick,lambda _:collision_setup_pullback(bp[1])[0],
            lambda _:_zeros(active),None)
        bp = _add(bp[0],collision_bar)
        by += jnp.sum(jnp.where((jnp.arange(len(ts)) < tape.initial_saves)[:, None], saved_bar, 0.), axis=0)
        iterator = iter(bp)
        raw_bar = tree_util.tree_unflatten(raw_structure,
            [next(iterator) if flag else _zeros(value) for value, flag in zip(raw_leaves, activity.flags)])
        return by, raw_bar, _zeros(controls)

    integrate.defvjp(fwd, bwd, symbolic_zeros=True)
    return integrate(y0, raw, controls)
