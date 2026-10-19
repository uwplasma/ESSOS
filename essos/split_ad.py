"""Checkpointed pathwise reverse AD through accepted orbit/collision maps.

The production forward solve chooses all times and draws all noise. Replay
uses only its accepted outer maps, inner RK schedules and Wiener increments.
Every collision coefficient, magnetic coordinate transform and physical kick
retains ordinary JAX derivatives. Controllers, rejected trials, error estimates
and event-time searches have no derivative path. No Brownian samples are drawn
in replay. The inner schedule is flat, rather than a product of step limits.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, tree_util
from jax.interpreters import partial_eval
from jax.custom_derivatives import SymbolicZero, custom_vjp_primal_tree_values

from .rk import _trial_step, _C, _A, _ERROR, _C8, _A8, _ERROR8, _rk45_dense, _dopri8_dense
from .rk_ad import _zeros, _add, _convert_all, _ParameterActivity, _step_pullback
from .split import solve_split, SplitStatus


def _checked_tape(tape):
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
    setup, stage, _, _ = partial_eval.partial_eval_jaxpr_nounits(
        graph, [True] * len(examples), instantiate=True)

    def prepare(raw):
        return jax.core.eval_jaxpr(setup.jaxpr, raw[0]), raw[1]

    def evaluate(arguments, parameters):
        residuals, constants = parameters
        out = jax.core.eval_jaxpr(stage.jaxpr, constants, *residuals, *arguments)
        return tree_util.tree_unflatten(structure, out)

    return (setup.consts, stage.consts), prepare, evaluate


def _orbit_replay(rhs, initial, parameters, tape, index, method, *, checkpoint_steps=0, workspace=0):
    """Replay one committed orbit, optionally collecting its segment starts."""
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    dense = _rk45_dense if method == "dopri5" else _dopri8_dense
    start, end = tape.offsets[index], tape.offsets[index+1]
    count = (workspace + checkpoint_steps - 1) // checkpoint_steps if checkpoint_steps else 0
    states = jnp.zeros((count, initial.size), initial.dtype)
    derivatives = jnp.zeros_like(states)
    f = rhs(tape.orbit_times[start, 0], initial, parameters)

    def step(local):
        i, y, f, states, derivatives = local
        if checkpoint_steps:
            def checkpoint(buffers):
                states, derivatives = buffers
                slot = (i-start) // checkpoint_steps
                return states.at[slot].set(y), derivatives.at[slot].set(f)
            states, derivatives = lax.cond((i-start) % checkpoint_steps == 0,
                checkpoint, lambda buffers: buffers, (states, derivatives))
        t, h = tape.orbit_times[i]
        trial = _trial_step(rhs, t, y, h, parameters, f, *constants)
        fraction = jnp.where(i == end-1, tape.fractions[index], 1.)
        terminal = dense(y, trial.y, trial.increments, fraction)
        return i+1, terminal, trial.f1, states, derivatives

    _, final, _, states, derivatives = lax.while_loop(
        lambda local: local[0] < end, step, (start, initial, f, states, derivatives))
    return final, (states, derivatives)


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
        y, _ = _orbit_replay(rhs_bound, y, None, tape, index, method)
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

    def prepare(raw):
        return prepare_rhs(raw[0]), prepare_collision(raw[1])

    def orbit_rhs(t, y, parameters):
        return evaluate_rhs((t, y), parameters)

    def forward(initial, parameters):
        return solve_split(lambda t, y, p: orbit_rhs(t, y, p[0]),
            lambda t, y, h, dw, p: evaluate_collision((t, y, h, dw), p[1]),
            initial, parameters, **options, _record_ad=True)

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
        return out, (initial, raw, controls, tape, activity)

    def bwd(residual, bar):
        initial, raw, controls, tape, activity = residual
        tape = _checked_tape(tape)
        raw_leaves, raw_structure = tree_util.tree_flatten(raw)
        active = tuple(value for value, flag in zip(raw_leaves, activity.flags) if flag)

        def prepare_active(values):
            iterator = iter(values)
            full = tree_util.tree_unflatten(raw_structure,
                [next(iterator) if flag else value for value, flag in zip(raw_leaves, activity.flags)])
            return prepare(full)

        rp, cp = prepare_active(active)
        _, rhs_setup_pullback = jax.vjp(lambda values: prepare_active(values)[0], active)
        ts = controls[1]
        saved_bar = jnp.zeros((len(ts), initial.size), initial.dtype) if isinstance(bar.ys, SymbolicZero) else bar.ys
        terminal_bar = jnp.zeros_like(initial) if isinstance(bar.y, SymbolicZero) else bar.y
        terminal_bar += jnp.sum(jnp.where((jnp.arange(len(ts)) >= tape.saved_count)[:, None], saved_bar, 0.), axis=0)

        def reverse_map(carry, initial):
            index, by, bp = carry
            index -= 1
            orbit_end, checkpoints = _orbit_replay(orbit_rhs, initial, rp, tape, index, method,
                checkpoint_steps=inner_segment, workspace=workspace)
            t, end = tape.times[index]
            by += jnp.sum(jnp.where(((ts > t) & (ts <= end))[:, None], saved_bar, 0.), axis=0)

            def kick_reverse(values):
                by, bp = values
                # Transpose the state result together with its preparation.
                # Error-only preparation must be dead before AD: an explicit
                # zero cotangent times a singular error derivative can be NaN.
                _, pullback = jax.vjp(lambda y, values: evaluate_collision(
                    (end, y, end-t, tape.increments[index]), prepare_active(values)[1]).y,
                    orbit_end, active)
                dy, dp = pullback(by)
                return dy, _add(bp, dp)

            by, bp = lax.cond(tape.kicks[index], kick_reverse, lambda values: values, (by, bp))
            orbit_start, orbit_stop = tape.offsets[index], tape.offsets[index+1]
            states, derivatives = checkpoints

            def reverse_orbit_segment(local):
                stop, by, bf, bp = local
                segment = jnp.maximum((stop-orbit_start-1) // inner_segment, 0)
                start = orbit_start + segment*inner_segment
                y, f = states[segment], derivatives[segment]
                state_bank = jnp.zeros((inner_segment, initial.size), initial.dtype)
                endpoint_bank = jnp.zeros_like(state_bank)
                increment_bank = jnp.zeros((inner_segment, len(constants[2]), initial.size), initial.dtype)

                def reconstruct(values):
                    i, y, f, state_bank, endpoint_bank, increment_bank = values
                    slot = i-start
                    t, h = tape.orbit_times[i]
                    trial = _trial_step(orbit_rhs, t, y, h, rp, f, *constants)
                    return (i+1, trial.y, trial.f1, state_bank.at[slot].set(y),
                        endpoint_bank.at[slot].set(trial.y), increment_bank.at[slot].set(trial.increments))

                _, _, _, state_bank, endpoint_bank, increment_bank = lax.while_loop(
                    lambda values: values[0] < stop, reconstruct,
                    (start, y, f, state_bank, endpoint_bank, increment_bank))

                def reverse_step(values):
                    i, by, bf, bp = values
                    i -= 1
                    slot = i-start
                    t, h = tape.orbit_times[i]
                    fraction = jnp.where(i == orbit_stop-1, tape.fractions[index], 1.)
                    dy, df, dp = _step_pullback(orbit_rhs, state_bank[slot], t, h,
                        endpoint_bank[slot], increment_bank[slot], rp, fraction,
                        jnp.empty((0,), initial.dtype), jnp.empty((0,), bool), jnp.empty((0,), bool),
                        by, bf, jnp.empty((0, initial.size), initial.dtype), method, "standard",
                        initial_fsal=(i == orbit_start))
                    return i, dy, df, _add(bp, rhs_setup_pullback(dp)[0])

                return lax.while_loop(lambda values: values[0] > start, reverse_step, local)

            _, by, _, bp = lax.while_loop(lambda values: values[0] > orbit_start,
                reverse_orbit_segment, (orbit_stop, by, jnp.zeros_like(initial), bp))
            return index, by, bp

        def reverse(carry):
            stop = carry[0]
            segment = jnp.maximum((stop-1) // outer_segment, 0)
            start = segment*outer_segment
            initial = tape.states[segment]
            if outer_segment == 1:
                return reverse_map(carry, initial)
            bank = jnp.zeros((outer_segment, initial.size), initial.dtype)

            def reconstruct(values):
                i, y, bank = values
                bank = bank.at[i-start].set(y)
                y, _ = _orbit_replay(orbit_rhs, y, rp, tape, i, method)
                t, end = tape.times[i]
                y = lax.cond(tape.kicks[i],
                    lambda y: evaluate_collision((end, y, end-t, tape.increments[i]), cp).y,
                    lambda y: y, y)
                return i+1, y, bank

            _, _, bank = lax.while_loop(lambda values: values[0] < stop, reconstruct, (start, initial, bank))
            return lax.while_loop(lambda values: values[0] > start,
                lambda values: reverse_map(values, bank[values[0]-1-start]), carry)

        _, by, bp = lax.while_loop(lambda values: values[0] > 0, reverse,
            (tape.count, terminal_bar, _zeros(active)))
        by += jnp.sum(jnp.where((jnp.arange(len(ts)) < tape.initial_saves)[:, None], saved_bar, 0.), axis=0)
        iterator = iter(bp)
        raw_bar = tree_util.tree_unflatten(raw_structure,
            [next(iterator) if flag else _zeros(value) for value, flag in zip(raw_leaves, activity.flags)])
        return by, raw_bar, _zeros(controls)

    integrate.defvjp(fwd, bwd, symbolic_zeros=True)
    return integrate(y0, raw, controls)
