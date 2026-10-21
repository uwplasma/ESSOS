"""Collisionless native checkpointed reverse timings and local RHS diagnostics.

No Diffrax or collision comparisons. Uses production GC/Lorentz RHS, field
classes, accepted-step adjoint, and frozen-grid replay. Independent local
kernels explain costs but are not an additive profile of the fused solver.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")

import jax
import jax.numpy as jnp
import numpy as np

from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import GuidingCenter, Lorentz, Particles
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart
from essos.orbit_error import OrbitErrorNorm
from essos.particle_losses import mean_final_radius
from essos.rk import solve_rk, _partition_rhs, _trial_step, _C8, _A8, _ERROR8
from essos.rk_ad import replay_accepted

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("essos/rk.py", "essos/rk_ad.py", "essos/rk_ad_rhs.py", "essos/rk_ad_biot_savart.py", "essos/dynamics.py", "essos/fields.py",
           "essos/coils.py", "essos/orbit_error.py", "essos/particle_losses.py",
           "essos/electric_field.py", "examples/benchmarks/profile_native_reverse.py")


def fingerprints():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def timed(function, arguments, repeats):
    """Compile/warm before taking blocked medians; retain each sample."""
    function = jax.jit(function)
    start = perf_counter()
    executable = function.lower(*arguments).compile()
    compile_seconds = perf_counter()-start
    value = jax.block_until_ready(function(*arguments))
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(function(*arguments))
        samples.append(1000*(perf_counter()-start))
    memory = executable.memory_analysis()
    return function, value, dict(median_ms=float(np.median(samples)), samples_ms=samples,
        compile_seconds=compile_seconds, compiler_buffer_bytes=int(memory.argument_size_in_bytes
            + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes))


def build(options, model, horizon):
    base = Coils.from_json(str(ROOT/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"))
    theta = jnp.asarray(base.dofs)
    reference = BiotSavart(base)
    axis = float(reference.r_axis), float(reference.z_axis)
    points = jnp.column_stack((jnp.linspace(1.23, 1.27, options.particles),
                               jnp.zeros((options.particles, 2))))
    pitches = jnp.linspace(.35, .65, options.particles)
    speed = float(np.sqrt(2*4000*ONE_EV/M))
    norm = OrbitErrorNorm(options.tolerance, options.tolerance, speed,
                         velocity_components=1 if model == "gc" else 3)
    times = jnp.linspace(0., horizon, options.saves)
    scale = jnp.array([1.]*3+[speed]*(1 if model == "gc" else 3))

    def initialize(parameters):
        field = BiotSavart(base.with_dofs(parameters))
        birth = Particles(points, mass=M, charge=Q, energy=4000*ONE_EV,
            initial_vparallel_over_v=pitches, field=field if model == "lorentz" else None,
            phase_angle_full_orbit=jnp.pi)
        if model == "gc":
            initial = jnp.column_stack((points, birth.initial_vparallel))
            rhs = lambda t, y, a: GuidingCenter(t, y, (field, birth, Electric_field_zero()))
        else:
            initial = jnp.column_stack((birth.initial_xyz_fullorbit, birth.initial_vxvyvz))
            rhs = lambda t, y, a: Lorentz(t, y, (field, birth))
        return initial, rhs

    def run(parameters, adjoint="none", record=False, initial_override=None):
        initial, rhs = initialize(parameters)
        if initial_override is not None:
            initial = initial_override
        def one(y):
            return solve_rk(rhs, y, t1=horizon, dt0=horizon/4, ts=times,
                method="dopri8", dense_output="standard", rtol=options.tolerance, atol=options.tolerance,
                error_norm=norm, max_steps=options.max_steps, adjoint=adjoint,
                adjoint_checkpoint_steps=options.checkpoint_steps,
                adjoint_parameter_accumulation=getattr(options, "parameter_accumulation", "segment"),
                adjoint_max_accepted_steps=options.accepted_capacity, _record_ad=record)
        return jax.vmap(one)(initial)

    def cost(parameters, adjoint):
        out = run(parameters, adjoint)
        return mean_final_radius(out.ys, *axis), (out.ys, out.stats, out.status)

    def frozen(parameters, tapes, initial_override=None):
        initial, rhs = initialize(parameters)
        if initial_override is not None:
            initial = initial_override
        ys, _ = jax.vmap(lambda y, tape: replay_accepted(rhs, y, None, tape, times,
                                                       method="dopri8"))(initial, tapes)
        return mean_final_radius(ys, *axis), ys

    def state_cost(states):
        out = run(theta, "accepted", initial_override=states)
        return mean_final_radius(out.ys, *axis)

    return theta, cost, run, frozen, initialize, scale, state_cost


def native_cache_diagnostics(initialize, theta):
    """Record the actual compact residuals used by the production adjoint."""
    from essos.rk_ad_rhs import prepare_reverse_rhs, prepare_rhs_linearization
    initial, rhs = initialize(theta)
    setup, stage = _partition_rhs(rhs, jnp.array(0.), initial[0], None)
    raw = setup.consts, stage.consts
    def prepare(raw):
        return jax.core.eval_jaxpr(setup.jaxpr, raw[0]), raw[1]
    def evaluate(t, y, parameters):
        values, constants = parameters
        return jax.core.eval_jaxpr(stage.jaxpr, constants, *values, t, y)[0]
    prepare, evaluate = prepare_reverse_rhs(prepare, evaluate, jnp.array(0.), initial[0], raw)
    parameters = prepare(raw)
    cached = prepare_rhs_linearization(evaluate, jnp.array(0.), initial[0], parameters)
    disabled = prepare_rhs_linearization(evaluate, jnp.array(0.), initial[0], parameters,
        cache_field_outputs=False)
    values = () if cached is None else jax.tree.leaves(cached[2])
    return dict(enabled=cached is not None, packed_shapes=[list(value.shape) for value in values],
        scalars=sum(int(value.size) for value in values),
        bytes=sum(int(value.size*value.dtype.itemsize) for value in values),
        enabled_without_named_outputs=disabled is not None,
        scope="Per particle per RK stage; contains small state/field values, not field-sized geometry")


def local_kernels(initialize, theta, repeats):
    """Stage costs at both birth states; all physical coil DOFs remain active.

    Full-RHS VJP includes coil setup per call; prepared-stage VJP exposes all
    prepared cotangents. Production instead accumulates these per RK step.
    Thus these kernels measure arithmetic costs, not additive solver phases.
    """
    initial, rhs = initialize(theta)
    t = jnp.array(0., dtype=theta.dtype)
    setup, stage = _partition_rhs(rhs, t, initial[0], None)
    parameters = (jax.core.eval_jaxpr(setup.jaxpr, setup.consts), stage.consts)

    def prepared_rhs(y, parameters):
        p, constants = parameters
        return jax.core.eval_jaxpr(stage.jaxpr, constants, *p, t, y)[0]

    seeds = jnp.ones_like(initial)
    def full_rhs(states, theta):
        _, f = initialize(theta)
        return jax.vmap(lambda y: f(t, y, None))(states)
    def full_vjp(states, theta, seeds):
        return jax.vjp(full_rhs, states, theta)[1](seeds)
    def state_vjp(states, parameters, seeds):
        return jax.vjp(lambda states: jax.vmap(prepared_rhs, in_axes=(0, None))(
            states, parameters), states)[1](seeds)[0]
    def prepared_vjp(states, parameters, seeds):
        return jax.vjp(lambda states, parameters: jax.vmap(prepared_rhs, in_axes=(0, None))(
            states, parameters), states, parameters)[1](seeds)
    def trial(states, parameters, h):
        def one(y):
            f0 = prepared_rhs(y, parameters)
            return _trial_step(lambda t, y, p: prepared_rhs(y, p), t, y, h,
                               parameters, f0, _C8, _A8, _ERROR8)
        return jax.vmap(one)(states)
    result = {}
    for name, function, arguments in (
        ("rhs_prepared_forward", lambda y, p: jax.vmap(prepared_rhs, in_axes=(0, None))(y, p), (initial, parameters)),
        ("rhs_state_vjp", state_vjp, (initial, parameters, seeds)),
        ("rhs_state_and_prepared_parameter_vjp", prepared_vjp, (initial, parameters, seeds)),
        ("rhs_with_coil_setup_forward", full_rhs, (initial, theta)),
        ("rhs_with_coil_setup_vjp", full_vjp, (initial, theta, seeds)),
        ("rk_trial_with_initial_rhs", trial, (initial, parameters, jnp.array(1e-9))),
    ):
        _, value, measurement = timed(function, arguments, repeats)
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(value)):
            raise RuntimeError(f"nonfinite kernel {name}")
        result[name] = measurement
    result["prepared_parameter_shapes"] = [list(x.shape) for x in jax.tree_util.tree_leaves(parameters)]
    # A local pullback with stored residuals distinguishes derivative arithmetic
    # from rebuilding RHS intermediates. Production deliberately does not keep
    # these field-sized residuals across every RK stage.
    joint_rhs = jax.jit(lambda y, p: jax.vmap(prepared_rhs, in_axes=(0, None))(y, p))
    _, rhs_pullback = jax.block_until_ready(jax.vjp(joint_rhs, initial, parameters))
    _, _, result["rhs_cached_pullback"] = timed(lambda pb, seed: pb(seed),
        (rhs_pullback, seeds), repeats)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--horizon-factors", type=float, nargs="+", default=[1., 10.])
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--saves", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--gc-horizon", type=float, default=2e-6)
    parser.add_argument("--lorentz-horizon", type=float, default=2e-7)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--accepted-capacity", type=int, default=1024)
    parser.add_argument("--checkpoint-steps", type=int, default=16)
    parser.add_argument("--parameter-accumulation", choices=("step", "segment"), default="segment")
    options = parser.parse_args()
    if options.output.exists() or jax.default_backend() != "cpu":
        parser.error("choose a fresh output path and the CPU backend")
    if not 1 <= options.particles <= 8 or options.saves < 2 or options.repeats < 3:
        parser.error("use 1..8 particles, >=2 saves and >=3 repeats")
    positive = (*options.horizon_factors, options.gc_horizon, options.lorentz_horizon, options.tolerance,
                options.max_steps, options.accepted_capacity, options.checkpoint_steps)
    if any(not np.isfinite(x) or x <= 0 for x in positive):
        parser.error("horizons, tolerances, factors and work/storage limits must be positive finite")
    hashes = fingerprints()
    report = dict(complete=False, metadata=dict(backend=jax.default_backend(), jax=jax.__version__,
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
        source_sha256=hashes,
        input_sha256=hashlib.sha256((ROOT/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json").read_bytes()).hexdigest(),
        method="Native adaptive Dopri8, I controller, standard dense output; accepted-only checkpointed AD",
        parameters="All 136 bundled QA coil Fourier/normalized-current DOFs; fixed births and reference axis",
        births="x=linspace(1.23,1.27) m; y=z=0; pitch=linspace(.35,.65); 4 keV alphas; Lorentz gyrophase pi",
        gc_state="Production collisionless energy-based (x,y,z,v_parallel); split-collision GC uses evolving mu instead",
        scope="Blocked warm CPU calls, no collisions, no Diffrax, no particle chunking; compilation separate",
        phase_policy="Reverse reuses a baseline pullback; linearization uses public jax.vjp. Independent phases need not sum to fused value_and_grad.",
        kernel_policy="Independent local kernels at birth states use ordinary field AD, not the native-only prepared Biot-Savart transpose. Full RHS VJP repeats coil setup; prepared VJP exposes all prepared cotangents. Not an additive breakdown of the whole backward pass.",
        state_only_policy="Diagnostic reverse with identical fields, initial states, accepted schedules and objective; differentiate initial states only, holding coils fixed. This is not the coil-optimization gradient.",
        work_count_policy="Nominal algorithmic counts per particle. Vmapped masked lanes and compiler dead-code elimination can change physical kernel execution.",
        memory_scope="Compiler buffers only, not process peak RSS"), rows=[])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for model in options.models:
        horizon0 = options.gc_horizon if model == "gc" else options.lorentz_horizon
        for factor in options.horizon_factors:
            horizon = horizon0*factor
            print(f"Measuring collisionless {model}: horizon {horizon:g} s", flush=True)
            theta, cost, run, frozen, initialize, scale, state_cost = build(options, model, horizon)
            primal, (value, aux), forward_time = timed(lambda p: cost(p, "none"), (theta,), options.repeats)
            accepted = jax.jit(lambda p: cost(p, "accepted"))
            linear_value, pullback, linear_aux = jax.block_until_ready(jax.vjp(accepted, theta, has_aux=True))
            _, reverse_gradient, reverse_time = timed(lambda pb, seed: pb(seed)[0],
                (pullback, jnp.ones_like(value)), options.repeats)
            _, ((fused_value, fused_aux), gradient), fused_time = timed(
                jax.value_and_grad(lambda p: cost(p, "accepted"), has_aux=True), (theta,), options.repeats)
            linear_samples = []
            for _ in range(options.repeats):
                start = perf_counter()
                jax.block_until_ready(jax.vjp(accepted, theta, has_aux=True))
                linear_samples.append(1000*(perf_counter()-start))
            np.testing.assert_allclose(linear_value, value, rtol=2e-12, atol=1e-25)
            np.testing.assert_allclose(fused_value, value, rtol=2e-12, atol=1e-25)
            np.testing.assert_allclose(reverse_gradient, gradient, rtol=2e-9, atol=1e-11)
            ys, stats, status = aux
            for candidate in (linear_aux, fused_aux):
                np.testing.assert_allclose((ys-candidate[0])/scale, 0., rtol=0., atol=2e-12)
                for a, b in zip(jax.tree_util.tree_leaves((stats, status)), jax.tree_util.tree_leaves(candidate[1:])):
                    np.testing.assert_array_equal(a, b)
            if np.any(status) or not np.isfinite(gradient).all():
                raise RuntimeError("failed solve or nonfinite gradient")
            _, tapes = jax.block_until_ready(jax.jit(lambda p: run(p, record=True))(theta))
            replay, (replay_value, replay_ys), replay_time = timed(frozen, (theta, tapes), options.repeats)
            np.testing.assert_allclose((ys-replay_ys)/scale, 0., rtol=0., atol=2e-12)
            direction = jnp.asarray(np.random.default_rng(12).normal(size=theta.size))
            direction /= jnp.linalg.norm(direction)
            expected = float(jnp.dot(gradient, direction))
            _, tangent = jax.block_until_ready(jax.jit(lambda p, d: jax.jvp(
                lambda p: frozen(p, tapes)[0], (p,), (d,)))(theta, direction))
            np.testing.assert_allclose(tangent, expected, rtol=2e-7, atol=1e-10)
            fd = []
            for epsilon in (1e-5, 3e-6, 1e-6):
                finite = float((replay(theta+epsilon*direction, tapes)[0]
                               - replay(theta-epsilon*direction, tapes)[0])/(2*epsilon))
                fd.append(dict(epsilon=epsilon, ad=expected, fd=finite,
                    relative_error=abs(expected-finite)/max(abs(expected), abs(finite), 1e-30)))
            if min(item["relative_error"] for item in fd) > 2e-4:
                raise RuntimeError(f"frozen-grid directional FD failed: {fd}")
            initial, _ = initialize(theta)
            state_value, state_pullback = jax.block_until_ready(jax.vjp(jax.jit(state_cost), initial))
            _, state_gradient, state_reverse_time = timed(lambda pb, seed: pb(seed)[0],
                (state_pullback, jnp.ones_like(state_value)), options.repeats)
            state_direction = jnp.asarray(np.random.default_rng(23).normal(size=initial.shape))*scale
            _, state_tangent = jax.block_until_ready(jax.jit(lambda y, d: jax.jvp(
                lambda y: frozen(theta, tapes, y)[0], (y,), (d,)))(initial, state_direction))
            np.testing.assert_allclose(state_value, value, rtol=2e-12, atol=1e-25)
            np.testing.assert_allclose(jnp.sum(state_gradient*state_direction), state_tangent,
                                       rtol=2e-7, atol=1e-10)
            row = dict(model=model, horizon=horizon, horizon_factor=factor, parameters=int(theta.size),
                value=float(value), gradient=np.asarray(gradient).tolist(), gradient_l2=float(jnp.linalg.norm(gradient)),
                stats={k: np.asarray(v).tolist() for k, v in stats._asdict().items()}, status=np.asarray(status).tolist(),
                forward=forward_time, forward_linearization=dict(median_ms=float(np.median(linear_samples)), samples_ms=linear_samples),
                reverse=reverse_time, value_and_grad=fused_time, frozen_replay=replay_time,
                reverse_over_forward=reverse_time["median_ms"]/forward_time["median_ms"],
                value_and_grad_over_forward=fused_time["median_ms"]/forward_time["median_ms"],
                directional_jvp=float(tangent), directional_fd=fd,
                fixed_coils_initial_state_reverse=state_reverse_time,
                initial_state_gradient=np.asarray(state_gradient).tolist(),
                initial_state_directional_jvp=float(state_tangent),
                forward_rhs_calls_per_particle=(1+13*np.asarray(stats.steps)).tolist(),
                backward_replay_rhs_calls_per_particle=(13*np.asarray(stats.accepted)).tolist(),
                backward_rhs_vjp_calls_per_particle=(1+13*np.asarray(stats.accepted)).tolist(),
                backward_setup_vjp_calls_per_particle=((np.asarray(stats.accepted)+options.checkpoint_steps-1)
                    // options.checkpoint_steps if options.parameter_accumulation == "segment"
                    else np.asarray(stats.accepted)).tolist())
            if factor == options.horizon_factors[0]:
                print(f"Profiling {model} local RHS kernels", flush=True)
                row["local_kernels"] = local_kernels(initialize, theta, max(options.repeats, 21))
                row["native_rhs_cache"] = native_cache_diagnostics(initialize, theta)
            report["rows"].append(row)
            options.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(json.dumps({k: row[k] for k in ("model", "horizon", "stats", "reverse_over_forward", "value_and_grad_over_forward")}
                | {k+"_ms": row[k]["median_ms"] for k in ("forward", "reverse", "value_and_grad", "frozen_replay")}), flush=True)
    if fingerprints() != hashes:
        raise RuntimeError("sources changed during measurement")
    report["complete"] = True
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
