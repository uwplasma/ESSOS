"""Paired particle-step experiments with identical complete solve settings.

This pairs synchronized warm runs in one process. Before and after use the
same physical norm, I-controller, FSAL, saves and events. Diffrax is a reference
backend for stage experiments; adaptive Diffrax uses its ESSOS PI controller.
The default compares legacy derivative storage with the current native RK
increment loop, keeping the field classes and RHS fixed for GC and Lorentz.
--comparison rhs_setup isolates generic RHS preparation: before recomputes
parameter-only work inside the driver; after prepares it once per solve.
Other stage comparisons preserve rejected experiments for review. With
--derivatives, also compare value-and-reverse-derivative costs for one RK step,
including state and RHS time-scale sensitivities. This is not a full adjoint.
--comparison execution measures old/new native driver stage execution and the
Diffrax counterpart, all with the current unchanged field provider. The public
accepted-step derivative map is unchanged by that forward execution comparison.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from time import perf_counter

from examples.benchmarks.benchmark_orbit_integrators import (
    check_output, compile_solve, make_case, make_solve, metadata)
import jax
import jax.numpy as jnp
import numpy as np

from essos.rk import RKStep
import essos.rk as rk
from examples.benchmarks.profile_orbit_stages import accumulate_step


def accumulation_kernel(rhs, t, y, h, args, f0, nodes, rows, error_weights):
    f0 = rhs(t, y, args) if f0 is None else f0
    return accumulate_step(lambda t, y, unused: rhs(t, y, args), t, y, h, f0, nodes, rows, error_weights, unroll=4)


def original_kernel(rhs, t, y, h, args, f0, nodes, rows, error_weights):
    """The native map before optimization, preserved for paired measurements."""
    stages = [rhs(t, y, args) if f0 is None else f0]
    for c, weights in zip(nodes, rows):
        trial = y + h * sum(a * k for a, k in zip(weights, stages) if a != 0)
        stages.append(rhs(t + c * h, trial, args))
    error = h * sum(a * k for a, k in zip(error_weights, stages) if a != 0)
    return RKStep(trial, error, jnp.stack(stages))


def derivative_storage_trial(rhs, t, y, h, args, f0, nodes, rows, error_weights):
    """Adapt an old derivative-storage experiment to the forward trial API."""
    step = rk._rk_step(rhs, t, y, h, args, f0, nodes, rows, error_weights)
    return rk._RKTrial(step.y, step.error, h * step.stages, step.stages[-1])


@contextmanager
def kernel_for_compilation(variant, comparison="execution", candidate="trial_production"):
    # This isolated benchmark process substitutes only the numerical map while
    # tracing. AOT executables then retain that map; warm calls need no patches.
    current = rk._rk_step
    current_trial = rk._trial_step
    current_prepare = rk._prepare_rhs
    if comparison == "stages":
        rk._rk_step = original_kernel if variant != "after" else accumulation_kernel
        rk._trial_step = derivative_storage_trial
    elif comparison == "rhs_setup":
        if variant == "before":
            rk._prepare_rhs = lambda rhs, t0, y0, args: rhs
    elif comparison == "execution":
        from examples.benchmarks.profile_rk_candidates import candidate_kernel
        if candidate.startswith("trial_"):
            from examples.benchmarks.profile_rk_increment_storage import increment_kernel
            rk._rk_step = original_kernel
            rk._trial_step = ((current_trial if candidate == "trial_production" else increment_kernel(candidate))
                              if variant == "after" else derivative_storage_trial)
        else:
            rk._rk_step = original_kernel if variant != "after" else candidate_kernel(candidate)
            rk._trial_step = derivative_storage_trial
    try:
        yield
    finally:
        rk._rk_step = current
        rk._trial_step = current_trial
        rk._prepare_rhs = current_prepare


def measure(executables, initial, repeats, rng):
    samples = {name: [] for name in executables}
    for executable in executables.values():
        jax.block_until_ready(executable(initial))
    for _ in range(repeats):
        for name in rng.permutation(tuple(executables)):
            start = perf_counter()
            jax.block_until_ready(executables[name](initial))
            samples[name].append(1000 * (perf_counter() - start))
    return {name: dict(samples_ms=values, median_ms=float(np.median(values)),
                       q25_ms=float(np.quantile(values, .25)), q75_ms=float(np.quantile(values, .75)))
            for name, values in samples.items()}


def scales_for(case):
    return jnp.array([1.] * 3 + [case["speed"]] * case["velocity_components"])


def derivative_function(case, method, prepare=False):
    constants = (rk._C, rk._A, rk._ERROR) if method == "dopri5" else (rk._C8, rk._A8, rk._ERROR8)
    scales = scales_for(case)
    weights = jnp.linspace(.5, 1., len(scales))
    def objective(normalized, parameter):
        y = normalized * scales
        rhs = lambda t, y, args: parameter * case["rhs"](t, y, args)
        if prepare:
            rhs = rk._prepare_rhs(rhs, jnp.asarray(0.), y, None)
        out = rk._trial_step(rhs, 0., y, case["dt0"], None, rhs(0., y, None), *constants)
        return jnp.dot(out.y / scales, weights)
    return jax.jit(jax.vmap(lambda y: jax.value_and_grad(objective, argnums=(0, 1))(y, jnp.asarray(1.))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--methods", nargs="+", choices=("dopri5", "dopri8"), default=["dopri5", "dopri8"])
    parser.add_argument("--modes", nargs="+", choices=("fixed", "adaptive"), default=["fixed", "adaptive"])
    parser.add_argument("--field", choices=("qa", "small"), default="qa")
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--derivatives", action="store_true")
    parser.add_argument("--comparison", choices=("stages", "execution", "rhs_setup"), default="execution")
    parser.add_argument("--candidate", default="trial_production", help="execution experiment kernel")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "native_stage_modes.json")
    options = parser.parse_args()
    if options.comparison == "execution" and options.candidate.startswith("while") and options.derivatives:
        parser.error("the direct while-loop experiment does not support reverse AD")
    if options.particles < 1 or options.repeats < 3:
        parser.error("use positive particle counts and at least three repeats")
    options.error_control, options.length_scale, options.accuracy_limit = "physics", 1., 1e-6
    report = dict(metadata=metadata(options), rows=[], derivative_rows=[])
    report["metadata"]["scope"] = __doc__
    report["metadata"]["comparison"] = options.comparison
    if options.comparison == "rhs_setup":
        report["metadata"]["native_rhs_preparation"] = {"before": False, "after": True}
    report["metadata"]["execution_candidate"] = options.candidate if options.comparison == "execution" else None
    report["metadata"]["production_stage_mode"] = "same RK increment loop for GC and Lorentz; legacy stage experiments are separate"
    report["metadata"]["source_sha256"]["examples/benchmarks/benchmark_native_stage_modes.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["metadata"]["source_sha256"]["examples/benchmarks/profile_rk_candidates.py"] = hashlib.sha256(Path(__file__).with_name("profile_rk_candidates.py").read_bytes()).hexdigest()
    report["metadata"]["source_sha256"]["examples/benchmarks/profile_rk_increment_storage.py"] = hashlib.sha256(Path(__file__).with_name("profile_rk_increment_storage.py").read_bytes()).hexdigest()
    rng = np.random.default_rng(42)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        options.output.write_text(json.dumps(report, indent=2) + "\n")
    for model in options.models:
        case = make_case(model, options.particles, options.field)
        print(f"{model}: compiling tight Dopri8 reference...", flush=True)
        reference_fn, reference_out, _ = compile_solve(make_solve(case, "diffrax_dopri8", True, 1e-12), case["y0"])
        reference, _ = check_output(reference_out, case["maxtime"])
        del reference_fn, reference_out
        for method in options.methods:
            for mode in options.modes:
                executables, rows, trajectories = {}, {}, {}
                for variant in ("before", "after", "diffrax"):
                    print(f"{model}/{method}/{mode}/{variant}: compiling...", flush=True)
                    implementation = f"{'diffrax' if variant == 'diffrax' else 'native'}_{method}"
                    fn = make_solve(case, implementation, mode == "adaptive", options.tolerance, "physics")
                    with kernel_for_compilation(variant, options.comparison, options.candidate):
                        executable, output, timing = compile_solve(fn, case["y0"])
                    trajectory, counts = check_output(output, case["maxtime"])
                    difference = np.abs(trajectory - reference)
                    position_error = float(np.max(difference[..., :3]))
                    velocity_error = float(np.max(difference[..., 3:]) / case["speed"])
                    if max(position_error, velocity_error) > options.accuracy_limit:
                        raise RuntimeError(f"{variant} failed the accuracy gate")
                    if mode == "fixed" and not np.all(counts[:, 1] == 256):
                        raise RuntimeError("fixed comparison must use 256 accepted steps")
                    rows[variant] = dict(model=model, method=method, mode=mode, variant=variant, **timing,
                                         counts_per_particle=counts.tolist(), accepted_mean=float(counts[:, 1].mean()),
                                         rejected_mean=float(counts[:, 2].mean()), position_error_m=position_error,
                                         velocity_error_over_speed=velocity_error,
                                         velocity_error_m_s=velocity_error * case["speed"])
                    executables[variant], trajectories[variant] = executable, trajectory
                # Isolate implementation changes from changes in mathematical
                # method or error control, even on adaptively chosen time grids.
                np.testing.assert_allclose(trajectories["after"] / scales_for(case),
                                           trajectories["before"] / scales_for(case), rtol=0., atol=1e-10)
                timings = measure(executables, case["y0"], options.repeats, rng)
                for variant, row in rows.items():
                    row.update(timings[variant])
                    report["rows"].append(row)
                    print(f"  {variant:8s} {row['median_ms']:9.3f} ms; "
                          f"steps {row['accepted_mean']:.3f}+{row['rejected_mean']:.3f}; "
                          f"position error {row['position_error_m']:.2e} m", flush=True)
                save()
                executables.clear()
            if options.derivatives:
                executables, outputs, rows = {}, {}, {}
                initial = case["y0"] / scales_for(case)
                for variant in ("before", "after"):
                    print(f"{model}/{method}/{variant}: compiling physical step derivatives...", flush=True)
                    fn = derivative_function(case, method, prepare=options.comparison == "rhs_setup")
                    with kernel_for_compilation(variant, options.comparison, options.candidate):
                        executable, output, timing = compile_solve(fn, initial)
                    outputs[variant], executables[variant] = output, executable
                    memory = executable.memory_analysis()
                    rows[variant] = dict(model=model, method=method, variant=variant, **timing,
                                         temporary_bytes=getattr(memory, "temp_size_in_bytes", None),
                                         scope="one RK step: scalar value and reverse derivatives wrt initial state and RHS time scale")
                for a, b in zip(jax.tree.leaves(outputs["after"]), jax.tree.leaves(outputs["before"])):
                    np.testing.assert_allclose(a, b, rtol=3e-10, atol=2e-12)
                timings = measure(executables, initial, options.repeats, rng)
                for variant, row in rows.items():
                    row.update(timings[variant])
                    report["derivative_rows"].append(row)
                    print(f"  derivative/{variant:8s} {row['median_ms']:9.3f} ms; "
                          f"XLA {row['compile_s']:.3f} s; temp bytes {row['temporary_bytes']}", flush=True)
                save()
                executables.clear()
    print(f"Saved {options.output}", flush=True)


if __name__ == "__main__":
    main()
