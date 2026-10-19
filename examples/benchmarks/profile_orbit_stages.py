"""Isolate RK stage costs using identical fixed-step scans and FSAL reuse.

The sparse kernel is the shared production map. Other variants are experiments,
separate from the production solver. Diffrax's
low-level ``solver.step`` supplies a reference.
All kernels return the final state, accumulated embedded-error squares and a
finite-stage check. This excludes saving, events and adaptive controllers.
Run with ``python -m examples.benchmarks.profile_orbit_stages``.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from examples.benchmarks.benchmark_orbit_integrators import compile_solve, make_case, metadata
import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from essos.rk import RKStep, _A, _A8, _C, _C8, _ERROR, _ERROR8


def unrolled_step(rhs, t, y, h, f0, nodes, rows, error_weights, *, sparse=False):
    stages = [f0]
    for c, weights in zip(nodes, rows):
        trial = y + h * sum(a * k for a, k in zip(weights, stages) if not sparse or a != 0)
        stages.append(rhs(t + c * h, trial, None))
    y1 = trial if sparse else y + h * sum(a * k for a, k in zip(rows[-1], stages))
    error = h * sum(a * k for a, k in zip(error_weights, stages) if not sparse or a != 0)
    return RKStep(y1, error, jnp.stack(stages))


def rolled_step(rhs, t, y, h, f0, nodes, rows, error_weights, *, unroll=1):
    n = len(error_weights)
    coefficients = np.zeros((n, n))
    for i, row in enumerate(rows, 1):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, dtype=y.dtype)
    times = jnp.asarray((0.,) + nodes, dtype=y.dtype)
    stages = jnp.zeros((n,) + y.shape, dtype=y.dtype).at[0].set(f0)

    def stage(i, carry):
        stages, _ = carry
        trial = y + h * jnp.tensordot(coefficients[i], stages, axes=1)
        f = rhs(t + times[i] * h, trial, None)
        return stages.at[i].set(f), trial

    stages, y1 = jax.lax.fori_loop(1, n, stage, (stages, y), unroll=unroll)
    error = h * jnp.tensordot(jnp.asarray(error_weights, dtype=y.dtype), stages, axes=1)
    return RKStep(y1, error, stages)


def dot_step(rhs, t, y, h, f0, nodes, rows, error_weights):
    stages = [f0]
    for c, weights in zip(nodes, rows):
        trial = y + jnp.tensordot(h * jnp.asarray(weights, dtype=y.dtype), jnp.stack(stages), axes=1)
        stages.append(rhs(t + c * h, trial, None))
    error = jnp.tensordot(h * jnp.asarray(error_weights, dtype=y.dtype), jnp.stack(stages), axes=1)
    return RKStep(trial, error, jnp.stack(stages))


def increment_step(rhs, t, y, h, f0, nodes, rows, error_weights, *, barrier=False):
    """Classical RK increments; keep original derivatives for dense output/FSAL."""
    stages, increments = [f0], [h * f0]
    for c, weights in zip(nodes, rows):
        trial = y + sum(a * k for a, k in zip(weights, increments) if a != 0)
        f = rhs(t + c * h, trial, None)
        if barrier:
            f = jax.lax.optimization_barrier(f)
        stages.append(f)
        increments.append(h * f)
    error = sum(a * k for a, k in zip(error_weights, increments) if a != 0)
    return RKStep(trial, error, jnp.stack(stages))


def increment_loop_step(rhs, t, y, h, f0, nodes, rows, error_weights, *, unroll=1):
    n = len(error_weights)
    coefficients = np.zeros((n, n))
    for i, row in enumerate(rows, 1):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, dtype=y.dtype)
    times = jnp.asarray((0.,) + nodes, dtype=y.dtype)
    stages = jnp.zeros((n,) + y.shape, dtype=y.dtype).at[0].set(f0)
    increments = jnp.zeros_like(stages).at[0].set(h * f0)

    def stage(i, carry):
        stages, increments, _ = carry
        trial = y + jnp.tensordot(coefficients[i], increments, axes=1)
        f = rhs(t + times[i] * h, trial, None)
        return stages.at[i].set(f), increments.at[i].set(h * f), trial

    stages, increments, y1 = jax.lax.fori_loop(1, n, stage, (stages, increments, y), unroll=unroll)
    error = jnp.tensordot(jnp.asarray(error_weights, dtype=y.dtype), increments, axes=1)
    return RKStep(y1, error, stages)


def barrier_step(rhs, t, y, h, f0, nodes, rows, error_weights):
    stages = [f0]
    for c, weights in zip(nodes, rows):
        trial = y + h * sum(a * k for a, k in zip(weights, stages) if a != 0)
        stages.append(jax.lax.optimization_barrier(rhs(t + c * h, trial, None)))
    error = h * sum(a * k for a, k in zip(error_weights, stages) if a != 0)
    return RKStep(trial, error, jnp.stack(stages))


def pairwise_step(rhs, t, y, h, f0, nodes, rows, error_weights, *, increments=False):
    def combine(weights, values):
        terms = [a * k for a, k in zip(weights, values) if a != 0]
        while len(terms) > 1:
            terms = [terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i]
                     for i in range(0, len(terms), 2)]
        return terms[0]
    stages, values = [f0], [h * f0 if increments else f0]
    for c, weights in zip(nodes, rows):
        delta = combine(weights, values)
        trial = y + (delta if increments else h * delta)
        f = rhs(t + c * h, trial, None)
        stages.append(f)
        values.append(h * f if increments else f)
    error = combine(error_weights, values)
    return RKStep(trial, error if increments else h * error, jnp.stack(stages))


def accumulate_step(rhs, t, y, h, f0, nodes, rows, error_weights, *, unroll=1):
    """Accumulate each new derivative into future stages; no stage dot products."""
    n = len(error_weights)
    coefficients = np.zeros((n + 1, n))
    for i, row in enumerate(rows, 1):
        coefficients[i, :len(row)] = row
    coefficients[-1] = error_weights
    coefficients = jnp.asarray(coefficients, dtype=y.dtype)
    times = jnp.asarray((0.,) + nodes, dtype=y.dtype)
    stages = jnp.zeros((n,) + y.shape, dtype=y.dtype).at[0].set(f0)
    pending = coefficients[:, 0, None] * (h * f0)
    def stage(i, carry):
        stages, pending, _ = carry
        trial = y + pending[i]
        derivative = rhs(t + times[i] * h, trial, None)
        pending = pending + coefficients[:, i, None] * (h * derivative)
        return stages.at[i].set(derivative), pending, trial
    stages, pending, y1 = jax.lax.fori_loop(1, n, stage, (stages, pending, y), unroll=unroll)
    return RKStep(y1, pending[-1], stages)


def trial_step(kernel, rhs, t, y, h, f, nodes, rows, weights):
    if kernel == "dot":
        return dot_step(rhs, t, y, h, f, nodes, rows, weights)
    if kernel in ("increments", "increments_barrier"):
        return increment_step(rhs, t, y, h, f, nodes, rows, weights, barrier=kernel.endswith("barrier"))
    if kernel.startswith("increment_loop"):
        unroll = (True if kernel == "increment_loop_all" else
                  int(kernel.removeprefix("increment_loop")) if kernel != "increment_loop" else 1)
        return increment_loop_step(rhs, t, y, h, f, nodes, rows, weights, unroll=unroll)
    if kernel.startswith("pairwise"):
        return pairwise_step(rhs, t, y, h, f, nodes, rows, weights, increments=kernel.endswith("increments"))
    if kernel.startswith("accumulate"):
        unroll = int(kernel.removeprefix("accumulate")) if kernel != "accumulate" else 1
        return accumulate_step(rhs, t, y, h, f, nodes, rows, weights, unroll=unroll)
    if kernel == "barrier":
        return barrier_step(rhs, t, y, h, f, nodes, rows, weights)
    if kernel in ("unrolled", "sparse"):
        return unrolled_step(rhs, t, y, h, f, nodes, rows, weights, sparse=kernel == "sparse")
    unroll = int(kernel.removeprefix("rolled")) if kernel != "rolled" else 1
    return rolled_step(rhs, t, y, h, f, nodes, rows, weights, unroll=unroll)


def check_derivatives(method, kernel):
    """Check y/t/h/parameter derivatives with consistent FSAL input."""
    nodes, rows, weights = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    def evaluate(x, impl):
        y, t, h, parameter = x[:2], x[2], x[3], x[4]
        rhs = lambda t, y, args: parameter * jnp.array([jnp.sin(y[1] + t), -y[0]**2 + jnp.cos(t)])
        out = trial_step(impl, rhs, t, y, h, rhs(t, y, None), nodes, rows, weights)
        return jnp.concatenate([out.y, out.error, out.stages.ravel()])
    x = jnp.array([.7, -.3, .2, .1, .8])
    reference = jax.jacfwd(lambda x: evaluate(x, "sparse"))(x)
    forward = jax.jacfwd(lambda x: evaluate(x, kernel))(x)
    reverse = jax.jacrev(lambda x: evaluate(x, kernel))(x)
    np.testing.assert_allclose(forward, reference, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(reverse, reference, rtol=2e-10, atol=2e-12)
    return dict(forward_max_difference=float(jnp.max(jnp.abs(forward - reference))),
                reverse_max_difference=float(jnp.max(jnp.abs(reverse - reference))))


def make_scan(case, method, kernel, steps, *, batch_size=None):
    rhs, h = case["rhs"], case["maxtime"] / steps
    nodes, rows, weights = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    solver = diffrax.Dopri5() if method == "dopri5" else diffrax.Dopri8()
    term = diffrax.ODETerm(rhs)

    def single(y0):
        def step(carry, index):
            y, f, errors, finite = carry
            t = index * h
            if kernel == "diffrax":
                y1, error, dense, state, _ = solver.step(term, t, t + h, y, None, (False, f), False)
                f1 = state[1]
                all_finite = jnp.isfinite(dense["k"]).all()
            else:
                trial = trial_step(kernel, rhs, t, y, h, f, nodes, rows, weights)
                y1, error, f1 = trial.y, trial.error, trial.stages[-1]
                all_finite = jnp.isfinite(trial.stages).all()
            return (y1, f1, errors + jnp.sum(error**2), finite & all_finite), None

        initial = y0, rhs(0., y0, None), jnp.zeros((), y0.dtype), jnp.asarray(True)
        (y, _, errors, finite), _ = jax.lax.scan(step, initial, jnp.arange(steps))
        return y, errors, finite

    if batch_size is None:
        return jax.jit(jax.vmap(single))
    return jax.jit(lambda ys: jax.lax.map(single, ys, batch_size=batch_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--field", choices=("qa", "small"), default="qa")
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--kernels", nargs="+", choices=("unrolled", "sparse", "dot", "rolled", "rolled2", "rolled4", "diffrax",
                                                         "increments", "increments_barrier", "increment_loop", "increment_loop2", "barrier",
                                                         "increment_loop4", "increment_loop8", "increment_loop_all", "pairwise", "pairwise_increments",
                                                         "accumulate", "accumulate2", "accumulate4", "accumulate8"),
                        default=["sparse", "accumulate4", "diffrax"])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "orbit_stage_profile.json")
    options = parser.parse_args()
    if min(options.particles, options.steps) < 1 or options.repeats < 3:
        parser.error("use positive particle/step counts and at least three repeats")
    options.tolerance, options.accuracy_limit = None, None
    report = dict(metadata=metadata(options), steps=options.steps, rows=[])
    report["metadata"]["scope"] = __doc__
    report["metadata"]["source_sha256"]["examples/benchmarks/profile_orbit_stages.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for unused in ("adaptive_controllers", "tolerance", "reference_tolerance", "accuracy_limit"):
        report["metadata"].pop(unused, None)
    rng = np.random.default_rng(42)
    derivative_checks = {}
    for method in ("dopri5", "dopri8"):
        for kernel in options.kernels:
            if kernel != "diffrax":
                print(f"{method}/{kernel}: checking forward/reverse derivatives...", flush=True)
                derivative_checks[method, kernel] = check_derivatives(method, kernel)
    for model in options.models:
        case = make_case(model, options.particles, options.field)
        for method in ("dopri5", "dopri8"):
            executables, outputs, rows = {}, {}, {}
            for kernel in options.kernels:
                print(f"{model}/{method}/{kernel}: compiling...", flush=True)
                fn = make_scan(case, method, kernel, options.steps)
                executable, output, timing = compile_solve(fn, case["y0"])
                y, errors, finite = jax.device_get(output)
                if not np.all(finite) or not np.isfinite(y).all() or not np.isfinite(errors).all():
                    raise RuntimeError("non-finite profile orbit")
                executables[kernel], outputs[kernel] = executable, y
                rows[kernel] = dict(model=model, method=method, kernel=kernel, **timing, samples_ms=[],
                                    derivative_checks=derivative_checks.get((method, kernel)),
                                    rhs_evaluations_per_particle=1 + options.steps * (6 if method == "dopri5" else 13),
                                    hlo_characters=len(executable.as_text()), cost_analysis=executable.cost_analysis())
            for kernel, y in outputs.items():
                delta = np.abs(y - outputs.get("diffrax", next(iter(outputs.values()))))
                # Stage reduction order changes roundoff, but not the RK formula.
                if max(np.max(delta[:, :3]), np.max(delta[:, 3:]) / case["speed"]) > 1e-10:
                    raise RuntimeError(f"{kernel} failed trajectory parity")
                rows[kernel]["position_difference_m"] = float(np.max(delta[:, :3]))
                rows[kernel]["velocity_difference_over_speed"] = float(np.max(delta[:, 3:]) / case["speed"])
                jax.block_until_ready(executables[kernel](case["y0"]))
            for _ in range(options.repeats):
                for kernel in rng.permutation(tuple(executables)):
                    start = perf_counter()
                    jax.block_until_ready(executables[kernel](case["y0"]))
                    rows[kernel]["samples_ms"].append(1000 * (perf_counter() - start))
            for row in rows.values():
                row["median_ms"] = float(np.median(row["samples_ms"]))
                row["q25_ms"], row["q75_ms"] = [float(x) for x in np.quantile(row["samples_ms"], [.25, .75])]
                report["rows"].append(row)
                print(f"  {row['kernel']:10s} {row['median_ms']:9.3f} ms; XLA {row['compile_s']:.3f} s; "
                      f"HLO {row['hlo_characters']} chars", flush=True)
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {options.output}", flush=True)


if __name__ == "__main__":
    main()
