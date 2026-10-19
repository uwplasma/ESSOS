"""Fresh-process CPU RAM and value/gradient timing versus particle count.

Uses the same QA orbit/objective construction as benchmark_native_ad. Each
worker compiles one scalar value_and_grad, verifies complete finite results,
and records its full gradient and accepted/rejected counts. --reference also
checks values, counts and gradients against an earlier implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")

import jax
import numpy as np

from .benchmark_native_ad import build_case, compiler_memory, json_safe, ROOT, NAMES
from .measure_native_ad_memory import _resident_memory, _reset_resident_peak
from .validate_coil_orbit_loss import reference_wall


def fingerprints():
    names = ["essos/rk.py", "essos/rk_ad.py", "essos/dynamics.py",
             "essos/fields.py", "essos/coils.py", "essos/particle_losses.py",
             "essos/orbit_error.py", "essos/electric_field.py",
             "examples/benchmarks/measure_native_ad_memory.py",
             "examples/benchmarks/benchmark_native_ad.py",
             "examples/benchmarks/benchmark_native_ad_scaling.py"]
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def key(row):
    return tuple(row[name] for name in ("particles", "model", "method", "backend"))


def worker(args):
    import resource
    if sys.platform != "linux" or jax.default_backend() != "cpu":
        raise ValueError("this RSS benchmark requires Linux CPU")
    boundary = None if args.objective_index in (0, 1) else reference_wall(
        angular_interpolation="periodic_cubic").classifier
    theta, cost, _, _, horizon, labels = build_case(
        args, args.model, args.method, args.backend, boundary,
        objective_index=args.objective_index)
    def objective(theta):
        value, (_, counts, status) = cost(theta)
        return value, (counts, status)
    evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
    setup_peak = _resident_memory()["VmHWM"]
    start = perf_counter()
    executable = evaluate.lower(theta).compile()
    compile_seconds = perf_counter()-start
    compile_peak = max(_resident_memory()["VmHWM"],
                       int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
    after_compile_rss = _reset_resident_peak()
    result = jax.block_until_ready(evaluate(theta))
    (value, (counts, status)), gradient = jax.tree.map(np.asarray, result)
    first_peak = _resident_memory()["VmHWM"]
    if np.any(status) or not np.isfinite(gradient).all() or not np.isfinite(value):
        raise RuntimeError("incomplete/nonfinite scaling case")
    row = dict(particles=args.particles, model=args.model, method=args.method,
               backend=args.backend, horizon=horizon, objective=NAMES[args.objective_index],
               value=float(value), gradient=gradient.tolist(), parameter_labels=labels,
               counts_per_particle=counts.tolist(), status=status.tolist(),
               setup_peak_rss_bytes=setup_peak, compile_seconds=compile_seconds,
               after_compile_peak_rss_bytes=compile_peak,
               after_compile_resident_rss_bytes=after_compile_rss,
               first_execution_peak_rss_bytes=first_peak,
               compiler_memory=compiler_memory(executable))
    if args.reference:
        reference = json.loads(args.reference.read_text())
        previous = next(item for item in reference["rows"] if key(item) == key(row))
        np.testing.assert_array_equal(counts, previous["counts_per_particle"])
        np.testing.assert_allclose(value, previous["value"], rtol=2e-12, atol=1e-25)
        expected = np.asarray(previous["gradient"])
        difference = float(np.linalg.norm(gradient-expected)/max(np.linalg.norm(expected), 1e-30))
        if not np.isfinite(difference) or difference > 2e-9:
            raise RuntimeError(f"reference gradient mismatch: {difference}")
        row["reference_gradient_relative_l2"] = difference
    row["warm_baseline_rss_bytes"] = _reset_resident_peak()
    samples = []
    for _ in range(args.repeats):
        start = perf_counter()
        jax.block_until_ready(evaluate(theta))
        samples.append((perf_counter()-start)*1000)
    row.update(warm_samples_ms=samples, warm_median_ms=float(np.median(samples)),
               warm_peak_rss_bytes=_resident_memory()["VmHWM"], complete=True)
    row["peak_rss_bytes"] = max(setup_peak, compile_peak, first_peak, row["warm_peak_rss_bytes"])
    memory = row["compiler_memory"]
    row["compiler_buffer_bytes"] = (memory["argument_size_in_bytes"]+memory["output_size_in_bytes"]
                                    +memory["temp_size_in_bytes"]-memory["alias_size_in_bytes"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--particle-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--models", nargs="+", choices=["gc", "lorentz"], default=["gc", "lorentz"])
    parser.add_argument("--methods", nargs="+", choices=["dopri5", "dopri8"], default=["dopri8"])
    parser.add_argument("--backends", nargs="+", choices=["native", "diffrax"], default=["native", "diffrax"])
    parser.add_argument("--objective-index", type=int, choices=range(7), default=0)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--checkpoint-steps", type=int, default=64)
    parser.add_argument("--accepted-capacity", type=int)
    parser.add_argument("--saves", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--gc-horizon", type=float, default=5e-5)
    parser.add_argument("--lorentz-horizon", type=float, default=5e-6)
    parser.add_argument("--initial-step-fraction", type=float, default=.5)
    parser.add_argument("--parameter-family", choices=["coils", "birth", "joint"], default="joint")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", choices=["gc", "lorentz"], help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=["dopri5", "dopri8"], help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["native", "diffrax"], help=argparse.SUPPRESS)
    parser.add_argument("--particles", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3 or any(n < 1 or n > 8 for n in args.particle_counts):
        parser.error("use a fresh output, >=3 repeats and particle counts in 1..8")
    if (args.max_steps < 1 or args.checkpoint_steps < 1 or args.saves < 2 or args.tolerance <= 0
            or min(args.gc_horizon, args.lorentz_horizon) <= 0 or not 0 < args.initial_step_fraction <= 1
            or (args.accepted_capacity is not None and args.accepted_capacity < 1)):
        parser.error("positive bounds, horizons, tolerance and capacities, >=2 saves, 0<initial-step-fraction<=1 required")
    if args.reference:
        previous = json.loads(args.reference.read_text())
        if not previous.get("complete"):
            parser.error("reference must be complete")
        comparable = ("objective_index", "saves", "tolerance", "gc_horizon", "lorentz_horizon",
                      "initial_step_fraction", "parameter_family", "max_steps")
        for name in comparable:
            if previous["metadata"]["settings"][name] != getattr(args, name):
                parser.error(f"reference has different {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.model:
        args.output.write_text(json.dumps(json_safe(worker(args)), indent=2)+"\n")
        return
    source_hashes = fingerprints()
    report = dict(complete=False, metadata=dict(
        settings={name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        source_sha256=source_hashes, jax=jax.__version__, backend="cpu",
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None,
        memory_scope="Own Linux process RSS; compile peak includes setup; VmHWM reset after compile and first execution",
        timing_scope="Three blocked warm scalar value_and_grad calls including the forward solve; compilation excluded"), rows=[])
    with tempfile.TemporaryDirectory(prefix="essos-ad-scaling-") as directory:
        for n in args.particle_counts:
            for model in args.models:
                for method in args.methods:
                    for backend in args.backends:
                        output = Path(directory)/"case.json"
                        output.unlink(missing_ok=True)
                        command = [sys.executable, "-m", "examples.benchmarks.benchmark_native_ad_scaling",
                            "--output", str(output), "--particles", str(n), "--model", model,
                            "--method", method, "--backend", backend]
                        for name in ("objective_index", "max_steps", "checkpoint_steps", "accepted_capacity",
                                     "saves", "tolerance", "gc_horizon", "lorentz_horizon",
                                     "initial_step_fraction", "parameter_family", "repeats", "reference"):
                            value = getattr(args, name)
                            if value is not None:
                                command.extend(["--"+name.replace("_", "-"), str(value)])
                        print(f"Measuring {n} particles: {model} {method} {backend}", flush=True)
                        child = subprocess.run(command, capture_output=True, text=True)
                        if child.returncode:
                            raise RuntimeError(child.stdout+"\n"+child.stderr)
                        row = json.loads(output.read_text())
                        if fingerprints() != source_hashes:
                            raise RuntimeError("source changed during scaling measurement")
                        report["rows"].append(row)
                        args.output.write_text(json.dumps(report, indent=2)+"\n")
                        print(json.dumps({name: row[name] for name in
                            ("particles", "model", "method", "backend", "warm_median_ms", "warm_peak_rss_bytes", "compiler_buffer_bytes")}), flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
