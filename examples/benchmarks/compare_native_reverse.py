"""Interleaved warm CPU timings for two native collisionless adjoints.

Compile both implementations before sampling in randomized order. The reference
rk_ad.py comes from a local commit; physical fields/RHS and inputs are shared.
Executable buffers are measured separately from time. ``--isolated-memory``
instead runs one fused objective per fresh Linux process, measuring compilation
and warm RSS separately. RSS includes resident JAX/executable/allocator memory.
"""
import argparse
from copy import copy
from functools import partial
import hashlib
import importlib.util
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
import jax.numpy as jnp
import numpy as np
import essos
from .profile_native_reverse import build, ROOT, SOURCES


def _load(source):
    spec = importlib.util.spec_from_file_location("essos.rk_ad", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["essos.rk_ad"] = module
    setattr(essos, "rk_ad", module)
    spec.loader.exec_module(module)
    return module


def _memory(executable):
    value = executable.memory_analysis()
    return int(value.argument_size_in_bytes+value.output_size_in_bytes
               +value.temp_size_in_bytes-value.alias_size_in_bytes)


def _configure(module, options):
    if not getattr(options, "field_cache", True):
        module.prepare_rhs_linearization = partial(module.prepare_rhs_linearization, cache_field_outputs=False)
    if not getattr(options, "field_transpose", True):
        module.prepare_reverse_rhs = partial(module.prepare_reverse_rhs, specialize_biot_savart=False)
    if not getattr(options, "rhs_inlining", True):
        module.prepare_reverse_rhs = lambda prepare, evaluate, *unused: (prepare, evaluate)


def _compile(options, model, horizon):
    theta, cost, _, _, _, scale, _ = build(options, model, horizon)
    forward = jax.jit(lambda p: cost(p, "none"))
    linear = jax.jit(lambda p: cost(p, "accepted"))
    value, pullback, aux = jax.block_until_ready(jax.vjp(linear, theta, has_aux=True))
    seed = jnp.ones_like(value)
    reverse = jax.jit(lambda pb, seed: pb(seed)[0])
    start = perf_counter()
    reverse_executable = reverse.lower(pullback, seed).compile()
    reverse_compile = perf_counter()-start
    gradient = jax.block_until_ready(reverse(pullback, seed))
    total = jax.jit(jax.value_and_grad(lambda p: cost(p, "accepted"), has_aux=True))
    start = perf_counter()
    executable = total.lower(theta).compile()
    total_compile = perf_counter()-start
    (fused_value, fused_aux), fused_gradient = jax.block_until_ready(total(theta))
    ordinary_value, ordinary_aux = jax.block_until_ready(forward(theta))
    np.testing.assert_allclose(gradient, fused_gradient, rtol=2e-9, atol=1e-11)
    np.testing.assert_allclose(value, fused_value, rtol=2e-12, atol=1e-25)
    np.testing.assert_allclose(value, ordinary_value, rtol=2e-12, atol=1e-25)
    for candidate in (fused_aux, ordinary_aux):
        np.testing.assert_allclose((aux[0]-candidate[0])/scale, 0., rtol=0., atol=2e-12)
        for a, b in zip(jax.tree.leaves(aux[1:]), jax.tree.leaves(candidate[1:])):
            np.testing.assert_array_equal(a, b)
    if np.any(aux[-1]) or not np.isfinite(gradient).all():
        raise RuntimeError("failed solve or nonfinite gradient")
    result = dict(value=float(value), gradient=np.asarray(gradient).tolist(),
        stats={k: np.asarray(v).tolist() for k, v in aux[1]._asdict().items()},
        status=np.asarray(aux[-1]).tolist(), total_compiler_buffer_bytes=_memory(executable),
        reverse_compiler_buffer_bytes=_memory(reverse_executable),
        reverse_compile_seconds=reverse_compile, total_compile_seconds=total_compile, samples_ms=[])
    return result, (lambda: forward(theta), lambda: reverse(pullback, seed), lambda: total(theta))


def _measure_memory(options, model, horizon):
    """One fused optimizer evaluation, with no other compiled case resident."""
    import resource
    if sys.platform != "linux":
        raise ValueError("isolated memory measurement requires Linux/WSL")

    def resident():
        return {key: int(value.split()[0])*1024
                for line in Path("/proc/self/status").read_text().splitlines()
                for key, _, value in [line.partition(":")] if key in ("VmRSS", "VmHWM")}

    def reset_peak():
        Path("/proc/self/clear_refs").write_text("5\n")
        return resident()["VmRSS"]

    theta, cost, *_ = build(options, model, horizon)
    evaluate = jax.jit(jax.value_and_grad(lambda p: cost(p, "accepted"), has_aux=True))
    start = perf_counter()
    executable = evaluate.lower(theta).compile()
    compile_seconds = perf_counter()-start
    compile_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    compile_resident = reset_peak()
    (value, aux), gradient = jax.block_until_ready(evaluate(theta))
    first_peak = resident()["VmHWM"]
    if np.any(aux[-1]) or not np.isfinite(gradient).all():
        raise RuntimeError("failed solve or nonfinite memory-replay gradient")
    warm_baseline = reset_peak()
    for _ in range(options.repeats):
        jax.block_until_ready(evaluate(theta))
    warm_peak = resident()["VmHWM"]
    return dict(value=float(value), gradient=np.asarray(gradient).tolist(),
        stats={k: np.asarray(v).tolist() for k, v in aux[1]._asdict().items()},
        status=np.asarray(aux[-1]).tolist(), total_compiler_buffer_bytes=_memory(executable),
        total_compile_seconds=compile_seconds, compile_peak_rss_bytes=compile_peak,
        after_compile_resident_rss_bytes=compile_resident, first_execution_peak_rss_bytes=first_peak,
        warm_baseline_rss_bytes=warm_baseline, warm_peak_rss_bytes=warm_peak,
        warm_additional_peak_rss_bytes=max(0, warm_peak-warm_baseline),
        process_peak_rss_bytes=max(compile_peak, first_peak, warm_peak))


def _isolated_memory(options, source, model, horizon):
    code = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
from examples.benchmarks.compare_native_reverse import _load, _configure, _measure_memory
options = SimpleNamespace(**json.loads(sys.argv[2]))
module = _load(Path(sys.argv[1]))
_configure(module, options)
result = _measure_memory(options, sys.argv[3], float(sys.argv[4]))
Path(sys.argv[5]).write_text(json.dumps(result, allow_nan=False))
"""
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
    with tempfile.TemporaryDirectory(prefix="essos-native-memory-") as directory:
        destination = Path(directory)/"memory.json"
        process = subprocess.Popen([sys.executable, "-c", code, str(source), json.dumps(settings),
                                    model, str(horizon), str(destination)], cwd=ROOT)
        try:
            if process.wait():
                raise RuntimeError("isolated memory worker failed")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        return json.loads(destination.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-revision", default="4abaf51")
    parser.add_argument("--compare-accumulation", action="store_true",
        help="Compare the current source in step and segment modes, instead of a committed adjoint")
    parser.add_argument("--compare-rhs-inlining", action="store_true",
        help="Compare current code with reverse RHS inlining disabled/enabled; no field/RHS rule changes")
    parser.add_argument("--compare-field-transpose", action="store_true",
        help="Compare ordinary field AD with the native-only exact prepared Biot-Savart transpose")
    parser.add_argument("--compare-field-cache", action="store_true",
        help="Compare the current exact field transposes with small field-output retention disabled/enabled")
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--horizon-factors", nargs="+", type=float, default=[1., 10.])
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--checkpoint-steps", type=int, default=16)
    parser.add_argument("--parameter-accumulation", choices=("step", "segment", "recommended"), default="segment",
        help="Default: segment for both models; recommended reproduces the historical mixed-mode benchmark")
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--accepted-capacity", type=int, default=1024)
    parser.add_argument("--saves", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--gc-horizon", type=float, default=2e-6)
    parser.add_argument("--lorentz-horizon", type=float, default=2e-7)
    parser.add_argument("--isolated-memory", action="store_true",
        help="Fresh Linux processes for fused-executable compile/first/warm RSS; no timing comparison")
    options = parser.parse_args()
    if options.output.exists() or jax.default_backend() != "cpu":
        parser.error("choose a fresh output path and CPU backend")
    if not 1 <= options.particles <= 8 or options.repeats < 5 or options.saves < 2:
        parser.error("use 1..8 particles, >=5 repeats and >=2 saves")
    if options.isolated_memory and sys.platform != "linux":
        parser.error("isolated memory measurement requires Linux/WSL")
    if options.compare_accumulation and options.parameter_accumulation != "segment":
        parser.error("--compare-accumulation requires --parameter-accumulation segment")
    if sum((options.compare_accumulation, options.compare_rhs_inlining,
            options.compare_field_transpose, options.compare_field_cache)) > 1:
        parser.error("isolate one comparison at a time")
    positive = (*options.horizon_factors, options.checkpoint_steps, options.max_steps,
        options.accepted_capacity, options.tolerance, options.gc_horizon, options.lorentz_horizon)
    if any(not np.isfinite(x) or x <= 0 for x in positive):
        parser.error("positive finite horizons, tolerances and work/storage bounds required")
    current = (ROOT/"essos/rk_ad.py").read_bytes()
    if (options.compare_accumulation or options.compare_rhs_inlining
            or options.compare_field_transpose or options.compare_field_cache):
        revision, baseline = None, current
    else:
        revision = subprocess.check_output(["git", "rev-parse", f"{options.baseline_revision}^{{commit}}"],
                                           cwd=ROOT, text=True).strip()
        baseline = subprocess.check_output(["git", "show", f"{revision}:essos/rk_ad.py"], cwd=ROOT)
    shared = set(SOURCES)-{"essos/rk_ad.py"}
    shared.add("examples/benchmarks/compare_native_reverse.py")
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(shared)}
    report = dict(complete=False, metadata=dict(jax=jax.__version__, backend=jax.default_backend(),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
        baseline_revision=revision, shared_source_sha256=hashes,
        reverse_sha256={"before": hashlib.sha256(baseline).hexdigest(), "after": hashlib.sha256(current).hexdigest()},
        input_sha256=hashlib.sha256((ROOT/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json").read_bytes()).hexdigest(),
        policy="Identical QA-coil births, 136 coil DOFs, mean final minor radius; adaptive native Dopri8/I, standard dense output, frozen accepted grids; no collisions, Diffrax or particle chunks",
        timing="21 by default: blocked warm forward, standalone reverse with reused pullback, and fused value_and_grad; both versions compiled first, seeded randomized order within each repeat",
        memory="Executable arguments + outputs + temporaries - aliases; process peak RSS is not measured",
        cache="After: compact RHS reduction residuals in each checkpoint segment; field-sized arrays rematerialized. The shared cache helper is used only by the after implementation.",
        parameter_accumulation="Before: once per step. After: "+options.parameter_accumulation), rows=[])
    if options.compare_accumulation:
        report["metadata"]["cache"] = "Both modes use the same current compact RHS cache; only parameter accumulation changes"
    if options.compare_rhs_inlining:
        report["metadata"].update(cache="Both modes use the same compact RHS cache and parameter accumulation",
            parameter_accumulation="Both: "+options.parameter_accumulation,
            rhs_inlining="Before: prepare_reverse_rhs is an identity; after: inline jit wrappers and repartition the original RHS for reverse replay; field specialization disabled in both")
    if options.compare_field_transpose:
        report["metadata"].update(cache="Both modes use the same compact-cache policy; selected residuals can differ with the field rule",
            parameter_accumulation="Both: "+options.parameter_accumulation,
            field_transpose="Before: ordinary field AD and generic reverse preparation; after: exact stock prepared float64 Biot-Savart B or B/J linearization when its source/signature match; other fields/calls keep ordinary AD; field classes and forward RHS unchanged")
    if options.compare_field_cache:
        report["metadata"].update(
            cache="Before: named field-output retention disabled. After: retain small named B or B/J outputs plus the existing reduction policy. Field-sized geometry arrays rematerialized in both.",
            parameter_accumulation="Both: "+options.parameter_accumulation,
            field_transpose="Both: the same exact native stock B/B-J transposes; forward RHS and RK maps unchanged")
    if options.isolated_memory:
        report["metadata"].update(
            timing="No timing comparison: compile seconds are diagnostic only",
            memory="Fresh Linux process per model/implementation: compile peak includes imports/setup/compilation; "
                "first/warm VmHWM reset via /proc/self/clear_refs=5; resident JAX/executable/allocator memory remains included; "
                "additional RSS growth is not the solver working-set size; executable buffers reported separately")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="essos-native-reference-") as directory:
        source = Path(directory)/"rk_ad.py"
        source.write_bytes(baseline)
        for model in options.models:
            for factor in options.horizon_factors:
                horizon = factor*(options.gc_horizon if model == "gc" else options.lorentz_horizon)
                cases = []
                for label, path in (("before", source), ("after", ROOT/"essos/rk_ad.py")):
                    print(f"Compiling {model} {horizon:g} s: {label}", flush=True)
                    settings = copy(options)
                    settings.field_cache = not (options.compare_field_cache and label == "before")
                    if options.parameter_accumulation == "recommended":
                        settings.parameter_accumulation = "step" if model == "gc" else "segment"
                    if options.compare_accumulation and label == "before":
                        settings.parameter_accumulation = "step"
                    settings.rhs_inlining = not (options.compare_rhs_inlining and label == "before")
                    settings.field_transpose = not options.compare_rhs_inlining and not (
                        options.compare_field_transpose and label == "before")
                    if options.isolated_memory:
                        cases.append((_isolated_memory(settings, path, model, horizon), ()))
                    else:
                        module = _load(path)
                        _configure(module, settings)
                        cases.append(_compile(settings, model, horizon))
                before, after = [case[0] for case in cases]
                np.testing.assert_allclose(after["gradient"], before["gradient"], rtol=2e-8, atol=1e-11)
                np.testing.assert_allclose(after["value"], before["value"], rtol=2e-12, atol=1e-25)
                for name in before["stats"]:
                    np.testing.assert_array_equal(after["stats"][name], before["stats"][name])
                if options.isolated_memory:
                    row = dict(model=model, horizon=horizon, horizon_factor=factor, before=before, after=after,
                        parameter_accumulation=settings.parameter_accumulation,
                        compiler_buffer_ratio=after["total_compiler_buffer_bytes"]/before["total_compiler_buffer_bytes"],
                        peak_rss_ratio=after["process_peak_rss_bytes"]/before["process_peak_rss_bytes"],
                        warm_rss_ratio=after["warm_peak_rss_bytes"]/before["warm_peak_rss_bytes"],
                        max_abs_gradient_difference=float(np.max(np.abs(np.asarray(after["gradient"])-before["gradient"]))))
                    report["rows"].append(row)
                    options.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
                    print(json.dumps({k: row[k] for k in ("model", "horizon", "compiler_buffer_ratio",
                        "peak_rss_ratio", "warm_rss_ratio", "max_abs_gradient_difference")}), flush=True)
                    continue
                random = np.random.default_rng(23)
                print(f"Interleaved timing {model} {horizon:g} s", flush=True)
                for _ in range(options.repeats):
                    for which in random.permutation(2):
                        result, functions = cases[which]
                        sample = {}
                        for name, function in zip(("forward", "reverse", "total"), functions):
                            start = perf_counter()
                            jax.block_until_ready(function())
                            sample[name] = 1000*(perf_counter()-start)
                        result["samples_ms"].append(sample)
                for result, _ in cases:
                    result["medians_ms"] = {name: float(np.median([s[name] for s in result["samples_ms"]]))
                                            for name in ("forward", "reverse", "total")}
                row = dict(model=model, horizon=horizon, horizon_factor=factor, before=before, after=after,
                    parameter_accumulation=settings.parameter_accumulation,
                    reverse_speedup=before["medians_ms"]["reverse"]/after["medians_ms"]["reverse"],
                    total_speedup=before["medians_ms"]["total"]/after["medians_ms"]["total"],
                    compiler_buffer_ratio=after["total_compiler_buffer_bytes"]/before["total_compiler_buffer_bytes"],
                    max_abs_gradient_difference=float(np.max(np.abs(np.asarray(after["gradient"])-before["gradient"]))))
                report["rows"].append(row)
                options.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
                print(json.dumps({k: row[k] for k in ("model", "horizon", "reverse_speedup", "total_speedup",
                    "compiler_buffer_ratio", "max_abs_gradient_difference")}), flush=True)
    if (ROOT/"essos/rk_ad.py").read_bytes() != current or any(
        hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError("sources changed during comparison")
    report["complete"] = True
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
