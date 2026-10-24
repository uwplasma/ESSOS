"""Profile the existing independent-orbit Dopri8 baseline, without variants.

Capture a JAX/CUPTI GPU timeline after compilation and warmup, using installed
packages. This does not change the field, tableau, controller, stopping
surface or arithmetic. Launch resources are not hardware stall counters.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from essos.boozer import _native_orbits, boozer_compiler_options
from essos.rk_boozer_gpu_orbit import prepare_orbit_solver
from .benchmark_native_boozer_alpha import PARAMETERS, ROOT, MASS, CHARGE, setup


def kernel_resources(metadata):
    """Read optional collector-supplied resources; never infer missing values."""
    details = metadata.get("kernel_details", "")
    result = {}
    for source, target in (("regs", "registers_per_thread"),
            ("static_shared", "static_shared_bytes"), ("dynamic_shared", "dynamic_shared_bytes")):
        match = re.search(rf"\b{source}:(\d+)\b", details)
        if match:
            result[target] = int(match[1])
    for source, target in (("grid", "grid_blocks"), ("block", "block_threads")):
        match = re.search(rf"\b{source}:(\d+),(\d+),(\d+)\b", details)
        if match:
            result[target] = list(map(int, match.groups()))
    occupancy = metadata.get("theoretical_occupancy_pct")
    if occupancy is None:
        match = re.search(r"\bocc_pct:([\d.]+)", details)
        occupancy = match[1] if match else None
    if occupancy is not None and np.isfinite(float(occupancy)):
        result["theoretical_occupancy_pct"] = float(occupancy)
    return result


def summarize_trace(path):
    """Summarize actual GPU stream events; never count host launches as kernels."""
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    processes, threads = {}, {}
    for event in events:
        if event.get("ph") == "M":
            name = event.get("args", {}).get("name", "")
            if event.get("name") == "process_name":
                processes[event["pid"]] = name
            elif event.get("name") == "thread_name":
                threads[event["pid"], event.get("tid")] = name
    gpu_pids = {pid for pid, name in processes.items()
                if re.search(r"(?:/device:GPU:|\bGPU\b)", name, re.IGNORECASE)}
    grouped, intervals, transfers = defaultdict(list), [], []
    for event in events:
        if event.get("ph") != "X" or event.get("pid") not in gpu_pids:
            continue
        track = threads.get((event["pid"], event.get("tid")), "")
        # XProf also has aggregate GPU-op tracks; these duplicate real streams.
        if not re.search(r"\bstream\b", track, re.IGNORECASE):
            continue
        name = event.get("name", "")
        if re.search(r"memcpy|memset", name, re.IGNORECASE):
            transfers.append(event)
            continue
        if "ts" not in event or not np.isfinite(event.get("dur", -1)) or event["dur"] < 0:
            continue
        grouped[name].append(event)
        intervals.append((event["ts"], event["ts"]+event["dur"]))
    total = sum(e["dur"] for group in grouped.values() for e in group)
    rows = []
    for name, group in grouped.items():
        durations = [e["dur"] for e in group]
        # Preserve collector-provided launch resources verbatim. Not counters
        # for executed spills, achieved occupancy, bandwidth or stall reasons.
        keys = ("kernel_details", "theoretical_occupancy_pct", "hlo_module", "hlo_op", "name")
        metadata = {json.dumps({key:e.get("args", {})[key] for key in keys
                    if key in e.get("args", {})}, sort_keys=True) for e in group}
        rows.append(dict(name=name, calls=len(group), total_ms=sum(durations)/1000,
            median_ms=float(np.median(durations))/1000,
            percent_of_summed_kernel_time=100*sum(durations)/total if total else 0.,
            launch_metadata=[json.loads(item) for item in sorted(metadata)],
            launch_resources=[kernel_resources(json.loads(item)) for item in sorted(metadata)]))
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    union = sum(end-start for start,end in merged)/1000
    span = (max(end for _,end in intervals)-intervals[0][0])/1000 if intervals else None
    return dict(gpu_stream_events_present=bool(rows),
        gpu_processes=[processes[pid] for pid in sorted(gpu_pids)],
        kernels=sorted(rows, key=lambda row: row["total_ms"], reverse=True),
        summed_kernel_time_ms=total/1000,
        kernel_interval_union_ms=union, first_to_last_kernel_ms=span,
        time_without_a_kernel_within_span_ms=span-union if span is not None else None,
        gap_scope="Gaps can include copies, launch/synchronization overhead and idle time; not measured instruction stalls.",
        memcpy_or_memset_events=len(transfers),
        memcpy_or_memset_sum_ms=sum(event.get("dur", 0) for event in transfers)/1000,
        scope="GPU stream durations; kernel sums can overlap. No instruction-stall, executed-spill or throughput counters.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--horizon", type=float, default=2e-4)
    parser.add_argument("--saves", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=3,
                        help="Unprofiled blocked warm calls; profile contains one additional call")
    parser.add_argument("--field-cache", type=Path, default=ROOT/
        "examples/benchmarks/results/gpu_boozer_fixed_20261022/coordinate_inputs.npz")
    parser.add_argument("--output", type=Path, default=Path("output/boozer_dopri8_profile"))
    options = parser.parse_args(argv)
    if options.particles < 1 or options.saves < 2 or options.repeats < 1 or not (
            np.isfinite(options.horizon) and options.horizon > 0):
        parser.error("Require positive particles/repeats/horizon and saves >= 2")
    if not options.field_cache.is_file():
        parser.error(f"Missing field cache: {options.field_cache}; use an existing boozer_inputs.npz")
    if options.output.exists() and any(options.output.iterdir()):
        parser.error("Choose a fresh output directory")
    compiler_options = boozer_compiler_options()
    if not compiler_options:
        parser.error("This profile requires the machine's normal JAX backend to be CUDA")
    options.output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(**(dict(PARAMETERS) | dict(particles=options.particles,
        horizon=options.horizon, saves=options.saves, field_cache=options.field_cache)))
    print(f"Profiling unchanged Dopri8 baseline: {jax.devices()[0].device_kind}; JAX {jax.__version__}", flush=True)
    field, initial, mu, speed, _, _ = setup(settings)
    times = jnp.linspace(0., settings.horizon, settings.saves)
    solve = prepare_orbit_solver(field, MASS, CHARGE, speed, times,
        settings.timestep, settings.rtol, settings.atol, settings.max_steps)

    def run(y, moment):
        return _native_orbits(field, y, moment, MASS, CHARGE, speed, times,
            settings.timestep, settings.rtol, settings.atol, "dopri8", True, "i",
            .4, .3, "standard", settings.max_steps, "unrolled", _orbit_solver=solve)

    jax.block_until_ready((initial, mu, times))
    report = dict(complete=False, capture="jax_cupti", device=jax.devices()[0].device_kind,
        jax=jax.__version__, particles=options.particles, horizon_s=options.horizon,
        saves=options.saves, field_cache=str(options.field_cache), field_modes=int(field.xm.size),
        rtol=settings.rtol, atol=settings.atol, timestep=settings.timestep,
        energy_ev=settings.energy_ev, birth_s=settings.birth_s, seed=settings.seed,
        method="dopri8", controller="i", dense_output="standard", independent_orbits=True,
        collisions=False, reverse_ad=False, particle_chunking=False, arithmetic_changes=False,
        compiler_options=compiler_options, hardware_stall_counters_measured=False,
        resource_scope="Launch registers/shared memory and theoretical occupancy, when provided by CUPTI; not achieved occupancy, executed spill traffic or instruction stalls.",
        birth_sha256=hashlib.sha256(np.asarray(initial).tobytes()+np.asarray(mu).tobytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("essos/boozer.py", "essos/rk_boozer_gpu_orbit.py", "essos/rk_boozer_gpu_trial.py")})

    def save():
        (options.output/"profile_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    print("Compiling, then warming twice outside the capture", flush=True)
    started = perf_counter()
    executable = jax.jit(run, compiler_options=compiler_options).lower(initial, mu).compile()
    report["compile_s"] = perf_counter()-started
    (options.output/"compiled_hlo.txt").write_text(executable.as_text())
    memory = executable.memory_analysis()
    report["compiled_buffers_bytes"] = int(memory.argument_size_in_bytes+memory.output_size_in_bytes+
        memory.temp_size_in_bytes-memory.alias_size_in_bytes)
    for _ in range(2):
        result = jax.block_until_ready(executable(initial, mu))
    report["unprofiled_warm_ms"] = []
    for _ in range(options.repeats):
        started = perf_counter()
        result = jax.block_until_ready(executable(initial, mu))
        report["unprofiled_warm_ms"].append(1000*(perf_counter()-started))
    report["unprofiled_warm_median_ms"] = float(np.median(report["unprofiled_warm_ms"]))
    baseline = tuple(np.asarray(array) for array in result)
    save()
    print(f"Unprofiled warm median: {report['unprofiled_warm_median_ms']:.3f} ms; capturing one warm call", flush=True)
    profiler_options = jax.profiler.ProfileOptions()
    profiler_options.host_tracer_level = 2
    profiler_options.python_tracer_level = 0
    profiler_options.device_tracer_level = 1
    with jax.profiler.trace(str(options.output/"trace"), create_perfetto_trace=True,
                            create_perfetto_link=False, profiler_options=profiler_options):
        with jax.profiler.TraceAnnotation("native_boozer_dopri8_profile"):
            started = perf_counter()
            result = jax.block_until_ready(executable(initial, mu))
            report["profiled_call_wall_ms"] = 1000*(perf_counter()-started)
    paths = list((options.output/"trace").rglob("*.trace.json.gz"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one JAX trace, found {len(paths)}; inspect {options.output/'trace'}")
    report["trace_path"] = str(paths[0])
    report["gpu_trace"] = summarize_trace(paths[0])
    states, losses, failures, energy, counts, status = map(np.asarray, result)
    report["profiled_outputs_bitwise_equal"] = all(np.array_equal(a,b) for a,b in zip(baseline,result))
    report["counts"] = {key:int(counts[:,i].sum()) for i,key in enumerate(("trials", "accepted", "rejected"))}
    report["lost"] = int(np.sum(losses >= 0))
    report["failed"] = int(np.sum(failures >= 0))
    report["max_relative_energy_error"] = float(energy.max()) if np.isfinite(energy).all() else None
    report["complete"] = bool(report["profiled_outputs_bitwise_equal"] and not report["failed"] and
        np.isfinite(states).all() and np.isfinite(energy).all() and np.isin(status, [0,1]).all() and
        report["gpu_trace"]["gpu_stream_events_present"])
    save()
    for row in report["gpu_trace"]["kernels"][:8]:
        print(f"GPU {row['total_ms']:.3f} ms ({row['percent_of_summed_kernel_time']:.1f}% of kernel sum), "
              f"{row['calls']} call(s): {row['name']}", flush=True)
        print(f"  Launch resources: {row['launch_resources']}", flush=True)
    trace = report["gpu_trace"]
    print(f"GPU kernel interval union: {trace['kernel_interval_union_ms']:.3f} ms; "
          f"first-to-last kernel span: {trace['first_to_last_kernel_ms']} ms; "
          f"copy/memset events: {trace['memcpy_or_memset_events']}", flush=True)
    print(f"Saved {options.output.resolve()/'profile_summary.json'}", flush=True)
    if not report["gpu_trace"]["gpu_stream_events_present"]:
        raise RuntimeError("No GPU stream events captured. This is NOT a valid GPU profile. Check CUPTI availability/permissions and profiler stderr; keep the saved host trace for diagnosis.")
    if not report["complete"]:
        raise RuntimeError("Profile changed outputs or solve failed; inspect profile_summary.json")
    return report


if __name__ == "__main__":
    main()
