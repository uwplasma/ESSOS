"""Measure each saved QA timing case in a fresh CPU process.

Compiler buffers describe the integration executable's forward arrays.
Linux peak RSS includes Python/JAX, input setup, compilation and execution.
No process/thread enumeration or GPU execution is used. Each replay must
recover the saved step counts and terminal state before its memory is accepted.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")

import jax
import jax.numpy as jnp
import numpy as np

from .benchmark_collision_tracing import make_solver
from essos.background_species import BackgroundSpecies
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import Particles
from essos.fields import BiotSavart


def measure(report, label):
    import resource
    if sys.platform != "linux" or jax.default_backend() != "cpu":
        raise ValueError("this RSS measurement requires Linux and CPU")
    options = argparse.Namespace(**report["metadata"]["settings"])
    root = Path(__file__).resolve().parents[2]
    coil = root/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
    if hashlib.sha256(coil.read_bytes()).hexdigest() != report["metadata"]["coil_sha256"]:
        raise ValueError("coil input changed")
    field = BiotSavart(Coils.from_json(str(coil)))
    points = jnp.column_stack((jnp.linspace(1.23, 1.27, options.particles),
        jnp.zeros((options.particles, 2))))
    birth = Particles(points, mass=M, charge=Q, energy=4000*ONE_EV)
    keys = jax.random.split(jax.random.key(42), options.particles)
    background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([1e20]), jnp.array([1000.]))
    times = jnp.linspace(0., options.horizon, options.saves)
    np.testing.assert_array_equal(points, report["metadata"]["initial_xyz"])
    np.testing.assert_array_equal(birth.initial_vparallel_over_v, report["metadata"]["initial_pitch"])
    jax.block_until_ready((points, keys, times))
    peak = lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    baseline_rss = peak()
    function = make_solver(label, field, background, birth, times, options)
    executable = function.lower(points, keys).compile()
    after_compile_rss = peak()
    output = jax.device_get(jax.block_until_ready(executable(points, keys)))
    after_execute_rss = peak()
    ys, final, end, status, counts = output
    if np.any(status != 0) or not np.isfinite(ys).all():
        raise RuntimeError("incomplete/nonfinite memory replay")
    previous = next(row for row in report["rows"] if row["case"] == label)
    np.testing.assert_array_equal(counts, previous["counts_per_particle"])
    np.testing.assert_allclose(final, previous["terminal_state"], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(end, options.horizon, rtol=1e-12, atol=0.)
    analysis = executable.memory_analysis()
    if analysis is None:
        raise RuntimeError("compiler memory analysis is unavailable")
    fields = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
    sizes = {name: int(getattr(analysis, name)) for name in fields}
    buffers = sizes[fields[0]]+sizes[fields[1]]+sizes[fields[2]]-sizes[fields[3]]
    return dict(case=label, compiler_buffer_bytes=buffers, compiler_memory=sizes,
        setup_peak_rss_bytes=baseline_rss, after_compile_peak_rss_bytes=after_compile_rss,
        after_execute_peak_rss_bytes=after_execute_rss, peak_rss_bytes=after_execute_rss,
        max_terminal_state_difference=float(np.max(abs(final-np.asarray(previous["terminal_state"])))),
        counts_match=True, complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="Measure one case only; used by isolated child processes")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path")
    original = json.loads(args.timings.read_text())
    if not original.get("complete") or original["metadata"]["backend"] != "cpu":
        parser.error("a complete CPU timing report is required")
    root = Path(__file__).resolve().parents[2]
    for name, expected in original["metadata"]["source_sha256"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"timed implementation changed: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.case:
        result = measure(original, args.case)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        return
    result = dict(complete=False, metadata=dict(original["metadata"],
        memory_scope="Separate fresh Linux CPU process per case; one blocked replay; no AD",
        compiler_memory="Arguments + outputs + temporaries - aliases; excludes executable code, runtime and host allocations",
        peak_rss="Linux ru_maxrss for the child process; includes imports, inputs, compilation and execution; not runtime-only integration memory",
        memory_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        timing_input=dict(path=str(args.timings), sha256=hashlib.sha256(args.timings.read_bytes()).hexdigest()), rows=[])
    with tempfile.TemporaryDirectory(prefix="essos-qa-memory-") as directory:
        for row in original["rows"]:
            label = row["case"]
            print(f"Measuring isolated CPU memory: {label}", flush=True)
            child_output = Path(directory)/(label+".json")
            command = [sys.executable, "-m", "examples.benchmarks.measure_collision_tracing_memory",
                "--timings", str(args.timings.resolve()), "--output", str(child_output), "--case", label]
            child = subprocess.run(command, capture_output=True, text=True)
            if child.returncode:
                raise RuntimeError(f"memory replay failed for {label}:\n{child.stdout}\n{child.stderr}")
            memory = json.loads(child_output.read_text())
            result["rows"].append(dict(row, memory=memory))
            args.output.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps(memory), flush=True)
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    table = [dict(case=row["case"], warm_median_ms=row["warm_median_ms"],
        compiler_buffer_mib=row["memory"]["compiler_buffer_bytes"]/2**20,
        process_peak_rss_mib=row["memory"]["peak_rss_bytes"]/2**20,
        setup_peak_rss_mib=row["memory"]["setup_peak_rss_bytes"]/2**20,
        after_compile_peak_rss_mib=row["memory"]["after_compile_peak_rss_bytes"]/2**20,
        max_terminal_state_difference=row["memory"]["max_terminal_state_difference"]) for row in result["rows"]]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(dict(complete=True, memory=table)), flush=True)


if __name__ == "__main__":
    main()
