"""Controlled 64-particle adaptive Dopri8 comparison of inline and deferred roots.

Both cases use the production Boozer kernel, identical births, physical RHS,
tableau, I-controller, tolerances and save times. Use --case inline/replay in
separate processes for independent peak-memory measurements. This short local
workload does not predict the 8192-particle, 20 ms benchmark runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from essos.boozer import _native_orbits
from .benchmark_native_boozer_alpha import PARAMETERS, ROOT, setup, MASS, CHARGE, rss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("both", "inline", "replay"), default="both")
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--horizon", type=float, default=2e-4)
    parser.add_argument("--saves", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--field-cache", type=Path, default=ROOT/
        "examples/benchmarks/results/gpu_boozer_fixed_20261022/coordinate_inputs.npz")
    parser.add_argument("--output", type=Path, default=Path("output/boozer_adaptive_events"))
    options = parser.parse_args()
    if options.particles < 1 or options.horizon <= 0 or options.saves < 2 or options.repeats < 1:
        parser.error("particles, horizon and repeats must be positive; saves >= 2")
    settings = SimpleNamespace(**(dict(PARAMETERS) | dict(particles=options.particles,
        horizon=options.horizon, saves=options.saves, field_cache=options.field_cache)))
    field, initial, mu, speed, births, _ = setup(settings)
    times = jnp.linspace(0., settings.horizon, settings.saves)
    cases = ("inline", "replay") if options.case == "both" else (options.case,)
    report = dict(backend=jax.default_backend(), device=jax.devices()[0].device_kind,
        jax=jax.__version__, particles=options.particles, horizon_s=options.horizon,
        saves=options.saves, repeats=options.repeats, energy_ev=settings.energy_ev,
        seed=settings.seed, rtol=settings.rtol, atol=settings.atol,
        controller=settings.controller, method="dopri8", stage_execution="unrolled",
        field_modes=int(field.xm.size), particle_chunking=False,
        birth_sha256=hashlib.sha256(np.asarray(initial).tobytes()+np.asarray(mu).tobytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT/"essos"/name).read_bytes()).hexdigest()
                       for name in ("boozer.py", "rk.py", "rk_events.py")}, rows={})
    executables, results = {}, {}
    for name in cases:
        jax.clear_caches()
        def run(y, moment):
            return _native_orbits(field, y, moment, MASS, CHARGE, speed, times, settings.timestep,
                settings.rtol, settings.atol, "dopri8", True, settings.controller,
                .4, .3, "standard", settings.max_steps, "unrolled",
                _defer_event_root=(None if name == "replay" else False))
        print(f"Compiling {name}: {options.particles} particles, adaptive Dopri8", flush=True)
        start = perf_counter()
        executable = jax.jit(run).lower(initial, mu).compile()
        memory = executable.memory_analysis()
        report["rows"][name] = dict(compile_s=perf_counter()-start, samples_ms=[],
            compiled_buffers_bytes=int(memory.argument_size_in_bytes+memory.output_size_in_bytes+
                                      memory.temp_size_in_bytes-memory.alias_size_in_bytes))
        executables[name] = executable
        results[name] = jax.block_until_ready(executable(initial, mu))
        for _ in range(2):
            jax.block_until_ready(executable(initial, mu))
    for repeat in range(options.repeats):
        for name in (cases if repeat % 2 == 0 else tuple(reversed(cases))):
            start = perf_counter()
            jax.block_until_ready(executables[name](initial, mu))
            report["rows"][name]["samples_ms"].append(1000*(perf_counter()-start))
    for name, row in report["rows"].items():
        values, losses, failures, energy, counts, status = map(np.asarray, results[name])
        if not np.isfinite(values).all() or not np.isin(status, [0, 1]).all() or np.any(failures >= 0):
            raise AssertionError(f"{name}: unexpected failure/nonfinite output")
        row.update(warm_median_ms=float(np.median(row["samples_ms"])),
            trials=int(counts[:, 0].sum()), accepted=int(counts[:, 1].sum()),
            rejected=int(counts[:, 2].sum()), max_trials=int(counts[:, 0].max()),
            lost=int(np.sum(losses >= 0)), failed=int(np.sum(failures >= 0)),
            max_relative_energy_error=float(energy.max()))
    if len(cases) == 2:
        # Check all saved and terminal diagnostics, including both speeds.
        for a, b in zip(results["inline"], results["replay"]):
            np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-9)
        np.testing.assert_array_equal(results["inline"][4], results["replay"][4])
        np.testing.assert_array_equal(results["inline"][5], results["replay"][5])
        delta = np.asarray(results["inline"][0])-np.asarray(results["replay"][0])
        report.update(max_saved_component_difference=np.max(abs(delta), axis=(0, 1)).tolist(),
            time_decrease_percent=100*(1-report["rows"]["replay"]["warm_median_ms"]/
                                      report["rows"]["inline"]["warm_median_ms"]))
    report["peak_memory_scope"] = "independent process" if len(cases) == 1 else "cumulative across both executables"
    report["gpu_memory"] = jax.devices()[0].memory_stats()
    report["host_memory"] = rss()
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output/f"{options.case}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
