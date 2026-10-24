"""Final paired Dopri8 RHS check; no Nsight, solver defaults or field edits.

Compare ordinary FP64 division with cached reciprocals plus explicit fused
rounding correction and an exceptional-value fallback. This is a discarded
local candidate being checked on the target GPU, not an installed solver
optimization. Both cases must agree bitwise before considering a speed gain.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import triton as plt

from essos.boozer import _native_orbits, boozer_compiler_options, guiding_center_rhs
from essos.rk import _C8, _A8, _ERROR8
import essos.rk_boozer_gpu_orbit as orbit_module
import essos.rk_boozer_gpu_trial as trial_module
from .benchmark_native_boozer_alpha import PARAMETERS, ROOT, MASS, CHARGE, setup, rss


def _fma(a, b, c):
    a, b, c = jnp.broadcast_arrays(a, b, c)
    scalar = a.shape == ()
    if scalar:
        a, b, c = (jnp.reshape(x, (1,)) for x in (a, b, c))
    value, = plt.elementwise_inline_asm("fma.rn.f64 $0, $1, $2, $3;",
        args=[a, b, c], constraints="=d,d,d,d", pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, a.dtype)])
    # Pallas represents an inline-assembly rank-zero result differently from
    # an ordinary scalar. A one-element reduction avoids that incompatibility.
    return jnp.sum(value) if scalar else value


@contextmanager
def _corrected_rhs():
    """Override only tracing in this sequential diagnostic; restore on exit."""
    original_factory = trial_module._kernel_rhs
    original_bind = trial_module._bind_primitive

    def all_true(value):
        return jnp.min(value.astype(jnp.int32)) > 0

    def factory(*args):
        ordinary = original_factory(*args)

        def evaluate(y):
            inverse_cache, guards = {}, []

            def bind(primitive, values, params):
                if (primitive.name != "div" or
                        not isinstance(values[1], jax.core.Tracer) or
                        values[1].dtype != jnp.float64):
                    return original_bind(primitive, values, params)
                numerator, denominator = values
                key = id(denominator)
                if key not in inverse_cache:
                    inverse = 1./denominator
                    inverse_cache[key] = inverse
                    guards.append(all_true(trial_module._kernel_is_finite(inverse) &
                        (jnp.abs(inverse) >= np.finfo(np.float64).tiny)))
                inverse = inverse_cache[key]
                quotient = numerator*inverse
                guards.append(all_true(trial_module._kernel_is_finite(quotient)))
                residual = _fma(-quotient, denominator, numerator)
                return _fma(residual, inverse, quotient)

            # The interpreter sees immutable SSA values. Its environment keeps
            # them alive for the complete call, so identity is a safe cache key.
            with patch.object(trial_module, "_bind_primitive", bind):
                value = ordinary(y)
            safe = jnp.asarray(True)
            for guard in guards:
                safe = safe & guard
            return jax.lax.cond(safe, lambda _: value, lambda _: ordinary(y), None)

        return evaluate

    with patch.object(orbit_module, "_kernel_rhs", factory), patch.object(trial_module, "_kernel_rhs", factory):
        yield


def _orbit(field, speed, times, settings):
    solver = orbit_module.prepare_orbit_solver(field, MASS, CHARGE, speed, times,
        settings.timestep, settings.rtol, settings.atol, settings.max_steps)

    def run(y, moment):
        return _native_orbits(field, y, moment, MASS, CHARGE, speed, times,
            settings.timestep, settings.rtol, settings.atol, "dopri8", True, "i",
            .4, .3, "standard", settings.max_steps, "unrolled", _orbit_solver=solver)
    return run


def _trials(field, speed, settings, steps):
    factory = trial_module.prepare_trial_factory(field, MASS, CHARGE)
    units = jnp.array([1., 1., 1., speed])
    h = jnp.asarray(settings.timestep)

    def one(y, moment, first):
        step = factory(moment)

        def advance(i, carry):
            point, derivative, valid, error = carry
            trial = step(None, i*h, point, h, None, derivative, _C8, _A8, _ERROR8)
            finite = (jnp.isfinite(trial.y).all() & jnp.isfinite(trial.error).all() &
                jnp.isfinite(trial.increments).all() & jnp.isfinite(trial.f1).all())
            return (trial.y, trial.f1, valid & finite,
                jnp.maximum(error, jnp.max(jnp.abs(trial.error/units))))

        return jax.lax.fori_loop(0, steps, advance,
            (y, first, jnp.asarray(True), jnp.zeros((), y.dtype)))
    return jax.vmap(one)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--horizon", type=float, default=2e-4)
    parser.add_argument("--saves", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--trial-steps", type=int, default=128)
    parser.add_argument("--field-cache", type=Path, default=ROOT/
        "examples/benchmarks/results/gpu_boozer_fixed_20261022/coordinate_inputs.npz")
    parser.add_argument("--output", type=Path, default=Path("output/boozer_dopri8_rhs_final"))
    options = parser.parse_args()
    if min(options.particles, options.repeats, options.trial_steps) < 1 or options.saves < 2 or not (
            np.isfinite(options.horizon) and options.horizon > 0):
        parser.error("Require positive particles/repeats/trial-steps/horizon and saves >= 2")
    if (options.output/"comparison.json").exists():
        parser.error("Choose a fresh output directory")
    if not boozer_compiler_options():
        parser.error("This test requires the machine's default JAX backend to be CUDA")
    options.output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(**(dict(PARAMETERS) | dict(particles=options.particles,
        horizon=options.horizon, saves=options.saves, field_cache=options.field_cache)))
    field, initial, mu, speed, _, _ = setup(settings)
    times = jnp.linspace(0., settings.horizon, settings.saves)
    first = jax.vmap(lambda y,m: guiding_center_rhs(field,y,m,MASS,CHARGE))(initial,mu)
    jax.block_until_ready((initial, mu, times, first))
    report = dict(complete=False, device=jax.devices()[0].device_kind, jax=jax.__version__,
        particles=options.particles, horizon_s=options.horizon, saves=options.saves,
        rtol=settings.rtol, atol=settings.atol, controller="i", method="dopri8",
        energy_ev=settings.energy_ev, seed=settings.seed, field_modes=int(field.xm.size),
        trial_steps=options.trial_steps, trial_dt_s=settings.timestep, repeats=options.repeats,
        profiler_required=False, hardware_counters_measured=False,
        particle_chunking=False, collisions=False, reverse_ad=False,
        compiler_options=boozer_compiler_options(),
        birth_sha256=hashlib.sha256(np.asarray(initial).tobytes()+np.asarray(mu).tobytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("essos/boozer.py", "essos/rk.py", "essos/rk_boozer_gpu_orbit.py",
             "essos/rk_boozer_gpu_trial.py", "examples/benchmarks/benchmark_boozer_rhs_final.py")}, rows={})

    def save():
        (options.output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")

    executables, arguments, results = {}, {}, {}
    for kind in ("orbit", "trials"):
        for variant in ("baseline", "corrected"):
            name = kind+"_"+variant
            print(f"Compiling {name}: {options.particles} particles",flush=True)
            args = (initial,mu) if kind == "orbit" else (initial,mu,first)
            started = perf_counter()
            if variant == "corrected":
                with _corrected_rhs():
                    run = _orbit(field,speed,times,settings) if kind == "orbit" else _trials(
                        field,speed,settings,options.trial_steps)
                    executable = jax.jit(run,compiler_options=boozer_compiler_options()).lower(*args).compile()
            else:
                run = _orbit(field,speed,times,settings) if kind == "orbit" else _trials(
                    field,speed,settings,options.trial_steps)
                executable = jax.jit(run,compiler_options=boozer_compiler_options()).lower(*args).compile()
            row = dict(compile_s=perf_counter()-started,samples_ms=[])
            memory = executable.memory_analysis()
            row["compiled_buffers_bytes"] = int(memory.argument_size_in_bytes+memory.output_size_in_bytes+
                memory.temp_size_in_bytes-memory.alias_size_in_bytes)
            started = perf_counter()
            results[name] = jax.block_until_ready(executable(*args))
            row["first_ms"] = 1000*(perf_counter()-started)
            for _ in range(2):
                jax.block_until_ready(executable(*args))
            executables[name], arguments[name] = executable, args
            report["rows"][name] = row
            save()
    for repeat in range(options.repeats):
        names = list(executables)
        for name in names if repeat % 2 == 0 else reversed(names):
            started = perf_counter()
            jax.block_until_ready(executables[name](*arguments[name]))
            report["rows"][name]["samples_ms"].append(1000*(perf_counter()-started))
    valid = True
    for name,row in report["rows"].items():
        row["warm_median_ms"] = float(np.median(row["samples_ms"]))
        row["warm_q25_ms"], row["warm_q75_ms"] = map(float,np.quantile(row["samples_ms"],[.25,.75]))
        baseline = name.split("_")[0]+"_baseline"
        row["bitwise_equal_to_baseline"] = all(np.array_equal(a,b) for a,b in
            zip(results[baseline],results[name]))
        if name.startswith("orbit"):
            states,losses,failures,energy,counts,status = map(np.asarray,results[name])
            okay = np.isfinite(states).all() and np.isfinite(energy).all() and np.isin(status,[0,1]).all()
            row.update(trials=int(counts[:,0].sum()),accepted=int(counts[:,1].sum()),
                rejected=int(counts[:,2].sum()),lost=int(np.sum(losses >= 0)),
                failed=int(np.sum(failures >= 0)),max_relative_energy_error=float(energy.max()))
            okay = okay and not np.any(failures >= 0)
        else:
            endpoint,derivative,finite,error = map(np.asarray,results[name])
            okay = np.isfinite(endpoint).all() and np.isfinite(derivative).all() and finite.all()
            row["executed_trial_slots"] = options.particles*options.trial_steps
        valid = bool(valid and okay)
        print(f"{name}: warm {row['warm_median_ms']:.3f} ms; "
            f"bitwise equality {row['bitwise_equal_to_baseline']}",flush=True)
    report["timing_scope"] = "Blocked device-resident calls, alternating case order. Compilation, setup and comparisons excluded."
    report["trials_scope"] = "Equal-count stage recurrence, embedded estimates and finite guards; no adaptive controller, saves or stopping. Initial FSAL RHS excluded."
    report["memory_scope"] = "Compiled buffers per executable. Allocator/host peaks are cumulative across all four cases; not separate per-case peak VRAM."
    report["device_memory"] = jax.devices()[0].memory_stats()
    report["host_memory"] = rss()
    report["complete"] = valid
    for kind in ("orbit","trials"):
        before,after=(report["rows"][kind+"_"+v] for v in ("baseline","corrected"))
        report[kind+"_time_decrease_percent"] = 100*(1-after["warm_median_ms"]/before["warm_median_ms"])
        report[kind+"_paired_median_ratio"] = float(np.median(np.asarray(after["samples_ms"])/before["samples_ms"]))
    report["eligible_for_retention"] = bool(valid and all(row["bitwise_equal_to_baseline"]
        for row in report["rows"].values()) and report["orbit_time_decrease_percent"] >= 3.)
    report["retention_screen"] = "Bitwise equality and >=3% lower warm adaptive median; inspect paired samples and memory before retaining. No automatic solver change."
    save()
    print(f"Saved {options.output.resolve()/'comparison.json'}",flush=True)
    if not valid:
        raise RuntimeError("Incomplete or nonfinite traces; inspect comparison.json")
    if not all(row["bitwise_equal_to_baseline"] for row in report["rows"].values()):
        raise RuntimeError("Candidate did not preserve bitwise outputs; do not treat its timing as an equivalent-result speedup")
    return report


if __name__ == "__main__":
    main()
