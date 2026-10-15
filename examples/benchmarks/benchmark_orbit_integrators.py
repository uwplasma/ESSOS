"""Compare native and Diffrax orbit solves, including compilation and accuracy.

Run from an installed ESSOS checkout, for example:
    python examples/benchmarks/benchmark_orbit_integrators.py --repeats 7

The benchmark uses the ESSOS GuidingCenter/Lorentz equations and bundled QA
coils. Both backends return trajectories on the same save grid, terminal states
and step counts. This measures JIT-compiled, vmapped integration, excluding Tracing's
Python construction, coordinate postprocessing, and VMEC/exterior events.

Fixed mode uses 256 steps for every particle/backend. Adaptive mode uses the
controllers configured by ESSOS (native I; Diffrax PI), so equal tolerances do
not imply equal work. Both backends use the selected physical block norm or
standard componentwise RMS norm. Errors use a tighter standard Dopri8 reference.
Persistent compilation caching is disabled, JAX caches are cleared before each
lower/compile, and all timed executions block until their outputs are ready.
Warm measurements interleave the four implementations in a seeded random order.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["ESSOS_XLA_CACHE"] = ""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS
from essos.dynamics import GuidingCenter, Lorentz, Particles
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart
from essos.rk import solve_rk
from essos.orbit_error import OrbitErrorNorm, orbit_block_norm

jax.config.update("jax_enable_compilation_cache", False)


METHODS = ("native_dopri5", "diffrax_dopri5", "native_dopri8", "diffrax_dopri8")


def make_case(model, particles_count, field_name):
    if field_name == "qa":
        path = Path(__file__).resolve().parents[1] / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
        field = BiotSavart(Coils.from_json(str(path)))
        xyz = jnp.stack([jnp.linspace(1.21, 1.25, particles_count), jnp.zeros(particles_count),
                         jnp.linspace(-.005, .005, particles_count)], axis=1)
    else:
        field = BiotSavart(Coils(CreateEquallySpacedCurves(2, 1, 1., .4, 16, 2, True), [1e6, 1e6]))
        xyz = jnp.stack([jnp.linspace(.95, 1.05, particles_count), jnp.zeros(particles_count),
                         jnp.linspace(-.005, .005, particles_count)], axis=1)
    particles = Particles(xyz, initial_vparallel_over_v=jnp.linspace(.5, .9, particles_count),
                          mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, energy=5000*ONE_EV, field=field)
    if model == "gc":
        y0 = jnp.column_stack([xyz, particles.initial_vparallel])
        args = (field, particles, Electric_field_zero())
        vector_field, maxtime = GuidingCenter, 2e-6
    else:
        y0 = jnp.column_stack([particles.initial_xyz_fullorbit, particles.initial_vxvyvz])
        args = (field, particles)
        vector_field, maxtime = Lorentz, 2e-7
    rhs = lambda t, y, unused: vector_field(t, y, args)
    jax.block_until_ready(y0)
    return dict(rhs=rhs, y0=y0, maxtime=maxtime, dt0=maxtime/256,
                times=jnp.linspace(0., maxtime, 64), speed=float(particles.total_speed),
                velocity_components=1 if model == "gc" else 3, field_object=field)


def make_solve(case, implementation, adaptive, tolerance, error_control="standard", length_scale=1.,
               native_controller="i", pcoeff=.4, icoeff=.3, *, rtol=None, atol=None,
               native_dense_output="standard"):
    rtol = tolerance if rtol is None else rtol
    atol = tolerance if atol is None else atol
    rhs, t1, dt0, times = (case[k] for k in ("rhs", "maxtime", "dt0", "times"))
    method = implementation.split("_", 1)[1]
    never = lambda t, y, args, **kwargs: False
    physical = OrbitErrorNorm(rtol, atol, case["speed"], length_scale,
                              case["velocity_components"]) if error_control == "physics" else None
    if implementation.startswith("native"):
        def single(y0):
            out = solve_rk(rhs, y0, t1=t1, dt0=dt0, ts=times, method=method,
                           adaptive=adaptive, rtol=rtol, atol=atol,
                           event=never, root_find=False, max_steps=100_000, error_norm=physical,
                           controller=native_controller, pcoeff=pcoeff, icoeff=icoeff,
                           dense_output=native_dense_output)
            return out.ys, out.y, out.t, out.status, jnp.stack(out.stats)
    else:
        solver = diffrax.Dopri5() if method == "dopri5" else diffrax.Dopri8()
        controller = (diffrax.PIDController(pcoeff=pcoeff, icoeff=icoeff, dcoeff=0., rtol=rtol, atol=atol)
                      if adaptive else diffrax.ConstantStepSize())
        if adaptive and physical is not None:
            # Cartesian benchmark: Diffrax's public atol/norm options express
            # exactly the same physical budget, without a custom controller.
            scales = jnp.array([length_scale] * 3 + [case["speed"]] * case["velocity_components"])
            controller = diffrax.PIDController(
                pcoeff=pcoeff, icoeff=icoeff, dcoeff=0., rtol=0., atol=(rtol + atol) * scales,
                norm=lambda error: orbit_block_norm(error, case["velocity_components"]))
        def single(y0):
            out = diffrax.diffeqsolve(
                diffrax.ODETerm(rhs), solver, t0=0., t1=t1, dt0=dt0, y0=y0,
                saveat=diffrax.SaveAt(subs=[diffrax.SubSaveAt(ts=times), diffrax.SubSaveAt(t1=True)]),
                stepsize_controller=controller, event=diffrax.Event(never),
                max_steps=100_000, throw=False)
            counts = jnp.stack([out.stats[k] for k in ("num_steps", "num_accepted_steps", "num_rejected_steps")])
            status = jnp.where(out.result == diffrax.RESULTS.successful, 0, 1)
            return out.ys[0], out.ys[1][-1], out.ts[1][-1], status, counts
    return jax.jit(jax.vmap(single))


def compile_solve(function, y0):
    jax.clear_caches()
    started = perf_counter()
    lowered = function.lower(y0)
    lower_seconds = perf_counter() - started
    started = perf_counter()
    executable = lowered.compile()
    compile_seconds = perf_counter() - started
    started = perf_counter()
    output = jax.block_until_ready(executable(y0))
    execute_seconds = perf_counter() - started
    return executable, output, dict(lower_s=lower_seconds, compile_s=compile_seconds,
                                    first_execute_s=execute_seconds,
                                    first_solve_s=lower_seconds + compile_seconds + execute_seconds)


def check_output(output, t1):
    trajectory, terminal, end_time, status, counts = jax.device_get(output)
    if not (np.all(status == 0) and np.isfinite(trajectory).all() and np.isfinite(terminal).all()):
        raise RuntimeError(f"Failed or non-finite benchmark solve: statuses={status}")
    np.testing.assert_allclose(end_time, t1, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(trajectory[:, -1], terminal, rtol=1e-12, atol=1e-10)
    return np.asarray(trajectory), np.asarray(counts)


def metadata(options):
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    root = Path(__file__).resolve().parents[2]
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("essos/rk.py", "essos/orbit_error.py", "essos/fields.py", "essos/dynamics.py",
                            "examples/benchmarks/benchmark_orbit_integrators.py")}
    return dict(utc=datetime.now(timezone.utc).isoformat(), cpu=cpu, platform=platform.platform(), source_sha256=sources,
                devices=[str(d) for d in jax.devices()], x64=bool(jax.config.jax_enable_x64),
                persistent_compilation_cache=bool(jax.config.jax_enable_compilation_cache),
                versions={p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "diffrax")},
                particles=options.particles, field=options.field, repeats=options.repeats,
                scope="compiled vmapped ESSOS RHS integration; excludes Tracing construction/postprocessing and events",
                adaptive_controllers={"native": "I", "diffrax": "ESSOS PI (PIDController with p=.4, i=.3, d=0)"},
                tolerance=options.tolerance, reference_tolerance=1e-12,
                error_control=getattr(options, "error_control", "standard"),
                reference_length_m=getattr(options, "length_scale", 1.),
                accuracy_limit=options.accuracy_limit, units="SI; velocity error also normalized by total speed",
                field_geometry="standard ESSOS field object; no geometry snapshot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--field", choices=("qa", "small"), default="qa")
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--modes", nargs="+", choices=("fixed", "adaptive"), default=["fixed", "adaptive"])
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--error-control", choices=("physics", "standard"), default="physics")
    parser.add_argument("--length-scale", type=float, default=1.)
    parser.add_argument("--accuracy-limit", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "orbit_integrator_timings.json")
    options = parser.parse_args()
    if options.particles < 1 or options.repeats < 3:
        parser.error("use at least one particle and three timing repeats")
    report = dict(metadata=metadata(options), rows=[])
    print(f"{report['metadata']['cpu']}; {options.particles} particles; "
          f"{options.repeats} repeats; field={options.field}; tolerance={options.tolerance:.0e}", flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for model in options.models:
        case = make_case(model, options.particles, options.field)
        print(f"{model}: compiling tight reference...", flush=True)
        reference_fn = make_solve(case, "diffrax_dopri8", True, 1e-12)
        compiled, result, reference_timing = compile_solve(reference_fn, case["y0"])
        reference, _ = check_output(result, case["maxtime"])
        del compiled, result, reference_fn
        for mode in options.modes:
            compiled_methods, rows = {}, {}
            for implementation in METHODS:
                print(f"{model}/{mode}/{implementation}: compile and validate...", flush=True)
                function = make_solve(case, implementation, mode == "adaptive", options.tolerance,
                                      options.error_control, options.length_scale)
                compiled, result, timing = compile_solve(function, case["y0"])
                trajectory, counts = check_output(result, case["maxtime"])
                error = np.abs(trajectory - reference)
                position_error = float(np.max(error[..., :3]))
                velocity_error = float(np.max(error[..., 3:]) / case["speed"])
                if max(position_error, velocity_error) > options.accuracy_limit:
                    raise RuntimeError(f"{implementation} failed accuracy gate: {position_error=}, {velocity_error=}")
                if mode == "fixed" and not np.all(counts[:, 1] == 256):
                    raise RuntimeError(f"Fixed-grid comparison has unequal work: {counts}")
                row = dict(model=model, mode=mode, implementation=implementation, **timing,
                           maxtime=case["maxtime"], dt0=case["dt0"], saves=len(case["times"]),
                           position_error_m=position_error, velocity_error_over_speed=velocity_error,
                           velocity_error_m_s=velocity_error * case["speed"],
                           counts_per_particle=counts.tolist(), accepted_mean=float(counts[:, 1].mean()),
                           accepted_max=int(counts[:, 1].max()), rejected_mean=float(counts[:, 2].mean()),
                           rejected_max=int(counts[:, 2].max()), reference_timing=reference_timing,
                           warm_samples_s=[])
                # Warm the device allocator and instruction/data caches as well.
                jax.block_until_ready(compiled(case["y0"]))
                compiled_methods[implementation], rows[implementation] = compiled, row
            print(f"{model}/{mode}: interleaved warm timings...", flush=True)
            # All four executables must be warm after the last cache clear.
            for executable in compiled_methods.values():
                jax.block_until_ready(executable(case["y0"]))
            for _ in range(options.repeats):
                for implementation in rng.permutation(METHODS):
                    started = perf_counter()
                    jax.block_until_ready(compiled_methods[implementation](case["y0"]))
                    rows[implementation]["warm_samples_s"].append(perf_counter() - started)
            for implementation in METHODS:
                row = rows[implementation]
                samples = np.asarray(row["warm_samples_s"])
                row.update(warm_median_ms=float(1000*np.median(samples)), warm_min_ms=float(1000*samples.min()),
                           warm_q25_ms=float(1000*np.quantile(samples, .25)), warm_q75_ms=float(1000*np.quantile(samples, .75)))
                report["rows"].append(row)
                print(f"  {implementation:16s} {row['warm_median_ms']:9.3f} ms; "
                      f"accepted {row['accepted_mean']:.1f}, rejected {row['rejected_mean']:.1f}; "
                      f"position error {row['position_error_m']:.2e} m", flush=True)
            options.output.write_text(json.dumps(report, indent=2) + "\n")
            compiled_methods.clear()
            del compiled, executable, function, result
            jax.clear_caches()
            gc.collect()
    print(f"Saved {options.output}", flush=True)


if __name__ == "__main__":
    main()
