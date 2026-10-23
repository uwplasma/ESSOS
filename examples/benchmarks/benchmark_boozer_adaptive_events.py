"""Controlled 64-particle adaptive Dopri5/8 GPU execution comparison.

The orbit comparisons use the production Boozer field and physical RHS,
identical births, tableau, I-controller, tolerances and save times. The replay case now retains
the crossing stages and retries in the same driver; no RHS replay is needed.
Use --case inline/replay in
separate processes for independent peak-memory measurements. This short local
workload does not predict the 8192-particle, 20 ms benchmark runtime.
``--comparison execution`` compares compiler defaults with GPU loop command
buffers, using retained crossing stages in both cases. The default event
comparison holds compiler options at their defaults to isolate event handling.
``--comparison reductions`` compares separate and grouped stage reductions
with the same CUDA loop command buffers enabled in both cases.
``--comparison fusion`` compares the standard trial with an opt-in fused
CUDA trial. Add --audit-reference for an untimed, twice-refined Diffrax
accuracy check after the timing and memory snapshots.
``--comparison trial_work`` measures controlled, equal-count RK recurrences
without adaptive control, saves or loss handling. It is a kernel diagnostic,
not an orbit-accuracy or fixed-versus-adaptive production comparison.
``--comparison orbit_fusion`` compares the fused trial plus ordinary driver
with one independent adaptive GPU program per orbit, using the same method.
Use --method dopri5 with fusion/orbit_fusion to select Dopri5. Add
--audit-rk4 with --audit-reference to measure fixed RK4 trajectory errors at
the benchmark timestep and two halvings; these are untimed accuracy solves.
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

from essos.boozer import _native_orbits, boozer_compiler_options
from .benchmark_native_boozer_alpha import PARAMETERS, ROOT, setup, MASS, CHARGE, rss


def benchmark_trial_work(options, field, initial, mu, speed):
    """Measure recurrence costs on the target GPU instead of extrapolating.

    RK4 calls the production _rk_step; Dopri8 calls the production ordinary
    or fused trial. Complete Dopri8 trial data are consumed through finite
    checks and the embedded estimate. All cases execute the same count,
    including beyond the equilibrium boundary: this diagnostic has no events.
    """
    from essos.boozer import _rk_step, _chart, guiding_center_rhs
    from essos.rk import _prepare_rhs, _trial_step_unrolled, _C8, _A8, _ERROR8
    from essos.rk_boozer_gpu_trial import prepare_trial_factory

    names = ("rk4", "dopri8_standard", "dopri8_fused")
    if options.case != "both" and options.case not in names:
        raise ValueError(f"trial_work --case must be both or one of {names}")
    cases = names if options.case == "both" else (options.case,)
    factory = prepare_trial_factory(field, MASS, CHARGE) if "dopri8_fused" in cases else None
    f0 = jax.vmap(lambda y, m: guiding_center_rhs(field, y, m, MASS, CHARGE))(initial, mu)
    jax.block_until_ready((initial, mu, f0))
    dt = jnp.asarray(options.trial_timestep, initial.dtype)
    units = jnp.asarray([1., 1., 1., speed])
    report = dict(comparison="trial_work", backend=jax.default_backend(),
        device=jax.devices()[0].device_kind, jax=jax.__version__, particles=options.particles,
        trials_per_particle=options.trial_steps, dt_s=options.trial_timestep,
        recurrence_duration_s=options.trial_steps*options.trial_timestep,
        seed=PARAMETERS["seed"], energy_ev=PARAMETERS["energy_ev"], field_modes=int(field.xm.size),
        repeats=options.repeats, particle_chunking=False, collisions=False, reverse_ad=False,
        scope="Equal-count stage/recurrence diagnostic, no adaptive control, saves, energy diagnostic or loss handling. "
              "Dopri8 embedded estimates and complete trial finite checks are included. "
              "The initial FSAL RHS is prepared outside timing. Kernel materialization and loop execution are included.",
        birth_sha256=hashlib.sha256(np.asarray(initial).tobytes()+np.asarray(mu).tobytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("essos/boozer.py", "essos/rk.py", "essos/rk_boozer_gpu_trial.py",
             "examples/benchmarks/benchmark_boozer_adaptive_events.py")}, rows={})
    executables, results = {}, {}
    for name in cases:
        def one(point, moment, first):
            def physical(t, y, unused):
                value = guiding_center_rhs(field, y, moment, MASS, CHARGE)
                _, r, theta, *_ = _chart(y)
                magnetic = field.modB_derivatives(r, theta, y[2])[0]
                return jnp.where(jnp.isfinite(magnetic) & (magnetic > 0), value, jnp.nan)
            rhs = _prepare_rhs(physical, 0., point, None, coalesce_views=True) if name == "dopri8_standard" else physical
            step = factory(moment) if name == "dopri8_fused" else _trial_step_unrolled
            def advance(index, carry):
                y, derivative, valid, max_error = carry
                if name == "rk4":
                    endpoint = _rk_step(lambda x: guiding_center_rhs(field, x, moment, MASS, CHARGE), y, dt, "rk4")
                    return endpoint, derivative, valid & jnp.isfinite(endpoint).all(), max_error
                trial = step(rhs, index*dt, y, dt, None, derivative, _C8, _A8, _ERROR8)
                finite = (jnp.isfinite(trial.y).all() & jnp.isfinite(trial.error).all() &
                    jnp.isfinite(trial.increments).all() & jnp.isfinite(trial.f1).all())
                return trial.y, trial.f1, valid & finite, jnp.maximum(max_error, jnp.max(abs(trial.error/units)))
            return jax.lax.fori_loop(0, options.trial_steps, advance,
                (point, first, jnp.asarray(True), jnp.zeros((), point.dtype)))
        run = jax.jit(jax.vmap(one), compiler_options=boozer_compiler_options())
        print(f"Compiling trial diagnostic {name}: {options.particles} particles, {options.trial_steps} trials", flush=True)
        start = perf_counter()
        executable = run.lower(initial, mu, f0).compile()
        memory = executable.memory_analysis()
        row = report["rows"][name] = dict(compile_s=perf_counter()-start, samples_ms=[],
            compiler_options=boozer_compiler_options(),
            compiled_buffers_bytes=int(memory.argument_size_in_bytes+memory.output_size_in_bytes+
                memory.temp_size_in_bytes-memory.alias_size_in_bytes))
        executables[name] = executable
        start = perf_counter()
        results[name] = jax.block_until_ready(executable(initial, mu, f0))
        row["first_execution_ms"] = 1000*(perf_counter()-start)
        for _ in range(2):
            jax.block_until_ready(executable(initial, mu, f0))
    for repeat in range(options.repeats):
        for name in (cases if repeat % 2 == 0 else tuple(reversed(cases))):
            start = perf_counter()
            jax.block_until_ready(executables[name](initial, mu, f0))
            report["rows"][name]["samples_ms"].append(1000*(perf_counter()-start))
    for name, row in report["rows"].items():
        endpoint, _, finite, error = map(np.asarray, results[name])
        if not finite.all() or not np.isfinite(endpoint).all():
            raise AssertionError(f"{name}: nonfinite trial diagnostic")
        fresh = 4 if name == "rk4" else 13
        row.update(warm_median_ms=float(np.median(row["samples_ms"])),
            executed_trial_slots=options.particles*options.trial_steps,
            fresh_rhs_per_trial=fresh, timed_rhs_calls=options.particles*options.trial_steps*fresh,
            finite_particles=int(finite.sum()),
            final_states_outside_equilibrium=int(np.sum(np.sum(endpoint[:, :2]**2, axis=1) >= 1)),
            max_birth_scaled_embedded_estimate=float(error.max()) if name != "rk4" else None)
        row["amortized_us_per_trial_batch"] = 1000*row["warm_median_ms"]/options.trial_steps
        row["amortized_us_per_rhs_batch"] = row["amortized_us_per_trial_batch"]/fresh
    if "rk4" in report["rows"]:
        for row in report["rows"].values():
            row["warm_ratio_to_rk4"] = row["warm_median_ms"]/report["rows"]["rk4"]["warm_median_ms"]
    report["peak_memory_scope"] = "cumulative across diagnostic executables and setup"
    report["gpu_memory"] = jax.devices()[0].memory_stats()
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output/"trial_work.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


def audit_reference(field, initial, mu, speed, times, settings, results, *, rk4_timestep=None, native_method="dopri8"):
    """Compare common pre-loss saves with two independently refined solves.

    Positions are measured in the regular flux chart, not Cartesian metres.
    The velocity denominator is birth speed, including near v_parallel=0.
    This diagnostic does not interpret a local tolerance as a global bound.
    """
    import diffrax
    from essos.boozer import guiding_center_rhs
    from essos.dynamics import _EVENT_ROOT_FINDER

    results = dict(results)
    rk4_rows = {}
    if rk4_timestep is not None:
        from .benchmark_native_boozer_alpha import standard_rk4_orbits
        if native_method == "dopri5":
            from essos.rk_boozer_gpu_orbit import prepare_orbit_solver
            d8_solver = prepare_orbit_solver(field, MASS, CHARGE, speed, times,
                settings.timestep, settings.rtol, settings.atol, settings.max_steps, method="dopri8")
            def d8_run(y, moment):
                return _native_orbits(field, y, moment, MASS, CHARGE, speed, times,
                    settings.timestep, settings.rtol, settings.atol, "dopri8", True, "i", .4, .3,
                    "standard", settings.max_steps, "unrolled", _orbit_solver=d8_solver)
            print("Untimed native Dopri8 control for the RK4/Dopri5 accuracy audit", flush=True)
            control = jax.block_until_ready(jax.jit(
                d8_run, compiler_options=boozer_compiler_options())(initial, mu))
            if not np.isfinite(np.asarray(control[0])).all() or not np.isin(np.asarray(control[5]), [0, 1]).all():
                raise AssertionError("Native Dopri8 accuracy control failed")
            results["dopri8_orbit"] = control
        for name, divisor in (("rk4_dt", 1), ("rk4_half_dt", 2), ("rk4_quarter_dt", 4)):
            run, dt, slots = standard_rk4_orbits(field, initial, mu, speed, times,
                rk4_timestep/divisor, settings.seed)
            print(f"Untimed RK4 accuracy solve: {name}, actual dt={dt:g}, {slots} grid steps", flush=True)
            output = jax.block_until_ready(jax.jit(run, compiler_options=boozer_compiler_options())(initial, mu))
            if not np.isfinite(np.asarray(output[0])).all() or not np.isin(np.asarray(output[5]), [0, 1]).all():
                raise AssertionError(f"{name}: fixed RK4 accuracy solve failed")
            results[name] = output
            rk4_rows[name] = dict(requested_dt_s=rk4_timestep/divisor, actual_dt_s=dt,
                grid_steps_per_particle=slots, lost=int(np.sum(np.asarray(output[1]) >= 0)),
                failed=int(np.sum(np.asarray(output[2]) >= 0)), scope="Untimed trajectory-accuracy solve")

    units = jnp.array([1., 1., 1., speed])
    def reference(y0, moments, tolerance):
        def one(y, moment):
            rhs = lambda t, y, args: guiding_center_rhs(field, y, moment, MASS, CHARGE)
            event = lambda t, y, args, **kwargs: 1-y[0]**2-y[1]**2
            out = diffrax.diffeqsolve(diffrax.ODETerm(rhs), diffrax.Dopri8(),
                t0=0., t1=times[-1], dt0=settings.timestep, y0=y,
                stepsize_controller=diffrax.PIDController(rtol=0., atol=tolerance*units,
                    pcoeff=.4, icoeff=.3),
                saveat=diffrax.SaveAt(subs=[diffrax.SubSaveAt(ts=times), diffrax.SubSaveAt(t1=True)]),
                event=diffrax.Event(event, root_finder=_EVENT_ROOT_FINDER),
                max_steps=settings.max_steps, throw=False)
            saved, terminal = out.ys[0], out.ys[1][-1]
            saved = jnp.where(jnp.isfinite(saved).all(axis=1)[:, None], saved, terminal)
            status = jnp.where(out.result == diffrax.RESULTS.successful, 0,
                jnp.where(out.result == diffrax.RESULTS.event_occurred, 1, 4))
            loss = jnp.where(status == 1, out.ts[1][-1], -1.)
            return saved/units, loss, status, jnp.array([out.stats[k] for k in
                ("num_steps", "num_accepted_steps", "num_rejected_steps")])
        return jax.vmap(one)(y0, moments)

    reference = jax.jit(reference, compiler_options=boozer_compiler_options())
    refs = {}
    for tolerance in (1e-12, 1e-13):
        print(f"Untimed independent reference: scaled absolute tolerance {tolerance:g}", flush=True)
        refs[tolerance] = tuple(map(np.asarray, reference(initial, mu, jnp.asarray(tolerance))))
        if not np.isin(refs[tolerance][2], [0, 1]).all() or not np.isfinite(refs[tolerance][0]).all():
            raise AssertionError("Independent reference failed or returned nonfinite states")

    end = np.full(initial.shape[0], settings.horizon)
    for result in (*results.values(), *refs.values()):
        end = np.minimum(end, np.where(np.asarray(result[1]) >= 0, result[1], settings.horizon))
    valid = np.asarray(times)[None, :] <= end[:, None]
    strict = refs[1e-13][0]
    audit = dict(reference="Independent Diffrax Dopri8, PI p=.4/i=.3; two refinements",
        reference_error_norm="RMS with rtol=0 and atol=tolerance*(1,1,1,birth_speed)",
        diagnostic="Common saved times before corresponding first loss; regular flux chart, not metres",
        velocity_scale="birth speed; not relative to instantaneous parallel velocity",
        saved_errors_are_global=True, rows={})
    for name, result in results.items():
        values = np.asarray(result[0])
        r = np.sqrt(np.maximum(values[..., 0], 0))
        chart = np.stack((r*np.cos(values[..., 1]), r*np.sin(values[..., 1]),
            values[..., 2], values[..., 3]/speed), axis=-1)
        delta = np.where(valid[..., None], chart-strict, 0.)
        audit["rows"][name] = dict(
            max_chart_radial_error=float(np.max(np.linalg.norm(delta[..., :2], axis=-1))),
            max_zeta_error_rad=float(np.max(abs(delta[..., 2]))),
            max_birth_speed_scaled_parallel_velocity_error=float(np.max(abs(delta[..., 3]))),
            max_parallel_velocity_error_m_s=float(np.max(abs(delta[..., 3]))*speed),
            reference_status_disagreements=int(np.sum(np.asarray(result[5]) != refs[1e-13][2])))
    delta = np.where(valid[..., None], refs[1e-12][0]-strict, 0.)
    audit["reference_refinement_difference"] = np.max(abs(delta), axis=(0, 1)).tolist()
    audit["reference_refinement_components"] = ["u", "w", "zeta", "v_parallel/birth_speed"]
    audit["reference_work"] = {str(k): dict(accepted=int(v[3][:, 1].sum()),
        rejected=int(v[3][:, 2].sum())) for k, v in refs.items()}
    if rk4_rows:
        audit["rk4_refinement"] = dict(rows=rk4_rows,
            scope="Same births, fields, horizon and common pre-loss saves; no timing comparison or tolerance guarantee",
            rk4_error_control="Fixed steps have no rtol/atol controller",
            reference_uncertainty="The two tighter-reference differences are diagnostics, not rigorous error bounds")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", choices=("events", "execution", "reductions", "fusion", "trial_work", "orbit_fusion"), default="events")
    parser.add_argument("--case", default="both")
    parser.add_argument("--method", choices=("dopri5", "dopri8"), default="dopri8")
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--horizon", type=float, default=2e-4)
    parser.add_argument("--saves", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--trial-steps", type=int, default=128,
        help="trial_work only: controlled recurrence count per particle")
    parser.add_argument("--trial-timestep", type=float, default=1.25e-7,
        help="trial_work only: same recurrence timestep for all methods")
    parser.add_argument("--audit-reference", action="store_true",
        help="Untimed two-refinement Diffrax accuracy audit; fusion/orbit_fusion with both cases only")
    parser.add_argument("--audit-rk4", action="store_true",
        help="With --audit-reference: untimed fixed RK4 accuracy at dt, dt/2 and dt/4")
    parser.add_argument("--field-cache", type=Path, default=ROOT/
        "examples/benchmarks/results/gpu_boozer_fixed_20261022/coordinate_inputs.npz")
    parser.add_argument("--output", type=Path, default=Path("output/boozer_adaptive_events"))
    options = parser.parse_args()
    if options.audit_reference and (options.comparison not in ("fusion", "orbit_fusion") or options.case != "both"):
        parser.error("--audit-reference requires --comparison fusion/orbit_fusion --case both")
    if options.audit_rk4 and not options.audit_reference:
        parser.error("--audit-rk4 requires --audit-reference")
    if options.method == "dopri5" and options.comparison not in ("fusion", "orbit_fusion"):
        parser.error("--method dopri5 is supported for fusion/orbit_fusion")
    if options.particles < 1 or options.horizon <= 0 or options.saves < 2 or options.repeats < 1:
        parser.error("particles, horizon and repeats must be positive; saves >= 2")
    if options.trial_steps < 1 or not np.isfinite(options.trial_timestep) or options.trial_timestep <= 0:
        parser.error("trial-steps and trial-timestep must be positive")
    settings = SimpleNamespace(**(dict(PARAMETERS) | dict(particles=options.particles,
        horizon=options.horizon, saves=options.saves, field_cache=options.field_cache)))
    field, initial, mu, speed, births, _ = setup(settings)
    if options.comparison == "trial_work":
        return benchmark_trial_work(options, field, initial, mu, speed)
    times = jnp.linspace(0., settings.horizon, settings.saves)
    names = {"events": ("inline", "replay"), "execution": ("default", "device_loops"),
             "reductions": ("separate", "grouped"), "fusion": ("standard", "fused"),
             "orbit_fusion": ("trial", "orbit")}[options.comparison]
    if options.case != "both" and options.case not in names:
        parser.error(f"--case must be both, {names[0]} or {names[1]} for this comparison")
    cases = names if options.case == "both" else (options.case,)
    report = dict(comparison=options.comparison, backend=jax.default_backend(), device=jax.devices()[0].device_kind,
        jax=jax.__version__, particles=options.particles, horizon_s=options.horizon,
        saves=options.saves, repeats=options.repeats, energy_ev=settings.energy_ev,
        seed=settings.seed, rtol=settings.rtol, atol=settings.atol,
        controller=settings.controller, method=options.method, stage_execution="unrolled",
        field_modes=int(field.xm.size), particle_chunking=False, particle_compaction=False,
        birth_sha256=hashlib.sha256(np.asarray(initial).tobytes()+np.asarray(mu).tobytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT/"essos"/name).read_bytes()).hexdigest()
                       for name in ("boozer.py", "rk.py", "rk_events.py", "rk_rhs.py")}, rows={})
    executables, results = {}, {}
    for name in cases:
        jax.clear_caches()
        compiler_options = boozer_compiler_options() if (name == "device_loops" or
            options.comparison in ("reductions", "fusion", "orbit_fusion")) else {}
        trial_factory = None
        orbit_solver = None
        if name in ("fused", "trial", "orbit"):
            from essos.rk_boozer_gpu_trial import prepare_trial_factory
            if name != "orbit":
                trial_factory = prepare_trial_factory(field, MASS, CHARGE, method=options.method)
            report["source_sha256"]["rk_boozer_gpu_trial.py"] = hashlib.sha256(
                (ROOT/"essos/rk_boozer_gpu_trial.py").read_bytes()).hexdigest()
        if name == "orbit":
            from essos.rk_boozer_gpu_orbit import prepare_orbit_solver
            orbit_solver = prepare_orbit_solver(field, MASS, CHARGE, speed, times,
                settings.timestep, settings.rtol, settings.atol, settings.max_steps, method=options.method)
            report["source_sha256"]["rk_boozer_gpu_orbit.py"] = hashlib.sha256(
                (ROOT/"essos/rk_boozer_gpu_orbit.py").read_bytes()).hexdigest()
        def run(y, moment):
            return _native_orbits(field, y, moment, MASS, CHARGE, speed, times, settings.timestep,
                settings.rtol, settings.atol, options.method, True, settings.controller,
                .4, .3, "standard", settings.max_steps, "unrolled",
                _defer_event_root=(False if name == "inline" else None),
                _group_rhs_reductions=(name == "grouped") if options.comparison == "reductions" else None,
                _trial_factory=trial_factory, _orbit_solver=orbit_solver)
        print(f"Compiling {name}: {options.particles} particles, adaptive {options.method}", flush=True)
        start = perf_counter()
        executable = jax.jit(run, compiler_options=compiler_options).lower(initial, mu).compile()
        memory = executable.memory_analysis()
        report["rows"][name] = dict(compile_s=perf_counter()-start, samples_ms=[], compiler_options=compiler_options,
            compiled_buffers_bytes=int(memory.argument_size_in_bytes+memory.output_size_in_bytes+
                                      memory.temp_size_in_bytes-memory.alias_size_in_bytes))
        executables[name] = executable
        start = perf_counter()
        results[name] = jax.block_until_ready(executable(initial, mu))
        report["rows"][name]["first_execution_ms"] = 1000*(perf_counter()-start)
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
        before, after = names
        # Check all saved and terminal diagnostics, including both speeds.
        if options.comparison not in ("fusion", "orbit_fusion"):
            for a, b in zip(results[before], results[after]):
                np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-9)
            np.testing.assert_array_equal(results[before][4], results[after][4])
        np.testing.assert_array_equal(results[before][5], results[after][5])
        delta = np.asarray(results[before][0])-np.asarray(results[after][0])
        delta[...,1] = np.arctan2(np.sin(delta[...,1]),np.cos(delta[...,1]))
        report.update(max_saved_component_difference=np.max(abs(delta), axis=(0, 1)).tolist(),
            max_scaled_saved_difference=float(np.max(abs(delta)/np.array([1.,1.,1.,speed,speed]))),
            counts_identical=bool(np.array_equal(results[before][4],results[after][4])),
            time_decrease_percent=100*(1-report["rows"][after]["warm_median_ms"]/
                                      report["rows"][before]["warm_median_ms"]))
    report["peak_memory_scope"] = "independent process" if len(cases) == 1 else "cumulative across both executables"
    report["gpu_memory"] = jax.devices()[0].memory_stats()
    report["host_memory"] = rss()
    if options.audit_reference:
        report["accuracy_audit"] = audit_reference(field, initial, mu, speed, times, settings, results,
            rk4_timestep=options.trial_timestep if options.audit_rk4 else None, native_method=options.method)
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output/f"{options.case}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
