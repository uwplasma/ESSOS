"""CATAPULT-sized alpha workload using native Dopri5/8 or standard Boozer RK4.

Run as a module from the repository root. Edit PARAMETERS or override by CLI.
JAX chooses the machine's normal backend. No collisions, AD or compaction.
The adaptive tolerance values match the published CATAPULT comparison; RK4
uses a fixed timestep without tolerance control. Field representation, error
norm and newly sampled births are recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from essos.boozer import (BoozerField, _native_orbits, _start, _advance,
                         boozer_compiler_options, psi0_from_vmec, trace_boozer)
from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE, ONE_EV
from essos.rk import RKStatus

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = dict(
    particles=8192, horizon=.02, saves=101, energy_ev=3.52e6,
    birth_s=.3, seed=42, timestep=1.25e-7,
    integrator="essos_dopri8",
    rtol=1e-10, atol=1e-10, controller="i", stage_execution="unrolled", max_steps=1_000_000,
    mboz=32, nboz=32, mode_cut=1e-4, repeats=2,
    wout=ROOT/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc",
    output=Path("output/native_boozer_alpha"),
)
CATAPULT_SOURCE = "https://github.com/ColumbiaStellaratorTheory/firm3d/blob/4dbeb5e29fdb4fca9179b2111cb897c576ecf42a/src/firm3dpp/cuda_kernel.cu"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss():
    """Host memory only; this is not a GPU VRAM measurement."""
    path = Path("/proc/self/status")
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM"):
                values[name] = int(value.split()[0])*1024
    return values


def setup(options):
    if options.field_cache is None:
        from booz_xform_jax import Booz_xform
        transform = Booz_xform(verbose=0, mboz=options.mboz, nboz=options.nboz)
        transform.read_wout(str(options.wout), flux=False)
        transform.run()
        with Dataset(options.wout) as data:
            psi0 = psi0_from_vmec(data["phi"][-1])
        values = {name:np.asarray(getattr(transform, key)) for name,key in (
            ("s_grid","s_b"),("bmnc","bmnc_b"),("xm","xm_b"),("xn","xn_b"),
            ("iota","iota"),("G","Boozer_G"),("I","Boozer_I"))}
        sine = getattr(transform, "bmns_b", None)
        if sine is not None:
            values["bmns"] = np.asarray(sine)
        values.update(psi0=psi0, nfp=int(transform.nfp), mboz=options.mboz,
                      nboz=options.nboz, wout_sha256=digest(options.wout))
    else:
        with np.load(options.field_cache) as cache:
            names = ("s_grid","bmnc","xm","xn","iota","G","I","psi0","nfp","mboz","nboz","wout_sha256")
            values = {name:cache[name] for name in names}
            if "bmns" in cache:
                values["bmns"] = cache["bmns"]
        if (str(values["wout_sha256"]) != digest(options.wout) or
                int(values["mboz"]) != options.mboz or int(values["nboz"]) != options.nboz):
            raise ValueError("Cached transform does not match --wout/--mboz/--nboz")
    field = BoozerField.from_booz(values["s_grid"], values["bmnc"], values["xm"], values["xn"],
        values["iota"], values["G"], values["I"], float(values["psi0"]), int(values["nfp"]),
        mode_tolerance=options.mode_cut, bmns=values.get("bmns"))
    rng = np.random.default_rng(options.seed)
    s = np.full(options.particles, options.birth_s)
    theta = rng.uniform(0., 2*np.pi, options.particles)
    zeta = rng.uniform(0., 2*np.pi/field.nfp, options.particles)
    pitch = rng.uniform(-1., 1., options.particles)
    speed = float(np.sqrt(2*options.energy_ev*ONE_EV/MASS))
    magnetic = jax.vmap(field.modB)(jnp.asarray(s), jnp.asarray(theta), jnp.asarray(zeta))
    if not np.all(np.isfinite(magnetic) & (np.asarray(magnetic) > 0)):
        raise ValueError("Birth field must be positive and finite")
    initial = jnp.asarray(np.column_stack((np.sqrt(s)*np.cos(theta), np.sqrt(s)*np.sin(theta), zeta, speed*pitch)))
    mu = speed**2*(1-jnp.asarray(pitch)**2)/(2*magnetic)
    births = dict(s=s, theta=theta, zeta=zeta, pitch=pitch, initial=np.asarray(initial), mu=np.asarray(mu))
    return field, initial, mu, speed, births, values


def plots(folder, times, states, loss_times, errors, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    lost = (loss_times[:,None] >= 0) & (loss_times[:,None] <= times)
    axes[0,0].plot(times*1e3, 100*lost.mean(axis=0))
    axes[0,0].set(xlabel="Time (ms)", ylabel="Lost (%)", title="QA equilibrium boundary: s=1")
    for q in (50, 95, 100):
        axes[0,1].plot(times*1e3, np.percentile(errors,q,axis=0), label=f"{q}th percentile")
    axes[0,1].set(xlabel="Time (ms)", ylabel="Relative energy drift at saved states")
    axes[0,1].legend()
    axes[1,0].hist(counts[:,1], bins=40, alpha=.7, label="Accepted")
    axes[1,0].hist(counts[:,2], bins=40, alpha=.7, label="Rejected")
    axes[1,0].set(xlabel="Steps per particle", ylabel="Particles")
    axes[1,0].legend()
    angles = np.linspace(0., 2*np.pi, 201)
    axes[1,1].plot(np.cos(angles), np.sin(angles), "k--", lw=.8)
    selected = np.random.default_rng(17).choice(len(states), min(10,len(states)), replace=False)
    for i in selected:
        radius = np.sqrt(np.maximum(states[i,:,0],0.))
        axes[1,1].plot(radius*np.cos(states[i,:,1]), radius*np.sin(states[i,:,1]), lw=.8)
    axes[1,1].set(xlabel="sqrt(s) cos(theta)", ylabel="sqrt(s) sin(theta)", aspect="equal",
                  title="Up to ten trajectories: flux chart, not metres")
    fig.savefig(folder/"benchmark.png", dpi=160)
    plt.close(fig)


def standard_rk4_orbits(field, initial, mu, speed, times, timestep, seed):
    """Call the existing production RK4 driver; no new RK or field formula."""
    intervals = len(times)-1
    n_sub = max(1, int(np.ceil(float(times[-1])/(intervals*timestep)-1e-9)))
    dt = float(times[-1])/(intervals*n_sub)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(initial))

    def run(y0, mu0):
        carry, first = _start(field, y0, mu0)
        carry, values = _advance(field, dt, n_sub, MASS, CHARGE, None, 1.5,
            carry, keys, 0, intervals, "rk4", 1e-10)
        states = jnp.concatenate((first[:, None], values), axis=1)
        status = jnp.where(carry[6] >= 0, RKStatus.NONFINITE,
            jnp.where(carry[4] >= 0, RKStatus.EVENT, RKStatus.SUCCESS))
        counts = jnp.stack((carry[10], carry[10], jnp.zeros_like(carry[10])), axis=-1)
        return states, carry[4], carry[6], carry[7], counts, status

    return run, dt, n_sub*intervals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key,value in PARAMETERS.items():
        parser.add_argument("--"+key.replace("_","-"), type=type(value), default=value)
    parser.add_argument("--field-cache", type=Path, help="Reuse a saved Boozer transform; births are sampled afresh")
    parser.add_argument("--setup-only", action="store_true")
    experimental = parser.add_mutually_exclusive_group()
    experimental.add_argument("--experimental-fused-trial", action="store_true",
        help="Opt-in CUDA Dopri5/8 stage kernel; forward only, fixed Fourier field")
    experimental.add_argument("--experimental-fused-orbit", action="store_true",
        help="Opt-in independent adaptive CUDA orbit kernels; Dopri5/8, I control, standard output, fixed Fourier field, forward only")
    options = parser.parse_args(argv)
    positive = (options.particles, options.horizon, options.energy_ev, options.timestep, options.atol,
                options.max_steps, options.mboz, options.nboz, options.repeats)
    if not all(np.isfinite(x) and x>0 for x in positive) or not np.isfinite(options.rtol) or options.rtol<0:
        parser.error("Counts, horizon, energy, timestep and atol must be positive; rtol >= 0")
    if not 0<options.birth_s<1 or options.saves<2 or not 0<=options.mode_cut<1 or options.controller not in ("i","pi"):
        parser.error("Require 0 < birth-s < 1, saves >= 2, 0 <= mode-cut < 1, controller i/pi")
    if options.stage_execution not in ("loop","unrolled"):
        parser.error("stage-execution must be loop/unrolled")
    if options.integrator not in ("essos_rk45", "essos_dopri8", "rk4"):
        parser.error("integrator must be essos_rk45 (Dopri5), essos_dopri8 or rk4")
    method = {"essos_rk45": "dopri5", "essos_dopri8": "dopri8", "rk4": "rk4"}[options.integrator]
    if options.experimental_fused_trial and (method not in ("dopri5", "dopri8") or options.stage_execution != "unrolled"):
        parser.error("The experimental fused trial requires essos_rk45/essos_dopri8 with unrolled stages")
    if options.experimental_fused_orbit and (method not in ("dopri5", "dopri8") or options.controller != "i" or
            options.stage_execution != "unrolled"):
        parser.error("The experimental fused orbit requires essos_rk45/essos_dopri8, I control and unrolled stages")
    if (options.output/"benchmark.json").exists():
        parser.error("Choose a fresh --output directory")
    options.output.mkdir(parents=True, exist_ok=True)
    print(f"JAX backend: {jax.default_backend()}; devices: {jax.devices()}", flush=True)
    print("Preparing QA Boozer field and births (excluded from tracing timings)", flush=True)
    start = perf_counter()
    field, initial, mu, speed, births, tables = setup(options)
    times = jnp.linspace(0.,options.horizon,options.saves)
    jax.block_until_ready((field,initial,mu,times))
    report = dict(complete=False, jax=jax.__version__, backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()], device_kinds=[d.device_kind for d in jax.devices()],
        platform=platform.platform(),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        setup_seconds=perf_counter()-start, collisions=False, reverse_ad=False, particle_chunking=False,
        particle_compaction=False, integrator=options.integrator, adaptive=method != "rk4",
        component_stage_arithmetic=(method != "rk4" and options.stage_execution == "unrolled"
                                    and not (options.experimental_fused_trial or options.experimental_fused_orbit)),
        field=dict(representation="Production Boozer K=0 GC field from QA equilibrium; no coils",
            modes=int(field.xm.size), nfp=field.nfp, angular_table=field.table is not None,
            wout_sha256=digest(options.wout)),
        births="Fixed s shell; uniform Boozer theta, zeta over one field period; uniform pitch [-1,1]; seed recorded",
        stopping=("Standard RK4 endpoint exit at s>=1; later saves hold that state; no compaction"
            if method == "rk4" else "Root-found exit at s=1; later saves hold the boundary state; inactive lanes are not compacted"),
        error_norm=("Fixed RK4: no embedded error control; rtol/atol are unused" if method == "rk4" else
            "max(norm(e_u,e_w), abs(e_zeta), abs(e_vpar)/birth_speed)/(rtol+atol), u/w=sqrt(s)*cos/sin(theta)"),
        catapult_reference=dict(source=CATAPULT_SOURCE, tolerance=1e-10, atol=1e-10, rtol=1e-10,
            distinction=("Standard fixed RK4, no tolerance controller. This matches the current native field/births, not the exact published CATAPULT inputs."
                if method == "rk4" else "Same tolerance values, not the same error norm: CATAPULT uses atol_i + rtol*(abs(y_i)+dt*abs(f_i)); native block scales are fixed. This is a CATAPULT-sized workload, not identical field/births or a direct speed comparison.")),
        cache_directory=jax.config.jax_compilation_cache_dir,
        event_root_strategy=("retained crossing stages; retry in the same adaptive loop"
            if method == "dopri8" and options.stage_execution == "unrolled" and jax.default_backend() == "gpu"
            else "fixed RK4 endpoint check" if method == "rk4" else "inline root search"),
        rhs_execution="ordinary stage reductions",
        trial_kernel=("experimental Pallas independent adaptive orbit loops" if options.experimental_fused_orbit else
            "experimental Pallas stage loop" if options.experimental_fused_trial else "standard JAX stages"),
        source_sha256={name:digest(ROOT/name) for name in ("essos/boozer.py","essos/rk.py","essos/rk_events.py","essos/rk_fixed.py","essos/rk_rhs.py",
            "essos/rk_dense.py","examples/benchmarks/benchmark_native_boozer_alpha.py")})
    if options.experimental_fused_orbit:
        report["event_root_strategy"] = "standard interpolant root search inside each independent GPU orbit"
    report["setup_device_memory_stats"] = [d.memory_stats() for d in jax.devices()]
    birth_digest = hashlib.sha256()
    for name, values in births.items():
        birth_digest.update(name.encode())
        birth_digest.update(np.ascontiguousarray(values).tobytes())
    report["births_sha256"] = birth_digest.hexdigest()
    def save():
        (options.output/"benchmark.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    np.savez_compressed(options.output/"boozer_inputs.npz", **tables)
    np.savez_compressed(options.output/"births.npz", **births)
    print(f"QA Boozer: {report['field']['modes']} retained modes, mode-cut={options.mode_cut:g}; "
          f"{'angular table' if field.table is not None else 'direct Fourier sums'}; K=0",flush=True)
    stepping = (f"fixed standard RK4; requested dt={options.timestep:g}; no tolerance controller"
        if method == "rk4" else f"adaptive native {method} ({options.controller}); "
        f"rtol={options.rtol:g}, atol={options.atol:g}; stages={options.stage_execution}")
    print(f"{options.particles} alpha GCs, {options.energy_ev:g} eV, s={options.birth_s:g}, "
          f"{options.horizon:g} s, {options.saves} saves; {stepping}",flush=True)
    print(f"Loss roots: {report['event_root_strategy']}", flush=True)
    print(f"RHS execution: {report['rhs_execution']}", flush=True)
    print(f"Trial kernel: {report['trial_kernel']}", flush=True)
    if options.setup_only:
        print(f"Setup saved in {options.output.resolve()}",flush=True)
        return report
    trial_factory = orbit_solver = None
    if options.experimental_fused_trial:
        from essos.rk_boozer_gpu_trial import prepare_trial_factory
        trial_factory = prepare_trial_factory(field, MASS, CHARGE, method=method)
        report["source_sha256"]["essos/rk_boozer_gpu_trial.py"] = digest(ROOT/"essos/rk_boozer_gpu_trial.py")
    if options.experimental_fused_orbit:
        from essos.rk_boozer_gpu_orbit import prepare_orbit_solver
        orbit_solver = prepare_orbit_solver(field, MASS, CHARGE, speed, times,
            options.timestep, options.rtol, options.atol, options.max_steps, method=method)
        for name in ("rk_boozer_gpu_trial.py", "rk_boozer_gpu_orbit.py"):
            report["source_sha256"]["essos/"+name] = digest(ROOT/"essos"/name)
    def run(y0,mu0):
        return _native_orbits(field,y0,mu0,MASS,CHARGE,speed,times,options.timestep,
            options.rtol,options.atol,method,True,options.controller,.4,.3,"standard",options.max_steps,options.stage_execution,
            _trial_factory=trial_factory, _orbit_solver=orbit_solver)
    if method == "rk4":
        run, actual_dt, grid_steps = standard_rk4_orbits(field, initial, mu, speed, times,
            options.timestep, options.seed)
        report.update(fixed_dt=actual_dt, fixed_grid_steps=grid_steps,
            count_semantics="RK4 executed grid slots per lane, including inactive held lanes; no error rejections",
            energy_check_scope="Standard production RK4 checks every executed active step")
        print(f"RK4 actual dt: {actual_dt:g} s; grid slots per lane: {grid_steps}", flush=True)
    compiler_options = (boozer_compiler_options() if method == "rk4" or options.experimental_fused_orbit or options.experimental_fused_trial or
                        (method == "dopri8" and options.stage_execution == "unrolled") else {})
    report["compiler_options"] = compiler_options
    print(f"GPU execution options: {compiler_options}", flush=True)
    print("Lowering and compiling the production Boozer kernel",flush=True)
    start = perf_counter()
    executable = jax.jit(run, compiler_options=compiler_options).lower(initial,mu).compile()
    report["lower_and_compile_seconds"] = perf_counter()-start
    report["compile_device_memory_stats"] = [d.memory_stats() for d in jax.devices()]
    memory = executable.memory_analysis()
    if memory is not None:
        parts = {key:int(getattr(memory,key)) for key in ("argument_size_in_bytes","output_size_in_bytes",
                    "temp_size_in_bytes","alias_size_in_bytes")}
        report["compiled_buffer_components"] = parts
        report["compiled_buffer_bytes"] = parts["argument_size_in_bytes"]+parts["output_size_in_bytes"]+parts["temp_size_in_bytes"]-parts["alias_size_in_bytes"]
    report["compile_host_rss_bytes"] = rss()
    save()
    print(f"Lowering/compilation: {report['lower_and_compile_seconds']:.3f} s; starting first execution",flush=True)
    start = perf_counter()
    result = jax.block_until_ready(executable(initial,mu))
    report["first_execution_seconds"] = perf_counter()-start
    report["first_device_memory_stats"] = [d.memory_stats() for d in jax.devices()]
    report["first_trace_including_compile_seconds"] = report["lower_and_compile_seconds"]+report["first_execution_seconds"]
    report["first_host_rss_bytes"] = rss()
    print(f"First execution: {report['first_execution_seconds']:.3f} s",flush=True)
    failed = np.asarray(result[2]) >= 0
    first_counts = np.asarray(result[4])
    report["first_step_counts"] = {name:dict(total=int(first_counts[:,i].sum()),
        min=int(first_counts[:,i].min()), median=float(np.median(first_counts[:,i])),
        max=int(first_counts[:,i].max())) for i,name in enumerate(("trials","accepted","rejected"))}
    fresh_rhs = {"rk4": 4, "dopri5": 6, "dopri8": 13}[method]
    counted_trials = int(first_counts[:,0].sum())
    minimum_batch_slots = (None if options.experimental_fused_orbit else
                           options.particles*int(first_counts[:,0].max()))
    initial_rhs = 0 if method == "rk4" else options.particles
    report["rhs_work_estimate"] = dict(fresh_rhs_per_trial=fresh_rhs,
        initial_rhs_evaluations=initial_rhs,
        counted_trial_slots=counted_trials,
        minimum_batched_trial_slots=minimum_batch_slots,
        counted_rhs_evaluations=initial_rhs+fresh_rhs*counted_trials,
        minimum_batched_rhs_evaluations=(None if minimum_batch_slots is None else
                                         initial_rhs+fresh_rhs*minimum_batch_slots),
        active_slot_fraction_upper_bound=(counted_trials/minimum_batch_slots if minimum_batch_slots else None),
        scope=("Independent per-particle GPU loops; counted RHS work excludes energy checks and root interpolation; "
               "no lockstep batch-slot estimate applies." if options.experimental_fused_orbit else
              "Logical RHS-lane work estimates for the current vmapped driver, not GPU instruction counts. "
              "Root retries can add batch iterations; field-only energy checks are excluded. "
              "RK4 counts already include held inactive grid slots."))
    print(f"First-call steps: {report['first_step_counts']}; failures: {int(failed.sum())}", flush=True)
    print(f"RHS work estimate: {report['rhs_work_estimate']}", flush=True)
    report["warm_seconds"] = []
    report["timing_scope"] = "Blocked device-resident solve calls; setup, output transfers and plots excluded. First trace adds lowering/compilation; persistent cache may be hit. Production saved-state energy diagnostics are included."
    save()
    if not failed.any():
        for i in range(options.repeats):
            start = perf_counter()
            repeated = jax.block_until_ready(executable(initial,mu))
            elapsed = perf_counter()-start
            del repeated
            report["warm_seconds"].append(elapsed)
            print(f"Warm {i+1}/{options.repeats}: {elapsed:.3f} s",flush=True)
            save()
    report["peak_host_rss_bytes"] = rss()
    report["device_memory_stats"] = [d.memory_stats() for d in jax.devices()]
    states,loss_times,failed_times,energy,counts,status = map(np.asarray,result)
    errors = abs(states[...,4]**2/speed**2-1)
    finite = bool(np.isfinite(states).all() and np.isfinite(energy).all())
    report.update(warm_median_seconds=float(np.median(report["warm_seconds"])) if report["warm_seconds"] else None,
        failed_particles=int(failed.sum()), lost_particles=int(np.sum(loss_times>=0)), finite_states=finite,
        complete=bool(not failed.any() and finite and np.isin(status,[RKStatus.SUCCESS,RKStatus.EVENT]).all()),
        max_saved_and_terminal_relative_energy_drift=float(energy.max()) if np.isfinite(energy).all() else None,
        step_counts={name:dict(total=int(counts[:,i].sum()), min=int(counts[:,i].min()),
                    median=float(np.median(counts[:,i])), max=int(counts[:,i].max()))
                    for i,name in enumerate(("trials","accepted","rejected"))},
        status_counts={RKStatus(int(k)).name:int(v) for k,v in zip(*np.unique(status,return_counts=True))})
    np.savez_compressed(options.output/"traces.npz",times=np.asarray(times),states=states,loss_times=loss_times,
        failed_times=failed_times,energy_error=energy,counts=counts,status=status,saved_relative_energy_error=errors)
    save()
    if not report["complete"]:
        raise RuntimeError("Incomplete/nonfinite traces; warm repeats skipped on failure. Inspect benchmark.json/traces.npz")
    if not options.experimental_fused_trial and options.horizon<=1e-4 and (options.particles<=8 or
            (method == "rk4" and options.particles<=64)):
        reference = trace_boozer(field,births["s"],births["theta"],births["zeta"],births["pitch"],
            speed=speed,mass=MASS,charge=CHARGE,tmax=options.horizon,timestep=options.timestep,n_save=options.saves,
            integrator=None if method == "rk4" else options.integrator,
            method="rk4" if method == "rk4" else None,
            rtol=options.rtol,atol=options.atol,native_controller=options.controller,
            native_stage_execution=options.stage_execution,
            max_steps=options.max_steps,progress=False,compact=False,devices=[jax.devices()[0]])
        np.testing.assert_allclose(states,reference.states,rtol=0,atol=1e-8)
        np.testing.assert_allclose(loss_times,reference.loss_times,rtol=0,atol=1e-13)
        np.testing.assert_allclose(failed_times,reference.failed_times,rtol=0,atol=1e-13)
        if method != "rk4":
            np.testing.assert_array_equal(counts,reference.step_stats)
            np.testing.assert_array_equal(status,reference.solver_status)
        report["production_api_check"] = dict(max_state_difference=float(np.max(abs(states-reference.states))),
            scope="Untimed public trace_boozer check after timing and memory snapshots; small runs only")
        save()
    plots(options.output,np.asarray(times),states,loss_times,errors,counts)
    print(f"Warm median: {report['warm_median_seconds']:.3f} s; lost: {report['lost_particles']}/{options.particles}; "
          f"failed: {report['failed_particles']}; accepted steps min/median/max: "
          f"{report['step_counts']['accepted']}",flush=True)
    print(f"Saved {options.output.resolve()}",flush=True)
    return report


if __name__ == "__main__":
    main()
