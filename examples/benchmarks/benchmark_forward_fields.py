"""Forward-only time/memory matrix for coils, VMEC and Boozer GC fields.

Each measured case runs in a fresh process, sequentially. The Boozer transform
and the common coordinate-field births are prepared separately and saved.
No reverse executable, particle chunks or alternate collision sampler is used.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[2]
CASES = ("native_orbit","diffrax_orbit","native_split_adaptive","native_split_fixed",
         "diffrax_sde_fixed","boozer_rk4","native_orbit_fixed_dopri5","native_orbit_fixed_dopri8",
         "diffrax_orbit_fixed_dopri8")


def sources():
    names = ("essos/rk.py","essos/rk_fixed.py","essos/rk_rhs.py","essos/rk_dense.py","essos/rk_dense_high.py",
        "essos/split.py","essos/collisions.py","essos/collision_controller.py","essos/collision_error.py",
        "essos/dynamics.py","essos/boozer.py","essos/fields.py","essos/coils.py","essos/orbit_error.py",
        "essos/background_species.py","essos/coulomb.py","essos/electric_field.py",
        "examples/benchmarks/benchmark_ad_backends.py","examples/benchmarks/forward_field_cases.py",
        "examples/benchmarks/benchmark_forward_fields.py")
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def resident():
    result = {}
    if sys.platform=="linux":
        for line in Path("/proc/self/status").read_text().splitlines():
            name,_,value = line.partition(":")
            if name in ("VmRSS","VmHWM"):
                result[name] = int(value.split()[0])*1024
    return result


def worker(options):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import diffrax
    from .benchmark_ad_backends import memory
    field,model,case = options.worker_field,options.worker_model,options.worker_case
    target = options.output/f"{field}_{model}_{case}.json"
    report = dict(complete=False,field=field,model=model,case=case,source_sha256=sources(),
        jax=jax.__version__,diffrax=diffrax.__version__,backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],device_kinds=[d.device_kind for d in jax.devices()],
        particle_chunking=False,reverse_ad=False)
    print(f"JAX backend: {report['backend']}; devices: {report['devices']}",flush=True)
    def save():
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    if field=="coils":
        from .benchmark_ad_backends import build
        parameters,cost,scale,metadata = build(options,model,case)
        def forward(p):
            value,(states,counts,status) = cost(p,"none")
            return value,(states,counts,status,states[:,-1],jnp.full(options.particles,metadata["horizon"]))
        metadata.update(field="QA BiotSavart coils",input_sha256=hashlib.sha256(
            (ROOT/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json").read_bytes()).hexdigest())
    else:
        from .forward_field_cases import build
        parameters,forward,scale,metadata = build(options,field,case)
    report["metadata"] = metadata
    report["setup_rss_bytes"] = resident()
    report["setup_device_memory_stats"] = [d.memory_stats() for d in jax.devices()]
    save()
    function = jax.jit(forward)
    start = perf_counter()
    executable = function.lower(parameters).compile()
    report["compile_seconds"] = perf_counter()-start
    report["compile_device_memory_stats"] = [d.memory_stats() for d in jax.devices()]
    report["compiled_memory"] = memory(executable)
    save()
    start = perf_counter()
    result = jax.block_until_ready(executable(parameters))
    report["first_execution_seconds"] = perf_counter()-start
    report["first_device_memory_stats"] = [d.memory_stats() for d in jax.devices()]
    report["first_peak_rss_bytes"] = resident()
    baseline = resident()
    if sys.platform=="linux":
        Path("/proc/self/clear_refs").write_text("5\n")
    samples = []
    for _ in range(options.repeats):
        start = perf_counter()
        repeated = jax.block_until_ready(executable(parameters))
        samples.append(1000*(perf_counter()-start))
        del repeated
    # Capture the complete timing/memory evidence before validation creates
    # any additional executable. These are host RSS, not peak GPU VRAM.
    report.update(warm_samples_ms=samples,warm_median_ms=float(np.median(samples)),
        warm_baseline_rss_bytes=baseline,warm_peak_rss_bytes=resident(),
        device_memory_stats=[d.memory_stats() for d in jax.devices()],
        memory_scope="Fresh worker: imports/field setup/one compile/forward executions only. RSS is CPU host memory, not GPU VRAM; compiled buffers are a separate estimate.",
        timing_scope="Blocked warm device calls; compile, host copies, validation and field transform excluded. Scalar objective and normal output reconstruction included.")
    value,(states,counts,status,terminal,end) = result
    states,counts,status,terminal,end = map(np.asarray,(states,counts,status,terminal,end))
    report.update(value=float(value),counts_per_particle=counts.tolist(),status=status.tolist(),
        terminal_times=end.tolist(),finite_states=bool(np.isfinite(states).all() and np.isfinite(terminal).all()),
        complete=bool(np.isin(status,[0,1]).all() and np.isfinite(states).all() and np.isfinite(terminal).all()))
    report["peak_rss_bytes"] = max(report["first_peak_rss_bytes"].get("VmHWM",0),report["warm_peak_rss_bytes"].get("VmHWM",0))
    report["saved_states"] = states.tolist()
    report["state_scale"] = np.asarray(scale).tolist()
    report["production_api_check"] = None
    save()
    if not report["complete"]:
        raise RuntimeError("Incomplete/nonfinite forward solve; inspect statuses")
    if options.particles<=8 and field!="coils" and (case=="native_orbit" or case.startswith("native_orbit_fixed_")):
        from .forward_field_cases import WOUT
        from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
        from essos.dynamics import Particles,Tracing
        from essos.fields import Vmec
        points = np.asarray(metadata["birth_xyz_or_sthetazeta"])
        pitch = np.asarray(metadata["birth_pitch"])
        adaptive = case == "native_orbit"
        integrator = "essos_rk45" if case == "native_orbit_fixed_dopri5" else "essos_dopri8"
        timestep = options.gc_horizon/4 if adaptive else options.gc_horizon/options.fixed_gc_steps
        if field=="vmec":
            actual = Vmec(str(WOUT),ntheta=8,nphi=8)
            birth = Particles(jnp.asarray(points),mass=M,charge=Q,energy=4000*ONE_EV,initial_vparallel_over_v=pitch)
            reference = np.asarray(Tracing(field=actual,particles=birth,
                model="GuidingCenterAdaptative" if adaptive else "GuidingCenter",
                integrator=integrator,maxtime=options.gc_horizon,timestep=timestep,
                times_to_trace=options.saves,rtol=options.tolerance,atol=options.tolerance,
                max_steps=options.max_steps,progress=False,devices=[jax.devices()[0]]).trajectories)
        else:
            from essos.boozer import trace_boozer
            # Reuse the same saved tables and production constructor.
            from essos.boozer import BoozerField
            with np.load(options.field_cache) as data:
                actual = BoozerField.from_booz(data["s_grid"],data["bmnc"],data["xm"],data["xn"],data["iota"],data["G"],
                    data["I"],float(data["psi0"]),int(data["nfp"]),mode_tolerance=options.mode_cut,
                    bmns=data["bmns"] if "bmns" in data else None)
            reference = trace_boozer(actual,*points.T,pitch,speed=float(np.sqrt(2*4000*ONE_EV/M)),mass=M,charge=Q,
                tmax=options.gc_horizon,timestep=timestep,n_save=options.saves,integrator=integrator,adaptive=adaptive,
                rtol=options.tolerance,atol=options.tolerance,max_steps=options.max_steps,
                native_stage_execution=options.stage_execution,
                progress=False,compact=False,devices=[jax.devices()[0]]).states
        valid = np.isfinite(reference).all(axis=-1)
        delta = np.where(valid[...,None],states-reference,0.)/np.asarray(scale)
        if field in ("vmec","boozer"):
            delta[...,1] = np.where(valid,np.arctan2(np.sin(states[...,1]-reference[...,1]),
                                                       np.cos(states[...,1]-reference[...,1])),0.)
        maximum = float(np.max(abs(delta)))
        report["production_api_check"] = dict(max_scaled_state_difference=maximum,
            scope="Untimed comparison with existing public native Tracing/trace_boozer; finite pre-loss saves")
        if maximum>3e-8 or not valid.any():
            report["complete"] = False
            save()
            raise RuntimeError("Benchmark kernel disagrees with public production tracer")
    save()


def supported(field,model,case):
    if field!="coils" and model!="gc":
        return False,"VMEC/Boozer entry points support GC, not Cartesian Lorentz"
    if case=="boozer_rk4" and field!="boozer":
        return False,"Specialized RK4 tracer exists only in Boozer coordinates"
    if case=="native_orbit_fixed_dopri5" and field=="coils":
        return False,"Coil timing comparison selects Dopri8"
    if case=="diffrax_sde_fixed" and field=="boozer":
        return False,"No existing Diffrax Boozer SDE entry point; native Milstein split is retained"
    return True,None


def parent(options):
    import numpy as np
    options.output.mkdir(parents=True,exist_ok=True)
    report_path = options.output/"comparison.json"
    if report_path.exists():
        raise ValueError("Choose a fresh output directory")
    environment = dict(os.environ,ESSOS_CPU_DEVICES="1",ESSOS_PROGRESS="0",ESSOS_XLA_CACHE="")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    if options.platform!="auto":
        environment["JAX_PLATFORMS"] = options.platform
    snapshot = sources()
    report = dict(complete=False,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        source_sha256=snapshot,rows=[],skipped=[],orbit_comparisons=[])
    def save():
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    settings = []
    for key in ("particles","repeats","gc_horizon","lorentz_horizon","fixed_gc_steps","fixed_lorentz_steps",
        "saves","tolerance","collision_tolerance","max_steps","max_orbit_steps","accepted_capacity","orbit_capacity",
        "checkpoint_steps","seed","mboz","nboz","mode_cut","platform","split_integrator","stage_execution"):
        settings += ["--"+key.replace("_","-"),str(getattr(options,key))]
    cached_inputs = options.field_cache
    options.field_cache = options.output.resolve()/"coordinate_inputs.npz"
    if any(f!="coils" for f in options.fields):
        if cached_inputs is not None:
            from .forward_field_cases import WOUT
            with np.load(cached_inputs) as data:
                if (str(data["wout_sha256"]) != hashlib.sha256(WOUT.read_bytes()).hexdigest()
                        or len(data["pitch"]) != options.particles
                        or int(data["seed"]) != options.seed
                        or int(data["mboz"]) != options.mboz or int(data["nboz"]) != options.nboz):
                    raise ValueError("Cached coordinate inputs do not match equilibrium/particles/seed/transform settings")
            if cached_inputs.resolve()!=options.field_cache:
                shutil.copyfile(cached_inputs,options.field_cache)
            print("Reusing exact saved coordinate inputs: "+str(cached_inputs),flush=True)
        else:
            print("Preparing one QA Boozer transform and matched VMEC/Boozer births (outside measurements)",flush=True)
            command = [sys.executable,"-m","examples.benchmarks.benchmark_forward_fields","--prepare",
                       "--output",str(options.output.resolve()),"--field-cache",str(options.field_cache),*settings]
            with (options.output/"setup.log").open("w") as stream:
                subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    for field in options.fields:
        for model in options.models:
            for case in options.cases:
                valid,reason = supported(field,model,case)
                if not valid:
                    report["skipped"].append(dict(field=field,model=model,case=case,reason=reason))
                    continue
                name = f"{field}_{model}_{case}"
                print("START "+name,flush=True)
                command = [sys.executable,"-m","examples.benchmarks.benchmark_forward_fields","--output",str(options.output.resolve()),
                    "--field-cache",str(options.field_cache),"--worker-field",field,"--worker-model",model,"--worker-case",case,*settings]
                start = perf_counter()
                with (options.output/(name+".log")).open("w") as stream:
                    child = subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                path = options.output/(name+".json")
                row = json.loads(path.read_text()) if path.exists() else dict(complete=False,field=field,model=model,case=case)
                row.update(worker_exit_code=child.returncode,worker_wall_seconds=perf_counter()-start)
                if child.returncode:
                    row["complete"] = False
                if row["complete"]:
                    if row["source_sha256"] != snapshot:
                        raise RuntimeError("Numerical sources changed during benchmark")
                    print(f"{name} ({row['backend']}): {row['warm_median_ms']:.3f} ms; buffers {row['compiled_memory']['bytes']/2**20:.3f} MiB; "
                          f"host peak {row['peak_rss_bytes']/2**20:.1f} MiB",flush=True)
                else:
                    print("FAILED "+name+"; see "+str(options.output/(name+".log")),flush=True)
                report["rows"].append(row)
                save()
    for field in options.fields:
        for model in options.models:
            rows = {r["case"]:r for r in report["rows"] if r["field"]==field and r["model"]==model and r["complete"]}
            if "native_orbit" not in rows or "diffrax_orbit" not in rows:
                continue
            a,b = (np.asarray(rows[k]["saved_states"]) for k in ("native_orbit","diffrax_orbit"))
            delta = a-b
            if field!="coils":
                delta[...,1] = np.arctan2(np.sin(delta[...,1]),np.cos(delta[...,1]))
            speed = rows["native_orbit"]["state_scale"][3]
            report["orbit_comparisons"].append(dict(field=field,model=model,
                max_position_or_coordinate_component_difference=np.max(abs(delta[...,:3]),axis=(0,1)).tolist(),
                max_vparallel_or_velocity_error_m_per_s=float(np.max(abs(delta[...,3:4 if model=="gc" else 6]))),
                max_velocity_error_relative_to_birth_speed=float(np.max(abs(delta[...,3:4 if model=="gc" else 6]))/speed),
                interpretation="Same physical RHS/field/births; production controller/norm differences are labelled in each row; these are saved-state global differences, not embedded estimates"))
    report["complete"] = all(r["complete"] for r in report["rows"]) and bool(report["rows"])
    save()
    from .benchmark_ad_backends import device_peak_mib
    fields = ("field","model","case","backend","warm_ms","compile_s","buffers_MiB","peak_host_MiB","warm_peak_host_MiB","peak_JAX_device_MiB","accepted_steps","rejected_steps")
    with (options.output/"comparison.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            if row["complete"]:
                counts = np.asarray(row["counts_per_particle"])
                writer.writerow(dict(field=row["field"],model=row["model"],case=row["case"],backend=row["backend"],warm_ms=row["warm_median_ms"],
                    compile_s=row["compile_seconds"],buffers_MiB=row["compiled_memory"]["bytes"]/2**20,
                    peak_host_MiB=row["peak_rss_bytes"]/2**20,warm_peak_host_MiB=row["warm_peak_rss_bytes"].get("VmHWM",0)/2**20,
                    peak_JAX_device_MiB=device_peak_mib(row["device_memory_stats"]),
                    accepted_steps=json.dumps(counts[:,1].tolist()),rejected_steps=json.dumps(counts[:,2].tolist())))
    if sources()!=snapshot:
        raise RuntimeError("Sources changed during benchmark")
    if not report["complete"]:
        raise RuntimeError("Some forward cases failed; inspect the saved logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fields",nargs="+",choices=("coils","vmec","boozer"),default=["coils","vmec","boozer"])
    parser.add_argument("--models",nargs="+",choices=("gc","lorentz"),default=["gc"])
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    parser.add_argument("--platform",choices=("cpu","cuda","auto"),default="auto",
        help="auto respects JAX's normal backend selection; cuda requires a working NVIDIA backend")
    parser.add_argument("--split-integrator",choices=("essos_dopri8","essos_rk45"),default="essos_dopri8")
    parser.add_argument("--stage-execution",choices=("loop","unrolled"),default="loop")
    for name,value in dict(particles=2,repeats=7,gc_horizon=2e-5,lorentz_horizon=2e-6,fixed_gc_steps=224,
        fixed_lorentz_steps=1792,saves=8,tolerance=1e-9,collision_tolerance=.003,max_steps=4096,max_orbit_steps=512,
        accepted_capacity=128,orbit_capacity=256,checkpoint_steps=16,seed=42,mboz=32,nboz=32,mode_cut=1e-4).items():
        parser.add_argument("--"+name.replace("_","-"),type=type(value),default=value)
    parser.add_argument("--field-cache",type=Path)
    parser.add_argument("--prepare",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--worker-field",choices=("coils","vmec","boozer"),help=argparse.SUPPRESS)
    parser.add_argument("--worker-model",choices=("gc","lorentz"),help=argparse.SUPPRESS)
    parser.add_argument("--worker-case",choices=CASES,help=argparse.SUPPRESS)
    options = parser.parse_args()
    import math
    if options.particles<1 or options.repeats<3 or options.saves<2:
        parser.error("Use positive particles, >=3 repeats and >=2 saves")
    if any(not math.isfinite(x) or x<=0 for x in (options.gc_horizon,options.lorentz_horizon,options.tolerance,options.collision_tolerance)):
        parser.error("Positive finite horizons/tolerances required")
    if min(options.fixed_gc_steps,options.fixed_lorentz_steps,options.max_steps,options.max_orbit_steps,options.mboz,options.nboz)<1:
        parser.error("Positive work/transform limits required")
    if not 0<=options.mode_cut<1:
        parser.error("mode-cut must be in [0,1)")
    if options.prepare:
        from .forward_field_cases import prepare_boozer
        prepare_boozer(options.field_cache,particles=options.particles,seed=options.seed,mboz=options.mboz,nboz=options.nboz)
    elif options.worker_case:
        try:
            worker(options)
        except Exception as error:
            path = options.output/f"{options.worker_field}_{options.worker_model}_{options.worker_case}.json"
            report = json.loads(path.read_text()) if path.exists() else {}
            report.update(complete=False,error=repr(error),traceback=traceback.format_exc())
            path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
            raise
    else:
        parent(options)


if __name__=="__main__":
    main()
