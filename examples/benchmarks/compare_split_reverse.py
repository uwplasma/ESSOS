"""Compare a committed split adjoint with the working implementation on CPU.

By default each implementation/model runs in a fresh process; interleaved mode
compiles both adjoints before alternating warm calls. The baseline split driver
and adjoint come from a commit; an optional adjoint-source override isolates a
kinetic-rule change from earlier checkpoint improvements. Physical kernels,
inputs and benchmark code are shared. Peak RSS includes compilation and execution. Linux/WSL
/proc monitoring reads only the worker spawned by this script, without threads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import sleep


ROOT = Path(__file__).resolve().parents[2]
WORKER = """
import importlib.util
import sys
source = sys.argv[1]
if source:
    import essos
    from pathlib import Path
    for name in ('split', 'split_ad'):
        spec = importlib.util.spec_from_file_location('essos.' + name, Path(source)/(name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules['essos.' + name] = module
        setattr(essos, name, module)
        spec.loader.exec_module(module)
sys.argv = ['benchmark_split_ad'] + sys.argv[2:]
from examples.benchmarks.benchmark_split_ad import main
main()
"""


def paired(options, sources, output):
    """Compile both adjoints before alternating blocked warm measurements.

    The physical driver must be identical; only the reverse module is replaced.
    Memory here is executable buffers, since both versions share one process.
    Use the default isolated mode separately for process peak RSS.
    """
    import importlib.util
    from time import perf_counter
    from types import SimpleNamespace
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
    os.environ.setdefault("ESSOS_PROGRESS", "0")
    import essos
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .benchmark_split_ad import build_case, SOURCES
    if jax.default_backend() != "cpu":
        raise ValueError("paired split measurements require CPU")
    if sources["essos/split.py"] != (ROOT/"essos/split.py").read_bytes():
        raise ValueError("paired mode requires the identical forward split driver")
    shared = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES
              if name != "essos/split_ad.py"}
    shared["examples/benchmarks/compare_split_reverse.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings = SimpleNamespace(particles=options.particles,saves=8,method="dopri8",controller="conservative",
        gc_horizon=options.gc_horizon,lorentz_horizon=options.lorentz_horizon,tolerance=1e-9,
        collision_tolerance=.003,max_steps=128,max_orbit_steps=128,checkpoint_steps=16,
        orbit_checkpoint_steps=16,accepted_capacity=128,orbit_capacity=2048,fixed_orbit_steps=64)
    report=dict(complete=False,metadata=dict(backend="cpu",jax=jax.__version__,settings=vars(settings),
        repeats=options.repeats,shared_source_sha256=shared,
        input_sha256=hashlib.sha256((ROOT/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json").read_bytes()).hexdigest(),
        timing="Both versions warmed before seeded alternating blocked calls; compilation excluded; reverse reuses baseline residuals",
        memory="Fused executable buffers; process RSS is not measured in paired mode",no_particle_chunking=True,
        baseline_source="Explicit source override or baseline commit; hashes below identify the actual reverse code"),rows=[])

    def save():
        (output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")

    def compile_case(path, model):
        spec=importlib.util.spec_from_file_location("essos.split_ad",path)
        module=importlib.util.module_from_spec(spec)
        sys.modules["essos.split_ad"]=module
        setattr(essos,"split_ad",module)
        spec.loader.exec_module(module)
        theta,cost,scale,_=build_case(settings,model,True,"mean_final_radius",None)
        forward=jax.jit(lambda p:cost(p,"none"))
        linear=jax.jit(lambda p:cost(p,"accepted"))
        value,pullback,aux=jax.block_until_ready(jax.vjp(linear,theta,has_aux=True))
        seed=jnp.ones_like(value)
        reverse=jax.jit(lambda pb,seed:pb(seed)[0])
        reverse_gradient=jax.block_until_ready(reverse(pullback,seed))
        total=jax.jit(jax.value_and_grad(lambda p:cost(p,"accepted"),has_aux=True))
        start=perf_counter()
        executable=total.lower(theta).compile()
        compilation=perf_counter()-start
        (fused_value,fused_aux),gradient=jax.block_until_ready(total(theta))
        ordinary_value,ordinary_aux=jax.block_until_ready(forward(theta))
        for v in (fused_value,ordinary_value):
            np.testing.assert_allclose(value,v,rtol=2e-12,atol=1e-25)
        np.testing.assert_allclose(gradient,reverse_gradient,rtol=2e-8,atol=1e-11)
        for candidate in (fused_aux,ordinary_aux):
            np.testing.assert_allclose((aux[0]-candidate[0])/scale,0.,atol=2e-12,rtol=0.)
            for a,b in zip(jax.tree.leaves(aux[1:]),jax.tree.leaves(candidate[1:])):
                np.testing.assert_allclose(a,b,rtol=2e-12,atol=1e-25)
        if np.any(aux[2]) or not np.isfinite(gradient).all():
            raise RuntimeError("failed or nonfinite split gradient")
        memory=executable.memory_analysis()
        result=dict(value=float(value),gradient=np.asarray(gradient).tolist(),
            stats={k:np.asarray(v).tolist() for k,v in aux[1]._asdict().items()},status=np.asarray(aux[2]).tolist(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),compile_seconds=compilation,
            compiler_buffer_bytes=int(memory.argument_size_in_bytes+memory.output_size_in_bytes+memory.temp_size_in_bytes-memory.alias_size_in_bytes),
            samples_ms=[])
        return result,(lambda:forward(theta),lambda:reverse(pullback,seed),lambda:total(theta)),(theta,cost,scale,aux[0])

    with tempfile.TemporaryDirectory(prefix="essos-split-paired-") as directory:
        baseline=Path(directory)/"split_ad.py"
        baseline.write_bytes(sources["essos/split_ad.py"])
        for model in options.models:
            cases=[]
            for label,path in (("before",baseline),("after",ROOT/"essos/split_ad.py")):
                print(f"Compiling paired {model}: {label}",flush=True)
                cases.append(compile_case(path,model))
            before,after=[case[0] for case in cases]
            np.testing.assert_allclose(before["gradient"],after["gradient"],rtol=2e-8,atol=1e-11)
            np.testing.assert_allclose(before["value"],after["value"],rtol=2e-12,atol=1e-25)
            for key in before["stats"]:
                np.testing.assert_allclose(before["stats"][key],after["stats"][key],rtol=2e-12,atol=1e-25)
            random=np.random.default_rng(23)
            for _ in range(options.repeats):
                for which in random.permutation(2):
                    result,functions,_=cases[which]
                    sample={}
                    for name,function in zip(("forward","reverse","total"),functions):
                        start=perf_counter();jax.block_until_ready(function())
                        sample[name]=1000*(perf_counter()-start)
                    result["samples_ms"].append(sample)
            for result,_,_ in cases:
                result["medians_ms"]={key:float(np.median([r[key] for r in result["samples_ms"]]))
                                      for key in ("forward","reverse","total")}
                result["reverse_over_forward"]=result["medians_ms"]["reverse"]/result["medians_ms"]["forward"]
                result["total_over_forward"]=result["medians_ms"]["total"]/result["medians_ms"]["forward"]
            if options.validate_replay:
                theta,cost,scale,ys=cases[1][2]
                recorded,tapes=jax.block_until_ready(jax.jit(lambda p:cost(p,"none",record=True))(theta))
                frozen=jax.jit(lambda p:cost(p,"none",replay_tapes=tapes))
                replay_value,(states,final)=jax.block_until_ready(frozen(theta))
                np.testing.assert_allclose((states-ys)/scale,0.,rtol=0.,atol=2e-12)
                np.testing.assert_allclose((final-recorded.y)/scale,0.,rtol=0.,atol=2e-12)
                np.testing.assert_allclose(replay_value,after["value"],rtol=2e-12,atol=1e-25)
                direction=jnp.asarray(np.random.default_rng(12).normal(size=theta.size))
                direction/=jnp.linalg.norm(direction)
                expected=float(jnp.dot(jnp.asarray(after["gradient"]),direction))
                tangent=jax.block_until_ready(jax.jit(lambda p,d:jax.jvp(
                    lambda p:cost(p,"none",replay_tapes=tapes)[0],(p,),(d,))[1])(theta,direction))
                np.testing.assert_allclose(tangent,expected,rtol=2e-7,atol=1e-10)
                fd=[]
                for epsilon in (1e-5,3e-6,1e-6):
                    finite=float((frozen(theta+epsilon*direction)[0]-frozen(theta-epsilon*direction)[0])/(2*epsilon))
                    fd.append(dict(epsilon=epsilon,ad=expected,fd=finite,
                        relative_error=abs(finite-expected)/max(abs(finite),abs(expected),1e-30)))
                if min(point["relative_error"] for point in fd)>2e-5:
                    raise RuntimeError("frozen-grid/noise directional gradient mismatch")
                after.update(frozen_directional_jvp=float(tangent),frozen_directional_fd=fd,
                    max_normalized_replay_difference=float(jnp.max(jnp.abs((states-ys)/scale))))
            row=dict(model=model,before=before,after=after,
                reverse_speedup=before["medians_ms"]["reverse"]/after["medians_ms"]["reverse"],
                total_speedup=before["medians_ms"]["total"]/after["medians_ms"]["total"],
                compiler_buffer_ratio=after["compiler_buffer_bytes"]/before["compiler_buffer_bytes"],
                max_abs_gradient_difference=float(np.max(np.abs(np.asarray(before["gradient"])-after["gradient"]))))
            report["rows"].append(row);save()
            print(json.dumps({key:row[key] for key in ("model","reverse_speedup","total_speedup","compiler_buffer_ratio","max_abs_gradient_difference")}
                | {"after_medians_ms":after["medians_ms"],"reverse_over_forward":after["reverse_over_forward"],"total_over_forward":after["total_over_forward"]}),flush=True)
    for name,digest in shared.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"shared source changed during benchmark: {name}")
    if hashlib.sha256((ROOT/"essos/split_ad.py").read_bytes()).hexdigest()!=report["rows"][-1]["after"]["source_sha256"]:
        raise RuntimeError("reverse source changed during benchmark")
    if hashlib.sha256((ROOT/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json").read_bytes()).hexdigest()!=report["metadata"]["input_sha256"]:
        raise RuntimeError("physical inputs changed during benchmark")
    report["complete"]=True;save()


def worker_rss(pid):
    try:
        peak = 0
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:","VmHWM:")):
                peak = max(peak,1024*int(line.split()[1]))
        return peak
    except (FileNotFoundError,ProcessLookupError):
        pass
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-revision",required=True,help="Commit containing the reference split_ad.py")
    parser.add_argument("--baseline-reverse-source",type=Path,
        help="Use this split_ad.py with the committed split.py, to isolate the GC kinetic rule")
    parser.add_argument("--baseline-driver-source",type=Path,
        help="Override baseline split.py as well; use the current driver when comparing reverse implementations only")
    parser.add_argument("--interleaved",action="store_true",
        help="Compile both adjoints in one process and interleave warm timings; compiler buffers only, not RSS")
    parser.add_argument("--validate-replay",action="store_true",
        help="Validate gradients against production frozen-grid/noise replay JVPs and finite differences")
    parser.add_argument("--output",type=Path,required=True,help="New results directory")
    parser.add_argument("--models",nargs="+",choices=("gc","lorentz"),default=["gc","lorentz"])
    parser.add_argument("--particles",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--gc-horizon",type=float,default=2e-6)
    parser.add_argument("--lorentz-horizon",type=float,default=2e-7)
    options = parser.parse_args(argv)
    if sys.platform != "linux" or not Path("/proc/self/status").is_file():
        parser.error("run under Linux/WSL for worker peak-RSS sampling")
    if options.output.exists() or not 1 <= options.particles <= 8 or options.repeats < 3:
        parser.error("choose a new output directory, 1..8 particles, and at least three repeats")
    revision = subprocess.check_output(["git","rev-parse",f"{options.baseline_revision}^{{commit}}"],
        cwd=ROOT,text=True).strip()
    reference_sources = {name:subprocess.check_output(["git","show",f"{revision}:{name}"],cwd=ROOT)
        for name in ("essos/split.py", "essos/split_ad.py")}
    if options.baseline_reverse_source is not None:
        if not options.baseline_reverse_source.is_file():
            parser.error("baseline reverse source must be an existing Python file")
        reference_sources["essos/split_ad.py"] = options.baseline_reverse_source.read_bytes()
    if options.baseline_driver_source is not None:
        if not options.baseline_driver_source.is_file():
            parser.error("baseline driver source must be an existing Python file")
        reference_sources["essos/split.py"] = options.baseline_driver_source.read_bytes()
    current_sources = {name:(ROOT/name).read_bytes() for name in reference_sources}
    source_hashes = {label:{name:hashlib.sha256(source).hexdigest() for name,source in sources.items()}
        for label,sources in (("before",reference_sources),("after",current_sources))}
    options.output.mkdir(parents=True)
    output = options.output.resolve()
    if options.interleaved:
        paired(options,reference_sources,output)
        return
    report = dict(complete=False,baseline_revision=revision,particles=options.particles,
        reverse_source_sha256={label:hashes["essos/split_ad.py"] for label,hashes in source_hashes.items()},
        compared_source_sha256=source_hashes,
        baseline_reverse_source=str(options.baseline_reverse_source) if options.baseline_reverse_source else None,
        baseline_driver_source=str(options.baseline_driver_source) if options.baseline_driver_source else None,
        rows=[],scope="Same CPU physical kernels/inputs; split AD driver and derivative rules are compared",
        rss_scope="Maximum observed worker VmHWM/VmRSS, 10 ms polling; fresh process per model/implementation; includes compilation and execution, not a warm-only peak",
        buffer_scope="XLA executable argument/output/temporary bytes minus aliases; separate from process RSS")
    def save():
        (output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    environment = dict(os.environ,JAX_PLATFORMS="cpu",ESSOS_CPU_DEVICES="1",ESSOS_PROGRESS="0",ESSOS_XLA_CACHE="")
    with tempfile.TemporaryDirectory(prefix="essos-split-reference-") as temporary:
        for name,source in reference_sources.items():
            (Path(temporary)/Path(name).name).write_bytes(source)
        for model in options.models:
            cases = {}
            metadata = {}
            for label in ("before","after"):
                destination = output/f"{label}_{model}.json"
                arguments = ["--output",str(destination),"--models",model,"--modes","adaptive",
                    "--separate-phases","--particles",str(options.particles),"--repeats",str(options.repeats),
                    "--gc-horizon",str(options.gc_horizon),"--lorentz-horizon",str(options.lorentz_horizon)]
                if options.validate_replay:
                    arguments.append("--validate-replay")
                print(f"Running isolated {label}: {model}",flush=True)
                process = subprocess.Popen([sys.executable,"-c",WORKER,
                    temporary if label == "before" else "",*arguments],cwd=ROOT,env=environment)
                peak = 0
                try:
                    while process.poll() is None:
                        peak = max(peak,worker_rss(process.pid))
                        sleep(.01)
                finally:
                    # Cancellation must not leave a benchmark child running.
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill();process.wait()
                if process.returncode:
                    raise RuntimeError(f"{label} {model} worker exited {process.returncode}")
                if not peak:
                    raise RuntimeError("worker RSS could not be measured")
                result = json.loads(destination.read_text())
                if not result["complete"]:
                    raise RuntimeError("worker report is incomplete")
                result["metadata"]["source_sha256"].update(source_hashes[label])
                result["metadata"].update(reverse_baseline_override=(label == "before"),
                    reference_commit=revision if label == "before" else None,process_peak_rss_bytes=peak)
                destination.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
                cases[label] = dict(result["rows"][0],process_peak_rss_bytes=peak)
                metadata[label] = result["metadata"]
            for name,digest in metadata["before"]["source_sha256"].items():
                if name not in reference_sources and metadata["after"]["source_sha256"][name] != digest:
                    raise RuntimeError(f"source changed between workers: {name}")
            if metadata["before"]["input_sha256"] != metadata["after"]["input_sha256"]:
                raise RuntimeError("physical inputs changed between workers")
            import numpy as np
            before,after = cases["before"],cases["after"]
            np.testing.assert_allclose(after["value"],before["value"],rtol=2e-12,atol=1e-25)
            np.testing.assert_allclose(after["gradient"],before["gradient"],rtol=2e-8,atol=1e-11)
            for key in before["stats"]:
                np.testing.assert_allclose(after["stats"][key],before["stats"][key],rtol=2e-12,atol=1e-25)
            row = dict(model=model,before=before,after=after,
                total_speedup=before["warm_median_ms"]/after["warm_median_ms"],
                reverse_speedup=before["phase_medians_ms"]["reverse_ms"]/after["phase_medians_ms"]["reverse_ms"],
                compiler_buffer_ratio=after["compiler_buffer_bytes"]/before["compiler_buffer_bytes"],
                peak_rss_ratio=after["process_peak_rss_bytes"]/before["process_peak_rss_bytes"],
                max_abs_gradient_difference=float(np.max(np.abs(np.asarray(after["gradient"])-before["gradient"]))))
            report["rows"].append(row);save()
            print(json.dumps({k:row[k] for k in ("model","total_speedup","reverse_speedup",
                "compiler_buffer_ratio","peak_rss_ratio","max_abs_gradient_difference")}),flush=True)
    if any((ROOT/name).read_bytes() != source for name,source in current_sources.items()):
        raise RuntimeError("working split sources changed during comparison")
    report["complete"] = True;save()


if __name__ == "__main__":
    main()
