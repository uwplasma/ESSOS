"""Replay saved collisionless AD cases in fresh Linux CPU processes for RSS.

Each child constructs the shared graph or one scalar objective and checks its
values, counts and gradients against the timing report. Whole-process RSS
includes setup and compilation; optional reset counters separately measure
first and warm execution peaks. Compiler buffers describe the selected
reverse or whole value_and_grad executable.
"""
import argparse
import hashlib
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

from .benchmark_native_ad import build_case, compiler_memory, json_safe, ROOT
from .validate_coil_orbit_loss import reference_wall
import jax
import jax.numpy as jnp
import numpy as np


def label(row):
    return '_'.join(row[k] for k in ('model','method','backend'))


def measure(report, key, objective_index=None, runtime_peak=False):
    import resource
    if sys.platform != 'linux' or jax.default_backend() != 'cpu':
        raise ValueError('isolated RSS measurement requires Linux CPU')
    previous = next(row for row in report['rows'] if label(row) == key)
    options = argparse.Namespace(**report['metadata']['settings'])
    # Radius objectives do not use a classifier or a hard-wall event. Avoid
    # constructing an unrelated distance grid when profiling that scalar cost.
    boundary = None if objective_index in (0,1) else reference_wall(angular_interpolation='periodic_cubic').classifier
    theta,cost,_,_,_,_ = build_case(options,previous['model'],previous['method'],previous['backend'],boundary,
        objective_index=objective_index)
    peak = lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    setup_peak = peak()
    primal = jax.jit(cost)
    primal.lower(theta).compile()
    values,pullback,aux = jax.block_until_ready(jax.vjp(primal,theta,has_aux=True))
    forward_peak = peak()
    ys,counts,status = map(np.asarray,aux)
    if np.any(status) or not np.isfinite(ys).all():
        raise RuntimeError('incomplete/nonfinite AD memory replay')
    np.testing.assert_array_equal(counts,previous['counts_per_particle'])
    expected_values = previous['values'] if objective_index is None else previous['values'][objective_index]
    np.testing.assert_allclose(values,expected_values,rtol=2e-12,atol=1e-25)
    reverse = jax.jit(lambda pullback,seed: pullback(seed)[0])
    seeds = tuple(jnp.eye(len(values),dtype=theta.dtype) if objective_index is None else jnp.ones((1,),theta.dtype))
    executable = reverse.lower(pullback,seeds[0]).compile()
    compile_peak = peak()
    runtime = {}
    if runtime_peak:
        runtime['after_compile_resident_rss_bytes'] = _reset_resident_peak()
    gradients = np.stack([np.asarray(jax.block_until_ready(reverse(pullback,seed))) for seed in seeds])
    if runtime_peak:
        runtime['first_execution_peak_rss_bytes'] = _resident_memory()['VmHWM']
    expected_gradients = previous['gradients'] if objective_index is None else [previous['gradients'][objective_index]]
    difference = np.linalg.norm(gradients-np.asarray(expected_gradients),axis=1)
    relative = difference/np.maximum(np.linalg.norm(expected_gradients,axis=1),1e-30)
    if not np.isfinite(relative).all() or np.max(relative)>2e-9:
        raise RuntimeError(f'memory replay gradient mismatch: {relative}')
    if runtime_peak:
        runtime['warm_baseline_rss_bytes'] = _reset_resident_peak()
        for seed in seeds:
            jax.block_until_ready(reverse(pullback,seed))
        runtime['warm_peak_rss_bytes'] = _resident_memory()['VmHWM']
        runtime['warm_additional_peak_rss_bytes'] = max(0,
            runtime['warm_peak_rss_bytes']-runtime['warm_baseline_rss_bytes'])
    memory = compiler_memory(executable)
    buffers = memory['argument_size_in_bytes']+memory['output_size_in_bytes']+memory['temp_size_in_bytes']-memory['alias_size_in_bytes']
    return dict(case=key,objective='joint_seven' if objective_index is None else report['metadata']['objective_names'][objective_index],
        complete=True,counts_match=True,gradients_match=True,gradient_relative_l2=relative.tolist(),
        compiler_memory=memory,compiler_buffer_bytes=buffers,
        setup_peak_rss_bytes=setup_peak,after_linearization_peak_rss_bytes=forward_peak,
        after_reverse_compile_peak_rss_bytes=compile_peak,
        peak_rss_bytes=max(setup_peak,forward_peak,compile_peak,peak(),
            runtime.get('first_execution_peak_rss_bytes',0),runtime.get('warm_peak_rss_bytes',0)),**runtime)


def _resident_memory():
    """Read only this process's current and peak resident RAM on Linux."""
    fields = {}
    with Path('/proc/self/status').open() as stream:
        for line in stream:
            name,_,value = line.partition(':')
            if name in ('VmRSS','VmHWM'):
                fields[name] = int(value.split()[0])*1024
    return fields


def _reset_resident_peak():
    # Linux clear_refs=5 resets RSS accounting, without changing allocations.
    # https://www.kernel.org/doc/html/latest/filesystems/proc.html
    Path('/proc/self/clear_refs').write_text('5\n')
    return _resident_memory()['VmRSS']


def measure_value_and_grad(report, key, objective_index, runtime_peak=False):
    """Measure one optimizer-style executable, rather than a reusable pullback."""
    import resource
    if sys.platform != 'linux' or jax.default_backend() != 'cpu':
        raise ValueError('isolated RSS measurement requires Linux CPU')
    previous = next(row for row in report['rows'] if label(row) == key)
    options = argparse.Namespace(**report['metadata']['settings'])
    boundary = None if objective_index in (0,1) else reference_wall(angular_interpolation='periodic_cubic').classifier
    theta,cost,_,_,_,_ = build_case(options,previous['model'],previous['method'],previous['backend'],boundary,
        objective_index=objective_index)
    peak = lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    setup_peak = peak()
    def objective(parameters):
        value,(_,counts,status) = cost(parameters)
        # Keep diagnostics, but do not retain the saved trajectory as an output.
        return value,(counts,status)
    evaluate = jax.jit(jax.value_and_grad(objective,has_aux=True))
    executable = evaluate.lower(theta).compile()
    compile_peak = peak()
    runtime = {}
    if runtime_peak:
        runtime['after_compile_resident_rss_bytes'] = _reset_resident_peak()
    (value,(counts,status)),gradient = jax.block_until_ready(evaluate(theta))
    if runtime_peak:
        runtime['first_execution_peak_rss_bytes'] = _resident_memory()['VmHWM']
    if np.any(np.asarray(status)) or not np.isfinite(gradient).all():
        raise RuntimeError('incomplete/nonfinite value_and_grad memory replay')
    np.testing.assert_array_equal(counts,previous['counts_per_particle'])
    np.testing.assert_allclose(value,previous['values'][objective_index],rtol=2e-12,atol=1e-25)
    expected = np.asarray(previous['gradients'][objective_index])
    relative = float(np.linalg.norm(np.asarray(gradient)-expected)/max(np.linalg.norm(expected),1e-30))
    if not np.isfinite(relative) or relative>2e-9:
        raise RuntimeError(f'value_and_grad memory replay gradient mismatch: {relative}')
    if runtime_peak:
        runtime['warm_baseline_rss_bytes'] = _reset_resident_peak()
    timings=[]
    for _ in range(3):
        start=perf_counter()
        jax.block_until_ready(evaluate(theta))
        timings.append((perf_counter()-start)*1000)
    if runtime_peak:
        runtime['warm_peak_rss_bytes'] = _resident_memory()['VmHWM']
        runtime['warm_additional_peak_rss_bytes'] = max(0,
            runtime['warm_peak_rss_bytes']-runtime['warm_baseline_rss_bytes'])
    memory = compiler_memory(executable)
    buffers = memory['argument_size_in_bytes']+memory['output_size_in_bytes']+memory['temp_size_in_bytes']-memory['alias_size_in_bytes']
    return dict(case=key,objective=report['metadata']['objective_names'][objective_index],
        complete=True,counts_match=True,gradients_match=True,gradient_relative_l2=[relative],
        compiler_memory=memory,compiler_buffer_bytes=buffers,
        setup_peak_rss_bytes=setup_peak,after_compile_peak_rss_bytes=compile_peak,
        peak_rss_bytes=max(setup_peak,compile_peak,peak(),runtime.get('first_execution_peak_rss_bytes',0),
            runtime.get('warm_peak_rss_bytes',0)),
        warm_samples_ms=timings,warm_median_ms=float(np.median(timings)),**runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',help='Optional model_method_backend labels')
    parser.add_argument('--objective-index',type=int,choices=range(7),
        help='Compile one scalar objective separately; default replays the shared seven-objective graph')
    parser.add_argument('--mode',choices=['split_vjp','value_and_grad'],default='split_vjp',
        help='Reusable pullback, or one jitted scalar value_and_grad including the forward solve')
    parser.add_argument('--runtime-peak',action='store_true',
        help='Reset this Linux process RSS high-water mark after compilation and after first execution')
    parser.add_argument('--process-repeats',type=int,default=1,
        help='Independent fresh Linux processes per case (default 1)')
    parser.add_argument('--case',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode=='value_and_grad' and args.objective_index is None:
        parser.error('value_and_grad requires --objective-index')
    if args.process_repeats<1:
        parser.error('--process-repeats must be positive')
    if args.output.exists():
        parser.error('use a fresh output path')
    original = json.loads(args.timings.read_text())
    if not original.get('complete') or not original.get('all_cases_passed') or original['metadata']['backend'] != 'cpu':
        parser.error('a complete passing CPU AD timing report is required')
    for kind in ('source_sha256','input_sha256'):
        for name,expected in original['metadata'][kind].items():
            actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'timed source/input changed: {name}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.case:
        result = (measure(original,args.case,args.objective_index,args.runtime_peak) if args.mode=='split_vjp' else
            measure_value_and_grad(original,args.case,args.objective_index,args.runtime_peak))
        args.output.write_text(json.dumps(json_safe(result),indent=2)+'\n')
        return
    available = [label(row) for row in original['rows']]
    cases = args.cases or available
    if not cases or any(key not in available for key in cases):
        parser.error(f'cases must be among {available}')
    result = dict(complete=False,metadata=dict(
        timing_report=str(args.timings),timing_sha256=hashlib.sha256(args.timings.read_bytes()).hexdigest(),
        measurement_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mode=args.mode,
        process_repeats=args.process_repeats,
        runtime_peak_scope=('VmHWM reset via /proc/self/clear_refs=5 after compile and after first execution; '
            + ('three blocked warm value_and_grad calls' if args.mode=='value_and_grad' else
                'one blocked warm reverse sweep over the objective seeds')
            + '; includes resident JAX/executable/allocator memory; '
            'extra RSS growth is not solver working-set size' if args.runtime_peak else None),
        scope=('Fresh Linux CPU process per case; gradients verified against saved report; no finite differences; '
            + ('three warm value_and_grad repeats' if args.mode=='value_and_grad' else 'no timing repeats')),
        objective='joint_seven' if args.objective_index is None else original['metadata']['objective_names'][args.objective_index],
        compiler_scope=(('Whole value_and_grad' if args.mode=='value_and_grad' else 'Reverse')
            +' executable arguments + outputs + temporaries - aliases; excludes runtime/compiler/host allocations'),
        rss_scope='Whole-process Linux ru_maxrss including imports, case setup, compilation and execution; not runtime-only integration memory'),rows=[])
    with tempfile.TemporaryDirectory(prefix='essos-ad-memory-') as directory:
        for replicate in range(1,args.process_repeats+1):
            for key in cases:
                print(f'Measuring isolated AD memory: {key}, process {replicate}/{args.process_repeats}',flush=True)
                output = Path(directory)/(key+f'_{replicate}.json')
                command = [sys.executable,'-m','examples.benchmarks.measure_native_ad_memory',
                    '--timings',str(args.timings.resolve()),'--output',str(output),'--case',key,'--mode',args.mode]
                if args.objective_index is not None:
                    command.extend(['--objective-index',str(args.objective_index)])
                if args.runtime_peak:
                    command.append('--runtime-peak')
                child = subprocess.run(command,capture_output=True,text=True)
                if child.returncode:
                    raise RuntimeError(f'memory replay failed for {key}:\n{child.stdout}\n{child.stderr}')
                row = json.loads(output.read_text())
                row['replicate'] = replicate
                result['rows'].append(row)
                args.output.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(row),flush=True)
    result['complete'] = True
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
