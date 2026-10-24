"""Paired target-GPU check of four Dopri8 register-allocation alternatives.

Uses the production independent-orbit baseline and isolated diagnostic copies
of that kernel. No public solver, field class or default is modified. GPU
timings exclude compilation, warmup, transfers, comparisons and profiling.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from essos.boozer import _native_orbits, boozer_compiler_options
import essos.rk_boozer_gpu_orbit as orbit_module
import essos.rk_boozer_gpu_trial as trial_module
from .benchmark_native_boozer_alpha import PARAMETERS, ROOT, MASS, CHARGE, setup, digest
from .profile_native_boozer_alpha import xplane_resources


class _LazyField(trial_module._KernelField):
    """Diagnostic-only Ref adapter; the public field class is unchanged."""
    def __init__(self, refs, psi0):
        self._radial, self.b_coef, self._flux, self.profile_coef, self._xm, self._xn = refs
        self.psi0 = psi0

    @property
    def r_knots(self):
        return self._radial[...]

    @property
    def s_knots(self):
        return self._flux[...]

    @property
    def xm(self):
        return self._xm[...]

    @property
    def xn(self):
        return self._xn[...]


def _factories():
    """Clone trusted local source for the experiment; never patch production."""
    original = inspect.getsource(orbit_module)
    before = '''                def stage(i, stage_carry):
                    bank, _, _ = stage_carry
                    yi = y+jnp.sum(aref[i, :][:, None]*bank, axis=0)
                    fi = evaluate(yi)
                    return jnp.where(indexbank == i, h*fi[None, :], bank), yi, fi

                bank, endpoint, f1 = jax.lax.fori_loop(1, stages, stage, (bank, y, f0))'''
    after = '''                def stage(i, stage_carry):
                    bank, _ = stage_carry
                    yi = y+jnp.sum(aref[i, :][:, None]*bank, axis=0)
                    fi = evaluate(yi)
                    return jnp.where(indexbank == i, h*fi[None, :], bank), fi

                bank, f1 = jax.lax.fori_loop(1, stages, stage, (bank, f0))
                endpoint = y+jnp.sum(aref[stages-1, :][:, None]*bank, axis=0)'''
    if original.count(before) != 1 or original.count('plt.CompilerParams(num_warps=1)') != 1:
        raise RuntimeError("Kernel source changed; review diagnostic replacements before running")
    factories = dict(baseline=orbit_module.prepare_orbit_solver)
    hashes = dict(baseline=hashlib.sha256(original.encode()).hexdigest())
    for name in ('pipeline1', 'pipeline2', 'lazy_constants', 'lean_carry'):
        source = original
        if name.startswith('pipeline'):
            source = source.replace('plt.CompilerParams(num_warps=1)',
                f'plt.CompilerParams(num_warps=1, num_stages={name[-1]})')
        elif name == 'lean_carry':
            source = source.replace(before, after)
        namespace = dict(__name__='essos._register_diagnostic_'+name, __package__='essos')
        exec(compile(source, orbit_module.__file__, 'exec'), namespace)
        if name == 'lazy_constants':
            namespace['_KernelField'] = _LazyField
        factories[name] = namespace['prepare_orbit_solver']
        hashes[name] = hashlib.sha256((source+(inspect.getsource(_LazyField)
            if name == 'lazy_constants' else '')).encode()).hexdigest()
    return factories, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, default=8192)
    parser.add_argument('--horizon', type=float, default=2e-4)
    parser.add_argument('--saves', type=int, default=101)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--field-cache', type=Path, default=ROOT/
        'examples/benchmarks/results/gpu_boozer_fixed_20261022/coordinate_inputs.npz')
    parser.add_argument('--output', type=Path, default=Path('output/boozer_register_alternatives'))
    options = parser.parse_args(argv)
    if options.particles < 1 or options.saves < 2 or options.repeats < 1 or not (
            np.isfinite(options.horizon) and options.horizon > 0):
        parser.error('Require positive particles/repeats/horizon and saves >= 2')
    if not options.field_cache.is_file():
        parser.error(f'Missing field cache: {options.field_cache}; use an existing boozer_inputs.npz')
    if options.output.exists() and any(options.output.iterdir()):
        parser.error('Choose a fresh output directory')
    compiler_options = boozer_compiler_options()
    if not compiler_options:
        parser.error("This comparison requires the machine's normal JAX backend to be CUDA")
    factories, hashes = _factories()
    options.output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(**(dict(PARAMETERS) | dict(particles=options.particles,
        horizon=options.horizon, saves=options.saves, field_cache=options.field_cache)))
    print(f'{jax.devices()[0].device_kind}; JAX {jax.__version__}; {options.particles} particles; '
          f'{options.horizon:g} s; {options.saves} saves; {options.repeats} warm repeats/case', flush=True)
    field, initial, mu, speed, _, _ = setup(settings)
    times = jnp.linspace(0., settings.horizon, settings.saves)
    jax.block_until_ready((initial, mu, times))
    report = dict(complete=False, device=jax.devices()[0].device_kind, jax=jax.__version__,
        particles=options.particles, horizon_s=options.horizon, saves=options.saves,
        repeats=options.repeats, rtol=settings.rtol, atol=settings.atol, timestep=settings.timestep,
        energy_ev=settings.energy_ev, birth_s=settings.birth_s, seed=settings.seed,
        field_modes=int(field.xm.size), method='dopri8', controller='i', dense_output='standard',
        collisions=False, reverse_ad=False, particle_chunking=False, production_changes=False,
        field_cache=str(options.field_cache), field_cache_sha256=digest(options.field_cache),
        compiler_options=compiler_options, candidate_source_sha256=hashes,
        birth_sha256=hashlib.sha256(np.asarray(initial).tobytes()+np.asarray(mu).tobytes()).hexdigest(),
        source_sha256={name:digest(ROOT/name) for name in ('essos/boozer.py',
            'essos/rk_boozer_gpu_orbit.py', 'essos/rk_boozer_gpu_trial.py',
            'examples/benchmarks/benchmark_boozer_register_final.py')},
        timing_scope='Blocked device-resident calls after compilation and three warmups; rotating start order and alternating forward/reverse order. Transfers, comparisons and profiles excluded.',
        memory_scope='Compiled buffers per executable. Device allocator peaks are cumulative across all five cases, not per-case peak VRAM. No executed-spill or instruction-stall counters.',
        candidates=dict(pipeline1='Compiler pipeline depth 1 instead of default 3; unchanged RK stage count',
            pipeline2='Compiler pipeline depth 2 instead of default 3; unchanged RK stage count',
            lazy_constants='Kernel-private adapter loads immutable constants nearer use',
            lean_carry='Smaller stage carry, followed by a repeated endpoint sum'), rows={})

    def save():
        (options.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

    executables, baseline = {}, None
    for name,prepare in factories.items():
        row = dict(samples_ms=[])
        report['rows'][name] = row
        save()
        try:
            solve = prepare(field,MASS,CHARGE,speed,times,settings.timestep,
                settings.rtol,settings.atol,settings.max_steps)

            def run(y, moment):
                return _native_orbits(field,y,moment,MASS,CHARGE,speed,times,settings.timestep,
                    settings.rtol,settings.atol,'dopri8',True,'i',.4,.3,'standard',
                    settings.max_steps,'unrolled',_orbit_solver=solve)

            print('Compiling '+name, flush=True)
            started = perf_counter()
            executable = jax.jit(run,compiler_options=compiler_options).lower(initial,mu).compile()
            row['compile_s'] = perf_counter()-started
            memory = executable.memory_analysis()
            row['compiled_buffer_components'] = {key:int(getattr(memory,key)) for key in
                ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
            row['compiled_buffers_bytes'] = int(memory.argument_size_in_bytes+memory.output_size_in_bytes+
                memory.temp_size_in_bytes-memory.alias_size_in_bytes)
            for _ in range(3):
                result = jax.block_until_ready(executable(initial,mu))
            arrays = tuple(np.asarray(array) for array in result)
            if name == 'baseline':
                baseline = arrays
            if baseline is None:
                raise RuntimeError('Baseline did not complete; an equivalent comparison is unavailable')
            row.update(bitwise_equal=all(np.array_equal(a,b) for a,b in zip(baseline,arrays)),
                counts=[int(value) for value in arrays[4].sum(axis=0)],
                counts_identical=bool(np.array_equal(baseline[4],arrays[4])),
                lost=int(np.sum(arrays[1]>=0)), failed=int(np.sum(arrays[2]>=0)),
                finite=bool(np.isfinite(arrays[0]).all() and np.isfinite(arrays[3]).all()),
                status_valid=bool(np.isin(arrays[5],[0,1]).all()),
                max_relative_energy_error=float(arrays[3].max()) if np.isfinite(arrays[3]).all() else None,
                max_saved_component_difference=np.max(np.abs(arrays[0]-baseline[0]),axis=(0,1)).tolist()
                    if np.isfinite(arrays[0]).all() and np.isfinite(baseline[0]).all() else None)
            if not (row['finite'] and row['status_valid'] and not row['failed']):
                raise RuntimeError('Incomplete/nonfinite solve; warm timings skipped for this case')
            executables[name] = executable
            print(f"{name}: bitwise equality {row['bitwise_equal']}; counts {row['counts']}",flush=True)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            print(f"{name}: {row['error']}",flush=True)
        save()
        if name == 'baseline' and name not in executables:
            raise RuntimeError('Baseline failed; inspect comparison.json')

    names = list(executables)
    for repeat in range(options.repeats):
        offset = repeat % len(names)
        order = names[offset:]+names[:offset]
        for name in order if repeat % 2 == 0 else reversed(order):
            started = perf_counter()
            jax.block_until_ready(executables[name](initial,mu))
            report['rows'][name]['samples_ms'].append(1000*(perf_counter()-started))
    baseline_times = np.asarray(report['rows']['baseline']['samples_ms'])
    for name in names:
        row = report['rows'][name]
        row['warm_median_ms'] = float(np.median(row['samples_ms']))
        row['warm_q25_ms'],row['warm_q75_ms'] = map(float,np.quantile(row['samples_ms'],[.25,.75]))
        row['paired_median_ratio'] = float(np.median(np.asarray(row['samples_ms'])/baseline_times))
        row['time_decrease_percent'] = 100*(1-row['warm_median_ms']/float(np.median(baseline_times)))
        save()
    # Capture resources only after every unprofiled timing has finished.
    for name in names:
        row = report['rows'][name]
        try:
            profiler_options = jax.profiler.ProfileOptions()
            profiler_options.python_tracer_level = 0
            with jax.profiler.trace(str(options.output/name),profiler_options=profiler_options):
                jax.block_until_ready(executables[name](initial,mu))
            path, = (options.output/name).rglob('*.xplane.pb')
            row['native_profile'] = xplane_resources(path)
            row['orbit_launches'] = [item for item in row['native_profile']['launches']
                                    if item['name'].startswith('kernel')]
        except Exception as error:
            row['resource_profile_error'] = f'{type(error).__name__}: {error}'
        save()
    report['complete'] = all('error' not in row and 'warm_median_ms' in row
                             for row in report['rows'].values())
    report['all_outputs_bitwise_equal'] = all(row.get('bitwise_equal',False) for row in report['rows'].values())
    report['device_allocator_stats_cumulative'] = jax.devices()[0].memory_stats()
    save()
    print('\nCase                 Warm ms   Compile s   Buffers MiB   Regs/thread   Equal',flush=True)
    for name,row in report['rows'].items():
        if 'warm_median_ms' not in row:
            print(f'{name:20} FAILED: {row.get("error","unknown error")}',flush=True)
            continue
        registers = sorted({item['resources']['registers_per_thread'] for item in row.get('orbit_launches',[])
                            if 'registers_per_thread' in item['resources']})
        regs = ','.join(map(str,registers)) or 'unavailable'
        print(f"{name:20} {row['warm_median_ms']:8.3f} {row['compile_s']:11.3f} "
              f"{row['compiled_buffers_bytes']/2**20:13.3f} {regs:>13} {str(row['bitwise_equal']):>7}",flush=True)
    print('Saved '+str((options.output/'comparison.json').resolve()),flush=True)
    if not report['complete']:
        raise RuntimeError('One or more cases failed; inspect the saved comparison')
    if not report['all_outputs_bitwise_equal']:
        raise RuntimeError('Output equivalence failed; do not promote a timing result without an accuracy audit')
    return report


if __name__ == '__main__':
    main()
