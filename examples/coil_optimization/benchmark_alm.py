"""Compare isolated ALM solves on Linux/WSL, optionally against a saved module.

Example: python -m examples.coil_optimization.benchmark_alm --platforms cpu cuda --backends scipy jaxopt
Add --baseline-module /path/to/previous/augmented_lagrangian.py for a comparison.
The baseline's dependencies must be available only when that option is used.
"""
import argparse
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def worker(args):
    import resource
    import statistics
    import jax
    import jax.numpy as jnp
    import scipy
    import essos  # establish ESSOS_CPU_DEVICES before backend initialization

    jax.config.update('jax_enable_x64', True)
    if args.module:
        spec = importlib.util.spec_from_file_location('alm_baseline', args.module)
        alm = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = alm
        spec.loader.exec_module(alm)
    else:
        from essos import augmented_lagrangian as alm
    device = jax.devices()[0]
    started = perf_counter()
    if args.case == 'quadratic':
        x = jnp.full(args.size, .25)
        target = jnp.linspace(.5, 1.5, args.size)
        diagonal = jnp.linspace(1., 10., args.size)
        objective = lambda y: 1 + .5 * jnp.sum(diagonal * (y - target)**2)
        constraints = alm.combine(alm.eq(lambda y: y[:2] - .75, omega=1e-6),
                                  alm.ineq(lambda y: 1 - y, omega=1e-6))
    else:
        from examples.coil_optimization import optimize_coils_surface_augmented_lagrangian as example
        example.alm = alm
        field, surface, iota = example.initial_configuration(args.case, smoke=True)
        x, constraints, objective, _, _ = example.build_problem(field, surface, args.case,
            ntrace=9, nalpha=2, npitch=2, iota_periods=1, iota_target=iota)
    params = x, constraints.init(x)
    if args.case != 'quadratic':
        params = x, alm.initialize_constraint_weights(constraints, x, params[1], loss=objective,
            use_perturbation=True, seed=42)
    factory = alm.ALM_model_jaxopt_lbfgsb if args.module or args.backend == 'jaxopt' else alm.ALM_model_scipy
    inner_results = []
    diagnostics = {'inner_callback': inner_results.append} if not args.module and args.backend == 'scipy' else {}
    if args.scipy_pinned_host and not args.module and args.backend == 'scipy':
        diagnostics['scipy_pinned_host'] = True
    solver = factory(constraints, loss=objective, model_mu='Mu_Adaptative_2',
                     inner_maxiter=40 if args.case == 'quadratic' else 2,
                     jit_update=args.jit_update, **diagnostics)
    state, grad, info = solver.init(params)
    jax.block_until_ready((params, grad, info))
    setup_seconds = perf_counter() - started
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def timed_update():
        before = perf_counter()
        result = solver.update(params, state, grad, info)
        jax.block_until_ready(result)
        return perf_counter() - before, result

    first_seconds, result = timed_update()
    warm = [timed_update()[0] for _ in range(args.repeats)]
    memory = device.memory_stats() or {}
    report = dict(backend='baseline' if args.module else args.backend, platform=args.platform,
        case=args.case, dofs=int(x.size), dtype=str(x.dtype), device=device.device_kind,
        jax=jax.__version__, scipy=scipy.__version__,
        jaxopt=version('jaxopt') if args.module or args.backend == 'jaxopt' else None,
        python=sys.version.split()[0], setup_s=setup_seconds,
        first_update_s=first_seconds, warm_update_s=warm,
        warm_median_s=statistics.median(warm), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        rss_before_solve_mib=before_rss, peak_device_mib=memory.get('peak_bytes_in_use', 0) / 2**20,
        cost=float(result[3][0]), infeasibility=float(alm.norm_constraints(result[3][2])),
        inner_maxiter=40 if args.case == 'quadratic' else 2, repeats=args.repeats,
        inner_iterations=int(inner_results[-1].nit) if inner_results else None,
        function_evaluations=int(inner_results[-1].nfev) if inner_results else None,
        jit_update=args.jit_update, persistent_compilation_cache=False)
    report['scipy_pinned_host'] = bool(diagnostics.get('scipy_pinned_host', False))
    if not all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(result)):
        raise FloatingPointError('Nonfinite benchmark iterate')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platforms', nargs='+', default=['cpu'], choices=['cpu', 'cuda'])
    parser.add_argument('--backends', nargs='+', default=['scipy'], choices=['scipy', 'jaxopt'])
    parser.add_argument('--cases', nargs='+', default=['quadratic', 'qs', 'qi'], choices=['quadratic', 'qs', 'qi'])
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 512])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--scipy-pinned-host', action='store_true',
                        help='use pinned host output buffers for SciPy GPU workers')
    parser.add_argument('--jit-update', action=argparse.BooleanOptionalAction, default=True,
                        help='compile the dual update (SciPy) or full update (JAXopt); default on')
    parser.add_argument('--baseline-module', type=Path)
    parser.add_argument('--output', type=Path, default=Path('alm_benchmark.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--module', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--case', help=argparse.SUPPRESS)
    parser.add_argument('--size', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--platform', help=argparse.SUPPRESS)
    parser.add_argument('--backend', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repeats <= 0 or any(n < 2 for n in args.sizes):
        parser.error('repeats must be positive and sizes at least two')
    if args.scipy_pinned_host and 'scipy' in args.backends and 'cpu' in args.platforms:
        parser.error('--scipy-pinned-host requires GPU workers; use --platforms cuda')
    results = []
    for platform in args.platforms:
        for case in args.cases:
            for size in args.sizes if case == 'quadratic' else [0]:
                comparisons = ([(args.baseline_module, 'baseline')] if args.baseline_module else [])
                comparisons += [(None, backend) for backend in args.backends]
                for module, backend in comparisons:
                    command = [sys.executable, '-m', 'examples.coil_optimization.benchmark_alm', '--worker', '--case', case,
                               '--size', str(size), '--platform', platform, '--repeats', str(args.repeats),
                               '--backend', backend]
                    if module:
                        command.extend(['--module', str(module.resolve())])
                    command.append('--jit-update' if args.jit_update else '--no-jit-update')
                    if args.scipy_pinned_host:
                        command.append('--scipy-pinned-host')
                    env = {**os.environ, 'JAX_PLATFORMS': platform, 'ESSOS_CPU_DEVICES': '1',
                           'JAX_ENABLE_COMPILATION_CACHE': 'false', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}
                    completed = subprocess.run(command, capture_output=True, text=True, env=env)
                    if completed.returncode:
                        raise RuntimeError(completed.stderr or completed.stdout)
                    report = json.loads(completed.stdout.strip().splitlines()[-1])
                    results.append(report)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(results, indent=2) + '\n')
                    print(f"{platform} {case} n={report['dofs']} {report['backend']}: "
                          f"first {report['first_update_s']:.3f}s, warm {report['warm_median_s']:.4f}s, "
                          f"peak RSS {report['peak_rss_mib']:.1f} MiB", flush=True)


if __name__ == '__main__':
    main()
