Adaptive Boozer Dopri8 GPU loop execution, 2026-10-23
==================================================

Both paths use the same production native solver with retained crossing stages
and same-loop root retries. Only the outermost JAX compilation options differ:

```python
{
    "xla_gpu_enable_command_buffer": "FUSION,WHILE,CONDITIONAL",
    "xla_gpu_graph_min_graph_size": 1,
}
```

The numerical RK stages, physical RHS, field objects, precision, error norm,
controller and save grid are identical. CUDA command buffers change execution
of the existing JAX control-flow graph. They reduce host launch/control overhead;
this is not a faster RK formula or a field approximation. Other JAX solvers
can benefit from the same compiler policy, which must be held equal in future
cross-library comparisons.

Controlled workload: **64 particles**, QA Boozer, 16 direct Fourier modes,
3.52 MeV alpha GCs at `s=0.3`, seed 42, 0.2 ms, 101 saves, adaptive Dopri8,
I control, `rtol=atol=1e-10`, standard interpolation, explicit stages. No particle
chunks or compaction. GPU: NVIDIA GeForce RTX 5060 Laptop GPU; JAX/JAXlib 0.11.2.

| Metric | Compiler defaults | CUDA loop command buffers |
| --- | ---: | ---: |
| Warm median, ms | 358.670 | 193.032 |
| Fresh-process first execution, ms | 394.183 | 222.813 |
| Fresh-process lowering and compilation, s | 6.169 | 6.314 |
| Compiled buffers, MiB | 1.466 | 1.466 |
| JAX allocator peak, MiB | 208.338 | 208.338 |
| Peak host RSS, MiB | 1031.945 | 1031.754 |
| Total accepted / rejected steps | 28,084 / 4,685 | 28,084 / 4,685 |
| Lost / failed particles | 1 / 0 | 1 / 0 |
| Maximum relative energy error | 4.78934e-10 | 4.78934e-10 |

Warm execution decreased **46.18%**. Saved flux-chart coordinates, parallel
velocities and speeds were identical; per-particle step counts and solver
statuses were identical. Timings are nine interleaved blocked warm samples per
case. Compilation values above come from independent fresh workers, avoiding
cross-case compiler reuse. The raw paired compilation times are also retained
in `both.json` and the CSV. Fresh workers measured the memory peaks, with three
warm samples each. GPU peaks are JAX allocator peaks including setup, compilation
and tracing; they exclude CUDA context/driver and graph-management allocations.
They are not whole-process VRAM measurements.

`both.json`, `default.json`, `device_loops.json` and `comparison.csv` contain the
raw timing/memory records and source/birth hashes. `entrypoint64.json` checks the
exact large-run benchmark entrypoint, still with 64 particles: 6.258 s compile,
0.220 s first execution, 0.197 s warm median, one loss and no failures. It
confirms the compiler options are enabled by the existing Dopri8 command.
This does not establish the gain for the remote 8192-particle, 20 ms run.

Reproduce from the repository root in the normal GPU JAX environment:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison execution --particles 64 --horizon 2e-4 --saves 101 --repeats 9 \
  --output output/boozer_gpu_execution

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison execution --case default --repeats 3 --output output/boozer_gpu_execution

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison execution --case device_loops --repeats 3 --output output/boozer_gpu_execution
```

The large-run example applies this policy automatically to CUDA adaptive
Dopri8 with explicit stages. Public single-device `trace_boozer` calls use the
same policy for that case. The default backend is respected; CPU and non-CUDA
backends retain their compiler defaults. When embedding the composable private
kernel in an outer JAX jit, apply `essos.boozer.boozer_compiler_options()` to
that outermost compilation. Per-jit compiler options are available in
[JAX 0.7.2](https://github.com/jax-ml/jax/blob/jax-v0.7.2/jax/_src/api.py#L139);
the remote machine's execution benefit remains to be measured.

Verification: 96 solver checks and three CUDA-specific checks passed. The latter
compare mixed initial events, survivors, recoverable rejected roots, I/PI history
and the production Boozer CUDA entrypoint against ordinary compilation. For
these CUDA tests, all returned arrays agree bitwise.

```sh
python -m pytest tests/test_rk_events.py tests/test_rk.py \
  tests/test_rk_ad.py::test_unrolled_execution_checkpointed_reverse_matches_its_fixed_map_jvp -q
```
