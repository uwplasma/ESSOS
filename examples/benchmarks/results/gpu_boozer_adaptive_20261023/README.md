64-particle adaptive Boozer Dopri8, GPU, 2026-10-23
================================================

Historical event-replay implementation. It has been superseded by
[retained crossing stages and retry in one adaptive driver](../gpu_boozer_adaptive_trial_reuse_20261023/README.md),
which removes the additional RHS replay and separately compiled fallback.
The measurements and source hashes below describe the earlier implementation.

The complete production forward kernel is timed, including saved-state energy
diagnostics. This is a local development workload: 64 collisionless 3.52 MeV
alpha guiding centres, QA Boozer field with 16 direct Fourier modes, births at
`s=0.3`, seed 42, a 0.2 ms horizon, 101 saves, `rtol=atol=1e-10`, I control,
standard interpolation and explicit stages. No particle chunks or compaction.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; JAX/JAXlib 0.11.2.

| Metric | Inline roots | Deferred terminal-root replay |
| --- | ---: | ---: |
| Warm median, ms | 418.127 | 369.639 |
| Lowering and compilation, s | 5.691 | 12.973 |
| Compiled buffers, MiB | 1.464 | 1.494 |
| JAX allocator peak, MiB | 208.338 | 208.338 |
| Peak host RSS, MiB | 1004.230 | 1251.676 |
| Total accepted steps | 28,084 | 28,084 |
| Total rejected steps | 4,685 | 4,685 |
| Lost / failed particles | 1 / 0 | 1 / 0 |
| Maximum relative energy error | 4.78934e-10 | 4.78934e-10 |

Warm execution decreased **11.60%**. Both paths produced identical saved
`s, theta, zeta, v_parallel, speed` values in this run. Per-particle step counts
and solver statuses were identical. Both solve the same RK formula and RHS;
the change removes root-search bookkeeping from ordinary adaptive trials.
The final crossing step is replayed once to find the same loss root, using
the same dense interpolant. Invalid interior roots use the original inline
algorithm, preserving rejection and retry semantics. The existing accepted-step
checkpoint and collision paths retain their defaults.

Compilation roughly doubled and peak host memory increased about 25%. Compiled
buffers increased 2.02%. GPU peaks above are JAX allocator peaks measured in
separate fresh processes, including setup, compilation and tracing. They exclude
CUDA context/driver allocations and are not whole-process VRAM measurements.
The common setup peak dominates this small workload. These results do not
establish the memory or timing change for the remote 8192-particle, 20 ms case.

`both.json` contains nine blocked interleaved warm samples per case and the
saved-state comparison. `inline.json` and `replay.json` contain independent
process memory measurements, with three warm samples each; their separate
timings are not used for the controlled timing comparison. `comparison.csv`
combines the paired timings and independent peaks. Each JSON records source
hashes and a birth-array hash. Failed scratch variants are not included.

From the repository root, using the normal JAX GPU environment:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --particles 64 --horizon 2e-4 --saves 101 --repeats 9 \
  --output output/boozer_adaptive_events

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --case inline --particles 64 --horizon 2e-4 --saves 101 --repeats 3 \
  --output output/boozer_adaptive_events

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --case replay --particles 64 --horizon 2e-4 --saves 101 --repeats 3 \
  --output output/boozer_adaptive_events
```

The field cache defaults to the bundled QA transform from the fixed-field
benchmark. The original field class, spline representation, Fourier spectrum,
controller, tolerances and births are used in both cases. `replay` selects the
production default on GPU; CPU defaults retain inline searches.

Validation includes rejected steps, mixed boolean and floating-point events,
initial events, earliest-event ordering, simultaneous ties, duplicate saves,
empty save grids, accepted-time histories, invalid interior roots and recorded
schedule metadata. Both methods and I/PI controllers are checked by
`tests/test_rk_events.py`.

GPU verification: **94 tests passed** (8 focused event tests and 86 existing
forward/checkpointed reverse checks):

```sh
python -m pytest tests/test_rk_events.py tests/test_rk.py \
  tests/test_rk_ad.py::test_unrolled_execution_checkpointed_reverse_matches_its_fixed_map_jvp -q
```
