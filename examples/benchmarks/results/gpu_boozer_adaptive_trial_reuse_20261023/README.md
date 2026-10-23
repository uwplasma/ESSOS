Adaptive Boozer Dopri8 with retained crossing stages, 2026-10-23
==============================================================

This table isolates event handling with ordinary compiler defaults. The
[subsequent CUDA loop-execution comparison](../gpu_boozer_device_loops_20261023/README.md)
measures the additional compiler policy now used by the production GPU example.

This replaces the earlier event-replay implementation, which compiled both a
terminal RHS replay and an entire fallback solver. The driver now retains the
crossing trial's stages, refines the root outside the ordinary trial loop, and
resumes that same loop if the root is invalid. Prior saves, FSAL derivative,
accepted-time history and PI history are preserved across a rejection.

Controlled local GPU workload: 64 collisionless 3.52 MeV alpha GCs, QA Boozer
field with 16 direct Fourier modes, birth `s=0.3`, seed 42, 0.2 ms horizon,
101 saves, `rtol=atol=1e-10`, I control and standard dense output. No particle
chunks. NVIDIA GeForce RTX 5060 Laptop GPU; JAX/JAXlib 0.11.2.

| Metric | Original inline roots | Retained stages, same-loop retry |
| --- | ---: | ---: |
| Warm median, ms | 433.054 | 371.083 |
| Lowering and compilation, s | 5.600 | 6.016 |
| Compiled buffers, MiB | 1.464 | 1.466 |
| JAX allocator peak, MiB | 208.338 | 208.338 |
| Peak host RSS, MiB | 1003.996 | 1034.648 |
| Total accepted / rejected steps | 28,084 / 4,685 | 28,084 / 4,685 |
| Lost / failed particles | 1 / 0 | 1 / 0 |
| Maximum relative energy error | 4.78934e-10 | 4.78934e-10 |

Both paths produced identical saved flux-chart coordinates, parallel velocities
and speeds, per-particle step counts and solver statuses. Warm execution was
14.31% faster than inline roots in this interleaved run. The mathematical RK
stages, field, tolerances and controller were unchanged. The improvement is
adaptive event handling, not a different RK formula or field approximation.

The [superseded replay record](../gpu_boozer_adaptive_20261023/README.md) compiled
in about 13 seconds locally. The new implementation removes that doubled
compilation cost; warm runtime remains around 0.37 seconds. The local 14.31%
comparison against inline roots is not an additional 14.31% gain over the
previous replay implementation. The new implementation has not yet been
measured on the remote 8192-particle, 20 ms workload.

`both.json` has nine blocked interleaved warm samples per path. `inline.json`
and `replay.json` are independent fresh-process peak measurements, each with
three warm repetitions. GPU peaks are JAX allocator peaks including setup,
compilation and tracing, excluding CUDA context/driver allocations. The setup
peak dominates this small workload. Host peaks include imports and compilation.
`comparison.csv` combines the paired timings with the independent peaks.
The `replay` CLI label is retained for compatibility; it now uses stored stages
and does not replay the RHS. JSON reports record core source and birth hashes.

Reproduce from the repository root in the normal GPU JAX environment:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --particles 64 --horizon 2e-4 --saves 101 --repeats 9 \
  --output output/boozer_adaptive_trial_reuse

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --case inline --repeats 3 --output output/boozer_adaptive_trial_reuse

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --case replay --repeats 3 --output output/boozer_adaptive_trial_reuse
```

The field cache is the bundled QA transform used for the preceding fixed-field
benchmark. The original field class, spline representation and Fourier
spectrum are used in both paths. The accelerated default remains limited to
GPU Boozer adaptive Dopri8 with explicit stages and standard interpolation.
The accepted-step AD recorder and collision solver paths retain their defaults.

Verification of this driver: 96 GPU checks passed (10 event/retry tests and 86
existing forward and checkpointed reverse checks).
