Adaptive Boozer Dopri8: grouped stage reductions, 2026-10-23
=========================================================

The production solver groups independent small floating-point sums in the
prepared RHS before expanding the RK stages. This exposes the joint sums to
the GPU compiler at every stage. The field class, physical RHS, tableau,
controller, tolerances, dense output and event handling are unchanged.
Custom derivative rules remain opaque. Parameter, state and time derivatives
are checked against the original map. This preparation is enabled only for
the GPU adaptive Boozer Dopri8 forward path with unrolled stages; the coil,
split-collision and accepted-step reverse paths retain their existing setup.

This comparison uses **64 particles** on an RTX 5060 Laptop GPU, JAX/JAXlib
0.11.2: QA Boozer with 16 direct Fourier modes, 3.52 MeV alpha GCs, s=0.3,
seed 42, 0.2 ms, 101 saves, I control, rtol=atol=1e-10. No particle chunks.
**CUDA loop command buffers remain enabled in both cases.**

| Metric | Separate reductions | Grouped reductions |
| --- | ---: | ---: |
| Warm median, ms | 194.781 | 171.926 |
| First execution, ms | 225.722 | 200.152 |
| Paired lowering/compilation, s | 6.399 | 6.212 |
| Compiled buffers, bytes | 1,537,504 | 1,538,528 |
| Accepted / rejected steps | 28,084 / 4,685 | 28,084 / 4,685 |
| Lost / failed particles | 1 / 0 | 1 / 0 |
| Maximum relative energy drift | 4.78934e-10 | 4.78934e-10 |

Warm execution decreased **11.73%** in nine interleaved blocked samples.
Saved states and velocities had zero differences, and per-particle step counts
and statuses matched. The compiled buffer increase is 1 KiB. Compilation is
a paired measurement, not independent cold-process compilation. The GPU and
host peaks in `both.json` are cumulative across both executables, not a claim
of separate per-case peaks or whole-process VRAM use.

This is a local result. The 8192-particle remote benefit remains unmeasured.
The preceding command-buffer-only change did not speed up that remote run;
see the [remote measurements](../gpu_boozer_device_loops_20261023/README.md).
Do not extrapolate this local percentage to the remote machine.

Reproduce the production comparison:

```sh
python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison reductions --particles 64 --horizon 2e-4 --saves 101 \
  --repeats 9 --output output/boozer_stage_reductions_64
```

The large-run entrypoint enables the new preparation automatically and prints
`RHS execution: grouped independent stage reductions` alongside the unchanged
GPU execution options. Its run settings and output source hashes record the
actual implementation. `both.json` contains the local raw samples and hashes.

Eight targeted checks passed: values and forward/reverse derivatives of the
prepared RHS, dependent reduction groups, opaque custom derivatives, the
Boozer error budget and the existing LCFS root/hold cases.
