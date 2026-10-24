# Native Boozer Dopri8 baseline GPU profile

Use the [profiling script](../../profile_native_boozer_alpha.py) to collect
an actual JAX/CUPTI timeline of the unchanged independent-orbit solver.
It uses installed packages and does not require Nsight, TensorBoard, a web
server or an interactive link. Compilation, two warmups, three blocked warm
timings and output transfers finish before the single captured solve.
The ordinary I controller, rtol=atol=1e-10, standard dense output, s=1 stopping
and QA Fourier field are retained. There are no alternative solver kernels,
arithmetic modifications, collisions, reverse AD or particle chunks.

On the A100, after syncing this branch, run:

```sh
python -m examples.benchmarks.profile_native_boozer_alpha \
  --particles 8192 --horizon 2e-4 --saves 101 --repeats 3 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_dopri8_profile_a100_run1
```

This profiles 0.2 ms of physical time, not the full 20 ms performance case.
The script checks that the cache exists before GPU setup. Use a different
existing `boozer_inputs.npz` path if that output directory is unavailable.

Outputs in the chosen directory:

- `profile_summary.json`: unprofiled warm timing, actual per-kernel device
  durations, launch registers/shared memory/grid/block sizes and theoretical
  occupancy when the collector provides them, hashes, counts and validation.
- `trace/plugins/profile/.../perfetto_trace.json.gz`: open this local file
  in [Perfetto](https://ui.perfetto.dev) for CPU/GPU timeline inspection.
- The original `*.trace.json.gz`, `*.xplane.pb` and `compiled_hlo.txt` are
  retained for detailed inspection. Large generated traces stay in `output/`.

The printed kernel percentages divide by summed kernel duration, not CPU
wall time; overlapping kernels can make the sum exceed the interval union.
Aggregate GPU operation tracks are excluded so they do not double-count GPU
stream events. A CPU-only capture raises an error with `complete=false`.
All six returned result arrays, including step counts, must remain bitwise
equal to the unprofiled baseline.

This trace can establish which kernels dominate and their launch resources.
It does not measure instruction stalls, executed spill traffic, achieved
occupancy or FP64/DRAM throughput. Low theoretical occupancy is a clue to
inspect register use and work distribution, not proof of the limiting factor.
The fused kernel contains the RHS, RK stages and controller together; its
duration does not separate those internal operations. Do not substitute the
profiled timing for the unprofiled warm performance result.

The [local validation record](local64_summary.json) uses 64 particles on
RTX 5060 Laptop/JAX 0.11.2. The GPU capture put about 99.6% of summed kernel duration in `kernel`, with
128 registers/thread, one 32-thread block per orbit, no shared memory and
33.33% theoretical occupancy. All outputs agreed bitwise and no particle
failed. The unprofiled warm median was 91.435 ms. The profiled orbit kernel
took 79.334 ms; all GPU kernels occupied a 79.626 ms interval union within
a 91.315 ms first-to-last-kernel span. The 11.689 ms without kernels can
include transfers, launch/synchronization overhead and idle time; it is not
an instruction-stall measurement. Thus the 99.6% kernel share is not a claim
that 99.6% of total runtime is in that kernel. There were 102 copy/memset
events. These local timings are not used to predict A100 behavior.
The validation command set `XLA_PYTHON_CLIENT_PREALLOCATE=false` locally to
avoid reserving most of the laptop GPU; the script does not change allocators
or backend selection.
The target-machine profile has not yet been supplied.

The separate corrected-reciprocal experiment was rejected on the A100:
[record](../gpu_boozer_dopri5_20261023/rhs_final_a100_console.json). No solver
optimization is retained from that experiment.

API and scope references: [JAX 0.7.2 profiling source](https://github.com/jax-ml/jax/blob/jax-v0.7.2/jax/_src/profiler.py),
[JAX GPU profiling guidance](https://docs.jax.dev/en/latest/201/profiling.html).
