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
  Native XPlane launch resources are also read automatically because older
  JSON exporters can omit `kernel_details`. An unavailable native reader is
  recorded explicitly; it never fabricates resource values.
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

## Completed A100 capture

The [completed console record](a100_console.json) uses the same command above
on NVIDIA A100-SXM4-40GB/JAX 0.7.2. The unprofiled warm median is 187.143 ms.

| Profile measurement | Time, ms |
|---|---:|
| Independent orbit kernel `kernel__1` | 185.928 |
| Union of all kernel intervals | 187.290 |
| First-to-last-kernel span | 187.914954 |
| Time without a kernel inside that span | 0.624954 |

The orbit kernel accounts for 99.3% of summed kernel duration and approximately
98.9% of the captured device span. Even eliminating all work outside that kernel
would recover only about 1.1% of this interval. This directs further inspection
inside the orbit kernel; it does not quantify its individual RHS, stage,
controller or event costs. These short-run percentages are not a decomposition
of the full 20 ms physical-horizon run.

The native XPlane statistics subsequently supplied by the user recover the
resources omitted from the JSON event. The summary parser was not discarding
these fields; they are present in XPlane but absent from the JSON export.

| A100 orbit launch resource | Native profile value |
|---|---:|
| Registers per thread (32-bit registers) | 128 |
| Threads per block | 32 |
| Grid blocks | 8192 |
| Static / dynamic shared memory | 0 / 0 bytes |
| Theoretical occupancy | 25% |

The A100 has 65536 32-bit registers, up to 64 resident warps and up to 32
resident blocks per SM, according to the
[NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/).
Here a block uses 128*32 = 4096 registers. The register-pool calculation gives
16 resident blocks, one warp each: 16/64 = 25%, consistent with the measured
theoretical occupancy. Registers therefore explain a residency limit for this
launch. This does not establish achieved occupancy, executed spills or whether
occupancy limits throughput.

Even eliminating register pressure would not allow 100% occupancy with one
warp per block: the 32-block limit caps this layout at 32/64 = 50%. Increasing
block width is a separate layout change, not a guaranteed speedup. The
`occupancy_suggested_block_size=512` value is an occupancy API recommendation,
not the actual launch block size or a measured optimization result. A useful
candidate is reducing live values in the stage/RHS kernel while preserving
the method and arithmetic; forcing a register cap can instead introduce
spills. No solver modification or speedup is established by this profile.

The native XPlane file already exists alongside the JSON trace. JAX exports
`profiler.ProfileData` in version 0.7.2; the following reader was verified on
the saved local profile without any solver execution. It checks whether the
native event has details absent from the JSON export. On this A100 run it
successfully recovered the resources above, without new particle tracing.

```sh
python - <<'PY'
import json
from pathlib import Path
from jax.profiler import ProfileData

if ProfileData is None:
    raise RuntimeError("This jaxlib build does not expose the native profile reader")
folder = Path("output/boozer_dopri8_profile_a100_run1/trace")
path, = folder.rglob("*.xplane.pb")
profile = ProfileData.from_file(str(path))
for plane in profile.planes:
    if "GPU" in plane.name:
        for line in plane.lines:
            for event in line.events:
                if event.name == "kernel__1":
                    print(json.dumps(dict(event.stats), indent=2, default=str))
PY
```

The Python `event_stats` deprecation warning occurs while reading the profile;
the native statistics were recovered successfully. Executed spill traffic,
instruction stalls, achieved occupancy and FP64/DRAM throughput remain
unmeasured.

Two TensorFlow Python-hook import warnings did not prevent GPU stream capture.
The run finished without the script's host-only or output-equivalence failure.
The console does not expose the raw counts, hashes, memory or equality arrays;
those remain in the remote `profile_summary.json` and have not been supplied.

The separate corrected-reciprocal experiment was rejected on the A100:
[record](../gpu_boozer_dopri5_20261023/rhs_final_a100_console.json). No solver
optimization is retained from that experiment.

## Final register-scheduling attempt

The [saved local record](final_register_try_local.json) tests a single final
candidate: reorder independent pure RHS JAXPR equations using a greedy
last-use schedule, retaining every primitive and operand and preserving effect
barriers. The physical field methods, stage recurrence, controller, tolerances,
dense output and stopping remain unchanged. The candidate is confined to a
scratch module; no production kernel or public solver variant is changed.

This uses the same local 64-particle QA case with 101 saves, a 0.2 ms horizon,
rtol=atol=1e-10 and 15 alternating blocked warm samples per case. Compilation,
three warmups, result transfers and separate native profiling are excluded
from warm timing.

| Measurement, local RTX 5060 Laptop | Baseline | Candidate |
|---|---:|---:|
| Warm median, ms | 91.523 | 92.693 |
| Registers/thread | 128 | 146 |
| Theoretical occupancy | 33.33% | 25.00% |
| Compiled buffers, bytes | 1464936 | 1464936 |
| Lowering/compilation, s | 3.750 | 3.191 |

All six result arrays agree bitwise, with 32716 trials, 28082 accepted steps,
4634 rejections and no failures in both cases. The scheduler's approximate
live-array high-water estimate also fails to decrease (2504 -> 2512 bytes);
that graph estimate is not a register allocator model. Actual native profiling
shows more registers and lower theoretical occupancy, while the warm median
is 1.28% slower in this pair. There is no demonstrated performance gain.
Equal compiled-buffer estimates do not establish equal runtime peak VRAM or
executed spill traffic.

The candidate is rejected. The current production kernel is retained, and no
additional A100 run is requested for this local candidate. This does not prove
that every possible kernel optimization has been exhausted.

## Pipeline, loading and stage-carry alternatives

The separate [four-candidate comparison](final_alternatives_local.json) covers
three additional approaches on the same 64-particle case: compiler pipeline
depths 1 and 2, immutable field-constant loads nearer use, and a smaller stage
loop carry with the endpoint sum repeated after the loop. Compiler pipeline
depth is not the number of RK stages; every case uses the same Dopri8 tableau.
There are 15 blocked warm samples per case with rotating start order and
alternating forward/reverse ordering. Compilation, three warmups, output
comparisons and separate resource profiles are excluded from warm timing.

| Case, local RTX 5060 Laptop | Warm median, ms | Registers/thread | Theoretical occupancy |
|---|---:|---:|---:|
| Baseline | 91.824 | 128 | 33.33% |
| Compiler pipeline depth 1 | 91.936 | 128 | 33.33% |
| Compiler pipeline depth 2 | 91.795 | 128 | 33.33% |
| Load constants nearer use | 92.999 | 152 | 25.00% |
| Smaller stage carry | 91.633 | 128 | 33.33% |

All six result arrays agree bitwise. Every case has 32716 trials, 28082
accepted steps, 4634 rejections and no failures. The pipeline-depth and carry
timing differences are below 0.3% and show no useful demonstrated gain; none
reduces registers. Moving field loads increases registers and lowers
theoretical occupancy. The record retains all individual samples, compilation
times, compiled-buffer estimates and native launch resources.

All four candidates are rejected. They stay outside the public solver API and
no production kernel change is retained. These are local measurements, not
A100 performance claims.

## Reproduce the alternatives on A100

The [standalone paired driver](../../benchmark_boozer_register_final.py)
compares the same five cases on the target machine. It reuses production
kernel source in isolated diagnostic namespaces, checks the expected source
structure before making each replacement, and does not patch any production
module or field class. Its default workload is 8192 particles over 0.2 ms
with 101 saves and 15 warm samples per case. This is a short hardware check,
not five full 20 ms tracing benchmarks.

After syncing `en/split_collisions_ad` on the A100:

```sh
python -m examples.benchmarks.benchmark_boozer_register_final \
  --particles 8192 --horizon 2e-4 --saves 101 --repeats 15 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_register_alternatives_a100_run1
```

If that old cache directory is unavailable, point `--field-cache` at another
existing `boozer_inputs.npz`, or omit it to use the tracked QA cache. Missing
cache files and nonempty output directories are rejected before tracing.
JAX's normal backend must be CUDA; no backend, allocator or package setting
is changed by the script.

The final table and `comparison.json` separate blocked warm time, lowering/
compilation wall time, compiled-buffer bytes, native register counts and
bitwise agreement. Compilation may use existing caches and is not described
as guaranteed cold compilation. Native resource profiles are captured only
after all warm timing samples, using installed JAX/CUPTI packages. Case order
rotates and alternates direction to reduce timing-order bias. All six returned
arrays are checked, and per-particle step counts are compared before summing.
An output-equivalence failure is reported explicitly and requires an accuracy
audit before promoting a timing result.

All individual warm samples, quartiles, paired ratios, physical parameters,
field/birth/source hashes and native launch resources are saved. Compiled
buffers are per executable; allocator peaks collected in this shared process
are cumulative across all five cases, not separate per-case peak VRAM.
The four alternative kernels remain diagnostic-only pending target-machine
evidence. No A100 result from this driver has yet been supplied.
The packaged driver was validated locally with 64 particles and three warm
repeats per case: all five completed, all six outputs matched bitwise and
native resource extraction recovered the expected 128/152 register counts.

API and scope references: [JAX 0.7.2 profiling source](https://github.com/jax-ml/jax/blob/jax-v0.7.2/jax/_src/profiler.py),
[JAX GPU profiling guidance](https://docs.jax.dev/en/latest/201/profiling.html).
