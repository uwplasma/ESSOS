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
The four alternative kernels remain diagnostic-only. The completed A100
comparison is recorded below.
The packaged driver was validated locally with 64 particles and three warm
repeats per case: all five completed, all six outputs matched bitwise and
native resource extraction recovered the expected 128/152 register counts.

## Completed A100 alternatives comparison

The [console evidence](final_alternatives_a100_console.json) records the
completed 8192-particle, 0.2 ms run on A100-SXM4-40GB with JAX 0.7.2. All five
cases produce bitwise-identical outputs and the same counts: 4,169,494 trials,
3,539,926 accepted steps and 629,568 rejections.

| Case | Warm median, ms | Change from baseline | Compile, s | Compiled buffers, MiB | Registers/thread |
|---|---:|---:|---:|---:|---:|
| Baseline | 187.365 | 0% | 4.979 | 96.112 | 128 |
| Pipeline depth 1 | 187.680 | +0.17% | 4.745 | 96.112 | 128 |
| Pipeline depth 2 | 188.467 | +0.59% | 4.853 | 96.112 | 128 |
| Load constants nearer use | 192.641 | +2.82% | 5.228 | 96.112 | 124 |
| Smaller stage carry | 186.170 | -0.64% | 4.896 | 96.112 | 128 |

Smaller stage carry has a 1.195 ms lower reported median, without reducing
registers or compiled buffers. The supplied console does not include the
15 timing samples or quartiles needed to assess variability. Loading constants
nearer use reduces registers by four on this GPU but runs slower; lower
register count alone does not establish a throughput gain. Buffer figures
are rounded compiler estimates, not measured per-case peak VRAM.

Retain the production baseline. The short comparison establishes no substantial
gain from these alternatives and does not predict a full 20 ms speedup. Ten
TensorFlow Python-hook warnings did not prevent completion or native register
extraction. No new packages are needed to interpret these results.

## Completed full 20 ms A100 comparison

The [full-horizon console record](final_alternatives_a100_full_console.json)
uses the same 8192 particles, 101 saves, 15 warm samples per case and
rtol=atol=1e-10 over 20 ms. Every case agrees bitwise and reports 413,252,254
trials, 351,025,279 accepted steps and 62,226,975 rejections.

| Case | Warm median, s | Change from baseline | Compile, s | Compiled buffers, MiB | Registers/thread |
|---|---:|---:|---:|---:|---:|
| Baseline | 18.459165 | 0% | 5.034 | 96.112 | 128 |
| Pipeline depth 1 | 18.359990 | -0.54% | 4.789 | 96.112 | 128 |
| Pipeline depth 2 | 18.430769 | -0.15% | 4.928 | 96.112 | 128 |
| Load constants nearer use | 18.997608 | +2.92% | 5.081 | 96.112 | 124 |
| Smaller stage carry | 18.337851 | -0.66% | 4.909 | 96.112 | 128 |

Smaller stage carry has the lowest warm median in both short and full runs.
In the full run it saves 0.121314 s per warm call and 0.125 s of reported
compilation versus baseline, without changing the printed buffer estimate or
register count. It is the preferred candidate for repeated tracing if one
alternative is retained. The gain remains small; the supplied console has no
individual samples or quartiles for assessing variability.

For a single call, reported compilation plus one warm median is 23.148990 s
for pipeline depth 1, 23.246851 s for smaller stage carry and 23.493165 s for
baseline. These are estimates, not measured first-execution totals. Pipeline
depth 1 compiles 0.120 s faster than smaller carry, but each warm solve is
0.022139 s slower. Smaller carry overtakes it from the sixth call using these
point estimates. Pipeline depth 2 and near-use loading offer no advantage over
these two candidates. Production remains unchanged by this diagnostic.

### Full-horizon runtime and progress

Changing `--horizon` from `2e-4` to `2e-2` gives every solve the full 20 ms
physical horizon. It does not divide the horizon across repeats. Five cases
with three warmups, 15 timed calls and one profiling call each execute 95
full solves. At approximately 18-19 seconds per solve, expect roughly
29-30 minutes plus compilation and other overhead. After all five initial
output checks print, 75 timed solves still remain (about 23 minutes), followed
by five profiling solves. This accounts for a long GPU-busy interval without
new console output in the original driver.

The driver now prints the solve budget, each warmup, each completed timing
and each profiling call. It saves completed samples after each timed call
and on a timing-loop interruption. These messages and file writes occur
outside the measured interval. Existing running processes retain the old
logging behavior; updating the file does not add progress to a running job.
The completed full-horizon table is preserved above.

## Combined smaller carry and pipeline depth 1

The [local combined-candidate record](lean_pipeline_local.json) compares
baseline, pipeline depth 1, smaller stage carry and their combination using
64 particles on RTX 5060 Laptop/JAX 0.11.2. The workload is 0.2 ms, 101 saves,
rtol=atol=1e-10 and 15 rotating/alternating blocked warm samples per case.
Persistent JAX compilation caching was disabled for this invocation; other
compiler caches and initialization effects are not guaranteed absent.

The driver now separates tracing/lowering from `lowered.compile()`. The
existing JSON `compile_s` field remains their combined wall time for comparison
with older records; `trace_and_lower_s` and `executable_compile_s` hold the two
phases. Field setup and preparation are outside those timers.

| Local case | Trace/lower, s | Executable compile, s | Combined, s | Warm median, ms |
|---|---:|---:|---:|---:|
| Baseline | 1.607 | 2.195 | 3.803 | 91.906 |
| Pipeline depth 1 | 1.495 | 1.618 | 3.113 | 91.658 |
| Smaller stage carry | 1.456 | 1.689 | 3.145 | 91.825 |
| Smaller carry + pipeline depth 1 | 1.484 | 1.714 | 3.198 | 91.944 |

All six output arrays and per-particle counts agree bitwise, with 32,716
trials, 28,082 accepted steps and 4,634 rejections in every case. All four use
1,464,936 compiled-buffer bytes and 128 registers/thread; all report no
failures. Combining the changes does not improve the measured compilation
time over smaller carry alone, and the warm medians differ by only 0.13%.
These compilation phase measurements are one pass, not repeated cold-process
statistics. There is no demonstrated benefit from promoting the combination.
Production kernels and field classes remain unchanged. No A100 measurement
of this combination has been supplied.

The original five default cases are preserved. Select only these four cases
for a short target-machine check with:

```sh
python -m examples.benchmarks.benchmark_boozer_register_final \
  --cases baseline pipeline1 lean_carry lean_pipeline1 \
  --particles 8192 --horizon 2e-4 --saves 101 --repeats 15 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_lean_pipeline_a100_run1
```

This command uses the machine's existing cache configuration, which is recorded
in the report. A local validation run used the same options with 64 particles
and command-only `JAX_ENABLE_COMPILATION_CACHE=false` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; script defaults are unchanged. The new
`--cases` option always includes baseline first, and prints the selected solve
budget before compilation. A full 20 ms horizon would again multiply the cost
of every warmup, timed call and profile; the command above retains 0.2 ms.

API and scope references: [JAX 0.7.2 profiling source](https://github.com/jax-ml/jax/blob/jax-v0.7.2/jax/_src/profiler.py),
[JAX GPU profiling guidance](https://docs.jax.dev/en/latest/201/profiling.html).

## Further lean-carry compilation experiments

The [local evidence](lean_compile_local.json) preserves four 64-particle
comparison passes on RTX 5060 Laptop/JAX 0.11.2. Each successful case uses
15 blocked warm samples, the same 0.2 ms QA workload, 101 saves, Dopri8,
rtol=atol=1e-10 and I control. Persistent JAX compilation caching is disabled
for these invocations; compiler caches and first-case initialization effects
are not guaranteed absent. Compilations are single measurements per case per
pass, not repeated independent cold-process statistics.

The main candidate evaluates the existing dense-output polynomial using a
Horner loop rather than expanding its coefficients in a Python loop. It
retains the coefficients, evaluation order and interpolant. A second candidate
loops only the two loss-root evaluations and keeps ordinary saved output
unrolled. Both include the smaller RK stage carry; neither changes the
physical RHS, tableau, error estimator, I controller or requested saves.

| Final pass, local 64 particles | Trace/lower, s | Executable compile, s | Combined, s | Warm median, ms |
|---|---:|---:|---:|---:|
| Baseline | 1.623 | 2.139 | 3.761 | 91.818 |
| Smaller stage carry | 1.490 | 1.672 | 3.162 | 91.466 |
| Loop polynomial only at loss roots | 1.323 | 1.481 | 2.804 | 92.000 |
| Loop polynomial for saves and roots | 1.252 | 1.343 | 2.595 | 92.323 |

Compared with smaller carry in this pass, root-only looping saves about 11%
of combined compilation with a 0.6% higher warm median; looping at every dense
evaluation saves about 18% with a 0.9% higher warm median. The fully looped
candidate's compilation saving ranges from 12% to 18% over four passes, with
warm median increases of about 0.7-1.0%. The record retains individual samples,
quartiles and paired ratios, including their timing variation.

All successful physical cases agree bitwise in all six returned arrays and
per-particle counts: 32,716 trials, 28,082 accepted steps and 4,634 rejections.
All report zero failures, 1,464,936 compiled-buffer bytes and 128
registers/thread. These buffer estimates do not measure separate peak VRAM.
The two promising candidates also pass four existing analytical checks each:
linear stopping with initial events and duplicate saves, finite rejection at
the work limit, nonfinite rejection and polynomial dense saves. Reproduce the
eight checks with:

```sh
python -m pytest -q tests/test_rk_boozer_gpu_orbit.py \
  -k looped_dense_candidate_analytical_checks
```

Other attempts are rejected: removing the nested JIT wrapper has no compilation
gain; boolean finite-mask reduction cannot lower through Pallas Triton;
generic `fori_loop(unroll=2/4)` also cannot lower on this backend. Those
unsupported forms were removed. Explicit fixed-size Horner blocks of two
and four do compile and preserve outputs, but offer no better tradeoff than
the candidates above. The explicit-block cases remain diagnostic-only for
reproducing the saved measurements.

These are compilation/runtime tradeoffs on the local GPU, not A100 gains.
Production remains unchanged. The short workload saves 101 points over
0.2 ms; the full benchmark saves the same number over 20 ms, so dense-output
frequency relative to RK work differs. A full-run runtime penalty cannot be
inferred directly from these short timings. Test the two candidates against
their controls on A100 before choosing one:

```sh
JAX_ENABLE_COMPILATION_CACHE=false python -m examples.benchmarks.benchmark_boozer_register_final \
  --cases baseline lean_carry lean_dense_roots lean_dense_loop \
  --particles 8192 --horizon 2e-4 --saves 101 --repeats 15 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_lean_compile_a100_run1
```

The cache override affects only this invocation, without selecting a device
or changing solver defaults. This is four cases over the short horizon,
not another full-horizon 30-minute sweep. No A100 result for these candidates
has been supplied, and neither implements a reverse rule.
