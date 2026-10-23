# Experimental fused adaptive Dopri8 trial, 2026-10-23

This record contains a local 64-particle CUDA comparison and a remote
8192-particle A100 comparison, both over 0.2 ms, plus a partial console report
from the full 20 ms fused workload. [Local timing and accuracy report](both.json),
[user-pasted A100 short report](a100_short.json),
[full-horizon partial evidence](remote_full_partial.json).
The [remote RK4 comparison](../gpu_boozer_rk4_remote_20261023/README.md)
records why reducing adaptive trial execution is necessary.

## Scope

Both cases use the production `_native_orbits` driver, QA direct Fourier
field with 16 modes, 3.52 MeV alphas, s=0.3, seed 42, 0.2 ms horizon,
101 saves, I control, rtol=atol=1e-10 and standard dense output. CUDA loop
command buffers are enabled in both cases. There is no particle chunking,
compaction, collision update or reverse AD. Timing uses blocked calls on
device-resident inputs; nine warm measurements alternate case order.

The ordinary trial is the current unrolled JAX stage implementation.
The opt-in trial uses a compact Pallas loop to evaluate all Dopri8 stages
inside one GPU kernel. It retains the original tableau and physical
`guiding_center_rhs`, with the adaptive error norm, controller, saves and
root handling still supplied by the existing driver. It requires a fixed,
cosine Fourier Boozer field on CUDA. Field objects retain their existing
structure. The kernel makes a padded view of their spline coefficients;
reduction order and spline recentering can change floating-point results.
This forward-only implementation has no reverse rule and is disabled by
default. Coil, Lorentz, collision and accepted-step reverse paths retain
their existing implementation.

## Local result

RTX 5060 Laptop GPU, JAX/JAXlib 0.11.2, float64:

| Measurement | Standard trial | Fused trial |
|---|---:|---:|
| Warm forward median, ms | 198.47 | 116.25 |
| First execution, ms | 228.39 | 137.29 |
| Lowering/compilation, s | 6.22 | 2.66 |
| Compiled buffers, MiB | 1.466 | 1.454 |
| Total accepted trials | 28084 | 28090 |
| Total rejected trials | 4685 | 4708 |
| Maximum trials per particle | 718 | 718 |
| Lost / failed particles | 1 / 0 | 1 / 0 |
| Maximum relative energy drift | 4.79e-10 | 5.18e-10 |

Warm time decreased by 41.4% (1.71x). The fused case takes slightly more
total trials, with the same worst-case trial count. This measurement
therefore demonstrates faster trial execution rather than fewer steps.
Compilation times are from the paired process, not independent cold workers.
The cumulative JAX allocator peak across both executables is 208.34 MiB;
it does not establish an independent VRAM reduction. Compiled-buffer
estimates do not include every driver allocation or Pallas register spill.

## Accuracy audit

An untimed independent Diffrax Dopri8 solve uses PI p=0.4/i=0.3, rtol=0
and absolute component scales (1,1,1,birth speed), refined from 1e-12 to
1e-13. Compare only common saves before the corresponding first loss.
Positions below are measured in the regular flux chart (u,w,zeta), not
Cartesian metres. Velocity errors are divided by birth speed, not by
instantaneous parallel velocity.

| Maximum saved error against the 1e-13 reference | Standard | Fused |
|---|---:|---:|
| norm(delta u, delta w) | 1.28e-7 | 1.80e-7 |
| abs(delta zeta), rad | 4.93e-7 | 6.98e-7 |
| abs(delta v_parallel) / birth speed | 2.43e-8 | 3.43e-8 |
| abs(delta v_parallel), m/s | 0.316 | 0.448 |
| Particle status disagreements | 0 | 0 |

Reference refinement changes (u,w,zeta,v_parallel/birth speed) by at most
(3.05e-9,1.30e-9,1.20e-8,5.89e-10). The fused run has larger maximum saved
errors in this test, about 1.4x, and is not bitwise identical to the
ordinary trial. In particular, embedded local tolerances are not bounds
on global errors at interpolated saves. The report retains these errors;
no tolerance was relaxed to obtain the speedup.

The isolated GPU test compares endpoints, embedded estimates, RK
increments and final derivatives across 64 states including the axis and
spline edges. Together with existing unrolled event/save and checkpointed
reverse tests, eight checks passed. The new trial cannot enter accepted-step
reverse replay.

## Reproduce locally

```sh
python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison fusion --particles 64 --horizon 2e-4 --saves 101 \
  --repeats 9 --audit-reference --output output/boozer_fused_trial_audit_64
```

The accuracy audit runs after timing and memory snapshots. Omit
`--audit-reference` for a timing-only comparison. Use `--case standard`
or `--case fused` in separate processes for independent memory snapshots.

## Target-machine check before another long run

The first remote attempt failed during fused-kernel compilation with
`Unimplemented primitive in Pallas GPU lowering: is_finite`; it produced no
fused solve timing. The kernel adapter now expresses this guard through an
exact IEEE exponent-bit check. JAX 0.7.2's Triton lowering supports the
required [bitcast](https://github.com/jax-ml/jax/blob/jax-v0.7.2/jax/_src/pallas/triton/lowering.py#L2702)
and [integer mask/comparison](https://github.com/jax-ml/jax/blob/jax-v0.7.2/jax/_src/pallas/triton/lowering.py#L1209)
operations. All ordinary solver and field checks retain their implementation.

Four local GPU checks pass with the `is_finite` lowering deliberately
removed: the complete Dopri8 trial across 64 axis/spline-edge states,
float32 and float64 finite guards (including signed zero, subnormals,
maximum finite values, infinities and NaNs), and the reverse replay guard.
This reproduces the specific missing-primitive restriction; it is not a
full test of the remote JAX environment. The subsequent A100 retry below
compiled and completed on JAX 0.7.2.

The [post-fix full adaptive check](compatibility.json), with 64 particles
and three interleaved warm repeats, measured 194.29 ms standard versus
115.62 ms fused (40.5% reduction). Accepted/rejected counts, statuses and
maximum energy drift are identical to the earlier local fused result.

Keep the remote workload's full 8192-particle batch shape, but use 0.2 ms
instead of 20 ms. Both cases use command buffers and the same input cache:

```sh
python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison fusion --particles 8192 --horizon 2e-4 --saves 101 --repeats 3 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_fused_trial_8192_short_run1
```

## A100 short comparison: remote result supplied by the user

The retry at commit fc99028 completed on an NVIDIA A100-SXM4-40GB with JAX
0.7.2. These are the pasted report's measurements, not local reruns. Its
8192-particle batch, 0.2 ms horizon, 101 saves, seed 42, I controller and
rtol=atol=1e-10 are identical between cases. Three blocked warm samples
alternate case order. Remote output: `output/boozer_fused_trial_8192_short_run2`.

| Measurement | Standard | Fused |
|---|---:|---:|
| Warm forward median, ms | 600.28 | 553.32 |
| Lowering/compilation, s | 14.35 | 5.04 |
| Compiled buffers, MiB | 149.103 | 149.094 |
| Accepted trials | 3539584 | 3539670 |
| Rejected trials | 628909 | 628398 |
| Maximum trials per particle | 760 | 759 |
| Lost / failed particles | 49 / 0 | 49 / 0 |
| Maximum relative energy drift | 8.81e-10 | 8.49e-10 |

Warm time decreased 7.82% (1.085x), and paired compilation time decreased
64.89%. The local 41% runtime reduction did not transfer at the same scale.
Total trials decreased only 0.0102%, so fewer counted trials cannot explain
most of the measured timing gain. This result does not isolate the relative
effects of GPU model, particle count and JAX version.

Compiled buffers decreased by only 9.5 KiB, effectively unchanged. The
213.27 MiB JAX allocator peak is cumulative across both executables and
cannot establish a per-case peak VRAM improvement. Pool reservation is
different from bytes in use.

All particle statuses agree. Maximum saved differences are 1.03 m/s in
parallel velocity and 1.63e-6 rad in zeta. These compare the two numerical
solutions; they are not errors against an independent reference. The A100
run did not request the tight-reference accuracy audit. Similar loss counts
and energy drift do not by themselves establish equal orbit accuracy.

## Full 20 ms fused run: first execution and first warm sample

The subsequent user-pasted console output reports completed full-horizon
solves: 8192 particles, 3.52 MeV, s=0.3, 101 saves, 16 direct Fourier modes,
I control, rtol=atol=1e-10 and the experimental Pallas trial. The partial log
does not print the GPU model or JAX version; the preceding short comparison
was on an A100 with JAX 0.7.2.

| Measurement, seconds | Earlier ordinary Dopri8 | Fused |
|---|---:|---:|
| Lowering/compilation | 14.800 | 4.842 |
| First execution | 54.168 | 48.185 |
| Warm result | 54.367, median of two | 48.246, first of two |

The observed first warm time is 11.26% lower, a 6.121 s reduction, compared
with the earlier ordinary median. These are separate runs, not an interleaved
full-horizon comparison. The second fused warm call and final median were
not supplied. This is evidence of a full-horizon gain, not the 41% measured
locally. The fixed RK4 first warm sample remains 20.215 s, so this fused
sample still takes 2.39x as long; the methods have different error control
and boundary localization.

The fused run reports 351031456 accepted and 62208718 rejected trials,
with no failures. Its accepted count is only 0.0147% higher than the earlier
ordinary run's 350979822. Maximum trials are 75835, requiring at least
985856 logical fresh RHS calls per lane including the initial call, versus
640000 for fixed RK4. The resulting minimum batched RHS estimate is
8076132352, about 1.54x RK4's 5242880000. This is a logical-work comparison,
not a GPU instruction count or a wall-time decomposition.

The final loss count, energy/trajectory accuracy and full-run memory
statistics have not been supplied. The short test's memory figures cannot
substitute for the full-run measurements. Keep the experiment opt-in.
The alpha entrypoint selects it through `--experimental-fused-trial`; its
default command retains ordinary stages. Earlier event and command-buffer
gains did not transfer to the large run, and grouped sums regressed there.

## Further cost investigation: no additional kernel change retained

Subsequent 64-particle experiments changed warp count, stage-bank orientation,
stage grouping, coefficient layout, element selection and GPU block packing.
None showed a reliable local reduction over the current fused kernel.
These candidates remain in ignored scratch files and are not enabled in
the solver or exposed as production variants. All three production sources
(`boozer.py`, `rk.py`, `rk_boozer_gpu_trial.py`) retained the hashes recorded in
the successful A100 short comparison at that point. The later opt-in orbit
experiment below adds a private dispatch in `boozer.py`; its default path,
`rk.py` and the fused trial kernel retain their existing implementation.

Profiling the local current fused solve with graph capture disabled to
expose individual launches recorded 76.10 ms in the trial kernel out of
99.54 ms of summed GPU activity. This diagnostic is not the command-buffer
wall time, and it is not an A100 profile. It identifies the trial as a major
local cost without explaining the target machine's complete timing gap.

An isolated recurrence diagnostic now uses the production `_rk_step`,
ordinary native Dopri8 trial and fused trial, all with the same fixed count
and dt. It includes Dopri8's embedded estimate and complete trial finite
checks, while excluding adaptive decisions, saves, energy diagnostics and
loss handling. It does not run three production fixed/adaptive tracers and
does not establish equal orbit accuracy. Initial FSAL derivatives are
prepared and synchronized outside timing. Complete GPU calls are blocked;
seven warm measurements alternate case order.

The [verified local diagnostic](trial_work_local.json), RTX 5060 Laptop,
JAX 0.11.2, 64 particles, 128 recurrences, dt=1.25e-7 s:

| Recurrence | Warm median, ms | Relative to RK4 | Compiled buffers, bytes |
|---|---:|---:|---:|
| Standard RK4 | 11.57 | 1.00x | 56456 |
| Ordinary native Dopri8 | 30.47 | 2.63x | 97728 |
| Current fused Dopri8 | 16.09 | 1.39x | 45712 |

All 64 endpoints were finite and inside the equilibrium. The source hashes,
raw samples and work counts are in the JSON. These local ratios cannot be
used to predict the A100 workload. The same-batch target diagnostic below
measured a substantially different ratio.

```sh
python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison trial_work --particles 8192 \
  --trial-steps 128 --trial-timestep 1.25e-7 --repeats 7 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_trial_work_8192_run1
```

This executes 128 recurrences per particle, not a 20 ms trace. There is no
boundary stopping in this diagnostic; the report counts final states outside
the equilibrium. Do not use it for confinement or loss-fraction results.
Its `amortized_us_per_rhs_batch` includes recurrence/loop and data costs
divided by the RHS count; it is not an isolated single-RHS latency.
`--case rk4`, `--case dopri8_standard` or `--case dopri8_fused` selects one
case for separate-process memory measurements. Biot-Savart, VMEC, collisions,
reverse AD and ordinary solver defaults retain their existing implementation.

## A100 equal-count trial diagnostic

The user ran the command above at commit `5b722e0` on an A100-SXM4-40GB,
JAX 0.7.2. The [unmodified supplied report](a100_trial_work.json) records
8192 particles, 128 recurrences each, dt=1.25e-7 s and seven interleaved,
blocked warm samples. Its birth hash matches the earlier A100 short run.

| Recurrence | Warm median, ms | Relative to RK4 | Amortized us / fresh RHS batch | Compile, s | Compiled buffers, MiB |
|---|---:|---:|---:|---:|---:|
| Standard RK4 | 14.828 | 1.000x | 28.961 | 1.928 | 3.950 |
| Ordinary native Dopri8 | 45.471 | 3.067x | 27.327 | 9.578 | 8.549 |
| Current fused Dopri8 | 39.096 | 2.637x | 23.495 | 1.287 | 5.454 |

Fusion reduces Dopri8 recurrence time by 14.02% on this machine. The fused
amortized cost per fresh RHS batch is 18.88% lower than RK4's, but it needs
13 fresh evaluations per trial rather than 4. The resulting 2.64x per-trial
ratio differs from the local 1.39x. The lower compilation and buffer figures
in this table belong to these isolated recurrences, not the full tracer.

All 8192 endpoints were finite and inside the equilibrium. This diagnostic
includes the embedded estimate and trial finite checks, but has no tolerance
controller or stopping surface. It does not demonstrate orbit convergence
or equal accuracy with RK4. The allocator's 10.43 MiB peak is cumulative
across setup and all three executables, not an independent per-case peak.

Multiplying these times by the full runs' minimum batch iteration counts
(75835 fused trials and 160000 RK4 slots) gives about 23.16 s and 18.53 s.
The reported full first warm calls were 48.246 s and 20.215 s. This rough
scaling motivates inspecting the adaptive driver, but does not measure a
25 s driver cost: particle states, compiler consumers and scheduling differ
between the recurrence diagnostic and the full integration.

Two additional local saved-output experiments retained every saved value
and accepted/rejected count exactly. Moving saves to an outer scan slowed
the 64-particle full adaptive case from 113.35 to 144.63 ms. Explicitly
masking small particle state and using masked save scatters avoided some
whole-history selections, but did not show a reliable timing improvement.
Neither experiment is retained or enabled. These tests do not establish
which driver operation dominates on the A100; that remains unresolved.

Moving the original finite checks and normalized embedded error inside
the fused trial also gave no local gain: 113.81 ms before versus 114.03 ms
after, with 13 interleaved samples. Reordering the norm calculation changed
some adaptive decisions (28090/4708 accepted/rejected became 28082/4634).
This candidate is also rejected. No additional solver implementation change
is retained from these experiments.

## Independent adaptive orbit kernels: local candidate

The later isolated candidate in `essos/rk_boozer_gpu_orbit.py` keeps each
particle's full adaptive loop inside one CUDA program. It reuses the same
field/RHS adapter, Dopri8 tableau, Boozer block norm, I controller, standard
dense polynomial and 52-iteration boundary-root search. An invalid root
rejects the crossing and retries with .2*h. Saved times do not constrain
steps. Particles stop independently; there is no chunking or compaction.
This changes execution scheduling as well as floating-point evaluation order.

The [local controlled report](orbit_fusion_local.json), RTX 5060 Laptop,
JAX 0.11.2, 64 particles, 0.2 ms, 101 saves, rtol=atol=1e-10, seed 42,
compares the current fused trial plus ordinary driver with the candidate.
Fifteen blocked warm samples alternate case order; compilation and two
independently refined reference solves are outside the warm timing.

| Measurement | Fused trial + ordinary driver | Independent orbit kernels |
|---|---:|---:|
| Warm forward median, ms | 115.197 | 91.623 |
| Lowering/compilation, s | 3.319 | 3.716 |
| Compiled buffers, bytes | 1524952 | 1464936 |
| Accepted / rejected trials | 28090 / 4708 | 28082 / 4634 |
| Maximum trials per particle | 718 | 718 |
| Lost / failed | 1 / 0 | 1 / 0 |
| Maximum relative energy drift | 5.177e-10 | 5.165e-10 |
| Maximum saved flux-chart radial error | 1.774e-7 | 1.898e-8 |
| Maximum saved zeta error, rad | 6.866e-7 | 9.125e-8 |
| Maximum saved parallel-velocity error, m/s | 0.4400 | 0.0448 |
| Parallel-velocity error / birth speed | 3.377e-8 | 3.438e-9 |

Warm time is 20.46% lower in this paired run; earlier paired local comparisons
showed 19--21% reductions. Compilation increases by 0.40 s in this run.
Absolute local timings vary between runs, so compare
the paired rows rather than unrelated earlier local measurements. Total trial
counts decrease only 0.25%, which cannot explain most of this measured gain.
Compiled buffers decrease 3.94%; this is not a separate-process peak VRAM result.

The accuracy audit uses common pre-loss saves against independent Diffrax
Dopri8 at scaled absolute tolerances 1e-12 and 1e-13. The two references differ
by at most 4.62e-9 rad in zeta and 2.28e-10 in parallel velocity / birth speed.
All particle statuses agree. The candidate has smaller global saved errors
in this particular case; rounding changes the adaptive mesh, so this is not
a general claim of higher accuracy. These errors are not local acceptance
estimates or errors relative to instantaneous parallel velocity.

The candidate is opt-in, CUDA-only, forward-only and limited to fixed cosine
Fourier Boozer fields, adaptive Dopri8, I control and standard dense output.
The ordinary default, Biot-Savart, VMEC, collisions and reverse AD retain their
existing execution paths. No A100 timing or full 20 ms gain has been measured
for this candidate. Check the remote JAX 0.7.2 environment and the short
controlled comparison before a full run:

```sh
python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison orbit_fusion --particles 8192 \
  --horizon 2e-4 --saves 101 --repeats 5 --audit-reference \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_fused_orbit_8192_short_run1
```

Use `--case trial` and `--case orbit` separately, omitting `--audit-reference`,
for independent-process peak-memory measurements. For the full alpha example,
`--experimental-fused-orbit` selects this candidate; it is mutually exclusive
with the earlier stage-only `--experimental-fused-trial` option.

Four focused GPU tests cover independent surviving/lost lanes against an
analytical linear trajectory, initial edge hits, NaN births, duplicate saves,
finite/nonfinite rejections and work-limit exits, and unsupported option
guards. Birth validity is checked once with ordinary JAX before entering the
persistent kernel; an invalid birth returns immediately. The existing fused
trial/finite-guard checks and public Boozer entrypoint check also passed.
These are local JAX 0.11.2 checks, including the earlier missing-primitive
proxy; they do not substitute for running the new kernel on JAX 0.7.2.
