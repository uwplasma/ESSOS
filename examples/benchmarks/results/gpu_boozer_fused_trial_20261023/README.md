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
(`boozer.py`, `rk.py`, `rk_boozer_gpu_trial.py`) retain the hashes recorded in
the successful A100 short comparison.

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
used to predict the A100 workload: applying even the earlier 1.50x local
trial ratio to 75835 versus 160000 iterations would predict about 0.71x
RK4's pure recurrence time, which does not explain the observed 2.39x
full-run time. A same-batch diagnostic on the target machine is required.

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
case for separate-process memory measurements. No further speedup on the
A100 is claimed without its result. Biot-Savart, VMEC, collisions, reverse AD
and ordinary solver defaults retain their existing implementation.
