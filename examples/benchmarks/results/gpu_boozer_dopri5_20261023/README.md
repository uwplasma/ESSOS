Dopri5 independent adaptive GPU orbits and RK4 accuracy audit, 2026-10-23
=====================================================================

The existing CUDA forward-only implementation now selects either Dopri5 or
Dopri8 at setup. Both use the same physical Boozer RHS, adaptive driver,
block error norm, I controller, stopping surface and root retry policy.
Dopri5 uses the original seven-stage tableau and embedded estimator from
`essos/rk.py`, six fresh RHS evaluations per trial, controller exponent -1/5,
and the original quartic dense extension. Dopri8 retains its existing stages
and dense output. Field classes and the ordinary solver defaults are unchanged.
There is no collision or reverse-AD rule for these experimental kernels.

Measurements below use the RTX 5060 Laptop GPU, JAX 0.11.2, 64 identical
3.52 MeV alpha births at s=0.3, seed 42, 16 direct Fourier modes, a 0.2 ms
horizon and 101 saves. Both native methods use adaptive I control and
rtol=atol=1e-10. No particle chunking or compaction is used.

Timing and buffer results
-------------------------

Each paired comparison has 15 blocked warm samples with alternating case
order. Compilation, setup and accuracy-reference calculations are excluded
from warm timings. Each comparison runs in a fresh process.

| Paired Dopri5 comparison | Before warm, ms | After warm, ms | Reduction | Before buffers, bytes | After buffers, bytes |
|---|---:|---:|---:|---:|---:|
| Ordinary stages -> fused stages | 292.637 | 197.679 | 32.45% | 1530272 | 1522072 |
| Fused stages + ordinary driver -> independent orbit kernels | 194.872 | 115.102 | 40.93% | 1522072 | 1464936 |

The records are [stage fusion](fusion_local.json) and
[whole-orbit fusion with the accuracy audit](orbit_fusion_local.json).
These are two separate paired comparisons; use the paired control rather
than multiplying speedup percentages or mixing unrelated timing samples.
For whole-orbit fusion compilation changes 3.469 -> 3.890 s. Compiled buffers
decrease about 3.75%; allocator peaks are cumulative across a comparison's
executables and setup, not separate per-case peak VRAM measurements.

All three Dopri5 paths complete 71401 accepted and 4838 rejected trials,
with maximum 1931 trials per particle, one lost particle and zero failures.
The improvements do not result from fewer active integration trials.
The later full target-machine result is recorded below; these local paired
reductions should not be substituted for the remote timings.

Full target-machine result
--------------------------

The user's [completed console log](remote_full_orbit_console.json) reports
8192 particles over 20 ms with the same cached fields, births, saves and
native tolerances. Device model and JAX version were not printed in this
full log. The earlier short Dopri8 run identified an A100 and JAX 0.7.2.

| Method | Compile, s | Warm median, s | Accepted steps | Rejected trials | Lost / failed |
|---|---:|---:|---:|---:|---:|
| Independent adaptive Dopri5 | 4.247 | 21.554 | 895113025 | 63143676 | 110 / 0 |
| Independent adaptive Dopri8, earlier run | 5.069 | 18.448 | 351025279 | 62226975 | 111 / 0 |
| Fixed RK4, completed run 3 | 2.728 | 20.244 | 1310720000 grid slots | 0 | 108 / 0 |

Dopri5's warm samples are 21.321 and 21.787 s. It takes 2.55 times as many
accepted steps as Dopri8 and about 7.0% more fresh RHS evaluations in total,
despite needing only six per trial versus thirteen. Its runtime is 16.8%
longer. The rejection fraction is smaller (6.6% versus 15.1%); smaller accepted
steps at the tight tolerance dominate this comparison. These are separate
runs, not a paired interleaved test. RK4 grid counts include held inactive
lanes, and the methods do not have identical stopping resolution or proven
equal full-horizon accuracy. Full-run peak memory and independent error
diagnostics were not supplied.

Achieved trajectory errors: RK4 versus Dopri5 and Dopri8
------------------------------------------------------

Fixed RK4 has no rtol/atol controller. Its timestep is not an achieved error
budget. This audit runs the production fixed RK4 tracer at the benchmark
timestep and two successive halvings, with the same fields, births, horizon
and saved times. It also runs the native independent-orbit Dopri8 control.
These are untimed accuracy solves, separate from the two Dopri5 timing rows.

All errors below are maximum errors over common pre-loss saved times against
an independent Diffrax Dopri8 reference at scaled absolute tolerance 1e-13.
A second reference uses 1e-12. The reference's RMS norm scales the regular
chart coordinates by (1,1,1,birth_speed). It differs from the native block
norm but supplies a more tightly resolved independent solution.

| Method | Fixed dt, s / adaptive tolerances | Max radial chart error | Max zeta error, rad | Max v_parallel error, m/s | v_parallel error / birth speed |
|---|---|---:|---:|---:|---:|
| Fixed RK4 | dt=1.25e-7 | 7.466e-6 | 2.559e-6 | 0.88519 | 6.794e-8 |
| Fixed RK4 | dt=6.25e-8 | 4.789e-7 | 1.440e-7 | 0.05579 | 4.282e-9 |
| Fixed RK4 | dt=3.125e-8 | 3.033e-8 | 8.498e-9 | 0.00350 | 2.688e-10 |
| Adaptive native Dopri5, fused orbit | rtol=atol=1e-10 | 1.168e-7 | 4.504e-7 | 0.28887 | 2.217e-8 |
| Adaptive native Dopri8, fused orbit | rtol=atol=1e-10 | 1.897e-8 | 9.125e-8 | 0.04479 | 3.438e-9 |

Radial chart error means norm(delta_u,delta_w), with
(u,w)=sqrt(s)*(cos(theta),sin(theta)); it is not a Cartesian error in metres.
Velocity is scaled by birth speed, not by instantaneous v_parallel near a
turning point. These are achieved global saved-point errors in this case,
not embedded estimates or guaranteed global tolerances.

RK4 velocity and radial errors decrease by approximately a factor of 16 per
halving, consistent with its fourth-order convergence in this test. RK4 at
half the benchmark timestep has similar velocity error to native Dopri8,
but larger radial chart error; there is no single matched tolerance across
all variables. Both native methods have smaller errors than RK4 at the
benchmark dt in this sample, so the existing timing comparison does not
establish equal accuracy.

The two reference refinements differ by up to 1.204e-8 rad in zeta and
5.890e-10 of birth speed in parallel velocity (about 0.0077 m/s). The finest
RK4 zeta/velocity errors are at or below those differences; do not interpret
that row as a precise accuracy ranking below the reference uncertainty.
The refinement difference is itself a diagnostic, not a rigorous error bound.
All methods agree with reference loss statuses in this 64-particle case.
The full 8192-particle, 20 ms workload still needs its own error audit.

Dopri8 regression and validation
--------------------------------

The [before/after record](dopri8_regression_local.json) compares the shared
driver with its Dopri8-only implementation at commit `e291a7c`. Every saved
state, diagnostic, status and accepted/rejected count agrees bitwise.
Compiled buffers remain 1464936 bytes. Paired warm medians are 93.992 and
94.058 ms (0.07% difference), with no observed material timing regression.

Fifteen targeted GPU tests pass, including both tableaus and FSAL values at
axis/spline edges, finite guards, rejected trials, initial boundary hits,
duplicate saves, work limits, a quartic analytical trajectory sampled inside
one accepted step, unsupported method/controller guards and the ordinary
Boozer public entrypoint. The normal alpha CLI also completes for Dopri5
with 64 particles, one lost and zero failed.

The A100/JAX 0.7.2 short Dopri8 result remains historical evidence. The later
full Dopri5 console result above confirms completion on the large machine,
without printing its exact software stack. Biot-Savart, VMEC, collisions and
the checkpointed reverse solver retain their existing paths.

Reproduction
------------

Local paired stages:

```sh
python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison fusion --method dopri5 --particles 64 \
  --horizon 2e-4 --saves 101 --repeats 15 \
  --output output/boozer_dopri5_stage_fusion_64
```

Local paired driver plus RK4/Dopri5/Dopri8 accuracy audit:

```sh
python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison orbit_fusion --method dopri5 --particles 64 \
  --horizon 2e-4 --saves 101 --repeats 15 \
  --audit-reference --audit-rk4 --trial-timestep 1.25e-7 \
  --output output/boozer_dopri5_orbit_accuracy_64
```

The same short comparison on the large machine, retaining the cached fields:

```sh
python -m examples.benchmarks.benchmark_boozer_adaptive_events \
  --comparison orbit_fusion --method dopri5 --particles 8192 \
  --horizon 2e-4 --saves 101 --repeats 5 --audit-reference --audit-rk4 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_dopri5_orbit_accuracy_8192_short_run1
```

After the short target check, the full native Dopri5 command is:

```sh
python -m examples.benchmarks.benchmark_native_boozer_alpha \
  --integrator essos_rk45 --experimental-fused-orbit \
  --particles 8192 --horizon 0.02 --saves 101 --repeats 2 \
  --rtol 1e-10 --atol 1e-10 \
  --controller i --stage-execution unrolled --seed 42 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/native_boozer_dopri5_8192_fused_orbit_run1
```

The standard full RK4 timing command is:

```sh
python -m examples.benchmarks.benchmark_native_boozer_alpha \
  --integrator rk4 --particles 8192 --horizon 0.02 --saves 101 \
  --timestep 1.25e-7 --seed 42 --repeats 2 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_rk4_alpha_8192_run2
```

Further Dopri8 tuning: no retained solver change
-----------------------------------------------

The [local follow-up record](dopri8_followup_local.json) preserves discarded
experiments on the same 64-particle, 0.2 ms, 101-save case. Each paired
comparison uses 15 interleaved blocked warm samples, with identical fields,
births, method, controller and rtol=atol=1e-10. Production source modules were
not modified; variants were generated in a task-local scratch harness.

Stage grouping, two warps per orbit, bit-preserving selection alternatives,
Fourier recurrences, explicit field-value caching and a transposed stage bank
did not improve warm execution. Full stage unrolling was interrupted after
excessive compilation and has no timing result. These variants are not added
to the production API or CLI.

The promising change replaces FP64 division by multiplication with a shared
reciprocal, falling back to the original RHS when reciprocals are not finite
and normal. The same physical equations, tableau, tolerances and error norm
are retained; ordinary floating-point rounding changes the adaptive mesh.
The final decision pair and its untimed two-refinement Diffrax audit show:

| Version | Warm median, ms | Compile, s | Trials | Max radial chart error | Max parallel-velocity error, m/s |
|---|---:|---:|---:|---:|---:|
| Current independent orbit kernel | 92.383 | 4.481 | 32716 | 1.898e-8 | 0.04480 |
| Dead-code pruning only | 92.138 | 4.001 | 32716 | 1.898e-8 | 0.04480 |
| Guarded reciprocal multiplication | 85.092 | 5.383 | 32854 | 7.720e-8 | 0.19009 |

The faster candidate saves 7.89% in this pair but has about 4.24 times the
maximum saved parallel-velocity error. This single-case difference is not
proof of systematic bias or violation of the embedded local tolerance; it
does not establish an equal-accuracy speedup either. All three have one lost
particle, zero failures and no disagreement with reference loss statuses.

Adding explicit fused rounding correction reproduces all six result arrays
bitwise, including saved states and step counts, but changes warm execution
91.326 -> 93.248 ms and compilation 4.249 -> 6.118 s in its own pair. Applying
that correction only to live/shared denominators also fails to improve time.
Dead-code pruning's 0.27% timing difference is not a useful demonstrated gain.
All cases report 1464936 compiled-buffer bytes; these estimates do not establish
equal per-case peak VRAM or register-spill use.

No production solver change is retained from this investigation. The observed
full remote Dopri8 median remains 18.448 s. The corrected-reciprocal candidate
was subsequently checked on the large machine as recorded below. The other
local candidates have no target-machine result.

Final target-GPU RHS check without Nsight
----------------------------------------

When Nsight is unavailable, run the standalone
[paired RHS diagnostic](../../benchmark_boozer_rhs_final.py). It uses the
current independent-orbit Dopri8 solver and its original field classes.
The candidate caches reciprocals, corrects each quotient with two explicit
FP64 fused operations, and falls back to ordinary division for exceptional
denominators or overflowing initial quotients. Overrides are confined to
sequential tracing inside this diagnostic and restored after each compilation.
There is no production solver change or new solver default.

The diagnostic measures both the complete short adaptive solve and an
equal-count 128-trial recurrence. Only the adaptive test includes controller,
saves, energy diagnostics and stopping. The stage test includes the embedded
estimate and finite checks. Both use the same fields, births and stage formula;
the initial FSAL derivative is excluded from stage timings. Every saved state,
diagnostic and count must agree bitwise with its baseline. Timings alternate
case order and block on GPU completion. The optional candidate is not promoted
automatically; the reported retention screen requires bitwise equality and at
least 3% lower adaptive warm median, followed by review of samples and memory.

The [64-particle local smoke result](rhs_final_local.json), with nine repeats,
is deliberately retained as negative evidence:

| Case | Baseline warm, ms | Corrected warm, ms | Bitwise equality |
|---|---:|---:|---|
| Adaptive orbit, 0.2 ms and 101 saves | 94.373 | 96.506 | Yes, all six output arrays |
| 128 stage trials | 16.389 | 17.257 | Yes, all four output arrays |

The local result does not demonstrate a speedup. The completed
[A100 console record](rhs_final_a100_console.json) also rejects this candidate:

| Case, 8192 particles | Baseline warm, ms | Corrected warm, ms | Time increase | Bitwise equality |
|---|---:|---:|---:|---|
| Adaptive orbit, 0.2 ms and 101 saves | 186.523 | 202.319 | 8.5% | Yes |
| 128 stage trials | 39.163 | 46.311 | 18.3% | Yes |

These are paired arithmetic benchmarks, not hardware profiles. They establish
that the corrected division is slower; they do not identify instruction stalls,
executed spill traffic, achieved occupancy or per-case peak VRAM. Only the
completed console was supplied; raw samples, compilation and memory values
remain unavailable for this target-machine run.

```sh
python -m examples.benchmarks.benchmark_boozer_rhs_final \
  --particles 8192 --horizon 2e-4 --saves 101 --repeats 9 \
  --field-cache output/native_boozer_dopri8_8192_event_replay_run1/boozer_inputs.npz \
  --output output/boozer_dopri8_rhs_final_a100_run1
```

The four-row timing table is printed and saved to `comparison.json`. This is
a short target-GPU diagnostic, not another 20 ms full-horizon performance run.
Do not combine its milliseconds with full-horizon seconds in one timing row.

The separate [baseline profiling workflow](../gpu_boozer_profile_20261023/README.md)
now captures an actual JAX/CUPTI GPU timeline and launch resources without
installing Nsight or changing solver arithmetic.

Consistency of the full 18.448 s result
-------------------------------------

The earlier [A100 stage diagnostic](../gpu_boozer_fused_trial_20261023/a100_trial_work.json)
measured 39.096 ms for 128 fused Dopri8 trials across 8192 particles. Scaling
that throughput by the full independent run's 413252254 active trials gives
39.096e-3 * 413252254 / (8192*128) = 15.408 s. The observed 18.448 s is about
19.7% larger. Controller work, output/event handling and workload imbalance
are plausible additional costs, but this is not a measured decomposition:
the stage diagnostic and the independent-orbit solver have different execution
layouts and varying field access patterns. The estimate is not a lower bound
or a guaranteed optimization target. It does show that 18.448 s is consistent
with the measured order of stage cost, without evidence of the former large
outer-driver overhead. The 16 ms trial figure from earlier discussion was not
this recorded A100 measurement.
