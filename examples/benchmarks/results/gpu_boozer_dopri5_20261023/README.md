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
No Dopri5 A100 or full 8192-particle timing is claimed yet.

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

The A100/JAX 0.7.2 Dopri8 result remains historical evidence. This generalized
Dopri5 kernel has not yet run on that stack. The existing finite-guard proxy
tests do not replace a target-machine test. Biot-Savart, VMEC, collisions and
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
