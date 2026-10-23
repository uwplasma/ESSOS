# Experimental fused adaptive Dopri8 trial, 2026-10-23

This is a local 64-particle CUDA measurement, not a measurement or prediction
of the user's 8192-particle run. [Raw timing and accuracy report](both.json).
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
full test of the remote JAX environment. The remote timing comparison still
needs to be retried after updating the checkout.

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

No target-machine fused result is available yet. Earlier event and command
buffer gains on the local GPU did not transfer to the large machine, and
grouped sums regressed there. Keep this experiment opt-in pending the
matched target-machine measurement. The alpha entrypoint exposes it as
`--experimental-fused-trial`; its default command retains ordinary stages.
