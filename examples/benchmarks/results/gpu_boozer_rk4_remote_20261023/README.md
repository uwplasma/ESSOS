Matched alpha RK4 and adaptive Dopri8: remote GPU evidence, 2026-10-23
===================================================================

These measurements were pasted by the user from the large machine. Both
commands use the current QA Boozer transform cache (16 direct Fourier modes),
8192 alpha GCs at 3.52 MeV, s=0.3, seed 42, a 20 ms horizon and 101 saves.
Both enable the same CUDA loop command-buffer options. There are no collisions
or lost-particle compaction. The GPU model was not present in the pasted log.

| Path | Lowering/compilation, s | First solve, s | Warm result, s |
| --- | ---: | ---: | ---: |
| Adaptive Dopri8, separate sums | 14.800 | 54.168 | 54.367 median of two |
| Adaptive Dopri8, experimental grouped sums | 14.550 | 62.301 | Not supplied |
| Adaptive Dopri8, experimental fused trial | 4.842 | 48.185 | 48.246, first warm sample |
| Standard fixed RK4 | 2.653 | 20.259 | 20.215, first warm sample |

The second RK4 and fused warm samples and their final medians were not
supplied. The [fused record](../gpu_boozer_fused_trial_20261023/remote_full_partial.json)
preserves the full-horizon partial console evidence. The fused first warm
sample is 11.26% below the earlier ordinary median but still 2.39x RK4's
first warm sample. Final full-run loss, accuracy and memory diagnostics
were not supplied for the fused run. The RK4
log reports no failures and 160,000 executed grid slots per lane. These count
held inactive lanes as well. The earlier separate-sum Dopri8 run accepted
350,979,822 steps in total, with a maximum of 70,886 accepted steps per particle;
it lost 110 particles and had no failures. Its rejected-trial maximum was not
included in the pasted summary. Grouping remains disabled by default.

The step counts are not equal-cost units. RK4 makes four fresh RHS calls per
fixed step. Native Dopri8 makes thirteen per trial, reusing its initial FSAL
derivative. The vmapped adaptive loop executes until its slowest lane finishes;
masked lanes still incur stage computation in the existing driver. Consequently,
even ignoring rejections, the observed maximum accepted count implies at least
70,886*13 = 921,518 RHS evaluations per lane in that execution layout, compared
with 160,000*4 = 640,000 for RK4. This is a lower bound on logical lane work,
not a count of emitted GPU instructions or a precise timing decomposition.
Energy-only field evaluations are additional work in both tracers.

The alpha benchmark now reports `rhs_work_estimate` immediately after its first
blocked solve. It distinguishes counted active trial slots from a lower bound
on batched slots using the maximum trial count. This diagnoses padding in the
adaptive driver without changing the numerical method, tolerance or controller.

These timings do not establish equal orbit accuracy. RK4 uses fixed dt=1.25e-7 s
without error-based rejection; Dopri8 uses rtol=atol=1e-10. RK4 checks boundary
crossing at the step endpoint, whereas native Dopri8 refines the crossing with
its dense interpolant. Full saved-state/energy comparison requires the raw
`traces.npz` and `benchmark.json` outputs from the remote runs.

Existing remote output folders:

```text
output/native_boozer_dopri8_8192_device_loops_run1
output/boozer_rk4_alpha_8192_run1
```
