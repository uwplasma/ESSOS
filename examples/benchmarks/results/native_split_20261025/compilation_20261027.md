# Cold compilation comparison, 27 October 2026

Local GPU: NVIDIA GeForce RTX 5060 Laptop GPU; JAX/JAXLIB 0.11.2,
Python 3.14 under WSL. Each row uses a fresh process with the persistent JAX
solver compilation cache disabled and a separate Triton cache directory. These are
single cold measurements, not averages or A100 measurements.

The split-native rows use adaptive Dopri8, the retained moment predictor
(`moment_tolerance=0.005`), collision tolerance `0.001`, and orbit
`rtol=atol=1e-10`. Shared and independent execution use the same particles,
field and numerical method within each workload. Cartesian GC includes its
existing spatial collision update. VMEC/Boozer use their existing velocity/pitch
collision operators with spatial diffusion explicitly disabled.

Two 4 keV alpha particles, seed 42, a 1 ns horizon, three saves, 32 maximum
collision trials and 64 maximum inner orbit trials keep execution short while
compiling the full selected solver. A uniform prescribed background has density
`1e20 m^-3` and temperature `1000 eV`. Coil geometry is the bundled
Landreman/Paul QA coil set; VMEC and Boozer use the bundled QA equilibrium.

Field loading/construction, imports and particle setup are excluded. Lowering
includes tracing and solver graph preparation; backend compilation is measured
by `lowered.compile()`. The first execution is timed separately and synchronized
with `jax.block_until_ready`. Cold total means lowering plus backend compilation,
without execution. These small-population measurements do not establish
production execution speed, population scaling or physical accuracy.

| Field / model | Execution | Lowering s | Backend compile s | Cold total s | First execution s |
|---|---|---:|---:|---:|---:|
| Coils / GC | Shared JAX | 2.91 | 8.79 | 11.70 | 0.039 |
| Coils / GC | Specialized GPU | 5.14 | 65.32 | 70.46 | 0.093 |
| Coils / Lorentz | Shared JAX | 2.82 | 8.81 | 11.63 | 0.028 |
| Coils / Lorentz | Specialized GPU | 6.97 | 346.09 | 353.06 | 0.108 |
| VMEC / GC | Shared JAX | 9.25 | 13.63 | 22.88 | 0.041 |
| VMEC / GC | Specialized GPU | 15.89 | 260.03 | 275.92 | 0.041 |
| Boozer / GC | Shared JAX | 3.22 | 10.26 | 13.47 | 0.025 |
| Boozer / GC | Specialized GPU | 5.07 | 66.88 | 71.95 | 0.039 |

These baseline measurements locate most of the specialized implementation's
extra cost in backend compilation. Subsequent compiler simplifications are
compared against these same native workloads. No Diffrax comparison is included.

## Exact shared field Jacobians

The predictor's private adapter now shares one field-value/Jacobian evaluation
across its Lorentz derivative directions. The custom JVP uses the existing exact
native coil Jacobian. These
are changes to code generation within the solver adapter; the selected moment
predictor and orbit/collision maps are unchanged. No public field classes change.

| Workload | Original cold total s | Shared-Jacobian cold total s | Reduction |
|---|---:|---:|---:|
| Coils / Lorentz, predictor enabled | 353.06 | 126.27 | 64.2% |

Lorentz backend compilation decreased from 346.09 to 119.67 seconds. Saved
states and step counts were identical in the matched check; compiled executable
buffers stayed at 7,056 bytes. This tiny-population check establishes equivalence
for these inputs, not a general production runtime benchmark.

A fresh confirmation with the corrected cold-cache policy below measured
5.69 seconds lowering and 99.27 seconds backend compilation (104.96 seconds
total). The helper code was unchanged from the earlier 126.27-second run;
the difference between those runs is not an additional optimization claim.
Warm median was 6.047 ms versus the historical 5.964 ms, with only three/one
samples respectively. Saved states, terminal states and all counters were
bitwise identical; executable buffers remained 7,056 bytes. The process-wide
JAX allocator peak was 143,360 bytes, including setup, compilation and execution.

As a diagnostic only, the same native Lorentz workload with conservative
collision control and no moment predictor took 2.81 seconds lowering and 5.07
seconds backend compilation (7.88 seconds total). The production default remains
the moment predictor. This comparison isolates the additional compiler burden;
it is not a recommendation to disable the predictor.

## GC symmetric covariance reduction

The GC predictor now evaluates the maximum over the unique entries of its
symmetric projected covariance, rather than both triangles. Every unique
covariance constraint remains present. This reduces the isolated scalar budget
graph from 908 to 712 expressions without changing coefficients, field queries,
the collision map or the selected controller.

The following fresh measurements use the corrected cold-cache policy described
below. Their full-matrix repeat is the appropriate baseline for this candidate;
the historical 70.46-second measurement above is retained separately.

| GC / direct coils | Lowering s | Backend compile s | Cold total s | Compiled buffers bytes | Allocator peak bytes |
|---|---:|---:|---:|---:|---:|
| Full covariance reduction | 5.25 | 73.79 | 79.04 | 135,544 | 83,988,736 |
| Unique symmetric entries | 5.26 | 66.35 | 71.61 | 135,544 | 83,988,736 |

This pair gives a 9.4% cold-total reduction. A separate warm comparison alternated
30 calls per version in one process: median 12.710 versus 12.683 ms, with
interquartile ranges 12.634–12.792 and 12.633–12.766 ms, respectively. All returned
arrays were bitwise identical. These two-particle checks support unchanged warm
cost for this workload; they do not establish population scaling. The allocator
peak includes setup, compilation and execution, and is not a per-particle cost.

The expanded budget is checked against the independent dense matrix calculation,
including mixed covariances, nonzero velocity curvature, pitch endpoints and
zero/tiny noise. This extends an existing test; the native split suite still has
19 test functions.

After the earlier covariance-symmetry change, `pytest tests/test_native_split.py -q` passed all
19 tests in 164.19 seconds on the local CPU. Actual GPU checks above validate
the coil kernels separately; the CPU suite does not measure GPU performance.

## Cold-cache validation

In the installed JAX version, changing the cache-enable setting after setup does
not invalidate its memoized cache-use decision. An attempted full-matrix repeat
loaded an executable in 0.040 seconds despite the setting reporting disabled;
that cached result is excluded from the cold comparisons. The benchmark now
calls the public experimental `compilation_cache.reset_cache()` after disabling
the cache and before solver lowering. Setup can still use the persistent cache.
No disk caches are deleted. Distinct Triton cache directories remain required.

## Further retained coil reductions

These follow-up measurements use the same GPU, two-particle inputs, controls,
spatial-diffusion settings and corrected cold-cache policy. The before column
uses the preceding retained implementations, rather than the original baseline.

| Direct coil field / model | Before cold total s | After this stage s | Additional reduction | Compiled buffers bytes | Allocator peak bytes |
|---|---:|---:|---:|---:|---:|
| GC | 71.61 | 66.54 | 7.1% | 135,544, unchanged | 83,988,736, unchanged |
| Lorentz | 104.96 | 83.10 | 20.8% | 7,056, unchanged | 143,360, unchanged |

At this stage GC lowering/backend compilation took 4.81/61.73 seconds; Lorentz took
5.01/78.08 seconds. These remain single cold measurements, not timing averages.
Paired warm checks alternated 30 calls per implementation in one process:

| Model | Before median ms (interquartile range) | After median ms (interquartile range) |
|---|---:|---:|
| GC | 12.677 (12.593–12.806) | 12.652 (12.561–12.740) |
| Lorentz | 4.885 (4.828–5.069) | 4.921 (4.835–5.099) |

The warm checks show no meaningful runtime change for this workload; they are
not evidence of a production throughput gain. Saved states, terminal states,
times, statuses and counters were bitwise identical in both paired checks.

GC now reuses reciprocal normalization scales, reducing scalar divisions from
35 to 11; its scalar expression count changes from 712 to 718. Lorentz evaluates
the Ito covariance in blocks and reduces its symmetric covariance maximum over
unique entries, reducing the scalar budget graph from 5,073 to 3,966 expressions.
The block covariance agrees with the original dense product in value, Jacobian
and Hessian, including unconstrained field-direction derivatives, pitch endpoints
and zero diffusion. The predictor, numerical scheme and public field classes
are unchanged. No finite-difference derivatives or fitted calibration enter.

Reciprocal normalization was also tested for Lorentz: 84.84 seconds cold versus
83.10 seconds for covariance blocks alone. It was reverted. A GC audit that
differentiated physical variances directly passed both Coulomb models with
smooth varying profiles, low speeds and vacuum, but removed only 58 scalar
expressions; that alternative was not added to production.

The audit also exposed a pre-existing interpolation-lowering incompatibility
with JAX 0.11 scan metadata. The private adapter now accepts both metadata forms
and stacks scalar tracers without converting them to host values. Forward and
reverse scans, interpolation and its forward derivative are covered within the
existing scalar-lowering test; the suite still contains 19 test functions.

After all the retained follow-up changes, `pytest tests/test_native_split.py -q`
passed all 19 tests in 174.35 seconds on the local CPU. Source whitespace checks
passed, and the orbit/collision stepper file matched its pre-optimization hash.

The follow-up raw results are under `output/compile_optimization_20261027/`
in `gc_inverse_scale`, `lorentz_covariance_blocks`, and the two
`{gc,lorentz}_algebra_warm_pair.json` reports. Generated raw files remain ignored.

## Exact Lorentz velocity Jacobians

The predictor now uses exact formulas for the partial velocity Jacobians of its
collision drift and orbit flow. These replace two small forward-AD calculations;
they do not change the tracing RHS, collision update, estimator or field classes.
The formulas retain field/profile dependence and remain differentiable in their
inputs. Independent AD comparisons cover values and higher velocity derivatives,
nonzero electric fields, pitch endpoints, zero rates and both orbit settings.
The scalar Lorentz budget graph decreases from 3,966 to 3,711 expressions.

| Latest retained coil implementation | Previous cold total s | Retained cold total s | Additional reduction |
|---|---:|---:|---:|
| GC | 66.54 | 66.54 (unchanged; preceding measurement) | None |
| Lorentz | 83.10 | 78.00 | 6.1% |

The new Lorentz cold run measured 5.50 seconds lowering and 72.50 seconds backend
compilation. Compiled buffers remained 7,056 bytes and the fresh-process JAX
allocator peak remained 143,360 bytes. The inputs and controls match the earlier
two-particle adaptive Dopri8/direct-coil workload; this is one cold measurement.

Thirty alternating blocked warm calls gave medians 4.867 ms before and 4.782 ms
after, with interquartile ranges 4.764–5.032 and 4.738–4.976 ms. No separate warm
throughput improvement is claimed. Every returned array, including saved and
terminal states, times, statuses and counters, was bitwise identical.

A shared reciprocal for the GC field magnitude and velocity-jet denominator
reduced the isolated division count from 32 to 3 and passed independent-reference
checks. Its measured cold total was 68.91 seconds versus 66.54 seconds, so it was
reverted. A scalar-identity pruning audit removed only 12/15 expressions from
the GC/Lorentz budgets and was not added to production.

The final suite passed all 19 tests in 122.13 seconds on CPU, while untimed GPU
executable preparation ran separately. GPU warm timing waited until CPU tests
finished. Results are in `lorentz_analytic_velocity` and
`lorentz_jacobian_warm_pair.json` beneath the same ignored output directory;
the discarded GC trial is `gc_shared_velocity_denominators`.

## Three additional passes: no further retained change

Three additional investigations kept the same predictor, collision update,
orbit method and field classes. None passed both the performance and correctness
requirements, so the retained cold totals remain **66.54 s for GC** and
**78.00 s for Lorentz**. Those are the preceding measurements, not new timings.

1. **Covariance contractions.** Factoring GC noise amplitudes increased the
   scalar budget graph from 718 to 728 expressions. A direct Lorentz covariance
   contraction reduced 3,711 expressions to 3,637, but its measured cold total
   with the experimental two-warp layout was 71.05 s versus 71.10 s for that
   layout alone. This does not establish a useful gain, and neither variant was
   retained.
2. **Shared coefficient derivatives.** Sharing primal values through ordinary
   nested JVPs left the scalar coefficient graph sizes unchanged. Low-speed
   derivative comparisons failed the existing strict reference tolerances for
   both GC and Lorentz, with both Coulomb-log models. These variants were
   rejected; no derivative tolerance was relaxed. Grouping source locations also
   reduced generated IR size by less than one percent and was not retained.
3. **GPU layout.** Two warps instead of one gave the following preliminary
   two-particle cold timings, with unchanged compiled buffers and allocator
   peaks. A subsequent 64-particle check invalidated this candidate.

| Direct coil model | Validated one-warp cold total s | Experimental two-warp cold total s | Decision |
|---|---:|---:|---|
| GC | 66.54 | 53.86 | Rejected: GPU correctness failure |
| Lorentz | 78.00 | 71.10 | Not retained: proposed layout failed GC validation |

The initial two-particle outputs agreed to roundoff and did not expose the GPU
layout defect. At 64 particles, repeated calls to the same two-warp GC executable
produced different outputs in both fixed and adaptive modes. The one-warp
executable was reproducible in the same checks. Separately, 16 consecutive
production Brownian draws for each of 64 keys agreed bitwise between layouts,
for noise dimensions two, three and five. Thus an adaptive Brownian-path change
does not explain the fixed-step failure. The underlying layout/code-generation
mechanism has not been isolated. No two-warp speedup is considered validated;
the launch change was completely reverted, and a larger-population Lorentz
comparison was not pursued.

No production source or tests remain changed by these three passes. The source
files were compared byte-for-byte with their preceding validated snapshots.
The suite still has 19 test functions; all 19 passed in 114.65 s during this
investigation. The restored source also matches the earlier validated version
whose full suite passed in 122.13 s. Diagnostic outputs remain ignored beneath
`output/compile_optimization_20261027/three_pass_*`; the experiment scripts are
workspace scratch files, not added production dependencies.

## Discarded alternatives

The following experiments are not production dependencies:

* Simplifying the expanded predictor again, sharing immutable table reads, and
  exposing literal padding zeros did not yield a shorter cold compile. These
  experiments were stopped without completed timings and were reverted; their
  partial runs are not presented as measured compile durations.
* An exact custom JVP for the six Lorentz collision moments passed first- and
  second-derivative comparisons against ordinary AD. After simplification, the
  complete local Lorentz budget had 5,183 scalar expressions instead of 5,073.
  This rule was discarded before another expensive GPU compilation.
* A joint Taylor-mode evaluation of radial coefficient derivatives produced
  2,454 scalar expressions instead of 1,255 in the NRL coefficient audit and
  failed the stringent low-speed derivative comparison. It was not added to
  production.
* Reducing the compiler's software-pipeline depth from three stages to one did
  not produce a shorter compile before the experiment was stopped. The default
  compiler setting was restored.
* Sharing the VMEC position/Jacobian in the same way exceeded the original
  compilation cost before completing. That variant was reverted; the existing
  analytic VMEC position derivative remains in use.
* Reusing the Cartesian GC predictor's magnitude from its full geometry query
  increased cold total from 70.46 to 91.02 seconds (backend: 65.32 to 82.94
  seconds). That variant was also reverted. GC/coils and Boozer retain their
  original field adapters; the GC improvement above changes covariance algebra.
* Four-warps GC launch layouts, with three/one software-pipeline stages, took
  77.15/74.07 seconds cold, versus the original 70.46 seconds. Two-particle warm
  calls took 6.37/6.95 ms versus 12.69 ms, but larger-population runtime was not
  validated. These launch settings are not retained for the compilation target.

An additional JAX compiler trace of unchanged GC code recorded about 77.9 seconds
in backend compilation, with 0.59 seconds in ordinary HLO optimization. The long
`CompileModuleToLlvmIr` span includes kernel code generation and nested assembly
work; the largest nested `PTX->CUBIN` event was 16.69 seconds. These spans overlap
and must not be summed. This is a diagnostic capture, not a new unprofiled timing
baseline. The remaining GC kernel-generation cost has not been eliminated.

A custom VJP does not supply the forward-mode and second derivatives used by
this predictor. Custom JVPs can provide those derivatives, but the tested rule
shows that replacing AD alone does not guarantee a smaller generated kernel.

## Corrected coordinate spatial kernels, 28 October

These follow-up adapter checks include Cartesian spatial kicks transformed
nonlinearly into VMEC and Boozer charts. They use 64 protons at 5 keV,
adaptive Dopri8, moment budget .01, native collision tolerance .001,
orbit rtol=atol=1e-9, three saves and a .4 ns horizon. Both implementations
execute on the local GPU. Setup is excluded from the timer; lowering and
backend compilation are included, execution is excluded. Persistent caching
is allowed and no cold-cache claim is made.

| Field / model | Shared native JAX s | Independent GPU s |
|---|---:|---:|
| Coils / GC | 12.9327 | 63.5113 |
| Coils / Lorentz | 8.4850 | 81.4998 |
| VMEC / GC | 9.4936 | 369.9863 |
| Boozer / GC | 12.2298 | 209.4124 |

All four pairs passed shared/GPU state agreement and exact repeatability of
repeated GPU calls at 64 particles. The new coordinate-spatial VMEC kernel
uses one warp: its previous four-warp launch failed the larger check despite
passing the initial two-particle check. The collision map and predictor were
unchanged. This is a layout correction, not a fallback to shared execution.
Boozer geometry table reads remain vectorized in its private solver adapter.

These are not before/after matches to the earlier cold alpha measurements:
population, species, controls and coordinate spatial physics differ. In
particular, the VMEC and Boozer compilation costs remain large. These adapter
checks also do not establish statistical physics accuracy in the QA equilibrium.

Reproduce with `benchmark_split_coordinate_physics --study adapters --particles
64 --clocks adaptive --tolerance .01 --output output/equilibrium_adapters`.
The optional `--adapter-fields` selects subsets. The source exports Boozer
spatial geometry from the same transform as its magnetic tables.

## Reproduction

From the repository root, start a fresh Python process for each row. For example:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
TRITON_CACHE_DIR=output/compile_gc_independent/triton_cache \
python -m examples.benchmarks.benchmark_split_orbits \
  --field coils --model gc --case native_independent --stepping adaptive \
  --particles 2 --horizon 1e-9 --saves 3 --steps 2 --repeats 1 \
  --max-steps 32 --max-orbit-steps 64 --rtol 1e-10 --atol 1e-10 \
  --collision-tolerance .001 --moment-tolerance .005 --cold-compile \
  --output output/compile_gc_independent
```

Repeat with `--case native_shared` and a fresh output/cache directory for the
shared implementation. Use `--model lorentz` for coil full orbits. For VMEC or
Boozer use `--field vmec` or `--field boozer`, `--model gc` and
`--no-spatial-diffusion`. Boozer accepts `--field-cache PATH/boozer_inputs.npz`
from the native Boozer alpha example; field preparation is excluded either way.

Raw reports and logs are generated under `output/compilation_comparison_20261027/`
and `output/compile_optimization_20261027/`;
they are deliberately not versioned.
