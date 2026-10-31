# Native split benchmark reproduction

Thermalization, first passage and the coordinate moment studies were rerun on
2026-10-28 with 2,000 particles per configuration. Older multirate/convergence
figures remain labelled separately. The paired convergence figure was also
regenerated with 64 common-Brownian paths at .01 and .005. These physics studies are not new performance
measurements of the specialized GPU integration. Numerical reports and samples
are generated under `output/`; no prior JSON/CSV/SVG/PPTX files are required in
the repository. The scripts select the machine's normal JAX backend. Population
and independent PDE reference studies are deliberately separate from CI.

The split branch is based on `feat/native_dopri` (commit `3d20e36`). Its CI
suite adds 19 test functions, using small deterministic problems and one/two
particles rather than population studies or reference-mesh sweeps. Public field
classes are unchanged; geometry conversion stays in the split solver modules.
Only scripts, Markdown and selected README PNGs belong in this addition.
Generated TMP/SVG/PPTX/CSV/JSON files stay under ignored `output/`.

### Presentation refresh

The primary uniform/helix views use 2,000 particles, budget .01 and
--save-points 201 for actual time resolution. Production and independent
references are evaluated at matching times. --charts cartesian selects one
curve per physical model. Black curves retain the distinct GC and full-Lorentz
references; grey/blue bands retain their separate sampling uncertainties.
Panels show means and standard deviations; variance confidence intervals are
transformed by the square root with their lower endpoint clipped at zero.
The six-save diagnostics and their numerical tables below remain historical
results, rather than statistics of the new save grid.

Thermalization and first-stopping reports can be rendered again with
--plot-only, without tracing. PNG export is 320 dpi, with a PDF companion.
Thermalization histograms retain the independent finite-time speed and pitch
theory, with blue birth-value markers. First stopping
plots only the tight adaptive histogram and black reference. Data are not
pooled across methods or models.
Smooth reference PDFs are scaled by particle count times bin width for the
count histograms. The Lorentz moments are instantaneous samples, not gyro-
averages: the 201-point output grid does not resolve every fast gyroperiod
in the helix, even though the adaptive orbit integration resolves gyromotion.

After generating the reports, regenerate only the figures with:

~~~sh
python -m examples.benchmarks.benchmark_split_coordinate_physics --plot-only --study diffusion --charts cartesian --output output/uniform
python -m examples.benchmarks.benchmark_split_coordinate_physics --plot-only --study coupled --charts cartesian --output output/coupled_moments
python -m examples.benchmarks.validate_beam_relaxation --plot-only --output output/thermalization.json
python -m examples.benchmarks.validate_production_tolerance --plot-only --output output/first_passage.json
~~~

## Physics studies

Run from the repository root. These commands launch population studies; they
are provided for later reproduction and are not run by installing the package.

```sh
python -m examples.benchmarks.validate_beam_relaxation \
  --particles 2000 --batch-size 2000 --coefficient-model impact \
  --methods conservative --settings fine --output output/split_thermalization.json
python -m examples.benchmarks.validate_production_tolerance \
  --particles 2000 --batch-size 2000 --output output/split_first_passage.json
python -m examples.benchmarks.benchmark_split_coordinate_physics \
  --study diffusion --particles 2000 --tolerance .01 --output output/coordinate_diffusion
python -m examples.benchmarks.benchmark_split_coordinate_physics \
  --study coupled --models gc lorentz --particles 2000 --tolerance .01 --output output/coordinate_moments
python -m examples.benchmarks.benchmark_split_multirate \
  --particles 2000 --models gc lorentz --output output/split_multirate
```

Thermalization and first passage are collision-operator checks in a fixed bath.
The multirate script uses production `trace_split` and measures speed/pitch and
spatial observables from the same trajectories. It includes GC spatial diffusion
and finite-gyrofrequency Lorentz motion. Its original fixed/conservative controls
remain explicit, so changing the public default does not relabel old results.

The new coordinate script evolves the same prescribed mirror/helix equations
in Cartesian, VMEC regular and Boozer-style regular charts. The latter include
an angle gauge and toroidal shift, respectively. They are manufactured charts,
not VMEC equilibria or Boozer K=0 magnetic spectra. All three use the production
collision maps, moment predictor, retained Brownian driver and nonlinear
Cartesian spatial-kick inversion. The population reference path uses shared
JAX execution. The mirror/helix axial reference generators omit perpendicular
spatial kicks, while the particles include them; the black curves are therefore
strong-magnetization references. The uniform-field diffusion reference includes
the tested spatial operator. Its Lorentz reference retains finite gyrofrequency.

Actual QA field-table checks, separate from those independent physics references:

```sh
python -m examples.benchmarks.benchmark_split_coordinate_physics \
  --study adapters --particles 64 --clocks fixed adaptive --tolerance .01 \
  --output output/equilibrium_adapters
```

This checks coils/GC, coils/Lorentz, VMEC/GC and Boozer/GC against shared JAX,
including GPU repeatability when a GPU is available. These checks establish
adapter agreement, not independent statistical accuracy in the QA equilibrium.
Boozer geometry is generated from the same transform as its magnetic tables;
`--mboz` and `--nboz` default to 4 for this small adapter check. VMEC uses its
existing geometric tables. Missing Boozer spatial geometry raises explicitly.

Black lines show ensemble expectations from independent references. Sampling
intervals, particle variances and paired numerical-refinement differences are
different quantities. A smaller predictor tolerance does not force an individual
finite-population curve closer to its ensemble expectation at every time.
Reference-mesh refinement and paired Brownian-path refinement must be assessed
separately from those sampling intervals.

For the recorded .01 run, the following final variance gaps are relative to the
independent axial reference. Both numbers are percentages of the reference
variance: the first is the sample gap and the second is its marginal 95%
sampling half-width. Neither number is the controller's estimated error.

| Field / chart | Position variance gap % | Pitch variance gap % | Parallel-velocity variance gap % |
|---|---:|---:|---:|
| Mirror / Cartesian | -6.73 +/- 5.78 | -9.66 +/- 5.88 | -10.51 +/- 5.66 |
| Mirror / VMEC chart | -6.73 +/- 5.78 | -9.65 +/- 5.88 | -10.51 +/- 5.66 |
| Mirror / Boozer chart | -6.73 +/- 5.78 | -9.66 +/- 5.88 | -10.51 +/- 5.66 |
| Helix / Cartesian | +0.71 +/- 6.00 | +3.95 +/- 6.32 | +0.64 +/- 6.05 |
| Helix / VMEC chart | +0.71 +/- 6.00 | +4.76 +/- 6.37 | +1.38 +/- 6.10 |
| Helix / Boozer chart | +0.68 +/- 5.99 | +3.91 +/- 6.38 | +1.10 +/- 6.14 |

Reference-refinement indicators for these variances reach 0.23% in the mirror
and 0.006% in the helix; they are not included in the sampling half-widths above.
The mirror offsets exceed marginal sampling intervals in this seed. Agreement
between charts does not resolve that physical-reference discrepancy, and this
run is not a general validation of a .01 budget. Independent-seed and numerical
refinement checks are separate from increasing the interval width by combining
many comparisons.

An independent mirror ensemble (seed 8707, same .01 budget and .0025 cap)
gave final variance gaps of `+1.85 +/- 6.17%` in position,
`-0.08 +/- 6.30%` in pitch, and `-1.89 +/- 6.06%` in parallel velocity.
Its speed variance was `-7.53 +/- 5.53%`. The changing offsets illustrate why a
single population/reference gap cannot identify numerical bias. These two
ensembles are reported separately; the primary figure has not been replaced
by a more favourable seed. Reproduce with `--fields mirror --charts cartesian
--seed 8707` and the same budget.

Uniform-field perpendicular spreading at `Omega/nu0=2048` gave
`-0.88 +/- 4.48%` for GC in each chart and `+0.41 +/- 4.62%` for Lorentz.
The same trajectories gave speed-variance gaps `+9.19 +/- 7.23%` (GC) and
`-11.32 +/- 5.46%` (Lorentz). These velocity offsets require further numerical
and ensemble convergence; the recorded .01 run is not an all-statistics pass.

The uniform figure runs to `t*nu0=0.1`; the mirror/helix figures end at 0.025.
The following additional ensembles use 2,000 particles, moment budget
.01 and cap .0025. Each entry is the final speed-variance gap plus/minus its
marginal 95% sampling half-width, in percent of the independent reference.

| Duration t*nu0 | Seed | GC gap % | Lorentz gap % |
|---|---:|---:|---:|
| 0.1, primary figure | 7706 | +9.19 +/- 7.23 | -11.32 +/- 5.46 |
| 0.1, independent ensemble | 8706 | -1.00 +/- 6.30 | -2.50 +/- 6.14 |
| 0.025, matched mirror duration | 8706 | -2.36 +/- 5.96 | +0.19 +/- 6.11 |

Changing only the seed at the original duration changes both offsets
substantially. These results do not support a claim that uniform-field speed
physics is intrinsically less accurate, but they do not separate all numerical
bias from sampling either. The primary figure has not been replaced. Reproduce
the extra ensembles with:

```sh
python -m examples.benchmarks.benchmark_split_coordinate_physics \
  --study diffusion --models gc lorentz --charts cartesian --particles 2000 \
  --tolerance .01 --seed 8706 --horizon .1 --output output/uniform_seed8706
python -m examples.benchmarks.benchmark_split_coordinate_physics \
  --study diffusion --models gc lorentz --charts cartesian --particles 2000 \
  --tolerance .01 --seed 8706 --horizon .025 --output output/uniform_matched_duration
```

Additional uniform Lorentz ensembles at the original duration gave speed
variance gaps `+2.01 +/- 6.34%` (seed 7707), `-3.47 +/- 6.15%` (7708),
`-1.63 +/- 6.15%` (7709), and `+4.59 +/- 6.18%` (7710). Each uses 2,000
particles and the unchanged .01 budget/.0025 cap; reproduce by changing
`--seed` in the uniform command above. All four marginal intervals include
the reference; the original seed 7706 remains recorded as an outlier.

A separate diagnostic retained the original 2,000-particle Brownian increments
and globally continued the production Lorentz map through conditional
subintervals. Subdivision factors 2, 4 and 8 changed the final speed variance
by `+0.019 +/- 0.082`, `+0.040 +/- 0.096` and `+0.049 +/- 0.108` percentage
points of the theoretical variance (coarse minus refined, paired 95% sampling
intervals). The coarse ensemble exactly reproduced the original variance;
the eightfold-refined ensemble remained 11.37% below theory. Doubling the
radial reference mesh from 768 to 1,536 cells changed its final variance by
0.014%. These checks support ensemble fluctuation over collision-step bias
as the explanation of this particular deficit; they do not certify every
phase-space statistic or eliminate all possible systematic errors.

The coupled script also supports Cartesian Lorentz mirror/helix runs through
`--models lorentz`. Velocities are aligned to give birth pitch 0.6 in each
local field, with the leading-order gyrocentre at the GC birth point. It keeps
the GC benchmark's gyrofrequency ratios: 2048 in the mirror and 2,000,000 in
the helix. Resolving the latter's gyromotion requires many orbit substeps per
collision interval. The speed reference is independent for both models, since
the field does no work in this homogeneous bath. Position, pitch and parallel
moments now use a separately implemented full-Lorentz numerical reference in
both the mirror and helix. The earlier GC-limit curves have been replaced.

The independent reference uses centred exact-angle magnetic rotations and Cartesian
Euler--Maruyama updates of the independently evaluated proton Coulomb Ito
generator. Its Gaussian noise is carried in a transported orthogonal frame;
this changes neither the physical drift nor diffusion tensor and makes its
paired refinements less sensitive to fast gyrophase differences. Production
RK, collision maps, predictor and Brownian memory are unchanged. Reference
controls halve the orbit phase limit, then halve the collision interval on
the same reference Brownian path. Defaults are 2,000 independent particles,
400 coarse collision steps per save, five save intervals of .005, and phase
limits .1/.05 radians. The finest collision interval is 6.25e-6/nu0.

```sh
python -m examples.benchmarks.lorentz_collision_reference \
  --particles 2000 --steps 400 --phase .1 --output output/lorentz_reference.json
python -m examples.benchmarks.benchmark_split_coordinate_physics \
  --study coupled --models lorentz --particles 2000 --tolerance .01 \
  --lorentz-reference output/lorentz_reference.json --output output/lorentz_moments
```

Use `--plot-only --lorentz-reference PATH` on an existing coupled report to
correct its reference and regenerate figures without retracing particles.
Old Lorentz reports cannot silently plot GC curves as Lorentz references.
Grey bands show reference sampling uncertainty, blue bands show production
sampling uncertainty. Gaps between independent sample means use the combined
standard error; reference step-refinement uncertainty remains separate. These
are model-matched numerical ensemble references, not analytic solutions.

The 2,000-particle Lorentz runs completed without solver failures. Final speed
variance gaps are `-0.28 +/- 6.09%` in the mirror and `-3.00 +/- 6.25%` in the
helix. Mean accepted orbit steps per collision are 1.006 and 256.77, respectively;
the gyrofrequency parameters were retained. These production trajectories
are unchanged by the corrected reference calculation.

Final production-minus-reference gaps, normalized by the absolute reference
value, are below. Intervals combine independent production and reference
sampling standard errors; deterministic radial theory is used for speed.

| Field | Mean z gap % | Mean pitch gap % | Var z gap % | Var pitch gap % |
|---|---:|---:|---:|---:|
| Mirror | +0.44 +/- 0.82 | +0.16 +/- 1.06 | +1.64 +/- 8.82 | -1.51 +/- 9.08 |
| Helix | -0.11 +/- 0.33 | -0.13 +/- 0.52 | -7.90 +/- 8.28 | -2.67 +/- 8.48 |

The helix reference's final paired orbit-refinement changes are
`-0.00103 +/- 0.00011%` for mean z and `-0.00489 +/- 0.00413%` for Var z.
Collision-interval refinement gives `+0.0107 +/- 0.0234%` and
`-0.2263 +/- 0.8591%`, respectively. These reference changes are substantially
smaller than the production/reference sampling intervals, but are not global
error bounds. For the mirror, collision refinement changes Var z by
`-0.0279 +/- 0.0048%`: this small nonzero numerical difference remains recorded.
The final variance measurements agree within their combined intervals, whose
widths are too large to certify 1% variance accuracy. Several earlier helix
variance/covariance gaps remain outside pointwise 95% intervals: Var z gaps
are `-9.00 +/- 8.21%`, `-9.91 +/- 8.13%`, `-10.91 +/- 8.06%` and
`-10.17 +/- 8.10%` at t*nu0=.005/.01/.015/.02. These are unresolved physical
statistics discrepancies; the corrected references do not make the entire
benchmark an all-moments pass. Finite-sample agreement is not a general
guarantee for the .01 controller budget.

The reference magnetic rotation uses the exact angle |Omega|*dt. The earlier
Cayley-angle trial had a gyrophase error visible in the helix pitch transient,
so it is not used for the retained figures. Collisionless exact-angle controls
converge to a tight full-Lorentz Dopri8 solve: at phase limit .05 the helix
relative z error is 3.26e-6 and the absolute pitch error is 6.70e-6. Both errors
fall by approximately four on halving that phase limit. This checks the orbit
reference separately from its Monte Carlo sampling and collision resolution.

The old Lorentz helix mean axial position was 1.94% +/- 0.22% below the GC-limit
curve. A collisionless control with the same field and birth convention
already differs by 1.76% from the exact GC orbit. Tightening Lorentz orbit
tolerances from 1e-10 to 1e-12 changes its final axial position by less than
2e-11 in relative terms; the speed error decreases from 1.44e-7 to 8.67e-10.
Thus most of this spatial gap precedes collisions and cannot be attributed
to the collision controller. Here k*rho at birth is approximately .0253;
the GC and Lorentz states are matched using only a leading-order gyrocentre
conversion. The test does not separately resolve finite-gyro model effects
from birth-coordinate conversion. This GC/Lorentz diagnostic is separate
from comparison with the new matched numerical reference.

## Retained predictor and controller experiments

Production uses the original orbit-then-collision map and retained Brownian
rejection memory. The selected moment predictor adds a state-dependent coupled
mean/covariance interval ceiling. Native collision and embedded RK checks remain
active. The default moment budget is `0.005`, collision tolerance is `0.001`,
and orbit tolerances are separate. These numbers are not interchangeable error
bounds and do not promise a global percentage error in physical statistics.

Earlier comparisons included conservative coefficient control, paper Wiener
look-ahead, shared-clock controls, full-step/two-half-step combined control,
local trajectory-RMS and coupling estimates, accumulated-history budgets,
scheduled tolerances and local quadrature probes. Combined control remains an
expensive convergence comparator. The retained moment predictor avoids the
comparison trajectory. Discarded history/scheduling/quadrature variants are
not production defaults or dependencies of the GPU kernel. No change to Strang
splitting, fitted calibration or finite-difference derivatives is included.

Generate other moment-budget checks without a saved numerical archive:

```sh
python -m examples.benchmarks.benchmark_split_coordinate_physics \
  --particles 2000 --tolerance .005 --output output/moment_005
```

The script regenerates its independent references by default. `--reference`
optionally reuses an ignored reference file from a previous invocation. `--cap`
and `--seed` provide explicit convergence and independent-ensemble controls;
the documented .01 run retains cap .0025. Paired numerical refinement is
provided separately and also regenerates its independent normalization reference:

```sh
python -m examples.benchmarks.benchmark_split_path_convergence \
  --control moments --particles 64 --tolerances .01 .005 \
  --refinements 8 16 32 --max-steps 2000 --max-roots 4096 \
  --output output/paired_convergence
```

`--reference` optionally supplies the reference generated by the coordinate
script. This check uses the axial model with spatial kicks disabled and retains
the same Brownian realization within each coarse/refined pair. It does not
regenerate the population confidence bands. Its .005 cases roughly double
collision updates compared with .01 and reduce position and speed path errors
in the recorded run; the physical sampling offsets above are a separate result.

## Timing and memory

See the [cold compilation comparison](compilation_20261027.md) for separately
measured lowering, backend compilation and first execution on the local GPU.

`examples.benchmarks.benchmark_split_orbits` separates compilation from blocked
warm tracing, records compiler buffers and allocator peaks, and supports the
shared and specialized native implementations. Use separate fresh processes
for allocator-peak comparisons:

```sh
python -m examples.benchmarks.benchmark_split_orbits --field coils --model gc \
  --case native_independent --stepping adaptive --particles 64 \
  --output output/split_gpu_gc
python -m examples.benchmarks.benchmark_split_orbits --field coils --model gc \
  --case native_shared --stepping adaptive --particles 64 \
  --output output/split_shared_gc
```

Use `--model lorentz` for full orbit or `--stepping fixed` for fixed clocks.
The legacy Diffrax GC reference omits spatial diffusion; use the explicit
`--no-spatial-diffusion` diagnostic for a matched comparison to that operator.
Coordinate-field GC diffusion uses the nonlinear Cartesian kick inversion.
Boozer callers supply `BoozerGeometry.from_booz_xform(transform)` through
`spatial_geometry`; the magnetic spectrum alone contains insufficient geometry.
