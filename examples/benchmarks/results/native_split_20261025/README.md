# Native forward split tracing

This directory retains compact evidence for the forward split addition on top
of the native-orbit branch. Full per-particle histories and exploratory HLO,
controller and interpolation experiments are excluded. Original measurements
remain identifiable by their source hashes and original-file hashes. No AD
benchmark or reverse rule belongs to this PR.

Impact-parameter Coulomb helpers now live in `essos/background_species.py`.
Archived source hashes retain the earlier `essos/coulomb.py` filename because
those measurements predate this move; the helper formulas are unchanged.

## Fixed forward timing and memory

64 alpha particles, 4 keV, seed 42, eight saves, RTX 5060 Laptop GPU,
JAX/JAXlib 0.11.2. Each case has a fresh process, compilation, two warm-ups and
three device-blocked warm samples; setup is excluded from execution timings.
GC duration is 20 us with 224 intervals; Lorentz duration is 2 us with 1792.
Fixed orbit and collision intervals have the same dt in this timing control.
Coils use bundled unscaled QA coils; VMEC/Boozer use bundled reactor-scale QA.
Compare solvers within each row, rather than comparing different field workloads.

| Field / model | Native collisionless ms | Native split ms | Existing fixed reference ms |
|---|---:|---:|---:|
| coils / gc | 420.381 | 581.817 | 1212.780 |
| coils / lorentz | 781.367 | 1211.415 | 5365.934 |
| vmec / gc | 204.456 | 140.117 | 434.764 |
| boozer / gc | 24.907 | 40.209 | 57.870 |

References are the existing Diffrax StratonovichMilstein GC and SPaRK Lorentz
maps, with constant SDE steps. Boozer has no equivalent Diffrax collisional
entry point: its separately labelled reference is legacy fixed Dopri8/Euler
collisions with thermal stopping disabled. These stochastic schemes use
different noise constructions. Throughput comparisons do not establish equal
weak error, equal sample paths or equal integration order.

| Field / model | Native / reference compiler buffers MiB | Native / reference allocator peak MiB |
|---|---:|---:|
| coils / gc | 0.668 / 16.106 | 0.772 / 160.070 |
| coils / lorentz | 0.066 / 0.737 | 0.655 / 288.098 |
| vmec / gc | 0.089 / 1.065 | 1.224 / 129.093 |
| boozer / gc | 0.067 / 0.104 | 0.310 / 64.041 |

Peak means the fresh-process JAX allocator high-water mark across setup,
compilation and warm solves. It does not include every CUDA allocation or
isolate a warm solve's transient peak. The larger reference peaks can include
compilation workspace. Compiler buffers mean arguments + outputs + temporaries
minus aliases, excluding resident field data/executable constants. Reserved
pool is recorded separately. Counts, compilation and all warm samples are in
[`forward.json`](forward.json); [`forward.csv`](forward.csv) is the flat table.

Native split GC evolves physical mu, while collisionless GC reconstructs mu
from a fixed energy. The drivers and compiled graphs differ. VMEC's split time
being below its collisionless time is therefore not a negative collision cost.

| Field / model | Collisionless compile s | Native split compile s | Reference compile s |
|---|---:|---:|---:|
| coils / gc | 4.688 | 15.894 | 15.947 |
| coils / lorentz | 2.398 | 6.561 | 11.719 |
| vmec / gc | 9.337 | 30.346 | 16.116 |
| boozer / gc | 2.794 | 9.049 | 3.679 |

## Adaptive native control

The orbit controller is I, rtol=atol=1e-9. The collision controller is
conservative with tolerance .003, initial interval H/4 and cap H/2. Both are
adaptive; no adaptive/fixed mixture or adaptive Diffrax SDE is shown. Requested
saves are exact split boundaries and constrain these short workloads.

| Field / model | Native collisionless ms | Native split ms | Mean accepted collision / orbit steps |
|---|---:|---:|---:|
| coils / gc | 84.380 | 137.528 | 7.2 / 37.5 |
| coils / lorentz | 29.353 | 41.482 | 7.0 / 62.7 |
| vmec / gc | 5.918 | 13.629 | 7.2 / 7.2 |
| boozer / gc | 1.313 | 4.851 | 7.2 / 7.2 |

All timing cases completed without failed particles. Accepted adaptive collision
estimates were <=1. The reported tolerances are local budgets, not bounds on
global stochastic or splitting errors. `orbit_steps` counts inner trials;
subtract `orbit_rejected` for accepted inner steps.

Run on the normal JAX backend; independent execution is selected on GPU and
shared execution on CPU. All cases run sequentially in fresh child processes:

```sh
python -m examples.benchmarks.benchmark_split_orbits \
  --field coils --model gc --stepping fixed --output output/split_coils_gc_fixed
python -m examples.benchmarks.benchmark_split_orbits \
  --field coils --model lorentz --stepping fixed --output output/split_coils_fo_fixed
python -m examples.benchmarks.benchmark_split_orbits \
  --field vmec --model gc --stepping fixed --output output/split_vmec_fixed
python -m examples.benchmarks.benchmark_split_orbits \
  --field boozer --model gc --stepping fixed --output output/split_boozer_fixed
```

Repeat with `--stepping adaptive` and a distinct output directory. That runs
the two native cases only. `--resume` skips completed matching settings;
`--case native_shared` is an optional implementation diagnostic. Default
benchmark settings respect the machine backend; local measurement used
`XLA_PYTHON_CLIENT_PREALLOCATE=false` to distinguish live usage from a reserved
pool. The committed scripts do not force a device or allocator configuration.

## Physical checks

All tests start from E=5T and pitch .6, without an alpha source. The bath is
held fixed: these are test-particle checks, not a self-consistent collision
operator conserving the combined test/bath energy. Theory is computed by
independent NumPy/SciPy radial/angular generators and refined spatial meshes.
Configurations share repeatable ordered random keys; differing meshes do not
guarantee identical Brownian paths. Time samples are not pooled as independent
particles. Marginal 95% sampling intervals are not simultaneous confidence
bands or numerical error bars.

### Archived accepted statistics

[`thermalization.json`](thermalization.json), [`first_stopping.json`](first_stopping.json)
and [`lorentz_spatial.json`](lorentz_spatial.json) compact the accepted earlier
2,000-particle measurements. Original source hashes are preserved; these
ensembles were not rerun after module consolidation. The thermal check uses
the actual production collision maps, retained Brownian stack and impact-model
coefficients in normalized reporting units. First stopping uses SI coefficients
and the production sampler; Lorentz spatial statistics resolve orbit motion.

Thermalization uses the conservative controller, tolerance .001, cap .000625
and t*nu0=20, starting from a non-Maxwellian beam. Both models' final speed/pitch
CDFs pass the recorded screens. The worst moment discrepancies over all saved
times are 2.95 and 2.89 standard errors, below the declared familywise screen
4.29. GC has a reflecting floor at .05*sqrt(2)*sigma and its reference uses the
same wall. Long-time curves approach the corresponding Maxwellian moments.

First stopping is first entry into E<=2T, not zero speed or final thermal
equilibrium. Fixed and adaptive tolerances .001/.00025 keep the same collision
cap .00078125 and boundary refinement interval 1.220703125e-5 (in 1/nu0 units).
All six CDF screens pass; mean and variance offsets stay below 1.27 standard
errors. Event refinement is a diagnostic near the absorbing speed boundary;
it does not replace the velocity-space collision update.

The older Lorentz spatial study varies Omega/nu0=32,64 and collision caps
.0025/.000625. Its inner orbit is adaptive even in rows labelled fixed
collisions; that provenance is retained explicitly. The new multirate study
below supplies fixed/fixed and adaptive/adaptive controls.

Regenerate the accepted families with the current consolidated implementation:

```sh
python -m examples.benchmarks.validate_beam_relaxation \
  --coefficient-model impact --methods conservative --settings fine \
  --particles 2000 --batch-size 50 --output output/thermalization.json
python -m examples.benchmarks.validate_production_tolerance \
  --particles 2000 --batch-size 50 --output output/first_stopping.json
python -m examples.benchmarks.validate_lorentz_coulomb_spatial \
  --coulomb-log-model impact --particles 2000 --batch-size 50 \
  --output output/lorentz_spatial.json
```

To reproduce the plots alone, pass a copied compact thermalization or
first-stopping report as `--output ... --plot-only` to its corresponding
module. The first-stopping reproducer now generates fresh complete ensembles
rather than requiring obsolete raw baseline files or matching old file-layout
hashes. It records the current source hashes. Archived plots remain archived;
fresh results should not be relabelled as their original measurements.

The numerical construction follows the [adaptive collision paper](https://arxiv.org/abs/1701.05043)
with nonrelativistic coefficients. Conservative control, the first-passage
diagnostic refinement and the finite-Omega reference are documented ESSOS
choices; they are not claimed to reproduce every controller or validation
setting in the relativistic paper.

### New production multirate spatial study

[`multirate.json`](multirate.json) and [`multirate.csv`](multirate.csv) record
2,000 particles per setting, seed 7706, batches of 100, five saves, t*nu0=.1,
Omega/nu0=128,512,2048 and caps .0025/.00125. This run calls `trace_split`
with actual SI uniform fields, `Particles`, `BackgroundSpecies`, collision
coefficients and the production Brownian memory. It does not generate an
alternative Wiener process for the benchmark.

Fixed orbit dt starts at min(collision cap,.1/Omega); adaptive orbit
rtol=atol=1e-10 and conservative collision tolerance=.001. The reference is
finite-Omega full Lorentz guiding-centre MSD, not the asymptotic GC diffusion
formula. Refining its radial mesh 384->768 changes the final result by at
most .027% across these frequencies.

| Omega/nu0 | Fixed orbit/collision ratio, coarse / fine | Adaptive ratio, coarse / fine | Fine fixed / adaptive final MSD z |
|---|---:|---:|---:|
| 128 | 4 / 2 | 1.01 / 1.00 | .79 / -.78 |
| 512 | 13 / 7 | 1.94 / 1.34 | .27 / .68 |
| 2048 | 52 / 26 | 5.17 / 3.05 | 1.82 / -.01 |

All refined endpoints lie within marginal 95% sampling intervals. The coarse
fixed Omega=2048 endpoint has z=2.45; it is retained rather than hidden. This
shows that clock separation works and supports the refined tested regime,
while leaving splitting/finite-sampling errors visible. Local tolerances
alone cannot certify arbitrary interval separation.

```sh
python -m examples.benchmarks.benchmark_split_multirate \
  --particles 2000 --batch-size 100 --output output/split_multirate
```

Use `--plot-only` to regenerate `multirate.png` from a copied report named
`comparison.json` in the selected output directory. The saved file here is
called `multirate.json` for clarity. Scripts also export SVGs when regenerating;
only the README PNGs are retained in the PR.

### Retained exploratory convergence control

[`controller_convergence.csv`](controller_convergence.csv) and its compact
JSON retain the 512-particle fixed/conservative/paper/candidate tolerance/cap
scan. This predates the accepted boundary-refinement diagnostics and uses NRL
coefficients. It is not an additional accepted validation or a controller
ranking. Regenerate with:

```sh
python -m examples.benchmarks.converge_collisional_slowing_down \
  --paths 512 --batch-size 64 --output output/controller_convergence.json
```

## Fast automated checks

`pytest -q tests/test_native_split.py` collects 19 CPU-capable tests covering
coefficient/divergence and zero-flux identities, collision covariance and
reflection, Brownian conditioning/rejection memory, error/control bounds,
separate clocks, events/failure statuses, public Cartesian/coordinate tracing,
private field parity and interpreted GPU execution. The native-orbit suite is
also rerun for the small shared RK extensions. The large statistical ensembles
and GPU performance workloads are explicit examples, not GitHub Actions tests.
