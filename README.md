# ESSOS

<p>
    <img src="https://img.shields.io/github/license/uwplasma/ESSOS?style=default&color=0080ff" alt="license">
    <img src="https://github.com/uwplasma/ESSOS/actions/workflows/build_test.yml/badge.svg" alt="Build Status">
    <img src="https://codecov.io/gh/uwplasma/ESSOS/branch/main/graph/badge.svg" alt="Coverage">
    <img src="https://readthedocs.org/projects/essos/badge/?version=latest" alt="Documentation">
</p>

Stellarator coil and particle optimization in JAX. Coil geometry, Biot-Savart
fields, and the JAX orbit models run on CPU or GPU and support derivatives of
smooth design objectives. The fast Boozer tracer returns NumPy diagnostics;
see its capabilities in the comparison table below.

```sh
pip install essos
```

## What it does

- **Differentiable optimization models.** `jax.grad` works through coil geometry,
  the field, and JAX orbit models for smooth objectives.
- **Coil optimization.** Fit coils to a plasma boundary under length, curvature,
  separation, coil-surface-distance and force constraints, with `least_squares`,
  an augmented Lagrangian, multi-objective (Pareto) search, or stochastic
  optimization over Gaussian coil perturbations. Coils can also target a
  near-axis field, particle confinement, or a finite-beta (VMEX) boundary.
- **Particle tracing.** Guiding-centre and full-orbit (Boris) models, with
  Monte Carlo collisions on background species with density and temperature
  profiles, electric fields and alpha-loss diagnostics.
- **Boozer-coordinate tracing.** A guiding-centre tracer that needs only the
  Boozer `|B|` spectrum and the flux functions `iota`, `G` and `I`; about 68x
  faster than VMEC-coordinate tracing in the 128-alpha example below.
- **Field-line tracing.** Adaptive, arclength and toroidal-angle models, with
  Poincare sections.
- **Fields.** Biot-Savart from coils, VMEC equilibria (analytic derivatives,
  optional `mode_tolerance` truncation), near-axis expansions, and
  `CombinedField` to trace a sum of fields as one.
- **VMEC MGRID.** Export coil fields and load MGRID files as JAX-compatible
  three-dimensional magnetic fields.
- **Surfaces.** Fourier-represented toroidal surfaces, from a VMEC `wout` or
  built directly.
- **Parallel.** JAX sharding across the visible devices; pass `devices=` to pick.
- **Checked against SIMSOPT.** [`examples/comparisons_simsopt`](examples/comparisons_simsopt)
  compares coils, surfaces, VMEC import, field lines, guiding-centre and
  full-orbit tracing, and losses with SIMSOPT on the same inputs.

## Coil optimization

Fit coils to a VMEC boundary, trading normal-field error against coil length and
curvature. Losses compose with `+`, and `L.grad` is the exact gradient.

![Coils fitted to a QA boundary](docs/readme_coil_optimization.png)

```python
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.losses import custom_loss
from essos.surfaces import BdotN_over_B, SurfaceRZFourier
from scipy.optimize import least_squares

surface = SurfaceRZFourier.from_wout_file("wout.nc", s=1, ntheta=30, nphi=30,
                                          range_torus="half period")
coils = Coils(curves=CreateEquallySpacedCurves(3, 3, 10.0, 5.6, n_segments=45,
                                               nfp=2, stellsym=True),
              currents=[1.0] * 3)

L = (custom_loss(lambda field, surface: abs(BdotN_over_B(surface, field)).sum(),
                 "field", surface=surface)
     + custom_loss(lambda field: (field.coils.length - 32.0).clip(0).mean(), "field")
     + custom_loss(lambda field: (field.coils.curvature - 0.1).clip(0).mean(), "field"))
L.dependencies = {"field": BiotSavart(coils)}

result = least_squares(L, L.starting_dofs, L.grad, max_nfev=400)
optimized = L.dofs_to_pytree(result.x)["field"].coils
```

More in [`examples/coil_optimization`](examples/coil_optimization).

## Field-line tracing

![Poincare section of a coil field](docs/readme_fieldlines.png)

```python
import jax.numpy as jnp
from essos.coils import Coils
from essos.dynamics import Tracing
from essos.fields import BiotSavart

field = BiotSavart(Coils.from_json("coils.json"))
R0 = jnp.linspace(1.21, 1.40, 8)
seeds = jnp.array([R0, jnp.zeros_like(R0), jnp.zeros_like(R0)]).T

tracing = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=seeds,
                  maxtime=8000, times_to_trace=40000, atol=1e-8, rtol=1e-8)
tracing.poincare_plot(shifts=[0.0])
```

More in [`examples/fieldline_tracing`](examples/fieldline_tracing).

## VMEC MGRID fields

Run [`examples/simple_examples/mgrid_from_coils.py`](examples/simple_examples/mgrid_from_coils.py)
to export the included Landreman-Paul QA coils, load the file as a magnetic
field, and compare its interpolated field with direct Biot-Savart. It prints
the differences and saves a plot; the maximum difference at its four sample
points is about 0.2%. Accuracy depends on grid resolution and distance from
the coils, so compare in your region of interest before using a grid.

The cylindrical grid covers one field period in phi. The field repeats across
periods, and R and Z queries are clamped to the grid bounds, so set the bounds
to cover the region you intend to evaluate.

## Particle tracing

![Guiding-centre alpha orbits](docs/readme_particles.png)

```python
from essos.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, ONE_EV
from essos.dynamics import Particles, Tracing

particles = Particles(initial_xyz=seeds, mass=ALPHA_PARTICLE_MASS,
                      charge=ALPHA_PARTICLE_CHARGE, energy=4000 * ONE_EV)
tracing = Tracing(field=field, model="GuidingCenterAdaptative", particles=particles,
                  maxtime=1e-4, times_to_trace=800, atol=1e-7, rtol=1e-7)
tracing.plot()
print(tracing.loss_fractions)
```

More in [`examples/particle_tracing`](examples/particle_tracing), including
full-orbit, collisional and electric-field variants.

## Boozer tracing

Transform a VMEC WOUT with [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax), then trace guiding centres from its `|B|` spectrum and flux functions. The fixed-step RK4 tracer supports optional Monte Carlo collisions; check the mode cut for each equilibrium.

```python
import numpy as np
from netCDF4 import Dataset
from booz_xform_jax import Booz_xform
from essos.boozer import BoozerField, trace_boozer
from essos.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, ONE_EV

booz = Booz_xform(verbose=0, mboz=32, nboz=32)
booz.read_wout("wout.nc", flux=False)
booz.run()
with Dataset("wout.nc") as wout:
    phi_edge = float(wout.variables["phi"][-1])
field = BoozerField.from_booz_xform(booz, psi0=-phi_edge / (2 * np.pi),
                                    mode_tolerance=1e-4)

n = 1000
rng = np.random.default_rng(42)
speed = np.sqrt(2 * 3.5e6 * ONE_EV / ALPHA_PARTICLE_MASS)
result = trace_boozer(field, s=np.full(n, 0.25), theta=rng.uniform(0, 2*np.pi, n),
                      zeta=rng.uniform(0, 2*np.pi / field.nfp, n),
                      pitch=rng.uniform(-1, 1, n), speed=speed,
                      mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                      tmax=1e-2, timestep=1e-7)
print(result.lost.mean(), result.loss_fractions())
```

VMEC's flux convention requires `psi0 = -phi_edge / (2*pi)`. A loss occurs at `s >= 1`; `result.loss_times` and `result.states` record the event and orbit. See the [VMEX cutoff study](https://github.com/uwplasma/vmex/pull/516) before interpreting small changes in loss fraction.

![Boozer and VMEC tracing times for 128 alphas](docs/readme_boozer_speed.png)

In the [128-alpha ARIES-CS example](examples/particle_tracing/trace_particles_boozer_vs_vmec.py), both methods use the same births for 0.1 ms on eight Apple M2 CPU devices. Warm times exclude JIT; `±` is the one-sigma binomial sampling error `sqrt(f(1-f)/N)`.

| ESSOS tracer | Lost / 128 | Warm trace | Maximum energy drift |
|---|---:|---:|---:|
| Boozer RK4 | 14 (10.9% ± 2.8%) | 0.221 s | 2.39e-5, every step |
| VMEC adaptive | 17 (13.3% ± 3.0%) | 14.94 s | not recorded |

The GPU comparison uses 4,096 common alpha births in a reactor-scaled VMEX equilibrium (`ns=31`, `mpol=5`, `ntor=5`) for 10 ms. ESSOS uses 12 Boozer modes and `dt=1.25e-7 s`; CATAPULT uses a 25³ tricubic table and adaptive DP5 at `1e-10` tolerance. Times are one first and one repeated trace on the same RTX A4000, excluding field setup. Both request 101 times; CATAPULT truncates lost paths while ESSOS returns 101 states per particle.

| RTX A4000 tracer | Lost / 4,096 | Labels matching ESSOS | First trace | Repeated trace | Maximum energy drift |
|---|---:|---:|---:|---:|---:|
| ESSOS GPU lookup ([#98](https://github.com/uwplasma/ESSOS/pull/98)) | 3,340 | 4,096 | 33.31 s | 25.63 s | 1.05e-5, every step |
| CATAPULT released | 3,371 | 4,065 | 11.53 s | 11.58 s | 1.97e-2, saved confined paths |

At 20 ms, the loss counts are unchanged. CATAPULT's three repeated calls take 21.97–22.14 s; ESSOS's repeated calls range from 50.95 to 103.49 s across two A4000 runs despite bitwise-identical states. That spread prevents a single 20 ms speed ratio. Thirty of the 31 released CATAPULT-only losses reach `s<0.01` in ESSOS. The draft [FIRM3D #90](https://github.com/ColumbiaStellaratorTheory/firm3d/pull/90) patch matches 4,094 labels but worsens some higher-resolution near-axis energy errors.

In a separate QA equilibrium, 4,096 births at `s=0.25` traced for 10 ms
with 101 requested times give 16 ESSOS and 21 released CATAPULT losses
(4,081 matching labels). First/repeated traces take 31.76/24.82 s for
ESSOS and 160.51/161.47 s for CATAPULT;
maximum relative energy drift is 9.46e-5 at every ESSOS step and 3.03e-2
on saved confined CATAPULT paths. These loss labels have not been converged.

For the first 64 births, ESSOS, SIMPLE and SIMSOPT agree on all resolved loss labels. Six SIMSOPT orbits reach the `s=0.001` stop surface and remain unresolved; the CPU times below use the same i7-3820, with code-specific output and parallelism.

| 64-birth CPU trace | Loss result | Warm trace | Maximum energy drift |
|---|---:|---:|---:|
| ESSOS RK4, eight devices | 55 | 1.701 s | 1.66e-6, every step |
| SIMPLE midpoint, eight threads | 55 | 5.247 s | 1.20e-5, 401 saved states |
| SIMSOPT `gc_noK`, eight workers, axis stop ([flux fix #664](https://github.com/hiddenSymmetries/simsopt/pull/664)) | 50/58 resolved; six stops | 0.415 s | 6.77e-4, resolved path states |

On the RTX A4000, DESC 0.17.1 also matches all 64 labels (55 losses); its warmed 101-time trace takes 21.40 s, versus 1.94 s for ESSOS #98 and 2.63 s for the draft FIRM3D #90 patch. DESC's surviving-endpoint energy drift reaches 2.62e-4. The [VMEX comparison guide](https://github.com/uwplasma/vmex/pull/516) gives the WOUT recipe, orbit checks and full timing conditions.

| Code | Orbit models | CPU / GPU | Collisions | Differentiation | Measured result here |
|---|---|---|---|---|---|
| ESSOS Boozer | guiding centre, RK4 | both (JAX) | yes | JAX RHS; public trace returns NumPy | 3,340/4,096; 25.63 s A4000, 10 ms |
| ESSOS VMEC/coil | guiding centre; full orbit | both (JAX) | yes | JAX trajectories | 17/128; Boozer 68× faster in that case |
| [SIMPLE](https://github.com/itpplasma/SIMPLE) | guiding centre, symplectic or adaptive | CPU / NVIDIA GPU | no | none documented | 55/64; midpoint 5.247 s CPU |
| [SIMSOPT](https://simsopt.readthedocs.io/v0.9.4/tracing.html) | guiding centre; full orbit | CPU | no in tested model | none documented | 58 matched labels; six axis stops |
| [FIRM3D](https://firm3d.readthedocs.io/) / [CATAPULT](https://arxiv.org/abs/2604.07617) | guiding centre, adaptive or symplectic | CPU / NVIDIA GPU | not tested | none documented | released: 3,371/4,096; 11.58 s A4000, 10 ms |
| [DESC](https://desc-docs.readthedocs.io/en/latest/_api/particles/desc.particles.trace_particles.html) | guiding centre, adaptive Diffrax | both (JAX) | no in documented model | JAX adjoints | 55/64; 21.40 s A4000 |

## Tracing notes

- **VMEC magnetic axis.** VMEC guiding centres are integrated in
  `sqrt(s) (cos theta, sin theta)`, which is regular on the axis, so orbits
  cross it; trajectories are returned in `(s, theta, phi, ...)` with `theta`
  in `[0, 2 pi)`. A trace stops at `s >= 1` and reports it through
  `tracing.boundary_hits`. Supplying `condition` replaces that event; it is
  evaluated on `(s, theta, phi, ...)`.
- **Stopping coil-field traces.** Pass `stopping_criteria=LevelsetStoppingCriterion(...)`
  to end Cartesian traces once they leave a prescribed distance from a surface,
  and read the per-line mask from `tracing.boundary_hits`.
- **Step budget.** `max_steps` (default `1_000_000`) bounds every Diffrax solve,
  so a trace that cannot finish returns instead of running unbounded.
- **Progress bars** are off by default; pass `progress=True` when interactive.
- **Model choice.** `FieldLineArclength` traces a fixed physical length so
  rescaling `B` does not change the run; `FieldLineToroidal` sets coverage
  directly in toroidal angle for flux-coordinate fields.

## Optional: near-axis fields

Near-axis expansions need [pyQSC_JAX](https://github.com/uwplasma/pyQSC_JAX),
which is not on PyPI and therefore cannot be a declared dependency:

```sh
pip install git+https://github.com/uwplasma/pyQSC_JAX.git
```

Everything else works without it; the near-axis entry points raise with this
command if it is missing.

## Examples

| Folder | Contents |
|---|---|
| [`coil_optimization`](examples/coil_optimization) | VMEC and near-axis targets, augmented Lagrangian, stochastic, multi-objective, particle confinement, finite beta |
| [`fieldline_tracing`](examples/fieldline_tracing) | Coil and VMEC field lines, Poincare sections, connection length |
| [`particle_tracing`](examples/particle_tracing) | Guiding-centre and full-orbit tracing, classifiers, electric fields, Boozer vs VMEC |
| [`particle_tracing_collisions`](examples/particle_tracing_collisions) | Collisional tracing and velocity-distribution statistics |
| [`comparisons_simsopt`](examples/comparisons_simsopt) | Side-by-side runs with SIMSOPT |
| [`simple_examples`](examples/simple_examples) | Coils from near-axis or BOOZ_XFORM, perturbed coils, combined fields, MGRID |
| [`paper`](examples/paper) | Integrator, gradient and Poincare figures |

## Testing

```sh
pytest
```

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgments

Developed by the [UWPlasma](https://rogerio.physics.wisc.edu/) group at the
University of Wisconsin-Madison, with support from Simons Foundation grant
560651.
