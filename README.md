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
  Boozer `|B|` spectrum and the flux functions `iota`, `G` and `I`; about 66x faster than VMEC-coordinate tracing and
  3.8x faster than SIMPLE and 7.4x faster than SIMSOPT on the same alphas (see below).
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

## Boozer-coordinate tracing

For alpha-loss studies in a VMEC equilibrium, transform it to Boozer
coordinates once (with [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax))
and trace there. `essos.boozer` integrates the guiding-centre equations in a
chart regular on the magnetic axis, with fixed-step RK4 vectorized and sharded
over particles, and optional Monte Carlo collisions (pitch-angle scattering,
slowing down and energy diffusion on `BackgroundSpecies`).

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
    phi_edge = float(wout.variables["phi"][-1])  # boundary toroidal flux
field = BoozerField.from_booz_xform(booz, psi0=phi_edge / (2 * np.pi),
                                    mode_tolerance=1e-3)

n = 1000
speed = np.sqrt(2 * 3.5e6 * ONE_EV / ALPHA_PARTICLE_MASS)
result = trace_boozer(field, s=np.full(n, 0.25), theta=np.random.uniform(0, 2*np.pi, n),
                      zeta=np.random.uniform(0, 2*np.pi, n), pitch=np.random.uniform(-1, 1, n),
                      speed=speed, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                      tmax=1e-2, timestep=1e-7)      # species=BackgroundSpecies(...) adds collisions
print(result.lost.mean(), result.loss_fractions())
```

`psi0` is the boundary toroidal flux over `2 pi`. A particle
is lost at `s = 1`; `result.loss_times` and `result.states` hold when and where.

![Boozer vs VMEC-coordinate and cross-code tracing times](docs/readme_boozer_speed.png)

| Case (8 CPU cores, compile excluded) | Tracer | Lost | Wall time |
|---|---|---|---|
| 128 ARIES-CS alphas, 0.1 ms | ESSOS Boozer (RK4) | 14.8% ± 3.1% | 0.91 s |
| | ESSOS VMEC coordinates (adaptive) | 13.3% ± 3.0% | 60.5 s |
| 1000 alphas, 10 ms (VMEX benchmark) | ESSOS Boozer | 12.8% | 146 s |
| | SIMPLE | 12.4% | 556 s |
| | SIMSOPT | 11.9% | 1079 s |

The first case is [`examples/particle_tracing/trace_particles_boozer_vs_vmec.py`](examples/particle_tracing/trace_particles_boozer_vs_vmec.py):
the loss fractions agree within their binomial errors and Boozer tracing is
about 66x faster. The figure is redrawn from these numbers by
`python docs/make_readme_boozer_figure.py`.

The 1000-particle rows above use one WOUT, the same births and energy, a 10 ms
horizon, and eight CPU cores; compilation and field setup are excluded
([protocol](https://github.com/uwplasma/vmex/blob/main/benchmarks/trace_cross_code.py)).
Capabilities differ, so the measured speed ratios apply only to this case:

| tracer | particle model and method | hardware | collisions | orbit derivatives | matched loss / run time | ESSOS speedup |
|---|---|---|---|---|---|---|
| ESSOS Boozer | guiding centre, fixed RK4 | JAX CPU/GPU | yes | current host API returns NumPy | 12.8% / 146 s | 1× |
| ESSOS VMEC/coil | guiding centre, adaptive; full orbit, Boris/adaptive | JAX CPU/GPU | yes | JAX trajectories | separate 128-particle case above | 66× on that case |
| [SIMPLE](https://github.com/itpplasma/SIMPLE) | guiding centre, symplectic CPU or CUDA Dormand–Prince | CPU/GPU | collisionless in matched run | no tracing AD documented | 12.4% / 556 s | 3.8× |
| [SIMSOPT](https://simsopt.readthedocs.io/v0.9.4/tracing.html) | guiding centre or full orbit, adaptive | CPU | collisionless in matched run | no tracing AD documented | 11.9% / 1079 s | 7.4× |
| [CATAPULT](https://arxiv.org/abs/2604.07617) | guiding centre, adaptive tricubic field | NVIDIA GPU | collisionless in published study | no tracing AD documented | not run on matched case | unknown |
| [DESC](https://desc-docs.readthedocs.io/en/latest/_api/particles/desc.particles.trace_particles.html) | vacuum guiding centre, adaptive Diffrax | JAX CPU/GPU | collisionless in documented model | JAX trajectory adjoints | not run on matched case | unknown |

CATAPULT reports converged loss estimates for 32,768 particles over 1 ms and
5–10× throughput versus 128 CPU cores on an A100; its [study](https://arxiv.org/abs/2604.07617)
uses different fields, births, horizons and hardware, so those numbers cannot
be divided by the eight-core ESSOS time above. DESC's [particle API](https://desc-docs.readthedocs.io/en/latest/api_particles.html)
provides a vacuum guiding-centre tracer, but no same-WOUT loss fraction or
runtime has been established here. A numerical ranking needs identical
initial conditions, field accuracy, loss surface and time horizon.

**Why it is fast.** In Boozer coordinates the guiding-centre equations of
motion depend only on `|B|` and the flux functions `G`, `I` and `iota`, not on
the full vector field or its Cartesian gradients. Each step evaluates one
scalar Fourier series, `|B| = sum b_mn(s) cos(m theta - n zeta)`, and its three
derivatives (about 1 us per particle with 135 modes), instead of the 3-D
field and its gradient from the VMEC geometry (8-10 us).

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
