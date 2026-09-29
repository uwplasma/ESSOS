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
field = BoozerField.from_booz_xform(booz, psi0=-phi_edge / (2 * np.pi),
                                    mode_tolerance=1e-4)

n = 1000
speed = np.sqrt(2 * 3.5e6 * ONE_EV / ALPHA_PARTICLE_MASS)
result = trace_boozer(field, s=np.full(n, 0.25), theta=np.random.uniform(0, 2*np.pi, n),
                      zeta=np.random.uniform(0, 2*np.pi, n), pitch=np.random.uniform(-1, 1, n),
                      speed=speed, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                      tmax=1e-2, timestep=1e-7)      # species=BackgroundSpecies(...) adds collisions
print(result.lost.mean(), result.loss_fractions())
```

For a VMEC WOUT, `psi0` is the negative of its boundary `phi` over `2 pi`
because VMEC uses a negative coordinate Jacobian. A particle
is lost at `s = 1`; `result.loss_times` and `result.states` hold when and where.
Check the mode cut on the intended equilibrium: small discarded harmonics can
change individual loss labels ([VMEX cutoff study](https://github.com/uwplasma/vmex/pull/516)).

![Boozer and VMEC-coordinate tracing times for the corrected 128-alpha example](docs/readme_boozer_speed.png)

The [128-alpha ARIES-CS example](examples/particle_tracing/trace_particles_boozer_vs_vmec.py)
uses the corrected VMEC flux sign, 0.1 ms, and eight Apple M2 CPU devices.
Times exclude JIT compilation. The two loss fractions agree within their
binomial errors; the Boozer call is 68 times faster on this workload.

| ESSOS tracer | Lost / 128 | Warm trace time | Maximum Boozer energy drift |
|---|---:|---:|---:|
| Boozer RK4 | 14 (10.9% ± 2.8%) | 0.221 s | 2.39e-5 |
| VMEC adaptive | 17 (13.3% ± 3.0%) | 14.94 s | not recorded |

A second comparison uses the same 1,024 fusion-alpha births in a reactor-scaled,
nonoptimized NFP=2 vacuum VMEX equilibrium (`ns=31`, `mpol=5`, `ntor=5`),
traced for 2 ms. ESSOS uses 12 retained Boozer modes, fixed RK4 steps of
`1.25e-7 s`; CATAPULT uses a 25×25×25 tricubic field table and adaptive DP5 at
`1e-10` tolerance. These warmed GPU timings were made
on the same NVIDIA GTX TITAN X; Boozer transform, field setup and JIT
compilation are excluded. Both request 101 sample times, but CATAPULT
truncates lost trajectories (median four stored rows across this ensemble)
while ESSOS returns 101 states per particle.

| Tracer | Lost / 1,024 | Labels matching ESSOS | Warm GPU time | Speed vs ESSOS scan | Maximum confined-orbit energy drift |
|---|---:|---:|---:|---:|---:|
| ESSOS Boozer, default scan ([kernel PR #95](https://github.com/uwplasma/ESSOS/pull/95)) | 795 | 1,024 / 1,024 | 14.72 s | 1.0× | 2.08e-6 |
| ESSOS Boozer, [GPU lookup PR #98](https://github.com/uwplasma/ESSOS/pull/98) | 795 | 1,024 / 1,024 | 3.81 s | 3.9× | 2.08e-6 |
| CATAPULT, released radial interpolation | 802 | 1,017 / 1,024 | 4.88 s | 3.0× | 7.84e-3 |
| CATAPULT, [axis fix PR #90](https://github.com/ColumbiaStellaratorTheory/firm3d/pull/90) | 795 | 1,024 / 1,024 | 4.68 s | 3.1× | 3.54e-4 |

The ESSOS lookup change leaves loss times, saved states and energy errors
bitwise identical on this field and on larger 100- and 300-knot fields.
With the same field table, births and requested output on one host
i7-3820 CPU core, patched FIRM3D's serial particle loop takes 74.67 and
74.70 s in two warmed runs. It loses the same 795 particles and has maximum
all-path energy drift `3.55e-4`. CPU and GPU setup times are excluded.
On the same host, ESSOS takes 18.15 s using eight CPU devices, with the
same loss labels and energy diagnostics. Its one-device run takes 260.76 s;
CPU comparisons depend strongly on particle parallelism.

The seven released-CATAPULT disagreements all cross `s < 0.03`. Its radial
interpolant assigns a nonzero `m=1` field harmonic on the magnetic axis,
where regularity requires zero. A temporary axis-regularized version changes
all seven to confined and lowers energy drift; the regularization is proposed in
[FIRM3D PR #90](https://github.com/ColumbiaStellaratorTheory/firm3d/pull/90) and remains experimental until merged. An independent [DESC](https://desc-docs.readthedocs.io/en/latest/_api/particles/desc.particles.trace_particles.html)
trace keeps those seven confined. On a separate 64-birth subset, ESSOS,
CATAPULT, FIRM3D CPU and DESC agree on all 64 loss labels (55 losses).
The patch still uses a cubic spline in `s`, so its near-axis `m=1` radial
scaling is approximate even though these loss labels agree.
DESC's warmed 64-birth CPU trace takes 32.46 s on an Apple M2 with endpoint
output; its maximum surviving-particle endpoint energy error is 2.59e-4.
ESSOS takes about 1.55 s on the M2 with 101 saved states. The different
output policies and hardware across rows preclude a general speed ranking.

On the same 64 Boozer births, SIMPLE's direct-Boozer `startmode=6` and the
[SIMSOPT VMEC-flux fix](https://github.com/hiddenSymmetries/simsopt/pull/664)
each reproduce all 55 ESSOS losses. The unpatched SIMSOPT field agreed on
only 44 of 64 labels because its radial drift had the opposite sign. SIMPLE's
symplectic Euler run (`npoiper2=512`) has maximum endpoint energy drift
`8.28e-4`; its midpoint option (`npoiper2=256`) lowers that to `8.03e-6`
with the same labels. Corrected SIMSOPT's maximum endpoint drift is
`3.00e-4`. These endpoint checks sample less of the orbit than ESSOS's
every-step diagnostic. The [VMEX comparison guide](https://github.com/uwplasma/vmex/pull/516)
gives the seed-WOUT recipe and a fail-closed three-code benchmark script.

| Same 64 births, i7-3820 CPU | Lost / 64 | Warm trace-only time | Maximum reported energy drift |
|---|---:|---:|---:|
| ESSOS Boozer RK4, eight devices | 55 | 1.701 s | 1.66e-6, every step |
| SIMPLE symplectic Euler, eight threads | 55 | 1.898 s | 8.28e-4, endpoint |
| SIMPLE symplectic midpoint, eight threads | 55 | 3.469 s | 8.03e-6, endpoint |
| SIMSOPT `gc_noK`, eight workers | 55 | 0.400 s | 3.00e-4, endpoint |

Field construction and compilation are excluded; SIMSOPT's interpolation
table alone takes roughly two minutes on this host. These times and the GPU
table above describe their specific output and solver policies.

| Code | Orbit models and solver | CPU / GPU | Collisions | Trajectory differentiation | Matched accuracy and speed evidence |
|---|---|---|---|---|---|
| ESSOS Boozer | guiding centre, fixed RK4 | both (JAX) | yes | host result is NumPy | 795/1,024; 3.81 s GPU, 3.9× default scan |
| ESSOS VMEC/coil | guiding centre, adaptive; full orbit | both (JAX) | yes | JAX trajectories | 17/128; Boozer 68× faster on that M2 case |
| [SIMPLE](https://github.com/itpplasma/SIMPLE) | guiding centre, symplectic CPU or CUDA Dormand–Prince | CPU (OpenMP) / NVIDIA GPU | no | none documented | 55/64; 1.898 s, eight CPU threads |
| [SIMSOPT](https://simsopt.readthedocs.io/v0.9.4/tracing.html) | guiding centre or full orbit, adaptive | CPU | no in this model | none documented | [sign fixed](https://github.com/hiddenSymmetries/simsopt/pull/664): 55/64; 0.400 s, eight CPU workers |
| [FIRM3D](https://firm3d.readthedocs.io/) / [CATAPULT](https://arxiv.org/abs/2604.07617) | guiding centre, adaptive or symplectic; GPU DP5 | CPU / NVIDIA GPU | no in this comparison | none documented | axis fixed: 795/1,024; 4.68 s GPU, 74.7 s CPU core |
| [DESC](https://desc-docs.readthedocs.io/en/latest/_api/particles/desc.particles.trace_particles.html) | vacuum guiding centre, adaptive Diffrax | both (JAX) | no in documented model | JAX adjoints | 55/64; 32.46 s M2 CPU, endpoint output |

SIMPLE's documented production CUDA mode is collisionless Boozer tracing
without orbit output or classifiers; its CPU mode has a broader solver set.
The measured rows above are case-specific, not throughput claims across different
physics or devices. Recheck field interpolation, energy and individual labels
when moving to another equilibrium, particle ensemble or time horizon.

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
