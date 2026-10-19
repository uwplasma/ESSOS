# ESSOS

<p>
    <img src="https://img.shields.io/github/license/uwplasma/ESSOS?style=default&color=0080ff" alt="license">
    <img src="https://github.com/uwplasma/ESSOS/actions/workflows/build_test.yml/badge.svg" alt="Build Status">
    <img src="https://codecov.io/gh/uwplasma/ESSOS/branch/main/graph/badge.svg" alt="Coverage">
    <img src="https://readthedocs.org/projects/essos/badge/?version=latest" alt="Documentation">
</p>

Stellarator coil optimization and particle tracing in JAX, on CPU or GPU.
Coil objectives and adaptive tracing support automatic differentiation.

```sh
pip install essos
```

## What it does

- **Automatic differentiation.** `jax.grad` works through coil geometry,
  Biot-Savart fields and adaptive tracing.
- **Coil optimization.** Fit coils to a plasma boundary under length, curvature,
  separation, coil-surface-distance and force constraints, with `least_squares`,
  an augmented Lagrangian, multi-objective (Pareto) search, or stochastic
  optimization over Gaussian coil perturbations. Coils can also target a
  near-axis field, particle confinement, or a finite-beta boundary (VMEX, or
  the plasma field from [virtual_casing_jax](https://github.com/uwplasma/virtual_casing_jax)
  in [`optimize_coils_virtual_casing.py`](examples/coil_optimization/optimize_coils_virtual_casing.py)).
- **Particle tracing.** Guiding-centre and full-orbit (Boris) models, with
  Monte Carlo collisions on background species with density and temperature
  profiles (NRL Coulomb logarithm per species pair), electric fields and
  alpha-loss diagnostics.
- **Boozer-coordinate tracing.** Guiding-centre diagnostics from the Boozer
  `|B|` spectrum and flux functions `iota`, `G` and `I`, with fixed-step
  (RK4, Dopri5, Dopri8) or adaptive Dormand-Prince stepping and optional collisions.
- **Field-line tracing.** Adaptive, arclength and toroidal-angle models, with
  Poincare sections.
- **Fields.** Biot-Savart from coils, VMEC equilibria (stellarator-symmetric or
  not, analytic derivatives, optional `mode_tolerance` truncation), near-axis
  expansions, and `CombinedField` to trace a sum of fields as one.
- **Interpolated fields.** `InterpolatedField(source, R=..., Z=..., nr, nz, nphi, nfp)`
  tabulates any Cartesian field (coils, dipoles, MGRID, combined) on a
  cylindrical grid and evaluates `B`, `dB/dX` and guiding-centre terms from a
  tricubic spline; it saves to `.npz` or MGRID and works under `jit` and
  `grad`. See [`trace_particles_interpolated_dipoles.py`](examples/particle_tracing/trace_particles_interpolated_dipoles.py).
- **VMEC MGRID.** Export coil fields and load MGRID files as JAX-compatible
  three-dimensional magnetic fields.
- **Surfaces.** Fourier-represented toroidal surfaces, from a VMEC `wout` or
  built directly.
- **Parallel by default.** ESSOS asks JAX for one device per CPU core and
  shards particles over every CPU core or GPU (`ESSOS_CPU_DEVICES` overrides
  the count; `devices=` picks). Compiled traces are cached on disk
  (`ESSOS_XLA_CACHE`, default `~/.cache/essos/xla`).
- **Checked against SIMSOPT.** [`examples/comparisons_simsopt`](examples/comparisons_simsopt)
  compares coils, surfaces, VMEC import, field lines, guiding-centre and
  full-orbit tracing, and losses with SIMSOPT on the same inputs. Alpha
  losses are also compared with SIMSOPT, FIRM3D and SIMPLE
  ([below](#comparison-with-other-codes)).

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

Pass `integrator="essos_rk45"` for native JAX Dormand--Prince 5(4), or
`integrator="essos_dopri8"` for native Dormand--Prince 8(7), the same RK formula
as ESSOS's default Diffrax solver. Both support `GuidingCenter`,
`GuidingCenterAdaptative`, `FullOrbit` and `FullOrbitAdaptative`;
the `Adaptative` models control step size with
`rtol`/`atol`, while the others use fixed `timestep` steps. Saved times use
dense interpolation and do not set the integration steps. VMEC axis handling,
LCFS crossings and exterior continuation are supported. The default remains
`integrator="diffrax"`; `solver=` selects a solver for that default backend.

VMEC guiding centers use the same `integrator="essos_rk45"` or
`"essos_dopri8"` selection in `Tracing(field=Vmec(...), ...)`. Particle births
are supplied as `(s, theta, phi)` in `Particles.initial_xyz`; integration uses
the existing regular axis chart and measures the embedded position error in
Cartesian metres. Boozer guiding centers use these selections in
[`trace_boozer`](#boozer-coordinate-tracing), with their existing field and drift
equations. Flux-coordinate support here is for guiding centers; the Lorentz
equations require a field that supplies Cartesian magnetic vectors.

The native solvers currently implement forward collisionless orbits. Split
collision updates and a custom derivative rule that replays accepted steps
are subsequent phases. The native driver does not yet support reverse-mode
AD. The standalone [`essos.rk.solve_rk`](essos/rk.py) (and its `solve_rk45` and
`solve_dopri8` entry points) also exposes termination status, accepted/rejected
trial counts and optional accepted-time history.
The comparison scripts in [`examples/paper`](examples/paper) include both
backends; their reported times include JAX compilation.

GC and Lorentz use the same native RK increment loop, separate from adaptive
retries, events and saving. It stores `k_i = h*f_i` for stage combinations and
dense output, and keeps the final derivative separately for FSAL reuse. No
division by `h` is needed. The stage loop and the public derivative-based
`rk45_step`/`dopri8_step` maps have ordinary forward/reverse JAX derivatives
in state, time, step size, vector-field parameters and the FSAL cache;
their values and derivatives are checked against each other, including `h=0`.
The full adaptive driver still needs the later accepted-step replay adjoint.

At solve setup, native RK separates RHS work that depends only on parameters
from work that depends on time or the particle state. The parameter work runs
once per solve, so coil geometry is not rebuilt at every timestep. This generic
JAX partial evaluation leaves the field objects and RK formulas unchanged;
time, the complete state, and parameter derivatives remain live. It uses JAX's
partial-evaluation interface in the isolated `_prepare_rhs` helper in `rk.py`.

Native adaptive orbits default to `native_error_control="physics"`. The
embedded error is measured in separate Cartesian-position and velocity blocks:
`max(norm(dx)/L, norm(dv)/V) / (atol + rtol)`. The position reference
`orbit_length_scale=L` defaults to **1 metre**; choose a relevant device or
orbit length when this is unsuitable. `orbit_speed_scale=V` defaults to the
particle's total speed, so a parallel turning point or a zero Cartesian
velocity component does not shrink its error budget. Both tolerances apply to
these nondimensional physical blocks. These are **local estimated errors**;
they do not bound the accumulated trajectory error.

VMEC position errors compare the embedded endpoints after mapping to Cartesian
space, including the regular chart angle. Constant magnetic moment and failure
flags do not enter the norm. Fixed steps do not evaluate the error policy.
`native_error_control="standard"` restores componentwise scaling
`atol + rtol*max(abs(y_old), abs(y_trial))` with an RMS norm. The bare
`solve_rk` API keeps that generic default and accepts a separate `error_norm`
callable; [`essos/orbit_error.py`](essos/orbit_error.py) contains the physical
policy without a Diffrax dependency.

The native driver defaults to `native_controller="i"` and `rtol=atol=1e-9`.
Explicit tolerances override these defaults. Diffrax tracing keeps its existing
`1e-7` defaults. Select
`native_controller="pi"`, `native_pcoeff=.4`, `native_icoeff=.3` in `Tracing`
to compare with ESSOS's Diffrax PI coefficients. The standalone `solve_rk`
keywords are `controller="pi"`, `pcoeff=.4`, `icoeff=.3`. PI uses the previous
accepted error, safety factor .9, accepted growth limits [1, 10] and rejected
growth limits [.2, .9]. Fixed stepping ignores the controller. PI does not
change the embedded error or acceptance threshold.

Diffrax's ESSOS path keeps its existing componentwise RMS/PI configuration
(`PIDController` with `dcoeff=0`). Native controller and error-policy options
do not change Diffrax, Boris or collisions. Comparing step counts also requires
the same error norm, initial step and tolerances. Neither native method is
symplectic.

An optional `native_dense_output="optimized"` changes Dopri8's saved-time
polynomial weights. The standalone solver keyword is `dense_output="optimized"`.
It reuses the existing RK stages, with **no additional field/RHS evaluations**,
and prepares six polynomial coefficient vectors once per trial for all saves
in that step. GC and Lorentz use the same implementation. The integration
formula, embedded estimator, controller and event-root interpolation are
unchanged; Dopri5 retains its standard quartic for either setting.

The weights in [`essos/rk_dense.py`](essos/rk_dense.py) preserve the continuous
RK conditions through order five and endpoint values/slopes. They reduce the
integrated order-six/seven B-series error norm without increasing the formal
dense order. The calculation is independent of any orbit data and can be
reproduced with `python -m examples.benchmarks.derive_rk_dense_output`.
Accuracy gains depend on the ODE, so `native_dense_output="standard"` remains
the default. This option improves saved output and does not impose a global
error bound from `rtol`/`atol`.

To compare standard I, optimized-output I, native PI and Diffrax PI together:

```bash
python -m examples.benchmarks.compare_orbit_controllers --compare-dense-output --repeats 9 --output examples/benchmarks/results/orbit_dense_output_20261015.json
```

The comparison verifies unchanged per-particle step counts and terminal states
within floating-point roundoff, and records position/velocity errors and warm
timings separately from compilation.
The [2026-10-15 dense-output comparison](examples/benchmarks/results/orbit_dense_output_20261015.json)
and [CSV](examples/benchmarks/results/orbit_dense_output_20261015.csv) preserve
the nine-repeat QA-coil run at `rtol=atol=1e-9`. Dopri8 Lorentz saved position
and velocity errors fell by about sevenfold at essentially unchanged cost;
GC gains were smaller. All per-particle step counts matched the standard I run.

For higher continuous order on both methods, select
`native_dense_output="higher_order"` in `Tracing`, or `dense_output="higher_order"`
in `solve_rk`. Dopri5 adds two output-only RHS evaluations to increase dense
order from four to five. Dopri8 adds three to increase dense order from five
to seven. Both GC and Lorentz use the same implementation and original RHS.
The new stages never feed the orbit endpoint, embedded estimator, controller,
or FSAL cache. Event roots retain the standard interpolant and terminal state.

Output stages run once per accepted step containing an interior save; all
saves in that step reuse the polynomial. Rejected trials and steps with only
endpoint saves skip them. Under particle batching, masked particles may share
the evaluations required by another particle in the same batch. Temporary
stage/coefficient storage is reused each step; no additional history is kept.
The pure output map supports forward/reverse AD, including derivatives in
step size and parameters. This does not implement the adaptive driver's later
replay adjoint. Non-finite extra evaluations return a failure status without
retrying the accepted orbit step.

The coefficients in [`essos/rk_dense_high.py`](essos/rk_dense_high.py) are
derived independently of orbit data by
`python -m examples.benchmarks.derive_rk_higher_dense_output`. Tests verify every
rooted-tree condition through the claimed continuous order, nonlinear
convergence, ordinary JAX derivatives, evaluation counts, and unchanged orbit
steps/events. Higher dense order reduces interpolation error; accumulated
integration error remains. Lorentz Dopri5 was already close to that floor in
the QA benchmark, so extra interpolation work brings little benefit there.
The default remains `"standard"`, with the I-controller at `rtol=atol=1e-9`.
The construction uses the additional-stage bootstrapping strategy described
in [Shampine and Jay, Continuous Extensions](https://homepage.math.uiowa.edu/~ljay/publications.dir/ContinuousExtensions.pdf).

To compare both output options, native PI and Diffrax PI, including compiler
buffer-memory estimates for the forward solve:

```bash
python -m examples.benchmarks.compare_orbit_controllers --compare-dense-output --compare-higher-order --repeats 9 --output examples/benchmarks/results/orbit_higher_dense_output_20261015.json
```

The [saved higher-order comparison](examples/benchmarks/results/orbit_higher_dense_output_20261015.json)
and [CSV](examples/benchmarks/results/orbit_higher_dense_output_20261015.csv)
contain eight QA-coil particles, 64 saves, nine interleaved warm repeats, and
`rtol=atol=1e-9`. Below, arrows compare native I standard to higher-order output.
Errors are maximum absolute saved component errors against a 1024-fixed-step
Diffrax Dopri8 reference; GC velocity is the parallel component.

| Orbit/method | Warm median (ms) | Position error (m) | Velocity error (m/s) |
| --- | ---: | ---: | ---: |
| GC Dopri5 | 29.61 → 41.01 | 5.68e-9 → 1.91e-9 | 1.45e-2 → 1.10e-3 |
| GC Dopri8 | 35.11 → 44.63 | 1.86e-9 → 1.17e-10 | 3.53e-3 → 2.39e-4 |
| Lorentz Dopri5 | 29.15 → 40.81 | 9.77e-11 → 9.77e-11 | 3.05e-3 → 2.86e-3 |
| Lorentz Dopri8 | 13.49 → 15.98 | 3.66e-9 → 5.57e-11 | 9.91e-2 → 1.53e-3 |

Every particle retained its orbit step counts and terminal state to roundoff.
The compiler's argument/output/temporary buffer estimate increased by only
192 bytes for the whole eight-particle GC batch and 128 bytes for Lorentz,
for either method. These are forward executable estimates, not process RSS
or future adjoint memory. The suite passed 239 solver/tracing tests.

For an isolated adaptive controller comparison (native I, native PI and
Diffrax PI with matched physical budgets and PI settings), run:

```bash
python -m examples.benchmarks.compare_orbit_controllers
```

It records per-particle accepted/rejected counts, warm timings and maximum
saved position/velocity errors in
`examples/benchmarks/output/orbit_controller_comparison.json`.
The preserved [2026-10-15 baseline](examples/benchmarks/results/orbit_controller_baseline_20261015.json)
and its [CSV table](examples/benchmarks/results/orbit_controller_baseline_20261015.csv)
record the eight-particle QA-coil comparison at `rtol=atol=1e-9`.

To compare native I at tighter tolerances alongside the baseline controllers:

```bash
python -m examples.benchmarks.compare_orbit_controllers --native-i-tolerances 1e-10 1e-9 --native-i-tolerances 1e-10 1e-10 --output examples/benchmarks/results/orbit_controller_tight_i_20261015.json
```

Each extra pair is `rtol atol`. With the physical norm, changing only rtol
reduces the budget from `2e-9` to `1.1e-9`; changing both to `1e-10` gives
`2e-10`. JSON and CSV records include both tolerances, the budget, timings,
per-particle counts and maximum saved position/velocity errors.
The [tighter native I results](examples/benchmarks/results/orbit_controller_tight_i_20261015.json)
and [CSV table](examples/benchmarks/results/orbit_controller_tight_i_20261015.csv)
include both variants and baseline controllers timed together in the same run.

For synchronized forward timings, run from the repository root:

```bash
python -m examples.benchmarks.benchmark_orbit_integrators --repeats 7
```

This compares native and Diffrax Dopri5/Dopri8 using the bundled QA coils,
reports compilation separately from repeated execution, checks trajectory
errors against a tighter reference, and records step counts in
`examples/benchmarks/output/orbit_integrator_timings.json`. Fixed stepping
uses the same number of steps; adaptive stepping uses each backend's ESSOS
controller. By default it applies the same physical norm to both backends;
`--error-control standard` reproduces the earlier RMS comparison. The benchmark
uses the existing field classes and GC/Lorentz RHS for both backends. It reports maximum
position error in metres and velocity error in m/s as well as normalized by
the initial total speed. The benchmark
measures compiled batched orbit integration and
excludes `Tracing` construction, postprocessing and VMEC boundary events.

For coil fields, GC quantities share the analytic spatial Jacobian of the
sampled Biot--Savart kernel. This remains differentiable in position, coil
shape and currents, and is used by both native and Diffrax solvers. The field
constructor, coil representation and PyTree structure are unchanged.

To isolate stage costs, run `python -m examples.benchmarks.profile_orbit_stages`.
This compares mathematical kernel variants in the same fixed-step scan, with
identical FSAL reuse and no saving, events or adaptive controller. It checks
final-state agreement and records warm timings and compilation sizes in JSON.
The RK map stays separate from acceptance, error scaling and output handling
so a later AD rule can replay accepted steps without differentiating retries.

For paired measurements of the previous and current native time-step map, run
`python -m examples.benchmarks.benchmark_native_stage_modes --derivatives`.
It holds the native error policy/controller fixed, checks trajectory agreement,
and reports complete solve timings separately from one-step reverse-derivative
timings. Field classes and RHS calculations are held fixed.
`--comparison rhs_setup` instead compares the driver before/after invariant
RHS preparation, with Diffrax alongside and the same field object for all three.
`python -m examples.benchmarks.audit_orbit_error_control` audits individual
native adaptive trials over a tolerance sweep. It checks acceptance, compares
the native and Diffrax maps on identical steps, and separates endpoint and
interpolated-output errors using independently refined Diffrax references.
Its host callbacks are diagnostic; use the timing benchmarks for performance.
`--comparison stages` reproduces the rejected stage-loop experiment;
the production RK method remains the same for both GC and Lorentz.

`python -m examples.benchmarks.profile_rk_candidates --kernels sparse trial_production diffrax --repeats 7`
compares the public derivative-based RK map, production increment-based trial
and Diffrax in an identical fixed-step scan. It checks stage values and
forward/reverse derivatives before timing. `--one-step --calls-per-block 7`
instead measures compiled single-step calls with an existing FSAL input cache.
Every kernel receives the original field object without a geometry snapshot.
Other named candidates are experiments and do not modify production.

`python -m examples.benchmarks.benchmark_native_stage_modes --comparison execution --candidate trial_production --repeats 11`
compares the legacy derivative-storage map and production increment-storage
map in the current full driver, with Diffrax alongside. Fixed mode uses 256
steps for every particle/backend; adaptive counts and errors are also recorded.

## Tracing to the wall

Inside the last closed flux surface (LCFS), guiding centres move in VMEC
coordinates. An orbit that crosses the LCFS continues in Cartesian coordinates
in a field you supply for the outside (coils, an mgrid, or a VMEX
`VmecExtender`) until it strikes the wall or comes back in. Below, 3 keV
protons in a VMEX free-boundary equilibrium of the Landreman-Paul QA coils,
with a wall 3 cm outside the LCFS.

![Guiding centres traced through the LCFS to a wall](docs/readme_wall.png)

```python
vmec = vmex.essos_vmec_field(wout)                     # or Vmec("wout_*.nc")
outside = vmex.VmecExtender.from_wout(wout, external_field=coil_grid)
tracing = Tracing(field=vmec, model="GuidingCenterAdaptative", particles=particles,
                  maxtime=4e-5, exterior_field=outside, wall=0.03)  # gap [m] or a surface
print(tracing.wall_hits.mean(), tracing.wall_positions[tracing.wall_hits])
```

The full script is
[`trace_particles_vmec_to_wall.py`](examples/particle_tracing/trace_particles_vmec_to_wall.py);
the figure comes from `python docs/make_readme_figures.py wall`.

## Boozer-coordinate tracing

Transform a VMEC equilibrium with [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax),
then trace guiding centres in an axis-regular chart with optional Monte Carlo
collisions. `method="dopri8"` (default), `"dopri5"` or `"rk4"` take fixed
steps; `"adaptive"` (Dopri5) and `"adaptive8"` (Dopri8) control each
particle's step to `tolerance`. This tracer provides forward diagnostics.

Install the transform extra with `pip install 'essos[booz]'`.

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
field = BoozerField.from_booz_xform(booz, psi0=-phi_edge / (2 * np.pi))

n = 1000
speed = np.sqrt(2 * 3.5e6 * ONE_EV / ALPHA_PARTICLE_MASS)
result = trace_boozer(field, s=np.full(n, 0.25), theta=np.random.uniform(0, 2*np.pi, n),
                      zeta=np.random.uniform(0, 2*np.pi, n), pitch=np.random.uniform(-1, 1, n),
                      speed=speed, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                      tmax=1e-2)      # species=BackgroundSpecies(...) adds collisions
print(result.lost.mean(), result.loss_fractions())
```

To select the shared native RK driver, omit `method` and pass
`integrator="essos_rk45"` or `"essos_dopri8"`:

```python
result = trace_boozer(
    field, s=[0.25], theta=[0.0], zeta=[0.0], pitch=[0.5],
    speed=speed, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
    tmax=1e-4, timestep=1e-8, integrator="essos_dopri8",
    adaptive=True, rtol=1e-9, atol=1e-9,
    native_controller="i", native_dense_output="standard",
)
```

Native stepping defaults to adaptive I control and standard interpolation;
`adaptive=False` selects fixed steps. `timestep` is the first adaptive trial
or fixed step, independent of `n_save`. PI and the optional dense-output modes
use the same native options as `Tracing`. Each orbit is solved over the full
horizon, with no controller resets at saves or progress boundaries. Legacy
interval compaction is unused in this path, and progress reports completion.
`result.step_stats[:, :]` contains trials/accepted/rejected counts and
`result.solver_status` contains native `RKStatus` codes. Losses are root-found
at `s=1`, later saves hold the crossing state, and failed solves remain failures.
Native Boozer stepping currently supports collisionless orbits; `species`
continues to select collisions on the existing legacy path.

The Boozer field provides `|B|` and profiles without a Cartesian position map.
Its native embedded error is
`max(norm(du,dw), abs(dzeta), abs(dv_parallel)/speed) / (rtol+atol)` in the
regular chart `u=sqrt(s)*cos(theta), w=sqrt(s)*sin(theta)`. The reference scales
are unit flux radius, one radian and birth speed. These are coordinate error
budgets, rather than the metre-based position budget used for VMEC. Angular
winding and parallel turning points do not change these scales. Native energy
diagnostics sample saved and terminal states. The existing
[`Boozer/VMEC comparison example`](examples/particle_tracing/trace_particles_boozer_vs_vmec.py)
selects the same native integrator for both fields.

`psi0` is minus VMEC's boundary toroidal flux over `2 pi`. A particle
is lost at `s = 1`; `result.loss_times` and `result.states` hold when and where.

Boozer tracing evaluates `|B| = sum [bc_mn(s) cos(m theta - n zeta) + bs_mn(s) sin(m theta - n zeta)]`
and its derivatives, together with the flux functions `G`, `I` and `iota`.

## Comparison with other codes

256 collisionless 3.5 MeV alphas per equilibrium, started at `s = 0.5` with
uniform Boozer angles and pitch, traced for 10 ms and lost at `s = 1`, in the
Landreman-Paul QA reactor-scale `wout`, the ARIES-CS `n3are` `wout`, and the
W7-X standard configuration scaled to the same minor radius (1.70 m) and
volume-averaged `|B|` (5.86 T). Apple M4 (load about 3 from other processes), 8 CPU
cores per code; compilation and field set-up excluded. DESC traces the first 32
alphas only, on `VMECIO.load(wout, L=24, M=8, N=8)` with the half-mesh lambda fix of
PlasmaControl/DESC#2352, and its times were taken with other DESC runs sharing the
machine. Run with
[`examples/cross_code_benchmark/benchmark.py`](examples/cross_code_benchmark/benchmark.py).

![Alpha losses against time in three equilibria](docs/cross_code_losses.png)

![Run time, energy error and fate agreement per code](docs/cross_code_bars.png)

| Equilibrium | Code | Method | Run time | Lost | Same fate as SIMPLE and FIRM3D | max \|E/E0 - 1\| |
|---|---|---|---|---|---|---|
| QA | ESSOS `trace_boozer` | Dopri8, `dt = 2.1e-07 s` (default) | 19 s | 7.0 ± 1.6% | 100.0% | 1e-05 |
|  | SIMPLE | symplectic, defaults | 17 s | 7.0 ± 1.6% | - | not reported |
|  | FIRM3D | RK45, tol 1e-9, `gc_noK` | 48 s | 7.4 ± 1.6% | - | 4e-05 |
|  | SIMSOPT | RK45, tol 1e-9, `gc_noK` | 17 s | 8.2 ± 1.7% | 85.4% | 5e-05 |
|  | DESC | Tsit5, tol 1e-7, `L = 24, M = N = 8` | 3859 s | 6.2 ± 4.3% (32) | 100.0% | 2e-05 |
| ARIES-CS | ESSOS `trace_boozer` | Dopri8, `dt = 1.1e-07 s` (default) | 36 s | 22.7 ± 2.6% | 98.3% | 1e-05 |
|  | SIMPLE | symplectic, defaults | 24 s | 23.4 ± 2.6% | - | not reported |
|  | FIRM3D | RK45, tol 1e-9, `gc_noK` | 146 s | 21.5 ± 2.6% | - | 5e-03 |
|  | SIMSOPT | RK45, tol 1e-9, `gc_noK` | 64 s | 23.8 ± 2.7% | 93.7% | 1e-04 |
|  | DESC | Tsit5, tol 1e-7, `L = 24, M = N = 8` | 23150 s | 34.4 ± 8.4% (32) | 90.3% | 7e-05 |
| W7-X | ESSOS `trace_boozer` | Dopri8, `dt = 1.4e-07 s` (default) | 26 s | 25.8 ± 2.7% | 97.1% | 7e-04 |
|  | SIMPLE | symplectic, defaults | 191 s | 27.3 ± 2.8% | - | not reported |
|  | FIRM3D | RK45, tol 1e-9, `gc_noK` | 191 s | 27.0 ± 2.8% | - | 3e-03 |
|  | SIMSOPT | RK45, tol 1e-9, `gc_noK` | 111 s | 28.9 ± 2.8% | 99.6% | 1e-03 |
|  | DESC | Tsit5, tol 1e-7, `L = 24, M = N = 8` | 21807 s | 40.6 ± 8.7% (32) | 89.7% | 2e-04 |

The fate agreement is the share of particles lost or confined as in SIMPLE and
FIRM3D, over the particles on which those two agree. The bar plot scales DESC's
time from 32 to 256 particles. ESSOS, SIMSOPT and FIRM3D use the same 32x32
`booz_xform` spectrum (ESSOS keeps modes above its default `1e-6` of `B00` and,
above 48 modes, evaluates them from a table in the Boozer angles); SIMPLE reads
the `wout` directly and DESC fits it. ± is the binomial standard error.

## Tracing notes

- **VMEC magnetic axis.** VMEC guiding centres are integrated in
  `sqrt(s) (cos theta, sin theta)`, which is regular on the axis, so orbits
  cross it; trajectories are returned in `(s, theta, phi, ...)` with `theta`
  in `[0, 2 pi)`.
- **LCFS and wall.** A VMEC guiding-centre trace stops at the exact LCFS
  crossing (`tracing.lcfs_times`, `lcfs_positions`, `lcfs_energies`), found by
  root finding. Pass `exterior_field=` (ESSOS coils, a VMEX `VmecExtender` or
  `MgridField`, or any batched `xyz -> B`) and `wall=` (a `SurfaceClassifier`
  or a signed-distance callable, positive inside) to follow the orbit outside
  in Cartesian coordinates until it strikes the wall (`wall_times`,
  `wall_positions`, `wall_energies`) or comes back inside, where it returns to
  flux coordinates (`returns`). `tracing.status` gives each outcome
  (`essos.dynamics.VMEC_STATUS`); `trajectories_xyz` and `region` hold the
  whole orbit. A non-finite step ends that orbit as `failed`, with its time.
  Loss fractions use the exact wall, or LCFS, times. Collisions are switched
  off outside the LCFS. Supplying `condition` replaces these events; it is
  evaluated on `(s, theta, phi, ...)`. See
  [`examples/particle_tracing/trace_particles_vmec_to_wall.py`](examples/particle_tracing/trace_particles_vmec_to_wall.py).
- **Stopping coil-field traces.** Pass `stopping_criteria=LevelsetStoppingCriterion(...)`
  to end Cartesian traces once they leave a prescribed distance from a surface,
  and read the per-line mask from `tracing.boundary_hits`.
- **Step budget.** `max_steps` (default `1_000_000`) bounds every orbit solve,
  so a trace that cannot finish returns instead of running unbounded.
- **Progress bars** are off by default; pass `progress=True` when interactive.
  With `particle_batch_size=n` particles are traced `n` at a time and the bar
  counts completed particles (batches are bypassed under `jax.jit`/`grad`).
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
| [`cross_code_benchmark`](examples/cross_code_benchmark) | Alpha losses against SIMSOPT, FIRM3D and SIMPLE |
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
