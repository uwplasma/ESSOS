"""Shared setup for the two accepted-step reverse-AD coil examples.

Run an objective-specific entry point with ``python -m``; importing this module
does not trace particles or start an optimization. All particles are traced
together. Birth parameters, reference axis and classifier remain fixed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, ONE_EV
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.particle_losses import mean_final_radius
from essos.surfaces import BdotN_over_B, SurfaceClassifier, SurfaceRZFourier

ROOT = Path(__file__).resolve().parents[2]
COILS = ROOT / "examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
WOUT = ROOT / "examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"


def fixed_surface(wout, *, ntheta=64, nphi=64, range_torus="full torus"):
    """Scaled QA reference shape, held fixed during coil optimization."""
    surface = SurfaceRZFourier.from_wout_file(str(wout))
    scale = surface.rc[0]
    return SurfaceRZFourier(surface.rc / scale, surface.zs / scale,
        surface.nfp, surface.mpol, surface.ntor, ntheta=ntheta, nphi=nphi,
        range_torus=range_torus)


def fixed_classifier(wout):
    """Fixed reference surface classifier used by the orbit benchmarks.

    This is a geometric confinement target, not an equilibrium recalculated
    after a coil update. Periodic cubic angular interpolation also matches
    derivatives at the toroidal seam. Distances are positive inside.
    """
    return SurfaceClassifier(fixed_surface(wout), h=.03, padding=.16,
                             angular_interpolation="periodic_cubic")


def coil_penalties(field, surface, options):
    """Scalar penalties matching optimize_coils_vmec_surface.py.

    Length and pointwise curvature use the mean positive excess above their
    targets. The normal-field term is the sum of abs(B dot n / |B|) on the
    fixed surface. Zero weights omit the corresponding calculation entirely.
    """
    return dict(
        length_penalty=jnp.mean(jnp.maximum(field.coils.length - options.length_target, 0.))
            if options.length_weight else jnp.array(0.),
        curvature_penalty=jnp.mean(jnp.maximum(field.coils.curvature - options.curvature_target, 0.))
            if options.curvature_weight else jnp.array(0.),
        bdotn_penalty=jnp.sum(jnp.abs(BdotN_over_B(surface, field)))
            if options.bdotn_weight else jnp.array(0.))


def build_objective(options, objective):
    """Return the coil-only scalar objective with saved trajectories as aux."""
    base = Coils.from_json(str(options.coils))
    theta0 = jnp.asarray(base.dofs)
    initial_field = BiotSavart(base)
    axis = (float(initial_field.r_axis), float(initial_field.z_axis))
    if options.length_target is None:
        options.length_target = 1.05 * float(jnp.max(base.length))
    if options.curvature_target is None:
        options.curvature_target = 1.05 * float(jnp.max(base.curvature))
    classifier = fixed_classifier(options.wout) if objective == "loss_fraction" else None
    surface = fixed_surface(options.wout, ntheta=options.surface_ntheta,
        nphi=options.surface_nphi, range_torus="half period") if options.bdotn_weight else None
    xyz = jnp.column_stack((jnp.linspace(1.23, 1.27, options.particles),
                            jnp.zeros((options.particles, 2))))
    pitches = jnp.linspace(.35, .65, options.particles)
    model = "GuidingCenterAdaptative" if options.model == "gc" else "FullOrbitAdaptative"
    integrator = "essos_dopri8" if options.method == "dopri8" else "essos_rk45"

    def evaluate(delta):
        # Coils.dofs already applies the public curve/current normalization.
        # A dimensionless bounded displacement makes max_change an explicit
        # absolute limit in those DOFs, including initially zero coefficients.
        field = BiotSavart(base.with_dofs(theta0 + options.max_change * delta))
        birth = Particles(xyz, initial_vparallel_over_v=pitches,
            mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
            energy=options.energy_ev * ONE_EV,
            field=field if options.model == "lorentz" else None,
            phase_angle_full_orbit=jnp.pi)
        trace = Tracing(field=field, particles=birth, model=model,
            integrator=integrator, native_adjoint="accepted",
            native_adjoint_checkpoint_steps=options.checkpoint_steps,
            native_adjoint_max_accepted_steps=options.accepted_capacity,
            native_controller="i", native_dense_output="standard",
            rtol=options.tolerance, atol=options.tolerance,
            maxtime=options.horizon, timestep=options.horizon / 2,
            times_to_trace=options.saves, max_steps=options.max_steps,
            progress=False)
        if objective == "loss_fraction":
            metric = trace.loss_fraction_classifier_differentiable(
                classifier, softness=options.softness)
        else:
            metric = mean_final_radius(trace.trajectories, *axis)
        components = dict(particle_metric=metric,
            **coil_penalties(field, surface, options),
            regularization_penalty=jnp.mean(delta**2))
        cost = (options.particle_weight * metric
            + options.length_weight * components["length_penalty"]
            + options.curvature_weight * components["curvature_penalty"]
            + options.bdotn_weight * components["bdotn_penalty"]
            + options.regularization * components["regularization_penalty"])
        return cost, (components, trace.trajectories)

    return base, theta0, axis, xyz, pitches, evaluate


def run(objective, argv=None):
    if objective not in ("loss_fraction", "final_radius"):
        raise ValueError("unknown particle objective")
    parser = argparse.ArgumentParser(description=(
        "Optimize coil geometry and currents using native accepted-step reverse AD. "
        + ("Minimize the mean maximum sigmoid outside probability."
           if objective == "loss_fraction" else "Minimize mean final minor radius in metres.")))
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--coils", type=Path, default=COILS)
    parser.add_argument("--wout", type=Path, default=WOUT, help="Fixed scaled reference shape for loss fraction and B dot n / |B|")
    parser.add_argument("--model", choices=("gc", "lorentz"), default="gc")
    parser.add_argument("--method", choices=("dopri5", "dopri8"), default="dopri8")
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--horizon", type=float, help="Seconds; default GC 20 us, Lorentz 2 us")
    parser.add_argument("--saves", type=int, default=32)
    parser.add_argument("--energy-ev", type=float, default=4000.)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--checkpoint-steps", type=int, default=16)
    parser.add_argument("--accepted-capacity", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--max-change", type=float, default=.01,
                        help="Absolute displacement bound in each public normalized coil DOF")
    parser.add_argument("--regularization", type=float, default=0.,
                        help="Optional mean-square bounded-displacement penalty")
    parser.add_argument("--particle-weight", type=float, default=1.)
    parser.add_argument("--length-weight", type=float, default=1., help="Weight of mean positive coil-length excess")
    parser.add_argument("--curvature-weight", type=float, default=1., help="Weight of mean positive pointwise curvature excess")
    parser.add_argument("--bdotn-weight", type=float, default=1., help="Weight of summed abs(B dot n / |B|) on the fixed surface")
    parser.add_argument("--length-target", type=float, help="Metres; default 1.05 times the largest initial coil length")
    parser.add_argument("--curvature-target", type=float, help="1/metre; default 1.05 times the largest initial pointwise curvature")
    parser.add_argument("--surface-ntheta", type=int, default=30, help="Poloidal points for the normal-field objective")
    parser.add_argument("--surface-nphi", type=int, default=30, help="Toroidal points over a half field period for the normal-field objective")
    parser.add_argument("--softness", type=float, default=10., help="Loss sigmoid slope in 1/metre")
    options = parser.parse_args(argv)
    if options.horizon is None:
        options.horizon = 2e-5 if options.model == "gc" else 2e-6
    positive = (options.horizon, options.energy_ev, options.tolerance,
                options.max_change, options.softness)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        parser.error("horizon, energy, tolerance, max-change and softness must be finite and positive")
    if min(options.particles, options.checkpoint_steps, options.accepted_capacity,
           options.max_steps, options.maxiter) < 1 or min(options.saves,
               options.surface_ntheta, options.surface_nphi) < 2:
        parser.error("positive particle/step/iteration counts and at least two saves and surface points per direction required")
    weights = (options.particle_weight, options.length_weight,
               options.curvature_weight, options.bdotn_weight, options.regularization)
    if any(not np.isfinite(value) or value < 0 for value in weights) or not any(weights):
        parser.error("weights must be finite and nonnegative, with at least one nonzero weight")
    targets = (options.length_target, options.curvature_target)
    if any(value is not None and (not np.isfinite(value) or value < 0) for value in targets):
        parser.error("length and curvature targets must be finite and nonnegative")
    if options.output.exists():
        parser.error("output directory already exists; choose a new path")
    options.output.mkdir(parents=True)
    print(f"Building {objective}, {options.model}, {options.particles} particles; coil DOFs only", flush=True)
    base, theta0, axis, xyz, pitches, objective_fn = build_objective(options, objective)
    print(f"Length target {options.length_target:.6g} m; curvature target "
          f"{options.curvature_target:.6g} /m; particle/length/curvature/Bn weights "
          f"{weights[:4]}", flush=True)
    evaluate = jax.jit(jax.value_and_grad(objective_fn, has_aux=True))
    history = []
    cached = None

    with (options.output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("evaluation", "cost", "particle_metric",
            "length_penalty", "curvature_penalty", "bdotn_penalty", "regularization_penalty",
            "gradient_l2", "max_abs_displacement", "seconds"))
        writer.writeheader()

        def value_gradient(delta):
            nonlocal cached
            delta = np.asarray(delta, dtype=np.float64)
            if cached is not None and np.array_equal(delta, cached[0]):
                return cached[1], cached[2]
            start = perf_counter()
            (value, (components, trajectories)), gradient = jax.block_until_ready(evaluate(jnp.asarray(delta)))
            value = float(value)
            components = {name: float(term) for name, term in components.items()}
            metric = components["particle_metric"]
            gradient, trajectories = np.asarray(gradient), np.asarray(trajectories)
            if not (np.isfinite(value) and all(np.isfinite(list(components.values())))
                    and np.isfinite(gradient).all() and np.isfinite(trajectories).all()):
                raise RuntimeError("nonfinite orbit, objective or gradient")
            row = dict(evaluation=len(history), cost=value, **components,
                gradient_l2=float(np.linalg.norm(gradient)),
                max_abs_displacement=float(options.max_change * np.max(np.abs(delta))),
                seconds=perf_counter() - start)
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(f"eval {row['evaluation']:3d}: metric={metric:.10g}, "
                  f"cost={value:.10g}, |grad|={row['gradient_l2']:.4g}, "
                  f"{row['seconds']:.2f} s", flush=True)
            cached = (delta.copy(), value, gradient, components, trajectories)
            return value, gradient

        delta0 = np.zeros(theta0.size)
        initial_value, initial_gradient = value_gradient(delta0)
        initial_components, initial_trajectories = cached[3], cached[4].copy()
        result = minimize(value_gradient, delta0, method="L-BFGS-B", jac=True,
            bounds=[(-1., 1.)] * theta0.size,
            options=dict(maxiter=options.maxiter, maxls=12, ftol=1e-12, gtol=1e-10))
        final_value, final_gradient = value_gradient(result.x)
        final_components, final_trajectories = cached[3], cached[4]

    final_dofs = theta0 + options.max_change * jnp.asarray(result.x)
    base.with_dofs(final_dofs).to_json(str(options.output / "optimized_coils.json"))
    np.savez_compressed(options.output / "orbits.npz",
        times=np.linspace(0., options.horizon, options.saves),
        initial_trajectories=initial_trajectories, final_trajectories=final_trajectories,
        initial_coil_dofs=np.asarray(theta0), final_coil_dofs=np.asarray(final_dofs),
        birth_xyz=np.asarray(xyz), birth_pitch=np.asarray(pitches))
    report = dict(objective=objective, metric_units="probability" if objective == "loss_fraction" else "metres",
        initial_cost=initial_value, final_cost=final_value,
        initial_particle_metric=initial_components["particle_metric"],
        final_particle_metric=final_components["particle_metric"],
        initial_components=initial_components, final_components=final_components,
        loss_weights=dict(particle_metric=options.particle_weight, length_penalty=options.length_weight,
            curvature_penalty=options.curvature_weight, bdotn_penalty=options.bdotn_weight,
            regularization_penalty=options.regularization),
        loss_definitions=dict(length_penalty="mean(max(coil_length_m - length_target_m, 0))",
            curvature_penalty="mean(max(pointwise_curvature_per_m - curvature_target_per_m, 0))",
            bdotn_penalty="sum(abs(B dot unitnormal / |B|)) over fixed half-period surface mesh",
            regularization_penalty="mean(delta**2)"),
        initial_gradient_l2=float(np.linalg.norm(initial_gradient)),
        final_gradient_l2=float(np.linalg.norm(final_gradient)),
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        iterations=int(result.nit), function_evaluations=int(result.nfev),
        recorded_evaluations=len(history), coil_dof_names=list(base.dof_names),
        reference_axis_r_z_m=axis, backend=jax.default_backend(), jax=jax.__version__,
        settings={name: str(value.resolve()) if isinstance(value, Path) else value
                  for name, value in vars(options).items()},
        derivative_policy="Accepted-step reverse AD with frozen adaptive times; segment checkpoints; full population vmap",
        birth_policy="Fixed GC birth XYZ, pitch, energy and gyrophase; Lorentz conversion remains field-dependent",
        target_policy="Fixed initial reference axis and fixed scaled QA surface; no hard wall stopping",
        gradient_coordinates="Dimensionless delta; coil DOFs = initial DOFs + max_change * delta",
        timing_scope="First evaluation includes compilation; later evaluations reuse compiled value_and_grad")
    (options.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Cost {initial_value:.10g} -> {final_value:.10g}; particle metric "
          f"{initial_components['particle_metric']:.10g} -> {final_components['particle_metric']:.10g}; {result.message}; "
          f"saved {options.output}", flush=True)

