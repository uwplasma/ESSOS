"""Shared setup for accepted-step reverse-AD coil optimization examples.

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

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, ONE_EV
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.particle_losses import mean_final_radius, minor_radius
from essos.surfaces import BdotN_over_B, SurfaceClassifier, SurfaceRZFourier

ROOT = Path(__file__).resolve().parents[2]
COILS = ROOT / "examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
WOUT = ROOT / "examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
PARTICLE_OBJECTIVES = ("loss_fraction", "final_radius", "radial_increment")


def save_optimization_plots(output, initial_coils, optimized_coils,
                            initial_trajectories, final_trajectories, *, axis,
                            count=10, seed=42):
    """Save matched 3D and R-Z views of a random subset of stored trajectories.

    R-Z panels are poloidal projections across all toroidal angles, not
    Poincare sections. No additional particle tracing or random births occur.
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    before = np.asarray(initial_trajectories)[..., :3]
    after = np.asarray(final_trajectories)[..., :3]
    if before.ndim != 3 or before.shape != after.shape or not len(before) or count < 1:
        raise ValueError("plotting requires matching nonempty trajectories and a positive particle count")
    indices = np.random.default_rng(seed).choice(len(before), min(count, len(before)), replace=False)
    curves = [np.asarray(coils.gamma) for coils in (initial_coils, optimized_coils)]
    xyz = np.concatenate([curves[0].reshape(-1,3), curves[1].reshape(-1,3),
                          before[indices].reshape(-1,3), after[indices].reshape(-1,3)])
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    spans = np.maximum(hi-lo, 1e-6)
    rz = [np.stack((np.hypot(paths[indices,...,0], paths[indices,...,1]),
                    paths[indices,...,2]), axis=-1) for paths in (before, after)]
    rz_all = np.concatenate([values.reshape(-1,2) for values in rz] + [np.asarray(axis)[None,:]])
    rz_lo, rz_hi = rz_all.min(axis=0), rz_all.max(axis=0)
    rz_pad = .08*np.maximum(rz_hi-rz_lo, 1e-6)
    figure = Figure(figsize=(14,11))
    FigureCanvasAgg(figure)  # Save headlessly without changing the user's pyplot backend.
    axes3d = [figure.add_subplot(2,2,i+1, projection="3d") for i in range(2)]
    axesrz = [figure.add_subplot(2,2,i+3) for i in range(2)]
    for column, (title, paths) in enumerate((("Initial", before), ("Optimized", after))):
        ax3d, axrz = axes3d[column], axesrz[column]
        for curve in curves[column]:
            closed = np.concatenate((curve, curve[:1]))
            ax3d.plot(*closed.T, color="0.45", linewidth=1., alpha=.65)
        for rank, index in enumerate(indices):
            color = colormaps["tab10"](rank % 10)
            points = paths[index]
            ax3d.plot(*points.T, color=color, linewidth=1.3)
            ax3d.scatter(*points[0], color=color, s=12)
            axrz.plot(*rz[column][rank].T, color=color, linewidth=1.3, label=f"Particle {index}")
            axrz.scatter(*rz[column][rank,0], color=color, s=12)
        ax3d.set(title=f"{title} coils and trajectories", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        ax3d.set_xlim(lo[0]-.05*spans[0], hi[0]+.05*spans[0])
        ax3d.set_ylim(lo[1]-.05*spans[1], hi[1]+.05*spans[1])
        ax3d.set_zlim(lo[2]-.05*spans[2], hi[2]+.05*spans[2])
        ax3d.set_box_aspect(spans)
        ax3d.view_init(elev=25, azim=-55)
        axrz.scatter(*axis, marker="+", color="black", s=40, zorder=4)
        axrz.set(title=f"{title}: R-Z poloidal projection", xlabel="R [m]", ylabel="Z [m]",
                 xlim=(rz_lo[0]-rz_pad[0],rz_hi[0]+rz_pad[0]),
                 ylim=(rz_lo[1]-rz_pad[1],rz_hi[1]+rz_pad[1]))
        axrz.set_aspect("equal", adjustable="box")
        axrz.grid(alpha=.25)
    figure.suptitle(f"Coil optimization: {len(indices)} sampled particle trajectories")
    figure.legend(*axesrz[0].get_legend_handles_labels(), loc="lower center",
                  ncol=min(5,len(indices)), frameon=False)
    figure.subplots_adjust(left=.07, right=.97, bottom=.10, top=.94, hspace=.25, wspace=.20)
    paths = [Path(output)/f"orbit_comparison.{suffix}" for suffix in ("png", "svg")]
    try:
        for path in paths:
            figure.savefig(path, dpi=160)
    finally:
        figure.clear()
    return dict(files=[path.name for path in paths], seed=seed, requested=count,
        particle_indices=indices.tolist(),
        selection="Same random subset of the optimization population before and after optimization",
        poloidal_view="R-Z projection across all toroidal angles")


def radial_increment_cost(trajectories, axis_r, axis_z):
    """Reference full-orbit metric, with Cartesian/time indexing corrected.

    This is sum_i(mean_j(r[i,j+1]-r[i,j]))**2, without division by dt.
    It depends on save count and measures net drift, not oscillatory excursions.
    """
    radius = minor_radius(trajectories, axis_r, axis_z)
    return jnp.sum(jnp.mean(jnp.diff(radius, axis=1), axis=1)**2)


def fixed_surface(wout, *, ntheta=64, nphi=64, range_torus="full torus", major_radius=1.):
    """Scaled QA reference shape, held fixed during coil optimization."""
    surface = SurfaceRZFourier.from_wout_file(str(wout))
    scale = surface.rc[0]
    return SurfaceRZFourier(surface.rc * (major_radius / scale), surface.zs * (major_radius / scale),
        surface.nfp, surface.mpol, surface.ntor, ntheta=ntheta, nphi=nphi,
        range_torus=range_torus)


def fixed_classifier(wout, *, major_radius=1.):
    """Fixed reference surface classifier used by the orbit benchmarks.

    This is a geometric confinement target, not an equilibrium recalculated
    after a coil update. Periodic cubic angular interpolation also matches
    derivatives at the toroidal seam. Distances are positive inside.
    """
    return SurfaceClassifier(fixed_surface(wout, major_radius=major_radius),
                             h=.03 * major_radius, padding=.16 * major_radius,
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
    if objective not in PARTICLE_OBJECTIVES:
        raise ValueError("unknown particle objective")
    if options.coils is None:
        # Higher Fourier modes start at zero; optimization can deform circles.
        curves = CreateEquallySpacedCurves(n_curves=options.coil_count,
            order=options.coil_order, R=options.coil_major_radius, r=options.coil_minor_radius,
            n_segments=options.coil_segments, nfp=options.coil_nfp, stellsym=options.coil_stellsym,
            scaling_type="L2", scaling_factor=options.coil_scaling_factor)
        base = Coils(curves, jnp.full(options.coil_count, options.coil_current))
    else:
        base = Coils.from_json(str(options.coils))
    theta0 = jnp.asarray(base.dofs)
    initial_field = BiotSavart(base)
    axis = (float(initial_field.r_axis), float(initial_field.z_axis))
    if options.length_target is None:
        options.length_target = 1.05 * float(jnp.max(base.length))
    if options.curvature_target is None:
        options.curvature_target = 1.05 * float(jnp.max(base.curvature))
    classifier = fixed_classifier(options.wout, major_radius=options.reference_major_radius) if objective == "loss_fraction" else None
    surface = fixed_surface(options.wout, ntheta=options.surface_ntheta,
        nphi=options.surface_nphi, range_torus="half period",
        major_radius=options.reference_major_radius) if options.bdotn_weight else None
    if options.particle_setup == "axis":
        # Match the reference examples, including the duplicated 0/2pi birth.
        phi = jnp.linspace(0., 2*jnp.pi, options.particles)
        xyz = jnp.column_stack((axis[0]*jnp.cos(phi), axis[0]*jnp.sin(phi),
                               jnp.full_like(phi, axis[1])))
        pitches = jax.random.uniform(jax.random.PRNGKey(options.birth_seed), (options.particles,),
            minval=options.pitch_min, maxval=options.pitch_max)
    else:
        xyz = jnp.column_stack((jnp.linspace(1.23, 1.27, options.particles),
                                jnp.zeros((options.particles, 2))))
        pitches = jnp.linspace(.35, .65, options.particles)
    model = "GuidingCenterAdaptative" if options.model == "gc" else "FullOrbitAdaptative"
    integrator = "essos_dopri8" if options.method == "dopri8" else "essos_rk45"
    phi_axis = jnp.linspace(0., 2*jnp.pi, options.axis_field_points)
    axis_points = jnp.column_stack((axis[0]*jnp.cos(phi_axis), axis[0]*jnp.sin(phi_axis),
                                   jnp.full_like(phi_axis, axis[1])))

    def evaluate(delta):
        # Coils.dofs already applies the public curve/current normalization.
        # A dimensionless bounded displacement makes max_change an explicit
        # absolute limit in those DOFs, including initially zero coefficients.
        field = BiotSavart(base.with_dofs(theta0 + options.max_change * delta))
        birth = Particles(xyz, initial_vparallel_over_v=pitches,
            mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
            energy=options.energy_ev * ONE_EV,
            field=field if options.model == "lorentz" else None,
            phase_angle_full_orbit=options.gyrophase)
        trace = Tracing(field=field, particles=birth, model=model,
            integrator=integrator, native_adjoint="accepted",
            native_adjoint_checkpoint_steps=options.checkpoint_steps,
            native_adjoint_max_accepted_steps=options.accepted_capacity,
            native_controller="i", native_dense_output="standard",
            rtol=options.tolerance, atol=options.tolerance,
            maxtime=options.horizon,
            timestep=options.initial_step if options.initial_step is not None else options.horizon / 2,
            times_to_trace=options.saves, max_steps=options.max_steps,
            progress=False)
        if objective == "loss_fraction":
            metric = trace.loss_fraction_classifier_differentiable(
                classifier, softness=options.softness)
        elif objective == "final_radius":
            metric = mean_final_radius(trace.trajectories, *axis)
        else:
            metric = radial_increment_cost(trace.trajectories, *axis)
        components = dict(particle_metric=metric,
            **coil_penalties(field, surface, options),
            axis_field_penalty=jnp.abs(jnp.mean(jax.vmap(field.AbsB)(axis_points)) - options.axis_field_target)
                if options.axis_field_weight else jnp.array(0.),
            regularization_penalty=jnp.mean(delta**2))
        cost = (options.particle_weight * metric
            + options.length_weight * components["length_penalty"]
            + options.curvature_weight * components["curvature_penalty"]
            + options.bdotn_weight * components["bdotn_penalty"]
            + options.axis_field_weight * components["axis_field_penalty"]
            + options.regularization * components["regularization_penalty"])
        return cost, (components, trace.trajectories)

    return base, theta0, axis, xyz, pitches, evaluate


def run(objective, argv=None, *, defaults=None):
    """Run a selected native objective with optional editable script defaults."""
    defaults = {} if defaults is None else dict(defaults)
    if objective not in PARTICLE_OBJECTIVES:
        raise ValueError("unknown particle objective")
    parser = argparse.ArgumentParser(description=
        "Optimize coil geometry and currents using native accepted-step reverse AD.")
    parser.add_argument("--objective", choices=PARTICLE_OBJECTIVES, default=objective)
    parser.add_argument("--output", type=Path, required="output" not in defaults, help="New output directory")
    coil_source = parser.add_mutually_exclusive_group()
    coil_source.add_argument("--coils", type=Path, default=COILS, help="Initial coil JSON")
    coil_source.add_argument("--circular-coils", dest="coils", action="store_const", const=None,
        help="Create equally spaced planar circular coils")
    parser.add_argument("--coil-count", type=int, default=3, help="Independent coils per half field period")
    parser.add_argument("--coil-order", type=int, default=6)
    parser.add_argument("--coil-major-radius", type=float, default=7.74)
    parser.add_argument("--coil-minor-radius", type=float, default=4.5)
    parser.add_argument("--coil-current", type=float, default=1.84e7, help="Amperes per independent coil")
    parser.add_argument("--coil-nfp", type=int, default=2)
    parser.add_argument("--coil-stellsym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coil-segments", type=int, default=60)
    parser.add_argument("--coil-scaling-factor", type=float, default=0.)
    parser.add_argument("--wout", type=Path, default=WOUT, help="Fixed scaled reference shape for loss fraction and B dot n / |B|")
    parser.add_argument("--model", choices=("gc", "lorentz"), default="gc")
    parser.add_argument("--method", choices=("dopri5", "dopri8"), default="dopri8")
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--particle-setup", choices=("line", "axis"), default="line")
    parser.add_argument("--birth-seed", type=int, default=42)
    parser.add_argument("--pitch-min", type=float, default=-1.)
    parser.add_argument("--pitch-max", type=float, default=1.)
    parser.add_argument("--gyrophase", type=float, default=float(jnp.pi), help="Radians")
    parser.add_argument("--horizon", type=float, help="Seconds; default GC 20 us, Lorentz 2 us")
    parser.add_argument("--initial-step", type=float, help="Initial adaptive step in seconds; default horizon/2")
    parser.add_argument("--saves", type=int, default=32)
    parser.add_argument("--energy-ev", type=float, default=4000.)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--checkpoint-steps", type=int, default=16)
    parser.add_argument("--accepted-capacity", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--max-evaluations", type=int, help="L-BFGS-B function evaluation limit")
    parser.add_argument("--max-change", type=float, default=.01,
                        help="Absolute displacement bound in each public normalized coil DOF")
    parser.add_argument("--regularization", type=float, default=0.,
                        help="Optional mean-square bounded-displacement penalty")
    parser.add_argument("--particle-weight", type=float, default=1.)
    parser.add_argument("--length-weight", type=float, default=1., help="Weight of mean positive coil-length excess")
    parser.add_argument("--curvature-weight", type=float, default=1., help="Weight of mean positive pointwise curvature excess")
    parser.add_argument("--bdotn-weight", type=float, default=1., help="Weight of summed abs(B dot n / |B|) on the fixed surface")
    parser.add_argument("--axis-field-weight", type=float, default=0., help="Weight of abs(mean axis |B| - target)")
    parser.add_argument("--axis-field-target", type=float, default=5.7, help="Tesla")
    parser.add_argument("--axis-field-points", type=int, default=15)
    parser.add_argument("--length-target", type=float, help="Metres; default 1.05 times the largest initial coil length")
    parser.add_argument("--curvature-target", type=float, help="1/metre; default 1.05 times the largest initial pointwise curvature")
    parser.add_argument("--surface-ntheta", type=int, default=30, help="Poloidal points for the normal-field objective")
    parser.add_argument("--surface-nphi", type=int, default=30, help="Toroidal points over a half field period for the normal-field objective")
    parser.add_argument("--reference-major-radius", type=float, default=1., help="Metres; scale the fixed QA target to this major radius")
    parser.add_argument("--softness", type=float, default=10., help="Loss sigmoid slope in 1/metre")
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-particles", type=int, default=10, help="Random subset size, capped by traced population")
    parser.add_argument("--plot-seed", type=int, default=42)
    unknown = defaults.keys() - {action.dest for action in parser._actions}
    if unknown:
        raise ValueError(f"unknown native optimization settings: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    options = parser.parse_args(argv)
    for action in parser._actions:
        if action.choices is not None and getattr(options, action.dest) not in action.choices:
            parser.error(f"invalid {action.dest} in script settings; choose from {action.choices}")
    objective = options.objective
    if options.horizon is None:
        options.horizon = 2e-5 if options.model == "gc" else 2e-6
    positive = (options.horizon, options.energy_ev, options.tolerance,
                options.max_change, options.softness, options.reference_major_radius,
                options.coil_major_radius, options.coil_minor_radius)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        parser.error("horizon, energy, tolerance, max-change, softness and radii must be finite and positive")
    if options.initial_step is not None and (not np.isfinite(options.initial_step) or options.initial_step <= 0):
        parser.error("initial-step must be finite and positive")
    if (not -1 <= options.pitch_min < options.pitch_max <= 1 or
        not np.isfinite(options.gyrophase) or not np.isfinite(options.coil_current) or
        options.coil_current == 0 or not np.isfinite(options.coil_scaling_factor)):
        parser.error("invalid pitch range, gyrophase, coil current or Fourier scaling")
    if (min(options.coil_count, options.coil_order, options.coil_nfp) < 1 or
        options.coil_segments < 3 or options.coil_minor_radius >= options.coil_major_radius or
        (options.max_evaluations is not None and options.max_evaluations < 1)):
        parser.error("positive coil counts/order, at least three segments, major > minor radius and positive evaluation limit required")
    if min(options.particles, options.checkpoint_steps, options.accepted_capacity,
           options.max_steps, options.maxiter, options.plot_particles) < 1 or min(options.saves,
               options.surface_ntheta, options.surface_nphi, options.axis_field_points) < 2:
        parser.error("positive particle/step/iteration counts and at least two saves and surface points per direction required")
    weights = (options.particle_weight, options.length_weight,
               options.curvature_weight, options.bdotn_weight, options.axis_field_weight, options.regularization)
    if any(not np.isfinite(value) or value < 0 for value in weights) or not any(weights):
        parser.error("weights must be finite and nonnegative, with at least one nonzero weight")
    targets = (options.length_target, options.curvature_target, options.axis_field_target)
    if any(value is not None and (not np.isfinite(value) or value < 0) for value in targets):
        parser.error("length, curvature and axis-field targets must be finite and nonnegative")
    if options.output.exists():
        parser.error("output directory already exists; choose a new path")
    options.output.mkdir(parents=True)
    print(f"Building {objective}, {options.model}, {options.particles} particles; coil DOFs only", flush=True)
    base, theta0, axis, xyz, pitches, objective_fn = build_objective(options, objective)
    base.to_json(str(options.output / "initial_coils.json"))
    print("Initial coils: " + (str(options.coils) if options.coils is not None else
        f"{base.gamma.shape[0]} planar circles ({options.coil_count} independent), "
        f"R={options.coil_major_radius:g} m, r={options.coil_minor_radius:g} m, "
        f"current={options.coil_current:g} A"), flush=True)
    print(f"Length target {options.length_target:.6g} m; curvature target "
          f"{options.curvature_target:.6g} /m; particle/length/curvature/Bn/axis weights "
          f"{weights[:5]}", flush=True)
    evaluate = jax.jit(jax.value_and_grad(objective_fn, has_aux=True))
    history = []
    cached = None

    with (options.output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("evaluation", "cost", "particle_metric",
            "length_penalty", "curvature_penalty", "bdotn_penalty", "axis_field_penalty", "regularization_penalty",
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
        optimizer_options = dict(maxiter=options.maxiter, maxls=12, ftol=1e-12, gtol=1e-10)
        if options.max_evaluations is not None:
            optimizer_options["maxfun"] = options.max_evaluations
        result = minimize(value_gradient, delta0, method="L-BFGS-B", jac=True,
            bounds=[(-1., 1.)] * theta0.size,
            options=optimizer_options)
        final_value, final_gradient = value_gradient(result.x)
        final_components, final_trajectories = cached[3], cached[4]

    final_dofs = theta0 + options.max_change * jnp.asarray(result.x)
    base.with_dofs(final_dofs).to_json(str(options.output / "optimized_coils.json"))
    np.savez_compressed(options.output / "orbits.npz",
        times=np.linspace(0., options.horizon, options.saves),
        initial_trajectories=initial_trajectories, final_trajectories=final_trajectories,
        initial_coil_dofs=np.asarray(theta0), final_coil_dofs=np.asarray(final_dofs),
        birth_xyz=np.asarray(xyz), birth_pitch=np.asarray(pitches))
    plots = save_optimization_plots(options.output, base, base.with_dofs(final_dofs),
        initial_trajectories, final_trajectories, axis=axis,
        count=options.plot_particles, seed=options.plot_seed) if options.save_plots else None
    units = {"loss_fraction": "probability", "final_radius": "metres", "radial_increment": "metres squared"}
    report = dict(objective=objective, metric_units=units[objective],
        coil_initialization="planar_circular" if options.coils is None else "json_file",
        initial_cost=initial_value, final_cost=final_value,
        initial_particle_metric=initial_components["particle_metric"],
        final_particle_metric=final_components["particle_metric"],
        initial_components=initial_components, final_components=final_components,
        loss_weights=dict(particle_metric=options.particle_weight, length_penalty=options.length_weight,
            curvature_penalty=options.curvature_weight, bdotn_penalty=options.bdotn_weight,
            axis_field_penalty=options.axis_field_weight,
            regularization_penalty=options.regularization),
        loss_definitions=dict(length_penalty="mean(max(coil_length_m - length_target_m, 0))",
            curvature_penalty="mean(max(pointwise_curvature_per_m - curvature_target_per_m, 0))",
            bdotn_penalty="sum(abs(B dot unitnormal / |B|)) over fixed half-period surface mesh",
            axis_field_penalty="abs(mean(|B| on fixed initial axis circle) - axis_field_target_T)",
            radial_increment="sum_i(mean_j(r[i,j+1]-r[i,j]))**2; no division by dt; save-count dependent",
            regularization_penalty="mean(delta**2)"),
        initial_gradient_l2=float(np.linalg.norm(initial_gradient)),
        final_gradient_l2=float(np.linalg.norm(final_gradient)),
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        iterations=int(result.nit), function_evaluations=int(result.nfev),
        recorded_evaluations=len(history), coil_dof_names=list(base.dof_names),
        reference_axis_r_z_m=axis, backend=jax.default_backend(), jax=jax.__version__,
        settings={name: str(value.resolve()) if isinstance(value, Path) else value
                  for name, value in vars(options).items()},
        plots=plots,
        derivative_policy="Accepted-step reverse AD with frozen adaptive times; segment checkpoints; full population vmap",
        birth_policy="Fixed GC birth XYZ, pitch, energy and gyrophase; Lorentz conversion remains field-dependent",
        target_policy="Fixed initial reference axis and fixed scaled QA surface; no hard wall stopping",
        gradient_coordinates="Dimensionless delta; coil DOFs = initial DOFs + max_change * delta",
        timing_scope="First evaluation includes compilation; later evaluations reuse compiled value_and_grad")
    (options.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Cost {initial_value:.10g} -> {final_value:.10g}; particle metric "
          f"{initial_components['particle_metric']:.10g} -> {final_components['particle_metric']:.10g}; {result.message}; "
          f"saved {options.output}", flush=True)

