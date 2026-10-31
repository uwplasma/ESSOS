"""Shared setup for accepted-step reverse-AD coil optimization examples.

Run an objective-specific entry point with ``python -m``; importing this module
does not trace particles or start an optimization. All particles are traced
together; birth parameters, reference axis and classifier stay fixed. Every
setting in ``DEFAULTS`` is a command-line option and an editable script default.
"""
import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("ESSOS_PROGRESS", "0")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from essos.background_species import BackgroundSpecies
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, ONE_EV
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.particle_losses import mean_final_radius, minor_radius, classifier_loss_fraction
from essos.split import trace_split, SplitStatus
from essos.surfaces import BdotN_over_B, SurfaceClassifier, SurfaceRZFourier

INPUT = Path(__file__).resolve().parents[1] / "input_files"
PARTICLE_OBJECTIVES = ("loss_fraction", "final_radius", "radial_increment")
DEFAULTS = dict(
    output=None, coils=INPUT / "ESSOS_biot_savart_LandremanPaulQA.json",  # coils=None: planar circles
    coil_count=3, coil_order=6, coil_major_radius=7.74, coil_minor_radius=4.5, coil_current=1.84e7,
    coil_nfp=2, coil_stellsym=True, coil_segments=60, coil_scaling_factor=0.,
    wout=INPUT / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc", target_surface="qa",
    target_minor_radius=.2, reference_major_radius=1., safety_minor_radius=None,
    model="gc", method="dopri8", particles=8, particle_setup="line", birth_seed=42,
    pitch_min=-1., pitch_max=1., gyrophase=float(np.pi), horizon=None, initial_step=None, saves=32,
    energy_ev=4000., tolerance=1e-9, checkpoint_steps=16, accepted_capacity=512, max_steps=4096,
    collisions=False, collision_adaptive=True, collision_controller="moments", collision_tolerance=.001,
    moment_tolerance=.005, collision_step=1e-7, collision_dtmax=1e-6, collision_spatial_diffusion=None,
    collision_checkpoint_steps=16, collision_accepted_capacity=4096, max_orbit_steps=4096,
    background_density=1e20, background_temperature_ev=1000., coulomb_log_model="nrl",
    maxiter=10, max_change=.01, particle_weight=1., length_weight=1., length_target=None,
    curvature_weight=1., curvature_target=None, bdotn_weight=1., surface_ntheta=30, surface_nphi=30,
    axis_field_weight=0., axis_field_target=5.7, axis_field_points=15, softness=10., save_plots=True)


def save_optimization_plots(output, initial_coils, optimized_coils, initial_trajectories,
                            final_trajectories, *, axis, count=10, seed=42):
    """Save matched 3D and R-Z views of one random subset of particles before/after."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    paths = [np.asarray(t)[..., :3] for t in (initial_trajectories, final_trajectories)]
    indices = np.random.default_rng(seed).choice(len(paths[0]), min(count, len(paths[0])), replace=False)
    figure = Figure(figsize=(14, 11), layout="constrained")
    FigureCanvasAgg(figure)  # headless: leaves the user's pyplot backend alone
    for column, (title, coils, trajectories) in enumerate(
            zip(("Initial", "Optimized"), (initial_coils, optimized_coils), paths)):
        ax3d = figure.add_subplot(2, 2, column+1, projection="3d")
        axrz = figure.add_subplot(2, 2, column+3)
        for curve in np.asarray(coils.gamma):
            ax3d.plot(*np.concatenate((curve, curve[:1])).T, color="0.45", linewidth=1., alpha=.65)
        for rank, index in enumerate(indices):
            points, color = trajectories[index], f"C{rank % 10}"
            ax3d.plot(*points.T, color=color, linewidth=1.3)
            axrz.plot(np.hypot(points[:, 0], points[:, 1]), points[:, 2], color=color, linewidth=1.3,
                      label=f"Particle {index}")
        ax3d.set(title=f"{title} coils and trajectories", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        ax3d.set_box_aspect(np.ptp(np.concatenate([np.asarray(coils.gamma).reshape(-1, 3), trajectories[indices].reshape(-1, 3)]), axis=0))
        axrz.scatter(*axis, marker="+", color="black", s=40, zorder=4)
        axrz.set(title=f"{title}: R-Z projection over all toroidal angles", xlabel="R [m]", ylabel="Z [m]")
        axrz.set_aspect("equal", adjustable="datalim")
    figure.legend(*axrz.get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
    figure.savefig(Path(output) / "orbit_comparison.png", dpi=160)


def radial_increment_cost(trajectories, axis_r, axis_z):
    """sum_i(mean_j(r[i,j+1]-r[i,j]))**2: net radial drift, dependent on the save count."""
    return jnp.sum(jnp.mean(jnp.diff(minor_radius(trajectories, axis_r, axis_z), axis=1), axis=1)**2)


def fixed_surface(wout, *, ntheta=64, nphi=64, range_torus="full torus", major_radius=1.,
                  shape="qa", minor_radius=.2, nfp=1):
    """QA shape rescaled to ``major_radius``, or a circular torus, held fixed during optimization."""
    if shape == "torus":
        return SurfaceRZFourier(jnp.array([major_radius, minor_radius]), jnp.array([0., minor_radius]),
            nfp, 1, 0, ntheta=ntheta, nphi=nphi, range_torus=range_torus)
    surface = SurfaceRZFourier.from_wout_file(str(wout))
    scale = major_radius / surface.rc[0]
    return SurfaceRZFourier(surface.rc*scale, surface.zs*scale, surface.nfp, surface.mpol, surface.ntor,
        ntheta=ntheta, nphi=nphi, range_torus=range_torus)


def fixed_classifier(wout, *, major_radius=1., **target):
    """Geometric confinement target (positive inside); not recomputed after a coil update."""
    return SurfaceClassifier(fixed_surface(wout, major_radius=major_radius, **target), h=.03*major_radius,
        padding=.16*major_radius)


def toroidal_safety_stop(major_radius, minor_radius, axis_z=0.):
    """Analytic hard-stop level set, positive inside the safety torus and clamped to zero outside."""
    def stop(t, y, args, **kwargs):
        return jnp.maximum(0., minor_radius**2 - (jnp.hypot(y[0], y[1])-major_radius)**2 - (y[2]-axis_z)**2)
    return stop


def build_objective(o, objective):
    """Return the coil-only scalar objective ``delta -> (cost, (components, states))``."""
    if o.coils is None:  # higher Fourier modes start at zero, so optimization can deform the circles
        curves = CreateEquallySpacedCurves(n_curves=o.coil_count, order=o.coil_order, R=o.coil_major_radius,
            r=o.coil_minor_radius, n_segments=o.coil_segments, nfp=o.coil_nfp, stellsym=o.coil_stellsym,
            scaling_type="L2", scaling_factor=o.coil_scaling_factor)
        base = Coils(curves, jnp.full(o.coil_count, o.coil_current))
    else:
        base = Coils.from_json(str(o.coils))
    theta0, initial_field = jnp.asarray(base.dofs), BiotSavart(base)
    axis = (float(initial_field.r_axis), float(initial_field.z_axis))
    o.length_target = 1.05*float(jnp.max(base.length)) if o.length_target is None else o.length_target
    o.curvature_target = 1.05*float(jnp.max(base.curvature)) if o.curvature_target is None else o.curvature_target
    target = dict(major_radius=o.reference_major_radius, shape=o.target_surface,
                  minor_radius=o.target_minor_radius, nfp=base.nfp)
    classifier = fixed_classifier(o.wout, **target) if objective == "loss_fraction" else None
    surface = fixed_surface(o.wout, ntheta=o.surface_ntheta, nphi=o.surface_nphi, range_torus="half period",
        **target) if o.bdotn_weight else None
    if o.particle_setup == "axis":  # equally spaced around the axis, including the duplicated 0/2pi birth
        phi = jnp.linspace(0., 2*jnp.pi, o.particles)
        xyz = jnp.column_stack((axis[0]*jnp.cos(phi), axis[0]*jnp.sin(phi), jnp.full_like(phi, axis[1])))
        pitches = jax.random.uniform(jax.random.PRNGKey(o.birth_seed), (o.particles,),
            minval=o.pitch_min, maxval=o.pitch_max)
    else:
        xyz = jnp.column_stack((jnp.linspace(1.23, 1.27, o.particles), jnp.zeros((o.particles, 2))))
        pitches = jnp.linspace(.35, .65, o.particles)
    phi = jnp.linspace(0., 2*jnp.pi, o.axis_field_points)
    axis_points = jnp.column_stack((axis[0]*jnp.cos(phi), axis[0]*jnp.sin(phi), jnp.full_like(phi, axis[1])))
    stop = (toroidal_safety_stop(axis[0], o.safety_minor_radius, axis[1])
            if o.safety_minor_radius is not None else None)
    integrator = "essos_dopri8" if o.method == "dopri8" else "essos_rk45"
    initial_step = o.initial_step if o.initial_step is not None else o.horizon/2
    if o.collisions:
        background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([o.background_density]),
            jnp.array([o.background_temperature_ev]), coulomb_log_model=o.coulomb_log_model)

    def evaluate(delta):
        # Coils.dofs already carries the public normalization: max_change bounds those DOFs directly.
        field = BiotSavart(base.with_dofs(theta0 + o.max_change*delta))
        birth = Particles(xyz, initial_vparallel_over_v=pitches, mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE, energy=o.energy_ev*ONE_EV,
            field=field if o.model == "lorentz" else None, phase_angle_full_orbit=o.gyrophase)
        work = {}
        if o.collisions:
            out = trace_split(field, birth, background, model="GuidingCenter" if o.model == "gc" else "FullOrbit",
                integrator=integrator, tmax=o.horizon, n_save=o.saves, timestep=o.collision_step,
                collision_dtmax=o.collision_dtmax, collision_tolerance=o.collision_tolerance,
                collision_controller=o.collision_controller, adaptive=o.collision_adaptive,
                moment_tolerance=o.moment_tolerance, orbit_dt0=initial_step,
                spatial_diffusion=o.collision_spatial_diffusion, rtol=o.tolerance, atol=o.tolerance, event=stop,
                max_steps=o.max_steps, max_orbit_steps=o.max_orbit_steps,
                execution="independent" if jax.default_backend() == "gpu" and o.method == "dopri8" and o.collision_adaptive else "shared",
                adjoint="accepted", adjoint_checkpoint_steps=o.collision_checkpoint_steps,
                adjoint_orbit_checkpoint_steps=o.checkpoint_steps,
                adjoint_max_accepted_steps=o.collision_accepted_capacity, adjoint_max_orbit_steps=o.accepted_capacity)
            states = eqx.error_if(out.ys, jnp.any(out.status > SplitStatus.EVENT),
                "split optimization solve failed; inspect collision/orbit work limits")
            stop_fraction = jnp.mean((out.status == SplitStatus.EVENT).astype(float))
            work = {f"mean_{name}": jnp.mean(getattr(out.stats, name).astype(float))
                    for name in ("accepted", "rejected", "orbit_steps", "orbit_rejected")}
        else:
            trace = Tracing(field=field, particles=birth,
                model="GuidingCenterAdaptative" if o.model == "gc" else "FullOrbitAdaptative",
                integrator=integrator, native_adjoint="accepted", native_adjoint_checkpoint_steps=o.checkpoint_steps,
                native_adjoint_max_accepted_steps=o.accepted_capacity, native_controller="i", rtol=o.tolerance,
                atol=o.tolerance, maxtime=o.horizon, timestep=initial_step, times_to_trace=o.saves,
                max_steps=o.max_steps, stopping_criteria=stop, progress=False)
            states, stop_fraction = trace.trajectories, jnp.mean(trace.boundary_hits.astype(float))
        metric = (classifier_loss_fraction(states, classifier, softness=o.softness) if objective == "loss_fraction"
                  else mean_final_radius(states, *axis) if objective == "final_radius"
                  else radial_increment_cost(states, *axis))
        zero = jnp.array(0.)
        components = dict(particle_metric=metric, safety_stop_fraction=stop_fraction, **work,
            length_penalty=jnp.mean(jnp.maximum(field.coils.length-o.length_target, 0.)) if o.length_weight else zero,
            curvature_penalty=jnp.mean(jnp.maximum(field.coils.curvature-o.curvature_target, 0.)) if o.curvature_weight else zero,
            bdotn_penalty=jnp.sum(jnp.abs(BdotN_over_B(surface, field))) if o.bdotn_weight else zero,
            axis_field_penalty=jnp.abs(jnp.mean(jax.vmap(field.AbsB)(axis_points))-o.axis_field_target)
                if o.axis_field_weight else zero)
        cost = (o.particle_weight*metric + o.length_weight*components["length_penalty"]
            + o.curvature_weight*components["curvature_penalty"] + o.bdotn_weight*components["bdotn_penalty"]
            + o.axis_field_weight*components["axis_field_penalty"])
        return cost, (components, states)

    return base, theta0, axis, evaluate


def run(objective, argv=None, *, defaults=None):
    """Run a selected native objective with optional editable script defaults."""
    settings = dict(DEFAULTS, **(defaults or {}))
    if objective not in PARTICLE_OBJECTIVES or settings.keys() - DEFAULTS.keys():
        raise ValueError(f"unknown objective or settings: {objective}, {sorted(settings.keys() - DEFAULTS.keys())}")
    parser = argparse.ArgumentParser(description="Optimize coil geometry and currents with accepted-step reverse AD.")
    parser.add_argument("--objective", choices=PARTICLE_OBJECTIVES, default=objective)
    for name, value in settings.items():
        kwargs = dict(default=value)
        if isinstance(value, bool) or name == "collision_spatial_diffusion":
            kwargs["action"] = argparse.BooleanOptionalAction
        elif isinstance(value, Path) or name in ("output", "coils"):
            kwargs["type"] = Path
        elif value is None:
            kwargs["type"] = float
        else:
            kwargs["type"] = type(value)
        if name == "coils":
            kwargs["help"] = "Initial coil JSON; --circular-coils creates equally spaced planar circles"
        if name == "collision_controller":
            kwargs["choices"] = ("moments", "conservative")
        if name in ("model", "method", "target_surface", "particle_setup"):
            kwargs["choices"] = dict(model=("gc", "lorentz"), method=("dopri5", "dopri8"),
                target_surface=("qa", "torus"), particle_setup=("line", "axis"))[name]
        parser.add_argument("--"+name.replace("_", "-"), **kwargs)
    parser.add_argument("--circular-coils", dest="coils", action="store_const", const=None)
    o = parser.parse_args(argv)
    if o.output is None:
        parser.error("--output is required")
    if o.output.exists():
        parser.error("output directory already exists; choose a new path")
    objective = o.objective
    if o.horizon is None:
        o.horizon = 2e-5 if o.model == "gc" else 2e-6
    print(f"JAX backend: {jax.default_backend()}; building {objective}, {o.model}, {o.particles} particles", flush=True)
    o.output.mkdir(parents=True)
    base, theta0, axis, objective_fn = build_objective(o, objective)
    base.to_json(str(o.output/"initial_coils.json"))
    evaluate = jax.jit(jax.value_and_grad(objective_fn, has_aux=True))
    history, cached = [], None
    stream = (o.output/"history.csv").open("w", newline="")
    writer = None

    def value_gradient(delta):
        nonlocal cached, writer
        delta = np.asarray(delta, dtype=np.float64)
        if cached is not None and np.array_equal(delta, cached[0]):
            return cached[1], cached[2]
        start = perf_counter()
        (value, (components, trajectories)), gradient = jax.block_until_ready(evaluate(jnp.asarray(delta)))
        components = {name: float(term) for name, term in components.items()}
        gradient, trajectories = np.asarray(gradient), np.asarray(trajectories)
        if not (np.isfinite(float(value)) and np.isfinite(list(components.values())).all()
                and np.isfinite(gradient).all() and np.isfinite(trajectories).all()):
            raise RuntimeError("nonfinite orbit, objective or gradient")
        row = dict(evaluation=len(history), cost=float(value), **components,
            gradient_l2=float(np.linalg.norm(gradient)), seconds=perf_counter()-start)
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        history.append(row)
        writer.writerow(row)
        stream.flush()
        print(f"eval {row['evaluation']:3d}: metric={components['particle_metric']:.10g}, cost={row['cost']:.10g}, "
              f"|grad|={row['gradient_l2']:.4g}, safety stops={components['safety_stop_fraction']:.1%}, "
              f"{row['seconds']:.2f} s", flush=True)
        cached = (delta.copy(), row["cost"], gradient, components, trajectories)
        return row["cost"], gradient

    delta0 = np.zeros(theta0.size)
    value_gradient(delta0)
    initial_components, initial_trajectories = cached[3], cached[4].copy()
    result = minimize(value_gradient, delta0, method="L-BFGS-B", jac=True, bounds=[(-1., 1.)]*theta0.size,
        options=dict(maxiter=o.maxiter, maxls=12, ftol=1e-12, gtol=1e-10))
    value_gradient(result.x)
    stream.close()
    final_components, final_trajectories = cached[3], cached[4]
    final_dofs = theta0 + o.max_change*jnp.asarray(result.x)
    base.with_dofs(final_dofs).to_json(str(o.output/"optimized_coils.json"))
    np.savez_compressed(o.output/"orbits.npz", initial_trajectories=initial_trajectories,
        final_trajectories=final_trajectories, initial_coil_dofs=np.asarray(theta0), final_coil_dofs=np.asarray(final_dofs))
    if o.save_plots:
        save_optimization_plots(o.output, base, base.with_dofs(final_dofs), initial_trajectories,
                                final_trajectories, axis=axis)
    report = dict(objective=objective, initial_components=initial_components, final_components=final_components,
        optimizer_success=bool(result.success), optimizer_message=str(result.message), iterations=int(result.nit),
        function_evaluations=int(result.nfev), backend=jax.default_backend(), jax=jax.__version__,
        settings={name: str(v) if isinstance(v, Path) else v for name, v in vars(o).items()},
        note="Gradients freeze adaptive step times, stopping decisions and saved-time masks; collision runs "
             "differentiate coefficients and kicks with the recorded Wiener increments. "
             "Coil DOFs = initial DOFs + max_change * delta.")
    (o.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Particle metric {initial_components['particle_metric']:.10g} -> "
          f"{final_components['particle_metric']:.10g}; {result.message}; saved {o.output}", flush=True)
