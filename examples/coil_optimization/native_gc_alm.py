"""Shared QI augmented-Lagrangian setup for the native guiding-center examples.

Three entry scripts select the particle constraint (loss fraction, distance to a boundary
line, heat proxy). Native accepted-map reverse AD supplies the gradients. Importing this
module never starts tracing; all particles and random paths remain fixed.
"""
import argparse
import csv
import json
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from essos import augmented_lagrangian as alm
from essos.background_species import BackgroundSpecies
from essos.coils import Coils
from essos.constants import (ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY, ELECTRON_MASS, PROTON_MASS, ONE_EV)
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.native_ad import gpu_jit_options
from essos.particle_losses import classifier_loss_fraction, classifier_line_loss, classifier_energy_cost
from essos.rk import _fourier_geometry, _coil_geometry, _coil_field_values
from essos.split import ClassifierStop, trace_split, SplitStatus
from essos.surfaces import SurfaceRZFourier, SurfaceClassifier
from .native_particle_optimization import save_optimization_plots

INPUT = Path(__file__).resolve().parents[1]/"input_files"
DEFAULTS = dict(
    coils=INPUT/"QI_opt_coils_essos_5.json",
    wout=INPUT/"wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc",
    particles=20, seed=42, birth_fraction=.7, energy_ev=FUSION_ALPHA_PARTICLE_ENERGY/ONE_EV,
    horizon=1e-4, initial_step=1e-8, saves=10, rtol=1e-7, atol=1e-7,
    checkpoint_steps=16, accepted_capacity=32768, max_steps=65536,
    length_target=45., curvature_target=1., axis_field_target=5.7,
    softness=30., location_softness=10., line_mode="fixed_theta",
    line_theta=1.5*np.pi, line_phi=np.pi, bdotn_target=1e-3, bdotn_weight=0.,
    surface_points=60, classifier_h=.1, safety_margin=.2, birth_axis_samples=1000,
    maxiter=100, inner_maxiter=50, max_change=.05, penalty=1., beta=1.01,
    eta_tol=1e-5, omega_tol=1e-7, model_mu="Mu_Adaptative_2",
    collisions=True, execution="auto", collision_step=1e-8, collision_dtmax=1e-6,
    collision_capacity=4096, max_orbit_steps=4096,
    collision_tolerance=.001, moment_tolerance=.01, collision_seed=42,
    background_density=4.21e20, background_temperature_ev=17.8e3,
    optimizer="optimistix", scipy_ftol=1e-12,
    save_plots=True, output=Path("output/native_gc_alm"))


def surface_point(surface, theta, phi):
    angle = surface.xm*theta-surface.xn*phi
    radius, height = jnp.sum(surface.rc*jnp.cos(angle)), jnp.sum(surface.zs*jnp.sin(angle))
    return jnp.array([radius*jnp.cos(phi), radius*jnp.sin(phi), height])


def boundary_line(surface, theta, phi, mode):
    """Infinite tangent line at a fixed-theta or fixed-phi boundary curve."""
    if mode not in ("fixed_theta", "fixed_phi"):
        raise ValueError("line_mode must be fixed_theta or fixed_phi")
    direction = jnp.array([0., 1.]) if mode == "fixed_theta" else jnp.array([1., 0.])
    point, tangent = jax.jvp(lambda a: surface_point(surface, *a), (jnp.array([theta, phi]),), (direction,))
    return point, tangent/jnp.linalg.norm(tangent)


def make_constraints(geometry, particle, options):
    """alm.ineq uses g >= 0, so upper limits enter as target minus value."""
    p = dict(penalty=options.penalty, omega=1/options.penalty, eta=1/options.penalty**.1)
    constraints = [alm.eq(lambda x: geometry(x)["axis"], **p),
        alm.ineq(lambda x: geometry(x)["length"], **p),
        alm.ineq(lambda x: geometry(x)["curvature"], **p), alm.eq(particle, **p)]
    if options.bdotn_weight:
        constraints.append(alm.ineq(lambda x: geometry(x)["bdotn"], weight=options.bdotn_weight, **p))
    return alm.combine(*constraints)


def compile_optimizer(optimizer, *, independent=True):
    """Apply the GPU command-buffer policy of the split reverse to ALM's outer jitted functions."""
    options = gpu_jit_options() if independent else {}
    if not options:
        return optimizer
    update = optimizer.update
    return optimizer._replace(init=jax.jit(optimizer.init.__wrapped__, **options),
        update=partial(jax.jit(update.func.__wrapped__, static_argnums=(4, 5, 6, 7, 8, 9, 10), **options),
                       *update.args, **update.keywords))


def scipy_optimizer(constraints, options):
    """Host L-BFGS-B inner solves with ESSOS ALM constraints and multiplier updates.

    One native value/gradient executable is compiled and reused; SciPy enforces the
    [-1, 1] scaled-DOF box directly (gradient tolerance omega_tol, ftol selectable).
    """
    import optax
    from scipy.optimize import minimize
    policy = gpu_jit_options() if options.collisions and options.execution != "shared" else {}
    def lagrangian(x, lp):
        value, violations = constraints.loss(lp, x)
        return value, (jnp.zeros_like(value), value, violations)
    value_grad = jax.jit(jax.value_and_grad(lagrangian, argnums=(0, 1), has_aux=True), **policy)
    executable = None
    statistics = dict(evaluations=0, compile_s=0., inner_solves=[])

    def evaluate(x, lp):
        (value, info), gradient = jax.block_until_ready(executable(jnp.asarray(x), lp))
        if not np.isfinite(value) or not np.isfinite(gradient[0]).all():
            raise RuntimeError("nonfinite SciPy native value/gradient")
        statistics["evaluations"] += 1
        return float(value), gradient, info

    def init(params):
        nonlocal executable
        start = perf_counter()
        executable = value_grad.lower(*params).compile()
        statistics["compile_s"] = perf_counter()-start
        print(f"Native objective/gradient compiled in {statistics['compile_s']:.2f} s", flush=True)
        _, gradient, info = evaluate(*params)
        return statistics, gradient, info

    @jax.jit
    def update_multipliers(lp, gradient, info, old_info):
        kwargs = dict(model_lagrangian="Standard", beta=options.beta, eta_tol=options.eta_tol, omega_tol=options.omega_tol)
        if options.model_mu == "Mu_Tolerance":
            updates = alm.apply_mu_tolerance_all_constraints(lp, grad_dicts_map=gradient, constraint_infos_map=info, **kwargs)
        else:
            updates = jax.tree_util.tree_map(lambda p, g, new, old: alm.apply_mu_tolerance_per_constraint(
                p, g, new, constraint_info_prev=old, model_mu=options.model_mu, decrease_tol=.75, **kwargs),
                lp, gradient, info, old_info, is_leaf=lambda x: isinstance(x, dict))
        return optax.apply_updates(lp, updates)

    def update(params, state, gradient, info):
        x, lp = params
        old_info, latest, start = info[2], {}, perf_counter()
        def fun(candidate):
            value, grad, aux = evaluate(candidate, lp)
            latest.update(x=candidate.copy(), gradient=grad, info=aux)
            return value, np.asarray(grad[0], dtype=np.float64)
        result = minimize(fun, np.asarray(x, dtype=np.float64), method="L-BFGS-B", jac=True,
            bounds=[(-1., 1.)]*x.size,
            options=dict(maxiter=options.inner_maxiter, gtol=options.omega_tol, ftol=options.scipy_ftol))
        x = jnp.asarray(result.x)
        if np.array_equal(result.x, latest.get("x")):
            gradient, info = latest["gradient"], latest["info"]
        else:
            _, gradient, info = evaluate(x, lp)
        lp = update_multipliers(lp, gradient[1], info[2], old_info)
        _, gradient, info = evaluate(x, lp)
        statistics["inner_solves"].append(dict(iterations=int(result.nit), evaluations=int(result.nfev),
            message=str(result.message), seconds=perf_counter()-start))
        print(f"  SciPy inner solve: {result.message}; {result.nit} iterations, {result.nfev} evaluations", flush=True)
        return (x, lp), statistics, gradient, info
    return alm.ALM(init, update)


def field_normal_ratio(surface, field):
    """B.n/|B| on a surface through the native direct-coil evaluator."""
    arrays, nc, nq = _coil_geometry(field)
    magnetic = jax.vmap(lambda x: jnp.asarray(_coil_field_values(arrays, tuple(x), nc, nq, False)[0]))(
        surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    return jnp.sum(magnetic*surface.unitnormal, axis=-1)/jnp.linalg.norm(magnetic, axis=-1)


def build_problem(o, objective):
    if objective not in ("loss_fraction", "distance", "heat_flux"):
        raise ValueError("unknown QI particle constraint")
    for path in (o.coils, o.wout):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing physical input: {path}; use --coils/--wout")
    base = Coils.from_json(str(o.coils))
    angles = dict(ntheta=o.surface_points, nphi=o.surface_points, range_torus="full torus")
    surface = SurfaceRZFourier.from_wout_file(str(o.wout), s=1, **angles)
    axis = SurfaceRZFourier.from_wout_file(str(o.wout), s=0, **angles)
    axis_points = jax.vmap(lambda p: surface_point(axis, 0., p))(jnp.linspace(0., 2*jnp.pi, 200))
    # Normalize the field after physical scaling so current DOFs stay of order one.
    factor = o.axis_field_target/jnp.mean(jax.vmap(BiotSavart(base).AbsB)(axis_points))
    base = Coils(base.curves, base.dofs_currents_raw*factor)
    theta0, ncoils = jnp.asarray(base.dofs), len(base.dofs_currents)
    boundary = SurfaceClassifier(surface, h=o.classifier_h, padding=max(.1, 1.5*o.safety_margin))
    particles = Particles.InitializeParticlesAroundSurfaceAxis(axis, o.particles, boundary_surface=surface,
        distance_mode="fraction_to_boundary", distance_from_axis=o.birth_fraction, random_seed=o.seed,
        n_arc_samples=o.birth_axis_samples, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
        energy=o.energy_ev*ONE_EV)
    particles.random_keys = jax.random.split(jax.random.key(o.collision_seed), o.particles)
    safety = ClassifierStop(boundary, o.safety_margin)
    if bool(jnp.any(jax.vmap(boundary.evaluate_xyz)(particles.initial_xyz) <= -o.safety_margin)):
        raise ValueError("birth outside safety surface; check classifier/surface resolution and input geometry")
    print(f"{ncoils} independent coils at {o.axis_field_target} T; {theta0.size} scaled DOFs", flush=True)
    point, direction = boundary_line(surface, o.line_theta, o.line_phi, o.line_mode)
    field_at = lambda x: BiotSavart(base.with_dofs(theta0+o.max_change*x))

    def geometry(x):
        field = field_at(x)
        _, tangent, second = _fourier_geometry(field.coils.curves, second=True)
        speed = jnp.linalg.norm(tangent, axis=-1)
        curvature = jnp.linalg.norm(jnp.cross(tangent, second), axis=-1)/speed**3
        result = dict(axis=jnp.mean(jax.vmap(field.AbsB)(axis_points))-o.axis_field_target,
            length=o.length_target-jnp.mean(speed, axis=-1)[:ncoils], curvature=o.curvature_target-curvature[:ncoils])
        if o.bdotn_weight:
            values = jnp.abs(field_normal_ratio(surface, field))-o.bdotn_target
            sharpness = 20/o.bdotn_target
            result["bdotn"] = -(jax.scipy.special.logsumexp(sharpness*values)-jnp.log(values.size))/sharpness
        return result

    execution = ("independent" if jax.default_backend() == "gpu" else "shared") if o.execution == "auto" else o.execution
    if o.collisions:
        background = BackgroundSpecies(2, jnp.array([ELECTRON_MASS/PROTON_MASS, 2.]), jnp.array([-1., 1.]),
            jnp.full(2, o.background_density), jnp.full(2, o.background_temperature_ev))
        print(f"Native split execution: forward=reverse={execution}", flush=True)

    def trace(x):
        field = field_at(x)
        if o.collisions:
            result = trace_split(field, particles, background, model="GuidingCenter", integrator="essos_dopri8",
                tmax=o.horizon, n_save=o.saves, timestep=o.collision_step, collision_dtmax=o.collision_dtmax,
                collision_tolerance=o.collision_tolerance, moment_tolerance=o.moment_tolerance,
                orbit_dt0=o.initial_step, rtol=o.rtol, atol=o.atol, event=safety, max_steps=o.max_steps,
                max_orbit_steps=o.max_orbit_steps, adjoint="accepted", adjoint_checkpoint_steps=o.checkpoint_steps,
                adjoint_orbit_checkpoint_steps=o.checkpoint_steps, adjoint_max_accepted_steps=o.collision_capacity,
                adjoint_max_orbit_steps=o.accepted_capacity, execution=execution,
                reverse_execution="independent" if execution == "independent" else "shared")
            states = eqx.error_if(result.ys, jnp.any(result.status > SplitStatus.EVENT),
                "native split solve failed; inspect work/tape capacities")
        else:
            states = Tracing(field=field, particles=particles, model="GuidingCenterAdaptative",
                integrator="essos_dopri8", native_adjoint="accepted", native_controller="i",
                native_adjoint_checkpoint_steps=o.checkpoint_steps, native_adjoint_max_accepted_steps=o.accepted_capacity,
                maxtime=o.horizon, timestep=o.initial_step, times_to_trace=o.saves, rtol=o.rtol, atol=o.atol,
                max_steps=o.max_steps, stopping_criteria=safety, progress=False).trajectories
        return eqx.error_if(states, ~jnp.isfinite(states).all(), "nonfinite native GC trajectory")

    measures = dict(
        loss_fraction=lambda x, states: classifier_loss_fraction(states, safety, o.softness),
        distance=lambda x, states: classifier_line_loss(states, point, direction, safety, o.softness, o.location_softness),
        heat_flux=lambda x, states: classifier_energy_cost(states, field_at(x), particles.mass, particles.energy,
            "GuidingCenterAdaptative", safety, softness=o.softness))
    metrics = lambda x, states: {name: measure(x, states) for name, measure in measures.items()}
    particle = lambda x: measures[objective](x, trace(x))  # one trace, only the selected constraint
    return SimpleNamespace(base=base, theta0=theta0, particles=particles, field=field_at, surface=surface,
        axis_points=axis_points, boundary=boundary, line=(point, direction), trace=trace, metrics=metrics,
        constraints=make_constraints(geometry, particle, o), initial=jnp.zeros_like(theta0))


def run(objective, defaults=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in dict(DEFAULTS, **(defaults or {})).items():
        kwargs = dict(default=value)
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = Path if isinstance(value, Path) else type(value)
        parser.add_argument("--"+name.replace("_", "-"), **kwargs)
    parser.add_argument("--evaluate-only", action="store_true", help="One value/gradient; skip ALM updates")
    o = parser.parse_args(argv)
    if min(o.particles, o.saves-1, o.surface_points-1, o.maxiter, o.inner_maxiter) < 1:
        parser.error("particle, save, surface and iteration counts are too small")
    if o.execution not in ("auto", "shared", "independent") or o.optimizer not in ("optimistix", "scipy"):
        parser.error("execution is auto/shared/independent; optimizer is optimistix/scipy")
    if o.execution == "independent" and not o.collisions:
        parser.error("independent split execution requires --collisions")
    if (o.output/"summary.json").exists():
        parser.error("choose a fresh output directory")
    o.output.mkdir(parents=True, exist_ok=True)
    print(f"JAX backend: {jax.default_backend()}; QI GC ALM {objective}; collisions={o.collisions}", flush=True)
    problem = build_problem(o, objective)
    problem.base.to_json(str(o.output/"initial_coils.json"))
    constraints = problem.constraints
    params = problem.initial, jax.jit(constraints.init)(problem.initial)  # shapes only; tracing is pruned
    gpu = o.collisions and o.execution != "shared"
    if o.optimizer == "scipy":
        optimizer = scipy_optimizer(constraints, o)
    else:
        optimizer = compile_optimizer(alm.ALM_model_lbfgs(constraints, beta=o.beta, model_mu=o.model_mu,
            eta_tol=o.eta_tol, omega_tol=o.omega_tol, inner_maxiter=o.inner_maxiter,
            bounds=(-jnp.ones_like(params[0]), jnp.ones_like(params[0]))), independent=gpu)
    start = perf_counter()
    state, gradient, info = jax.block_until_ready(optimizer.init(params))
    records = []
    def record(iteration, seconds):
        values = dict(iteration=iteration, lagrangian=float(info[1]), particle_metric=float(info[2][3]),
            constraint_norm=float(alm.norm_constraints(info[2])), gradient_norm=float(jnp.linalg.norm(gradient[0])),
            seconds=seconds)
        if not np.isfinite(list(values.values())).all():
            raise RuntimeError("nonfinite ALM value or gradient")
        records.append(values)
        print(f"ALM {iteration}: particle={values['particle_metric']:.8g}; L={values['lagrangian']:.8g}; "
              f"constraints={values['constraint_norm']:.4g}; |grad|={values['gradient_norm']:.4g}; {seconds:.2f} s", flush=True)
        with (o.output/"history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=values)
            writer.writeheader()
            writer.writerows(records)
    record(0, perf_counter()-start)
    for iteration in range(1, 1+(0 if o.evaluate_only else o.maxiter)):
        if records[-1]["constraint_norm"] <= o.eta_tol and records[-1]["gradient_norm"] <= o.omega_tol:
            break
        start = perf_counter()
        params, state, gradient, info = jax.block_until_ready(optimizer.update(params, state, gradient, info))
        record(iteration, perf_counter()-start)
    trace = jax.jit(problem.trace, **(gpu_jit_options() if gpu else {}))
    initial, final = jax.block_until_ready((trace(problem.initial), trace(params[0])))
    metrics = [{k: float(v) for k, v in problem.metrics(x, states).items()}
               for x, states in ((problem.initial, initial), (params[0], final))]
    print(f"Particle cost: logged {records[0]['particle_metric']:.9g} -> {records[-1]['particle_metric']:.9g}; "
          f"retraced {metrics[0][objective]:.9g} -> {metrics[1][objective]:.9g}", flush=True)
    optimized = problem.field(params[0]).coils
    optimized.to_json(str(o.output/"optimized_coils.json"))
    np.savez_compressed(o.output/"optimization.npz", parameters=params[0], initial_dofs=problem.theta0,
        optimized_dofs=optimized.dofs, initial_trajectories=initial, optimized_trajectories=final)
    if o.save_plots:
        radius, height = jnp.mean(jnp.linalg.norm(problem.axis_points[:, :2], axis=1)), jnp.mean(problem.axis_points[:, 2])
        save_optimization_plots(o.output, problem.base, optimized, initial, final, axis=(float(radius), float(height)), seed=o.seed)
    report = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(o).items()},
        objective=objective, initial=metrics[0], final=metrics[1], history=records, backend=jax.default_backend(),
        jax=jax.__version__, note="distance: escape-centroid distance to an infinite boundary tangent line; "
        "heat_flux: first-loss-weighted E_total*v_perp normalized by birth E*v (a dimensionless proxy)")
    if o.optimizer == "scipy":
        report["optimizer_statistics"] = state
    (o.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    if o.save_plots:
        from .plot_native_gc_alm import plot_run
        plot_run(o.output, problem.surface, ClassifierStop(problem.boundary), vars(o))
    print(f"Saved {o.output.resolve()}", flush=True)
