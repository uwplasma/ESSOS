"""Shared QI ALM setup recovered from en/jax_13 (638916d).

Three entry scripts select the particle constraint. The current ESSOS ALM
and accepted-map native reverse replace the historical optimizer/custom VJPs.
Imports never start tracing. All particles and random paths remain fixed.
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
import jaxlib
import jax.numpy as jnp
import numpy as np
from essos import augmented_lagrangian as alm
from essos.coils import Coils
from essos.constants import (ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY, ELECTRON_MASS, PROTON_MASS, ONE_EV)
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.particle_losses import (classifier_loss_fraction, classifier_line_loss,
    classifier_energy_cost)
from essos.surfaces import SurfaceRZFourier, SurfaceClassifier
from essos.split import ClassifierStop
from essos.native_ad import gpu_jit_options
from essos.rk import _fourier_geometry,_coil_geometry,_coil_field_values
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
    angles = jnp.array([theta, phi])
    direction = jnp.array([0., 1.]) if mode == "fixed_theta" else jnp.array([1., 0.])
    point, tangent = jax.jvp(lambda a: surface_point(surface, *a), (angles,), (direction,))
    return point, tangent/jnp.linalg.norm(tangent)


def make_constraints(geometry, particle, options):
    """Current alm.ineq uses g >= 0: upper limits are target minus value."""
    p = dict(penalty=options.penalty, omega=1/options.penalty, eta=1/options.penalty**.1)
    constraints = [alm.eq(lambda x: geometry(x)["axis"], **p),
        alm.ineq(lambda x: geometry(x)["length"], **p),
        alm.ineq(lambda x: geometry(x)["curvature"], **p), alm.eq(particle, **p)]
    if options.bdotn_weight:
        constraints.append(alm.ineq(lambda x: geometry(x)["bdotn"],
            weight=options.bdotn_weight, **p))
    return alm.combine(*constraints)


def compile_optimizer(optimizer,*,independent=True):
    """Apply the split GPU capture policy to ALM's outer compiled functions."""
    options=gpu_jit_options() if independent else {}
    if not options:return optimizer
    update=optimizer.update
    return optimizer._replace(
        init=jax.jit(optimizer.init.__wrapped__,**options),
        update=partial(jax.jit(update.func.__wrapped__,static_argnums=(4,5,6,7,8,9,10),**options),
                       *update.args,**update.keywords))


def scipy_optimizer(constraints, options):
    """Host L-BFGS-B, retaining ESSOS ALM constraints and multiplier updates.

    Only one native value/gradient executable is compiled. SciPy enforces the
    same [-1, 1] scaled-DOF box directly, rather than through tanh coordinates.
    Its inner gradient tolerance is omega_tol; ftol is independently selectable.
    """
    import optax
    from scipy.optimize import minimize
    policy = gpu_jit_options() if options.collisions and options.execution != "shared" else {}
    def lagrangian(x, lp):
        value, violations = constraints.loss(lp, x)
        return value, (jnp.zeros_like(value), value, violations)
    value_grad = jax.jit(jax.value_and_grad(lagrangian, argnums=(0, 1), has_aux=True), **policy)
    executable = None
    statistics = dict(evaluations=0, compile_s=0., first_evaluation_s=0., inner_solves=[])

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
        print(f"Native objective/gradient lowering+compilation: {statistics['compile_s']:.2f} s; starting first evaluation", flush=True)
        start = perf_counter()
        _, gradient, info = evaluate(*params)
        statistics["first_evaluation_s"] = perf_counter()-start
        print(f"First native value/gradient execution: {statistics['first_evaluation_s']:.2f} s", flush=True)
        return statistics, gradient, info

    @jax.jit
    def update_multipliers(lp, gradient, info, old_info):
        kwargs = dict(model_lagrangian="Standard", beta=options.beta,
                      eta_tol=options.eta_tol, omega_tol=options.omega_tol)
        if options.model_mu == "Mu_Tolerance":
            updates = alm.apply_mu_tolerance_all_constraints(lp,
                grad_dicts_map=gradient, constraint_infos_map=info, **kwargs)
        else:
            updates = jax.tree_util.tree_map(lambda p, g, new, old:
                alm.apply_mu_tolerance_per_constraint(p, g, new, constraint_info_prev=old,
                    model_mu=options.model_mu, decrease_tol=.75, **kwargs),
                lp, gradient, info, old_info, is_leaf=lambda x:isinstance(x, dict))
        return optax.apply_updates(lp, updates)

    def update(params, state, gradient, info):
        x, lp = params
        old_info, latest, steps = info[2], {}, []
        start = perf_counter()
        def fun(candidate):
            value, grad, aux = evaluate(candidate, lp)
            latest.update(x=candidate.copy(), gradient=grad, info=aux)
            return value, np.asarray(grad[0], dtype=np.float64)
        def callback(candidate):
            steps.append(None)
            print(f"  SciPy L-BFGS-B inner {len(steps)}: "
                  f"{statistics['evaluations']} total value/gradient calls; {perf_counter()-start:.2f} s", flush=True)
        result = minimize(fun, np.asarray(x, dtype=np.float64), method="L-BFGS-B", jac=True,
            bounds=[(-1., 1.)]*x.size, callback=callback,
            options=dict(maxiter=options.inner_maxiter, gtol=options.omega_tol, ftol=options.scipy_ftol))
        x = jnp.asarray(result.x)
        if np.array_equal(result.x, latest.get("x")):
            gradient, info = latest["gradient"], latest["info"]
        else:
            _, gradient, info = evaluate(x, lp)
        lp = update_multipliers(lp, gradient[1], info[2], old_info)
        _, gradient, info = evaluate(x, lp)
        statistics["inner_solves"].append(dict(iterations=int(result.nit), evaluations=int(result.nfev),
            status=int(result.status), message=str(result.message), seconds=perf_counter()-start))
        print(f"  SciPy inner solve: {result.message}; {result.nit} iterations, {result.nfev} evaluations", flush=True)
        return (x, lp), statistics, gradient, info
    return alm.ALM(init, update)


def field_normal_ratio(surface,field):
    """Use the native direct-coil evaluator for the unchanged B.n/|B| constraint."""
    arrays,nc,nq=_coil_geometry(field)
    magnetic=jax.vmap(lambda x:jnp.asarray(_coil_field_values(arrays,tuple(x),nc,nq,False)[0]))(
        surface.gamma.reshape(-1,3)).reshape(surface.gamma.shape)
    return jnp.sum(magnetic*surface.unitnormal,axis=-1)/jnp.linalg.norm(magnetic,axis=-1)


def save_coil_iteration(output,iteration,coil_dofs,parameters):
    """Store public coil DOFs and optimizer coordinates without particle tracing."""
    directory=Path(output)/"iterations"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/f"{iteration:06d}.npz",iteration=iteration,
        coil_dofs=np.asarray(coil_dofs),parameters=np.asarray(parameters))


def build_problem(options, objective):
    if objective not in ("loss_fraction", "distance", "heat_flux"):
        raise ValueError("unknown QI particle constraint")
    for path in (options.coils, options.wout):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing physical input: {path}; use --coils/--wout")
    base = Coils.from_json(str(options.coils))
    surface = SurfaceRZFourier.from_wout_file(str(options.wout), s=1,
        ntheta=options.surface_points, nphi=options.surface_points, range_torus="full torus")
    axis = SurfaceRZFourier.from_wout_file(str(options.wout), s=0,
        ntheta=options.surface_points, nphi=options.surface_points, range_torus="full torus")
    phi = jnp.linspace(0., 2*jnp.pi, 200)
    axis_points = jax.vmap(lambda p: surface_point(axis, 0., p))(phi)
    initial_field = BiotSavart(base)
    factor = options.axis_field_target/jnp.mean(jax.vmap(initial_field.AbsB)(axis_points))
    # Fix normalization after physical scaling so current DOFs remain order one.
    base = Coils(base.curves, base.dofs_currents_raw*factor)
    theta0 = jnp.asarray(base.dofs)
    print(f"Preparing fixed QI classifier: {options.surface_points} surface samples/angle; h={options.classifier_h} m", flush=True)
    boundary = SurfaceClassifier(surface, h=options.classifier_h,
        padding=max(.1, 1.5*options.safety_margin))
    print(f"Seeding {options.particles} fixed births at boundary fraction {options.birth_fraction}", flush=True)
    particles = Particles.InitializeParticlesAroundSurfaceAxis(axis, options.particles,
        boundary_surface=surface, distance_mode="fraction_to_boundary",
        distance_from_axis=options.birth_fraction, random_seed=options.seed,
        n_arc_samples=options.birth_axis_samples,
        mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
        energy=options.energy_ev*ONE_EV)
    particles.random_keys = jax.random.split(jax.random.key(options.collision_seed), options.particles)
    if bool(jnp.any(jax.vmap(boundary.evaluate_xyz)(particles.initial_xyz)<=-options.safety_margin)):
        raise ValueError("birth outside safety surface; check classifier/surface resolution and input geometry")
    print(f"Prepared {len(base.dofs_currents)} independent coils at {options.axis_field_target} T; {theta0.size} scaled DOFs", flush=True)
    point, direction = boundary_line(surface, options.line_theta, options.line_phi, options.line_mode)
    ncoils = len(base.dofs_currents)
    field_at = lambda x: BiotSavart(base.with_dofs(theta0+options.max_change*x))

    def geometry(x):
        field = field_at(x)
        _,tangent,second=_fourier_geometry(field.coils.curves,second=True)
        speed=jnp.linalg.norm(tangent,axis=-1)
        length=jnp.mean(speed,axis=-1)
        curvature=jnp.linalg.norm(jnp.cross(tangent,second),axis=-1)/speed**3
        result = dict(axis=jnp.mean(jax.vmap(field.AbsB)(axis_points))-options.axis_field_target,
            length=options.length_target-length[:ncoils],
            curvature=options.curvature_target-curvature[:ncoils])
        if options.bdotn_weight:
            values = jnp.abs(field_normal_ratio(surface, field))-options.bdotn_target
            sharpness = 20/options.bdotn_target
            result["bdotn"] = -(jax.scipy.special.logsumexp(sharpness*values)
                -jnp.log(values.size))/sharpness
        return result

    safety = ClassifierStop(boundary, options.safety_margin)

    if options.collisions:
        from essos.background_species import BackgroundSpecies
        background = BackgroundSpecies(2, jnp.array([ELECTRON_MASS/PROTON_MASS, 2.]),
            jnp.array([-1., 1.]), jnp.full(2, options.background_density),
            jnp.full(2, options.background_temperature_ev))
    execution = ("independent" if jax.default_backend()=="gpu" else "shared") if options.execution=="auto" else options.execution
    if options.collisions:print(f"Native split execution: forward={execution}; reverse={execution}",flush=True)

    def trace(x):
        field = field_at(x)
        if options.collisions:
            from essos.split import trace_split, SplitStatus
            result = trace_split(field, particles, background, model="GuidingCenter",
                integrator="essos_dopri8", tmax=options.horizon, n_save=options.saves,
                timestep=options.collision_step, collision_dtmax=options.collision_dtmax,
                adaptive=True, orbit_adaptive=True,
                collision_tolerance=options.collision_tolerance, collision_controller="moments",
                moment_tolerance=options.moment_tolerance, orbit_dt0=options.initial_step,
                rtol=options.rtol, atol=options.atol, event=safety,
                max_steps=options.max_steps, max_orbit_steps=options.max_orbit_steps,
                adjoint="accepted", adjoint_checkpoint_steps=options.checkpoint_steps,
                adjoint_orbit_checkpoint_steps=options.checkpoint_steps,
                adjoint_max_accepted_steps=options.collision_capacity,
                adjoint_max_orbit_steps=options.accepted_capacity,
                execution=execution, reverse_execution="independent" if execution=="independent" else "shared")
            states = eqx.error_if(result.ys, jnp.any(result.status>SplitStatus.EVENT),
                "native split solve failed; inspect work/tape capacities")
        else:
            result = Tracing(field=field, particles=particles, model="GuidingCenterAdaptative",
                integrator="essos_dopri8", native_adjoint="accepted", native_controller="i",
                native_adjoint_checkpoint_steps=options.checkpoint_steps,
                native_adjoint_max_accepted_steps=options.accepted_capacity,
                maxtime=options.horizon, timestep=options.initial_step, times_to_trace=options.saves,
                rtol=options.rtol, atol=options.atol, max_steps=options.max_steps,
                stopping_criteria=safety, progress=False)
            states = result.trajectories
        return eqx.error_if(states, ~jnp.isfinite(states).all(), "nonfinite native GC trajectory")

    def metrics(x, states):
        field = field_at(x)
        return dict(loss_fraction=classifier_loss_fraction(states, safety, options.softness),
            distance=classifier_line_loss(states, point, direction, safety,
                options.softness, options.location_softness),
            heat_flux=classifier_energy_cost(states, field, particles.mass, particles.energy,
                "GuidingCenterAdaptative", safety, softness=options.softness))

    def particle(x):
        states = trace(x)
        # Trace only once and evaluate only the selected particle constraint.
        if objective == "loss_fraction":
            return classifier_loss_fraction(states, safety, options.softness)
        if objective == "distance":
            return classifier_line_loss(states, point, direction, safety,
                options.softness, options.location_softness)
        return classifier_energy_cost(states, field_at(x), particles.mass, particles.energy,
            "GuidingCenterAdaptative", safety, softness=options.softness)

    constraints = make_constraints(geometry, particle, options)
    return SimpleNamespace(base=base, theta0=theta0, particles=particles, field=field_at,
        surface=surface, axis_points=axis_points, boundary=boundary, line=(point, direction),
        geometry=geometry, particle=particle, trace=trace, metrics=metrics,
        constraints=constraints, initial=jnp.zeros_like(theta0))


def run(objective, defaults=None, argv=None):
    settings = dict(DEFAULTS, **(defaults or {}))
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in settings.items():
        kwargs = dict(default=value)
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = Path if isinstance(value, Path) else type(value)
        parser.add_argument("--"+name.replace("_", "-"), **kwargs)
    parser.add_argument("--evaluate-only", action="store_true", help="One value/gradient; skip ALM updates")
    o = parser.parse_args(argv)
    for name in ("particles", "saves", "surface_points", "birth_axis_samples", "maxiter", "inner_maxiter",
                 "accepted_capacity", "collision_capacity", "max_steps", "max_orbit_steps"):
        if getattr(o, name) < (2 if name in ("saves", "surface_points", "birth_axis_samples") else 1):
            parser.error(name+" is too small")
    if not (0<=o.birth_fraction<1 and 0<o.initial_step<=o.horizon and o.max_change>0
            and o.classifier_h>0 and o.safety_margin>0 and o.penalty>0 and o.bdotn_target>0):
        parser.error("invalid birth fraction, time, scale, classifier, penalty or Bn target")
    if o.execution not in ("auto", "shared", "independent"):
        parser.error("execution must be auto, shared or independent")
    if o.optimizer not in ("optimistix", "scipy") or not 0 < o.scipy_ftol < 1:
        parser.error("optimizer must be optimistix or scipy; scipy-ftol must lie between 0 and 1")
    if o.execution=="independent" and not o.collisions:
        parser.error("independent split execution requires --collisions")
    o.output.mkdir(parents=True, exist_ok=True)
    if (o.output/"summary.json").exists() or (o.output/"iterations").exists():
        parser.error("choose a fresh output directory")
    print(f"JAX backend: {jax.default_backend()}; devices: {jax.devices()}", flush=True)
    print(f"QI GC ALM: {objective}; native Dopri8 accepted-map AD; collisions={o.collisions}", flush=True)
    problem = build_problem(o, objective)
    problem.base.to_json(str(o.output/"initial_coils.json"))
    print("Initializing ALM multipliers and native value/gradient", flush=True)
    constraints = problem.constraints
    # Multiplier setup needs output shapes only; JIT prunes particle tracing.
    params = problem.initial, jax.jit(constraints.init)(problem.initial)
    if o.optimizer == "scipy":
        optimizer = scipy_optimizer(constraints, o)
    else:
        optimizer = alm.ALM_model_lbfgs(constraints, beta=o.beta, model_mu=o.model_mu,
            eta_tol=o.eta_tol, omega_tol=o.omega_tol, inner_maxiter=o.inner_maxiter,
            bounds=(-jnp.ones_like(params[0]), jnp.ones_like(params[0])))
        optimizer=compile_optimizer(optimizer,independent=o.collisions and o.execution!='shared')
    start = perf_counter()
    state, gradient, info = jax.block_until_ready(optimizer.init(params))
    initialization_s = perf_counter()-start
    records = []
    def record(iteration, seconds):
        values = dict(iteration=iteration, lagrangian=float(info[1]),
            particle_metric=float(info[2][3]),
            constraint_norm=float(alm.norm_constraints(info[2])),
            gradient_norm=float(jnp.linalg.norm(gradient[0])), seconds=seconds)
        if not np.isfinite(list(values.values())).all():
            raise RuntimeError("nonfinite ALM value or gradient")
        save_coil_iteration(o.output,iteration,
            np.asarray(problem.theta0)+o.max_change*np.asarray(params[0]),params[0])
        records.append(values)
        print(f"ALM {iteration}: particle={values['particle_metric']:.8g}; L={values['lagrangian']:.8g}; constraints={values['constraint_norm']:.4g}; "
            f"|grad|={values['gradient_norm']:.4g}; {seconds:.2f} s", flush=True)
        with (o.output/"history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=values)
            writer.writeheader(); writer.writerows(records)
    record(0, initialization_s)
    for iteration in range(1, 1+(0 if o.evaluate_only else o.maxiter)):
        if records[-1]["constraint_norm"]<=o.eta_tol and records[-1]["gradient_norm"]<=o.omega_tol:
            break
        phase="; first call also traces/compiles the update" if iteration==1 and o.optimizer=="optimistix" else ""
        print(f"ALM {iteration}: starting inner L-BFGS (max {o.inner_maxiter} steps){phase}",flush=True)
        start = perf_counter()
        params, state, gradient, info = jax.block_until_ready(optimizer.update(params, state, gradient, info))
        record(iteration, perf_counter()-start)
    print("Tracing initial/final coils for saved diagnostics", flush=True)
    trace = jax.jit(problem.trace, **(gpu_jit_options() if o.collisions and o.execution!='shared' else {}))
    initial, final = jax.block_until_ready((trace(problem.initial), trace(params[0])))
    initial_metrics = problem.metrics(problem.initial, initial)
    final_metrics = problem.metrics(params[0], final)
    print(f"Particle cost consistency: logged {records[0]['particle_metric']:.9g} -> {records[-1]['particle_metric']:.9g}; "
          f"retraced {float(initial_metrics[objective]):.9g} -> {float(final_metrics[objective]):.9g}", flush=True)
    optimized = problem.field(params[0]).coils
    optimized.to_json(str(o.output/"optimized_coils.json"))
    np.savez_compressed(o.output/"optimization.npz", parameters=params[0],
        initial_dofs=problem.theta0, optimized_dofs=optimized.dofs,
        births=problem.particles.initial_xyz, pitches=problem.particles.initial_vparallel_over_v,
        initial_trajectories=initial, optimized_trajectories=final, line_point=problem.line[0],
        line_direction=problem.line[1], lagrange=jax.flatten_util.ravel_pytree(params[1])[0])
    if o.save_plots:
        axis_rz = np.asarray(problem.axis_points).mean(axis=0)
        save_optimization_plots(o.output, problem.base, optimized, initial, final,
            axis=(float(jnp.mean(jnp.linalg.norm(problem.axis_points[:, :2], axis=1))),float(axis_rz[2])), seed=o.seed)
    report = dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(o).items()},
        objective=objective, source_branch="en/jax_13", source_commit="638916db823c0fe4689913b779eb473d55d6e3ae",
        initial={k:float(v) for k,v in initial_metrics.items()},
        final={k:float(v) for k,v in final_metrics.items()}, history=records,
        solver="essos_dopri8", adjoint="accepted", backend=jax.default_backend(),
        distance_definition="Per-save escape-centroid distance to an infinite boundary tangent line, as in classic jax_13",
        heat_definition="First-loss-weighted E_total*v_perp normalized by birth E*v; dimensionless proxy",
        optimizer=("ESSOS ALM with host SciPy L-BFGS-B and a reused native value/gradient executable"
                   if o.optimizer=="scipy" else "Current ESSOS ALM with bounded Optimistix LBFGS; no historical frozen-trajectory custom VJP"))
    report["cost_retrace"] = dict(logged_initial=records[0]["particle_metric"],
        retraced_initial=float(initial_metrics[objective]), logged_final=records[-1]["particle_metric"],
        retraced_final=float(final_metrics[objective]))
    report["runtime"] = dict(jax=jax.__version__, jaxlib=jaxlib.__version__,
        devices=[str(device) for device in jax.devices()], prng=str(jax.config.jax_default_prng_impl),
        threefry_partitionable=bool(jax.config.jax_threefry_partitionable))
    if o.optimizer == "scipy":
        report["optimizer_statistics"] = state
    (o.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    if o.save_plots:
        from .plot_native_gc_alm import plot_run
        plot_run(o.output, problem.surface, ClassifierStop(problem.boundary), vars(o))
    print(f"Saved {o.output.resolve()}", flush=True)
