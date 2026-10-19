"""Small production QA-coil split-AD gradient and warm value/gradient benchmark.

Fixed cases freeze both grids by construction and compare directional finite
differences using the unchanged production Gaussian/bridge sampler. Adaptive
cases check complete finite gradients and forward equivalence; their exact
frozen-schedule gradient oracle is exercised by tests/test_split_ad.py. This
report does not compare different SDE discretizations or noise generators.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")

import jax
import jax.numpy as jnp
import numpy as np

from essos.background_species import BackgroundSpecies
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import Particles
from essos.fields import BiotSavart
from essos.particle_losses import (mean_final_radius, classifier_loss_fraction,
    classifier_loss_location, classifier_energy_cost)
from essos.split import trace_split
from .validate_coil_orbit_loss import reference_wall

ROOT = Path(__file__).resolve().parents[2]
OBJECTIVES = ("mean_final_radius", "loss_fraction", "loss_location", "heat_proxy")


def build_case(options, model, adaptive, objective, boundary):
    base = Coils.from_json(str(ROOT / "examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"))
    parameters = jnp.asarray(base.dofs)
    reference = BiotSavart(base)
    axis = (float(reference.r_axis), float(reference.z_axis))
    points = jnp.column_stack((jnp.linspace(1.23, 1.27, options.particles),
                              jnp.zeros((options.particles, 2))))
    pitch = jnp.linspace(.35, .65, options.particles)
    energy = 4000*ONE_EV
    speed = float(np.sqrt(2*energy/M))
    horizon = options.gc_horizon if model == "gc" else options.lorentz_horizon
    physical_model = "GuidingCenter" if model == "gc" else "FullOrbit"
    background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
                                   jnp.array([1e20]), jnp.array([1000.]))
    state_scale = jnp.array([1., 1., 1., speed, energy] if model == "gc" else [1.]*3+[speed]*3)

    def cost(theta, adjoint):
        field = BiotSavart(base.with_dofs(theta))
        birth = Particles(points, initial_vparallel_over_v=pitch, energy=energy, mass=M, charge=Q,
            field=field if model == "lorentz" else None, phase_angle_full_orbit=jnp.pi)
        out = trace_split(field, birth, background, model=physical_model, tmax=horizon,
            timestep=horizon/(4 if adaptive else 8), collision_dtmax=horizon/(2 if adaptive else 8),
            orbit_dt0=horizon/(4 if adaptive else options.fixed_orbit_steps),
            adaptive=adaptive, orbit_adaptive=adaptive, collision_tolerance=options.collision_tolerance,
            collision_controller=options.controller, integrator="essos_dopri8" if options.method == "dopri8" else "essos_rk45",
            times=jnp.linspace(0., horizon, options.saves), rtol=options.tolerance, atol=options.tolerance,
            max_steps=options.max_steps, max_orbit_steps=options.max_orbit_steps,
            adjoint=adjoint, adjoint_checkpoint_steps=options.checkpoint_steps,
            adjoint_orbit_checkpoint_steps=options.orbit_checkpoint_steps,
            adjoint_max_accepted_steps=options.accepted_capacity,
            adjoint_max_orbit_steps=options.orbit_capacity)
        if objective == "mean_final_radius":
            value = mean_final_radius(out.ys, *axis)
        elif objective == "loss_fraction":
            value = classifier_loss_fraction(out.ys, boundary)
        elif objective == "loss_location":
            value = classifier_loss_location(out.ys, jnp.array([1.1,.1,0.]), boundary)
        else:
            value = classifier_energy_cost(out.ys, field, M, energy, physical_model,
                                            boundary, kind="heat_jax10")
        return value, (out.ys, out.stats, out.status)

    return parameters, cost, state_scale, base.dof_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--modes", nargs="+", choices=("fixed", "adaptive"), default=["fixed", "adaptive"])
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=["mean_final_radius"])
    parser.add_argument("--method", choices=("dopri5", "dopri8"), default="dopri8")
    parser.add_argument("--controller", choices=("conservative", "paper", "candidate"), default="conservative")
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--saves", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gc-horizon", type=float, default=2e-6)
    parser.add_argument("--lorentz-horizon", type=float, default=2e-7)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--collision-tolerance", type=float, default=.003)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--max-orbit-steps", type=int, default=128)
    parser.add_argument("--accepted-capacity", type=int, default=128)
    parser.add_argument("--orbit-capacity", type=int, default=2048)
    parser.add_argument("--checkpoint-steps", type=int, default=16)
    parser.add_argument("--orbit-checkpoint-steps", type=int, default=16)
    parser.add_argument("--fixed-orbit-steps", type=int, default=64)
    parser.add_argument("--separate-phases", action="store_true",
        help="Also time ordinary forward, forward linearization and a standalone reverse pullback")
    options = parser.parse_args()
    if options.output.exists() or jax.default_backend() != "cpu":
        parser.error("choose a new output path and CPU backend")
    if not 1 <= options.particles <= 8 or options.saves < 2 or options.repeats < 3:
        parser.error("use 1..8 particles, at least two saves and at least three repeats")
    bounds = (options.max_steps, options.max_orbit_steps, options.accepted_capacity, options.orbit_capacity,
              options.checkpoint_steps, options.orbit_checkpoint_steps, options.fixed_orbit_steps)
    if min(bounds) < 1 or any(not np.isfinite(x) or x <= 0 for x in
        (options.gc_horizon, options.lorentz_horizon, options.tolerance, options.collision_tolerance)):
        parser.error("positive finite tolerances/horizons and positive storage/work bounds required")
    sources = ("essos/split.py", "essos/split_ad.py", "essos/rk.py", "essos/rk_ad.py",
        "essos/collisions.py", "essos/collision_controller.py", "essos/collision_error.py",
        "essos/background_species.py", "essos/coulomb.py", "essos/fields.py", "essos/coils.py",
        "essos/dynamics.py", "essos/particle_losses.py", "essos/surfaces.py",
        "examples/benchmarks/benchmark_split_ad.py", "examples/benchmarks/validate_coil_orbit_loss.py")
    fingerprints = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    report = dict(complete=False, metadata=dict(backend=jax.default_backend(), jax=jax.__version__,
        settings={name: str(value) if isinstance(value,Path) else value for name,value in vars(options).items()},
        source_sha256=fingerprints,
        input_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json",
             "examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc")},
        derivative_policy="Discrete accepted orbit/collision maps; frozen schedules and exact production Wiener increments",
        validation_policy="Fixed-case FD uses both fixed grids and identical per-particle keys. Adaptive-case frozen replay is covered by unit tests; ordinary adaptive FD is not used as its oracle.",
        timing_scope=("Blocked warm value_and_grad includes forward solve and excludes compilation. "
            "Optional phase timings report ordinary forward, public jax.vjp of a jitted cost "
            "(forward linearization includes tape construction/host dispatch), and a separately jitted "
            "pullback reusing baseline residuals. Separate phases need not sum to fused value_and_grad."),
        memory_scope="Compiled value_and_grad arguments + outputs + temporaries - aliases; not process peak RSS",
        parameters="136 bundled QA coil Fourier/normalized-current DOFs; fixed GC births, energy, pitch and gyrophase; fixed reference axis/surface"), rows=[])
    boundary = reference_wall(angular_interpolation="periodic_cubic").classifier if any(
        name != "mean_final_radius" for name in options.objectives) else None
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for model in options.models:
        for mode in options.modes:
            for objective in options.objectives:
                print(f"Checking {model} {mode} {objective}", flush=True)
                theta, cost, scale, names = build_case(options, model, mode == "adaptive", objective, boundary)
                evaluate = jax.jit(jax.value_and_grad(lambda p:cost(p,"accepted"), has_aux=True))
                start = perf_counter()
                executable = evaluate.lower(theta).compile()
                compile_seconds = perf_counter()-start
                (value,(ys,stats,status)), gradient = jax.block_until_ready(evaluate(theta))
                primal = jax.jit(lambda p:cost(p,"none"))
                reference,(reference_ys,reference_stats,reference_status) = jax.block_until_ready(primal(theta))
                state_difference = float(jnp.max(jnp.abs(ys-reference_ys)/scale))
                np.testing.assert_array_equal(status, reference_status)
                for name in stats._fields:
                    if name == "max_accepted_error":
                        np.testing.assert_allclose(getattr(stats,name),getattr(reference_stats,name),rtol=2e-12,atol=1e-25)
                    else:
                        np.testing.assert_array_equal(getattr(stats,name),getattr(reference_stats,name))
                np.testing.assert_allclose(value, reference, rtol=2e-12, atol=1e-25)
                if np.any(status) or not np.isfinite(gradient).all() or state_difference > 2e-12:
                    raise RuntimeError("failed/nonfinite solve or forward replay mismatch")
                row = dict(model=model,mode=mode,objective=objective,value=float(value),
                    gradient=np.asarray(gradient).tolist(),parameter_labels=list(names),
                    gradient_l2=float(jnp.linalg.norm(gradient)),status=np.asarray(status).tolist(),
                    stats={name:np.asarray(value).tolist() for name,value in stats._asdict().items()},
                    max_normalized_forward_difference=state_difference,compile_seconds=compile_seconds)
                if mode == "fixed":
                    direction=jnp.asarray(np.random.default_rng(12).normal(size=theta.size))
                    direction/=jnp.linalg.norm(direction)
                    expected=float(jnp.dot(gradient,direction))
                    fd=[]
                    for eps in (1e-5,3e-6,1e-6):
                        finite=float((primal(theta+eps*direction)[0]-primal(theta-eps*direction)[0])/(2*eps))
                        fd.append(dict(epsilon=eps,ad=expected,finite_difference=finite,
                            relative_error=abs(finite-expected)/max(abs(finite),abs(expected),1e-30)))
                    if min(item["relative_error"] for item in fd) > 2e-5:
                        raise RuntimeError(f"fixed-grid coil-gradient FD mismatch: {fd}")
                    row["directional_fd"]=fd
                samples=[]
                for _ in range(options.repeats):
                    start=perf_counter()
                    jax.block_until_ready(evaluate(theta))
                    samples.append((perf_counter()-start)*1000)
                memory=executable.memory_analysis()
                row.update(warm_samples_ms=samples,warm_median_ms=float(np.median(samples)),
                    compiler_buffer_bytes=int(memory.argument_size_in_bytes+memory.output_size_in_bytes
                        +memory.temp_size_in_bytes-memory.alias_size_in_bytes),complete=True)
                if options.separate_phases:
                    # Match the collisionless benchmark's public JAX VJP path.
                    # Returning a Python pullback through jit is avoided on
                    # JAX 0.11.2 because it miscounts some lifted constants.
                    accepted_cost = jax.jit(lambda p:cost(p,"accepted"))
                    linear_value, pullback, linear_aux = jax.block_until_ready(
                        jax.vjp(accepted_cost,theta,has_aux=True))
                    reverse = jax.jit(lambda pb,seed:pb(seed)[0])
                    seed = jnp.ones_like(linear_value)
                    reverse_gradient = jax.block_until_ready(reverse(pullback,seed))
                    np.testing.assert_allclose(linear_value,value,rtol=2e-12,atol=1e-25)
                    np.testing.assert_allclose(reverse_gradient,gradient,rtol=2e-8,atol=1e-11)
                    phase_samples = []
                    for _ in range(options.repeats):
                        start = perf_counter()
                        jax.block_until_ready(primal(theta))
                        ordinary_ms = (perf_counter()-start)*1000
                        start = perf_counter()
                        jax.block_until_ready(jax.vjp(accepted_cost,theta,has_aux=True))
                        linear_ms = (perf_counter()-start)*1000
                        # Reuse the original pullback: a new Python pullback can
                        # retrigger compilation and spoil a warm measurement.
                        start = perf_counter()
                        jax.block_until_ready(reverse(pullback,seed))
                        reverse_ms = (perf_counter()-start)*1000
                        phase_samples.append(dict(ordinary_forward_ms=ordinary_ms,
                            forward_linearization_ms=linear_ms,reverse_ms=reverse_ms))
                    medians = {k:float(np.median([s[k] for s in phase_samples])) for k in phase_samples[0]}
                    row.update(phase_samples_ms=phase_samples,phase_medians_ms=medians,
                        total_over_ordinary_forward=row["warm_median_ms"]/medians["ordinary_forward_ms"],
                        reverse_over_ordinary_forward=medians["reverse_ms"]/medians["ordinary_forward_ms"])
                report["rows"].append(row)
                options.output.write_text(json.dumps(report,indent=2)+"\n")
                print(json.dumps({name:row[name] for name in
                    ("model","mode","objective","value","gradient_l2","warm_median_ms","compiler_buffer_bytes",
                     "phase_medians_ms","total_over_ordinary_forward","reverse_over_ordinary_forward") if name in row}),flush=True)
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest for name,digest in fingerprints.items()):
        raise RuntimeError("benchmark sources changed during measurement")
    report["complete"]=True
    options.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
