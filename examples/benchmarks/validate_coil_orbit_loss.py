"""Reproduce the QA-coil orbit failure and verify production wall stopping.

The original eight-particle fixed-collision launch has no loss boundary. One
escaped guiding center approaches the poorly resolved near-coil field, where
the configured RHS becomes singular. Independent SciPy ODE solves diagnose
the same interval. Wall runs use the unchanged production tracer, Gaussian
sampler, fields and tolerances, with the reference wall from the existing
connection_length_coils.py example. This is a regression, not a timing study.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from .benchmark_collision_tracing import make_solver
from essos.background_species import BackgroundSpecies
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import Particles, LevelsetStoppingCriterion, _guiding_center_velocity
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart
from essos.orbit_error import OrbitErrorNorm
from essos.rk import RKStatus, solve_rk
from essos.split import SplitStatus, trace_split
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier

ROOT = Path(__file__).resolve().parents[2]
COIL = ROOT/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
WOUT = ROOT/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
SPEED = float(np.sqrt(2*4000*ONE_EV/M))


def setup():
    """Exact original eight-particle launches and production noise keys."""
    options = argparse.Namespace(particles=8, horizon=1e-4, fixed_dt=1e-7,
        adaptive_ceiling=1e-6, orbit_tolerance=1e-9, collision_tolerance=.001)
    field = BiotSavart(Coils.from_json(str(COIL)))
    points = jnp.column_stack((jnp.linspace(1.23, 1.27, 8), jnp.zeros((8, 2))))
    birth = Particles(points, mass=M, charge=Q, energy=4000*ONE_EV)
    keys = jax.random.split(jax.random.key(42), 8)
    birth.random_keys = keys
    background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([1e20]), jnp.array([1000.]))
    return options, field, birth, background, points, keys


def reference_wall(*, angular_interpolation="linear"):
    """Same scaled boundary, 8 cm clearance and grid as connection-length example.

    This is an explicit reference geometry, not a newly solved coil LCFS or a
    measured vessel. The field itself remains the bundled 75-point coil field.
    """
    surface = SurfaceRZFourier.from_wout_file(str(WOUT))
    scale = surface.rc[0]
    boundary = SurfaceRZFourier(surface.rc/scale, surface.zs/scale, surface.nfp,
        surface.mpol, surface.ntor, ntheta=64, nphi=64)
    return LevelsetStoppingCriterion(SurfaceClassifier(boundary, h=.03, padding=.16,
        angular_interpolation=angular_interpolation),
        maximum_distance=.08)


def bounded_trace(options, field, birth, background, wall, *, adaptive, saves):
    return trace_split(field, birth, background, model="GuidingCenter",
        integrator="essos_dopri8", tmax=options.horizon, timestep=options.fixed_dt,
        collision_dtmax=options.adaptive_ceiling if adaptive else options.fixed_dt,
        adaptive=adaptive, times=jnp.linspace(0., options.horizon, saves),
        orbit_dt0=options.fixed_dt, rtol=options.orbit_tolerance,
        atol=options.orbit_tolerance, collision_tolerance=options.collision_tolerance,
        collision_controller="conservative", event=wall, max_steps=20_000)


def check_wall_result(out, options, wall, *, adaptive):
    status, end = np.asarray(out.status), np.asarray(out.t)
    assert np.all(np.isin(status, [SplitStatus.SUCCESS, SplitStatus.EVENT])), status
    assert np.isfinite(out.y).all() and np.isfinite(out.ys).all()
    np.testing.assert_array_equal(end[status == SplitStatus.SUCCESS], options.horizon)
    if not adaptive:
        assert status[3] == SplitStatus.EVENT and 0 < end[3] < 7.87e-5
    # Cartesian GC collision kicks do not move position in this setup. All
    # wall hits must therefore be root-found orbit events, not kick crossings.
    levels = np.asarray(jax.vmap(lambda y: wall(0., y, None))(out.y))
    np.testing.assert_allclose(levels[status == SplitStatus.EVENT], 0., atol=2e-10, rtol=0.)
    assert np.all(levels[status == SplitStatus.SUCCESS] > 0)
    ts = np.asarray(out.ts)
    for i in np.flatnonzero(status == SplitStatus.EVENT):
        np.testing.assert_array_equal(np.asarray(out.ys)[i, ts >= end[i]],
            np.broadcast_to(np.asarray(out.y)[i], (np.count_nonzero(ts >= end[i]), out.y.shape[-1])))
    return levels


def diagnostics(field, y):
    b, _, magnitude, grad, curl, _, _ = field.gc_quantities(jnp.asarray(y[:3]))
    b, grad, curl = map(np.asarray, (b, grad, curl))
    magnitude = float(magnitude)
    rho = np.sqrt(2*M*float(y[4])*magnitude)/(abs(Q)*magnitude)
    gamma = np.asarray(field.coils.gamma)
    segment = np.roll(gamma, -1, axis=1)-gamma
    delta = np.asarray(y[:3])-gamma
    fraction = np.clip(np.sum(delta*segment, axis=-1)/np.sum(segment**2, axis=-1), 0., 1.)
    distance = np.min(np.linalg.norm(delta-fraction[..., None]*segment, axis=-1))
    return dict(B_tesla=magnitude, Bstar_parallel_over_B=float(np.dot(b,
        b+float(y[3])*M/Q*curl)/np.dot(b, b)), gyroradius_m=float(rho),
        rho_over_LB=float(rho*np.linalg.norm(grad)/magnitude),
        nearest_coil_polyline_m=float(distance),
        energy_ev=float((M*float(y[3])**2/2+float(y[4])*magnitude)/ONE_EV))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or jax.default_backend() != "cpu":
        parser.error("a fresh output path and CPU are required")
    options, field, birth, background, points, keys = setup()
    sources = ("essos/split.py", "essos/rk.py", "essos/collisions.py", "essos/dynamics.py",
        "essos/fields.py", "essos/coils.py", "essos/surfaces.py", "essos/orbit_error.py",
        "examples/benchmarks/benchmark_collision_tracing.py",
        "examples/benchmarks/validate_coil_orbit_loss.py",
        "examples/fieldline_tracing/connection_length_coils.py")
    report = dict(complete=False, metadata=dict(backend=jax.default_backend(),
        settings=vars(options), seed=42, initial_xyz=np.asarray(points).tolist(),
        initial_pitch=np.asarray(birth.initial_vparallel_over_v).tolist(),
        source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        coil_sha256=hashlib.sha256(COIL.read_bytes()).hexdigest(),
        wall_input_sha256=hashlib.sha256(WOUT.read_bytes()).hexdigest(),
        wall="Reference scaled QA boundary + 0.08 m, as connection_length_coils.py; h=0.03 m, padding=0.16 m",
        scope="Unchanged production Brownian sampler, orbit/collision maps and coil field; optional wall event only"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Reproducing unbounded eight-particle fixed collisions", flush=True)
    unbounded = make_solver("native_collisions_fixed", field, background, birth,
        jnp.array([0., options.horizon]), options)(points, keys)
    _, final, end, status, counts = jax.device_get(unbounded)
    np.testing.assert_array_equal(status, [0, 0, 0, SplitStatus.MIN_STEP, 0, 0, 0, 0])
    initial, start = jnp.asarray(final[3]), float(end[3])
    finish = (int(round(start/options.fixed_dt))+1)*options.fixed_dt
    electric = Electric_field_zero()

    def rhs(t, y, unused):
        dx, dv = _guiding_center_velocity(field, electric, Q, M, y[:3], y[3], mu=y[4])
        return jnp.concatenate([dx, jnp.array([dv, 0.])])

    native = jax.jit(lambda y: solve_rk(rhs, y, t0=start, t1=finish, dt0=finish-start,
        method="dopri8", error_norm=OrbitErrorNorm(1e-9, 1e-9, SPEED), max_steps=10_000))(initial)
    native = jax.device_get(native)
    assert native.status == RKStatus.MIN_STEP
    report["unbounded"] = dict(status=status.tolist(), end=end.tolist(), counts=counts.tolist(),
        failing_particle_index=3, state_before_failed_interval=np.asarray(initial).tolist(),
        isolated_terminal_time=float(native.t), isolated_terminal_state=np.asarray(native.y).tolist(),
        isolated_stats=[int(x) for x in native.stats],
        initial_diagnostics=diagnostics(field, initial), terminal_diagnostics=diagnostics(field, native.y))

    scale = jnp.array([1., 1., 1., SPEED])
    scaled_rhs = jax.jit(lambda t, z: rhs(t, jnp.append(z*scale, initial[4]), None)[:4]/scale)
    report["independent_ode_references"] = []
    for method in ("DOP853", "RK45"):
        reference = solve_ivp(lambda t, z: np.asarray(scaled_rhs(t, jnp.asarray(z))),
            (start, finish), np.asarray(initial[:4]/scale), method=method, rtol=1e-9, atol=1e-9)
        assert not reference.success and "spacing" in reference.message
        assert abs(reference.t[-1]-float(native.t)) < 1e-12
        state = jnp.append(jnp.asarray(reference.y[:, -1])*scale, initial[4])
        row = dict(method=method, success=reference.success, message=reference.message,
            terminal_time=float(reference.t[-1]), rhs_evaluations=reference.nfev,
            terminal_diagnostics=diagnostics(field, state))
        report["independent_ode_references"].append(row)
        print(json.dumps(row), flush=True)

    # Vary only diagnostic copies of the coil quadrature at the same point.
    # This is a sensitivity check, not a converged field reference or a change
    # to any production orbit/collision trajectory below.
    report["near_coil_quadrature_sensitivity"] = []
    for segments in (75, 150, 300, 600):
        coils = Coils.from_json(str(COIL))
        coils.n_segments = segments
        report["near_coil_quadrature_sensitivity"].append(dict(segments=segments,
            **diagnostics(BiotSavart(coils), native.y)))

    wall = reference_wall()
    birth_levels = np.asarray(jax.vmap(lambda p: wall(0., p, None))(points))
    assert np.all(birth_levels > 0)
    report["initial_wall_levels_m"] = birth_levels.tolist()
    report["wall_cases"] = []
    for saves in (2, 101):
        for adaptive in (False, True):
            print(f"Production wall run: adaptive={adaptive}, saves={saves}", flush=True)
            out = jax.device_get(bounded_trace(options, field, birth, background, wall,
                adaptive=adaptive, saves=saves))
            levels = check_wall_result(out, options, wall, adaptive=adaptive)
            row = dict(adaptive=adaptive, saves=saves, status=out.status.tolist(),
                terminal_time=out.t.tolist(), terminal_state=out.y.tolist(),
                wall_levels_m=levels.tolist(),
                stats={name: np.asarray(value).tolist() for name, value in out.stats._asdict().items()})
            report["wall_cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps({k: row[k] for k in ("adaptive", "saves", "status", "terminal_time")}), flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("Complete: both independent ODE references confirm the singular interval; all four wall runs terminate successfully", flush=True)


if __name__ == "__main__":
    main()
