"""Four production integration cases on the bundled QA-coil GC example.

Uses the field, alpha species, 4 keV energy and radial launch interval from
examples/particle_tracing/trace_particles_coils_guidingcenter.py. CPU timings
exclude construction/compilation, block on completion and interleave cases.
This is a tracing cost comparison; the small ensemble is not a physics test.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

for name, value in (("JAX_PLATFORMS", "cpu"), ("ESSOS_CPU_DEVICES", "1"),
        ("ESSOS_PROGRESS", "0"), ("ESSOS_XLA_CACHE", "")):
    os.environ.setdefault(name, value)

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from essos.background_species import BackgroundSpecies
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV, SPEED_OF_LIGHT
from essos.dynamics import Particles, GuidingCenter, _SDE_MODELS
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart
from essos.orbit_error import OrbitErrorNorm
from essos.rk import solve_rk
from essos.split import trace_split


def make_solver(label, field, background, birth, times, options):
    horizon = options.horizon
    electric = Electric_field_zero()
    speed = float(np.sqrt(2*4000*ONE_EV/M))
    if label.startswith("native_collisions"):
        adaptive = label.endswith("adaptive")
        def solve(points, keys):
            particles = Particles(points, mass=M, charge=Q, energy=4000*ONE_EV,
                initial_vparallel_over_v=birth.initial_vparallel_over_v)
            particles.random_keys = keys
            out = trace_split(field, particles, background, model="GuidingCenter",
                integrator="essos_dopri8", tmax=horizon, timestep=options.fixed_dt,
                collision_dtmax=options.adaptive_ceiling if adaptive else options.fixed_dt,
                adaptive=adaptive, times=times, orbit_dt0=options.fixed_dt,
                rtol=options.orbit_tolerance, atol=options.orbit_tolerance,
                collision_tolerance=options.collision_tolerance, collision_controller="conservative",
                max_steps=20_000)
            counts = jnp.stack((out.stats.steps, out.stats.accepted, out.stats.rejected,
                out.stats.orbit_steps, out.stats.orbit_rejected), axis=1)
            return out.ys, out.y, out.t, out.status, counts
        return jax.jit(solve)
    if label == "native_no_collisions":
        def solve(points, keys):
            initial = jnp.column_stack((points, birth.initial_vparallel))
            def one(y):
                out = solve_rk(lambda t, y, a: GuidingCenter(t, y, (field, birth, electric)), y,
                    t1=horizon, dt0=options.fixed_dt, ts=times, method="dopri8", adaptive=True,
                    rtol=options.orbit_tolerance, atol=options.orbit_tolerance,
                    error_norm=OrbitErrorNorm(options.orbit_tolerance, options.orbit_tolerance, speed),
                    max_steps=20_000)
                counts = jnp.array([out.stats.steps, out.stats.accepted, out.stats.rejected,
                    out.stats.steps, out.stats.rejected])
                return out.ys, out.y, out.t, out.status, counts
            return jax.vmap(one)(initial)
        return jax.jit(solve)
    drift, noise, solver, dim = _SDE_MODELS["GuidingCenterCollisionsMuFixed"]
    args = (field, birth, electric, background, 1.)
    scale = jnp.array([1., 1., 1., SPEED_OF_LIGHT, SPEED_OF_LIGHT**2*M])
    def solve(points, keys):
        magnetic = jax.vmap(field.AbsB)(points)
        initial = jnp.column_stack((points, birth.initial_vparallel,
            M*birth.initial_vperpendicular**2/(2*magnetic)))
        def one(y, key):
            bm = diffrax.VirtualBrownianTree(0., horizon, tol=options.fixed_dt*.5,
                shape=(dim,), key=key, levy_area=diffrax.SpaceTimeTimeLevyArea)
            terms = diffrax.MultiTerm(diffrax.ODETerm(lambda t, y, a: drift(t, y, args)),
                diffrax.ControlTerm(lambda t, y, a: noise(t, y, args), bm))
            out = diffrax.diffeqsolve(terms, solver(), t0=0., t1=horizon, dt0=options.fixed_dt,
                y0=y/scale, saveat=diffrax.SaveAt(ts=times),
                stepsize_controller=diffrax.ConstantStepSize(), max_steps=20_000, throw=False)
            ys = out.ys*scale
            counts = jnp.array([out.stats["num_steps"], out.stats["num_accepted_steps"],
                out.stats["num_rejected_steps"], 0, 0])
            return ys, ys[-1], out.ts[-1], jnp.where(out.result == diffrax.RESULTS.successful, 0, 1), counts
        return jax.vmap(one)(initial, keys)
    return jax.jit(solve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--horizon", type=float, default=1e-4)
    parser.add_argument("--fixed-dt", type=float, default=1e-7)
    parser.add_argument("--adaptive-ceiling", type=float, default=1e-6)
    parser.add_argument("--saves", type=int, default=101)
    parser.add_argument("--orbit-tolerance", type=float, default=1e-9)
    parser.add_argument("--collision-tolerance", type=float, default=.001)
    options = parser.parse_args()
    if options.output.exists() or jax.default_backend() != "cpu":
        parser.error("fresh output path and CPU required")
    if not 2 <= options.particles <= 128 or options.repeats < 3 or options.saves < 2:
        parser.error("2..128 particles, >=3 repeats and >=2 saves required")
    root = Path(__file__).resolve().parents[2]
    coil_path = root/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
    field = BiotSavart(Coils.from_json(str(coil_path)))
    points = jnp.column_stack((jnp.linspace(1.23, 1.27, options.particles), jnp.zeros((options.particles, 2))))
    birth = Particles(points, mass=M, charge=Q, energy=4000*ONE_EV)
    keys = jax.random.split(jax.random.key(42), options.particles)
    background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([1e20]), jnp.array([1000.]))
    times = jnp.linspace(0., options.horizon, options.saves)
    names = ("native_collisions_adaptive", "native_collisions_fixed", "diffrax_collisions_fixed", "native_no_collisions")
    sources = ("essos/split.py", "essos/collisions.py", "essos/rk.py", "essos/dynamics.py",
        "essos/background_species.py", "essos/fields.py", "essos/coils.py", "essos/orbit_error.py",
        "examples/benchmarks/benchmark_collision_tracing.py",
        "examples/particle_tracing/trace_particles_coils_guidingcenter.py")
    report = dict(complete=False, metadata=dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
        backend=jax.default_backend(), jax=jax.__version__, diffrax=diffrax.__version__,
        field="Unchanged bundled Landreman-Paul QA BiotSavart coils", mass=float(M), charge=float(Q), energy_ev=4000.,
        initial_xyz=np.asarray(points).tolist(), initial_pitch=np.asarray(birth.initial_vparallel_over_v).tolist(),
        seed=42, background="Production default NRL; protons, 1 keV, 1e20 m^-3",
        coil_sha256=hashlib.sha256(coil_path.read_bytes()).hexdigest(),
        scope="Compiled production integration kernels; no plotting/postprocessing/compilation in warm time; no events; no spatial kicks",
        example_changes="Original two-particle default; 101 saves, fixed dt=1e-7 and adaptive collision ceiling=1e-6; original field/alpha energy/radial launches/horizon",
        count_columns=["trials", "accepted", "rejected", "native_orbit_trials", "native_orbit_rejected"],
        interpretation="Collision runs have different physical dynamics from the collisionless cost baseline; timings alone do not establish equal statistical accuracy",
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}), rows=[])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    executables = {}
    for name in names:
        print(f"Compiling {name}", flush=True)
        function = make_solver(name, field, background, birth, times, options)
        started = perf_counter()
        executable = function.lower(points, keys).compile()
        compile_seconds = perf_counter()-started
        output = jax.device_get(jax.block_until_ready(executable(points, keys)))
        ys, final, end, status, counts = output
        if np.any(status != 0) or not np.isfinite(ys).all():
            report["failure"] = dict(case=name, status=status.tolist(), end=end.tolist(),
                counts=counts.tolist(), terminal_state=final.tolist())
            options.output.write_text(json.dumps(report, indent=2)+"\n")
            raise RuntimeError(f"incomplete/nonfinite case: {name}, statuses={status}")
        np.testing.assert_allclose(end, options.horizon, rtol=1e-12, atol=0.)
        np.testing.assert_allclose(ys[:, -1], final, rtol=0., atol=0.)
        accepted = float(np.mean(counts[:, 1]))
        row = dict(case=name, compile_seconds=compile_seconds, counts_per_particle=counts.tolist(),
            accepted_mean=accepted, rejected_mean=float(np.mean(counts[:, 2])),
            orbit_trials_mean=float(np.mean(counts[:, 3])), orbit_rejected_mean=float(np.mean(counts[:, 4])),
            effective_mean_interval_seconds=options.horizon/accepted, warm_samples_ms=[],
            terminal_state=np.asarray(final).tolist(), failures=int(np.count_nonzero(status)))
        report["rows"].append(row)
        executables[name] = executable
        options.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k: v for k, v in row.items() if k not in ("counts_per_particle", "terminal_state")}), flush=True)
    rng = np.random.default_rng(2711)
    for repeat in range(options.repeats):
        for index in rng.permutation(len(names)):
            name = names[index]
            started = perf_counter()
            jax.block_until_ready(executables[name](points, keys))
            report["rows"][index]["warm_samples_ms"].append(1000*(perf_counter()-started))
        print(f"Warm repeat {repeat+1}/{options.repeats} complete", flush=True)
    for row in report["rows"]:
        row["warm_median_ms"] = float(np.median(row["warm_samples_ms"]))
        row["warm_min_ms"] = float(np.min(row["warm_samples_ms"]))
    report["complete"] = True
    options.output.write_text(json.dumps(report, indent=2)+"\n")
    table = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in report["rows"]]
    with options.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(dict(complete=True, rows=table)), flush=True)


if __name__ == "__main__":
    main()
