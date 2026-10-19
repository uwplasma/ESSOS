"""Batched 2,000-particle physical validation, with checkpointed results.

No Diffrax solves, source, field changes or timing repetitions. Thermal tests
hold ln Lambda constant, giving an exact Maxwellian control. Beam tests retain
ESSOS's speed-dependent physical coefficients and an independent backward-PDE
reference. Every configuration has 2,000 independent particles; configurations
share ordered keys but do not claim a common Brownian path across step grids.
"""
import argparse
import json
from math import erf
from pathlib import Path
from time import perf_counter

from .validate_collisional_slowing_down import native_solver
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest, maxwell, t as student_t

from essos.background_species import chandrasekhar, d_chandrasekhar
from essos.collisions import (CollisionCoefficients, guiding_center_collision_step,
    particle_collision_step, guiding_center_spatial_diffusion, guiding_center_spatial_increment)
from essos.split import solve_split
from .nonrelativistic_collision_reference import ProtonBackground, proton_first_passage, passage_statistics
from .spatial_collision_reference import stopped_spatial_reference, stopped_spatial_moments, fixed_time_spatial_msd
from .analyze_collisional_convergence import conditional_spatial_statistics


METHODS = ("fixed", "conservative", "paper", "candidate")
COLORS = dict(fixed="#0072B2", conservative="#D55E00", paper="#009E73", candidate="#CC79A7")


def summary(values, expected):
    x = np.asarray(values, dtype=float)
    se = x.std(ddof=1)/np.sqrt(len(x))
    half = student_t.ppf(.975, len(x)-1)*se
    return dict(mean=float(x.mean()), expected=float(expected), se=float(se), half95=float(half),
        difference=float(x.mean()-expected), relative_difference=float(x.mean()/expected-1) if expected else None,
        z=float((x.mean()-expected)/se) if se else None)


def constant_log_coefficients(speed, *, full_orbit=False):
    """Nonrelativistic equal-proton Maxwellian coefficients with fixed ln Lambda.

    Velocity is in sigma=sqrt(T/m); time is in 1/nu_s(sigma). Spatial units
    are rho_sigma. This control leaves the production coefficients untouched.
    """
    normalizer = chandrasekhar(jnp.asarray(1/np.sqrt(2.)))
    def values(v):
        x = v/jnp.sqrt(2.)
        g = chandrasekhar(x)
        dp = g/(v*normalizer)
        dt = (jax.scipy.special.erf(x)-g)/(2*v*normalizer)
        ddp = (d_chandrasekhar(x)/(jnp.sqrt(2.)*v)-g/v**2)/normalizer
        return jnp.array([dp, dt, ddp, -v*dp])
    value, slope = jax.jvp(values, (speed,), (jnp.ones_like(speed),))
    dp, dt, ddp, q = value
    dk = slope[3]+slope[2]+2*(slope[0]-slope[1])/speed-2*(dp-dt)/speed**2 if full_orbit else jnp.zeros_like(speed)
    return CollisionCoefficients(dp, dt, ddp, q, slope[3], dk)


def thermal_solver(model, method, horizon, times, *, coefficient_law=constant_log_coefficients):
    """Use the actual native collision maps and driver for the analytic control."""
    gc, adaptive = model == "gc", method != "fixed"
    controller = method if adaptive else "conservative"
    def batch(initial, keys, tolerance, cap, first):
        def one(y0, key):
            def kick(t, y, h, dw, args):
                speed = y[3] if gc else jnp.linalg.norm(y)
                c = coefficient_law(speed, full_orbit=not gc)
                kwargs = dict(tolerance=tolerance, return_error_model=adaptive and controller == "candidate")
                if gc:
                    trial = guiding_center_collision_step(speed, y[4], h, dw[-2:], c,
                        speed_floor=.05*np.sqrt(2.), **kwargs)
                    dx = guiding_center_spatial_increment(y[4], dw[:3], c, jnp.array([0., 0., 1.]), 1.)
                    value = y.at[:3].add(dx).at[3:5].set(trial.y)
                    value = value.at[5].add(4*guiding_center_spatial_diffusion(y[4], c, 1.)*h)
                    return trial._replace(y=value)
                return particle_collision_step(y, h, dw, c, **kwargs)
            return solve_split(lambda t, y, a: jnp.zeros_like(y), kick, y0,
                t1=horizon, dt0=first, dtmax=cap, ts=jnp.asarray(times), key=key,
                noise_dim=5 if gc else 3, adaptive=adaptive, collision_controller=controller,
                collision_wiener_norm=(lambda w: jnp.linalg.norm(w[-2:])) if gc else (lambda w: abs(w[2])),
                method="dopri5", max_steps=100_000)
        return jax.vmap(one)(initial, keys)
    return jax.jit(batch)


def equilibrium_reference(floor):
    """Exactly conditioned Maxwell speed for the GC reflecting lower wall."""
    excluded = maxwell.cdf(floor)
    raw = [maxwell.expect(lambda r, k=k: r**k, lb=floor, conditional=True) for k in range(1, 5)]
    # Isotropic pitch average of Eq. (36), not a drag-only approximation.
    g1 = float(chandrasekhar(jnp.asarray(1/np.sqrt(2.))))
    diffusivity = quad(lambda r: maxwell.pdf(r)*erf(r/np.sqrt(2.)) / (3*r*g1),
        floor, np.inf, epsabs=1e-11, epsrel=1e-11)[0]/(1-excluded)
    return dict(excluded_maxwell_mass=float(excluded), speed_raw_moments=raw,
        speed_mean=raw[0], speed_variance=raw[1]-raw[0]**2, speed2=raw[1],
        pitch_mean=0., pitch_variance=1/3, parallel_mean=0., parallel_variance=raw[1]/3,
        spatial_diffusivity=diffusivity,
        cdf=lambda r: np.maximum((maxwell.cdf(r)-excluded)/(1-excluded), 0.))


def thermal_statistics(speed, pitch, reference):
    r, xi = np.asarray(speed), np.asarray(pitch)
    metrics = dict(speed_mean=summary(r, reference["speed_mean"]),
        speed_variance=summary((r-reference["speed_mean"])**2, reference["speed_variance"]),
        speed2=summary(r*r, reference["speed2"]), pitch_mean=summary(xi, 0.),
        pitch_variance=summary(xi*xi, 1/3), parallel_mean=summary(r*xi, 0.),
        parallel_variance=summary((r*xi)**2, reference["parallel_variance"]))
    for name, values, cdf in (("speed_cdf", r, reference["cdf"]), ("pitch_cdf", xi, lambda z: np.clip((z+1)/2, 0., 1.))):
        check = kstest(values, cdf)
        metrics[name] = dict(distance=float(check.statistic), pvalue=float(check.pvalue))
    return metrics


def run_batches(executable, batches, parameters):
    outputs = []
    start = perf_counter()
    for inputs in batches:
        outputs.append(jax.device_get(jax.block_until_ready(executable(*inputs, *parameters))))
    output = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *outputs)
    if np.any((output.status != 0) & (output.status != 1)):
        raise RuntimeError(f"invalid physical solve: {np.unique(output.status, return_counts=True)}")
    return output, perf_counter()-start


def work(output):
    return dict(accepted_mean=float(output.stats.accepted.mean()), rejected_mean=float(output.stats.rejected.mean()),
        trials_mean=float(output.stats.steps.mean()), max_accepted_error=float(output.stats.max_accepted_error.max()),
        peak_pending=int(output.stats.peak_pending.max()))


def spatial_statistics(q, msd, reference_mean, reference_second=None):
    values = dict(integral=summary(q, reference_mean), displacement2=summary(msd, reference_mean),
        noise=conditional_spatial_statistics(msd, q), integral_samples=np.asarray(q).tolist(),
        displacement2_samples=np.asarray(msd).tolist())
    if reference_second is not None:
        values.update(integral_second=summary(np.asarray(q)**2, reference_second),
            displacement4=summary(np.asarray(msd)**2, 2*reference_second))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3706)
    parser.add_argument("--phases", nargs="+", choices=("thermal", "slowing", "spatial"), default=["thermal", "slowing", "spatial"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.particles <= 2000 or not 2 <= args.batch_size <= args.particles or args.particles % args.batch_size:
        parser.error("use at most the authorized 2000 particles and a dividing batch size")
    report = dict(complete=False, metadata=dict(particles=args.particles, batch_size=args.batch_size,
        seed=args.seed, phases=args.phases,
        thermal="Fixed ln Lambda analytic Maxwell control; native maps and split driver; normalized units",
        beam="Actual ESSOS speed-dependent coefficients, SI; independent backward-PDE first passage and spatial rewards",
        uncertainty="Independent particles within rows; shared ordered keys between settings, not identical Brownian paths; marginal 95% mean intervals and 99% CDF checks",
        interpretation="No finite-ensemble check certifies global error equal to local collision tolerance"), rows=[])
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
        for k in ("particles", "batch_size", "seed", "phases"):
            if report["metadata"][k] != getattr(args, k):
                raise ValueError(f"resume metadata mismatch: {k}")
        report["complete"] = False
    elif args.output.exists():
        raise ValueError("output exists; use --resume or a fresh output")
    completed = {r["id"] for r in report["rows"]}
    def save(row=None):
        if row:
            report["rows"].append(row)
            completed.add(row["id"])
            brief = {k: row[k] for k in ("id", "elapsed_seconds", "trials_mean", "max_accepted_error")}
            if row["phase"] == "thermal":
                stats = row["evolution"][-1]["statistics"]
                brief.update(speed_mean_difference=stats["speed_mean"]["relative_difference"],
                    speed_variance_difference=stats["speed_variance"]["relative_difference"],
                    speed_cdf=stats["speed_cdf"]["distance"], pitch_cdf=stats["pitch_cdf"]["distance"])
            if row["phase"] == "slowing":
                brief.update(stopping_mean_difference=row["statistics"]["moments"]["restricted_mean"]["value"]/
                    row["statistics"]["moments"]["restricted_mean"]["expected"]-1,
                    stopping_cdf=row["statistics"]["cdf"]["distance"])
            if "spatial" in row:
                brief.update(diffusion_integral_difference=row["spatial"]["integral"]["relative_difference"],
                    displacement2_difference=row["spatial"]["displacement2"]["relative_difference"])
            print(json.dumps(brief), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(args.seed), i))(jnp.arange(args.particles)).reshape((-1, args.batch_size))
    count = args.particles//args.batch_size
    physical = ProtonBackground()
    save()
    if "thermal" in args.phases:
        horizon, times = 2., np.array([0., .5, 1., 2.])
        rng = np.random.default_rng(args.seed+1)
        u, xi, phi = rng.uniform(size=(3, args.particles))
        xi, phi = 2*xi-1, 2*np.pi*phi
        for model in ("gc", "lorentz"):
            reference = equilibrium_reference(.05*np.sqrt(2.) if model == "gc" else 0.)
            excluded = reference["excluded_maxwell_mass"]
            speed = maxwell.ppf(excluded+(1-excluded)*u)
            initial = np.column_stack((np.zeros((args.particles, 3)), speed, xi, np.zeros(args.particles))) if model == "gc" else np.column_stack((speed*np.sqrt(1-xi*xi)*np.cos(phi), speed*np.sqrt(1-xi*xi)*np.sin(phi), speed*xi))
            initial = jnp.asarray(initial.reshape((count, args.batch_size, -1)))
            for method in METHODS:
                settings = (("coarse", .003, .0025, .0025), ("fine", .001, .000625, .000625)) if method == "fixed" else (("coarse", .003, .05, .005), ("fine", .001, .003125, .003125))
                pending = [s for s in settings if f"thermal/{model}/{method}/{s[0]}" not in completed]
                if not pending:
                    continue
                function = thermal_solver(model, method, horizon, times)
                executable = function.lower(initial[0], keys[0], *map(jnp.asarray, pending[0][1:])).compile()
                for setting, tol, cap, first in pending:
                    print(f"Running thermal/{model}/{method}/{setting}: N={args.particles}", flush=True)
                    output, elapsed = run_batches(executable, list(zip(initial, keys)), tuple(map(jnp.asarray, (tol, cap, first))))
                    if np.any(output.status != 0):
                        raise RuntimeError("thermal particles must reach the common physical horizon")
                    y = output.ys
                    r = y[:, :, 3] if model == "gc" else np.linalg.norm(y, axis=-1)
                    pitch = y[:, :, 4] if model == "gc" else y[:, :, 2]/r
                    row = dict(id=f"thermal/{model}/{method}/{setting}", phase="thermal", model=model,
                        method=method, setting=setting, tolerance=tol if method != "fixed" else None, cap=cap,
                        elapsed_seconds=elapsed, **work(output),
                        reference={k: v for k, v in reference.items() if k != "cdf"},
                        evolution=[dict(time=float(t), statistics=thermal_statistics(r[:, i], pitch[:, i], reference)) for i, t in enumerate(times)],
                        speed_samples=r[:, -1].tolist(), pitch_samples=pitch[:, -1].tolist())
                    if model == "gc":
                        row["spatial"] = spatial_statistics(output.y[:, 5], np.sum(output.y[:, :2]**2, axis=1), 4*reference["spatial_diffusivity"]*horizon)
                    if method != "fixed" and row["max_accepted_error"] > 1:
                        raise RuntimeError("accepted thermal estimate exceeds one")
                    save(row)
    if "slowing" in args.phases:
        reference = proton_first_passage(physical)
        horizon = 6*reference.mean
        ts = np.linspace(0., horizon, 21)
        spatial = stopped_spatial_moments(physical, [horizon])
        spectral = float(stopped_spatial_reference(physical).msd(horizon))
        fine = stopped_spatial_moments(physical, [horizon], nodes=1024, upper=10.)
        if max(abs(spatial["mean"][0]/spectral-1), abs(spatial["second_moment"][0]/fine["second_moment"][0]-1)) > 3e-4:
            raise RuntimeError("stopped spatial moments are not converged")
        report["stopping_reference"] = dict(horizon=horizon, mean=reference.mean, variance=reference.variance,
            restricted_mean=float(reference.restricted_moments(horizon)[0]), spatial_mean=spectral,
            spatial_integral_second=float(spatial["second_moment"][0]))
        from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
        rho = physical.sigma*PROTON_MASS/ELEMENTARY_CHARGE
        for model in ("gc", "lorentz"):
            for method in METHODS:
                settings = (("coarse", .003, .01, .01), ("fine", .001, .000625, .000625)) if method == "fixed" else (("coarse", .003, .05, .005), ("fine", .001, .003125, .003125))
                for shifted in (False, True):
                    pending = [s for s in settings if (not shifted or s[0] == "fine") and f"slowing/{model}/{method}/{s[0]}/{'shifted' if shifted else 'endpoint'}" not in completed]
                    if not pending:
                        continue
                    function = native_solver(model, method if method != "fixed" else "conservative", method != "fixed", physical, horizon, 2., spatial=model == "gc", times=ts, boundary_shift=shifted)
                    executable = function.lower(keys[0], *map(jnp.asarray, pending[0][1:])).compile()
                    for setting, tol, cap, first in pending:
                        identifier = f"slowing/{model}/{method}/{setting}/{'shifted' if shifted else 'endpoint'}"
                        print(f"Running {identifier}: N={args.particles}", flush=True)
                        output, elapsed = run_batches(executable, [(k,) for k in keys], tuple(map(jnp.asarray, (tol, cap, first))))
                        t, hit = np.minimum(output.t, horizon), output.status == 1
                        row = dict(id=identifier, phase="slowing", model=model, method=method, setting=setting,
                            boundary_shift=shifted, tolerance=tol if method != "fixed" else None, cap=cap,
                            elapsed_seconds=elapsed, **work(output), times=t.tolist(), hit=hit.tolist(),
                            statistics=passage_statistics(t, hit, reference, horizon))
                        if model == "gc":
                            row["spatial"] = spatial_statistics(output.y[:, 5]/rho**2, np.sum(output.y[:, :2]**2, axis=1)/rho**2, spectral, float(spatial["second_moment"][0]))
                        if shifted:
                            original = next(r for r in report["rows"] if r["id"] == identifier.replace("shifted", "endpoint"))
                            if np.any(t > np.asarray(original["times"])+1e-12):
                                raise RuntimeError("shifted stopping is not a prefix of the same path")
                            row["paired_time_change"] = summary(t-np.asarray(original["times"]), 0.)
                        if method != "fixed" and row["max_accepted_error"] > 1:
                            raise RuntimeError("accepted beam estimate exceeds one")
                        save(row)
    if "spatial" in args.phases:
        from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
        rho = physical.sigma*PROTON_MASS/ELEMENTARY_CHARGE
        horizon = .1
        expected = float(fixed_time_spatial_msd(physical, [horizon])[0])
        report["short_spatial_reference"] = expected
        for method in METHODS:
            identifier = f"spatial/gc/{method}/fine"
            if identifier in completed:
                continue
            adaptive = method != "fixed"
            cap = .003125 if adaptive else .000625
            function = native_solver("gc", method if adaptive else "conservative", adaptive, physical, horizon, 0., spatial=True, times=[0., horizon])
            params = tuple(map(jnp.asarray, (.001, cap, cap)))
            executable = function.lower(keys[0], *params).compile()
            print(f"Running {identifier}: N={args.particles}", flush=True)
            output, elapsed = run_batches(executable, [(k,) for k in keys], params)
            if np.any(output.status != 0):
                raise RuntimeError("unabsorbed short test did not reach its horizon")
            row = dict(id=identifier, phase="spatial", model="gc", method=method, setting="fine",
                elapsed_seconds=elapsed, **work(output), spatial=spatial_statistics(output.y[:, 5]/rho**2,
                    np.sum(output.y[:, :2]**2, axis=1)/rho**2, expected))
            save(row)
    report["complete"] = True
    save()
    print(f"Completed {len(report['rows'])} rows: {args.output}", flush=True)


if __name__ == "__main__":
    main()
