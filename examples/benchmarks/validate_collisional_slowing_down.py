"""Native-only nonrelativistic first-passage statistics, with a SciPy reference.

This is a source-free 5 keV proton beam in a 1 keV proton background. First
passage means first detected entry into E<=2T, NOT reaching zero speed or
equilibrium. Endpoint monitoring is refined separately from collision tolerance.
No Diffrax solves are performed. Default ensembles contain only 32 paths.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np

from essos.background_species import BackgroundSpecies
from essos.collisions import (collision_coefficients, guiding_center_collision_step, particle_collision_step,
    guiding_center_spatial_diffusion, guiding_center_spatial_increment)
from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q
from essos.split import solve_split, SplitStatus
from .nonrelativistic_collision_reference import ProtonBackground, proton_first_passage, quadrature_mean, passage_statistics
from .collision_event_diagnostic import boundary_step_limit


def native_solver(model, controller, adaptive, physical, horizon, threshold, *,
                  spatial=False, times=None, magnetic_field=1., boundary_shift=False,
                  event_interval=None, event_safety=.2, record_steps=False):
    """Native benchmark, optionally testing a discrete-exit boundary correction.

    ``boundary_shift`` tests the Gobet--Menozzi shift at each actual step h;
    it is not enabled in production tracing. The event flag is auxiliary and
    does not enter the velocity/spatial update or the local error estimates.
    ``event_interval`` enables boundary-aware stepping and refinement of
    detected crossing intervals. It is in collision-time units (h*nu0),
    independently of the collision tolerance. No boundary shift is needed.
    """
    if spatial and model != "gc":
        raise ValueError("explicit spatial kicks require the GC model")
    if event_interval is not None:
        if not np.isfinite(event_interval) or event_interval <= 0 or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("event_interval and the thermal threshold must be positive and finite")
        if not np.isfinite(event_safety) or not 0 < event_safety < 1:
            raise ValueError("event_safety must lie strictly between zero and one")
        if boundary_shift:
            raise ValueError("compare event refinement and the experimental shift separately")
    species = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([physical.density]), jnp.array([physical.temperature_ev]),
        coulomb_log_model=physical.coulomb_log_model)
    gc = model == "gc"
    v0 = physical.sigma*np.sqrt(10.)
    y0 = jnp.array([v0, .6]) if gc else jnp.array([.8*v0, 0., .6*v0])
    if spatial:
        # Position, speed, pitch, accumulated 4*D_X*dt. All units are SI.
        y0 = jnp.array([0., 0., 0., v0, .6, 0.])
    physical_size = y0.size
    if boundary_shift:
        y0 = jnp.append(y0, v0/physical.sigma-threshold)
        boundary_diffusion = float(physical.coefficients(threshold*physical.sigma)["parallel"])
    nu = physical.nu
    save_times = jnp.empty((0,)) if times is None else jnp.asarray(times)/nu
    if times is not None:
        # JAX division may use a reciprocal multiplication, differing from
        # Python's horizon/nu by one ulp. A save at the requested horizon must
        # use the exact same SI endpoint as t1, rather than be rejected as >t1.
        save_times = jnp.where(jnp.asarray(times) == horizon, jnp.asarray(horizon/nu), save_times)

    def batch(keys, tolerance, cap, initial):
        def one(key):
            def event_limit(t, y, args):
                speed = y[3] if spatial else (y[0] if gc else jnp.linalg.norm(y))
                c = collision_coefficients(species, M, Q, speed, jnp.zeros(3))
                drift = c.friction+c.parallel_derivative+2*c.parallel/speed
                return boundary_step_limit(speed, threshold*physical.sigma, c.parallel,
                    drift, event_interval/nu, safety=event_safety)

            def kick(t, y, h, dw, args):
                y = y[:physical_size]
                speed = y[3] if spatial else (y[0] if gc else jnp.linalg.norm(y))
                pitch = y[4] if spatial else y[1]
                c = collision_coefficients(species, M, Q, speed, jnp.zeros(3), full_orbit=not gc)
                kwargs = dict(tolerance=tolerance, return_error_model=adaptive and controller == "candidate")
                # Keep driver, Wiener increments and candidate-error metadata
                # in SI units. Only reporting converts to collision time.
                result = (guiding_center_collision_step(speed, pitch, h, dw[-2:], c,
                            speed_floor=.05*np.sqrt(2)*physical.sigma, **kwargs) if gc else
                        particle_collision_step(y, h, dw, c, **kwargs))
                if spatial:
                    omega = Q*magnetic_field/M
                    displacement = guiding_center_spatial_increment(pitch, dw[:3], c, jnp.array([0., 0., 1.]), omega)
                    value = y.at[:3].add(displacement).at[3:5].set(result.y)
                    value = value.at[5].add(4*guiding_center_spatial_diffusion(pitch, c, omega)*h)
                    result = result._replace(y=value)
                if boundary_shift:
                    value = result.y
                    next_speed = value[3] if spatial else (value[0] if gc else jnp.linalg.norm(value))
                    level = (next_speed-threshold*physical.sigma
                        -.5825971579390108*jnp.sqrt(2*boundary_diffusion*h))/physical.sigma
                    result = result._replace(y=jnp.append(value, level))
                return result
            return solve_split(lambda t, y, args: jnp.zeros_like(y), kick, y0,
                t1=horizon/nu, dt0=initial/nu, dtmax=cap/nu, ts=save_times,
                key=key, noise_dim=(5 if spatial else 2) if gc else 3, method="dopri5", adaptive=adaptive,
                collision_controller=controller,
                collision_wiener_norm=(lambda w: jnp.linalg.norm(w[-2:])) if gc else (lambda w: jnp.abs(w[2])),
                event=lambda t, y, args: y[-1] if boundary_shift else
                    (y[3] if spatial else (y[0] if gc else jnp.linalg.norm(y)))/physical.sigma-threshold,
                max_steps=100_000, record_steps=record_steps,
                step_limit=event_limit if event_interval is not None else None,
                event_dtmax=None if event_interval is None else event_interval/nu)
        out = jax.vmap(one)(keys)
        return out._replace(t=out.t*nu)
    return jax.jit(batch)


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(fixed="#0072B2", conservative="#D55E00", paper="#009E73", candidate="#CC79A7")
    labels = dict(fixed="Fixed", conservative="Conservative", paper="Paper", candidate="Candidate")
    markers = dict(fixed_coarse="o", fixed_fine="s", base="o", tolerance="^", cap="s")
    offsets = dict(fixed_coarse=-.12, fixed_fine=.12, base=-.18, tolerance=0., cap=.18)
    methods = ["fixed"]+report["metadata"]["controllers"]
    reference = report["reference"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    t = np.asarray(reference["times"])
    f = np.asarray(reference["cdf"])
    for i, model in enumerate(("gc", "lorentz")):
        title = "Guiding centre" if model == "gc" else "Lorentz"
        axes[i, 0].plot(t, f, "k--", label="Backward FP reference", lw=2)
        width = np.sqrt(np.log(2/.01)/(2*report["metadata"]["paths"]))
        axes[i, 0].fill_between(t, np.maximum(f-width, 0), np.minimum(f+width, 1),
            color=".8", alpha=.4, label="99% DKW sampling band")
        rows = [r for r in report["rows"] if r["model"] == model]
        for row in rows:
            method = row["controller"] if row["adaptive"] else "fixed"
            x = methods.index(method)+offsets[row["setting"]]
            stats = row["statistics"]
            for j, quantity in enumerate(("restricted_mean", "restricted_variance"), 1):
                m = stats["moments"][quantity]
                axes[i, j].errorbar(x, m["value"], yerr=m["mc_se"], marker=markers[row["setting"]],
                    color=colors[method], capsize=3, ms=5, linestyle="none")
            if row["setting"] in ("cap", "fixed_fine"):
                values = np.sort(np.asarray(row["times"])[np.asarray(row["hit"], dtype=bool)])
                axes[i, 0].step(np.r_[0., values, report["metadata"]["horizon"]],
                    np.r_[0., np.arange(1, len(values)+1)/len(row["times"]), len(values)/len(row["times"])],
                    where="post", color=colors[method], label=labels[method], lw=1.3)
        axes[i, 0].set(title=f"{title}: thermal-band entry CDF", xlabel=r"Time $t\nu_0$",
                       ylabel="Cumulative probability", ylim=(-.02, 1.02), xlim=(0, 4*reference["mean"]))
        for j, quantity in enumerate(("restricted_mean", "restricted_variance"), 1):
            axes[i, j].axhline(reference[quantity], color="k", ls="--", lw=1.5)
            axes[i, j].set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=15)
            axes[i, j].set(title=f"{title}: restricted {'mean' if j == 1 else 'variance'}", ylabel=r"$t\nu_0$" if j == 1 else r"$(t\nu_0)^2$")
        for ax in axes[i]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    for marker, label in (("o", "Base / coarse fixed"), ("^", "Tighter tolerance"), ("s", "Smaller cap / fine fixed")):
        axes[0, 1].plot([], [], marker=marker, color=".3", ls="none", label=label)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Source-free nonrelativistic slowing down: 5 keV protons, 1 keV background\n"
        f"First detected entry into E <= 2T; N={report['metadata']['paths']} per configuration; bars = 1 MC SE", fontsize=13)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--controllers", nargs="+", choices=("conservative", "paper", "candidate"), default=["conservative", "paper", "candidate"])
    parser.add_argument("--repeats", type=int, default=3)
    event_options = parser.add_mutually_exclusive_group()
    event_options.add_argument("--event-interval", type=float,
                               help="boundary monitoring resolution in collision-time units (default: 0.003125/256)")
    event_options.add_argument("--endpoint-only", dest="event_interval", action="store_const", const=None,
                               help="disable boundary refinement to reproduce the original diagnostic")
    parser.set_defaults(event_interval=.003125/256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true", help="render the existing output JSON without particle integrations")
    args = parser.parse_args()
    if args.plot_only:
        plot_report(json.loads(args.output.read_text()), args.output.with_suffix(".png"))
        return
    if not 2 <= args.paths <= 128 or args.repeats < 1:
        parser.error("use 2..128 paths and positive repeats")
    physical = ProtonBackground()
    coarse = proton_first_passage(physical, nodes=384)
    reference = proton_first_passage(physical, nodes=768)
    extended = proton_first_passage(physical, nodes=1024, upper=10.)
    exact_mean = quadrature_mean(physical, reference.boundary, reference.start)
    horizon = 6*reference.mean
    times = np.linspace(0, horizon, 401)
    refinement = dict(mean_relative=abs(coarse.mean/reference.mean-1),
        variance_relative=abs(coarse.variance/reference.variance-1),
        cdf_max=float(np.max(np.abs(coarse.cdf(times)-reference.cdf(times)))),
        extended_cdf_max=float(np.max(np.abs(extended.cdf(times)-reference.cdf(times)))),
        quadrature_mean_relative=abs(reference.mean/exact_mean-1))
    if max(refinement.values()) > 5e-4:
        raise RuntimeError(f"reference not sufficiently converged: {refinement}")
    raw = reference.restricted_moments(horizon)
    root = Path(__file__).resolve().parents[2]
    sources = ("essos/collisions.py", "essos/split.py", "essos/collision_controller.py", "essos/collision_error.py",
               "examples/benchmarks/nonrelativistic_collision_reference.py", "examples/benchmarks/validate_collisional_slowing_down.py")
    report = dict(complete=False, metadata=dict(paths=args.paths, controllers=args.controllers,
        horizon=horizon, nu0_per_second=physical.nu, initial_energy_over_T=5., threshold_energy_over_T=2.,
        backend=jax.default_backend(), jax=jax.__version__,
        reference="Independent NumPy coefficients and backward FP; mesh/upper-boundary/quadrature cross-checks",
        interpretation="First detected E<=2T, not exact equilibrium or zero-speed arrival; no source",
        uncertainty="Independent paths within each row; configurations share seeds, not exact Brownian paths; 1-SE bars and 99% DKW marginal CDF bands",
        event_interval=args.event_interval,
        event_bias="Boundary-aware refinement is optional; endpoint detection is approximate and requires an event-resolution convergence study",
        precision="32 paths are a pilot for gross statistical errors, not evidence of 0.1% weak accuracy",
        source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}),
        reference=dict(mean=reference.mean, variance=reference.variance, quadrature_mean=exact_mean,
            restricted_mean=float(raw[0]), restricted_variance=float(raw[1]-raw[0]**2),
            cdf_at_horizon=float(reference.cdf(horizon)), refinement=refinement,
            times=times.tolist(), cdf=reference.cdf(times).tolist()), rows=[])

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        flat = []
        for row in report["rows"]:
            entry = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
            for quantity, m in row["statistics"]["moments"].items():
                entry.update({quantity+"_"+k: v for k, v in m.items()})
            entry.update(hits=row["statistics"]["hits"], censored=row["statistics"]["censored"],
                cdf_distance=row["statistics"]["cdf"]["distance"], cdf_limit=row["statistics"]["cdf"]["limit"],
                diagnostics_passed=row["statistics"]["passed"])
            flat.append(entry)
        if flat:
            with args.output.with_suffix(".csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
    save()
    keys = jax.random.split(jax.random.key(706), args.paths)
    print(json.dumps(dict(reference=report["reference"] | {"times": None, "cdf": None})), flush=True)
    for model in ("gc", "lorentz"):
        for method in ["fixed"]+args.controllers:
            adaptive = method != "fixed"
            controller = method if adaptive else "conservative"
            compiled = native_solver(model, controller, adaptive, physical, horizon, reference.boundary,
                event_interval=args.event_interval).lower(
                keys, jnp.array(.003), jnp.array(.05), jnp.array(.005)).compile()
            settings = (("base", .003, .05, .005), ("tolerance", .001, .05, .005), ("cap", .003, .0125, .005)) if adaptive else (
                ("fixed_coarse", .003, .01, .01), ("fixed_fine", .003, .0025, .0025))
            for name, tolerance, cap, initial in settings:
                inputs = (keys, jnp.array(tolerance), jnp.array(cap), jnp.array(initial))
                out = jax.block_until_ready(compiled(*inputs))
                warm = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    jax.block_until_ready(compiled(*inputs))
                    warm.append(1000*(perf_counter()-start))
                out = jax.device_get(out)
                if np.any((out.status != SplitStatus.SUCCESS) & (out.status != SplitStatus.EVENT)):
                    raise RuntimeError(f"invalid native solve: {model}/{method}/{name}: {np.unique(out.status)}")
                if adaptive and np.max(out.stats.max_accepted_error) > 1:
                    raise RuntimeError("accepted collision estimate exceeds one")
                hit = out.status == SplitStatus.EVENT
                statistics = passage_statistics(out.t, hit, reference, horizon)
                row = dict(model=model, controller=controller, adaptive=adaptive, setting=name,
                    tolerance=tolerance if adaptive else None, dtmax=cap, initial_step=initial,
                    warm_median_ms=float(np.median(warm)), accepted_mean=float(out.stats.accepted.mean()),
                    rejected_mean=float(out.stats.rejected.mean()), trials_mean=float(out.stats.steps.mean()),
                    max_accepted_error=float(out.stats.max_accepted_error.max()),
                    times=out.t.tolist(), hit=hit.tolist(), statistics=statistics)
                report["rows"].append(row)
                save()
                print(json.dumps({k: v for k, v in row.items() if k not in ("times", "hit")}), flush=True)
    report["complete"] = True
    save()
    plot_report(report, args.output.with_suffix(".png"))
    print(f"Saved {args.output} and plots", flush=True)


if __name__ == "__main__":
    main()
