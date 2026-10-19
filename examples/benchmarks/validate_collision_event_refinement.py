"""Check optional thermal-event resolution against saved endpoint baselines.

Reuses the ordered keys of the completed physical study, up to 2,000 particles.
No timing repetitions or Diffrax solves. Changed monitoring grids generally
change Brownian paths after refinement starts: these are not prefix replays.
The minimum policy interval and maximum detected-crossing interval are refined
independently of the collision tolerance and the ordinary maximum interval.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import t as student_t

from .nonrelativistic_collision_reference import ProtonBackground, proton_first_passage, passage_statistics
from .validate_collisional_slowing_down import native_solver
from .validate_collision_physics import run_batches, work, spatial_statistics

COLORS = dict(fixed="#0072B2", conservative="#D55E00", paper="#009E73", candidate="#CC79A7")


def plot(report, output):
    import matplotlib.pyplot as plt
    models = report["metadata"]["models"]
    methods = report["metadata"]["methods"]
    levels = [None, *report["metadata"]["event_intervals"]]
    fig, axes = plt.subplots(len(models), 3, figsize=(16, 5*len(models)), squeeze=False, constrained_layout=True)
    times = np.asarray(report["reference"]["cdf_times"])
    exact = np.asarray(report["reference"]["cdf_values"])
    n = report["metadata"]["particles"]
    band = np.sqrt(np.log(2/.01)/(2*n))
    for i, model in enumerate(models):
        axes[i, 0].fill_between(times, -band, band, color=".85", alpha=.5, label="Individual 99% sampling band")
        for row in report["rows"]:
            if row["model"] != model:
                continue
            level = levels.index(row["event_interval"])
            t = np.asarray(row["times"])
            hit = np.asarray(row["hit"])
            cdf = np.searchsorted(np.sort(t[hit]), times, side="right")/n
            axes[i, 0].plot(times, cdf-exact, color=COLORS[row["method"]],
                            ls=("--", "-", ":")[level % 3], alpha=.8)
            x = methods.index(row["method"])+(level-(len(levels)-1)/2)*.18
            for j, value in ((1, row["statistics"]["moments"]["restricted_mean"]),
                              (2, row["statistics"]["moments"]["restricted_variance"])):
                if j == 2 and model == "gc":
                    s = row["spatial"]["integral"]
                    difference, half = s["relative_difference"], s["half95"]/s["expected"]
                else:
                    difference = value["value"]/value["expected"]-1
                    half = student_t.ppf(.975, n-1)*value["mc_se"]/value["expected"]
                axes[i, j].errorbar(x, 100*difference, yerr=100*half,
                    color=COLORS[row["method"]], marker=("o", "s", "D")[level % 3], capsize=3, ls="none")
        axes[i, 0].set(title=f"{model.upper()}: stopping CDF minus continuous reference",
                       xlabel=r"$t\nu_0$", ylabel="CDF difference", xlim=(0., 6.))
        for j, name in ((1, "Stopping-time mean"),
                        (2, "Stopped spatial integral Q" if model == "gc" else "Stopping-time variance")):
            axes[i, j].set_xticks(range(len(methods)), methods, rotation=20)
            axes[i, j].set(title=f"{model.upper()}: {name}", ylabel="Difference from reference (%)")
        for ax in axes[i]:
            ax.axhline(0., color="k", ls="--")
            ax.grid(alpha=.2)
    for method in methods:
        axes[0, 0].plot([], [], color=COLORS[method], label=method)
    axes[0, 0].legend(fontsize=8)
    for k, interval in enumerate(levels):
        label = "Saved endpoint baseline" if interval is None else f"Event resolution {interval:g}"
        axes[0, 1].plot([], [], color=".3", marker=("o", "s", "D")[k % 3], ls="none", label=label)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"N={n}: refine the thermal-event diagnostic, not the collision tolerance\n"
                 "Same initial conditions and keys; retained Brownian retries; physical endpoint states; marginal 95% intervals")
    fig.savefig(output.with_suffix(".png"), dpi=170)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--methods", nargs="+", choices=tuple(COLORS), default=list(COLORS))
    parser.add_argument("--event-intervals", nargs="+", type=float, default=[.003125/64, .003125/256])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(json.loads(args.output.read_text()), args.output)
        return
    source = json.loads(args.source.read_text())
    if not source.get("complete"):
        raise ValueError("baseline study is incomplete")
    n = args.particles
    if not 2 <= n <= min(2000, source["metadata"]["particles"]) or not 2 <= args.batch_size <= n or n % args.batch_size:
        parser.error("use at most the saved/authorized 2000 particles and a dividing batch size")
    if not args.event_intervals or any(not np.isfinite(x) or x <= 0 for x in args.event_intervals):
        parser.error("event intervals must be finite and positive")
    if len(set(args.event_intervals)) != len(args.event_intervals):
        parser.error("event intervals must be distinct")
    physical = ProtonBackground()
    reference = proton_first_passage(physical)
    horizon = source["stopping_reference"]["horizon"]
    ts = np.linspace(0., horizon, 21)
    expected_q = source["stopping_reference"]["spatial_mean"]
    from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
    rho = physical.sigma*PROTON_MASS/ELEMENTARY_CHARGE
    metadata = dict(particles=n, batch_size=args.batch_size, seed=source["metadata"]["seed"],
        models=args.models, methods=args.methods, event_intervals=args.event_intervals,
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        scope="Optional boundary monitoring, no new thermal cutoff in ordinary tracing",
        interpretation="Endpoint monitoring with controlled resolution, not exact continuous first passage; changed grids need not share full Brownian paths",
        timing="One execution per configuration after compilation; baseline timings are historical, not matched speed benchmarks")
    raw = reference.restricted_moments(horizon)
    cdf_grid = np.linspace(0., horizon, 601)
    report = dict(complete=False, metadata=metadata,
        reference=dict(horizon=horizon, restricted_mean=float(raw[0]),
            restricted_variance=float(raw[1]-raw[0]**2), spatial_mean=expected_q,
            cdf_times=cdf_grid.tolist(), cdf_values=reference.cdf(cdf_grid).tolist()), rows=[])
    if args.output.exists():
        if not args.resume:
            raise ValueError("output exists; use --resume or a fresh output")
        report = json.loads(args.output.read_text())
        if report["metadata"] != metadata:
            raise ValueError("resume metadata differs")
        report["complete"] = False
    completed = {r["id"] for r in report["rows"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(row=None):
        if row is not None:
            report["rows"].append(row)
            completed.add(row["id"])
            brief = {k: row[k] for k in ("id", "trials_mean", "elapsed_seconds") if k in row}
            s = row["statistics"]
            brief.update(stopping_mean_difference=s["moments"]["restricted_mean"]["value"]/s["moments"]["restricted_mean"]["expected"]-1,
                         cdf_distance=s["cdf"]["distance"], cdf_limit=s["cdf"]["limit"])
            if "spatial" in row:
                brief["spatial_integral_difference"] = row["spatial"]["integral"]["relative_difference"]
            print(json.dumps(brief), flush=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    save()
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(metadata["seed"]), i))(jnp.arange(n)).reshape((-1, args.batch_size))
    for model in args.models:
        for method in args.methods:
            identifier = f"{model}/{method}/endpoint"
            if identifier not in completed:
                old = next(r for r in source["rows"] if r["id"] == f"slowing/{model}/{method}/fine/endpoint")
                t, hit = np.asarray(old["times"][:n]), np.asarray(old["hit"][:n])
                row = dict(id=identifier, model=model, method=method, event_interval=None,
                    times=t.tolist(), hit=hit.tolist(), statistics=passage_statistics(t, hit, reference, horizon))
                if model == "gc":
                    row["spatial"] = spatial_statistics(old["spatial"]["integral_samples"][:n],
                        old["spatial"]["displacement2_samples"][:n], expected_q)
                save(row)
            for interval in args.event_intervals:
                identifier = f"{model}/{method}/event_{interval:g}"
                if identifier in completed:
                    continue
                adaptive = method != "fixed"
                cap = .003125 if adaptive else .000625
                params = tuple(map(jnp.asarray, (.001, cap, cap)))
                function = native_solver(model, method if adaptive else "conservative", adaptive,
                    physical, horizon, 2., spatial=model == "gc", times=ts, event_interval=interval)
                print(f"Running {identifier}: N={n}", flush=True)
                executable = function.lower(keys[0], *params).compile()
                output, elapsed = run_batches(executable, [(k,) for k in keys], params)
                if adaptive and np.max(output.stats.max_accepted_error) > 1:
                    raise RuntimeError("accepted collision error exceeds one")
                hit = output.status == 1
                speed = output.y[:, 3] if model == "gc" else np.linalg.norm(output.y, axis=1)
                if np.any(speed[hit] > 2*physical.sigma*(1+1e-13)):
                    raise RuntimeError("event returned a state above the physical threshold")
                t = np.minimum(output.t, horizon)
                row = dict(id=identifier, model=model, method=method, event_interval=interval,
                    collision_tolerance=.001 if adaptive else None, ordinary_cap=cap,
                    elapsed_seconds=elapsed, **work(output), times=t.tolist(), hit=hit.tolist(),
                    terminal_speed_over_sigma=(speed/physical.sigma).tolist(),
                    statistics=passage_statistics(t, hit, reference, horizon))
                if model == "gc":
                    row["spatial"] = spatial_statistics(output.y[:, 5]/rho**2,
                        np.sum(output.y[:, :2]**2, axis=1)/rho**2, expected_q)
                save(row)
    report["complete"] = True
    save()
    flat = []
    for row in report["rows"]:
        s = row["statistics"]
        item = {k: row[k] for k in ("id", "model", "method", "event_interval")}
        item.update(stopping_mean_difference=s["moments"]["restricted_mean"]["value"]/s["moments"]["restricted_mean"]["expected"]-1,
                    cdf_distance=s["cdf"]["distance"], hits=s["hits"], censored=s["censored"],
                    spatial_difference=row.get("spatial", {}).get("integral", {}).get("relative_difference"))
        flat.append(item)
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot(report, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
