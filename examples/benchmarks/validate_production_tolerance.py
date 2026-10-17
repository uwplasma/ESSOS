"""First-stopping tolerance comparison using the production Brownian memory.

Reuse the original native benchmark wrapper unchanged: solve_split generates
and retains every Wiener increment and passes it directly to the collision
maps. Replay saved fixed/base-tolerance ensembles before reuse, then run only
the tighter tolerance. No common-path sampler or paired-error claim is used.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for name, value in (("JAX_PLATFORMS", "cpu"), ("ESSOS_CPU_DEVICES", "1"),
        ("ESSOS_PROGRESS", "0"), ("ESSOS_XLA_CACHE", ""), ("MPLBACKEND", "Agg")):
    os.environ.setdefault(name, value)

import jax
import jax.numpy as jnp
import numpy as np

from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
from .nonrelativistic_collision_reference import ProtonBackground, proton_first_passage, passage_statistics
from .validate_collisional_slowing_down import native_solver
from .validate_collision_physics import run_batches, work, spatial_statistics
from .validate_beam_relaxation import estimate


def plot(report, reference, output):
    import matplotlib.pyplot as plt
    meta = report["metadata"]
    grid = np.linspace(0., meta["horizon"], 1601)
    cdf = reference.cdf(grid)
    right = float(np.ceil(np.interp(.995, cdf, grid)))
    edges = np.linspace(0., right, 46)
    density = np.diff(reference.cdf(edges))/np.diff(edges)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    moments, moment_axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    labels = ("Fixed", f"Adaptive\ntol {meta['base_tolerance']:g}",
        f"Adaptive\ntol {meta['tight_tolerance']:g}")
    for i, model in enumerate(("gc", "lorentz")):
        rows = [row for row in report["rows"] if row["model"] == model]
        for j, row in enumerate(rows):
            samples = np.asarray(row["times"])
            if len(samples) != meta["particles"] or row["statistics"]["censored"]:
                raise ValueError("plots require complete uncensored ensembles")
            np.testing.assert_allclose([samples.mean(), samples.var(ddof=1)],
                [row["mean"]["value"], row["variance"]["value"]], rtol=1e-12)
            color, style = f"C{j}", ("-", "--", "-.")[j]
            axes[i, 0].step(np.r_[0., np.sort(samples)],
                np.r_[0., np.arange(1, len(samples)+1)/len(samples)], where="post",
                color=color, ls=style, label=labels[j].replace("\n", ", "))
            hist = np.histogram(samples, bins=edges)[0]/(len(samples)*np.diff(edges))
            axes[i, 1].stairs(hist, edges, color=color, ls=style,
                label=labels[j].replace("\n", ", "))
            row["plot"] = dict(histogram_density=hist.tolist(),
                fraction_beyond_display=float(np.mean(samples > right)))
            for k, name in enumerate(("mean", "variance")):
                moment_axes[i, k].errorbar(j, row[name]["value"],
                    yerr=row[name]["half95"], fmt="o", color=color, capsize=5)
        if len(rows) != len(labels):
            raise ValueError("fixed, base-adaptive and tight-adaptive rows required")
        axes[i, 0].plot(grid, cdf, "k", lw=2, label="Independent first-passage theory")
        axes[i, 1].stairs(density, edges, color="k", lw=2, label="Theory, same bin averages")
        for k, title in enumerate(("stopping-time CDF", "stopping-time density")):
            axes[i, k].set(title=model.upper()+": "+title, xlabel=r"First-stopping time $\tau\nu_0$",
                ylabel="Cumulative probability" if k == 0 else "Probability density", xlim=(0., right))
            axes[i, k].grid(alpha=.2)
        for k, name in enumerate(("mean", "variance")):
            ax = moment_axes[i, k]
            ax.axhline(report["reference"]["restricted_"+name], color="k", lw=2,
                label="Independent first-passage theory")
            ax.set(title=model.upper()+": first-stopping "+name,
                xticks=range(3), xticklabels=labels, xlim=(-.4, 2.4),
                ylabel=r"$\tau\nu_0$" if k == 0 else r"$(\tau\nu_0)^2$")
            ax.grid(alpha=.2)
    subtitle = (f"Production Brownian sampler; CPU; N={meta['particles']:,} per configuration; "
        f"ceiling {meta['cap']:g} unchanged")
    fig.suptitle("First slowing down: E=5T to first E<=2T; no source\n"+subtitle)
    moments.suptitle("First-stopping mean and variance; empirical 95% sampling intervals\n"+subtitle)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    moment_axes[0, 0].legend(fontsize=8)
    for suffix, figure in (("distribution", fig), ("moments", moments)):
        for extension in ("png", "svg"):
            figure.savefig(output.with_name(output.stem+"_"+suffix).with_suffix("."+extension), dpi=170)
        plt.close(figure)
    report["plot_reference"] = dict(times=grid.tolist(), cdf=cdf.tolist(),
        histogram_edges=edges.tolist(), histogram_density=density.tolist(),
        survival_beyond_display=float(1-reference.cdf(right)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3706)
    parser.add_argument("--tight-tolerance", type=float, default=.00025)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        parser.error("CPU only")
    p = ProtonBackground(coulomb_log_model="impact")
    reference = proton_first_passage(p, nodes=1536)
    if args.plot_only:
        report = json.loads(args.output.read_text())
        if not report.get("complete"):
            parser.error("plot-only requires a complete report")
        plot(report, reference, args.output)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        return
    if args.output.exists():
        parser.error("use a fresh output path")
    baseline = json.loads(args.baseline.read_text())
    old = baseline["metadata"]
    if not baseline.get("complete") or old["coulomb_log_model"] != "impact":
        parser.error("a complete production impact-model baseline is required")
    if args.seed not in old["seeds"] or not 0 < args.tight_tolerance < old["tolerance"]:
        parser.error("reuse an existing seed and choose a tighter positive tolerance")
    n, batch = old["particles_per_seed"], old["batch_size"]
    if not 2 <= n <= 2000 or n % batch:
        parser.error("at most 2000 particles per configuration, preserving the original batch size")
    root = Path(__file__).resolve().parents[2]
    hashes = {}
    for name, expected in baseline["source_sha256"].items():
        hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if hashes[name] != expected:
            raise ValueError(f"production baseline source changed: {name}")
    own = "examples/benchmarks/validate_production_tolerance.py"
    hashes[own] = hashlib.sha256((root/own).read_bytes()).hexdigest()
    raw = reference.restricted_moments(old["horizon"])
    np.testing.assert_allclose([raw[0], raw[1]-raw[0]**2],
        [baseline["reference"]["restricted_mean"], baseline["reference"]["restricted_variance"]], rtol=1e-10)
    metadata = dict(particles=n, batch_size=batch, seed=args.seed, backend=jax.default_backend(),
        jax_version=jax.__version__, cap=old["cap"], initial_step=old["cap"],
        horizon=old["horizon"], event_interval=old["event_interval"], save_times=old["save_times"],
        base_tolerance=old["tolerance"], tight_tolerance=args.tight_tolerance,
        coulomb_log_model="impact", controller="conservative", orbit=False,
        framework="Unchanged native_solver -> solve_split -> production brownian_propose/brownian_accept; direct driver dW, SI units",
        noise_dimensions=dict(gc=5, lorentz=3),
        scope="Collision-only first-passage diagnostic; original save anchors and event hooks retained; no custom Wiener process",
        uncertainty="Empirical marginal 95% sampling intervals; same ordered keys across tolerances do not guarantee the same Wiener paths; no paired-error inference")
    report = dict(complete=False, metadata=metadata, source_sha256=hashes,
        baseline=dict(path=str(args.baseline), sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest()),
        reference=dict(baseline["reference"]), replay_controls=[], rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n")

    save()
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(args.seed), i))(
        jnp.arange(n)).reshape((-1, batch))
    base_params = tuple(map(jnp.asarray, (old["tolerance"], old["cap"], old["cap"])))
    for model in ("gc", "lorentz"):
        for method in ("fixed", "conservative"):
            previous = next(row for row in baseline["rows"]
                if (row["model"], row["method"], row["seed"]) == (model, method, args.seed))
            print(f"Compiling production {model}/{method}; replaying {batch} baseline paths", flush=True)
            fn = native_solver(model, "conservative", method != "fixed", p, old["horizon"], 2.,
                spatial=model == "gc", times=old["save_times"], event_interval=old["event_interval"])
            executable = fn.lower(keys[0], *base_params).compile()
            checks = jax.device_get(jax.block_until_ready(executable(keys[0], *base_params)))
            np.testing.assert_array_equal(checks.status, np.ones(batch, int))
            np.testing.assert_allclose(checks.t, previous["times"][:batch], rtol=0., atol=2e-10)
            if model == "gc":
                rho = p.sigma*PROTON_MASS/ELEMENTARY_CHARGE
                np.testing.assert_allclose(checks.y[:, 5]/rho**2,
                    previous["spatial"]["integral_samples"][:batch], rtol=2e-9, atol=2e-9)
                np.testing.assert_allclose(np.sum(checks.y[:, :2]**2, axis=1)/rho**2,
                    previous["spatial"]["displacement2_samples"][:batch], rtol=2e-9, atol=2e-9)
            report["replay_controls"].append(dict(model=model, method=method, paths=batch,
                max_time_difference=float(np.max(abs(checks.t-np.asarray(previous["times"][:batch]))))))
            row = dict(previous, tolerance=None if method == "fixed" else old["tolerance"], reused=True)
            # Recompute the same CDF screen level for all six displayed rows.
            row["statistics"] = passage_statistics(row["times"], row["hit"], reference, old["horizon"], alpha=.01/6)
            report["rows"].append(row)
            save()
            print(f"Production baseline replay passed: {model}/{method}", flush=True)
            if method == "fixed":
                continue
            params = tuple(map(jnp.asarray, (args.tight_tolerance, old["cap"], old["cap"])))
            print(f"Running production {model}: N={n}, tolerance={args.tight_tolerance:g}, unchanged cap={old['cap']:g}", flush=True)
            out, elapsed = run_batches(executable, [(key,) for key in keys], params)
            if np.any(out.status != 1) or np.max(out.stats.max_accepted_error) > 1:
                raise RuntimeError("incomplete first passage or invalid collision acceptance")
            row = dict(model=model, method="conservative", seed=args.seed,
                tolerance=args.tight_tolerance, reused=False, elapsed_seconds=elapsed,
                **work(out), times=out.t.tolist(), hit=(out.status == 1).tolist(),
                mean=estimate(out.t, float(raw[0])), variance=estimate(out.t, float(raw[1]-raw[0]**2), variance=True),
                statistics=passage_statistics(out.t, out.status == 1, reference, old["horizon"], alpha=.01/6))
            if model == "gc":
                rho = p.sigma*PROTON_MASS/ELEMENTARY_CHARGE
                row["spatial"] = spatial_statistics(out.y[:, 5]/rho**2,
                    np.sum(out.y[:, :2]**2, axis=1)/rho**2, report["reference"]["spatial_mean"])
            report["rows"].append(row)
            save()
            print(json.dumps({name: row[name] for name in ("model", "tolerance", "elapsed_seconds", "mean", "variance", "statistics")}), flush=True)
    plot(report, reference, args.output)
    table = [dict(model=r["model"], method=r["method"], tolerance=r["tolerance"],
        particles=n, cap=old["cap"], mean=r["mean"]["value"], mean_half95=r["mean"]["half95"],
        variance=r["variance"]["value"], variance_half95=r["variance"]["half95"],
        accepted_mean=r["accepted_mean"], rejected_mean=r["rejected_mean"], reused=r["reused"]) for r in report["rows"]]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    report["complete"] = True
    save()
    print(json.dumps(dict(complete=True, production_sampler=True, configurations=len(table),
        replayed_paths=sum(r["paths"] for r in report["replay_controls"]), new_ensembles=2)), flush=True)


if __name__ == "__main__":
    main()
