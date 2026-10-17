"""Plot the saved matched impact-model first-stopping ensembles; no tracing.

Pool only the declared independent seeds within each model/method. Histogram
densities divide by the full particle count, including the displayed tail.
The black density curve contains reference probabilities in those same bins.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from .nonrelativistic_collision_reference import ProtonBackground, proton_first_passage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output stem")
    args = parser.parse_args()
    if args.input.resolve() == args.output.with_suffix(".json").resolve():
        parser.error("plot metadata must not overwrite the input")
    report = json.loads(args.input.read_text())
    if not report.get("complete") or report["metadata"]["coulomb_log_model"] != "impact":
        raise ValueError("a complete matched impact report is required")
    reference = proton_first_passage(ProtonBackground(coulomb_log_model="impact"), nodes=1536)
    np.testing.assert_allclose(reference.mean, report["reference"]["mean"], rtol=2e-5)
    np.testing.assert_allclose(reference.variance, report["reference"]["variance"], rtol=1e-10)
    horizon = report["metadata"]["horizon"]
    times = np.linspace(0., horizon, 1601)
    cdf = reference.cdf(times)
    # Display the bulk and retain the full horizon in the exported data.
    right = float(np.ceil(np.interp(.995, cdf, times)))
    edges = np.linspace(0., right, 46)
    density = np.diff(reference.cdf(edges))/np.diff(edges)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    moments, moment_axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    summary = []
    counts = set()
    for i, model in enumerate(("gc", "lorentz")):
        for j, method in enumerate(("fixed", "conservative")):
            row = next(r for r in report["pooled"] if (r["model"], r["method"]) == (model, method))
            samples = np.asarray(row["times"])
            n = len(samples)
            counts.add(n)
            if n != row["particles"] or row["statistics"]["censored"]:
                raise ValueError("the matched plots require complete, uncensored ensembles")
            ordered = np.concatenate([next(r for r in report["rows"] if
                (r["model"], r["method"], r["seed"]) == (model, method, seed))["times"]
                for seed in report["metadata"]["seeds"]])
            np.testing.assert_array_equal(samples, ordered)
            if np.any(~np.isfinite(samples) | (samples < 0) | (samples > horizon)):
                raise ValueError("invalid saved stopping times")
            events = np.sort(samples)
            empirical = np.r_[0., np.arange(1, n+1)/n]
            axes[i, 0].step(np.r_[0., events], empirical, where="post", color=f"C{j}", label=method)
            hist = np.histogram(samples, bins=edges)[0]/(n*np.diff(edges))
            axes[i, 1].stairs(hist, edges, color=f"C{j}", label=method)
            for k, name in enumerate(("mean", "variance")):
                estimate = row[name]
                moment_axes[i, k].errorbar(j, estimate["value"], yerr=estimate["half95"],
                    fmt="o", color=f"C{j}", capsize=5)
            summary.append(dict(model=model, method=method, particles=n,
                histogram_density=hist.tolist(), fraction_beyond_plot=float(np.mean(samples > right)),
                cdf=row["statistics"]["cdf"], mean=row["mean"], variance=row["variance"]))
        axes[i, 0].plot(times, cdf, "k", lw=2, label="Independent first-passage theory")
        axes[i, 1].stairs(density, edges, color="k", lw=2, label="Theory, same bin averages")
        for k, title in enumerate(("stopping-time CDF", "stopping-time density")):
            axes[i, k].set(title=model.upper()+": "+title, xlabel=r"First-stopping time $\tau\nu_0$",
                ylabel="Cumulative probability" if k == 0 else "Probability density", xlim=(0., right))
            axes[i, k].grid(alpha=.2)
        for k, name in enumerate(("mean", "variance")):
            ax = moment_axes[i, k]
            ax.axhline(report["reference"]["restricted_"+name], color="k", lw=2,
                label="Independent first-passage theory")
            ax.set(title=model.upper()+": first-stopping "+name, xticks=(0, 1),
                xticklabels=("fixed", "conservative"), xlim=(-.4, 1.4),
                ylabel=r"$\tau\nu_0$" if k == 0 else r"$(\tau\nu_0)^2$")
            ax.grid(alpha=.2)
    if counts != {report["metadata"]["particles_per_seed"]*len(report["metadata"]["seeds"])}:
        raise ValueError("ensemble counts differ")
    count = counts.pop()
    subtitle = (f"Impact model; {count:,} paths per model/method "
        f"({len(report['metadata']['seeds'])} independent "
        f"{report['metadata']['particles_per_seed']:,}-particle seeds); no source")
    fig.suptitle("First slowing down from E=5T to E<=2T\n"+subtitle)
    moments.suptitle("First-stopping mean and variance; marginal 95% sampling intervals\n"+subtitle)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    moment_axes[0, 0].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, figure in (("distribution", fig), ("moments", moments)):
        for extension in ("png", "svg"):
            figure.savefig(args.output.with_name(args.output.stem+"_"+suffix).with_suffix("."+extension), dpi=170)
        plt.close(figure)
    root = Path(__file__).resolve().parents[2]
    source_names = ("examples/benchmarks/plot_matched_first_passage.py",
        "examples/benchmarks/nonrelativistic_collision_reference.py", "essos/constants.py")
    output = dict(complete=True, input=dict(path=str(args.input), sha256=hashlib.sha256(args.input.read_bytes()).hexdigest()),
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_names},
        description="Saved samples only; no particle tracing. Reference recomputed at 1536 nodes; histogram densities use all particles.",
        reference=dict(times=times.tolist(), cdf=cdf.tolist(), histogram_density=density.tolist(),
            histogram_edges=edges.tolist(), survival_beyond_plot=float(1-reference.cdf(right)),
            mean=report["reference"]["restricted_mean"], variance=report["reference"]["restricted_variance"]),
        configurations=summary)
    args.output.with_suffix(".json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(dict(complete=True, configurations=len(summary), particles_per_configuration=count,
        display_until=right, theoretical_fraction_beyond_plot=output["reference"]["survival_beyond_plot"],
        max_cdf_distance=max(r["cdf"]["distance"] for r in summary))))


if __name__ == "__main__":
    main()
