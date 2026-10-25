"""Source-free first stopping with production collision maps and Brownian memory.

Fresh fixed/base/tight ensembles use the same cap and event resolution. Keys
are repeatable; different adaptive meshes need not produce common paths.
Reports retain CDF/histogram and moment summaries, not raw particle histories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .nonrelativistic_collision_reference import ProtonBackground, proton_first_passage, passage_statistics
from .validate_collisional_slowing_down import native_solver
from .validate_collision_physics import run_batches, work
from .validate_beam_relaxation import estimate


def plot(report, reference, output):
    import matplotlib.pyplot as plt
    meta = report["metadata"]
    grid = np.asarray(report["plot_reference"]["times"])
    cdf = reference.cdf(grid)
    edges = np.asarray(report["plot_reference"]["histogram_edges"])
    right = edges[-1]
    density = np.diff(reference.cdf(edges))/np.diff(edges)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    moments, moment_axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    labels = ("Fixed", f"Adaptive\ntol {meta['base_tolerance']:g}",
        f"Adaptive\ntol {meta['tight_tolerance']:g}")
    for i, model in enumerate(("gc", "lorentz")):
        rows = [row for row in report["rows"] if row["model"] == model]
        for j, row in enumerate(rows):
            color, style = f"C{j}", ("-", "--", "-.")[j]
            axes[i, 0].step(grid, row["empirical_cdf"], where="post", color=color,
                ls=style, label=labels[j].replace("\n", ", "))
            axes[i, 1].stairs(row["plot"]["histogram_density"], edges, color=color,
                ls=style, label=labels[j].replace("\n", ", "))
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
    subtitle = (f"Production Brownian sampler; {meta.get('backend', 'unspecified')}; N={meta['particles']:,} per configuration; "
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3706)
    parser.add_argument("--cap", type=float, default=.00078125)
    parser.add_argument("--base-tolerance", type=float, default=.001)
    parser.add_argument("--tight-tolerance", type=float, default=.00025)
    parser.add_argument("--horizon", type=float, default=40.)
    parser.add_argument("--event-interval", type=float, default=1.220703125e-5)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    p = ProtonBackground(coulomb_log_model="impact")
    reference = proton_first_passage(p, nodes=1536)
    if args.plot_only:
        report = json.loads(args.output.read_text())
        if not report.get("complete"): parser.error("plot-only requires a complete report")
        plot(report, reference, args.output)
        return
    n, batch = args.particles, args.batch_size
    if not 2 <= n <= 2000 or not 2 <= batch <= n or n % batch:
        parser.error("use 2..2000 particles and a dividing batch size >=2")
    values = (args.cap, args.base_tolerance, args.tight_tolerance, args.horizon, args.event_interval)
    if not all(np.isfinite(x) and x > 0 for x in values) or args.tight_tolerance >= args.base_tolerance:
        parser.error("require positive finite parameters and tight tolerance < base tolerance")
    if args.output.exists(): parser.error("use a fresh output path")
    times = np.r_[np.linspace(0., min(10.370951771498717, args.horizon), 21), args.horizon]
    times = np.unique(times)
    grid = np.linspace(0., args.horizon, 201)
    right = float(np.ceil(np.interp(.995, reference.cdf(grid), grid)))
    edges = np.linspace(0., right, 46)
    raw = reference.restricted_moments(args.horizon)
    report = dict(complete=False, metadata=dict(particles=n, batch_size=batch, seed=args.seed,
        backend=jax.default_backend(), jax_version=jax.__version__, cap=args.cap, horizon=args.horizon,
        initial_step=args.cap, event_interval=args.event_interval, save_times=times.tolist(),
        base_tolerance=args.base_tolerance, tight_tolerance=args.tight_tolerance,
        coulomb_log_model="impact", controller="conservative", orbit=False,
        framework="native_solver -> solve_split -> production retained Brownian memory; SI coefficients",
        uncertainty="Independent particles; marginal 95% intervals; common seeds do not imply identical Wiener paths"),
        reference=dict(restricted_mean=float(raw[0]), restricted_variance=float(raw[1]-raw[0]**2)),
        plot_reference=dict(times=grid.tolist(), cdf=reference.cdf(grid).tolist(),
            histogram_edges=edges.tolist()), rows=[])
    root = Path(__file__).resolve().parents[2]
    names = ("essos/split.py", "essos/coulomb.py", "essos/background_species.py",
        "examples/benchmarks/validate_production_tolerance.py", "examples/benchmarks/validate_collisional_slowing_down.py",
        "examples/benchmarks/nonrelativistic_collision_reference.py")
    report["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: args.output.write_text(json.dumps(report, indent=2)+"\n")
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(args.seed), i))(jnp.arange(n)).reshape((-1, batch))
    for model in ("gc", "lorentz"):
        for adaptive in (False, True):
            function = native_solver(model, "conservative", adaptive, p, args.horizon, 2.,
                spatial=model=="gc", times=times, event_interval=args.event_interval)
            parameters = tuple(map(jnp.asarray, (args.base_tolerance, args.cap, args.cap)))
            executable = function.lower(keys[0], *parameters).compile()
            tolerances = (args.base_tolerance, args.tight_tolerance) if adaptive else (args.base_tolerance,)
            for tolerance in tolerances:
                print(f"Running {model}: adaptive={adaptive}, tolerance={tolerance:g}, N={n}", flush=True)
                parameters = tuple(map(jnp.asarray, (tolerance, args.cap, args.cap)))
                out, elapsed = run_batches(executable, [(key,) for key in keys], parameters)
                if np.any(out.status != 1):
                    raise RuntimeError("first-stopping plot requires complete uncensored trajectories; increase horizon")
                if adaptive and np.max(out.stats.max_accepted_error) > 1:
                    raise RuntimeError("accepted collision estimate exceeds one")
                samples = np.asarray(out.t)
                row = dict(model=model, method="conservative" if adaptive else "fixed",
                    tolerance=tolerance if adaptive else None, elapsed_seconds=elapsed, **work(out),
                    mean=estimate(samples,float(raw[0])), variance=estimate(samples,float(raw[1]-raw[0]**2),variance=True),
                    statistics=passage_statistics(samples, out.status==1, reference, args.horizon, alpha=.01/6),
                    empirical_cdf=(np.searchsorted(np.sort(samples),grid,side="right")/n).tolist(),
                    plot=dict(histogram_density=(np.histogram(samples,edges)[0]/(n*np.diff(edges))).tolist(),
                        fraction_beyond_display=float(np.mean(samples>right))))
                report["rows"].append(row); save()
    report["complete"] = True; save(); plot(report, reference, args.output)
    print("Saved "+str(args.output), flush=True)


if __name__ == "__main__": main()
