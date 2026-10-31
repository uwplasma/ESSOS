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
from .collision_plotting import presentation, histogram, save


@presentation
def plot(report, reference, output):
    import matplotlib.pyplot as plt
    meta=report["metadata"]
    edges=np.asarray(report["plot_reference"]["histogram_edges"])
    density=np.diff(reference.cdf(edges))/np.diff(edges)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout="constrained")
    for ax,model in zip(axes,("gc","lorentz")):
        row=min((r for r in report["rows"] if r["model"]==model),
                key=lambda r:(r["method"]=="fixed",r["tolerance"] if r["tolerance"] is not None else np.inf))
        counts=np.asarray(row["plot"]["histogram_density"])*meta["particles"]*np.diff(edges)
        grid=np.linspace(0.,edges[-1],1001)
        histogram(ax,edges,counts,density,particles=meta["particles"],curve=(grid,np.gradient(reference.cdf(grid),grid)))
        ax.set(title=model.upper()+": first-stopping distribution",xlabel=r"$\tau\nu_0$",xlim=(0.,edges[-1]))
        ax.legend()
    fig.suptitle(f"First crossing from 5T to 2T · N={meta['particles']:,}")
    save(fig,output.with_name(output.stem+"_distribution"))


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
    names = ("essos/split.py", "essos/background_species.py",
        "examples/benchmarks/validate_production_tolerance.py", "examples/benchmarks/validate_collisional_slowing_down.py",
        "examples/benchmarks/nonrelativistic_collision_reference.py")
    report["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: args.output.write_text(json.dumps(report, indent=2)+"\n")
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(args.seed), i))(jnp.arange(n)).reshape((-1, batch))
    for model in ("gc", "lorentz"):
        for adaptive in (False, True):
            function = native_solver(model, adaptive, p, args.horizon, 2.,
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
