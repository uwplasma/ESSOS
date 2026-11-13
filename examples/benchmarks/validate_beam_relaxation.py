"""Validate hot-beam relaxation and unabsorbed GC spatial diffusion.

Run up to 2,000 independent particles per configuration, one execution only,
using native fixed/conservative stepping at two caps. This declared fixed-
Coulomb-log control has exact Maxwell equilibrium; ``--coefficient-model
physical`` uses the production speed-dependent coefficients and an independent
explicit-drift reference. Both predict the transient, not merely equilibrium.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import kstest, norm, t as student_t

from .thermal_relaxation_reference import RelaxationReference, PhysicalRelaxationReference, impact_coefficients
from .validate_collision_physics import thermal_solver, run_batches, work
from .analyze_collisional_convergence import conditional_spatial_statistics
from .collision_plotting import presentation, series, histogram, save


def estimate(values, expected, *, variance=False):
    x = np.asarray(values, float)
    n = len(x)
    if variance:
        influence = n/(n-1)*(x-x.mean())**2
        value = x.var(ddof=1)
    else:
        influence, value = x, x.mean()
    se = influence.std(ddof=1)/np.sqrt(n)
    half = student_t.ppf(.975, n-1)*se
    return dict(value=float(value), reference=float(expected), difference=float(value-expected),
        se=float(se), half95=float(half), z=float((value-expected)/se) if se > 1e-14 else None)


def statistics(r, xi, reference):
    return dict(speed_mean=estimate(r, reference["speed_mean"]),
        speed_variance=estimate(r, reference["speed_variance"], variance=True),
        speed2=estimate(r*r, reference["speed2"]),
        pitch_mean=estimate(xi, reference["pitch_mean"]),
        pitch_variance=estimate(xi, reference["pitch_variance"], variance=True),
        parallel_mean=estimate(r*xi, reference["parallel_mean"]),
        parallel_variance=estimate(r*xi, reference["parallel_variance"], variance=True))


def physical_native_coefficients(coulomb_log_model="nrl"):
    from essos.background_species import BackgroundSpecies
    from essos.split import collision_coefficients, CollisionCoefficients
    from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
    from .nonrelativistic_collision_reference import ProtonBackground
    p = ProtonBackground(coulomb_log_model=coulomb_log_model)
    species = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([p.density]), jnp.array([p.temperature_ev]),
        coulomb_log_model=coulomb_log_model)
    def law(r, *, full_orbit=False):
        c = collision_coefficients(species, PROTON_MASS, ELEMENTARY_CHARGE, p.sigma*r, jnp.zeros(3), full_orbit=full_orbit)
        return CollisionCoefficients(c.parallel/(p.sigma**2*p.nu), c.perpendicular/(p.sigma**2*p.nu),
            c.parallel_derivative/(p.sigma*p.nu), c.friction/(p.sigma*p.nu),
            c.friction_derivative/p.nu, c.particle_drift_derivative/p.nu)
    return law


def prepare_references(times, coefficient_model="constant_log"):
    references, checks = {}, {}
    for model in ("gc", "lorentz"):
        lower = .05*np.sqrt(2.) if model == "gc" else 0.
        cls = PhysicalRelaxationReference if coefficient_model == "physical" else RelaxationReference
        kwargs = dict(law=impact_coefficients) if coefficient_model == "impact" else {}
        coarse = cls(lower=lower, nodes=512, **kwargs)
        fine = cls(lower=lower, nodes=1024, **kwargs)
        upper = cls(lower=lower, upper=10., nodes=1280, **kwargs)
        a, b, c = (ref.moments(times) for ref in (coarse, fine, upper))
        changes = {}
        for name in b:
            scale = max(1., np.max(np.abs(b[name])))
            changes[name] = dict(mesh=float(np.max(abs(a[name]-b[name]))/scale),
                upper=float(np.max(abs(c[name]-b[name]))/scale))
        grid = np.linspace(-1., 1., 201)
        pitch8 = fine.pitch_cdf(times[-1], harmonics=8)
        pitch12 = fine.pitch_cdf(times[-1], harmonics=12)
        angular_error = float(np.max(abs(pitch8(grid)-pitch12(grid))))
        maximum = max(v[k] for v in changes.values() for k in ("mesh", "upper"))
        if maximum > 3e-4 or angular_error > 1e-6:
            raise RuntimeError(f"reference refinement failed: {model}, {changes}, angular={angular_error}")
        references[model] = (fine, b, pitch12)
        checks[model] = dict(changes=changes, max_scaled_change=maximum, pitch_harmonic_change=angular_error,
            equilibrium=fine.equilibrium(), maxwell_equilibrium=RelaxationReference(lower=lower, nodes=32).equilibrium(),
            transient={name: value.tolist() for name, value in b.items()})
        speed_edges, pitch_edges = np.linspace(0., 6., 37), np.linspace(-1., 1., 25)
        checks[model]["distribution"] = dict(speed_edges=speed_edges.tolist(), pitch_edges=pitch_edges.tolist(),
            speed_density=(np.diff(fine.speed_cdf(times[-1])(speed_edges))/np.diff(speed_edges)).tolist(),
            pitch_density=(np.diff(pitch12(pitch_edges))/np.diff(pitch_edges)).tolist())
        print(f"{model} reference: maximum scaled refinement change {maximum:.3g}", flush=True)
    return references, checks


@presentation
def plot(report, output):
    import matplotlib.pyplot as plt
    times=np.asarray(report["metadata"]["times"]);meta=report["metadata"]
    panels=(("speed",r"v/\sigma_T"),("pitch",r"\xi"),("parallel",r"v_\parallel/\sigma_T"))
    for model in ("gc","lorentz"):
        ref=report["references"][model]
        rows=[r for r in report["rows"] if r["model"]==model]
        selected=[r for r in rows if r["setting"]=="fine"] or rows
        fig,axes=plt.subplots(2,3,figsize=(15,7),layout="constrained")
        for j,(variable,symbol) in enumerate(panels):
            for i,stat in enumerate(("mean","variance")):
                ax=axes[i,j];name=variable+"_"+stat
                for k,row in enumerate(selected):
                    series(ax,times,[s[name]["value"] for s in row["evolution"]],[s[name]["half95"] for s in row["evolution"]],
                           std=i==1,label=model.upper(),color=f"C{k}")
                series(ax,times,ref["transient"][name],std=i==1,reference=True,label="Independent theory")
                ax.set(title=r"$\langle "+symbol+r"\rangle$" if i==0 else r"$\sigma("+symbol+r")$",xlabel=r"$t\nu_0$")
        axes[0,0].legend()
        fig.suptitle(f"{model.upper()} · Thermalization · N={meta['particles']:,}")
        save(fig,output.with_name(output.stem+"_"+model))
    # Keep the unabsorbed spatial-diffusion diagnostic.
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout="constrained")
    ref=np.asarray(report["references"]["gc"]["transient"]["spatial_integral"])
    for row in report["rows"]:
        if row["model"]!="gc":continue
        for ax,name in zip(axes[:2],("spatial_integral","displacement2")):
            series(ax,times,[s[name]["value"] for s in row["spatial"]],[s[name]["half95"] for s in row["spatial"]],
                   label=row["method"]+" / "+row["setting"])
        if row["setting"]=="fine":
            axes[2].plot(times,[s.get("noise",{}).get("mean",1.) for s in row["spatial"]],label=row["method"])
    for ax,title in zip(axes[:2],("Accumulated diffusion","Spatial spreading")):
        series(ax,times,ref,reference=True,label="Independent theory")
        ax.set(title=title,xlabel=r"$t\nu_0$",ylabel=r"$\langle R_\bot^2\rangle/\rho_\sigma^2$")
    half=1.96/np.sqrt(meta["particles"])
    axes[2].axhspan(1-half,1+half,color=".8",alpha=.3)
    axes[2].axhline(1.,color="k",ls="--");axes[2].set(title="Conditional-noise mean",xlabel=r"$t\nu_0$")
    axes[0].legend();save(fig,output.with_name(output.stem+"_spatial"))
    if not all("distribution" in report["references"][model] for model in ("gc","lorentz")):return
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for i,model in enumerate(("gc","lorentz")):
        ref=report["references"][model]["distribution"]
        row=min((r for r in report["rows"] if r["model"]==model),key=lambda r:(r["method"]=="fixed",r["cap"]))
        # Raw samples allow finer bins, evaluated against the same independent generator.
        family=meta.get("coefficient_model","constant_log")
        cls=PhysicalRelaxationReference if family=="physical" else RelaxationReference
        reference=cls(lower=.05*np.sqrt(2.) if model=="gc" else 0.,nodes=1024,
                      **(dict(law=impact_coefficients) if family=="impact" else {}))
        for j,name in enumerate(("speed","pitch")):
            ax=axes[i,j];edges=np.asarray(ref[name+"_edges"])
            if "histograms" in row:
                counts=np.asarray(row["histograms"][name]["counts"]);count=row["histograms"][name]["particles"]
                density=np.asarray(ref[name+"_density"])
            else:
                samples=row["final_"+name];edges=np.linspace(0.,6.,61) if name=="speed" else np.linspace(-1.,1.,51)
                counts=np.histogram(samples,edges)[0];count=len(samples)
                cdf=reference.speed_cdf(times[-1]) if name=="speed" else reference.pitch_cdf(times[-1],harmonics=12)
                density=np.diff(cdf(edges))/np.diff(edges)
            grid=np.linspace(edges[0],edges[-1],501)
            cdf=reference.speed_cdf(times[-1]) if name=="speed" else reference.pitch_cdf(times[-1],harmonics=12)
            histogram(ax,edges,counts,density,particles=count,initial=meta.get("initial_"+name,np.sqrt(10.) if name=="speed" else .6),
                      curve=(grid,np.gradient(cdf(grid),grid)))
            ax.set(title=model.upper()+": "+("Speed distribution" if name=="speed" else "Pitch distribution"),
                   xlabel=r"$v/\sigma_T$" if name=="speed" else r"$\xi$")
            ax.legend(fontsize=10)
            if name=="pitch":ax.set_ylim(top=max(counts)*1.4)
    fig.suptitle(f"Thermalized distributions · N={meta['particles']:,} · "+r"$t\nu_0=$"+f"{times[-1]:g}")
    save(fig,output.with_name(output.stem+"_distribution"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--horizon", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=6706)
    parser.add_argument("--caps", type=float, nargs=2, default=[.0025, .000625])
    parser.add_argument("--coefficient-model", choices=("constant_log", "physical", "impact"), default="constant_log")
    parser.add_argument("--methods", choices=("fixed", "conservative"), nargs="+", default=["fixed", "conservative"])
    parser.add_argument("--settings", choices=("coarse", "fine"), nargs="+", default=["coarse", "fine"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(json.loads(args.output.read_text()), args.output)
        return
    n, batch_size = args.particles, args.batch_size
    if not 2 <= n <= 2000 or not 2 <= batch_size <= n or n % batch_size:
        parser.error("use at most 2000 particles and a dividing batch size")
    if not np.isfinite(args.horizon) or args.horizon <= 0 or not all(np.isfinite(h) and h > 0 for h in args.caps) or args.caps[0] <= args.caps[1]:
        parser.error("positive horizon and coarse>fine>0 caps required")
    times = np.linspace(0., args.horizon, 41)
    metadata = dict(particles=n, batch_size=batch_size, seed=args.seed, times=times.tolist(), caps=args.caps,
        collision_tolerance=.001, initial_speed=float(np.sqrt(10.)), initial_pitch=.6,
        coefficient_model=args.coefficient_model, methods=args.methods, settings=args.settings,
        scope="Native collision-only relaxation from a monoenergetic anisotropic beam; declared coefficient law; no source or stopping",
        uncertainty="Independent particles; configurations share ordered keys, not exact Brownian paths; time samples are not pooled")
    report = dict(complete=False, metadata=metadata, rows=[])
    if args.output.exists():
        if not args.resume:
            raise ValueError("output exists; use --resume or a new path")
        report = json.loads(args.output.read_text())
        if "coefficient_model" not in report["metadata"]:
            # Completed early control files predate the explicit family flags.
            report["metadata"].update(coefficient_model="constant_log", methods=["fixed", "conservative"], settings=["coarse", "fine"],
                scope=metadata["scope"])
        if report["metadata"] != metadata:
            raise ValueError("resume metadata differs")
        report["complete"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    references, report["references"] = prepare_references(times, args.coefficient_model)
    save()
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(args.seed), i))(jnp.arange(n)).reshape((-1, batch_size))
    completed = {r["id"] for r in report["rows"]}
    cdf_alpha = .01/(2*2*len(args.methods)*len(args.settings))
    limit = float(np.sqrt(np.log(2/cdf_alpha)/(2*n)))
    for model in ("gc", "lorentz"):
        ref, expected, pitch_cdf = references[model]
        initial = ([0., 0., 0., np.sqrt(10.), .6, 0.] if model == "gc" else [np.sqrt(10.)*.8, 0., np.sqrt(10.)*.6])
        initial = jnp.broadcast_to(jnp.asarray(initial), (n, len(initial))).reshape((-1, batch_size, len(initial)))
        for method in args.methods:
            pending = [(setting, cap) for setting, cap in zip(("coarse", "fine"), args.caps)
                if setting in args.settings and f"{model}/{method}/{setting}" not in completed]
            if not pending:
                continue
            kwargs = (dict(coefficient_law=physical_native_coefficients("impact" if args.coefficient_model == "impact" else "nrl"))
                if args.coefficient_model != "constant_log" else {})
            function = thermal_solver(model, method, args.horizon, times, **kwargs)
            parameters = tuple(map(jnp.asarray, (.001, pending[0][1], pending[0][1])))
            executable = function.lower(initial[0], keys[0], *parameters).compile()
            for setting, cap in pending:
                identifier = f"{model}/{method}/{setting}"
                print(f"Running {identifier}: N={n}, horizon={args.horizon}, cap={cap}", flush=True)
                output, elapsed = run_batches(executable, list(zip(initial, keys)), tuple(map(jnp.asarray, (.001, cap, cap))))
                if np.any(output.status != 0):
                    raise RuntimeError("all unabsorbed thermal particles must reach the fixed horizon")
                if method != "fixed" and output.stats.max_accepted_error.max() > 1:
                    raise RuntimeError("an accepted adaptive collision error exceeds one")
                r = output.ys[:, :, 3] if model == "gc" else np.linalg.norm(output.ys, axis=-1)
                xi = output.ys[:, :, 4] if model == "gc" else output.ys[:, :, 2]/r
                row = dict(id=identifier, model=model, method=method, setting=setting, cap=cap,
                    elapsed_seconds=elapsed, **work(output),
                    evolution=[statistics(r[:, i], xi[:, i], {k: v[i] for k, v in expected.items()}) for i in range(len(times))],
                    final_speed=r[:, -1].tolist(), final_pitch=xi[:, -1].tolist(), cdf={})
                for name, values, cdf in (("speed", r[:, -1], ref.speed_cdf(args.horizon)), ("pitch", xi[:, -1], pitch_cdf)):
                    check = kstest(values, cdf)
                    row["cdf"][name] = dict(distance=float(check.statistic), pvalue=float(check.pvalue),
                        familywise_limit=limit, within_limit=bool(check.statistic <= limit))
                if model == "gc":
                    row["spatial"] = []
                    for i, t in enumerate(times):
                        q, msd = output.ys[:, i, 5], np.sum(output.ys[:, i, :2]**2, axis=1)
                        value = dict(spatial_integral=estimate(q, expected["spatial_integral"][i]),
                            displacement2=estimate(msd, expected["spatial_integral"][i]))
                        if t:
                            value["noise"] = {k: v for k, v in conditional_spatial_statistics(msd, q).items() if k != "ratio"}
                        row["spatial"].append(value)
                report["rows"].append(row)
                completed.add(identifier)
                save()
                print(json.dumps(dict(id=identifier, elapsed=elapsed,
                    final_moments={k: dict(difference=v["difference"], z=v["z"]) for k, v in row["evolution"][-1].items()}, cdf=row["cdf"])), flush=True)
    screens = []
    count = len(report["rows"])*(len(times)-1)*7
    critical = float(norm.ppf(1-.01/(2*count)))
    for row in report["rows"]:
        deviations = [(abs(value["z"]), float(t), name) for t, sample in zip(times[1:], row["evolution"][1:])
            for name, value in sample.items() if value["z"] is not None]
        worst = max(deviations)
        screens.append(dict(id=row["id"], maximum_abs_z=worst[0], time=worst[1], quantity=worst[2],
            critical=critical, within_familywise_screen=bool(worst[0] <= critical)))
    report["moment_screens"] = screens
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        "examples/benchmarks/validate_beam_relaxation.py", "examples/benchmarks/thermal_relaxation_reference.py",
        "examples/benchmarks/validate_collision_physics.py", "essos/split.py",
        "essos/background_species.py",
        "examples/benchmarks/nonrelativistic_collision_reference.py")}
    report["interpretation"] = "CDF and moment screens test statistical consistency, not a numerical-accuracy bound; compare both caps and report achieved uncertainty."
    report["complete"] = True
    save()
    table = [dict(id=row["id"], time=float(t), quantity=name, **value)
        for row in report["rows"] for t, sample in zip(times, row["evolution"]) for name, value in sample.items()]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    plot(report, args.output)
    print(json.dumps(dict(complete=True, moment_screens=screens)), flush=True)


if __name__ == "__main__":
    main()
