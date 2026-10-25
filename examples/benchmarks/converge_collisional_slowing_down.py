"""Source-free native controller convergence, including stopped GC diffusion.

512 independent particles per configuration by default, in batches of 64.
Repeated timing calls reuse those particles; they do not increase sample size.
No Diffrax solves, source, or Lorentz gyromotion benchmark is included.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

# Share the production collision/Brownian wrapper; JAX selects the backend.
from .validate_collisional_slowing_down import native_solver
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import t as student_t, kstest, chi2

from essos.split import (CollisionCoefficients, guiding_center_spatial_diffusion,
    guiding_center_spatial_increment)
from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q
from essos.split import SplitStatus
from .nonrelativistic_collision_reference import (ProtonBackground, proton_first_passage,
    passage_statistics, quadrature_mean)
from .spatial_collision_reference import stopped_spatial_reference


COLORS = dict(fixed="#0072B2", conservative="#D55E00", paper="#009E73", candidate="#CC79A7")
SETTINGS = {
    "tol_loose": (.01, .05), "base": (.003, .05), "tol_tight": (.001, .05),
    "cap_mid": (.003, .0125), "cap_fine": (.003, .003125),
    "joint_fine": (.001, .003125),
}
FAMILIES = dict(tolerance=["tol_loose", "base", "tol_tight"],
                cap=["base", "cap_mid", "cap_fine"])


def estimate(values, expected=None):
    """Independent-particle mean and exploratory two-sided 95% t interval."""
    x = np.asarray(values, dtype=float)
    n = x.shape[0]
    mean = x.mean(axis=0)
    se = x.std(axis=0, ddof=1)/np.sqrt(n)
    half = student_t.ppf(.975, n-1)*se
    out = dict(mean=mean.tolist(), se=se.tolist(), half95=half.tolist(),
               interval95=np.stack([mean-half, mean+half], axis=-1).tolist())
    if expected is not None:
        out["relative_bias"] = (mean/np.asarray(expected)-1).tolist()
    return out


def paired_difference(left, right):
    """Seed-paired weak observable, without claiming identical Brownian paths."""
    x, y = np.asarray(left), np.asarray(right)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("paired observations must have matching one-dimensional shapes")
    return estimate(x-y)


def frozen_spatial_check(physical, paths, seed):
    """One frozen kick: zero drift, perpendicular covariance and B^-2 law."""
    h, pitch, omega = .01/physical.nu, .6, Q/M
    values = physical.coefficients(np.sqrt(10)*physical.sigma)
    c = CollisionCoefficients(*(jnp.asarray(values.get(name, 0.)) for name in CollisionCoefficients._fields))
    dw = jax.random.normal(jax.random.key(seed), (paths, 3), dtype=jnp.float64)*np.sqrt(h)
    b = jnp.array([0., 0., 1.])
    advance = jax.jit(jax.vmap(lambda w, om: guiding_center_spatial_increment(pitch, w, c, b, om), in_axes=(0, None)))
    x = np.asarray(advance(dw, omega))
    twice_b = np.asarray(advance(dw, 2*omega))
    dx = float(guiding_center_spatial_diffusion(pitch, c, omega))
    variance = 2*dx*h
    # Only the frozen one-component squared increment / local variance is chi2_1.
    ks = kstest(x[:, 0]**2/variance, chi2(1).cdf)
    return dict(paths=paths, h_seconds=h, field_tesla=1., local_D_m2_per_second=dx,
        component_variance_theory_m2=variance, component_mean=estimate(x),
        component_second_moment_ratio=(np.mean(x*x, axis=0)/variance).tolist(),
        component_cross_moment_ratio=float(np.mean(x[:, 0]*x[:, 1])/variance),
        parallel_max_abs_m=float(abs(x[:, 2]).max()),
        paired_double_B_displacement_residual_m=float(abs(twice_b-x/2).max()),
        chi_square_1_ks_distance=float(ks.statistic), chi_square_1_ks_pvalue=float(ks.pvalue))


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    flat = []
    for row in report["rows"]:
        r = {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
        r.update(censored=row["statistics"]["censored"], cdf_distance=row["statistics"]["cdf"]["distance"])
        for name, m in row["statistics"]["moments"].items():
            r.update({name+"_"+k: v for k, v in m.items()})
        for name in ("terminal_msd", "integrated_diffusion", "msd_minus_integral"):
            stats = row.get("spatial", {}).get(name, {})
            for k in ("mean", "se", "relative_bias"):
                r[name+"_"+k] = stats.get(k)
        flat.append(r)
    if flat:
        with output.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def add_contrasts(report):
    comparisons = []
    for model in ("gc", "lorentz"):
        rows = [r for r in report["rows"] if r["model"] == model]
        for method in COLORS:
            active = [r for r in rows if r["method"] == method]
            pairs = [(a, b) for a, b in zip(active, active[1:])] if method == "fixed" else [
                (next(r for r in active if r["setting"] == a), next(r for r in active if r["setting"] == b))
                for family in FAMILIES.values() for a, b in zip(family, family[1:])]
            for a, b in pairs:
                entry = dict(model=model, method=method, coarse=a["setting"], fine=b["setting"],
                             stopping_mean_difference=paired_difference(a["times"], b["times"]))
                if model == "gc":
                    entry["integrated_diffusion_difference"] = paired_difference(
                        a["spatial"]["integral_samples"], b["spatial"]["integral_samples"])
                comparisons.append(entry)
        finals = [r for r in rows if r["setting"] == "joint_fine"]
        for i, a in enumerate(finals):
            for b in finals[i+1:]:
                comparisons.append(dict(model=model, left=a["method"], right=b["method"], setting="joint_fine",
                    stopping_mean_difference=paired_difference(a["times"], b["times"])))
    report["paired_contrasts"] = comparisons


def plot(report, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for i, model in enumerate(("gc", "lorentz")):
        rows = [r for r in report["rows"] if r["model"] == model]
        for method, color in COLORS.items():
            active = [r for r in rows if r["method"] == method]
            for j, family in enumerate(("tolerance", "cap")):
                if method == "fixed" and family == "tolerance":
                    continue
                selected = active if method == "fixed" else [next(r for r in active if r["setting"] == s) for s in FAMILIES[family]]
                x = [r["tolerance"] if family == "tolerance" else r["dtmax"] for r in selected]
                for col, quantity in ((j, "restricted_mean"), (2, "restricted_variance")):
                    if col == 2 and family != "cap":
                        continue
                    m = [r["statistics"]["moments"][quantity] for r in selected]
                    axes[i, col].errorbar(x, [100*(z["value"]/z["expected"]-1) for z in m],
                        yerr=[196*z["mc_se"]/z["expected"] for z in m], color=color, marker="o", capsize=3, label=method)
        for j, ax in enumerate(axes[i]):
            ax.axhline(0., color="k", ls="--")
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.grid(alpha=.2)
            ax.set(xlabel="Collision tolerance (cap = 0.05)" if j == 0 else r"Maximum interval $h\nu_0$ (adaptive tol = 0.003)",
                ylabel="Relative difference from reference (%)",
                title=f"{model.upper()}: {'stopping mean' if j < 2 else 'stopping variance'}")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"Source-free slowing down: N={report['metadata']['paths']} per configuration\n"
        "Bars = 1.96 reference MC SE; tolerance and interval cap varied separately; no rate inferred from noisy sample errors")
    fig.savefig(output.with_suffix(".png"), dpi=170)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, model in enumerate(("gc", "lorentz")):
        for r in report["rows"]:
            if r["model"] != model or r["setting"] not in ("joint_fine", "fixed_fine"):
                continue
            for j, quantity in enumerate(("restricted_mean", "restricted_variance")):
                samples = r["particle_prefixes"]
                m = [p["statistics"]["moments"][quantity] for p in samples]
                axes[i, j].errorbar([p["paths"] for p in samples], [100*(v["value"]/v["expected"]-1) for v in m],
                    yerr=[196*v["mc_se"]/v["expected"] for v in m], marker="o", capsize=3, color=COLORS[r["method"]], label=r["method"])
        for j, ax in enumerate(axes[i]):
            ax.axhline(0., color="k", ls="--")
            ax.set(xscale="log", xlabel="Particles (nested subsets of the same ensemble)",
                ylabel="Relative difference from reference (%)", title=f"{model.upper()}: {'mean' if j == 0 else 'variance'}")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Sampling convergence at jointly refined adaptive settings and finest fixed step\n"
        "Bars = 1.96 reference MC SE; nested prefixes are correlated, not independent repeats")
    fig.savefig(output.with_name(output.stem+"_sampling.png"), dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ts = np.asarray(report["metadata"]["save_times"])
    expected = np.asarray(report["reference"]["spatial_msd"])
    for ax in axes[0]:
        ax.plot(ts, expected, "k--", lw=2, label="Independent backward FP")
    for method, color in COLORS.items():
        rows = [r for r in report["rows"] if r["model"] == "gc" and r["method"] == method]
        final = next(r for r in rows if r["setting"] == ("fixed_fine" if method == "fixed" else "joint_fine"))
        s = final["spatial"]["evolution"]
        for j, name in enumerate(("msd", "integral")):
            mean, half = np.asarray(s[name]["mean"]), np.asarray(s[name]["half95"])
            axes[0, j].plot(ts, mean, color=color, label=method)
            axes[0, j].fill_between(ts, mean-half, mean+half, color=color, alpha=.12)
        selected = rows if method == "fixed" else [next(r for r in rows if r["setting"] == s) for s in FAMILIES["cap"]]
        for j, name in enumerate(("terminal_msd", "integrated_diffusion")):
            axes[1, j].errorbar([r["dtmax"] for r in selected],
                [100*r["spatial"][name]["relative_bias"] for r in selected],
                yerr=[100*r["spatial"][name]["half95"]/expected[-1] for r in selected],
                color=color, marker="o", capsize=3, label=method)
    for j, title in enumerate(("Stopped perpendicular mean squared displacement", "Expected variance from accumulated 4 D_X dt")):
        axes[0, j].set(title=title, xlabel=r"Time $t\nu_0$", ylabel=r"Squared distance / $\rho_\sigma^2$")
        axes[1, j].set(title=f"Terminal {'displacement' if j == 0 else 'diffusion integral'}: cap refinement",
            xlabel=r"Maximum interval $h\nu_0$", ylabel="Relative difference from reference (%)", xscale="log")
        axes[1, j].invert_xaxis()
        axes[1, j].axhline(0., color="k", ls="--")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f"Uniform-field GC spatial diffusion, B=1 T, N={report['metadata']['paths']}\n"
        "Evolution: adaptive tol=0.001, cap=0.003125; fixed h=0.000625. Cap sweep: adaptive tol=0.003.\n"
        "Stopped at first E <= 2T or H; shading/bars = exploratory 95% particle-mean t intervals", fontsize=11)
    fig.savefig(output.with_name(output.stem+"_spatial.png"), dpi=170)
    fig.savefig(output.with_name(output.stem+"_spatial.svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1706)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(json.loads(args.output.read_text()), args.output)
        return
    if not 2 <= args.paths <= 512 or not 2 <= args.batch_size <= args.paths or args.paths % args.batch_size or args.repeats < 1:
        parser.error("use 2..512 particles, a dividing batch size >=2, and positive timing repeats")
    physical = ProtonBackground()
    reference = proton_first_passage(physical)
    horizon = 6*reference.mean
    ts = np.linspace(0., horizon, 21)
    sr = stopped_spatial_reference(physical)
    spatial = sr.msd(ts)
    sc = stopped_spatial_reference(physical, nodes=384).msd(ts)
    se = stopped_spatial_reference(physical, upper=10., nodes=1024).msd(ts)
    pc = proton_first_passage(physical, nodes=384)
    pe = proton_first_passage(physical, upper=10., nodes=1024)
    checks = dict(spatial_mesh_relative_max=float(abs(sc-spatial).max()/spatial[-1]),
        spatial_upper_relative_max=float(abs(se-spatial).max()/spatial[-1]),
        passage_mean_mesh_relative=abs(pc.mean/reference.mean-1),
        passage_cdf_mesh_max=float(abs(pc.cdf(ts)-reference.cdf(ts)).max()),
        passage_cdf_upper_max=float(abs(pe.cdf(ts)-reference.cdf(ts)).max()),
        passage_mean_quadrature_relative=abs(reference.mean/quadrature_mean(physical, 2., np.sqrt(10.))-1))
    if max(checks.values()) > 5e-4:
        raise RuntimeError(f"reference not converged: {checks}")
    raw = reference.restricted_moments(horizon)
    root = Path(__file__).resolve().parents[2]
    sources = ["essos/split.py"]
    sources += ["examples/benchmarks/"+s for s in ("nonrelativistic_collision_reference.py", "spatial_collision_reference.py",
        "validate_collisional_slowing_down.py", "converge_collisional_slowing_down.py")]
    report = dict(complete=False, metadata=dict(paths=args.paths, batch_size=args.batch_size,
        seed=args.seed, timing_repeats=args.repeats, controllers=list(COLORS)[1:],
        backend=jax.default_backend(), jax=jax.__version__, horizon=horizon, nu0_per_second=physical.nu,
        sigma_m_per_second=physical.sigma, magnetic_field_tesla=1., rho_sigma_m=physical.sigma*M/Q,
        initial_energy_over_T=5., threshold_energy_over_T=2., pitch=.6, save_times=ts.tolist(),
        uncertainty="Independent particles; shared ordered keys across settings, not identical Brownian paths; exploratory marginal intervals, not simultaneous rankings",
        sampling="Nested 32/128/512-particle prefixes and disjoint execution blocks; timing repeats reuse keys",
        spatial="GC Eq. (36)/(69), uniform B along z; collision-only perpendicular displacement, frozen after thermal-band entry",
        lorentz="Velocity-space slowing down only; no Lorentz orbit/spatial-transport claim",
        stepping="Save times force common split boundaries; endpoint-detected crossings, no Brownian crossing correction; no source",
        settings=SETTINGS, families=FAMILIES,
        source_sha256={s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources}),
        reference=dict(mean=reference.mean, variance=reference.variance,
            restricted_mean=float(raw[0]), restricted_variance=float(raw[1]-raw[0]**2),
            cdf_at_horizon=float(reference.cdf(horizon)), refinement=checks,
            spatial_msd=spatial.tolist()), frozen_spatial=frozen_spatial_check(physical, args.paths, args.seed+1), rows=[])
    save(report, args.output)
    keys = jax.random.split(jax.random.key(args.seed), args.paths).reshape((-1, args.batch_size))
    rho = physical.sigma*M/Q
    print(json.dumps(dict(reference=report["reference"], frozen_spatial=report["frozen_spatial"])), flush=True)
    for model in ("gc", "lorentz"):
        for method in COLORS:
            adaptive = method != "fixed"
            controller = method if adaptive else "conservative"
            start = perf_counter()
            compiled = native_solver(model, controller, adaptive, physical, horizon, reference.boundary,
                spatial=model == "gc", times=ts).lower(keys[0], jnp.array(.003), jnp.array(.05), jnp.array(.005)).compile()
            compile_seconds = perf_counter()-start
            print(f"Compiled {model}/{method}: {compile_seconds:.2f} s", flush=True)
            settings = SETTINGS if adaptive else {"fixed_coarse": (None, .01), "fixed_mid": (None, .0025), "fixed_fine": (None, .000625)}
            for name, (tolerance, cap) in settings.items():
                initial = min(.005, cap) if adaptive else cap
                params = tuple(jnp.asarray(v) for v in (tolerance or .003, cap, initial))
                # Warm once. Subsequent complete 512-path execution medians exclude JIT.
                jax.block_until_ready(compiled(keys[0], *params))
                warm, outputs = [], []
                for repeat in range(args.repeats):
                    start = perf_counter()
                    batch_outputs = [jax.block_until_ready(compiled(k, *params)) for k in keys]
                    warm.append(1000*(perf_counter()-start))
                    if repeat == 0:
                        outputs = [jax.device_get(o) for o in batch_outputs]
                out = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *outputs)
                if np.any((out.status != SplitStatus.SUCCESS) & (out.status != SplitStatus.EVENT)):
                    raise RuntimeError(f"failed {model}/{method}/{name}: {np.unique(out.status, return_counts=True)}")
                if adaptive and np.max(out.stats.max_accepted_error) > 1:
                    raise RuntimeError("accepted collision error exceeds one")
                # Clamp only the final SI->collision-time roundoff at H.
                if np.any(out.t > horizon*(1+1e-14)):
                    raise RuntimeError("stopping time outside reporting horizon")
                times = np.minimum(out.t, horizon)
                hit = out.status == SplitStatus.EVENT
                row = dict(model=model, method=method, controller=controller, adaptive=adaptive,
                    setting=name, tolerance=tolerance, dtmax=cap, initial_step=initial,
                    compile_seconds=compile_seconds, warm_median_ms=float(np.median(warm)), warm_ms=warm,
                    accepted_mean=float(out.stats.accepted.mean()), rejected_mean=float(out.stats.rejected.mean()),
                    trials_mean=float(out.stats.steps.mean()), max_accepted_error=float(out.stats.max_accepted_error.max()),
                    times=times.tolist(), hit=hit.tolist(), statistics=passage_statistics(times, hit, reference, horizon),
                    particle_prefixes=[dict(paths=n, statistics=passage_statistics(times[:n], hit[:n], reference, horizon))
                        for n in sorted(set([n for n in (32, 128, args.paths) if n <= args.paths]))],
                    execution_block_means=times.reshape((-1, args.batch_size)).mean(axis=1).tolist())
                if model == "gc":
                    position = out.ys[:, :, :3]/rho
                    msd = np.sum(position[:, :, :2]**2, axis=2)
                    integral = out.ys[:, :, 5]/rho**2
                    row["spatial"] = dict(terminal_msd=estimate(msd[:, -1], spatial[-1]),
                        integrated_diffusion=estimate(integral[:, -1], spatial[-1]),
                        msd_minus_integral=estimate(msd[:, -1]-integral[:, -1]),
                        terminal_position_mean=estimate(position[:, -1]),
                        max_parallel_displacement_m=float(abs(out.y[:, 2]).max()),
                        terminal_msd_samples=msd[:, -1].tolist(), integral_samples=integral[:, -1].tolist(),
                        evolution=dict(msd=estimate(msd), integral=estimate(integral)))
                report["rows"].append(row)
                save(report, args.output)
                print(json.dumps({k: v for k, v in row.items() if k not in
                    ("times", "hit", "spatial", "particle_prefixes", "execution_block_means")}), flush=True)
    add_contrasts(report)
    report["complete"] = True
    save(report, args.output)
    plot(report, args.output)
    print(f"Saved {len(report['rows'])} configurations to {args.output}", flush=True)


if __name__ == "__main__":
    main()
