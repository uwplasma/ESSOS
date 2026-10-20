"""Paul et al. (2023), Table 2, vacuum QA / perfect QS, on CPU or GPU.

From the checkout root:
    python -m examples.particle_tracing.analytical_alfven_paper_benchmark

Defaults to 32 markers per case, 64 trajectories total, at most 120 seconds including
compilation. Uses the existing adaptive8 GC integrator. The low-resolution
equilibrium and axis-loss convention differ from the paper, so numerical
checks and published-value comparisons are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import tomllib

from examples.benchmarks.sampling import estimate_interval
from examples.benchmarks.paul_conventions import (CONVENTIONS, convert_births, convert_mode,
                                                 load_essos_births, particle_parameters)

def write_comparison(output, reference, results):
    """Persist published values beside ESSOS estimates without a physics pass/fail gate."""
    rows = []
    for case, result in zip(reference["cases"], results):
        for metric in ("loss_fraction", "mean_abs_delta_s"):
            estimate = result[metric]
            rows.append(dict(case=case["name"], metric=metric, paper=case[metric],
                             essos=estimate["value"], difference=estimate["value"]-case[metric],
                             approximate_95_low=estimate["interval_95"][0],
                             approximate_95_high=estimate["interval_95"][1],
                             reference_in_sampling_interval=bool(estimate["interval_95"][0]
                                                                 <= case[metric]
                                                                 <= estimate["interval_95"][1])))
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_report(output, summary):
    reference, rows = summary["reference"], summary["comparisons"]
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    report = ["# Paul et al. (2023) vacuum-QA transport comparison", "",
              f"Source: [Table 2, Perfect QS]({reference['source']}).", "",
              "| Case | Quantity | Paper | ESSOS | Approx. 95% sampling interval |",
              "| --- | --- | ---: | ---: | --- |"]
    for row in rows:
        report.append(f"| {row['case']} | {row['metric']} | {row['paper']:.3f} | "
                      f"{row['essos']:.4f} | [{row['approximate_95_low']:.4f}, "
                      f"{row['approximate_95_high']:.4f}] |")
    report += ["", f"{summary['total_trajectories']} trajectories total; "
               f"{summary['tmax']*1e3:g} ms physical time.", "",
               "Numerical conservation checks passed. Quantitative agreement with the paper "
               "is not established by this small comparison.", ""]
    report.extend(f"- {item}" for item in summary["limitations"])
    (output/"comparison.md").write_text("\n".join(report)+"\n")


def report_saved(output, reference):
    """Recover tabulated diagnostics without tracing or initializing JAX."""
    import numpy as np

    with np.load(output/"trajectories.npz") as archive:
        weights, times = archive["birth_weights"], archive["times"]
        results = []
        for index, case in enumerate(reference["cases"]):
            states, lcfs = archive[f"states_{index}"], archive[f"loss_times_{index}"]
            axes = archive[f"axis_stop_times_{index}"] if f"axis_stop_times_{index}" in archive else np.full_like(lcfs, -1.)
            losses = np.where(axes >= 0, axes, lcfs)
            drift = archive[f"invariant_drift_{index}"]
            valid = (losses[:, None] < 0) | (times < losses[:, None])
            max_drift = float(np.max(np.where(valid, drift, 0.)))
            if not np.isfinite(states).all() or not np.isfinite(max_drift) or max_drift > 1e-6:
                raise RuntimeError("Saved trajectories fail the numerical checks")
            # This archive is written only after both cases pass the failed-marker
            # and conservation checks. Frozen finite failed markers cannot reach it.
            results.append(dict(name=case["name"], particles=len(weights),
                                delta_bhat_psi=case["delta_bhat_psi"], lost=int((losses >= 0).sum()),
                                lcfs_exits=int((lcfs >= 0).sum()), axis_stops=int((axes >= 0).sum()),
                                failed=0, loss_fraction=estimate_interval(losses >= 0, weights, binary=True),
                                mean_abs_delta_s=estimate_interval(archive[f"displacement_{index}"], weights),
                                max_normalized_invariant_drift=max_drift,
                                minimum_saved_s=float(states[:, :, 0].min())))
        rows = write_comparison(output, reference, results)
        summary_file = output/"summary.json"
        # Preserve original provenance and timing when a prior summary exists.
        if summary_file.exists():
            saved_metadata = json.loads(summary_file.read_text())
        elif "metadata_json" in archive:
            saved_metadata = json.loads(str(archive["metadata_json"]))
        else:
            saved_metadata = None
        summary = saved_metadata if saved_metadata is not None else dict(
            reference=reference, particles_per_case=len(weights), total_trajectories=2*len(weights),
            tmax=float(times[-1]), saved_times=len(times), numerical_checks_passed=True,
            quantitative_paper_validation="not established",
            setup_metadata="Not stored in the original archive; see the benchmark command/defaults",
            limitations=["Low-resolution shipped QA equilibrium; differs from the authors' exact data",
                         f"Importance-weighted sampling; {len(weights)} proposals per case versus 5000 in the paper",
                         "Approximate intervals include sampling noise only",
                         "ESSOS counts LCFS exits; paper also counts s=0 with its singular potential",
                         "No equilibrium-resolution or timestep convergence study"])
        summary.update(results=results, comparisons=rows, report_rebuilt_from_saved_trajectories=True)
        write_report(output, summary)
        print(f"Rebuilt comparison tables from saved data in {output}; no particles traced", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=32, help="per case; two cases total")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--diagnostic-batch-size", type=int, default=128,
                        help="Bound the device memory used by invariant diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--births", type=Path, help="Use a prefix of an already-converted saved marker pool")
    parser.add_argument("--wout", type=Path, help="Explicit QA equilibrium; bundled reactor-scale input by default")
    parser.add_argument("--source-convention", choices=CONVENTIONS, default="essos",
                        help="Convention of the source births and wave; converted together to ESSOS")
    parser.add_argument("--particle-model", choices=CONVENTIONS, default="essos",
                        help="Measured alpha mass by default; explicit historical library-mass control")
    parser.add_argument("--axis-stop-s", type=float, default=1e-8,
                        help="inner stopping surface for the singular mode; refine for cutoff convergence")
    parser.add_argument("--wall-seconds", type=float, default=120.)
    parser.add_argument("--report-only", action="store_true", help="rebuild tables from saved data; no tracing")
    parser.add_argument("--case", choices=("both", "mode"), default="both",
                        help="mode runs only the perturbed case and saves diagnostic data")
    parser.add_argument("--output", type=Path,
                        default=Path("examples/benchmarks/results/analytical_alfven_paper_benchmark"))
    args = parser.parse_args()
    if args.particles < 3 or args.diagnostic_batch_size < 1:
        parser.error("Need particles >= 3 and diagnostic-batch-size >= 1")
    if not 0 < args.wall_seconds < float("inf"):
        parser.error("wall-seconds must be positive and finite")
    os.environ.update(JAX_PLATFORMS="cuda" if args.platform == "gpu" else "cpu", ESSOS_CPU_DEVICES="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.platform == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.perf_counter()

    def timeout():
        print(f"Stopped at the {args.wall_seconds:g}-second wall-time limit", flush=True)
        os._exit(124)

    deadline = threading.Timer(args.wall_seconds, timeout)
    deadline.daemon = True
    deadline.start()
    try:
        checkout = Path(__file__).resolve().parents[2]
        reference_file = checkout/"examples/benchmarks/references/paul2023_vacuum_qa.toml"
        reference = tomllib.loads(reference_file.read_text())
        if args.report_only:
            report_saved(args.output, reference)
            return
        import jax
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from essos.constants import ALPHA_PARTICLE_MASS, ONE_EV
        from .analytical_alfven_transport import (load_field, make_mode, trace,
                                                  invariant_errors)
        particle = particle_parameters(args.particle_model, measured_mass=ALPHA_PARTICLE_MASS,
            energy_joule=reference["energy_MeV"]*1e6*ONE_EV)
        mass, speed = particle["mass_kg"], particle["birth_speed_m_s"]
        assert all(device.platform == args.platform for device in jax.devices())
        args.wout = args.wout or checkout/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
        args.helicity, args.m, args.mboz, args.nboz = 0, 15, 16, 16
        args.regular_axis = False
        args.tmax, args.saves = reference["tmax_seconds"], 1001
        args.method, args.timestep, args.tolerance = "adaptive8", 5e-8, 1e-9
        args.output.mkdir(parents=True, exist_ok=True)
        case_count = 2 if args.case == "both" else 1
        print(f"{args.platform.upper()}; {args.particles} markers x {case_count} cases; tmax={args.tmax:g} s; "
              f"adaptive8; hard wall limit {args.wall_seconds:g} s", flush=True)
        field, provenance = load_field(args)
        provenance["wout_sha256"] = hashlib.sha256(args.wout.read_bytes()).hexdigest()
        print(f"Field ready after {time.perf_counter()-started:.1f} s", flush=True)

        rng = np.random.default_rng(args.seed)
        if args.births is None:
            source_births = (rng.uniform(reference["s_min"], reference["s_max"], args.particles),
                             rng.uniform(0, 2*np.pi, args.particles), rng.uniform(0, 2*np.pi, args.particles),
                             rng.uniform(-1, 1, args.particles))
            births = (*convert_births(*source_births, source=args.source_convention), speed)
        else:
            births = (*load_essos_births(args.births, args.particles, source=args.source_convention), speed)
        profiles, _ = jax.vmap(field.profiles)(births[0])
        iota, G, I = np.asarray(profiles).T
        B = np.asarray(jax.vmap(field.modB)(*births[:3]))
        # Uniform coordinate proposals, with the equilibrium Boozer Jacobian
        # as importance weights, target uniform physical volume. Keep the
        # same proposals, pitches and weights in both cases.
        weights = np.abs(float(field.psi0)*(G+iota*I)/B**2)
        weights /= weights.sum()
        arrays = dict(birth_s=births[0], birth_theta=births[1], birth_zeta=births[2],
                      birth_pitch=births[3], birth_weights=weights)
        results = []
        for index, case in enumerate(reference["cases"]):
            if args.case == "mode" and index == 0:
                continue
            source_wave = make_mode(field, args, case["delta_bhat_psi"], case["n"], case["omega_rad_s"])
            wave = convert_mode(source_wave, source=args.source_convention)
            print(f"Tracing {case['name']}; elapsed {time.perf_counter()-started:.1f} s", flush=True)
            result = trace(field, args, births, wave, mass=mass)
            # Keep raw evidence even when a subsequent validity check fails.
            np.savez_compressed(args.output/f"case_{index}.npz", times=result.times,
                                 states=result.states, loss_times=result.loss_times,
                                 failed_times=result.failed_times, axis_stop_times=result.axis_stop_times,
                                 **{key: value for key, value in arrays.items() if key.startswith("birth_")})
            if result.failed.any() or not np.isfinite(result.states).all():
                failed = np.flatnonzero(result.failed)
                lower = float(weights@(result.lost | result.axis_stopped))
                unknown = float(weights@result.failed)
                diagnostic = dict(case=case["name"], particles=args.particles, seed=args.seed, **particle,
                                  backend=str(jax.devices()[0]), tmax=args.tmax,
                                  method=args.method, tolerance=args.tolerance,
                                  provenance=provenance, lost=int(result.lost.sum()),
                                  axis_stops=int(result.axis_stopped.sum()), axis_stop_s=args.axis_stop_s,
                                  failed=int(result.failed.sum()),
                                  numerical_checks_passed=False,
                                  empirical_loss_fraction_bounds=[lower, lower+unknown],
                                  bounds_note="Unresolved failures may or may not be lost; these are not confidence intervals",
                                  failed_markers=[dict(index=int(i), birth_s=float(births[0][i]),
                                                       birth_pitch=float(births[3][i]),
                                                       failed_time=float(result.failed_times[i]),
                                                       last_finite_s=float(result.states[i, -1, 0]),
                                                       minimum_saved_s=float(result.states[i, :, 0].min()))
                                                  for i in failed])
                (args.output/f"case_{index}_diagnostic.json").write_text(json.dumps(diagnostic, indent=2)+"\n")
                print(json.dumps(diagnostic), flush=True)
                raise RuntimeError(f"{case['name']}: {int(result.failed.sum())} failed markers; "
                                   "loss comparison would be invalid")
            drift = invariant_errors(field, wave, result, args.helicity,
                                     batch_size=args.diagnostic_batch_size, mass=mass)
            stop_times = np.where(result.axis_stopped, result.axis_stop_times, result.loss_times)
            valid = (stop_times[:, None] < 0) | (result.times < stop_times[:, None])
            max_drift = float(np.max(np.where(valid, drift, 0.)))
            if max_drift > 1e-6:
                raise RuntimeError(f"{case['name']}: invariant drift {max_drift:g} exceeds 1e-6")
            displacement = np.abs(result.states[:, -1, 0]-births[0])
            loss_curve = weights@((stop_times[:, None] >= 0) & (stop_times[:, None] <= result.times))
            results.append(dict(name=case["name"], particles=args.particles, **particle,
                                delta_bhat_psi=case["delta_bhat_psi"],
                                source_omega_rad_s=float(source_wave.omega),
                                essos_omega_rad_s=float(wave.omega),
                                source_potential_volts=float(source_wave.amplitude),
                                potential_volts=float(wave.amplitude), lost=int((result.lost | result.axis_stopped).sum()),
                                lcfs_exits=int(result.lost.sum()), axis_stops=int(result.axis_stopped.sum()),
                                axis_stop_s=args.axis_stop_s,
                                failed=int(result.failed.sum()),
                                loss_fraction=estimate_interval(result.lost | result.axis_stopped, weights, binary=True),
                                mean_abs_delta_s=estimate_interval(displacement, weights),
                                max_normalized_invariant_drift=max_drift,
                                minimum_saved_s=float(result.states[:, :, 0].min())))
            for key, value in dict(states=result.states, loss_times=result.loss_times,
                                   axis_stop_times=result.axis_stop_times,
                                   loss_curve=loss_curve, displacement=displacement,
                                   invariant_drift=drift).items():
                arrays[f"{key}_{index}"] = value
            (args.output/f"case_{index}_summary.json").write_text(json.dumps(results[-1], indent=2)+"\n")
            print(json.dumps(results[-1]), flush=True)

        if args.case == "mode":
            (args.output/"mode_summary.json").write_text(json.dumps(results[-1], indent=2)+"\n")
            return
        rows = write_comparison(args.output, reference, results)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        colors = ("tab:blue", "tab:orange")
        for index, (case, color) in enumerate(zip(reference["cases"], colors)):
            axes[0].plot(result.times*1e3, 100*arrays[f"loss_curve_{index}"],
                         color=color, label=f"ESSOS: {case['name']}")
            axes[0].scatter([1.], [100*case["loss_fraction"]], marker="D", s=40,
                            color=color, edgecolor="black", zorder=5,
                            label=f"Paper endpoint: {case['name']}")
        axes[0].set(xlabel="Time (ms)", ylabel="Volume-weighted loss fraction (%)",
                    title="ESSOS curves; published endpoints")
        axes[0].legend(fontsize=7)
        for ax, metric, scale, title in zip(axes[1:], ("loss_fraction", "mean_abs_delta_s"),
                                            (100., 1.), ("Loss fraction at 1 ms (%)", r"Mean $|\Delta s|$ at 1 ms")):
            for index, (case, measured, color) in enumerate(zip(reference["cases"], results, colors)):
                estimate = measured[metric]
                low, high = estimate["interval_95"]
                ax.scatter(index-.1, scale*case[metric], marker="D", color="black", s=40,
                            label="Paper, Table 2" if index == 0 else None)
                ax.errorbar(index+.1, scale*estimate["value"],
                             yerr=scale*np.array([[estimate["value"]-low], [high-estimate["value"]]]),
                             fmt="o", color=color, capsize=4,
                             label="ESSOS, approx. 95% interval" if index == 0 else None)
            ax.set(xticks=[0, 1], xticklabels=["Unperturbed", "m=15 mode"], title=title,
                    xlim=(-.5, 1.5))
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f"Paul et al. (2023), Table 2: vacuum QA / perfect QS\n"
                     f"ESSOS: {args.particles} markers/case, {args.platform.upper()} adaptive8; "
                     "low-resolution equilibrium, sampling intervals only")
        for extension in ("png", "svg"):
            fig.savefig(args.output/f"paper_comparison.{extension}", dpi=180)
        plt.close(fig)
        summary = dict(reference=reference, reference_file=str(reference_file.relative_to(checkout)),
                       provenance=provenance, backend=str(jax.devices()[0]),
                       particles_per_case=args.particles, total_trajectories=2*args.particles,
                       tmax=args.tmax, saved_times=len(result.times), seed=args.seed, **particle,
                       source_convention=args.source_convention, birth_coordinate_convention="essos",
                       birth_pool=None if args.births is None else str(args.births.resolve()),
                       method=args.method, tolerance=args.tolerance, initial_timestep=args.timestep,
                       axis_stop_s=args.axis_stop_s, loss_definition="LCFS exits plus singular-mode axis stops",
                       platform=args.platform, diagnostic_batch_size=args.diagnostic_batch_size,
                       numerical_checks_passed=True, quantitative_paper_validation="not established",
                       results=results, comparisons=rows, wall_seconds=time.perf_counter()-started,
                       wall_limit_seconds=args.wall_seconds,
                       limitations=[f"{args.particles} markers per case versus the paper's 5000; intervals are approximate",
                                    "Importance-weighted proposals approximate uniform physical volume",
                                    "Exact published equilibrium/runtime unconfirmed; no equilibrium/step convergence study",
                                    "Field normalized to innermost B00=5.86 T and minor radius=1.70 m",
                                    "Historical SIMSOPT convention is explicit; the paper's exact runtime is unconfirmed",
                                    "Axis stops use a finite cutoff and chord interpolation; refine cutoff and timestep",
                                    "Intervals include sampling noise only, not model or numerical uncertainty"])
        np.savez_compressed(args.output/"trajectories.npz", times=result.times,
                             metadata_json=json.dumps(summary), **arrays)
        write_report(args.output, summary)
        print(f"Saved comparison plot, CSV, report and trajectories in {args.output}; "
              f"total {summary['wall_seconds']:.1f} s", flush=True)
    finally:
        deadline.cancel()


if __name__ == "__main__":
    main()
