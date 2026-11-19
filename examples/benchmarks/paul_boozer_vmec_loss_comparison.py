"""Plot four loss curves from completed Boozer and VMEC cases.

No particle tracing or JAX import. Boozer is the Paul / Perfect QS
benchmark; VMEC retains the full equilibrium. Published Table 2 endpoints
use the corresponding columns. Birth proposals and pitches must match.
"""

import argparse
import csv
import json
from pathlib import Path


def compare(boozer, vmec, output, *, particles=None):
    import numpy as np
    from examples.benchmarks.sampling import estimate_interval

    boozer_summary = json.loads((boozer/"summary.json").read_text())
    vmec_summary = json.loads((vmec/"summary.json").read_text())
    if particles is None:
        particles = boozer_summary["particles_per_case"]
    if boozer_summary["particles_per_case"] != particles or vmec_summary["particles_per_case"] != particles:
        raise ValueError(f"Both inputs must contain exactly {particles} markers per case")
    if not boozer_summary["numerical_checks_passed"] or not vmec_summary["numerical_checks_passed"]:
        raise ValueError("Both comparisons must pass their numerical checks")
    if not vmec_summary["all_orbits_resolved"]:
        raise ValueError("Unresolved VMEC orbits cannot be presented as a completed loss comparison")
    source_convention = boozer_summary.get("source_convention", "essos")
    if vmec_summary.get("source_convention", "essos") != source_convention:
        raise ValueError("The two formulations use different source conventions")
    for key in ("mass_kg", "energy_joule"):
        if key in boozer_summary or key in vmec_summary:
            if key not in boozer_summary or key not in vmec_summary:
                raise ValueError(f"Both formulations must record {key} to compare explicit particle models")
            if not np.isclose(boozer_summary[key], vmec_summary[key], rtol=1e-12, atol=0.):
                raise ValueError(f"Different particle parameter {key}")
    representation = vmec_summary.get("representation", "raw")
    b_hash = boozer_summary.get("provenance", {}).get("wout_sha256")
    v_hash = vmec_summary.get("wout_sha256")
    if b_hash is not None and v_hash is not None and b_hash != v_hash:
        raise ValueError("The equilibrium source hashes differ, even if filenames match")
    for bcase, vcase in zip(boozer_summary["results"], vmec_summary["cases"]):
        for key in ("essos_omega_rad_s", "potential_volts"):
            if key in bcase and key in vcase:
                np.testing.assert_allclose(bcase[key], vcase[key], rtol=1e-12, atol=1e-12,
                                           err_msg=f"Different actual ESSOS wave parameter {key}")
    if len(boozer_summary["reference"]["cases"]) != 2 or len(vmec_summary["reference"]["cases"]) != 2:
        raise ValueError("Expected unperturbed and perturbed cases for each formulation")
    for bcase, vcase in zip(boozer_summary["reference"]["cases"], vmec_summary["reference"]["cases"]):
        for key in ("m", "n", "omega_rad_s", "delta_bhat_psi"):
            if bcase[key] != vcase[key]:
                raise ValueError(f"The two formulations use different wave parameter {key}")
    if not np.isclose(boozer_summary["tmax"], .001) or not np.isclose(vmec_summary["tmax_s"], .001):
        raise ValueError("This Paul comparison requires the full 1 ms duration")
    for key in ("B_scale", "length_scale"):
        np.testing.assert_allclose(boozer_summary["provenance"][key], vmec_summary[key], rtol=1e-13)
    if Path(boozer_summary["provenance"]["wout"]).name != Path(vmec_summary["wout"]).name:
        raise ValueError("The equilibrium source files differ")
    if boozer_summary["axis_stop_s"] != 1e-8:
        raise ValueError("The saved Boozer case must use the common s=1e-8 inner cutoff")
    output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0., .001, 1001)
    curves, rows, data = [], [], {}
    with np.load(boozer/"trajectories.npz") as old, np.load(vmec/"births.npz") as native:
        for key in ("birth_s", "birth_theta", "birth_zeta", "birth_pitch"):
            np.testing.assert_allclose(old[key], native[key], rtol=0, atol=1e-13,
                                       err_msg=f"Different initial {key}")
            if len(old[key]) != particles:
                raise ValueError("Birth archive marker count does not match its summary")
        for model, summary, weights in (("Boozer / Perfect QS", boozer_summary, old["birth_weights"]),
                                        ("VMEC / full equilibrium", vmec_summary, native["birth_weights"])):
            for index, case in enumerate(summary["reference"]["cases"]):
                if model.startswith("Boozer"):
                    lcfs, inner = old[f"loss_times_{index}"], old[f"axis_stop_times_{index}"]
                    lcfs, inner = np.where(lcfs >= 0, lcfs, np.inf), np.where(inner >= 0, inner, np.inf)
                    if summary["results"][index]["failed"]:
                        raise ValueError("The saved Boozer case has unresolved markers")
                else:
                    with np.load(vmec/f"case_{index}.npz") as saved:
                        lcfs, inner = saved["lcfs_times"], saved["axis_stop_times"]
                        if np.any(saved["status"] == 4):
                            raise ValueError("The saved VMEC case has unresolved markers")
                        if saved["times"][-1] != times[-1]:
                            raise ValueError("VMEC duration differs from the comparison")
                        np.testing.assert_allclose(saved["weights"], weights, rtol=1e-13)
                if len(lcfs) != particles or len(inner) != particles:
                    raise ValueError("Per-case marker count does not match the requested comparison")
                stop_times = np.minimum(lcfs, inner)
                weights = weights/weights.sum()
                curve = weights@(stop_times[:, None] <= times[None, :])
                estimate = estimate_interval(np.isfinite(stop_times), weights, binary=True)
                measured = (summary["results"][index]["loss_fraction"]["value"] if model.startswith("Boozer")
                            else summary["cases"][index]["weighted_confirmed_loss"])
                np.testing.assert_allclose(estimate["value"], measured, rtol=1e-12, atol=1e-14)
                curves.append((model, index, curve, case["loss_fraction"]))
                rows.append(dict(model=model, case=case["name"], particles=particles,
                                 lcfs_exits=int(np.isfinite(lcfs).sum()),
                                 inner_stops=int(np.isfinite(inner).sum()), failed=0,
                                 loss_fraction=estimate["value"],
                                 sampling_95_low=estimate["interval_95"][0],
                                 sampling_95_high=estimate["interval_95"][1],
                                 paper_loss_fraction=case["loss_fraction"]))
                data[f"loss_curve_{len(curves)-1}"] = curve
    _plot(output, times, curves, particles, source_convention=source_convention, representation=representation)
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output/"loss_curves.npz", times=times, **data)
    report = ["# Boozer and VMEC alpha loss fractions", "",
              f"{particles} particles per curve; 1 ms; collisionless adaptive GC. "
              "Perturbed cases: m=15, n=7, source omega=399000 rad/s, edge amplitude 1e-3.", "",
              f"Source convention: `{source_convention}`; actual ESSOS omega: "
              f"{boozer_summary['results'][1].get('essos_omega_rad_s', 399000.):g} rad/s. "
              f"VMEC field representation: `{representation}`.", "",
              "Identical initial coordinate proposals and pitches, with physical-volume "
              "importance weights computed for each background.", "",
              "| Model | Case | Loss | Approx. 95% sampling interval | LCFS exits | Inner stops | Paper |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for row in rows:
        report.append(f"| {row['model']} | {row['case']} | {100*row['loss_fraction']:.2f}% | "
                      f"[{100*row['sampling_95_low']:.2f}%, {100*row['sampling_95_high']:.2f}%] | "
                      f"{row['lcfs_exits']} | {row['inner_stops']} | {100*row['paper_loss_fraction']:.1f}% |")
    representation_note = (
        "This comparison uses raw VMEC tables; it does not establish convergence in "
        "equilibrium resolution or interpolation. "
    )
    report.extend(["", "Boozer uses the earlier Paul benchmark with QS-breaking harmonics filtered out; "
                   "VMEC retains the full equilibrium. This comparison includes that background "
                   "difference as well as the coordinate formulation. Paper endpoints use the matching "
                   "Perfect QS and Equil. columns of Table 2.", "",
                   "All four cases have zero unresolved failures. Inner stops at s=1e-8 are counted "
                   "as paper-style losses, separately from LCFS exits. The exact published "
                   "equilibrium/runtime remains unconfirmed; the paper used 5000 markers.", "",
                   "The historical SIMSOPT convention is an explicit comparison assumption; "
                   "the exact paper runtime has not been established. " + representation_note +
                   "Sampling intervals exclude these systematic uncertainties.", "",
                   f"Source: {boozer_summary['reference']['source']}", ""])
    (output/"comparison.md").write_text("\n".join(report))
    (output/"summary.json").write_text(json.dumps(dict(particles_per_case=particles,
        tmax_s=.001, boozer_source=str(boozer), vmec_source=str(vmec), cases=rows,
        source_convention=source_convention, vmec_representation=representation,
        background_difference="Boozer Perfect QS; VMEC full equilibrium"), indent=2)+"\n")
    print(json.dumps(rows, indent=2), flush=True)


def _plot(output, times, curves, particles, *, source_convention="essos", representation="raw", stem=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    for k, (model, index, curve, paper) in enumerate(curves):
        color = ("#2878b5", "#004c8c", "#e58b26", "#a83f16")[k]
        condition = "unperturbed" if index == 0 else "m=15 wave"
        ax.step(times*1e3, 100*curve, where="post", color=color,
                linestyle="--" if index == 0 else "-", linewidth=2,
                label=f"{model}, {condition} ({100*curve[-1]:.2f}%)")
        ax.plot(1., 100*paper, "x", color=color, markersize=9, markeredgewidth=2)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="x", linestyle="none", color="black", markersize=8))
    labels.append("Paul Table 2 endpoint (matching background)")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0., -.15),
              fontsize=9, frameon=False)
    ax.set(xlabel="Time [ms]", ylabel="Volume-weighted loss fraction [%]", xlim=(0, 1.025),
           title=f"Paul alpha transport: {particles} particles per curve\n"
                 f"Source: {source_convention}; VMEC: {representation}")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    stem = stem or f"boozer_vmec_loss_{particles}"
    for extension in ("png", "svg"):
        fig.savefig(output/f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozer", type=Path, default=root/"examples/benchmarks/results/paul_100_rerun/boozer")
    parser.add_argument("--vmec", type=Path, default=root/"examples/benchmarks/results/paul_100_rerun/vmec")
    parser.add_argument("--output", type=Path, default=root/"examples/benchmarks/results/paul_100_rerun/comparison")
    parser.add_argument("--particles", type=int, help="Require this marker count; otherwise infer it from saved results")
    args = parser.parse_args()
    compare(args.boozer, args.vmec, args.output, particles=args.particles)


if __name__ == "__main__":
    main()
