"""v1 (original) vs v2 (periodicity + matmul-kernel optimized) Fourier
induction-matrix comparison: wall-time and fidelity, side by side.

Reads v1 data from examples/output/winding_surface_fourier_comparison_v1_snapshot
(a full snapshot taken before the optimization was applied) and v2 data from
examples/output/winding_surface_fourier_comparison (the current, optimized run).
Writes figures to examples/winding_surface_fourier_comparison/figures/, alongside
the standard vs-other-methods figures winding_surface_fourier_comparison_plots.py
produces.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(ROOT, "output", "winding_surface_fourier_comparison_v1_snapshot")
V2_DIR = os.path.join(ROOT, "output", "winding_surface_fourier_comparison")
FIGURES_DIR = os.path.join(ROOT, "winding_surface_fourier_comparison", "figures")

CASES = ("Landreman-Paul QA", "Landreman-Paul QH", "W7-X")


def read_csv(directory, name):
    with open(os.path.join(directory, name), newline="") as stream:
        return list(csv.DictReader(stream))


def find_row(rows, case, surface_method):
    for row in rows:
        if row["configuration"] == case and row["surface_method"] == surface_method:
            return row
    return None


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    v1_48 = read_csv(V1_DIR, "comparison_metrics_combined.csv")
    v2_48 = read_csv(V2_DIR, "comparison_metrics_combined.csv")
    v1_96 = read_csv(V1_DIR, "surface_validation_96_combined.csv")
    v2_96 = read_csv(V2_DIR, "surface_validation_96_combined.csv")

    runtimes_v1, runtimes_v2, iters_v1, iters_v2 = [], [], [], []
    fb48_v1, fb48_v2, fb96_v1, fb96_v2 = [], [], [], []
    for case in CASES:
        r1 = find_row(v1_48, case, "ESSOS Fourier entropy")
        r2 = find_row(v2_48, case, "ESSOS Fourier entropy")
        runtimes_v1.append(float(r1["surface_runtime_s"]))
        runtimes_v2.append(float(r2["surface_runtime_s"]))
        iters_v1.append(int(r1["surface_iterations"]))
        iters_v2.append(int(r2["surface_iterations"]))
        fb48_v1.append(float(r1["sheet_f_B_T2_m2"]))
        fb48_v2.append(float(r2["sheet_f_B_T2_m2"]))
        n1 = find_row(v1_96, case, "ESSOS Fourier entropy")
        n2 = find_row(v2_96, case, "ESSOS Fourier entropy")
        fb96_v1.append(float(n1["sheet_f_B_T2_m2"]))
        fb96_v2.append(float(n2["sheet_f_B_T2_m2"]))

    speedups = [a / b for a, b in zip(runtimes_v1, runtimes_v2)]

    # --- Figure 1: wall-time / speedup ---
    x = np.arange(len(CASES))
    width = 0.35
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axis = axes[0]
    axis.bar(x - width / 2, runtimes_v1, width, label="v1 (original)", color="#D55E00")
    axis.bar(x + width / 2, runtimes_v2, width, label="v2 (periodicity + matmul kernel)",
             color="#0072B2")
    for xi, (r1, r2, s, i1, i2) in enumerate(zip(runtimes_v1, runtimes_v2, speedups,
                                                  iters_v1, iters_v2)):
        assert i1 == i2, "iteration count changed -- optimizer path differs!"
        axis.text(xi, max(r1, r2) * 1.03, f"{s:.2f}x", ha="center", fontsize=10,
                  fontweight="bold")
    axis.set_xticks(x, CASES)
    axis.set_ylabel("Surface optimization runtime [s]")
    axis.set_title("Wall-time: same iteration count both versions\n"
                   "(optimizer took the identical path)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1]
    axis.bar(x, speedups, 0.5, color="#009E73")
    for xi, s in enumerate(speedups):
        axis.text(xi, s * 1.02, f"{s:.2f}x", ha="center", fontsize=10, fontweight="bold")
    axis.set_xticks(x, CASES)
    axis.set_ylabel("Speedup (v1 runtime / v2 runtime)")
    axis.set_title("v2 speedup over v1")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(os.path.join(FIGURES_DIR, "fourier_v1_vs_v2_speedup.png"), dpi=180)
    plt.close(figure)

    # --- Figure 2: fidelity (sheet fB should be ~identical) ---
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, (fb_v1, fb_v2, title) in zip(
            axes, ((fb48_v1, fb48_v2, "48x48 sheet $f_B$"),
                  (fb96_v1, fb96_v2, "96x96 (validation) sheet $f_B$"))):
        relative_diff = [abs(a - b) / abs(a) for a, b in zip(fb_v1, fb_v2)]
        axis.bar(x - width / 2, fb_v1, width, label="v1", color="#D55E00")
        axis.bar(x + width / 2, fb_v2, width, label="v2", color="#0072B2")
        for xi, rd in enumerate(relative_diff):
            axis.text(xi, max(fb_v1[xi], fb_v2[xi]) * 1.03, f"Δ={rd:.1e}",
                      ha="center", fontsize=8)
        axis.set_xticks(x, CASES)
        axis.set_yscale("log")
        axis.set_title(f"{title} [$T^2m^2$]\n(v1 vs v2 relative difference annotated)")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Fidelity check: v2 reproduces v1's physical results "
                    "(same resolution, algorithm only)")
    figure.savefig(os.path.join(FIGURES_DIR, "fourier_v1_vs_v2_fidelity.png"), dpi=180)
    plt.close(figure)

    print(f"Saved v1-vs-v2 figures to {FIGURES_DIR}")
    print("\nSummary:")
    for case, r1, r2, s in zip(CASES, runtimes_v1, runtimes_v2, speedups):
        print(f"  {case}: {r1:.2f}s -> {r2:.2f}s ({s:.2f}x speedup)")
    for case, a48, b48, a96, b96 in zip(CASES, fb48_v1, fb48_v2, fb96_v1, fb96_v2):
        print(f"  {case}: 48x48 fB {a48:.6e} vs {b48:.6e} (Δrel={abs(a48-b48)/a48:.2e})  "
             f"96x96 fB {a96:.6e} vs {b96:.6e} (Δrel={abs(a96-b96)/a96:.2e})")


if __name__ == "__main__":
    main()
