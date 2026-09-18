"""Plots for the Fourier-coefficient winding-surface comparison, styled to
match winding_surface_comparison_8_coils.py's own validate_surfaces() /
sheet_resolution_study() figures.

Reads the combined CSVs winding_surface_fourier_comparison.py produces
(examples/output/winding_surface_fourier_comparison/*_combined.csv) and
writes PNGs to examples/winding_surface_fourier_comparison/figures/.

This script only reads CSVs; it does not rerun any optimization or solve.
Run after winding_surface_fourier_comparison.py has completed.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "output", "winding_surface_fourier_comparison")
FIGURES_DIR = os.path.join(ROOT, "winding_surface_fourier_comparison", "figures")

CASES = ("Landreman-Paul QA", "Landreman-Paul QH", "W7-X")

# Methods shown in the comparison, in display order. "normal offset",
# "ESSOS entropy" and "ESSOS Fourier entropy" are freshly rerun on this
# machine this session (see winding_surface_fourier_comparison.py);
# "ESSOS Pareto" and "REGCOIL adjoint" are reused from the tracked,
# cross-machine PR data (rerunning them needs the ESSOS 32x32 Pareto solve
# and the legacy REGCOIL adjoint Fortran binary respectively, out of scope
# here) -- flagged with a lighter color/hatch since W7-X's tracked baseline
# was found to be stale for some other methods (see the PR comment/commit
# message); these two's cross-machine W7-X numbers have not been
# independently re-verified on this machine.
DISPLAY_METHODS = (
    ("normal offset (this machine)", "normal offset", "#777777", False),
    ("ESSOS entropy (this machine)", "ESSOS entropy (dipole)", "#56B4E9", False),
    ("ESSOS Fourier entropy", "ESSOS Fourier entropy (new)", "#D55E00", False),
    ("ESSOS Pareto", "ESSOS Pareto (tracked, cross-machine)", "#E69F00", True),
    ("REGCOIL adjoint", "REGCOIL adjoint (tracked, cross-machine)", "#0072B2", True),
)


def read_csv(name):
    path = os.path.join(DATA_DIR, name)
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def find_row(rows, case, surface_method, **extra):
    for row in rows:
        if row["configuration"] != case or row["surface_method"] != surface_method:
            continue
        if all(row.get(key) == str(value) for key, value in extra.items()):
            return row
    return None


def bar_plot(rows, filename, title_prefix):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    metrics = (("sheet_f_B_T2_m2", r"Resolved sheet $f_B$ [$T^2m^2$]", True),
               ("sheet_max_abs_Bn_over_B", r"Resolved max $|B_n|/B$", True),
               ("filament_f_B_T2_m2", r"Filament $f_B$ [$T^2m^2$]", True),
               ("filament_max_abs_Bn_over_B", r"Filament max $|B_n|/B$", True),
               ("achieved_or_minimum_Kmax_A_per_m",
                r"Achieved/minimum $K_{max}$ [MA/m]", False),
               ("surface_runtime_s", "Surface optimization runtime [s]", True))
    x = np.arange(len(CASES))
    width = 0.15
    center = (len(DISPLAY_METHODS) - 1) / 2
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for axis, (metric, title, logarithmic) in zip(axes.flat, metrics):
        for index, (surface_method, label, color, hatched) in enumerate(DISPLAY_METHODS):
            values = []
            feasible = []
            for case in CASES:
                row = find_row(rows, case, surface_method)
                value = float(row[metric]) if row and row.get(metric) not in (None, "", "nan") else np.nan
                if "Kmax" in metric and not np.isnan(value):
                    value /= 1e6
                values.append(value)
                feasible.append(row is None or row.get("feasible_at_Kmax", "True") == "True")
            positions = x + (index - center) * width
            axis.bar(positions, values, width, label=label, color=color,
                     hatch="//" if hatched else None,
                     edgecolor="black" if hatched else None, linewidth=0.4)
            for position, ok in zip(positions, feasible):
                if not ok:
                    axis.text(position, 0.03, "infeasible", rotation=90,
                             color="#AA0000", ha="center", va="bottom",
                             transform=axis.get_xaxis_transform(), fontsize=7)
        axis.set_xticks(x, CASES)
        axis.set_title(title)
        if logarithmic:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=1)
    figure.suptitle(title_prefix)
    figure.savefig(os.path.join(FIGURES_DIR, filename), dpi=180)
    plt.close(figure)


def resolution_convergence_plot(rows, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    figure, axes = plt.subplots(1, len(CASES), figsize=(15, 4), constrained_layout=True,
                                sharey=False)
    for axis, case in zip(axes, CASES):
        for surface_method, label, color, hatched in DISPLAY_METHODS:
            points = [(int(row["resolution"]), float(row["sheet_f_B_T2_m2"]))
                     for row in rows
                     if row["configuration"] == case
                     and row["surface_method"] == surface_method
                     and row.get("method", "REGCOIL") == "REGCOIL"]
            points.sort()
            if not points:
                continue
            resolutions, values = zip(*points)
            axis.plot(resolutions, values, "o--" if hatched else "o-",
                     color=color, label=label)
        axis.set_title(case)
        axis.set_xlabel("points per angle per field period")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"Sheet $f_B$ [$T^2m^2$]")
    axes[0].legend(fontsize=7)
    figure.suptitle("Resolution convergence: 48/56/64 (W7-X normal offset missing @64 -- "
                    "see README: watchdog aborted at the edge of available memory, "
                    "consistent across repeated retries)")
    figure.savefig(os.path.join(FIGURES_DIR, filename), dpi=180)
    plt.close(figure)


def cost_plot(rows_48, filename):
    """Optimization runtime and iteration count per method/case -- new plot,
    not present in the existing PR figures, since the Fourier method's cost
    profile relative to the dipole method is a key part of this comparison."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    x = np.arange(len(CASES))
    width = 0.15
    center = (len(DISPLAY_METHODS) - 1) / 2
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for index, (surface_method, label, color, hatched) in enumerate(DISPLAY_METHODS):
        runtimes = []
        iterations = []
        for case in CASES:
            row = find_row(rows_48, case, surface_method)
            runtimes.append(float(row["surface_runtime_s"]) if row else np.nan)
            iterations.append(float(row["surface_iterations"]) if row else np.nan)
        positions = x + (index - center) * width
        axes[0].bar(positions, runtimes, width, label=label, color=color,
                    hatch="//" if hatched else None,
                    edgecolor="black" if hatched else None, linewidth=0.4)
        axes[1].bar(positions, iterations, width, color=color,
                    hatch="//" if hatched else None,
                    edgecolor="black" if hatched else None, linewidth=0.4)
    axes[0].set_xticks(x, CASES)
    axes[0].set_title("Surface optimization runtime [s]")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_xticks(x, CASES)
    axes[1].set_title("L-BFGS-B iterations")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(os.path.join(FIGURES_DIR, filename), dpi=180)
    plt.close(figure)


def main():
    validation_96 = read_csv("surface_validation_96_combined.csv")
    comparison_48 = read_csv("comparison_metrics_combined.csv")
    resolution_convergence = read_csv("sheet_resolution_convergence_combined.csv")

    bar_plot(validation_96, "fourier_comparison_validation_96.png",
            "Resolved 96x96 REGCOIL validation: Fourier vs. dipole entropy vs. established baselines")
    resolution_convergence_plot(
        resolution_convergence, "fourier_comparison_resolution_convergence.png")
    cost_plot(comparison_48, "fourier_comparison_optimization_cost.png")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
