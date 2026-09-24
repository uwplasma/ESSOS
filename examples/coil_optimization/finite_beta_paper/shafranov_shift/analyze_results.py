"""Rebuild compact result tables and figures from the archived scan summaries.

The fits have a fixed zero intercept and include a quadratic pressure term:
``observable(alpha) = alpha * slope + alpha**2 * curvature``. The linear
coefficient is an empirical small-pressure estimate, not an uncertainty bar.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "lines.linewidth": 0.9,
    }
)

ROOT = Path(__file__).parent / "results"
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
RADII_MM = (12, 15, 18)


def _fit(alpha, values):
    design = np.column_stack((alpha, alpha**2))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    return coefficients[0], coefficients[1], residual


def _periodic_interp(query, source, values, period):
    return np.interp(
        query,
        np.r_[source, source[0] + period],
        np.r_[values, values[0]],
    )


def _save_figure(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        output = FIGURES / f"{stem}.{extension}"
        fig.savefig(
            output,
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
        )
        if extension == "svg":
            # Matplotlib emits trailing spaces on some continued path lines.
            # Strip them for clean review diffs; SVG whitespace is insignificant.
            lines = output.read_text(encoding="utf-8").splitlines()
            output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def _read_radius(radius_mm):
    folder = RUNS / f"qa{radius_mm}"
    manifest = json.loads((folder / "manifest.json").read_text())
    points = json.loads((folder / "pressure_points.json").read_text())
    with np.load(folder / "theory.npz") as archive:
        theory = {key: archive[key].copy() for key in archive.files}
    if manifest["status"] != "pressure_scan_complete":
        raise ValueError(f"{folder} is not a complete pressure scan")
    if len(points) != 5 or not all(point["accepted"] for point in points):
        raise ValueError(f"{folder} does not contain five accepted pressure points")
    return manifest, points, theory


def main():
    datasets = {radius: _read_radius(radius) for radius in RADII_MM}
    scalar_rows = []
    profile_rows = []
    residual_rows = []
    summaries = []

    for radius, (manifest, points, theory) in datasets.items():
        alpha = np.asarray([point["alpha"] for point in points])
        phi = np.asarray(points[0]["phi"])
        period = float(phi[-1] + phi[1] - phi[0])
        flux_radius = float(manifest["flux_radius_m"])
        radius_used = float(theory.get("radius_m", flux_radius))
        dR = np.asarray([point["delta_R"] for point in points])
        dZ = np.asarray([point["delta_Z"] for point in points])
        dL = np.asarray([point["delta_length_over_L"] for point in points])
        slope_R, curvature_R, residual_R = _fit(alpha, dR)
        slope_Z, curvature_Z, residual_Z = _fit(alpha, dZ)
        slope_L, curvature_L, residual_L = _fit(alpha, dL)
        theory_R = _periodic_interp(phi, theory["phi"], theory["delta_R"], period)
        theory_Z = _periodic_interp(phi, theory["phi"], theory["delta_Z"], period)
        theory_L = float(manifest["theory"]["length_slope_over_L"])

        for index, point in enumerate(points):
            scalar_rows.append(
                {
                    "radius_m": radius_used,
                    "flux_radius_m": flux_radius,
                    "alpha": alpha[index],
                    "beta_axis": point["beta_axis"],
                    "beta_total": point["beta_total"],
                    "delta_R_phi0_m": dR[index, 0],
                    "delta_Z_phi0_m": dZ[index, 0],
                    "max_abs_delta_R_m": float(np.max(np.abs(dR[index]))),
                    "max_abs_delta_Z_m": float(np.max(np.abs(dZ[index]))),
                    "delta_length_over_L": dL[index],
                    "iota_axis_lab": point["iota_axis_lab"],
                    "reported_toroidal_current_A": point["reported_toroidal_current_A"],
                    "wout_edge_flux_relative_error": point[
                        "wout_edge_flux_relative_error"
                    ],
                    "fsqr": point["fsqr"],
                    "fsqz": point["fsqz"],
                    "fsql": point["fsql"],
                    "interface_tangential_jump_max_over_B": point["interface"][
                        "tangential_jump_max_over_B"
                    ],
                    "interface_pressure_balance_max_rel": point["interface"][
                        "pressure_balance_max_rel"
                    ],
                }
            )
            fitted_R = alpha[index] * slope_R + alpha[index] ** 2 * curvature_R
            fitted_Z = alpha[index] * slope_Z + alpha[index] ** 2 * curvature_Z
            fitted_L = alpha[index] * slope_L + alpha[index] ** 2 * curvature_L
            for j, angle in enumerate(phi):
                profile_rows.append(
                    {
                        "radius_m": radius_used,
                        "flux_radius_m": flux_radius,
                        "phi_rad": angle,
                        "theory_delta_R_per_alpha_m": theory_R[j],
                        "fit_delta_R_per_alpha_m": slope_R[j],
                        "theory_delta_Z_per_alpha_m": theory_Z[j],
                        "fit_delta_Z_per_alpha_m": slope_Z[j],
                    }
                )
            residual_rows.append(
                {
                    "radius_m": radius_used,
                    "alpha": alpha[index],
                    "max_delta_R_fit_residual_m": float(
                        np.max(np.abs(residual_R[index]))
                    ),
                    "max_delta_Z_fit_residual_m": float(
                        np.max(np.abs(residual_Z[index]))
                    ),
                    "delta_length_fit_residual_over_L": float(residual_L[index]),
                    "delta_R_phi0_observed_m": float(dR[index, 0]),
                    "delta_R_phi0_quadratic_fit_m": float(fitted_R[0]),
                    "delta_Z_phi0_observed_m": float(dZ[index, 0]),
                    "delta_Z_phi0_quadratic_fit_m": float(fitted_Z[0]),
                    "delta_length_over_L_observed": float(dL[index]),
                    "delta_length_over_L_quadratic_fit": float(fitted_L),
                }
            )

        response_rms_R = float(np.sqrt(np.mean(slope_R**2)))
        theory_rms_R = float(np.sqrt(np.mean(theory_R**2)))
        response_rms_Z = float(np.sqrt(np.mean(slope_Z**2)))
        theory_rms_Z = float(np.sqrt(np.mean(theory_Z**2)))
        summaries.append(
            {
                "nominal_radius_m": radius * 1e-3,
                "flux_radius_m": flux_radius,
                "pressure_points": len(points),
                "all_accepted": all(point["accepted"] for point in points),
                "vacuum_iota_lab": manifest["vacuum"]["iota_axis_lab"],
                "direct_floquet_iota_lab": manifest["vacuum"]["traced_iota_lab"],
                "vacuum_iota_relative_gap": manifest["vacuum"][
                    "iota_relative_gap"
                ],
                "alpha_max": manifest["pressure_family"]["alpha_max"],
                "beta_axis_max": points[-1]["beta_axis"],
                "beta_total_max": points[-1]["beta_total"],
                "max_abs_current_A": max(
                    float(np.max(np.abs(point["current_profile"])))
                    for point in points
                ),
                "max_abs_edge_flux_relative_error": max(
                    abs(float(point["wout_edge_flux_relative_error"]))
                    for point in points
                ),
                "max_interface_tangential_jump_over_B": max(
                    float(point["interface"]["tangential_jump_max_over_B"])
                    for point in points
                ),
                "delta_R_phi0_fit_slope_m_per_alpha": float(slope_R[0]),
                "delta_R_phi0_theory_slope_m_per_alpha": float(theory_R[0]),
                "delta_R_phi0_fit_to_theory_ratio": float(
                    slope_R[0] / theory_R[0]
                ),
                "delta_R_profile_fit_rms_to_theory_rms_ratio": (
                    response_rms_R / theory_rms_R
                ),
                "delta_Z_profile_fit_rms_to_theory_rms_ratio": (
                    response_rms_Z / theory_rms_Z
                ),
                "axis_length_fit_slope_over_L": float(slope_L),
                "axis_length_theory_slope_over_L": theory_L,
                "axis_length_fit_to_theory_ratio": float(slope_L / theory_L),
                "axis_length_quadratic_coefficient": float(curvature_L),
                "maximum_abs_symmetry_plane_delta_Z_m": float(
                    np.max(np.abs(dZ[:, 0]))
                ),
                "uncertainty_status": "not_estimated; one primary resolution",
            }
        )

    ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (ROOT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for name, rows in (
        ("pressure_points.csv", scalar_rows),
        ("response_profiles.csv", profile_rows),
        ("fit_residuals.csv", residual_rows),
    ):
        with (ROOT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    resolution_rows = []
    resolution_sources = [(65, RUNS / "qa18")]
    resolution_sources.extend(
        (int(folder.name.removeprefix("qa18_ns")), folder)
        for folder in sorted(RUNS.glob("qa18_ns*"))
        if folder.is_dir()
    )
    for ns, folder in resolution_sources:
        if not (folder / "manifest.json").exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        points = json.loads((folder / "pressure_points.json").read_text())
        if manifest["status"] != "pressure_scan_complete" or not points:
            continue
        point = points[-1]
        alpha = float(point["alpha"])
        dR_slope = np.asarray(point["delta_R"], dtype=float) / alpha
        dZ_slope = np.asarray(point["delta_Z"], dtype=float) / alpha
        resolution_rows.append(
            {
                "ns": ns,
                "alpha": alpha,
                "vacuum_iota_lab": manifest["vacuum"]["iota_axis_lab"],
                "vacuum_iota_relative_gap": manifest["vacuum"][
                    "iota_relative_gap"
                ],
                "delta_R_phi0_per_alpha_m": float(dR_slope[0]),
                "delta_R_profile_rms_per_alpha_m": float(
                    np.sqrt(np.mean(dR_slope**2))
                ),
                "delta_Z_profile_rms_per_alpha_m": float(
                    np.sqrt(np.mean(dZ_slope**2))
                ),
                "delta_length_over_L_per_alpha": float(
                    point["delta_length_over_L"] / alpha
                ),
                "max_abs_current_A": float(
                    np.max(np.abs(point["current_profile"]))
                ),
                "wout_sha256": point["wout_sha256"],
                "input_sha256": point["input_sha256"],
            }
        )
    if resolution_rows:
        (ROOT / "resolution_check.json").write_text(
            json.dumps(resolution_rows, indent=2) + "\n"
        )
        with (ROOT / "resolution_check.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(resolution_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(resolution_rows)

    quadrature_path = ROOT / "quadrature_check.json"
    quadrature_rows = (
        json.loads(quadrature_path.read_text()) if quadrature_path.exists() else []
    )
    if quadrature_rows:
        with (ROOT / "quadrature_check.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(quadrature_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(quadrature_rows)
        spread = {}
        for field in (
            "target_field_rms_over_B0",
            "target_gradient_rms_R0_over_B0",
            "floquet_iota_lab",
            "return_map_determinant",
            "effective_flux_radius_m",
        ):
            values = np.asarray([row[field] for row in quadrature_rows], dtype=float)
            spread[field] = {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "absolute_range": float(np.ptp(values)),
                "relative_range": float(
                    np.ptp(values) / max(abs(float(np.mean(values))), 1e-30)
                ),
            }
        (ROOT / "quadrature_spread.json").write_text(
            json.dumps(spread, indent=2) + "\n"
        )

    colors = {12: "#0072B2", 15: "#D55E00", 18: "#009E73"}
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    for radius, (manifest, points, theory) in datasets.items():
        alpha = np.asarray([point["alpha"] for point in points])
        a = float(manifest["flux_radius_m"])
        ax.plot(
            alpha,
            [point["delta_R"][0] / a for point in points],
            "o",
            color=colors[radius],
            ms=3.4,
            label=f"VMEX, {radius} mm",
        )
        ax.plot(
            [0.0, alpha[-1]],
            [0.0, alpha[-1] * float(theory["delta_R"][0]) / a],
            color=colors[radius],
            ls="--",
            lw=1,
            label=f"first order, {radius} mm",
        )
    ax.axhline(0, color="0.25", lw=0.6)
    ax.set_xlabel(r"pressure multiplier $\alpha$")
    ax.set_ylabel(r"$\delta R(0)/a_{\rm flux}$")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        fontsize=5.7,
    )
    _save_figure(fig, "symmetry_plane_shift")

    for component, values, theory_key, ylabel in (
        ("R", "delta_R", "delta_R", r"$\delta R/\alpha$ (m)"),
        ("Z", "delta_Z", "delta_Z", r"$\delta Z/\alpha$ (m)"),
    ):
        fig, ax = plt.subplots(figsize=(3.45, 3.0))
        for radius, (manifest, points, theory) in datasets.items():
            alpha = np.asarray([point["alpha"] for point in points])
            profiles = np.asarray([point[values] for point in points])
            slope, _, _ = _fit(alpha, profiles)
            phi = np.asarray(points[0]["phi"])
            period = float(phi[-1] + phi[1] - phi[0])
            predicted = _periodic_interp(
                phi, theory["phi"], theory[theory_key], period
            )
            ax.plot(
                phi,
                predicted,
                color=colors[radius],
                lw=1,
                label=f"theory, {radius} mm",
            )
            ax.plot(
                phi,
                slope,
                color=colors[radius],
                ls="--",
                lw=1,
                marker="o",
                markevery=16,
                ms=2.4,
                label=f"VMEX fit, {radius} mm",
            )
        ax.set_xlabel(r"geometrical toroidal angle $\phi$ (rad)")
        ax.set_ylabel(ylabel)
        ax.legend(
            frameon=False,
            ncol=3,
            fontsize=5.7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
        )
        _save_figure(fig, f"axis_{component}_profile")

    fig, ax = plt.subplots(figsize=(3.45, 3.2))
    for radius, (manifest, points, _) in datasets.items():
        alpha = np.asarray([point["alpha"] for point in points])
        lengths = np.asarray([point["delta_length_over_L"] for point in points])
        ax.plot(
            alpha,
            lengths,
            "o",
            ms=3.4,
            color=colors[radius],
            label=f"VMEX, {radius} mm",
        )
        pred = float(manifest["theory"]["length_slope_over_L"])
        ax.plot(
            [0.0, alpha[-1]],
            [0.0, alpha[-1] * pred],
            color=colors[radius],
            lw=1,
            ls=":",
            label=f"first order, {radius} mm",
        )
    ax.axhline(0, color="0.25", lw=0.6)
    ax.set_xlabel(r"pressure multiplier $\alpha$")
    ax.set_ylabel(r"$(L_\alpha-L_0)/L_0$")
    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=6.3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
    )
    _save_figure(fig, "axis_length_change")

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.1), sharex=False)
    for radius, (manifest, points, _) in datasets.items():
        alpha = np.asarray([point["alpha"] for point in points])
        dr0_per_alpha = np.asarray(
            [point["delta_R"][0] for point in points]
        ) / alpha
        dL_per_alpha = np.asarray(
            [point["delta_length_over_L"] for point in points]
        ) / alpha
        axes[0].plot(
            alpha,
            dr0_per_alpha * 1e3,
            "o-",
            ms=3,
            lw=0.9,
            color=colors[radius],
            label=f"{radius} mm",
        )
        axes[1].plot(
            alpha,
            dL_per_alpha * 1e6,
            "o-",
            ms=3,
            lw=0.9,
            color=colors[radius],
            label=f"{radius} mm",
        )
    axes[0].set_ylabel(r"$\delta R(0)/\alpha$ (mm)")
    axes[1].set_ylabel(r"$10^6(L_\alpha-L_0)/(\alpha L_0)$")
    axes[1].set_xlabel(r"pressure multiplier $\alpha$")
    for ax in axes:
        ax.axhline(0, color="0.25", lw=0.6)
    axes[0].legend(
        frameon=False,
        fontsize=6,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
    )
    fig.tight_layout()
    _save_figure(fig, "pressure_step_response")

    radii = np.asarray([row["flux_radius_m"] * 1e3 for row in summaries])
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.plot(
        radii,
        [row["delta_R_phi0_fit_to_theory_ratio"] for row in summaries],
        "o-",
        lw=1,
        ms=3.5,
        label=r"$\delta R(0)$ slope ratio",
    )
    ax.plot(
        radii,
        [row["axis_length_fit_to_theory_ratio"] for row in summaries],
        "s--",
        lw=1,
        ms=3.5,
        label=r"axis-length slope ratio",
    )
    ax.axhline(1, color="0.25", lw=0.8, ls=":", label="agreement")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"measured vacuum flux radius $a_{\rm flux}$ (mm)")
    ax.set_ylabel("fitted VMEX slope / first-order prediction")
    ax.legend(
        frameon=False,
        fontsize=6,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
    )
    _save_figure(fig, "radius_response_ratio")

    if len(resolution_rows) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(3.45, 5.0), sharex=True)
        ns = [row["ns"] for row in resolution_rows]
        axes[0].plot(
            ns,
            [row["delta_R_profile_rms_per_alpha_m"] * 1e6 for row in resolution_rows],
            "o-",
            color="#0072B2",
            ms=3.5,
            lw=1,
        )
        axes[1].plot(
            ns,
            [row["delta_length_over_L_per_alpha"] * 1e6 for row in resolution_rows],
            "s--",
            color="#D55E00",
            ms=3.5,
            lw=1,
        )
        axes[1].axhline(0, color="0.25", lw=0.6)
        axes[2].plot(
            ns,
            [row["vacuum_iota_relative_gap"] * 100 for row in resolution_rows],
            "o-",
            color="#009E73",
            ms=3.5,
            lw=1,
        )
        axes[0].set_ylabel(r"R-profile RMS / $\alpha$ ($\mu$m)")
        axes[1].set_ylabel(r"$10^6\Delta L/(\alpha L)$")
        axes[2].set_xlabel("radial surfaces at highest resolution")
        axes[2].set_ylabel("vacuum transform gap (%)")
        fig.tight_layout()
        _save_figure(fig, "radial_resolution")

    errors = [
        {
            "contribution": "coil-field quadrature",
            "estimate": "bounded by 240/480/960 segment spread; see quadrature_check.csv",
            "evidence": "direct target-field, gradient, return-map transform, seed flux, and traced flux are invariant across the checked segment counts",
        },
        {
            "contribution": "pressure-step extrapolation",
            "estimate": "not isolated",
            "evidence": "five positive levels fit to fixed-zero linear + quadratic; no extra smaller levels",
        },
        {
            "contribution": "radial/Fourier/toroidal VMEX discretization",
            "estimate": "not converged; see resolution_check.csv",
            "evidence": "NS=65/129 changes the selected pressure response and flips the axis-length sign; NS=257 did not reach its first vacuum iteration within 15 min",
        },
        {
            "contribution": "vacuum interface / exterior response",
            "estimate": "diagnostic reported; no pressure-signal error bound",
            "evidence": "interface tangential and pressure-balance metrics are in pressure_points.csv",
        },
        {
            "contribution": "coil-reference/model mismatch",
            "estimate": "unresolved",
            "evidence": "the measured slopes disagree in sign or by large factors with the ideal first-order response",
        },
        {
            "contribution": "finite-radius truncation",
            "estimate": "unresolved",
            "evidence": "the three-radius trend is not a small correction to the first-order prediction",
        },
    ]
    with (ROOT / "error_budget.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(errors[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(errors)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
