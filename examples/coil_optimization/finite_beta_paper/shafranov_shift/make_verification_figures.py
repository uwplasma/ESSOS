"""Paper figures for the fixed-coil pressure-response verification.

Reads only the compact JSON files in ``results/verification`` and writes vector
PDFs plus compressed PNG previews to ``results/verification/figures``.

    python make_verification_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "verification"
OUT = DATA / "figures"
# Fixed categorical order (validated default palette); each series keeps its slot.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdad4"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.markersize": 4.5, "legend.frameon": False, "savefig.dpi": 200,
    "pdf.fonttype": 42,
})


def load(name):
    return json.loads((DATA / name).read_text())


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def panel_label(ax, text):
    ax.text(-0.16, 1.04, text, transform=ax.transAxes, fontweight="bold", color=INK)


def figure_source_operator():
    source = load("source_field_check.json")["near_axis_source"]
    operator = load("axis_operator_check.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    nphi = np.array([row["nphi"] for row in source["ladder"]])
    error = np.array([row["max_relative_difference"] for row in source["ladder"]])
    a.loglog(nphi, error, "o-", color=BLUE, label="Biot–Savart of near-axis current")
    a.loglog(nphi, error[-1] * (nphi / nphi[-1]) ** -2, "--", color=MUTED, lw=1,
             label=r"reference slope $n_\varphi^{-2}$")
    a.axhline(source["richardson_relative_difference"], color=ORANGE, lw=1.2)
    a.text(nphi[0], source["richardson_relative_difference"] * 1.4,
           f"Richardson: {source['richardson_relative_difference']:.1e}", color=INK, fontsize=7)
    a.set_xlabel(r"toroidal source samples per period $n_\varphi$")
    a.set_ylabel("relative difference from\nclosed-form on-axis field")
    a.legend(loc="upper right")
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xticks(nphi, [str(n) for n in nphi])
    panel_label(a, "(a)")

    eps = np.array(sorted(float(k) for k in operator["nonlinear_vs_linear_first_order_rel"]))
    rel = np.array([operator["nonlinear_vs_linear_first_order_rel"][repr(e)]
                    if repr(e) in operator["nonlinear_vs_linear_first_order_rel"]
                    else operator["nonlinear_vs_linear_first_order_rel"][f"{e:g}"] for e in eps])
    b.loglog(eps, rel, "o-", color=BLUE, label="nonlinear traced / exact linear")
    b.axhline(operator["richardson_vs_linear_rel"], color=AQUA, lw=1.2,
              label=f"Richardson ({operator['richardson_vs_linear_rel']:.1e})")
    b.axhline(operator["frenet_actual_gradient_vs_traced_rel"], color=ORANGE, lw=1.2,
              label="Frenet near-axis operator (%.2f%%)"
              % (100 * operator["frenet_actual_gradient_vs_traced_rel"]))
    b.set_xlabel(r"vertical test field $\epsilon B_0$ added to the coils")
    b.set_ylabel("relative axis-response difference")
    b.set_ylim(1e-9, 3e-2)
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xticks(eps, [f"{e:.0e}" for e in eps])
    b.legend(loc="center right")
    panel_label(b, "(b)")
    fig.tight_layout()
    save(fig, "source_and_operator_verification")


def vmec2000_series(runs, prefix, theory):
    cases = {c["case"]: c for c in runs["vmec2000"] if c["case"].startswith(prefix)}
    vacuum = next((c for c in cases.values() if c["pres_scale_Pa"] == 0 and c.get("ier_flag") == 0), None)
    if vacuum is None:
        return np.array([]), np.array([])
    p0 = theory["p0_star_Pa"] * theory["alpha_max"]
    x, y = [], []
    for c in cases.values():
        if c.get("ier_flag") != 0 or c["pres_scale_Pa"] == 0:
            continue
        x.append(c["pres_scale_Pa"] / p0)
        y.append(c["R_axis_phi0_m"] - vacuum["R_axis_phi0_m"])
    order = np.argsort(x)
    return np.array(x)[order], np.array(y)[order]


def vmex_series(runs, name, theory):
    run = next((r for r in runs["vmex"] if r["run"] == name), None)
    if run is None:
        return np.array([]), np.array([])
    pts = [p for p in run["points"] if p["accepted"]]
    return (np.array([p["alpha"] for p in pts]) / theory["alpha_max"],
            np.array([p["delta_R0_m"] for p in pts]))


def figure_pressure_response():
    runs = load("pressure_runs.json")
    theory = runs["theory"]
    slope = theory["delta_R0_per_alpha_m"] * theory["alpha_max"]  # metres per alpha_max
    trace = load("finite_amplitude_trace.json")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.6),
                                  gridspec_kw={"width_ratios": [1.3, 1, 1]})

    xs = np.linspace(0, 12.5, 50)
    a.plot(xs, 1e3 * slope * xs, color=INK, lw=1.0, label="first-order theory (untuned)")
    kept = [p for p in trace["points"] if p["multiple"] <= 8]
    tx = np.array([p["multiple"] for p in kept])
    ty = np.array([p["delta_R0_m"] for p in kept])
    a.plot(tx, 1e3 * ty, "--", color=MUTED, lw=1.1, label="coils + frozen near-axis current, traced")
    series = [
        ("VMEC2000 free, NS 65", vmec2000_series(runs, "v2k_ns/ns65_", theory), BLUE, "o"),
        ("VMEC2000 free, NS 129", vmec2000_series(runs, "v2k_ns/ns129_", theory), ORANGE, "s"),
        ("VMEX free, NS 65, cold", tuple(np.r_[u, v] for u, v in zip(
            vmex_series(runs, "qa18_cold65", theory),
            vmex_series(runs, "qa18_cold65_high", theory))), AQUA, "^"),
    ]
    for label, (x, y), color, marker in series:
        if len(x):
            a.plot(x, 1e3 * y, marker=marker, color=color, label=label)
    a.axvspan(0, 1, color=GRID, alpha=0.6, lw=0)
    a.text(0.08, 0.55, "linearity\nscreen", transform=a.get_xaxis_transform(), va="top",
           fontsize=6.5, color=MUTED)
    a.set_xlabel(r"pressure amplitude $\alpha/\alpha_{\max}$")
    a.set_ylabel(r"axis shift $\delta R(\phi=0)$ (mm)")
    a.set_xlim(0, 12.5)
    a.set_ylim(-0.3, 10.5)
    a.legend(loc="upper left", bbox_to_anchor=(0.1, 1.02), fontsize=6.0)
    panel_label(a, "(a)")

    # Response at the screen amplitude against resolution and restart policy.
    groups = []
    for ns in (33, 65, 129, 257):
        prefix = "v2k_smoke/frac_" if ns == 33 else f"v2k_ns/ns{ns}_"
        x, y = vmec2000_series(runs, prefix, theory)
        hit = np.isclose(x, 1.0)
        if hit.any():
            groups.append(("VMEC2000 cold", ns, float(y[hit][0] / slope)))
    for name, ns, policy in (("qa18", 65, "VMEX warm"), ("qa18_ns129", 129, "VMEX warm"),
                             ("qa18_cold65", 65, "VMEX cold"), ("qa18_cold129", 129, "VMEX cold")):
        x, y = vmex_series(runs, name, theory)
        if len(x):
            k = int(np.argmax(x))
            groups.append((policy, ns, float(y[k] / (slope * x[k]))))
    colors = {"VMEC2000 cold": BLUE, "VMEX warm": MAGENTA, "VMEX cold": AQUA}
    markers = {"VMEC2000 cold": "o", "VMEX warm": "v", "VMEX cold": "^"}
    for policy in colors:
        pts = [(ns, r) for p, ns, r in groups if p == policy]
        if pts:
            ns, r = zip(*pts)
            b.semilogx(ns, r, marker=markers[policy], color=colors[policy], label=policy, base=2)
    b.axhline(1, color=INK, lw=1.0)
    b.axhline(0, color=MUTED, lw=0.6)
    b.set_xlabel("radial surfaces NS")
    b.set_ylabel(r"$\delta R(0)/\delta R_{\rm theory}(0)$ at $\alpha\leq\alpha_{\max}$")
    b.set_xticks([33, 65, 129, 257], ["33", "65", "129", "257"])
    b.set_ylim(-0.2, 1.1)
    b.legend(loc="center right")
    panel_label(b, "(b)")

    # Vacuum-axis error against the directly traced coil axis.
    traced = theory["traced_vacuum_R_axis_phi0_m"]
    ladder = [c for c in runs["vmec2000"] if c["case"].startswith("v2k_ns/")
              and c["case"].endswith("f0.000000") and c.get("ier_flag") == 0]
    ladder += [c for c in runs["vmec2000"] if c["case"] == "v2k_smoke/frac_0.000000"]
    ladder.sort(key=lambda c: c["ns"])
    c.semilogx([k["ns"] for k in ladder], [1e3 * (k["R_axis_phi0_m"] - traced) for k in ladder],
               "o-", color=BLUE, base=2, label="VMEC2000 free")
    iters = [k for k in runs["vmec2000"] if k["case"].startswith("v2k_iter/")
             and k["case"].endswith("f0.000000") and k.get("ier_flag") == 0]
    c.semilogx([17] * len(iters), [1e3 * (k["R_axis_phi0_m"] - traced) for k in iters], "o",
               mfc="none", color=BLUE, base=2, label="NS 17, NITER 1k–64k")
    vmex = [(r["settings"]["ns"][-1], r["vacuum_R_axis_phi0_m"]) for r in runs["vmex"]
            if r["vacuum_R_axis_phi0_m"] is not None]
    ns_v, r_v = zip(*sorted(vmex))
    c.semilogx(ns_v, 1e3 * (np.array(r_v) - traced), "^", color=AQUA, base=2, label="VMEX free")
    fixed = next(k for k in runs["vmec2000"] if k["case"] == "v2k_fixed/f0.000000")
    c.semilogx([fixed["ns"]], [1e3 * (fixed["R_axis_phi0_m"] - traced)], "D", color=YELLOW,
               base=2, label="VMEC2000 fixed boundary")
    c.axhline(1e3 * slope, color=INK, lw=1.0, ls="--")
    c.text(17, 1e3 * slope * 1.06, r"theory shift at $\alpha_{\max}$", fontsize=6.5, color=INK,
           va="bottom")
    c.axhline(0, color=MUTED, lw=0.6)
    c.set_xticks([17, 33, 65, 129, 257], ["17", "33", "65", "129", "257"])
    c.set_xlabel("radial surfaces NS")
    c.set_ylabel(r"vacuum $R_{\rm axis}-R_{\rm traced}$ at $\phi=0$ (mm)")
    c.set_ylim(-0.45, 0.85)
    c.legend(loc="lower right", fontsize=5.6)
    panel_label(c, "(c)")
    fig.tight_layout()
    save(fig, "fixed_coil_pressure_response")


def figure_equilibrium_current():
    check = load("source_field_check.json")
    near_axis = np.asarray(check["near_axis_source"]["richardson_T"])[0]
    pairs = [p for p in check["pairs"] if p["name"].endswith("_a1")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    labels = ["near-axis\n(closed form)"]
    values, spreads = [near_axis[2]], [0.0]
    pretty = {"vmec2000_free_ns33_a1": "VMEC2000\nfree NS33", "vmec2000_free_ns65_a1": "VMEC2000\nfree NS65",
              "vmec2000_fixed_ns33_a1": "VMEC2000\nfixed NS33", "vmex_free_ns65_warm_a1": "VMEX\nfree NS65"}
    for p in pairs:
        z = [np.asarray(f["delta_B_per_alpha_T"])[0, 2] for f in p["equilibrium_current_field"]]
        labels.append(pretty.get(p["name"], p["name"]))
        values.append(z[-1])
        spreads.append(abs(z[-1] - z[-2]))
    x = np.arange(len(values))
    a.bar(x, 1e4 * np.array(values), color=[ORANGE] + [BLUE] * (len(values) - 1), width=0.6,
          yerr=1e4 * np.array(spreads), ecolor=MUTED, capsize=2)
    a.axhline(0, color=MUTED, lw=0.6)
    a.set_xticks(x, labels, fontsize=6.3)
    a.set_ylabel(r"vertical plasma field on axis, $\phi=0$" "\n" r"per unit $\alpha$ ($10^{-4}$ T)")
    panel_label(a, "(a)")

    names = [p["name"] for p in pairs]
    floor = [p["mid_radius"]["m1_Fs_vacuum_Pa"] for p in pairs]
    dp = [abs(p["mid_radius"]["dp_ds_Pa"]) for p in pairs]
    # Pfirsch-Schlueter part of the force at s = 1/2: 2*etabar*r/iota_N of |p'(s)|.
    ps = 2 * 0.83636 * np.sqrt(0.5) * check["radius_m"] / 0.21248
    xb = np.arange(len(names))
    b.bar(xb - 0.2, floor, width=0.38, color=MAGENTA, label="vacuum m=1 force residual")
    b.bar(xb + 0.2, np.array(dp) * ps, width=0.38, color=AQUA, label="expected PS force at α_max")
    b.set_xticks(xb, [pretty.get(n, n) for n in names], fontsize=6.3)
    b.set_ylabel(r"$m=1$ radial force at $s=1/2$ (Pa)")
    b.legend(loc="upper right")
    panel_label(b, "(b)")
    fig.tight_layout()
    save(fig, "equilibrium_current_diagnostic")


if __name__ == "__main__":
    figure_source_operator()
    figure_pressure_response()
    figure_equilibrium_current()
    print(sorted(p.name for p in OUT.iterdir()))
