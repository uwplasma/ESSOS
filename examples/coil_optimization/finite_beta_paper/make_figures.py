"""Regenerate the paper's contour figures from archived files only (no optimization, no solve).

python make_figures.py -> figures/*.png. Near-axis surfaces are rebuilt from the archived
optimized axis (axis_targets_optimized.npz); free-boundary surfaces are read from the wout files.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vmex as vj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("ESSOS_EXAMPLE_DIR", str(HERE.parent)))
import nearaxis_finite_beta_helpers as h
from pyqsc_jax.near_axis import near_axis
from vmex.core.plotting import surface_rz

RUNS, OUT = HERE / "runs", HERE / "figures"
OUT.mkdir(exist_ok=True)


def near_solution(name):
    summary = json.loads((RUNS / name / "summary.json").read_text())
    saved, inputs = np.load(RUNS / name / "axis_targets_optimized.npz"), summary["inputs"]
    return near_axis(rc=jnp.asarray(saved["rc"]), zs=jnp.asarray(saved["zs"]), etabar=float(saved["etabar"]),
                     nfp=inputs["nfp"], nphi=151, order="r3", B0=inputs["B0"], I2=inputs["I2"], p2=inputs["p2"],
                     B2c=inputs["B2c"]).solution, summary


def wout(name, route):
    path = RUNS / name / "vmex_fitted_optimized" / f"wout_{route}.nc"
    return vj.read_wout(path) if path.exists() else None


def contours(name, routes, path, levels=(0.0625, 0.25, 0.5625, 1.0)):
    solution, summary = near_solution(name)
    radius = summary["inputs"]["a"] * summary["config"]["inputs"].get("vmex_radius_fraction", 1.0)
    phi, period = np.asarray(solution.phi), 2 * np.pi / int(solution.inputs.axis.nfp)
    theta = np.arange(257) * 2 * np.pi / 256
    styles = {"direct": "--", "mgrid": ":"}
    with plt.rc_context(h.STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.0), layout="constrained")
        for ax, fraction in zip(axes.ravel(), (0.0, 0.25, 0.5, 0.75)):
            k = int(np.argmin(np.abs(phi - fraction * period)))
            for s in levels:
                near = h.flux_surface(solution, radius * np.sqrt(s), 256)
                ax.plot(np.r_[near["R"][:, k], near["R"][0, k]], np.r_[near["Z"][:, k], near["Z"][0, k]],
                        color=h.COLORS["near"], lw=1.3)
            for route, source in routes.items():
                equilibrium = wout(source, route)
                for index, _ in h.flux_indices(equilibrium, levels):
                    RV, ZV = surface_rz(equilibrium, s_index=index, theta=theta, phi=phi[k:k + 1])
                    ax.plot(RV[:, 0], ZV[:, 0], styles[route], color=h.COLORS[route], lw=1.3)
                RA, ZA = surface_rz(equilibrium, s_index=0, theta=np.zeros(1), phi=phi[k:k + 1])
                ax.plot(RA[0, 0], ZA[0, 0], "x", color=h.COLORS[route], ms=5, mew=1.4)
            ax.plot(float(solution.R0[k]), float(solution.Z0[k]), "+", color=h.COLORS["near"], ms=7, mew=1.4)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.6)
            ax.set_title(rf"$\phi$ = {fraction:g} field period", fontsize=9)
            ax.set_xlabel("R [m]", fontsize=9)
            ax.set_ylabel("Z [m]", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.locator_params(nbins=4)
        handles = [plt.Line2D([], [], color=h.COLORS["near"], lw=1.3, label="near axis")]
        handles += [plt.Line2D([], [], color=h.COLORS[r], ls=styles[r], lw=1.3,
                               label=f"free boundary, {'direct coils' if r == 'direct' else 'field file'}") for r in routes]
        fig.legend(handles=handles, loc="outside lower center", ncols=3, fontsize=8)
        fig.savefig(path)
    plt.close(fig)


contours("qa", {"direct": "qa", "mgrid": "qa_mgrid"}, OUT / "qa_cross_sections.png")
def converged(name, route="direct"):
    report = json.loads((RUNS / name / "summary.json").read_text())["vmex"]["fitted"]["optimized"].get(route, {})
    return bool(report.get("converged"))


# Only converged equilibria are drawn; a failed solve still writes a wout file.
for name, label in (("control_single_f9", "control"), ("nohess", "nohess")):
    if converged(name):
        contours(name, {"direct": name}, OUT / f"{label}_cross_sections.png")
for name, figure in (("qa", "coils_and_normal_field"), ("qa", "axis_fields"), ("vacuum_trace", "poincare")):
    source = RUNS / name / f"{figure}.png"
    if source.exists():
        shutil.copy(source, OUT / f"{'vacuum' if name == 'vacuum_trace' else name}_{figure}.png")
print(sorted(p.name for p in OUT.iterdir()))
