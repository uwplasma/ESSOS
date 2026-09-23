"""Regenerate every numerical table of the paper from the archived run outputs.

python make_tables.py  ->  results.json, tables.tex

Reads runs/*/summary.json and runs/*/axis_targets_optimized.npz only; nothing is re-optimized.
The displacement ratio is recomputed from the archived optimized axis at two axis resolutions.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pyqsc_jax.near_axis import near_axis

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CASES = ["qa", "nohess", "control_single", "control", "fixed_sub", "fixed_nosub", "vacuum", "axisym", "a010", "a015", "a020"]
NS33 = ["qa_ns33", "a010_ns33", "a015_ns33", "a020_ns33", "fixed_sub_ns33"]
LADDER = NS33 + ["V_ftol", "V_ns129", "L_ns129", "L_mode10", "L_nzeta64", "L_ftol", "L_coilquad", "L_grid65", "L_grid129", "L_gridwide"]


def load(name):
    path = RUNS / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def displacement_ratio(summary, nphi, ntheta=128):
    """max over the surface of a |x2| / |x1|: second- over first-order displacement at r = a."""
    saved = np.load(RUNS / summary["_name"] / "axis_targets_optimized.npz")
    inputs = summary["inputs"]
    sol = near_axis(rc=jnp.asarray(saved["rc"]), zs=jnp.asarray(saved["zs"]), etabar=float(saved["etabar"]),
                    nfp=inputs["nfp"], nphi=nphi, order="r2", B0=inputs["B0"], I2=inputs["I2"], p2=inputs["p2"],
                    B2c=inputs["B2c"]).solution
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)[:, None]
    c, s = np.cos(theta), np.sin(theta)
    g = lambda name: np.asarray(getattr(sol, name))[None, :]
    x1 = np.hypot(g("X1c") * c, g("Y1s") * s + g("Y1c") * c)
    so = sol.second_order
    h = lambda name: np.asarray(getattr(so, name))[None, :]
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    X2 = h("X20") + h("X2c") * c2 + h("X2s") * s2
    Y2 = h("Y20") + h("Y2c") * c2 + h("Y2s") * s2
    Z2 = h("Z20") + h("Z2c") * c2 + h("Z2s") * s2
    return float(np.max(inputs["a"] * np.sqrt(X2**2 + Y2**2 + Z2**2) / x1))


def free_boundary_row(summary, route="direct"):
    report = summary.get("vmex", {}).get("fitted", {}).get("optimized", {}).get(route)
    if not report:
        return None
    row = dict(converged=report["converged"], error=report.get("error"))
    if report["converged"]:
        near = report["near_axis"]
        row.update(iterations=report["iterations"], seconds=report["seconds"], fsqr=report["fsqr"],
                   benchmark_radius_m=near["benchmark_radius_m"], betatotal=report["betatotal"],
                   iota_lab_vmex=report["signed"]["vmex_iota_axis_lab"],
                   iota_lab_near_axis=report["signed"]["near_axis_iota_lab"],
                   iota_raw_vmex=report["iota_axis"], iota_raw_near_axis=report["iota_near_axis"],
                   axis_shift_max_m=near["axis_shift_max_m"],
                   axis_shift_over_benchmark_radius=near["axis_shift_over_benchmark_radius"],
                   surfaces=near["surfaces"], interface=report["interface"], runtime=report["runtime"],
                   grid_margin_m=report.get("grid_margin_m"))
    return row


results, missing = dict(runs={}, ladder={}), []
for name in CASES + LADDER:
    summary = load(name)
    if summary is None:
        missing.append(name)
        continue
    summary["_name"] = name
    opt = summary["states"]["optimized"]
    entry = dict(config_hash=summary.get("config_hash"), inputs=summary["inputs"], optimization=summary.get("optimization"),
                 iota_near_axis=opt["iota"], etabar=opt["etabar"], axis_match=opt["axis_match"],
                 boundary=opt["boundary"], initial_boundary=summary["states"]["initial"]["boundary"],
                 vmex_settings=summary["vmex_settings"],
                 summary_sha256=hashlib.sha256((RUNS / name / "summary.json").read_bytes()).hexdigest()[:16])
    for route in ("direct", "mgrid"):
        entry[route] = free_boundary_row(summary, route)
    if entry["direct"] is not None:
        entry["direct"]["ftol_final"] = summary["vmex_settings"]["ftol"][-1]
    relaxed = load(f"{name}_f9")  # The same solve with final FTOL 1e-9, used only if 1e-10 failed.
    if relaxed is not None:
        entry["direct_ftol_1e-9"] = free_boundary_row(relaxed, "direct")
        if entry["direct"] is not None and not entry["direct"]["converged"] and entry["direct_ftol_1e-9"]:
            entry["direct_ftol_1e-10_failure"] = entry["direct"]
            entry["direct"] = dict(entry["direct_ftol_1e-9"], ftol_final=1e-9)
    split = load(f"{name}_mgrid")  # A route run on its own, to keep every job under ten minutes.
    if split is not None and entry["mgrid"] is None:
        entry["mgrid"] = free_boundary_row(split, "mgrid")
        summary.setdefault("vmex", {}).setdefault("fitted", {}).setdefault("optimized", {})["mgrid_table"] = \
            split["vmex"]["fitted"]["optimized"].get("mgrid_table")
    fitted = summary.get("vmex", {}).get("fitted", {}).get("optimized", {})
    entry["mgrid_table"] = fitted.get("mgrid_table")
    entry["mgrid_vs_direct"] = fitted.get("mgrid_vs_direct")
    pair = [RUNS / n / "vmex_fitted_optimized" / f"wout_{r}.nc" for n, r in ((name, "direct"), (f"{name}_mgrid", "mgrid"))]
    if entry["mgrid_vs_direct"] is None and all(path.exists() for path in pair) and entry["mgrid"] and entry["mgrid"]["converged"]:
        import vmex as vj
        sys.path.insert(0, os.environ.get("ESSOS_EXAMPLE_DIR", str(HERE.parent)))
        import nearaxis_finite_beta_helpers as helpers
        entry["mgrid_vs_direct"] = helpers.compare_equilibria(
            *(vj.read_wout(path) for path in pair), entry["direct"]["benchmark_radius_m"],
            summary["vmex_settings"]["flux_levels"], summary["inputs"]["nfp"])
    trace = load(f"{name}_trace") or {}
    for key in ("poincare", "signed_iota_traced", "flux_check"):
        if key in summary or key in trace:
            entry[key] = summary.get(key, trace.get(key))
    if name in CASES:
        entry["displacement_ratio"] = {str(n): displacement_ratio(summary, n) for n in (151, 301)}
    (results["runs"] if name in CASES else results["ladder"])[name] = entry
results["missing"] = missing
Path("results.json").write_text(json.dumps(results, indent=1, default=float) + "\n")


# ------------------------------------ LaTeX ------------------------------------
def pct(x, digits=1):
    return "--" if x is None else f"{100 * x:.{digits}f}\\%"


def sci(x, digits=1):
    if x is None:
        return "--"
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


out = []
qa = results["runs"].get("qa")
if qa:
    m = qa["axis_match"]
    out += ["% Table: axis fit (qa)",
            f"Field $[\\mathrm T]$ & {sci(m['field_rms_T'])} & {sci(m['plasma_field_rms_T'])}\\\\",
            f"Gradient $[\\mathrm{{T\\,m^{{-1}}}}]$ & {sci(m['gradient_rms_T_per_m'])} & {sci(m['plasma_gradient_rms_T_per_m'])}\\\\",
            f"Hessian $[\\mathrm{{T\\,m^{{-2}}}}]$ & {sci(m['hessian_rms_T_per_m2'])} & ${m['plasma_hessian_rms_T_per_m2']:.1f}$\\\\",
            f"% target Hessian RMS {m['target_hessian_rms_T_per_m2']:.2f}; normal max {pct(qa['boundary']['normal_error_max'], 2)},"
            f" rms {pct(qa['boundary']['normal_error_rms'], 3)}; initial max {pct(qa['initial_boundary']['normal_error_max'])}",
            "% Table: surfaces (qa, direct)"]
    for row in qa["direct"]["surfaces"]:
        out.append(f"${row['s']:.4f}$ & {pct(row['rms_over_flux_radius'])} & {pct(row['shape_rms_over_flux_radius'])}\\\\")
labels = dict(qa="Finite pressure", nohess="No Hessian residual", control_single="Total field, joint",
              control="Total field, joint (segmented)",
              fixed_sub="Subtracted, fixed axis", fixed_nosub="Total field, fixed axis", vacuum="Vacuum",
              axisym="Axisymmetric", a010="$a=0.010$", a015="$a=0.015$", a020="$a=0.020$")
out.append("% Table: free-boundary cases (direct route)")
for name, entry in results["runs"].items():
    d = entry["direct"]
    if not d or not d["converged"]:
        out.append(f"% {name}: free boundary not converged ({d and d['error']})")
        continue
    mark = "$^\\dagger$" if d.get("ftol_final") == 1e-9 else ""
    out.append(f"{labels[name]}{mark} & ${d['benchmark_radius_m'] / entry['inputs']['a']:.2g}$ & ${d['iota_lab_vmex']:.4f}$ & "
               f"${d['iota_lab_near_axis']:.4f}$ & {pct(d['axis_shift_over_benchmark_radius'])} & "
               f"{pct(d['surfaces'][-1]['shape_rms_over_flux_radius'])} & {pct(entry['boundary']['normal_error_max'], 2)}\\\\")
out.append("% Table: radius sequence")
for name in ("a010", "a015", "a020", "qa"):
    e = results["runs"].get(name)
    if not e:
        continue
    d, m = e["direct"] or {}, e["axis_match"]
    out.append(f"${e['inputs']['a']:.3f}$ & ${e['displacement_ratio']['301']:.3f}$ & "
               f"${m['plasma_field_rms_T'] / m['field_rms_T']:.1f}$ & ${m['plasma_gradient_rms_T_per_m'] / m['gradient_rms_T_per_m']:.1f}$ & "
               f"{pct(e['boundary']['normal_error_max'], 2)} & {pct(d.get('axis_shift_over_benchmark_radius'))} & "
               f"{pct(d['surfaces'][-1]['shape_rms_over_flux_radius']) if d.get('surfaces') else '--'}\\\\")
out.append("% Table: resolution ladder (qa coils)")
for name, e in results["ladder"].items():
    d = e["direct"] or e["mgrid"]
    if not d or not d["converged"]:
        out.append(f"% {name}: not converged ({d and d['error']})")
        continue
    out.append(f"{name} & ${d['iota_lab_vmex']:.5f}$ & {pct(d['axis_shift_over_benchmark_radius'], 2)} & "
               f"{pct(d['surfaces'][-1]['shape_rms_over_flux_radius'], 2)} & {pct(d['interface']['tangential_jump_max_over_B'], 2)} & {d['iterations']}\\\\")
Path("tables.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
print("missing:", missing)
