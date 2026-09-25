"""Condense VMEC2000 and VMEX pressure-family runs into one compact JSON.

Every VMEC2000 case directory holds ``input.case`` and, if it finished,
``wout_case.nc``; every VMEX run directory is a ``scan.py`` output. Native
WOUT files stay outside Git; their SHA-256 hashes are recorded here.

    python collect_verification.py --vmec2000 RUNS/v2k_* --vmex RUNS/qa18_* \
        --output results/verification/pressure_runs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deck_value(deck, key):
    match = re.search(rf"^\s*{key}\s*=\s*(.*)$", deck, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None


def vmec2000_case(directory):
    deck = (directory / "input.case").read_text()
    wout = directory / "wout_case.nc"
    row = {"case": f"{directory.parent.name}/{directory.name}",
           "pres_scale_Pa": float(deck_value(deck, "PRES_SCALE")),
           "lfreeb": "TRUE" in deck_value(deck, "LFREEB").upper(),
           "ns_array": deck_value(deck, "NS_ARRAY"), "ftol_array": deck_value(deck, "FTOL_ARRAY"),
           "mgrid": deck_value(deck, "MGRID_FILE"), "input_sha256": sha(directory / "input.case"),
           "finished": wout.exists()}
    if wout.exists():
        with Dataset(wout) as nc:
            rc = np.asarray(nc["raxis_cc"][:])
            row.update(ns=int(nc["ns"][:]), ier_flag=int(nc["ier_flag"][:]),
                       fsqr=float(nc["fsqr"][:]), fsqz=float(nc["fsqz"][:]), fsql=float(nc["fsql"][:]),
                       betatotal=float(nc["betatotal"][:]), R_axis_phi0_m=float(np.sum(rc)),
                       iota_axis_raw=float(np.asarray(nc["iotaf"][:])[0]),
                       wout_sha256=sha(wout))
        timing = directory / "timings.txt"
        if timing.exists():
            match = re.search(r"^\s*total\s*:\s*([0-9.]+)", timing.read_text(), re.MULTILINE)
            row["seconds"] = float(match.group(1)) if match else None
    return row


def vmex_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    points = json.loads((directory / "pressure_points.json").read_text()) if (
        directory / "pressure_points.json").exists() else []
    vacuum = manifest.get("vacuum", {})
    return {"run": directory.name, "settings": manifest.get("settings"),
            "restart_policy": manifest.get("pressure_family", {}).get("restart_policy"),
            "alpha_max": manifest["theory"]["pressure_multiplier_max"],
            "vacuum_R_axis_phi0_m": vacuum.get("axis_R", [None])[0],
            "vacuum_iterations": vacuum.get("iterations"),
            "points": [{"alpha": p["alpha"], "accepted": p["accepted"], "iterations": p["iterations"],
                        "fsqr": p["fsqr"], "delta_R0_m": p["delta_R"][0],
                        "delta_length_over_L": p["delta_length_over_L"],
                        "betatotal": p["betatotal"], "wout_sha256": p["wout_sha256"]} for p in points]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmec2000", nargs="*", type=Path, default=[])
    parser.add_argument("--vmex", nargs="*", type=Path, default=[])
    parser.add_argument("--theory", type=Path, required=True, help="a scan manifest.json")
    parser.add_argument("--traced-axis", type=Path, required=True,
                        help="axis_operator_check.npz with the directly traced vacuum axis")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    theory = json.loads(args.theory.read_text())
    theory_npz = np.load(args.theory.parent / "theory.npz")
    cases = [vmec2000_case(d) for root in args.vmec2000
             for d in sorted(root.iterdir()) if (d / "input.case").exists()]
    runs = [vmex_run(d) for d in args.vmex if (d / "manifest.json").exists()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "theory": {"delta_R0_per_alpha_m": float(theory_npz["delta_R"][0]),
                   "alpha_max": theory["theory"]["pressure_multiplier_max"],
                   "length_slope_over_L": theory["theory"]["length_slope_over_L"],
                   "flux_radius_m": theory["flux_radius_m"],
                   "p0_star_Pa": 600000.0 * theory["flux_radius_m"] ** 2,
                   "coil_sha256": theory["coil_sha256"],
                   "traced_vacuum_R_axis_phi0_m": float(np.load(args.traced_axis)["traced_axis"][0, 0])},
        "vmec2000": cases, "vmex": runs}, indent=1) + "\n")
    print(f"{len(cases)} VMEC2000 cases, {len(runs)} VMEX runs")


if __name__ == "__main__":
    main()
