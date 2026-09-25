"""Independent VMEC2000 free-boundary solve of one fixed-coil pressure family.

Reads a VMEX run directory written by ``scan.py`` (its manifest and vacuum
runtime deck), tabulates the same coils into a MAKEGRID file, and runs
VMEC2000 cold from the same boundary guess at alpha = 0 and each requested
positive pressure fraction. Only PRES_SCALE, the MGRID file and the radial
ladder differ from the VMEX deck.

    python vmec2000_check.py --run /path/to/qa18 --output /path/to/qa18_vmec2000 \
        --xvmec /path/to/xvmec2000 --ns 17 33 65 --ftol 1e-8 1e-10 1e-12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import nearaxis_finite_beta_helpers as helpers

from scan import load_reference


def set_key(deck, key, value):
    pattern = re.compile(rf"^\s*{key}\s*=.*$", re.MULTILINE | re.IGNORECASE)
    line = f"  {key} = {value}"
    return pattern.sub(line, deck) if pattern.search(deck) else deck.replace(
        "&INDATA", "&INDATA\n" + line, 1)


def axis_from_wout(path, phi):
    with Dataset(path) as nc:
        raxis = np.asarray(nc["raxis_cc"][:])
        zaxis = np.asarray(nc["zaxis_cs"][:])
        nfp = int(nc["nfp"][:])
        record = {
            "ier_flag": int(nc["ier_flag"][:]),
            "fsqr": float(nc["fsqr"][:]), "fsqz": float(nc["fsqz"][:]),
            "fsql": float(nc["fsql"][:]),
            "betatotal": float(nc["betatotal"][:]),
            "iota_axis_raw": float(np.asarray(nc["iotaf"][:])[0]),
            "ctor_A": float(nc["ctor"][:]),
            "edge_flux_Wb": float(np.asarray(nc["phi"][:])[-1]),
        }
    n = np.arange(len(raxis)) * nfp
    R = np.cos(np.outer(phi, n)) @ raxis
    Z = -np.sin(np.outer(phi, n)) @ zaxis
    return R, Z, record


def axis_length(R, Z, nfp):
    k = nfp * np.fft.fftfreq(len(R), d=1 / len(R))
    d = lambda f: np.fft.ifft(1j * k * np.fft.fft(f)).real
    return float(2 * np.pi * np.mean(np.sqrt(d(R) ** 2 + R**2 + d(Z) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xvmec", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=HERE / "reference" / "vacuum_fitted_reference.json")
    parser.add_argument("--ns", nargs="+", type=int, default=[17, 33, 65])
    parser.add_argument("--ftol", nargs="+", type=float, default=[1e-8, 1e-10, 1e-12])
    parser.add_argument("--niter", type=int, default=40000)
    parser.add_argument("--mpol", type=int)
    parser.add_argument("--ntor", type=int)
    parser.add_argument("--grid", nargs=3, type=int, default=[161, 161, 64],
                        help="MAKEGRID R, Z and per-period phi points")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.0, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0])
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if len(args.ns) != len(args.ftol):
        parser.error("--ns and --ftol need the same length")

    manifest = json.loads((args.run / "manifest.json").read_text())
    deck0 = (args.run / "alpha_000" / "input.runtime").read_text()
    spec, coil_path, field, solution = load_reference(args.reference, 480)
    flux_radius = float(manifest["flux_radius_m"])
    alpha_max = float(manifest["theory"]["pressure_multiplier_max"])
    p0 = -float(spec["p2_star"]) * flux_radius**2
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    mgrid = out / "mgrid_essos.nc"
    surface = helpers.flux_surface(solution, float(manifest["radius_m"]))
    if not mgrid.exists():
        _, grid_record = helpers.write_coil_mgrid(field, surface, flux_radius,
                                                  tuple(args.grid), int(spec["nfp"]), mgrid)
        (out / "mgrid.json").write_text(json.dumps(grid_record, indent=2, default=float))
    grid_record = json.loads((out / "mgrid.json").read_text())

    deck0 = set_key(deck0, "MGRID_FILE", f"'{mgrid.name}'")
    deck0 = set_key(deck0, "NS_ARRAY", ", ".join(map(str, args.ns)))
    deck0 = set_key(deck0, "FTOL_ARRAY", ", ".join(f"{v:.3e}" for v in args.ftol))
    deck0 = set_key(deck0, "NITER_ARRAY", ", ".join([str(args.niter)] * len(args.ns)))
    deck0 = set_key(deck0, "NSTEP", "500")
    if args.mpol:
        deck0 = set_key(deck0, "MPOL", str(args.mpol))
    if args.ntor:
        deck0 = set_key(deck0, "NTOR", str(args.ntor))

    def run(fraction):
        alpha = alpha_max * fraction
        directory = out / f"frac_{fraction:.6f}"
        directory.mkdir(exist_ok=True)
        (directory / mgrid.name).unlink(missing_ok=True)
        (directory / mgrid.name).symlink_to(mgrid)
        deck = set_key(deck0, "PRES_SCALE", f"{alpha * p0:.17e}")
        (directory / "input.case").write_text(deck)
        start = time.perf_counter()
        wout = directory / "wout_case.nc"
        if not wout.exists():
            with open(directory / "vmec2000.log", "w") as log:
                subprocess.run([str(args.xvmec), "input.case"], cwd=directory,
                               stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter() - start
        if not wout.exists():
            return {"fraction": fraction, "alpha": alpha, "accepted": False,
                    "reason": "no wout", "seconds": elapsed}
        phi = 2 * np.pi / int(spec["nfp"]) * np.arange(256) / 256
        R, Z, record = axis_from_wout(wout, phi)
        record.update(fraction=fraction, alpha=alpha, seconds=elapsed,
                      pressure_axis_Pa=alpha * p0, axis_R=R.tolist(), axis_Z=Z.tolist(),
                      length_m=axis_length(R, Z, int(spec["nfp"])),
                      input_sha256=hashlib.sha256(deck.encode()).hexdigest(),
                      wout_sha256=hashlib.sha256(wout.read_bytes()).hexdigest(),
                      accepted=record["ier_flag"] == 0)
        return record

    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(run, args.fractions))
    vacuum = next(r for r in rows if r["fraction"] == 0.0)
    theory = np.load(args.run / "theory.npz")
    ideal_dR0 = float(theory["delta_R"][0])
    for r in rows:
        if r.get("accepted") and vacuum.get("accepted") and r["fraction"] > 0:
            r["delta_R0_m"] = r["axis_R"][0] - vacuum["axis_R"][0]
            r["delta_length_over_L"] = r["length_m"] / vacuum["length_m"] - 1
            r["delta_R0_over_alpha_theory"] = r["delta_R0_m"] / (r["alpha"] * ideal_dR0)
    summary = {
        "solver": "VMEC2000 (STELLOPT build), cold start at every point",
        "vmex_run": str(args.run),
        "coil_sha256": manifest["coil_sha256"],
        "flux_radius_m": flux_radius, "alpha_max": alpha_max, "p0_star_Pa": p0,
        "ns": args.ns, "ftol": args.ftol, "mgrid": grid_record,
        "theory_delta_R0_per_alpha_m": ideal_dR0,
        "theory_length_slope_over_L": float(manifest["theory"]["length_slope_over_L"]),
        "points": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    for r in rows:
        print(r["fraction"], r.get("accepted"), r.get("ier_flag"), r.get("fsqr"),
              r.get("delta_R0_over_alpha_theory"), round(r["seconds"], 1))


if __name__ == "__main__":
    main()
