"""Pressure-source checks: near-axis current, equilibrium current, local force.

1. Biot-Savart of the pyQSC_JAX positive-volume current measure on the
   first-order surfaces, against the closed-form on-axis plasma field, for a
   ladder of toroidal resolutions (the error falls as nphi**-2).
2. For each (vacuum, pressure) WOUT pair: the free-space field of the
   equilibrium's own current (Ampere's law on the covariant field) at the
   vacuum axis, differenced against the vacuum at the same resolution, and the
   local radial force residual F_s = sqrt(g)(J^u B^v - J^v B^u) - p'(s).

    python source_field_check.py --output results/source_field_check.json \
        --pair NAME VACUUM_WOUT PRESSURE_WOUT ALPHA ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from netCDF4 import Dataset
from pyqsc_jax.near_axis import near_axis
from pyqsc_jax.plasma import (
    evaluate_weighted_current,
    plasma_current_source,
    plasma_field_on_axis,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from equilibrium_plasma_field import MU0, biot_savart, current_elements

PHI_SAMPLES = np.array([0.0, np.pi / 4, np.pi / 2])


def near_axis_elements(solution, radius, nr=8, ntheta=32):
    """Full-torus Gauss-Legendre elements of mu0*J*dV / mu0 on first-order surfaces."""
    geometry = solution.geometry
    nfp = int(solution.inputs.axis.nfp)
    nphi = len(solution.phi)
    source = plasma_current_source(solution, formal_radius=radius)
    nodes, weights = np.polynomial.legendre.leggauss(nr)
    r, wr = (nodes + 1) * radius / 2, weights * radius / 2
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    normal = np.asarray(geometry.normal_cartesian)
    binormal = np.asarray(geometry.binormal_cartesian)
    axis = np.asarray(geometry.position_cartesian)
    dvarphi = np.asarray(geometry.d_varphi_d_phi) * 2 * np.pi / nfp / nphi
    X1c, Y1s, Y1c = (np.asarray(v) for v in (solution.X1c, solution.Y1s, solution.Y1c))
    x, w = [], []
    for ri, wi in zip(r, wr):
        for t in theta:
            W = np.asarray(evaluate_weighted_current(source, ri, t))
            e = X1c[:, None] * np.cos(t) * normal + (Y1s * np.sin(t) + Y1c * np.cos(t))[:, None] * binormal
            x.append(axis + ri * e)
            w.append(W * (wi * 2 * np.pi / ntheta) * dvarphi[:, None])
    x, w = np.concatenate(x), np.concatenate(w) * source.chi / MU0
    rotations = []
    for k in range(nfp):
        c, s = np.cos(2 * np.pi * k / nfp), np.sin(2 * np.pi * k / nfp)
        rotations.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    return (np.concatenate([x @ R.T for R in rotations]),
            np.concatenate([w @ R.T for R in rotations]))


def near_axis_ladder(spec, radius, nphis):
    kw = dict(rc=spec["rc"], zs=spec["zs"], nfp=spec["nfp"], etabar=spec["etabar"],
              order="r3", B0=spec["B0"], B2c=spec["B2c"], I2=0.0, p2=spec["p2_star"])
    rows = []
    for nphi in nphis:
        solution = near_axis(nphi=nphi, **kw).solution
        closed = np.asarray(plasma_field_on_axis(solution, formal_radius=radius).field)
        index = [0, nphi // 10, nphi // 4]
        x, w = near_axis_elements(solution, radius)
        direct = biot_savart(np.asarray(solution.geometry.position_cartesian)[index], x, w)
        rows.append({"nphi": nphi, "biot_savart_T": direct.tolist(),
                     "closed_form_T": closed[index].tolist(),
                     "max_relative_difference": float(
                         np.max(np.linalg.norm(direct - closed[index], axis=1))
                         / np.max(np.linalg.norm(closed[index], axis=1)))})
    a, b = np.asarray(rows[-2]["biot_savart_T"]), np.asarray(rows[-1]["biot_savart_T"])
    factor = (rows[-1]["nphi"] / rows[-2]["nphi"]) ** 2
    richardson = b + (b - a) / (factor - 1)
    closed = np.asarray(rows[-1]["closed_form_T"])
    return {"ladder": rows, "richardson_T": richardson.tolist(),
            "richardson_relative_difference": float(
                np.max(np.linalg.norm(richardson - closed, axis=1))
                / np.max(np.linalg.norm(closed, axis=1)))}


def wout_axis(path, phi):
    with Dataset(path) as nc:
        rc, zs = np.asarray(nc["raxis_cc"][:]), np.asarray(nc["zaxis_cs"][:])
        nfp = int(nc["nfp"][:])
    n = np.arange(len(rc)) * nfp
    R, Z = np.cos(np.outer(phi, n)) @ rc, -np.sin(np.outer(phi, n)) @ zs
    return np.stack((R * np.cos(phi), R * np.sin(phi), Z), -1)


def local_radial_force(path, ntheta=64, nzeta=64):
    """Angle-resolved F_s on interior full-mesh surfaces and p'(s)."""
    with Dataset(path) as nc:
        g = lambda k: np.asarray(nc[k][:])
        ns, nfp = int(g("ns")), int(g("nfp"))
        m, n = g("xm_nyq"), g("xn_nyq")
        th = 2 * np.pi * np.arange(ntheta) / ntheta
        ze = 2 * np.pi / nfp * np.arange(nzeta) / nzeta
        angle = m[:, None, None] * th[None, :, None] - n[:, None, None] * ze[None, None, :]
        cos = np.cos(angle)
        ev = lambda c: np.einsum("jm,mtz->jtz", c, cos)
        Bu, Bv = ev(g("bsubumnc")), ev(g("bsubvmnc"))
        Bs_u, Bs_v = ev(g("bsubsmns") * m), ev(-g("bsubsmns") * n)
        bu, bv = ev(g("bsupumnc")), ev(g("bsupvmnc"))
        pres = g("presf")
    ds = 1 / (ns - 1)
    j = np.arange(1, ns - 1)
    gJu = (Bs_v[j] - (Bv[j + 1] - Bv[j]) / ds) / MU0
    gJv = ((Bu[j + 1] - Bu[j]) / ds - Bs_u[j]) / MU0
    Fs = gJu * 0.5 * (bv[j] + bv[j + 1]) - gJv * 0.5 * (bu[j] + bu[j + 1])
    dpds = np.gradient(pres, ds)[j]
    return j * ds, Fs - dpds[:, None, None], dpds


def pair_report(name, vacuum, pressure, alpha, resolutions):
    points = wout_axis(vacuum, PHI_SAMPLES)
    fields = []
    for res in resolutions:
        x0, w0 = current_elements(vacuum, *res)
        x1, w1 = current_elements(pressure, *res)
        dB = (biot_savart(points, x1, w1, chunk=1) - biot_savart(points, x0, w0, chunk=1)) / alpha
        fields.append({"ntheta_nzeta_period": list(res), "delta_B_per_alpha_T": dB.tolist()})
    s, F, dp = local_radial_force(pressure)
    s0, F0, _ = local_radial_force(vacuum)
    k = len(s) // 2
    harmonic = lambda f: float(np.mean(np.abs(np.fft.fft(f, axis=0)[1])) * 2 / f.shape[0])
    with Dataset(vacuum) as a, Dataset(pressure) as b:
        dR = float(np.sum(b["raxis_cc"][:]) - np.sum(a["raxis_cc"][:]))
        ns = int(a["ns"][:])
    return {"name": name, "ns": ns, "alpha": alpha, "vacuum": str(vacuum), "pressure": str(pressure),
            "phi": PHI_SAMPLES.tolist(), "equilibrium_current_field": fields,
            "delta_R_axis_phi0_m": dR,
            "mid_radius": {"s": float(s[k]), "dp_ds_Pa": float(dp[k]),
                           "mean_Fs_Pa": float(F[k].mean()),
                           "m1_Fs_pressure_Pa": harmonic(F[k]),
                           "m1_Fs_vacuum_Pa": harmonic(F0[k])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=HERE / "reference" / "vacuum_fitted_reference.json")
    parser.add_argument("--radius", type=float, default=0.0178115)
    parser.add_argument("--nphi", nargs="+", type=int, default=[101, 201, 401, 801])
    parser.add_argument("--pair", nargs=4, action="append", default=[],
                        metavar=("NAME", "VACUUM", "PRESSURE", "ALPHA"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.reference.read_text())
    report = {"near_axis_source": near_axis_ladder(spec, args.radius, args.nphi),
              "radius_m": args.radius, "pairs": []}
    for name, vacuum, pressure, alpha in args.pair:
        report["pairs"].append(pair_report(name, Path(vacuum), Path(pressure), float(alpha),
                                           [(64, 100), (64, 200), (64, 400)]))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2)[:1500])


if __name__ == "__main__":
    main()
