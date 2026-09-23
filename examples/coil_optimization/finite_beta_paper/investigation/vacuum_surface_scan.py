"""Last closed coil-field surface of the vacuum coils: trace lines launched ever farther from the axis.

For each launch radius r (outboard, phi = 0 plane, from the near-axis magnetic axis) the line is
traced for many toroidal turns. Reported: whether it stays within 3 r of the axis, its laboratory
rotation per turn, and the smoothness of its phi = 0 punctures -- the RMS residual of a 6-harmonic
fit of the puncture radius about the axis as a function of angle, relative to r. A closed surface
gives a tiny residual; an island chain or chaotic band a large one.
"""
import json, sys
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.dynamics import Tracing
from pyqsc_jax.near_axis import near_axis

run = Path(sys.argv[1]); radii = np.array([float(x) for x in sys.argv[2].split(",")])
summary = json.loads((run / "summary.json").read_text()); inp = summary["inputs"]
axis = np.load(run / "axis_targets_optimized.npz")
sol = near_axis(rc=jnp.asarray(axis["rc"]), zs=jnp.asarray(axis["zs"]), etabar=float(axis["etabar"]), nfp=inp["nfp"],
                nphi=151, order="r3", B0=inp["B0"], I2=inp["I2"], p2=inp["p2"], B2c=inp["B2c"]).solution
field = BiotSavart(Coils.from_json(str(run / "coils_optimized.json")))
nfp, grid = inp["nfp"], np.asarray(sol.phi); period = 2 * np.pi / nfp
R0, Z0 = np.asarray(sol.R0), np.asarray(sol.Z0)
starts = [[R0[0] + r, 0.0, Z0[0]] for r in radii]
tr = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=jnp.asarray(starts),
             maxtime=float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0, times_to_trace=40000, atol=1e-10, rtol=1e-10)
rows = []
for r, line in zip(radii, np.asarray(tr.trajectories_xyz)):
    x, y, z = line[:, :3].T
    ok = np.all(np.isfinite(line[:, :3]), axis=1); x, y, z = x[ok], y[ok], z[ok]
    phi = np.unwrap(np.arctan2(y, x)); R = np.hypot(x, y)
    RA = np.interp(np.mod(phi, period), grid, R0, period=period); ZA = np.interp(np.mod(phi, period), grid, Z0, period=period)
    dist = np.hypot(R - RA, z - ZA)
    confined = bool(dist.max() < 3 * r)
    winding = np.unwrap(np.arctan2(z - ZA, R - RA))
    turns = abs(phi[-1] - phi[0]) / 2 / np.pi
    iota = float((winding[-1] - winding[0]) / (phi[-1] - phi[0]))
    # phi = 0 (mod period) punctures
    k = np.flatnonzero(np.floor(phi[1:] / period) != np.floor(phi[:-1] / period))
    w = (np.round(phi[k + 1] / period) * period - phi[k]) / (phi[k + 1] - phi[k])
    pR, pZ = R[k] + w * (R[k + 1] - R[k]), z[k] + w * (z[k + 1] - z[k])
    ang, rad = np.arctan2(pZ - Z0[0], pR - R0[0]), np.hypot(pR - R0[0], pZ - Z0[0])
    A = np.stack([np.ones_like(ang)] + [f(m * ang) for m in range(1, 7) for f in (np.cos, np.sin)], 1)
    fit = np.linalg.lstsq(A, rad, rcond=None)[0] if len(ang) > 20 else None
    smooth = float(np.sqrt(np.mean((A @ fit - rad) ** 2)) / r) if fit is not None else float("nan")
    # Toroidal coil flux through the puncture curve (12-harmonic fit about the axis), as a flux radius.
    r_eff = float("nan")
    if len(ang) > 40 and confined:
        A12 = np.stack([np.ones_like(ang)] + [f(m * ang) for m in range(1, 13) for f in (np.cos, np.sin)], 1)
        c12 = np.linalg.lstsq(A12, rad, rcond=None)[0]
        th = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        T12 = np.stack([np.ones_like(th)] + [f(m * th) for m in range(1, 13) for f in (np.cos, np.sin)], 1)
        edge = T12 @ c12
        g, gw = np.polynomial.legendre.leggauss(24); g, gw = (g + 1) / 2, gw / 2
        rr = g[:, None] * edge[None, :]
        Rq, Zq = R0[0] + rr * np.cos(th), Z0[0] + rr * np.sin(th)
        B = np.asarray(jax.vmap(field.B)(jnp.asarray(np.stack((Rq, 0 * Rq, Zq), -1).reshape(-1, 3)))).reshape(Rq.shape + (3,))
        flux = abs(np.sum(gw[:, None] * B[..., 1] * rr * edge[None, :]) * 2 * np.pi / th.size)
        r_eff = float(np.sqrt(flux / np.pi / inp["B0"]))
    rows.append(dict(r_flux=r_eff, r_flux_over_a=r_eff / inp["a"], r=float(r), r_over_a=float(r / inp["a"]), confined=confined, turns=float(turns), iota_lab=iota,
                     punctures=int(len(ang)), puncture_fit_residual_over_r=smooth, max_distance_over_r=float(dist.max() / r)))
    print(f"r = {r:.4f} m ({r / inp['a']:.2f} a), flux radius {r_eff / inp['a']:.3f} a: confined {confined}, {turns:5.0f} turns, iota {iota:+.4f}, "
          f"puncture residual / r {smooth:.2e}, max distance / r {dist.max() / r:.2f}", flush=True)
Path(f"{run.name}_surface_scan.json").write_text(json.dumps(rows, indent=1))
