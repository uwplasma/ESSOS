"""Coil-field transform on VMEX vacuum surfaces, with the true Jacobian, and traced from VMEX surfaces."""
import json, sys
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex as vj
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.dynamics import Tracing

run = Path(sys.argv[1]); route = sys.argv[2] if len(sys.argv) > 2 else "direct"
field = BiotSavart(Coils.from_json(str(run / "coils_optimized.json")))
Bfun = jax.jit(jax.vmap(field.B))
wout = vj.read_wout(run / "vmex_fitted_optimized" / f"wout_{route}.nc")
nfp, ns = int(wout.nfp), int(wout.ns)
xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
th = np.arange(128) * 2 * np.pi / 128
ph = np.arange(128) * 2 * np.pi / nfp / 128
ang = xm[:, None, None] * th[None, :, None] - xn[:, None, None] * ph[None, None, :]
C, S = np.cos(ang), np.sin(ang)
P = np.broadcast_to(ph[None, :], (th.size, ph.size))
c, s = np.cos(P), np.sin(P)

def geometry(js):
    rmnc, zmns = np.asarray(wout.rmnc)[js], np.asarray(wout.zmns)[js]
    R = np.einsum("m,mtp->tp", rmnc, C); Z = np.einsum("m,mtp->tp", zmns, S)
    Ru = np.einsum("m,mtp->tp", -rmnc * xm, S); Zu = np.einsum("m,mtp->tp", zmns * xm, C)
    Rv = np.einsum("m,mtp->tp", rmnc * xn, S); Zv = np.einsum("m,mtp->tp", -zmns * xn, C)
    xyz = np.stack((R * c, R * s, Z), -1)
    eu = np.stack((Ru * c, Ru * s, Zu), -1)
    ev = np.stack((Rv * c - R * s, Rv * s + R * c, Zv), -1)
    return xyz, eu, ev

rows = []
for js in range(4, ns - 1, max(1, ns // 10)):
    xyz, eu, ev = geometry(js)
    es = (geometry(js + 1)[0] - geometry(js - 1)[0]) / 2  # d x / d js; constant factor cancels in the ratio
    sqrtg = np.sum(es * np.cross(eu, ev), -1)
    B = np.asarray(Bfun(jnp.asarray(xyz.reshape(-1, 3)))).reshape(xyz.shape)
    # Contravariant components: B^u = B . (e_v x e_s)/sqrtg, B^v = B . (e_s x e_u)/sqrtg
    Bu = np.sum(B * np.cross(ev, es), -1) / sqrtg
    Bv = np.sum(B * np.cross(es, eu), -1) / sqrtg
    iota_coil = float(np.mean(sqrtg * Bu) / np.mean(sqrtg * Bv))
    rows.append(dict(s=js / (ns - 1), iota_vmex=float(np.asarray(wout.iotaf)[js]), iota_coil=iota_coil))
    print(f"s={js/(ns-1):.3f} iota VMEX {rows[-1]['iota_vmex']:+.5f}  coil field on VMEX surface {iota_coil:+.5f}", flush=True)

# Trace from VMEX surface points at phi = 0, u = 0; winding about the VMEX axis at the line's phi.
axis_R = lambda p: float(np.sum(np.asarray(wout.rmnc)[0] * np.cos(-xn * p)))
axis_Z = lambda p: float(np.sum(np.asarray(wout.zmns)[0] * np.sin(-xn * p)))
starts, labels = [], []
for js in (ns // 8, ns // 4, ns // 2, 3 * ns // 4):
    R0 = float(np.sum(np.asarray(wout.rmnc)[js])); starts.append([R0, 0.0, 0.0]); labels.append(js / (ns - 1))
tr = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=jnp.asarray(starts),
             maxtime=400.0, times_to_trace=8000, atol=1e-10, rtol=1e-10)
traced = []
for line, sv in zip(np.asarray(tr.trajectories_xyz), labels):
    x, y, z = line[:, :3].T
    phi = np.unwrap(np.arctan2(y, x)); R = np.hypot(x, y)
    RA = np.array([axis_R(p) for p in phi]); ZA = np.array([axis_Z(p) for p in phi])
    w = np.unwrap(np.arctan2(z - ZA, R - RA))
    it = float((w[-1] - w[0]) / (phi[-1] - phi[0]))
    traced.append(dict(s=sv, iota_traced_lab=it, turns=float(abs(phi[-1] - phi[0]) / 2 / np.pi),
                       iota_vmex=float(np.interp(sv, np.linspace(0, 1, ns), np.asarray(wout.iotaf)))))
    print(f"traced from VMEX surface s={sv:.3f}: lab rotation {it:+.5f} over {traced[-1]['turns']:.0f} turns; VMEX iota there {traced[-1]['iota_vmex']:+.5f}", flush=True)
json.dump(dict(surfaces=rows, traced=traced), open(f"{run.name}_{route}_vacuum_iota.json", "w"), indent=1)
