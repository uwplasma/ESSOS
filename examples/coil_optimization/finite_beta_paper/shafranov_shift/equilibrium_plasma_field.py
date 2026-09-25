"""Free-space field of a VMEC-format equilibrium's own current, at chosen points.

The current follows from Ampere's law on the covariant field in the WOUT file,
    mu0 sqrt(g) J^u = d_v B_s - d_s B_v,   mu0 sqrt(g) J^v = d_s B_u - d_u B_s,
so J dV = (sqrt(g) J^u e_u + sqrt(g) J^v e_v) ds du dv and no Jacobian division
is needed. The field is the Biot-Savart volume integral over the full torus.
A vacuum equilibrium supplies the quadrature/representation floor.
"""

from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

MU0 = 4e-7 * np.pi


def _read(path):
    with Dataset(path) as nc:
        get = lambda k: np.asarray(nc[k][:])
        return {k: get(k) for k in ("xm", "xn", "rmnc", "zmns", "xm_nyq", "xn_nyq",
                                   "bsubumnc", "bsubvmnc", "bsubsmns", "nfp", "ns")}


def current_elements(path, ntheta=96, nzeta_period=48):
    """Positions and J dV [A m] on interior full-mesh surfaces of the whole torus."""
    w = _read(path)
    nfp, ns = int(w["nfp"]), int(w["ns"])
    s = np.linspace(0, 1, ns)
    ds = s[1] - s[0]
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2 * np.pi * np.arange(nzeta_period * nfp) / (nzeta_period * nfp)
    ang = lambda m, n: m[:, None, None] * theta[None, :, None] - n[:, None, None] * zeta[None, None, :]
    a, an = ang(w["xm"], w["xn"]), ang(w["xm_nyq"], w["xn_nyq"])
    ca, sa, can, san = np.cos(a), np.sin(a), np.cos(an), np.sin(an)
    ev = lambda c, basis: np.einsum("jm,mtz->jtz", c, basis)
    m, n = w["xm"], w["xn"]
    R, Z = ev(w["rmnc"], ca), ev(w["zmns"], sa)
    Ru, Rv = ev(-w["rmnc"] * m, sa), ev(w["rmnc"] * n, sa)
    Zu, Zv = ev(w["zmns"] * m, ca), ev(-w["zmns"] * n, ca)
    mq, nq = w["xm_nyq"], w["xn_nyq"]
    Bu_h, Bv_h = ev(w["bsubumnc"], can), ev(w["bsubvmnc"], can)  # half mesh, row 0 unused
    Bs_u = ev(w["bsubsmns"] * mq, can)
    Bs_v = ev(-w["bsubsmns"] * nq, can)
    # Interior full-mesh surfaces j = 1..ns-2: radial derivative of half-mesh data.
    j = np.arange(1, ns - 1)
    dBu_ds = (Bu_h[j + 1] - Bu_h[j]) / ds
    dBv_ds = (Bv_h[j + 1] - Bv_h[j]) / ds
    sqrtg_Ju = (Bs_v[j] - dBv_ds) / MU0
    sqrtg_Jv = (dBu_ds - Bs_u[j]) / MU0
    cz, sz = np.cos(zeta), np.sin(zeta)
    e_u = np.stack((Ru[j] * cz, Ru[j] * sz, Zu[j]), -1)
    e_v = np.stack((Rv[j] * cz - R[j] * sz, Rv[j] * sz + R[j] * cz, Zv[j]), -1)
    x = np.stack((R[j] * cz, R[j] * sz, Z[j]), -1)
    weight = ds * (2 * np.pi / ntheta) * (2 * np.pi / len(zeta))
    JdV = (sqrtg_Ju[..., None] * e_u + sqrtg_Jv[..., None] * e_v) * weight
    return x.reshape(-1, 3), JdV.reshape(-1, 3)


def biot_savart(points, x, JdV, chunk=64):
    points = np.atleast_2d(points)
    out = np.empty_like(points)
    for i in range(0, len(points), chunk):
        d = points[i:i + chunk, None, :] - x[None]
        r3 = np.linalg.norm(d, axis=-1) ** 3
        out[i:i + chunk] = MU0 / (4 * np.pi) * np.sum(np.cross(JdV[None], d) / r3[..., None], 1)
    return out
