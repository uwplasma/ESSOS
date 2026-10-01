#!/usr/bin/env python3
"""
Resolution-sweep benchmark: essos vs simsopt dipole field.

Times THREE evaluation paths at each surface resolution (PM dipole grid
only — no coils):

  1. simsopt DipoleField.B()      — reference C++ implementation
  2. essos   DipoleField.B()      — chunked evaluation (memory-safe):
       essos's B() is a double-vmap over (points x all symmetric dipole
       copies); evaluating all points at once materializes an
       (n_points, n_dipoles_full, 3) tensor that exceeds laptop RAM
       above ~256 points (9.8 GB at 1024 points for 397k dipoles),
       causing swap-driven 100x+ slowdowns. Chunking points bounds
       memory and reveals the true compute scaling.
  3. essos   G-path               — compute_interaction_matrix build
       (once) then Bn = G @ pho per evaluation. This is essos's intended
       fast path for optimization: G build is a one-time cost and each
       subsequent evaluation is a single matvec.

Accuracy (max |B_essos - B_simsopt|) is recorded per resolution using
the chunked essos B path.

Usage:
  python benchmark_dipole_sweep.py <surf_file> <mag_file>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

if len(sys.argv) < 3:
    sys.exit("Usage: python benchmark_dipole_sweep.py <surf_file> <mag_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])

SURFACE_RANGE = "half period"
RESOLUTIONS   = [8, 16, 32, 64]
CHUNK_SIZE    = 128

OUTPUT_DIR = Path(__file__).resolve().parent / "output_dipole_field_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ESSOS_ROOT  = Path(__file__).resolve().parents[2]
SIMSOPT_SRC = ESSOS_ROOT.parent / "simsopt" / "src"
for p in [str(ESSOS_ROOT), str(SIMSOPT_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from essos.fields import DipoleField as EssosDipoleField
from simsopt.field import DipoleField as SimsoptDipoleField
from simsopt.geo import SurfaceRZFourier
import simsoptpp as sopp


def load_surface(surf_file, surface_range, nphi, ntheta):
    try:
        surface = SurfaceRZFourier.from_vmec_input(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    except Exception:
        surface = SurfaceRZFourier.from_focus(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    return surface


def load_magnet_grid(mag_file):
    pos_list, mom_list, pho_list = [], [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.replace(",", " ").split()
            if len(tokens) < 12:
                continue
            try:
                x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
                m0      = float(tokens[7])
                pho     = float(tokens[8])
                az, pol = float(tokens[10]), float(tokens[11])
            except ValueError:
                continue
            pos_list.append((x, y, z))
            mom_list.append((m0*np.cos(az)*np.sin(pol), m0*np.sin(az)*np.sin(pol), m0*np.cos(pol)))
            pho_list.append(pho)
    return (
        np.asarray(pos_list, np.float64),
        np.asarray(mom_list, np.float64),
        np.asarray(pho_list, np.float64),
    )


def essos_B_chunked(df, pts, chunk_size):
    """Evaluate essos DipoleField.B in memory-bounded chunks of points."""
    out = np.empty((len(pts), 3), np.float64)
    for i in range(0, len(pts), chunk_size):
        chunk = jnp.asarray(pts[i:i+chunk_size], jnp.float64)
        out[i:i+chunk_size] = np.asarray(df.B(chunk), np.float64)
    return out


print(f"--- Loading magnet grid: {MAG_FILE.name} ---")
positions, moments_raw, pho = load_magnet_grid(MAG_FILE)
n_magnets = len(positions)
scaled_moments = moments_raw * pho[:, None]
print(f"{n_magnets} magnets  |  {int(np.sum(np.abs(pho) > 0.5))} active")

surface0 = load_surface(SURF_FILE, SURFACE_RANGE, RESOLUTIONS[0], RESOLUTIONS[0])
nfp, stellsym = int(surface0.nfp), bool(surface0.stellsym)
print(f"Surface nfp={nfp}  stellsym={stellsym}")
print(f"essos B() chunk size: {CHUNK_SIZE} points\n")

df_essos = EssosDipoleField(
    jnp.asarray(positions,      jnp.float64),
    jnp.asarray(scaled_moments, jnp.float64),
    jnp.zeros(n_magnets, jnp.float64),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)
df_sims = SimsoptDipoleField(
    positions,
    scaled_moments.flatten(),
    nfp=nfp,
    coordinate_flag="cartesian",
    m_maxima=np.linalg.norm(moments_raw, axis=1),
)

native_norms = np.linalg.norm(moments_raw, axis=1)
norms_safe = np.where(native_norms > 0, native_norms, 1.0)
orientations = np.where(native_norms[:, None] > 0, moments_raw / norms_safe[:, None], 0.0)
M0_SCALE = float(np.mean(native_norms[native_norms > 0]))
unit_moments = orientations * M0_SCALE

df_essos_G = EssosDipoleField(
    jnp.asarray(positions,    jnp.float32),
    jnp.asarray(unit_moments, jnp.float32),
    jnp.zeros(n_magnets, jnp.float32),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)

results = {"n_pts": [], "t_sims": [], "t_essos_B": [], "t_G_build": [], "t_G_matvec": [], "t_A_build": [], "t_A_matvec": [], "maxdiff": []}

for res in RESOLUTIONS:
    surface = load_surface(SURF_FILE, SURFACE_RANGE, res, res)
    surf_pts = np.asarray(surface.gamma(), np.float64).reshape(-1, 3)
    surf_n   = np.asarray(surface.unitnormal(), np.float64).reshape(-1, 3)
    n_pts = len(surf_pts)

    # Two-pass protocol for BOTH codes: run the complete evaluation once
    # (absorbing one-time costs: JIT compilation, allocations, caches),
    # then time the second identical pass. This is the fair per-evaluation
    # cost for repeated use, which is the optimizer's cost model.

    # 1. simsopt B — warm pass, then timed pass
    df_sims.set_points(surf_pts)
    _ = np.asarray(df_sims.B(), np.float64)
    t0 = time.time()
    df_sims.set_points(surf_pts)
    B_sims = np.asarray(df_sims.B(), np.float64)
    t_sims = time.time() - t0

    # 2. essos B, chunked — full warm pass, then timed full pass
    _ = essos_B_chunked(df_essos, surf_pts, CHUNK_SIZE)
    t0 = time.time()
    B_essos = essos_B_chunked(df_essos, surf_pts, CHUNK_SIZE)
    t_essos_B = time.time() - t0

    # 3. essos G path: build G (one-time), then time the matvec
    t0 = time.time()
    G = np.asarray(df_essos_G.compute_interaction_matrix(
        jnp.asarray(surf_pts, jnp.float32),
        jnp.asarray(surf_n,   jnp.float32)), np.float32)
    t_G_build = time.time() - t0

    G64 = G.astype(np.float64)
    _ = G64 @ pho
    t0 = time.time()
    Bn_G = G64 @ pho
    t_G_matvec = time.time() - t0

    # 4. simsopt A-matrix path (apples-to-apples with essos G path):
    # sopp.dipole_field_Bn is the C++ kernel PermanentMagnetGrid uses to
    # build A_obj (N_pts x 3*N_dipoles, maps moment vector m -> Bn, with
    # nfp/stellsym handled internally). Two-pass: warm, then timed.
    # NOTE: A has 3x the columns of essos's G (full vector moments vs
    # fixed-orientation pho) — annotated on the plot, not corrected for.
    b_dummy = np.zeros(n_pts)
    def build_A():
        return sopp.dipole_field_Bn(
            np.ascontiguousarray(surf_pts),
            np.ascontiguousarray(positions),
            np.ascontiguousarray(surf_n),
            nfp, int(stellsym),
            np.ascontiguousarray(b_dummy),
            "cartesian", 1.0)
    _ = build_A()
    t0 = time.time()
    A_obj = build_A()
    t_A_build = time.time() - t0
    A_obj = np.asarray(A_obj).reshape(n_pts, n_magnets * 3)

    m_vec = scaled_moments.flatten()
    _ = A_obj @ m_vec
    t0 = time.time()
    Bn_A = A_obj @ m_vec
    t_A_matvec = time.time() - t0

    max_diff = float(np.linalg.norm(B_essos - B_sims, axis=1).max())

    results["n_pts"].append(n_pts)
    results["t_sims"].append(t_sims)
    results["t_essos_B"].append(t_essos_B)
    results["t_G_build"].append(t_G_build)
    results["t_G_matvec"].append(t_G_matvec)
    results["t_A_build"].append(t_A_build)
    results["t_A_matvec"].append(t_A_matvec)
    results["maxdiff"].append(max_diff)

    print(f"{res:3d}x{res:<3d} ({n_pts:5d} pts):  simsopt B {t_sims:7.3f}s   "
          f"essos B(chunked) {t_essos_B:7.3f}s   "
          f"G build {t_G_build:6.2f}s   G matvec {t_G_matvec*1000:7.2f}ms   "
          f"A build {t_A_build:6.2f}s   A matvec {t_A_matvec*1000:7.2f}ms   "
          f"max|dB| {max_diff:.2e} T")


import matplotlib.pyplot as plt

n_pts_arr = np.array(results["n_pts"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(n_pts_arr, results["t_G_build"],  "^-", label="essos G build",   color="seagreen",  linewidth=2)
axes[0].plot(n_pts_arr, results["t_A_build"],  "v--", label="simsopt A build (3x cols)", color="darkred", linewidth=2)
for n, tg, ta in zip(n_pts_arr, results["t_G_build"], results["t_A_build"]):
    axes[0].annotate(f"{tg:.2f}s", (n, tg), textcoords="offset points", xytext=(5, 5),  fontsize=8, color="seagreen")
    axes[0].annotate(f"{ta:.2f}s", (n, ta), textcoords="offset points", xytext=(5, -12), fontsize=8, color="darkred")
axes[0].set_xlabel("number of evaluation points")
axes[0].set_ylabel("wall time [s]")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_title(f"Interaction matrix build time — {n_magnets} magnets (nfp={nfp}, stellsym={stellsym})")
axes[0].legend(fontsize=9)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].plot(n_pts_arr, results["maxdiff"], "d-", color="purple", linewidth=2)
axes[1].set_xlabel("number of evaluation points")
axes[1].set_ylabel("max |B_essos - B_simsopt| [T]")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_title("Accuracy vs resolution (essos B vs simsopt B)")
axes[1].grid(True, which="both", alpha=0.3)

plt.suptitle(f"Dipole field benchmark sweep — {MAG_FILE.name} on {SURF_FILE.name}", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dipole_benchmark_sweep.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"\nSaved {OUTPUT_DIR}/dipole_benchmark_sweep.png")
