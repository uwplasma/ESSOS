#!/usr/bin/env python3
"""
Benchmark: essos DipoleField vs simsopt DipoleField.

Computes the magnetic field vector B at the plasma-surface xyz points from
the permanent-magnet dipole grid ONLY — no TF coils, no normal projection.
Both codes get identical inputs (same magnet positions, moments from the
loaded pho, same evaluation points), so any difference is purely
implementation.

Outputs (in benchmark_output/):
  - dipole_benchmark.png : 3 panels — |B| essos, |B| simsopt, |B_essos - B_simsopt|
                           plus a timing bar chart
  - printed summary: max/mean |dB|, relative error, wall times

Usage:
  python benchmark_dipole_field.py <surf_file> <mag_file>

Example (MUSE):
  python benchmark_dipole_field.py \
      /Users/joshuabourassa/essos_new/essos/input.muse \
      /Users/joshuabourassa/simsopt/tests/test_files/zot80.focus
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

if len(sys.argv) < 3:
    sys.exit("Usage: python benchmark_dipole_field.py <surf_file> <mag_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])

SURFACE_RANGE  = "half period"
SURFACE_NPHI   = 16
SURFACE_NTHETA = 16

OUTPUT_DIR = Path(__file__).resolve().parent / "output_dipole_field"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ESSOS_ROOT  = Path(__file__).resolve().parents[2]
SIMSOPT_SRC = ESSOS_ROOT.parent / "simsopt" / "src"
for p in [str(ESSOS_ROOT), str(SIMSOPT_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ============================================================
# LOAD SURFACE (evaluation points only — no normals needed)
# ============================================================
def load_surface_points(surf_file, surface_range, nphi, ntheta):
    from simsopt.geo import SurfaceRZFourier
    try:
        surface = SurfaceRZFourier.from_vmec_input(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    except Exception:
        surface = SurfaceRZFourier.from_focus(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    surf_pts = np.asarray(surface.gamma(), np.float64).reshape(-1, 3)
    return surface, surf_pts


# ============================================================
# LOAD MAGNET GRID (positions, moments, pho — robust parsing)
# ============================================================
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


print(f"--- Loading surface: {SURF_FILE.name} ({SURFACE_NPHI}x{SURFACE_NTHETA}, {SURFACE_RANGE}) ---")
surface, surf_pts = load_surface_points(SURF_FILE, SURFACE_RANGE, SURFACE_NPHI, SURFACE_NTHETA)
nfp      = int(surface.nfp)
stellsym = bool(surface.stellsym)
n_pts    = len(surf_pts)
print(f"Surface nfp={nfp}  stellsym={stellsym}  |  {n_pts} evaluation points")

print(f"\n--- Loading magnet grid: {MAG_FILE.name} ---")
positions, moments_raw, pho = load_magnet_grid(MAG_FILE)
n_magnets = len(positions)
print(f"{n_magnets} magnets  |  {int(np.sum(np.abs(pho) > 0.5))} active (|pho|>0.5)")

# The actual physical moment of each magnet is its file moment scaled by pho.
# Both codes get exactly these same scaled moments so the comparison is fair.
scaled_moments = moments_raw * pho[:, None]


# ============================================================
# ESSOS: B at surface points from DipoleField
# ============================================================
print(f"\n--- essos DipoleField ---")
from essos.fields import DipoleField as EssosDipoleField

t0 = time.time()
df_essos = EssosDipoleField(
    jnp.asarray(positions,      jnp.float64),
    jnp.asarray(scaled_moments, jnp.float64),
    jnp.zeros(n_magnets, jnp.float64),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)
t_essos_init = time.time() - t0

# Warm-up call: JAX JIT-compiles on first evaluation; time the SECOND
# (warm) call so we benchmark execution, not one-time compilation.
pts_jax = jnp.asarray(surf_pts, jnp.float64)
_ = np.asarray(df_essos.B(pts_jax))
t0 = time.time()
B_essos = np.asarray(df_essos.B(pts_jax), np.float64)
t_essos_eval = time.time() - t0
print(f"init: {t_essos_init:.2f}s   B eval: {t_essos_eval:.2f}s   B shape: {B_essos.shape}")


# ============================================================
# SIMSOPT: B at the same points from DipoleField
# ============================================================
print(f"\n--- simsopt DipoleField ---")
from simsopt.field import DipoleField as SimsoptDipoleField

t0 = time.time()
df_sims = SimsoptDipoleField(
    positions,
    scaled_moments.flatten(),
    nfp=nfp,
    coordinate_flag="cartesian",
    m_maxima=np.linalg.norm(moments_raw, axis=1),
)
t_sims_init = time.time() - t0

t0 = time.time()
df_sims.set_points(surf_pts)
B_sims = np.asarray(df_sims.B(), np.float64)
t_sims_eval = time.time() - t0
print(f"init: {t_sims_init:.2f}s   B eval: {t_sims_eval:.2f}s   B shape: {B_sims.shape}")


# ============================================================
# COMPARE
# ============================================================
dB      = B_essos - B_sims
mag_e   = np.linalg.norm(B_essos, axis=1)
mag_s   = np.linalg.norm(B_sims,  axis=1)
mag_d   = np.linalg.norm(dB,      axis=1)

max_diff = float(mag_d.max())
mean_diff = float(mag_d.mean())
max_B     = float(max(mag_e.max(), mag_s.max()))
rel_err   = max_diff / max_B if max_B > 0 else 0.0

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"max |B| (either code):     {max_B:.4e} T")
print(f"max |B_essos - B_simsopt|: {max_diff:.4e} T")
print(f"mean |dB|:                 {mean_diff:.4e} T")
print(f"relative error:            {rel_err:.4e}")
print(f"\nessos:   init {t_essos_init:6.2f}s   eval {t_essos_eval:6.2f}s   total {t_essos_init+t_essos_eval:6.2f}s")
print(f"simsopt: init {t_sims_init:6.2f}s   eval {t_sims_eval:6.2f}s   total {t_sims_init+t_sims_eval:6.2f}s")
speedup = (t_sims_init + t_sims_eval) / max(t_essos_init + t_essos_eval, 1e-12)
print(f"essos speedup vs simsopt: {speedup:.1f}x")


# ============================================================
# PLOT: |B| essos / |B| simsopt / |dB| side-by-side + timing bars
# ============================================================
import matplotlib.pyplot as plt

mag_e_2d = mag_e.reshape(SURFACE_NPHI, SURFACE_NTHETA)
mag_s_2d = mag_s.reshape(SURFACE_NPHI, SURFACE_NTHETA)
mag_d_2d = mag_d.reshape(SURFACE_NPHI, SURFACE_NTHETA)

phi   = np.linspace(0, 1, SURFACE_NPHI)
theta = np.linspace(0, 1, SURFACE_NTHETA)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

vmax_B = max(mag_e_2d.max(), mag_s_2d.max())
cf0 = axes[0].contourf(phi, theta, mag_e_2d.T, levels=30, cmap="viridis", vmin=0, vmax=vmax_B)
axes[0].set_title("|B| essos DipoleField")
axes[0].set_xlabel("phi"); axes[0].set_ylabel("theta")
plt.colorbar(cf0, ax=axes[0], label="|B| [T]")

cf1 = axes[1].contourf(phi, theta, mag_s_2d.T, levels=30, cmap="viridis", vmin=0, vmax=vmax_B)
axes[1].set_title("|B| simsopt DipoleField")
axes[1].set_xlabel("phi")
plt.colorbar(cf1, ax=axes[1], label="|B| [T]")

cf2 = axes[2].contourf(phi, theta, mag_d_2d.T, levels=30, cmap="magma")
axes[2].set_title(f"|B_essos - B_simsopt|  max={max_diff:.2e} T")
axes[2].set_xlabel("phi")
plt.colorbar(cf2, ax=axes[2], label="|dB| [T]")

labels = ["essos", "simsopt"]
init_times = [t_essos_init, t_sims_init]
eval_times = [t_essos_eval, t_sims_eval]
x = np.arange(len(labels))
axes[3].bar(x, init_times, 0.5, label="init", color="steelblue")
axes[3].bar(x, eval_times, 0.5, bottom=init_times, label="B eval", color="coral")
axes[3].set_xticks(x); axes[3].set_xticklabels(labels)
axes[3].set_ylabel("wall time [s]")
ratio_txt = f"essos {speedup:.1f}x faster" if speedup >= 1 else f"simsopt {1/speedup:.1f}x faster"
axes[3].set_title(f"Timing ({n_magnets} magnets, {n_pts} pts, warm JIT)\n{ratio_txt}")
axes[3].legend()
for i, total in enumerate([t_essos_init+t_essos_eval, t_sims_init+t_sims_eval]):
    axes[3].text(i, total, f"{total:.1f}s", ha="center", va="bottom")

plt.suptitle(f"Dipole field benchmark — {MAG_FILE.name} on {SURF_FILE.name} "
             f"(nfp={nfp}, stellsym={stellsym})  |  rel. err = {rel_err:.1e}", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dipole_benchmark.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"\nSaved {OUTPUT_DIR}/dipole_benchmark.png")
