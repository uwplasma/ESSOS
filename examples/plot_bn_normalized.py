#!/usr/bin/env python3
"""
Bn/|B| normalized contour plot -- the standard relative normal-field-
error convention in stellarator PM optimization literature, rather
than raw Bn in Tesla. Bn/|B| is dimensionless (often shown as a
percentage): how much of the total local field strength is "leaking"
through the plasma boundary.

Usage:
  python plot_bn_normalized.py <surf_file> <mag_file> <coil_file>
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

if len(sys.argv) != 4:
    sys.exit("Usage: python plot_bn_normalized.py <surf_file> <mag_file> <coil_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])
COIL_FILE = Path(sys.argv[3])

SURFACE_RANGE  = "half period"
SURFACE_NPHI   = 64
SURFACE_NTHETA = 64

RESULTS_DIR = Path(__file__).resolve().parent / "pm_opt_custom_loss_output"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from essos.fields import DipoleField


def load_surface(surf_file, surface_range, nphi, ntheta):
    from simsopt.geo import SurfaceRZFourier
    try:
        surface = SurfaceRZFourier.from_vmec_input(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    except Exception:
        surface = SurfaceRZFourier.from_focus(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    return surface


def load_coils_essos(coil_file):
    from simsopt.field import Coil, Current
    from simsopt.util.permanent_magnet_helper_functions import read_focus_coils
    from essos.coils import Coils_from_simsopt

    base_curves, base_currents0, ncoils = read_focus_coils(str(coil_file))
    total_current = float(np.sum([c.get_value() for c in base_currents0]))
    all_coils = [Coil(base_curves[i], Current(total_current / ncoils)) for i in range(ncoils)]
    return Coils_from_simsopt(all_coils, nfp=1, stellsym=False)


def compute_B_vec_essos(coils, surf_pts):
    from essos.fields import BiotSavart
    field = BiotSavart(coils)
    surf_pts_jax = jnp.asarray(surf_pts, jnp.float64)
    B_at_pts = jax.vmap(field.B)(surf_pts_jax)
    return np.asarray(B_at_pts, np.float64)


def load_magnet_grid(mag_file):
    pos_list, mom_list = [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.replace(",", " ").split()
            if len(tokens) < 12:
                continue
            try:
                x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
                m0      = float(tokens[7])
                az, pol = float(tokens[10]), float(tokens[11])
            except ValueError:
                continue
            pos_list.append((x, y, z))
            mom_list.append((m0*np.cos(az)*np.sin(pol), m0*np.sin(az)*np.sin(pol), m0*np.cos(pol)))
    return np.asarray(pos_list, np.float64), np.asarray(mom_list, np.float64)


print("--- Loading surface, coils ---")
surface = load_surface(SURF_FILE, SURFACE_RANGE, SURFACE_NPHI, SURFACE_NTHETA)
nfp, stellsym = int(surface.nfp), bool(surface.stellsym)

surf_xyz = np.asarray(surface.gamma(), np.float64)
surf_pts = surf_xyz.reshape(-1, 3)
surf_n   = np.asarray(surface.unitnormal(), np.float64).reshape(-1, 3)

essos_coils = load_coils_essos(COIL_FILE)
B_coils_vec = compute_B_vec_essos(essos_coils, surf_pts)   # (n_pts, 3)
del essos_coils
gc.collect()

print("--- Loading magnet grid + optimized pho ---")
positions, moments_raw = load_magnet_grid(MAG_FILE)
n_magnets = len(positions)
native_norms = np.linalg.norm(moments_raw, axis=1)
norms_safe = np.where(native_norms > 0, native_norms, 1.0)
orientations = np.where(native_norms[:, None] > 0, moments_raw / norms_safe[:, None], 0.0)

pho_optimized = np.load(RESULTS_DIR / "pho_optimized.npy")
scaled_moments = orientations * float(np.mean(native_norms[native_norms > 0])) * pho_optimized[:, None]

print("--- Building dipole field, evaluating B (chunked) ---")
dipole_field = DipoleField(
    jnp.asarray(positions, jnp.float32),
    jnp.asarray(scaled_moments, jnp.float32),
    jnp.zeros(n_magnets, jnp.float32),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)

CHUNK_SIZE = 128
surf_pts_jax = jnp.asarray(surf_pts, jnp.float64)
B_pm_vec = np.empty((len(surf_pts), 3), np.float64)
for i in range(0, len(surf_pts), CHUNK_SIZE):
    chunk = surf_pts_jax[i:i+CHUNK_SIZE]
    B_pm_vec[i:i+CHUNK_SIZE] = np.asarray(dipole_field.B(chunk), np.float64)
del dipole_field
gc.collect()

B_total_vec = B_coils_vec + B_pm_vec

# Bn = B . n_hat;  |B| = vector magnitude;  Bn/|B| = normalized relative error
Bn_coils = np.sum(B_coils_vec * surf_n, axis=1)
absB_coils = np.linalg.norm(B_coils_vec, axis=1)
BnoverB_coils = Bn_coils / absB_coils

Bn_total = np.sum(B_total_vec * surf_n, axis=1)
absB_total = np.linalg.norm(B_total_vec, axis=1)
BnoverB_total = Bn_total / absB_total

nphi_s, ntheta_s = surf_xyz.shape[0], surf_xyz.shape[1]
BnoverB_coils_2d = BnoverB_coils.reshape(nphi_s, ntheta_s)
BnoverB_total_2d = BnoverB_total.reshape(nphi_s, ntheta_s)

import matplotlib.pyplot as plt

phi_coords   = np.linspace(0, 1, nphi_s)
theta_coords = np.linspace(0, 1, ntheta_s)

abs_max_before = np.abs(BnoverB_coils_2d).max()
abs_max_after  = np.abs(BnoverB_total_2d).max()
levels_before  = np.linspace(-abs_max_before, abs_max_before, 21)
levels_after   = np.linspace(-abs_max_after, abs_max_after, 21)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
im1 = axes[0].contourf(phi_coords, theta_coords, BnoverB_coils_2d.T, levels=levels_before, cmap="RdBu_r", extend="both")
axes[0].set_title("Coils only (before)", fontsize=12)
axes[0].set_xlabel("phi"); axes[0].set_ylabel("theta")
axes[0].set_aspect("auto")
cbar1 = plt.colorbar(im1, ax=axes[0], label="B$_n$/|B|")
cbar1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
rms_before = float(np.sqrt(np.mean(BnoverB_coils_2d**2)))
axes[0].text(0.02, 0.97, f"RMS = {rms_before*100:.3f}%", transform=axes[0].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

im2 = axes[1].contourf(phi_coords, theta_coords, BnoverB_total_2d.T, levels=levels_after, cmap="RdBu_r", extend="both")
axes[1].set_title("Coils + optimized PMs (after, custom_loss cold-start)", fontsize=12)
axes[1].set_xlabel("phi")
axes[1].set_aspect("auto")
cbar2 = plt.colorbar(im2, ax=axes[1], label="B$_n$/|B|")
cbar2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
rms_after = float(np.sqrt(np.mean(BnoverB_total_2d**2)))
axes[1].text(0.02, 0.97, f"RMS = {rms_after*100:.3f}%", transform=axes[1].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.suptitle(f"Normalized normal field error B$_n$/|B|:  RMS {rms_before*100:.3f}% \u2192 {rms_after*100:.3f}%  "
             f"({rms_before/rms_after:.1f}\u00d7 reduction)", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "Bn_over_B_normalized.png", dpi=200, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/Bn_over_B_normalized.png")
print(f"\nRMS Bn/|B|: before={rms_before*100:.4f}%  after={rms_after*100:.4f}%")
