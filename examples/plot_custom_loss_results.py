#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if len(sys.argv) != 4:
    sys.exit("Usage: python plot_custom_loss_results.py <surf_file> <mag_file> <coil_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])
COIL_FILE = Path(sys.argv[3])

SURFACE_RANGE  = "half period"
SURFACE_NPHI   = 64
SURFACE_NTHETA = 64

RESULTS_DIR = Path(__file__).resolve().parent / "pm_opt_custom_loss_output"
FB_ONLY_STEPS = 8000 
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


def compute_Bn_fixed_essos(coils, surf_pts, surf_n):
    from essos.fields import BiotSavart
    field = BiotSavart(coils)
    surf_pts_jax = jnp.asarray(surf_pts, jnp.float64)
    B_at_pts = jax.vmap(field.B)(surf_pts_jax)
    return np.asarray(jnp.sum(B_at_pts * jnp.asarray(surf_n, jnp.float64), axis=1), np.float64)


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


import matplotlib.pyplot as plt

# 1. Convergence history (fB, fV, fD)

print("--- Plotting convergence history ---")
hist = np.load(RESULTS_DIR / "convergence_history.npz")
hist_fB, hist_fV, hist_fD = hist["fB"], hist["fV"], hist["fD"]
steps_axis = np.arange(1, len(hist_fB) + 1)

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(steps_axis, hist_fB, color="steelblue", linewidth=1)
axes[0].set_yscale("log")
axes[0].set_ylabel("fB [T$^2$m$^2$]")
axes[0].set_title("custom_loss cold-start, volume-matched: convergence history", fontsize=13)
axes[0].grid(alpha=0.3)

axes[1].plot(steps_axis, hist_fV, color="seagreen", linewidth=1)
axes[1].axhline(811.25, color="darkred", linestyle=":", label="Volume target (811.25 cm$^3$)")
axes[1].set_ylabel("fV [cm$^3$] (unique domain)")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

axes[2].plot(steps_axis, hist_fD, color="purple", linewidth=1)
axes[2].set_yscale("log")
axes[2].set_ylabel("fD (discreteness residual)")
axes[2].set_xlabel("iteration")
axes[2].grid(alpha=0.3)

for ax in axes:
    ax.axvline(FB_ONLY_STEPS, color="gray", linestyle="--", alpha=0.7)
axes[0].text(FB_ONLY_STEPS, axes[0].get_ylim()[1], "  Stage 2 (annealing) starts",
             va="top", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "convergence_history_detailed.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {RESULTS_DIR}/convergence_history_detailed.png")



print("\n--- Loading physics setup for Bn plot ---")
surface = load_surface(SURF_FILE, SURFACE_RANGE, SURFACE_NPHI, SURFACE_NTHETA)
nfp, stellsym = int(surface.nfp), bool(surface.stellsym)

surf_xyz = np.asarray(surface.gamma(), np.float64)
surf_pts = surf_xyz.reshape(-1, 3)
surf_n   = np.asarray(surface.unitnormal(), np.float64).reshape(-1, 3)

essos_coils = load_coils_essos(COIL_FILE)
Bn_fixed = compute_Bn_fixed_essos(essos_coils, surf_pts, surf_n)

positions, moments_raw, _ = load_magnet_grid(MAG_FILE)
n_magnets = len(positions)
native_norms = np.linalg.norm(moments_raw, axis=1)
norms_safe = np.where(native_norms > 0, native_norms, 1.0)
orientations = np.where(native_norms[:, None] > 0, moments_raw / norms_safe[:, None], 0.0)

pho_optimized = np.load(RESULTS_DIR / "pho_optimized.npy")
scaled_moments = orientations * float(np.mean(native_norms[native_norms > 0])) * pho_optimized[:, None]

dipole_field = DipoleField(
    jnp.asarray(positions, jnp.float32),
    jnp.asarray(scaled_moments, jnp.float32),
    jnp.zeros(n_magnets, jnp.float32),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)
B_dipole = np.asarray(dipole_field.B(jnp.asarray(surf_pts, jnp.float64)), np.float64)
Bn_pm = np.sum(B_dipole * surf_n, axis=1)

nphi_s, ntheta_s = surf_xyz.shape[0], surf_xyz.shape[1]
Bn_before = Bn_fixed.reshape(nphi_s, ntheta_s)
Bn_after  = (Bn_fixed + Bn_pm).reshape(nphi_s, ntheta_s)

phi_coords   = np.linspace(0, 1, nphi_s)
theta_coords = np.linspace(0, 1, ntheta_s)

abs_max_before = np.abs(Bn_before).max()
abs_max_after  = np.abs(Bn_after).max()
levels_before  = np.linspace(-abs_max_before, abs_max_before, 21)
levels_after   = np.linspace(-abs_max_after, abs_max_after, 21)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
im1 = axes[0].contourf(phi_coords, theta_coords, Bn_before.T, levels=levels_before, cmap="RdBu_r", extend="both")
axes[0].set_title("Coils only (before)", fontsize=12)
axes[0].set_xlabel("phi"); axes[0].set_ylabel("theta")
axes[0].set_aspect("auto")
plt.colorbar(im1, ax=axes[0], label="B\u00b7n [T]")
rms_before = float(np.sqrt(np.mean(Bn_before**2)))
axes[0].text(0.02, 0.97, f"RMS = {rms_before:.3e} T", transform=axes[0].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

im2 = axes[1].contourf(phi_coords, theta_coords, Bn_after.T, levels=levels_after, cmap="RdBu_r", extend="both")
axes[1].set_title("Coils + optimized PMs (after, custom_loss cold-start)", fontsize=12)
axes[1].set_xlabel("phi")
axes[1].set_aspect("auto")
plt.colorbar(im2, ax=axes[1], label="B\u00b7n [T]")
rms_after = float(np.sqrt(np.mean(Bn_after**2)))
axes[1].text(0.02, 0.97, f"RMS = {rms_after:.3e} T", transform=axes[1].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.suptitle(f"RMS: {rms_before:.3e} \u2192 {rms_after:.3e} T  ({rms_before/rms_after:.1f}\u00d7 reduction)", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "Bn_before_after.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {RESULTS_DIR}/Bn_before_after.png")



# 3. 3D magnet map

print("\n--- Plotting 3D magnet map ---")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

active = np.abs(pho_optimized) > 0.5
sc = ax.scatter(
    positions[active, 0], positions[active, 1], positions[active, 2],
    c=pho_optimized[active], cmap='RdBu_r', s=8, vmin=-1, vmax=1,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('pho')

try:
    for c in essos_coils:
        g = np.asarray(c.gamma[0])
        ax.plot(g[:, 0], g[:, 1], g[:, 2], color='gold', linewidth=1.5)
except Exception as e:
    print(f"(coils not plotted: {e})")

ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_axis_off()

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0]-radius, origin[0]+radius])
    ax.set_ylim3d([origin[1]-radius, origin[1]+radius])
    ax.set_zlim3d([origin[2]-radius, origin[2]+radius])

set_axes_equal(ax)
ax.view_init(elev=45, azim=45)
n_pos = int(np.sum(pho_optimized > 0.5))
n_neg = int(np.sum(pho_optimized < -0.5))
ax.set_title(f"custom_loss cold-start solution: {n_pos+n_neg} active magnets "
             f"(+{n_pos} / -{n_neg}) out of {n_magnets}", fontsize=12)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "magnet_map_3d.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {RESULTS_DIR}/magnet_map_3d.png")

print(f"\nAll plots saved to {RESULTS_DIR}/")
