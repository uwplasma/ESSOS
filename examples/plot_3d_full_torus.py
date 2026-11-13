#!/usr/bin/env python3
"""
Full-torus 3D magnet map -- expands the unique (quarter-period) magnet
grid to all symmetric copies (nfp rotations x stellarator-symmetry
reflection) before plotting, matching the transform used in
compute_G_symmetric.py / DipoleField.compute_interaction_matrix
(verified against simsopt to ~1e-9).

pho is NOT sign-flipped across symmetric copies: it is a scalar
strength relative to each copy's own (correctly transformed) local
orientation, not a lab-frame sign -- the same convention already used
when G sums contributions from all copies with a single shared pho.

Usage:
  python plot_3d_full_torus.py <mag_file> [surf_file]

If surf_file is given, nfp/stellsym are read from it; otherwise
defaults to MUSE's nfp=2, stellsym=True.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if len(sys.argv) not in (2, 3):
    sys.exit("Usage: python plot_3d_full_torus.py <mag_file> [surf_file]")

MAG_FILE = Path(sys.argv[1])
RESULTS_DIR = Path(__file__).resolve().parent / "pm_opt_custom_loss_output"

if len(sys.argv) == 3:
    SURF_FILE = Path(sys.argv[2])
    from simsopt.geo import SurfaceRZFourier
    try:
        surface = SurfaceRZFourier.from_focus(str(SURF_FILE), range="full torus", nphi=8, ntheta=8)
    except Exception:
        surface = SurfaceRZFourier.from_vmec_input(str(SURF_FILE), range="full torus", nphi=8, ntheta=8)
    nfp, stellsym = int(surface.nfp), bool(surface.stellsym)
else:
    nfp, stellsym = 2, True
    print("No surface file given -- defaulting to MUSE's nfp=2, stellsym=True")

print(f"nfp={nfp}  stellsym={stellsym}  ->  {nfp * (2 if stellsym else 1)} symmetric copies")

import matplotlib.pyplot as plt

print("--- Loading magnet grid (unique domain) ---")
pos_list = []
with open(str(MAG_FILE), encoding="utf-8") as f:
    for line in f.readlines():
        tokens = line.replace(",", " ").split()
        if len(tokens) < 12:
            continue
        try:
            x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
        except ValueError:
            continue
        pos_list.append((x, y, z))
positions_unique = np.asarray(pos_list, np.float64)
n_unique = len(positions_unique)

pho_unique = np.load(RESULTS_DIR / "pho_optimized.npy")
print(f"{n_unique} unique magnet sites, {int(np.sum(np.abs(pho_unique) > 0.5))} active")


def expand_to_full_torus(positions, pho, nfp, stellsym):
    """Same transform as compute_G_symmetric.py: reflect y,z for
    stellarator symmetry, then rotate nfp times about the z-axis.
    pho is repeated unchanged for each copy (scalar strength relative
    to each copy's own local orientation)."""
    stell_list = [1.0, -1.0] if stellsym else [1.0]
    all_pos, all_pho = [], []
    for stell in stell_list:
        pos_s = positions * np.array([1.0, stell, stell])
        for i in range(nfp):
            angle = 2 * np.pi * i / nfp
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            pos_r = pos_s @ R.T
            all_pos.append(pos_r)
            all_pho.append(pho)
    return np.concatenate(all_pos, axis=0), np.concatenate(all_pho, axis=0)


print("--- Expanding to full torus ---")
positions_full, pho_full = expand_to_full_torus(positions_unique, pho_unique, nfp, stellsym)
print(f"{len(positions_full)} total magnet sites after symmetry expansion "
      f"({int(np.sum(np.abs(pho_full) > 0.5))} active)")

print("--- Plotting ---")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

active = np.abs(pho_full) > 0.5
sc = ax.scatter(
    positions_full[active, 0], positions_full[active, 1], positions_full[active, 2],
    c=pho_full[active], cmap='RdBu_r', s=4, vmin=-1, vmax=1,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('pho')

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

n_pos = int(np.sum(pho_full > 0.5))
n_neg = int(np.sum(pho_full < -0.5))
n_total_sites = len(positions_full)
ax.set_title(f"Full-torus magnet map (nfp={nfp}, stellsym={stellsym}): "
             f"{n_pos+n_neg} active (+{n_pos} / -{n_neg}) out of {n_total_sites}", fontsize=12)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "magnet_map_3d_full_torus.png", dpi=200, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/magnet_map_3d_full_torus.png")
