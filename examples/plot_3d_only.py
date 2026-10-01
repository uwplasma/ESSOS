#!/usr/bin/env python3
"""
Standalone 3D magnet map plot -- minimal memory footprint, run
independently of the other plot scripts.

Usage:
  python plot_3d_only.py <mag_file>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if len(sys.argv) != 2:
    sys.exit("Usage: python plot_3d_only.py <mag_file>")

MAG_FILE = Path(sys.argv[1])
RESULTS_DIR = Path(__file__).resolve().parent / "pm_opt_custom_loss_output"

import matplotlib.pyplot as plt

print("--- Loading magnet grid ---")
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
positions = np.asarray(pos_list, np.float64)
n_magnets = len(positions)

pho_optimized = np.load(RESULTS_DIR / "pho_optimized.npy")
print(f"{n_magnets} magnet sites, {int(np.sum(np.abs(pho_optimized) > 0.5))} active")

print("--- Plotting 3D magnet map ---")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

active = np.abs(pho_optimized) > 0.5
sc = ax.scatter(
    positions[active, 0], positions[active, 1], positions[active, 2],
    c=pho_optimized[active], cmap='RdBu_r', s=8, vmin=-1, vmax=1,
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
n_pos = int(np.sum(pho_optimized > 0.5))
n_neg = int(np.sum(pho_optimized < -0.5))
ax.set_title(f"custom_loss cold-start solution: {n_pos+n_neg} active magnets "
             f"(+{n_pos} / -{n_neg}) out of {n_magnets}", fontsize=12)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "magnet_map_3d.png", dpi=200, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/magnet_map_3d.png")
