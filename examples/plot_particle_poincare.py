#!/usr/bin/env python3
"""
Standalone plotting script for particle_trajectories.npy. Run as a
SEPARATE process (not imported/exec'd into the tracing script) so it
gets a fresh, small memory footprint -- JAX's own allocator does not
release memory back to the OS on `del`/`gc.collect()` within a single
process, which is why in-process plotting after a big trace has
repeatedly been OS-killed under memory pressure tonight, even after
explicitly deleting the large field/tracing objects.
"""
import sys
from pathlib import Path
import numpy as np

RESULTS_DIR = Path(sys.argv[1])
MAXTIME = float(sys.argv[2])
ENERGY_EV = float(sys.argv[3])
R0_VALS_STR = sys.argv[4]  # comma-separated string, just for the plot title

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from essos.dynamics import Tracing

traj = jnp.asarray(np.load(RESULTS_DIR / "particle_trajectories.npy"))
print(f"Loaded trajectories: shape={traj.shape}")

obj = object.__new__(Tracing)
obj.trajectories = traj
obj.times = jnp.linspace(0, MAXTIME, traj.shape[1], endpoint=True)

fig, ax = plt.subplots(figsize=(9, 9))
obj.poincare_plot(ax=ax, show=False)
ax.set_xlabel(r"$R$ [m]")
ax.set_ylabel(r"$Z$ [m]")
ax.set_title(f"Guiding Center particle trace, {ENERGY_EV:.0f}eV electron, R0=[{R0_VALS_STR}]")
ax.set_aspect("equal")
ax.grid(True, alpha=0.5)
plt.tight_layout()
out_path = RESULTS_DIR / "particle_poincare_plot.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
