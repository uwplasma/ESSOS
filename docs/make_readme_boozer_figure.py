"""Plot matched 8,192-alpha, 20 ms cold/warm RTX A4000 trace times."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ESSOS 7200cc7, JAX 0.9.2, fresh processes with compilation caches disabled.
# CATAPULT released and opt-in FIRM3D #90; field setup excluded.
names = ["ESSOS lookup", "ESSOS compact", "CATAPULT", "CATAPULT regular axis"]
cold = [76.521, 49.589, 34.17, 36.61]
warm = [69.178, 37.354, 34.48, 36.88]
y = np.arange(len(names))
fig, ax = plt.subplots(figsize=(6.5, 2.8), constrained_layout=True)
for offset, values, color, label in [(-0.18, cold, "#86b6ef", "Cold"),
                                      (0.18, warm, "#2a78d6", "Warm")]:
    bars = ax.barh(y + offset, values, height=0.34, color=color, label=label)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlim(0, max(cold) * 1.3)
ax.set_xlabel("Trace wall time [s]; field setup excluded")
ax.set_title("8,192 common alpha births, 20 ms; RTX A4000")
ax.legend(loc="lower right", fontsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.savefig(Path(__file__).with_name("readme_boozer_speed.png"), dpi=140)
