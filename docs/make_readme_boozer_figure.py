"""Plot matched 8,192-alpha, 20 ms cold/warm RTX A4000 trace times."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ESSOS 1307356 + mass fix 9ab3e36, JAX 0.9.2; compilation caches disabled.
# FIRM3D 4dbeb5e + non-MPI guard #91; mass 6.6446573450e-27 kg; setup excluded.
# Record: https://github.com/uwplasma/vmex/pull/516 (benchmarks/trace_accuracy.json).
names = ["ESSOS lookup", "ESSOS compact", "CATAPULT regular axis"]
cold = [76.399, 49.287, 36.694]
warm = [69.062, 37.418, 37.044]
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
