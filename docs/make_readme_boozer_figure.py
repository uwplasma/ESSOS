"""Plot the recorded Boozer-tracing benchmarks for the README (no tracing is run).

    python docs/make_readme_boozer_figure.py

Numbers: examples/particle_tracing/trace_particles_boozer_vs_vmec.py (128 ARIES-CS
alphas, 0.1 ms) and the VMEX cross-code benchmark (1000 alphas, 10 ms). Wall
times exclude JIT compilation; all runs on 8 CPU cores.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

panels = [
    ("128 alphas, 0.1 ms (ARIES-CS)",
     [("ESSOS VMEC", 60.5, "13.3%"), ("ESSOS Boozer", 0.91, "14.8%")]),
    ("1000 alphas, 10 ms (VMEX benchmark)",
     [("SIMSOPT", 1079, "11.9%"), ("SIMPLE", 556, "12.4%"), ("ESSOS Boozer", 146, "12.8%")]),
]

fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.2), constrained_layout=True)
for ax, (title, rows) in zip(axes, panels):
    names = [r[0] for r in rows]
    times = [r[1] for r in rows]
    colors = ["#1f6fb4" if "Boozer" in n else "#9aa5b1" for n in names]
    bars = ax.barh(names, times, color=colors)
    for bar, (_, t, loss) in zip(bars, rows):
        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
                f"{t:g} s  (lost {loss})", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(min(times) / 2, max(times) * 30)
    ax.set_xlabel("wall time [s], compile excluded", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

out = Path(__file__).parent / "readme_boozer_speed.png"
fig.savefig(out, dpi=110)
try:
    from PIL import Image
    Image.open(out).convert("RGB").quantize(colors=32).save(out, optimize=True)
except ImportError:
    pass
