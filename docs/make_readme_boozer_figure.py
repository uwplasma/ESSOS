"""Plot the corrected 128-alpha Boozer/VMEC example for the README.

    python docs/make_readme_boozer_figure.py

Numbers: examples/particle_tracing/trace_particles_boozer_vs_vmec.py (128
ARIES-CS alphas, 0.1 ms, corrected VMEC toroidal-flux sign). Times exclude
JIT compilation; eight Apple M2 CPU devices and the compiled kernel of PR #95.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [("ESSOS VMEC", 14.94, "13.3%"), ("ESSOS Boozer", 0.221, "10.9%")]

fig, ax = plt.subplots(figsize=(4.6, 2.2), constrained_layout=True)
names = [r[0] for r in rows]
times = [r[1] for r in rows]
bars = ax.barh(names, times, color=["#9aa5b1", "#1f6fb4"])
for bar, (_, t, loss) in zip(bars, rows):
    ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
            f"{t:g} s  (lost {loss})", va="center", fontsize=8)
ax.set_xscale("log")
ax.set_xlim(min(times) / 2, max(times) * 15)
ax.set_xlabel("wall time [s], compilation excluded", fontsize=8)
ax.set_title("128 ARIES-CS alphas, 0.1 ms", fontsize=9)
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
