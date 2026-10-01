"""Render matched collisionless loss curves and CPU/GPU trace timings."""

from pathlib import Path
import sys
import json
from urllib.request import urlopen
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RECORD = "https://raw.githubusercontent.com/uwplasma/vmex/045db9f6/benchmarks/trace_accuracy.json"
record_text = Path(sys.argv[1]).read_text() if len(sys.argv)>1 else urlopen(RECORD, timeout=30).read()
BLUE, BLUE_LIGHT = "#2a78d6", "#86b6ef"

def make_trace_comparison_figure(out: Path) -> None:
    record = json.loads(record_text)["long_gpu"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), layout="constrained")
    curve = record["loss_curve"]
    for name, color in [("ESSOS", BLUE), ("CATAPULT", "#d89039")]:
        f = np.asarray(curve[name + "_lost"]) / record["particles"]
        error = np.sqrt(f * (1-f) / record["particles"])
        axes[0].plot(1e6*np.asarray(curve["times_s"]), 100*f, label=name, color=color)
        axes[0].fill_between(1e6*np.asarray(curve["times_s"]), 100*(f-error), 100*(f+error), color=color, alpha=.2)
    axes[0].set(xlabel="Time [µs]", ylabel="Lost [%]", title="8,192 births; 20 ms horizon")
    axes[0].legend(fontsize=8)
    rows = [record["results"][1], record["results"][2]]
    for offset, column, color, label in [(-.18, 3, BLUE_LIGHT, "Cold"), (.18, 4, BLUE, "Warm")]:
        bars = axes[1].barh(np.arange(2)+offset, [row[column] for row in rows], height=.34, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axes[1].set(yticks=[0,1], yticklabels=["ESSOS", "CATAPULT"], xlabel="Trace time [s]", title="8,192 births, 20 ms; RTX A4000", xlim=(0, 62))
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=130, pil_kwargs={"lossless": True})
    plt.close(fig)
    with Image.open(out) as image:
        image.convert("RGB").quantize(colors=64).convert("RGB").save(out, lossless=True)


make_trace_comparison_figure(Path(__file__).with_name("readme_boozer_speed.png"))
