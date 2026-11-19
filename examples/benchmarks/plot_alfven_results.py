"""Regenerate the documentation figures from committed data; trace no particles.

From the checkout root: python -m examples.benchmarks.plot_alfven_results
"""

import argparse
import json
import os
from pathlib import Path


def plot_smoke(output, arrays):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = arrays["times"]*1e9
    birth_energy = arrays["equilibrium_energy"][:, :1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for marker in range(2):
        for name, style in (("equilibrium", "--"), ("wave", "-")):
            axes[0].plot(times, arrays[f"{name}_states"][marker, :, 0], style,
                         color=f"C{marker}", label=f"Marker {marker+1}: {name}")
            axes[1].plot(times, arrays[f"{name}_energy"][marker]/birth_energy[marker]-1,
                         style, color=f"C{marker}")
        axes[2].plot(times, arrays["wave_B_s"][marker], color=f"C{marker}")
    axes[0].set(ylabel="Toroidal flux s", title="Existing adaptive VMEC GC solver")
    axes[0].legend(fontsize=7)
    axes[1].set(ylabel="Kinetic energy change / birth energy", title="Wave work is allowed")
    axes[2].set(ylabel="Covariant radial B_s [T m]", title="Radial covariant field retained")
    for ax in axes:
        ax.set_xlabel("Time [ns]")
        ax.grid(alpha=.2)
    fig.suptitle("VMEC analytical-wave smoke test: four CPU trajectories")
    for extension in ("png", "svg"):
        fig.savefig(output/f"vmec_alfven_smoke.{extension}", dpi=160)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root/"examples/benchmarks/references")
    parser.add_argument("--output", type=Path, default=root/"docs/_static/analytical_perturbations")
    args = parser.parse_args()
    os.environ.update(JAX_PLATFORMS="cpu", ESSOS_CPU_DEVICES="1", CUDA_VISIBLE_DEVICES="-1")
    import numpy as np
    from examples.benchmarks.alfven_gc_verification import plot as plot_verification
    from examples.benchmarks.paul_boozer_vmec_loss_comparison import _plot as plot_losses

    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.data/"paul_loss_100.json").read_text())
    with np.load(args.data/"paul_loss_100.npz") as arrays:
        curves = [(row["model"], index % 2, arrays[f"loss_curve_{index}"], row["paper_loss_fraction"])
                  for index, row in enumerate(summary["cases"])]
        plot_losses(args.output, arrays["times"], curves, summary["particles_per_case"],
                    source_convention=summary["source_convention"],
                    representation=summary["vmec_representation"], stem="paul_loss_100")
    summary = json.loads((args.data/"gc_verification.json").read_text())
    with np.load(args.data/"gc_verification.npz") as arrays:
        plot_verification(args.output, summary, dict(arrays))
    with np.load(args.data/"vmec_alfven_smoke.npz") as arrays:
        plot_smoke(args.output, arrays)
    print(f"Regenerated three figures (PNG and SVG) in {args.output}; zero trajectories traced.")


if __name__ == "__main__":
    main()
