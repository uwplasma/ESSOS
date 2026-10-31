"""Plot QI ALM runs (coils, loss fraction, heat proxy at first loss) without tracing again.

Loss means first saved position outside the fixed classifier (signed distance <= 0), not the
soft optimization score. The heat proxy is E_total*v_perp [W m], not a surface heat-flux density.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS
from essos.fields import BiotSavart
from essos.split import ClassifierStop
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier

INPUT = Path(__file__).resolve().parents[1]/"input_files"
LABELS, COLORS = ("Initial", "Optimized"), ("royalblue", "goldenrod")


def diagnostics(states, coils, boundary):
    """Cumulative loss fraction and E*v_perp at first loss for physical-mu GC states (particles, saves, 5)."""
    states = np.asarray(states)
    distance = np.asarray(jax.jit(jax.vmap(boundary.evaluate_xyz))(
        jnp.asarray(states[..., :3].reshape(-1, 3)))).reshape(states.shape[:2])
    escaped = np.maximum.accumulate(distance <= 0., axis=1)
    lost = escaped[:, -1]
    hits = states[lost][np.arange(lost.sum()), np.argmax(escaped[lost], axis=1)]
    heat = np.empty(0)
    if len(hits):
        magnitude = np.asarray(jax.jit(jax.vmap(BiotSavart(coils).AbsB))(jnp.asarray(hits[:, :3])))
        perpendicular = np.maximum(hits[:, 4]*magnitude, 0.)
        heat = (.5*ALPHA_PARTICLE_MASS*hits[:, 3]**2+perpendicular)*np.sqrt(2*perpendicular/ALPHA_PARTICLE_MASS)
    return dict(fraction=escaped.mean(axis=0), hits=hits[:, :3], heat=heat, count=int(lost.sum()))


def plot_run(folder, surface, boundary, settings):
    coils = [Coils.from_json(str(folder/f"{name}_coils.json")) for name in ("initial", "optimized")]
    with np.load(folder/"optimization.npz") as data:
        states = [data[f"{name}_trajectories"] for name in ("initial", "optimized")]
    rows = [diagnostics(y, c, boundary) for y, c in zip(states, coils)]
    times = np.linspace(0., settings["horizon"], states[0].shape[1])
    surf = np.asarray(surface.gamma)
    heat = np.concatenate([r["heat"] for r in rows])
    norm = colors.Normalize(vmin=0., vmax=max(float(heat.max()) if heat.size else 0., 1e-30))
    fig = Figure(figsize=(17, 12), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    ax = fig.add_subplot(grid[0, :2], projection="3d")
    for model, label, color in zip(coils, LABELS, COLORS):
        for i, curve in enumerate(np.asarray(model.gamma)):
            ax.plot(*np.concatenate((curve, curve[:1])).T, color=color, label=label if i == 0 else None, alpha=.85)
    ax.set_title("Initial and optimized coils")
    ax.legend(loc="upper left")
    ax = fig.add_subplot(grid[0, 2:4])
    for row, label, color in zip(rows, LABELS, COLORS):
        ax.step(times, row["fraction"], where="post", color=color, linewidth=2., label=f"{label}: {row['count']}/{len(states[0])}")
    ax.set(title="Loss fraction vs time", xlabel="Time [s]", ylabel="Cumulative lost fraction")
    ax.legend()
    ax = fig.add_subplot(grid[0, 4:])
    for row, label, color in zip(rows, LABELS, COLORS):
        ax.hist(row["heat"], bins=np.linspace(norm.vmin, norm.vmax, 21), color=color, alpha=.6, label=label)
    ax.set(title="Heat proxy at first recorded loss", xlabel=r"$E\,v_\perp$ [W m]", ylabel="Particle count")
    ax.legend()
    axes = [fig.add_subplot(grid[1, 3*i:3*(i+1)], projection="3d") for i in range(2)]
    for ax, row, label in zip(axes, rows, LABELS):
        ax.plot_surface(*np.moveaxis(surf, -1, 0), color="0.7", alpha=.18, linewidth=0., shade=False, rcount=60, ccount=60)
        ax.scatter(*row["hits"].T, c=row["heat"], cmap="turbo", norm=norm, s=26, edgecolors="0.25", depthshade=False)
        ax.set_title(f"{label}: {row['count']} lost particles")
    fig.colorbar(ScalarMappable(norm=norm, cmap="turbo"), ax=axes, label=r"$E\,v_\perp$ [W m]", shrink=.65)
    fig.suptitle(f"QI GC: {json.loads((folder/'summary.json').read_text())['objective'].replace('_', ' ')} optimization; "
                 f"N={len(states[0])}; colours show E v_perp [W m], not heat flux per area")
    fig.savefig(folder/"overview.png", dpi=150)
    print(f"Saved {(folder/'overview.png').resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--wout", type=Path, help="Local copy of the run's classifier equilibrium")
    args = parser.parse_args()
    for folder in args.runs:
        settings = json.loads((folder/"summary.json").read_text())["settings"]
        wout = args.wout or Path(settings["wout"])
        wout = wout if wout.is_file() else INPUT/wout.name
        surface = SurfaceRZFourier.from_wout_file(str(wout), s=1, ntheta=settings["surface_points"],
            nphi=settings["surface_points"], range_torus="full torus")
        boundary = ClassifierStop(SurfaceClassifier(surface, h=settings["classifier_h"],
            padding=max(.1, 1.5*settings["safety_margin"])))
        plot_run(folder, surface, boundary, settings)


if __name__ == "__main__":
    main()
