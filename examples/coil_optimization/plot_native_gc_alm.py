"""Plot downloaded QI ALM runs without tracing or differentiating again.

Loss means first saved position outside the fixed classifier (signed distance
<= 0), not the soft optimization score or an exact wall-impact diagnostic.
The heat proxy is E_total*v_perp [W m], not a surface heat-flux density.
"""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

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
PAIR_COLORS = ("royalblue", "goldenrod")
LABELS = ("Initial", "Optimized")


def diagnostics(states, coils, boundary, softness):
    """Use the physical-mu GC states and the run's original signed distance."""
    states = np.asarray(states)
    if states.ndim != 3 or states.shape[-1] != 5 or not np.isfinite(states).all():
        raise ValueError("Expected finite (particles, saves, 5) Cartesian physical-mu GC states")
    distance = np.asarray(jax.jit(jax.vmap(boundary.evaluate_xyz))(
        jnp.asarray(states[..., :3].reshape(-1, 3)))).reshape(states.shape[:2])
    escaped = np.maximum.accumulate(distance <= 0., axis=1)
    lost = escaped[:, -1]
    first = np.argmax(escaped[lost], axis=1)
    hits = states[lost][np.arange(np.count_nonzero(lost)), first]
    if len(hits):
        magnitude = np.asarray(jax.jit(jax.vmap(BiotSavart(coils).AbsB))(jnp.asarray(hits[:, :3])))
        perpendicular = np.maximum(hits[:, 4]*magnitude, 0.)
        energy = .5*ALPHA_PARTICLE_MASS*hits[:, 3]**2+perpendicular
        heat = energy*np.sqrt(2*perpendicular/ALPHA_PARTICLE_MASS)
    else:
        heat = np.empty(0)
    # Stable sigmoid, matching the soft score but keeping it distinct from counts.
    z = -softness*distance
    probability = np.exp(-np.logaddexp(0., -z))
    return dict(fraction=escaped.mean(axis=0), hits=hits[:, :3], heat=heat,
                count=int(lost.sum()), soft=float(probability.max(axis=1).mean()))


def view3d(ax, points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = .06*np.maximum(hi-lo, 1.)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
           xlim=(lo[0]-pad[0], hi[0]+pad[0]),
           ylim=(lo[1]-pad[1], hi[1]+pad[1]),
           zlim=(lo[2]-pad[2], hi[2]+pad[2]))
    ax.set_box_aspect(np.maximum(hi-lo, 1.), zoom=.88)
    ax.view_init(elev=25, azim=-55)


def save(fig, path):
    # Atomic replacement also allows regenerating a PNG displayed by Windows.
    with NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        fig.savefig(temporary, dpi=180, facecolor="white", bbox_inches="tight")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        fig.clear()


def plot_run(folder, surface, boundary, settings):
    coils = [Coils.from_json(str(folder/f"{name}_coils.json"))
             for name in ("initial", "optimized")]
    with np.load(folder/"optimization.npz") as data:
        states = [data[f"{name}_trajectories"] for name in ("initial", "optimized")]
        for model, name in zip(coils, ("initial", "optimized")):
            np.testing.assert_allclose(model.dofs, data[f"{name}_dofs"], rtol=1e-12, atol=1e-12)
    if states[0].shape != states[1].shape or not np.array_equal(states[0][:, 0, :3], states[1][:, 0, :3]):
        raise ValueError("Initial and optimized runs must use matching saved particles and birth positions")
    if states[0].shape[:2] != (settings["particles"], settings["saves"]):
        raise ValueError("Saved trajectory dimensions disagree with summary settings")
    rows = [diagnostics(y, c, boundary, settings["softness"]) for y, c in zip(states, coils)]
    report = json.loads((folder/"summary.json").read_text())
    for name, row in zip(("initial", "final"), rows):
        np.testing.assert_allclose(row["soft"], report[name]["loss_fraction"], rtol=1e-6, atol=1e-8,
                                   err_msg="Reconstructed classifier disagrees with saved run")
    times = np.linspace(0., settings["horizon"], states[0].shape[1])
    curves = [np.asarray(c.gamma) for c in coils]
    surf = np.asarray(surface.gamma)
    coil_points = np.concatenate([g.reshape(-1, 3) for g in curves])
    hit_points = np.concatenate([surf.reshape(-1, 3)]+[r["hits"] for r in rows])
    all_heat = np.concatenate([r["heat"] for r in rows])
    norm = colors.Normalize(vmin=0., vmax=max(float(all_heat.max()) if all_heat.size else 0., 1e-30))
    bins = np.linspace(norm.vmin, norm.vmax, 21)

    def draw_coils(ax):
        for g, label, color in zip(curves, LABELS, PAIR_COLORS):
            for i, curve in enumerate(g):
                ax.plot(*np.concatenate((curve, curve[:1])).T, color=color,
                        label=label if i == 0 else None, linewidth=1., alpha=.85)
        view3d(ax, coil_points)
        ax.set_title("Initial and optimized coils")
        ax.legend(loc="upper left")

    def draw_loss(ax):
        for row, label, color in zip(rows, LABELS, PAIR_COLORS):
            ax.step(times, row["fraction"], where="post", color=color, linewidth=2.,
                    label=f"{label}: {row['count']}/{len(states[0])}")
        ax.set(title="Loss fraction vs time", xlabel="Time [s]", ylabel="Cumulative lost fraction", ylim=(0., max(.05, 1.12*max(r["fraction"][-1] for r in rows))))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.grid(alpha=.2)
        ax.legend()

    def draw_histogram(ax):
        for row, label, color in zip(rows, LABELS, PAIR_COLORS):
            ax.hist(row["heat"], bins=bins, color=color, alpha=.6, label=label, edgecolor="white")
        ax.set(title="Heat proxy at first recorded loss", xlabel=r"$E\,v_\perp$ [W m]", ylabel="Particle count")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend()

    def draw_hits(ax, i):
        ax.plot_surface(*np.moveaxis(surf, -1, 0), color="0.7", alpha=.18,
                        linewidth=0., shade=False, rcount=60, ccount=60)
        row = rows[i]
        ax.scatter(*row["hits"].T, c=row["heat"], cmap="turbo", norm=norm,
                   s=26, edgecolors="0.25", linewidths=.4, depthshade=False)
        view3d(ax, hit_points)
        ax.set_title(f"{LABELS[i]}: {row['count']} lost particles")

    out = folder/"comparison_plots"
    out.mkdir(exist_ok=True)
    objective = report["objective"]
    title = f"QI GC + collisions: {objective.replace('_', ' ')} optimization; N={len(states[0])}"
    note = ("Loss locations/energies: first saved point outside the physical classifier; "
            f"{len(times)} saves, spacing {np.diff(times)[0]*1e6:.2f} microseconds.\n"
            "Colours show E v_perp [W m], not heat flux per unit area.")

    fig = Figure(figsize=(10, 8), layout="constrained")
    draw_coils(fig.add_subplot(projection="3d"))
    save(fig, out/"coil_comparison.png")
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    draw_loss(fig.add_subplot())
    save(fig, out/"loss_fraction.png")
    fig = Figure(figsize=(17, 6.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=.15, wspace=.08)
    draw_histogram(fig.add_subplot(131))
    axes = [fig.add_subplot(132+i, projection="3d") for i in range(2)]
    for i, ax in enumerate(axes):
        draw_hits(ax, i)
    fig.colorbar(ScalarMappable(norm=norm, cmap="turbo"), ax=axes,
                 label=r"$E\,v_\perp$ [W m]", shrink=.65, pad=.02)
    fig.suptitle(title+"\n"+note, fontsize=11)
    save(fig, out/"lost_heat_comparison.png")

    fig = Figure(figsize=(17, 12), layout="constrained")
    fig.get_layout_engine().set(w_pad=.15, wspace=.08)
    grid = fig.add_gridspec(2, 6)
    draw_coils(fig.add_subplot(grid[0, :2], projection="3d"))
    draw_loss(fig.add_subplot(grid[0, 2:4]))
    draw_histogram(fig.add_subplot(grid[0, 4:]))
    axes = [fig.add_subplot(grid[1, 3*i:3*(i+1)], projection="3d") for i in range(2)]
    for i, ax in enumerate(axes):
        draw_hits(ax, i)
    fig.colorbar(ScalarMappable(norm=norm, cmap="turbo"), ax=axes,
                 label=r"$E\,v_\perp$ [W m]", shrink=.65, pad=.02)
    fig.suptitle(title+"\n"+note, fontsize=12)
    save(fig, out/"overview.png")
    for label, row in zip(LABELS, rows):
        print(f"{folder.name}: {label}: losses {row['count']}/{len(states[0])}; "
              f"soft score {row['soft']:.9f}; mean lost E*v_perp "
              f"{row['heat'].mean() if row['heat'].size else float('nan'):.9g} W m", flush=True)
    print(f"Saved {out.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--wout", type=Path, help="Local copy of the run's unchanged classifier equilibrium")
    args = parser.parse_args()
    cache = {}
    for folder in args.runs:
        settings = json.loads((folder/"summary.json").read_text())["settings"]
        wout = args.wout or Path(settings["wout"])
        if not wout.is_file():
            wout = INPUT/wout.name
        if not wout.is_file():
            raise FileNotFoundError("Original classifier equilibrium missing; supply --wout")
        key = (str(wout.resolve()), settings["surface_points"], settings["classifier_h"], settings["safety_margin"])
        if key not in cache:
            print(f"Preparing the original fixed classifier from {wout.name}", flush=True)
            surface = SurfaceRZFourier.from_wout_file(str(wout), s=1,
                ntheta=key[1], nphi=key[1], range_torus="full torus")
            boundary = ClassifierStop(SurfaceClassifier(surface, h=key[2], padding=max(.1, 1.5*key[3])))
            cache[key] = surface, boundary
        plot_run(folder, *cache[key], settings)


if __name__ == "__main__":
    main()
