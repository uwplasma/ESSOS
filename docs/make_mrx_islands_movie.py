"""3D movie: protons through the magnetic islands of a relaxed MRX state, next to the same protons in VMEX.

Left: the li383 state MRX relaxed after seeding the (5,1), (6,1) and (7,1) island chains. Right: VMEX's li383
equilibrium (ns = 65), which has nested surfaces. Both fields are scaled to 1 T on the axis. Six 10 keV
passing protons start on phi = 0 across the (6,1) chain; their guiding centers are drawn in 3D, and their
crossings of phi = 0 accumulate over each field's Poincare section. Writes island_movie.mp4.
"""
import os
os.environ["MRX_DTYPE"] = "float64"
import time
import jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from essos.fields import MRXField, Vmec
from essos.dynamics import Tracing, Particles
from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV

HERE = os.path.dirname(os.path.abspath(__file__))
# MRX_DIR: a clone of https://github.com/ToBlick/mrx after its tutorials 3 and 4; VMEX_WOUT: VMEX li383 at ns = 65.
MRX_DIR = os.environ.get("MRX_DIR", "mrx")
VMEX_WOUT = os.environ.get("VMEX_WOUT", "wout_li383_ns65.nc")
RUN = f"{MRX_DIR}/outputs/tutorials/4_li383_island_seed"
TMAX, SAVES, FRAMES, PERIODS = 3e-4, 6000, 240, 300
r0 = 0.552 + np.linspace(-0.12, 0.12, 6)            # across the (6,1) chain at r = 0.55, width 0.19
fields = {"MRX, relaxed with islands": MRXField.from_mrx(f"{MRX_DIR}/data/wout_li383_low_res_reference.nc",
                                                         f"{RUN}/checkpoints/state_000005.h5"),
          "VMEX (ns = 65), nested surfaces": Vmec(VMEX_WOUT)}
data = {}
for name, raw in fields.items():
    field = (1.0 / raw.AbsB(jnp.array([1e-6, 0., 0.]))) * raw
    s0 = jnp.asarray(r0**2)
    x0 = jnp.stack([s0, 0 * s0, 0 * s0], 1)
    t = time.perf_counter()
    lines = Tracing(field=field, model="FieldLineToroidal", initial_conditions=jnp.asarray(np.linspace(0.02, 0.95, 40)**2)[:, None] * jnp.array([1., 0, 0]),
                    maxtime=2 * np.pi * PERIODS / raw.nfp, times_to_trace=PERIODS + 1, atol=1e-10, rtol=1e-10)
    particles = Particles(initial_xyz=x0, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, energy=1e4 * ONE_EV, field=field,
                          initial_vparallel_over_v=jnp.full(6, 0.9))
    gc = Tracing(field=field, model="GuidingCenterAdaptative", particles=particles, maxtime=TMAX, times_to_trace=SAVES,
                 atol=1e-9, rtol=1e-9)
    xyz = np.asarray(gc.trajectories_xyz)
    print(name, f"traced in {time.perf_counter() - t:.0f} s", flush=True)
    theta = jnp.linspace(0, 2 * np.pi, 60)
    phi = jnp.linspace(0, 2 * np.pi, 120)
    surface = np.asarray(jax.vmap(lambda p: jax.vmap(lambda t: field.to_xyz(jnp.array([1., t, p])))(theta))(phi))
    # crossings of phi = 0 (mod 2 pi / nfp), linearly interpolated
    period = np.unwrap(np.arctan2(xyz[..., 1], xyz[..., 0]), axis=1) * raw.nfp / (2 * np.pi)
    i, j = np.nonzero(np.diff(np.floor(period), axis=1))
    w = ((np.floor(period[i, j + 1]) - period[i, j]) / (period[i, j + 1] - period[i, j]))[:, None]
    cross = (1 - w) * xyz[i, j] + w * xyz[i, j + 1]
    P = np.asarray(lines.trajectories_xyz)
    data[name] = dict(xyz=xyz, surface=surface, cross_R=np.hypot(cross[:, 0], cross[:, 1]), cross_Z=cross[:, 2],
                      cross_particle=i, cross_index=j, poincare=(np.hypot(P[..., 0], P[..., 1]).ravel(), P[..., 2].ravel()))

fig = plt.figure(figsize=(14, 10))
colors = plt.cm.plasma(np.linspace(0.05, 0.85, 6))
axes = []
for k, (name, d) in enumerate(data.items()):
    ax3 = fig.add_subplot(2, 2, k + 1, projection="3d")
    S = d["surface"]
    ax3.plot_surface(S[..., 0], S[..., 1], S[..., 2], color="lightsteelblue", alpha=0.12, linewidth=0)
    ax3.set_box_aspect((1, 1, 0.45)); ax3.set_axis_off(); ax3.set_title(name, fontsize=13)
    lim = np.abs(S[..., :2]).max(); ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-0.6, 0.6)
    trails = [ax3.plot([], [], [], color=c, lw=1.1)[0] for c in colors]
    heads = [ax3.plot([], [], [], "o", color=c, ms=4)[0] for c in colors]
    ax2 = fig.add_subplot(2, 2, k + 3)
    ax2.scatter(*d["poincare"], s=0.15, c="0.75", rasterized=True)
    dots = ax2.scatter([], [], s=6)
    ax2.set_aspect("equal"); ax2.set_xlabel("R [m]"); ax2.set_ylabel("Z [m]")
    ax2.set_title(r"field lines (grey) and proton crossings of $\phi = 0$", fontsize=10)
    axes.append((d, trails, heads, dots, ax3))
title = fig.suptitle("")


def draw(f):
    n = int(SAVES * (f + 1) / FRAMES)
    for d, trails, heads, dots, ax3 in axes:
        for p, (trail, head) in enumerate(zip(trails, heads)):
            x = d["xyz"][p, max(0, n - 600):n]
            trail.set_data_3d(x[:, 0], x[:, 1], x[:, 2])
            head.set_data_3d(x[-1:, 0], x[-1:, 1], x[-1:, 2])
        seen = d["cross_index"] < n
        dots.set_offsets(np.c_[d["cross_R"][seen], d["cross_Z"][seen]])
        dots.set_color(colors[d["cross_particle"][seen]])
        ax3.view_init(elev=28, azim=-60 + 0.5 * f)
    title.set_text(f"10 keV protons, 1 T on axis, t = {TMAX * (f + 1) / FRAMES * 1e3:.3f} ms")


writer = FFMpegWriter(fps=24, bitrate=4000)
with writer.saving(fig, f"{HERE}/readme_mrx_islands_full.mp4", dpi=110):
    for f in range(FRAMES):
        draw(f)
        writer.grab_frame()
fig.savefig(f"{HERE}/readme_mrx_islands.png", dpi=110)
print("wrote readme_mrx_islands_full.mp4; compress it to readme_mrx_islands.mp4 with ffmpeg -crf 30")
