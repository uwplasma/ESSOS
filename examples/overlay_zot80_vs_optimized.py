import sys
sys.path.insert(0, '/Users/joshuabourassa/essos_new')
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.dynamics import Tracing

OPT_DIR = '/Users/joshuabourassa/essos_new/examples/pm_opt_custom_loss_output'
ZOT80_DIR = '/Users/joshuabourassa/essos_new/examples/trace_zot80_output'

MAXTIME = 1290.0  # must match whatever maxtime the two saved traces were run with

traj_opt = jnp.asarray(np.load(f'{OPT_DIR}/trajectories.npy'))
traj_zot80 = jnp.asarray(np.load(f'{ZOT80_DIR}/trajectories.npy'))
R0_opt = np.load(f'{OPT_DIR}/trace_R0.npy')
R0_zot80 = np.load(f'{ZOT80_DIR}/trace_R0.npy')

print(f"Optimized: shape={traj_opt.shape}  R0={R0_opt}")
print(f"zot80:     shape={traj_zot80.shape}  R0={R0_zot80}")
print(f"R0 arrays match: {np.array_equal(R0_opt, R0_zot80)}")
if not np.array_equal(R0_opt, R0_zot80):
    print("WARNING: R0 arrays do not match -- this is not an apples-to-apples comparison.")
    print("Re-run both trace_fieldlines_custom_loss_result.py and trace_fieldlines_zot80.py")
    print("with the same R0 range before trusting this overlay.")

obj_opt = object.__new__(Tracing)
obj_opt.trajectories = traj_opt
obj_opt.times = jnp.linspace(0, MAXTIME, traj_opt.shape[1], endpoint=True)

obj_zot80 = object.__new__(Tracing)
obj_zot80.trajectories = traj_zot80
obj_zot80.times = jnp.linspace(0, MAXTIME, traj_zot80.shape[1], endpoint=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

for shift, ax, label in [(jnp.pi/2, axes[0], r"$\phi=\pi/2$"), (0.0, axes[1], r"$\phi=0$")]:
    obj_zot80.poincare_plot(shifts=[shift], ax=ax, show=False, color='red', s=6)
    obj_opt.poincare_plot(shifts=[shift], ax=ax, show=False, color='blue', s=6)
    ax.set_xlabel(r"$R$ [m]")
    ax.set_ylabel(r"$Z$ [m]")
    ax.set_title(f"{label}  (red=zot80, blue=optimized custom_loss)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.5)

plt.tight_layout()
OUT_PATH = '/Users/joshuabourassa/essos_new/examples/poincare_overlay_zot80_vs_optimized.png'
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f"Saved {OUT_PATH}")
