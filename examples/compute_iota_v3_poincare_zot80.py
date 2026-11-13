import sys
sys.path.insert(0, '/Users/joshuabourassa/essos_new')
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from essos.dynamics import roots

RESULTS_DIR = '/Users/joshuabourassa/essos_new/examples/trace_zot80_output'
trajectories = np.load(f'{RESULTS_DIR}/trajectories.npy')
R0 = np.load(f'{RESULTS_DIR}/trace_R0.npy')

print(f"trajectories shape: {trajectories.shape}")
print(f"R0: {R0}")

maxtime = 1290.0
n_pts = trajectories.shape[1]
times = jnp.linspace(0, maxtime, n_pts, endpoint=True)

SHIFT = 0.0  # phi=0 slice

def get_crossings(traj):
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    phi = jnp.arctan2(y, x)
    T_cross = roots(times, phi, shift=SHIFT)
    # de-duplicate trailing zero-padding roots() can leave behind
    T_cross = np.asarray(T_cross)
    T_cross = np.unique(T_cross[T_cross > 0])
    R_cross = np.interp(T_cross, np.asarray(times), np.sqrt(x**2 + y**2))
    Z_cross = np.interp(T_cross, np.asarray(times), z)
    return T_cross, R_cross, Z_cross

# Find crossings for every trajectory, and how many crossings each has
# (a proxy for how many toroidal transits it actually completed).
all_crossings = []
n_crossings = []
for i in range(len(R0)):
    T_c, R_c, Z_c = get_crossings(trajectories[i])
    all_crossings.append((T_c, R_c, Z_c))
    n_crossings.append(len(T_c))
n_crossings = np.array(n_crossings)

print(f"\nCrossing counts per R0: {n_crossings}")

# Use the trajectory with the most crossings (best-confined) to estimate
# the LOCAL axis position at this phi=0 slice specifically.
best_idx = np.argmax(n_crossings)
_, R_c_best, Z_c_best = all_crossings[best_idx]
R_axis = R_c_best.mean()
Z_axis = Z_c_best.mean()
print(f"Using R0={R0[best_idx]:.4f} ({n_crossings[best_idx]} crossings) for axis estimate")
print(f"Local axis at phi=0: R_axis={R_axis:.4f}, Z_axis={Z_axis:.4f}")

print(f"\n{'R0':>8} {'n_crossings':>12} {'iota':>10}  note")
for i, r0 in enumerate(R0):
    T_c, R_c, Z_c = all_crossings[i]
    if len(R_c) < 10:
        print(f"{r0:>8.4f} {len(R_c):>12}  {'--':>10}  too few crossings (escaped)")
        continue
    theta = np.arctan2(Z_c - Z_axis, R_c - R_axis)
    theta_unwrapped = np.unwrap(theta)
    # iota = average poloidal angle advance PER CROSSING (i.e. per toroidal transit) / 2pi
    iota = (theta_unwrapped[-1] - theta_unwrapped[0]) / (len(theta_unwrapped) - 1) / (2 * np.pi)
    print(f"{r0:>8.4f} {len(R_c):>12} {iota:>10.5f}")

print("\nLow-order rationals for reference:")
for name, val in [("1/2", 0.5), ("1/3", 1/3), ("2/5", 0.4), ("1/5", 0.2),
                   ("2/7", 2/7), ("1/4", 0.25), ("3/7", 3/7), ("1/6", 1/6),
                   ("1/7", 1/7), ("1/8", 1/8)]:
    print(f"  {name} = {val:.5f}")
