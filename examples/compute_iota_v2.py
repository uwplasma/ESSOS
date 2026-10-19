import numpy as np

RESULTS_DIR = '/Users/joshuabourassa/essos_new/examples/pm_opt_custom_loss_output'
trajectories = np.load(f'{RESULTS_DIR}/trajectories.npy')
R0 = np.load(f'{RESULTS_DIR}/trace_R0.npy')

print(f"trajectories shape: {trajectories.shape}")
print(f"R0: {R0}")

def unwrapped_phi(traj):
    x, y = traj[:, 0], traj[:, 1]
    phi = np.arctan2(y, x)
    return np.unwrap(phi)

# Pick the trajectory with the MOST toroidal transits as a well-confined
# reference for estimating the axis -- NOT necessarily the innermost R0,
# since an escaped/scattered trajectory's time-average is meaningless.
transit_counts = []
for i in range(len(R0)):
    phi_unw = unwrapped_phi(trajectories[i])
    n_transits = abs(phi_unw[-1] - phi_unw[0]) / (2 * np.pi)
    transit_counts.append(n_transits)
transit_counts = np.array(transit_counts)

best_idx = np.argmax(transit_counts)
print(f"\nUsing R0={R0[best_idx]:.4f} (most toroidal transits: {transit_counts[best_idx]:.1f}) for axis estimation")

ref_traj = trajectories[best_idx]
R_ref = np.sqrt(ref_traj[:, 0]**2 + ref_traj[:, 1]**2)
Z_ref = ref_traj[:, 2]
R_axis = R_ref.mean()
Z_axis = Z_ref.mean()
print(f"Estimated axis: R_axis={R_axis:.4f}, Z_axis={Z_axis:.4f}")

def compute_iota(traj, R_axis, Z_axis):
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z - Z_axis, R - R_axis)

    phi_unwrapped = np.unwrap(phi)
    theta_unwrapped = np.unwrap(theta)

    n = len(phi_unwrapped)
    fit_slice = slice(n // 20, n - n // 20)
    coeffs = np.polyfit(phi_unwrapped[fit_slice], theta_unwrapped[fit_slice], 1)
    iota = coeffs[0]

    n_toroidal_transits = abs(phi_unwrapped[-1] - phi_unwrapped[0]) / (2 * np.pi)
    n_poloidal_transits = abs(theta_unwrapped[-1] - theta_unwrapped[0]) / (2 * np.pi)

    return iota, n_toroidal_transits, n_poloidal_transits

print(f"\n{'R0':>8} {'iota':>10} {'toroidal transits':>18} {'poloidal transits':>18}  {'note':>10}")
for i, r0 in enumerate(R0):
    iota, n_tor, n_pol = compute_iota(trajectories[i], R_axis, Z_axis)
    note = "ESCAPED?" if n_tor < 50 else ""
    print(f"{r0:>8.4f} {iota:>10.5f} {n_tor:>18.2f} {n_pol:>18.2f}  {note:>10}")

print("\nLow-order rationals for reference:")
for name, val in [("1/2", 0.5), ("1/3", 1/3), ("2/5", 0.4), ("1/5", 0.2),
                   ("2/7", 2/7), ("1/4", 0.25), ("3/7", 3/7), ("1/6", 1/6)]:
    print(f"  {name} = {val:.5f}")
