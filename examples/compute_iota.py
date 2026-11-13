import numpy as np

RESULTS_DIR = '/Users/joshuabourassa/essos_new/examples/pm_opt_custom_loss_output'
trajectories = np.load(f'{RESULTS_DIR}/trajectories.npy')
R0 = np.load(f'{RESULTS_DIR}/trace_R0.npy')

print(f"trajectories shape: {trajectories.shape}")
print(f"R0: {R0}")

# Estimate the magnetic axis position (R_axis, Z_axis) from the innermost
# trajectory's time-averaged (R, Z) -- a field line very close to the axis
# stays close to it throughout, so its mean position is a good axis estimate.
innermost = trajectories[0]  # smallest R0
R_innermost = np.sqrt(innermost[:, 0]**2 + innermost[:, 1]**2)
Z_innermost = innermost[:, 2]
R_axis = R_innermost.mean()
Z_axis = Z_innermost.mean()
print(f"\nEstimated axis: R_axis={R_axis:.4f}, Z_axis={Z_axis:.4f}")

def compute_iota(traj, R_axis, Z_axis):
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z - Z_axis, R - R_axis)

    # Unwrap both angles to track total winding, not just [-pi, pi] wrapped values
    phi_unwrapped = np.unwrap(phi)
    theta_unwrapped = np.unwrap(theta)

    # iota = net poloidal angle traversed / net toroidal angle traversed
    # Use a linear fit (theta vs phi) for robustness against local wiggles.
    # Fit only over the well-sampled middle portion to avoid edge effects.
    n = len(phi_unwrapped)
    fit_slice = slice(n // 20, n - n // 20)
    coeffs = np.polyfit(phi_unwrapped[fit_slice], theta_unwrapped[fit_slice], 1)
    iota = coeffs[0]

    n_toroidal_transits = (phi_unwrapped[-1] - phi_unwrapped[0]) / (2 * np.pi)
    n_poloidal_transits = (theta_unwrapped[-1] - theta_unwrapped[0]) / (2 * np.pi)

    return iota, n_toroidal_transits, n_poloidal_transits

print(f"\n{'R0':>8} {'iota':>10} {'toroidal transits':>18} {'poloidal transits':>18}")
for i, r0 in enumerate(R0):
    iota, n_tor, n_pol = compute_iota(trajectories[i], R_axis, Z_axis)
    print(f"{r0:>8.4f} {iota:>10.5f} {n_tor:>18.2f} {n_pol:>18.2f}")

print("\nLow-order rationals for reference:")
for name, val in [("1/2", 0.5), ("1/3", 1/3), ("2/5", 0.4), ("1/5", 0.2),
                   ("2/7", 2/7), ("1/4", 0.25), ("3/7", 3/7), ("1/6", 1/6)]:
    print(f"  {name} = {val:.5f}")
