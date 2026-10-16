"""Debug why surface doesn't vary with theta."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

def helical_axis(phi):
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

x1_coeffs = np.zeros(13)
x1_coeffs[0] = 0.0
x1_coeffs[2] = 0.2
x2_coeffs = np.zeros(13)

surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_function=helical_axis,
    ntheta=10,
    nphi=10
)

print("Checking theta2d and phi2d:")
print(f"theta2d shape: {surface.theta2d.shape}")
print(f"phi2d shape: {surface.phi2d.shape}")

print(f"\ntheta2d[0,:] (should vary with theta):")
print(surface.theta2d[0, :])

print(f"\nphi2d[:,0] (should vary with phi):")
print(surface.phi2d[:, 0])

print(f"\ntheta2d[:,0] (should be constant):")
print(surface.theta2d[:, 0])

print(f"\nphi2d[0,:] (should be constant):")
print(surface.phi2d[0, :])

# Check angles
angles = surface.angles
print(f"\nangles shape: {angles.shape}")
print(f"angles[2, :, :] (mode m=1, n=-1):")
print(angles[2, :, :])

# Check X1
cos_angles = jnp.cos(angles)
X1 = jnp.einsum('i,ijk->jk', surface.x1_coeffs, cos_angles)
print(f"\nX1 shape: {X1.shape}")
print(f"X1[0,:] (should vary with theta):")
print(X1[0, :])

print(f"\nX1[:,0] (should be constant or vary slowly with phi):")
print(X1[:, 0])

# Manually compute what X1 should be
# x1_coeffs[2] = 0.2 corresponds to m=1, n=-1
# So X1 should have a term: 0.2 * cos(m*theta - n*phi) = 0.2 * cos(theta + phi)
theta_vals = surface.theta2d[0, :]
phi_val = surface.phi2d[0, 0]
X1_expected = x1_coeffs[2] * np.cos(1 * theta_vals - (-1) * phi_val)
print(f"\nExpected X1[0,:] based on x1_coeffs[2]=0.2, m=1, n=-1:")
print(X1_expected)

# Check mode numbers
print(f"\nxm (poloidal mode numbers): {surface.xm}")
print(f"xn (toroidal mode numbers): {surface.xn}")
