"""Test einsum for angles computation."""

import jax.numpy as jnp
import numpy as np

# Create simple test grids
ntheta = 5
nphi = 4
theta_vals = jnp.linspace(0, 2*jnp.pi, ntheta, endpoint=False)
phi_vals = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)

theta2d, phi2d = jnp.meshgrid(theta_vals, phi_vals)

print("theta_vals:", theta_vals)
print("phi_vals:", phi_vals)
print("\ntheta2d shape:", theta2d.shape)
print("theta2d:")
print(theta2d)
print("\nphi2d:")
print(phi2d)

# Mode numbers
xm = jnp.array([0, 1])
xn = jnp.array([0, -1])

print("\nxm:", xm)
print("xn:", xn)

# Test the einsum
angles = jnp.einsum('i,jk->ijk', xm, theta2d) - jnp.einsum('i,jk->ijk', xn, phi2d)

print("\nangles shape:", angles.shape)
print("\nangles[0, :, :] (mode 0, m=0, n=0):")
print(angles[0, :, :])

print("\nangles[1, :, :] (mode 1, m=1, n=-1):")
print(angles[1, :, :])

print("\nManual calculation for mode 1:")
print("angles[1, i, j] = xm[1] * theta2d[i, j] - xn[1] * phi2d[i, j]")
print("                = 1 * theta2d[i, j] - (-1) * phi2d[i, j]")
print("                = theta2d[i, j] + phi2d[i, j]")

manual = theta2d + phi2d
print("\nManual calculation result:")
print(manual)

print("\nDo they match?", jnp.allclose(angles[1, :, :], manual))
