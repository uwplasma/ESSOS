"""Check meshgrid orientation."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import jax.numpy as jnp

# Test how meshgrid works
theta_vals = jnp.linspace(0, 2*jnp.pi, 5)
phi_vals = jnp.linspace(0, 2*jnp.pi, 5)

print("theta_vals:", theta_vals)
print("phi_vals:", phi_vals)

theta2d, phi2d = jnp.meshgrid(theta_vals, phi_vals)
print("\ntheta2d shape:", theta2d.shape)
print("phi2d shape:", phi2d.shape)

print("\ntheta2d:")
print(theta2d)
print("\nphi2d:")
print(phi2d)

print("\ntheta2d[0, :]:", theta2d[0, :])
print("phi2d[0, :]:", phi2d[0, :])

print("\ntheta2d[:, 0]:", theta2d[:, 0])
print("phi2d[:, 0]:", phi2d[:, 0])
