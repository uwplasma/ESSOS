"""Detailed diagnostic for Frenet frame computation."""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/exouser/ESSOS')

def helical_axis(phi):
    """Helical axis that closes on itself."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

# Test axis evaluation
print("Testing axis function...")
phi_test = jnp.array([0.0, jnp.pi/4, jnp.pi/2, jnp.pi])
axis_result = helical_axis(phi_test)
print(f"phi_test shape: {phi_test.shape}")
print(f"axis_result shape: {axis_result.shape}")
print(f"axis values:\n{axis_result}")

# Test derivative computation
print("\n\nTesting tangent computation...")
eps = 1e-8
axis_plus = helical_axis(phi_test + eps)
axis_minus = helical_axis(phi_test - eps)
tangent = (axis_plus - axis_minus) / (2 * eps)
print(f"tangent shape: {tangent.shape}")
print(f"tangent values:\n{tangent}")

tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
print(f"tangent_norm shape: {tangent_norm.shape}")
print(f"tangent_norm values: {tangent_norm.flatten()}")

tangent_normalized = tangent / jnp.maximum(tangent_norm, 1e-14)
print(f"tangent_normalized:\n{tangent_normalized}")

# Test normal computation
print("\n\nTesting normal computation...")
n_pts = phi_test.shape[0]
ref_z = jnp.zeros((n_pts, 3))
ref_z = ref_z.at[:, 2].set(1.0)
print(f"ref_z:\n{ref_z}")

dot_product = jnp.sum(ref_z * tangent_normalized, axis=-1, keepdims=True)
print(f"dot_product: {dot_product.flatten()}")

n1 = ref_z - dot_product * tangent_normalized
print(f"n1 (before normalization):\n{n1}")

n1_norm = jnp.linalg.norm(n1, axis=-1, keepdims=True)
print(f"n1_norm: {n1_norm.flatten()}")

n1_normalized = n1 / jnp.maximum(n1_norm, 1e-14)
print(f"n1_normalized:\n{n1_normalized}")

# Check for NaN
print(f"\nHas NaN in n1_normalized: {jnp.any(jnp.isnan(n1_normalized))}")

# Compute n2
n2 = jnp.cross(tangent_normalized, n1_normalized)
print(f"\nn2 (before normalization):\n{n2}")

n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)
print(f"n2_norm: {n2_norm.flatten()}")

n2_normalized = n2 / jnp.maximum(n2_norm, 1e-14)
print(f"n2_normalized:\n{n2_normalized}")

print(f"\nHas NaN in n2_normalized: {jnp.any(jnp.isnan(n2_normalized))}")

# Check orthonormality
print("\n\nChecking orthonormality...")
print(f"n1 · n1 = {jnp.sum(n1_normalized * n1_normalized, axis=-1)}")
print(f"n2 · n2 = {jnp.sum(n2_normalized * n2_normalized, axis=-1)}")
print(f"n1 · n2 = {jnp.sum(n1_normalized * n2_normalized, axis=-1)}")
print(f"n1 · T = {jnp.sum(n1_normalized * tangent_normalized, axis=-1)}")
print(f"n2 · T = {jnp.sum(n2_normalized * tangent_normalized, axis=-1)}")
