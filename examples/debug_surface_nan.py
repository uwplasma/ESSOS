"""Debug where NaN is introduced in surface computation."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, vmap

def helical_axis(phi):
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

# Simulate what the surface code does
nphi = 30
ntheta = 30
phi_vals = jnp.linspace(0, 2*jnp.pi, nphi)
theta_vals = jnp.linspace(0, 2*jnp.pi, ntheta)

# Create 2D grid
phi2d, theta2d = jnp.meshgrid(phi_vals, theta_vals, indexing='ij')
phi_flat = phi2d.flatten()

print(f"phi_flat shape: {phi_flat.shape}")
print(f"phi_flat range: [{phi_flat.min():.4f}, {phi_flat.max():.4f}]")

# Step 1: Compute axis
print("\nStep 1: Computing axis...")
axis_result = helical_axis(phi_flat)
print(f"axis shape: {axis_result.shape}")
print(f"axis has NaN: {jnp.any(jnp.isnan(axis_result))}")

# Step 2: Compute tangent using autodiff
print("\nStep 2: Computing tangent...")
tangent_func = jacfwd(helical_axis)
tangent = vmap(tangent_func)(phi_flat)
print(f"tangent shape: {tangent.shape}")
print(f"tangent has NaN: {jnp.any(jnp.isnan(tangent))}")
print(f"tangent sample: {tangent[0]}, {tangent[100]}, {tangent[500]}")

tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
print(f"tangent_norm has NaN: {jnp.any(jnp.isnan(tangent_norm))}")
print(f"tangent_norm min/max: {tangent_norm.min():.6f} / {tangent_norm.max():.6f}")

tangent_normalized = tangent / jnp.maximum(tangent_norm, 1e-14)
print(f"tangent_normalized has NaN: {jnp.any(jnp.isnan(tangent_normalized))}")

# Step 3: Compute n1
print("\nStep 3: Computing n1...")
n_pts = phi_flat.shape[0]
ref_z = jnp.zeros((n_pts, 3))
ref_z = ref_z.at[:,2].set(1.0)

dot_product = jnp.sum(ref_z * tangent_normalized, axis=-1, keepdims=True)
print(f"dot_product has NaN: {jnp.any(jnp.isnan(dot_product))}")

n1 = ref_z - dot_product * tangent_normalized
print(f"n1 (before norm) has NaN: {jnp.any(jnp.isnan(n1))}")

n1_norm = jnp.linalg.norm(n1, axis=-1, keepdims=True)
print(f"n1_norm has NaN: {jnp.any(jnp.isnan(n1_norm))}")
print(f"n1_norm min/max: {n1_norm.min():.6f} / {n1_norm.max():.6f}")
print(f"n1_norm < 1e-6 count: {jnp.sum(n1_norm[:, 0] < 1e-6)}")

# Alternative n1
is_small = n1_norm[:, 0] < 1e-6
ref_x = jnp.zeros((n_pts, 3))
ref_x = ref_x.at[:, 0].set(1.0)
dot_product_x = jnp.sum(ref_x * tangent_normalized, axis=-1, keepdims=True)
n1_alt = ref_x - dot_product_x * tangent_normalized
n1_alt_norm = jnp.linalg.norm(n1_alt, axis=-1, keepdims=True)

n1 = jnp.where(is_small[:, None], n1_alt, n1)
n1_norm = jnp.where(is_small[:, None], n1_alt_norm, n1_norm)

print(f"n1 (after alt) has NaN: {jnp.any(jnp.isnan(n1))}")
print(f"n1_norm (after alt) has NaN: {jnp.any(jnp.isnan(n1_norm))}")

n1_normalized = n1 / jnp.maximum(n1_norm, 1e-14)
print(f"n1_normalized has NaN: {jnp.any(jnp.isnan(n1_normalized))}")

# Step 4: Compute n2
print("\nStep 4: Computing n2...")
n2 = jnp.cross(tangent_normalized,n1_normalized)
print(f"n2 (before norm) has NaN: {jnp.any(jnp.isnan(n2))}")

n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)
print(f"n2_norm has NaN: {jnp.any(jnp.isnan(n2_norm))}")
print(f"n2_norm min/max: {n2_norm.min():.6f} / {n2_norm.max():.6f}")

n2_normalized = n2 / jnp.maximum(n2_norm, 1e-14)
print(f"n2_normalized has NaN: {jnp.any(jnp.isnan(n2_normalized))}")

# Step 5: Reshape and compute surface point
print("\nStep 5: Computing surface point...")
n1_reshaped = n1_normalized.reshape(nphi, ntheta, 3)
n2_reshaped = n2_normalized.reshape(nphi, ntheta, 3)
axis_reshaped = axis_result.reshape(nphi, ntheta, 3)

# Sample surface point calculation
x1_coeff = 0.15  # Minor radius
theta_sample = jnp.pi/2
X1_sample = x1_coeff * jnp.cos(theta_sample)

surface_point = axis_reshaped[0, 0, :] + X1_sample * n1_reshaped[0, 0, :]
print(f"Sample surface point (0,0): {surface_point}")
print(f"Has NaN: {jnp.any(jnp.isnan(surface_point))}")

print("\n" + "="*70)
if not jnp.any(jnp.isnan(n1_normalized)) and not jnp.any(jnp.isnan(n2_normalized)):
    print("SUCCESS! No NaN values found in Frenet frame")
else:
    print("FAILURE! NaN values detected")
