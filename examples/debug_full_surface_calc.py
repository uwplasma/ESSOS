"""Check Frenet frame reshape and surface computation."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, vmap

def helical_axis(phi):
    """Helical axis that closes on itself."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

# Simulate surface computation
nphi = 10
ntheta = 10

# Create mesh
theta_vals = jnp.linspace(0, 2*jnp.pi, ntheta,endpoint=False)
phi_vals = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
theta2d, phi2d = jnp.meshgrid(theta_vals, phi_vals)

print("theta2d shape:", theta2d.shape)
print("phi2d shape:", phi2d.shape)
print("phi2d first row:", phi2d[0, :])
print("phi2d first col:", phi2d[:, 0])

# Flatten
phi_flat = phi2d.flatten()
print("\nphi_flat shape:", phi_flat.shape)
print("phi_flat first 10:", phi_flat[:10])

# Compute axis
axis_result = helical_axis(phi_flat)
print("\naxis_result shape:", axis_result.shape)
print("axis_result has NaN:", jnp.any(jnp.isnan(axis_result)))

# Reshape axis
axis_curve = axis_result.reshape(nphi, ntheta, 3)
print("\naxis_curve shape:", axis_curve.shape)
print("axis_curve[0, 0, :]:", axis_curve[0, 0, :])
print("axis_curve[1, 0, :]:", axis_curve[1, 0, :])
print("axis_curve has NaN:", jnp.any(jnp.isnan(axis_curve)))

# Compute tangent
tangent_func = jacfwd(helical_axis)
tangent = vmap(tangent_func)(phi_flat)
print("\ntangent shape:", tangent.shape)
print("tangent has NaN:", jnp.any(jnp.isnan(tangent)))

tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
tangent_normalized = tangent / jnp.maximum(tangent_norm, 1e-14)
print("tangent_normalized has NaN:", jnp.any(jnp.isnan(tangent_normalized)))

# Compute n1
n_pts = phi_flat.shape[0]
ref_z = jnp.zeros((n_pts, 3))
ref_z = ref_z.at[:, 2].set(1.0)

dot_product = jnp.sum(ref_z * tangent_normalized, axis=-1, keepdims=True)
n1 = ref_z - dot_product * tangent_normalized
n1_norm = jnp.linalg.norm(n1, axis=-1, keepdims=True)

is_small = n1_norm[:, 0] < 1e-6
ref_x = jnp.zeros((n_pts, 3))
ref_x = ref_x.at[:, 0].set(1.0)
dot_product_x = jnp.sum(ref_x * tangent_normalized, axis=-1, keepdims=True)
n1_alt = ref_x - dot_product_x * tangent_normalized
n1_alt_norm = jnp.linalg.norm(n1_alt, axis=-1, keepdims=True)

n1 = jnp.where(is_small[:, None], n1_alt, n1)
n1_norm = jnp.where(is_small[:, None], n1_alt_norm, n1_norm)

n1_normalized = n1 / jnp.maximum(n1_norm, 1e-14)
print("\nn1_normalized has NaN:", jnp.any(jnp.isnan(n1_normalized)))

# Compute n2
n2 = jnp.cross(tangent_normalized, n1_normalized)
n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)
n2_normalized = n2 / jnp.maximum(n2_norm, 1e-14)
print("n2_normalized has NaN:", jnp.any(jnp.isnan(n2_normalized)))

# Reshape
n1_reshaped = n1_normalized.reshape(nphi, ntheta, 3)
n2_reshaped = n2_normalized.reshape(nphi, ntheta, 3)

print("\nn1_reshaped shape:", n1_reshaped.shape)
print("n1_reshaped has NaN:", jnp.any(jnp.isnan(n1_reshaped)))
print("n2_reshaped has NaN:", jnp.any(jnp.isnan(n2_reshaped)))

# Compute X1, X2
x1_coeffs = np.zeros(13)
x1_coeffs[0] = 1.0
x1_coeffs[2] = 0.15
x2_coeffs = np.zeros(13)

# Mode numbers (mpol=2, ntor=2)
xm = np.array([0, 1, 1, 1, 2, 2, 2, 0, 0, -1, -1, -2, -2])
xn = np.array([0, -2, -1, 0, -2, -1, 0, 1, 2, 1, 2, 1, 2])

# Compute angles
angles = jnp.einsum('i,jk->ijk', xm, theta2d) - jnp.einsum('i,jk->ijk', xn, phi2d)
cos_angles = jnp.cos(angles)
sin_angles = jnp.sin(angles)

X1 = jnp.einsum('i,ijk->jk', x1_coeffs, cos_angles)
X2 = jnp.einsum('i,ijk->jk', x2_coeffs, sin_angles)

print("\nX1 shape:", X1.shape)
print("X1 has NaN:", jnp.any(jnp.isnan(X1)))
print("X2 has NaN:", jnp.any(jnp.isnan(X2)))

X1_expanded = X1[..., None]
X2_expanded = X2[..., None]

print("\nX1_expanded shape:", X1_expanded.shape)

# Compute final surface
gamma = axis_curve + X1_expanded * n1_reshaped + X2_expanded * n2_reshaped

print("\ngamma shape:", gamma.shape)
print("gamma has NaN:", jnp.any(jnp.isnan(gamma)))

if jnp.any(jnp.isnan(gamma)):
    nan_mask = jnp.isnan(gamma[:, :, 0])
    print(f"NaN count: {jnp.sum(nan_mask)} / {nan_mask.size}")
    print("First few gamma values:")
    for i in range(min(3, nphi)):
        for j in range(min(3, ntheta)):
            print(f"  gamma[{i},{j},:] = {gamma[i,j,:]}")
else:
    print("No NaN! Surface looks good.")
    print(f"X range: [{gamma[:,:,0].min():.3f}, {gamma[:,:,0].max():.3f}]")
