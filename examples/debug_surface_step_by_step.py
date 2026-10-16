"""Debug surface computation step by step."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

def helical_axis(phi):
    """Helical axis that closes on itself."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

x1_coeffs = np.zeros(13)
x1_coeffs[0] = 1.0
x1_coeffs[2] = 0.15
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

print("Checking surface intermediate values...")

# Check angles
angles = surface.angles
print(f"angles shape: {angles.shape}")
print(f"angles has NaN: {jnp.any(jnp.isnan(angles))}")
print(f"angles sample [0]: {angles[:,0,0]}")

# Check sin/cos
sin_angles = jnp.sin(angles)
cos_angles = jnp.cos(angles)
print(f"\ncos_angles has NaN: {jnp.any(jnp.isnan(cos_angles))}")
print(f"sin_angles has NaN: {jnp.any(jnp.isnan(sin_angles))}")

# Check X1 and X2
X1 = jnp.einsum('i,ijk->jk', surface.x1_coeffs, cos_angles)
X2 = jnp.einsum('i,ijk->jk', surface.x2_coeffs, sin_angles)
print(f"\nX1 shape: {X1.shape}")
print(f"X1 has NaN: {jnp.any(jnp.isnan(X1))}")
print(f"X1 sample: {X1[0, :]}")
print(f"X2 has NaN: {jnp.any(jnp.isnan(X2))}")

# Check theta2d and phi2d
print(f"\ntheta2d shape: {surface.theta2d.shape}")
print(f"phi2d shape: {surface.phi2d.shape}")
print(f"theta2d has NaN: {jnp.any(jnp.isnan(surface.theta2d))}")
print(f"phi2d has NaN: {jnp.any(jnp.isnan(surface.phi2d))}")
print(f"phi2d sample [0,:]: {surface.phi2d[0, :]}")

# Check axis curve
phi_flat = surface.phi2d.flatten()
axis_result = helical_axis(phi_flat)
print(f"\naxis_result shape: {axis_result.shape}")
print(f"axis_result has NaN: {jnp.any(jnp.isnan(axis_result))}")
print(f"axis_result[0]: {axis_result[0]}")
print(f"axis_result[1]: {axis_result[1]}")

# Check tangent
from jax import jacfwd, vmap
tangent_func = jacfwd(helical_axis)
tangent = vmap(tangent_func)(phi_flat)
print(f"\ntangent shape: {tangent.shape}")
print(f"tangent has NaN: {jnp.any(jnp.isnan(tangent))}")
print(f"tangent[0]: {tangent[0]}")
print(f"tangent[1]: {tangent[1]}")

# Check norm
tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
print(f"\ntangent_norm has NaN: {jnp.any(jnp.isnan(tangent_norm))}")
print(f"tangent_norm[0:5]: {tangent_norm[0:5, 0]}")

tangent_normalized = tangent / jnp.maximum(tangent_norm, 1e-14)
print(f"tangent_normalized has NaN: {jnp.any(jnp.isnan(tangent_normalized))}")
