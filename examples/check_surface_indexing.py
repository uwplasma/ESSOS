"""Check if the surface is correctly computed around the axis."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

def helical_axis(phi):
    """Simple helical axis."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

# Create simple surface with obvious parameters
x1_coeffs = np.zeros(13)
x1_coeffs[0] = 0.0  # No major radius offset
x1_coeffs[2] = 0.2  # Minor radius: 0.2 * cos(theta)
x2_coeffs = np.zeros(13)

surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_function=helical_axis,
    ntheta=20,
    nphi=20
)

gamma = np.array(surface.gamma)

# Compute axis at same phi values for direct comparison
phi_vals = np.linspace(0, 2*np.pi, 20, endpoint=False)
axis_pts = np.array(helical_axis(jnp.array(phi_vals)))

print("Comparing surface points to axis:")
print(f"Axis shape: {axis_pts.shape}")
print(f"Surface shape: {gamma.shape}")

# Check a few specific points
for i in [0, 5, 10, 15]:
    # Surface at theta=0 (should be axis + X1*n1)
    surf_pt = gamma[i, 0, :]
    axis_pt = axis_pts[i]
    dist = np.linalg.norm(surf_pt - axis_pt)
    
    print(f"\nPhi index {i}:")
    print(f"  Axis point: [{axis_pt[0]:.3f}, {axis_pt[1]:.3f}, {axis_pt[2]:.3f}]")
    print(f"  Surface[{i},0]: [{surf_pt[0]:.3f}, {surf_pt[1]:.3f}, {surf_pt[2]:.3f}]")
    print(f"  Distance: {dist:.3f}")
    print(f"  Expected: ~{abs(x1_coeffs[2]):.3f}")

# Check theta variation at one phi
print(f"\nAt phi index 0, varying theta:")
for j in [0, 5, 10, 15]:
    surf_pt = gamma[0, j, :]
    axis_pt = axis_pts[0]
    dist = np.linalg.norm(surf_pt - axis_pt)
    print(f"  theta_idx={j}: surface=[{surf_pt[0]:.3f}, {surf_pt[1]:.3f}, {surf_pt[2]:.3f}], dist={dist:.3f}")

# Check indexing: which index varies with phi?
print(f"\nChecking which index corresponds to phi:")
print(f"gamma[0,0,:] = {gamma[0,0,:]}")
print(f"gamma[1,0,:] = {gamma[1,0,:]}")
print(f"gamma[2,0,:] = {gamma[2,0,:]}")
print(f"\naxis[0] = {axis_pts[0]}")
print(f"axis[1] = {axis_pts[1]}")
print(f"axis[2] = {axis_pts[2]}")

# Is the first index phi or theta?
dist_00_to_10 = np.linalg.norm(gamma[0,0,:] - gamma[1,0,:])
dist_00_to_01 = np.linalg.norm(gamma[0,0,:] - gamma[0,1,:])
print(f"\nDistance gamma[0,0] to gamma[1,0]: {dist_00_to_10:.4f}")
print(f"Distance gamma[0,0] to gamma[0,1]: {dist_00_to_01:.4f}")

axis_dist_0_to_1 = np.linalg.norm(axis_pts[0] - axis_pts[1])
print(f"Distance axis[0] to axis[1]: {axis_dist_0_to_1:.4f}")

if abs(dist_00_to_10 - axis_dist_0_to_1) < 0.5:
    print("\n>>> First index appears to correspond to PHI")
else:
    print("\n>>> First index does NOT correspond to PHI")
