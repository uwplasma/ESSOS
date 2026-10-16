"""Verify that coils wrap around surfaces properly."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier
from essos.coils import CreateCoilsAroundAxis

def figure_eight_axis(phi):
    a = 10.0
    x = a * jnp.cos(phi)
    y = a * jnp.sin(phi) * jnp.cos(phi)
    z = 0.2 * jnp.sin(3 * phi)
    return jnp.stack([x, y, z], axis=-1)

# Create surface
x1_coeffs = np.zeros(13)
x1_coeffs[5] = 1.0
x2_coeffs = np.zeros(13)
x2_coeffs[5] = 1.0

surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_function=figure_eight_axis,
    ntheta=20,
    nphi=20
)

# Create coils
coils = CreateCoilsAroundAxis(
    n_coils=8,
    order=4,
    coil_radius=1.5,
    axis_function=figure_eight_axis,
    axis_shape='custom',
    nfp=1,
    stellsym=False
)

print("="*70)
print("COIL-SURFACE POSITIONING VERIFICATION")
print("="*70)

# Get coil and surface data
coil_gamma = np.array(coils.gamma)
surface_gamma = np.array(surface.gamma)

print(f"\nSurface minor radius: ~1.0")
print(f"Coil radius: 1.5")
print(f"Number of coils: {len(coils)}")

# Check distances from coils to axis
print("\nChecking coil positions relative to axis:")
for coil_idx in range(min(3, len(coils))):
    # Get one point from this coil
    coil_pt = coil_gamma[coil_idx, 0, :]
    
    # Find closest axis point
    phi_coil = (coil_idx + 0.5) * 2 * np.pi / 8
    axis_pt = np.array(figure_eight_axis(jnp.array(phi_coil)))
    
    dist = np.linalg.norm(coil_pt - axis_pt)
    print(f"  Coil {coil_idx}: distance from axis = {dist:.3f} (expected ~1.5)")

# Check minimum distance from coils to surface
print("\nChecking coil-surface clearance:")
min_distances = []
for coil_idx in range(len(coils)):
    # Sample coil points
    coil_pts = coil_gamma[coil_idx, ::10, :]  # Sample every 10th point
    
    min_dist_for_coil = float('inf')
    for coil_pt in coil_pts:
        # Find distance to closest surface point
        dists = np.linalg.norm(surface_gamma.reshape(-1, 3) - coil_pt, axis=1)
        min_dist = np.min(dists)
        min_dist_for_coil = min(min_dist_for_coil, min_dist)
    
    min_distances.append(min_dist_for_coil)
    if coil_idx < 3:
        print(f"  Coil {coil_idx}: min distance to surface = {min_dist_for_coil:.3f}")

min_distances = np.array(min_distances)
print(f"\nOverall statistics:")
print(f"  Mean minimum clearance: {np.mean(min_distances):.3f}")
print(f"  Min clearance: {np.min(min_distances):.3f}")
print(f"  Max clearance: {np.max(min_distances):.3f}")

# Expected clearance = coil_radius - surface_radius = 1.5 - 1.0 = 0.5
expected_clearance = 0.5
if np.min(min_distances) > 0.0:
    print(f"\n✅ Coils are outside the surface!")
    print(f"   Expected clearance: ~{expected_clearance:.1f}, observed: {np.mean(min_distances):.3f}")
else:
    print(f"\n⚠️  Coils may intersect the surface")

print("="*70)
