"""Verify surfaces have proper 2D structure."""

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

# Create surface with CORRECT coefficient index
x1_coeffs = np.zeros(13)
x1_coeffs[0] = 0.0   # Zero offset (surface starts at axis)
x1_coeffs[5] = 0.2   # Minor radius (m=1,n=0)
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

gamma = np.array(surface.gamma)

print("="*70)
print("SURFACE STRUCTURE VERIFICATION")
print("="*70)
print(f"\nSurface shape: {gamma.shape}")
print(f"Has NaN: {np.any(np.isnan(gamma))}")

# Check variation with theta (should vary)
print("\n1. Check variation with THETA (column index):")
print("   At phi_idx=0, checking gamma[0, j, :] for different j:")
for j in [0, 3, 6, 9]:
    pt = gamma[0, j, :]
    print(f"     j={j}: [{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]")

distances_theta = []
for j in range(9):
    dist = np.linalg.norm(gamma[0, j+1, :] - gamma[0, j, :])
    distances_theta.append(dist)
print(f"   Average distance between adjacent theta points: {np.mean(distances_theta):.4f}")
print(f"   >>> Surface {'DOES' if np.mean(distances_theta) > 0.01 else 'DOES NOT'} vary with theta ✓" if np.mean(distances_theta) > 0.01 else "   >>> Surface DOES NOT vary with theta ✗")

# Check variation with phi (should vary)
print("\n2. Check variation with PHI (row index):")
print("   At theta_idx=0, checking gamma[i, 0, :] for different i:")
for i in [0, 3, 6, 9]:
    pt = gamma[i, 0, :]
    print(f"     i={i}: [{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]")

distances_phi = []
for i in range(9):
    dist = np.linalg.norm(gamma[i+1, 0, :] - gamma[i, 0, :])
    distances_phi.append(dist)
print(f"   Average distance between adjacent phi points: {np.mean(distances_phi):.4f}")
print(f"   >>> Surface {'DOES' if np.mean(distances_phi) > 0.01 else 'DOES NOT'} vary with phi ✓" if np.mean(distances_phi) > 0.01 else "   >>> Surface DOES NOT vary with phi ✗")

# Check distance from axis
phi_vals = np.linspace(0, 2*np.pi, 10, endpoint=False)
axis_pts = np.array(helical_axis(jnp.array(phi_vals)))

print(f"\n3. Check distance from axis:")
distances_from_axis = []
for i in range(10):
    dist = np.linalg.norm(gamma[i, 0, :] - axis_pts[i])
    distances_from_axis.append(dist)
print(f"   Average distance from axis: {np.mean(distances_from_axis):.4f}")
print(f"   Expected (|x1_coeffs[5]|): {abs(x1_coeffs[5]):.4f}")
print(f"   >>> Distance matches expected: {'YES ✓' if abs(np.mean(distances_from_axis) - abs(x1_coeffs[5])) < 0.05 else 'NO ✗'}")

print(f"\n{'='*70}")
if np.mean(distances_theta) > 0.01 and np.mean(distances_phi) > 0.01:
    print("✅ SUCCESS: Surface is a proper 2D surface!")
    print("   The surface varies in both theta and phi directions.")
else:
    print("✗ FAILED: Surface is still collapsed!")
print(f"{'='*70}")
