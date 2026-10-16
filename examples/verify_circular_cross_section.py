"""Verify that surfaces now have proper circular cross-sections."""

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

# Create surface with CORRECT circular cross-section
x1_coeffs = np.zeros(13)
x1_coeffs[0] = 0.0   # Zero offset
x1_coeffs[5] = 0.15  # X1 = r*cos(theta)
x2_coeffs = np.zeros(13)
x2_coeffs[5] = 0.15  # X2 = r*sin(theta)

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

print("="*70)
print("CIRCULAR CROSS-SECTION VERIFICATION")
print("="*70)

# Check cross-section at phi_idx=5
phi_idx = 5
axis_pt = np.array(helical_axis(jnp.array(2*np.pi * phi_idx / 20)))

print(f"\nAt phi_idx={phi_idx} (phi={2*np.pi*phi_idx/20:.3f}):")
print(f"Axis point: [{axis_pt[0]:.3f}, {axis_pt[1]:.3f}, {axis_pt[2]:.3f}]")

print("\nDistances from axis at different theta values:")
distances = []
for theta_idx in range(20):
    surf_pt = gamma[phi_idx, theta_idx, :]
    dist = np.linalg.norm(surf_pt - axis_pt)
    distances.append(dist)
    if theta_idx % 5 == 0:
        print(f"  theta_idx={theta_idx:2d} (theta={2*np.pi*theta_idx/20:.3f}): distance = {dist:.4f}")

distances = np.array(distances)
print(f"\nStatistics:")
print(f"  Mean distance: {np.mean(distances):.4f}")
print(f"  Std deviation: {np.std(distances):.6f}")
print(f"  Min distance: {np.min(distances):.4f}")
print(f"  Max distance: {np.max(distances):.4f}")
print(f"  Expected (minor radius): {x1_coeffs[5]:.4f}")

if np.std(distances) < 0.01 and abs(np.mean(distances) - x1_coeffs[5]) < 0.01:
    print("\n✅ SUCCESS: Cross-section is circular!")
    print(f"   All points are at approximately constant distance {np.mean(distances):.4f} from axis")
else:
    print(f"\n⚠️  Cross-section may not be perfectly circular")
    print(f"   Standard deviation: {np.std(distances):.6f}")

print("="*70)
