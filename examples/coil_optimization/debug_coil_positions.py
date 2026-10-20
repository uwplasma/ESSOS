#!/usr/bin/env python
"""Debug coil positions to find overlaps"""
import os
import sys
sys.path.insert(0, '/home/exouser/ESSOS')

from essos.surfaces import SurfaceRZFourier
from essos.coils import CreateCoilsAroundAxis, Coils
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load surface
input_filepath = '../input_files'
vmec_input = os.path.join(input_filepath, 'wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc')
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=60, nphi=60, range_torus='full torus')

# Parameters
N_COILS = 4
FOURIER_ORDER = 6
SMALL_R = 5.7
N_SEGMENTS = 60
NFP = 2
STELLSYM = True
COIL_CURRENT = 1.0

# Create coils
init_curves = CreateCoilsAroundAxis(
    n_coils=N_COILS, 
    order=FOURIER_ORDER,
    coil_radius=SMALL_R, 
    n_segments=N_SEGMENTS, 
    nfp=NFP, 
    stellsym=STELLSYM, 
    surface=surface
)

gamma = np.array(init_curves.gamma)  # (n_curves, n_samples, 3)
n_coils = gamma.shape[0]

print(f"Total coils: {n_coils}")
print(f"Expected: {N_COILS * NFP * (2 if STELLSYM else 1)}")

# Compute coil centers (means of all points)
centers = np.mean(gamma, axis=1)  # (n_curves, 3)

print("\n" + "="*70)
print("COIL CENTER POSITIONS (phi, z coordinates in cylindrical)")
print("="*70)

for i, center in enumerate(centers):
    x, y, z = center
    # Convert to cylindrical
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2*np.pi
    
    print(f"Coil {i:2d}: R={R:8.4f}  φ={phi:7.4f} rad ({np.degrees(phi):7.2f}°)  Z={z:8.4f}")

# Find duplicates
print("\n" + "="*70)
print("DETECTING OVERLAPS (spatial distance between centers)")
print("="*70)

tolerance = 0.1  # 0.1 unit tolerance
duplicates = []
for i in range(n_coils):
    for j in range(i+1, n_coils):
        dist = np.linalg.norm(centers[i] - centers[j])
        if dist < tolerance:
            duplicates.append((i, j, dist))
            print(f"WARNING: Coil {i} and {j} are close (distance: {dist:.6f})")

if not duplicates:
    print("No overlaps detected (distance > 0.1)")
else:
    print(f"\nFound {len(duplicates)} pairs of coils that are too close!")

# Plot coil centers in 3D
fig = plt.figure(figsize=(15, 5))

# 3D plot
ax1 = fig.add_subplot(131, projection='3d')
for i, center in enumerate(centers):
    ax1.scatter(center[0], center[1], center[2], s=100, alpha=0.6)
    ax1.text(center[0], center[1], center[2], f'  {i}', fontsize=8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Coil Centers (3D)')

# XY plane
ax2 = fig.add_subplot(132)
for i, center in enumerate(centers):
    ax2.scatter(center[0], center[1], s=100, alpha=0.6)
    ax2.text(center[0], center[1], f'{i}', fontsize=8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('XY Plane View')

# Toroidal angle distribution
ax3 = fig.add_subplot(133)
phis = []
Zs = []
for i, center in enumerate(centers):
    x, y, z = center
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2*np.pi
    phis.append(np.degrees(phi))
    Zs.append(z)
    ax3.scatter(phis[-1], Zs[-1], s=100, alpha=0.6)
    ax3.text(phis[-1], Zs[-1], f'{i}', fontsize=8)

ax3.set_xlabel('Toroidal Angle φ (degrees)')
ax3.set_ylabel('Z')
ax3.grid(True, alpha=0.3)
ax3.set_title('Coil Distribution (φ vs Z)')

plt.tight_layout()
plt.savefig('coil_positions_debug.png', dpi=150)
print("\n✓ Saved: coil_positions_debug.png")

# Check if coils are evenly distributed toroidally
print("\n" + "="*70)
print("TOROIDAL DISTRIBUTION ANALYSIS")
print("="*70)

# Sort by toroidal angle
phis_array = np.array(phis)
sorted_indices = np.argsort(phis_array)
sorted_phis = phis_array[sorted_indices]

print("Coils in toroidal order:")
for idx, phi in zip(sorted_indices, sorted_phis):
    print(f"  Coil {idx:2d}: φ = {phi:7.2f}°")

print("\nGaps between consecutive coils (in toroidal angle):")
gaps = []
for i in range(len(sorted_phis)):
    next_i = (i + 1) % len(sorted_phis)
    if next_i == 0:
        gap = (360 - sorted_phis[i]) + sorted_phis[next_i]
    else:
        gap = sorted_phis[next_i] - sorted_phis[i]
    gaps.append(gap)
    print(f"  {sorted_indices[i]} -> {sorted_indices[next_i]}: {gap:7.2f}°")

print(f"\nExpected gap (uniform): {360.0 / n_coils:.2f}°")
print(f"Actual gaps - min: {min(gaps):.2f}°, max: {max(gaps):.2f}°, std: {np.std(gaps):.2f}°")
