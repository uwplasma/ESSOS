#!/usr/bin/env python
"""Diagnose coil count issue"""
import os
from essos.surfaces import SurfaceRZFourier
from essos.coils import CreateCoilsAroundAxis, Coils
import jax.numpy as jnp
import numpy as np

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

print(f"Surface NFP: {surface.nfp}")
print(f"Requested NFP: {NFP}")
print(f"STELLSYM: {STELLSYM}")
print(f"N_COILS (base): {N_COILS}")

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

print(f"\nCurves stats:")
print(f"  Number of base curves: {init_curves.n_base_curves}")
print(f"  Curves.nfp: {init_curves.nfp}")
print(f"  Curves.stellsym: {init_curves.stellsym}")
print(f"  Total curves after symmetry: {len(init_curves.gamma)}")

# Check gamma shape
print(f"\nGamma shape: {init_curves.gamma.shape}")

# Create coils object
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)

print(f"\nCoils stats:")
print(f"  Number of base currents: {len(init_coils.dofs_currents_raw)}")
print(f"  Total currents after symmetry: {len(init_coils.currents)}")

# Check for duplicates by looking at centers
print("\n--- Checking for duplicate/identical coils ---")
gamma = init_curves.gamma  # (n_curves, n_samples, 3)
centers = jnp.mean(gamma, axis=1)  # Approximate center (n_curves, 3)

print(f"Coil centers (first 5):")
for i in range(min(5, len(centers))):
    print(f"  Coil {i}: {centers[i]}")

# Check distances between consecutive coil centers
print(f"\nDistances between consecutive coil centers:")
for i in range(min(15, len(centers)-1)):
    dist = jnp.linalg.norm(centers[i+1] - centers[i])
    print(f"  Coil {i} -> {i+1}: {float(dist):.6f}")

# Find truly unique coils (allowing small tolerance)
tolerance = 1e-6
unique_coils = []
for i in range(len(centers)):
    is_duplicate = False
    for j in unique_coils:
        if jnp.linalg.norm(centers[i] - centers[j]) < tolerance:
            is_duplicate = True
            print(f"WARNING: Coil {i} is duplicate of coil {j}")
            break
    if not is_duplicate:
        unique_coils.append(i)

print(f"\nUnique coils found: {len(unique_coils)} out of {len(centers)}")
