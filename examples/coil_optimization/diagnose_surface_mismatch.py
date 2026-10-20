#!/usr/bin/env python3
"""Diagnose geometric mismatch between RZ and GFrame ellipse"""
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=1'

import sys
sys.path.insert(0, '/home/exouser/ESSOS')

import jax.numpy as jnp
from essos.surfaces import SurfaceRZFourier, SurfaceGFrameFourier

# Setup
ntheta = 60
nphi = 60
NFP = 2
input_filepath = os.path.join(os.path.dirname(__name__), "../input_files")
elipse_input = os.path.join(input_filepath, 'input.rotating_ellipse')

# Load RZ ellipse
print("="*80)
print("LOADING RZ ELLIPSE")
print("="*80)
init_ellipse_RZ = SurfaceRZFourier.from_input_file(
    elipse_input, ntheta=ntheta, nphi=nphi, close=True, range_torus='full period',
    scaling_factor=1.2, scaling_type=jnp.inf, scale_fixed=1.)

print(f"\nRZ Fourier Coefficients:")
print(f"  xm (poloidal modes): {init_ellipse_RZ.xm}")
print(f"  xn (toroidal modes): {init_ellipse_RZ.xn}")
print(f"  rc (major radius): {init_ellipse_RZ.rc}")
print(f"  zs (vertical): {init_ellipse_RZ.zs}")

# Build GFrame from RZ
def build_gframe_from_rz_surface(rz_surface):
    """Build GFrame coefficients from RZ surface with correct mode mapping."""
    mpol = int(rz_surface.mpol)
    ntor = int(rz_surface.ntor)
    nfp = int(rz_surface.nfp)
    axis_ntor = ntor
    
    # Build GFrame mode arrays (matches SurfaceGFrameFourier constructor)
    xm_gframe = jnp.repeat(jnp.arange(mpol + 1), 2 * ntor + 1)[ntor:]
    xn_gframe = nfp * jnp.tile(jnp.arange(-ntor, ntor + 1), mpol + 1)[ntor:]
    
    n_gframe_modes = len(xm_gframe)
    x1 = jnp.zeros(n_gframe_modes, dtype=jnp.float64)
    x2 = jnp.zeros(n_gframe_modes, dtype=jnp.float64)
    axis_rc = jnp.zeros(2 * axis_ntor + 1, dtype=jnp.float64)
    axis_zs = jnp.zeros(2 * axis_ntor + 1, dtype=jnp.float64)
    
    print(f"\nGFrame mapping (NFP={nfp}):")
    print(f"  mpol={mpol}, ntor={ntor}, axis_ntor={axis_ntor}")
    print(f"  GFrame xm: {xm_gframe}")
    print(f"  GFrame xn: {xn_gframe}")
    print(f"  Total GFrame surface modes: {n_gframe_modes}")
    
    # Map each RZ mode to the correct GFrame index
    for i in range(len(rz_surface.xm)):
        m = int(rz_surface.xm[i])
        n_phys = int(rz_surface.xn[i] // nfp)
        
        if m == 0:
            idx_axis = n_phys + axis_ntor
            if 0 <= idx_axis < axis_rc.size:
                axis_rc = axis_rc.at[idx_axis].set(rz_surface.rc[i])
                axis_zs = axis_zs.at[idx_axis].set(rz_surface.zs[i])
                print(f"    RZ[{i}] (n={n_phys},m={m}) → axis[{idx_axis}]: rc={rz_surface.rc[i]:.6f}, zs={rz_surface.zs[i]:.6f}")
        else:
            # Find matching (m, n*nfp) in GFrame basis
            matches = jnp.where((xm_gframe == m) & (xn_gframe == rz_surface.xn[i]))[0]
            if len(matches) > 0:
                idx_gframe = int(matches[0])
                x1 = x1.at[idx_gframe].set(rz_surface.rc[i])
                x2 = x2.at[idx_gframe].set(rz_surface.zs[i])
                if rz_surface.rc[i] != 0 or rz_surface.zs[i] != 0:
                    print(f"    RZ[{i}] (n={n_phys},m={m}) → GF[{idx_gframe}]: x1={rz_surface.rc[i]:.6f}, x2={rz_surface.zs[i]:.6f}")
    
    return x1, x2, axis_rc, axis_zs, mpol, ntor, axis_ntor

ellipse_x1, ellipse_x2, ellipse_axis_rc, ellipse_axis_zs, ellipse_mpol, ellipse_ntor, ellipse_axis_ntor = \
    build_gframe_from_rz_surface(init_ellipse_RZ)

print(f"\nExtracted GFrame coefficients:")
print(f"  axis_rc: {ellipse_axis_rc}")
print(f"  axis_zs: {ellipse_axis_zs}")
print(f"  x1 (nonzero): {ellipse_x1[jnp.nonzero(ellipse_x1, size=10)]}")
print(f"  x2 (nonzero): {ellipse_x2[jnp.nonzero(ellipse_x2, size=10)]}")

# Create GFrame surface
init_ellipse_gframe = SurfaceGFrameFourier(
    x1_coeffs=ellipse_x1, x2_coeffs=ellipse_x2, nfp=NFP, mpol=ellipse_mpol, ntor=ellipse_ntor,
    ntheta=ntheta, nphi=nphi, close=True, range_torus='full period',
    axis_major_radius=float(ellipse_axis_rc[ellipse_axis_ntor]),
    axis_function=None, enable_axis_dofs=True,
    axis_rc_coeffs=ellipse_axis_rc, axis_zs_coeffs=ellipse_axis_zs,
    axis_ntor=ellipse_axis_ntor, axis_shape='fourier',
    scaling_factor=1.2, scaling_type=jnp.inf, scale_fixed=1.0)

# Compare surfaces
print("\n" + "="*80)
print("SURFACE GEOMETRY COMPARISON")
print("="*80)

gamma_rz = init_ellipse_RZ.gamma
gamma_gf = init_ellipse_gframe.gamma

# Extract R and Z from gamma (assuming gamma = [x, y, z] in Cartesian, convert to R)
R_rz = jnp.sqrt(gamma_rz[..., 0]**2 + gamma_rz[..., 1]**2)
Z_rz = gamma_rz[..., 2]

R_gf = jnp.sqrt(gamma_gf[..., 0]**2 + gamma_gf[..., 1]**2)
Z_gf = gamma_gf[..., 2]

print(f"\nRZ Surface:")
print(f"  R: min={float(jnp.min(R_rz)):.6f}, max={float(jnp.max(R_rz)):.6f}, mean={float(jnp.mean(R_rz)):.6f}")
print(f"  Z: min={float(jnp.min(Z_rz)):.6f}, max={float(jnp.max(Z_rz)):.6f}, mean={float(jnp.mean(Z_rz)):.6f}")

print(f"\nGFrame Surface:")
print(f"  R: min={float(jnp.min(R_gf)):.6f}, max={float(jnp.max(R_gf)):.6f}, mean={float(jnp.mean(R_gf)):.6f}")
print(f"  Z: min={float(jnp.min(Z_gf)):.6f}, max={float(jnp.max(Z_gf)):.6f}, mean={float(jnp.mean(Z_gf)):.6f}")

# Point-by-point difference
dR = R_rz - R_gf
dZ = Z_rz - Z_gf
distance = jnp.sqrt(dR**2 + dZ**2)

print(f"\nPoint-wise differences (R,Z):")
print(f"  ΔR: min={float(jnp.min(dR)):.10e}, max={float(jnp.max(dR)):.10e}, mean={float(jnp.mean(jnp.abs(dR))):.10e}")
print(f"  ΔZ: min={float(jnp.min(dZ)):.10e}, max={float(jnp.max(dZ)):.10e}, mean={float(jnp.mean(jnp.abs(dZ))):.10e}")
print(f"  Distance: min={float(jnp.min(distance)):.10e}, max={float(jnp.max(distance)):.10e}, mean={float(jnp.mean(distance)):.10e}")

# Check worst-case points
worst_idx = jnp.argmax(distance)
phi_worst, theta_worst = jnp.unravel_index(worst_idx, distance.shape)
print(f"\nWorst mismatch at (phi,theta) = ({int(phi_worst)}, {int(theta_worst)}):")
print(f"  RZ point: R={float(R_rz[phi_worst, theta_worst]):.10f}, Z={float(Z_rz[phi_worst, theta_worst]):.10f}")
print(f"  GF point: R={float(R_gf[phi_worst, theta_worst]):.10f}, Z={float(Z_gf[phi_worst, theta_worst]):.10f}")
print(f"  Distance: {float(distance[phi_worst, theta_worst]):.10e}")

# Check at specific theta=0 slice (different phi)
print(f"\nCross-section at theta=0 (varying phi):")
print(f"{'phi':<8} {'R_RZ':<15} {'Z_RZ':<15} {'R_GF':<15} {'Z_GF':<15} {'ΔR':<15} {'ΔZ':<15} {'dist':<15}")
print("-"*110)
for phi_idx in range(0, nphi, nphi//5):
    print(f"{phi_idx:<8} {float(R_rz[phi_idx,0]):<15.10f} {float(Z_rz[phi_idx,0]):<15.10f} "
          f"{float(R_gf[phi_idx,0]):<15.10f} {float(Z_gf[phi_idx,0]):<15.10f} "
          f"{float(dR[phi_idx,0]):<15.10e} {float(dZ[phi_idx,0]):<15.10e} "
          f"{float(distance[phi_idx,0]):<15.10e}")

# Check area elements and metrics
print(f"\nMetric comparison:")
ae_rz = init_ellipse_RZ.area_element
ae_gf = init_ellipse_gframe.area_element
print(f"  Area element RZ: mean={float(jnp.mean(ae_rz)):.10e}, std={float(jnp.std(ae_rz)):.10e}")
print(f"  Area element GF: mean={float(jnp.mean(ae_gf)):.10e}, std={float(jnp.std(ae_gf)):.10e}")
print(f"  Total area RZ:   {float(init_ellipse_RZ.area):.10f}")
print(f"  Total area GF:   {float(init_ellipse_gframe.area):.10f}")

# Check normal vectors
normal_rz = init_ellipse_RZ.unitnormal
normal_gf = init_ellipse_gframe.unitnormal
normal_diff = jnp.linalg.norm(normal_rz - normal_gf, axis=-1)
print(f"\nNormal vector differences:")
print(f"  Max difference: {float(jnp.max(normal_diff)):.10e}")
print(f"  Mean difference: {float(jnp.mean(normal_diff)):.10e}")

print("\n" + "="*80)
