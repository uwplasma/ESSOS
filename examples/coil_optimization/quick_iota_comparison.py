#!/usr/bin/env python3
"""Quick iota comparison between RZ and GFrame surfaces"""
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=1'

import sys
sys.path.insert(0, '/home/exouser/ESSOS')

import jax
import jax.numpy as jnp
from jax import vmap
from essos.surfaces import SurfaceRZFourier, SurfaceGFrameFourier, B_on_surface
from essos.fields import BiotSavart
from essos.coils import Coils, CreateEquallySpacedCurves

# Setup parameters
ntheta = 60
nphi = 60
NFP = 2
COIL_CURRENT = 1.0
N_COILS = 4
FOURIER_ORDER = 6
LARGE_R = 10.0
SMALL_R = 5.7
N_SEGMENTS = 60
STELLSYM = True

input_filepath = os.path.join(os.path.dirname(__name__), "../input_files")
elipse_input = os.path.join(input_filepath, 'input.rotating_ellipse')
toroidal_input = os.path.join(input_filepath, 'input.toroidal_surface_RZ')

# Create coils and field
init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, 
                                        n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM,
                                        scaling_factor=1.2, scaling_type=jnp.inf, scale_fixed=1.)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)

# Load RZ ellipse
init_ellipse_RZ = SurfaceRZFourier.from_input_file(elipse_input, ntheta=ntheta, nphi=nphi,
                                                     close=True, range_torus='full period',
                                                     scaling_factor=1.2, scaling_type=jnp.inf, scale_fixed=1.)

# Load RZ torus
init_surface_2_RZ = SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,
                                                       close=True, range_torus='full period',
                                                       scaling_factor=1.2, scaling_type=jnp.inf, scale_fixed=1.)

def build_gframe_from_rz_surface(rz_surface):
    """Build GFrame coefficients directly from an existing SurfaceRZFourier object."""
    mpol = int(rz_surface.mpol)
    ntor = int(rz_surface.ntor)
    nfp = int(rz_surface.nfp)
    axis_ntor = ntor
    x1 = jnp.zeros_like(rz_surface.rc, dtype=jnp.float64)
    x2 = jnp.zeros_like(rz_surface.zs, dtype=jnp.float64)
    axis_rc = jnp.zeros(2 * axis_ntor + 1, dtype=jnp.float64)
    axis_zs = jnp.zeros(2 * axis_ntor + 1, dtype=jnp.float64)
    for i in range(len(rz_surface.xm)):
        m = int(rz_surface.xm[i])
        n = int(rz_surface.xn[i] // nfp)
        if m == 0:
            idx_axis = n + axis_ntor
            if 0 <= idx_axis < axis_rc.size:
                axis_rc = axis_rc.at[idx_axis].set(rz_surface.rc[i])
                axis_zs = axis_zs.at[idx_axis].set(rz_surface.zs[i])
        else:
            x1 = x1.at[i].set(rz_surface.rc[i])
            x2 = x2.at[i].set(rz_surface.zs[i])
    return x1, x2, axis_rc, axis_zs, mpol, ntor, axis_ntor

# Create GFrame ellipse
ellipse_x1, ellipse_x2, ellipse_axis_rc, ellipse_axis_zs, ellipse_mpol, ellipse_ntor, ellipse_axis_ntor = \
    build_gframe_from_rz_surface(init_ellipse_RZ)

init_ellipse_gframe = SurfaceGFrameFourier(
    x1_coeffs=ellipse_x1, x2_coeffs=ellipse_x2, nfp=NFP, mpol=ellipse_mpol, ntor=ellipse_ntor,
    ntheta=ntheta, nphi=nphi, close=True, range_torus='full period',
    axis_major_radius=float(ellipse_axis_rc[ellipse_axis_ntor]),
    axis_function=None, enable_axis_dofs=True,
    axis_rc_coeffs=ellipse_axis_rc, axis_zs_coeffs=ellipse_axis_zs,
    axis_ntor=ellipse_axis_ntor, axis_shape='fourier',
    scaling_factor=1.2, scaling_type=jnp.inf, scale_fixed=1.0)

# Create GFrame torus  
torus_x1, torus_x2, torus_axis_rc, torus_axis_zs, torus_mpol, torus_ntor, torus_axis_ntor = \
    build_gframe_from_rz_surface(init_surface_2_RZ)

init_surface_2 = SurfaceGFrameFourier(
    x1_coeffs=torus_x1, x2_coeffs=torus_x2, nfp=NFP, mpol=torus_mpol, ntor=torus_ntor,
    ntheta=ntheta, nphi=nphi, close=True, range_torus='full period',
    axis_major_radius=float(torus_axis_rc[torus_axis_ntor]),
    axis_function=None, enable_axis_dofs=True,
    axis_rc_coeffs=torus_axis_rc, axis_zs_coeffs=torus_axis_zs,
    axis_ntor=torus_axis_ntor, axis_shape='fourier',
    scaling_factor=1.2, scaling_type=jnp.inf, scale_fixed=1.0)

# Define iota function
@jax.jit
def iota_function(field, surface):
    B = B_on_surface(surface, field)
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
    area_element = jnp.linalg.norm(n_vec_raw, axis=-1)
    
    g_tt = jnp.sum(e_theta**2, axis=-1)
    g_pp = jnp.sum(e_phi**2, axis=-1)
    g_tp = jnp.sum(e_theta * e_phi, axis=-1)
    det_g2d = g_tt * g_pp - g_tp**2
    
    b_cov_theta = jnp.sum(B * e_theta, axis=-1)
    b_cov_phi = jnp.sum(B * e_phi, axis=-1)
    b_theta = (g_pp * b_cov_theta - g_tp * b_cov_phi) / (det_g2d + 1e-8)
    b_phi = (g_tt * b_cov_phi - g_tp * b_cov_theta) / (det_g2d + 1e-8)
    
    b_phi_safe = jnp.sqrt(b_phi**2 + 1e-28)
    iota_local = b_theta / b_phi_safe
    
    b_phi_avg_safe = jnp.sqrt(jnp.average(b_phi * area_element)**2 + 1e-28)
    iota_average = jnp.average(b_theta * area_element) / b_phi_avg_safe
    return iota_local, iota_average

# Run comparison
print('\n' + '='*80)
print('IOTA COMPARISON: RZFourier vs GFrameFourier')
print('='*80)

print('\n--- ROTATING ELLIPSE ---')
iota_local_rz, iota_avg_rz = iota_function(init_field, init_ellipse_RZ)
iota_local_gf, iota_avg_gf = iota_function(init_field, init_ellipse_gframe)
print(f'RZ Ellipse     ι_avg: {float(iota_avg_rz):.15f}')
print(f'GFrame Ellipse ι_avg: {float(iota_avg_gf):.15f}')
print(f'Absolute difference:  {float(jnp.abs(iota_avg_rz - iota_avg_gf)):.15e}')
print(f'Relative difference:  {100*float(jnp.abs(iota_avg_rz - iota_avg_gf)/jnp.abs(iota_avg_rz)):.6e}%')

print('\n--- TORUS ---')
iota_local_rz2, iota_avg_rz2 = iota_function(init_field, init_surface_2_RZ)
iota_local_gf2, iota_avg_gf2 = iota_function(init_field, init_surface_2)
print(f'RZ Torus       ι_avg: {float(iota_avg_rz2):.15f}')
print(f'GFrame Torus   ι_avg: {float(iota_avg_gf2):.15f}')
print(f'Absolute difference:  {float(jnp.abs(iota_avg_rz2 - iota_avg_gf2)):.15e}')
print(f'Relative difference:  {100*float(jnp.abs(iota_avg_rz2 - iota_avg_gf2)/jnp.abs(iota_avg_rz2)):.6e}%')

print('\n' + '='*80)
print('✓ Comparison complete')
print('='*80)
