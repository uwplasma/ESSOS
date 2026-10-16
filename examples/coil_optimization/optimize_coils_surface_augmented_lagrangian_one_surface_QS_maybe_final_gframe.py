import os
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

# Enable JAX persistent compilation cache to avoid recompiling
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from time import time
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from essos.optimization import optimize_loss_function
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, SurfaceGFrameFourier, BdotN_over_B, toroidal_flux, B_on_surface
from essos.losses import custom_loss, base_loss

import jax
import essos.augmented_lagrangian as alm
from functools import partial


# Optimization parameters
maximum_function_evaluations=10
ntheta = 60
nphi=60

input_filepath = os.path.join(os.path.dirname(__name__), "../input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
filename_coils=os.path.join(input_filepath,'LandremanPaul2021_QA_coils.json')
elipse_input=os.path.join(input_filepath, 'input.rotating_ellipse')
elipse_input_axis=os.path.join(input_filepath, 'input.rotating_ellipse_axis')
toroidal_input=os.path.join(input_filepath, 'input.toroidal_surface_RZ')
toroidal_input_axis=os.path.join(input_filepath, 'input.toroidal_surface_axis')

target_coils=Coils.from_json(filename_coils)
target_field = BiotSavart(Coils.from_json(filename_coils))
#target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='half period',close=True)
#target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full torus',close=True)
target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full period',close=True)
init_surface_2_RZ=SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='full period',scaling_factor=1.2,scaling_type=jnp.inf,scale_fixed=1.)
init_ellipse_RZ=SurfaceRZFourier.from_input_file(elipse_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='full period',scaling_factor=1.2,scaling_type=jnp.inf,scale_fixed=1.)

print("\n=== RZ ellipse diagnostics ===")
print(f"RZ rc nonzero: {int(jnp.count_nonzero(init_ellipse_RZ.rc))}, zs nonzero: {int(jnp.count_nonzero(init_ellipse_RZ.zs))}")
print(f"RZ rc range: [{float(jnp.min(init_ellipse_RZ.rc)):.6f}, {float(jnp.max(init_ellipse_RZ.rc)):.6f}]")
print(f"RZ zs range: [{float(jnp.min(init_ellipse_RZ.zs)):.6f}, {float(jnp.max(init_ellipse_RZ.zs)):.6f}]")
print(f"RZ gamma |max|: {float(jnp.max(jnp.abs(init_ellipse_RZ.gamma))):.6f}")

""" Creating starting coils and surface """
N_COILS = 2; FOURIER_ORDER = 6; LARGE_R = 10.; SMALL_R = 5.7; NFP = 2; N_SEGMENTS = 60; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM,scaling_factor=1.2,scaling_type=jnp.inf,scale_fixed=1.)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)
surface_major_radius = 10.0
surface_minor_radius = 4.0
surface_mpol = 2
surface_ntor = 2
axis_ntor = 2


def build_gframe_ellipse_from_rz_input(input_file, nfp):
    """
    Convert RZ Fourier input file to GFrame format.
    
    RZ format: Surface is R(theta,phi) and Z(theta,phi) with modes indexed by (n,m)
    GFrame format: 
      - Axis: m=0 modes (toroidal Fourier) 
      - Surface: m≥1 modes (poloidal-toroidal expansion)
    
    Returns:
      (x1_coeffs, x2_coeffs, axis_rc_coeffs, axis_zs_coeffs, mpol_surf, ntor_surf, axis_ntor)
    """
    import re
    rc_dict = {}  # RBC(n,m)
    zs_dict = {}  # ZBS(n,m)
    with open(input_file, 'r') as f:
        content = f.read()
    # Parse RBC and ZBS values: RBC( n, m) = value
    rbc_pattern = r'RBC\(\s*(-?\d+),\s*(\d+)\s*\)\s*=\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
    zbs_pattern = r'ZBS\(\s*(-?\d+),\s*(\d+)\s*\)\s*=\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
    for match in re.finditer(rbc_pattern, content):
        n, m, val = int(match.group(1)), int(match.group(2)), float(match.group(3))
        rc_dict[(n, m)] = val
    for match in re.finditer(zbs_pattern, content):
        n, m, val = int(match.group(1)), int(match.group(2)), float(match.group(3))
        zs_dict[(n, m)] = val
    # Extract mpol, ntor, nfp from input
    nfp_pattern = r'NFP\s*=\s*(\d+)'
    mpol_pattern = r'MPOL\s*=\s*(\d+)'
    ntor_pattern = r'NTOR\s*=\s*(\d+)'
    nfp_file = int(re.search(nfp_pattern, content).group(1))
    mpol_file = int(re.search(mpol_pattern, content).group(1))
    ntor_file = int(re.search(ntor_pattern, content).group(1))
    mpol_surf = mpol_file
    ntor_surf = ntor_file
    axis_ntor = ntor_file
    # Match SurfaceGFrameFourier mode convention exactly (same as surfaces.py):
    # xm = repeat(0..mpol, each 2*ntor+1)[ntor:]
    # xn = nfp * tile(-ntor..ntor, mpol+1)[ntor:]
    # This cropped basis is why len(coeffs) = (mpol+1)*(2*ntor+1) - ntor.
    xm = jnp.repeat(jnp.arange(mpol_surf + 1), 2 * ntor_surf + 1)[ntor_surf:]
    xn = nfp * jnp.tile(jnp.arange(-ntor_surf, ntor_surf + 1), mpol_surf + 1)[ntor_surf:]
    if nfp_file != nfp:
        raise ValueError(
            f"NFP mismatch: input file has NFP={nfp_file}, but function received nfp={nfp}."
        )
    x1 = jnp.zeros_like(xm, dtype=jnp.float64)
    x2 = jnp.zeros_like(xm, dtype=jnp.float64)
    # Map RZ (n,m) to GFrame basis entries. For this convention:
    # x1 <- RBC(n,m), x2 <- ZBS(n,m), only for m>=1 (m=0 is axis).
    for idx in range(len(xm)):
        m = int(xm[idx])
        n = int(xn[idx] // nfp)
        if m >= 1:
            x1 = x1.at[idx].set(rc_dict.get((n, m), 0.0))
            x2 = x2.at[idx].set(zs_dict.get((n, m), 0.0))
    # Build axis cofficients (m=0 modes)
    n_axis_modes = 2 * axis_ntor + 1
    axis_rc = jnp.zeros(n_axis_modes, dtype=jnp.float64)
    axis_zs = jnp.zeros(n_axis_modes, dtype=jnp.float64)
    for n_idx, n in enumerate(range(-axis_ntor, axis_ntor + 1)):
        if (n, 0) in rc_dict:
            axis_rc = axis_rc.at[n_idx].set(rc_dict[(n, 0)])
        if (n, 0) in zs_dict:
            axis_zs = axis_zs.at[n_idx].set(zs_dict[(n, 0)])
    return x1, x2, axis_rc, axis_zs, mpol_surf, ntor_surf, axis_ntor


def build_gframe_torus_coeffs(mpol, ntor, nfp, minor_radius):
    xm = jnp.repeat(jnp.arange(mpol + 1), 2 * ntor + 1)[ntor:]
    xn = nfp * jnp.tile(jnp.arange(-ntor, ntor + 1), mpol + 1)[ntor:]
    x1 = jnp.zeros_like(xm, dtype=jnp.float64)
    x2 = jnp.zeros_like(xm, dtype=jnp.float64)
    idx = int(jnp.where((xm == 1) & (xn == 0))[0][0])
    x1 = x1.at[idx].set(minor_radius)
    x2 = x2.at[idx].set(minor_radius)
    return x1, x2

def build_axis_coeffs(axis_ntor, nfp, major_radius):
    """Build axis Fourier coefficients (axis is 1D, only toroidal modes)."""
    n_axis_modes = 2 * axis_ntor + 1
    rc = jnp.zeros(n_axis_modes, dtype=jnp.float64)
    zs = jnp.zeros(n_axis_modes, dtype=jnp.float64)
    # Set the n=0 mode to major_radius
    rc = rc.at[axis_ntor].set(major_radius)
    return rc, zs


def align_gframe_coeffs(x1, x2, mpol, ntor):
    """Force coeff vectors to SurfaceGFrameFourier expected size/order length."""
    expected_nmodes = (mpol + 1) * (2 * ntor + 1) - ntor
    x1 = jnp.asarray(x1, dtype=jnp.float64).ravel()
    x2 = jnp.asarray(x2, dtype=jnp.float64).ravel()
    if x1.size < expected_nmodes:
        x1 = jnp.pad(x1, (0, expected_nmodes - x1.size))
    else:
        x1 = x1[:expected_nmodes]
    if x2.size < expected_nmodes:
        x2 = jnp.pad(x2, (0, expected_nmodes - x2.size))
    else:
        x2 = x2[:expected_nmodes]
    return x1, x2


def build_gframe_from_rz_surface(rz_surface):
    """Build GFrame coefficients directly from an existing SurfaceRZFourier object.
    
    CRITICAL: GFrame has a different mode ordering than RZ because it crops the
    first ntor entries (which would be m=0, n<0 modes that are handled by axis).
    We must find the correct GFrame index for each (n,m) pair.
    """
    mpol = int(rz_surface.mpol)
    ntor = int(rz_surface.ntor)
    nfp = int(rz_surface.nfp)
    axis_ntor = ntor
    
    # Build GFrame mode arrays (matches SurfaceGFrameFourier constructor)
    # Full basis: xm = [0,0,0,...,1,1,1,...,2,2,2,...]  for each m, all n from -ntor to +ntor
    # Cropped basis: remove first ntor entries (m=0, n<0 modes)
    xm_gframe = jnp.repeat(jnp.arange(mpol + 1), 2 * ntor + 1)[ntor:]
    xn_gframe = nfp * jnp.tile(jnp.arange(-ntor, ntor + 1), mpol + 1)[ntor:]
    
    n_gframe_modes = len(xm_gframe)
    x1 = jnp.zeros(n_gframe_modes, dtype=jnp.float64)
    x2 = jnp.zeros(n_gframe_modes, dtype=jnp.float64)
    axis_rc = jnp.zeros(2 * axis_ntor + 1, dtype=jnp.float64)
    axis_zs = jnp.zeros(2 * axis_ntor + 1, dtype=jnp.float64)
    
    # Map each RZ mode to the correct GFrame index
    for i in range(len(rz_surface.xm)):
        m = int(rz_surface.xm[i])
        n_phys = int(rz_surface.xn[i] // nfp)  # Physical toroidal mode number
        
        if m == 0:
            # Axis modes (m=0)
            idx_axis = n_phys + axis_ntor
            if 0 <= idx_axis < axis_rc.size:
                axis_rc = axis_rc.at[idx_axis].set(rz_surface.rc[i])
                axis_zs = axis_zs.at[idx_axis].set(rz_surface.zs[i])
        else:
            # Surface modes (m >= 1): find matching (m, n*nfp) in GFrame basis
            # Search for the index where both xm and xn match
            matches = jnp.where((xm_gframe == m) & (xn_gframe == rz_surface.xn[i]))[0]
            if len(matches) > 0:
                idx_gframe = int(matches[0])
                x1 = x1.at[idx_gframe].set(rz_surface.rc[i])
                x2 = x2.at[idx_gframe].set(rz_surface.zs[i])
    
    return x1, x2, axis_rc, axis_zs, mpol, ntor, axis_ntor

# === Create GFrame ROTATING ELLIPSE surface from input file ===
def build_init_ellipse_gframe():
    ellipse_x1, ellipse_x2, ellipse_axis_rc, ellipse_axis_zs, ellipse_mpol, ellipse_ntor, ellipse_axis_ntor = \
        build_gframe_from_rz_surface(init_ellipse_RZ)
    print(f"\n✓ Loaded GFrame ellipse from SurfaceRZFourier input ({elipse_input})")
    print(f"  - mpol={ellipse_mpol}, ntor={ellipse_ntor}, axis_ntor={ellipse_axis_ntor}")
    print(f"  - x1 size: {ellipse_x1.size}, x2 size: {ellipse_x2.size}")
    print(f"  - x1 nonzero: {int(jnp.count_nonzero(ellipse_x1))}, x2 nonzero: {int(jnp.count_nonzero(ellipse_x2))}")
    print(f"  - x1 range: [{float(jnp.min(ellipse_x1)):.6f}, {float(jnp.max(ellipse_x1)):.6f}]")
    print(f"  - x2 range: [{float(jnp.min(ellipse_x2)):.6f}, {float(jnp.max(ellipse_x2)):.6f}]")
    print(f"  - axis_rc range: [{float(jnp.min(ellipse_axis_rc)):.6f}, {float(jnp.max(ellipse_axis_rc)):.6f}]")
    print(f"  - axis_zs range: [{float(jnp.min(ellipse_axis_zs)):.6f}, {float(jnp.max(ellipse_axis_zs)):.6f}]")
    return SurfaceGFrameFourier(
        x1_coeffs=ellipse_x1,
        x2_coeffs=ellipse_x2,
        nfp=NFP,
        mpol=ellipse_mpol,
        ntor=ellipse_ntor,
        ntheta=ntheta,
        nphi=nphi,
        close=True,
        range_torus='full period',
        axis_major_radius=float(ellipse_axis_rc[ellipse_axis_ntor]),  # n=0 component
        axis_function=None,
        # IMPORTANT: keep True for Fourier axis so pytree unflatten preserves axis coeffs
        # (with False, axis coeffs can become zeros in jitted paths, collapsing gamma)
        enable_axis_dofs=True,
        axis_rc_coeffs=ellipse_axis_rc,
        axis_zs_coeffs=ellipse_axis_zs,
        axis_ntor=ellipse_axis_ntor,
        axis_shape='fourier',  # Use Fourier axis with coefficients
        scaling_factor=1.2,
        scaling_type=jnp.inf,
        scale_fixed=1.0,
    )


init_ellipse_gframe = build_init_ellipse_gframe()

print("\n=== GFrame ellipse diagnostics ===")
print(f"GFrame x1 nonzero: {int(jnp.count_nonzero(init_ellipse_gframe.x1_coeffs))}, x2 nonzero: {int(jnp.count_nonzero(init_ellipse_gframe.x2_coeffs))}")
print(f"GFrame axis_rc: {init_ellipse_gframe.axis_rc_coeffs}")
print(f"GFrame axis_zs: {init_ellipse_gframe.axis_zs_coeffs}")
print(f"GFrame gamma |max|: {float(jnp.max(jnp.abs(init_ellipse_gframe.gamma))):.6f}")

# === Create standard TORUS for comparison ===
x1_coeffs, x2_coeffs = build_gframe_torus_coeffs(surface_mpol, surface_ntor, NFP, surface_minor_radius)
x1_coeffs, x2_coeffs = align_gframe_coeffs(x1_coeffs, x2_coeffs, surface_mpol, surface_ntor)
axis_rc_coeffs, axis_zs_coeffs = build_axis_coeffs(axis_ntor, NFP, surface_major_radius)

init_surface_2 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=NFP,
    mpol=surface_mpol,
    ntor=surface_ntor,
    ntheta=ntheta,
    nphi=nphi,
    close=True,
    range_torus='full period',
    axis_major_radius=surface_major_radius,
    axis_function=None,
    enable_axis_dofs=True,
    axis_rc_coeffs=axis_rc_coeffs,
    axis_zs_coeffs=axis_zs_coeffs,
    axis_ntor=axis_ntor,
    scaling_factor=1.2,
    scaling_type=jnp.inf,
    scale_fixed=1.0,
)

print(f"\n✓ GFrame torus created:")
print(f"  - Major radius R0 = {float(init_surface_2.axis_rc_coeffs[axis_ntor]):.6f}")
print(f"  - Minor radius (m=1,n=0) = {float(jnp.max(jnp.abs(init_surface_2.x1_coeffs))):.6f}")

# # ========== DIAGNOSTIC: Compare RZ vs GFrame surfaces ==========
# print("\n" + "="*80)
# print("SURFACE COMPARISON: RZFourier vs GFrameFourier")
# print("="*80)

# print(f"\n{'Property':<30} {'RZFourier':<25} {'GFrameFourier':<25} {'Diff %':<15}")
# print("-"*95)

# # Basic properties
# area_rz = float(init_surface_2_RZ.area)
# area_gf = float(init_surface_2.area)
# vol_rz = float(init_surface_2_RZ.volume)
# vol_gf = float(init_surface_2.volume)

# diff_area = 100 * abs(area_rz - area_gf) / abs(area_rz) if area_rz != 0 else float('inf')
# diff_vol = 100 * abs(vol_rz - vol_gf) / abs(vol_rz) if vol_rz != 0 else float('inf')

# print(f"{'Area':<30} {area_rz:<25.10f} {area_gf:<25.10f} {diff_area:<15.2e}")
# print(f"{'Volume':<30} {vol_rz:<25.10f} {vol_gf:<25.10f} {diff_vol:<15.2e}")

# # Area element statistics
# ae_rz = init_surface_2_RZ.area_element
# ae_gf = init_surface_2.area_element
# print(f"{'Area element mean':<30} {float(jnp.mean(ae_rz)):<25.10f} {float(jnp.mean(ae_gf)):<25.10f}")
# print(f"{'Area element min':<30} {float(jnp.min(ae_rz)):<25.10f} {float(jnp.min(ae_gf)):<25.10f}")
# print(f"{'Area element max':<30} {float(jnp.max(ae_rz)):<25.10f} {float(jnp.max(ae_gf)):<25.10f}")

# # Gamma position comparison
# gamma_rz = init_surface_2_RZ.gamma
# gamma_gf = init_surface_2.gamma
# r_rz = jnp.sqrt(gamma_rz[:,:,0]**2 + gamma_rz[:,:,1]**2)
# r_gf = jnp.sqrt(gamma_gf[:,:,0]**2 + gamma_gf[:,:,1]**2)

# print(f"{'Mean major radius (R)':<30} {float(jnp.mean(r_rz)):<25.10f} {float(jnp.mean(r_gf)):<25.10f}")
# print(f"{'Mean Z coordinate':<30} {float(jnp.mean(gamma_rz[:,:,2])):<25.10f} {float(jnp.mean(gamma_gf[:,:,2])):<25.10f}")

# # Normal vector comparison
# normal_rz = init_surface_2_RZ.normal
# normal_gf = init_surface_2.normal
# normal_mag_rz = jnp.linalg.norm(normal_rz, axis=2)
# normal_mag_gf = jnp.linalg.norm(normal_gf, axis=2)

# print(f"{'Normal magnitude mean':<30} {float(jnp.mean(normal_mag_rz)):<25.10f} {float(jnp.mean(normal_mag_gf)):<25.10f}")

# # Check if normals point in the same direction (should have positive dot product)
# normal_dot = jnp.sum(normal_rz * normal_gf, axis=2) / (normal_mag_rz * normal_mag_gf + 1e-12)
# print(f"{'Normal dot product mean':<30} {float(jnp.mean(normal_dot)):<25.10f} {'(should be ~1.0)':<25}")

# # Derivative comparison
# gd_theta_rz = init_surface_2_RZ.gammadash_theta
# gd_theta_gf = init_surface_2.gammadash_theta
# gd_phi_rz = init_surface_2_RZ.gammadash_phi
# gd_phi_gf = init_surface_2.gammadash_phi

# print(f"{'|∂γ/∂θ| mean':<30} {float(jnp.mean(jnp.linalg.norm(gd_theta_rz, axis=2))):<25.10f} {float(jnp.mean(jnp.linalg.norm(gd_theta_gf, axis=2))):<25.10f}")
# print(f"{'|∂γ/∂φ| mean':<30} {float(jnp.mean(jnp.linalg.norm(gd_phi_rz, axis=2))):<25.10f} {float(jnp.mean(jnp.linalg.norm(gd_phi_gf, axis=2))):<25.10f}")

# # B field comparison
# print("\n--- MAGNETIC FIELD COMPARISON ---")
# B_rz = B_on_surface(init_surface_2_RZ, init_field)
# B_gf = B_on_surface(init_surface_2, init_field)
# modB_rz = jnp.linalg.norm(B_rz, axis=-1)
# modB_gf = jnp.linalg.norm(B_gf, axis=-1)

# print(f"{'|B| mean':<30} {float(jnp.mean(modB_rz)):<25.10f} {float(jnp.mean(modB_gf)):<25.10f}")
# print(f"{'|B| min':<30} {float(jnp.min(modB_rz)):<25.10f} {float(jnp.min(modB_gf)):<25.10f}")
# print(f"{'|B| max':<30} {float(jnp.max(modB_rz)):<25.10f} {float(jnp.max(modB_gf)):<25.10f}")

# # Check if B vectors match
# B_diff = jnp.linalg.norm(B_rz - B_gf, axis=-1)
# print(f"{'|B_rz - B_gf| mean':<30} {float(jnp.mean(B_diff)):<25.10e} {'(should be ~0)':<25}")
# print(f"{'|B_rz - B_gf| max':<30} {float(jnp.max(B_diff)):<25.10e}")

# print("="*95)
# print()
# # ========== END DIAGNOSTIC ==========


@jax.jit
def iota_function(field, surface):
    # 1. Elemento de área (Jacobiano de superfície)
    # n_vec_raw = e_theta x e_phi
    B= B_on_surface(surface, field)
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
    area_element = jnp.linalg.norm(n_vec_raw, axis=-1)
    # 2. Métrica 2D da superfície (igual ao anterior)
    g_tt = jnp.sum(e_theta**2, axis=-1)
    g_pp = jnp.sum(e_phi**2, axis=-1)
    g_tp = jnp.sum(e_theta * e_phi, axis=-1)
    det_g2d = g_tt * g_pp - g_tp**2
    # 3. Componentes contravariantes via métrica inversa
    b_cov_theta = jnp.sum(B * e_theta, axis=-1)
    b_cov_phi = jnp.sum(B * e_phi, axis=-1)
    b_theta = (g_pp * b_cov_theta - g_tp * b_cov_phi) / (det_g2d + 1e-8)
    b_phi = (g_tt * b_cov_phi - g_tp * b_cov_theta) / (det_g2d + 1e-8)
    # 4. Integração usando o elemento de área
    # iota = <B^theta * area> / <B^phi * area>
    # CRITICAL: Use smooth safe division to protect gradients
    # Using sqrt(x^2 + eps) instead of abs(x) provides smooth gradients everywhere
    b_phi_safe = jnp.sqrt(b_phi**2 + 1e-28)  # Smooth absolute value for gradient stability
    iota_local = b_theta / b_phi_safe
    
    # Similarly protect the average calculation
    b_phi_avg_safe = jnp.sqrt(jnp.average(b_phi * area_element)**2 + 1e-28)
    iota_average = jnp.average(b_theta * area_element) / b_phi_avg_safe
    return iota_local, iota_average


iota_function(init_field,init_ellipse_gframe)
# # Compute iota for both surfaces
# print("\n" + "="*80)
# print("IOTA (ROTATIONAL TRANSFORM) COMPARISON")
# print("="*80)
# iota_local_rz, iota_avg_rz = iota_function(init_field, init_surface_2_RZ)
# iota_local_gf, iota_avg_gf = iota_function(init_field, init_surface_2)

# print(f"\n{'Property':<30} {'RZFourier':<25} {'GFrameFourier':<25} {'Diff %':<15}")
# print("-"*95)
# print(f"{'Iota average':<30} {float(iota_avg_rz):<25.10f} {float(iota_avg_gf):<25.10f} {100*abs(iota_avg_rz-iota_avg_gf)/abs(iota_avg_rz):<15.2e}")
# print(f"{'Iota local mean':<30} {float(jnp.mean(iota_local_rz)):<25.10f} {float(jnp.mean(iota_local_gf)):<25.10f}")
# print(f"{'Iota local std dev':<30} {float(jnp.std(iota_local_rz)):<25.10f} {float(jnp.std(iota_local_gf)):<25.10f}")
# print(f"{'Iota local min':<30} {float(jnp.min(iota_local_rz)):<25.10f} {float(jnp.min(iota_local_gf)):<25.10f}")
# print(f"{'Iota local max':<30} {float(jnp.max(iota_local_rz)):<25.10f} {float(jnp.max(iota_local_gf)):<25.10f}")

# # Check for NaN values
# has_nan_rz = jnp.any(jnp.isnan(iota_local_rz)) or jnp.isnan(iota_avg_rz)
# has_nan_gf = jnp.any(jnp.isnan(iota_local_gf)) or jnp.isnan(iota_avg_gf)
# print(f"\n{'Contains NaN - RZFourier:':<30} {has_nan_rz}")
# print(f"{'Contains NaN - GFrameFourier:':<30} {has_nan_gf}")

# # Check for Inf values
# has_inf_rz = jnp.any(jnp.isinf(iota_local_rz)) or jnp.isinf(iota_avg_rz)
# has_inf_gf = jnp.any(jnp.isinf(iota_local_gf)) or jnp.isinf(iota_avg_gf)
# print(f"{'Contains Inf - RZFourier:':<30} {has_inf_rz}")
# print(f"{'Contains Inf - GFrameFourier:':<30} {has_inf_gf}")

# # Direct iota value comparison
# iota_diff = jnp.linalg.norm(iota_local_rz - iota_local_gf)
# print(f"\n{'Iota local |difference|':<30} {float(iota_diff):<25.10e}")

# print("="*95)
# print("\n✓ Surface comparison complete!\n")
# print("="*95)
# print()

# # Original calls retained for compatibility
# iota_function(init_field, init_surface_2)
# iota_function(init_field, init_surface_2_RZ)

def loss_quasisymmetry(field,surface):
    # Calculating ∇(B · ∇|B|) on the surface
    gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
    modB = vmap(field.AbsB)(gamma_reshaped)
    B_average = jnp.mean(modB)
    grad_B_dot_grad_absB_on_surface = vmap(field.grad_B_dot_GradAbsB)(gamma_reshaped).reshape(surface.nphi, surface.ntheta, 3)/(B_average**2)
    # Calculating ∇ψ · ∇|B| on the surface
    grad_absB_on_surface = vmap(field.dAbsB_by_dX)(gamma_reshaped).reshape(surface.nphi, surface.ntheta, 3)/B_average 
    normal_cross_grad_absB_on_surface = jnp.cross(surface.unitnormal, grad_absB_on_surface,axis=-1)
    qs_residual=jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)
    loss_qs=jnp.abs(qs_residual)
    #loss_qs=jnp.linalg.norm(qs_residual)
    #loss_qs=jnp.sqrt(jnp.sum(jnp.square(qs_residual) * surface.area_element) / (jnp.sum(surface.area_element)  + 1e-12))
    return loss_qs



def metric_aspect_ratio(surface):
    norm_theta = jnp.linalg.norm(surface.gammadash_theta, axis=-1)
    norm_phi = jnp.linalg.norm(surface.gammadash_phi, axis=-1)
    # Rácio entre as dimensões locais da superfície
    ratio = norm_theta / (norm_phi + 1e-8)
    # Penaliza usando log-space penalty (smooth, symmetric, differentiable everywhere)
    # Instead of: ratio**2 + (1/ratio)**2  which has dangerous reciprocal
    ratio_safe = jnp.sqrt(ratio**2 + 1e-12)
    log_ratio = jnp.log(ratio_safe)
    penalty = jnp.mean(log_ratio**2)  # Zero at ratio=1, grows smoothly away
    return penalty


# --- 1. CÁLCULO DO GRADIENTE RADIAL (REAL) ---
@jax.jit
def get_dB_dnormal(field, surface):
    gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
    grad_B= vmap(field.dAbsB_by_dX)(gamma_reshaped).reshape(surface.nphi, surface.ntheta, 3)
    # Vetor normal unitário (direção do gradiente de fluxo psi)
    return jnp.sum(grad_B * surface.unitnormal, axis=-1)
    
@jax.jit
def vacuum_well(field, surface):
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    dS = surface.area_element
    unit_normal = surface.unitnormal
    # 1. Métrica e Segundas Derivadas para Curvatura Média H
    E = jnp.sum(e_theta * e_theta, axis=-1)
    F = jnp.sum(e_theta * e_phi, axis=-1)
    G = jnp.sum(e_phi * e_phi, axis=-1)
    det_g = E * G - F**2
    L = jnp.sum(surface.gammadashdash_theta * unit_normal, axis=-1)
    M = jnp.sum(surface.gammadashdash_theta_phi * unit_normal, axis=-1)
    N = jnp.sum(surface.gammadashdash_phi * unit_normal, axis=-1)
    # Curvatura Média (H = (kappa_1 + kappa_2)/2)
    H = (E * N - 2 * F * M + G * L) / (2 * det_g + 1e-12)
    # 2. O Módulo de B
    b_grid = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
    dBdn = get_dB_dnormal(field, surface)
    # 3. O termo combinado (Variação de B + Variação de dl)
    # No vácuo, o poço é dominado por < (1/B^2) * (dB/dn + B * H) >
    # Note que se dB/dn = -B * kappa_n, e H contém kappa_n, 
    # estes termos reforçam-se para descrever a expansão do volume.
    well_integrand = -( (dBdn / (b_grid + 1e-12)) + H ) * (dS / b_grid)
    well_measure = jnp.sum(well_integrand) / jnp.sum(dS / b_grid)
    return well_measure

@jax.jit
def loss_minmax_toroidal(field,surface):
    """
    Força a fase do primeiro harmónico toroidal a ser um cosseno positivo.
    Garante máximo em 0 e mínimo em pi.
    """
    b_grid = jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
    b_grid = b_grid / jnp.mean(b_grid)  # Normalizar para evitar escalas muito grandes
    # FFT apenas no eixo toroidal (phi)
    coeffs_phi = jnp.fft.rfft2(b_grid)
    # O modo n=1 é o fundamental.
    # Penaliza modos toroidais mais altos para manter o perfil simples
    loss_toroidal = jnp.sqrt(jnp.sum(jnp.abs(coeffs_phi[1:, :])**2))
    # 1. Penalizar o desvio do Máximo esperado (em phi=0, theta=0)
    # Assumindo que o índice [0,0] corresponde a (theta=0, phi=0)
    val_at_zero = b_grid[0, 0]
    loss_min = val_at_zero - jnp.min(b_grid)
    # 2. Penalizar o desvio do Mínimo esperado (em phi=pi)
    mid_theta = b_grid.shape[1] // 2
    val_at_pi = b_grid[0, mid_theta]
    loss_max = val_at_pi - jnp.max(b_grid)
#   return jnp.array([loss_poloidal, loss_phase, loss_sign])
    return loss_toroidal, loss_min, loss_max

# @jax.jit
# def vacuum_well_loss_general(field, surface):
#     # 1. Vetores tangentes e Normal
#     e_theta = surface.gammadash_theta
#     e_phi = surface.gammadash_phi
#     n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
#     dS = jnp.sqrt(jnp.sum(n_vec_raw**2, axis=-1))
#     unit_normal = n_vec_raw / (dS[..., None] + 1e-12)

#     # 2. Primeira Forma Fundamental (Métrica)
#     E = jnp.sum(e_theta * e_theta, axis=-1)
#     F = jnp.sum(e_theta * e_phi, axis=-1)
#     G = jnp.sum(e_phi * e_phi, axis=-1)
#     det_g = E * G - F**2

#     # 3. Segundas Derivadas (usando jnp.gradient ou diferenciação automática)
#     e_theta_theta = surface.gammadashdash_theta
#     e_phi_phi = surface.gammadashdash_phi
#     e_theta_phi = surface.gammadashdash_theta_phi

#     # 4. Segunda Forma Fundamental
#     L = jnp.sum(e_theta_theta * unit_normal, axis=-1)
#     M = jnp.sum(e_theta_phi * unit_normal, axis=-1)
#     N = jnp.sum(e_phi_phi * unit_normal, axis=-1)

#     # 5. Curvatura Média Rigorosa (H)
#     H = (E * N - 2 * F * M + G * L) / (2 * det_g + 1e-12)
    
#     # 6. Integral do Poço
#     b_mag = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
#     well_integral = jnp.sum(-b_mag * H * dS) / jnp.sum(dS)
    
#     return jnp.square(jnp.maximum(0, 0.05 - well_integral))





""" Setting the losses weights and targets """
Enforce_R0_magnitude=1000.
Enforce_R1_magnitude=1.
Enforce_Volume_magnitude=1.
Enforce_Area_magnitude=1.
Enforce_Toroidal_Flux_magnitude=1.
LENGTH_WEIGHT = 1.; LENGTH_TARGET = 40.
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 0.5
NORMAL_FIELD_WEIGHT = 1.
BDOTN_TARGET=1.e-3
Area_Target=target_surface_2.area
Volume_Target=target_surface_2.volume
Toroidal_Flux_target=toroidal_flux(target_surface_2,target_field)
R0_target=surface_major_radius
QS_target=0.01*jnp.max(loss_quasisymmetry(target_field,target_surface_2))
iota_precisision_target=0.0001
iota_target_local,iota_target_avg=iota_function(target_field, target_surface_2)
AR_target=metric_aspect_ratio(target_surface_2)
contours_target, _, _ = loss_minmax_toroidal(target_field, target_surface_2)

#Weights for scaling in case it is necessary
weight_bdotn=1.#7.334582e-05
weight_length=1.
weight_curvature=1.
weight_R0=Enforce_R0_magnitude
weight_volume=Enforce_Volume_magnitude
weight_area=Enforce_Area_magnitude
weight_toroidal_flux=Enforce_Toroidal_Flux_magnitude
weight_iota=1.
weight_qs= 1.
weight_aspect_ratio=1.
weight_contour=1.
weight_min=1.
weight_max=1.
weight_iota_variance_axis=1.
weight_mirror=100.
weight_iota_axis=1.






# def B_on_surface_TEST_2(surface, field, nphi_fine=None, ntheta_fine=None):
#     gamma = surface.gamma  # (nphi, ntheta, 3)
#     nphi, ntheta = surface.nphi, surface.ntheta
#     if nphi_fine and ntheta_fine:
#         # Criar os índices para a nova grelha (de 0 a n-1)
#         phi_idx = jnp.linspace(0, nphi - 1, nphi_fine)
#         theta_idx = jnp.linspace(0, ntheta - 1, ntheta_fine)
#         PHI_IDX, THETA_IDX = jnp.meshgrid(phi_idx, theta_idx, indexing='ij')
#         coords = jnp.stack([PHI_IDX, THETA_IDX])
#         # Interpolar gamma e unitnormal (vmap sobre a componente XYZ)
#         # mode='wrap' garante a periodicidade toroidal/poloidal
#         interp_fn = lambda data: map_coordinates(data, coords, order=1, mode='wrap')
#         # Aplicamos aos eixos x, y, z (eixo 2)
#         gamma_fine = jnp.stack([interp_fn(gamma[..., i]) for i in range(3)], axis=-1)
#         unitnormal_fine = jnp.stack([interp_fn(surface.unitnormal[..., i]) for i in range(3)], axis=-1)
#         current_gamma = gamma_fine.reshape(-1, 3)
#         current_normal = unitnormal_fine
#         c_nphi, c_ntheta = nphi_fine, ntheta_fine
#     else:
#         current_gamma = gamma.reshape(-1, 3)
#         current_normal = surface.unitnormal
#         c_nphi, c_ntheta = nphi, ntheta
#     # Cálculo do campo
#     B_on_surface = vmap(field.B)(current_gamma).reshape(c_nphi, c_ntheta, 3)
#     # B.n / |B|
#     bdotn_over_b = jnp.sum(B_on_surface * current_normal, axis=2) / jnp.linalg.norm(B_on_surface, axis=2)
#     return jnp.max(jnp.abs(bdotn_over_b))




""" Creating the loss functions """
def loss_bdotn(field, surface):
    return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))


def BdotN_constraint(field,surface,target_tol=1.e-6,P=50.):
    bdotn_over_b = BdotN_over_B(surface, field)
    g_local =jnp.abs(bdotn_over_b) - target_tol
    # Ajustamos P para a escala física (ex: se target é 1e-4, P será 200.000)
    P = 20.0 / jnp.maximum(1e-8, target_tol) 
    # LogSumExp estável
    #g_max = (1.0 / P) * jax.scipy.special.logsumexp(P * g_local)
    g_max=(1.0 / P) *  jax.scipy.special.logsumexp(P * g_local) - (jnp.log(bdotn_over_b.size) / P)
    return g_max

def loss_length_constraint(field,target_length=40.0,N_coils=4):
    #return jnp.maximum(0, field.coils.length - target_length)
    #This was divided by target_length to make it scale invariant, but it seems to cause issues with the optimization, so I am trying without it for now
    #return jnp.maximum(0, (field.coils.length - target_length)/target_length)
    return field.coils.length[0:N_coils] - target_length

def loss_curvature_contraint(field,target_curvature=0.5,N_coils=4):
    #return jnp.maximum(0, field.coils.curvature - target_curvature)
    #This was divided by target_curvature to make it scale invariant, but it seems to cause issues with the optimization, so I am trying without it for now
    #return jnp.maximum(0, (field.coils.curvature - target_curvature)/target_curvature)
    return field.coils.curvature[0:N_coils] - target_curvature





""" Surface Constraints """

def loss_area_contraint(surface,target_area=900.):
#    return jnp.maximum(0, -(surface.area - target_area)/target_area)
    return (surface.area - target_area)#/target_area

def loss_volume_contraint(surface,target_volume=2000):
#    return jnp.maximum(0, -(surface.volume - target_volume)/target_volume)
    return (surface.volume - target_volume)#/target_volume

def loss_toroidal_flux_contraint(field,surface,target_toroidal_flux=1.e-5):
#    return jnp.maximum(0, (toroidal_flux(surface,field) - target_toroidal_flux)/jnp.abs(target_toroidal_flux))
    return (toroidal_flux(surface,field) - target_toroidal_flux)#/jnp.abs(target_toroidal_flux)

def loss_R0_contraint(surface,target_R0):
    # For GFrame surfaces with Fourier axis, constrain the n=0 (average) radius mode
    # Axis modes are: n = nfp * [-axis_ntor, ..., -1, 0, 1, ..., +axis_ntor]
    # So n=0 is at the CENTER index of the array, not at position 0
    if hasattr(surface, "axis_rc_coeffs") and surface.axis_rc_coeffs is not None and len(surface.axis_rc_coeffs) > 0:
        axis_ntor = surface.axis_ntor
        # The n=0 mode is at index axis_ntor (center of the mode array)
        # This is because the indices go from -axis_ntor to +axis_ntor
        n0_index = axis_ntor  
        return (surface.axis_rc_coeffs[n0_index] - target_R0)
    # For RZ surfaces, use rc[0] (which is the constant R part)
    if hasattr(surface, "rc"):
        return (surface.rc[0] - target_R0)
    # Fallback: use axis major radius directly
    return (surface._axis_major_radius - target_R0)




# def loss_quasisymmetry_constraint(field,surface,target_QS=0.00001):
#     # Calculate residual once and reuse
#     qs_residual_local = loss_quasisymmetry(field,surface)
#     g_local = qs_residual_local - target_QS
#     # Ajustamos P para a escala física (ex: se target é 1e-4, P será 200.000)
#     P = 20.0 / jnp.maximum(1e-8, target_QS) 
#     # LogSumExp estável
#     g_max = (1.0 / P) * jax.scipy.special.logsumexp(P * g_local) - (jnp.log(qs_residual_local.size) / P)    
#     return g_max

def loss_quasisymmetry_constraint(field,surface,target_QS=0.00001):
    # Calculate residual once and reuse
    qs_residual_local = loss_quasisymmetry(field,surface)
    surface_area_element=surface.area_element
    qs_average = jnp.sum(qs_residual_local * surface_area_element) / (jnp.sum(surface_area_element) + 1e-12)
    return qs_average



@jax.jit
def loss_minmax_constraint(field,surface, M_target=1,N_target=0):
    """
    Força a fase do primeiro harmónico toroidal a ser um cosseno positivo.
    Garante máximo em 0 e mínimo em pi.
    """
    b_grid = jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
    b_grid = b_grid / jnp.mean(b_grid)  # Normalizar para evitar escalas muito grandes
    # FFT apenas no eixo toroidal (phi)
    coeffs_phi = jnp.fft.rfft2(b_grid)
    # O modo n=1 é o fundamental.
    # Queremos que ele seja puramente Real e Positivo: B ~ cos(phi)
    mode_1 = coeffs_phi[M_target,N_target]
    # Penaliza a parte imaginária (desvio de fase)
    loss_phase = jnp.abs(jnp.imag(mode_1)) 
    # Penaliza se a parte real for negativa (mínimo vira máximo)
    loss_sign = jnp.maximum(0, -jnp.real(mode_1)) 
    # Penaliza modos toroidais mais altos para manter o perfil simples
    loss_poloidal = jnp.sqrt(jnp.sum(jnp.abs(coeffs_phi[:, 1:])**2))
    # 1. Penalizar o desvio do Máximo esperado (em phi=0, theta=0)
    # Assumindo que o índice [0,0] corresponde a (theta=0, phi=0)
    val_at_zero = b_grid[0, 0]
    loss_max = jnp.abs(val_at_zero - jnp.max(b_grid))/jnp.abs(jnp.max(b_grid))
    # 2. Penalizar o desvio do Mínimo esperado (em phi=pi)
    mid_phi = b_grid.shape[0] // 2
    val_at_pi = b_grid[mid_phi, 0]
    loss_min = jnp.abs(val_at_pi - jnp.min(b_grid))/jnp.abs(jnp.min(b_grid))
#   return jnp.array([loss_poloidal, loss_phase, loss_sign])
    return jnp.array([loss_poloidal*0.1, loss_min*10., loss_max*10.])



def toroidal_contours_penalty(field,surface,target_contours=430.):
    loss_toroidal, _, _ = loss_minmax_toroidal(field, surface)
    return loss_toroidal-target_contours

def min_penalty(field,surface):
    _, loss_min, _ = loss_minmax_toroidal(field, surface)
    return loss_min

def max_penalty(field,surface):
    _, _, loss_max = loss_minmax_toroidal(field, surface)
    return loss_max



@jax.jit
def symmetry_penalty(field, surface,  M=0, N=1):
    """
    Calcula o custo de quasisimetria para qualquer tipo (QA, QP, QH).
    
    grad_B: (Nth, Nph, 3) - Gradiente 3D de B
    B_dot_grad_B: (Nth, Nph, 3) - Termo de aceleração (B . grad) B
    gamma_grid: (Nth, Nph, 3) - Coordenadas (x, y, z)
    e_theta, e_phi: (Nth, Nph, 3) - Vetores tangentes poloidal e toroidal lógico
    sym_type: "QA", "QP" ou "QH"
    M, N: Números de modo para a simetria (N=0 para QA, M=0 para QP, M,N para QH)
    """
    # 2. Definir o Vetor de Simetria Físico (V_sym) no Laboratório
    x, y = surface.gamma[..., 0], surface.gamma[..., 1]
    v_tor_lab = jnp.stack([-y, x, jnp.zeros_like(x)], axis=-1) # Direção toroidal real
    v_pol_lab = surface.gammadash_theta  # Direção poloidal real
    v_target = M * v_pol_lab + N * v_tor_lab
    grad_B= vmap(field.dAbsB_by_dX)(gamma_reshaped).reshape(surface.nphi, surface.ntheta, 3)
    # 3. Projeção Tangencial (O filtro para não matar o gradiente radial)
    # v_proj = v_target - (v_target . n) * n
    v_proj = v_target - jnp.einsum('ijk,ijk->ij', v_target, surface.unitnormal)[..., jnp.newaxis] * surface.unitnormal
    v_proj_unit = v_proj / (jnp.linalg.norm(v_proj, axis=-1, keepdims=True) + 1e-8)
    # 4. Penalização Direcional (A "Bússola")
    # Força o gradiente de B a ser zero APENAS na direção v_proj
    f_direction = jnp.einsum('ijk,ijk->ij', grad_B, v_proj_unit)
    cost_direction = jnp.sum(f_direction**2 * surface.area_element) / (jnp.sum(surface.area_element) + 1e-12)
    # Custo Total: Física da Simetria + Direcionamento
    return cost_direction




@jax.jit
def calculate_local_shear(e_theta, e_phi, normal, iota_local):
    """
    Calcula o cisalhamento local (Local Shear) na superfície.
    e_theta, e_phi: (3, N_phi, N_theta)
    normal: (3, N_phi, N_theta)
    iota_local: (N_phi, N_theta)
    """
    # 1. Definir o vetor unitário ao longo da linha de campo (b_hat)
    # No sistema de coordenadas da superfície: B_vec = e_phi + iota_local * e_theta
    b_vec = e_phi + iota_local * e_theta
    b_mag = jnp.sqrt(jnp.sum(b_vec**2, axis=0))
    b_hat = b_vec / b_mag
    # 2. Calcular o gradiente do ângulo da linha de campo (beta)
    # beta é a inclinação local: arctan(iota_local * |e_theta| / |e_phi|)
    # Simplificação: gradiente do iota_local em relação aos índices da grelha
    diota_dphi, diota_dtheta = jnp.gradient(iota_local)
    # 3. O Local Shear (S) relaciona-se com a variação de iota na superfície
    # Integrando com a métrica da superfície (g_ij)
    g_tt = jnp.sum(e_theta * e_theta, axis=0)
    g_pp = jnp.sum(e_phi * e_phi, axis=0)
    # Proxy de Shear: quão rápido a inclinação muda entre linhas vizinhas
    # Escalado pela métrica local para ser independente da parametrização
    local_shear = jnp.abs(diota_dtheta / jnp.sqrt(g_tt) - diota_dphi / jnp.sqrt(g_pp))
    return local_shear







def loss_mirror(field,surface,target_mirror=0.29):
    modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1).T
    # 4. Mirror Ratio (Profundidade do Poço)
    b_min, b_max = jnp.min(modB), jnp.max(modB)
    mirror_val = (b_max - b_min) / (b_max + b_min)
    return mirror_val - target_mirror




""" Creating the loss functions """
def loss_iota(field, surface,target_iota_local=0.41,target_iota_avg=0.41):
    iota,iota_average= iota_function(field, surface)
    iota_variance=0.3
    iota_precision=0.0001
    iota_max=0.47
    iota_min=0.405
    #iota_residual=(jnp.abs((iota-target_iota_local))-iota_precision)/iota_precision*234
    #iota_residual=jnp.abs((iota-target_iota_local))/target_iota_local
    #iota_residual=(jnp.abs((iota-iota_target))-iota_precision)/iota_precision
    #iota_residual=(jnp.append(jnp.abs(iota_average-iota_target)/jnp.abs(iota_target),jnp.maximum(jnp.abs(iota-iota_target)/jnp.abs(iota_target)-iota_variance,0.0)/iota_variance)-iota_precision)/iota_precision
    #iota_residual=jnp.append(jnp.abs(iota_average-target_iota_avg),(jnp.maximum(0.0,iota_min-jnp.min(iota))+jnp.maximum(0.0,jnp.max(iota)-iota_max)))
    #iota_residual=jnp.append(jnp.abs(iota_average-target_iota_avg),jnp.maximum(0.0,iota_min-jnp.min(iota))+jnp.maximum(0.0,jnp.max(iota)-iota_max))*1000.
    #iota_residual=jnp.sum(jnp.append(jnp.abs(iota_average-target_iota_avg)/jnp.abs(target_iota_avg),(jnp.maximum(0.0,iota_min-iota)/jnp.abs(iota_min)+jnp.maximum(0.0,iota-iota_max)/jnp.abs(iota_max))))
    #iota_residual=(jnp.append(jnp.abs(iota_average-iota_target_avg)/jnp.abs(iota_target_avg),jnp.maximum(jnp.abs(iota-iota_target_avg)/jnp.abs(iota_target_avg)-iota_variance,0.0)/iota_variance)-iota_precision)/iota_precision    
    #iota_residual=(jnp.abs((iota_average-iota_target))-iota_precision)/iota_precision
    ##iota_residual=jnp.abs(jnp.append(jnp.abs(iota_average-target_iota_avg),jnp.maximum(0.0,iota_min-iota)+jnp.maximum(0.0,iota-iota_max))-iota_precision)/iota_precision    
    #residual_iota_average=jnp.maximum(jnp.abs(iota_average-target_iota_avg)/jnp.abs(target_iota_avg)-iota_precision,0.0)/iota_precision
    #residual_iota_average=jnp.abs(iota_average-target_iota_avg)/jnp.abs(target_iota_avg)/iota_precision
    residual_iota_average=(iota_average-target_iota_avg)#/jnp.abs(target_iota_avg)#iota_precision
    ###residual_iota_max=jnp.maximum(0.0,jnp.max(iota)-iota_max)/jnp.abs(iota_max)
    ###residual_iota_min=jnp.maximum(0.0,iota_min-jnp.min(iota))/jnp.abs(iota_min)
    residual_iota_variance=jnp.sum(surface.area_element*(iota-iota_average)**2)/jnp.sum(surface.area_element)
    return residual_iota_average,residual_iota_variance
    #return jnp.array([residual_iota_average,residual_iota_max,residual_iota_min])


def loss_iota_average(field, surface,target_iota_local=0.41,target_iota_avg=0.41):
    residual_iota_average,_=loss_iota(field, surface,target_iota_local,target_iota_avg)
    return residual_iota_average

def loss_iota_variance(field, surface,target_iota_local=0.41,target_iota_avg=0.41):
    _, residual_iota_variance=loss_iota(field, surface,target_iota_local,target_iota_avg)
    return residual_iota_variance

def loss_well_constraint(field, surface, target_well=0.05):
    well_value = vacuum_well(field, surface)
    # Penaliza se o poço for menor que o alvo (queremos poço profundo)
    loss_well = jnp.maximum(0, target_well - well_value)/target_well 
    return loss_well

def metric_aspect_ratio_penalty(surface,target_AR=500.0):
    penalty = metric_aspect_ratio(surface)
    return penalty - target_AR

def loss_normal(surface):
    residual=jnp.sum((surface.area_element-jnp.mean(surface.area_element))**2)/jnp.sum(surface.area_element)
    return residual




def boozer_penalty(field, surface):
    """
    Calcula o resíduo da condição de Boozer: J * B^2 = constante.
    Parâmetros:
    r_x, r_y, r_z: Grids 2D com as coordenadas cartesianas da superfície.
    b_mag: Grid 2D com o módulo do campo magnético B(phi, theta).
    phi, theta: Vetores 1D das coordenadas angulares.
    """
    # 1. Calcular os vetores tangentes (derivadas parciais)
    B_grid =jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
    j_surf = jnp.linalg.norm(n_vec_raw, axis=-1)
    # 3. Calcular a métrica K = J * B^2
    # Numa superfície de Boozer pura, isto deve ser constante
    k_local = j_surf * (B_grid**2)
    # 4. Calcular o resíduo (erro relativo em relação à média)
    k_mean = jnp.mean(k_local)
    residuos = (k_local / k_mean) - 1.0
    # Métrica escalar para o otimizador (RMS)
    boozer_error = jnp.sqrt(jnp.mean(residuos**2))
    return boozer_error






# Create the constraints
penalty = 1. #Intial penalty values
multiplier=0.0 #Initial lagrange multiplier values
#Initializing first tolerances for the inner minimisation loop iteration
omega=1./penalty
eta=1./penalty**0.1
model_lagrangian='Standard'  #Use standard augmented lagragian suitable for bounded optimizers 
#Since we are using LBFGS-B from jaxopt, model_mu will be updated with tolerances so we do not need to difinte the model
model_mu='Mu_Adaptative_2'




beta=1.01                                     #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-5   #desired contraint tolerance, associated with variation of contraints
decrease_tol=0.99

""" Defining constraints"""
#Coils

# field_constraint = alm.ScaledConstraint(alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_bdotn ),auto_scale=True)
# length_constraint=alm.ScaledConstraint(alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_length),auto_scale=True)
# curvature_constraint=alm.ScaledConstraint(alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_curvature),auto_scale=True)

# area_constraint=alm.ScaledConstraint(alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_area),auto_scale=True)
# volume_constraint=alm.ScaledConstraint(alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_volume),auto_scale=True)
# toroidal_flux_constraint=alm.ScaledConstraint(alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_toroidal_flux),auto_scale=True)
# R0_constraint=alm.ScaledConstraint(alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_R0),auto_scale=True)
# aspect_ratio_constraint=alm.ScaledConstraint(alm.eq(metric_aspect_ratio_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_aspect_ratio),auto_scale=True)

# QS_constraint=alm.ScaledConstraint(alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_qs),auto_scale=True)
# iota_constraint=alm.ScaledConstraint(alm.eq(loss_iota_2,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_iota),auto_scale=True)

eta_tol_field=1.e-5*BDOTN_TARGET
eta_tol_field_axis=1.e-5*BDOTN_TARGET
eta_tol_length=1.e-5*LENGTH_TARGET
eta_tol_curvature=1.e-5*CURVATURE_TARGET


eta_tol_area=1.e-5*Area_Target
eta_tol_volume=1.e-5*Volume_Target
eta_tol_toroidal_flux=1.e-5*jnp.abs(Toroidal_Flux_target)
eta_tol_R0=1.e-5*R0_target
eta_tol_aspect_ratio=1.e-5*AR_target
eta_tol_QS=1.e-5*QS_target
eta_tol_iota=1.e-5*jnp.abs(iota_target_avg)
eta_tol_iota_variance_axis=1.e-5
eta_tol_contour=1.e-5*contours_target
eta_tol_contour_eq=1.e-10
eta_tol_boozer=1.e-10
eta_tol_mirror=1.e-10


field_constraint = alm.ScaledConstraint(alm.ineq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_bdotn,eta_tol=eta_tol_field,use_analytical_penalty=True),auto_scale=False)
length_constraint=alm.ScaledConstraint(alm.ineq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_length,eta_tol=eta_tol_length,use_analytical_penalty=True),auto_scale=False)
curvature_constraint=alm.ScaledConstraint(alm.ineq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_curvature,eta_tol=eta_tol_curvature,use_analytical_penalty=True),auto_scale=False)
field_constraint_axis = alm.ScaledConstraint(alm.ineq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_bdotn,eta_tol=eta_tol_field_axis,use_analytical_penalty=True),auto_scale=False)

area_constraint=alm.ScaledConstraint(alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_area,eta_tol=eta_tol_area),auto_scale=False)
volume_constraint=alm.ScaledConstraint(alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_volume,eta_tol=eta_tol_volume),auto_scale=False)
toroidal_flux_constraint=alm.ScaledConstraint(alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_toroidal_flux,eta_tol=eta_tol_toroidal_flux),auto_scale=False)
R0_constraint=alm.ScaledConstraint(alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_R0,eta_tol=eta_tol_R0),auto_scale=False)
aspect_ratio_constraint=alm.ScaledConstraint(alm.ineq(metric_aspect_ratio_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_aspect_ratio,eta_tol=eta_tol_aspect_ratio,use_analytical_penalty=True),auto_scale=False)
iota_variance_constraint=alm.ScaledConstraint(alm.eq(loss_iota_variance,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_iota_variance_axis,eta_tol=eta_tol_iota_variance_axis),auto_scale=False)


#QS_constraint=alm.ScaledConstraint(alm.ineq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_qs,eta_tol=eta_tol_QS,use_analytical_penalty=True),auto_scale=False)
QS_constraint=alm.ScaledConstraint(alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_qs,eta_tol=eta_tol_QS),auto_scale=False)
iota_constraint=alm.ScaledConstraint(alm.eq(loss_iota_average,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_iota,eta_tol=eta_tol_iota),auto_scale=False)
toroidal_contours_constraint=alm.ScaledConstraint(alm.ineq(toroidal_contours_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_contour,eta_tol=eta_tol_contour,use_analytical_penalty=True),auto_scale=False)
toroidal_contours_constraint_eq=alm.ScaledConstraint(alm.eq(toroidal_contours_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_contour,eta_tol=eta_tol_contour_eq),auto_scale=False)
boozer_constraint=alm.ScaledConstraint(alm.eq(boozer_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=1.,eta_tol=eta_tol_boozer),auto_scale=False)




C_normal_field_constraint = alm.SelectiveConstraint(field_constraint, "field", "surface", target_tol=BDOTN_TARGET)
C_length_constraint = alm.SelectiveConstraint(length_constraint, "field", target_length=LENGTH_TARGET,N_coils=N_COILS)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field", target_curvature=CURVATURE_TARGET,N_coils=N_COILS)
C_area_constraint = alm.SelectiveConstraint(area_constraint, "surface", target_area=Area_Target)
C_volume_constraint = alm.SelectiveConstraint(volume_constraint, "surface", target_volume=Volume_Target)
C_toroidal_flux_constraint = alm.SelectiveConstraint(toroidal_flux_constraint, "field", "surface", target_toroidal_flux=Toroidal_Flux_target)
C_R0_constraint = alm.SelectiveConstraint(R0_constraint,  "surface", target_R0=R0_target)
C_aspect_ratio_constraint = alm.SelectiveConstraint(aspect_ratio_constraint,  "surface", target_AR=AR_target)
C_QS_constraint = alm.SelectiveConstraint(QS_constraint,"field", "surface", target_QS=QS_target)
C_iota_constraint = alm.SelectiveConstraint(iota_constraint,  "field", "surface",target_iota_local=iota_target_local, target_iota_avg=iota_target_avg)
C_toroidal_contours_constraint = alm.SelectiveConstraint(toroidal_contours_constraint, "field", "surface", target_contours=contours_target)
C_toroidal_contours_constraint_eq = alm.SelectiveConstraint(toroidal_contours_constraint_eq, "field", "surface", target_contours=0.0)
C_boozer_constraint = alm.SelectiveConstraint(boozer_constraint, "field", "surface")
C_iota_variance_constraint = alm.SelectiveConstraint(iota_variance_constraint,  "field", "surface",target_iota_local=iota_target_local, target_iota_avg=iota_target_avg)


##TEST WITHOUT QS
C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,C_R0_constraint,
C_area_constraint,C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,
C_iota_constraint,C_QS_constraint,C_toroidal_contours_constraint)

# C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,C_R0_constraint,
# C_area_constraint,C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,
# C_iota_constraint,C_toroidal_contours_constraint_eq, C_boozer_constraint)



C_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_area_constraint.dependencies = {"surface": init_surface_2}
C_volume_constraint.dependencies = {"surface": init_surface_2}
C_toroidal_flux_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_R0_constraint.dependencies = {"surface": init_surface_2}
C_QS_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_iota_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_aspect_ratio_constraint.dependencies = {"surface": init_surface_2}
C_toroidal_contours_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_boozer_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_toroidal_contours_constraint_eq.dependencies = {"field": init_field, "surface": init_surface_2}   
C_iota_variance_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

C_Total_constraint.dependencies = {"field": init_field, "surface": init_surface_2}  
#C_Total_constraint.dependencies = {"field": init_field, "surface": init_surface_2}  


#If loss=cost_function(x) is not prescribed, f(x)=0 is considered, uncomment second line to use B dot N as a loss and not a constraint
#ALM=alm.ALM_model_jaxopt_lbfgsb(constraints,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)

# Per-constraint weighting options:
#   False: no weighting (default, fastest)
#   'constraint': one weight per constraint (fast, ~seconds, low memory)
#   'element': element-wise weights (slow, ~minutes, higher memory, most precise)
# TEMPORARY FIX: Disabled per-constraint weights and scaled params to avoid NaN gradients
ALM = alm.ALM_model_jaxopt_lbfgsb(
    constraints=C_Total_constraint,
    model_lagrangian=model_lagrangian,
    model_mu=model_mu,
    beta=beta,
    mu_max=mu_max,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon,
    eta_tol=eta_tol,
    omega_tol=omega_tol,
    decrease_tol=decrease_tol,
    use_per_constraint_weights='element',  # DISABLED - causes NaN gradients
    weight_g_epsilon=1.e-4,
    use_perturbation_weights=True,  # TEST: Disable perturbations to check if NaN persists
    perturbation_magnitude=0.01,
    perturbation_mode='relative',
    perturbation_seed=42,
    use_scaled_main_params=True,  # DISABLED - causes NaN gradients
)




# Get starting DOFs - this will respect frozen DOFs because init_surface_2 is already frozen
# The starting_dofs combines coils.dofs and surface.dofs (which excludes frozen DOFs)
starting_dofs = C_Total_constraint.starting_dofs
starting_dofs_scaled = alm._get_constraints_scaled_dofs(C_Total_constraint)
print(f"Starting DOFs shape: {starting_dofs.shape}")
print(f"Starting DOFs scaled shape: {starting_dofs_scaled.shape}")

#Initializing lagrange multipliers
lagrange_params = C_Total_constraint.init(starting_dofs)








# Print auto-scaling factors for each constraint
def print_weights():
    print("\n" + "="*70)
    print("AUTO-SCALING FACTORS FOR CONSTRAINTS")
    print("="*70)
    constraint_names = [
        "Normal Field (BdotN)",
        "Coil Length",
        "Coil Curvature", 
        "Major Radius R0",
        "Volume",
        "Area",
        "Toroidal Flux",
        "Aspect Ratio",
        "Iota",
        "Quasisymmetry"
    ]
    selective_constraints = [
        C_normal_field_constraint,
        C_length_constraint,
        C_curvature_constraint,
        C_R0_constraint,
        C_volume_constraint,
        C_area_constraint,
        C_toroidal_flux_constraint,
        C_aspect_ratio_constraint,
        C_iota_constraint,
        C_QS_constraint
    ]
    for name, sel_constraint in zip(constraint_names, selective_constraints):
        # Access the underlying ScaledConstraint
        scaled_constraint = sel_constraint.constraint
        #print(f"\n{name}:")
        #print(f"  auto_scale: {scaled_constraint.auto_scale}")
        #print(f"  max_scale_factor: {scaled_constraint.max_scale_factor}")
        if hasattr(scaled_constraint, '_initial_scale_scalar'):
            if scaled_constraint._initial_scale_scalar is not None:
                scale = float(scaled_constraint._initial_scale_scalar)
                print(f"{name}:  → scale factor = {scale:.6e} (auto-scaled)")
            else:
                print(f"{name}:  → scale factor = NOT COMPUTED (auto_scale=True but _initial_scale_scalar=None)")
                print(f"{name}:      This usually means auto-scaling evaluation failed!")
        elif hasattr(scaled_constraint, 'scale_factor'):
            print(f"{name}:  → scale factor = {scaled_constraint.scale_factor:.6e} (manual fallback)")
        else:
            print(f"{name}:  → no scaling info available")
    print("="*70)

print_weights()

#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
##params = starting_dofs, lagrange_params
params = starting_dofs_scaled, lagrange_params
#This is just to initialize an empty state for the lagrange multiplier update and get some information
params, lag_state, grad, info = ALM.init(params)
scale = alm._get_constraints_scaling_factors(C_Total_constraint)
C_Total_constraint.loss(params[1],params[0] / jnp.maximum(scale, 1e-12))

print(f'grad_coils: {jnp.linalg.norm(grad[0][:160]):g}, grad_surface: {jnp.linalg.norm(grad[0][160:186]):g},grad_surface axis: {jnp.linalg.norm(grad[0][186:]):g}')




""" Optimizing  with alm"""
t_start = time()

i=0
while i<=60000 and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info= ALM.update(params,lag_state,grad,info)    
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2],params[1]):g}, grad_coils: {jnp.linalg.norm(grad[0][0:160]):g}, grad_surface: {jnp.linalg.norm(grad[0][160:186]):g}, grad_surface axis: {jnp.linalg.norm(grad[0][186:]):g}')
    i=i+1

t_end = time()


params, lag_state,grad,info= ALM.update(params,lag_state,grad,info)  
print(f' loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2], params[1]):g}, grad_surface: {jnp.linalg.norm(grad[0][0:82]):g}, grad_coils: {jnp.linalg.norm(grad[0][82:-1]):g}')

C_Total_constraint.loss(params[1],params[0])


opt_field_alm = C_Total_constraint.dofs_to_pytree(params[0] / jnp.maximum(scale, 1e-12))[0]
opt_coils_alm = opt_field_alm.coils

opt_surface_alm_2= C_Total_constraint.dofs_to_pytree(params[0] / jnp.maximum(scale, 1e-12))[1]
opt_surface_alm_1= C_Total_constraint.dofs_to_pytree(params[0] / jnp.maximum(scale, 1e-12))[2]
# Verify frozen DOF was preserved
print(f"\n=== Frozen DOF Verification ===")
print(f"Initial axis R0 (frozen): {init_surface_2.axis_rc_coeffs[0]:.6f}")
print(f"Optimized axis R0 (should be same): {opt_surface_alm_2.axis_rc_coeffs[0]:.6f}")
print(f"Axis rc coeffs shape: {opt_surface_alm_2.axis_rc_coeffs.shape}")
print(f"Axis zs coeffs shape: {opt_surface_alm_2.axis_zs_coeffs.shape}")
print(f"Free DOFs in optimized surface: {len(opt_surface_alm_2.dofs)}")


print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial B dot N:", jnp.max(BdotN_over_B(init_surface_2, init_field)))    
print("B dot N after optimization alm:", jnp.max(BdotN_over_B(opt_surface_alm_2, opt_field_alm)))
print("B dot N after optimization alm axis:", jnp.max(BdotN_over_B(opt_surface_alm_1, opt_field_alm)))
print("Initial curvature :", jnp.average(init_field.coils.curvature,axis=0))    
print("Curvature after optimization alm:",jnp.average(opt_field_alm.coils.curvature,axis=0))
print("Curvature target:",CURVATURE_TARGET)
print("Initial length :", init_field.coils.length)    
print("Length after optimization alm:",opt_field_alm.coils.length)
print("Length target:",LENGTH_TARGET)




fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(131, projection='3d')
init_coils.plot(ax=ax1, show=False,label='Initial coils')
init_surface_2.plot(ax=ax1, show=False)
ax2 = fig.add_subplot(132, projection='3d')
opt_surface_alm_2.plot(ax=ax2, show=False,color='red')
ax3 = fig.add_subplot(133, projection='3d')
opt_coils_alm.plot(ax=ax3, show=False,label='ALM optimized coils')
opt_surface_alm_2.plot(ax=ax3, show=False)
plt.legend()
plt.tight_layout()
plt.savefig('coils_and_surfaces.pdf')

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(131, projection='3d')
init_coils.plot(ax=ax1, show=False,label='Initial coils')
init_surface_1.plot(ax=ax1, show=False)
ax2 = fig.add_subplot(132, projection='3d')
opt_surface_alm_1.plot(ax=ax2, show=False,color='red')
ax3 = fig.add_subplot(133, projection='3d')
opt_coils_alm.plot(ax=ax3, show=False,label='ALM optimized coils')
opt_surface_alm_1.plot(ax=ax3, show=False)
plt.legend()
plt.tight_layout()
plt.savefig('coils_and_surfaces_1.pdf')

iota_function(opt_field_alm, opt_surface_alm_2)
iota_function(opt_field_alm, opt_surface_alm_1)
C_Total_constraint.loss(params[1],params[0] / jnp.maximum(scale, 1e-12))




B_final= B_on_surface(opt_surface_alm_2, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(opt_surface_alm_2.phi2d,opt_surface_alm_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB.png', dpi=300)


B_final= B_on_surface(opt_surface_alm_1, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(opt_surface_alm_1.phi2d,opt_surface_alm_1.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB_1.png', dpi=300)



filename = f"alm_QS_mn2_ncoils6_gframe.pkl"
params, lag_state, grad, eta, omega, info = alm.load_checkpoint_for_reconstruction(
    filename,
    C_Total_constraint  # Your freshly created constraint object
)


filename = f"alm_QS_mn2_ncoils6_gframe.pkl"

alm.save_checkpoint_with_constraints(
    filename, params, lag_state, grad, eta, omega, info, 
    C_Total_constraint, 
    save_constraint_object=False  # Autoscales still saved!
)



# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing


phi_poincare=jnp.pi / 4
tmax = 100000000000
nfieldlines_per_core = 40
nfieldlines = nfieldlines_per_core * number_of_processors_to_use
#R0 = jnp.linspace(11.2, 14.9, nfieldlines)
R0 = jnp.linspace(11.2, 12.2, nfieldlines)
#R0 = jnp.linspace(11.7, 13.3, nfieldlines)
trace_tolerance = 1e-7
num_steps = 60000

Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T
time0 = time()

tracing = block_until_ready(Tracing(
    field=opt_field_alm,
    model='FieldLineAdaptative',
    initial_conditions=initial_xyz,
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance
))

print(f"ESSOS tracing took {time() - time0:.2f} seconds")

trajectories = tracing.trajectories
traj = trajectories[0]
R, phi, Z = traj[:, 0], traj[:, 1], traj[:, 2]

phi_u = jnp.unwrap(phi)
phi0_cross = jnp.where((phi_u[:-1] < 0) & (phi_u[1:] >= 0))[0]
phi90_cross = jnp.where((phi_u[:-1] < phi_poincare) & (phi_u[1:] >= phi_poincare))[0]

theta = jnp.linspace(0, 2 * jnp.pi, 200)

def compute_rz_on_phi(surface, theta, phi=0.0):
    if hasattr(surface, "rc"):
        angles = jnp.outer(theta, surface.xm) - phi * surface.xn
        R = jnp.sum(surface.rc * jnp.cos(angles), axis=1)
        Z = jnp.sum(surface.zs * jnp.sin(angles), axis=1)
        return R, Z
    phi_grid = surface.phi2d[:, 0]
    idx = jnp.argmin(jnp.abs(phi_grid - phi))
    gamma_slice = surface.gamma[idx, :, :]
    R = jnp.sqrt(gamma_slice[:, 0]**2 + gamma_slice[:, 1]**2)
    Z = gamma_slice[:, 2]
    return R, Z

# # Contours from optimized surface
R0_opt, Z0_opt = compute_rz_on_phi(opt_surface_alm_2, theta, phi=0.0)
R90_opt, Z90_opt = compute_rz_on_phi(opt_surface_alm_2, theta, phi=phi_poincare)
# # Contours from true VMEC surface
R0_true, Z0_true = compute_rz_on_phi(target_surface_2, theta, phi=0.0)
R90_true, Z90_true = compute_rz_on_phi(target_surface_2, theta, phi=phi_poincare)

# R0_3, Z0_3 = compute_rz_on_phi(init_ellipse_RZ, theta, phi=0.0)
# R90_3, Z90_3 = compute_rz_on_phi(init_ellipse_RZ, theta, phi=phi_poincare)

# R0_4, Z0_4 = compute_rz_on_phi(init_ellipse_gframe, theta, phi=0.0)
# R90_4, Z90_4 = compute_rz_on_phi(init_ellipse_gframe, theta, phi=phi_poincare)

fig, ax = plt.subplots(figsize=(6, 6))

tracing.poincare_plot(ax=ax, show=False, shifts=[0, phi_poincare])
ax.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 0$")
ax.plot(R90_opt, Z90_opt, color='black', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2$")

#ax.plot(R0_opt_2, Z0_opt_2, color='red', linewidth=1.5, label=r"Optimized @ $\phi = 0 s=0.98$")
#ax.plot(R90_opt_2, Z90_opt_2, color='red', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2, s=0.98$")
ax.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
ax.plot(R90_true, Z90_true, color='blue', linestyle='--', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")
# ax.plot(R0_3, Z0_3, color='green', linewidth=1.2, label=r"Initial Ellipse RZ @ $\phi = 0$")
# ax.plot(R90_3, Z90_3, color='green', linestyle='--', linewidth=1.2, label=r"Initial Ellipse RZ @ $\phi = \pi/2$")
# ax.plot(R0_4, Z0_4, color='orange', linewidth=1.2, label=r"Initial Ellipse GFrame @ $\phi = 0$")
# ax.plot(R90_4, Z90_4, color='orange', linestyle='--', linewidth=1.2, label=r"Initial Ellipse GFrame @ $\phi = \pi/2$")


ax.set_xlabel("R")
ax.set_ylabel("Z")
ax.set_title("Poincaré + Surfaces Comparison @ φ = 0 and π/2")
ax.legend()
ax.axis("equal")
plt.tight_layout()
plt.savefig('optimize_qfm_surface_poincare.png', dpi=300)



init_coils.to_json("init_coils_vmec_surface.json")
opt_coils_alm.to_json( "opt_coils_vmec_surface.json")

init_field.coils.to_vtk('coils_initial')
opt_coils_alm.to_vtk('coils_optimized')
#opt_surface_alm_1.to_vtk('surface_1_optimized')
opt_surface_alm_2.to_vtk('surface_2_optimized',field=opt_field_alm)


import pyvista as pv
coils_init_file='coils_initial.vtu'
coils_opt_file='coils_optimized.vtu'
#surf_1_file='surface_1_optimized.vts'
surf_2_file='surface_2_optimized.vts'

# Load the coil and field files
#surf1_mesh = pv.read(surf_1_file)
surf2_mesh = pv.read(surf_2_file)
coils_mesh = pv.read(coils_opt_file)
coils_init_mesh = pv.read(coils_init_file)
#Function to use with pyvista plotter to change rcamera rotation based on aximuthal angle and elevation angle
def camera_rotation(pl,azim_angle=0,elev_angle=0):
    cam = pl.camera
    # Load current camera state
    pos = jnp.array(cam.position)
    f = jnp.array(cam.focal_point)
    up = jnp.array(cam.up)
    # Camera vector (from focal point to camera)
    v = pos - f
    r = jnp.linalg.norm(v)
    # Convert to spherical coordinates
    x, y, z = v
    theta = jnp.arctan2(y, x)        # azimuth angle
    phi   = jnp.arccos(z / r)        # polar angle
    # Apply changes (in radians)
    theta += jnp.radians(azim_angle)
    phi   += jnp.radians(elev_angle)
    # Clamp phi to avoid flipping over the poles
    phi = jnp.clip(phi, 1e-3, jnp.pi - 1e-3)
    # Convert back to Cartesian, maintaining SAME radius (zoom)
    vx = r * jnp.sin(phi) * jnp.cos(theta)
    vy = r * jnp.sin(phi) * jnp.sin(theta)
    vz = r * jnp.cos(phi)
    # Update camera
    cam.position = f + jnp.array([vx, vy, vz])
    cam.up = up  # preserve the original up vector




# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf2_mesh,show_scalar_bar=False,color='red')
pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color='gold', line_width=5)
camera_rotation(pl,-100.,elev_angle=12)
pl.render()
#pl.show()
pl.screenshot('coils_paraviw_style.png')


B_final= B_on_surface(opt_surface_alm_2, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(opt_surface_alm_2.phi2d,opt_surface_alm_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB.png', dpi=300)

B_final= B_on_surface(init_surface_2, init_field)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(init_surface_2.phi2d,init_surface_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB_init.png', dpi=300)

B_final= B_on_surface(target_surface_2, target_field)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(target_surface_2.phi2d,target_surface_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('target_modB.png', dpi=300)
