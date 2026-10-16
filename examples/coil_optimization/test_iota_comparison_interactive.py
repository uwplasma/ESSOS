#!/usr/bin/env python
"""
Test script to compare iota calculations between RZ and GFrame surfaces.
Run this from the Python REPL in the coil_optimization directory.
"""

import jax.numpy as jnp
from jax import vmap
from essos.surfaces import B_on_surface

# Iota calculation function
def iota_function_test(field, surface):
    """Calculate rotational transform (iota)."""
    B = B_on_surface(surface, field)
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
    area_element = jnp.linalg.norm(n_vec_raw, axis=-1)
    
    # 2D metric
    g_tt = jnp.sum(e_theta**2, axis=-1)
    g_pp = jnp.sum(e_phi**2, axis=-1)
    g_tp = jnp.sum(e_theta * e_phi, axis=-1)
    det_g2d = g_tt * g_pp - g_tp**2
    
    # Contravariant components
    b_cov_theta = jnp.sum(B * e_theta, axis=-1)
    b_cov_phi = jnp.sum(B * e_phi, axis=-1)
    b_theta = (g_pp * b_cov_theta - g_tp * b_cov_phi) / (det_g2d + 1e-8)
    b_phi = (g_tt * b_cov_phi - g_tp * b_cov_theta) / (det_g2d + 1e-8)
    
    # Safe division for gradients
    b_phi_safe = jnp.sqrt(b_phi**2 + 1e-28)
    iota_local = b_theta / b_phi_safe
    
    b_phi_avg_safe = jnp.sqrt(jnp.average(b_phi * area_element)**2 + 1e-28)
    iota_average = jnp.average(b_theta * area_element) / b_phi_avg_safe
    
    return iota_local, iota_average


def compare_iota_values(surf_rz, surf_gf, field, name="Surface"):
    """Compare iota between RZ and GFrame surfaces."""
    print(f"\n{'='*80}")
    print(f"IOTA COMPARISON: {name}")
    print(f"{'='*80}\n")
    
    # Calculate iota
    iota_local_rz, iota_avg_rz = iota_function_test(field, surf_rz)
    iota_local_gf, iota_avg_gf = iota_function_test(field, surf_gf)
    
    print(f"{'Property':<35} {'RZFourier':<25} {'GFrameFourier':<25} {'Rel Diff %':<15}")
    print("-"*100)
    
    rel_diff = 100 * abs(iota_avg_rz - iota_avg_gf) / abs(iota_avg_rz)
    
    print(f"{'ι_average':<35} {float(iota_avg_rz):<25.15f} {float(iota_avg_gf):<25.15f} {rel_diff:<15.6e}")
    print(f"{'ι_local mean':<35} {float(jnp.mean(iota_local_rz)):<25.15f} {float(jnp.mean(iota_local_gf)):<25.15f}")
    print(f"{'ι_local std':<35} {float(jnp.std(iota_local_rz)):<25.15f} {float(jnp.std(iota_local_gf)):<25.15f}")
    print(f"{'ι_local min':<35} {float(jnp.min(iota_local_rz)):<25.15f} {float(jnp.min(iota_local_gf)):<25.15f}")
    print(f"{'ι_local max':<35} {float(jnp.max(iota_local_rz)):<25.15f} {float(jnp.max(iota_local_gf)):<25.15f}")
    
    iota_local_diff = jnp.abs(iota_local_rz - iota_local_gf)
    print(f"{'|ι_rz - ι_gf| mean':<35} {float(jnp.mean(iota_local_diff)):<25.12e}")
    print(f"{'|ι_rz - ι_gf| max':<35} {float(jnp.max(iota_local_diff)):<25.12e}")
    
    # Geometry comparison
    gamma_rz, gamma_gf = surf_rz.gamma, surf_gf.gamma
    gamma_diff = jnp.linalg.norm(gamma_rz - gamma_gf, axis=-1)
    print(f"\n{'|gamma_rz - gamma_gf| max':<35} {float(jnp.max(gamma_diff)):<25.12e}")
    
    # Metric comparison
    gd_theta_rz, gd_theta_gf = surf_rz.gammadash_theta, surf_gf.gammadash_theta
    gd_phi_rz, gd_phi_gf = surf_rz.gammadash_phi, surf_gf.gammadash_phi
    
    g_tt_rz = jnp.sum(gd_theta_rz**2, axis=-1)
    g_pp_rz = jnp.sum(gd_phi_rz**2, axis=-1)
    g_tp_rz = jnp.sum(gd_theta_rz * gd_phi_rz, axis=-1)
    
    g_tt_gf = jnp.sum(gd_theta_gf**2, axis=-1)
    g_pp_gf = jnp.sum(gd_phi_gf**2, axis=-1)
    g_tp_gf = jnp.sum(gd_theta_gf * gd_phi_gf, axis=-1)
    
    det_g_rz = g_tt_rz * g_pp_rz - g_tp_rz**2
    det_g_gf = g_tt_gf * g_pp_gf - g_tp_gf**2
    
    print(f"{'|det(g)_rz - det(g)_gf| max':<35} {float(jnp.max(jnp.abs(det_g_rz-det_g_gf))):<25.12e}")
    
    if rel_diff > 0.1:
        print(f"\n⚠️  WARNING: Significant iota difference ({rel_diff:.4f}% > 0.1%)")
    else:
        print(f"\n✓ Iota values agree well ({rel_diff:.6f}% < 0.1%)")
    
    return {
        'iota_avg_rz': float(iota_avg_rz),
        'iota_avg_gf': float(iota_avg_gf),
        'rel_diff_pct': rel_diff
    }


# To use in interactive Python session:
# results_ellipse = compare_iota_values(init_ellipse_RZ, init_ellipse_gframe, init_field, "Ellipse")
# results_torus = compare_iota_values(init_surface_2_RZ, init_surface_2, init_field, "Torus")

print("Iota comparison functions loaded. Use:")
print("  compare_iota_values(init_ellipse_RZ, init_ellipse_gframe, init_field, 'Ellipse')")
print("  compare_iota_values(init_surface_2_RZ, init_surface_2, init_field, 'Torus')")
