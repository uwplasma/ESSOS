#!/usr/bin/env python
"""Quick iota comparison between RZ and GFrame surfaces."""

# Import from the main script environment
import jax.numpy as jnp
from essos.surfaces import B_on_surface

# Load the script to get all variables
exec(open('optimize_coils_surface_augmented_lagrangian_one_surface_QS_maybe_final_gframe.py').read())

print("\n" + "="*80)
print("IOTA COMPARISON TEST: Ellipse")
print("="*80)

# Compare ellipse surfaces
iota_local_rz, iota_avg_rz = iota_function(init_field, init_ellipse_RZ)
iota_local_gf, iota_avg_gf = iota_function(init_field, init_ellipse_gframe)

print(f"\n{'Property':<35} {'RZFourier':<25} {'GFrameFourier':<25} {'Rel Diff %':<15}")
print("-"*100)

rel_diff_ellipse = 100 * abs(iota_avg_rz - iota_avg_gf) / abs(iota_avg_rz)

print(f"{'ι_average':<35} {float(iota_avg_rz):<25.15f} {float(iota_avg_gf):<25.15f} {rel_diff_ellipse:<15.6e}")
print(f"{'ι_local mean':<35} {float(jnp.mean(iota_local_rz)):<25.15f} {float(jnp.mean(iota_local_gf)):<25.15f}")
print(f"{'ι_local std':<35} {float(jnp.std(iota_local_rz)):<25.15f} {float(jnp.std(iota_local_gf)):<25.15f}")
print(f"{'ι_local min':<35} {float(jnp.min(iota_local_rz)):<25.15f} {float(jnp.min(iota_local_gf)):<25.15f}")
print(f"{'ι_local max':<35} {float(jnp.max(iota_local_rz)):<25.15f} {float(jnp.max(iota_local_gf)):<25.15f}")

iota_local_diff = jnp.abs(iota_local_rz - iota_local_gf)
print(f"{'|ι_rz - ι_gf| mean':<35} {float(jnp.mean(iota_local_diff)):<25.12e}")
print(f"{'|ι_rz - ι_gf| max':<35} {float(jnp.max(iota_local_diff)):<25.12e}")

# Geometry comparison
gamma_rz, gamma_gf = init_ellipse_RZ.gamma, init_ellipse_gframe.gamma
gamma_diff = jnp.linalg.norm(gamma_rz - gamma_gf, axis=-1)
print(f"\n{'Geometry Differences':<35}")
print("-"*100)
print(f"{'|gamma_rz - gamma_gf| max':<35} {float(jnp.max(gamma_diff)):<25.12e}")

if rel_diff_ellipse > 0.1:
    print(f"\n⚠️  WARNING: Significant iota difference ({rel_diff_ellipse:.4f}% > 0.1%)")
else:
    print(f"\n✓ Iota values agree well ({rel_diff_ellipse:.6f}% < 0.1%)")

print("\n" + "="*80)
print("IOTA COMPARISON TEST: Torus")
print("="*80)

# Compare torus surfaces
iota_local_rz2, iota_avg_rz2 = iota_function(init_field, init_surface_2_RZ)
iota_local_gf2, iota_avg_gf2 = iota_function(init_field, init_surface_2)

print(f"\n{'Property':<35} {'RZFourier':<25} {'GFrameFourier':<25} {'Rel Diff %':<15}")
print("-"*100)

rel_diff_torus = 100 * abs(iota_avg_rz2 - iota_avg_gf2) / abs(iota_avg_rz2)

print(f"{'ι_average':<35} {float(iota_avg_rz2):<25.15f} {float(iota_avg_gf2):<25.15f} {rel_diff_torus:<15.6e}")
print(f"{'ι_local mean':<35} {float(jnp.mean(iota_local_rz2)):<25.15f} {float(jnp.mean(iota_local_gf2)):<25.15f}")

if rel_diff_torus > 0.1:
    print(f"\n⚠️  WARNING: Significant iota difference ({rel_diff_torus:.4f}% > 0.1%)")
else:
    print(f"\n✓ Iota values agree well ({rel_diff_torus:.6f}% < 0.1%)")

print("\n" + "="*80)
