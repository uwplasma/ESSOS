"""Verify that coil planes are perpendicular to the axis direction."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from jax import jacfwd
from essos.coils import CreateCoilsAroundAxis

def figure_eight_axis(phi):
    a = 10.0
    x = a * jnp.cos(phi)
    y = a * jnp.sin(phi) * jnp.cos(phi)
    z = 0.2 * jnp.sin(3 * phi)
    return jnp.stack([x, y, z], axis=-1)

def helical_axis(phi):
    R = 10.0
    n_periods = 2
    r_mod = R + 3.0 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 4.0 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

print("="*70)
print("COIL PERPENDICULARITY VERIFICATION")
print("="*70)

for name, axis_func in [("Figure-Eight", figure_eight_axis), ("Helical", helical_axis)]:
    print(f"\n{name} Axis:")
    print("-" * 40)
    
    # Create coils
    coils = CreateCoilsAroundAxis(
        n_coils=4,
        order=4,
        coil_radius=1.5,
        axis_function=axis_func,
        axis_shape='custom',
        nfp=1,
        stellsym=False
    )
    
    coil_gamma = np.array(coils.gamma)
    
    # Check each coil
    for coil_idx in range(min(2, len(coils))):
        # Position where this coil is centered
        phi_coil = (coil_idx + 0.5) * 2 * np.pi / 4
        
        # Compute axis tangent at this position
        tangent_func = jacfwd(axis_func)
        tangent = np.array(tangent_func(phi_coil))
        tangent = tangent / np.linalg.norm(tangent)
        
        # Get coil center and a point on the coil
        axis_center = np.array(axis_func(phi_coil))
        coil_pt = coil_gamma[coil_idx, 0, :]
        
        # Vector from axis to coil point (should be perpendicular to tangent)
        radial_vec = coil_pt - axis_center
        radial_vec = radial_vec / np.linalg.norm(radial_vec)
        
        # Dot product should be ~0 if perpendicular
        dot_product = np.dot(tangent, radial_vec)
        
        print(f"  Coil {coil_idx} (phi={phi_coil:.3f}):")
        print(f"    Tangent: [{tangent[0]:.3f}, {tangent[1]:.3f}, {tangent[2]:.3f}]")
        print(f"    Radial:  [{radial_vec[0]:.3f}, {radial_vec[1]:.3f}, {radial_vec[2]:.3f}]")
        print(f"    Dot product: {dot_product:.6f}")
        
        if abs(dot_product) < 0.01:
            print(f"    ✅ Coil plane is perpendicular to axis!")
        else:
            print(f"    ⚠️  Coil plane may not be perpendicular")
    
    # Check the coil plane normal for first coil
    coil_idx = 0
    phi_coil = (coil_idx + 0.5) * 2 * np.pi / 4
    
    # Get three points on the coil to compute the plane
    p1 = coil_gamma[coil_idx, 0, :]
    p2 = coil_gamma[coil_idx, len(coil_gamma[coil_idx])//4, :]
    p3 = coil_gamma[coil_idx, len(coil_gamma[coil_idx])//2, :]
    
    # Coil plane normal = (p2-p1) × (p3-p1)
    v1 = p2 - p1
    v2 = p3 - p1
    coil_normal = np.cross(v1, v2)
    coil_normal = coil_normal / np.linalg.norm(coil_normal)
    
    # Axis tangent
    tangent_func = jacfwd(axis_func)
    tangent = np.array(tangent_func(phi_coil))
    tangent = tangent / np.linalg.norm(tangent)
    
    # Dot product of coil normal with tangent should be ~±1 if coil is perpendicular
    alignment = abs(np.dot(coil_normal, tangent))
    
    print(f"\n  Coil plane alignment with axis tangent:")
    print(f"    |normal · tangent| = {alignment:.6f} (should be ~1.0)")
    if alignment > 0.95:
        print(f"    ✅ Coil plane is perpendicular to axis direction!")
    else:
        print(f"    ⚠️  Coil plane alignment may be off")

print("\n" + "="*70)
