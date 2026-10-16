#!/usr/bin/env python3
"""
Examples of custom axis functions for SurfaceGFrameFourier.

This demonstrates how to create arbitrary axis curves by providing
JAX-compatible functions to the axis_function parameter.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

print("=" * 70)
print("CUSTOM AXIS FUNCTIONS FOR SurfaceGFrameFourier")
print("=" * 70)

# Create base coefficients for all examples
mpol, ntor, nfp = 1, 1, 2
x1_coeffs = jnp.array([0.1, 0.05, -0.02], dtype=jnp.float64)
x2_coeffs = jnp.array([0.08, 0.03, -0.01], dtype=jnp.float64)

# ==============================================================================
# Example 1: Figure-Eight Axis
# ==============================================================================

print("\n" + "=" * 70)
print("Example 1: Figure-Eight Axis")
print("=" * 70)

def figure_eight_axis(phi):
    """
    Create a figure-eight shaped axis.
    The axis traces a 2-petaled rose curve in the XY plane.
    """
    R = 1.0  # Major radius
    r = 0.3  # Modulation amplitude
    x = (R + r * jnp.cos(2 * phi)) * jnp.cos(phi)
    y = (R + r * jnp.cos(2 * phi)) * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

print("\nCreating surface with figure-eight axis...")
surf_fig8 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=20,
    nphi=20,
    axis_shape='custom',
    axis_function=figure_eight_axis
)

gamma_fig8 = surf_fig8.gamma
print(f"✓ Figure-eight surface created")
print(f"  Gamma shape: {gamma_fig8.shape}")
print(f"  X-range: [{jnp.min(gamma_fig8[..., 0]):.4f}, {jnp.max(gamma_fig8[..., 0]):.4f}]")
print(f"  Y-range: [{jnp.min(gamma_fig8[..., 1]):.4f}, {jnp.max(gamma_fig8[..., 1]):.4f}]")

# ==============================================================================
# Example 2: Rose Curve Axis (3 petals)
# ==============================================================================

print("\n" + "=" * 70)
print("Example 2: Rose Curve Axis (3 petals)")
print("=" * 70)

def rose_curve_axis_3petals(phi):
    """
    Create a 3-petaled rose curve axis.
    r(φ) = a·cos(3φ) in polar coordinates.
    """
    a = 1.0  # Petal amplitude
    r_radius = a * jnp.abs(jnp.cos(3 * phi))
    x = r_radius * jnp.cos(phi)
    y = r_radius * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

print("\nCreating surface with 3-petal rose axis...")
surf_rose3 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs * 0.8,
    x2_coeffs=x2_coeffs * 0.8,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=20,
    nphi=25,
    axis_shape='custom',
    axis_function=rose_curve_axis_3petals
)

gamma_rose3 = surf_rose3.gamma
print(f"✓ 3-petal rose surface created")
print(f"  Volume: {surf_rose3.volume:.4f}")
print(f"  Area: {surf_rose3.area:.4f}")

# ==============================================================================
# Example 3: Lemniscate (Figure-Eight in 2D)
# ==============================================================================

print("\n" + "=" * 70)
print("Example 3: Lemniscate Axis")
print("=" * 70)

def lemniscate_axis(phi):
    """
    Create a lemniscate (figure-8 curve) as the axis.
    Parametric form: (cos²(t)·cos(t), cos²(t)·sin(t), 0) with scaling.
    """
    a = 0.8  # Scale factor
    cos2_phi = jnp.cos(phi) ** 2
    x = a * cos2_phi * jnp.cos(phi)
    y = a * cos2_phi * jnp.sin(phi)
    z = 0.1 * jnp.sin(2 * phi)  # Small vertical component
    return jnp.stack([x, y, z], axis=-1)

print("\nCreating surface with lemniscate axis...")
surf_lemn = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=20,
    nphi=20,
    axis_shape='custom',
    axis_function=lemniscate_axis
)

gamma_lemn = surf_lemn.gamma
print(f"✓ Lemniscate surface created")
print(f"  Z-extent: [{jnp.min(gamma_lemn[..., 2]):.4f}, {jnp.max(gamma_lemn[..., 2]):.4f}]")

# ==============================================================================
# Example 4: Spiral Curve (Archimedean Spiral)
# ==============================================================================

print("\n" + "=" * 70)
print("Example 4: Archimedean Spiral Axis")
print("=" * 70)

def spiral_axis(phi):
    """
    Create an Archimedean spiral as the axis.
    r(t) = a·t (in polar coords), then convert to Cartesian.
    """
    a = 0.15  # Spiral tightness
    radius = a * phi / jnp.pi  # Increases with phi
    x = radius * jnp.cos(phi)
    y = radius * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

print("\nCreating surface with spiral axis...")
surf_spiral = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs * 0.7,
    x2_coeffs=x2_coeffs * 0.7,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=15,
    nphi=20,
    axis_shape='custom',
    axis_function=spiral_axis
)

gamma_spiral = surf_spiral.gamma
print(f"✓ Spiral surface created")
print(f"  Radial extent: {jnp.linalg.norm(gamma_spiral[0, 0, :2]):.4f}")

# ==============================================================================
# Example 5: Parametric Curve (Star-shaped)
# ==============================================================================

print("\n" + "=" * 70)
print("Example 5: Star-Shaped Axis")
print("=" * 70)

def star_axis(phi):
    """
    Create a star-shaped axis with 5 points.
    """
    n_points = 5  # Number of star points
    r_min = 0.5
    r_max = 1.0
    # Smoothly vary between r_min and r_max
    r = r_min + (r_max - r_min) * (1 + jnp.cos(n_points * phi)) / 2
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

print("\nCreating surface with star-shaped axis...")
surf_star = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=20,
    nphi=20,
    axis_shape='custom',
    axis_function=star_axis
)

gamma_star = surf_star.gamma
print(f"✓ Star surface created")
print(f"  Volume: {surf_star.volume:.4f}")

# ==============================================================================
# Example 6: 3D Helix with Custom Function
# ==============================================================================

print("\n" + "=" * 70)
print("Example 6: 3D Helix (Custom Function Alternative)")
print("=" * 70)

def helix_custom(phi):
    """
    Create a helix as a custom axis function.
    This is more flexible than axis_shape='helical' since it allows
    customization of pitch and radius via closure variables.
    """
    R = 1.0   # Major radius
    pitch = 0.4  # Vertical rise per 2π
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    z = pitch * phi / (2 * jnp.pi)
    return jnp.stack([x, y, z], axis=-1)

print("\nCreating surface with 3D helix axis (custom function)...")
surf_helix = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=20,
    nphi=20,
    axis_shape='custom',
    axis_function=helix_custom
)

gamma_helix = surf_helix.gamma
print(f"✓ Helix surface created")
print(f"  Z-extent: [{jnp.min(gamma_helix[..., 2]):.4f}, {jnp.max(gamma_helix[..., 2]):.4f}]")

# ==============================================================================
# Example 7: Changing Custom Functions Dynamically
# ==============================================================================

print("\n" + "=" * 70)
print("Example 7: Dynamic Custom Function Assignment")
print("=" * 70)

# Create surface with initial custom function
def circle_func(phi):
    R = 1.0
    return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), jnp.zeros_like(phi)], axis=-1)

surf_dynamic = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=15,
    nphi=15,
    axis_shape='custom',
    axis_function=circle_func
)

print(f"\nInitial volume (circle axis): {surf_dynamic.volume:.4f}")

# Change to figure-eight
surf_dynamic.axis_function = figure_eight_axis
print(f"After changing to figure-eight: {surf_dynamic.volume:.4f}")

# Change to rose curve
surf_dynamic.axis_function = rose_curve_axis_3petals
print(f"After changing to 3-petal rose: {surf_dynamic.volume:.4f}")

# ==============================================================================
# Example 8: Gradient Computation Through Custom Axis
# ==============================================================================

print("\n" + "=" * 70)
print("Example 8: Gradient Computation with Custom Axis")
print("=" * 70)

def volume_with_custom_axis(coeffs, axis_fn):
    """Compute volume for given coefficients and custom axis function."""
    x1 = coeffs[:3]
    x2 = coeffs[3:]
    surf = SurfaceGFrameFourier(
        x1_coeffs=x1,
        x2_coeffs=x2,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ntheta=10,
        nphi=10,
        axis_shape='custom',
        axis_function=axis_fn
    )
    return surf.volume

coeffs = jnp.hstack([x1_coeffs, x2_coeffs])

# Compute gradients with respect to coefficients (with figure-eight axis)
grad_fn = jax.grad(volume_with_custom_axis, argnums=0)
grads = grad_fn(coeffs, figure_eight_axis)

print(f"✓ Gradients computed successfully!")
print(f"  Volume: {volume_with_custom_axis(coeffs, figure_eight_axis):.6f}")
print(f"  Gradient magnitude: {jnp.linalg.norm(grads):.6f}")
print(f"  Gradient shape: {grads.shape}")

print("\n" + "=" * 70)
print("ALL CUSTOM AXIS EXAMPLES COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Key takeaways for custom axis functions:

1. Functions must be JAX-compatible (use jnp, not np)
2. Signature: func(phi) -> [x, y, z] where phi is an array or scalar
3. Returns 3D Cartesian coordinates as jnp array
4. Can be changed dynamically via axis_function property
5. Enables optimization through arbitrary axis shapes
6. Works with JAX grad, vmap, jit transformations

Common patterns:
- Closed curves: Use cos/sin for periodic shapes
- Variable radius: Modulate with cos/sin of multiples of phi
- Vertical component: Add z = f(phi) for non-planar axes
- Closures: Define functions with captured parameters for flexibility
""")
