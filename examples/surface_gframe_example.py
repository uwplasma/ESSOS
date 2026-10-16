"""
Example demonstrating the use of SurfaceGFrameFourier (G-frame) surfaces.

This example shows how to create and manipulate surfaces using the generalized
Frenet frame (G-frame) approach inspired by GVEC, in contrast to the standard
RZ Fourier representation used in VMEC.

Key differences from RZFourier:
- G-frame uses a central axis curve plus perpendicular directions (Frenet frame)
- Expands X1, X2 coefficients instead of R, Z
- More flexible for non-convex or complex-shaped surfaces
- Better for equilibria with magnetic islands or other exotic geometries
"""

import jax
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier, SurfaceRZFourier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# Example 1: Create a simple circular torus using G-frame
# ==============================================================================

print("="*70)
print("Example 1: Simple Circular Torus using G-frame")
print("="*70)

# Define mode numbers
nfp = 2  # 2 field periods
mpol = 1  # maximum poloidal mode
ntor = 1  # maximum toroidal mode

# Number of Fourier coefficients
n_coeffs = (mpol + 1) * (2 * ntor + 1) - ntor

# Initialize X1 and X2 coefficients
# For a simple example, we'll create a small deviation from the axis
x1_coeffs = jnp.array([0.2, 0.05, 0.02])  # m=0,1; n=0,±1
x2_coeffs = jnp.array([0.0, 0.03, -0.02])

# Create the surface
major_radius = 1.0
surf_gframe = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=30,
    nphi=30,
    close=True,
    range_torus='full torus',
    axis_major_radius=major_radius,
    axis_shape='circle',
    scaling_type=2,
    scaling_factor=0.0,
    scale_fixed=1.0
)

print(f"Created G-frame surface with:")
print(f"  - Number of field periods: {surf_gframe.nfp}")
print(f"  - Poloidal modes: 0 to {surf_gframe.mpol}")
print(f"  - Toroidal modes: -{surf_gframe.ntor} to {surf_gframe.ntor}")
print(f"  - Grid points: {surf_gframe.ntheta} (poloidal) x {surf_gframe.nphi} (toroidal)")
print(f"  - Central axis major radius: {major_radius}")
print(f"  - Number of DOFs: {len(surf_gframe.dofs)}")

# Compute surface properties
print("\nComputing surface properties...")
gamma = surf_gframe.gamma
print(f"  - Surface position shape: {gamma.shape}")
print(f"  - Surface volume: {surf_gframe.volume:.4f}")
print(f"  - Surface area: {surf_gframe.area:.4f}")

# ==============================================================================
# Example 2: Modify coefficients
# ==============================================================================

print("\n" + "="*70)
print("Example 2: Modifying G-frame Coefficients")
print("="*70)

# Create modified coefficients
x1_coeffs_modified = jnp.array([0.3, 0.08, 0.01])
x2_coeffs_modified = jnp.array([0.05, 0.04, -0.01])

surf_gframe.x1_coeffs = x1_coeffs_modified
surf_gframe.x2_coeffs = x2_coeffs_modified

print(f"Modified X1 coefficients: {surf_gframe.x1_coeffs}")
print(f"Modified X2 coefficients: {surf_gframe.x2_coeffs}")
print(f"New surface volume: {surf_gframe.volume:.4f}")
print(f"New surface area: {surf_gframe.area:.4f}")

# ==============================================================================
# Example 3: Compare G-frame and RZ Fourier approaches
# ==============================================================================

print("\n" + "="*70)
print("Example 3: Conceptual Comparison of G-frame vs RZ Fourier")
print("="*70)

print("""
RZ Fourier Approach (SurfaceRZFourier):
  - Represents surface as: R(θ, φ) and Z(θ, φ)
  - Fourier expands: R = Σ rc_i cos(m_i θ - n_i φ)
                    Z = Σ zs_i sin(m_i θ - n_i φ)
  - Cartesian: X = R(θ,φ) cos(φ), Y = R(θ,φ) sin(φ), Z = Z(θ,φ)
  - Advantages: Standard in VMEC, efficient for nearly circular surfaces
  - Disadvantages: Difficult for non-convex or complex-shaped surfaces

G-frame Approach (SurfaceGFrameFourier):
  - Represents surface as: r = r_axis(φ) + X1(θ,φ) n1(φ) + X2(θ,φ) n2(φ)
  - r_axis: central axis curve (typically circular torus axis)
  - n1, n2: orthonormal Frenet frame vectors at each point on axis
  - X1, X2: Fourier expanded in poloidal angle θ
  - Advantages: More versatile for complex shapes, used by GVEC
  - Disadvantages: More complex derivative computations
  
Mathematical relationship:
  The G-frame approach expands deviations from the axis in the local frame,
  whereas RZ Fourier expands the global cylindrical coordinates directly.
""")

# ==============================================================================
# Example 4: Using G-frame surface with field calculations
# ==============================================================================

print("\n" + "="*70)
print("Example 4: G-frame Surface Properties")
print("="*70)

# Get surface properties
normal_vector = surf_gframe.normal
unit_normal = surf_gframe.unitnormal
area_elem = surf_gframe.area_element

print(f"Normal vector shape: {normal_vector.shape}")
print(f"Unit normal (normalized) shape: {unit_normal.shape}")
print(f"Area element shape: {area_elem.shape}")
print(f"Average area element: {jnp.mean(area_elem):.6f}")
print(f"Min/Max area element: {jnp.min(area_elem):.6f} / {jnp.max(area_elem):.6f}")

# ==============================================================================
# Example 5: Access DOFs
# ==============================================================================

print("\n" + "="*70)
print("Example 5: Accessing and Modifying DOFs")
print("="*70)

dofs = surf_gframe.dofs
print(f"Current DOFs shape: {dofs.shape}")
print(f"Current DOFs: {dofs}")

# Modify DOFs
new_dofs = dofs * 1.1  # Scale by 10%
surf_gframe.dofs = new_dofs
print(f"\nAfter scaling DOFs by 1.1:")
print(f"New X1 coefficients: {surf_gframe.x1_coeffs}")
print(f"New X2 coefficients: {surf_gframe.x2_coeffs}")

# ==============================================================================
# Example 6: Visualize if matplotlib is available
# ==============================================================================

print("\n" + "="*70)
print("Example 6: Surface Visualization (if plotting is available)")
print("="*70)

try:
    fig = plt.figure(figsize=(12, 5))
    
    # Plot G-frame surface
    ax1 = fig.add_subplot(121, projection='3d')
    gamma = surf_gframe.gamma
    ax1.plot_surface(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2],
                     alpha=0.8, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('G-frame Surface')
    
    # Plot the central axis
    ax2 = fig.add_subplot(122, projection='3d')
    phi_axis = jnp.linspace(0, 2*jnp.pi, 100)
    axis_curve = jnp.array([major_radius * jnp.cos(phi_axis),
                            major_radius * jnp.sin(phi_axis),
                            jnp.zeros_like(phi_axis)]).T
    ax2.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'r-', linewidth=2, label='Central Axis')
    ax2.plot_surface(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2],
                     alpha=0.6, cmap='plasma')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('G-frame Surface with Central Axis')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/home/exouser/ESSOS/examples/gframe_surface_example.png', dpi=150)
    print("Plot saved to: examples/gframe_surface_example.png")
    plt.close()
    
except Exception as e:
    print(f"Could not generate plots: {e}")

# ==============================================================================
# Example 8: Helical Axis with Pitch and Twist Rate
# ==============================================================================

print("\n" + "="*70)
print("Example 8: Helical Axis with Pitch and Twist Rate")
print("="*70)

print("""
The G-frame allows for flexible axis shapes beyond simple circles/ellipses.
This example demonstrates a helical axis with vertical pitch and frame twist.
""")

# Create a helical axis surface
nfp = 2
mpol = 1
ntor = 1

x1_coeffs = jnp.array([0.15, 0.04, 0.01])
x2_coeffs = jnp.array([0.1, 0.02, -0.01])

print("\n1. Helical axis (no frame twist):")
surf_helical_1 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=25,
    nphi=25,
    close=True,
    axis_shape='helical',
    axis_major_radius=1.0,
    axis_pitch=0.2,      # Rise 0.2 units per 2π toroidal turn
    axis_twist_rate=0.0  # No frame rotation
)

gamma1 = surf_helical_1.gamma
z_extent_1 = float(jnp.max(gamma1[..., 2])) - float(jnp.min(gamma1[..., 2]))
print(f"   - Axis pitch: 0.2")
print(f"   - Frame twist rate: 0.0 rad/turn")
print(f"   - Surface z-extent: {z_extent_1:.4f}")
print(f"   - Volume: {surf_helical_1.volume:.4f}")

print("\n2. Helical axis with frame twist:")
surf_helical_2 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=25,
    nphi=25,
    close=True,
    axis_shape='helical',
    axis_major_radius=1.0,
    axis_pitch=0.3,        # Rise 0.3 units per 2π toroidal turn
    axis_twist_rate=0.5    # Frame rotates 0.5 rad per toroidal turn
)

gamma2 = surf_helical_2.gamma
z_extent_2 = float(jnp.max(gamma2[..., 2])) - float(jnp.min(gamma2[..., 2]))
print(f"   - Axis pitch: 0.3")
print(f"   - Frame twist rate: 0.5 rad/turn")
print(f"   - Surface z-extent: {z_extent_2:.4f}")
print(f"   - Volume: {surf_helical_2.volume:.4f}")

print("\n3. Steeper helical axis:")
surf_helical_3 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs * 0.8,  # Slightly smaller cross-section
    x2_coeffs=x2_coeffs * 0.8,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=25,
    nphi=25,
    close=True,
    axis_shape='helical',
    axis_major_radius=1.0,
    axis_pitch=0.5,        # Steep rise: 0.5 units per 2π toroidal turn
    axis_twist_rate=1.0    # Strong frame twist: 1.0 rad per toroidal turn
)

gamma3 = surf_helical_3.gamma
z_extent_3 = float(jnp.max(gamma3[..., 2])) - float(jnp.min(gamma3[..., 2]))
print(f"   - Axis pitch: 0.5")
print(f"   - Frame twist rate: 1.0 rad/turn")
print(f"   - Surface z-extent: {z_extent_3:.4f}")
print(f"   - Volume: {surf_helical_3.volume:.4f}")

print("\nKey parameters for helical axis:")
print("  - axis_pitch: Controls vertical rise per 2π toroidal rotation")
print("    * 0.0 = standard torus (circular axis)")
print("    * 0.1-0.3 = gradual helix")
print("    * 0.5+ = steep helix")
print("  - axis_twist_rate: Controls Frenet frame rotation around axis")
print("    * 0.0 = no frame rotation (standard)")
print("    * 0.5-1.0 = noticeable twist")
print("    * >1.0 = strong twist over one period")

print("\nApplications of helical axes:")
print("  - Stellarators with toroidal field (TF coil rotation)")
print("  - Twisted magnetic surfaces in non-planar geometries")
print("  - Modeling coil imperfections and misalignments")
print("  - Creating exotic equilibrium shapes for stability studies")

# ==============================================================================
# Example 9: Custom Axis Functions
# ==============================================================================

print("\n" + "="*70)
print("Example 9: Custom Axis Functions")
print("="*70)

print("""
Custom axis functions allow arbitrary axis shapes by providing a JAX-compatible
function r(φ) -> [x, y, z]. This enables modeling of non-standard geometries.
""")

# Define a custom figure-eight axis
def figure_eight_axis(phi):
    """Custom axis: figure-eight shape."""
    R = 1.0
    r = 0.3
    x = (R + r * jnp.cos(2 * phi)) * jnp.cos(phi)
    y = (R + r * jnp.cos(2 * phi)) * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

print("\n1. Figure-Eight Axis:")
surf_fig8 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=25,
    nphi=25,
    close=True,
    axis_shape='custom',
    axis_function=figure_eight_axis
)

gamma_fig8 = surf_fig8.gamma
print(f"   - Custom function: figure_eight_axis(phi)")
print(f"   - Surface volume: {surf_fig8.volume:.4f}")
print(f"   - Surface area: {surf_fig8.area:.4f}")

# Define a second custom axis: rose curve
def rose_curve_axis(phi):
    """Custom axis: 3-petal rose curve."""
    a = 1.0
    r = a * jnp.abs(jnp.cos(3 * phi))
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

print("\n2. Rose Curve Axis (3 petals):")
surf_rose = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs * 0.8,
    x2_coeffs=x2_coeffs * 0.8,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=25,
    nphi=25,
    close=True,
    axis_shape='custom',
    axis_function=rose_curve_axis
)

print(f"   - Custom function: rose_curve_axis(phi)")
print(f"   - Surface volume: {surf_rose.volume:.4f}")

print("\n3. Dynamic custom axis assignment:")
# Create surface and change its axis function
def circle_func(phi):
    R = 1.0
    return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), jnp.zeros_like(phi)], axis=-1)

surf_dyn = SurfaceGFrameFourier(
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

vol_circle = surf_dyn.volume
surf_dyn.axis_function = figure_eight_axis
vol_fig8 = surf_dyn.volume

print(f"   - Initial (circle): volume = {vol_circle:.4f}")
print(f"   - After change to figure-eight: volume = {vol_fig8:.4f}")

print("\nKey features of custom axis functions:")
print("  - Must be JAX-compatible (use jnp, not np)")
print("  - Signature: func(phi) -> array([x, y, z])")
print("  - Can be changed dynamically via axis_function property")
print("  - Enables optimization through JAX grad/vmap")
print("  - Supports arbitrary geometric shapes")

print("\nSee test_custom_axis.py for 8 detailed custom axis examples!")

# ==============================================================================
# Example 10: Key References
# ==============================================================================

print("\n" + "="*70)
print("References for G-frame Surface Representation")
print("="*70)

print("""
The G-frame (generalized Frenet frame) approach for surface representation
is implemented in GVEC (Galerkin Variational Equilibrium Code).

Key Reference:
  Hindenlang, F., Plunk, G. G., & Maj, O. (2025).
  "Computing MHD equilibria of stellarators with a flexible coordinate frame"
  Plasma Physics and Controlled Fusion, 67(4), 045002
  https://doi.org/10.1088/1361-6587/adba11

GVEC project:
  https://github.com/gvec-group/gvec
  https://gvec.readthedocs.io/latest/

The SurfaceGFrameFourier class in ESSOS provides a JAX-based implementation
of this approach, enabling automatic differentiation for optimization.

Helical axis coordinates are useful for:
  - Representing twisted magnetic geometries
  - Modeling stellarator configurations with toroidal field
  - Optimization of non-planar device designs
""")

print("\n" + "="*70)
print("Examples completed successfully!")
print("="*70)
print("\nYou have now seen 10 examples of SurfaceGFrameFourier usage:")
print("  1. Simple circular torus with G-frame")
print("  2. Modifying coefficients")
print("  3. Comparing G-frame and RZ Fourier")
print("  4. Computing surface properties (volume, area)")
print("  5. Accessing DOFs and mode information")
print("  6. Plotting surfaces and axis curves")
print("  7. DOF manipulation and vector operations")
print("  8. Helical axes with pitch and twist rate")
print("  9. Custom axis functions (arbitrary shapes)")
print("  10. References and further reading")
print("\nNext steps:")
print("  - Experiment with different mode numbers (mpol, ntor)")
print("  - Try different axis shapes: 'circle', 'ellipse', 'helical', 'custom'")
print("  - Create your own custom axis functions for specialized geometries")
print("  - Use in optimization: gradients work through JAX via vmap/grad")
print("  - See test_custom_axis.py for 8 detailed custom axis examples")
print("="*70)
