"""
Examples of creating coils around custom axes using CreateCoilsAroundAxis.

This demonstrates how to generalize coil generation beyond simple circular axes
to support elliptical, helical, and completely custom axis shapes—just like
the SurfaceGFrameFourier surface class.

The function CreateCoilsAroundAxis:
- Takes coil positions equally spaced around any axis type
- Creates circular coils perpendicular to the axis (in Frenet frame)
- Fits Fourier series to the discrete coil curves
- Returns a Curves object with full JAX compatibility
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import from ESSOS
import sys
sys.path.insert(0, '/home/exouser/ESSOS')
from essos.coils import CreateCoilsAroundAxis, CreateEquallySpacedCurves


# ============================================================================
# Example 1: Coils around circular axis (baseline - matches CreateEquallySpacedCurves)
# ============================================================================
def example_1_circular_axis():
    """
    Create coils around a circular axis.
    This should produce equivalent results to CreateEquallySpacedCurves(R=1.0, r=0.1).
    """
    print("\n" + "="*70)
    print("Example 1: Coils around CIRCULAR axis")
    print("="*70)
    
    coils = CreateCoilsAroundAxis(
        n_coils=4,
        order=4,
        coil_radius=0.1,
        n_samples=150,
        axis_major_radius=1.0,
        axis_shape='circle',
        nfp=1,
        stellsym=False
    )
    
    print(f"Created {len(coils)} coils")
    print(f"Coil order: {coils.order}")
    print(f"Discretization: {coils.n_segments} points")
    print(f"Coil shape: {coils.gamma.shape}")  # (n_curves, n_segments, 3)
    
    # Compute coil length
    length = coils.length
    print(f"Coil lengths (mm): {length * 1000}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    coils.plot(ax=ax, show=False, color='blue', linewidth=2)
    ax.set_title('Circular Axis - Coils')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Save
    plt.savefig('/home/exouser/ESSOS/examples/coils_example_1_circular.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: coils_example_1_circular.png")
    plt.close()


# ============================================================================
# Example 2: Coils around elliptical axis
# ============================================================================
def example_2_elliptical_axis():
    """
    Create coils around an elliptical axis.
    The axis is elliptical in the xy-plane.
    """
    print("\n" + "="*70)
    print("Example 2: Coils around ELLIPTICAL axis")
    print("="*70)
    
    coils = CreateCoilsAroundAxis(
        n_coils=4,
        order=4,
        coil_radius=0.1,
        n_samples=150,
        axis_major_radius=1.0,
        axis_shape='ellipse',
        nfp=1,
        stellsym=False
    )
    
    print(f"Created {len(coils)} coils around elliptical axis")
    print(f"Coil shape: {coils.gamma.shape}")
    
    length = coils.length
    print(f"Coil lengths (mm): {length * 1000}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    coils.plot(ax=ax, show=False, color='green', linewidth=2)
    ax.set_title('Elliptical Axis - Coils')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig('/home/exouser/ESSOS/examples/coils_example_2_elliptical.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: coils_example_2_elliptical.png")
    plt.close()


# ============================================================================
# Example 3: Coils around helical axis
# ============================================================================
def example_3_helical_axis():
    """
    Create coils around a helical axis.
    The axis spirals upward with pitch = rise per 2π toroidal turn.
    """
    print("\n" + "="*70)
    print("Example 3: Coils around HELICAL axis")
    print("="*70)
    
    coils = CreateCoilsAroundAxis(
        n_coils=4,
        order=4,
        coil_radius=0.1,
        n_samples=150,
        axis_major_radius=1.0,
        axis_shape='helical',
        axis_pitch=0.3,  # Rise 0.3 units per 2π
        axis_twist_rate=0.0,  # No frame twist (yet)
        nfp=1,
        stellsym=False
    )
    
    print(f"Created {len(coils)} coils around helical axis (pitch=0.3)")
    print(f"Coil shape: {coils.gamma.shape}")
    
    length = coils.length
    print(f"Coil lengths (mm): {length * 1000}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    coils.plot(ax=ax, show=False, color='red', linewidth=2)
    ax.set_title('Helical Axis (pitch=0.3) - Coils')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig('/home/exouser/ESSOS/examples/coils_example_3_helical.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: coils_example_3_helical.png")
    plt.close()


# ============================================================================
# Example 4: Coils around helical axis with twist rate
# ============================================================================
def example_4_helical_twisted():
    """
    Create coils around a helical axis with frame twist.
    The Frenet frame rotates as you go around the axis.
    """
    print("\n" + "="*70)
    print("Example 4: Coils around HELICAL axis with TWIST RATE")
    print("="*70)
    
    coils = CreateCoilsAroundAxis(
        n_coils=4,
        order=5,
        coil_radius=0.1,
        n_samples=150,
        axis_major_radius=1.0,
        axis_shape='helical',
        axis_pitch=0.3,
        axis_twist_rate=0.5,  # Frame rotates 0.5 rad per 2π
        nfp=1,
        stellsym=False
    )
    
    print(f"Created {len(coils)} coils around helical axis (pitch=0.3, twist_rate=0.5)")
    print(f"Coil shape: {coils.gamma.shape}")
    
    length = coils.length
    print(f"Coil lengths (mm): {length * 1000}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    coils.plot(ax=ax, show=False, color='purple', linewidth=2)
    ax.set_title('Helical Axis (twist_rate=0.5) - Coils')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig('/home/exouser/ESSOS/examples/coils_example_4_helical_twisted.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: coils_example_4_helical_twisted.png")
    plt.close()


# ============================================================================
# Example 5: Coils around figure-eight axis (custom function)
# ============================================================================
def example_5_figure_eight():
    """
    Create coils around a figure-eight (lemniscate) axis.
    Custom function: r(φ) = R(√2·cos(φ)) / (1+sin²(φ)) * [cos(φ), sin(φ), 0]
    """
    print("\n" + "="*70)
    print("Example 5: Coils around FIGURE-EIGHT axis (custom function)")
    print("="*70)
    
    # Define the figure-eight axis
    def figure_eight_axis(phi):
        """Lemniscate parametrization."""
        R = 1.0
        denom = 1.0 + jnp.sin(phi)**2
        r = R * jnp.sqrt(2.0) * jnp.cos(phi) / denom
        x = r * jnp.cos(phi)
        y = r * jnp.sin(phi)
        z = jnp.zeros_like(phi)
        return jnp.array([x, y, z])
    
    coils = CreateCoilsAroundAxis(
        n_coils=6,
        order=4,
        coil_radius=0.08,
        n_samples=150,
        axis_function=figure_eight_axis,
        axis_shape='custom',
        nfp=1,
        stellsym=False
    )
    
    print(f"Created {len(coils)} coils around figure-eight axis")
    print(f"Coil shape: {coils.gamma.shape}")
    
    length = coils.length
    print(f"Coil lengths (mm): {length * 1000}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    coils.plot(ax=ax, show=False, color='orange', linewidth=2)
    ax.set_title('Figure-Eight Axis - Coils')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig('/home/exouser/ESSOS/examples/coils_example_5_figure_eight.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: coils_example_5_figure_eight.png")
    plt.close()


# ============================================================================
# Example 6: Coils around rose curve axis (custom function)
# ============================================================================
def example_6_rose_curve():
    """
    Create coils around a rose curve (k=3 petals).
    Rose curve: r(θ) = cos(k·θ) for k=3
    """
    print("\n" + "="*70)
    print("Example 6: Coils around ROSE CURVE axis (custom function)")
    print("="*70)
    
    def rose_curve_axis(phi):
        """Rose curve with 3 petals: r(φ) = cos(3φ)."""
        k = 3
        r = jnp.cos(k * phi)
        # Offset to keep positive
        r = r + 1.1  # Shift so all positive
        x = r * jnp.cos(phi)
        y = r * jnp.sin(phi)
        z = jnp.zeros_like(phi)
        return jnp.array([x, y, z])
    
    coils = CreateCoilsAroundAxis(
        n_coils=6,
        order=5,
        coil_radius=0.08,
        n_samples=150,
        axis_function=rose_curve_axis,
        axis_shape='custom',
        nfp=1,
        stellsym=False
    )
    
    print(f"Created {len(coils)} coils around 3-petal rose curve")
    print(f"Coil shape: {coils.gamma.shape}")
    
    length = coils.length
    print(f"Coil lengths (mm): {length * 1000}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    coils.plot(ax=ax, show=False, color='brown', linewidth=2)
    ax.set_title('3-Petal Rose Curve - Coils')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig('/home/exouser/ESSOS/examples/coils_example_6_rose.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: coils_example_6_rose.png")
    plt.close()


# ============================================================================
# Example 7: Coils around spiral axis (custom function)
# ============================================================================
def example_7_spiral():
    """
    Create coils around an Archimedean spiral axis.
    Spiral: r(φ) = a + b·φ, giving an outward spiral pattern.
    """
    print("\n" + "="*70)
    print("Example 7: Coils around SPIRAL axis (custom function)")
    print("="*70)
    
    def spiral_axis(phi):
        """Archimedean spiral: r(φ) = 0.5 + 0.5·(φ/(2π))."""
        # Normalize phi to [0, 1] for spiral parametrization
        a, b = 0.5, 0.5  # base radius and spiral rate
        spiral_r = a + b * (phi / (2 * jnp.pi))
        x = spiral_r * jnp.cos(phi)
        y = spiral_r * jnp.sin(phi)
        z = 0.1 * phi / (2 * jnp.pi)  # Slight vertical variation
        return jnp.array([x, y, z])
    
    coils = CreateCoilsAroundAxis(
        n_coils=5,
        order=4,
        coil_radius=0.08,
        n_samples=150,
        axis_function=spiral_axis,
        axis_shape='custom',
        nfp=1,
        stellsym=False
    )
    
    print(f"Created {len(coils)} coils around spiral axis")
    print(f"Coil shape: {coils.gamma.shape}")
    
    length = coils.length
    print(f"Coil lengths (mm): {length * 1000}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    coils.plot(ax=ax, show=False, color='cyan', linewidth=2)
    ax.set_title('Spiral Axis - Coils')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig('/home/exouser/ESSOS/examples/coils_example_7_spiral.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: coils_example_7_spiral.png")
    plt.close()


# ============================================================================
# Example 8: Compare circular and custom axes side-by-side
# ============================================================================
def example_8_comparison():
    """
    Create coils around different axes and compare geometries.
    """
    print("\n" + "="*70)
    print("Example 8: COMPARISON of different axes")
    print("="*70)
    
    # Create several coil sets with different axes
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Circular
    coils_circular = CreateCoilsAroundAxis(
        n_coils=4, order=3, coil_radius=0.1,
        axis_shape='circle', axis_major_radius=1.0
    )
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    coils_circular.plot(ax=ax1, show=False, color='blue', linewidth=1.5)
    ax1.set_title('Circular Axis', fontsize=12, fontweight='bold')
    
    # 2. Elliptical
    coils_ellipse = CreateCoilsAroundAxis(
        n_coils=4, order=3, coil_radius=0.1,
        axis_shape='ellipse', axis_major_radius=1.0
    )
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    coils_ellipse.plot(ax=ax2, show=False, color='green', linewidth=1.5)
    ax2.set_title('Elliptical Axis', fontsize=12, fontweight='bold')
    
    # 3. Helical (no twist)
    coils_helical = CreateCoilsAroundAxis(
        n_coils=4, order=3, coil_radius=0.1,
        axis_shape='helical', axis_pitch=0.3, axis_twist_rate=0.0
    )
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    coils_helical.plot(ax=ax3, show=False, color='red', linewidth=1.5)
    ax3.set_title('Helical (pitch=0.3)', fontsize=12, fontweight='bold')
    
    # 4. Helical (with twist)
    coils_helical_twist = CreateCoilsAroundAxis(
        n_coils=4, order=4, coil_radius=0.1,
        axis_shape='helical', axis_pitch=0.3, axis_twist_rate=0.3
    )
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    coils_helical_twist.plot(ax=ax4, show=False, color='purple', linewidth=1.5)
    ax4.set_title('Helical (twist_rate=0.3)', fontsize=12, fontweight='bold')
    
    # 5. Figure-eight
    def fig8(phi):
        R = 1.0
        denom = 1.0 + jnp.sin(phi)**2
        r = R * jnp.sqrt(2.0) * jnp.cos(phi) / denom
        return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), jnp.zeros_like(phi)])
    
    coils_fig8 = CreateCoilsAroundAxis(
        n_coils=6, order=4, coil_radius=0.08,
        axis_function=fig8, axis_shape='custom'
    )
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    coils_fig8.plot(ax=ax5, show=False, color='orange', linewidth=1.5)
    ax5.set_title('Figure-Eight', fontsize=12, fontweight='bold')
    
    # 6. Rose curve
    def rose(phi):
        r = 1.1 + jnp.cos(3 * phi)
        return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), jnp.zeros_like(phi)])
    
    coils_rose = CreateCoilsAroundAxis(
        n_coils=6, order=4, coil_radius=0.08,
        axis_function=rose, axis_shape='custom'
    )
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    coils_rose.plot(ax=ax6, show=False, color='brown', linewidth=1.5)
    ax6.set_title('3-Petal Rose', fontsize=12, fontweight='bold')
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('/home/exouser/ESSOS/examples/coils_example_8_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: coils_example_8_comparison.png")
    plt.close()


# ============================================================================
# Example 9: Gradient computation and differentiability
# ============================================================================
def example_9_gradients():
    """
    Demonstrate JAX differentiation through coil geometry.
    Compute gradient of total coil length w.r.t. axis parameters.
    """
    print("\n" + "="*70)
    print("Example 9: GRADIENT computation (JAX differentiation)")
    print("="*70)
    
    # Function to compute total coil length as function of axis pitch
    def total_coil_length(axis_pitch_val):
        coils = CreateCoilsAroundAxis(
            n_coils=3,
            order=3,
            coil_radius=0.1,
            axis_shape='helical',
            axis_pitch=axis_pitch_val,
            axis_twist_rate=0.0
        )
        return jnp.mean(coils.length)  # Mean coil length
    
    # Compute gradient using JAX autodiff
    pitch_value = 0.2
    grad_length = jax.grad(total_coil_length)(pitch_value)
    
    print(f"Coil radius: 0.1")
    print(f"Axis pitch: {pitch_value}")
    print(f"Mean coil length: {total_coil_length(pitch_value):.6f}")
    print(f"∂(length)/∂(pitch) = {grad_length:.6f}")
    print("\n✓ JAX autodiff works through coil creation!")


# ============================================================================
# Main execution
# ============================================================================
def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "CUSTOM AXIS COIL GENERATION EXAMPLES" + " "*17 + "║")
    print("╚" + "="*68 + "╝")
    
    example_1_circular_axis()
    example_2_elliptical_axis()
    example_3_helical_axis()
    example_4_helical_twisted()
    example_5_figure_eight()
    example_6_rose_curve()
    example_7_spiral()
    example_8_comparison()
    example_9_gradients()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
