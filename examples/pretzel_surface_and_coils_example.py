"""
Example: Pretzel Axis with Surface and Coils

This example demonstrates:
1. Define a pretzel-shaped (twisted knot-like) axis
2. Create a G-frame surface around the axis using SurfaceGFrameFourier
3. Create coils around the same axis using CreateCoilsAroundAxis
4. Export to VTK for visualization in ParaView
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0, '/home/exouser/ESSOS')

from essos.surfaces import SurfaceGFrameFourier
from essos.coils import CreateCoilsAroundAxis


# ============================================================================
# Define the Pretzel Axis
# ============================================================================

def pretzel_axis(phi):
    """
    Parametric pretzel (twisted knot) axis.
    
    A pretzel has multiple loops that twist around each other,
    creating an intricate 3D curve.
    
    Parameters:
        phi: parameter angle
        
    Returns:
        Cartesian coordinates [x, y, z] with shape (..., 3)
    """
    # Use trigonometric functions to create a twisted loop structure
    # This creates a knot-like shape with multiple passes
    
    # Main loop structure (figure-8 like base)
    r_loop = 2.0 + 1.5 * jnp.cos(2 * phi)
    x = r_loop * jnp.cos(phi)
    y = r_loop * jnp.sin(phi)
    
    # Vertical twist with additional modulation
    # Creates the pretzel-like twisting
    a = 1.5  # amplitude of twist
    z = a * jnp.sin(3 * phi) + 0.5 * jnp.sin(5 * phi)
    
    return jnp.stack([x, y, z], axis=-1)


# ============================================================================
# Create Surface around Pretzel Axis
# ============================================================================

def create_pretzel_surface():
    """Create a G-frame surface using the pretzel axis."""
    
    mpol = 2
    ntor = 2
    nfp = 1
    
    coeff_size = (mpol + 1) * (2 * ntor + 1) - ntor
    
    x1_coeffs = jnp.zeros(coeff_size)
    x1_coeffs = x1_coeffs.at[0].set(0.0)      # Zero offset
    x1_coeffs = x1_coeffs.at[5].set(0.6)      # Minor radius X1
    
    x2_coeffs = jnp.zeros(coeff_size)
    x2_coeffs = x2_coeffs.at[5].set(0.6)      # Minor radius X2
    
    surface = SurfaceGFrameFourier(
        x1_coeffs=x1_coeffs,
        x2_coeffs=x2_coeffs,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        axis_function=pretzel_axis,
        ntheta=60,
        nphi=60
    )
    
    return surface


# ============================================================================
# Create Coils around Pretzel Axis
# ============================================================================

def create_pretzel_coils():
    """Create coils positioned around the pretzel axis."""
    
    coils = CreateCoilsAroundAxis(
        n_coils=10,
        order=4,
        coil_radius=1.0,  # Just outside surface
        axis_function=pretzel_axis,
        axis_shape='custom',
        nfp=1,
        stellsym=False
    )
    
    return coils


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_pretzel_system():
    """Create comprehensive visualizations of the pretzel system."""
    
    print("Creating pretzel surface...")
    surface = create_pretzel_surface()
    
    print("Creating pretzel coils...")
    coils = create_pretzel_coils()
    
    print("Generating plot...")
    fig = plt.figure(figsize=(18, 12))
    
    # Prepare axis curve
    phi_axis = np.linspace(0, 2*np.pi, 200)
    axis_curve = np.array(pretzel_axis(jnp.array(phi_axis)))
    
    # ========== Subplot 1: Surface only ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surface.plot(ax=ax1, show=False, cmap='plasma')
    ax1.set_title('Pretzel Surface', fontweight='bold', fontsize=11)
    ax1.view_init(elev=20, azim=45)
    
    # ========== Subplot 2: Surface + Coils ==========
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surface.plot(ax=ax2, show=False, cmap='plasma', alpha=0.8)
    coils.plot(ax=ax2, show=False, color='darkblue', linewidth=2)
    ax2.set_title('Surface + Coils', fontweight='bold', fontsize=11)
    ax2.view_init(elev=20, azim=45)
    
    # ========== Subplot 3: Axis curve and coil centers ==========
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k-', linewidth=3, label='Axis', alpha=0.7)
    
    # Plot coil centers
    n_coils = len(coils)
    coil_phis = (np.arange(n_coils) + 0.5) * (2 * np.pi) / n_coils
    axis_centers = np.array(pretzel_axis(jnp.array(coil_phis)))
    ax3.scatter(axis_centers[:, 0], axis_centers[:, 1], axis_centers[:, 2],
                c='red', s=100, marker='o', label='Coil Centers', alpha=0.7)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Axis and Coil Positions', fontweight='bold', fontsize=11)
    ax3.legend()
    ax3.view_init(elev=20, azim=45)
    
    # ========== Subplot 4: Top-down view ==========
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surface.plot(ax=ax4, show=False, cmap='plasma', alpha=0.8)
    coils.plot(ax=ax4, show=False, color='darkblue', linewidth=1.5, alpha=0.7)
    ax4.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2, alpha=0.5)
    ax4.set_title('Top-Down View', fontweight='bold', fontsize=11)
    ax4.view_init(elev=90, azim=0)
    
    # ========== Subplot 5: Front view ==========
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    surface.plot(ax=ax5, show=False, cmap='plasma', alpha=0.8)
    coils.plot(ax=ax5, show=False, color='darkblue', linewidth=1.5, alpha=0.7)
    ax5.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2, alpha=0.5)
    ax5.set_title('Front View (XY)', fontweight='bold', fontsize=11)
    ax5.view_init(elev=0, azim=0)
    
    # ========== Subplot 6: Side view ==========
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    surface.plot(ax=ax6, show=False, cmap='plasma', alpha=0.8)
    coils.plot(ax=ax6, show=False, color='darkblue', linewidth=2)
    ax6.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2, alpha=0.5)
    ax6.set_title('Side View (XZ)', fontweight='bold', fontsize=11)
    ax6.view_init(elev=0, azim=90)
    
    plt.tight_layout()
    plt.savefig('/home/exouser/ESSOS/examples/pretzel_surface_and_coils.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: pretzel_surface_and_coils.png")
    plt.close()


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    """Print configuration summary."""
    
    surface = create_pretzel_surface()
    coils = create_pretzel_coils()
    
    print("\n" + "="*70)
    print("PRETZEL AXIS SYSTEM SUMMARY")
    print("="*70)
    
    print("\nSURFACE PROPERTIES:")
    print(f"  Axis shape: custom (pretzel knot)")
    print(f"  Poloidal modes (mpol): {surface.mpol}")
    print(f"  Toroidal modes (ntor): {surface.ntor}")
    print(f"  Discretization: {surface.ntheta} θ × {surface.nphi} φ")
    print(f"  Surface DOFs per coordinate: {len(surface.x1_coeffs)}")
    print(f"  Total DOFs: {len(surface.dofs)}")
    
    print("\nCOILS PROPERTIES:")
    print(f"  Number of coils: {len(coils)}")
    print(f"  Fourier order: {coils.order}")
    print(f"  DOFs per coil: {coils.dofs.shape[1] * coils.dofs.shape[2]}")
    print(f"  Total DOFs: {np.prod(coils.dofs.shape)}")
    
    print("\nCOIL GEOMETRY:")
    coil_lengths = np.array(coils.length)
    print(f"  Coil lengths (m): {coil_lengths}")
    print(f"  Mean length: {np.mean(coil_lengths):.6f} m")
    print(f"  Min/Max length: {np.min(coil_lengths):.6f} / {np.max(coil_lengths):.6f} m")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "PRETZEL AXIS WITH SURFACE AND COILS" + " "*18 + "║")
    print("╚" + "="*68 + "╝\n")
    
    print_summary()
    
    print("Generating main visualization...")
    plot_pretzel_system()
    
    print("\nExporting to VTK format...")
    surface = create_pretzel_surface()
    coils = create_pretzel_coils()
    surface.to_vtk('pretzel_surface')
    coils.to_vtk('pretzel_coils')
    print("✓ Saved: pretzel_surface.vts")
    print("✓ Saved: pretzel_coils.vtu")
    
    print("\n✅ All visualizations complete!")
    print("\nGenerated files:")
    print("  - pretzel_surface_and_coils.png")
    print("  - pretzel_surface.vts (VTK for ParaView)")
    print("  - pretzel_coils.vtu (VTK for ParaView)")


if __name__ == "__main__":
    main()
