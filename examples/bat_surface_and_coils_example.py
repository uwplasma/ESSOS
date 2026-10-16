"""
Example: Bat-Shaped Axis with Surface and Coils

This example demonstrates:
1. Define a bat-shaped axis
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
# Define the Bat-Shaped Axis
# ============================================================================

def bat_axis(phi):
    """
    Parametric bat-shaped axis.
    
    Creates a 3D bat curve with wings, body, and ears.
    
    Parameters:
        phi: parameter angle (0 to 2π)
        
    Returns:
        Cartesian coordinates [x, y, z] with shape (..., 3)
    """
    t = phi
    
    # Create bat shape using trigonometric functions
    # The bat has wings that spread out and ears at the top
    
    # Main body and wings parametrization
    # Use abs(cos(t)) to create symmetric wings
    r = 3.0 * jnp.abs(jnp.cos(t))**1.5 + 1.0 * jnp.sin(t)**2
    
    # Cartesian coordinates
    x = r * jnp.cos(t)
    y = r * jnp.sin(t)
    
    # Add ears at the top (when sin(t) is large and positive)
    # Ears are created by adding a bump at t ~ π/2
    ear_height = 1.5 * jnp.sin(t)**2 * jnp.cos(2*t)**2
    y = y + ear_height
    
    # Add z-component for 3D effect - subtle wave
    z = 0.5 * jnp.sin(t)**2 * jnp.sin(2*t)
    
    return jnp.stack([x, y, z], axis=-1)


# ============================================================================
# Create Surface around Bat Axis
# ============================================================================

def create_bat_surface():
    """Create a G-frame surface using the bat axis."""
    
    mpol = 2
    ntor = 2
    nfp = 1
    
    coeff_size = (mpol + 1) * (2 * ntor + 1) - ntor
    
    x1_coeffs = jnp.zeros(coeff_size)
    x1_coeffs = x1_coeffs.at[0].set(0.0)      # Zero offset
    x1_coeffs = x1_coeffs.at[5].set(0.5)      # Minor radius X1
    
    x2_coeffs = jnp.zeros(coeff_size)
    x2_coeffs = x2_coeffs.at[5].set(0.5)      # Minor radius X2
    
    surface = SurfaceGFrameFourier(
        x1_coeffs=x1_coeffs,
        x2_coeffs=x2_coeffs,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        axis_function=bat_axis,
        ntheta=60,
        nphi=60
    )
    
    return surface


# ============================================================================
# Create Coils around Bat Axis
# ============================================================================

def create_bat_coils():
    """Create coils positioned around the bat axis."""
    
    coils = CreateCoilsAroundAxis(
        n_coils=14,
        order=4,
        coil_radius=0.8,  # Just outside surface
        axis_function=bat_axis,
        axis_shape='custom',
        nfp=1,
        stellsym=False
    )
    
    return coils


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_bat_system():
    """Create comprehensive visualizations of the bat system."""
    
    print("Creating bat surface...")
    surface = create_bat_surface()
    
    print("Creating bat coils...")
    coils = create_bat_coils()
    
    print("Generating plot...")
    fig = plt.figure(figsize=(18, 12))
    
    # Prepare axis curve
    phi_axis = np.linspace(0, 2*np.pi, 200)
    axis_curve = np.array(bat_axis(jnp.array(phi_axis)))
    
    # ========== Subplot 1: Surface only ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surface.plot(ax=ax1, show=False, cmap='Greys')
    ax1.set_title('Bat Surface', fontweight='bold', fontsize=11)
    ax1.view_init(elev=20, azim=45)
    
    # ========== Subplot 2: Surface + Coils ==========
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surface.plot(ax=ax2, show=False, cmap='Greys', alpha=0.8)
    coils.plot(ax=ax2, show=False, color='black', linewidth=2)
    ax2.set_title('Surface + Coils', fontweight='bold', fontsize=11)
    ax2.view_init(elev=20, azim=45)
    
    # ========== Subplot 3: Axis curve and coil centers ==========
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k-', linewidth=3, label='Bat Axis', alpha=0.7)
    
    # Plot coil centers
    n_coils = len(coils)
    coil_phis = (np.arange(n_coils) + 0.5) * (2 * np.pi) / n_coils
    axis_centers = np.array(bat_axis(jnp.array(coil_phis)))
    ax3.scatter(axis_centers[:, 0], axis_centers[:, 1], axis_centers[:, 2],
                c='red', s=100, marker='o', label='Coil Centers', alpha=0.7)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Axis and Coil Positions', fontweight='bold', fontsize=11)
    ax3.legend()
    ax3.view_init(elev=20, azim=45)
    
    # ========== Subplot 4: Front view (wings) ==========
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surface.plot(ax=ax4, show=False, cmap='Greys', alpha=0.8)
    coils.plot(ax=ax4, show=False, color='black', linewidth=1.5, alpha=0.7)
    ax4.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2, alpha=0.5)
    ax4.set_title('Front View (Wings)', fontweight='bold', fontsize=11)
    ax4.view_init(elev=85, azim=0)
    
    # ========== Subplot 5: Side view ==========
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    surface.plot(ax=ax5, show=False, cmap='Greys', alpha=0.8)
    coils.plot(ax=ax5, show=False, color='black', linewidth=1.5, alpha=0.7)
    ax5.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2, alpha=0.5)
    ax5.set_title('Side View (XZ)', fontweight='bold', fontsize=11)
    ax5.view_init(elev=0, azim=0)
    
    # ========== Subplot 6: Angled view ==========
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    surface.plot(ax=ax6, show=False, cmap='Greys', alpha=0.8)
    coils.plot(ax=ax6, show=False, color='black', linewidth=2)
    ax6.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2, alpha=0.5)
    ax6.set_title('3D Perspective View', fontweight='bold', fontsize=11)
    ax6.view_init(elev=30, azim=120)
    
    plt.tight_layout()
    plt.savefig('/home/exouser/ESSOS/examples/bat_surface_and_coils.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: bat_surface_and_coils.png")
    plt.close()


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    """Print configuration summary."""
    
    surface = create_bat_surface()
    coils = create_bat_coils()
    
    print("\n" + "="*70)
    print("BAT-SHAPED AXIS SYSTEM SUMMARY")
    print("="*70)
    
    print("\nSURFACE PROPERTIES:")
    print(f"  Axis shape: custom (bat-shaped)")
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
    
    # Axis bounds
    phi_test = np.linspace(0, 2*np.pi, 100)
    axis_pts = np.array(bat_axis(jnp.array(phi_test)))
    print("\nAXIS GEOMETRY:")
    print(f"  X extent: [{np.min(axis_pts[:,0]):.2f}, {np.max(axis_pts[:,0]):.2f}]")
    print(f"  Y extent: [{np.min(axis_pts[:,1]):.2f}, {np.max(axis_pts[:,1]):.2f}]")
    print(f"  Z extent: [{np.min(axis_pts[:,2]):.2f}, {np.max(axis_pts[:,2]):.2f}]")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*19 + "BAT-SHAPED AXIS WITH SURFACE AND COILS" + " "*11 + "║")
    print("╚" + "="*68 + "╝\n")
    
    print_summary()
    
    print("Generating main visualization...")
    plot_bat_system()
    
    print("\nExporting to VTK format...")
    surface = create_bat_surface()
    coils = create_bat_coils()
    surface.to_vtk('bat_surface')
    coils.to_vtk('bat_coils')
    print("✓ Saved: bat_surface.vts")
    print("✓ Saved: bat_coils.vtu")
    
    print("\n✅ All visualizations complete!")
    print("\nGenerated files:")
    print("  - bat_surface_and_coils.png")
    print("  - bat_surface.vts (VTK for ParaView)")
    print("  - bat_coils.vtu (VTK for ParaView)")


if __name__ == "__main__":
    main()
