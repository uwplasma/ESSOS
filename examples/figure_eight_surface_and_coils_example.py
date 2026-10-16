"""
Example: Figure-Eight Axis with Both Surface and Coils

This example demonstrates:
1. Define a custom figure-eight (lemniscate) axis
2. Create a G-frame surface around this axis using SurfaceGFrameFourier
3. Create coils around the same axis using CreateCoilsAroundAxis
4. Plot both together with matplotlib to visualize the coordinate system
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

import sys
sys.path.insert(0, '/home/exouser/ESSOS')

from essos.surfaces import SurfaceGFrameFourier
from essos.coils import CreateCoilsAroundAxis


# ============================================================================
# Define the Figure-Eight (Lemniscate) Axis
# ============================================================================

# def figure_eight_axis(phi):
#     """
#     Lemniscate (figure-eight) parametrization: r² = cos(2φ)
    
#     This creates a proper figure-eight shape traced in the xy-plane.
#     """
#     # Lemniscate in polar coordinates: r = sqrt(|cos(2*phi)|)
#     cos_2phi = jnp.cos(2 * phi)
#     r = jnp.sqrt(jnp.abs(cos_2phi) + 1e-10)  # Add small epsilon to avoid NaN
    
#     x = r * jnp.cos(phi)
#     y = r * jnp.sin(phi)
#     z = jnp.zeros_like(4.*phi)
    
#     return jnp.stack([x, y, z], axis=-1)


def figure_eight_axis(phi):
    """
    Figure-eight axis that doesn't self-intersect.
    
    The two lobes are separated in z-direction when they would cross in XY plane.
    """
    a = 10.0
    x = a * jnp.cos(phi)
    y = a * jnp.sin(phi) * jnp.cos(phi)
    # z varies with phi to separate the crossing points
    # At phi=π/2 (first crossing), z is positive
    # At phi=3π/2 (second crossing), z is negative
    z = 3.0 * jnp.sin(phi)
    return jnp.stack([x, y, z], axis=-1)

# ============================================================================
# Create Surface around Figure-Eight Axis
# ============================================================================

def create_figure_eight_surface():
    """Create a G-frame surface using the figure-eight axis."""
    
    # Fourier coefficients for X1 and X2 (poloidal modes)
    # Simple case: just constant cross-section
    mpol = 2  # poloidal order
    ntor = 2  # toroidal order
    nfp = 1   # number of field periods
    
    # Create simple X1, X2 coefficients (flat cross-section)
    # Size: (mpol+1)*(2*ntor+1) - ntor
    coeff_size = (mpol + 1) * (2 * ntor + 1) - ntor
    x1_coeffs = jnp.zeros(coeff_size)
    x2_coeffs = jnp.zeros(coeff_size)
    
    # X1: Set minor radius using poloidal mode (m=1, n=0)
    # Mode (m=1, n=0) is at index: (mpol+1)*ntor + m = (2+1)*2 + 1 = 7... wait, let me calculate properly
    # Modes are ordered: m from 0 to mpol, n from -ntor to +ntor (skipping when m=0 and n<0)
    # For mpol=2, ntor=2: (m=0,n=0), (m=0,n=1), (m=0,n=2), (m=1,n=-2), (m=1,n=-1), (m=1,n=0), ...
    # Index 5 corresponds to (m=1, n=0) - the pure poloidal mode
    x1_coeffs = x1_coeffs.at[0].set(0.0)     # Zero offset (m=0, n=0)
    x1_coeffs = x1_coeffs.at[5].set(1.0)    # Minor radius X1 = r*cos(theta) (m=1, n=0)
    x2_coeffs = x2_coeffs.at[0].set(0.0)
    x2_coeffs = x2_coeffs.at[5].set(1.0)    # Minor radius X2 = r*sin(theta) (m=1, n=0) - creates circular cross-section
    
    # Create surface
    surface = SurfaceGFrameFourier(
        x1_coeffs=x1_coeffs,
        x2_coeffs=x2_coeffs,
        nfp=nfp,
        ntheta=60,
        nphi=60,
        mpol=mpol,
        ntor=ntor,
        axis_shape='custom',
        axis_function=figure_eight_axis,
        axis_major_radius=20.0
    )
    
    return surface


# ============================================================================
# Create Coils around Figure-Eight Axis
# ============================================================================

def create_figure_eight_coils():
    """Create coils positioned around the figure-eight axis."""
    
    coils = CreateCoilsAroundAxis(
        n_coils=8,
        order=4,
        coil_radius=4.0,  # Just outside the surface (surface minor radius = 1.0)
        axis_function=figure_eight_axis,
        axis_shape='custom',
        nfp=1,
        stellsym=False
    )
    
    return coils


# ============================================================================
# Plotting Utilities
# ============================================================================

class Arrow3D(FancyArrowPatch):
    """Helper class for 3D arrows in matplotlib."""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        l, m, n = self._dxdydz
        xs = [x1, x1+l]
        ys = [y1, y1+m]
        zs = [z1, z1+n]

        xs, ys, zs = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions(((xs[0], ys[0]), (xs[1], ys[1])))
        return FancyArrowPatch.draw(self, renderer)


def plot_figure_eight_system():
    """Plot the complete figure-eight system with surface and coils."""
    
    print("Creating figure-eight surface...")
    surface = create_figure_eight_surface()
    
    print("Creating figure-eight coils...")
    coils = create_figure_eight_coils()
    
    print("Generating plot...")
    fig = plt.figure(figsize=(16, 12))
    
    # ========== Prepare axis and surface data ==========
    phi_axis = np.linspace(0, 2*np.pi, 200)
    axis_curve = np.array(figure_eight_axis(jnp.array(phi_axis)))  # Shape: (N, 3)
    
    # ========== Subplot 1: Surface only ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surface.plot(ax=ax1, show=False, cmap='plasma')
    ax1.set_title('Surface (G-frame, Figure-Eight Axis)', fontweight='bold', fontsize=11)
    ax1.view_init(elev=20, azim=45)
    
    # ========== Subplot 2: Surface + Coils ==========
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surface.plot(ax=ax2, show=False, cmap='plasma', alpha=0.9)
    coils.plot(ax=ax2, show=False, color='blue', linewidth=2.5)
    ax2.set_title('Surface + Coils (Colored Separately)', fontweight='bold', fontsize=11)
    ax2.view_init(elev=20, azim=45)
    
    # ========== Subplot 3: Both together ==========
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surface.plot(ax=ax3, show=False, cmap='plasma', alpha=0.9)
    coils.plot(ax=ax3, show=False, color='darkred', linewidth=1.5, alpha=0.7)
    ax3.set_title('Surface + Coils (Combined View)', fontweight='bold', fontsize=11)
    ax3.view_init(elev=20, azim=45)
    
    # ========== Subplot 4: Axis curve visualization ==========
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Plot axis curve
    ax4.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k-', linewidth=3, label='Axis')
    
    # Plot coil centers (at axis positions)
    n_coils = len(coils)
    coil_phis = (np.arange(n_coils) + 0.5) * (2 * np.pi) / n_coils
    axis_centers = np.array(figure_eight_axis(jnp.array(coil_phis)))  # Shape: (N, 3)
    ax4.scatter(axis_centers[:, 0], axis_centers[:, 1], axis_centers[:, 2],
                c='red', s=100, marker='o', label='Coil Centers')
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Axis Curve & Coil Positions', fontweight='bold', fontsize=11)
    ax4.legend()
    ax4.view_init(elev=20, azim=45)
    
    # ========== Subplot 5: Top-down view (X-Y plane) ==========
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    surface.plot(ax=ax5, show=False, cmap='plasma', alpha=0.9)
    coils.plot(ax=ax5, show=False, color='darkred', linewidth=1.5, alpha=0.7)
    ax5.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=1.5, alpha=0.7)
    ax5.set_title('Top-Down View', fontweight='bold', fontsize=11)
    ax5.view_init(elev=90, azim=0)  # Looking down
    
    # ========== Subplot 6: Side view (X-Z plane) ==========
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    surface.plot(ax=ax6, show=False, cmap='plasma', alpha=0.8)
    coils.plot(ax=ax6, show=False, color='darkred', linewidth=2)
    ax6.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=1.5, alpha=0.7)
    ax6.set_title('Side View', fontweight='bold', fontsize=11)
    ax6.view_init(elev=0, azim=0)  # Side view
    
    plt.tight_layout()
    plt.savefig('/home/exouser/ESSOS/examples/figure_eight_surface_and_coils.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: figure_eight_surface_and_coils.png")
    plt.close()


# ============================================================================
# Detailed Analysis Plots
# ============================================================================

def plot_detailed_analysis():
    """Create detailed analysis plots of the figure-eight system."""
    
    print("\nCreating detailed analysis plots...")
    
    surface = create_figure_eight_surface()
    coils = create_figure_eight_coils()
    
    fig = plt.figure(figsize=(16, 10))
    
    # ========== Coil geometry analysis ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot coils with different colors for each
    n_coils = len(coils)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_coils))
    
    for i, (gamma_coil, color) in enumerate(zip(coils.gamma, colors)):
        gamma_coil = np.array(gamma_coil)
        # Close the loop for visualization
        gamma_coil_closed = np.vstack([gamma_coil, gamma_coil[0:1]])
        ax1.plot(gamma_coil_closed[:, 0], gamma_coil_closed[:, 1], gamma_coil_closed[:, 2],
                color=color, linewidth=1.5, label=f'Coil {i+1}')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Individual Coils (Color-Coded)', fontweight='bold', fontsize=11)
    ax1.view_init(elev=20, azim=45)
    
    # ========== Coil length distribution ==========
    ax2 = fig.add_subplot(2, 3, 2)
    
    coil_lengths = np.array(coils.length)
    ax2.bar(range(len(coil_lengths)), coil_lengths, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Coil Index')
    ax2.set_ylabel('Length (m)')
    ax2.set_title('Coil Lengths', fontweight='bold', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    # ========== Surface cross-section at equators ==========
    ax3 = fig.add_subplot(2, 3, 3)
    
    gamma = np.array(surface.gamma)
    # Plot a few poloidal cross-sections at different toroidal angles
    n_phi_sections = 6
    phi_indices = np.linspace(0, gamma.shape[0]-1, n_phi_sections, dtype=int)
    
    for idx, phi_idx in enumerate(phi_indices):
        cross_section = gamma[phi_idx, :, :2]  # X-Y plane
        ax3.plot(cross_section[:, 0], cross_section[:, 1], 'o-', 
                label=f'φ section {idx}', alpha=0.7)
    
    # Plot axis in X-Y plane
    phi_axis = np.linspace(0, 2*np.pi, 200)
    axis_curve_full = np.array(figure_eight_axis(jnp.array(phi_axis)))  # Shape: (N, 3)
    ax3.plot(axis_curve_full[:, 0], axis_curve_full[:, 1], 'k--', linewidth=2, label='Axis')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Surface Cross-Sections (X-Y plane)', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.axis('equal')
    
    # ========== Coil positions around axis ==========
    ax4 = fig.add_subplot(2, 3, 4)
    
    n_coils = len(coils)
    coil_phis = (np.arange(n_coils) + 0.5) * (2 * np.pi) / n_coils
    axis_centers = np.array(figure_eight_axis(jnp.array(coil_phis)))  # Shape: (N, 3)
    
    # Plot axis
    ax4.plot(axis_curve_full[:, 0], axis_curve_full[:, 1], 'k-', linewidth=2.5, label='Axis')
    
    # Plot coil centers
    ax4.scatter(axis_centers[:, 0], axis_centers[:, 1], c=list(range(n_coils)),
               cmap='rainbow', s=200, marker='*', edgecolors='black', linewidths=1.5,
               label='Coil Centers', zorder=5)
    
    # Add labels
    for i, (x, y) in enumerate(zip(axis_centers[:, 0], axis_centers[:, 1])):
        ax4.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Coil Positions Around Axis (Top View)', fontweight='bold', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.axis('equal')
    
    # ========== 3D scatter of coil centers vs axis ==========
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    # Dense axis samples
    phi_dense = np.linspace(0, 2*np.pi, 300)
    axis_dense = np.array(figure_eight_axis(jnp.array(phi_dense)))  # Shape: (N, 3)
    ax5.plot(axis_dense[:, 0], axis_dense[:, 1], axis_dense[:, 2],
            'k-', linewidth=2, label='Axis', zorder=10)
    
    # Coil centers
    ax5.scatter(axis_centers[:, 0], axis_centers[:, 1], axis_centers[:, 2],
               c=list(range(n_coils)), cmap='rainbow', s=300, marker='*',
               edgecolors='black', linewidths=2, label='Coil Centers', zorder=20)
    
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('Coil Centers vs Axis (3D)', fontweight='bold', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.view_init(elev=30, azim=45)
    
    # ========== Fourier mode content ==========
    ax6 = fig.add_subplot(2, 3, 6)
    
    order = coils.order
    mode_indices = np.arange(2*order + 1)
    
    # Show coefficient magnitudes for first coil
    dofs_first = np.array(coils.dofs[0])  # (3, 2*order+1)
    
    # Combine all coordinates
    dofs_mag = np.max(np.abs(dofs_first), axis=0)
    
    ax6.bar(mode_indices, dofs_mag, color='coral', edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Fourier Mode Index')
    ax6.set_ylabel('Coefficient Magnitude')
    ax6.set_title(f'Fourier Spectrum (Coil 1, Order={order})', fontweight='bold', fontsize=11)
    ax6.set_yscale('log')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/exouser/ESSOS/examples/figure_eight_detailed_analysis.png',
                dpi=150, bbox_inches='tight')
    print("✓ Saved: figure_eight_detailed_analysis.png")
    plt.close()


# ============================================================================
# Print Summary Statistics
# ============================================================================

def print_summary():
    """Print summary statistics about the system."""
    
    print("\n" + "="*70)
    print("FIGURE-EIGHT SYSTEM SUMMARY")
    print("="*70)
    
    surface = create_figure_eight_surface()
    coils = create_figure_eight_coils()
    
    print("\nSURFACE PROPERTIES:")
    print(f"  Axis shape: custom (figure-eight/lemniscate)")
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
    print("║" + " "*12 + "FIGURE-EIGHT AXIS WITH SURFACE AND COILS" + " "*16 + "║")
    print("╚" + "="*68 + "╝\n")
    
    print_summary()
    
    print("Generating main visualization...")
    plot_figure_eight_system()
    
    print("\nGenerating detailed analysis...")
    plot_detailed_analysis()
    
    print("\nExporting to VTK format...")
    surface = create_figure_eight_surface()
    coils = create_figure_eight_coils()
    surface.to_vtk('figure_eight_surface')
    coils.to_vtk('figure_eight_coils')
    print("✓ Saved: figure_eight_surface.vts")
    print("✓ Saved: figure_eight_coils.vtu")
    
    print("\n✅ All visualizations complete!")
    print("\nGenerated files:")
    print("  - figure_eight_surface_and_coils.png")
    print("  - figure_eight_detailed_analysis.png")
    print("  - figure_eight_surface.vts (VTK for ParaView)")
    print("  - figure_eight_coils.vtu (VTK for ParaView)")


if __name__ == "__main__":
    main()
