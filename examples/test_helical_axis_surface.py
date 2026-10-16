"""
Test surface and coils around a helical axis.
This is a simpler test case to verify the custom axis functionality.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
import sys
sys.path.insert(0, '/home/exouser/ESSOS')

from essos.surfaces import SurfaceGFrameFourier
from essos.coils import CreateCoilsAroundAxis


# ============================================================================
# Helical Axis Definition
# ============================================================================

def helical_axis(phi):
    """
    Parametrize a helical axis that closes on itself.
    
    The axis spirals in 3D space with radius modulation and helical twist,
    but returns to the starting point when phi goes from 0 to 2π.
    
    Parameters:
        phi: poloidal angle (array or scalar)
        
    Returns:
        Cartesian coordinates [x, y, z] with shape (..., 3)
    """
    R = 10.0  # Major radius
    n_periods = 2  # Number of helical periods as we go around
    
    # Radius modulation: varies sinusoidally as we go around
    r_mod = R + 3.0 * jnp.sin(n_periods * phi)
    
    # Toroidal position (major circle)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    
    # Helical twist: z modulation that gives helical character
    z = 4.0 * jnp.sin(n_periods * phi)
    
    return jnp.stack([x, y, z], axis=-1)


# ============================================================================
# Surface Creation
# ============================================================================

def create_helical_surface():
    """Create a surface around a helical axis using G-frame parametrization."""
    
    print("\nSurface Configuration:")
    print("  - Axis: Helical (R=1.0, pitch=0.3)")
    print("  - mpol: 2, ntor: 2, nfp: 1")
    
    mpol = 2
    ntor = 2
    nfp = 1
    
    # Coefficient array for surface in G-frame coordinates
    # Shape: (2, coeff_size) for x1 and x2 coordinates (x3 = 0 for axis-aligned frame)
    coeff_size = (mpol + 1) * (2 * ntor + 1) - ntor
    print(f"  - Coefficient size: {coeff_size}")
    
    x1_coeffs = np.zeros(coeff_size)
    x1_coeffs[0] = 0.0      # Zero offset (m=0,n=0)
    x1_coeffs[5] = 0.15     # Minor radius X1 = r*cos(theta) (m=1,n=0)
    
    x2_coeffs = np.zeros(coeff_size)
    x2_coeffs[5] = 0.15     # Minor radius X2 = r*sin(theta) (m=1,n=0) - creates circular cross-section
    
    # Create surface with custom axis
    surface = SurfaceGFrameFourier(
        x1_coeffs=x1_coeffs,
        x2_coeffs=x2_coeffs,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        axis_function=helical_axis,
        ntheta=60,
        nphi=60
    )
    
    return surface


# ============================================================================
# Coil Creation
# ============================================================================

def create_helical_coils(n_coils=6, order=4):
    """Create coils placed symmetrically around the helical axis."""
    
    print("\nCoil Configuration:")
    print(f"  - Number of coils: {n_coils}")
    print(f"  - Fourier order: {order}")
    
    # Use CreateCoilsAroundAxis with helical axis
    coils = CreateCoilsAroundAxis(
        n_coils=n_coils,
        order=order,
        coil_radius=4.0,  # Just outside the surface (surface minor radius = 0.15)
        axis_shape='custom',
        axis_function=helical_axis,
        nfp=1,
        stellsym=False
    )
    
    return coils


# ============================================================================
# Plotting
# ============================================================================

def plot_helical_system():
    """Plot the helical system with surface and coils."""
    
    print("\nCreating surface...")
    surface = create_helical_surface()
    
    # Check if surface has NaN
    import numpy as np
    gamma = np.array(surface.gamma)
    print(f"Surface gamma shape: {gamma.shape}")
    print(f"Surface has NaN: {np.any(np.isnan(gamma))}")
    if not np.any(np.isnan(gamma)):
        print(f"Surface range: X=[{gamma[:,:,0].min():.2f}, {gamma[:,:,0].max():.2f}], " +
              f"Y=[{gamma[:,:,1].min():.2f}, {gamma[:,:,1].max():.2f}], " +
              f"Z=[{gamma[:,:,2].min():.2f}, {gamma[:,:,2].max():.2f}]")
    
    print("Creating coils...")
    coils = create_helical_coils()
    
    print("Generating plots...")
    fig = plt.figure(figsize=(16, 10))
    
    # Prepare axis curve
    phi_axis = np.linspace(0, 4*np.pi, 300)  # Two full periods
    axis_curve = np.array(helical_axis(jnp.array(phi_axis)))  # Shape: (N, 3)
    
    # ========== Subplot 1: Surface only ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surface.plot(ax=ax1, show=False, cmap='viridis')
    ax1.set_title('Surface (G-frame, Helical Axis)', fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    
    # ========== Subplot 2: Surface + Coils ==========
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surface.plot(ax=ax2, show=False, cmap='viridis', alpha=0.8)
    coils.plot(ax=ax2, show=False, color='red', linewidth=2)
    ax2.set_title('Surface + Coils', fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    
    # ========== Subplot 3: Coils only ==========
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    coils.plot(ax=ax3, show=False, color='red', linewidth=2.5)
    ax3.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2, label='Helix axis')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Coils Around Axis', fontweight='bold')
    ax3.legend()
    ax3.view_init(elev=20, azim=45)
    
    # ========== Subplot 4: Axis visualization ==========
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k-', linewidth=3, label='Helical axis')
    
    # Plot coil centers
    n_coils = len(coils)
    coil_phis = (np.arange(n_coils) / n_coils) * 2 * np.pi
    axis_centers = np.array(helical_axis(jnp.array(coil_phis)))
    ax4.scatter(axis_centers[:, 0], axis_centers[:, 1], axis_centers[:, 2],
                c='red', s=100, marker='o', label='Coil centers')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Axis and Coil Positions', fontweight='bold')
    ax4.legend()
    ax4.view_init(elev=20, azim=45)
    
    # ========== Subplot 5: Side view (X-Z plane) ==========
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    surface.plot(ax=ax5, show=False, cmap='viridis', alpha=0.8)
    coils.plot(ax=ax5, show=False, color='red', linewidth=2)
    ax5.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2)
    ax5.set_title('Side View (X-Z)', fontweight='bold')
    ax5.view_init(elev=0, azim=0)
    
    # ========== Subplot 6: Top view (X-Y plane) ==========
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    surface.plot(ax=ax6, show=False, cmap='viridis', alpha=0.8)
    coils.plot(ax=ax6, show=False, color='red', linewidth=2)
    ax6.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2],
             'k--', linewidth=2)
    ax6.set_title('Top View (X-Y)', fontweight='bold')
    ax6.view_init(elev=90, azim=0)
    
    plt.tight_layout()
    plt.savefig('/home/exouser/ESSOS/examples/test_helical_axis.png', dpi=150)
    print("✓ Saved: test_helical_axis.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("HELICAL AXIS TEST - SURFACE AND COILS")
    print("="*70)
    
    plot_helical_system()
    
    print("\nExporting to VTK format...")
    surface = create_helical_surface()
    coils = create_helical_coils()
    surface.to_vtk('helical_surface')
    coils.to_vtk('helical_coils')
    print("✓ Saved: helical_surface.vts")
    print("✓ Saved: helical_coils.vtu")
    
    print("\n✅ Test complete!")
