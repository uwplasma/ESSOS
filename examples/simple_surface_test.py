"""Simple test to verify surface plotting works."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def helical_axis(phi):
    """Simple helical axis."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

# Create surface
x1_coeffs = np.zeros(13)
x1_coeffs[0] = 1.0
x1_coeffs[2] = 0.3  # Larger minor radius
x2_coeffs = np.zeros(13)

surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_function=helical_axis,
    ntheta=30,
    nphi=30
)

gamma = np.array(surface.gamma)
print(f"Surface shape: {gamma.shape}")
print(f"Has NaN: {np.any(np.isnan(gamma))}")
print(f"X range: [{gamma[:,:,0].min():.3f}, {gamma[:,:,0].max():.3f}]")
print(f"Y range: [{gamma[:,:,1].min():.3f}, {gamma[:,:,1].max():.3f}]")
print(f"Z range: [{gamma[:,:,2].min():.3f}, {gamma[:,:,2].max():.3f}]")

# Plot axis for reference
phi_axis = np.linspace(0, 2*np.pi, 100)
axis_curve = np.array(helical_axis(jnp.array(phi_axis)))

print("\n" + "="*70)
print("Creating 3 plots to compare:")
print("1. Manual plot_surface (should definitely show)")
print("2. Using surface.plot() method")
print("3. Both together")
print("="*70)

fig = plt.figure(figsize=(18, 6))

# Plot 1: Manual plot_surface
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2],
                 cmap='viridis', alpha=1.0, edgecolor='k', linewidth=0.3)
ax1.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2], 
         'r-', linewidth=3, label='Axis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Manual plot_surface\n(BASELINE - should be visible)', fontweight='bold')
ax1.legend()

# Plot 2: Using surface.plot() method
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
print("\nCalling surface.plot()...")
surface.plot(ax=ax2, show=False, cmap='viridis', alpha=1.0)
ax2.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2], 
         'r-', linewidth=3, label='Axis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Using surface.plot() method\n(Test - is surface visible?)', fontweight='bold')
ax2.legend()

# Plot 3: Check what happens with default alpha
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surface.plot(ax=ax3, show=False, cmap='plasma')
ax3.plot(axis_curve[:, 0], axis_curve[:, 1], axis_curve[:, 2], 
         'r-', linewidth=3, label='Axis')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('surface.plot() with defaults\n(Default alpha=0.6)', fontweight='bold')
ax3.legend()

plt.tight_layout()
plt.savefig('/home/exouser/ESSOS/examples/simple_surface_comparison.png', dpi=150)
print("\n✓ Saved: simple_surface_comparison.png")
print("\nPlease check if:")
print("  1. Surface is visible in plot 1 (manual)")
print("  2. Surface is visible in plot 2 (using .plot() method)")
print("  3. Surface is visible in plot 3 (with default alpha)")
