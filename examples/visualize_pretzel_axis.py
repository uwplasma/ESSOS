"""Visualize the pretzel axis structure."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pretzel_axis(phi):
    """Parametric pretzel axis."""
    # Main loop structure
    r_loop = 2.0 + 1.5 * jnp.cos(2 * phi)
    x = r_loop * jnp.cos(phi)
    y = r_loop * jnp.sin(phi)
    
    # Vertical twist with additional modulation
    a = 1.5
    z = a * jnp.sin(3 * phi) + 0.5 * jnp.sin(5 * phi)
    
    return jnp.stack([x, y, z], axis=-1)

# Sample the curve
n_samples = 300
phi_vals = np.linspace(0, 2*np.pi, n_samples, endpoint=True)
curve_pts = np.array([pretzel_axis(jnp.array(phi)) for phi in phi_vals])

fig = plt.figure(figsize=(16, 5))

# 3D view
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2], 'b-', linewidth=2.5, label='Axis curve')

# Color by parameter value
scatter = ax1.scatter(curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2], 
                     c=phi_vals, cmap='hsv', s=5, alpha=0.5)

ax1.set_xlabel('X', fontsize=11)
ax1.set_ylabel('Y', fontsize=11)
ax1.set_zlabel('Z', fontsize=11)
ax1.set_title('3D Pretzel Axis', fontsize=13, fontweight='bold')
ax1.view_init(elev=25, azim=45)
ax1.grid(True, alpha=0.3)

# Top view (XY plane)
ax2 = fig.add_subplot(132)
ax2.plot(curve_pts[:, 0], curve_pts[:, 1], 'b-', linewidth=2)
scatter2 = ax2.scatter(curve_pts[:, 0], curve_pts[:, 1], c=phi_vals, cmap='hsv', s=10, alpha=0.6)
plt.colorbar(scatter2, ax=ax2, label='Parameter φ')

ax2.set_xlabel('X', fontsize=11)
ax2.set_ylabel('Y', fontsize=11)
ax2.set_title('Top View (XY): Pretzel Shape', fontsize=13, fontweight='bold')
ax2.axis('equal')
ax2.grid(True, alpha=0.3)

# Side view (XZ plane)
ax3 = fig.add_subplot(133)
ax3.plot(curve_pts[:, 0], curve_pts[:, 2], 'b-', linewidth=2)
ax3.scatter(curve_pts[:, 0], curve_pts[:, 2], c=phi_vals, cmap='hsv', s=10, alpha=0.6)

ax3.set_xlabel('X', fontsize=11)
ax3.set_ylabel('Z (vertical twist)', fontsize=11)
ax3.set_title('Side View (XZ): Twisting', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/exouser/ESSOS/examples/pretzel_axis_structure.png', dpi=150, bbox_inches='tight')
print("✅ Saved: pretzel_axis_structure.png")

print("\nPretzel axis characteristics:")
print(f"  - Shape: Twisted knot with multiple loops")
print(f"  - XY extent: [-{max(abs(curve_pts[:,0])):.1f}, {max(abs(curve_pts[:,0])):.1f}]")
print(f"  - Z extent (vertical twist): [{np.min(curve_pts[:,2]):.2f}, {np.max(curve_pts[:,2]):.2f}]")
print(f"  - Number of vertical oscillations: 3 complete cycles in 2π parameter range")
