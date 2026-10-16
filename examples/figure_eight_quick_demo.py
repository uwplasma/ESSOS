"""
Quick Start: Figure-Eight Surface + Coils
Minimal code to visualize a custom axis with both surface and coils.
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


# Define figure-eight axis
def fig8_axis(phi):
    R = 1.0
    r = R * jnp.sqrt(2.0) * jnp.cos(phi) / (1.0 + jnp.sin(phi)**2)
    return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), 0.0])


# Create surface
x1 = jnp.zeros(9); x1 = x1.at[4].set(0.15)  # Constant cross-section
surface = SurfaceGFrameFourier(x1, x1, nfp=1, mpol=2, ntor=2,
    ntheta=40, nphi=50, axis_shape='custom', axis_function=fig8_axis)

# Create coils
coils = CreateCoilsAroundAxis(n_coils=8, order=4, coil_radius=0.12,
    axis_function=fig8_axis, axis_shape='custom')

# Plot
fig = plt.figure(figsize=(16, 5))

# Subplot 1: Surface
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
gamma = np.array(surface.gamma)
ax1.plot_surface(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], cmap='viridis', alpha=0.7)
ax1.set_title('Surface'); ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

# Subplot 2: Coils
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
coils.plot(ax=ax2, show=False, color='blue', linewidth=2)
ax2.set_title('Coils'); ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

# Subplot 3: Combined
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2], cmap='viridis', alpha=0.5)
coils.plot(ax=ax3, show=False, color='darkred', linewidth=2)
ax3.set_title('Combined'); ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')

plt.tight_layout()
plt.savefig('/home/exouser/ESSOS/examples/fig8_quick_demo.png', dpi=150, bbox_inches='tight')
print("✓ Saved: fig8_quick_demo.png")
plt.show()
