"""Show the fixed surfaces - demonstrate they revolve around the axis."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def helical_axis(phi):
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

def figure_eight_axis(phi):
    a = 1.0
    x = a * jnp.cos(phi)
    y = a * jnp.sin(phi) * jnp.cos(phi)
    z = 0.2 * jnp.sin(3 * phi)
    return jnp.stack([x, y, z], axis=-1)

# Create surfaces with fixed coefficients
configs = [
    ("Helical Axis", helical_axis, 0.15),
    ("Figure-Eight Axis", figure_eight_axis, 0.25),
]

fig = plt.figure(figsize=(14, 6))

for idx, (name, axis_func, minor_r) in enumerate(configs, 1):
    x1_coeffs = np.zeros(13)
    x1_coeffs[0] = 0.0      # Zero offset
    x1_coeffs[5] = minor_r  # Minor radius
    x2_coeffs = np.zeros(13)
    
    surface = SurfaceGFrameFourier(
        x1_coeffs=x1_coeffs,
        x2_coeffs=x2_coeffs,
        mpol=2,
        ntor=2,
        nfp=1,
        axis_function=axis_func,
        ntheta=60,
        nphi=60
    )
    
    gamma = np.array(surface.gamma)
    
    ax = fig.add_subplot(1, 2, idx, projection='3d')
    
    # Plot surface
    surface.plot(ax=ax, show=False,  alpha=0.7, edgecolor='none')
    
    # Plot axis
    phi_axis = np.linspace(0, 2*np.pi, 100)
    axis_pts = np.array(axis_func(jnp.array(phi_axis)))
    ax.plot(axis_pts[:, 0], axis_pts[:, 1], axis_pts[:, 2], 
            'r-', linewidth=3, label='Axis')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{name}\nSurface revolves around axis', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 1])

plt.tight_layout()
plt.savefig('/home/exouser/ESSOS/examples/surface_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: surface_comparison.png")
print("\nThe surfaces now properly revolve around their axes!")
