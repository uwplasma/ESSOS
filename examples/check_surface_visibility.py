"""Check if surfaces are actually visible."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def helical_axis(phi):
    """Helical axis that closes on itself."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

def figure_eight_axis(phi):
    """Figure-eight axis."""
    R = 1.0
    denom = 1.0 + jnp.sin(phi)**2
    r = R * jnp.sqrt(2.0) * jnp.cos(phi) / denom
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

print("="*70)
print("HELICAL SURFACE TEST")
print("="*70)

# Create helical surface
x1_coeffs = np.zeros(13)
x1_coeffs[0] = 1.0
x1_coeffs[2] = 0.15
x2_coeffs = np.zeros(13)

surface_helical = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_function=helical_axis,
    ntheta=30,
    nphi=30
)

gamma_h = np.array(surface_helical.gamma)
print(f"Shape: {gamma_h.shape}")
print(f"Has NaN: {np.any(np.isnan(gamma_h))}")
print(f"X range: [{gamma_h[:,:,0].min():.3f}, {gamma_h[:,:,0].max():.3f}]")
print(f"Y range: [{gamma_h[:,:,1].min():.3f}, {gamma_h[:,:,1].max():.3f}]")
print(f"Z range: [{gamma_h[:,:,2].min():.3f}, {gamma_h[:,:,2].max():.3f}]")

# Compute axis for comparison
phi_axis = np.linspace(0, 2*np.pi, 100)
axis_h = np.array(helical_axis(jnp.array(phi_axis)))
print(f"\nAxis X range: [{axis_h[:,0].min():.3f}, {axis_h[:,0].max():.3f}]")
print(f"Axis Y range: [{axis_h[:,1].min():.3f}, {axis_h[:,1].max():.3f}]")
print(f"Axis Z range: [{axis_h[:,2].min():.3f}, {axis_h[:,2].max():.3f}]")

# Check distance from axis
distances = []
for i in range(0, 30, 5):
    surf_pt = gamma_h[i, 0, :]
    axis_pt = axis_h[i*100//30, :]
    dist = np.linalg.norm(surf_pt - axis_pt)
    distances.append(dist)
    print(f"Distance at phi_idx={i}: {dist:.4f}")

print(f"Mean distance from axis: {np.mean(distances):.4f}")

print("\n" + "="*70)
print("FIGURE-EIGHT SURFACE TEST")
print("="*70)

# Create figure-eight surface
x1_coeffs_fig8 = np.zeros(13)
x1_coeffs_fig8[0] = 1.0
x1_coeffs_fig8[2] = 0.25
x2_coeffs_fig8 = np.zeros(13)

surface_fig8 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs_fig8,
    x2_coeffs=x2_coeffs_fig8,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_function=figure_eight_axis,
    ntheta=60,
    nphi=60
)

gamma_f = np.array(surface_fig8.gamma)
print(f"Shape: {gamma_f.shape}")
print(f"Has NaN: {np.any(np.isnan(gamma_f))}")
print(f"X range: [{gamma_f[:,:,0].min():.3f}, {gamma_f[:,:,0].max():.3f}]")
print(f"Y range: [{gamma_f[:,:,1].min():.3f}, {gamma_f[:,:,1].max():.3f}]")
print(f"Z range: [{gamma_f[:,:,2].min():.3f}, {gamma_f[:,:,2].max():.3f}]")

# Compute axis for comparison
axis_f = np.array(figure_eight_axis(jnp.array(phi_axis)))
print(f"\nAxis X range: [{axis_f[:,0].min():.3f}, {axis_f[:,0].max():.3f}]")
print(f"Axis Y range: [{axis_f[:,1].min():.3f}, {axis_f[:,1].max():.3f}]")
print(f"Axis Z range: [{axis_f[:,2].min():.3f}, {axis_f[:,2].max():.3f}]")

# Check distance from axis
distances_f = []
for i in range(0, 60, 10):
    surf_pt = gamma_f[i, 0, :]
    axis_pt = axis_f[i*100//60, :]
    dist = np.linalg.norm(surf_pt - axis_pt)
    distances_f.append(dist)
    print(f"Distance at phi_idx={i}: {dist:.4f}")

print(f"Mean distance from axis: {np.mean(distances_f):.4f}")

# Try manual plotting
print("\n" + "="*70)
print("CREATING TEST PLOT")
print("="*70)

fig = plt.figure(figsize=(16, 8))

# Helical surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(gamma_h[:, :, 0], gamma_h[:, :, 1], gamma_h[:, :, 2],
                 cmap='viridis', alpha=0.9, edgecolor='k', linewidth=0.3)
ax1.plot(axis_h[:, 0], axis_h[:, 1], axis_h[:, 2], 'r-', linewidth=3, label='Axis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Helical Surface (Manual Plot)')
ax1.legend()

# Figure-eight surface
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(gamma_f[:, :, 0], gamma_f[:, :, 1], gamma_f[:, :, 2],
                 cmap='plasma', alpha=0.9, edgecolor='k', linewidth=0.3)
ax2.plot(axis_f[:, 0], axis_f[:, 1], axis_f[:, 2], 'r-', linewidth=3, label='Axis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Figure-8 Surface (Manual Plot)')
ax2.legend()

plt.tight_layout()
plt.savefig('/home/exouser/ESSOS/examples/surface_visibility_test.png', dpi=150)
print("Saved: surface_visibility_test.png")
