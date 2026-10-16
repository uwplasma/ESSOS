"""
Plot |B| on a surface using SIMSOPT with coils optimized in ESSOS
"""
import numpy as np
import matplotlib.pyplot as plt
from simsopt import load
from simsopt.mhd import Vmec
from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart

# ========== LOAD COILS ==========
print("Loading optimized coils...")
bs = load('optimized_coils_simsopt.json')
coils = bs.coils
print(f"Loaded {len(coils)} coils")

# ========== CREATE SURFACE FROM VMEC ==========
print("Loading VMEC surface...")
vmec_file = '../input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'
from simsopt.mhd import Vmec

vmec = Vmec(vmec_file)

# Get boundary surface and set resolution
surf = vmec.boundary
nphi = 60
ntheta = 60
surf = surf.copy(range="half period", nphi=nphi, ntheta=ntheta)

print(f"Surface: nphi={surf.quadpoints_phi.size}, ntheta={surf.quadpoints_theta.size}")

# ========== COMPUTE |B| ON SURFACE ==========
print("Computing magnetic field on surface...")

# Get surface coordinates
gamma = surf.gamma()  # Shape: (nphi, ntheta, 3)
nphi_actual = gamma.shape[0]
ntheta_actual = gamma.shape[1]

# Reshape to flat list of points and compute B
points = gamma.reshape(-1, 3)
bs.set_points(points)
B_flat = bs.B()
B_field = B_flat.reshape(nphi_actual, ntheta_actual, 3)

# Compute |B|
modB = np.linalg.norm(B_field, axis=2)

print(f"|B| statistics:")
print(f"  min: {np.min(modB):.6e}")
print(f"  max: {np.max(modB):.6e}")
print(f"  mean: {np.mean(modB):.6e}")
print(f"  ripple: {(np.max(modB)-np.min(modB))/np.mean(modB)*100:.2f}%")

# ========== COMPUTE B.n/|B| ==========
# Get unit normal
unitnormal = surf.unitnormal()  # Shape: (nphi, ntheta, 3)

# Compute B.n / |B|
bdotn_over_b = np.sum(B_field * unitnormal, axis=2) / modB

print(f"\nB.n/|B| statistics:")
print(f"  max |B.n/|B||: {np.max(np.abs(bdotn_over_b)):.6e}")
print(f"  mean |B.n/|B||: {np.mean(np.abs(bdotn_over_b)):.6e}")

# ========== CREATE COORDINATE GRIDS ==========
# SIMSOPT uses normalized quadpoints [0, 1]
phi_grid = surf.quadpoints_phi * 2 * np.pi
theta_grid = surf.quadpoints_theta * 2 * np.pi

# Create 2D meshgrids for plotting
phi_2d, theta_2d = np.meshgrid(phi_grid, theta_grid, indexing='ij')

# ========== PLOTTING ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: |B| contour
ax = axes[0, 0]
cs1 = ax.contour(phi_2d, theta_2d, modB, levels=30, cmap='viridis')
plt.colorbar(cs1, ax=ax, label='|B| [T]')
ax.set_xlabel('φ (toroidal angle)')
ax.set_ylabel('θ (poloidal angle)')
ax.set_title(f'|B| on surface (SIMSOPT)\nripple={(np.max(modB)-np.min(modB))/np.mean(modB)*100:.2f}%')

# Plot 2: B.n/|B| contour
ax = axes[0, 1]
cs2 = ax.contour(phi_2d, theta_2d, bdotn_over_b, levels=30, cmap='RdBu_r', 
                   vmin=-np.max(np.abs(bdotn_over_b)), 
                   vmax=np.max(np.abs(bdotn_over_b)))
plt.colorbar(cs2, ax=ax, label='B·n/|B|')
ax.set_xlabel('φ (toroidal angle)')
ax.set_ylabel('θ (poloidal angle)')
ax.set_title(f'Normal field error\nmax |B·n/|B||={np.max(np.abs(bdotn_over_b)):.3e}')

# Plot 3: Line profiles of |B|
ax = axes[1, 0]
# Plot at several toroidal angles
for i_phi in [0, nphi_actual//4, nphi_actual//2, 3*nphi_actual//4]:
    phi_val = phi_grid[i_phi]
    ax.plot(theta_grid, modB[i_phi, :], 
            label=f'φ={phi_val:.3f}', alpha=0.7)
ax.set_xlabel('θ (poloidal angle)')
ax.set_ylabel('|B| [T]')
ax.legend()
ax.grid()
ax.set_title('|B| line profiles at different φ')

# Plot 4: 3D coils + surface
ax = axes[1, 1]
ax.remove()
ax = fig.add_subplot(224, projection='3d')

# Plot surface
X = gamma[:, :, 0]
Y = gamma[:, :, 1]
Z = gamma[:, :, 2]
ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', edgecolor='none')

# Plot coils
for coil in coils:
    curve = coil.curve
    curve_gamma = curve.gamma()
    ax.plot(curve_gamma[:, 0], curve_gamma[:, 1], curve_gamma[:, 2], 
            'r-', linewidth=2, alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Coils + Surface')
ax.set_box_aspect([1, 1, 0.5])

plt.tight_layout()
plt.savefig('modB_simsopt.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved plot: modB_simsopt.png")
plt.savefig('sismot_modB_simsopt.pdf', dpi=300, bbox_inches='tight')

# ========== OPTIONAL: SAVE DATA ==========
np.savez('modB_data.npz',
         modB=modB,
         bdotn_over_b=bdotn_over_b,
         phi_grid=phi_grid,
         theta_grid=theta_grid,
         gamma=gamma)
print("✓ Saved data: modB_data.npz")
