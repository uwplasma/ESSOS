"""
SIMSOPT: Plot |B| on a VMEC surface in (theta, phi) coordinates
"""
import numpy as np
import matplotlib.pyplot as plt
from simsopt import load
from simsopt.geo import SurfaceRZFourier

print("="*70)
print("SIMSOPT: Computing |B| on VMEC surface")
print("="*70)

# ========== STEP 1: Load optimized coils ==========
print("\n1. Loading coils from SIMSOPT file...")
bs = load('optimized_coils_simsopt.json')
print(f"   ✓ Loaded BiotSavart field with {len(bs.coils)} coils")

# ========== STEP 2: Create surface from VMEC ==========
print("\n2. Creating VMEC surface...")
from simsopt.mhd import Vmec

# Read VMEC output file
vmec = Vmec('../input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc')

# Get the boundary surface (s=1) from VMEC
surf = vmec.boundary
print(f"   ✓ Surface loaded from VMEC")

# Optionally change resolution
nphi = 60
ntheta = 60
surf = surf.copy(range="half period", nphi=nphi, ntheta=ntheta)
print(f"   ✓ Surface resolution: {surf.quadpoints_phi.size} × {surf.quadpoints_theta.size} grid")

# ========== STEP 3: Get surface coordinates ==========
print("\n3. Getting surface coordinates...")
gamma = surf.gamma()  # Shape: (nphi, ntheta, 3)
nphi, ntheta = gamma.shape[0], gamma.shape[1]
print(f"   ✓ Gamma shape: {gamma.shape}")
print(f"   ✓ Grid points: {nphi} (phi) × {ntheta} (theta)")

# ========== STEP 4: Compute B at every point on surface ==========
print("\n4. Computing magnetic field on surface grid...")

# Reshape gamma to a flat list of points
points = gamma.reshape(-1, 3)  # Shape: (nphi*ntheta, 3)

# Set points in BiotSavart and compute B
bs.set_points(points)
B_flat = bs.B()  # Shape: (nphi*ntheta, 3)

# Reshape back to grid
B_field = B_flat.reshape(nphi, ntheta, 3)
modB = np.linalg.norm(B_field, axis=2)  # |B| at each point: shape (nphi, ntheta)

print(f"   ✓ |B| computed on {nphi}×{ntheta} grid")
print(f"   ✓ |B| range: [{modB.min():.3e}, {modB.max():.3e}] Tesla")

# ========== STEP 5: Create (phi, theta) coordinate grids for plotting ==========
print("\n5. Creating coordinate grids for plotting...")
# SIMSOPT stores normalized quadpoints [0, 1], convert to radians
phi_1d = surf.quadpoints_phi * 2 * np.pi      # Toroidal angle
theta_1d = surf.quadpoints_theta * 2 * np.pi  # Poloidal angle

# Create 2D meshgrids for contour plot
PHI, THETA = np.meshgrid(phi_1d, theta_1d, indexing='ij')
print(f"   ✓ Coordinate grids: PHI {PHI.shape}, THETA {THETA.shape}")

# ========== STEP 6: Create 2D contour plots ==========
print("\n6. Creating plots...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: |B| contour plot
ax = axes[0]
cs = ax.contourf(PHI, THETA, modB, levels=50, cmap='viridis')
plt.colorbar(cs, ax=ax, label='|B| [Tesla]')
ax.set_xlabel('φ (toroidal angle) [rad]', fontsize=12)
ax.set_ylabel('θ (poloidal angle) [rad]', fontsize=12)
ax.set_title(f'SIMSOPT: |B| on VMEC surface\nRipple: {(modB.max()-modB.min())/modB.mean()*100:.2f}%')

# Plot 2: Line cuts through the surface
ax = axes[1]
# Plot several toroidal cuts
for i_phi in [0, nphi//4, nphi//2, 3*nphi//4]:
    phi_val = phi_1d[i_phi]
    ax.plot(theta_1d, modB[i_phi, :], label=f'φ={phi_val:.2f}', linewidth=2)

ax.set_xlabel('θ (poloidal angle) [rad]', fontsize=12)
ax.set_ylabel('|B| [Tesla]', fontsize=12)
ax.set_title('Line cuts at different toroidal angles')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modB_simsopt.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: modB_simsopt.png")

# ========== STEP 7: Print statistics ==========
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"|B| Statistics:")
print(f"  Minimum: {modB.min():.6e} Tesla")
print(f"  Maximum: {modB.max():.6e} Tesla")
print(f"  Mean:    {modB.mean():.6e} Tesla")
print(f"  Ripple:  {(modB.max()-modB.min())/modB.mean()*100:.3f}%")
print("="*70)
