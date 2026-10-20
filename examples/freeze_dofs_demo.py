"""
Demo script showing how to freeze certain DOFs in a SurfaceRZFourier object.

This demonstrates three methods for freezing DOFs:
1. By mode numbers (m, n) - most intuitive for stellarator work
2. By direct indices - for fine-grained control
3. By boolean mask - for conditional freezing
"""

import jax.numpy as jnp
from essos.surfaces import SurfaceRZFourier
import os

# Create or load a surface
input_file = os.path.join('input_files', 'input.rotating_ellipse')

# If you have a VMEC input file, use:
# surface = SurfaceRZFourier.from_input_file(input_file, ntheta=30, nphi=30)

# Or create from scratch with dummy data
nfp = 2
mpol = 3
ntor = 2

# Create dummy Fourier coefficients
n_modes = (mpol + 1) * (2 * ntor + 1) - ntor
rc = jnp.ones(n_modes) * 10.0  # Major radius ~ 10
zs = jnp.ones(n_modes) * 0.1   # Some vertical asymmetry

surface = SurfaceRZFourier(rc, zs, nfp, mpol, ntor, ntheta=30, nphi=30)

print("=" * 70)
print("DEMO: Freezing Surface DOFs")
print("=" * 70)

# Show initial state
print(f"\nInitial surface DOFs:")
print(f"  Total DOFs: {len(surface.all_dofs)}")
print(f"  RC coefficients: {surface.rc.size}")
print(f"  ZS coefficients: {surface.zs.size}")
print(f"  Mode numbers (m, n):")
for i in range(min(5, len(surface.xm))):
    m = int(surface.xm[i])
    n = int(surface.xn[i] / surface.nfp)
    print(f"    Mode {i}: (m={m}, n={n})")

print("\n" + "=" * 70)
print("METHOD 1: Freeze by mode numbers (m, n)")
print("=" * 70)

# Freeze rc[m=0, n=0] (major radius) and zs[m=1, n=0]
surface.freeze_dofs_by_modes(
    rc_modes=[(0, 0)],
    zs_modes=[(1, 0)]
)

print(f"\nFrozen rc[m=0, n=0] and zs[m=1, n=0]")
print(f"  Free DOFs: {len(surface.dofs)} (was {len(surface.all_dofs)})")
print(f"  Frozen DOFs: {jnp.sum(surface.frozen_dofs_mask)}")

# Get the specific mode index
idx_rc_00 = surface.get_mode_index(0, 0, 'rc')
idx_zs_10 = surface.get_mode_index(1, 0, 'zs')
print(f"  rc[m=0, n=0] at index: {idx_rc_00}")
print(f"  zs[m=1, n=0] at index: {idx_zs_10}")

# Demonstrate that frozen DOFs don't change during optimization
old_free_dofs = surface.dofs.copy()
new_free_dofs = old_free_dofs * 1.1  # Simulate optimizer changing free DOFs

surface.dofs = new_free_dofs  # Update with new values

print(f"\nAfter 'optimization' (multiplied free DOFs by 1.1):")
print(f"  Frozen rc[m=0,n=0] unchanged: {surface.all_dofs[idx_rc_00]:.6f}")
print(f"  Frozen zs[m=1,n=0] unchanged: {surface.all_dofs[idx_zs_10]:.6f}")
print(f"  Other DOFs changed: ✓")

# Unfreeze for next demo
surface.unfreeze_all_dofs()
print("\n  All DOFs unfrozen")

print("\n" + "=" * 70)
print("METHOD 2: Freeze by direct indices")
print("=" * 70)

# Freeze first 3 DOFs by index
surface.freeze_dofs_by_indices([0, 1, 2])

print(f"\nFrozen DOFs at indices [0, 1, 2]")
print(f"  Free DOFs: {len(surface.dofs)}")
print(f"  Frozen DOFs: {jnp.sum(surface.frozen_dofs_mask)}")

surface.unfreeze_all_dofs()

print("\n" + "=" * 70)
print("METHOD 3: Freeze by conditional mask")
print("=" * 70)

# Freeze all DOFs with small magnitude
threshold = 0.5
mask = jnp.abs(surface.all_dofs) < threshold

surface.freeze_dofs_by_mask(mask)

print(f"\nFrozen all DOFs with |value| < {threshold}")
print(f"  Free DOFs: {len(surface.dofs)}")
print(f"  Frozen DOFs: {jnp.sum(surface.frozen_dofs_mask)}")

surface.unfreeze_all_dofs()

print("\n" + "=" * 70)
print("USAGE IN OPTIMIZATION")
print("=" * 70)

# Example optimization setup with frozen DOFs
surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])

print(f"\n# In your optimization code:")
print(f"# Freeze major radius rc[m=0, n=0]")
print(f"surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])")
print(f"\n# Get initial DOFs for optimizer (only free DOFs)")
print(f"initial_dofs = surface.dofs  # Shape: {surface.dofs.shape}")
print(f"\n# During optimization, set new free DOFs")
print(f"# surface.dofs = optimized_free_dofs")
print(f"# Frozen DOFs automatically stay constant!")
print(f"\n# Access all DOFs (including frozen) for evaluation")
print(f"# all_dofs = surface.all_dofs")

print("\n" + "=" * 70)
print("ADVANCED: Update frozen values during optimization")
print("=" * 70)

surface.unfreeze_all_dofs()
surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])
old_frozen_value = surface.all_dofs[surface.get_mode_index(0, 0, 'rc')]

print(f"\nInitial frozen rc[m=0,n=0] = {old_frozen_value:.6f}")

# Manually change rc (e.g., after a coarse optimization step)
surface._rc = surface._rc.at[0].set(11.0)
surface.reset_cache()

# Update the frozen values to new state
surface.update_frozen_values()
new_frozen_value = surface.all_dofs[surface.get_mode_index(0, 0, 'rc')]

print(f"After manual change and update_frozen_values()")
print(f"  New frozen rc[m=0,n=0] = {new_frozen_value:.6f}")

print("\n" + "=" * 70)
print("EXAMPLE: Freeze multiple modes at once")
print("=" * 70)

surface.unfreeze_all_dofs()

# Freeze several modes that define the basic shape
surface.freeze_dofs_by_modes(
    rc_modes=[(0, 0), (1, 0), (0, 1)],  # Basic circular shape
    zs_modes=[(1, 0), (1, 1)]  # Basic up-down asymmetry
)

print(f"\nFrozen multiple modes to preserve basic shape:")
print(f"  RC modes: (0,0), (1,0), (0,1)")
print(f"  ZS modes: (1,0), (1,1)")
print(f"  Free DOFs for optimization: {len(surface.dofs)}/{len(surface.all_dofs)}")
print(f"  ({100 * len(surface.dofs)/len(surface.all_dofs):.1f}% of DOFs free)")

print("\n" + "=" * 70)
print("Tips for efficient JAX optimization:")
print("=" * 70)
print("""
1. Freeze DOFs BEFORE creating your loss function:
   - Call surface.freeze_dofs_by_modes(...) early
   - Then pass surface to your loss function
   - JAX will only optimize free DOFs

2. The dofs property automatically handles masking:
   - surface.dofs returns only FREE DOFs (for optimizer)
   - surface.all_dofs returns ALL DOFs (for evaluation)
   
3. Frozen DOFs are preserved during updates:
   - Setting surface.dofs = new_values automatically inserts frozen values
   - No manual masking needed in your code!
   
4. Works seamlessly with JAX JIT compilation:
   - All operations are pure JAX operations
   - No Python control flow during optimization
   - Efficient gradient computation

5. Use get_mode_index() to find specific modes:
   - idx = surface.get_mode_index(m, n, 'rc')
   - Helpful for debugging or manual inspection
""")

print("=" * 70)
print("Demo complete!")
print("=" * 70)
