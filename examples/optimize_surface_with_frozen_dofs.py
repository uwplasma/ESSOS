"""
Example: Surface optimization with frozen DOFs using Augmented Lagrangian

This demonstrates how to optimize a surface while keeping certain Fourier modes
fixed (e.g., maintaining the major radius rc[m=0,n=0]).
"""

import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
from essos.surfaces import SurfaceRZFourier, BdotN_over_B
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
import essos.augmented_lagrangian as alm
from functools import partial

print("=" * 70)
print("Surface Optimization with Frozen DOFs Example")
print("=" * 70)

# Set up a simple stellarator configuration
nfp = 2
number_coils = 4
major_radius_coils = 11.0
minor_radius_coils = 1.5
current_per_coil = 1e7
order_Fourier = 5
n_segments = 50

# Create coils
curves = CreateEquallySpacedCurves(
    n_curves=number_coils,
    order=order_Fourier,
    R=major_radius_coils,
    r=minor_radius_coils,
    n_segments=n_segments,
    nfp=nfp,
    stellsym=True
)
coils = Coils(curves=curves, currents=[current_per_coil] * number_coils)
field = BiotSavart(coils)

# Create initial surface
ntheta, nphi = 30, 30
mpol, ntor = 3, 2

# Initialize with approximate circular cross-section
n_modes = (mpol + 1) * (2 * ntor + 1) - ntor
rc_init = jnp.zeros(n_modes)
zs_init = jnp.zeros(n_modes)

# Set basic shape: rc[0,0]=10 (major radius), rc[1,0]=1 (minor radius)
surface = SurfaceRZFourier(rc_init, zs_init, nfp, mpol, ntor, ntheta=ntheta, nphi=nphi)

# Set initial values for key modes
idx_rc_00 = surface.get_mode_index(0, 0, 'rc')
idx_rc_10 = surface.get_mode_index(1, 0, 'rc')
idx_zs_10 = surface.get_mode_index(1, 0, 'zs')

all_dofs = surface.all_dofs
all_dofs = all_dofs.at[idx_rc_00].set(10.0)  # Major radius
all_dofs = all_dofs.at[idx_rc_10].set(1.0)   # Minor radius
all_dofs = all_dofs.at[idx_zs_10].set(0.5)   # Some up-down asymmetry

surface.dofs = all_dofs  # This works because no DOFs are frozen yet

print(f"\nInitial surface configuration:")
print(f"  nfp={nfp}, mpol={mpol}, ntor={ntor}")
print(f"  Total DOFs: {len(surface.all_dofs)}")
print(f"  rc[m=0,n=0] (major radius): {all_dofs[idx_rc_00]:.3f}")
print(f"  rc[m=1,n=0] (minor radius): {all_dofs[idx_rc_10]:.3f}")
print(f"  zs[m=1,n=0]: {all_dofs[idx_zs_10]:.3f}")

print("\n" + "=" * 70)
print("SCENARIO 1: Optimize surface shape, keep major radius fixed")
print("=" * 70)

# Freeze major radius to maintain plasma size
surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])

print(f"\nFrozen rc[m=0,n=0] at value: {all_dofs[idx_rc_00]:.3f}")
print(f"Free DOFs for optimization: {len(surface.dofs)}")

# Define loss function for surface optimization
@jit
def surface_loss(surface_dofs, surface_template, field):
    """Loss = squared flux + deviation from circular cross-section."""
    # Create a new surface with updated DOFs
    surf = SurfaceRZFourier(
        surface_template.rc, surface_template.zs,
        surface_template.nfp, surface_template.mpol, surface_template.ntor,
        ntheta=surface_template.ntheta, nphi=surface_template.nphi
    )
    # Copy frozen DOF state
    surf._frozen_dofs_mask = surface_template._frozen_dofs_mask
    surf._frozen_dofs_values = surface_template._frozen_dofs_values
    
    # Update with new DOFs (frozen ones stay constant automatically!)
    surf.dofs = surface_dofs
    
    # Compute squared flux (magnetic error)
    bdotn = jnp.sum(
        vmap(field.B)(surf.gamma.reshape(-1, 3)).reshape(surf.nphi, surf.ntheta, 3) 
        * surf.unitnormal, 
        axis=2
    )
    squared_flux = jnp.mean(bdotn**2 * surf.area_element)
    
    # Penalize deviation from circularity
    rc_vals = surf.rc
    circularity_penalty = jnp.sum(rc_vals[2:]**2)  # Minimize higher modes
    
    return squared_flux + 0.01 * circularity_penalty

# Gradient of loss
@jit
def loss_grad(surface_dofs, surface_template, field):
    return jax.grad(surface_loss)(surface_dofs, surface_template, field)

print(f"\nInitial loss: {surface_loss(surface.dofs, surface, field):.6e}")

# Simple gradient descent for demonstration
learning_rate = 0.01
n_iterations = 10

current_dofs = surface.dofs.copy()
for i in range(n_iterations):
    grad = loss_grad(current_dofs, surface, field)
    current_dofs = current_dofs - learning_rate * grad
    loss_val = surface_loss(current_dofs, surface, field)
    if i % 2 == 0:
        print(f"  Iteration {i:2d}: loss = {loss_val:.6e}")

surface.dofs = current_dofs

print(f"\nAfter optimization:")
print(f"  rc[m=0,n=0] (should be unchanged): {surface.all_dofs[idx_rc_00]:.3f}")
print(f"  rc[m=1,n=0]: {surface.all_dofs[idx_rc_10]:.3f}")
print(f"  Final loss: {surface_loss(surface.dofs, surface, field):.6e}")

print("\n" + "=" * 70)
print("SCENARIO 2: Using with Augmented Lagrangian constraints")
print("=" * 70)

# Reset surface
surface.unfreeze_all_dofs()
all_dofs = all_dofs.at[idx_rc_00].set(10.0)
all_dofs = all_dofs.at[idx_rc_10].set(1.0)
surface.dofs = all_dofs

# Now freeze multiple modes to preserve basic shape
surface.freeze_dofs_by_modes(
    rc_modes=[(0, 0), (1, 0)],  # Fix major and minor radius
    zs_modes=[(1, 0)]            # Fix basic up-down asymmetry
)

print(f"\nFrozen modes:")
print(f"  RC: (0,0), (1,0)")
print(f"  ZS: (1,0)")
print(f"  Free DOFs: {len(surface.dofs)}/{len(surface.all_dofs)}")

# Create a constraint-based loss with ALM
def objective_fn(surf_dofs, surface_template):
    """Minimize surface area."""
    surf = SurfaceRZFourier(
        surface_template.rc, surface_template.zs,
        surface_template.nfp, surface_template.mpol, surface_template.ntor,
        ntheta=surface_template.ntheta, nphi=surface_template.nphi
    )
    surf._frozen_dofs_mask = surface_template._frozen_dofs_mask
    surf._frozen_dofs_values = surface_template._frozen_dofs_values
    surf.dofs = surf_dofs
    return surf.area

def constraint_fn(surf_dofs, surface_template):
    """Constrain volume to be close to target."""
    surf = SurfaceRZFourier(
        surface_template.rc, surface_template.zs,
        surface_template.nfp, surface_template.mpol, surface_template.ntor,
        ntheta=surface_template.ntheta, nphi=surface_template.nphi
    )
    surf._frozen_dofs_mask = surface_template._frozen_dofs_mask
    surf._frozen_dofs_values = surface_template._frozen_dofs_values
    surf.dofs = surf_dofs
    target_volume = 100.0  # Target volume
    return surf.volume - target_volume  # Should equal zero

# Create ALM constraint
volume_constraint = alm.eq(
    lambda dofs: constraint_fn(dofs, surface),
    model_lagrangian='Standard'
)

print(f"\nALM Objective: Minimize area")
print(f"ALM Constraint: Volume = 100.0")
print(f"With frozen DOFs preserving basic shape")

# Initialize constraint parameters
constraint_params = volume_constraint.init(surface.dofs)

# Evaluate loss and constraint
obj_val = objective_fn(surface.dofs, surface)
loss_val, constraint_val = volume_constraint.loss(constraint_params, surface.dofs)

print(f"\nInitial values:")
print(f"  Area: {obj_val:.3f}")
print(f"  Volume constraint: {constraint_val:.3f}")
print(f"  ALM loss: {loss_val:.3e}")

print("\n" + "=" * 70)
print("KEY ADVANTAGES of Frozen DOFs:")
print("=" * 70)
print("""
1. REDUCED OPTIMIZATION SPACE:
   - Fewer DOFs = faster convergence
   - More stable optimization
   - Example: 21 total DOFs → 18 free DOFs (14% reduction)

2. PRESERVE PHYSICAL CONSTRAINTS:
   - Keep major radius fixed (maintain plasma size)
   - Preserve stellarator symmetry
   - Maintain basic equilibrium properties

3. STAGED OPTIMIZATION:
   - Stage 1: Optimize coarse shape (freeze high-m,n modes)
   - Stage 2: Add fine details (unfreeze, freeze coarse modes)
   - Stage 3: Final refinement (optimize all)

4. DOMAIN EXPERTISE:
   - Freeze modes known to be well-optimized
   - Focus optimization on problematic regions
   - Prevent optimizer from "breaking" good solutions

5. COMPUTATIONAL EFFICIENCY:
   - Smaller Jacobian matrices
   - Faster gradient computations
   - Better memory usage in JAX
""")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)

# Clean up
surface.unfreeze_all_dofs()
