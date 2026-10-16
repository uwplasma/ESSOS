# Freezing Surface DOFs in ESSOS

## Overview

The `SurfaceRZFourier` class now supports freezing (fixing) certain degrees of freedom (DOFs) during optimization. This is useful for:

- **Preserving physical constraints**: Keep major radius fixed while optimizing shape details
- **Staged optimization**: Optimize coarse features first, then fine features
- **Reducing dimensionality**: Fewer free parameters → faster convergence
- **Incorporating domain knowledge**: Freeze well-optimized modes, focus on problematic ones

## Quick Start

```python
from essos.surfaces import SurfaceRZFourier

# Load or create a surface
surface = SurfaceRZFourier.from_input_file('input.vmec', ntheta=30, nphi=30)

# Freeze major radius rc[m=0, n=0]
surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])

# Now surface.dofs returns only FREE DOFs for optimization
initial_dofs = surface.dofs  # Smaller array, only free DOFs

# During optimization:
optimized_dofs = my_optimizer(initial_dofs, ...)
surface.dofs = optimized_dofs  # Frozen DOFs stay constant automatically!

# Access all DOFs (including frozen) for evaluation
all_dofs = surface.all_dofs
```

## Three Methods to Freeze DOFs

### Method 1: By Mode Numbers (Recommended)

Most intuitive for stellarator physics - specify which (m, n) modes to freeze:

```python
# Freeze major radius and basic shape
surface.freeze_dofs_by_modes(
    rc_modes=[(0, 0), (1, 0), (0, 1)],  # R00, R10, R01
    zs_modes=[(1, 0), (1, 1)]            # Z10, Z11
)
```

**Use cases:**
- Freeze `rc[m=0, n=0]` to maintain plasma size
- Freeze `rc[m=1, n=0]` and `zs[m=1, n=0]` to preserve basic cross-section
- Freeze high-m,n modes for coarse optimization stage

### Method 2: By Direct Indices

Fine-grained control using DOF array indices:

```python
# Freeze first 3 DOFs
surface.freeze_dofs_by_indices([0, 1, 2])

# Or find specific mode index first
idx = surface.get_mode_index(0, 0, 'rc')
surface.freeze_dofs_by_indices([idx])
```

**Use cases:**
- When you know exact DOF ordering
- For debugging or testing
- When mode numbers aren't available

### Method 3: By Boolean Mask

Conditional freezing based on DOF values:

```python
# Freeze all small coefficients
threshold = 1e-6
mask = jnp.abs(surface.all_dofs) < threshold
surface.freeze_dofs_by_mask(mask)

# Or create custom mask
mask = jnp.zeros(len(surface.all_dofs), dtype=bool)
mask = mask.at[my_indices].set(True)
surface.freeze_dofs_by_mask(mask)
```

**Use cases:**
- Freeze negligible modes automatically
- Complex conditional logic
- Adaptive freezing strategies

## Helper Methods

```python
# Find DOF index for specific mode
idx = surface.get_mode_index(m=0, n=0, coefficient='rc')
# Returns: integer index or None if mode doesn't exist

# Unfreeze all DOFs
surface.unfreeze_all_dofs()

# Update frozen values (keep same DOFs frozen, update their values)
surface.update_frozen_values()

# Check which DOFs are frozen
mask = surface.frozen_dofs_mask  # Boolean array

# Access DOFs
free_dofs = surface.dofs         # Only free (optimizable) DOFs
all_dofs = surface.all_dofs      # All DOFs including frozen
free_dofs2 = surface.free_dofs   # Alias for surface.dofs
```

## Integration with Optimizers

### With scipy.optimize

```python
from scipy.optimize import minimize
import jax.numpy as jnp
from jax import jit, grad

# Setup
surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])

@jit
def loss_fn(dofs_free, surface_template):
    # Create surface and update with free DOFs
    surf = SurfaceRZFourier(
        surface_template.rc, surface_template.zs,
        surface_template.nfp, surface_template.mpol, surface_template.ntor,
        ntheta=surface_template.ntheta, nphi=surface_template.nphi
    )
    # Copy frozen state
    surf._frozen_dofs_mask = surface_template._frozen_dofs_mask
    surf._frozen_dofs_values = surface_template._frozen_dofs_values
    surf.dofs = dofs_free  # Frozen values inserted automatically
    
    # Compute loss (e.g., squared flux)
    return compute_squared_flux(surf)

# Optimize
result = minimize(
    lambda x: loss_fn(x, surface),
    x0=surface.dofs,  # Only free DOFs
    jac=jit(grad(loss_fn)),
    method='L-BFGS-B'
)

# Update surface with optimized DOFs
surface.dofs = result.x
```

### With Augmented Lagrangian (ALM)

```python
import essos.augmented_lagrangian as alm

# Freeze DOFs before setting up optimization
surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])

# Create constraints using custom_loss with frozen surface
constraint = alm.eq(lambda dofs: my_constraint_fn(dofs, surface))

# Create composite constraint
C_Total = alm.combine(constraint)

# Set dependencies
C_Total.dependencies = {'surface': surface}

# Optimize with ALM - automatically uses only free DOFs
# (see optimize_surface_with_frozen_dofs.py for full example)
```

### With JAX Optimizers (Optax)

```python
import optax

surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(surface.dofs)

# Training loop
for step in range(n_steps):
    loss_val, grads = jax.value_and_grad(loss_fn)(surface.dofs, surface)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_dofs = optax.apply_updates(surface.dofs, updates)
    surface.dofs = new_dofs  # Frozen DOFs preserved automatically
```

## Staged Optimization Strategy

Progressively unfreeze DOFs for better convergence:

```python
# Stage 1: Coarse shape (only low-m,n modes free)
surface.freeze_dofs_by_modes(
    rc_modes=[(m, n) for m in range(2, mpol+1) for n in range(-ntor, ntor+1)],
    zs_modes=[(m, n) for m in range(2, mpol+1) for n in range(-ntor, ntor+1)]
)
optimize_stage_1(surface)

# Stage 2: Add medium modes
surface.freeze_dofs_by_modes(
    rc_modes=[(m, n) for m in range(3, mpol+1) for n in range(-ntor, ntor+1)],
    zs_modes=[(m, n) for m in range(3, mpol+1) for n in range(-ntor, ntor+1)]
)
optimize_stage_2(surface)

# Stage 3: All modes free
surface.unfreeze_all_dofs()
optimize_stage_3(surface)
```

## Advanced Usage

### Updating Frozen Values Mid-Optimization

```python
# Initial optimization with frozen major radius
surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])
optimize(surface)

# Adjust major radius based on results
new_major_radius = 11.0
idx = surface.get_mode_index(0, 0, 'rc')
all_dofs = surface.all_dofs.at[idx].set(new_major_radius)
surface.unfreeze_all_dofs()
surface.dofs = all_dofs

# Re-freeze at new value and continue optimization
surface.freeze_dofs_by_modes(rc_modes=[(0, 0)])
surface.update_frozen_values()  # Update to new major radius
continue_optimization(surface)
```

### Custom Freezing Patterns

```python
# Freeze based on mode magnitude
mode_magnitudes = jnp.sqrt(surface.rc**2 + surface.zs**2)
threshold = jnp.percentile(mode_magnitudes, 25)  # Bottom quartile
mask = mode_magnitudes < threshold

# Create full DOF mask (rc + zs)
full_mask = jnp.concatenate([mask, mask])
surface.freeze_dofs_by_mask(full_mask)

# Or freeze alternating modes for testing
mask = jnp.zeros(len(surface.all_dofs), dtype=bool)
mask = mask.at[::2].set(True)  # Every other DOF
surface.freeze_dofs_by_mask(mask)
```

## Performance Considerations

1. **JIT Compilation**: All DOF operations are pure JAX - fully JIT-compatible
   ```python
   @jit
   def loss_fn(dofs):
       surface.dofs = dofs  # JIT-safe
       return compute_loss(surface)
   ```

2. **Memory Efficiency**: Only free DOFs passed to optimizer
   - Original: 100 DOFs × 100 parameters = 10,000 elements in Jacobian
   - Frozen: 80 free DOFs × 100 parameters = 8,000 elements (20% reduction)

3. **Gradient Computation**: JAX automatically handles masking
   ```python
   # Gradients only computed for free DOFs
   grad_fn = jax.grad(loss_fn)  # Efficient!
   ```

## Common Patterns

### Preserve Stellarator Symmetry

```python
# Ensure n=0 modes stay fixed (toroidal symmetry)
rc_n0_modes = [(m, 0) for m in range(mpol+1)]
zs_n0_modes = [(m, 0) for m in range(1, mpol+1)]
surface.freeze_dofs_by_modes(rc_modes=rc_n0_modes, zs_modes=zs_n0_modes)
```

### Fix Equilibrium Properties

```python
# Keep modes that define pressure balance
surface.freeze_dofs_by_modes(
    rc_modes=[(0, 0), (1, 0)],  # R0, iota
    zs_modes=[(1, 0)]            # Vertical shift
)
```

### Optimize Only Surface Details

```python
# Freeze everything except high-m modes
low_m_modes = [(m, n) for m in range(0, 2) for n in range(-ntor, ntor+1)]
surface.freeze_dofs_by_modes(rc_modes=low_m_modes, zs_modes=low_m_modes)
```

## Troubleshooting

**Q: Why do I get "index out of bounds" when freezing modes?**

A: The mode (m, n) doesn't exist in your surface. Check available modes:
```python
for i, (m, n) in enumerate(zip(surface.xm, surface.xn / surface.nfp)):
    print(f"Mode {i}: m={int(m)}, n={int(n)}")
```

**Q: How do I check if DOFs are actually frozen?**

A: Compare before and after optimization:
```python
idx = surface.get_mode_index(0, 0, 'rc')
before = surface.all_dofs[idx]
optimize(surface)
after = surface.all_dofs[idx]
print(f"Changed: {not jnp.allclose(before, after)}")  # Should be False
```

**Q: Can I freeze DOFs for multiple surfaces independently?**

A: Yes! Each surface instance maintains its own frozen state:
```python
surface1.freeze_dofs_by_modes(rc_modes=[(0, 0)])
surface2.freeze_dofs_by_modes(rc_modes=[(1, 0)])  # Different!
```

## Examples

See the following example scripts:
- `examples/freeze_dofs_demo.py` - Basic demonstration of all methods
- `examples/optimize_surface_with_frozen_dofs.py` - Realistic optimization example
- `examples/coil_optimization/optimize_coils_surface_augmented_lagrangian_one_surface.py` - Integration with ALM

## API Reference

### Properties

- `surface.dofs` - Get/set free DOFs only (for optimization)
- `surface.all_dofs` - Get all DOFs including frozen
- `surface.free_dofs` - Alias for `surface.dofs`
- `surface.frozen_dofs_mask` - Boolean mask of frozen DOFs

### Methods

- `freeze_dofs_by_modes(rc_modes=None, zs_modes=None)` - Freeze by (m, n) mode numbers
- `freeze_dofs_by_indices(indices)` - Freeze by DOF indices
- `freeze_dofs_by_mask(mask)` - Freeze by boolean mask
- `unfreeze_all_dofs()` - Remove all freezing
- `update_frozen_values()` - Update frozen values to current state
- `get_mode_index(m, n, coefficient='rc')` - Find DOF index for mode

## Implementation Details

The freezing mechanism uses:
1. **Boolean mask** (`_frozen_dofs_mask`): Tracks which DOFs are frozen
2. **Frozen values** (`_frozen_dofs_values`): Stores frozen DOF values
3. **Automatic reconstruction**: Setting `dofs` automatically inserts frozen values

All operations are pure JAX for efficiency:
```python
# Internally uses:
all_dofs = all_dofs.at[frozen_mask].set(frozen_values)
all_dofs = all_dofs.at[~frozen_mask].set(free_values)
```

This ensures JIT compilation and automatic differentiation work seamlessly.
