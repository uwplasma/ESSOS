# Quick Reference: SurfaceGFrameFourier vs SurfaceRZFourier

## Quick Comparison Table

| Aspect | SurfaceRZFourier | SurfaceGFrameFourier |
|--------|-----------------|---------------------|
| **Coordinate System** | Cylindrical (R, Z, φ) | Cartesian (x, y, z) via Frenet frame |
| **Primary DOFs** | R coefficients (rc), Z coefficients (zs) | X1, X2 coefficients (perpendicular) |
| **Axis** | Implicit (R=0 is the axis) | Explicit central curve |
| **Fourier Basis** | R and Z separately | X1, X2 in local frame coordinates |
| **Geometry** | Expands global cylindrical coords | Expands deviations from axis |
| **Complexity** | Simple for nearly circular | Flexible for complex shapes |
| **Standard Use** | VMEC, SPEC equilibrium codes | GVEC equilibrium code |
| **Non-convex Support** | Limited | Good |
| **Magnetic Islands** | Difficult | Good |
| **Computational Cost** | Lower | Higher (Frenet frame computation) |
| **Derivatives** | Straightforward | More complex (involving axis curves) |

## Instantiation Comparison

### SurfaceRZFourier
```python
from essos.surfaces import SurfaceRZFourier
import jax.numpy as jnp

rc = jnp.array([1.0, 0.1, 0.05])  # R coefficients
zs = jnp.array([0.0, 0.08, -0.03])  # Z coefficients

surface_rz = SurfaceRZFourier(
    rc=rc,
    zs=zs,
    nfp=2,
    mpol=1,
    ntor=1,
    ntheta=30,
    nphi=30
)
```

### SurfaceGFrameFourier
```python
from essos.surfaces import SurfaceGFrameFourier
import jax.numpy as jnp

x1 = jnp.array([0.2, 0.05, 0.02])  # X1 coefficients (perpendicular)
x2 = jnp.array([0.0, 0.03, -0.02])  # X2 coefficients (perpendicular)

surface_gf = SurfaceGFrameFourier(
    x1_coeffs=x1,
    x2_coeffs=x2,
    nfp=2,
    mpol=1,
    ntor=1,
    ntheta=30,
    nphi=30,
    axis_major_radius=1.0,
    axis_shape='circle'
)
```

## Property Access Comparison

```python
# Both support similar property access:

# Cartesian surface position
gamma_rz = surface_rz.gamma  # (nphi, ntheta, 3)
gamma_gf = surface_gf.gamma  # (nphi, ntheta, 3)

# Surface normal
normal_rz = surface_rz.normal
normal_gf = surface_gf.normal

# Geometric properties
volume_rz = surface_rz.volume
volume_gf = surface_gf.volume

area_rz = surface_rz.area
area_gf = surface_gf.area

# Degrees of freedom access
dofs_rz = surface_rz.dofs  # rc and zs combined
dofs_gf = surface_gf.dofs  # x1 and x2 combined
```

## Mathematical Formulation Comparison

### SurfaceRZFourier Formulation
$$R(\theta, \phi) = \sum_i \text{rc}_i \cos(m_i \theta - n_i \phi)$$
$$Z(\theta, \phi) = \sum_i \text{zs}_i \sin(m_i \theta - n_i \phi)$$

**Cartesian transformation:**
$$\begin{align}
X &= R(\theta, \phi) \cos(\phi)\\
Y &= R(\theta, \phi) \sin(\phi)\\
Z &= Z(\theta, \phi)
\end{align}$$

### SurfaceGFrameFourier Formulation
$$\mathbf{r}(\theta, \phi) = \mathbf{r}_{\text{axis}}(\phi) + X_1(\theta, \phi) \mathbf{n}_1(\phi) + X_2(\theta, \phi) \mathbf{n}_2(\phi)$$

**Where:**
$$X_1(\theta, \phi) = \sum_i \text{x1}_i \cos(m_i \theta - n_i \phi)$$
$$X_2(\theta, \phi) = \sum_i \text{x2}_i \sin(m_i \theta - n_i \phi)$$

## Mode Structure

Both classes use the same mode numbering scheme:

```python
# Given mpol and ntor, the mode array indices are:
xm = jnp.repeat(jnp.arange(mpol + 1), 2 * ntor + 1)[ntor:]
xn = nfp * jnp.tile(jnp.arange(-ntor, ntor + 1), mpol + 1)[ntor:]

# Example: mpol=1, ntor=1, nfp=2
# Mode 0: (m=0, n=0)
# Mode 1: (m=0, n=±2)
# Mode 2: (m=1, n=-2)
# Mode 3: (m=1, n=0)
# Mode 4: (m=1, n=2)
```

## When to Use Each

### Use **SurfaceRZFourier** when:
- ✓ Working with standard VMEC/SPEC equilibria
- ✓ Surface is nearly circular/convex
- ✓ You need fast computation
- ✓ Interfacing with existing stellarator codes
- ✓ Simple torus geometry

### Use **SurfaceGFrameFourier** when:
- ✓ Surface has complex shape
- ✓ Magnetic islands present
- ✓ Non-convex geometry
- ✓ Using GVEC or implementing GVEC compatibility
- ✓ Flexible coordinate system needed
- ✓ Implementing near-axis expansions
- ✓ Need local frame-based parametrization

## Practical Examples

### Example 1: Standard Torus

Both can represent a standard torus, but with different adjustments:

```python
# RZFourier: Major radius ≈ R, minor radius ≈ r
surface_rz = SurfaceRZFourier(
    rc=jnp.array([1.0, 0.3]),  # Major radius, first deviation
    zs=jnp.array([0.0, 0.1]),  # Vertical asymmetry
    nfp=1, mpol=1, ntor=0
)

# GFrameFourier: Axis at major radius, then add minor radius as X1
major_R = 1.0
minor_r = 0.3
surface_gf = SurfaceGFrameFourier(
    x1_coeffs=jnp.array([minor_r, 0.02]),  # Minor radius + triangularity
    x2_coeffs=jnp.array([0.0, 0.02]),      # Vertical component
    nfp=1, mpol=1, ntor=0,
    axis_major_radius=major_R
)
```

### Example 2: Non-convex Surface

RZFourier struggles with non-convex surfaces, but G-frame handles them:

```python
# G-frame naturally supports complex shapes
surface_gf = SurfaceGFrameFourier(
    x1_coeffs=jnp.array([0.3, 0.15, 0.05]),  # Multiple harmonics
    x2_coeffs=jnp.array([0.0, 0.12, -0.08]),
    nfp=2, mpol=2, ntor=2,
    axis_major_radius=1.0,
    axis_shape='circle'
)
# The basis n1, n2 local frame naturally accommodates 
# non-convex deviations from the axis
```

## Code Migration

### Converting from RZFourier to GFrameFourier

The transformation isn't trivial, but you can:

1. Extract the RZ surface at the midplane
2. Fit an axis curve
3. Compute the Frenet frame
4. Project the surface onto local coordinates (X1, X2)
5. Perform Fourier analysis in the new basis

**Note:** This is a complex operation and would require careful interpolation.

## Performance Considerations

```python
# Computational complexity comparison:

# RZFourier:
# - Surface computation: O(mpol × ntor × ntheta × nphi)
# - Derivatives: Direct from Fourier formulation

# GFrameFourier:
# - Axis curve: Extra evaluation
# - Frenet frame: Extra orthonormalization
# - Surface: O(mpol × ntor × ntheta × nphi) + overhead
# - Derivatives: More complex, potentially slower

# Trade-off: Slightly slower but more flexible
```

## JAX Integration

Both classes support JAX transformations identically:

```python
import jax
import jax.numpy as jnp

# Automatic differentiation works for both
@jax.jit
def objective(surface):
    return surface.volume - surface.area

# Gradient with respect to DOFs
grad_obj = jax.grad(lambda dofs: objective(
    SurfaceGFrameFourier(..., x1_coeffs=dofs[:n], x2_coeffs=dofs[n:])
))

# Vectorization over multiple surfaces
surfaces = jax.vmap(
    lambda x1, x2: SurfaceGFrameFourier(x1, x2, ...),
    in_axes=(0, 0)
)(x1_batch, x2_batch)
```

## Common Pitfalls

### RZFourier
- ❌ Assuming R > 0 always (can cause issues with reflections)
- ❌ Not accounting for toroidal periodicity (2π/nfp)
- ❌ Forgetting that Z is sine-like while R is cosine-like

### GFrameFourier
- ❌ Not setting `axis_major_radius` appropriately
- ❌ Assuming axis is always circular (can customize)
- ❌ Forgetting that X1, X2 are local to the Frenet frame
- ❌ Not resetting cache after modifying axis parameters

## Debugging Tips

### Verify Mode Numbers
```python
# Both classes should have matching xm, xn arrays:
assert jnp.allclose(surface_rz.xm, surface_gf.xm)
assert jnp.allclose(surface_rz.xn, surface_gf.xn)
```

### Check Surface Smoothness
```python
# Plot to visualize surface quality
surface.plot()

# Check derivatives for smoothness
gamma = surface.gamma
normal = surface.normal
unitnormal = surface.unitnormal  # Should be normalized
```

### Verify Geometry
```python
# Check volume and area are reasonable
print(f"Volume: {surface.volume}")
print(f"Area: {surface.area}")
print(f"Volume/Area ratio (should be ~size): {surface.volume / surface.area}")
```

## Further Reading

- [SURFACE_GFRAME_DOCUMENTATION.md](./SURFACE_GFRAME_DOCUMENTATION.md): Detailed technical documentation
- [surface_gframe_example.py](./examples/surface_gframe_example.py): Complete working examples
- GVEC: https://github.com/gvec-group/gvec
- VMEC: https://princetonuniversity.github.io/STELLOPT/VMEC
