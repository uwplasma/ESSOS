# Custom Axis Functions Feature

## Overview
Extended `SurfaceGFrameFourier` to support **custom axis functions**, enabling users to define arbitrary axis curves for surface parametrization.

## New Parameter

### `axis_function` (callable, optional, default=None)
- Accepts a JAX-compatible function with signature: `func(phi) -> array([x, y, z])`
- Only used when `axis_shape='custom'`
- Can be dynamically changed via the `axis_function` property
- Enables modeling of arbitrary geometric shapes

## Mathematical Definition

For custom axis functions:
```
r(θ,φ) = r_axis(φ) + X₁(θ,φ)·n₁(φ) + X₂(θ,φ)·n₂(φ)
```

where `r_axis(φ) = axis_function(φ)` is user-supplied.

## Usage Examples

### 1. Figure-Eight Axis

```python
def figure_eight_axis(phi):
    R = 1.0
    r = 0.3
    x = (R + r * jnp.cos(2 * phi)) * jnp.cos(phi)
    y = (R + r * jnp.cos(2 * phi)) * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=2, mpol=1, ntor=1,
    axis_shape='custom',
    axis_function=figure_eight_axis
)
```

### 2. Rose Curve (N-petal)

```python
def rose_curve_axis(phi):
    n_petals = 3
    a = 1.0
    r = a * jnp.abs(jnp.cos(n_petals * phi))
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

surface = SurfaceGFrameFourier(
    ...,
    axis_shape='custom',
    axis_function=rose_curve_axis
)
```

### 3. Parametric Spiral

```python
def spiral_axis(phi):
    a = 0.15
    radius = a * phi / jnp.pi  # Increases with phi
    x = radius * jnp.cos(phi)
    y = radius * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

surface = SurfaceGFrameFourier(..., axis_shape='custom', axis_function=spiral_axis)
```

### 4. Dynamic Function Assignment

```python
# Create with initial function
surface = SurfaceGFrameFourier(
    ...,
    axis_shape='custom',
    axis_function=circle_func
)

# Change function dynamically
surface.axis_function = figure_eight_axis
new_volume = surface.volume  # Automatically recomputes

# Change again
surface.axis_function = rose_curve_axis
```

## Requirements for Custom Functions

### JAX Compatibility
Functions **must** use JAX operations for autodifferentiation:

✅ **Correct:**
```python
def my_axis(phi):
    x = 1.0 * jnp.cos(phi)
    y = 1.0 * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)
```

❌ **Incorrect (uses NumPy):**
```python
def my_axis(phi):
    x = 1.0 * np.cos(phi)  # BAD: use jnp instead of np
    y = 1.0 * np.sin(phi)
    z = np.zeros_like(phi)
    return np.stack([x, y, z], axis=-1)
```

### Function Signature
- **Input:** `phi` (can be scalar or array)
- **Output:** JAX array of shape `(3,)` or `(n, 3)` for array input
  - Represents `[x, y, z]` coordinates
  - Must handle both single values and batched arrays

### Supported Operations
Use JAX primitives:
- Trigonometric: `jnp.sin()`, `jnp.cos()`, `jnp.tan()`
- Arithmetic: `+`, `-`, `*`, `/`
- Array operations: `jnp.stack()`, `jnp.concatenate()`
- Logical: `jnp.where()` (for piecewise functions)
- Power: `**`, `jnp.sqrt()`
- And all standard JAX functions

## Advanced Patterns

### 1. Closures with Parameters

```python
def create_custom_axis(amplitude, frequency):
    """Factory function to create custom axes with parameters."""
    def axis(phi):
        x = (1.0 + amplitude * jnp.cos(frequency * phi)) * jnp.cos(phi)
        y = (1.0 + amplitude * jnp.cos(frequency * phi)) * jnp.sin(phi)
        z = jnp.zeros_like(phi)
        return jnp.stack([x, y, z], axis=-1)
    return axis

# Create axes with different parameters
axis_1 = create_custom_axis(0.3, 2)
axis_2 = create_custom_axis(0.5, 3)

surf1 = SurfaceGFrameFourier(..., axis_shape='custom', axis_function=axis_1)
surf2 = SurfaceGFrameFourier(..., axis_shape='custom', axis_function=axis_2)
```

### 2. Piecewise Functions

```python
def piecewise_axis(phi):
    """Define different shapes in different regions."""
    # Circle in region 1, rose curve in region 2
    is_region1 = phi < jnp.pi
    
    # Region 1: circle
    x1 = jnp.cos(phi)
    y1 = jnp.sin(phi)
    
    # Region 2: rose curve
    r2 = jnp.cos(2 * phi)
    x2 = r2 * jnp.cos(phi)
    y2 = r2 * jnp.sin(phi)
    
    x = jnp.where(is_region1, x1, x2)
    y = jnp.where(is_region1, y1, y2)
    z = jnp.zeros_like(phi)
    
    return jnp.stack([x, y, z], axis=-1)
```

### 3. Parametric Curves

```python
def lissajous_axis(phi):
    """Lissajous curve as axis."""
    A, B = 1.0, 0.8
    a, b = 3, 2
    delta = jnp.pi / 4
    
    x = A * jnp.sin(a * phi + delta)
    y = B * jnp.sin(b * phi)
    z = jnp.zeros_like(phi)
    
    return jnp.stack([x, y, z], axis=-1)
```

## Optimization with Custom Axes

Custom axis functions work seamlessly with JAX transformations:

```python
import jax

def volume_objective(coeffs, axis_fn):
    x1 = coeffs[:n_coeff]
    x2 = coeffs[n_coeff:]
    
    surf = SurfaceGFrameFourier(
        x1_coeffs=x1,
        x2_coeffs=x2,
        ...,
        axis_shape='custom',
        axis_function=axis_fn
    )
    return surf.volume

# Compute gradients
grad_fn = jax.grad(volume_objective)
gradients = grad_fn(coeffs, my_axis_function)

# Vectorize over multiple initial conditions
batch_volumes = jax.vmap(volume_objective, in_axes=(0, None))
```

## Available Predefined Custom Axes

See `test_custom_axis.py` for 8 complete examples:

1. **Figure-Eight Axis** - 2-petaled rose with XY modulation
2. **Rose Curve (3-petal)** - Classic rose curve r = cos(3φ)
3. **Lemniscate** - Figure-8 in 2D with optional z-component
4. **Archimedean Spiral** - Radius increases linearly with φ
5. **Star-shaped** - N-pointed star with smooth radius variation
6. **3D Helix** - Alternative to axis_shape='helical'
7. **Custom Helix** - Demonstrating closure pattern
8. **Dynamic Assignment** - Changing functions at runtime

## Performance Considerations

### JIT Compilation
- Custom functions must be traceable by JAX
- Avoid Python conditionals; use `jnp.where()` instead
- First call will compile; subsequent calls use cached version

### Vectorization
- Functions automatically vectorize over arrays
- Can be used with `vmap` for batch operations

### Differentiation
- Fully differentiable through JAX autodiff
- Gradient computation included in cost model
- No additional overhead vs. built-in axis shapes

## Files Modified

- **essos/surfaces.py**: Added `axis_function` parameter and support
- **examples/surface_gframe_example.py**: Added Example 9 (custom axes)
- **test_custom_axis.py**: 8 detailed custom axis examples

## Testing

```bash
# Run comprehensive custom axis tests
python /home/exouser/ESSOS/test_custom_axis.py

# Run main example file (includes custom axis example)
python /home/exouser/ESSOS/examples/surface_gframe_example.py
```

## Design Rationale

1. **Flexibility**: Users can define arbitrary axis geometries
2. **JAX Integration**: Full autodiff support for optimization
3. **Dynamic Assignment**: Axis function can be changed without recreating surface
4. **Backward Compatible**: Existing code continues to work
5. **Factory Pattern**: Closures enable parameterized axis families

## Limitations & Future Work

### Current Limitations
- Cannot be used with JIT directly (axis function must be pure)
- No analytical mode resolution for custom curves
- Numerical derivatives required for complex shapes

### Future Enhancements
- Support for analytically-specified derivatives in custom functions
- Built-in library of parametric curves
- Automatic differentiation of axis parameters
- Plotting utilities specific to custom axes

## References

Parametric curves used in examples:
- **Rose curves**: r(θ) = a·cos(nθ)
- **Lemniscate**: Classic figure-8 algebraic curve
- **Spiral**: Archimedean spiral r = aθ
- **Lissajous**: Parametric curves x = A·sin(at+δ), y = B·sin(bt)

See GVEC documentation for context on flexible coordinate frames:
> Hindenlang, F., Plunk, G. G., & Maj, O. (2025).  
> "Computing MHD equilibria of stellarators with a flexible coordinate frame"  
> *Plasma Physics and Controlled Fusion*, 67(4), 045002
