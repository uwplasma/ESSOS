# Custom Axis Coil Generation Guide

## Overview

`CreateCoilsAroundAxis` is a new function in the ESSOS coils module that generalizes coil generation beyond simple circular axes. It applies the same flexible axis parametrization approach used in `SurfaceGFrameFourier` to create coils around:

- **Circular axes** (major radius, simple torus)
- **Elliptical axes** (elongated major axis)
- **Helical axes** (spiral with adjustable pitch and twist)
- **Custom axes** (arbitrary user-defined functions)

## Motivation: From Curved Surfaces to Curved Axes for Coils

The original `CreateEquallySpacedCurves` function creates coils as small circles positioned around a single circular major axis:

```
r_coil(θ, φ) = r_axis(φ) + r_minor · (cos(θ)·n1 + sin(θ)·n2)
```

where:
- `r_axis(φ) = (R·cos(φ), R·sin(φ), 0)` is the circular axis
- `r_minor = 0.1` is the coil radius
- `θ` parametrizes each individual coil (0 to 2π)
- `φ` positions coils around the axis

The new `CreateCoilsAroundAxis` function extends this by allowing **any axis shape** while keeping the coil creation logic identical. This is conceptually the same as how `SurfaceGFrameFourier` uses different axis types for surfaces.

## API Reference

### Function Signature

```python
def CreateCoilsAroundAxis(
    n_coils: int,
    order: int,
    coil_radius: float,
    n_samples: int = 200,
    axis_major_radius: float = 1.0,
    axis_shape: str = 'circle',
    axis_pitch: float = 0.0,
    axis_twist_rate: float = 0.0,
    axis_function = None,
    n_segments: int = 100,
    nfp: int = 1,
    stellsym: bool = False,
    scaling_type: int = 2,
    scaling_factor: float = 0,
    scale_fixed: float = 1.0
) -> Curves
```

### Parameters

#### Core Coil Parameters
- **n_coils** (int): Number of coils to generate around the axis
- **order** (int): Fourier order for coil representation (2*order+1 coefficients per coordinate)
- **coil_radius** (float): Radius of each individual coil (minor radius)
- **n_samples** (int, default=200): Number of discretization points per coil for accurate Fourier fitting

#### Axis Definition
- **axis_shape** (str, 'circle'): Type of axis
  - `'circle'`: Circular axis with radius `axis_major_radius`
  - `'ellipse'`: Elliptical axis (aspect ratio hardcoded as 1.0 in current version)
  - `'helical'`: Helical spiral with pitch `axis_pitch`
  - `'custom'`: User-provided function via `axis_function`

#### Axis Parameters
- **axis_major_radius** (float, default=1.0): Major radius for circle/ellipse/helical axes
- **axis_pitch** (float, default=0.0): Vertical rise per 2π toroidal turn (helical axis only)
- **axis_twist_rate** (float, default=0.0): Frenet frame rotation rate (radians per 2π)
- **axis_function** (callable, default=None): Custom axis function
  - **Required if** `axis_shape == 'custom'`
  - **Signature**: `func(phi: float) -> jnp.ndarray[3]`
  - **Returns**: Position [x, y, z] on axis at toroidal angle `phi`
  - **JAX requirement**: Must use only JAX-compatible operations

#### Curve and Symmetry Parameters
- **n_segments** (int, default=100): Discretization points for Fourier fitting output
- **nfp** (int, default=1): Number of field periods (for symmetry)
- **stellsym** (bool, default=False): Apply stellarator symmetry
- **scaling_type** (int, default=2): Mode scaling type (1=L1, 2=L2, inf=L∞)
- **scaling_factor** (float, default=0): Exponential mode scaling factor
- **scale_fixed** (float, default=1.0): Fixed multiplier for all modes

### Returns

**Curves**: A `Curves` object with properties:
- `gamma`: Discretized coil positions `[n_curves, n_segments, 3]`
- `order`: Fourier order
- `dofs`: Fourier coefficients `[n_curves, 3, 2*order+1]`
- `length`: Mean length of each coil

## Implementation Details

### Architecture

The function implements the following steps:

1. **Define axis curve**: Create a parametric axis `r_axis(φ) → [x, y, z]`
2. **Compute Frenet frame**: At each coil position φ, compute orthonormal vectors (n1, n2) perpendicular to axis
3. **Sample coils**: For each coil position, create a circle `r_coil = r_axis + r_minor*(cos(θ)·n1 + sin(θ)·n2)`
4. **Fit to Fourier**: Convert discrete coil samples to Fourier coefficients using `fit_dofs_from_coils()`

### Frenet Frame Construction

For smooth axis curves, the Frenet frame is computed as:

$$n_1 = \frac{r_{axis}(φ) \text{ projected to xy-plane}}{|r_{axis}|_{xy}}$$

$$n_2 = \hat{t} × n_1$$

where $\hat{t}$ is the unit tangent to the axis.

With twist rate:

$$n_1^{twisted}(φ) = \cos(ω_{twist} \cdot φ) \cdot n_1 + \sin(ω_{twist} \cdot φ) \cdot n_2$$

$$n_2^{twisted}(φ) = -\sin(ω_{twist} \cdot φ) \cdot n_1 + \cos(ω_{twist} \cdot φ) \cdot n_2$$

This allows the coils to twist as they traverse the axis.

## Usage Examples

### Example 1: Circular Axis (Baseline)

```python
from essos.coils import CreateCoilsAroundAxis

# Equivalent to CreateEquallySpacedCurves(R=1.0, r=0.1, n_curves=4, order=4)
coils = CreateCoilsAroundAxis(
    n_coils=4,
    order=4,
    coil_radius=0.1,
    axis_major_radius=1.0,
    axis_shape='circle'
)

print(f"Coil lengths: {coils.length}")
coils.plot()
```

### Example 2: Helical Axis

Coils spiraling upward around a helical path:

```python
coils = CreateCoilsAroundAxis(
    n_coils=6,
    order=4,
    coil_radius=0.12,
    axis_major_radius=1.0,
    axis_shape='helical',
    axis_pitch=0.5  # Rise 0.5 units per 2π
)
```

### Example 3: Helical with Twist

Coils that both spiral upward AND twist:

```python
coils = CreateCoilsAroundAxis(
    n_coils=6,
    order=5,
    coil_radius=0.12,
    axis_shape='helical',
    axis_pitch=0.5,
    axis_twist_rate=0.3  # Frame rotates 0.3 rad per 2π
)
```

### Example 4: Figure-Eight Axis (Custom)

```python
def lemniscate_axis(phi):
    """Figure-eight (lemniscate) curve."""
    R = 1.0
    denom = 1.0 + jnp.sin(phi)**2
    r = R * jnp.sqrt(2.0) * jnp.cos(phi) / denom
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.array([x, y, z])

coils = CreateCoilsAroundAxis(
    n_coils=8,
    order=4,
    coil_radius=0.08,
    axis_function=lemniscate_axis,
    axis_shape='custom'
)
```

### Example 5: Rose Curve (Polar Flower)

```python
def rose_axis(phi):
    """3-petal rose curve: r(φ) = cos(3φ)."""
    r = 1.1 + jnp.cos(3 * phi)  # Offset to keep positive
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.array([x, y, z])

coils = CreateCoilsAroundAxis(
    n_coils=6,
    order=5,
    coil_radius=0.08,
    axis_function=rose_axis,
    axis_shape='custom'
)
```

### Example 6: Custom Scrolling Spiral

```python
def spiral_axis(phi):
    """Archimedean spiral."""
    a, b = 0.5, 0.5  # Base radius and spiral rate
    r = a + b * (phi / (2 * jnp.pi))
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = 0.3 * phi / (2 * jnp.pi)  # Slight vertical component
    return jnp.array([x, y, z])

coils = CreateCoilsAroundAxis(
    n_coils=5,
    order=4,
    coil_radius=0.08,
    axis_function=spiral_axis,
    axis_shape='custom'
)
```

## JAX Integration and Differentiation

`CreateCoilsAroundAxis` is fully compatible with JAX autodiff. You can compute gradients with respect to axis parameters:

```python
import jax

def total_coil_length(pitch_value):
    """Compute mean coil length as function of pitch."""
    coils = CreateCoilsAroundAxis(
        n_coils=4,
        order=3,
        coil_radius=0.1,
        axis_shape='helical',
        axis_pitch=pitch_value
    )
    return jnp.mean(coils.length)

# Compute gradient
grad_fn = jax.grad(total_coil_length)
grad_value = grad_fn(0.3)  # dL/d(pitch) at pitch=0.3

print(f"dL/d(pitch) = {grad_value}")
```

**Note**: Gradients through `axis_function` require that the function itself is JAX-compatible (uses `jnp` instead of `np`).

## Performance Considerations

### n_samples Parameter

The `n_samples` parameter (default=200) controls:
- **Accuracy**: Higher values ≈ better Fourier fitting for complex axes
- **Speed**: Scales linearly with fitting time

Recommendation:
- `n_samples = 100-150` for simple axes (circle, ellipse)
- `n_samples = 200+` for complex custom axes

### Order Parameter

Higher Fourier order captures finer geometry details but uses more DOFs:
- `order = 3-4` for smooth axes
- `order = 5+` for intricate shapes (rose curves, tight spirals)

## Comparison with SurfaceGFrameFourier

| Feature | SurfaceGFrameFourier | CreateCoilsAroundAxis |
|---------|---------------------|----------------------|
| **Axis types** | circle, ellipse, helical, custom | circle, ellipse, helical, custom |
| **Object** | Full surface (2D parametrization) | Circular coils (1D along axis) |
| **DOFs** | X1, X2 poloidal Fourier modes | Coil geometry Fourier coefficients |
| **Frame twist** | Yes (axis_twist_rate) | Yes (axis_twist_rate) |
| **Custom functions** | Yes (axis_function) | Yes (axis_function) |
| **JAX integration** | Full (vmap, grad, jit) | Full (vmap, grad, jit) |

## Common Patterns

### Circular Coils on Any Axis

The default pattern—create circles perpendicular to the axis—works universally:

```python
coils = CreateCoilsAroundAxis(
    n_coils=N,
    order=K,
    coil_radius=r,
    axis_shape=shape,          # 'circle', 'helical', or 'custom'
    axis_function=func         # For custom shapes
)
```

### Twisted Helical Lattice

Coils that twist and spiral simultaneously:

```python
coils = CreateCoilsAroundAxis(
    n_coils=8,
    order=5,
    coil_radius=0.1,
    axis_shape='helical',
    axis_pitch=0.4,            # Rise per revolution
    axis_twist_rate=0.2        # Frame twist per revolution
)
```

### Field-Aligned Coils

Custom axis matching stellarator geometry:

```python
def custom_stellarator_axis(phi):
    """Define axis matching your equilibrium shape."""
    # ... implement custom geometry
    return jnp.array([x, y, z])

coils = CreateCoilsAroundAxis(
    n_coils=N,
    order=K,
    coil_radius=r,
    axis_function=custom_stellarator_axis,
    axis_shape='custom'
)
```

## Limitations and Future Work

### Current Limitations

1. **Ellipse aspect ratio**: Hardcoded to 1.0; could be parameterized
2. **Frame computation**: Uses numerical derivatives (1e-8 step) for stability
3. **Edge cases**: Custom axes near z-axis may have numerical issues

### Potential Enhancements

1. Make ellipse aspect ratio a parameter
2. Add `compute_axis_curve` and `compute_frenet_frame` as public methods for custom analysis
3. Vectorized custom functions via vmap for batch axis generation
4. Integration with GVEC file I/O for geometry import/export

## Testing

Run the comprehensive example suite:

```bash
cd /home/exouser/ESSOS/examples
python coils_custom_axis_example.py
```

This generates 8 detailed examples with plotting and validates:
- Circular, elliptical, helical axes
- Helical twist behavior
- Figure-eight, rose, spiral custom axes
- Side-by-side comparison
- JAX gradient computation

## Summary

`CreateCoilsAroundAxis` brings the flexible axis parametrization of `SurfaceGFrameFourier` to coil generation. By supporting arbitrary axis shapes while maintaining simple circular coil geometry, it enables:

✓ Coil design for non-standard geometries  
✓ Stellarator and field-aligned applications  
✓ Parametric studies via JAX autodiff  
✓ Seamless integration with existing ESSOS infrastructure  

The implementation mirrors the proven approach used in surface representations, ensuring consistency and reliability across the codebase.
