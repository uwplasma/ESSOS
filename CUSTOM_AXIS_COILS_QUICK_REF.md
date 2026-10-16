# Quick Reference: CreateCoilsAroundAxis

## TL;DR

Create coils around **any axis shape** (circle, ellipse, helix, custom curve) using the same unified approach.

## Axis Types

| Type | Code | Formula | Parameters |
|------|------|---------|------------|
| **Circle** | `axis_shape='circle'` | $(R\cos φ, R\sin φ, 0)$ | `axis_major_radius=R` |
| **Ellipse** | `axis_shape='ellipse'` | $(R\cos φ, aR\sin φ, 0)$ | `axis_major_radius=R` |
| **Helix** | `axis_shape='helical'` | $(R\cos φ, R\sin φ, p·φ/2π)$ | `axis_pitch=p` |
| **Custom** | `axis_shape='custom'` | `func(φ) → [x,y,z]` | `axis_function=func` |

## Basic Patterns

### Circular (default)
```python
coils = CreateCoilsAroundAxis(n_coils=4, order=4, coil_radius=0.1)
```

### Helical
```python
coils = CreateCoilsAroundAxis(
    n_coils=4, order=4, coil_radius=0.1,
    axis_shape='helical',
    axis_pitch=0.3  # Rise per revolution
)
```

### Helical + Twist
```python
coils = CreateCoilsAroundAxis(
    n_coils=4, order=4, coil_radius=0.1,
    axis_shape='helical',
    axis_pitch=0.3,
    axis_twist_rate=0.2  # Frame twist per revolution
)
```

### Custom (Figure-Eight)
```python
def fig8(phi):
    R = 1.0
    r = R * jnp.sqrt(2) * jnp.cos(phi) / (1 + jnp.sin(phi)**2)
    return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), 0])

coils = CreateCoilsAroundAxis(
    n_coils=6, order=4, coil_radius=0.08,
    axis_function=fig8, axis_shape='custom'
)
```

## Common Custom Axes

### Rose Curve (k petals)
```python
def rose(phi, k=3):
    r = 1.1 + jnp.cos(k*phi)
    return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), 0])
```

### Spiral
```python
def spiral(phi):
    r = 0.5 + 0.3*(phi/(2*jnp.pi))
    z = 0.2*phi/(2*jnp.pi)
    return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), z])
```

### Lissajous
```python
def lissajous(phi):
    x = jnp.sin(phi)
    y = jnp.sin(2*phi)
    z = 0.5*jnp.sin(3*phi)
    return jnp.array([x, y, z])
```

## Key Parameters

```python
CreateCoilsAroundAxis(
    # Core
    n_coils=4,           # How many coils
    order=4,             # Fourier order
    coil_radius=0.1,     # Size of each coil
    
    # Axis
    axis_shape='circle',          # 'circle', 'ellipse', 'helical', 'custom'
    axis_major_radius=1.0,        # For circle/ellipse/helix
    axis_pitch=0.0,               # For helix: rise per 2π
    axis_twist_rate=0.0,          # Frame twist: radians per 2π
    axis_function=None,           # For custom: func(phi) -> [x,y,z]
    
    # Fitting
    n_samples=200,        # Discretization for Fourier fitting
    n_segments=100,       # Output discretization
)
```

## Output

Returns `Curves` object with:
- `coils.gamma` → Discretized positions `[n_coils, n_segments, 3]`
- `coils.order` → Fourier order
- `coils.dofs` → Fourier coefficients
- `coils.length` → Coil lengths

## Usage

```python
from essos.coils import CreateCoilsAroundAxis

# Create
coils = CreateCoilsAroundAxis(n_coils=4, order=4, coil_radius=0.1, 
                               axis_shape='helical', axis_pitch=0.3)

# Inspect
print(coils.length)        # Coil lengths
print(coils.order)         # Fourier order
print(coils.gamma.shape)   # (4, 100, 3)

# Visualize
coils.plot()

# Export
coils.to_vtk('coils.vtu')
coils.to_json('coils.json')
```

## JAX Integration

Works with autodiff:
```python
def cost(pitch):
    coils = CreateCoilsAroundAxis(..., axis_pitch=pitch)
    return jnp.mean(coils.length)

grad_fn = jax.grad(cost)
gradient = grad_fn(0.3)  # dL/d(pitch)
```

## File

Implementation: [essos/coils.py](essos/coils.py) (search: `CreateCoilsAroundAxis`)
