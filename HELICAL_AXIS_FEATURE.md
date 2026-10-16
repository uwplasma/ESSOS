# Helical Axis Feature Implementation

## Overview
Added support for **helical axis** parameterization to `SurfaceGFrameFourier` class, extending beyond the existing circular and elliptical axis shapes.

## New Parameters

### `axis_pitch` (float, default=0.0)
- Controls the vertical rise of the helical axis per 2π toroidal turn
- Units: Same as the radial units (e.g., if major radius is 1.0, pitch is also in same units)
- Examples:
  - `axis_pitch=0.0`: Standard circular torus (flat axis)
  - `axis_pitch=0.2`: Gradual helical rise
  - `axis_pitch=0.5`: Steep helix

### `axis_twist_rate` (float, default=0.0)
- Controls rotation of the generalized Frenet frame around the axis
- Units: Radians per toroidal turn (2π radians)
- The frame rotates as you traverse around the axis
- Examples:
  - `axis_twist_rate=0.0`: No frame rotation (standard)
  - `axis_twist_rate=0.5`: Frame rotates π/2 radians per period
  - `axis_twist_rate=1.0`: Frame rotates 2π radians per period

## Mathematical Implementation

### Helical Axis Curve
```
r_axis(φ) = [R₀·cos(φ), R₀·sin(φ), pitch·φ/(2π)]
```

### Frame Twist
The Frenet frame (n₁, n₂) is rotated by angle = `axis_twist_rate × φ` in their plane:
```
n₁_twisted = cos(twist_angle)·n₁ + sin(twist_angle)·n₂
n₂_twisted = -sin(twist_angle)·n₁ + cos(twist_angle)·n₂
```

### Surface Position
```
r(θ,φ) = r_axis(φ) + X₁(θ,φ)·n₁(φ) + X₂(θ,φ)·n₂(φ)
```
with the twist-adjusted frame.

## Usage Example

```python
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

# Create coefficients
x1_coeffs = jnp.array([0.1, 0.05, -0.02])
x2_coeffs = jnp.array([0.08, 0.03, -0.01])

# Create surface with helical axis
surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=2,
    mpol=1,
    ntor=1,
    axis_shape='helical',
    axis_major_radius=1.0,
    axis_pitch=0.25,        # Moderate helix
    axis_twist_rate=0.5     # Frame rotates
)

# Compute surface
gamma = surface.gamma  # Shape: (nphi, ntheta, 3)
volume = surface.volume
area = surface.area
```

## Physical Applications

### 1. Toroidal Field Coil Effects
- Stellarators with rotated TF coils naturally have twisted axes
- Pitch represents vertical offset between periods

### 2. Magnetic Island Modeling
- Helical axes useful for representing islands
- Twist creates the characteristic S-shape topology

### 3. Device Imperfections
- Model alignment errors and coil misalignments
- Represent ripple effects from discrete coils

### 4. Optimization
- Generate wider space of candidates for design exploration
- Better capture of non-planar geometries

## Implementation Details

### Modified Methods

1. **`__init__`**: Added `axis_pitch` and `axis_twist_rate` parameters
2. **`_compute_axis_curve`**: Extended to handle 'helical' shape
3. **`_compute_frenet_frame`**: Added twist rotation logic

### Compatibility
- ✅ JAX transformations (grad, vmap, jit)
- ✅ Automatic differentiation through entire surface
- ✅ Pytree registration (when added)
- ✅ Existing code (backward compatible)

## Files Modified

- `essos/surfaces.py`: Core implementation
- `examples/surface_gframe_example.py`: New Example 8 demonstrating usage

## Testing

A comprehensive test suite is available in `test_helical_axis.py`:

```bash
python /home/exouser/ESSOS/test_helical_axis.py
```

Tests include:
1. Basic helical axis creation
2. Helical axis with frame twist
3. Comparison of axis shapes (circle, ellipse, helical)
4. Gradient computation through helical surfaces

## Future Extensions

Possible enhancements:
1. Custom axis functions (user-defined v(φ))
2. Elliptical helix (different radii in different directions)
3. Multiple helix parameters (nested helices)
4. Analytical mode resolution for simple helical cases

## Notes

- The axis is **not** a DOF by design (avoids parameter non-uniqueness)
- Only X₁ and X₂ coefficients are DOFs
- Pitch and twist rate are geometry parameters (not optimized by default)
- Can be made DOF-like through wrapper layer if needed

## References

The helical geometry is part of the generalized coordinate frame approach used in GVEC:

> Hindenlang, F., Plunk, G. G., & Maj, O. (2025).  
> "Computing MHD equilibria of stellarators with a flexible coordinate frame"  
> *Plasma Physics and Controlled Fusion*, 67(4), 045002

GVEC: https://github.com/gvec-group/gvec
