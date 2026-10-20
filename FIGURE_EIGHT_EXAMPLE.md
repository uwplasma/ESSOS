# Figure-Eight Axis Example: Surface and Coils

## Overview

This example demonstrates how to:
1. **Define a custom figure-eight (lemniscate) axis** in 3D space
2. **Create a G-frame surface** wrapped around this axis using `SurfaceGFrameFourier`
3. **Create coils** positioned around the same axis using `CreateCoilsAroundAxis`
4. **Visualize both** together with comprehensive matplotlib plots

## Axis Definition

The **figure-eight** (lemniscate) axis is defined parametrically as:

$$r(\phi) = \frac{R \sqrt{2} \cos(\phi)}{1 + \sin^2(\phi)}$$

$$\vec{r}_{axis}(\phi) = \left[ r(\phi) \cos(\phi), r(\phi) \sin(\phi), 0 \right]$$

This creates a symmetric, smooth curve in the xy-plane that resembles the figure "8" or the infinity symbol ∞.

### Code
```python
def figure_eight_axis(phi):
    R = 1.0
    denom = 1.0 + jnp.sin(phi)**2
    r = R * jnp.sqrt(2.0) * jnp.cos(phi) / denom
    
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    
    return jnp.array([x, y, z])
```

## Surface Construction

The `SurfaceGFrameFourier` surface is built by:

1. **Specifying the axis**: Pass `axis_shape='custom'` and `axis_function=figure_eight_axis`
2. **Defining poloidal modes**: Poloidal (θ) Fourier expansion creates the cross-sectional shape
3. **Specifying toroidal modes**: Toroidal (φ) Fourier expansion allows shape variation around the axis
4. **Setting coefficients**: X1 and X2 represent the surface position in the Frenet frame

### Code
```python
surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=1,           # field periods
    mpol=2,          # poloidal order
    ntor=2,          # toroidal order
    ntheta=40,       # poloidal discretization
    nphi=50,         # toroidal discretization
    axis_shape='custom',
    axis_function=figure_eight_axis,
    axis_major_radius=1.0
)
```

## Coil Positioning

Coils are created using `CreateCoilsAroundAxis`:

1. **Positioning**: Coils are equally spaced around the figure-eight axis
2. **Geometry**: Each coil is circular, perpendicular to the axis (in the Frenet frame)
3. **Fourier representation**: Discrete coil curves are fitted to Fourier series for compact representation

### Code
```python
coils = CreateCoilsAroundAxis(
    n_coils=8,              # 8 coils around the axis
    order=4,                # Fourier order
    coil_radius=0.12,       # Individual coil radius
    n_samples=200,          # Sampling points per coil
    axis_function=figure_eight_axis,
    axis_shape='custom',
    nfp=1
)
```

## Visualization Components

The main visualization generates **6 subplots**:

### 1. Surface Only (Subplot 1)
Shows the G-frame surface wrapped around the figure-eight axis with viridis colormap.

### 2. Coils Only (Subplot 2)
Displays the 8 coils in dark blue, showing their circular geometry around the axis.

### 3. Combined View (Subplot 3)
Surface (semi-transparent) with coils overlaid, showing how they relate geometrically.

### 4. Axis Curve & Coil Positions (Subplot 4)
The figure-eight axis (black curve) with red markers showing where each coil is centered.

### 5. Top-Down View (Subplot 5)
Looking straight down (from +Z), showing the characteristic figure-eight shape and coil distribution.

### 6. Side View (Subplot 6)
Side perspective (from X direction), showing z-variations if any.

## Detailed Analysis Plots

A second visualization provides **6 analysis subplots**:

### 1. Individual Coils (Color-Coded)
Each coil shown in a different color so you can distinguish them.

### 2. Coil Length Distribution
Bar chart showing the length of each coil. For a symmetric figure-eight, all coils should have similar lengths.

### 3. Surface Cross-Sections
Poloidal cross-sections at different toroidal angles, overlaid with the axis curve.

### 4. Coil Positions (Top View)
Stars marking where each coil is centered, numbered for reference.

### 5. Coil Centers (3D)
3D scatter showing coil center positions relative to the dense axis curve.

### 6. Fourier Spectrum
Bar chart showing the magnitude of Fourier coefficients for the first coil, illustrating frequency content.

## Key Properties

After running, the example prints:

```
SURFACE PROPERTIES:
  Axis shape: custom (figure-eight/lemniscate)
  Poloidal modes (mpol): 2
  Toroidal modes (ntor): 2
  Discretization: 40 θ × 50 φ

COILS PROPERTIES:
  Number of coils: 8
  Fourier order: 4
  Total DOFs: ...

COIL GEOMETRY:
  Coil lengths (m): [values]
  Mean length: ...
```

## Running the Example

```bash
cd /home/exouser/ESSOS
python examples/figure_eight_surface_and_coils_example.py
```

This generates:
- `figure_eight_surface_and_coils.png` - Main 6-panel visualization
- `figure_eight_detailed_analysis.png` - Detailed analysis visualization

## Customization

### Change Coil Radius
```python
coils = CreateCoilsAroundAxis(..., coil_radius=0.15)  # larger coils
```

### Add More Coils
```python
coils = CreateCoilsAroundAxis(n_coils=12, ...)  # 12 instead of 8
```

### Modify Surface Cross-Section
Change X1 and X2 coefficients to create elliptical, flattened, or other poloidal shapes.

### Different Axis
Replace `figure_eight_axis` with any custom function:
```python
def custom_axis(phi):
    r = 1.0 + 0.3*jnp.sin(3*phi)  # 3-lobed shape
    return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), 0.2*phi/(2*jnp.pi)])
```

## Physical Interpretation

This example represents a **hypothetical plasma confinement system**:
- The **surface** could represent a plasma boundary
- The **coils** could represent external windings to create the magnetic field
- The unusual figure-eight geometry demonstrates that the system works for arbitrary, non-trivial axis shapes

This is useful for:
✓ Stellarator optimization studies  
✓ Testing custom geometries  
✓ Validating surface-coil relationships  
✓ Generating publication-quality visualizations  

## Technical Notes

1. **JAX Compatibility**: Both functions use JAX operations and work with autodiff
2. **Numerical Derivatives**: Frenet frame computation uses numerical derivatives (ε = 1e-8)
3. **Symmetry**: Currently set to `nfp=1, stellsym=False` for simplicity; can be extended
4. **Performance**: 8 coils with Fourier order 4 fits in ~seconds; scales to larger systems

## Related Examples

- `surface_gframe_example.py` - More surface variations
- `coils_custom_axis_example.py` - More coil geometries
- `test_custom_axis.py` - Additional axis definitions
