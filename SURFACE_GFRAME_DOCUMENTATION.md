# SurfaceGFrameFourier: G-frame Surface Representation in ESSOS

## Overview

The `SurfaceGFrameFourier` class implements a surface representation using the **generalized Frenet frame (G-frame)** approach, inspired by [GVEC](https://github.com/gvec-group/gvec) (Galerkin Variational Equilibrium Code). This enables more flexible surface parametrizations compared to the standard RZ Fourier representation.

## Mathematical Formulation

### Surface Parametrization

The surface is parametrized as:

$$\mathbf{r}(\theta, \phi) = \mathbf{r}_{\text{axis}}(\phi) + X_1(\theta, \phi) \mathbf{n}_1(\phi) + X_2(\theta, \phi) \mathbf{n}_2(\phi)$$

Where:
- $\mathbf{r}_{\text{axis}}(\phi)$ is the **central axis curve** (typically a circular or elliptical torus axis)
- $\mathbf{n}_1(\phi), \mathbf{n}_2(\phi)$ are orthonormal vectors forming the **generalized Frenet frame** at each point on the axis
- $X_1(\theta, \phi), X_2(\theta, \phi)$ are the perpendicular coordinates, **Fourier expanded in the poloidal angle $\theta$**

### Fourier Expansions

The perpendicular coordinates are expanded as:

$$X_1(\theta, \phi) = \sum_{m=0}^{m_{\text{pol}}} \sum_{n=-n_{\text{tor}}}^{n_{\text{tor}}} x1^{m,n} \cos(m\theta - n\phi)$$

$$X_2(\theta, \phi) = \sum_{m=0}^{m_{\text{pol}}} \sum_{n=-n_{\text{tor}}}^{n_{\text{tor}}} x2^{m,n} \sin(m\theta - n\phi)$$

Where:
- $x1^{m,n}, x2^{m,n}$ are the Fourier coefficients (degrees of freedom)
- $m$ is the poloidal mode number
- $n$ is the toroidal mode number

### Frenet Frame Construction

The generalized Frenet frame is constructed as follows:

1. **Tangent vector** at each point on the axis:
   $$\mathbf{t}(\phi) = \frac{d\mathbf{r}_{\text{axis}}}{d\phi} / \left| \frac{d\mathbf{r}_{\text{axis}}}{d\phi} \right|$$

2. **First normal** $\mathbf{n}_1$ (radial for circular axis):
   $$\mathbf{n}_1(\phi) = (\cos\phi, \sin\phi, 0)^T$$

3. **Second normal** $\mathbf{n}_2$ (perpendicular to both):
   $$\mathbf{n}_2(\phi) = \mathbf{t}(\phi) \times \mathbf{n}_1(\phi) / |\mathbf{t}(\phi) \times \mathbf{n}_1(\phi)|$$

This gives an orthonormal coordinate system at each point on the axis.

## Comparison: G-frame vs RZ Fourier

### RZ Fourier (Standard VMEC approach)

```python
# In RZFourier:
R(θ, φ) = Σ rc_i cos(m_i θ - n_i φ)
Z(θ, φ) = Σ zs_i sin(m_i θ - n_i φ)
X = R cos(φ), Y = R sin(φ), Z = Z
```

**Advantages:**
- Standard in plasma physics (VMEC, SPEC)
- Efficient for nearly circular surfaces
- Fewer DOFs for simple geometries

**Disadvantages:**
- Difficult for non-convex surfaces
- Problematic with magnetic islands
- Limited flexibility for complex shapes

### G-frame (GVEC approach)

```python
# In SurfaceGFrameFourier:
r = r_axis(φ) + X1(θ,φ) n1(φ) + X2(θ,φ) n2(φ)
X1, X2 are Fourier expanded in θ
```

**Advantages:**
- Flexible for complex-shaped surfaces
- Natural for near-axis expansions
- Better for magnetic island geometries
- Easier to handle topological changes

**Disadvantages:**
- More complex derivative computations
- Requires explicit axis specification
- Additional computational overhead for Frenet frame

## Usage

### Basic Example

```python
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

# Define mode numbers
nfp = 2  # 2 field periods
mpol = 1  # max poloidal mode
ntor = 1  # max toroidal mode

# Initialize coefficients
n_coeffs = (mpol + 1) * (2 * ntor + 1) - ntor
x1_coeffs = jnp.array([0.2, 0.05, 0.02])
x2_coeffs = jnp.array([0.0, 0.03, -0.02])

# Create surface
surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    nfp=nfp,
    mpol=mpol,
    ntor=ntor,
    ntheta=30,
    nphi=30,
    axis_major_radius=1.0,
    axis_shape='circle'
)

# Access surface properties
gamma = surface.gamma  # (nphi, ntheta, 3) - Cartesian coordinates
volume = surface.volume
area = surface.area
normal = surface.normal
```

### Advanced Usage

```python
# Modify coefficients
surface.x1_coeffs = new_x1_values
surface.x2_coeffs = new_x2_values

# Work with degrees of freedom
dofs = surface.dofs  # Combined X1 and X2 coefficients
surface.dofs = new_dofs

# Compute derivatives for optimization
# (JAX automatically supports grad, vmap, jit, etc.)

# Change grid resolution
surface.ntheta = 50
surface.nphi = 50
```

## Class Architecture

### Properties

**Static Properties:**
- `nfp`: Number of field periods
- `mpol`: Maximum poloidal mode
- `ntor`: Maximum toroidal mode
- `xm`, `xn`: Mode number arrays
- `scaling`: Mode scaling factors

**Dynamic Properties:**
- `x1_coeffs`, `x2_coeffs`: Fourier coefficients (setters trigger cache reset)
- `dofs`: Degrees of freedom (combined scaled coefficients)

**Computed Properties (cached):**
- `gamma`: Surface position in Cartesian coordinates (nphi, ntheta, 3)
- `gammadash_theta`: ∂r/∂θ derivative
- `gammadash_phi`: ∂r/∂φ derivative
- `normal`: Surface normal vector (unnormalized)
- `unitnormal`: Normalized surface normal
- `area_element`: Local area element
- `area`: Total surface area
- `volume`: Enclosed volume

### Methods

**Mesh and Grid:**
- `reset_cache()`: Clear computed properties
- `reset_mesh()`: Clear mesh-related cached values
- `_compute_meshgrid()`: Compute poloidal/toroidal grid

**Geometry:**
- `_compute_axis_curve(phi)`: Central axis curve
- `_compute_frenet_frame(phi)`: Frenet frame vectors
- `_compute_gamma()`: Main surface computation
- `_compute_properties()`: Normal and area element

**Utility:**
- `plot()`: 3D visualization
- `_tree_flatten()`/`_tree_unflatten()`: JAX pytree support

## Current Implementation Notes

### Limitations (for future enhancement)

1. **Axis curve**: Currently implements circular and elliptical axes. Can be extended with custom axis curves.

2. **Frenet frame derivatives**: Currently uses finite differences for some derivatives. Can be optimized with analytical expressions.

3. **Phi derivatives**: `gammadash_phi` is simplified. Full implementation requires computing derivatives of the axis curve and Frenet frame.

### Integration with JAX

The class fully supports JAX transformations:

```python
# Automatic differentiation
grad_surface = jax.grad(objective, argnums=0)(surface, ...)

# Just-in-time compilation
gvec_surface_jit = jax.jit(surface._compute_gamma)

# Vectorization
surfaces_batch = jax.vmap(lambda c1, c2: SurfaceGFrameFourier(c1, c2, ...))(x1_batch, x2_batch)
```

## References

### Key Papers

1. **Hindenlang, F., Plunk, G. G., & Maj, O. (2025)**
   - "Computing MHD equilibria of stellarators with a flexible coordinate frame"
   - Plasma Physics and Controlled Fusion, 67(4), 045002
   - https://doi.org/10.1088/1361-6587/adba11

2. **GVEC Documentation**
   - https://gvec.readthedocs.io/latest/
   - https://github.com/gvec-group/gvec

### Related Concepts

- **Frenet-Serret frame**: Classical differential geometry framework
- **Near-axis expansion**: Complementary expansion method for complex geometries
- **VMEC**: Variational Moments Equilibrium Code (standard stellarator solver)
- **SPEC**: Stepped Pressure Equilibrium Code (alternative equilibrium solver)

## Future Enhancements

1. **Custom axis curves**: Support user-defined axis parametrizations
2. **Analytical derivatives**: Compute phi-derivatives analytically instead of numerically
3. **Pressure and iota profiles**: Add radial structure for full equilibrium representation
4. **GVEC I/O**: Interface with GVEC input/output files
5. **Symmetry handling**: Explicit stellarator symmetry support
6. **Multi-surface**: Radial coordinate family implementation

## Example Scripts

See `/home/exouser/ESSOS/examples/surface_gframe_example.py` for comprehensive examples including:
- Creating a simple circular torus
- Modifying coefficients
- Computing surface properties
- Accessing DOFs
- Visualizing the surface

## Related Classes

- `SurfaceRZFourier`: Standard RZ Fourier representation (cylindrical coordinates)
- `SurfaceClassifier`: Signed distance function for surfaces
- `Vmec`: Reader for VMEC equilibrium files

## Contact & Contributions

For questions or contributions regarding the G-frame surface implementation:
- GitHub: https://github.com/uwplasma/ESSOS
- GVEC Project: https://gitlab.mpcdf.mpg.de/gvec-group/gvec
