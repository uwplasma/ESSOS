# Developer's Guide: Extending SurfaceGFrameFourier

## Overview

This guide explains how to extend and enhance the `SurfaceGFrameFourier` class with additional features.

## Current Architecture

```
SurfaceGFrameFourier
├── Initialization & Properties
│   ├── Coefficients (x1, x2)
│   ├── Mode numbers (mpol, ntor, nfp)
│   ├── Grid parameters (ntheta, nphi)
│   └── Axis parameters (major_radius, shape)
├── Axis Curve System
│   ├── _compute_axis_curve(phi)
│   ├── _compute_frenet_frame(phi)
│   └── [Currently: Circle/Ellipse]
├── Surface Computation
│   ├── _compute_meshgrid()
│   ├── _compute_gamma()
│   ├── _compute_properties()
│   └── [Cached properties]
├── Derivatives
│   ├── Fourier derivatives (analytical)
│   ├── Axis curve derivatives (finite diff)
│   └── [Can be optimized]
└── JAX Integration
    ├── pytree support
    ├── JIT compatibility
    └── Autodiff support
```

## Enhancement 1: Custom Axis Curves

### Goal
Allow users to specify arbitrary axis curve parametrizations beyond circle/ellipse.

### Implementation Strategy

**Option A: Registry Pattern (Recommended)**

```python
# Add to SurfaceGFrameFourier class

_AXIS_CURVE_REGISTRY = {}

@classmethod
def register_axis_curve(cls, name, func):
    """Register a custom axis curve function.
    
    Parameters:
    -----------
    name : str
        Name of the axis curve (e.g., 'circle', 'ellipse', 'helical')
    func : callable
        Function signature: func(phi, params) -> (x, y, z)
    """
    cls._AXIS_CURVE_REGISTRY[name] = func

def _compute_axis_curve(self, phi):
    """Compute the central axis curve."""
    if self._axis_shape in self._AXIS_CURVE_REGISTRY:
        func = self._AXIS_CURVE_REGISTRY[self._axis_shape]
        return func(phi, self._axis_params)
    else:
        # Fallback to built-in shapes
        return self._compute_axis_curve_builtin(phi)
```

**Option B: Callable Class Attribute**

```python
def __init__(self, ..., axis_curve_func=None, ...):
    """
    axis_curve_func : callable, optional
        Custom axis curve function: (phi) -> (3,) or (n, 3)
    """
    self._axis_curve_func = axis_curve_func or self._default_axis_curve
```

### Usage Example

```python
def custom_helical_axis(phi, major_R=1.0, pitch=0.1):
    """Helical axis curve."""
    x = major_R * jnp.cos(phi)
    y = major_R * jnp.sin(phi)
    z = pitch * phi
    return jnp.stack([x, y, z], axis=-1)

# Register for later use
SurfaceGFrameFourier.register_axis_curve('helical', custom_helical_axis)

# Or pass directly
surface = SurfaceGFrameFourier(
    ...,
    axis_curve_func=custom_helical_axis
)
```

### Code Changes Required

1. Modify `__init__` to accept `axis_curve_func` parameter
2. Add `_axis_params` dict to store axis-specific parameters
3. Refactor `_compute_axis_curve` to dispatch to custom function
4. Add validation for function signature

## Enhancement 2: Analytical Phi Derivatives

### Current Status
- `gammadash_phi` is simplified/incomplete
- Uses finite differences in some cases

### Goal
Compute exact analytical derivatives with respect to φ.

### Mathematical Foundation

```
∂r/∂φ = ∂r_axis/∂φ + ∂X1/∂φ * n1 + X1 * ∂n1/∂φ + ∂X2/∂φ * n2 + X2 * ∂n2/∂φ

Where:
- ∂r_axis/∂φ: Derivative of axis curve
- ∂X1/∂φ: Fourier mode derivatives
- ∂n1/∂φ, ∂n2/∂φ: Frame vector derivatives
```

### Implementation

```python
@jit
def _compute_gamma_full(self):
    """Complete surface computation with all derivatives."""
    angles = self.angles
    sin_angles = jnp.sin(angles)
    cos_angles = jnp.cos(angles)
    
    # X1, X2 Fourier expansions
    X1 = jnp.einsum('i,ijk->jk', self.x1_coeffs, cos_angles)
    X2 = jnp.einsum('i,ijk->jk', self.x2_coeffs, sin_angles)
    
    # Derivatives w.r.t. phi (toroidal)
    dX1_dphi = jnp.einsum('i,ijk->jk', -self.nfp * self.xn * self.x1_coeffs, sin_angles)
    dX2_dphi = jnp.einsum('i,ijk->jk', -self.nfp * self.xn * self.x2_coeffs, cos_angles)
    
    # Derivatives w.r.t. theta (poloidal) - already available
    dX1_dtheta = jnp.einsum('i,ijk->jk', -self.xm * self.x1_coeffs, sin_angles)
    dX2_dtheta = jnp.einsum('i,ijk->jk', self.xm * self.x2_coeffs, cos_angles)
    
    # Get axis and frame (vmapped for efficiency)
    phi_1d = self.phi2d.flatten()
    axis_curve = vmap(self._compute_axis_curve)(phi_1d).reshape(self.nphi, self.ntheta, 3)
    n1, n2 = vmap(self._compute_frenet_frame)(phi_1d)
    n1 = n1.reshape(self.nphi, self.ntheta, 3)
    n2 = n2.reshape(self.nphi, self.ntheta, 3)
    
    # Derivatives of axis and frame
    d_axis = vmap(jacfwd(self._compute_axis_curve))(phi_1d).reshape(self.nphi, self.ntheta, 3)
    # Frame derivatives require careful computation...
    
    # Surface position and derivatives
    gamma = axis_curve + X1[..., None] * n1 + X2[..., None] * n2
    
    gammadash_theta = (2 * jnp.pi) * (
        dX1_dtheta[..., None] * n1 + 
        dX2_dtheta[..., None] * n2
    )
    
    gammadash_phi = (2 * jnp.pi) * (
        d_axis + 
        dX1_dphi[..., None] * n1 + 
        dX2_dphi[..., None] * n2 +
        X1[..., None] * d_n1_dphi +
        X2[..., None] * d_n2_dphi
    )
    
    return gamma, gammadash_theta, gammadash_phi, ...
```

### Testing

```python
def test_phi_derivatives():
    """Verify phi derivatives using finite differences."""
    surface = SurfaceGFrameFourier(...)
    
    # Analytical derivative
    gammadash_phi_analytical = surface.gammadash_phi
    
    # Finite difference check
    eps = 1e-8
    # Modify phi values and compute finite diff
    # Compare with analytical
```

## Enhancement 3: GVEC I/O Interface

### Goal
Read/write surfaces from/to GVEC equilibrium files.

### Structure

```python
@classmethod
def from_gvec_file(cls, filename, surface_id=0):
    """Load surface from GVEC output file."""
    import h5py
    
    with h5py.File(filename, 'r') as f:
        # Parse GVEC HDF5 structure
        x1_coeffs = jnp.array(f['surfaces'][surface_id]['X1_coeffs'][:])
        x2_coeffs = jnp.array(f['surfaces'][surface_id]['X2_coeffs'][:])
        nfp = int(f['nfp'][()])
        mpol = int(f['mpol'][()])
        ntor = int(f['ntor'][()])
        # ... more parameters
    
    return cls(x1_coeffs, x2_coeffs, nfp, mpol, ntor, ...)

def to_gvec_file(self, filename, mode='w'):
    """Save surface to GVEC format."""
    import h5py
    
    with h5py.File(filename, mode) as f:
        f.create_dataset('X1_coeffs', data=np.array(self.x1_coeffs))
        f.create_dataset('X2_coeffs', data=np.array(self.x2_coeffs))
        f.attrs['nfp'] = self.nfp
        f.attrs['mpol'] = self.mpol
        f.attrs['ntor'] = self.ntor
        # ... more attributes
```

### Implementation Checklist
- [ ] Understand GVEC HDF5 file format
- [ ] Add h5py dependency check
- [ ] Implement reading logic
- [ ] Implement writing logic
- [ ] Add error handling and validation
- [ ] Test with real GVEC files

## Enhancement 4: Multi-Surface Radial Representation

### Goal
Represent a family of nested flux surfaces (radial coordinate)

```python
class SurfaceGFrameFourierRadial:
    """Multiple nested G-frame surfaces along radial coordinate."""
    
    def __init__(self, x1_coeffs_radial, x2_coeffs_radial, s_grid, ...):
        """
        x1_coeffs_radial: (n_surfaces, n_modes)
        x2_coeffs_radial: (n_surfaces, n_modes)
        s_grid: radial grid points [0, 1]
        """
        self.n_surfaces = len(s_grid)
        self.s_grid = s_grid
        
        # Create surfaces
        self.surfaces = [
            SurfaceGFrameFourier(x1_coeffs_radial[i], x2_coeffs_radial[i], ...)
            for i in range(self.n_surfaces)
        ]
    
    def surface_at_s(self, s):
        """Interpolate surface at arbitrary s value."""
        # Interpolate coefficients
        x1_interp = jnp.interp(s, self.s_grid, self.x1_coeffs_radial)
        x2_interp = jnp.interp(s, self.s_grid, self.x2_coeffs_radial)
        
        return SurfaceGFrameFourier(x1_interp, x2_interp, ...)
```

## Enhancement 5: Symmetry Support

### Goal
Explicitly enforce stellarator symmetry

```python
def __init__(self, ..., symmetry='full', ...):
    """
    symmetry : str
        'none': No symmetry
        'full': Full stellarator symmetry (R=symmetric, Z=antisymmetric)
        'custom': User-defined symmetry
    """
    self._symmetry = symmetry
    self._apply_symmetry_constraints()

def _apply_symmetry_constraints(self):
    """Apply symmetry constraints to coefficients."""
    if self._symmetry == 'full':
        # For stellarator symmetry:
        # - Even m modes should have same x1 sign
        # - Odd m modes should have opposite x1 sign
        # - Similar for x2 with sine/cosine combinations
        pass
```

## Testing Infrastructure

### Create Test Suite

```python
# File: tests/test_surface_gframe.py

import pytest
import jax
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

class TestSurfaceGFrameFourier:
    """Test suite for G-frame surfaces."""
    
    @pytest.fixture
    def basic_surface(self):
        """Create a basic test surface."""
        return SurfaceGFrameFourier(
            x1_coeffs=jnp.array([0.2, 0.05, 0.02]),
            x2_coeffs=jnp.array([0.0, 0.03, -0.02]),
            nfp=2, mpol=1, ntor=1
        )
    
    def test_surface_creation(self, basic_surface):
        """Test surface instantiation."""
        assert basic_surface.gamma.shape == (30, 30, 3)
    
    def test_volume_positive(self, basic_surface):
        """Volume should be positive."""
        assert basic_surface.volume > 0
    
    def test_area_positive(self, basic_surface):
        """Area should be positive."""
        assert basic_surface.area > 0
    
    def test_unit_normal_normalized(self, basic_surface):
        """Unit normal should have magnitude 1."""
        unitnormal = basic_surface.unitnormal
        norms = jnp.linalg.norm(unitnormal, axis=2)
        assert jnp.allclose(norms, 1.0, atol=1e-6)
    
    def test_jax_jit(self, basic_surface):
        """Test JIT compatibility."""
        @jax.jit
        def compute_volume(s):
            return s.volume
        
        vol = compute_volume(basic_surface)
        assert vol > 0
    
    def test_jax_grad(self, basic_surface):
        """Test autodiff compatibility."""
        def objective(dofs):
            s = SurfaceGFrameFourier(dofs[:3], dofs[3:], 2, 1, 1)
            return s.volume
        
        grad_obj = jax.grad(objective)
        g = grad_obj(basic_surface.dofs)
        assert g.shape == basic_surface.dofs.shape
```

### Run Tests

```bash
pytest tests/test_surface_gframe.py -v
```

## Performance Optimization

### Current Bottlenecks
1. Frenet frame computation (vmapped)
2. Finite difference derivatives
3. Meshgrid creation

### Optimization Strategies

```python
# Use lower-level JAX primitives where possible
@jit
def _compute_axis_curve_vectorized(phi_array):
    """Vectorized axis curve computation."""
    # Replace vmap with explicit vectorization
    # Can be more efficient for large arrays
    pass

# Cache expensive computations
def _cache_frenet_frame(self):
    """Pre-compute and cache Frenet frames."""
    phi_1d = self.phi2d.flatten()
    self._cached_n1 = vmap(self._compute_frenet_frame)[0](phi_1d)
    self._cached_n2 = vmap(self._compute_frenet_frame)[1](phi_1d)
```

### Benchmarking

```python
import time

def benchmark_gamma_computation():
    """Measure performance of surface computation."""
    surface = SurfaceGFrameFourier(...)
    
    start = time.time()
    for _ in range(100):
        _ = surface.gamma
    elapsed = time.time() - start
    
    print(f"Average time: {elapsed/100*1000:.2f} ms")
```

## Documentation Tasks

- [ ] Add docstring examples to each method
- [ ] Create Jupyter notebook with advanced usage
- [ ] Generate API documentation (Sphinx)
- [ ] Add type hints throughout

## Debugging & Validation

### Consistency Checks

```python
def validate_surface(self):
    """Comprehensive surface validation."""
    checks = []
    
    # Check volume and area have reasonable values
    if self.volume <= 0:
        checks.append("ERROR: Volume is non-positive")
    if self.area <= 0:
        checks.append("ERROR: Area is non-positive")
    
    # Check normal vectors
    norm_norms = jnp.linalg.norm(self.unitnormal, axis=2)
    if not jnp.allclose(norm_norms, 1.0):
        checks.append("WARNING: Unit normal not normalized")
    
    # Check for NaNs/Infs
    if jnp.any(jnp.isnan(self.gamma)):
        checks.append("ERROR: NaN in surface coordinates")
    
    return checks if checks else ["All checks passed"]
```

## Next Steps

### High Priority (Week 1-2)
1. Analytical phi derivatives
2. Custom axis curves
3. Unit tests

### Medium Priority (Week 3-4)
4. GVEC I/O interface
5. Performance optimization
6. Extended documentation

### Lower Priority (Ongoing)
7. Radial families
8. Symmetry enforcement
9. Integration with optimization routines

---

**Last Updated**: February 2026  
**Related Files**:
- Main implementation: `/home/exouser/ESSOS/essos/surfaces.py`
- User documentation: `/home/exouser/ESSOS/SURFACE_GFRAME_DOCUMENTATION.md`
- Examples: `/home/exouser/ESSOS/examples/surface_gframe_example.py`
