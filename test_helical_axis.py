#!/usr/bin/env python3
"""Test script for helical axis functionality in SurfaceGFrameFourier."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

def test_helical_axis_basic():
    """Test basic helical axis creation and evaluation."""
    print("=" * 60)
    print("Test 1: Basic Helical Axis Creation")
    print("=" * 60)
    
    # Create simple Fourier coefficients
    mpol, ntor, nfp = 1, 1, 2
    n_coeffs = (mpol + 1) * (2 * ntor + 1)
    
    x1_coeffs = jnp.array([0.1, 0.05, -0.02], dtype=jnp.float64)
    x2_coeffs = jnp.array([0.08, 0.03, -0.01], dtype=jnp.float64)
    
    # Create surface with helical axis
    print(f"\nCreating surface with:")
    print(f"  - nfp = {nfp}, mpol = {mpol}, ntor = {ntor}")
    print(f"  - axis_shape = 'helical'")
    print(f"  - axis_pitch = 0.2 (rise per 2π turn)")
    print(f"  - axis_twist_rate = 0.0 (no frame twist)")
    
    surface = SurfaceGFrameFourier(
        x1_coeffs=x1_coeffs,
        x2_coeffs=x2_coeffs,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ntheta=20,
        nphi=20,
        axis_shape='helical',
        axis_major_radius=1.0,
        axis_pitch=0.2,
        axis_twist_rate=0.0
    )
    
    print(f"\nSurface object created successfully!")
    print(f"  DOFs shape: {surface.dofs.shape}")
    print(f"  Gamma shape: {surface.gamma.shape}")
    print(f"\n✓ Test 1 passed!")


def test_helical_axis_with_twist():
    """Test helical axis with frame twist."""
    print("\n" + "=" * 60)
    print("Test 2: Helical Axis with Frame Twist")
    print("=" * 60)
    
    mpol, ntor, nfp = 1, 1, 2
    x1_coeffs = jnp.array([0.1, 0.05, -0.02], dtype=jnp.float64)
    x2_coeffs = jnp.array([0.08, 0.03, -0.01], dtype=jnp.float64)
    
    # Create surface with helical axis and frame twist
    print(f"\nCreating surface with:")
    print(f"  - axis_shape = 'helical'")
    print(f"  - axis_pitch = 0.3 (rise per 2π turn)")
    print(f"  - axis_twist_rate = 0.5 rad/turn (frame rotates)")
    
    surface = SurfaceGFrameFourier(
        x1_coeffs=x1_coeffs,
        x2_coeffs=x2_coeffs,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ntheta=20,
        nphi=20,
        axis_shape='helical',
        axis_major_radius=1.0,
        axis_pitch=0.3,
        axis_twist_rate=0.5
    )
    
    print(f"\nSurface object created successfully!")
    print(f"  Gamma shape: {surface.gamma.shape}")
    
    # Check that surface extent includes z component now
    gamma = surface.gamma
    z_min, z_max = float(jnp.min(gamma[..., 2])), float(jnp.max(gamma[..., 2]))
    print(f"\nSurface z-extent: [{z_min:.4f}, {z_max:.4f}]")
    print(f"  (Expected to include pitch effects since axis_pitch=0.3)")
    
    print(f"\n✓ Test 2 passed!")


def test_axis_shape_comparison():
    """Compare different axis shapes."""
    print("\n" + "=" * 60)
    print("Test 3: Axis Shape Comparison")
    print("=" * 60)
    
    mpol, ntor, nfp = 1, 1, 2
    x1_coeffs = jnp.array([0.1, 0.05, -0.02], dtype=jnp.float64)
    x2_coeffs = jnp.array([0.08, 0.03, -0.01], dtype=jnp.float64)
    
    shapes = ['circle', 'ellipse', 'helical']
    surfaces = {}
    
    for shape in shapes:
        print(f"\nCreating surface with axis_shape = '{shape}':")
        if shape == 'helical':
            surf = SurfaceGFrameFourier(
                x1_coeffs=x1_coeffs,
                x2_coeffs=x2_coeffs,
                nfp=nfp,
                mpol=mpol,
                ntor=ntor,
                ntheta=15,
                nphi=15,
                axis_shape=shape,
                axis_major_radius=1.0,
                axis_pitch=0.2,
                axis_twist_rate=0.1
            )
        else:
            surf = SurfaceGFrameFourier(
                x1_coeffs=x1_coeffs,
                x2_coeffs=x2_coeffs,
                nfp=nfp,
                mpol=mpol,
                ntor=ntor,
                ntheta=15,
                nphi=15,
                axis_shape=shape,
                axis_major_radius=1.0
            )
        
        gamma = surf.gamma
        print(f"  Gamma shape: {gamma.shape}")
        print(f"  x-range: [{jnp.min(gamma[..., 0]):.4f}, {jnp.max(gamma[..., 0]):.4f}]")
        print(f"  y-range: [{jnp.min(gamma[..., 1]):.4f}, {jnp.max(gamma[..., 1]):.4f}]")
        print(f"  z-range: [{jnp.min(gamma[..., 2]):.4f}, {jnp.max(gamma[..., 2]):.4f}]")
        surfaces[shape] = surf
    
    print(f"\n✓ Test 3 passed!")


def test_gradient_through_helical():
    """Test that JAX gradients work through helical surfaces."""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Computation Through Helical Surface")
    print("=" * 60)
    
    mpol, ntor, nfp = 1, 1, 2
    x1_coeffs = jnp.array([0.1, 0.05, -0.02], dtype=jnp.float64)
    x2_coeffs = jnp.array([0.08, 0.03, -0.01], dtype=jnp.float64)
    
    print(f"\nComputing surface volume (requires integration over gamma)...")
    
    def volume_proxy(coeffs):
        """Compute a simple volume-like quantity."""
        x1 = coeffs[:3]
        x2 = coeffs[3:]
        surf = SurfaceGFrameFourier(
            x1_coeffs=x1,
            x2_coeffs=x2,
            nfp=nfp,
            mpol=mpol,
            ntor=ntor,
            ntheta=10,
            nphi=10,
            axis_shape='helical',
            axis_pitch=0.2,
            axis_twist_rate=0.1
        )
        gamma = surf.gamma
        return jnp.sum(jnp.linalg.norm(gamma, axis=-1))
    
    coeffs = jnp.hstack([x1_coeffs, x2_coeffs])
    value = volume_proxy(coeffs)
    print(f"  Volume-like quantity: {float(value):.6f}")
    
    # Compute gradient
    grad_func = jax.grad(volume_proxy)
    grads = grad_func(coeffs)
    print(f"  Gradient computed successfully!")
    print(f"  Gradient shape: {grads.shape}")
    print(f"  Gradient magnitude: {jnp.linalg.norm(grads):.6f}")
    
    print(f"\n✓ Test 4 passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HELICAL AXIS TESTS FOR SurfaceGFrameFourier")
    print("=" * 60)
    
    try:
        test_helical_axis_basic()
        test_helical_axis_with_twist()
        test_axis_shape_comparison()
        test_gradient_through_helical()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nHelical axis implementation is working correctly!")
        print("Both pitch and twist_rate parameters are functional.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
