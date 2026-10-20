"""Check what axis_shape is set to."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp
from essos.surfaces import SurfaceGFrameFourier

def helical_axis(phi):
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

x1_coeffs = np.zeros(13)
x1_coeffs[0] = 1.0
x1_coeffs[2] = 0.15
x2_coeffs = np.zeros(13)

# Test 1: without axis_shape set
print("Test 1: axis_function provided, axis_shape not specified")
surface1 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_function=helical_axis,
    ntheta=10,
    nphi=10
)

print(f"axis_shape: '{surface1._axis_shape}'")
print(f"axis_function: {surface1._axis_function}")

# Test 2: with axis_shape='custom' explicitly set
print("\nTest 2: axis_function provided, axis_shape='custom' explicitly set")
surface2 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_shape='custom',
    axis_function=helical_axis,
    ntheta=10,
    nphi=10
)

print(f"axis_shape: '{surface2._axis_shape}'")
print(f"axis_function: {surface2._axis_function}")

# Compute gamma for both  
print("\nComputing surface 1 gamma...")
gamma1 = np.array(surface1.gamma)
print(f"Surface 1 has NaN: {np.any(np.isnan(gamma1))}")

print("\nComputing surface 2 gamma...")
gamma2 = np.array(surface2.gamma)
print(f"Surface 2 has NaN: {np.any(np.isnan(gamma2))}")

if not np.any(np.isnan(gamma2)):
    print(f"Surface 2 X range: [{gamma2[:,:,0].min():.3f}, {gamma2[:,:,0].max():.3f}]")
