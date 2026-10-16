"""Test if jacfwd import and usage works in surface context."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import jax.numpy as jnp
from jax import jacfwd, vmap

def helical_axis(phi):
    """Helical axis that closes on itself."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

print("Testing jacfwd in isolation...")
phi_test = jnp.linspace(0, 2*jnp.pi, 10)

try:
    tangent_func = jacfwd(helical_axis)
    tangent = vmap(tangent_func)(phi_test)
    print(f"SUCCESS: tangent shape = {tangent.shape}")
    print(f"Has NaN: {jnp.any(jnp.isnan(tangent))}")
    print(f"Sample: {tangent[0]}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nNow testing with surface...")
from essos.surfaces import SurfaceGFrameFourier
import numpy as np

x1_coeffs = np.zeros(13)
x1_coeffs[0] = 1.0
x1_coeffs[2] = 0.15
x2_coeffs = np.zeros(13)

try:
    surface = SurfaceGFrameFourier(
        x1_coeffs=x1_coeffs,
        x2_coeffs=x2_coeffs,
        mpol=2,
        ntor=2,
        nfp=1,
        axis_function=helical_axis,
        ntheta=10,
        nphi=10
    )
    
    gamma = np.array(surface.gamma)
    print(f"Surface created successfully")
    print(f"Gamma shape: {gamma.shape}")
    print(f"Has NaN: {np.any(np.isnan(gamma))}")
    if not np.any(np.isnan(gamma)):
        print(f"X range: [{gamma[:,:,0].min():.3f}, {gamma[:,:,0].max():.3f}]")
    else:
        # Check which points are NaN
        nan_mask = np.isnan(gamma[:,:,0])
        print(f"NaN count: {np.sum(nan_mask)} / {nan_mask.size}")
        print(f"First non-NaN indices: {np.argwhere(~nan_mask)[0] if np.any(~nan_mask) else 'all NaN'}")
        print(f"First NaN indices: {np.argwhere(nan_mask)[0] if np.any(nan_mask) else 'none'}")
        
except Exception as e:
    print(f"Surface creation FAILED: {e}")
    import traceback
    traceback.print_exc()
