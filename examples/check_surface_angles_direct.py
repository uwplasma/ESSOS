"""Direct check of surface internals."""

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
x1_coeffs[0] = 0.0
x1_coeffs[2] = 0.2
x2_coeffs = np.zeros(13)

surface = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=2,
    ntor=2,
    nfp=1,
    axis_function=helical_axis,
    ntheta=5,
    nphi=4
)

# Force computation
gamma = surface.gamma

print("surface.theta2d:")
print(surface.theta2d)

print("\nsurface.phi2d:")
print(surface.phi2d)

print("\nsurface.xm:", surface.xm)
print("surface.xn:", surface.xn)

angles = surface.angles

print(f"\nangles shape: {angles.shape}")
print(f"Mode 2: m={surface.xm[2]}, n={surface.xn[2]}")

print("\nangles[2, :, :]:")
print(angles[2, :, :])

# Manual computation
manual_angles_2 = surface.xm[2] * surface.theta2d - surface.xn[2] * surface.phi2d
print("\nManual angles[2, :, :]:")
print(manual_angles_2)

print("\nDo they match?", jnp.allclose(angles[2, :, :], manual_angles_2))

# Check if surface varies
print("\n\nChecking if surface varies with theta:")
for i in [0, 1]:
    print(f"\nphi index {i}:")
    for j in [0, 1, 2]:
        print(f"  gamma[{i},{j},:] = {np.array(gamma)[i,j,:]}")
