"""Quick diagnostic to check surface values."""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/exouser/ESSOS')

from essos.surfaces import SurfaceGFrameFourier

def helical_axis(phi):
    """Helical axis that closes on itself."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

def figure_eight_axis(phi):
    """Figure-eight axis."""
    R = 1.0
    denom = 1.0 + jnp.sin(phi)**2
    r = R * jnp.sqrt(2.0) * jnp.cos(phi) / denom
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = jnp.zeros_like(phi)
    return jnp.stack([x, y, z], axis=-1)

# Test helical axis
print("\n" + "="*70)
print("HELICAL AXIS SURFACE")
print("="*70)

mpol = 2
ntor = 2
nfp = 1
coeff_size = (mpol + 1) * (2 * ntor + 1) - ntor

x1_coeffs = np.zeros(coeff_size)
x1_coeffs[0] = 1.0
x1_coeffs[2] = 0.15

x2_coeffs = np.zeros(coeff_size)

surface_helical = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs,
    x2_coeffs=x2_coeffs,
    mpol=mpol,
    ntor=ntor,
    nfp=nfp,
    axis_function=helical_axis
)

gamma_helical = np.array(surface_helical.gamma)
print(f"Helical surface shape: {gamma_helical.shape}")
print(f"Has NaN: {np.any(np.isnan(gamma_helical))}")
print(f"Min values: [{gamma_helical[:,:,0].min():.3f}, {gamma_helical[:,:,1].min():.3f}, {gamma_helical[:,:,2].min():.3f}]")
print(f"Max values: [{gamma_helical[:,:,0].max():.3f}, {gamma_helical[:,:,1].max():.3f}, {gamma_helical[:,:,2].max():.3f}]")
print(f"Sample points:")
for i in [0, 15, 30]:
    for j in [0, 15, 30]:
        print(f"  gamma[{i},{j},:] = {gamma_helical[i,j,:]}")

# Test figure-eight axis
print("\n" + "="*70)
print("FIGURE-EIGHT AXIS SURFACE")
print("="*70)

x1_coeffs_fig8 = np.zeros(coeff_size)
x1_coeffs_fig8[0] = 1.0
x1_coeffs_fig8[2] = 0.25

x2_coeffs_fig8 = np.zeros(coeff_size)

surface_fig8 = SurfaceGFrameFourier(
    x1_coeffs=x1_coeffs_fig8,
    x2_coeffs=x2_coeffs_fig8,
    mpol=mpol,
    ntor=ntor,
    nfp=nfp,
    axis_function=figure_eight_axis
)

gamma_fig8 = np.array(surface_fig8.gamma)
print(f"Figure-8 surface shape: {gamma_fig8.shape}")
print(f"Has NaN: {np.any(np.isnan(gamma_fig8))}")
print(f"Min values: [{gamma_fig8[:,:,0].min():.3f}, {gamma_fig8[:,:,1].min():.3f}, {gamma_fig8[:,:,2].min():.3f}]")
print(f"Max values: [{gamma_fig8[:,:,0].max():.3f}, {gamma_fig8[:,:,1].max():.3f}, {gamma_fig8[:,:,2].max():.3f}]")
print(f"Sample points:")
for i in [0, 15, 30]:
    for j in [0, 15, 30]:
        print(f"  gamma[{i},{j},:] = {gamma_fig8[i,j,:]}")
