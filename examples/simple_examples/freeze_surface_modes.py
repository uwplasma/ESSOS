"""Select `SurfaceRZFourier` Fourier coefficients by their physical (m, n) mode."""
import jax.numpy as jnp

from essos.losses import custom_loss
from essos.surfaces import SurfaceRZFourier


surface = SurfaceRZFourier(
    rc=jnp.array([10.0, 1.0]), zs=jnp.array([0.0, 2.0]),
    nfp=1, mpol=1, ntor=0,
)

loss = custom_loss(lambda surface: jnp.sum(surface.rc**2) + jnp.sum(surface.zs**2), "surface")
loss.dependencies = {"surface": surface}

# Freeze RBC(0,1) and ZBS(0,1), expressed as (m, n) pairs.
loss.freeze_surface_modes("surface", rc=[(1, 0)], zs=[(1, 0)])

proposed = loss.starting_dofs + 1.0
frozen_surface = loss.dofs_to_pytree(proposed)["surface"]
print("Initial rc, zs:", surface.rc, surface.zs)
print("After a proposed update:", frozen_surface.rc, frozen_surface.zs)
