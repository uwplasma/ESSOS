"""Opt-in resolution of a positive thermal-speed first-passage diagnostic.

This limits monitoring intervals; it is not an exact Brownian crossing sampler.
The policy depends only on the current accepted state. Both an inward drift
budget and a diffusion-distance budget are used. A positive resolution floor
permits crossings, rather than approaching the boundary in infinitely many
steps. The split driver separately refines detected crossing intervals.
"""
import jax.numpy as jnp


def boundary_step_limit(speed, boundary, diffusion, drift, resolution, *, safety=.2):
    """Return a time interval using consistently scaled speed, D and drift.

    ``diffusion`` is D in dv=A dt+sqrt(2D)dW. ``resolution`` is the smallest
    boundary-policy interval and the detected-crossing ceiling, not a bound on
    the unknown continuous hit time. Missed returns require convergence checks.
    """
    distance = jnp.maximum(speed-boundary, 0.)
    diffusive = safety*distance**2/jnp.where(diffusion > 0, 2*diffusion, 1.)
    diffusive = jnp.where(diffusion > 0, diffusive, jnp.inf)
    inward = jnp.maximum(-drift, 0.)
    advective = safety*distance/jnp.where(inward > 0, inward, 1.)
    advective = jnp.where(inward > 0, advective, jnp.inf)
    return jnp.maximum(resolution, jnp.minimum(diffusive, advective))
