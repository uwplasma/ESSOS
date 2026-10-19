"""Frozen local collision-error coefficients, shared by GC and particle control.

This describes the estimates in Sarkimaki et al., Eqs. (60)--(62) and
(73)--(75), using the magnitude approximation after Eq. (55). It predicts
local error with coefficients frozen at a trial's initial state. It does not
predict the next state or bound orbit/collision splitting error.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class CollisionErrorModel(NamedTuple):
    abs_drift: jax.Array
    parallel: jax.Array
    drift: jax.Array                 # .5 |Q Q'| (GC), or .5 |K K'| (particle)
    diffusion: jax.Array             # (D_parallel')**2
    pitch_drift: jax.Array           # .5 |xi| nu**2; zero for particle control
    pitch_diffusion: jax.Array       # .5 sqrt((1-xi**2) nu) nu; zero for particle
    tolerance: jax.Array
    speed_noise_index: jax.Array     # -2 for GC (including spatial noise), 2 for particle


def _ratio(numerator, denominator):
    safe = jnp.where(denominator > 0, denominator, 1.)
    return jnp.where(denominator > 0, numerator/safe,
                     jnp.where(numerator == 0, 0., jnp.inf))


def collision_error_estimates(model, h, dw):
    """Evaluate both normalized estimates without another coefficient/RHS call."""
    eps_v = model.tolerance*(model.abs_drift*h+jnp.sqrt(jnp.maximum(2*model.parallel*h, 0.)))
    drift = jnp.maximum(_ratio(model.drift*h*h, eps_v),
                        model.pitch_drift*h*h/model.tolerance)
    diffusion = jnp.maximum(
        _ratio(model.diffusion*jnp.abs(dw[model.speed_noise_index])**3,
               6*eps_v*jnp.sqrt(jnp.maximum(model.parallel, 0.))),
        model.pitch_diffusion*h*(jnp.abs(dw[-1])+jnp.sqrt(h/3))/model.tolerance)
    return drift, diffusion


def collision_error_passes(model, h, dw, *, target=1.):
    """Check the same four bounds without forming normalized error ratios.

    Coefficients and tolerance are finite/nonnegative, with positive tolerance,
    as supplied by the physical collision maps. Zero numerators and zero speed
    diffusion retain the estimator's 0/0 -> 0 convention. The actual trial still
    computes its normalized errors and independently checks finiteness.
    """
    tolerance = target*model.tolerance
    speed_scale = tolerance*(model.abs_drift*h+jnp.sqrt(jnp.maximum(2*model.parallel*h, 0.)))
    return ((model.drift*h*h <= speed_scale)
            & (model.diffusion*jnp.abs(dw[model.speed_noise_index])**3
               <= 6*speed_scale*jnp.sqrt(jnp.maximum(model.parallel, 0.)))
            & (model.pitch_drift*h*h <= tolerance)
            & (model.pitch_diffusion*h*(jnp.abs(dw[-1])+jnp.sqrt(h/3)) <= tolerance))
