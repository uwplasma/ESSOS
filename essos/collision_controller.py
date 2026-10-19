"""Candidate-error timestep proposals with retained Brownian samples.

Both GC and particle paths use this same controller. Only the frozen local
error coefficients differ. A proposal never replaces the real-step acceptance
test in solve_split, and every probed Wiener endpoint remains in memory.
"""
import jax.numpy as jnp
from jax import lax

from .collision_error import collision_error_passes
from .collisions import brownian_sample


def _fit_error_step(model, bound, minimum, normal_magnitude, target):
    """Largest bracketed step for frozen coefficients and fixed |dW|/sqrt(h).

    Eight bisections avoid assuming a common error power for speed and pitch.
    This is a cheap prediction: the next actual Brownian sample is checked
    separately. A sub-grid positive fallback still goes through real acceptance.
    """
    # At fixed |dW|/sqrt(h), cancel the common sqrt(h) in the speed
    # numerators/denominator and compare cross-products. These are exactly the
    # same four error inequalities, including 0/0 -> 0 in vacuum, without
    # repeated divisions, noise indexing, or coefficient square roots.
    parallel = jnp.sqrt(jnp.maximum(model.parallel, 0.))
    thermal_noise = jnp.sqrt(jnp.maximum(2*model.parallel, 0.))
    speed_noise = model.diffusion*jnp.abs(normal_magnitude[model.speed_noise_index])**3
    pitch_noise = model.pitch_diffusion*(jnp.abs(normal_magnitude[-1])+1/jnp.sqrt(3.))
    tolerance = target*model.tolerance

    def passes(h):
        root_h = jnp.sqrt(h)
        scale = tolerance*(model.abs_drift*root_h+thermal_noise)
        return ((model.drift*h*root_h <= scale)
                & (speed_noise*h <= 6*parallel*scale)
                & (model.pitch_drift*h*h <= tolerance)
                & (pitch_noise*h*root_h <= tolerance))

    def bisect(_):
        def one(_, bracket):
            lo, hi = bracket
            middle = (lo+hi)/2
            passed = passes(middle)
            return jnp.where(passed, middle, lo), jnp.where(passed, hi, middle)
        lo, hi = lax.fori_loop(0, 8, one, (minimum, bound))
        return jnp.where(lo > 0, lo, hi/2)
    return lax.cond(passes(bound), lambda _: bound, bisect, None)


def candidate_collision_step_size(memory, h, error_model, accepted, *,
                                  dtmin, dtmax, barrier, target=.9):
    """Predict, probe and backtrack using both frozen normalized error estimates.

    Accepted trials can grow by at most two; rejected trials shrink by at least
    20%, subject to dtmin and save/final boundaries. The initial predictor uses
    unit Gaussian magnitudes. A failed probe refits using its realized Gaussian
    magnitudes and a smaller upper bound. At most four Wiener points are probed.
    If the budget is exhausted, the last sampled step is returned for the real
    acceptance test; no inaccurate step is forced through.
    """
    limit = jnp.minimum(dtmax, barrier-memory.t)
    bound = jnp.minimum(jnp.maximum(jnp.where(accepted, 2., .8)*h, dtmin), limit)
    minimum = jnp.minimum(dtmin, bound)
    unit_noise = jnp.ones(memory.increments.shape[1:], dtype=memory.increments.dtype)
    proposal = _fit_error_step(error_model, bound, minimum, unit_noise, target)

    def stochastic(_):
        def probe(carry):
            m, wanted, last, count, running, overflow = carry
            end = jnp.minimum(m.t+wanted, barrier)
            length = end-m.t
            m, dw, full = brownian_sample(m, end)
            passed = collision_error_passes(error_model, length, dw, target=target)
            retry = ~passed & ~full & (length > minimum) & (count < 3)

            def refit(_):
                smaller = jnp.maximum(.8*length, minimum)
                normal = jnp.abs(dw)/jnp.sqrt(length)
                return _fit_error_step(error_model, smaller, minimum, normal, target)
            next_h = lax.cond(retry, refit, lambda _: length, None)
            return m, next_h, length, count+1, retry, overflow | full
        m, _, last, _, _, overflow = lax.while_loop(
            lambda c: c[4], probe,
            (memory, proposal, proposal, jnp.array(0), jnp.array(True), jnp.array(False)))
        return m, last, overflow

    has_noise_error = (error_model.diffusion > 0) | (error_model.pitch_diffusion > 0)
    return lax.cond(has_noise_error, stochastic,
        lambda _: (memory, proposal, jnp.array(False)), None)
