"""Experimental RK increment storage; the production driver defines the result."""

import jax
import jax.numpy as jnp
import numpy as np

from essos.rk import _RKTrial


def unrolled_trial(rhs, t, y, h, args, f0, nodes, rows, errors):
    f0 = rhs(t, y, args) if f0 is None else f0
    increments = [h * f0]
    for c, weights in zip(nodes, rows):
        trial = y + sum(a * k for a, k in zip(weights, increments) if a != 0)
        final = rhs(t + c * h, trial, args)
        increments.append(h * final)
    error = sum(a * k for a, k in zip(errors, increments) if a != 0)
    return _RKTrial(trial, error, jnp.stack(increments), final)


def loop_trial(rhs, t, y, h, args, f0, nodes, rows, errors, *, components=False):
    f0 = rhs(t, y, args) if f0 is None else f0
    n = len(errors)
    coefficients = np.zeros((n - 2, n - 1))
    for i, row in enumerate(rows[:-1]):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, dtype=y.dtype)
    times = jnp.asarray(nodes[:-1], dtype=y.dtype)
    endpoint_weights = jnp.asarray(rows[-1], dtype=y.dtype)
    error_weights = jnp.asarray(errors[:-1], dtype=y.dtype)
    if components:
        initial = tuple(jnp.concatenate([h * f0[d:d + 1], jnp.zeros(n - 2, y.dtype)]) for d in range(len(y)))
        def point(weights, bank):
            return jnp.stack([y[d] + jnp.dot(weights, values) for d, values in enumerate(bank)])
        def stage(i, bank):
            trial = point(coefficients[i - 1], bank)
            increment = h * rhs(t + times[i - 1] * h, trial, args)
            return tuple(jax.lax.dynamic_update_index_in_dim(values, increment[d], i, axis=0)
                         for d, values in enumerate(bank))
    else:
        initial = jnp.concatenate([(h * f0)[None], jnp.zeros((n - 2,) + y.shape, y.dtype)])
        def point(weights, bank):
            return y + jnp.tensordot(weights, bank, axes=1)
        def stage(i, bank):
            trial = point(coefficients[i - 1], bank)
            increment = h * rhs(t + times[i - 1] * h, trial, args)
            return jax.lax.dynamic_update_index_in_dim(bank, increment, i, axis=0)
    bank = jax.lax.fori_loop(1, n - 1, stage, initial)
    endpoint = point(endpoint_weights, bank)
    final = rhs(t + h, endpoint, args)
    if components:
        error = jnp.stack([jnp.dot(error_weights, values) for values in bank])
        bank = jnp.stack(bank, axis=1)
    else:
        error = jnp.tensordot(error_weights, bank, axes=1)
    if errors[-1] != 0:
        error = error + errors[-1] * (h * final)
    increments = jnp.concatenate([bank, (h * final)[None]])
    return _RKTrial(endpoint, error, increments, final)


def increment_kernel(name):
    if name == "trial_unrolled":
        return unrolled_trial
    if name == "trial_loop":
        return loop_trial
    if name == "trial_components":
        return lambda *args: loop_trial(*args, components=True)
    if name == "trial_production":
        import essos.rk as rk
        return rk._trial_step
    raise ValueError(name)
