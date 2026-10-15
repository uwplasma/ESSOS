"""Derivative contract of the shared pure RK map, including FSAL carry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import rk45_step, dopri8_step, _trial_step, _C, _A, _ERROR, _C8, _A8, _ERROR8

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("step", [rk45_step, dopri8_step])
@pytest.mark.parametrize("cached", [False, True])
def test_state_time_step_parameter_and_fsal_derivatives(step, cached):
    def evaluate(x):
        y, t, h, parameter = x[:2], x[2], x[3], x[4]
        rhs = lambda t, y, args: parameter * jnp.array([jnp.sin(y[1] + t), -y[0]**2 + jnp.cos(t)])
        f0 = x[5:] if cached else None
        out = step(rhs, t, y, h, f0=f0)
        return jnp.concatenate([out.y, out.error, out.stages.ravel()])
    x = jnp.array([.7, -.3, .2, .1, .8, -.08, .392])
    forward, reverse = jax.jacfwd(evaluate)(x), jax.jacrev(evaluate)(x)
    np.testing.assert_allclose(forward, reverse, rtol=2e-10, atol=2e-12)
    eps = 1e-5
    finite = jnp.stack([(evaluate(x + direction * eps) - evaluate(x - direction * eps)) / (2 * eps)
                        for direction in jnp.eye(len(x))], axis=1)
    np.testing.assert_allclose(reverse, finite, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("h", [0., .1])
def test_forward_trial_and_its_derivatives_match_the_public_map(method, cached, h):
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    public = rk45_step if method == "dopri5" else dopri8_step
    def observe(x, forward):
        y, t, h, parameter = x[:2], x[2], x[3], x[4]
        rhs = lambda t, y, args: parameter * jnp.array([jnp.sin(y[1] + t), -y[0]**2 + jnp.cos(t)])
        f0 = x[5:] if cached else None
        if forward:
            step = _trial_step(rhs, t, y, h, None, f0, *constants)
            increments, final = step.increments, step.f1
        else:
            step = public(rhs, t, y, h, f0=f0)
            increments, final = h * step.stages, step.stages[-1]
        return jnp.concatenate([step.y, step.error, increments.ravel(), final])
    x = jnp.array([.7, -.3, .2, h, .8, -.08, .392])
    before = lambda x: observe(x, False)
    after = lambda x: observe(x, True)
    np.testing.assert_allclose(after(x), before(x), rtol=2e-12, atol=2e-12)
    reference = jax.jacfwd(before)(x)
    np.testing.assert_allclose(jax.jacfwd(after)(x), reference, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(jax.jacrev(after)(x), reference, rtol=2e-10, atol=2e-12)
