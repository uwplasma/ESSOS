"""Order, cost, derivative and integration contracts for optional dense output."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import (RKStatus, _A8, _DENSE8, _dopri8_output_coefficients, _polynomial_dense,
                      dopri8_step, solve_rk)
from essos.rk_dense import DOPRI8_OUTPUT

jax.config.update("jax_enable_x64", True)


def test_coefficients_reproduce_and_satisfy_all_fifth_order_conditions():
    from examples.benchmarks.derive_rk_dense_output import derive, rooted_trees, tree_order, tree_gamma
    derived = derive()
    np.testing.assert_allclose(DOPRI8_OUTPUT, derived["coefficients_descending"], atol=2e-8, rtol=2e-8)
    assert derived["integrated_bseries_error_after"] < .3 * derived["integrated_bseries_error_before"]
    a = np.zeros((14, 14))
    for i, row in enumerate(_A8):
        a[i+1, :len(row)] = row
    def elementary(tree):
        return np.prod([a @ elementary(child) for child in tree], axis=0) if tree else np.ones(14)
    coefficients = np.asarray(DOPRI8_OUTPUT)[:, ::-1]
    for tree in rooted_trees(5):
        target = np.zeros(6)
        target[tree_order(tree)-1] = 1/tree_gamma(tree)
        np.testing.assert_allclose(elementary(tree) @ coefficients, target, atol=2e-12, rtol=0.)
    np.testing.assert_allclose(coefficients.sum(axis=1), a[-1], atol=2e-12, rtol=0.)
    # Preserve the supplied standard coefficients' endpoint slopes, including
    # their existing ~3e-11 coefficient rounding residual.
    np.testing.assert_allclose(coefficients @ np.arange(1, 7),
                               np.asarray(_DENSE8)[:, ::-1] @ np.arange(1, 7), atol=2e-12, rtol=0.)


@pytest.mark.parametrize("power", range(5))
def test_saved_polynomial_is_exact_through_degree_five(power):
    rhs = lambda t, y, args: jnp.full_like(y, t**power)
    y, t, h = jnp.array([.3]), .2, .7
    trial = dopri8_step(rhs, t, y, h)
    coefficients = _dopri8_output_coefficients(h * trial.stages)
    points = jnp.linspace(0., 1., 17)
    actual = jax.vmap(lambda s: _polynomial_dense(y, trial.y, coefficients, s))(points)
    expected = y + ((t + points[:, None]*h)**(power+1) - t**(power+1))/(power+1)
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=0.)


def test_nonlinear_dense_error_converges_at_the_retained_order():
    errors = []
    y = jnp.array([.8])
    for h in (.5, .25, .125):
        trial = dopri8_step(lambda t, y, args: y*y, 0., y, h)
        coefficients = _dopri8_output_coefficients(h * trial.stages)
        points = jnp.array([.17, .37, .59, .83])
        actual = jax.vmap(lambda s: _polynomial_dense(y, trial.y, coefficients, s))(points)
        exact = .8/(1-.8*h*points[:, None])
        errors.append(float(jnp.max(jnp.abs(actual-exact))))
    assert errors[2] < errors[1] / 25 < errors[0] / 500


def test_dense_map_has_ordinary_forward_and_reverse_derivatives():
    def evaluate(x):
        rhs = lambda t, y, args: x[4] * jnp.array([y[1] + jnp.sin(t), -y[0]**2])
        trial = dopri8_step(rhs, x[2], x[:2], x[3])
        return _polynomial_dense(x[:2], trial.y, _dopri8_output_coefficients(x[3]*trial.stages), x[5])
    point = jnp.array([.7, -.3, .2, .1, .8, .47])
    reverse = jax.jacrev(evaluate)(point)
    np.testing.assert_allclose(jax.jacfwd(evaluate)(point), reverse, atol=2e-11, rtol=2e-10)
    eps = 1e-5
    finite = jnp.stack([(evaluate(point + eps*d)-evaluate(point - eps*d))/(2*eps)
                        for d in jnp.eye(len(point))], axis=1)
    np.testing.assert_allclose(reverse, finite, atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("dense_output", ["standard", "optimized"])
def test_dense_output_adds_no_rhs_evaluations(dense_output):
    calls = []
    def rhs(t, y, args):
        jax.debug.callback(lambda t: calls.append(float(t)), t, ordered=True)
        return jnp.array([y[1], -y[0]])
    out = jax.jit(lambda y: solve_rk(rhs, y, t1=3., dt0=2., method="dopri8",
        dense_output=dense_output, ts=jnp.linspace(0., 3., 201)))(jnp.array([1., 0.]))
    jax.block_until_ready(out)
    jax.effects_barrier()
    assert out.status == RKStatus.SUCCESS
    assert len(calls) == 1 + 13 * int(out.stats.steps)


@pytest.mark.parametrize("event", [None, lambda t, y, args: y[0]-.5])
def test_dense_output_preserves_retries_endpoints_and_event_times(event):
    def run(output):
        return solve_rk(lambda t, y, args: jnp.array([y[1], -y[0]]), jnp.array([1., 0.]),
                        t1=5., dt0=3., ts=jnp.linspace(0., 5., 61), method="dopri8",
                        dense_output=output, event=event, record_steps=True, max_steps=1000)
    standard, output = run("standard"), run("optimized")
    assert standard.status == output.status and output.stats.rejected > 0
    np.testing.assert_array_equal(output.stats, standard.stats)
    np.testing.assert_allclose(output.step_ts, standard.step_ts, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(output.y, standard.y, atol=2e-12, rtol=2e-12)
    assert output.t == pytest.approx(float(standard.t), abs=2e-12)


def test_invalid_dense_output_raises():
    with pytest.raises(ValueError, match="dense_output"):
        solve_rk(lambda t, y, args: y, jnp.ones(1), t1=1., dt0=.1, dense_output="unknown")
