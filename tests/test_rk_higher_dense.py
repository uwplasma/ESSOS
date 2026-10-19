"""Order, cost, AD and driver contracts for output-only extra RK stages."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import (RKStatus, _A, _A8, _higher_order_output_coefficients,
                      _polynomial_dense, rk45_step, dopri8_step, solve_rk)
from essos.rk_dense_high import HIGHER_ORDER

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("method,order", [("dopri5", 5), ("dopri8", 7)])
def test_all_continuous_order_conditions_and_endpoint_slopes(method, order):
    from examples.benchmarks.derive_rk_higher_dense_output import conditions, tableau, append_stage
    from examples.benchmarks.derive_rk_dense_output import tree_order, tree_gamma
    nodes, rows, output = HIGHER_ORDER[method]
    original = tableau(_A if method == "dopri5" else _A8)
    a = original.copy()
    for node, row in zip(nodes, rows):
        assert sum(row) == pytest.approx(node, abs=5e-13)
        a = append_stage(a, row)
    trees, matrix = conditions(a, order)
    output = np.asarray(output)[:, ::-1]
    expected = np.array([[float(tree_order(tree) == p)/tree_gamma(tree)
                           for p in range(1, order+1)] for tree in trees])
    np.testing.assert_allclose(matrix @ output, expected, atol=5e-12, rtol=0.)
    endpoint = np.pad(original[-1], (0, len(nodes)))
    np.testing.assert_allclose(output.sum(axis=1), endpoint, atol=1e-11, rtol=0.)
    np.testing.assert_allclose(output[:, 0], np.eye(len(a))[0], atol=1e-13, rtol=0.)
    np.testing.assert_allclose(output @ np.arange(1, order+1),
                               np.eye(len(a))[len(original)-1], atol=1e-11, rtol=0.)


def test_coefficients_can_be_reproduced_without_orbit_data():
    from examples.benchmarks.derive_rk_higher_dense_output import derive
    for method, derived in derive().items():
        nodes, rows, output = HIGHER_ORDER[method]
        np.testing.assert_allclose(nodes, derived["nodes"], atol=1e-14, rtol=0.)
        for actual, expected in zip(rows, derived["rows"]):
            np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(output, derived["coefficients_descending"], atol=2e-7, rtol=2e-8)


def output(rhs, t, y, h, method, fraction, args=None):
    step = (rk45_step if method == "dopri5" else dopri8_step)(rhs, t, y, h, args)
    coefficients = _higher_order_output_coefficients(rhs, t, y, h, h*step.stages, args, method)
    return _polynomial_dense(y, step.y, coefficients, fraction)


@pytest.mark.parametrize("method,power", [(method, power) for method, order in
                         [("dopri5", 5), ("dopri8", 7)] for power in range(order)])
def test_time_dependent_polynomials_through_continuous_order(method, power):
    rhs = lambda t, y, args: jnp.full_like(y, t**power)
    y, t, h = jnp.array([.3]), .2, .7
    points = jnp.linspace(0., 1., 17)
    actual = jax.vmap(lambda s: output(rhs, t, y, h, method, s))(points)
    expected = y + ((t + h*points[:, None])**(power+1) - t**(power+1))/(power+1)
    np.testing.assert_allclose(actual, expected, atol=4e-12, rtol=0.)


@pytest.mark.parametrize("method,min_ratio", [("dopri5", 35), ("dopri8", 90)])
def test_nonlinear_continuous_order(method, min_ratio):
    errors = []
    y = jnp.array([.7])
    points = jnp.array([.17, .37, .59, .83])
    for h in (.6, .3, .15):
        actual = jax.vmap(lambda s: output(lambda t, y, args: y*y, 0., y, h, method, s))(points)
        exact = .7/(1-.7*h*points[:, None])
        errors.append(float(jnp.max(jnp.abs(actual-exact))))
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[1]/errors[2] > min_ratio, errors


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("h", [0., .1])
def test_output_map_forward_reverse_and_finite_difference_derivatives(method, h):
    def evaluate(x):
        rhs = lambda t, y, args: args*jnp.array([y[1]+jnp.sin(t), -y[0]**2])
        return output(rhs, x[2], x[:2], x[3], method, x[5], x[4])
    point = jnp.array([.7, -.3, .2, h, .8, .47])
    reverse = jax.jacrev(evaluate)(point)
    np.testing.assert_allclose(jax.jacfwd(evaluate)(point), reverse, atol=3e-10, rtol=2e-9)
    eps = 1e-5
    finite = jnp.stack([(evaluate(point+eps*d)-evaluate(point-eps*d))/(2*eps)
                        for d in jnp.eye(len(point))], axis=1)
    np.testing.assert_allclose(reverse, finite, atol=3e-8, rtol=3e-6)


@pytest.mark.parametrize("method,per_trial,extra", [("dopri5", 6, 2), ("dopri8", 13, 3)])
@pytest.mark.parametrize("saves", [[], [0., 1., 2., 3.], [0., .2, .4, .8, 1., 1.5, 2., 3.]])
def test_extra_rhs_only_once_per_accepted_step_with_interior_saves(method, per_trial, extra, saves):
    calls = []
    def rhs(t, y, args):
        jax.debug.callback(lambda t: calls.append(float(t)), t, ordered=True)
        return jnp.array([y[1], -y[0]])
    result = jax.jit(lambda y: solve_rk(rhs, y, t1=3., dt0=1., adaptive=False,
        method=method, dense_output="higher_order", ts=jnp.asarray(saves)))(jnp.array([1., 0.]))
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert result.status == RKStatus.SUCCESS and result.stats.accepted == 3
    output_steps = 2 if len(saves) > 4 else 0
    assert len(calls) == 1 + per_trial*3 + extra*output_steps


@pytest.mark.parametrize("method,per_trial,extra", [("dopri5", 6, 2), ("dopri8", 13, 3)])
def test_rejected_steps_do_not_evaluate_output_rhs(method, per_trial, extra):
    calls = []
    def rhs(t, y, args):
        jax.debug.callback(lambda t: calls.append(float(t)), t, ordered=True)
        return jnp.array([y[1], -y[0]])
    saves = np.linspace(0., 3., 101)
    result = jax.jit(lambda y: solve_rk(rhs, y, t1=3., dt0=2., method=method,
        dense_output="higher_order", record_steps=True, max_steps=1000,
        ts=jnp.asarray(saves)))(jnp.array([1., 0.]))
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert result.status == RKStatus.SUCCESS and result.stats.rejected > 0
    endpoints = np.asarray(result.step_ts[:int(result.stats.accepted)+1])
    output_steps = sum(np.any((saves > lo) & (saves < hi)) for lo, hi in zip(endpoints[:-1], endpoints[1:]))
    assert len(calls) == 1 + per_trial*int(result.stats.steps) + extra*output_steps


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("event", [None, lambda t, y, args: y[0]-.5])
def test_same_orbit_steps_rejections_and_event_root(method, event):
    def run(mode):
        return solve_rk(lambda t, y, args: jnp.array([y[1], -y[0]]), jnp.array([1., 0.]),
            t1=5., dt0=3., ts=jnp.linspace(0., 5., 61), method=method,
            dense_output=mode, event=event, record_steps=True, max_steps=1000)
    standard, higher = run("standard"), run("higher_order")
    assert standard.status == higher.status and higher.stats.rejected > 0
    np.testing.assert_array_equal(higher.stats, standard.stats)
    np.testing.assert_allclose(higher.step_ts, standard.step_ts, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(higher.y, standard.y, atol=2e-12, rtol=2e-12)
    assert higher.t == pytest.approx(float(standard.t), abs=2e-12)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_output_failure_is_reported_without_changing_the_accepted_step(method):
    node = HIGHER_ORDER[method][0][0]
    rhs = lambda t, y, args: jnp.full_like(y, jnp.where(jnp.abs(t-node) < 1e-14, jnp.inf, 1.))
    options = dict(t1=1., dt0=1., adaptive=False, method=method, dense_output="higher_order")
    result = solve_rk(rhs, jnp.array([0.]), ts=jnp.array([.5, 1.]), **options)
    assert result.status == RKStatus.NONFINITE
    assert result.stats.accepted == 1 and result.t == 1.
    np.testing.assert_allclose(result.y, 1., atol=1e-13, rtol=0.)
    assert np.isinf(result.ys).all()
    terminal_only = solve_rk(rhs, jnp.array([0.]), ts=jnp.array([1.]), **options)
    assert terminal_only.status == RKStatus.SUCCESS
    np.testing.assert_allclose(terminal_only.ys, [[1.]], atol=1e-13, rtol=0.)
