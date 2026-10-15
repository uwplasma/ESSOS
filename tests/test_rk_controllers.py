"""Controller history, rejection recovery and independent PI comparisons."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import RKStatus, solve_rk

jax.config.update("jax_enable_x64", True)


def oscillator(t, y, args):
    return jnp.array([y[1], -y[0]])


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("dt0", [.05, 3.])
@pytest.mark.parametrize("pcoeff, icoeff", [(.4, .3), (.2, .5)])
def test_pi_matches_diffrax_with_the_same_coefficients(method, dt0, pcoeff, icoeff):
    y0 = jnp.array([1., 0.])
    options = dict(t0=0., t1=12., dt0=dt0, rtol=1e-7, atol=1e-7)
    native = jax.jit(lambda y: solve_rk(
        oscillator, y, **options, method=method, controller="pi",
        pcoeff=pcoeff, icoeff=icoeff, record_steps=True, max_steps=1000))(y0)
    reference = diffrax.diffeqsolve(
        diffrax.ODETerm(oscillator), diffrax.Dopri5() if method == "dopri5" else diffrax.Dopri8(),
        **{k: v for k, v in options.items() if k not in ("rtol", "atol")}, y0=y0,
        stepsize_controller=diffrax.PIDController(
            rtol=options["rtol"], atol=options["atol"], pcoeff=pcoeff, icoeff=icoeff),
        saveat=diffrax.SaveAt(steps=True), max_steps=1000)
    assert native.status == RKStatus.SUCCESS
    np.testing.assert_array_equal(native.stats, [reference.stats[k] for k in
                                  ("num_steps", "num_accepted_steps", "num_rejected_steps")])
    n = int(native.stats.accepted)
    np.testing.assert_allclose(native.step_ts[1:n+1], reference.ts[:n], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(native.y, reference.ys[n-1], atol=2e-11, rtol=2e-11)
    if dt0 == 3.:
        assert native.stats.rejected > 0


@pytest.mark.parametrize("method, q", [("dopri5", 5), ("dopri8", 8)])
def test_pi_retains_the_last_accepted_error_through_rejections(method, q):
    # A controlled error norm makes the expected time sequence independent of
    # RK roundoff. In particular, initial rejected trials must not enter history.
    norm = lambda y0, y1, error: (jnp.abs(y1[0] - y0[0]) / .2)**q
    out = solve_rk(lambda t, y, args: jnp.ones_like(y), jnp.zeros(1), t1=3., dt0=.8,
                   method=method, error_norm=norm, controller="pi", record_steps=True, max_steps=100)
    t, h, previous, rejected, times = 0., .8, 1., 0, [0.]
    while t < 3.:
        h = min(h, 3. - t)
        error = (h / .2)**q
        accept = error <= 1.
        factor = np.clip(.9 * error**(-.7/q) * previous**(.4/q),
                         1. if accept else .2, 10. if accept else .9)
        if accept:
            t += h
            times.append(t)
            previous = error
        else:
            rejected += 1
        h *= factor
    assert out.status == RKStatus.SUCCESS and out.stats.rejected == rejected > 0
    assert out.stats.accepted == len(times) - 1
    np.testing.assert_allclose(out.step_ts[:len(times)], times, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_pi_handles_zero_error_and_recovers_from_a_nonfinite_trial(method):
    zero = solve_rk(lambda t, y, args: jnp.zeros_like(y), jnp.ones(1), t1=1., dt0=.001,
                    method=method, controller="pi", record_steps=True, max_steps=100)
    assert zero.status == RKStatus.SUCCESS and zero.stats.accepted == 4 and zero.stats.rejected == 0
    np.testing.assert_allclose(zero.step_ts[:5], [0., .001, .011, .111, 1.], atol=2e-15)
    rhs = lambda t, y, args: -jnp.sqrt(y)
    out = solve_rk(rhs, jnp.ones(1), t1=1.8, dt0=1.8, method=method, controller="pi",
                   atol=1e-9, rtol=1e-9)
    assert out.status == RKStatus.SUCCESS and out.stats.rejected > 0
    np.testing.assert_allclose(out.y, [.01], atol=3e-9)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_default_i_and_fixed_steps_are_unchanged(method):
    options = dict(t1=4., dt0=.13, ts=jnp.linspace(0., 4., 19), method=method,
                   record_steps=True, max_steps=200)
    default = solve_rk(oscillator, jnp.array([1., 0.]), **options)
    explicit = solve_rk(oscillator, jnp.array([1., 0.]), controller="i", **options)
    for left, right in zip(jax.tree_util.tree_leaves(default), jax.tree_util.tree_leaves(explicit)):
        np.testing.assert_array_equal(left, right)
    fixed_i = solve_rk(oscillator, jnp.array([1., 0.]), adaptive=False, **options)
    fixed_pi = solve_rk(oscillator, jnp.array([1., 0.]), adaptive=False, controller="pi", **options)
    for left, right in zip(jax.tree_util.tree_leaves(fixed_i), jax.tree_util.tree_leaves(fixed_pi)):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("options", [dict(controller="pid"), dict(pcoeff=-.1), dict(icoeff=0.),
                                    dict(pcoeff=jnp.nan), dict(icoeff=jnp.inf), dict(pcoeff=[.4])])
def test_invalid_pi_configuration_raises(options):
    with pytest.raises(ValueError):
        solve_rk(oscillator, jnp.array([1., 0.]), t1=1., dt0=.1, **(dict(controller="pi") | options))


def test_invalid_dynamic_pi_coefficient_returns_status():
    run = jax.jit(lambda p: solve_rk(oscillator, jnp.array([1., 0.]), t1=1., dt0=.1,
                                   controller="pi", pcoeff=p))
    out = run(jnp.array(-.4))
    assert out.status == RKStatus.INVALID_INPUT and out.stats.steps == 0
