import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.dynamics import Tracing, _from_axis_regular
from essos.orbit_error import OrbitErrorNorm
from essos.rk import RKStatus, dopri8_step, rk45_step, solve_rk


def test_each_physical_block_has_its_own_budget():
    norm = OrbitErrorNorm(.1, .01, speed_scale=10., length_scale=2.)
    y = jnp.zeros(6)  # position, vpar, constant mu, failure flag
    for error in (jnp.array([.22, 0., 0., 0., 1e5, 1e5]),
                  jnp.array([0., 0., 0., 1.1, 0., 0.]),
                  jnp.array([.22, 0., 0., 1.1, 0., 0.])):
        np.testing.assert_allclose(norm(y, y, error), 1., atol=1e-15)
    assert norm(y, y, jnp.array([.22, .22, 0., 0., 0., 0.])) > 1


def test_cartesian_error_is_translation_rotation_and_units_invariant():
    norm = OrbitErrorNorm(1e-6, 1e-8, speed_scale=10., length_scale=2., velocity_components=3)
    y = jnp.array([4., -3., 2., 0., 1., -2.])
    error = jnp.array([2e-6, -1e-6, 3e-6, 4e-5, 0., -1e-5])
    expected = norm(y, y, error)
    rotation = jnp.array([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]])
    rotated = jnp.concatenate([rotation @ error[:3], rotation @ error[3:]])
    np.testing.assert_allclose(norm(y + 1000., y + 1000., rotated), expected, rtol=1e-15)
    unit_scale = jnp.array([1000.] * 3 + [.01] * 3)
    changed_units = OrbitErrorNorm(1e-6, 1e-8, speed_scale=.1, length_scale=2000., velocity_components=3)
    np.testing.assert_allclose(changed_units(y * unit_scale, y * unit_scale, error * unit_scale), expected)


def test_parallel_turning_point_uses_total_speed_reference():
    norm = OrbitErrorNorm(1e-6, 1e-8, speed_scale=1e6)
    error = jnp.array([0., 0., 0., .5])
    at_turn = jnp.zeros(4)
    away = at_turn.at[3].set(1e6)
    np.testing.assert_allclose(norm(at_turn, at_turn, error), norm(away, away, error))
    assert norm(at_turn, at_turn, error) < 1


def torus_position(y):
    s, theta, phi = _from_axis_regular(y)[:3]
    radius = 2 + jnp.sqrt(s) * jnp.cos(theta)
    return jnp.array([radius * jnp.cos(phi), radius * jnp.sin(phi), jnp.sqrt(s) * jnp.sin(theta)])


@pytest.mark.parametrize("radius", [0., 1e-6, .4])
def test_regular_chart_handles_axis_and_angle_wrapping(radius):
    y = jnp.array([radius, 0., .2, 1., 2 * jnp.pi - 1e-8])
    error = jnp.array([0., 0., 0., 0., -2e-8])
    norm = OrbitErrorNorm(1e-8, 1e-8, 1., position_map=torus_position)
    expected = np.linalg.norm(np.asarray(torus_position(y) - torus_position(y - error))) / 2e-8
    np.testing.assert_allclose(jax.jit(norm)(y, y, error), expected, atol=1e-8)
    assert np.isfinite(expected)
    if radius:
        assert expected > 0  # The auxiliary chart angle is physical, not a flag.


def test_segment_layout_maps_chart_but_excludes_mu_and_failure_flag():
    trace = object.__new__(Tracing)
    trace.native_error_control, trace._axis_regular = "physics", True
    trace.rtol = trace.atol = 1e-8
    trace.orbit_length_scale, trace.orbit_speed_scale = 1., 1.
    trace.model = "GuidingCenterAdaptative"
    trace.field = type("Torus", (), {"to_xyz": staticmethod(lambda q: jnp.array([
        (2 + jnp.sqrt(q[0]) * jnp.cos(q[1])) * jnp.cos(q[2]),
        (2 + jnp.sqrt(q[0]) * jnp.cos(q[1])) * jnp.sin(q[2]),
        jnp.sqrt(q[0]) * jnp.sin(q[1])]))})()
    y = jnp.array([.4, 0., .2, 0., .3, 0.])
    error = jnp.array([0., 0., 0., 0., 1e-8, 1e6])
    inside = trace._native_error_norm(flag=True)
    np.testing.assert_allclose(inside(y, y, error), .2, atol=1e-8)
    outside = trace._native_error_norm(exterior=True, flag=True)
    assert outside(y, y, error) == 0  # Here index 4 is constant mu.


@pytest.mark.parametrize("options", [dict(speed_scale=0.), dict(length_scale=-1.), dict(atol=0.),
                                      dict(rtol=-1.), dict(length_scale=np.nan), dict(speed_scale=[1., 2.])])
def test_invalid_physical_scales_raise(options):
    values = dict(rtol=1e-9, atol=1e-9, speed_scale=1.)
    with pytest.raises(ValueError, match="orbit"):
        OrbitErrorNorm(**(values | options))


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_physical_norm_controls_integration_and_is_independent_of_origin(method):
    def solve(offset, tolerance):
        rhs = lambda t, y, args: jnp.concatenate([y[3:], jnp.cross(y[3:], jnp.array([0., 0., 1.]))])
        y = jnp.array([offset, 0., 0., 0., 1., 0.])
        norm = OrbitErrorNorm(tolerance, tolerance, 1., velocity_components=3)
        return solve_rk(rhs, y, t1=10., dt0=.1, ts=jnp.linspace(0., 10., 21),
                        error_norm=norm, method=method, max_steps=10000)
    loose, tight, shifted = [jax.jit(solve, static_argnums=1)(offset, tol)
                             for offset, tol in [(0., 1e-5), (0., 1e-9), (100., 1e-9)]]
    t = np.linspace(0., 10., 21)
    expected = np.column_stack([1 - np.cos(t), np.sin(t), np.zeros_like(t)])
    errors = [np.max(np.linalg.norm(np.asarray(out.ys[:, :3]) - expected, axis=1)) for out in (loose, tight)]
    assert errors[1] < errors[0] * .1
    assert tight.status == shifted.status == RKStatus.SUCCESS
    np.testing.assert_allclose(shifted.ys[:, :3] - np.array([100., 0., 0.]), tight.ys[:, :3], atol=1e-10)


@pytest.mark.parametrize("step", [rk45_step, dopri8_step])
def test_pure_rk_map_keeps_forward_and_reverse_ad_consistent(step):
    def final_state(parameter):
        rhs = lambda t, y, args: parameter * jnp.array([y[1], -y[0]])
        y = jnp.array([1., .3])
        return step(rhs, 0., y, .05, f0=rhs(0., y, None)).y
    np.testing.assert_allclose(jax.jacfwd(final_state)(.7), jax.jacrev(final_state)(.7), rtol=1e-12, atol=1e-14)
    finite_difference = (final_state(.7 + 1e-5) - final_state(.7 - 1e-5)) / 2e-5
    np.testing.assert_allclose(jax.jacrev(final_state)(.7), finite_difference, rtol=1e-8, atol=1e-10)


def test_fixed_steps_skip_error_policy_and_invalid_adaptive_norm_is_rejected():
    rhs = lambda t, y, args: y
    def forbidden(*args):
        raise AssertionError("fixed steps evaluated the error policy")
    fixed = solve_rk(rhs, jnp.ones(1), t1=1., dt0=.1, adaptive=False, error_norm=forbidden)
    assert fixed.status == RKStatus.SUCCESS
    invalid = solve_rk(rhs, jnp.ones(1), t1=1., dt0=.1, dtmin=.1, error_norm=lambda *args: jnp.nan)
    assert invalid.status == RKStatus.NONFINITE and invalid.stats.accepted == 0
