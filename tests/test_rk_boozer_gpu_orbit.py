"""Analytical maps check the independent GPU loop's stopping and retry rules."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def field_and_module():
    from essos.boozer import BoozerField, boozer_compiler_options
    import essos.rk_boozer_gpu_orbit as module
    if not boozer_compiler_options():
        pytest.skip("CUDA-only orbit kernel")
    s = np.linspace(.005, .995, 8)
    field = BoozerField.from_booz(s, np.full((1, len(s)), 5.), [0], [0],
        np.full_like(s, .4), np.full_like(s, 40.), np.zeros_like(s), -40., 1, angle_grid=None)
    return field, module


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_independent_lanes_initial_events_analytic_exit_and_duplicate_saves(monkeypatch, method):
    field, module = field_and_module()
    # Isolate the loop against u(t)=u0+t; the production QA benchmark separately
    # checks the physical RHS against two tighter independent orbit solves.
    monkeypatch.setattr(module, "_kernel_rhs", lambda *args:
        lambda y: jnp.where(jnp.arange(4) == 0, jnp.ones_like(y), jnp.zeros_like(y)))
    from jax._src.pallas.triton import lowering
    monkeypatch.delitem(lowering.triton_lowering_rules, jax.lax.is_finite_p, raising=False)
    times = jnp.array([0., .1, .1, .25, .75, 1.])
    solve = module.prepare_orbit_solver(field, 1., 1., 1., times, .2, 1e-10, 1e-10, 128, method=method)
    births = np.zeros((64, 4))
    births[:, 0] = np.resize([-.8, .5, 1., np.nan], 64)
    result = jax.jit(jax.vmap(solve))(jnp.asarray(births), jnp.zeros(64))
    np.testing.assert_array_equal(result.status, np.resize([0, 1, 1, 4], 64),
        err_msg=f"NaN lane: steps={result.stats.steps[3]}, t={result.t[3]}, y={result.y[3]}")
    for lane in range(0, 64, 4):
        np.testing.assert_allclose(result.ys[lane, :, 0], -.8+times, rtol=0, atol=3e-14)
        np.testing.assert_allclose(result.t[lane+1], .5, rtol=0, atol=3e-14)
        np.testing.assert_allclose(result.y[lane+1, 0], 1., rtol=0, atol=3e-14)
        np.testing.assert_allclose(result.ys[lane+1, :4, 0], .5+times[:4], rtol=0, atol=3e-14)
        assert np.isinf(result.ys[lane+1, 4:]).all()
        assert np.isinf(result.ts[lane+1, 4:]).all()
        assert result.t[lane+2] == 0 and result.stats.steps[lane+2] == 0
        assert np.isinf(result.ys[lane+2]).all()
        assert result.t[lane+3] == 0 and result.stats.steps[lane+3] == 0
    np.testing.assert_array_equal(result.stats.steps, result.stats.accepted+result.stats.rejected)
    np.testing.assert_array_equal(result.ys[:, 1], result.ys[:, 2])


@pytest.mark.parametrize("nonfinite", [False, True])
@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_rejections_preserve_state_and_saved_prefix_at_work_limit(monkeypatch, nonfinite, method):
    field, module = field_and_module()
    def evaluate(y):
        return (jnp.full_like(y, jnp.nan) if nonfinite else
                jnp.where(jnp.arange(4) == 0, 20*y+1., jnp.zeros_like(y)))
    monkeypatch.setattr(module, "_kernel_rhs", lambda *args: evaluate)
    limit = 8 if nonfinite else 1
    solve = module.prepare_orbit_solver(field, 1., 1., 1., jnp.linspace(0., .2, 5),
        .2, 1e-12, 1e-12, limit, method=method)
    births = jnp.tile(jnp.array([.1, 0., 0., 0.]), (64, 1))
    result = jax.jit(jax.vmap(solve))(births, jnp.zeros(64))
    np.testing.assert_array_equal(result.status, 2)
    np.testing.assert_array_equal(result.stats.steps, limit)
    np.testing.assert_array_equal(result.stats.rejected, limit)
    np.testing.assert_array_equal(result.stats.accepted, 0)
    np.testing.assert_array_equal(result.t, 0.)
    np.testing.assert_array_equal(result.y, births)
    np.testing.assert_array_equal(result.ys[:, 0], births)
    assert np.isinf(result.ys[:, 1:]).all()


def test_factory_rejects_unsupported_time_grid_and_wrapper_controller():
    field, module = field_and_module()
    for times in ([.1, .2], [0., .2, .1], [0., np.inf]):
        with pytest.raises(ValueError, match="Save times"):
            module.prepare_orbit_solver(field, 1., 1., 1., times, .1, 1e-9, 1e-9, 32)
    solve = module.prepare_orbit_solver(field, 1., 1., 1., [0., .1], .01, 1e-9, 1e-9, 32)
    from essos.boozer import _native_orbits
    with pytest.raises(ValueError, match="I control"):
        _native_orbits(field, jnp.zeros((64, 4)), jnp.zeros(64), 1., 1., 1., jnp.array([0., .1]),
            .01, 1e-9, 1e-9, "dopri8", True, "pi", .4, .3, "standard", 32,
            _orbit_solver=solve)
    with pytest.raises(ValueError, match="does not match"):
        _native_orbits(field, jnp.zeros((64, 4)), jnp.zeros(64), 1., 1., 1., jnp.array([0., .1]),
            .01, 1e-9, 1e-9, "dopri5", True, "i", .4, .3, "standard", 32, _orbit_solver=solve)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_dense_saves_reproduce_polynomial_trajectory_with_one_accepted_step(monkeypatch, method):
    field, module = field_and_module()
    # u'=1, w'=2u, zeta'=3w, v'=4zeta gives (t,t^2,t^3,1+t^4).
    def evaluate(y):
        u,w,zeta=(module._select_one(y,i,0).reshape(()) for i in range(3))
        index=jnp.arange(4)
        return jnp.where(index==0,jnp.ones_like(y),jnp.where(index==1,2*u,
            jnp.where(index==2,3*w,4*zeta)))
    monkeypatch.setattr(module, "_kernel_rhs", lambda *args: evaluate)
    times=jnp.array([0.,.03,.09,.21,.3])
    solve=module.prepare_orbit_solver(field,1.,1.,1.,times,.3,1e-9,1e-9,8,method=method)
    initial=jnp.tile(jnp.array([0.,0.,0.,1.]),(64,1))
    result=jax.jit(jax.vmap(solve))(initial,jnp.zeros(64))
    expected=jnp.stack((times,times**2,times**3,1+times**4),axis=-1)
    np.testing.assert_allclose(result.ys,jnp.broadcast_to(expected,result.ys.shape),rtol=0,atol=3e-13)
    np.testing.assert_array_equal(result.stats.accepted,1)
    np.testing.assert_array_equal(result.stats.rejected,0)
    np.testing.assert_array_equal(result.status,0)
