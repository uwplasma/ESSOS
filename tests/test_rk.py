"""Forward solver contracts, including independent Diffrax/analytic references."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import (RKStatus, rk45_interpolate, rk45_step, solve_rk45,
                      dopri8_interpolate, dopri8_step, solve_dopri8)

jax.config.update("jax_enable_x64", True)
methods = pytest.mark.parametrize("solve", [solve_rk45, solve_dopri8], ids=["dopri5", "dopri8"])


def oscillator(t, y, args):
    return jnp.array([y[1], -y[0]])


@methods
@pytest.mark.parametrize("adaptive", [False, True])
def test_unrolled_execution_preserves_events_dense_saves_and_work(solve, adaptive):
    def rhs(t,y,args):
        return jnp.array([.6+.1*jnp.sin(t), -.3*y[1]+.05*jnp.cos(y[0])])
    options = dict(t1=1.,dt0=.17,ts=jnp.linspace(0.,1.,11),adaptive=adaptive,
        rtol=1e-9,atol=1e-9,event=lambda t,y,args:.4-y[0],max_steps=128)
    initial = jnp.array([.2,.5])
    normal = jax.jit(lambda y:solve(rhs,y,**options))(initial)
    unrolled = jax.jit(lambda y:solve(rhs,y,stage_execution="unrolled",**options))(initial)
    assert unrolled.status == normal.status == RKStatus.EVENT
    np.testing.assert_allclose(unrolled.y,normal.y,rtol=0,atol=2e-12)
    np.testing.assert_allclose(unrolled.t,normal.t,rtol=0,atol=2e-12)
    np.testing.assert_allclose(unrolled.ys,normal.ys,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(unrolled.stats,normal.stats)


@methods
def test_fixed_intervals_at_nonzero_times_do_not_add_roundoff_steps(solve):
    dt = 2e-5/224
    starts = jnp.arange(224)*dt
    ends = (jnp.arange(224)+1)*dt
    def one(start,end):
        return solve(lambda t,y,args:jnp.ones_like(y),jnp.zeros(1),
            t0=start,t1=end,dt0=jnp.minimum(dt,end-start),adaptive=False,max_steps=1)
    out = jax.jit(jax.vmap(one))(starts,ends)
    np.testing.assert_array_equal(out.status,RKStatus.SUCCESS)
    np.testing.assert_array_equal(out.stats.accepted,1)
    np.testing.assert_array_equal(out.stats.rejected,0)
    np.testing.assert_array_equal(out.t,ends)
    np.testing.assert_allclose(out.y[:,0],ends-starts,rtol=0,atol=1e-21)


@pytest.mark.parametrize("step_fn, interpolate, solver", [
    (rk45_step, rk45_interpolate, diffrax.Dopri5()),
    (dopri8_step, dopri8_interpolate, diffrax.Dopri8()),
], ids=["dopri5", "dopri8"])
def test_trial_and_dense_output_match_diffrax(step_fn, interpolate, solver):
    rhs = lambda t, y, args: jnp.array([y[1] + jnp.sin(t), -y[0]**2])
    y, t, h = jnp.array([.7, .3]), .2, .4
    term = diffrax.ODETerm(rhs)
    reference, error, info, *_ = solver.step(term, t, t + h, y, None,
                                             solver.init(term, t, t + h, y, None), False)
    trial = jax.jit(lambda y: step_fn(rhs, t, y, h))(y)
    np.testing.assert_allclose(trial.y, reference, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(trial.error, error, atol=2e-16, rtol=2e-12)
    interpolation = solver.interpolation_cls(t0=t, t1=t+h, **info)
    for fraction in (0., .13, .5, .89, 1.):
        np.testing.assert_allclose(interpolate(y, h, trial, fraction),
                                   interpolation.evaluate(t + fraction*h), atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("adaptive", [False, True])
@methods
def test_oscillator_accuracy_and_dense_saves(adaptive, solve):
    ts = jnp.linspace(0., 2*np.pi, 83)
    out = jax.jit(lambda y: solve(oscillator, y, t1=ts[-1], dt0=.07, ts=ts,
                                     adaptive=adaptive, rtol=1e-10, atol=1e-12))(jnp.array([1., 0.]))
    assert out.status == RKStatus.SUCCESS and out.t == ts[-1]
    np.testing.assert_allclose(out.ys, jnp.stack([jnp.cos(ts), -jnp.sin(ts)], axis=1), atol=1e-8, rtol=0.)
    assert out.stats.steps == out.stats.accepted + out.stats.rejected


@methods
def test_adaptation_rejects_trials_and_is_independent_of_save_grid(solve):
    def run(ts):
        return solve(oscillator, jnp.array([1., 0.]), t1=6., dt0=3., ts=ts,
                          atol=1e-11, rtol=1e-10, record_steps=True, max_steps=1000)
    coarse, fine = run(jnp.array([0., 6.])), run(jnp.linspace(0., 6., 101))
    assert coarse.status == RKStatus.SUCCESS and coarse.stats.rejected > 0
    np.testing.assert_array_equal(coarse.y, fine.y)
    np.testing.assert_array_equal(coarse.step_ts, fine.step_ts)
    n = int(coarse.stats.accepted) + 1
    grid = coarse.step_ts[:n]
    assert jnp.all(jnp.diff(grid) > 0) and grid[-1] == 6.
    # Replay exactly the accepted times using the independent solver.
    solver = diffrax.Dopri5() if solve is solve_rk45 else diffrax.Dopri8()
    replay = diffrax.diffeqsolve(diffrax.ODETerm(oscillator), solver,
                                 t0=0., t1=6., dt0=None, y0=jnp.array([1., 0.]),
                                 stepsize_controller=diffrax.StepTo(ts=grid), saveat=diffrax.SaveAt(t1=True))
    np.testing.assert_allclose(coarse.y, replay.ys[-1], atol=3e-14, rtol=3e-14)
    assert solve(oscillator, jnp.array([1., 0.]), t1=1., dt0=.1).step_ts.shape == (0,)


@methods
def test_tightening_tolerance_improves_global_error(solve):
    errors = []
    for tol in (1e-4, 1e-7, 1e-10):
        out = solve(oscillator, jnp.array([1., 0.]), t1=20., dt0=1., atol=tol, rtol=tol)
        assert out.status == RKStatus.SUCCESS
        errors.append(float(jnp.linalg.norm(out.y - jnp.array([jnp.cos(20.), -jnp.sin(20.)]))))
    assert errors[2] < errors[1] / 100 < errors[0] / 10000


@methods
def test_zero_span_duplicate_saves_and_empty_save_grid(solve):
    rhs = lambda t, y, args: jnp.ones_like(y)
    out = solve(rhs, jnp.array([2.]), t0=3., t1=3., dt0=.1, ts=jnp.array([3., 3.]))
    assert out.status == RKStatus.SUCCESS and out.stats.steps == 0
    np.testing.assert_array_equal(out.ys, [[2.], [2.]])
    out = solve(rhs, jnp.array([2.]), t1=.35, dt0=.1, ts=jnp.array([0., 0., .2, .2, .35]), adaptive=False)
    np.testing.assert_allclose(out.ys[:, 0], [2., 2., 2.2, 2.2, 2.35], atol=2e-15)
    assert out.stats.accepted == 4
    out = solve(rhs, jnp.array([2.]), t1=.35, dt0=.1)
    assert out.ys.shape == (0, 1) and out.y[0] == pytest.approx(2.35)


@methods
@pytest.mark.parametrize("span, start", [(2e-7, 0.), (2e-6, 0.), (2e-7, 1e-6), (1., .3)])
def test_fixed_clock_does_not_accumulate_an_extra_endpoint_step(solve, span, start):
    end = start + span
    out = solve(lambda t, y, args: jnp.ones_like(y)*1e6, jnp.ones(1), t0=start,
                t1=end, dt0=(end-start)/256, adaptive=False, record_steps=True, max_steps=300)
    assert out.status == RKStatus.SUCCESS and out.t == end
    assert out.stats.accepted == 256 and out.stats.rejected == 0
    np.testing.assert_allclose(out.y, 1 + (end-start)*1e6, atol=1e-8, rtol=1e-13)
    assert jnp.all(jnp.diff(out.step_ts[:257]) > 0) and out.step_ts[256] == end


@methods
def test_work_limit_and_minimum_step_do_not_force_acceptance(solve):
    out = solve(oscillator, jnp.array([1., 0.]), t1=10., dt0=.1,
                      adaptive=False, max_steps=2, ts=jnp.array([0., 10.]))
    assert out.status == RKStatus.MAX_STEPS and out.stats.steps == 2
    assert out.t == pytest.approx(.2) and jnp.isinf(out.ys[-1]).all()
    out = solve(oscillator, jnp.array([1., 0.]), t1=10., dt0=1., dtmin=.5,
                      atol=1e-14, rtol=1e-14)
    assert out.status == RKStatus.MIN_STEP and out.stats.accepted == 0
    assert out.t == 0. and out.stats.rejected > 0


@pytest.mark.parametrize("adaptive", [False, True])
@methods
def test_nonfinite_rhs_terminates_and_preserves_last_accepted_state(adaptive, solve):
    out = solve(lambda t, y, args: jnp.full_like(y, jnp.nan), jnp.array([1.]),
                      t1=1., dt0=.1, dtmin=.001, adaptive=adaptive, max_steps=100)
    assert out.status == RKStatus.NONFINITE and out.stats.accepted == 0
    assert out.t == 0. and out.y[0] == 1. and out.stats.steps <= 4


@methods
def test_invalid_trial_can_be_rejected_then_recovered(solve):
    # Large trial stages cross y=0; smaller accepted steps remain in the domain.
    rhs = lambda t, y, args: -jnp.sqrt(y)
    out = solve(rhs, jnp.array([1.]), t1=1.8, dt0=1.8, atol=1e-9, rtol=1e-9)
    assert out.status == RKStatus.SUCCESS and out.stats.rejected > 0
    np.testing.assert_allclose(out.y, [.01], atol=3e-9)


@methods
def test_refined_earliest_event_and_unreached_saves(solve):
    rhs = lambda t, y, args: jnp.array([2.])
    events = (lambda t, y, args: y[0] - 1.7, lambda t, y, args: y[0] - .37,
              lambda t, y, args: False)
    out = solve(rhs, jnp.zeros(1), t1=1., dt0=1., event=events, ts=jnp.array([0., .1, .5, 1.]))
    assert out.status == RKStatus.EVENT and out.event_mask == (False, True, False)
    assert out.t == pytest.approx(.185, abs=2e-15)
    np.testing.assert_allclose(out.y, [.37], atol=3e-15)
    np.testing.assert_allclose(out.ys[:2, 0], [0., .2], atol=1e-15)
    assert jnp.isinf(out.ys[2:]).all()


@methods
def test_boolean_endpoint_events_initial_events_and_no_root_finding(solve):
    rhs = lambda t, y, args: jnp.ones_like(y)
    out = solve(rhs, jnp.zeros(1), t1=1., dt0=.3, adaptive=False,
                      event=lambda t, y, args: y[0] > .5)
    assert out.status == RKStatus.EVENT and out.t == pytest.approx(.6)
    for event in (lambda t, y, args: True, lambda t, y, args: y[0]):
        out = solve(rhs, jnp.zeros(1), t1=1., dt0=.3, event=event, ts=[0., 1.])
        assert out.status == RKStatus.EVENT and out.t == 0. and out.stats.steps == 0
        assert jnp.isinf(out.ys).all() and jnp.isinf(out.ts).all()
    out = solve(rhs, jnp.zeros(1), t1=1., dt0=.3, adaptive=False,
                      event=lambda t, y, args: y[0] - .5, root_find=False)
    assert out.t == pytest.approx(.6)


@methods
def test_jit_vmap_particles_stop_independently(solve):
    def run(speed):
        return solve(lambda t, y, args: jnp.ones_like(y)*args, jnp.zeros(1), speed,
                          t1=2., dt0=.7, event=lambda t, y, args: y[0] - 1., max_steps=100)
    out = jax.jit(jax.vmap(run))(jnp.array([1., 2., 4.]))
    np.testing.assert_allclose(out.t, [1., .5, .25], atol=2e-15)
    np.testing.assert_allclose(out.y, 1., atol=3e-15)
    assert jnp.all(out.status == RKStatus.EVENT)


@pytest.mark.parametrize("kwargs", [dict(dt0=0.), dict(t1=-1.), dict(max_steps=0),
                                    dict(atol=0.), dict(dtmin=.2, dtmax=.1),
                                    dict(ts=[.2, .1]), dict(ts=[2.])])
@methods
def test_invalid_solver_inputs_raise(kwargs, solve):
    options = dict(t1=1., dt0=.1)
    options.update(kwargs)
    with pytest.raises(ValueError):
        solve(oscillator, jnp.array([1., 0.]), **options)


@methods
@pytest.mark.parametrize("options", [dict(max_steps=True), dict(dt0=[.1, .2]),
                                      dict(t0=jnp.inf), dict(dtmin=jnp.inf)])
def test_invalid_scalar_inputs_raise(solve, options):
    kwargs = dict(t1=1., dt0=.1)
    kwargs.update(options)
    with pytest.raises(ValueError):
        solve(oscillator, jnp.array([1., 0.]), **kwargs)


@methods
@pytest.mark.parametrize("options", [dict(t1=-1.), dict(dt0=0.), dict(dt0=jnp.nan),
                                      dict(ts=[.8, .2]), dict(ts=[2.]), dict(dtmin=.2)])
def test_invalid_dynamic_inputs_return_status_under_jit(solve, options):
    kwargs = dict(t1=1., dt0=.1, ts=jnp.array([0., 1.]))
    kwargs.update(options)
    kwargs = jax.tree_util.tree_map(jnp.asarray, kwargs)
    out = jax.jit(lambda values: solve(oscillator, jnp.array([1., 0.]), **values))(kwargs)
    assert out.status == RKStatus.INVALID_INPUT and out.stats.steps == 0


@methods
def test_nonfinite_event_is_not_reported_as_a_crossing(solve):
    rhs = lambda t, y, args: jnp.ones_like(y)
    # Initially finite/positive; the first trial lands outside its valid domain.
    event = lambda t, y, args: jnp.where(y[0] > .1, jnp.nan, 1.)
    out = solve(rhs, jnp.zeros(1), t1=1., dt0=.2, event=event, adaptive=False)
    assert out.status == RKStatus.NONFINITE and not out.event_mask
    assert out.t == 0. and out.stats.accepted == 0
    out = solve(rhs, jnp.zeros(1), t1=1., dt0=.2, event=lambda t, y, args: jnp.nan)
    assert out.status == RKStatus.NONFINITE and out.stats.steps == 0


@methods
@pytest.mark.parametrize("adaptive", [False, True])
def test_nonfinite_event_during_root_refinement_is_rejected(solve, adaptive):
    # Both endpoints are finite, but the apparent root lies in an invalid region.
    event = lambda t, y, args: jnp.where((y[0] > .4) & (y[0] < .6), jnp.nan, y[0] - .5)
    out = solve(lambda t, y, args: jnp.ones_like(y), jnp.zeros(1),
                t1=1., dt0=1., dtmin=.01, event=event, adaptive=adaptive, max_steps=100)
    assert out.status == RKStatus.NONFINITE and not out.event_mask
    assert out.y[0] <= .4 and jnp.isfinite(out.y).all() and out.stats.rejected > 0


@methods
@pytest.mark.parametrize("initial", [True, False])
def test_simultaneous_events_have_one_deterministic_mask(solve, initial):
    event = lambda t, y, args: jnp.asarray(initial) | (y[0] > .2)
    out = solve(lambda t, y, args: jnp.ones_like(y), jnp.zeros(1),
                t1=1., dt0=.3, adaptive=False, event=(event, event))
    assert out.status == RKStatus.EVENT and out.event_mask == (True, False)
