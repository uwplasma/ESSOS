"""Accepted-step reverse AD against analytic and frozen-schedule references."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import (solve_rk, rk45_step, dopri8_step, rk45_interpolate,
    dopri8_interpolate, RKStatus)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_unrolled_execution_checkpointed_reverse_matches_its_fixed_map_jvp(method):
    def rhs(t,y,p):
        return p[0]*jnp.array([y[1]+jnp.sin(t),-y[0]**2])+p[1]
    options = dict(t1=.4,dt0=.025,ts=jnp.array([0.,.13,.31,.4]),adaptive=False,
        method=method,stage_execution="unrolled",max_steps=64,adjoint_checkpoint_steps=3,
        adjoint_max_accepted_steps=32)
    def cost(x,adjoint):
        out = solve_rk(rhs,x[:2],x[2:],adjoint=adjoint,**options)
        return jnp.sum(out.ys**2)+.3*jnp.sum(out.y**2)
    x,direction = jnp.array([.7,-.2,.8,.1]),jnp.array([.2,-.1,.3,.4])
    value,gradient = jax.jit(jax.value_and_grad(lambda x:cost(x,"accepted")))(x)
    ordinary,tangent = jax.jit(lambda x,d:jax.jvp(lambda x:cost(x,"none"),(x,),(d,)))(x,direction)
    np.testing.assert_allclose(value,ordinary,rtol=2e-13,atol=1e-13)
    np.testing.assert_allclose(jnp.vdot(gradient,direction),tangent,rtol=2e-11,atol=1e-12)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("controller", ["i", "pi"])
@pytest.mark.parametrize("checkpoint_steps", [1, 3, 16])
def test_reverse_matches_frozen_accepted_steps_and_dense_saves(method, controller, checkpoint_steps):
    ts = jnp.array([0., 0., .17, .43, .79, 1.])
    def rhs(t, y, p):
        return p*jnp.array([y[1]+jnp.sin(t), -y[0]])
    initial = jnp.array([.7, -.2])
    options = dict(t1=1., dt0=1., ts=ts, method=method, controller=controller,
        rtol=1e-10, atol=1e-11, max_steps=100, record_steps=True)
    plain = solve_rk(rhs, initial, .8, **options)
    assert plain.status == RKStatus.SUCCESS and plain.stats.rejected > 0
    times = np.asarray(plain.step_ts)[:int(plain.stats.accepted)+1]
    step = rk45_step if method == "dopri5" else dopri8_step
    dense = rk45_interpolate if method == "dopri5" else dopri8_interpolate

    def replay(x):
        y, parameter = x[:2], x[2]
        saved = [y, y]
        index = 2
        f = rhs(times[0], y, parameter)
        for t, end in zip(times[:-1], times[1:]):
            trial = step(rhs, t, y, end-t, parameter, f0=f)
            while index < len(ts) and float(ts[index]) <= end:
                saved.append(dense(y, end-t, trial, (ts[index]-t)/(end-t)))
                index += 1
            y, f = trial.y, trial.stages[-1]
        return jnp.sum(jnp.stack(saved)**2)+.3*jnp.sum(y**2)

    def objective(x):
        out = solve_rk(rhs, x[:2], x[2], **options, adjoint="accepted",
            adjoint_checkpoint_steps=checkpoint_steps)
        return jnp.sum(out.ys**2)+.3*jnp.sum(out.y**2)
    x = jnp.append(initial, .8)
    value, gradient = jax.jit(jax.value_and_grad(objective))(x)
    np.testing.assert_allclose(value, replay(x), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gradient, jax.grad(replay)(x), rtol=2e-10, atol=2e-11)
    direction = jnp.array([.2, -.3, .4])
    eps = 1e-5
    fd = (replay(x+eps*direction)-replay(x-eps*direction))/(2*eps)
    np.testing.assert_allclose(gradient@direction, fd, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_closed_over_parameters_and_dynamic_controls_are_separate(method):
    def objective(p):
        # Parameter appears in the RHS, tolerance and event closure. Only the
        # RHS path is differentiated under the accepted-step contract.
        rhs = lambda t, y, unused: p[0]*y
        out = solve_rk(rhs, p[1:2], t1=1., dt0=.8, ts=jnp.array([0., .31, 1.]),
            atol=p[2], rtol=p[2], event=lambda t, y, a: 100.+p[3]-y[0],
            method=method, max_steps=100, adjoint="accepted")
        return out.y[0]+out.ys[0, 0]
    p = jnp.array([.7, 1.2, 1e-10, .5])
    gradient = jax.jit(jax.grad(objective))(p)
    expected = jnp.array([1.2*jnp.exp(.7), jnp.exp(.7)+1., 0., 0.])
    np.testing.assert_allclose(gradient, expected, rtol=3e-9, atol=3e-10)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("mode", ["standard", "optimized", "higher_order"])
def test_fixed_solve_dense_output_gradient_matches_finite_difference(method, mode):
    def objective(p, adjoint):
        out = solve_rk(lambda t, y, a: p*jnp.sin(y+t), jnp.array([.4]),
            t1=1., dt0=.2, ts=jnp.linspace(0., 1., 13), adaptive=False,
            method=method, dense_output=mode, max_steps=10, adjoint=adjoint)
        return jnp.sum(out.ys**2)
    value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, "accepted")))(.7)
    eps = 1e-5
    expected = (objective(.7+eps, "none")-objective(.7-eps, "none"))/(2*eps)
    np.testing.assert_allclose(value, objective(.7, "none"), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gradient, expected, rtol=3e-8, atol=3e-9)


def test_vmap_zero_span_and_failed_solve_contract():
    def objective(p):
        out = solve_rk(lambda t, y, a: p*y, jnp.array([1.]), t1=1., dt0=.1,
            max_steps=100, adjoint="accepted")
        return out.y[0]
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(objective)))(jnp.array([.2, .7])),
        jnp.exp(jnp.array([.2, .7])), rtol=2e-8)
    empty = lambda y: jnp.sum(solve_rk(lambda t, y, a: y, y, t0=1., t1=1., dt0=.1,
        ts=jnp.array([1., 1.]), max_steps=2, adjoint="accepted").ys)
    np.testing.assert_array_equal(jax.grad(empty)(jnp.array([2.])), [2.])
    invalid_unused_rhs = lambda y: solve_rk(lambda t,y,a:jnp.sqrt(-y),y,
        t1=0.,dt0=.1,max_steps=2,adjoint="accepted").y[0]
    np.testing.assert_array_equal(jax.grad(invalid_unused_rhs)(jnp.array([2.])),[1.])
    invalid_unused_parameter = lambda p: solve_rk(lambda t,y,a:jnp.sqrt(-p)*y,jnp.ones(1),
        t1=0.,dt0=.1,max_steps=2,adjoint="accepted").y[0]
    np.testing.assert_array_equal(jax.grad(invalid_unused_parameter)(1.),0.)
    failed = lambda p: solve_rk(lambda t, y, a: p*y, jnp.ones(1), t1=1., dt0=.1,
        adaptive=False, max_steps=1, adjoint="accepted").y[0]
    with pytest.raises(Exception, match="successful/event"):
        jax.jit(jax.grad(failed))(.7).block_until_ready()


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_rejected_nonfinite_trials_do_not_poison_reverse(method):
    def objective(p):
        return solve_rk(lambda t, y, a: -p*jnp.sqrt(y), jnp.array([1.]),
            t1=1.8, dt0=1.8, method=method, max_steps=200,
            atol=1e-11, rtol=1e-11, adjoint="accepted").y[0]
    value, derivative = jax.jit(jax.value_and_grad(objective))(1.)
    np.testing.assert_allclose([value, derivative], [.01, -.18], rtol=3e-8, atol=1e-9)


def test_event_time_is_frozen_not_implicitly_differentiated():
    def objective(p):
        return solve_rk(lambda t, y, a: jnp.ones_like(y)*p, jnp.zeros(1),
            t1=1., dt0=1., event=lambda t, y, a: .5-y[0], ts=jnp.array([0., .1]),
            max_steps=10, adjoint="accepted").y[0]
    # With the recorded hit time .25 held fixed, d(p*t_hit)/dp = .25.
    # This deliberately differs from d(boundary_position)/dp = 0.
    value, derivative = jax.jit(jax.value_and_grad(objective))(2.)
    np.testing.assert_allclose([value, derivative], [.5, .25], atol=2e-14)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("mode", ["standard", "optimized", "higher_order"])
def test_segmented_reverse_preserves_partial_segment_event_and_dense_gradients(method, mode):
    def objective(p, segment):
        out = solve_rk(lambda t, y, a: p[0]*jnp.sin(y+t), p[1:2],
            t1=1.5, dt0=.03, ts=jnp.linspace(0., .3, 11), adaptive=False,
            event=lambda t,y,a: 1.-y[0], method=method, dense_output=mode,
            max_steps=60, adjoint="accepted", adjoint_checkpoint_steps=segment)
        return jnp.sum(out.ys**2)+out.y[0]
    p = jnp.array([.7, .4])
    value, reference = jax.jit(jax.value_and_grad(lambda p: objective(p, 1)))(p)
    segmented_value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, 7)))(p)
    np.testing.assert_array_equal(segmented_value, value)
    np.testing.assert_allclose(gradient, reference, rtol=3e-12, atol=3e-12)


def test_segment_checkpoints_store_only_segment_starts_and_keep_identical_schedule():
    rhs = lambda t,y,p: p*jnp.sin(y+t)
    options = dict(t1=1., dt0=.03, adaptive=False, max_steps=40, _record_ad=True)
    full, tape = solve_rk(rhs, jnp.array([.4]), .7, **options)
    segmented, compact = solve_rk(rhs, jnp.array([.4]), .7, **options, _ad_checkpoint_steps=7)
    count = int(full.stats.accepted)
    assert count == 34 and compact.states.shape == (6, 1)
    np.testing.assert_array_equal(segmented.y, full.y)
    np.testing.assert_array_equal(compact.times, tape.times)
    np.testing.assert_array_equal(compact.states[:(count+6)//7], tape.states[:count:7])
    np.testing.assert_array_equal(compact.derivatives[:(count+6)//7], tape.derivatives[:count:7])


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("mode", ["standard", "optimized", "higher_order"])
@pytest.mark.parametrize("initial_fsal", [False, True])
def test_stage_pullback_matches_automatic_vjp_of_original_rk_map(method, mode, initial_fsal):
    from essos.rk import (_trial_step, _C, _A, _ERROR, _C8, _A8, _ERROR8,
        _rk45_dense, _dopri8_dense)
    from essos.rk_ad import _step_pullback, _samples
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    dense = _rk45_dense if method == "dopri5" else _dopri8_dense
    rhs = lambda t,y,p: jnp.array([p[0]*jnp.sin(y[1]+t), p[1]*jnp.cos(y[0]-t)])
    y, f, p = jnp.array([.7,-.2]), jnp.array([.3,.4]), jnp.array([.6,.8])
    t, h, fraction = .2, .13, .73
    fractions = jnp.array([0., .19, fraction, .87])
    owned = jnp.array([False,True,True,False])
    event_save = fractions == fraction
    by, bf = jnp.array([.3,-.4]), jnp.array([-.5,.2])
    bs = jnp.arange(8, dtype=float).reshape(4,2)/10
    def step(y,f,p):
        trial = _trial_step(rhs,t,y,h,p,f,*constants)
        terminal = dense(y,trial.y,trial.increments,fraction)
        saved = _samples(rhs,y,t,h,trial,p,fractions,owned,method,mode)
        if mode == "higher_order":
            saved = jnp.where(event_save[:,None],terminal,saved)
        return terminal,trial.f1,saved
    seed = (by,bf,jnp.where(owned[:,None],bs,0.))
    if initial_fsal:
        f = rhs(t,y,p)
        _, automatic = jax.vjp(lambda y,p: step(y,rhs(t,y,p),p),y,p)
        dy, dp = automatic(seed)
        expected = (dy,jnp.zeros_like(f),dp)
    else:
        _, automatic = jax.vjp(step,y,f,p)
        expected = automatic(seed)
    trial = _trial_step(rhs,t,y,h,p,f,*constants)
    actual = jax.jit(lambda: _step_pullback(rhs,y,t,h,trial.y,trial.increments,p,
        fraction,fractions,owned,event_save,by,bf,bs,method,mode,initial_fsal=initial_fsal))()
    for found, reference in zip(actual, expected):
        np.testing.assert_allclose(found,reference,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("nested", [False, True])
def test_fixed_setup_has_no_pullback_but_nonlinear_parameter_setup_differentiates(method, nested):
    @jax.custom_vjp
    def fixed_setup(value):
        return value[0]
    def forward(value):
        return value[0], ()
    def backward(residual, cotangent):
        raise AssertionError("fixed discretization data must not be differentiated")
    fixed_setup.defvjp(forward,backward)

    def objective(parameter):
        # Created inside the outer jit, so this is a tracer while recording
        # the RHS, but JAX's activity flag correctly identifies it as fixed.
        fixed = jnp.array([.7,.2])
        rhs = lambda t,y,a: fixed_setup(fixed)*jnp.exp(parameter)*y
        if nested:
            rhs = jax.jit(rhs)
        return solve_rk(rhs,jnp.array([.4]),t1=.7,dt0=.03,adaptive=False,
            method=method,max_steps=30,adjoint="accepted").y[0]
    p = .3
    value, gradient = jax.jit(jax.value_and_grad(objective))(p)
    expected = .4*np.exp(.7*.7*np.exp(p))
    np.testing.assert_allclose(value,expected,rtol=3e-10,atol=3e-12)
    np.testing.assert_allclose(gradient,expected*.7*.7*np.exp(p),rtol=3e-10,atol=3e-12)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_batched_checkpoint_replay_handles_finished_and_empty_particles(method):
    initial = jnp.array([.1,1.4,.4])
    horizons = jnp.array([.1,.9,0.])
    def one(p,y,horizon):
        return solve_rk(lambda t,y,a: 2*p*jnp.sin(y+t),jnp.array([y]),
            t1=horizon,dt0=.2,rtol=1e-8,atol=1e-8,method=method,max_steps=128,
            adjoint="accepted",adjoint_checkpoint_steps=3).y[0]
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(jax.vmap(lambda y,h: one(p,y,h))(initial,horizons))))(.7)
    individual = jax.jit(jax.grad(one))
    reference = sum(individual(.7,y,h) for y,h in zip(initial,horizons))
    np.testing.assert_allclose(gradient,reference,rtol=3e-11,atol=3e-12)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_accepted_capacity_changes_storage_without_changing_primal_or_gradient(method):
    options = dict(t1=1., dt0=.2, adaptive=False, max_steps=1000,
                   ts=jnp.linspace(0., 1., 9), method=method)
    def observe(p, capacity):
        out = solve_rk(lambda t,y,a:p*jnp.sin(y+t),jnp.array([.4]), **options,
            adjoint="accepted", adjoint_checkpoint_steps=3,
            adjoint_max_accepted_steps=capacity)
        return jnp.sum(out.ys**2), jnp.stack(out.stats)
    (value, counts), gradient = jax.jit(jax.value_and_grad(lambda p:observe(p,8),has_aux=True))(.7)
    (expected, reference_counts), reference = jax.jit(jax.value_and_grad(lambda p:observe(p,None),has_aux=True))(.7)
    np.testing.assert_array_equal(value,expected)
    np.testing.assert_array_equal(counts,reference_counts)
    np.testing.assert_allclose(gradient,reference,rtol=2e-13,atol=2e-13)
    _, tape = solve_rk(lambda t,y,a:.7*jnp.sin(y+t),jnp.array([.4]), **options,
        _record_ad=True,_ad_compact=True,_ad_checkpoint_steps=3,adjoint_max_accepted_steps=8)
    assert tape.step_sizes.shape == (8,) and tape.states.shape == (3,1)


def test_accepted_capacity_overflow_rejects_gradient_but_keeps_the_forward_solve():
    def observe(p):
        return solve_rk(lambda t,y,a:p*y,jnp.ones(1),t1=1.,dt0=.2,adaptive=False,
            max_steps=1000,adjoint="accepted",adjoint_max_accepted_steps=2)
    out = jax.jit(observe)(.7)
    assert out.status == RKStatus.SUCCESS and int(out.stats.accepted) == 5
    with pytest.raises(Exception,match="accepted-step AD capacity exceeded"):
        jax.jit(jax.grad(lambda p:observe(p).y[0]))(.7).block_until_ready()


@pytest.mark.parametrize("capacity", [0,-1,True,1.5])
def test_accepted_capacity_validation(capacity):
    with pytest.raises(ValueError,match="adjoint_max_accepted_steps"):
        solve_rk(lambda t,y,a:y,jnp.ones(1),t1=1.,dt0=.1,adjoint_max_accepted_steps=capacity)
