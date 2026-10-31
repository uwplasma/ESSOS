"""Accepted-map reverse AD of orbit and split orbit/collision solves.

Reverse gradients are checked against forward mode through the frozen accepted
schedule (the derivative the rule is meant to deliver), against finite differences
of that map, through the adjoint identity <w, Jv> = <J^T w, v>, against checkpoint
layout, and against exact conservation laws. The persistent GPU reverse kernel is
run on CPU through the Pallas interpreter and compared with shared-JAX reverse.
No long tracing or Monte Carlo runs.
"""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from essos.background_species import BackgroundSpecies
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import Particles
from essos.native_ad import replay_accepted, replay_accepted_split, _OrbitRecorder
from essos.rk import solve_rk
from essos.split import CollisionStep, solve_split, trace_split, move_in_cartesian
from tests.test_native_orbit import UniformField, coil_field
from tests.test_native_split import bath


def kernels():
    def rhs(t, y, p):
        return p[0]*jnp.sin(y+12*t)
    def kick(t, y, h, dw, p):
        return CollisionStep(y-p[2]*h*y+p[1]*(.7+.1*y)*dw, (h/(.04*jnp.exp(p[3])))**2, jnp.array(0.))
    return rhs, kick


def settings(method='dopri8', outer=3, inner=3):
    return dict(t1=.4, dt0=.3, ts=jnp.array([0., 0., .13, .4]), key=jax.random.key(71), noise_dim=1,
        method=method, rtol=1e-8, atol=1e-8, max_steps=64, max_orbit_steps=32, adjoint_checkpoint_steps=outer,
        adjoint_orbit_checkpoint_steps=inner, adjoint_max_accepted_steps=32, adjoint_max_orbit_steps=128)


def cost(ys, y):
    return jnp.sum(ys**2)+.3*jnp.sum(y**2)


@pytest.mark.parametrize('method', ['dopri5', 'dopri8'])
def test_split_reverse_equals_frozen_forward_mode_fd_and_adjoint_identity(method):
    rhs, kick = kernels()
    p, y0, o = jnp.array([.8, .1, .2, 0.]), jnp.array([.7]), settings(method)
    out, tape = jax.jit(lambda p: solve_split(rhs, kick, y0, p, _record_ad=True, **o))(p)
    assert out.status == 0 and out.stats.rejected > 0 and out.stats.orbit_rejected > 0
    frozen = lambda p: replay_accepted_split(rhs, kick, y0, p, tape, o['ts'], method=method)
    def reverse(p, **changes):
        out = solve_split(rhs, kick, y0, p, adjoint='accepted', **(o | changes))
        return out.ys, out.y
    d = jnp.array([.2, -.3, .4, .5])
    # Primal values are those of the production solve; reverse equals forward mode on the frozen schedule.
    np.testing.assert_allclose(jax.jit(reverse)(p)[1], out.y, rtol=3e-13)
    tangent = jax.jit(lambda p, d: jax.jvp(frozen, (p,), (d,))[1])(p, d)
    w = (jnp.cos(jnp.arange(4.))[:, None], jnp.array([1.7]))
    bar = jax.jit(lambda p, w: jax.vjp(reverse, p)[1](w)[0])(p, w)
    np.testing.assert_allclose(bar@d, sum(jnp.vdot(a, b) for a, b in zip(w, tangent)), rtol=3e-10, atol=1e-11)
    assert bar[3] == 0  # the acceptance-only parameter is frozen
    gradient = jax.jit(jax.grad(lambda p: cost(*reverse(p))))(p)
    # Finite differences of the frozen map converge at second order to the reverse derivative.
    errors = []
    for h in (1e-2, 1e-3, 1e-4):
        central = (cost(*frozen(p+h*d))-cost(*frozen(p-h*d)))/(2*h)
        errors.append(abs(central-gradient@d)/abs(gradient@d))
    assert errors[1] < errors[0]/30 and errors[2] < 3e-8
    # Checkpoint layout (including no checkpointing at all) never changes the gradient.
    for outer, inner in ((1, 1), (8, 2), (64, 64)):
        other = jax.jit(jax.grad(lambda p: cost(*reverse(p, adjoint_checkpoint_steps=outer,
            adjoint_orbit_checkpoint_steps=inner))))(p)
        np.testing.assert_allclose(other, gradient, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize('method', ['dopri5', 'dopri8'])
def test_orbit_reverse_with_dense_output_matches_forward_mode_for_every_checkpoint_size(method):
    rhs, y, p = (lambda t, y, p: p*jnp.sin(y+t)), jnp.array([.7]), jnp.array(.3)
    o = dict(t1=.4, dt0=.12, ts=jnp.array([0., .031, .17, .4]), method=method, max_steps=32)
    out, tape = jax.jit(lambda p: solve_rk(rhs, y, p, _recorder=_OrbitRecorder(32, 3), **o))(p)
    def frozen(p):
        saved, terminal = replay_accepted(rhs, y, p, tape, o['ts'], method=method)
        return cost(saved, terminal)
    expected, tangent = jax.jvp(frozen, (p,), (jnp.ones_like(p),))
    for segment in (1, 3, 8):
        value, gradient = jax.jit(jax.value_and_grad(lambda p: cost(*(lambda r: (r.ys, r.y))(solve_rk(
            rhs, y, p, adjoint='accepted', adjoint_checkpoint_steps=segment, **o)))))(p)
        np.testing.assert_allclose(value, expected, rtol=3e-13)
        np.testing.assert_allclose(gradient, tangent, rtol=2e-10)


@pytest.mark.parametrize('method', ['dopri5', 'dopri8'])
def test_gradient_of_a_conserved_energy_is_its_partial_derivative_at_the_start(method):
    # H = y1^2/2 + p y0^2/2 is conserved by y0' = y1, y1' = -p y0, so H(final) = H(start) for all
    # (y0, y1, p): the reverse gradient must equal the partial derivatives of H at t=0.
    def final_energy(x):
        out = solve_rk(lambda t, y, p: jnp.array([y[1], -p*y[0]]), x[:2], x[2], t1=1., dt0=.1,
            ts=jnp.array([1.]), method=method, rtol=1e-10, atol=1e-10, max_steps=1024, adjoint='accepted')
        return out.y[1]**2/2+x[2]*out.y[0]**2/2
    gradient = jax.jit(jax.grad(final_energy))(jnp.array([.7, -.3, 1.3]))
    np.testing.assert_allclose(gradient, [1.3*.7, -.3, .7**2/2], rtol=5e-8)


def test_chart_kick_has_exact_inverse_derivative_including_gauge():
    def position(y):
        u, w = y[0]*jnp.cos(y[5])-y[1]*jnp.sin(y[5]), y[1]*jnp.cos(y[5])+y[0]*jnp.sin(y[5])
        return jnp.array([(3+u)*jnp.cos(y[2]), (3+u)*jnp.sin(y[2]), w])
    y, delta = jnp.array([.2, .1, .3, 2., .4, .2]), jnp.array([.03, -.02, .04])
    def fn(y, delta):
        return position(move_in_cartesian(y, delta, position, implicit=True))
    by, bd = jax.jit(jax.jacrev(fn, argnums=(0, 1)))(y, delta)
    np.testing.assert_allclose(bd, jnp.eye(3), atol=5e-14)
    np.testing.assert_allclose(by, jax.jacfwd(position)(y), atol=8e-14)
    np.testing.assert_array_equal(move_in_cartesian(y, delta, position)[3:], y[3:])


def test_split_events_hold_saves_and_skip_collision_derivatives():
    rhs = lambda t, y, p: jnp.ones_like(y)*p[0]
    kick = lambda t, y, h, dw, p: CollisionStep(y+p[1]*dw, jnp.array(0.), jnp.array(0.))
    o = dict(t1=1., dt0=.5, ts=jnp.array([0., .7, 1.]), key=jax.random.key(1), noise_dim=1,
        event=lambda t, y, p: .2-y[0], adaptive=False, orbit_adaptive=False, max_steps=8, max_orbit_steps=8)
    def run(p):
        out = solve_split(rhs, kick, jnp.zeros(1), p, adjoint='accepted', **o)
        return cost(out.ys, out.y)
    value, g = jax.jit(jax.value_and_grad(run))(jnp.array([1., .3]))
    assert np.isfinite(value) and g[1] == 0
    # The crossing time is frozen: the endpoint sensitivity is that of y = p0 * t_cross.
    np.testing.assert_allclose(g[0], 2*.2*.2*2.3, atol=2e-13)


def test_failed_or_capacity_limited_reverse_raises():
    rhs, kick = kernels()
    o = settings() | dict(adaptive=False, orbit_adaptive=False, dt0=.1, orbit_dt0=.01, adjoint_max_orbit_steps=1)
    f = jax.jit(jax.grad(lambda p: cost(*(lambda r: (r.ys, r.y))(
        solve_split(rhs, kick, jnp.array([.7]), p, adjoint='accepted', **o)))))
    with pytest.raises(Exception, match='capacity exceeded'):
        jax.block_until_ready(f(jnp.array([.8, .1, .2, 0.])))


def test_ad_configuration_rejects_invalid_modes_and_capacities():
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            _OrbitRecorder(8, value)
    for options in (dict(adjoint='bad'), dict(reverse_execution='bad'),
                    dict(adjoint='accepted', adjoint_orbit_checkpoint_steps=0),
                    dict(adjoint='accepted', reverse_execution='independent'),
                    dict(adjoint='accepted', _replay_tapes=object())):
        with pytest.raises(ValueError):
            trace_split(None, None, None, tmax=.1, timestep=.1, **options)


def test_closure_parameters_initial_state_and_custom_jvp_are_differentiated():
    @jax.custom_jvp
    def nonlinear(y):
        return jnp.sin(y)
    @nonlinear.defjvp
    def tangent(primals, tangents):
        (y,), (dy,) = primals, tangents
        return nonlinear(y), 3*jnp.cos(y)*dy
    o = dict(t1=.2, dt0=.1, ts=jnp.array([0., .13, .2]), adaptive=False, max_steps=4)
    def run(p, **kw):
        return solve_rk(lambda t, y, a: p[0]*nonlinear(y), p[1:], **o, **kw)
    p, d = jnp.array([.3, .7]), jnp.array([.2, .4])
    _, tape = jax.jit(lambda p: run(p, _recorder=_OrbitRecorder(4, 2)))(p)
    frozen = lambda p: cost(*replay_accepted(lambda t, y, a: p[0]*nonlinear(y), p[1:], None, tape, o['ts'], method='dopri5'))
    gradient = jax.jit(jax.grad(lambda p: cost(*(lambda r: (r.ys, r.y))(run(p, adjoint='accepted')))))(p)
    np.testing.assert_allclose(gradient@d, jax.jvp(frozen, (p,), (d,))[1], rtol=3e-11)


def test_profile_and_collision_coefficient_derivatives_remain_active():
    from essos.split import collision_coefficients, guiding_center_collision_step
    o = dict(t1=2e-7, dt0=1e-7, ts=jnp.array([0., 2e-7]), key=jax.random.key(33), noise_dim=2, adaptive=False,
        orbit_adaptive=False, orbit_dt0=5e-8, method='dopri5', max_steps=8, max_orbit_steps=8, adjoint_max_orbit_steps=16)
    def problem(p):
        background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.exp(p[0])*jnp.array([[1e20, 2e20]]),
            jnp.array([[1000., 1200.]]), radial_grid=jnp.array([0., 1.]), coulomb_log_model='impact')
        def kick(t, y, h, dw, a):
            coefficients = collision_coefficients(background, M, Q, y[1], jnp.array([y[0], 0., 0.]))
            step = guiding_center_collision_step(y[1], y[2], h, dw, coefficients, speed_floor=1.)
            return step._replace(y=jnp.concatenate((y[:1], step.y)))
        return (lambda t, y, a: jnp.array([.02*y[1]*y[2]/4e5, 0., 0.])), kick, jnp.array([.3+p[1], 4e5*jnp.exp(p[2]), .4])
    p, d = jnp.array([.1, .02, .03]), jnp.array([.1, .2, .1])
    metric = lambda y: y[0]+y[1]/4e5+y[2]
    def reverse(p):
        rhs, kick, y = problem(p)
        return metric(solve_split(rhs, kick, y, adjoint='accepted', **o).y)
    rhs, kick, y = problem(p)
    _, tape = jax.jit(lambda: solve_split(rhs, kick, y, _record_ad=True, **o))()
    def frozen(p):
        rhs, kick, y = problem(p)
        return metric(replay_accepted_split(rhs, kick, y, None, tape, o['ts'], method='dopri5')[1])
    gradient = jax.jit(jax.grad(reverse))(p)
    np.testing.assert_allclose(gradient@d, jax.jvp(frozen, (p,), (d,))[1], rtol=2e-10, atol=1e-12)
    assert abs(gradient[0]) > 1e-8  # the background density is an active parameter


def coil_trace(model, coils, p, adaptive=False, **kwargs):
    from essos.fields import BiotSavart
    field = BiotSavart(coils.with_dofs(coils.dofs.at[0].add(p[0])))
    birth = Particles(jnp.array([[1., .03, .01], [1., .02, 0.]]), mass=M, charge=Q,
        energy=4000*ONE_EV*jnp.exp(p[1]), initial_vparallel_over_v=jnp.array([.5, .3])+p[2],
        field=field, phase_angle_full_orbit=0.)
    return trace_split(field, birth, bath(), model=model, tmax=2e-8, timestep=1e-8, n_save=3, max_steps=64,
        max_orbit_steps=64, spatial_diffusion=model == 'GuidingCenter', adaptive=adaptive, orbit_adaptive=adaptive,
        adjoint_orbit_checkpoint_steps=4,
        adjoint_max_accepted_steps=64, adjoint_max_orbit_steps=256, **kwargs)


coil_metric = lambda saved, final: jnp.sum(saved[..., :3]**2)+jnp.sum(final[..., 3]**2)/1e10
coil_point, coil_direction = jnp.array([.01, .02, .03]), jnp.array([.4, -.5, .3])


@pytest.mark.parametrize('model', ['GuidingCenter', 'FullOrbit'])
def test_coil_split_gradient_matches_frozen_forward_mode(model):
    coils, p, d = coil_field().coils, coil_point, coil_direction
    def objective(p):
        out = coil_trace(model, coils, p, adjoint='accepted')
        return coil_metric(out.ys, out.y)
    value, gradient = jax.jit(jax.value_and_grad(objective))(p)
    _, tape = jax.jit(lambda p: coil_trace(model, coils, p, _record_ad=True))(p)
    frozen = lambda p: coil_metric(*coil_trace(model, coils, p, _replay_tapes=tape))
    np.testing.assert_allclose(value, frozen(p), rtol=1e-12)
    np.testing.assert_allclose(gradient@d, jax.jvp(frozen, (p,), (d,))[1], rtol=5e-9)


@pytest.mark.parametrize('model,adaptive', [('GuidingCenter', True), ('FullOrbit', False)])
def test_persistent_reverse_kernel_gradient_equals_shared_reverse(model, adaptive):
    # The persistent GPU forward/reverse kernels run on CPU through the Pallas interpreter.
    coils = coil_field().coils
    def gradient(**kwargs):
        def objective(p):
            out = coil_trace(model, coils, p, adaptive, adjoint='accepted', **kwargs)
            return coil_metric(out.ys, out.y)
        return jax.jit(jax.value_and_grad(objective))(coil_point)
    shared, independent = gradient(), gradient(execution='independent', interpret=True)
    np.testing.assert_allclose(independent[0], shared[0], rtol=1e-11)
    np.testing.assert_allclose(independent[1], shared[1], rtol=1e-9)


def test_classifier_stop_is_the_surface_classifier_level_and_stops_both_executions_alike():
    from essos.split import ClassifierStop
    from essos.surfaces import SurfaceClassifier, SurfaceRZFourier
    torus = SurfaceRZFourier(jnp.array([1., .1]), jnp.array([0., .1]), 1, 1, 0, ntheta=24, nphi=24, range_torus='full torus')
    classifier = SurfaceClassifier(torus, h=.05, padding=.1)
    stop = ClassifierStop(classifier, .01)
    points = jnp.array([[1.02, .0, .01], [0., 1.05, -.02], [-1., .3, .0], [.5, .5, .03]])
    np.testing.assert_allclose(jax.vmap(stop.evaluate_xyz)(points), jax.vmap(classifier.evaluate_xyz)(points), atol=1e-12)
    np.testing.assert_allclose(jax.vmap(jax.grad(stop.evaluate_xyz))(points),
        jax.vmap(jax.grad(classifier.evaluate_xyz))(points), atol=1e-9)
    field = coil_field()
    birth = Particles(jnp.array([[1., .03, .01]]), mass=M, charge=Q, energy=4000*ONE_EV,
        initial_vparallel_over_v=jnp.array([.5]), field=field)
    kw = dict(tmax=2e-7, timestep=1e-8, n_save=3, max_steps=64, max_orbit_steps=64, adaptive=False,
              orbit_adaptive=False, event=ClassifierStop(classifier, -.05))
    shared = trace_split(field, birth, bath(), **kw)
    independent = trace_split(field, birth, bath(), execution='independent', interpret=True, **kw)
    assert shared.status[0] == 1 and np.all(independent.status == shared.status)
    np.testing.assert_allclose(independent.t, shared.t, rtol=1e-9)


def test_native_tracing_api_and_array_objectives_receive_physical_gradients():
    from essos.dynamics import Tracing
    from essos.particle_losses import (mean_final_radius, classifier_loss_fraction, classifier_loss_location,
        classifier_energy_cost, classifier_line_loss)
    def objective(v):
        p = Particles(jnp.zeros((1, 3)), initial_vparallel_over_v=[.2], mass=1., charge=1., energy=.5,
            initial_xyz_fullorbit=jnp.zeros((1, 3)), initial_vxvyvz=jnp.array([[v, 0., .2]]))
        out = Tracing(field=UniformField(), particles=p, model='FullOrbitAdaptative', integrator='essos_dopri8',
            native_adjoint='accepted', maxtime=.2, timestep=.1, times_to_trace=3, max_steps=16, progress=False)
        return jnp.sum(out.trajectories[0, -1, :3]**2)
    # Uniform B along z: x(t) = (v/B) sin(Bt) with B = 1, so d(x^2)/dv = 2 x sin(t) at t = .2 (x and y parts).
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(jnp.array(.3)), 8*.3*np.sin(.1)**2, rtol=2e-10)
    states = jnp.array([[[0., 0., 0., .2, .3, .4], [.2, 0., 0., .2, .3, .4]],
                        [[0., 0., 0., .2, .3, .4], [.3, 0., 0., .2, .3, .4]]])
    boundary = SimpleNamespace(evaluate_xyz=lambda x: .5-x[0])
    np.testing.assert_allclose(mean_final_radius(states, 0., 0.), .25, atol=2e-12)
    np.testing.assert_allclose(classifier_loss_fraction(states, boundary, softness=4.),
        jnp.mean(jax.nn.sigmoid(jnp.array([-1.2, -.8]))), rtol=1e-14)
    assert .7 < classifier_loss_location(states, jnp.array([1., 0., 0.]), boundary) < 1.
    energy = classifier_energy_cost(states, UniformField(), 1., 1., 'FullOrbit', boundary, kind='perpendicular_jax9')
    np.testing.assert_allclose(energy, .065, rtol=2e-7)
    assert np.any(np.asarray(jax.grad(lambda y: classifier_energy_cost(y, UniformField(), 1., 1., 'FullOrbit',
        boundary))(states))[..., 3:] != 0)
    # Line loss against its definition: probability-weighted escape centroids, distance to the line.
    wall = SimpleNamespace(evaluate_xyz=lambda x: -x[0])
    states = states.at[..., 0].set(jnp.array([[1., 2.], [.5, 1.5]])).at[..., 1].set(jnp.array([[.2, .4], [.3, .7]]))
    probability = jax.nn.sigmoid(30*states[..., 0])
    total = probability.sum(axis=0)
    centroids = (states[..., :3]*probability[..., None]).sum(axis=0)/(total[:, None]+1e-8)
    expected = jnp.sum(jnp.linalg.norm(centroids[:, 1:], axis=1)*(total/2)/(1+10*total/2))
    np.testing.assert_allclose(classifier_line_loss(states, jnp.zeros(3), jnp.array([1., 0., 0.]), wall), expected, rtol=1e-12)


def test_qi_alm_helpers_match_their_definitions():
    from examples.coil_optimization.native_gc_alm import (make_constraints, boundary_line, field_normal_ratio,
        scipy_optimizer, DEFAULTS)
    from essos import augmented_lagrangian as alm
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier, BdotN_over_B
    torus = SurfaceRZFourier(jnp.array([3., 1.]), jnp.array([0., 1.]), 1, 1, 0)
    point, direction = boundary_line(torus, 0., 0., 'fixed_theta')  # outer equator, tangent along phi
    np.testing.assert_allclose(point, [4., 0., 0.], atol=1e-14)
    np.testing.assert_allclose(direction, [0., 1., 0.], atol=1e-14)
    # The native direct-coil B.n/|B| equals the public formula, in value and in the coil-DOF gradient.
    coils = coil_field().coils
    def flux(x, native):
        field = BiotSavart(coils.with_dofs(coils.dofs+x))
        return jnp.sum((field_normal_ratio(torus, field) if native else BdotN_over_B(torus, field))**2)
    expected = jax.jit(jax.value_and_grad(lambda x: flux(x, False)))(jnp.zeros_like(coils.dofs))
    actual = jax.jit(jax.value_and_grad(lambda x: flux(x, True)))(jnp.zeros_like(coils.dofs))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-12)
    # Upper limits enter as "target minus value" (feasible when >= 0); the equality is the particle metric.
    geometry = lambda x: dict(axis=x[0], length=jnp.array([2.])-x[1], curvature=jnp.array([1.])-x[1])
    constraint = make_constraints(geometry, lambda x: x[0]**2, SimpleNamespace(penalty=1., bdotn_weight=0.))
    feasible, violated = jnp.zeros(2), jnp.array([0., 3.])
    multipliers = constraint.init(feasible)
    assert float(constraint.loss(multipliers, feasible)[0]) == 0 < float(constraint.loss(multipliers, violated)[0])
    # The host L-BFGS-B driver solves the multiplier-free box problem and compiles once.
    target = alm.combine(alm.eq(lambda x: x-jnp.array([2., .25])))
    options = SimpleNamespace(**dict(DEFAULTS, collisions=False, inner_maxiter=10))
    params = (jnp.zeros(2), target.init(jnp.zeros(2)))
    optimizer = scipy_optimizer(target, options)
    state, gradient, info = optimizer.init(params)
    compile_seconds = state['compile_s']
    for _ in range(2):
        params, state, gradient, info = optimizer.update(params, state, gradient, info)
        np.testing.assert_allclose(params[0], [1., .25], atol=1e-7)  # clipped to the [-1, 1] box
    assert state['compile_s'] == compile_seconds and len(state['inner_solves']) == 2
