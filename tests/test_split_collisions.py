"""Independent physics, Brownian conditioning and forward splitting checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.background_species import BackgroundSpecies, nu_D_ab, nu_par_ab, nu_s_ab
from essos.collisions import (CollisionCoefficients, CollisionStep, brownian_memory,
    brownian_propose, brownian_accept, collision_coefficients, guiding_center_collision_step,
    particle_collision_step, reflect_pitch, velocity_basis)
from essos.collisions import brownian_sample, brownian_consume, paper_collision_step_size
from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q
from essos.split import solve_split, SplitStatus, trace_split
from essos.collision_error import CollisionErrorModel, collision_error_estimates, collision_error_passes
from essos.collision_controller import candidate_collision_step_size, _fit_error_step


def background(density=1e20):
    return BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
                             jnp.array([density]), jnp.array([1e3]))


@pytest.mark.parametrize("v", [2e4, 3e5, 2e6])
def test_coefficients_match_independent_diffusion_divergence(v):
    species, point = background(), jnp.array([.3, .2, .1])
    def diffusion(velocity):
        speed = jnp.linalg.norm(velocity)
        direction = velocity/speed
        dp = .5*speed**2*nu_par_ab(M, Q, 0, speed, point, species)
        dt = .5*speed**2*nu_D_ab(M, Q, 0, speed, point, species)
        return dt*jnp.eye(3)+(dp-dt)*jnp.outer(direction, direction)
    velocity = jnp.array([.3, -.4, np.sqrt(.75)])*v
    D = diffusion(velocity)
    divergence = jnp.einsum('ijj->i', jax.jacfwd(diffusion)(velocity))
    drift = -nu_s_ab(M, Q, 0, v, point, species)*velocity + divergence
    c = collision_coefficients(species, M, Q, jnp.asarray(v), point, full_orbit=True)
    expected = (c.friction+c.parallel_derivative+2*(c.parallel-c.perpendicular)/v)*velocity/v
    np.testing.assert_allclose(expected, drift, rtol=3e-11)
    basis = velocity_basis(velocity)
    sigma = basis @ jnp.diag(jnp.sqrt(2*jnp.array([c.perpendicular, c.perpendicular, c.parallel])))
    np.testing.assert_allclose(sigma@sigma.T/2, D, rtol=1e-12)
    def scalar_drift(speed):
        vel = velocity/v*speed
        div = jnp.einsum('ijj->i', jax.jacfwd(diffusion)(vel))
        return jnp.dot(-nu_s_ab(M, Q, 0, speed, point, species)*vel+div, velocity/v)
    np.testing.assert_allclose(c.particle_drift_derivative, jax.grad(scalar_drift)(jnp.asarray(v)), rtol=1e-10)
    # Maxwellian detailed balance: Q = D_parallel d(log f_M)/dv.
    np.testing.assert_allclose(c.friction, -M*v*c.parallel/(1000*Q), rtol=2e-14)


def test_gc_map_and_error_estimates_match_paper_equations():
    c = CollisionCoefficients(.8, .4, .12, -.3, .07, 0.)
    v, xi, h, tol = 2., -.4, .03, .004
    dw = jnp.array([.1, -.2])
    got = guiding_center_collision_step(v, xi, h, dw, c, tolerance=tol)
    nu, ku = .2, -.3+.12+.8
    expected_v = v+ku*h+np.sqrt(1.6)*dw[0]+.06*(dw[0]**2-h)
    expected_xi = xi-xi*nu*h+np.sqrt((1-xi*xi)*nu)*dw[1]-.5*xi*nu*(dw[1]**2-h)
    np.testing.assert_allclose(got.y, [expected_v, expected_xi], rtol=1e-14)
    eps_v = tol*(abs(ku)*h+np.sqrt(1.6*h))
    ed = max(.5*abs(-.3*.07)*h*h/eps_v, .5*abs(xi)*nu**2*h*h/tol)
    es = max(.12**2*abs(dw[0])**3/(6*eps_v*np.sqrt(.8)),
        .5*np.sqrt(1-xi*xi)*nu**1.5*h*(abs(dw[1])+np.sqrt(h/3))/tol)
    np.testing.assert_allclose([got.drift_error, got.diffusion_error], [ed, es], rtol=1e-14)


@pytest.mark.parametrize("pitch_dw", [-.1, 0., .1])
def test_gc_pitch_error_uses_paper_eq55_magnitude_estimate(pitch_dw):
    # Isolate pitch diffusion. The text following Eq. (55) approximates
    # |J| = h/2 * (|dW| + sqrt(h/3)); a negative dW must not cancel the
    # second term as a literal reading of printed Eq. (75) would allow.
    c = CollisionCoefficients(.8, .4, 0., 0., 0., 0.)
    v, xi, h, tolerance = 2., .3, .03, .004
    got = guiding_center_collision_step(v, xi, h, jnp.array([0., pitch_dw]), c,
                                       tolerance=tolerance)
    nu = 2*c.perpendicular/v**2
    expected = .5*np.sqrt(1-xi**2)*nu**1.5*h*(abs(pitch_dw)+np.sqrt(h/3))/tolerance
    assert got.diffusion_error == pytest.approx(expected, rel=1e-14)
    assert got.diffusion_error > 0


@pytest.mark.parametrize("gc, vacuum", [(True, False), (False, False), (True, True), (False, True)])
def test_candidate_error_model_matches_actual_kicks(gc, vacuum):
    c = CollisionCoefficients(*([0.]*6 if vacuum else [.8, .4, .12, -.3, .07, .02]))
    for h, dw in [(1e-4, [-.02, .1, -.3]), (.03, [.1, -.2, .4]), (.2, [0., 0., 0.])]:
        dw = jnp.asarray(dw)
        if gc:
            actual = guiding_center_collision_step(2., -.4, h, dw[-2:], c,
                tolerance=.004, return_error_model=True)
            # Spatial GC noise is deliberately absent from these error estimates.
            noise = jnp.concatenate([jnp.array([100., -300., 500.]), dw[-2:]])
        else:
            actual = particle_collision_step(jnp.array([1.2, 0., 1.6]), h, dw, c,
                tolerance=.004, return_error_model=True)
            noise = dw
        predicted = collision_error_estimates(actual.error_model, h, noise)
        np.testing.assert_allclose(predicted, [actual.drift_error, actual.diffusion_error], rtol=2e-14)


def test_candidate_uses_both_errors_instead_of_uncontrolled_noise_norm():
    c = CollisionCoefficients(.8, .4, 0., 0., 0., 0.)
    previous = guiding_center_collision_step(2., .3, .03, jnp.array([0., .1]), c,
        tolerance=.004, return_error_model=True)
    m = brownian_memory(jax.random.key(11), 2, capacity=8)
    ends = jnp.array([.01, .02, .03, .04])
    values = jnp.array([[.3, .02], [.4, .04], [.5, .06], [.6, .08]])
    increments = jnp.diff(jnp.concatenate([jnp.zeros((1, 2)), values]), axis=0)
    m = m._replace(ends=m.ends.at[:4].set(ends[::-1]),
        increments=m.increments.at[:4].set(increments[::-1]), size=jnp.array(4), peak=jnp.array(4))
    old, paper_h, _ = paper_collision_step_size(m, .03, jnp.array([0., .1]),
        previous.drift_error, previous.diffusion_error, jnp.array(True), dtmin=0., dtmax=.04, barrier=.04)
    new, candidate_h, full = candidate_collision_step_size(m, .03, previous.error_model,
        jnp.array(True), dtmin=0., dtmax=.04, barrier=.04)
    assert not full and paper_h == pytest.approx(.01) and candidate_h == pytest.approx(.04)
    ed, es = collision_error_estimates(previous.error_model, candidate_h, values[-1])
    assert max(ed, es) == pytest.approx(.08339037495)
    for before, after in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
        if jax.dtypes.issubdtype(before.dtype, jax.dtypes.prng_key):
            before, after = jax.random.key_data(before), jax.random.key_data(after)
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("drift, expected", [(1., .9**(2/3)), (0., 1.)])
def test_candidate_deterministic_error_root_and_cap_need_no_random_samples(drift, expected):
    # eps_v=sqrt(h), E_drift=drift*h**1.5; known analytic root.
    model = CollisionErrorModel(0., .5, drift, 0., 0., 0., 1., jnp.array(0))
    m = brownian_memory(jax.random.key(3), 1)
    new, h, full = candidate_collision_step_size(m, .5, model, jnp.array(True),
        dtmin=0., dtmax=2., barrier=1.)
    assert not full and expected-1/256 <= h <= expected+1e-15
    assert new.size == 0
    np.testing.assert_array_equal(jax.random.key_data(new.key), jax.random.key_data(m.key))


@pytest.mark.parametrize("gc", [False, True])
@pytest.mark.parametrize("normal", [.2, 1.5, 5.])
def test_candidate_fit_obeys_original_normalized_error_limits(gc, normal):
    c = CollisionCoefficients(.8, .4, .12, -.3, .07, .02)
    if gc:
        model = guiding_center_collision_step(2., .3, .03, jnp.zeros(2), c,
            tolerance=.004, return_error_model=True).error_model
    else:
        model = particle_collision_step(jnp.array([1.2, 0., 1.6]), .03, jnp.zeros(3), c,
            tolerance=.004, return_error_model=True).error_model
    z = jnp.full((2 if gc else 3,), normal)
    bound, minimum, target = 1., .0002, .9
    h = _fit_error_step(model, bound, minimum, z, target)
    ed, es = collision_error_estimates(model, h, z*jnp.sqrt(h))
    assert max(ed, es) <= target+1e-14
    # If not capped, the next bisection grid point must exceed the limit.
    if h < bound:
        higher = h+(bound-minimum)/256
        ed, es = collision_error_estimates(model, higher, z*jnp.sqrt(higher))
        assert max(ed, es) > target-1e-14


@pytest.mark.parametrize("controller", ["candidate", "paper"])
def test_lookahead_rejection_preserves_all_terminal_brownian_parents(controller):
    # 32 toy paths, with a guaranteed rejected first trial at t=1.
    model = CollisionErrorModel(0., .5, .5, .3, 0., 0., .2, jnp.array(0))
    def one(key):
        def kick(t, y, h, dw, args):
            ed, es = collision_error_estimates(model, h, dw)
            return CollisionStep(y+dw, ed, es, model)
        out = solve_split(lambda t, y, args: jnp.zeros_like(y), kick, jnp.zeros(1),
            t1=1., dt0=1., dtmax=1., ts=jnp.array([1.]), key=key, noise_dim=1,
            method="dopri5", collision_controller=controller, max_steps=256)
        _, draw = jax.random.split(key)
        return out, jax.random.normal(draw, (1,), dtype=jnp.float64)
    out, parents = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(25), 32))
    assert np.all(out.status == SplitStatus.SUCCESS)
    assert np.all(out.stats.rejected >= 1)
    assert np.max(out.stats.max_accepted_error) <= 1
    np.testing.assert_allclose(out.y, parents, atol=1e-12, rtol=0.)


def test_candidate_requires_metadata_and_never_overrides_actual_acceptance():
    empty = CollisionErrorModel(0., 0., 0., 0., 0., 0., 1., jnp.array(0))
    def solve(model, **options):
        def kick(t, y, h, dw, args):
            # Deliberately optimistic surrogate; actual trial must still fail.
            return CollisionStep(y+dw, jnp.array(4.), jnp.array(0.), model)
        return solve_split(lambda t, y, args: y*0, kick, jnp.zeros(1),
            t1=1., dt0=.5, ts=jnp.array([1.]), key=jax.random.key(1), noise_dim=1,
            collision_controller="candidate", max_steps=3, **options)
    with pytest.raises(ValueError, match="requires CollisionStep.error_model"):
        solve(None)
    out = solve(empty)
    assert out.status == SplitStatus.MAX_STEPS
    assert out.stats.accepted == 0 and out.stats.rejected == 3
    np.testing.assert_array_equal(out.y, [0.])
    out = solve(empty, dtmin=.5)
    assert out.status == SplitStatus.MIN_STEP and out.stats.accepted == 0


def test_candidate_capacity_failure_retains_existing_brownian_values():
    m, _, _, _ = brownian_propose(brownian_memory(jax.random.key(3), 1, capacity=1), 1.)
    model = CollisionErrorModel(0., .5, 0., .3, 0., 0., .2, jnp.array(0))
    new, _, full = candidate_collision_step_size(m, 1., model, jnp.array(False),
        dtmin=0., dtmax=1., barrier=1.)
    assert full  # A bridge point inside the retained parent cannot fit.
    for before, after in zip(jax.tree.leaves(m), jax.tree.leaves(new)):
        if jax.dtypes.issubdtype(before.dtype, jax.dtypes.prng_key):
            before, after = jax.random.key_data(before), jax.random.key_data(after)
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("gc, vacuum", [(True, False), (False, False), (True, True), (False, True)])
def test_candidate_probe_predicate_matches_actual_normalized_errors(gc, vacuum):
    c = CollisionCoefficients(*([0.]*6 if vacuum else [.8, .4, .12, -.3, .07, .02]))
    for h, noise in [(.001, [.1, -.2, .3]), (.03, [1., .4, -.8]), (1., [-2., 3., -4.])]:
        dw = jnp.array(noise[-2:] if gc else noise)
        actual = (guiding_center_collision_step(2., -.4, h, dw, c,
                    tolerance=.004, return_error_model=True) if gc else
                  particle_collision_step(jnp.array([1.2, 0., 1.6]), h, dw, c,
                    tolerance=.004, return_error_model=True))
        error = float(max(actual.drift_error, actual.diffusion_error))
        for target in [.9, 1., .5*error+1e-12, 2*error+1e-12]:
            assert bool(collision_error_passes(actual.error_model, h, dw, target=target)) == (error <= target)


@pytest.mark.parametrize("noise", [0., 1.])
def test_candidate_probe_handles_zero_diffusion_denominator(noise):
    model = CollisionErrorModel(.1, 0., 0., .1, 0., 0., .004, jnp.array(0))
    assert bool(collision_error_passes(model, .03, jnp.array([noise]))) == (noise == 0.)


def test_candidate_orbit_event_skips_collision():
    model = CollisionErrorModel(0., 0., 0., 0., 0., 0., 1., jnp.array(0))
    collision = lambda t, y, h, dw, args: CollisionStep(jnp.full_like(y, jnp.nan),
        jnp.array(0.), jnp.array(0.), model)
    out = solve_split(lambda t, y, args: jnp.ones_like(y), collision, jnp.zeros(1),
        t1=1., dt0=.5, ts=jnp.array([0., .1, 1.]), key=jax.random.key(1), noise_dim=1,
        collision_controller="candidate", event=lambda t, y, args: .05-y[0])
    assert out.status == SplitStatus.EVENT
    assert out.t == pytest.approx(.05, abs=2e-14)
    np.testing.assert_allclose(out.ys[:, 0], [0., .05, .05], atol=2e-14)


@pytest.mark.parametrize("controller", ["conservative", "paper", "candidate"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_unused_candidate_metadata_is_allowed_with_other_step_modes(controller, adaptive):
    model = CollisionErrorModel(0., 0., 0., 0., 0., 0., 1., jnp.array(0))
    def collision(t, y, h, dw, args):
        return CollisionStep(y+h, jnp.array(0.), jnp.array(0.), model)
    out = solve_split(lambda t, y, args: y*0, collision, jnp.zeros(1),
        t1=1., dt0=.25, ts=jnp.array([1.]), key=jax.random.key(1), noise_dim=1,
        collision_controller=controller, adaptive=adaptive, dtmax=.25)
    assert out.status == SplitStatus.SUCCESS
    assert out.stats.accepted == 4
    np.testing.assert_allclose(out.y, [1.], atol=2e-15)


def test_reflection_handles_large_overshoots_without_clipping():
    np.testing.assert_allclose(reflect_pitch(jnp.array([-5.2, -1.2, -.3, 1.2, 2.4, 7.3])),
                               [-.8, -.8, -.3, .8, -.4, -.7], atol=1e-14)
    c = CollisionCoefficients(0., 0., 0., -4., 0., 0.)
    step = guiding_center_collision_step(1., .2, 1., jnp.zeros(2), c, speed_floor=.1)
    assert step.y[0] == pytest.approx(3.2)


def test_collision_maps_keep_state_and_coefficient_jvp_vjp():
    state = jnp.array([2., .3])
    def step(z, strength):
        c = CollisionCoefficients(.8*strength, .4*strength, .12*strength,
                                   -.3*strength, .07*strength, .02*strength)
        return guiding_center_collision_step(z[0], z[1], .03, jnp.array([.1, -.2]), c).y
    tangent = jnp.array([.2, -.1])
    value, jvp = jax.jvp(step, (state, jnp.array(1.)), (tangent, jnp.array(.4)))
    _, pullback = jax.vjp(step, state, jnp.array(1.))
    cotangent = jnp.array([.7, -.2])
    dz, ds = pullback(cotangent)
    np.testing.assert_allclose(jnp.dot(cotangent, jvp), jnp.dot(dz, tangent)+.4*ds, rtol=1e-13)
    eps = 1e-5
    finite_difference = (step(state+eps*tangent, 1+eps*.4)-step(state-eps*tangent, 1-eps*.4))/(2*eps)
    np.testing.assert_allclose(jvp, finite_difference, atol=2e-11, rtol=2e-9)
    assert np.isfinite(value).all()


def test_bridge_rejection_retains_parent_and_nested_right_intervals():
    m = brownian_memory(jax.random.key(4), 3, capacity=8)
    m, h, parent, overflow = brownian_propose(m, 1.)
    assert not overflow and h == 1
    same, _, repeated, _ = brownian_propose(m, 1.)
    np.testing.assert_array_equal(repeated, parent)
    np.testing.assert_array_equal(jax.random.key_data(same.key), jax.random.key_data(m.key))
    m, _, left, _ = brownian_propose(m, .3)
    m, _, nested, _ = brownian_propose(m, .1)
    m = brownian_accept(m)
    m, h, middle, _ = brownian_propose(m, 10.)
    assert h == pytest.approx(.2)
    m = brownian_accept(m)
    m, h, right, _ = brownian_propose(m, 10.)
    assert h == pytest.approx(.7)
    np.testing.assert_allclose(nested+middle+right, parent, atol=3e-16)
    np.testing.assert_allclose(nested+middle, left, atol=2e-16)
    assert m.peak == 3


def test_bridge_has_conditional_mean_variance_and_unconditional_independence():
    n, h = 32768, .3
    keys = jax.random.split(jax.random.key(10), n)
    def sample(key):
        m, _, parent, _ = brownian_propose(brownian_memory(key, 1, capacity=2), 1.)
        _, _, left, _ = brownian_propose(m, h)
        return jnp.array([parent[0], left[0], parent[0]-left[0]])
    parent, left, right = np.asarray(jax.jit(jax.vmap(sample))(keys)).T
    residual = left-h*parent
    assert abs(residual.mean()) < .008
    assert residual.var() == pytest.approx(h*(1-h), rel=.035)
    assert left.var() == pytest.approx(h, rel=.035)
    assert right.var() == pytest.approx(1-h, rel=.035)
    assert abs(np.cov(left, right)[0, 1]) < .012


def test_bridge_capacity_never_discards_stored_values():
    m, _, _, _ = brownian_propose(brownian_memory(jax.random.key(2), 2, capacity=1), 1.)
    after, _, _, overflow = brownian_propose(m, .5)
    assert overflow
    for before, now in zip(jax.tree.leaves(m), jax.tree.leaves(after)):
        if jax.dtypes.issubdtype(before.dtype, jax.dtypes.prng_key):
            before, now = jax.random.key_data(before), jax.random.key_data(now)
        np.testing.assert_array_equal(now, before)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_split_zero_collisions_matches_analytic_orbit(method):
    rhs = lambda t, y, args: jnp.array([y[1], -y[0]])
    collision = lambda t, y, h, dw, args: CollisionStep(y, jnp.array(0.), jnp.array(0.))
    times = jnp.linspace(0., 2., 9)
    solution = jax.jit(lambda: solve_split(rhs, collision, jnp.array([1., 0.]),
        t1=2., dt0=.3, ts=times, key=jax.random.key(0), noise_dim=1, method=method,
        rtol=1e-11, atol=1e-11))()
    assert solution.status == SplitStatus.SUCCESS
    np.testing.assert_allclose(solution.ys, jnp.stack([jnp.cos(times), -jnp.sin(times)], axis=1), atol=2e-11)
    assert solution.stats.rejected == 0


def test_reusing_orbit_step_reduces_retries_and_recomputes_rhs_after_kicks():
    # Rotation plus a jump in amplitude has an independent exact solution.
    # Reusing a pre-kick FSAL derivative would spoil this accuracy check.
    def rhs(t, y, args):
        return 20*jnp.array([-y[1], y[0]])
    def collision(t, y, h, dw, args):
        return CollisionStep(1.1*y, jnp.array(0.), jnp.array(0.))
    results = [jax.jit(lambda: solve_split(rhs, collision, jnp.array([1., 0.]),
        t1=2., dt0=.2, ts=jnp.linspace(0., 2., 11), key=jax.random.key(0),
        noise_dim=1, adaptive=False, orbit_adaptive=True, method="dopri5",
        reuse_orbit_step=reuse, rtol=1e-10, atol=1e-10))() for reuse in (False, True)]
    exact = 1.1**10*np.array([np.cos(40.), np.sin(40.)])
    for result in results:
        assert result.status == SplitStatus.SUCCESS
        np.testing.assert_allclose(result.y, exact, atol=2e-8, rtol=0)
    assert results[1].stats.orbit_rejected < results[0].stats.orbit_rejected/2
    assert results[1].stats.orbit_steps < results[0].stats.orbit_steps


@pytest.mark.parametrize("controller", ["conservative", "paper"])
def test_adaptive_rejection_preserves_additive_brownian_terminal_distribution(controller):
    # The deliberately noise-dependent acceptance tests the bias mechanism
    # described in the paper: rejected large increments must not be redrawn.
    def collision(t, y, h, dw, args):
        # Cubic in the Wiener increment, as in the paper's diffusion estimate;
        # the acceptance threshold remains |dW| <= .6 for both controllers.
        return CollisionStep(y+dw, jnp.array(0.), (jnp.abs(dw[0])/.6)**3)
    def one(key):
        return solve_split(lambda t, y, args: jnp.zeros_like(y), collision, jnp.zeros(1),
            t1=1., dt0=1., ts=jnp.array([1.]), key=key, noise_dim=1,
            max_steps=256, record_steps=False, collision_controller=controller)
    keys = jax.random.split(jax.random.key(32), 8192)
    result = jax.jit(jax.vmap(one))(keys)
    assert np.all(result.status == SplitStatus.SUCCESS), np.unique(result.status, return_counts=True)
    assert np.mean(result.stats.rejected) > 0
    assert np.max(result.stats.max_accepted_error) <= 1
    assert abs(np.mean(result.y)) < .04
    assert np.var(result.y) == pytest.approx(1., rel=.055)
    root = jax.jit(jax.vmap(lambda key: brownian_propose(brownian_memory(key, 1), 1.)[2]))(keys)
    np.testing.assert_allclose(result.y, root, atol=3e-14, rtol=0.)


def test_lookahead_keeps_samples_across_insertions_growth_and_consumption():
    m = brownian_memory(jax.random.key(4), 2, capacity=8)
    known = {}
    for end in (1., .3, 1.4, .1, .8):
        m, value, full = brownian_sample(m, end)
        assert not full
        known[end] = value
        for old, expected in known.items():
            unchanged, got, full = brownian_sample(m, old)
            np.testing.assert_allclose(got, expected, atol=4e-16, rtol=0)
            np.testing.assert_array_equal(jax.random.key_data(unchanged.key), jax.random.key_data(m.key))
    m = brownian_consume(m, .3)
    for end in (.8, 1., 1.4):
        m, got, _ = brownian_sample(m, end)
        np.testing.assert_allclose(got, known[end]-known[.3], atol=4e-16, rtol=0)
    assert m.size == 3


def test_lookahead_bridge_conditional_law_and_capacity():
    def sample(key):
        m, parent, _ = brownian_sample(brownian_memory(key, 1, capacity=2), 1.)
        m, left, _ = brownian_sample(m, .3)
        after, _, full = brownian_sample(m, .6)
        return parent[0], left[0], full, jnp.all(after.increments == m.increments)
    parent, left, full, same = np.asarray(jax.jit(jax.vmap(sample))(
        jax.random.split(jax.random.key(21), 16384))).reshape(4, -1)
    residual = left-.3*parent
    assert abs(residual.mean()) < .012
    assert residual.var() == pytest.approx(.21, rel=.045)
    assert np.all(full) and np.all(same)


@pytest.mark.parametrize("accepted, outlier, expected_limit", [(False, False, 2), (True, False, 4), (True, True, 6)])
def test_paper_controller_uses_thirds_and_retains_a_failing_probe(accepted, outlier, expected_limit):
    m = brownian_memory(jax.random.key(7), 1, capacity=16)
    dw = jnp.array([3. if outlier else 1.])
    # Precompute candidates independently from the controller and identify
    # the first threshold crossing, including the rejected candidate in memory.
    optimum = .9*np.linalg.norm(dw)
    expected, stored = 1/3, m
    for n in range(1, expected_limit+1):
        stored, value, full = brownian_sample(stored, n*(1/3))
        assert not full
        if np.linalg.norm(value) >= optimum:
            break
        expected = n/3
    got, length, full = paper_collision_step_size(m, 1., dw, 0., 1., jnp.array(accepted),
        dtmin=0., dtmax=10., barrier=10.)
    assert not full
    assert length == pytest.approx(expected)
    np.testing.assert_array_equal(got.ends, stored.ends)
    np.testing.assert_allclose(got.increments, stored.increments, atol=2e-16)
    np.testing.assert_array_equal(jax.random.key_data(got.key), jax.random.key_data(stored.key))


def test_explicit_normal_obeys_conditional_bridge_and_retains_parent():
    m = brownian_memory(jax.random.key(81), 2, capacity=4)
    parent = jnp.array([.3, -.7])
    m = m._replace(ends=m.ends.at[0].set(1.), increments=m.increments.at[0].set(parent),
                   size=jnp.array(1), peak=jnp.array(1))
    z = jnp.array([.2, -.4])
    after, left, full = brownian_sample(m, .4, normal=z)
    assert not full
    np.testing.assert_allclose(left, .4*parent+np.sqrt(.24)*z, rtol=1e-14)
    np.testing.assert_array_equal(jax.random.key_data(after.key), jax.random.key_data(m.key))
    # Conditional covariance is .24 I and conditional mean is .4 parent.
    derivative = jax.jacfwd(lambda normal: brownian_sample(m, .4, normal=normal)[1])(z)
    np.testing.assert_allclose(derivative @ derivative.T, .24*np.eye(2), atol=1e-15)
    after, root, full = brownian_sample(after, 1., normal=z*100)
    assert not full
    np.testing.assert_allclose(root, parent, atol=1e-15)
    tail = brownian_consume(after, .4)
    tail, remainder, full = brownian_sample(tail, 1., normal=z*100)
    assert not full
    np.testing.assert_allclose(left+remainder, parent, atol=1e-15)


@pytest.mark.parametrize("error, expected", [(.25, 1.5), (4., .45)])
def test_paper_controller_drift_limit_and_output_barrier(error, expected):
    m = brownian_memory(jax.random.key(8), 1, capacity=8)
    got, length, full = paper_collision_step_size(m, 1., jnp.array([2.]), error, 1e-18,
        jnp.array(error <= 1), dtmin=0., dtmax=10., barrier=10.)
    assert not full and length == pytest.approx(expected)
    assert got.size == 3
    capped, length, full = paper_collision_step_size(m, 1., jnp.array([2.]), error, 0.,
        jnp.array(error <= 1), dtmin=0., dtmax=10., barrier=.2)
    assert not full and length == pytest.approx(.2)
    assert capped.size == 0  # no speculative samples needed for zero diffusion error


@pytest.mark.parametrize("barrier, expected", [(2., .4), (.1, .1)])
def test_paper_zero_diffusion_proposal_respects_step_floor_and_barrier(barrier, expected):
    m = brownian_memory(jax.random.key(12), 1)
    after, h, full = paper_collision_step_size(m, .5, jnp.zeros(1), 4., 0.,
        jnp.array(False), dtmin=.4, dtmax=1., barrier=barrier)
    assert not full and h == pytest.approx(expected)
    assert after.size == 0
    np.testing.assert_array_equal(jax.random.key_data(after.key), jax.random.key_data(m.key))


def test_paper_zero_diffusion_cannot_accept_below_minimum_step():
    def kick(t, y, h, dw, args):
        return CollisionStep(y+1., 16*h*h, jnp.array(0.))
    out = solve_split(lambda t, y, args: jnp.zeros_like(y), kick, jnp.zeros(1),
        t1=1., dt0=.5, dtmin=.4, dtmax=1., ts=jnp.array([1.]),
        key=jax.random.key(13), noise_dim=1, collision_controller="paper", max_steps=4)
    # E(.5)=4 and E(.4)=2.56: neither is acceptable. A buggy fallback
    # tries h=.225, where E=.81, and silently advances below the step floor.
    assert out.status == SplitStatus.MIN_STEP
    assert out.stats.accepted == 0 and out.stats.rejected == 2
    np.testing.assert_array_equal(out.y, [0.])


@pytest.mark.parametrize("perpendicular_dw", [0., 10.])
def test_particle_paper_controller_matches_scalar_parallel_error(perpendicular_dw):
    # Pre-realized future values isolate the scalar Eq. (62) error control.
    # Large perpendicular increments, past or future, must not shorten its
    # proposal. The 1D reference shares the exact same parallel Brownian path.
    m = brownian_memory(jax.random.key(5), 3, capacity=8)
    m = m._replace(ends=m.ends.at[:2].set(jnp.array([2/3, 1/3])),
        increments=m.increments.at[:2].set(jnp.array([[.1, 0., .05], [.6, 0., .1]])),
        size=jnp.array(2), peak=jnp.array(2))
    dw = jnp.array([perpendicular_dw, 0., .5])
    reference = m._replace(increments=m.increments[:, 2:3])
    expected, scalar_h, scalar_full = paper_collision_step_size(reference, 1., dw[2:3],
        .2, 8., jnp.array(False), dtmin=0., dtmax=10., barrier=10.)
    got, h, full = paper_collision_step_size(m, 1., dw, .2, 8., jnp.array(False),
        dtmin=0., dtmax=10., barrier=10., wiener_norm=lambda w: jnp.abs(w[2]))
    assert not full and not scalar_full
    assert h == scalar_h == pytest.approx(2/3)
    np.testing.assert_array_equal(got.ends, expected.ends)
    np.testing.assert_allclose(got.increments[:, 2:3], expected.increments, atol=0.)


@pytest.mark.parametrize("controller", ["conservative", "paper"])
@pytest.mark.parametrize("options, status", [
    (dict(max_steps=1), SplitStatus.MAX_STEPS),
    (dict(max_pending=1), SplitStatus.BROWNIAN_CAPACITY),
    (dict(dtmin=.5), SplitStatus.MIN_STEP),
])
def test_split_limits_report_failure_and_hold_last_state(options, status, controller):
    collision = lambda t, y, h, dw, args: CollisionStep(y+dw, jnp.array(4.), jnp.array(0.))
    result = solve_split(lambda t, y, args: y*0, collision, jnp.ones(1),
        t1=1., dt0=.5, ts=jnp.array([0., 1.]), key=jax.random.key(1), noise_dim=1,
        collision_controller=controller, **options)
    assert result.status == status
    np.testing.assert_array_equal(result.ys, np.ones((2, 1)))


def test_orbit_event_is_root_found_and_skips_collision():
    collision = lambda t, y, h, dw, args: CollisionStep(jnp.full_like(y, jnp.nan), jnp.array(0.), jnp.array(0.))
    result = solve_split(lambda t, y, args: jnp.ones_like(y), collision, jnp.zeros(1),
        t1=1., dt0=.5, ts=jnp.array([0., .1, 1.]), key=jax.random.key(1), noise_dim=1,
        event=lambda t, y, args: .05-y[0])
    assert result.status == SplitStatus.EVENT
    assert result.t == pytest.approx(.05, abs=2e-14)
    np.testing.assert_allclose(result.ys[:, 0], [0., .05, .05], atol=2e-14)


class UniformField:
    def B_contravariant(self, point):
        return jnp.array([0., 0., 1.])
    B_covariant = B_contravariant
    def AbsB(self, point):
        return jnp.array(1.)
    def dAbsB_by_dX(self, point):
        return jnp.zeros(3)
    curl_b = kappa = dAbsB_by_dX
    def sqrtg(self, point):
        return jnp.array(1.)
    def to_xyz(self, point):
        return point


@pytest.mark.parametrize("model, spatial", [("GuidingCenter", False), ("GuidingCenter", True), ("FullOrbit", False)])
def test_trace_candidate_controller_uses_native_collision_metadata(model, spatial):
    from essos.dynamics import Particles
    particles = Particles(jnp.zeros((1, 3)), initial_vparallel_over_v=[.3],
        mass=M, charge=Q, energy=5000*Q, field=UniformField())
    out = trace_split(UniformField(), particles, background(), model=model,
        orbit=False, spatial_diffusion=spatial, tmax=2e-7, timestep=1e-7,
        collision_dtmax=1e-6, n_save=2, collision_tolerance=.02,
        collision_controller="candidate")
    assert np.all(out.status == SplitStatus.SUCCESS)
    assert np.max(out.stats.max_accepted_error) <= 1
    assert np.isfinite(out.y).all()


@pytest.mark.parametrize("model, spatial, noise, expected", [
    ("FullOrbit", False, [1000., 1000., .5], .5),
    ("GuidingCenter", False, [3., 4.], 5.),
    ("GuidingCenter", True, [1000., 1000., 1000., 3., 4.], 5.),
])
def test_trace_paper_controller_selects_only_error_controlled_noise(monkeypatch, model, spatial, noise, expected):
    import essos.split as split_module
    from essos.dynamics import Particles
    seen = []
    original = split_module.paper_collision_step_size
    def record(*args, **kwargs):
        seen.append(kwargs["wiener_norm"])
        return original(*args, **kwargs)
    monkeypatch.setattr(split_module, "paper_collision_step_size", record)
    particles = Particles(jnp.zeros((1, 3)), initial_vparallel_over_v=[.3],
                          mass=M, charge=Q, energy=5000*Q, field=UniformField())
    result = trace_split(UniformField(), particles, background(), model=model,
        orbit=False, spatial_diffusion=spatial, tmax=2e-7, timestep=1e-7,
        collision_dtmax=1e-6, n_save=2, collision_tolerance=.02,
        collision_controller="paper")
    assert np.all(result.status == SplitStatus.SUCCESS)
    assert np.max(result.stats.max_accepted_error) <= 1
    assert seen
    for norm in seen:
        assert norm(jnp.array(noise)) == pytest.approx(expected)


@pytest.mark.parametrize("model", ["GuidingCenter", "FullOrbit"])
@pytest.mark.parametrize("integrator", ["essos_rk45", "essos_dopri8"])
def test_trace_split_vacuum_recovers_native_orbits(model, integrator, monkeypatch):
    import essos.dynamics as dynamics
    def forbidden(*args, **kwargs):
        raise AssertionError("split path called Diffrax")
    monkeypatch.setattr(dynamics, "diffeqsolve", forbidden)
    particles = dynamics.Particles(jnp.zeros((2, 3)), initial_vparallel_over_v=[.3, -.7],
        mass=M, charge=Q, energy=1000*Q, field=UniformField())
    result = trace_split(UniformField(), particles, background(0.), model=model,
        integrator=integrator, tmax=1e-7, timestep=2e-8, n_save=5)
    reference = dynamics.Tracing(field=UniformField(), particles=particles,
        model=model+"Adaptative", integrator=integrator, maxtime=1e-7, timestep=1e-9, times_to_trace=5,
        rtol=1e-12, atol=1e-12, progress=False)
    assert np.all(result.status == SplitStatus.SUCCESS)
    np.testing.assert_allclose(result.ys[..., :3], reference.trajectories[..., :3], atol=2e-9)
    np.testing.assert_allclose(result.ys[..., 3:4] if model == "GuidingCenter" else result.ys[..., 3:],
        reference.trajectories[..., 3:], atol=2e-8*particles.total_speed, rtol=0.)


def test_trace_split_collision_only_changes_velocity_and_mu():
    from essos.dynamics import Particles
    particles = Particles(jnp.zeros((3, 3)), initial_vparallel_over_v=[-.9, 0., .9],
        mass=M, charge=Q, energy=5000*Q)
    result = trace_split(UniformField(), particles, background(), tmax=2e-6,
        timestep=5e-7, n_save=5, orbit=False)
    assert np.all(result.status == SplitStatus.SUCCESS)
    assert np.max(result.stats.max_accepted_error) <= 1
    np.testing.assert_array_equal(result.ys[..., :3], np.zeros((3, 5, 3)))
    assert np.all(result.ys[..., 4] >= 0)
    assert not np.allclose(result.ys[:, -1, 3], particles.initial_vparallel)


@pytest.mark.parametrize("model", ["gc", "particle"])
def test_velocity_kernels_relax_to_nonrelativistic_maxwellian(model):
    # Independent solvable Fokker-Planck model: D=I, Q=-v, so f_M~exp(-v^2/2).
    n, h, steps = 4096, .002, 2500
    initial = jnp.tile(jnp.array([3., .8] if model == "gc" else [0., 0., 3.]), (n, 1))
    def advance(carry, _):
        y, key = carry
        key, draw = jax.random.split(key)
        noise = jax.random.normal(draw, y.shape)*jnp.sqrt(h)
        def one(y, dw):
            v = y[0] if model == "gc" else jnp.linalg.norm(y)
            c = CollisionCoefficients(1., 1., 0., -v, -1., -1.)
            if model == "gc":
                return guiding_center_collision_step(v, y[1], h, dw, c, speed_floor=.05).y
            return particle_collision_step(y, h, dw, c).y
        return (jax.vmap(one)(y, noise), key), None
    y, _ = jax.jit(lambda: jax.lax.scan(advance, (initial, jax.random.key(5)), None, length=steps)[0])()
    speed2 = y[:, 0]**2 if model == "gc" else jnp.sum(y*y, axis=1)
    pitch = y[:, 1] if model == "gc" else y[:, 2]/jnp.sqrt(speed2)
    assert np.mean(speed2) == pytest.approx(3., abs=.14)
    assert abs(np.mean(pitch)) < .035
    assert np.mean(pitch*pitch) == pytest.approx(1/3, abs=.02)


def test_scalar_milstein_has_first_order_strong_convergence():
    # A geometric Brownian speed has a closed pathwise solution. Coefficients
    # are chosen so the GC speed Ito drift is a*v and diffusion is sigma*v.
    n, fine, a, sigma = 4096, 256, .2, .4
    noise = jax.random.normal(jax.random.key(9), (fine, n))/np.sqrt(fine)
    exact = jnp.exp(a-.5*sigma*sigma+sigma*jnp.sum(noise, axis=0))
    errors = []
    for steps in (8, 16, 32, 64):
        dw = noise.reshape((steps, fine//steps, n)).sum(axis=1)
        def advance(v, w):
            def one(v, w):
                c = CollisionCoefficients(.5*sigma*sigma*v*v, 0., sigma*sigma*v,
                                           (a-2*sigma*sigma)*v, a-2*sigma*sigma, 0.)
                return guiding_center_collision_step(v, 0., 1/steps, jnp.array([w, 0.]), c).y[0]
            return jax.vmap(one)(v, w), None
        actual, _ = jax.jit(lambda: jax.lax.scan(advance, jnp.ones(n), dw))()
        errors.append(float(jnp.sqrt(jnp.mean((actual-exact)**2))))
    assert errors[0]/errors[-1] > 6
    assert all(first > second for first, second in zip(errors, errors[1:]))


def test_cartesian_gc_spatial_diffusion_matches_paper_covariance():
    from essos.dynamics import Particles
    n, h, xi = 4096, 2e-9, .4
    particles = Particles(jnp.zeros((n, 3)), initial_vparallel_over_v=jnp.full(n, xi),
        mass=M, charge=Q, energy=5000*Q)
    species = background()
    c = collision_coefficients(species, M, Q, particles.total_speed, jnp.zeros(3))
    dx = (c.parallel*(1-xi*xi)/2+c.perpendicular*(1+xi*xi)/2)/(Q/M)**2
    result = trace_split(UniformField(), particles, species, tmax=h, timestep=h,
        n_save=2, orbit=False, adaptive=False, spatial_diffusion=True)
    assert np.all(result.status == SplitStatus.SUCCESS)
    position = np.asarray(result.y[:, :3])
    np.testing.assert_array_equal(position[:, 2], np.zeros(n))
    np.testing.assert_allclose(np.var(position[:, :2], axis=0)/(2*dx*h), 1., rtol=.065)


def test_boozer_split_vacuum_matches_existing_native_driver():
    from essos.boozer import BoozerField, trace_boozer
    from essos.dynamics import Particles
    s = np.linspace(.01, .99, 10)
    field = BoozerField.from_booz(s, np.stack([np.full_like(s, 5.), -.5*np.sqrt(s)]),
        [0, 1], [0, 0], np.full_like(s, .4), np.full_like(s, 40.), np.zeros_like(s), 40., 1,
        angle_grid=None)
    points = jnp.array([[.3, .4, .1], [.2, 1., .2]])
    particles = Particles(points, initial_vparallel_over_v=[.3, -.6], mass=M, charge=Q, energy=5000*Q)
    result = trace_split(field, particles, background(0.), tmax=1e-7, timestep=2e-8, n_save=5)
    reference = trace_boozer(field, *points.T, particles.initial_vparallel_over_v,
        speed=float(particles.total_speed), mass=M, charge=Q, tmax=1e-7, timestep=1e-9,
        n_save=5, integrator="essos_dopri8", progress=False)
    assert np.all(result.status == SplitStatus.SUCCESS)
    np.testing.assert_allclose(result.ys[..., :3], reference.states[..., :3], atol=3e-9)
    np.testing.assert_allclose(result.ys[..., 3]/particles.total_speed,
                               reference.states[..., 3]/particles.total_speed, atol=3e-9)
    np.testing.assert_allclose(result.ys[..., 4]/result.ys[:, :1, 4], 1., rtol=1e-12)


def test_vmec_split_vacuum_matches_existing_native_driver():
    from pathlib import Path
    from essos.fields import Vmec
    from essos.dynamics import Particles, Tracing
    path = Path(__file__).resolve().parents[1]/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
    field = Vmec(str(path), ntheta=8, nphi=8)
    particles = Particles(jnp.array([[.3, .4, .1]]), initial_vparallel_over_v=[.3],
        mass=M, charge=Q, energy=5000*Q)
    result = trace_split(field, particles, background(0.), tmax=2e-8, timestep=1e-8, n_save=3)
    reference = Tracing(field=field, particles=particles, model="GuidingCenterAdaptative",
        integrator="essos_dopri8", maxtime=2e-8, timestep=1e-9, times_to_trace=3, progress=False)
    assert np.all(result.status == SplitStatus.SUCCESS)
    np.testing.assert_allclose(result.ys[..., :3], reference.trajectories[..., :3], atol=3e-9)
    np.testing.assert_allclose(result.ys[..., 3]/particles.total_speed,
                               reference.trajectories[..., 3]/particles.total_speed, atol=3e-9)


@pytest.mark.parametrize("options, message", [
    (dict(integrator="diffrax"), "integrator"),
    (dict(collision_tolerance=0.), "collision_tolerance"),
    (dict(collision_controller="pid"), "collision_controller"),
    (dict(speed_floor=0.), "speed_floor"),
    (dict(model="FullOrbit", spatial_diffusion=True), "spatial_diffusion"),
])
def test_invalid_split_configuration_raises(options, message):
    from essos.dynamics import Particles
    particles = Particles(jnp.zeros((1, 3)), initial_vparallel_over_v=[.3], mass=M, charge=Q, energy=5000*Q)
    with pytest.raises(ValueError, match=message):
        trace_split(UniformField(), particles, background(), tmax=1e-6, timestep=1e-7, **options)


def test_lie_split_global_error_reduces_linearly_with_interval_size():
    # Two noncommuting, exactly solved nilpotent flows. The combined solution
    # is (cosh(t), sinh(t)), so this checks the separate splitting error.
    rhs = lambda t, y, args: jnp.array([y[1], 0.])
    collision = lambda t, y, h, dw, args: CollisionStep(
        y+jnp.array([0., h*y[0]]), jnp.array(0.), jnp.array(0.))
    errors = []
    for dt in (.1, .05, .025):
        result = solve_split(rhs, collision, jnp.array([1., 0.]), t1=1., dt0=dt,
            ts=jnp.array([1.]), key=jax.random.key(1), noise_dim=1, adaptive=False,
            record_steps=True)
        assert result.status == SplitStatus.SUCCESS
        assert result.stats.rejected == 0
        assert result.step_ts[int(result.stats.accepted)] == 1.
        errors.append(np.linalg.norm(result.y-np.array([np.cosh(1.), np.sinh(1.)])))
    assert errors[0]/errors[1] == pytest.approx(2., rel=.07)
    assert errors[1]/errors[2] == pytest.approx(2., rel=.04)


@pytest.mark.parametrize("times, dt, expected_steps", [
    (jnp.linspace(0., 1e-5, 9), 1e-5/64, 64),
    (jnp.array([0., .11, .33, 1.]), .25, 6),
])
def test_fixed_split_grid_keeps_save_roundoff_from_adding_noise_intervals(times, dt, expected_steps):
    collision = lambda t, y, h, dw, args: CollisionStep(y, jnp.array(0.), jnp.array(0.))
    result = solve_split(lambda t, y, args: jnp.ones_like(y), collision, jnp.zeros(1),
        t1=times[-1], dt0=dt, ts=times, key=jax.random.key(1), noise_dim=1,
        adaptive=False, record_steps=True)
    assert result.status == SplitStatus.SUCCESS
    assert result.stats.accepted == expected_steps
    np.testing.assert_allclose(result.ys[:, 0], times, atol=1e-15)


@pytest.mark.parametrize("controller", ["conservative", "paper"])
def test_adaptive_split_can_grow_beyond_initial_and_fixed_step(controller):
    def collision(t, y, h, dw, args):
        return CollisionStep(y+h, (h/.1)**2, jnp.array(0.))
    def solve(adaptive):
        return solve_split(lambda t, y, args: jnp.zeros_like(y), collision, jnp.zeros(1),
            t1=1., dt0=.01, dtmax=1., ts=jnp.array([1.]), key=jax.random.key(3),
            noise_dim=1, adaptive=adaptive, collision_controller=controller, record_steps=True, max_steps=256)
    fixed, adaptive = jax.jit(lambda: solve(False))(), jax.jit(lambda: solve(True))()
    assert fixed.status == adaptive.status == SplitStatus.SUCCESS
    assert fixed.stats.accepted == 100
    assert adaptive.stats.accepted < 25
    steps = np.diff(adaptive.step_ts[:int(adaptive.stats.accepted)+1])
    assert steps.max() > .05  # actual accepted intervals exceed the initial .01
    assert adaptive.stats.max_accepted_error <= 1
    np.testing.assert_allclose(adaptive.y, [1.], atol=2e-15)


@pytest.mark.parametrize("model", ["GuidingCenter", "FullOrbit"])
def test_split_example_uses_existing_external_field_adapter(monkeypatch, capsys, model):
    from examples.particle_tracing_collisions.trace_particles_split import main
    monkeypatch.setattr("sys.argv", ["trace_particles_split", "--model", model, "--particles", "2"])
    main()
    assert "'SUCCESS': 2" in capsys.readouterr().out


