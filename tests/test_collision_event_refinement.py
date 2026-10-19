"""Boundary monitoring, Brownian replay, and unchanged opt-in defaults."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.collision_error import CollisionErrorModel
from essos.collisions import CollisionStep
from essos.split import solve_split, SplitStatus
from examples.benchmarks.collision_event_diagnostic import boundary_step_limit


@pytest.mark.parametrize("adaptive,controller", [(False, "conservative"), (True, "conservative"),
                                                (True, "paper"), (True, "candidate")])
def test_refining_a_collision_event_retains_the_sampled_brownian_endpoint(adaptive, controller):
    # The first proposed interval spans the whole solve and samples W(.1).
    # Clock changes only in kicks, so RK event finding cannot intercept it.
    # Refinement must keep this W(.1), despite rejecting its original interval.
    model = CollisionErrorModel(*(jnp.asarray(x) for x in (0., 0., 0., 0., 0., 0., 1., 0)))
    def kick(t, y, h, dw, args):
        return CollisionStep(jnp.array([t, y[1]+dw[0]]), jnp.array(0.), jnp.array(0.),
                             model if controller == "candidate" else None)
    def solve(resolution):
        return solve_split(lambda t, y, a: jnp.zeros_like(y), kick, jnp.zeros(2),
            t1=.1, dt0=.1, dtmax=.1, ts=jnp.array([0., .1]), key=jax.random.key(73),
            noise_dim=1, adaptive=adaptive, collision_controller=controller,
            event=lambda t, y, a: .1-y[0], event_dtmax=resolution,
            record_steps=True, max_steps=256)
    original, refined = solve(None), solve(.001)
    assert original.status == refined.status == SplitStatus.EVENT
    assert refined.t == pytest.approx(.1)
    np.testing.assert_allclose(refined.y, original.y, atol=3e-15, rtol=3e-15)
    steps = np.asarray(refined.step_ts)[:int(refined.stats.accepted)+1]
    assert steps[-1]-steps[-2] <= .001*(1+1e-12)
    assert refined.stats.rejected > 0
    assert refined.stats.peak_pending > 1


def test_state_step_ceiling_subdivides_fixed_grid_without_moving_its_anchors():
    def kick(t, y, h, dw, args):
        return CollisionStep(y+h, jnp.array(0.), jnp.array(0.))
    solution = solve_split(lambda t, y, a: jnp.zeros_like(y), kick, jnp.zeros(1),
        t1=.3, dt0=.1, dtmax=.1, ts=jnp.array([0., .115, .3]), key=jax.random.key(19),
        noise_dim=1, adaptive=False, step_limit=lambda t, y, a: .03,
        record_steps=True, max_steps=64)
    assert solution.status == SplitStatus.SUCCESS
    steps = np.asarray(solution.step_ts)[:int(solution.stats.accepted)+1]
    assert np.max(np.diff(steps)) <= .03*(1+1e-12)
    for anchor in (.1, .115, .2, .3):
        assert np.min(abs(steps-anchor)) < 2e-16
    np.testing.assert_allclose(solution.ys[:, 0], [0., .115, .3], atol=2e-16)


@pytest.mark.parametrize("limit", [0., -1., np.nan])
def test_invalid_dynamic_step_ceiling_returns_without_advancing(limit):
    def kick(t, y, h, dw, args):
        return CollisionStep(y+h, jnp.array(0.), jnp.array(0.))
    solution = solve_split(lambda t, y, a: jnp.zeros_like(y), kick, jnp.zeros(1),
        t1=.1, dt0=.1, ts=jnp.array([0., .1]), key=jax.random.key(19),
        noise_dim=1, step_limit=lambda t, y, a: limit)
    assert solution.status == SplitStatus.INVALID_INPUT
    assert solution.t == 0
    assert solution.stats.steps == 0


def test_speed_boundary_limit_is_unit_consistent_and_handles_vanishing_diffusion():
    value = boundary_step_limit(2.1, 2., .5, -.3, .0001)
    assert value == pytest.approx(.002)
    # v -> alpha v and t -> beta t transforms D and A accordingly.
    alpha, beta = 7., 3.
    transformed = boundary_step_limit(2.1*alpha, 2.*alpha, .5*alpha**2/beta,
                                     -.3*alpha/beta, .0001*beta)
    assert transformed == pytest.approx(value*beta)
    assert boundary_step_limit(2., 2., .5, -.3, .0001) == pytest.approx(.0001)
    assert boundary_step_limit(2.1, 2., 0., -1., .0001) == pytest.approx(.02)
    assert np.isinf(boundary_step_limit(2.1, 2., 0., 1., .0001))


@pytest.mark.parametrize("model,spatial", [("gc", False), ("gc", True), ("lorentz", False)])
def test_physical_diagnostic_resolves_crossing_and_returns_matching_stopped_state(model, spatial):
    from examples.benchmarks.nonrelativistic_collision_reference import ProtonBackground
    from examples.benchmarks.validate_collisional_slowing_down import native_solver
    physical = ProtonBackground()
    resolution = 1e-5
    threshold = np.sqrt(10.)-1e-5
    keys = jax.random.split(jax.random.key(23), 8)
    output = native_solver(model, "conservative", True, physical, .004, threshold,
        spatial=spatial, event_interval=resolution, record_steps=True)(keys, .003, .001, .001)
    assert np.all((output.status == SplitStatus.SUCCESS) | (output.status == SplitStatus.EVENT))
    hits = np.asarray(output.status) == SplitStatus.EVENT
    assert np.any(hits)
    speed = output.y[:, 3] if spatial else (output.y[:, 0] if model == "gc" else np.linalg.norm(output.y, axis=1))
    assert np.all(speed[hits]/physical.sigma <= threshold+1e-13)
    for i in np.flatnonzero(hits):
        count = int(output.stats.accepted[i])
        times = np.asarray(output.step_ts[i, :count+1])*physical.nu
        assert times[-1]-times[-2] <= resolution*(1+1e-10)
        assert times[-1] == pytest.approx(output.t[i], abs=1e-15)
    assert np.max(output.stats.max_accepted_error) <= 1
    if spatial:
        assert np.all(output.y[:, 5] >= 0)


def test_event_refinement_requires_a_positive_boundary_and_separate_shift_experiment():
    from examples.benchmarks.nonrelativistic_collision_reference import ProtonBackground
    from examples.benchmarks.validate_collisional_slowing_down import native_solver
    with pytest.raises(ValueError, match="threshold"):
        native_solver("gc", "conservative", True, ProtonBackground(), 1., 0., event_interval=.001)
    with pytest.raises(ValueError, match="separately"):
        native_solver("gc", "conservative", True, ProtonBackground(), 1., 2.,
                      event_interval=.001, boundary_shift=True)
