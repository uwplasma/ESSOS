"""Deferred root replay must preserve the forward solver's event semantics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import RKStatus, solve_rk

jax.config.update("jax_enable_x64", True)


def assert_same(actual, expected):
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(actual.stats, expected.stats)
    np.testing.assert_array_equal(actual.status, expected.status)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("controller", ["i", "pi"])
def test_rejected_steps_mixed_events_dense_saves_and_history(method, controller):
    def rhs(t, y, args):
        return jnp.stack((1 + .2 * jnp.sin(9*t), y[1] * jnp.cos(8*t)))
    # Both crossings can fall in the same trial; earliest root, not leaf order,
    # must choose the winner. Includes a boolean event and an initial event.
    event = (lambda t, y, a: .9-y[0], lambda t, y, a: .65-y[0],
             lambda t, y, a: t > .95)
    options = dict(t1=1., dt0=.8, ts=jnp.array([0., .2, .2, .5, .7, 1.]),
        event=event, method=method, controller=controller, stage_execution="unrolled",
        rtol=1e-10, atol=1e-10, max_steps=128, record_steps=True)
    births = jnp.array([[-2., 1.], [.1, .7], [.65, 1.]])
    def run(defer):
        return jax.jit(jax.vmap(lambda y: solve_rk(rhs, y, _defer_event_root=defer, **options)))(births)
    inline, replay = run(False), run(True)
    assert_same(replay, inline)
    assert np.any(np.asarray(replay.stats.rejected) > 0)
    assert replay.status[-1] == RKStatus.EVENT and replay.stats.steps[-1] == 0
    assert replay.event_mask[1][1]


@pytest.mark.parametrize("adaptive", [False, True])
def test_nonfinite_interior_root_uses_inline_rejection_and_retry(adaptive):
    rhs = lambda t, y, args: jnp.ones_like(y)
    def event(t, y, args):
        return jnp.where((t > .26) & (t < .30), jnp.nan, .28-y[0])
    options = dict(t1=1., dt0=.125, ts=jnp.linspace(0., 1., 5), event=event,
        method="dopri8", stage_execution="unrolled", max_steps=16, adaptive=adaptive)
    def run(defer):
        return jax.jit(lambda y: solve_rk(rhs, y, _defer_event_root=defer, **options))(jnp.array([0.]))
    inline, replay = run(False), run(True)
    assert_same(replay, inline)
    assert replay.stats.rejected > 0
    assert replay.status != RKStatus.EVENT


def test_empty_saves_and_simultaneous_event_tie():
    rhs = lambda t, y, args: jnp.ones_like(y)
    options = dict(t1=1., dt0=.8, ts=(), event=(lambda t, y, a: .5-y[0],)*2,
                   method="dopri8", stage_execution="unrolled", max_steps=16)
    solutions = [jax.jit(lambda y: solve_rk(rhs, y, _defer_event_root=defer, **options))(
        jnp.array([0.])) for defer in (False, True)]
    assert_same(solutions[1], solutions[0])
    assert solutions[1].event_mask == (True, False)


def test_recorded_schedule_retains_inline_event_fraction():
    rhs = lambda t, y, args: jnp.ones_like(y)
    options = dict(t1=1., dt0=.8, event=lambda t, y, a: .5-y[0], method="dopri8",
                   stage_execution="unrolled", max_steps=16, _record_schedule=True)
    actual = solve_rk(rhs, jnp.array([0.]), _defer_event_root=True, **options)
    expected = solve_rk(rhs, jnp.array([0.]), **options)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("controller", ["i", "pi"])
def test_nonfinite_root_retry_recovers_and_preserves_saved_prefix(controller):
    # After an accepted step, the first crossing's bisection hits a bad value
    # away from the true root. A shorter retry must recover, keeping prior
    # saves, accepted-time history and nontrivial PI controller history.
    rhs = lambda t, y, args: jnp.ones_like(y)
    def event(t, y, args):
        return jnp.where((t > .59) & (t < .61), jnp.nan, .5-y[0])
    options = dict(t1=1., dt0=.2, ts=jnp.linspace(0., 1., 11), event=event,
        method="dopri8", stage_execution="unrolled", max_steps=32,
        controller=controller, record_steps=True)
    def run(defer):
        return jax.jit(lambda y: solve_rk(rhs, y, _defer_event_root=defer, **options))(jnp.array([0.]))
    inline, replay = run(False), run(True)
    assert_same(replay, inline)
    assert replay.status == RKStatus.EVENT and replay.stats.rejected > 0
    np.testing.assert_allclose(replay.t, .5, rtol=0, atol=2e-14)
    np.testing.assert_allclose(replay.ys[:5, 0], [.0, .1, .2, .3, .4], rtol=0, atol=2e-14)


@pytest.mark.parametrize("controller", ["i", "pi"])
def test_cuda_command_buffers_preserve_mixed_root_retries(controller):
    from essos.boozer import boozer_compiler_options
    compiler = boozer_compiler_options()
    if not compiler:
        pytest.skip("CUDA command-buffer comparison")
    rhs = lambda t, y, args: jnp.ones_like(y)
    def event(t, y, args):
        return jnp.where((t > .59) & (t < .61), jnp.nan, .5-y[0])
    options = dict(t1=1., dt0=.2, ts=jnp.linspace(0., 1., 11), event=event,
        method="dopri8", stage_execution="unrolled", max_steps=32,
        controller=controller, record_steps=True, _defer_event_root=True)
    births = jnp.array([[-1.], [0.], [.5]])
    solve = jax.vmap(lambda y: solve_rk(rhs, y, **options))
    ordinary = jax.jit(solve)(births)
    captured = jax.jit(solve, compiler_options=compiler)(births)
    for a, b in zip(jax.tree_util.tree_leaves(captured), jax.tree_util.tree_leaves(ordinary)):
        np.testing.assert_array_equal(a, b)
    assert captured.status[0] == RKStatus.SUCCESS
    assert captured.status[1] == RKStatus.EVENT and captured.stats.rejected[1] > 0
    assert captured.status[2] == RKStatus.EVENT and captured.stats.steps[2] == 0


def test_cuda_public_boozer_entrypoint_retains_loss_root_and_energy():
    from essos.boozer import BoozerField, _native_orbits, _native_orbits_cuda, boozer_compiler_options
    from essos.constants import ALPHA_PARTICLE_MASS as mass, ALPHA_PARTICLE_CHARGE as charge, ONE_EV
    if not boozer_compiler_options():
        pytest.skip("CUDA command-buffer comparison")
    speed = float(np.sqrt(2*3.52e6*ONE_EV/mass))
    s = np.linspace(.005, .995, 50)
    field = BoozerField.from_booz(s, np.stack((np.full_like(s, 5.), -1.25*np.sqrt(s))),
        [0, 1], [0, 0], np.full_like(s, .4), np.full_like(s, 40.), np.zeros_like(s), -40., 1,
        angle_grid=None)
    birth = jnp.array([[0., jnp.sqrt(.9999), 0., 0.], [jnp.sqrt(.2), 0., 0., .5*speed]])
    B = jax.vmap(field.modB)(jnp.array([.9999, .2]), jnp.array([jnp.pi/2, 0.]), jnp.zeros(2))
    mu = (speed**2-birth[:, 3]**2)/(2*B)
    args = (field, birth, mu, mass, charge, speed, jnp.linspace(0., 1e-6, 5),
            1.25e-7, 1e-9, 1e-9, "dopri8", True, "i", .4, .3, "standard", 32, "unrolled")
    ordinary, captured = _native_orbits(*args), _native_orbits_cuda(*args)
    for a, b in zip(captured, ordinary):
        np.testing.assert_array_equal(a, b)
    assert captured[1][0] > 0 and captured[-1][0] == RKStatus.EVENT
