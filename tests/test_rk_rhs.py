"""RHS preparation preserves state/time dependence and parameter sensitivities."""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from essos.rk import _prepare_rhs

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dependence", ["parameters", "time", "state", "both"])
@pytest.mark.parametrize("coalesce_views", [False, True])
def test_preparation_keeps_new_times_states_and_parameter_derivatives(dependence, coalesce_views):
    def observe(x, prepare):
        args = {"parameters": x[6:]}

        @jax.jit
        def rhs(t, y, args):
            p = args["parameters"]
            matrix = jnp.sin(p[:, None] + p[None, :]) + jnp.diag(p**2)
            value = jnp.ones_like(y)
            if dependence in ("time", "both"):
                value = value * jnp.cos(t)
            if dependence in ("state", "both"):
                value = value * jnp.sin(y)
            return matrix @ value

        prepared = _prepare_rhs(rhs, x[0], x[1:3], args,
                                coalesce_views=coalesce_views) if prepare else rhs
        # Evaluation deliberately uses a different time/state from setup.
        return prepared(x[3], x[4:6], args)

    x = jnp.array([.1, .2, .3, .7, -.4, .8, 1.1, .6])
    prepared, original = lambda x: observe(x, True), lambda x: observe(x, False)
    np.testing.assert_allclose(jax.jit(prepared)(x), original(x), rtol=1e-13, atol=1e-13)
    expected = jax.jacfwd(original)(x)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(prepared))(x), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.jacrev(prepared))(x), expected, rtol=1e-12, atol=1e-12)
    batch = jnp.stack([x, x + .04])
    np.testing.assert_allclose(jax.jit(jax.vmap(prepared))(batch), jax.vmap(original)(batch),
                               rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("model", ["gc", "lorentz"])
@pytest.mark.parametrize("coalesce_views", [False, True])
def test_prepared_orbit_rhs_keeps_coil_shape_current_and_position_sensitivities(model, coalesce_views):
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
    from essos.dynamics import GuidingCenter, Lorentz, Particles
    from essos.electric_field import Electric_field_zero
    from essos.fields import BiotSavart

    field = BiotSavart(Coils(CreateEquallySpacedCurves(2, 1, 1., .4, 16, 1, False), [1e6, .9e6]))
    particles = Particles(initial_xyz=jnp.array([[.95, .03, .02]]), initial_vparallel_over_v=[.6],
                          mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, energy=5000 * ONE_EV, field=field)
    speed = float(particles.total_speed)
    normalized = jnp.array([.95, .03, .02, .6] if model == "gc" else [.95, .03, .02, .3, -.2, .6])
    state_scale = jnp.array([1.] * 3 + [speed] * (len(normalized) - 3))
    rhs_scale = jnp.array([speed] * 3 + [speed / 1e-7] * (len(normalized) - 3))
    parameters, rebuild = ravel_pytree(field)
    x = jnp.concatenate([normalized, parameters])
    vector_field = GuidingCenter if model == "gc" else Lorentz

    def observe(values, prepare):
        state = values[:len(normalized)] * state_scale
        field = rebuild(values[len(normalized):])
        args = (field, particles, Electric_field_zero()) if model == "gc" else (field, particles)
        rhs = lambda t, y, args: vector_field(t, y, args)
        if prepare:
            rhs = _prepare_rhs(rhs, jnp.asarray(0.), state, args, coalesce_views=coalesce_views)
        return rhs(jnp.asarray(1e-8), state, args) / rhs_scale

    prepared, original = lambda x: observe(x, True), lambda x: observe(x, False)
    direction = jnp.linspace(-.002, .003, len(x))
    cotangent = jnp.linspace(.2, .7, len(normalized))
    value, tangent = jax.jvp(prepared, (x,), (direction,))
    expected, expected_tangent = jax.jvp(original, (x,), (direction,))
    np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-10, atol=2e-11)
    gradient = jax.vjp(prepared, x)[1](cotangent)[0]
    expected_gradient = jax.vjp(original, x)[1](cotangent)[0]
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-10, atol=2e-11)
    epsilon = 1e-5
    finite = (prepared(x + epsilon * direction) - prepared(x - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(tangent, finite, rtol=2e-5, atol=2e-8)


def test_coalesced_parameter_views_keep_nested_custom_derivative_rules():
    @jax.custom_jvp
    def law(x):
        return jnp.sin(x)

    @law.defjvp
    def law_jvp(primals, tangents):
        x, = primals
        dx, = tangents
        # Deliberately differs from the derivative of the primal formula.
        return law(x), 2*jnp.cos(x)*dx

    @jax.jit
    def rhs(t, y, matrix):
        return law(matrix @ y) + matrix[0]*jnp.cos(t) + matrix[:, 1]*y

    def observe(matrix, coalesce):
        t, y = jnp.array(.3), jnp.array([.2, -.4, .7])
        prepared = _prepare_rhs(rhs, t, y, matrix, coalesce_views=coalesce)
        return prepared(t+.2, y+.1, matrix)

    matrix = jnp.arange(9., dtype=float).reshape(3, 3)/10
    original = lambda matrix: rhs(jnp.array(.5), jnp.array([.3, -.3, .8]), matrix)
    prepared = lambda matrix: observe(matrix, True)
    np.testing.assert_allclose(jax.jit(prepared)(matrix), original(matrix), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(prepared))(matrix), jax.jacfwd(original)(matrix),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.jacrev(prepared))(matrix), jax.jacrev(original)(matrix),
                               rtol=1e-12, atol=1e-12)
