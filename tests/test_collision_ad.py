"""The kinetic AD rule preserves states, physical parameters and JVPs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.collision_ad import speed_pitch_linearization
from essos.collisions import CollisionCoefficients, guiding_center_collision_step


@pytest.mark.parametrize("pitch", [-1., -.4, .3, 1.])
@pytest.mark.parametrize("diffusion", [0., .2])
def test_kinetic_partials_and_captured_parameters_match_ordinary_ad(pitch, diffusion):
    def run(p, custom):
        def velocity(v, xi):
            c = CollisionCoefficients(diffusion*p[2]*v, .3*diffusion*p[2],
                diffusion*p[2], -.1*p[3]*v, -.1*p[3], 0.)
            return guiding_center_collision_step(v, xi, .03, jnp.array([.02, -.01]), c,
                                                 speed_floor=.05).y
        if custom:
            return speed_pitch_linearization(velocity, p[0], p[1])
        return velocity(p[0], p[1])
    p = jnp.array([2., pitch, 1., .7])
    direct = jax.jit(lambda p: run(p, False))
    custom = jax.jit(lambda p: run(p, True))
    np.testing.assert_array_equal(custom(p), direct(p))
    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(jax.jit(derivative(custom))(p),
                                   jax.jit(derivative(direct))(p), rtol=3e-13, atol=3e-13)


def test_parameter_free_map_and_batched_higher_derivatives():
    def run(p, custom):
        function = lambda v, xi: jnp.array([jnp.sin(v)*xi, jnp.exp(.1*v)+xi*xi])
        state = speed_pitch_linearization(function, p[0], p[1]) if custom else function(*p)
        return jnp.sum(state**2)
    points = jnp.array([[.3, -.2], [.7, .4]])
    expected = jax.jit(jax.vmap(jax.hessian(lambda p: run(p, False))))(points)
    actual = jax.jit(jax.vmap(jax.hessian(lambda p: run(p, True))))(points)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


def test_active_profile_vector_and_captured_integer_index():
    def run(p, custom):
        profile, index = p[2:], jnp.array(1)
        def velocity(v, xi):
            density = jnp.interp(.2*v+.1*xi, jnp.array([0., .5, 1.]), profile)
            return jnp.array([v+.03*density, xi+.02*profile[index]*v])
        if custom:
            return speed_pitch_linearization(velocity, p[0], p[1])
        return velocity(p[0], p[1])
    p = jnp.array([1.1, .2, .7, .9, 1.3])
    direct = jax.jit(lambda p: run(p, False))
    custom = jax.jit(lambda p: run(p, True))
    np.testing.assert_array_equal(custom(p), direct(p))
    np.testing.assert_allclose(jax.jit(jax.jacrev(custom))(p),
                               jax.jit(jax.jacrev(direct))(p), rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(custom))(p),
                               jax.jit(jax.jacfwd(direct))(p), rtol=3e-13, atol=3e-13)
