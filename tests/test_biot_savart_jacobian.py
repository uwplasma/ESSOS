"""Physical, AD and finite-difference checks of the fused Biot--Savart Jacobian."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from essos.coils import Coils, Curves, CreateEquallySpacedCurves
from essos.fields import BiotSavart, _gc_from_jacobian

jax.config.update("jax_enable_x64", True)


def make_field(stellsym=False):
    curves = CreateEquallySpacedCurves(2, 2, 1.7, .6, 32, 2 if stellsym else 1, stellsym)
    return BiotSavart(Coils(curves, [1.1e6, -.9e6]))


@pytest.mark.parametrize("stellsym", [False, True])
def test_fused_values_and_jacobians_match_the_original_field_ad(stellsym):
    field = make_field(stellsym)
    points = jnp.array([[1.4, .2, .1], [1.6, -.1, -.15], [1.8, .1, .2]])
    values, jacobians = jax.vmap(field.B_and_dB_by_dX)(points)
    np.testing.assert_allclose(values, jax.vmap(field.B)(points), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(jacobians, jax.vmap(jax.jacfwd(field.B))(points), rtol=2e-12, atol=2e-12)


def test_float32_currents_do_not_reduce_float64_jacobian_precision():
    class SampledCoils(NamedTuple):
        gamma: object
        gamma_dash: object
        currents: object

    curves = CreateEquallySpacedCurves(2, 2, 1.7, .6, 30, 1, False)
    field = BiotSavart(SampledCoils(curves.gamma, curves.gamma_dash,
                                  jnp.array([1.1e6, -.9e6], dtype=jnp.float32)))
    point = jnp.array([1.4, .2, .1], dtype=jnp.float64)
    magnetic, jacobian = field.B_and_dB_by_dX(point)
    assert jacobian.dtype == jnp.float64
    np.testing.assert_allclose(magnetic, field.B(point), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(jacobian, jax.jacfwd(field.B)(point), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("quantities", ["jacobian", "guiding_center"])
def test_position_shape_current_and_higher_derivatives_match_ad_and_finite_difference(quantities):
    flat, rebuild = ravel_pytree(make_field())
    initial = jnp.concatenate([jnp.array([1.4, .2, .1]), flat])

    def observe(values, fused):
        point, field = values[:3], rebuild(values[3:])
        if fused:
            result = field.B_and_dB_by_dX(point)
        else:
            jacobian, magnetic = jax.jacfwd(lambda x: (field.B(x), field.B(x)), has_aux=True)(point)
            result = magnetic, jacobian
        if quantities == "guiding_center":
            result = _gc_from_jacobian(*result)
        return ravel_pytree(result)[0]

    direction = jnp.linspace(-.02, .03, len(initial))
    value, tangent = jax.jvp(lambda x: observe(x, True), (initial,), (direction,))
    ref_value, ref_tangent = jax.jvp(lambda x: observe(x, False), (initial,), (direction,))
    np.testing.assert_allclose(value, ref_value, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tangent, ref_tangent, rtol=2e-10, atol=2e-10)
    cotangent = jnp.linspace(-.4, .6, len(value))
    gradient = jax.vjp(lambda x: observe(x, True), initial)[1](cotangent)[0]
    ref_gradient = jax.vjp(lambda x: observe(x, False), initial)[1](cotangent)[0]
    np.testing.assert_allclose(gradient, ref_gradient, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(jnp.vdot(tangent, cotangent), jnp.vdot(direction, gradient),
                               rtol=2e-11, atol=2e-11)
    epsilon = 1e-5
    finite = (observe(initial + epsilon * direction, True)
              - observe(initial - epsilon * direction, True)) / (2 * epsilon)
    np.testing.assert_allclose(tangent, finite, rtol=2e-6, atol=2e-8)


def test_circular_coil_axis_matches_the_analytic_field_and_jacobian():
    radius, current = 1.2, 2.3e5
    dofs = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(radius).at[0, 1, 1].set(radius)
    field = BiotSavart(Coils(Curves(dofs, n_segments=32, stellsym=False), jnp.array([current])))
    z = jnp.array([-.4, .1, .5])
    points = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z], axis=1)
    values, jacobians = jax.vmap(field.B_and_dB_by_dX)(points)
    bz = 2 * jnp.pi * 1e-7 * current * radius**2 / (radius**2 + z**2)**1.5
    dzz = -3 * z * bz / (radius**2 + z**2)
    expected = jax.vmap(jnp.diag)(jnp.stack([-.5 * dzz, -.5 * dzz, dzz], axis=1))
    np.testing.assert_allclose(values[:, :2], 0., atol=1e-15)
    np.testing.assert_allclose(values[:, 2], bz, rtol=2e-13, atol=1e-15)
    np.testing.assert_allclose(jacobians, expected, rtol=2e-13, atol=1e-15)
