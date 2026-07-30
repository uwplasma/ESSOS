from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from essos.coils import Coils, Curves
from essos.field_jet import (
    FieldJetResiduals,
    NearAxisFieldJetTarget,
    field_jet_loss_terms,
    loss_field_jet_coils_near_axis,
    near_axis_field_jet_target,
    normalized_field_jet_residuals,
    pack_symmetric_trace_free_rank2,
    pack_symmetric_trace_free_rank3,
)
from essos.fields import BiotSavart, MagneticField


@jax.tree_util.register_pytree_node_class
class PolynomialField(MagneticField):
    def __init__(self, offset):
        self.offset = jnp.asarray(offset)
        self.linear = jnp.asarray(
            [
                [1.0, 0.2, -0.1],
                [0.2, -0.4, 0.3],
                [-0.1, 0.3, -0.6],
            ]
        )
        self.quadratic = jnp.asarray(
            [
                [[0.3, -0.2, 0.1], [-0.2, 0.4, 0.0], [0.1, 0.0, -0.1]],
                [[-0.2, 0.4, 0.0], [0.4, -0.1, 0.2], [0.0, 0.2, -0.3]],
                [[0.1, 0.0, -0.1], [0.0, 0.2, -0.3], [-0.1, -0.3, -0.2]],
            ]
        )

    def B(self, point):
        return (
            self.offset
            + self.linear @ point
            + jnp.einsum("ijk,j,k->i", self.quadratic, point, point) / 2
        )

    def tree_flatten(self):
        return (self.offset,), {}

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(children[0])


def _matching_polynomial_target(field):
    points = jnp.asarray(
        [
            [0.8, 0.1, -0.2],
            [0.4, -0.3, 0.5],
        ]
    )
    values = jax.vmap(field.B)(points)
    gradients = jax.vmap(field.dB_by_dX)(points)
    hessians = jax.vmap(field.d2B_by_dXdX)(points)
    return NearAxisFieldJetTarget(
        points=points,
        field=values,
        gradient=gradients,
        hessian=hessians,
        gradient_independent=pack_symmetric_trace_free_rank2(gradients),
        hessian_independent=pack_symmetric_trace_free_rank3(hessians),
        reference_field=jnp.asarray(2.0),
        reference_length=jnp.asarray(3.0),
        is_external=False,
    )


def _vacuum_solution(nphi=31):
    return qsc.Qsc(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=0.64,
        B2c=-0.00322,
        nphi=nphi,
        order="r2",
    )


def _finite_current_solution(nphi=31):
    return qsc.Qsc(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=0.9,
        p2=-600000.0,
        B2c=-0.7,
        nphi=nphi,
        order="r2",
    )


def test_magnetic_field_hessian_matches_polynomial_coefficients():
    field = PolynomialField(jnp.asarray([0.2, -0.1, 0.4]))
    point = jnp.asarray([0.8, 0.1, -0.2])

    np.testing.assert_allclose(
        field.d2B_by_dXdX(point),
        field.quadratic,
    )


def test_matching_target_has_zero_normalized_3_5_7_residuals():
    field = PolynomialField(jnp.asarray([0.2, -0.1, 0.4]))
    target = _matching_polynomial_target(field)
    residuals = normalized_field_jet_residuals(field, target)

    assert isinstance(residuals, FieldJetResiduals)
    assert residuals.field.shape == (2, 3)
    assert residuals.gradient.shape == (2, 5)
    assert residuals.hessian.shape == (2, 7)
    np.testing.assert_allclose(residuals.field, 0.0)
    np.testing.assert_allclose(residuals.gradient, 0.0)
    np.testing.assert_allclose(residuals.hessian, 0.0)
    np.testing.assert_allclose(field_jet_loss_terms(field, target), 0.0)


def test_field_jet_loss_is_smooth_normalized_and_differentiable():
    reference = PolynomialField(jnp.asarray([0.2, -0.1, 0.4]))
    target = _matching_polynomial_target(reference)

    def loss(offset):
        return loss_field_jet_coils_near_axis(
            PolynomialField(offset),
            target,
            weights=(2.0, 3.0, 4.0),
        )

    offset = jnp.asarray([0.25, -0.07, 0.38])
    value = loss(offset)
    gradient = jax.grad(loss)(offset)
    compiled = jax.jit(loss)(offset)
    step = 1.0e-5
    finite_difference = (
        loss(offset.at[0].add(step)) - loss(offset.at[0].add(-step))
    ) / (2 * step)

    np.testing.assert_allclose(compiled, value)
    np.testing.assert_allclose(gradient[0], finite_difference, rtol=2.0e-10)
    assert value > 0


def test_vacuum_external_target_reduces_exactly_to_total_target():
    solution = _vacuum_solution()
    total = near_axis_field_jet_target(solution)
    external = near_axis_field_jet_target(
        solution,
        formal_radius=0.05,
    )

    assert not total.is_external
    assert external.is_external
    np.testing.assert_allclose(external.points, total.points)
    np.testing.assert_allclose(external.field, total.field)
    np.testing.assert_allclose(external.gradient, total.gradient)
    np.testing.assert_allclose(external.hessian, total.hessian)
    np.testing.assert_allclose(
        external.gradient_independent,
        total.gradient_independent,
    )
    np.testing.assert_allclose(
        external.hessian_independent,
        total.hessian_independent,
    )


def test_finite_current_target_uses_external_not_total_field_jet():
    solution = _finite_current_solution(nphi=61)
    total = near_axis_field_jet_target(solution)
    external = near_axis_field_jet_target(
        solution,
        formal_radius=0.05,
    )

    assert external.is_external
    assert external.gradient_independent.shape == (61, 5)
    assert external.hessian_independent.shape == (61, 7)
    assert np.max(np.abs(np.asarray(external.field - total.field))) > 1.0e-6
    assert np.max(np.abs(np.asarray(external.gradient - total.gradient))) > 1.0e-3
    assert np.max(np.abs(np.asarray(external.hessian - total.hessian))) > 1.0e-3


def test_legacy_near_axis_adapter_is_accepted():
    legacy = qsc.near_axis(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=0.64,
        B2c=-0.00322,
        nphi=31,
        order="r2",
    )
    target = near_axis_field_jet_target(legacy)

    np.testing.assert_allclose(
        target.points, legacy.solution.geometry.position_cartesian
    )
    np.testing.assert_allclose(target.field, legacy.solution.B_axis)


def test_actual_biot_savart_field_jet_has_coil_current_gradient():
    points = jnp.asarray([[0.0, 0.0, 0.2], [0.1, 0.0, -0.2]])
    zero_gradient = jnp.zeros((2, 3, 3))
    zero_hessian = jnp.zeros((2, 3, 3, 3))
    target = NearAxisFieldJetTarget(
        points=points,
        field=jnp.zeros((2, 3)),
        gradient=zero_gradient,
        hessian=zero_hessian,
        gradient_independent=pack_symmetric_trace_free_rank2(zero_gradient),
        hessian_independent=pack_symmetric_trace_free_rank3(zero_hessian),
        reference_field=jnp.asarray(1.0),
        reference_length=jnp.asarray(1.0),
        is_external=False,
    )

    def loss(parameters):
        dofs = jnp.zeros((1, 3, 3))
        dofs = dofs.at[0, 0, 2].set(parameters[0])
        dofs = dofs.at[0, 1, 1].set(1.0)
        curves = Curves(
            dofs,
            n_segments=24,
            nfp=1,
            stellsym=False,
        )
        coils = Coils(
            curves,
            jnp.atleast_1d(parameters[1]),
            currents_scale=1.0e6,
        )
        return loss_field_jet_coils_near_axis(BiotSavart(coils), target)

    parameters = jnp.asarray([1.0, 1.0e6])
    value, derivative = jax.value_and_grad(loss)(parameters)
    steps = (1.0e-5, 1.0)
    finite_difference = []
    for index, step in enumerate(steps):
        direction = jnp.zeros(2).at[index].set(step)
        finite_difference.append(
            (loss(parameters + direction) - loss(parameters - direction)) / (2 * step)
        )

    assert np.isfinite(value)
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=2.0e-7,
    )


def test_finite_current_objective_has_axis_and_near_axis_gradients():
    coil_field = PolynomialField(jnp.asarray([0.2, -0.1, 0.4]))

    def loss(parameters):
        solution = qsc.Qsc(
            rc=jnp.stack((jnp.asarray(1.0), parameters[0])),
            zs=[0.0, -0.09],
            nfp=2,
            etabar=parameters[1],
            I2=0.9,
            p2=-600000.0,
            B2c=-0.7,
            nphi=15,
            order="r2",
        )
        target = near_axis_field_jet_target(
            solution,
            formal_radius=0.05,
            angular_resolution=32,
        )
        return loss_field_jet_coils_near_axis(coil_field, target)

    parameters = jnp.asarray([0.09, 0.95])
    value, derivative = jax.value_and_grad(loss)(parameters)
    finite_difference = []
    for index in range(2):
        direction = jnp.zeros(2).at[index].set(1.0e-5)
        finite_difference.append(
            (loss(parameters + direction) - loss(parameters - direction)) / (2.0e-5)
        )

    assert np.isfinite(value)
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_target_and_weight_guards():
    first_order = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=15,
    )
    with pytest.raises(ValueError, match="r2 or r3"):
        near_axis_field_jet_target(first_order)
    with pytest.raises(ValueError, match="positive"):
        near_axis_field_jet_target(_vacuum_solution(), reference_field=0.0)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        loss_field_jet_coils_near_axis(
            PolynomialField(jnp.zeros(3)),
            _matching_polynomial_target(PolynomialField(jnp.zeros(3))),
            weights=(1.0, 2.0),
        )
