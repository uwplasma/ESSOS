"""Normalized coil objectives for near-axis vacuum field jets."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

import jax
import jax.numpy as jnp
from pyqsc_jax import (
    pack_symmetric_trace_free_rank2,
    pack_symmetric_trace_free_rank3,
)

ArrayLike = Any


def _positive_scalar(value: ArrayLike, *, name: str) -> jax.Array:
    value = jnp.asarray(value)
    if value.ndim:
        raise ValueError(f"{name} must be a scalar.")
    try:
        if bool(value <= 0):
            raise ValueError(f"{name} must be positive.")
    except jax.errors.TracerBoolConversionError:
        pass
    return value


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NearAxisFieldJetTarget:
    """Immutable 3+5+7 vacuum target sampled on a magnetic axis."""

    points: jax.Array
    field: jax.Array
    gradient: jax.Array
    hessian: jax.Array
    gradient_independent: jax.Array
    hessian_independent: jax.Array
    reference_field: jax.Array
    reference_length: jax.Array
    is_external: bool = dataclass_field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FieldJetResiduals:
    """Dimensionless field, gradient, and Hessian residual blocks."""

    field: jax.Array
    gradient: jax.Array
    hessian: jax.Array


def near_axis_field_jet_target(
    near_axis,
    *,
    formal_radius: ArrayLike | None = None,
    reference_field: ArrayLike | None = None,
    reference_length: ArrayLike | None = None,
    angular_resolution: int = 128,
) -> NearAxisFieldJetTarget:
    """Construct a total-vacuum or surface-free external 3+5+7 target.

    ``formal_radius=None`` uses the total near-axis jet and is intended for a
    vacuum solution. Supplying a positive formal radius subtracts the plasma
    field, gradient, and Hessian before constructing the coil target.
    """

    solution = getattr(near_axis, "solution", near_axis)
    if solution.field_jet is None:
        raise ValueError("A field-jet target requires an r2 or r3 near-axis solution.")

    points = solution.geometry.position_cartesian
    if formal_radius is None:
        target_field = solution.B_axis
        target_gradient = solution.grad_B_axis
        target_hessian = solution.grad_grad_B_axis
        is_external = False
    else:
        from pyqsc_jax import plasma_hessian_on_axis

        external = plasma_hessian_on_axis(
            solution,
            formal_radius=formal_radius,
            angular_resolution=angular_resolution,
        )
        target_field = external.field.external_field
        target_gradient = external.field.external_gradient
        target_hessian = external.external_hessian
        is_external = True

    if reference_field is None:
        reference_field = solution.inputs.B0
    if reference_length is None:
        reference_length = solution.geometry.abs_G0_over_B0

    return NearAxisFieldJetTarget(
        points=points,
        field=target_field,
        gradient=target_gradient,
        hessian=target_hessian,
        gradient_independent=pack_symmetric_trace_free_rank2(target_gradient),
        hessian_independent=pack_symmetric_trace_free_rank3(target_hessian),
        reference_field=_positive_scalar(reference_field, name="reference_field"),
        reference_length=_positive_scalar(reference_length, name="reference_length"),
        is_external=is_external,
    )


def normalized_field_jet_residuals(
    coil_field,
    target: NearAxisFieldJetTarget,
) -> FieldJetResiduals:
    """Evaluate dimensionless 3+5+7 residual blocks for a coil field."""

    coil_values = jax.vmap(coil_field.B)(target.points)
    coil_gradients = jax.vmap(coil_field.dB_by_dX)(target.points)
    coil_hessians = jax.vmap(coil_field.d2B_by_dXdX)(target.points)
    return FieldJetResiduals(
        field=(coil_values - target.field) / target.reference_field,
        gradient=(
            target.reference_length
            * (
                pack_symmetric_trace_free_rank2(coil_gradients)
                - target.gradient_independent
            )
            / target.reference_field
        ),
        hessian=(
            target.reference_length**2
            * (
                pack_symmetric_trace_free_rank3(coil_hessians)
                - target.hessian_independent
            )
            / target.reference_field
        ),
    )


def field_jet_loss_terms(
    coil_field,
    target: NearAxisFieldJetTarget,
) -> jax.Array:
    """Return mean-square ``(field, gradient, Hessian)`` objective terms."""

    residuals = normalized_field_jet_residuals(coil_field, target)
    return jnp.stack(
        (
            jnp.mean(jnp.square(residuals.field)),
            jnp.mean(jnp.square(residuals.gradient)),
            jnp.mean(jnp.square(residuals.hessian)),
        )
    )


def loss_field_jet_coils_near_axis(
    coil_field,
    target: NearAxisFieldJetTarget,
    *,
    weights: ArrayLike = (1.0, 1.0, 1.0),
) -> jax.Array:
    """Return a smooth weighted least-squares loss for a near-axis field jet."""

    weights = jnp.asarray(weights)
    if weights.shape != (3,):
        raise ValueError("weights must have shape (3,).")
    return jnp.dot(weights, field_jet_loss_terms(coil_field, target))
