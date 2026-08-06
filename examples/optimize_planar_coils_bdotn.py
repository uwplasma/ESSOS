#!/usr/bin/env python
"""Optimize a planar filament coil against a small normal-field target."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from essos.coil_design import make_planar_coil_design_field_builder  # noqa: E402
from essos.optimization import optimize_planar_residual  # noqa: E402
from essos.planar_coils import PlanarCoils, PlanarXYCurves  # noqa: E402


def main() -> None:
    curves = PlanarXYCurves(
        centers=jnp.asarray([[1.4, 0.0, 0.0]]),
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        xy_dofs=jnp.asarray([[[0.0, 0.28], [0.22, 0.0]]]),
        n_segments=48,
        stellsym=False,
    )
    coils = PlanarCoils(curves, jnp.asarray([1.1e5]))
    builder = make_planar_coil_design_field_builder(
        coils,
        current_groups=(0,),
        shape_dofs=((0, 0, 1),),
    )

    probe_points = jnp.asarray(
        [[1.4, 0.0, 0.16], [1.4, 0.0, 0.27], [1.4, 0.0, 0.40]]
    )
    probe_normals = jnp.asarray([[0.0, 0.0, 1.0]] * len(probe_points))

    def normal_field(parameters):
        magnetic_field = builder(parameters).B(probe_points)
        return jnp.sum(magnetic_field * probe_normals, axis=-1)

    target_parameters = jnp.asarray([0.12, 0.03])
    target_normal_field = normal_field(target_parameters)
    field_scale = jnp.max(jnp.abs(target_normal_field))

    def residual(parameters):
        return (normal_field(parameters) - target_normal_field) / field_scale

    initial_parameters = jnp.zeros(builder.parameter_shape)
    initial_objective = 0.5 * jnp.sum(residual(initial_parameters) ** 2)
    optimized_coils, result = optimize_planar_residual(
        residual,
        builder,
        initial_parameters,
        tolerance_optimization=1.0e-12,
        maximum_function_evaluations=40,
    )
    optimized_parameters = jnp.asarray(result.x)
    final_objective = 0.5 * jnp.sum(residual(optimized_parameters) ** 2)

    base_gamma = optimized_coils.gamma[: optimized_coils.planar_curves.n_base_curves]
    planarity_residual = jnp.einsum(
        "nc,nsc->ns",
        optimized_coils.normals,
        base_gamma - optimized_coils.centers[:, None, :],
    )
    max_planarity_residual = float(jnp.max(jnp.abs(planarity_residual)))

    if not result.success:
        raise RuntimeError(f"planar optimization failed: {result.message}")
    if not float(final_objective) < float(initial_objective):
        raise RuntimeError("planar optimization did not lower the objective")
    if max_planarity_residual > 1.0e-12:
        raise RuntimeError("optimized coils left the planar design manifold")

    print(f"optimizer success: {result.success}")
    print(f"initial objective: {float(initial_objective):.6e}")
    print(f"final objective: {float(final_objective):.6e}")
    print(f"optimized parameters: {np.asarray(optimized_parameters)}")
    print(f"max planarity residual [m]: {max_planarity_residual:.6e}")


if __name__ == "__main__":
    main()
