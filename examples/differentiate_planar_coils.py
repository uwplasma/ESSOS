#!/usr/bin/env python
"""Build and differentiate planar coils without leaving their moving planes."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from essos.coil_design import (  # noqa: E402
    PlanarCoilDesignFieldBuilder,
    make_planar_coil_design_field_builder,
)
from essos.planar_coils import PlanarCoils, PlanarXYCurves  # noqa: E402


def main() -> None:
    centers = jnp.asarray([[1.4, 0.0, 0.0], [0.0, 1.35, 0.1]])
    half_angle = jnp.pi / 4
    quaternions = jnp.asarray(
        [
            [jnp.cos(half_angle), 0.0, jnp.sin(half_angle), 0.0],
            [jnp.cos(half_angle), -jnp.sin(half_angle), 0.0, 0.0],
        ]
    )
    # Local ordering is [sin(1), cos(1), sin(2), cos(2), ...].
    xy_dofs = jnp.asarray(
        [
            [[0.0, 0.28, 0.015, -0.010], [0.22, 0.0, 0.005, 0.012]],
            [[0.0, 0.24, -0.010, 0.006], [0.20, 0.0, 0.008, -0.004]],
        ]
    )
    planar_curves = PlanarXYCurves(
        centers,
        quaternions,
        xy_dofs,
        n_segments=32,
        nfp=2,
        stellsym=True,
    )
    planar_coils = PlanarCoils(planar_curves, jnp.asarray([1.1e5, -7.5e4]))

    builder = make_planar_coil_design_field_builder(
        planar_coils,
        current_groups=(0,),
        shape_dofs=((0, 0, 1), (1, 1, 2)),
        center_dofs=((0, 0), (1, 2)),
        orientation_dofs=((0, 1), (1, 0)),
    )
    if not isinstance(builder, PlanarCoilDesignFieldBuilder):
        raise TypeError("factory did not return PlanarCoilDesignFieldBuilder")

    labels = (
        "fractional current group 0",
        "coil 0 local-X cos(1) [m]",
        "coil 1 local-Y sin(2) [m]",
        "coil 0 center-X [m]",
        "coil 1 center-Z [m]",
        "coil 0 rotation-Y [rad]",
        "coil 1 rotation-X [rad]",
    )
    parameters = jnp.asarray([0.02, 0.006, -0.004, 0.025, -0.015, 0.08, -0.06])
    direction = jnp.asarray([0.3, -0.2, 0.15, 0.4, -0.1, 0.5, -0.25])
    point = jnp.asarray([0.85, 0.10, 0.03])
    field = builder(parameters)
    value, tangent = jax.jvp(
        lambda values: builder(values).B(point),
        (parameters,),
        (direction,),
    )

    curves = builder.rebuild_curves(parameters)
    rebuilt_planar_coils = builder.rebuild_planar_coils(parameters)
    centers_at_parameters = builder.centers_at(parameters)
    normals = builder.normals_at(parameters)
    base_gamma = curves.gamma[: planar_curves.n_base_curves]
    planarity_residual = jnp.einsum(
        "nc,nsc->ns",
        normals,
        base_gamma - centers_at_parameters[:, None, :],
    )
    max_planarity_residual = float(jnp.max(jnp.abs(planarity_residual)))
    if max_planarity_residual > 1.0e-12:
        raise ValueError("planarity invariant was not preserved")
    if not np.allclose(rebuilt_planar_coils.gamma, curves.gamma, atol=5.0e-15):
        raise ValueError("native planar rebuild changed the physical geometry")
    if not np.all(np.isfinite(np.asarray(tangent))):
        raise ValueError("planar field JVP is non-finite")

    # SIMSOPT-style radial coefficients convert exactly into the native XY form.
    polar_circle = PlanarXYCurves.from_polar_radius(
        centers=jnp.asarray([[1.2, 0.0, 0.0]]),
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        radius_dofs=jnp.asarray([[0.25]]),
        n_segments=32,
        stellsym=False,
    )

    print(
        "planar builder blocks: "
        f"{builder.current_parameter_count} current + "
        f"{builder.shape_parameter_count} local shape + "
        f"{builder.center_parameter_count} center + "
        f"{builder.orientation_parameter_count} orientation"
    )
    print("planar parameter order:")
    for index, (label, parameter) in enumerate(zip(labels, parameters, strict=True)):
        print(f"  [{index}] {label}: {float(parameter):+.6e}")
    print(f"planar field B [T]: {np.asarray(field.B(point))}")
    print(f"planar field JVP [T]: {np.asarray(tangent)}")
    print(f"max planarity residual [m]: {max_planarity_residual:.6e}")
    print(f"polar compatibility XY order: {polar_circle.order}")


if __name__ == "__main__":
    main()
