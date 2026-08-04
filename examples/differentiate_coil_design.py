#!/usr/bin/env python
"""Walk through ESSOS' differentiable current and coil-shape field API.

This one fast example covers the complete public ``essos.coil_design`` surface
and ``FilamentaryBiotSavart`` without depending on an equilibrium solver. It
prints diagnostics only and writes no files.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from essos.coil_design import (  # noqa: E402
    CoilDesignFieldBuilder,
    ShapeDeformationMetrics,
    make_coil_design_field_builder,
    make_fractional_current_field_builder,
    make_shape_field_builder,
    shape_deformation_metrics,
)
from essos.coils import Coils_from_json  # noqa: E402
from essos.fields import FilamentaryBiotSavart  # noqa: E402


COILS_JSON = (
    Path(__file__).resolve().parent
    / "input_files"
    / "ESSOS_biot_savart_LandremanPaulQA.json"
)


def main() -> None:
    coils = Coils_from_json(COILS_JSON)
    print(f"independent physical currents [A]: {np.asarray(coils.base_currents)}")

    radius = jnp.asarray(0.90)
    phi = jnp.asarray(0.05)
    height = jnp.asarray(0.02)
    point = jnp.asarray([radius * jnp.cos(phi), radius * jnp.sin(phi), height])
    direct = FilamentaryBiotSavart.from_coils(coils)
    cartesian = direct.B(point)
    callable_cartesian = direct(point)
    br, bphi, bz = direct.b_cyl(
        radius,
        phi,
        height,
    )
    np.testing.assert_allclose(callable_cartesian, cartesian, rtol=0.0, atol=0.0)
    cylindrical_as_cartesian = jnp.asarray(
        [
            jnp.cos(phi) * br - jnp.sin(phi) * bphi,
            jnp.sin(phi) * br + jnp.cos(phi) * bphi,
            bz,
        ]
    )
    np.testing.assert_allclose(cylindrical_as_cartesian, cartesian, rtol=2.0e-15)
    print(f"direct Cartesian B [T]: {np.asarray(cartesian)}")
    print(f"direct cylindrical B [T]: {np.asarray([br, bphi, bz])}")

    current_only = make_fractional_current_field_builder(coils, groups=(2, 3))
    scalar_current = make_fractional_current_field_builder(coils, groups=2)
    shape_only = make_shape_field_builder(
        coils,
        shape_dofs=((2, 2, 1),),
    )
    scalar_current.field_from_scalar_current(jnp.asarray(0.01)).B(point)
    current_only(jnp.asarray([0.01, -0.02])).B(point)
    shape_only(jnp.asarray([0.001])).B(point)

    builder = make_coil_design_field_builder(
        coils,
        current_groups=(2, 3),
        shape_dofs=((2, 2, 1),),
    )
    if not isinstance(builder, CoilDesignFieldBuilder):
        raise TypeError("factory did not return CoilDesignFieldBuilder")
    parameters = jnp.asarray([0.01, -0.02, 0.001])
    curve_dofs = builder.curve_dofs_at(parameters)
    base_currents = builder.base_currents_at(parameters)
    expanded_currents = builder.expanded_currents_at(parameters)
    rebuilt_curves = builder.rebuild_curves(parameters)
    rebuilt_coils = builder.rebuild_coils(parameters)
    field = builder(parameters)
    np.testing.assert_allclose(rebuilt_curves.dofs, curve_dofs)
    np.testing.assert_allclose(rebuilt_coils.base_currents, base_currents)
    np.testing.assert_allclose(field.currents, expanded_currents)

    zero = jnp.zeros(builder.parameter_shape)
    shape_direction = jnp.asarray([0.0, 0.0, 1.0])
    shape_field, shape_tangent = jax.jvp(
        builder,
        (zero,),
        (shape_direction,),
    )
    deformation = shape_deformation_metrics(
        shape_field.gamma,
        shape_tangent.gamma,
        coil_index=2,
    )
    if not isinstance(deformation, ShapeDeformationMetrics):
        raise TypeError("shape diagnostic did not return ShapeDeformationMetrics")

    print(
        "joint builder: "
        f"{builder.current_parameter_count} current + "
        f"{builder.shape_parameter_count} shape parameters"
    )
    print(f"joint field B [T]: {np.asarray(field.B(point))}")
    print(
        "shape deformation: "
        f"pair-distance RMS={deformation.pair_distance_derivative_rms:.6e}, "
        f"rigid-fit ratio={deformation.rigid_motion_fit_residual_ratio:.6e}, "
        f"length derivative={deformation.length_derivative:+.6e}"
    )

    # A shape coordinate need not be one Fourier coefficient. Each row of an
    # explicit direction stack can describe an arbitrary chart direction over
    # several independent coils and coefficients. Current fractions still
    # come first in the parameter vector, followed by these shape amplitudes.
    shape_directions = np.zeros((2, *np.shape(coils.dofs_curves)))
    shape_directions[0, 0, 2, 1] = 1.0
    shape_directions[0, 1, 2, 1] = -0.5
    shape_directions[1, 2, 0, 2] = 0.75
    shape_directions[1, 3, 1, 2] = -0.25
    arbitrary_chart = make_coil_design_field_builder(
        coils,
        current_groups=(0, 2),
        shape_directions=shape_directions,
    )
    chart_labels = (
        "fractional current group 0",
        "fractional current group 2",
        "shape direction 0 [m]",
        "shape direction 1 [m]",
    )
    chart_parameters = jnp.asarray([0.015, -0.010, 0.0015, -0.0008])
    chart_direction = jnp.asarray([0.4, -0.3, 0.25, -0.2])
    if arbitrary_chart.parameter_shape != chart_parameters.shape:
        raise ValueError("arbitrary chart parameter ordering is inconsistent")

    chart_curves = arbitrary_chart.rebuild_curves(chart_parameters)
    chart_coils = arbitrary_chart.rebuild_coils(chart_parameters)
    chart_field = arbitrary_chart(chart_parameters)
    np.testing.assert_allclose(chart_coils.dofs_curves, chart_curves.dofs)
    np.testing.assert_allclose(
        chart_field.currents,
        arbitrary_chart.expanded_currents_at(chart_parameters),
    )
    chart_value, chart_tangent = jax.jvp(
        lambda values: arbitrary_chart(values).B(point),
        (chart_parameters,),
        (chart_direction,),
    )
    chart_tangent_norm = float(jnp.linalg.norm(chart_tangent))
    if not np.all(np.isfinite(np.asarray(chart_value))):
        raise ValueError("arbitrary chart field is non-finite")
    if not np.isfinite(chart_tangent_norm) or chart_tangent_norm <= 0.0:
        raise ValueError("arbitrary chart field JVP must be finite and nonzero")

    coil_lengths = np.asarray(chart_curves.length)
    if not np.all(np.isfinite(coil_lengths)):
        raise ValueError("rebuilt coil lengths are non-finite")
    print(
        "arbitrary multi-shape chart: "
        f"{arbitrary_chart.current_parameter_count} current + "
        f"{arbitrary_chart.shape_parameter_count} shape parameters"
    )
    print("arbitrary chart parameter order:")
    for index, (label, value) in enumerate(
        zip(chart_labels, chart_parameters, strict=True)
    ):
        print(f"  [{index}] {label}: {float(value):+.6e}")
    print(f"arbitrary chart field B [T]: {np.asarray(chart_value)}")
    print(f"arbitrary chart field JVP norm [T]: {chart_tangent_norm:.6e}")
    print(
        "rebuilt physical-coil length range [m]: "
        f"{float(np.min(coil_lengths)):.6e} to "
        f"{float(np.max(coil_lengths)):.6e}"
    )


if __name__ == "__main__":
    main()
