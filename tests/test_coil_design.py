import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import essos.coil_design as coil_design
from essos.coil_design import (
    PlanarCoilDesignFieldBuilder,
    make_coil_design_field_builder,
    make_fractional_current_field_builder,
    make_planar_coil_design_field_builder,
    make_shape_field_builder,
    shape_deformation_metrics,
)
from essos.coils import (
    Coils,
    CreateEquallySpacedCurves,
    Curves,
    apply_symmetries_to_currents,
)
from essos.fields import BiotSavart, FilamentaryBiotSavart
from essos.planar_coils import PlanarCoils, PlanarXYCurves


def test_coil_design_public_surface_is_explicit():
    assert coil_design.__all__ == [
        "CoilDesignFieldBuilder",
        "PlanarCoilDesignFieldBuilder",
        "ShapeDeformationMetrics",
        "make_coil_design_field_builder",
        "make_fractional_current_field_builder",
        "make_planar_coil_design_field_builder",
        "make_shape_field_builder",
        "shape_deformation_metrics",
    ]


def _coils(*, stellsym=True):
    curves = CreateEquallySpacedCurves(
        3,
        order=2,
        R=1.4,
        r=0.25,
        n_segments=24,
        nfp=2,
        stellsym=stellsym,
    )
    return Coils(curves, jnp.asarray([1.0e5, -8.0e4, 6.0e4]))


def _assert_same_field(first, second):
    np.testing.assert_array_equal(first.gamma, second.gamma)
    np.testing.assert_array_equal(first.gamma_dash, second.gamma_dash)
    np.testing.assert_array_equal(first.currents, second.currents)


def _planar_coils(*, stellsym=True):
    centers = jnp.asarray([[1.4, 0.0, 0.0], [0.0, 1.35, 0.1]])
    half_angle = jnp.pi / 4
    quaternions = jnp.asarray(
        [
            [jnp.cos(half_angle), 0.0, jnp.sin(half_angle), 0.0],
            [jnp.cos(half_angle), -jnp.sin(half_angle), 0.0, 0.0],
        ]
    )
    xy_dofs = jnp.asarray(
        [
            [[0.0, 0.28, 0.015, -0.010], [0.22, 0.0, 0.005, 0.012]],
            [[0.0, 0.24, -0.010, 0.006], [0.20, 0.0, 0.008, -0.004]],
        ]
    )
    planar = PlanarXYCurves(
        centers,
        quaternions,
        xy_dofs,
        n_segments=28,
        nfp=2,
        stellsym=stellsym,
    )
    coils = Coils(planar.as_curves(), jnp.asarray([1.1e5, -7.5e4]))
    return planar, coils


@pytest.mark.parametrize("stellsym", [False, True])
def test_fractional_current_builder_uses_essos_symmetry_and_is_jvp_traceable(stellsym):
    coils = _coils(stellsym=stellsym)
    builder = make_fractional_current_field_builder(coils, groups=(0, 2))
    parameters = jnp.asarray([0.10, -0.25])
    field = builder(parameters)

    expected_base = np.asarray(coils.base_currents).copy()
    expected_base[0] *= 1.10
    expected_base[2] *= 0.75
    expected_expanded = apply_symmetries_to_currents(
        expected_base, coils.nfp, coils.stellsym
    )
    np.testing.assert_allclose(
        field.currents, expected_expanded, rtol=2.0e-16, atol=0.0
    )
    assert builder.current_groups == (0, 2)
    assert builder.parameter_shape == (2,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        builder.current_groups = (1, 2)

    direction = jnp.asarray([0.5, -1.25])
    primal, tangent = jax.jvp(builder, (parameters,), (direction,))
    expected_tangent = direction @ builder.fractional_expanded_directions
    np.testing.assert_array_equal(tangent.gamma, np.zeros_like(tangent.gamma))
    np.testing.assert_array_equal(tangent.gamma_dash, np.zeros_like(tangent.gamma_dash))
    np.testing.assert_allclose(
        tangent.currents, expected_tangent, rtol=2.0e-16, atol=0.0
    )
    np.testing.assert_array_equal(
        jax.jit(builder)(parameters).currents, primal.currents
    )


def test_one_current_builder_has_a_scalar_adapter():
    builder = make_fractional_current_field_builder(_coils(), groups=1)
    scalar = jnp.asarray(0.125)

    _assert_same_field(builder.field_from_scalar_current(scalar), builder(scalar[None]))
    with pytest.raises(ValueError, match="scalar"):
        builder.field_from_scalar_current(jnp.asarray([0.125]))


def test_shape_builder_changes_selected_coefficients_and_not_currents():
    coils = _coils()
    shape_dofs = ((0, 2, 1), (1, 0, 2))
    builder = make_shape_field_builder(coils, shape_dofs=shape_dofs)
    parameters = jnp.asarray([0.004, -0.003])
    expected_dofs = np.asarray(coils.dofs_curves).copy()
    expected_dofs[shape_dofs[0]] += parameters[0]
    expected_dofs[shape_dofs[1]] += parameters[1]

    np.testing.assert_allclose(
        builder.curve_dofs_at(parameters), expected_dofs, rtol=0.0, atol=2.0e-16
    )
    primal, tangent = jax.jvp(
        builder,
        (parameters,),
        (jnp.asarray([0.7, -1.1]),),
    )
    np.testing.assert_array_equal(primal.currents, builder.nominal_expanded_currents)
    np.testing.assert_array_equal(tangent.currents, np.zeros_like(tangent.currents))
    assert np.linalg.norm(np.asarray(tangent.gamma)) > 0.0
    assert np.linalg.norm(np.asarray(tangent.gamma_dash)) > 0.0

    step = 1.0e-5
    direction = jnp.asarray([0.7, -1.1])
    plus = builder(parameters + step * direction)
    minus = builder(parameters - step * direction)
    fd_gamma = (plus.gamma - minus.gamma) / (2.0 * step)
    fd_gamma_dash = (plus.gamma_dash - minus.gamma_dash) / (2.0 * step)
    np.testing.assert_allclose(tangent.gamma, fd_gamma, rtol=2.0e-10, atol=2.0e-10)
    np.testing.assert_allclose(
        tangent.gamma_dash, fd_gamma_dash, rtol=2.0e-10, atol=2.0e-9
    )


def test_explicit_shape_direction_stack_supports_general_coordinates():
    coils = _coils()
    directions = np.zeros((2, *coils.dofs_curves.shape))
    directions[0, 0, 2, 1] = 1.0
    directions[0, 1, 2, 1] = -0.25
    directions[1, 2, 0, 2] = 0.5
    builder = make_shape_field_builder(coils, shape_directions=directions)
    parameters = jnp.asarray([0.004, -0.006])

    expected = np.asarray(coils.dofs_curves) + np.einsum(
        "s,sijk->ijk", parameters, directions
    )
    np.testing.assert_allclose(
        builder.curve_dofs_at(parameters), expected, rtol=0.0, atol=2.0e-16
    )
    assert builder.shape_dofs == ()
    assert builder.shape_parameter_count == 2


def test_shape_deformation_metrics_distinguish_deformation_from_translation():
    builder = make_shape_field_builder(_coils(), shape_dofs=((0, 2, 1),))
    zero = jnp.zeros((1,))
    field, tangent = jax.jvp(builder, (zero,), (jnp.ones_like(zero),))
    deformation = shape_deformation_metrics(field.gamma, tangent.gamma, coil_index=0)

    assert deformation.sampled_point_count == builder.n_segments
    assert (
        deformation.sampled_pair_count
        == builder.n_segments * (builder.n_segments - 1) // 2
    )
    assert deformation.pair_distance_derivative_rms > 1.0e-3
    assert deformation.rigid_motion_fit_residual_ratio > 0.1
    assert abs(deformation.length_derivative) > 1.0e-3

    translation = np.broadcast_to(
        np.asarray([0.4, -0.2, 0.1]),
        np.shape(field.gamma),
    )
    rigid = shape_deformation_metrics(field.gamma, translation, coil_index=0)
    assert rigid.pair_distance_derivative_rms < 1.0e-14
    assert rigid.rigid_motion_fit_residual_ratio < 1.0e-12
    assert abs(rigid.length_derivative) < 1.0e-14


def test_joint_builder_has_exact_current_and_shape_slices_and_field_parity():
    coils = _coils()
    groups = (0, 2)
    shape_dofs = ((0, 2, 1), (1, 0, 2))
    joint = make_coil_design_field_builder(
        coils,
        current_groups=groups,
        shape_dofs=shape_dofs,
    )
    current = make_fractional_current_field_builder(coils, groups=groups)
    shape = make_shape_field_builder(coils, shape_dofs=shape_dofs)

    current_parameters = jnp.asarray([0.12, -0.17])
    shape_parameters = jnp.asarray([0.004, -0.003])
    _assert_same_field(
        joint(jnp.concatenate((current_parameters, jnp.zeros(2)))),
        current(current_parameters),
    )
    _assert_same_field(
        joint(jnp.concatenate((jnp.zeros(2), shape_parameters))),
        shape(shape_parameters),
    )

    parameters = jnp.concatenate((current_parameters, shape_parameters))
    field = joint(parameters)
    independent = BiotSavart(joint.rebuild_coils(parameters))
    points = jnp.asarray(
        [
            [0.70, 0.10, -0.08],
            [0.85, -0.15, 0.03],
            [1.00, 0.05, 0.12],
        ]
    )
    expected = jax.vmap(independent.B)(points)
    np.testing.assert_allclose(field.B(points), expected, rtol=5.0e-15, atol=2.0e-15)

    current_direction = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    _primal, current_tangent = jax.jvp(joint, (parameters,), (current_direction,))
    np.testing.assert_array_equal(
        current_tangent.gamma, np.zeros_like(current_tangent.gamma)
    )
    np.testing.assert_array_equal(
        current_tangent.gamma_dash,
        np.zeros_like(current_tangent.gamma_dash),
    )
    assert np.linalg.norm(np.asarray(current_tangent.currents)) > 0.0

    shape_direction = jnp.asarray([0.0, 0.0, 1.0, 0.0])
    _primal, shape_tangent = jax.jvp(joint, (parameters,), (shape_direction,))
    np.testing.assert_array_equal(
        shape_tangent.currents, np.zeros_like(shape_tangent.currents)
    )
    assert np.linalg.norm(np.asarray(shape_tangent.gamma)) > 0.0
    assert np.linalg.norm(np.asarray(shape_tangent.gamma_dash)) > 0.0


def test_joint_field_jvp_matches_centered_parameter_difference():
    builder = make_coil_design_field_builder(
        _coils(),
        current_groups=(0, 2),
        shape_dofs=((0, 2, 1), (1, 0, 2)),
    )
    parameters = jnp.asarray([0.08, -0.06, 0.002, -0.001])
    direction = jnp.asarray([0.4, -0.7, 0.3, -0.2])
    points = jnp.asarray([[0.75, 0.10, 0.02], [0.95, -0.05, -0.04]])

    def evaluate(values):
        return builder(values).B(points)

    _primal, tangent = jax.jvp(evaluate, (parameters,), (direction,))
    step = 1.0e-5
    finite_difference = (
        evaluate(parameters + step * direction)
        - evaluate(parameters - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-8, atol=2.0e-12)


def test_planar_builder_zero_matches_nominal_and_exposes_parameter_blocks():
    planar, coils = _planar_coils()
    planar_coils = PlanarCoils(planar, coils.base_currents)
    builder = make_planar_coil_design_field_builder(
        planar_coils,
        current_groups=(0,),
        shape_dofs=((0, 0, 1),),
        center_dofs=((0, 2),),
        orientation_dofs=((1, 0),),
    )
    assert isinstance(builder, PlanarCoilDesignFieldBuilder)
    assert builder.parameter_shape == (4,)
    assert builder.current_parameter_slice == slice(0, 1)
    assert builder.shape_parameter_slice == slice(1, 2)
    assert builder.center_parameter_slice == slice(2, 3)
    assert builder.orientation_parameter_slice == slice(3, 4)

    zero = jnp.zeros(builder.parameter_shape)
    rebuilt_curves = builder.rebuild_curves(zero)
    rebuilt_coils = builder.rebuild_coils(zero)
    rebuilt_planar_coils = builder.rebuild_planar_coils(zero)
    field = builder(zero)
    assert isinstance(rebuilt_curves, Curves)
    assert isinstance(rebuilt_coils, Coils)
    assert isinstance(rebuilt_planar_coils, PlanarCoils)
    assert isinstance(field, FilamentaryBiotSavart)
    np.testing.assert_allclose(rebuilt_curves.dofs, coils.dofs_curves, atol=2.0e-15)
    np.testing.assert_allclose(rebuilt_coils.base_currents, coils.base_currents)
    np.testing.assert_allclose(rebuilt_planar_coils.centers, planar.centers)
    np.testing.assert_allclose(
        rebuilt_planar_coils.quaternions,
        planar.quaternions,
    )
    np.testing.assert_allclose(rebuilt_planar_coils.xy_dofs, planar.xy_dofs)
    np.testing.assert_allclose(
        rebuilt_planar_coils.base_currents,
        coils.base_currents,
    )
    np.testing.assert_allclose(field.gamma, coils.gamma, atol=5.0e-15)
    np.testing.assert_allclose(field.currents, coils.currents)


def test_unrestricted_builders_reject_planar_coils_unless_explicitly_lowered():
    planar, ordinary = _planar_coils(stellsym=False)
    planar_coils = PlanarCoils(planar, ordinary.base_currents)

    with pytest.raises(TypeError, match="make_planar_coil_design_field_builder"):
        make_coil_design_field_builder(
            planar_coils,
            shape_dofs=((0, 2, 1),),
        )
    with pytest.raises(TypeError, match="make_planar_coil_design_field_builder"):
        make_shape_field_builder(
            planar_coils,
            shape_dofs=((0, 2, 1),),
        )
    with pytest.raises(TypeError, match="make_planar_coil_design_field_builder"):
        make_fractional_current_field_builder(planar_coils, groups=(0,))

    explicitly_unrestricted = make_shape_field_builder(
        planar_coils.as_coils(),
        shape_dofs=((0, 2, 1),),
    )
    assert explicitly_unrestricted.parameter_shape == (1,)


def test_planar_builder_preserves_the_moving_plane_for_finite_steps_and_jvps():
    planar, coils = _planar_coils(stellsym=True)
    builder = make_planar_coil_design_field_builder(
        coils,
        plane_frames=planar.rotation_matrices[..., :2],
        shape_dofs=((0, 0, 1), (1, 1, 2)),
        center_dofs=((0, 0), (1, 2)),
        orientation_dofs=((0, 1), (1, 0)),
    )
    parameters = jnp.asarray([0.018, -0.012, 0.04, -0.03, 0.21, -0.17])
    curves = builder.rebuild_curves(parameters)
    planar_coils = builder.rebuild_planar_coils(parameters)
    frames = builder.frames_at(parameters)
    centers = builder.centers_at(parameters)
    normals = jnp.cross(frames[..., 0], frames[..., 1])
    base_gamma = curves.gamma[: planar.n_base_curves]
    residual = jnp.einsum("nc,nsc->ns", normals, base_gamma - centers[:, None, :])
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(planar_coils.gamma, curves.gamma, atol=3.0e-15)
    np.testing.assert_allclose(
        planar_coils.base_currents,
        builder.base_currents_at(parameters),
    )

    compiled_gamma = jax.jit(
        lambda values: builder.rebuild_planar_coils(values).gamma
    )(parameters)
    np.testing.assert_allclose(compiled_gamma, curves.gamma, atol=3.0e-15)

    for physical_gamma in np.asarray(curves.gamma):
        centered = physical_gamma - np.mean(physical_gamma, axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        assert singular_values[-1] < 2.0e-14

    direction = jnp.asarray([0.3, -0.2, 0.4, -0.1, 0.5, -0.6])

    def planarity_constraint(values):
        local_curves = builder.rebuild_curves(values)
        local_centers = builder.centers_at(values)
        local_normals = builder.normals_at(values)
        return jnp.einsum(
            "nc,nsc->ns",
            local_normals,
            local_curves.gamma[: planar.n_base_curves] - local_centers[:, None, :],
        )

    primal, tangent = jax.jvp(planarity_constraint, (parameters,), (direction,))
    np.testing.assert_allclose(primal, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(tangent, 0.0, atol=4.0e-15)


def test_planar_orientation_and_field_derivatives_are_finite_and_accurate():
    planar, coils = _planar_coils(stellsym=False)
    builder = make_planar_coil_design_field_builder(
        coils,
        plane_frames=planar.rotation_matrices[..., :2],
        current_groups=(0,),
        shape_dofs=((0, 0, 1),),
        center_dofs=((1, 2),),
        orientation_dofs=((0, 0), (0, 2)),
    )
    zero = jnp.zeros(builder.parameter_shape)
    direction = jnp.asarray([0.2, -0.3, 0.4, 0.5, -0.6])
    frames, frame_tangent = jax.jvp(builder.frames_at, (zero,), (direction,))
    assert np.all(np.isfinite(np.asarray(frame_tangent)))
    gram = jnp.einsum("nca,ncb->nab", frames, frames)
    np.testing.assert_allclose(
        gram,
        np.broadcast_to(np.eye(2), np.shape(gram)),
        atol=2.0e-15,
    )

    points = jnp.asarray([[0.8, 0.1, 0.02], [1.0, -0.15, -0.05]])

    def evaluate(values):
        return builder(values).B(points)

    _value, tangent = jax.jvp(evaluate, (zero,), (direction,))
    step = 1.0e-5
    finite_difference = (
        evaluate(zero + step * direction) - evaluate(zero - step * direction)
    ) / (2 * step)
    assert np.all(np.isfinite(np.asarray(tangent)))
    np.testing.assert_allclose(tangent, finite_difference, rtol=3.0e-8, atol=3.0e-12)


def test_planar_builder_uses_native_xy_coefficient_indices():
    planar, coils = _planar_coils()
    coordinates = tuple(
        (0, local_axis, xy_index)
        for local_axis in range(2)
        for xy_index in range(planar.xy_dofs.shape[-1])
    )
    builder = make_planar_coil_design_field_builder(
        PlanarCoils(planar, coils.base_currents),
        shape_dofs=coordinates,
    )
    parameters = jnp.arange(1, len(coordinates) + 1, dtype=float) * 1.0e-3
    actual = builder.local_dofs_at(parameters)
    expected = np.asarray(builder.nominal_local_dofs).copy()
    for value, (base_coil, local_axis, xy_index) in zip(
        parameters, coordinates, strict=True
    ):
        expected[base_coil, local_axis, xy_index + 1] += float(value)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_planar_native_rebuild_respects_an_explicit_in_plane_frame():
    planar, coils = _planar_coils(stellsym=False)
    frames = np.asarray(planar.rotation_matrices[..., :2]).copy()
    rotated_frames = np.stack((frames[..., 1], -frames[..., 0]), axis=-1)
    builder = make_planar_coil_design_field_builder(
        PlanarCoils(planar, coils.base_currents),
        plane_frames=rotated_frames,
    )

    rebuilt = builder.rebuild_planar_coils(jnp.zeros(builder.parameter_shape))
    np.testing.assert_allclose(rebuilt.gamma, coils.gamma, atol=4.0e-15)
    np.testing.assert_allclose(rebuilt.frames, rotated_frames, atol=3.0e-15)


def test_planar_builder_rejects_invalid_frames_planarity_and_coordinates():
    planar, coils = _planar_coils()
    frames = np.asarray(planar.rotation_matrices[..., :2])
    nonorthonormal = frames.copy()
    nonorthonormal[0, :, 1] = nonorthonormal[0, :, 0]
    with pytest.raises(ValueError, match="orthonormal"):
        make_planar_coil_design_field_builder(coils, plane_frames=nonorthonormal)

    nonplanar_dofs = np.asarray(coils.dofs_curves).copy()
    nonplanar_dofs[0, :, 3] += np.asarray(planar.normals[0]) * 1.0e-3
    nonplanar = Coils(
        Curves(
            nonplanar_dofs,
            coils.n_segments,
            coils.nfp,
            coils.stellsym,
        ),
        coils.base_currents,
    )
    with pytest.raises(ValueError, match="not planar"):
        make_planar_coil_design_field_builder(nonplanar, plane_frames=frames)
    with pytest.raises(IndexError, match="outside"):
        make_planar_coil_design_field_builder(
            coils,
            plane_frames=frames,
            shape_dofs=((0, 0, 4),),
        )
    with pytest.raises(IndexError, match="outside"):
        make_planar_coil_design_field_builder(
            coils,
            plane_frames=frames,
            center_dofs=((0, 3),),
        )

    builder = make_planar_coil_design_field_builder(
        coils,
        plane_frames=frames,
        orientation_dofs=((0, 1),),
    )
    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        builder(jnp.zeros(2))
    with pytest.raises(TypeError, match="floating dtype"):
        builder(jnp.zeros(1, dtype=int))


@pytest.mark.parametrize(
    ("groups", "error"),
    [
        ((0, 0), ValueError),
        ((-1,), IndexError),
        ((3,), IndexError),
        ((True,), TypeError),
        ((1.5,), TypeError),
    ],
)
def test_builder_rejects_invalid_current_groups(groups, error):
    with pytest.raises(error):
        make_fractional_current_field_builder(_coils(), groups=groups)


def test_builder_rejects_invalid_shape_contracts_and_parameters():
    coils = _coils()
    with pytest.raises(ValueError, match="either"):
        make_coil_design_field_builder(
            coils,
            shape_dofs=((0, 2, 1),),
            shape_directions=np.zeros((1, *coils.dofs_curves.shape)),
        )
    with pytest.raises(IndexError, match="outside"):
        make_shape_field_builder(coils, shape_dofs=((3, 2, 1),))
    with pytest.raises(ValueError, match="unique"):
        make_shape_field_builder(coils, shape_dofs=((0, 2, 1), (0, 2, 1)))
    with pytest.raises(ValueError, match="shape_directions"):
        make_shape_field_builder(coils, shape_directions=np.zeros((2, 3, 5)))

    builder = make_coil_design_field_builder(
        coils,
        current_groups=(0, 2),
        shape_dofs=((0, 2, 1),),
    )
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        builder(jnp.zeros(2))
    with pytest.raises(TypeError, match="floating dtype"):
        builder(jnp.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="real-valued"):
        builder(jnp.zeros(3, dtype=complex))
