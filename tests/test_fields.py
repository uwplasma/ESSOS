import pytest
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart, FilamentaryBiotSavart
import jax
import jax.numpy as jnp
from jax import random
import numpy as np


class MockCoils:
    def __init__(self):
        self.currents = jnp.array([1.0, 2.0, 3.0])
        self.gamma = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.gamma_dash = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.gamma_dashdash = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.dofs_curves = random.uniform(random.PRNGKey(0), (3, 3, 3))


def test_biot_savart_initialization():
    coils = MockCoils()
    biot_savart = BiotSavart(coils)
    assert biot_savart.coils == coils
    assert jnp.allclose(biot_savart.currents, coils.currents)
    assert jnp.allclose(biot_savart.gamma, coils.gamma)
    assert jnp.allclose(biot_savart.gamma_dash, coils.gamma_dash)


def _coils():
    curves = CreateEquallySpacedCurves(
        2,
        order=2,
        R=1.4,
        r=0.25,
        n_segments=32,
        nfp=2,
        stellsym=True,
    )
    return Coils(curves, jnp.asarray([1.0e5, -8.0e4]))


def test_filamentary_biot_savart_matches_legacy_cartesian_field():
    coils = _coils()
    legacy = BiotSavart(coils)
    field = FilamentaryBiotSavart.from_coils(coils)
    points = jnp.asarray(
        [
            [0.70, 0.10, -0.08],
            [0.85, -0.15, 0.03],
            [1.00, 0.05, 0.12],
        ]
    )

    expected = jax.vmap(legacy.B)(points)
    np.testing.assert_allclose(field.B(points), expected, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(field(points), expected, rtol=2.0e-15, atol=2.0e-15)


def test_filamentary_biot_savart_implements_the_cartesian_field_contract():
    coils = _coils()
    legacy = BiotSavart(coils)
    field = FilamentaryBiotSavart.from_coils(coils)
    point = jnp.asarray([0.82, 0.11, -0.06])

    for method_name in (
        "B_covariant",
        "B_contravariant",
        "AbsB",
        "dB_by_dX",
        "dAbsB_by_dX",
        "grad_B_covariant",
        "curl_B",
        "curl_b",
        "kappa",
    ):
        actual = getattr(field, method_name)(point)
        expected = getattr(legacy, method_name)(point)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-12)

    np.testing.assert_allclose(field.sqrtg(point), legacy.sqrtg(point))
    np.testing.assert_array_equal(field.to_xyz(point), point)
    compiled = jax.jit(lambda candidate, xyz: candidate.kappa(xyz))(field, point)
    np.testing.assert_allclose(compiled, field.kappa(point), rtol=2.0e-13)


def test_filamentary_biot_savart_runs_existing_fieldline_and_guidingcenter_rhs():
    from essos.dynamics import FieldLine, GuidingCenter

    class ParticleParameters:
        charge = 1.0
        mass = 1.0
        energy = 1.0

    class ZeroElectricField:
        def E_covariant(self, points):
            return jnp.zeros_like(points)

    field = FilamentaryBiotSavart.from_coils(_coils())
    point = jnp.asarray([0.82, 0.11, -0.06])
    fieldline_rhs = FieldLine(0.0, point, field)
    guidingcenter_rhs = GuidingCenter(
        0.0,
        jnp.concatenate((point, jnp.asarray([0.1]))),
        (field, ParticleParameters(), ZeroElectricField()),
    )

    assert fieldline_rhs.shape == (3,)
    assert guidingcenter_rhs.shape == (4,)
    assert np.all(np.isfinite(np.asarray(fieldline_rhs)))
    assert np.all(np.isfinite(np.asarray(guidingcenter_rhs)))


def test_filamentary_biot_savart_broadcast_cylindrical_roundtrip():
    field = FilamentaryBiotSavart.from_coils(_coils())
    r = jnp.asarray([[0.75], [0.95]])
    phi = jnp.asarray([0.05, 0.20, 0.35])
    z = jnp.asarray(0.04)
    br, bphi, bz = field.b_cyl(r, phi, z)
    rr, pp, zz = jnp.broadcast_arrays(r, phi, z)
    cos_phi = jnp.cos(pp)
    sin_phi = jnp.sin(pp)
    actual = jnp.stack(
        (
            cos_phi * br - sin_phi * bphi,
            sin_phi * br + cos_phi * bphi,
            bz,
        ),
        axis=-1,
    )
    xyz = jnp.stack((rr * cos_phi, rr * sin_phi, zz), axis=-1)

    assert br.shape == bphi.shape == bz.shape == (2, 3)
    np.testing.assert_allclose(actual, field.B(xyz), rtol=2.0e-15, atol=2.0e-15)


def test_filamentary_biot_savart_is_a_jittable_differentiable_pytree():
    field = FilamentaryBiotSavart.from_coils(_coils())
    leaves, definition = jax.tree_util.tree_flatten(field)
    rebuilt = jax.tree_util.tree_unflatten(definition, leaves)
    assert len(leaves) == 3
    np.testing.assert_array_equal(rebuilt.currents, field.currents)

    point = jnp.asarray([0.85, 0.10, 0.04])
    compiled = jax.jit(lambda item: item.B(point))(field)
    primal, tangent = jax.jvp(
        lambda scale: FilamentaryBiotSavart(
            field.gamma,
            field.gamma_dash,
            scale * field.currents,
        ).B(point),
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )
    np.testing.assert_allclose(compiled, primal, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(tangent, primal, rtol=2.0e-15, atol=2.0e-15)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.float32, 2.0e-6, 2.0e-7),
        (jnp.float64, 2.0e-14, 2.0e-14),
    ],
)
def test_filamentary_biot_savart_regularizes_a_point_on_the_filament(
    dtype,
    rtol,
    atol,
):
    gamma = jnp.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
        dtype=dtype,
    )
    gamma_dash = jnp.asarray(
        [[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]],
        dtype=dtype,
    )
    currents = jnp.asarray([2.0], dtype=dtype)
    field = FilamentaryBiotSavart(gamma, gamma_dash, currents)
    point = gamma[0, 0]

    value = field.B(point)
    compiled = jax.jit(field.B)(point)
    assert value.dtype == dtype
    assert compiled.dtype == dtype
    assert np.all(np.isfinite(np.asarray(value)))
    assert np.all(np.isfinite(np.asarray(compiled)))
    np.testing.assert_allclose(compiled, value, rtol=rtol, atol=atol)

    displacement = point - gamma[0, 1:]
    radius_squared = jnp.sum(displacement * displacement, axis=-1)
    remaining = (
        jnp.cross(gamma_dash[0, 1:], displacement)
        / radius_squared[:, None]
        / jnp.sqrt(radius_squared)[:, None]
    )
    expected = 1.0e-7 * currents[0] * jnp.sum(remaining, axis=0) / gamma.shape[1]
    np.testing.assert_allclose(value, expected, rtol=rtol, atol=atol)

    jacobian = jax.jit(jax.jacfwd(field.B))(point)
    _value, point_jvp = jax.jvp(
        field.B,
        (point,),
        (jnp.ones_like(point),),
    )
    _value, gamma_jvp = jax.jvp(
        lambda values: FilamentaryBiotSavart(
            values,
            field.gamma_dash,
            field.currents,
        ).B(point),
        (field.gamma,),
        (jnp.ones_like(field.gamma),),
    )
    _value, current_jvp = jax.jvp(
        lambda values: FilamentaryBiotSavart(
            field.gamma,
            field.gamma_dash,
            values,
        ).B(point),
        (field.currents,),
        (jnp.ones_like(field.currents),),
    )
    for transformed in (jacobian, point_jvp, gamma_jvp, current_jvp):
        assert np.all(np.isfinite(np.asarray(transformed)))


@pytest.mark.parametrize(
    ("radius", "current"),
    [
        (1.1e-15, 1.0),
        (2.0e-15, 1.0),
        (1.0e-14, 1.0),
        (1.0e-13, 1.0e5),
        (1.0e-12, 1.0e5),
    ],
)
def test_filamentary_biot_savart_float32_near_core_derivatives_are_finite(
    radius,
    current,
):
    dtype = jnp.float32
    gamma = jnp.zeros((1, 1, 3), dtype=dtype)
    gamma_dash = jnp.asarray([[[0.0, 1.0, 0.0]]], dtype=dtype)
    currents = jnp.asarray([current], dtype=dtype)
    field = FilamentaryBiotSavart(gamma, gamma_dash, currents)
    point = jnp.asarray([radius, 0.0, 0.0], dtype=dtype)
    point_direction = jnp.asarray([1.0, 0.0, 0.0], dtype=dtype)

    value = jax.jit(field.B)(point)
    _value, point_jvp = jax.jvp(
        field.B,
        (point,),
        (point_direction,),
    )
    jacobian_forward = jax.jacfwd(field.B)(point)
    jacobian_reverse = jax.jacrev(field.B)(point)
    scalar_gradient = jax.grad(lambda location: field.B(location)[2])(point)
    gamma_direction = jnp.zeros_like(gamma).at[0, 0, 0].set(1.0)
    _value, gamma_jvp = jax.jvp(
        lambda values: FilamentaryBiotSavart(
            values,
            field.gamma_dash,
            field.currents,
        ).B(point),
        (field.gamma,),
        (gamma_direction,),
    )
    _value, gamma_dash_jvp = jax.jvp(
        lambda values: FilamentaryBiotSavart(
            field.gamma,
            values,
            field.currents,
        ).B(point),
        (field.gamma_dash,),
        (field.gamma_dash,),
    )
    _value, current_jvp = jax.jvp(
        lambda values: FilamentaryBiotSavart(
            field.gamma,
            field.gamma_dash,
            values,
        ).B(point),
        (field.currents,),
        (field.currents,),
    )
    gamma_gradient = jax.grad(
        lambda values: FilamentaryBiotSavart(
            values,
            field.gamma_dash,
            field.currents,
        ).B(point)[2]
    )(field.gamma)
    gamma_dash_gradient = jax.grad(
        lambda values: FilamentaryBiotSavart(
            field.gamma,
            values,
            field.currents,
        ).B(point)[2]
    )(field.gamma_dash)
    current_gradient = jax.grad(
        lambda values: FilamentaryBiotSavart(
            field.gamma,
            field.gamma_dash,
            values,
        ).B(point)[2]
    )(field.currents)

    expected_value = jnp.asarray(
        [0.0, 0.0, -current * 1.0e-7 / radius**2],
        dtype=dtype,
    )
    expected_directional_derivative = jnp.asarray(
        [0.0, 0.0, current * 2.0e-7 / radius**3],
        dtype=dtype,
    )
    for transformed in (
        value,
        point_jvp,
        jacobian_forward,
        jacobian_reverse,
        scalar_gradient,
        gamma_jvp,
        gamma_dash_jvp,
        current_jvp,
        gamma_gradient,
        gamma_dash_gradient,
        current_gradient,
    ):
        assert np.all(np.isfinite(np.asarray(transformed)))
    np.testing.assert_allclose(value, expected_value, rtol=3.0e-6, atol=0.0)
    np.testing.assert_allclose(
        point_jvp,
        expected_directional_derivative,
        rtol=3.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jacobian_forward,
        jacobian_reverse,
        rtol=3.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        scalar_gradient,
        jacobian_forward[2],
        rtol=3.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        gamma_jvp,
        -point_jvp,
        rtol=3.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        gamma_dash_jvp,
        value,
        rtol=3.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        current_jvp,
        value,
        rtol=3.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jnp.vdot(gamma_gradient, gamma_direction),
        gamma_jvp[2],
        rtol=3.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jnp.vdot(gamma_dash_gradient, field.gamma_dash),
        gamma_dash_jvp[2],
        rtol=3.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jnp.vdot(current_gradient, field.currents),
        current_jvp[2],
        rtol=3.0e-6,
        atol=0.0,
    )


# def test_biot_savart_B():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B = biot_savart.B(points)
#     assert jnp.allclose(B, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_B_covariant():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B_covariant = biot_savart.B_covariant(points)
#     assert jnp.allclose(B_covariant, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_B_contravariant():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B_contravariant = biot_savart.B_contravariant(points)
#     assert jnp.allclose(B_contravariant, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_AbsB():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     AbsB = biot_savart.AbsB(points)
#     assert jnp.allclose(AbsB, 4.42495529e-06)

# def test_biot_savart_dB_by_dX():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     dB_by_dX = biot_savart.dB_by_dX(points)
#     assert jnp.allclose(dB_by_dX[0], jnp.array([6.80204469e-05, 2.29490027e-05, 7.88513155e-05]))

# def test_biot_savart_dAbsB_by_dX():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     dAbsB_by_dX = biot_savart.dAbsB_by_dX(points)
#     assert jnp.allclose(dAbsB_by_dX, jnp.array([7.16688661e-05, 3.82872752e-05, 1.01490560e-04]))

if __name__ == "__main__":
    pytest.main()
