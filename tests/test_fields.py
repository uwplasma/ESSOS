import pytest
from essos.coils import Coils, Curves
from essos.fields import BiotSavart
import jax
import jax.numpy as jnp
from jax import random

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
    assert jnp.allclose(biot_savart.coils.currents, coils.currents)
    assert jnp.allclose(biot_savart.coils.gamma, coils.gamma)
    assert jnp.allclose(biot_savart.coils.gamma_dash, coils.gamma_dash)


def test_biot_savart_cylindrical_interface_matches_cartesian_and_differentiates():
    dofs = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(1.0).at[0, 1, 1].set(1.0)
    field = BiotSavart(Coils(Curves(dofs, n_segments=32, stellsym=False),
                              jnp.array([1.0e5])))
    R = jnp.array([[0.7, 0.8], [0.9, 1.0]])
    phi = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    Z = jnp.array([[-0.2, -0.1], [0.1, 0.2]])
    br, bp, bz = field.b_cyl(R, phi, Z)
    xyz = jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1)
    B = jax.vmap(field.B)(xyz.reshape((-1, 3))).reshape(xyz.shape)
    expected = jnp.stack((B[..., 0] * jnp.cos(phi) + B[..., 1] * jnp.sin(phi),
                          -B[..., 0] * jnp.sin(phi) + B[..., 1] * jnp.cos(phi),
                          B[..., 2]))
    assert jnp.allclose(jnp.stack((br, bp, bz)), expected)
    derivative = jax.grad(lambda radius: jnp.sum(field.b_cyl(radius, phi, Z)[0]))(R)
    assert jnp.all(jnp.isfinite(derivative))

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


def test_combined_field_sums_correctly():
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import CombinedField

    curves = CreateEquallySpacedCurves(n_curves=2, order=1, R=1.0, r=0.3,
                                       n_segments=20, nfp=2, stellsym=True)
    field = BiotSavart(Coils(curves=curves, currents=[1e5] * 2))
    points = jnp.array([0.5, 0.5, 0.5])

    assert jnp.allclose(CombinedField(field, field).B(points), 2 * field.B(points))
    assert jnp.allclose(CombinedField(field, field, field).B(points), 3 * field.B(points))


def test_combined_field_requires_at_least_one_field():
    from essos.fields import CombinedField

    with pytest.raises(ValueError):
        CombinedField()


def _circular_loop(radius=1.3, current=2.5e5, n_segments=64):
    """One counterclockwise circular filament in the xy plane."""
    dofs = jnp.zeros((1, 3, 3))
    dofs = dofs.at[0, 0, 2].set(radius)  # x = R cos(theta)
    dofs = dofs.at[0, 1, 1].set(radius)  # y = R sin(theta)
    return Coils(
        Curves(dofs, n_segments=n_segments, nfp=1, stellsym=False),
        jnp.array([current]),
    )


def _axis_field_derivatives(radius, current, z):
    """Bz and its first three z derivatives for an ideal circular loop."""
    # mu_0 / 2 in SI units; this is independent of ESSOS's quadrature kernel.
    coefficient = 2.0 * jnp.pi * 1.0e-7 * current * radius**2
    q = radius**2 + z**2
    field = coefficient * q**(-1.5)
    first = -3.0 * coefficient * z * q**(-2.5)
    second = -3.0 * coefficient * (radius**2 - 4.0 * z**2) * q**(-3.5)
    third = 15.0 * coefficient * z * (3.0 * radius**2 - 4.0 * z**2) * q**(-4.5)
    return field, first, second, third


def _axis_cartesian_tensors(radius, current, z):
    """Full Cartesian B derivative tensors through third order on the axis.

    The entries follow from the source-free axisymmetric expansion

      Bz = F - (x**2 + y**2) F'' / 4 + ...
      (Bx, By) = -(x, y) F' / 2
                 + (x, y) (x**2 + y**2) F''' / 16 + ... .
    """
    field, first, second, third = _axis_field_derivatives(radius, current, z)
    tensors = [jnp.array([0.0, 0.0, field])]

    d1 = jnp.zeros((3, 3))
    d1 = d1.at[0, 0].set(-first / 2).at[1, 1].set(-first / 2)
    d1 = d1.at[2, 2].set(first)
    tensors.append(d1)

    d2 = jnp.zeros((3, 3, 3))
    for output, radial in ((0, 0), (1, 1)):
        d2 = d2.at[output, radial, 2].set(-second / 2)
        d2 = d2.at[output, 2, radial].set(-second / 2)
    d2 = d2.at[2, 0, 0].set(-second / 2)
    d2 = d2.at[2, 1, 1].set(-second / 2)
    d2 = d2.at[2, 2, 2].set(second)
    tensors.append(d2)

    d3 = jnp.zeros((3, 3, 3, 3))
    for output, radial, transverse in ((0, 0, 1), (1, 1, 0)):
        for indices in ((radial, 2, 2), (2, radial, 2), (2, 2, radial)):
            d3 = d3.at[(output,) + indices].set(-third / 2)
        d3 = d3.at[output, radial, radial, radial].set(3 * third / 8)
        for indices in ((radial, transverse, transverse),
                        (transverse, radial, transverse),
                        (transverse, transverse, radial)):
            d3 = d3.at[(output,) + indices].set(third / 8)
    d3 = d3.at[2, 2, 2, 2].set(third)
    for radial in (0, 1):
        for indices in ((radial, radial, 2), (radial, 2, radial),
                        (2, radial, radial)):
            d3 = d3.at[(2,) + indices].set(-third / 2)
    tensors.append(d3)
    return tensors


def test_biot_savart_circular_loop_axis_derivative_tensors():
    radius, current, z = 1.3, 2.5e5, 0.4
    field = BiotSavart(_circular_loop(radius, current))
    point = jnp.array([0.0, 0.0, z])

    actual = [field.B(point)]
    derivative = field.B
    for _ in range(3):
        derivative = jax.jacfwd(derivative)
        actual.append(derivative(point))

    expected = _axis_cartesian_tensors(radius, current, z)
    for order, (result, reference) in enumerate(zip(actual, expected)):
        assert result.shape == (3,) * (order + 1)
        atol = 2e-12 * jnp.max(jnp.abs(reference))
        assert jnp.allclose(result, reference, rtol=2e-11, atol=atol), order


def test_biot_savart_circular_loop_current_and_radius_sensitivities():
    radius, current, z = 1.3, 2.5e5, 0.4
    coils = _circular_loop(radius, current)
    point = jnp.array([0.0, 0.0, z])

    def field_from_dofs(dofs):
        return BiotSavart(coils.with_dofs(dofs)).B(point)

    # The public current DOF is normalized; this tangent represents +1 ampere.
    current_index = coils.dof_names.index("coil[0].current")
    current_direction = jnp.zeros_like(coils.dofs).at[current_index].set(
        1.0 / coils.currents_scale)
    radius_direction = jnp.zeros_like(coils.dofs)
    for name in ("coil[0].xc(1)", "coil[0].ys(1)"):
        radius_direction = radius_direction.at[coils.dof_names.index(name)].set(1.0)
    _, current_tangent = jax.jvp(field_from_dofs, (coils.dofs,), (current_direction,))
    _, radius_tangent = jax.jvp(field_from_dofs, (coils.dofs,), (radius_direction,))

    axis_field = _axis_field_derivatives(radius, current, z)[0]
    expected_current = jnp.array([0.0, 0.0, axis_field / current])
    q = radius**2 + z**2
    expected_radius_z = (2.0 * jnp.pi * 1.0e-7 * current * radius
                         * (2.0 * z**2 - radius**2) * q**(-2.5))
    expected_radius = jnp.array([0.0, 0.0, expected_radius_z])
    current_atol = 2e-12 * jnp.max(jnp.abs(expected_current))
    radius_atol = 2e-12 * jnp.max(jnp.abs(expected_radius))
    assert jnp.allclose(
        current_tangent, expected_current, rtol=2e-11, atol=current_atol)
    assert jnp.allclose(
        radius_tangent, expected_radius, rtol=2e-11, atol=radius_atol)
