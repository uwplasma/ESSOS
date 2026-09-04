import pytest
from essos.fields import BiotSavart
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
    from essos.fields import CombinedField
    from essos.coils import Coils, CreateEquallySpacedCurves

    curves = CreateEquallySpacedCurves(n_curves=2, order=1, R=1.0, r=0.3,
                                        n_segments=20, nfp=2, stellsym=True)
    coils = Coils(curves=curves, currents=[1e5] * 2)
    field = BiotSavart(coils)
    points = jnp.array([0.5, 0.5, 0.5])

    combined_two = CombinedField(field, field)
    assert jnp.allclose(combined_two.B(points), 2 * field.B(points))

    combined_three = CombinedField(field, field, field)
    assert jnp.allclose(combined_three.B(points), 3 * field.B(points))


def test_combined_field_requires_at_least_one_field():
    from essos.fields import CombinedField
    with pytest.raises(ValueError):
        CombinedField()
