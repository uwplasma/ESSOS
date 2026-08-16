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
