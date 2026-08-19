import os
import pytest
import jax
from essos.fields import BiotSavart, Vmec, VMEC_WOUT_ARRAYS
import jax.numpy as jnp
from jax import random, vmap

WOUT_FILE = os.path.join(os.path.dirname(__file__), "..", "examples", "input_files",
                         "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")

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

def test_vmec_from_arrays_matches_wout_file():
    vmec = Vmec(WOUT_FILE)
    rebuilt = Vmec.from_arrays(nfp=vmec.nfp, ns=vmec.ns,
                               **{name: getattr(vmec, name) for name in VMEC_WOUT_ARRAYS})
    points = jnp.array([[0.3, 0.4, 0.5], [0.7, 1.2, 0.2], [0.9, 3.0, 1.1]])

    assert (rebuilt.nfp, rebuilt.ns, rebuilt.mpol, rebuilt.ntor) == (vmec.nfp, vmec.ns, vmec.mpol, vmec.ntor)
    assert jnp.array_equal(vmap(rebuilt.B)(points), vmap(vmec.B)(points))
    assert jnp.array_equal(vmap(rebuilt.AbsB)(points), vmap(vmec.AbsB)(points))
    assert jnp.array_equal(rebuilt.surface.gamma, vmec.surface.gamma)

def test_vmec_from_arrays_is_differentiable_in_the_coefficients():
    vmec = Vmec(WOUT_FILE)
    arrays = {name: getattr(vmec, name) for name in VMEC_WOUT_ARRAYS}
    point = jnp.array([0.7, 1.2, 0.2])

    def AbsB_of_scale(scale):
        return Vmec.from_arrays(nfp=vmec.nfp, ns=vmec.ns, **{**arrays, 'bmnc': arrays['bmnc']*scale}).AbsB(point)

    assert jnp.isclose(jax.grad(AbsB_of_scale)(1.0), vmec.AbsB(point))

if __name__ == "__main__":
    pytest.main()
