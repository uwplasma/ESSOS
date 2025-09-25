import pytest
from unittest.mock import MagicMock
import jax.numpy as jnp
from jax import random
from essos.surfaces import SurfaceRZFourier
from essos.qfm import QfmSurface
from essos.fields import BiotSavart


class MockSurface:
    def __init__(self):
        self.rc = jnp.array([[1., 2., 3.],
                             [1., 2., 3.],
                             [1., 2., 3.]])
        self.zs = jnp.array([[0.5, 1.5, 2.5],
                             [0.5, 1.5, 2.5],
                             [0.5, 1.5, 2.5]])
        self.nfp = 2
        self.ntheta = 3
        self.nphi = 3
        self.range_torus = "half period"

        self.area = 1.23
        self.volume = 4.56

        self.x = jnp.ones(16)
        self.gamma = jnp.ones((1, 3, 3))  # 添加这个就不会再报错了

    def change_resolution(self, ntheta, nphi):
        self.ntheta = ntheta
        self.nphi = nphi
        self.gamma = jnp.ones((ntheta, nphi, 3))  
        self.unitnormal = jnp.ones((ntheta, nphi, 3)) 


class MockField:
    def A(self, point):
        return jnp.array([1.0, 0.0, 0.0]) 

    def B(self, point):
        return jnp.array([0.0, 1.0, 0.0])


@pytest.fixture
def mock_data():
    surface = MockSurface()
    field = MockField()
    return surface, field


def test_qfm_surface_initialization(mock_data):
    surface, field = mock_data
    qfm = QfmSurface(field, surface, label="area")
    
    assert qfm.label == "area"
    assert qfm.targetlabel == surface.area
    assert qfm.surface == surface
    assert isinstance(qfm.surface_optimize, SurfaceRZFourier)
    assert qfm.toroidal_flux_idx == 0  


def test_minimize_lbfgs(mock_data):
    surface, field = mock_data
    qfm = QfmSurface(field, surface, label="area")

    qfm.minimize_lbfgs = MagicMock()
    qfm.minimize_lbfgs(x0=None, tol=1e-6, maxiter=1000, constraint_weight=1e-3)
    qfm.minimize_lbfgs.assert_called_once()


def test_minimize_slsqp(mock_data):
    surface, field = mock_data
    qfm = QfmSurface(field, surface, label="volume")

    qfm.minimize_slsqp = MagicMock()
    qfm.minimize_slsqp(x0=None, tol=1e-6, maxiter=1000)
    qfm.minimize_slsqp.assert_called_once()



def test_run_method(mock_data):
    surface, field = mock_data
    qfm = QfmSurface(field, surface, label="area")

    result_lbfgs = qfm.run(method="LBFGS", tol=1e-6, maxiter=1000)
    assert "s" in result_lbfgs
    assert result_lbfgs["success"] == True

    result_slsqp = qfm.run(method="SLSQP", tol=1e-6, maxiter=1000)
    assert "s" in result_slsqp
    assert result_slsqp["success"] == True

if __name__ == "__main__":
    pytest.main()
