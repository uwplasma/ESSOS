import pytest
from unittest.mock import MagicMock
import jax.numpy as jnp
from jax import random
from essos.surfaces import SurfaceRZFourier
from essos.qfm import QfmSurface
from essos.fields import BiotSavart


class MockSurface:
    def __init__(self):
        self.nfp = 2
        self.ntheta = 3
        self.nphi = 3
        self.gamma = jnp.ones((self.ntheta, self.nphi, 3))  # Simulated 3D points
        self.unitnormal = jnp.ones((self.ntheta, self.nphi, 3))  # Simulated unit normal
        self.volume = 10.0
        self.area = 5.0
        self.x = jnp.ones(5)  # Example parameter array

    def change_resolution(self, ntheta, nphi):
        self.ntheta = ntheta
        self.nphi = nphi
        self.gamma = jnp.ones((ntheta, nphi, 3))  # Simulated 3D points
        self.unitnormal = jnp.ones((ntheta, nphi, 3))  # Simulated unit normal


class MockField:
    def A(self, point):
        return jnp.array([1.0, 0.0, 0.0])  # Mock field A function

    def B(self, point):
        return jnp.array([0.0, 1.0, 0.0])  # Mock field B function


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
    assert qfm.toroidal_flux_idx == 0  # Default value


def test_minimize_lbfgs(mock_data):
    surface, field = mock_data
    qfm = QfmSurface(field, surface, label="area")

    # Mock the optimizer
    qfm.minimize_lbfgs = MagicMock()
    qfm.minimize_lbfgs(x0=None, tol=1e-6, maxiter=1000, constraint_weight=1e3)

    # Call minimize_lbfgs and check if the function was called
    qfm.minimize_lbfgs(x0=None, tol=1e-6, maxiter=1000, constraint_weight=1e3)
    qfm.minimize_lbfgs.assert_called_once()


def test_minimize_slsqp(mock_data):
    surface, field = mock_data
    qfm = QfmSurface(field, surface, label="volume")

    # Mock the optimizer
    qfm.minimize_slsqp = MagicMock()
    qfm.minimize_slsqp(x0=None, tol=1e-6, maxiter=1000)

    # Call minimize_slsqp and check if the function was called
    qfm.minimize_slsqp(x0=None, tol=1e-6, maxiter=1000)
    qfm.minimize_slsqp.assert_called_once()


def test_callback(mock_data):
    surface, field = mock_data
    qfm = QfmSurface(field, surface, label="toroidal_flux")

    # Mock the callback
    info = {
        "objective": 1.0,
        "constraint": 0.5,
        "penalty": 0.2,
        "grad_norm": 0.1,
        "iter": 1,
    }

    # Test LBFGS callback
    qfm._callback(info)
    qfm._callback(info, printlog=False)


def test_run_method(mock_data):
    surface, field = mock_data
    qfm = QfmSurface(field, surface, label="area")

    # Mock the run method for LBFGS
    result_lbfgs = qfm.run(method="LBFGS", tol=1e-6, maxiter=1000)
    assert "s" in result_lbfgs
    assert result_lbfgs["success"] is True

    # Mock the run method for SLSQP
    result_slsqp = qfm.run(method="SLSQP", tol=1e-6, maxiter=1000)
    assert "s" in result_slsqp
    assert result_slsqp["success"] is True


if __name__ == "__main__":
    pytest.main()
