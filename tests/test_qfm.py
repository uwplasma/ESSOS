import pytest
import jax.numpy as jnp
from jax import device_get
from essos.surfaces import SurfaceRZFourier, BdotN_over_B, toroidal_flux
from essos.fields import BiotSavart
from essos.qfm import QfmSurface
from essos.coils import Coils_from_json
from unittest.mock import MagicMock

# Mock function to simulate VMEC
def mock_vmec():
    vmec = MagicMock()
    vmec.nfp = 2
    vmec.r_axis = 10.0
    vmec.surface = surface()  # Assume surface function is defined elsewhere
    return vmec

# Mock surface for testing
def surface():
    surface = MagicMock()
    surface.nphi = 3
    surface.ntheta = 3
    surface.gamma = jnp.ones((3, 3, 3))
    surface.unitnormal = jnp.ones((3, 3, 3))
    surface.volume = 1000
    surface.area = 500
    return surface

# Mock field for testing
def mock_field():
    coils = Coils_from_json("input_files/stellarator_coils.json")
    return BiotSavart(coils)

# Test QfmSurface class
def test_qfm_surface():
    # Setup
    vmec = mock_vmec()
    field = mock_field()
    surface_instance = vmec.surface
    label = "toroidal_flux"
    targetlabel = toroidal_flux(surface_instance, field)
    qfm = QfmSurface(field=field, surface=surface_instance, label=label, targetlabel=targetlabel)

    # Check initialization
    assert qfm.label == label
    assert qfm.targetlabel == targetlabel
    assert qfm.field == field
    assert qfm.surface == surface_instance

    # Test the optimization run
    method = "slsqp"  # or 'lbfgs'
    result = qfm.run(
        tol=1e-6,
        maxiter=1000,
        method=method,
        constraint_weight=1e-3,
        log_every=10
    )

    # Check if the optimization was successful
    assert result["success"]
    assert "s" in result  # Check if optimized surface is returned

    # Check if final objective and constraint values are within expected range
    x_opt = device_get(result["s"].x)
    qfm_loss = float(jnp.asarray(qfm.objective(x_opt)))
    c_loss = float(jnp.asarray(qfm.constraint(x_opt)))
    assert qfm_loss < 1e-3  # Expected value for the objective
    assert abs(c_loss) < 1e-3  # Expected value for the constraint

# Run test
if __name__ == "__main__":
    pytest.main()
