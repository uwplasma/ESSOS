import pytest
from unittest.mock import MagicMock, patch
from essos.multiobjectiveoptimizer import MultiObjectiveOptimizer
from essos.coils import Coils,Curves

# test_multiobjectiveoptimizer.py

import jax.numpy as jnp


@pytest.fixture
def mock_vmec():
    vmec = MagicMock()
    vmec.nfp = 2
    vmec.r_axis = 10.0
    vmec.surface = MagicMock()
    return vmec



@pytest.fixture
def dummy_loss_fn():
    def loss_fn(field=None, coils=None, vmec=None, surface=None, x=None, **kwargs):
        return jnp.sum(x)
    return loss_fn

@patch("essos.coils.Curves")
@patch("essos.coils.Coils")
@patch("essos.fields.BiotSavart")
def test_build_available_inputs(mock_BiotSavart, mock_Coils, mock_Curves, mock_vmec,  dummy_loss_fn):
    optimizer = MultiObjectiveOptimizer(
        loss_functions=[dummy_loss_fn],
        vmec=mock_vmec,
        coils_init=None,
        function_inputs={"extra": 42},
        opt_config={"order_Fourier": 2, "num_coils": 2}
    )
    x = jnp.arange(20, dtype=float)
    mock_Curves.return_value = MagicMock()
    mock_Coils.return_value = MagicMock()
    mock_BiotSavart.return_value = MagicMock()

    result = optimizer._build_available_inputs(x)

    expected_keys = {
        "field", "coils", "vmec", "surface", "x", "dofs_curves", "currents_scale", "nfp", "extra"
    }
    assert expected_keys.issubset(result.keys())
    assert isinstance(result["x"], jnp.ndarray)
    assert result["vmec"] is mock_vmec
    assert result["surface"] is mock_vmec.surface
    assert result["currents_scale"] == 1.0
    assert result["nfp"] == 2
    assert result["extra"] == 42
    assert result["dofs_curves"].shape == (2, 3)