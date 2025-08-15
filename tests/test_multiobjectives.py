import pytest
from unittest.mock import MagicMock, patch
from essos.multiobjectiveoptimizer import MultiObjectiveOptimizer
from essos.coils import Coils,Curves
from essos.fields import BiotSavart

# test_multiobjectiveoptimizer.py

import jax.numpy as jnp


def mock_vmec():
    vmec = MagicMock()
    vmec.nfp = 2
    vmec.r_axis = 10.0
    vmec.surface = MagicMock()
    return vmec



def dummy_loss_fn():
    def loss_fn(field=None, coils=None, vmec=None, surface=None, x=None):
        return jnp.sum(x)
    return loss_fn


def test_build_available_inputs( vmec=mock_vmec(),  dummy_loss_fn=dummy_loss_fn()):
    optimizer = MultiObjectiveOptimizer(
        loss_functions=[dummy_loss_fn],
        vmec=vmec,
        coils_init=None,
        function_inputs={"extra": 42},
        opt_config={"order_Fourier": 2, "num_coils": 2}
    )
    x = jnp.arange(32, dtype=float)

    result = optimizer._build_available_inputs(x)


    expected_keys = {
        "field", "coils", "vmec", "surface", "x", "dofs_curves", "currents_scale", "nfp", "extra"
    }
    assert expected_keys.issubset(result.keys())
    assert isinstance(result["x"], jnp.ndarray)
    assert result["vmec"] is vmec
    assert result["surface"] is vmec.surface
    assert result["currents_scale"] == 1.0
    assert result["nfp"] == 2
    assert result["extra"] == 42
    assert result["dofs_curves"].shape == (2, 3,5)

    weights=jnp.array([1.0])
    loss_result=optimizer._call_loss_fn(dummy_loss_fn,result)
    assert loss_result.shape == ()
    assert loss_result == 496
    loss_weight_result=optimizer.weighted_loss( x, weights)
    assert loss_weight_result.shape == ()
    assert loss_weight_result == 496

    optimization_result=optimizer.optimize_with_optax(weights, method="adam", lr=1e-2)
    assert optimization_result.currents_scale==0.01999998979999997872
    
    optimizer.run()