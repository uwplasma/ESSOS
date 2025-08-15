import pytest
from unittest.mock import MagicMock, patch
from essos.multiobjectiveoptimizer import MultiObjectiveOptimizer
from essos.coils import Coils,Curves
from essos.fields import BiotSavart
from essos.objective_functions import loss_bdotn_over_b, loss_coil_length, loss_coil_curvature, loss_normB_axis

# test_multiobjectiveoptimizer.py

import jax.numpy as jnp




def surface():
    surface.nphi=3
    surface.ntheta=3
    surface.gamma = jnp.ones((3, 3, 3))
    surface.unitnormal = jnp.ones((3, 3, 3))    
    return surface

def mock_vmec():
    vmec = MagicMock()
    vmec.nfp = 2
    vmec.r_axis = 10.0
    vmec.surface = surface()
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

    optimized_coils=optimizer.optimize_with_optax(weights, method="adam", lr=1e-2)
    assert optimized_coils.currents_scale==0.01999998979999997872

    dofs_curves=optimized_coils.dofs_curves
    currents_scale=optimized_coils.currents_scale
    nfp=optimized_coils.nfp
    n_segments=optimized_coils.n_segments
    stellsym=optimized_coils.stellsym
    x=optimized_coils.x
    bdotn_b=loss_bdotn_over_b(x,vmec=vmec,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)
    #assert bdotn_b==0.0000000000000037761977058799732810080238

    max_length=loss_coil_length(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)
    max_curvature=loss_coil_curvature(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)
    normB_axis=loss_normB_axis(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)

    optimizer.run()

