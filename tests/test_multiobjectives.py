import pytest
from unittest.mock import MagicMock, patch
from essos.losses import CustomLoss
from essos.multiobjectiveoptimizer import MultiObjectiveOptimizer
from essos.coils import Coils,Curves
from essos.fields import BiotSavart

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



def dummy_loss_fn(field=None, coils=None, vmec=None, surface=None, x=None):
    return jnp.sum(x)


def test_custom_loss_named_unraveler():
    def loss_fn(curve_dofs, current):
        return jnp.sum(curve_dofs**2) + jnp.sum(current)

    loss = CustomLoss(loss_fn, "curve_dofs", "current")
    loss.dependencies = {
        "curve_dofs": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "current": jnp.array([5.0]),
        "unused": jnp.array([99.0]),
    }

    dofs = loss.starting_dofs
    named_args = loss.dofs_to_pytree(dofs)
    tuple_args = tuple(named_args[name] for name in loss.args_names)

    assert set(named_args) == {"curve_dofs", "current"}
    assert jnp.array_equal(named_args["curve_dofs"], loss.dependencies["curve_dofs"])
    assert jnp.array_equal(named_args["current"], loss.dependencies["current"])
    assert loss(dofs) == loss_fn(named_args["curve_dofs"], named_args["current"])
    assert loss.call_pytree(named_args) == loss_fn(named_args["curve_dofs"], named_args["current"])
    assert loss.call_pytree(tuple_args) == loss_fn(named_args["curve_dofs"], named_args["current"])
    value, grad = loss.value_and_grad(dofs)
    assert value == loss_fn(named_args["curve_dofs"], named_args["current"]) and jnp.array_equal(grad, loss.grad(dofs))

    gradient = loss.grad_pytree(named_args)
    gradient_tuple = loss.grad_pytree(tuple_args)
    assert set(gradient) == {"curve_dofs", "current", "unused"}
    assert jnp.array_equal(gradient["curve_dofs"], 2 * named_args["curve_dofs"])
    assert jnp.array_equal(gradient["current"], jnp.ones_like(named_args["current"]))
    assert jnp.array_equal(gradient["unused"], jnp.zeros_like(loss.dependencies["unused"]))
    assert jnp.array_equal(gradient_tuple["curve_dofs"], gradient["curve_dofs"])
    assert jnp.array_equal(gradient_tuple["current"], gradient["current"])
    assert jnp.array_equal(gradient_tuple["unused"], gradient["unused"])


def test_weighted_loss_then_jitted_loss_keeps_static_currents_scale():
    # Regression: coils rebuilt inside the jitted weighted_loss used to recompute
    # currents_scale as a tracer, which then leaked into later jitted losses.
    from jax import jit
    import jax

    @jit
    def loss_coil_length_sum(coils):
        return jnp.sum(coils.length)

    optimizer = MultiObjectiveOptimizer(
        loss_functions=[loss_coil_length_sum],
        vmec=mock_vmec(),
        opt_config={"order_Fourier": 2, "num_coils": 2},
    )
    x = optimizer.initial_coils.dofs
    weighted = optimizer.weighted_loss(x, [1.0])
    jax.grad(lambda y: optimizer.weighted_loss(y, [1.0]))(x)
    direct = optimizer._call_loss_fn(loss_coil_length_sum, optimizer._build_available_inputs(x))
    assert jnp.isfinite(weighted)
    assert jnp.allclose(weighted, direct)


def test_plots_without_plotly_raise_clear_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(ImportError, match="essos\\[multiobjective\\]"):
        MultiObjectiveOptimizer.plot_param_importances(None)
