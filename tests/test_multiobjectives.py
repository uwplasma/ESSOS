import pytest
import numpy as np
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
        opt_config={"fourier_order": 2, "num_coils": 2},
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


def test_scaling_an_initialized_custom_loss_preserves_dependencies():
    loss = CustomLoss(lambda x: jnp.sum(x*x), 'x')
    loss.dependencies = {'x': jnp.array([2., 3.])}
    scaled = 3*loss
    np.testing.assert_allclose(scaled(scaled.starting_dofs), 39.)
    np.testing.assert_allclose(scaled.grad(scaled.starting_dofs), jnp.array([12., 18.]))


def test_first_jitted_call_does_not_cache_traced_starting_dofs():
    loss = CustomLoss(lambda x, y: jnp.sum(x*x)+jnp.sum(y), 'x', 'y')
    loss.dependencies = {'x': jnp.array([2., 3.]), 'y': jnp.array([4.])}
    np.testing.assert_allclose(loss(jnp.array([2., 3., 4.])), 17.)
    np.testing.assert_allclose(loss.starting_dofs + 0., jnp.array([2., 3., 4.]))


@pytest.mark.parametrize('first_access', ['call', 'starting_dofs'])
def test_composite_first_jit_access_preserves_dofs_and_dependency_gradients(first_access):
    import jax

    quadratic = CustomLoss(lambda x, y: jnp.sum(x*x) + jnp.sum(y), 'x', 'y')
    coupling = CustomLoss(lambda x: x[0]*x[1], 'x')
    total = quadratic + coupling
    total.dependencies = {'x': jnp.array([2., 3.]), 'y': jnp.array([4.]),
                          'unused': jnp.array([99.])}
    # Dictionary pytrees flatten their keys in sorted order.
    dofs = jnp.array([99., 2., 3., 4.])
    if first_access == 'call':
        np.testing.assert_allclose(total(dofs), 23.)
    else:
        np.testing.assert_allclose(jax.jit(lambda: total.starting_dofs)(), dofs)
    # The cached vector must remain usable outside the initial JIT trace.
    np.testing.assert_allclose(total.starting_dofs + 0., dofs)
    np.testing.assert_allclose(total(dofs), 23.)
    np.testing.assert_allclose(total.grad(dofs), jnp.array([0., 7., 8., 1.]))
    np.testing.assert_allclose(jax.grad(total)(dofs), total.grad(dofs))
