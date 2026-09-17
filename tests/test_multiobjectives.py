import pytest
from unittest.mock import MagicMock, patch
import essos.objective_functions as objf
from essos.losses import custom_loss
from essos.multiobjectiveoptimizer import MultiObjectiveOptimizer
from essos.coils import Coils,Curves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier
from essos.objective_functions import loss_coil_length, loss_coil_curvature
from essos.surfaces import BdotN_over_B

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

    loss = custom_loss(loss_fn, "curve_dofs", "current")
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


def test_custom_loss_freeze_dofs_keeps_pytree_and_zeroes_gradient():
    loss = custom_loss(lambda x: jnp.sum(x**2), "x")
    loss.dependencies = {"x": jnp.array([1.0, 2.0, 3.0])}
    loss.freeze_dofs(jnp.array([False, True, False]))

    proposed = jnp.array([4.0, 5.0, 6.0])
    assert loss(proposed) == 16.0 + 4.0 + 36.0
    assert jnp.array_equal(loss.dofs_to_pytree(proposed)["x"], jnp.array([4.0, 2.0, 6.0]))
    assert jnp.array_equal(loss.grad(proposed), jnp.array([8.0, 0.0, 12.0]))


def test_composite_loss_freeze_dofs_keeps_dependency_pytree():
    first = custom_loss(lambda x: jnp.sum(x**2), "x")
    second = custom_loss(lambda y: jnp.sum(y**2), "y")
    loss = first + second
    loss.dependencies = {"x": jnp.array([1.0]), "y": jnp.array([2.0])}
    loss.freeze_dofs(jnp.array([True, False]))

    proposed = jnp.array([10.0, 3.0])
    assert jnp.array_equal(loss.dofs_to_pytree(proposed)["x"], jnp.array([1.0]))
    assert jnp.array_equal(loss.grad(proposed), jnp.array([0.0, 6.0]))


def test_freeze_current_selects_one_base_coil_current():
    curves = Curves(jnp.zeros((2, 3, 3)), n_segments=10, nfp=1, stellsym=False)
    field = BiotSavart(Coils(curves, currents=jnp.array([1.0, 2.0])))
    loss = custom_loss(lambda field: jnp.sum(field.coils.dofs_currents**2), "field")
    loss.dependencies = {"field": field}
    loss.freeze_current("field", coil=0)

    proposed = loss.starting_dofs + 1.0
    optimized = loss.dofs_to_pytree(proposed)["field"]
    assert optimized.coils.dofs_currents[0] == field.coils.dofs_currents[0]
    assert optimized.coils.dofs_currents[1] == field.coils.dofs_currents[1] + 1.0


def test_freeze_coil_and_curve_dofs_select_geometry_leaves():
    initial_dofs = jnp.arange(18.0).reshape(2, 3, 3)
    field = BiotSavart(Coils(Curves(initial_dofs, n_segments=10, nfp=1, stellsym=False), currents=jnp.ones(2)))
    loss = custom_loss(lambda field: jnp.sum(field.coils.dofs_curves**2), "field")
    loss.dependencies = {"field": field}
    loss.freeze_coil("field", coil=0)
    loss.freeze_curve_dofs("field", coil=1, coordinates="z", modes=1)

    optimized = loss.dofs_to_pytree(loss.starting_dofs + 1.0)["field"]
    assert jnp.array_equal(optimized.coils.dofs_curves[0], field.coils.dofs_curves[0])
    assert optimized.coils.dofs_curves[1, 2, 1] == field.coils.dofs_curves[1, 2, 1]
    assert optimized.coils.dofs_curves[1, 0, 0] != field.coils.dofs_curves[1, 0, 0]


def test_freeze_surface_modes_selects_rc_and_zs_coefficients():
    surface = SurfaceRZFourier(
        rc=jnp.array([10.0, 1.0]), zs=jnp.array([0.0, 2.0]),
        nfp=1, mpol=1, ntor=0,
    )
    loss = custom_loss(lambda surface: jnp.sum(surface.rc**2) + jnp.sum(surface.zs**2), "surface")
    loss.dependencies = {"surface": surface}
    loss.freeze_surface_modes("surface", rc=[(1, 0)], zs=[(1, 0)])

    proposed = loss.starting_dofs + 1.0
    optimized = loss.dofs_to_pytree(proposed)["surface"]
    assert optimized.rc[1] == surface.rc[1]
    assert optimized.zs[1] == surface.zs[1]


@pytest.mark.xfail(reason='test_build_available_inputs uses the old optimizer loss API (x, dofs_curves=, currents_scale=); BdotN_over_B now lives in essos.surfaces with signature (surface, field). Needs rewrite to new API.', strict=False)
def test_build_available_inputs( vmec=mock_vmec(),  dummy_loss_fn=dummy_loss_fn):
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
    bdotn_b=objf.loss_bdotn_over_b(x,vmec=vmec,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)
    #assert bdotn_b==0.0000000000000037761977058799732810080238

    max_length=loss_coil_length(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)
    max_curvature=loss_coil_curvature(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)
    normB_axis=objf.loss_normB_axis(x,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)

    optimizer.run()
