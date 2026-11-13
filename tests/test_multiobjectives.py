import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import essos.objective_functions as objf
from essos.losses import custom_loss
from essos.multiobjectiveoptimizer import MultiObjectiveOptimizer
from essos.coils import Coils,Curves
from essos.fields import BiotSavart
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
    x = optimizer.initial_coils.x
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


def _length(coils):
    return jnp.sum(coils.length)


def _curvature(coils):
    return jnp.max(coils.curvature)


def _axis_vmec():
    import types
    return types.SimpleNamespace(nfp=2, r_axis=1.0, surface=None)


def test_optax_descent_shrinks_coils_and_the_optuna_study_picks_a_pareto_point(tmp_path, monkeypatch):
    """Descending on the coil length alone must shorten the coils (a circle of radius r has length 2 pi r), and a
    two-objective optuna study returns non-dominated trials from which one is selected and rebuilt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    opt = MultiObjectiveOptimizer([_length, _curvature], _axis_vmec(),
                                  opt_config={"order_Fourier": 1, "num_coils": 2, "maximum_function_evaluations": 4,
                                              "n_trials": 3, "optimizer_choices": ["adam", "sgd"]})
    assert opt.loss_names == ["_length", "_curvature"] and opt.initial_coils.nfp == 2
    start = float(jnp.sum(opt.initial_coils.length))
    shorter = opt.optimize_with_optax([1.0, 0.0], method="adam", lr=1e-2)
    assert float(jnp.sum(shorter.length)) < start
    with pytest.raises(ValueError, match="missing parameter"):
        opt._call_loss_fn(lambda missing: missing, opt._build_available_inputs(opt.initial_coils.x))

    opt.run(seed=1)
    assert len(opt.study.trials) == 3 and opt.study.user_attrs["seed"] == 1
    for t in opt.study.trials:
        assert len(t.values) == 2 and np.all(np.isfinite(t.values))
    best = opt.select_best_from_pareto(weights=[1.0, 0.0])
    assert best in opt.study.best_trials
    scores = [t.values[0] for t in opt.study.best_trials]
    assert best.values[0] == min(scores)  # weight only on the first objective
    assert opt.select_best_from_pareto(limits=[-1.0, None]) is None
    with pytest.raises(ValueError, match="No valid trial"):
        opt.rebuild_best_coils(limits=[-1.0, None])
    rebuilt = opt.rebuild_best_coils(weights=[1.0, 1.0])
    assert isinstance(rebuilt, Coils) and rebuilt is opt.best_coils
    opt.plot_pareto_fronts(save=str(tmp_path / "pareto.png"))
    plt.close("all")
