from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from essos.coil_design import make_planar_coil_design_field_builder
from essos.coils import Coils
from essos.io import load_coils_json
from essos.optimization import optimize_planar_residual
from essos.planar_coils import PlanarCoils, PlanarXYCurves


def _planar_coils():
    curves = PlanarXYCurves(
        centers=jnp.asarray([[1.4, 0.0, 0.0]]),
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        xy_dofs=jnp.asarray([[[0.0, 0.28], [0.22, 0.0]]]),
        n_segments=40,
        stellsym=False,
    )
    return PlanarCoils(curves, jnp.asarray([1.1e5]))


def test_optimize_planar_residual_lowers_objective_and_returns_native_coils():
    builder = make_planar_coil_design_field_builder(
        _planar_coils(),
        current_groups=(0,),
        shape_dofs=((0, 0, 1),),
    )
    points = jnp.asarray([[1.4, 0.0, 0.16], [1.4, 0.0, 0.29], [1.4, 0.0, 0.42]])
    normals = jnp.asarray([[0.0, 0.0, 1.0]] * len(points))

    def normal_field(parameters):
        return jnp.sum(builder(parameters).B(points) * normals, axis=-1)

    target_parameters = jnp.asarray([0.10, 0.025])
    target = normal_field(target_parameters)
    scale = jnp.max(jnp.abs(target))

    def residual(parameters):
        return (normal_field(parameters) - target) / scale

    initial = jnp.zeros(builder.parameter_shape)
    initial_objective = 0.5 * np.sum(np.asarray(residual(initial)) ** 2)
    optimized, result = optimize_planar_residual(
        residual,
        builder,
        initial,
        tolerance_optimization=1.0e-12,
        maximum_function_evaluations=40,
    )
    final_objective = 0.5 * np.sum(np.asarray(residual(result.x)) ** 2)

    assert result.success
    assert isinstance(optimized, PlanarCoils)
    assert final_objective < initial_objective * 1.0e-8
    np.testing.assert_allclose(result.x, target_parameters, atol=2.0e-9)
    np.testing.assert_allclose(
        optimized.gamma,
        builder.rebuild_coils(result.x).gamma,
        atol=3.0e-15,
    )


def test_optimize_planar_residual_validates_inputs():
    builder = make_planar_coil_design_field_builder(_planar_coils())
    with pytest.raises(TypeError, match="PlanarCoilDesignFieldBuilder"):
        optimize_planar_residual(lambda values: values, object())
    with pytest.raises(ValueError, match="shape"):
        optimize_planar_residual(
            lambda values: values,
            builder,
            np.zeros((1,)),
        )
    with pytest.raises(ValueError, match="at least one"):
        optimize_planar_residual(
            lambda values: jnp.zeros((0,)),
            builder,
        )


def test_canonical_io_loads_planar_and_ordinary_coil_json(tmp_path):
    planar = _planar_coils()
    planar_path = tmp_path / "planar.json"
    planar.to_json(planar_path)
    loaded_planar = load_coils_json(planar_path)
    assert isinstance(loaded_planar, PlanarCoils)
    np.testing.assert_allclose(loaded_planar.gamma, planar.gamma)

    ordinary = planar.as_coils()
    ordinary_path = tmp_path / "ordinary.json"
    ordinary.to_json(ordinary_path)
    loaded_ordinary = load_coils_json(ordinary_path)
    assert isinstance(loaded_ordinary, Coils)
    np.testing.assert_allclose(loaded_ordinary.gamma, ordinary.gamma)
