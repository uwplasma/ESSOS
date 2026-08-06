from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coil_design import make_planar_coil_design_field_builder
from essos.planar_coils import PlanarCoils, PlanarXYCurves


GPU_DEVICES = tuple(device for device in jax.devices() if device.platform == "gpu")


@pytest.mark.skipif(not GPU_DEVICES, reason="requires a JAX GPU device")
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_planar_combined_design_gradient_and_planarity_on_gpu(dtype):
    gpu = GPU_DEVICES[0]
    curves = PlanarXYCurves(
        centers=jnp.asarray([[1.4, 0.0, 0.0]], dtype=dtype),
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        xy_dofs=jnp.asarray([[[0.0, 0.28], [0.22, 0.0]]], dtype=dtype),
        n_segments=48,
        stellsym=False,
    )
    builder = make_planar_coil_design_field_builder(
        PlanarCoils(curves, jnp.asarray([1.1e5], dtype=dtype)),
        current_groups=(0,),
        shape_dofs=((0, 0, 1),),
        center_dofs=((0, 0),),
        orientation_dofs=((0, 1),),
    )
    parameters = jax.device_put(
        jnp.asarray([0.03, 0.005, 0.01, 0.04], dtype=dtype),
        gpu,
    )
    points = jax.device_put(
        jnp.asarray([[0.85, 0.10, 0.03], [1.0, -0.12, -0.04]], dtype=dtype),
        gpu,
    )

    def objective(values, evaluation_points):
        field = builder(values)
        return jnp.sum(field.B(evaluation_points) ** 2)

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    value, gradient = value_and_gradient(parameters, points)
    rebuilt = builder.rebuild_planar_coils(parameters)
    base_gamma = rebuilt.gamma[: rebuilt.planar_curves.n_base_curves]
    planarity = jnp.einsum(
        "nc,nsc->ns",
        rebuilt.normals,
        base_gamma - rebuilt.centers[:, None, :],
    )

    assert next(iter(value.devices())).platform == "gpu"
    assert next(iter(gradient.devices())).platform == "gpu"
    assert np.all(np.isfinite(np.asarray(value)))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 0.0
    tolerance = 2.0e-6 if dtype == jnp.float32 else 2.0e-14
    np.testing.assert_allclose(planarity, 0.0, atol=tolerance)

    cpu = next(device for device in jax.devices() if device.platform == "cpu")
    cpu_value, cpu_gradient = value_and_gradient(
        jax.device_put(parameters, cpu),
        jax.device_put(points, cpu),
    )
    parity_rtol = 2.0e-5 if dtype == jnp.float32 else 2.0e-12
    np.testing.assert_allclose(value, cpu_value, rtol=parity_rtol)
    np.testing.assert_allclose(gradient, cpu_gradient, rtol=parity_rtol)
