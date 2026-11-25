"""Coil designs and coil constraint rows (essos.coil_design) on the bundled Landreman-Paul QA coils."""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coil_design import CoilDesign, CoilLimits, coil_constraint
from essos.coils import Coils
from essos.fields import BiotSavart

DATA = Path(__file__).resolve().parents[1] / "examples" / "input_files"
LIMITS = CoilLimits(length=5.0, curvature=5.0, mean_squared_curvature=5.0, coil_distance=0.15,
                    curvature_points=64, distance_points=32)


@pytest.fixture(scope="module")
def coils():
    return Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))


def test_nominal_design_is_the_coils_and_their_field(coils):
    design = CoilDesign(coils, current_dofs=(0, 1))
    np.testing.assert_allclose(design.coils_from_x(design.x0).gamma, coils.gamma, atol=1e-14)
    points = jnp.asarray([[1.0, 0.0, 0.0], [1.1, 0.2, 0.05]])
    np.testing.assert_allclose(jax.vmap(design(design.x0).B)(points), jax.vmap(BiotSavart(coils).B)(points),
                               rtol=1e-13)
    x = design.x0.copy()
    x[1] = 0.1  # the second current coordinate: base coil 1 up 10%
    np.testing.assert_allclose(design.base_currents_at(x)[1], 1.1 * coils.dofs_currents_raw[1], rtol=1e-14)
    assert len(design.dof_names) == design.size
    # the field of a traced design passes into a jitted function (its pytree metadata stays hashable)
    inner = jax.jit(lambda field: field.B(points[0])[0])
    np.testing.assert_allclose(jax.jit(lambda x: inner(design(x)))(x), inner(design(x)), rtol=1e-13)


def test_extreme_rows_are_the_minima_of_the_point_rows(coils):
    extreme, point = np.asarray(LIMITS.rows(coils)), np.asarray(LIMITS.rows(coils, per_point=True))
    n = coils.dofs_curves.shape[0]
    curvature_points, distance_points = LIMITS.curvature_points, LIMITS.distance_points
    assert extreme.size == 4 * n + 2
    assert point.size == 3 * n + n * curvature_points + 2 * n * distance_points
    np.testing.assert_allclose(point[:n], extreme[:n])                                   # length
    np.testing.assert_allclose(point[n:n + n * curvature_points].reshape(n, -1).min(axis=1), extreme[n:2 * n])
    rest = point[n + n * curvature_points:]
    np.testing.assert_allclose(rest[:n], extreme[2 * n:3 * n])                         # mean squared curvature
    np.testing.assert_allclose(rest[n:n + n * distance_points].min(), extreme[3 * n], rtol=1e-12)
    np.testing.assert_allclose(rest[n + n * distance_points:-n].min(), extreme[3 * n + 1], rtol=1e-12)
    np.testing.assert_allclose(rest[-n:], extreme[-n:])                                # minimum speed


def test_coil_constraint_jacobians_match_differences(coils):
    design = CoilDesign(coils, current_dofs=(0,))
    rows = lambda x: LIMITS.rows(design.coils_from_x(x))  # noqa: E731
    reverse = coil_constraint(rows)
    forward = coil_constraint(rows, forward_chunk=16)
    rng = np.random.default_rng(0)
    x = 0.1 * design.scales * rng.standard_normal(design.size)
    J = reverse.jac(x)
    np.testing.assert_allclose(forward.jac(x), J, rtol=1e-10, atol=1e-12)
    v = design.scales * rng.standard_normal(design.size)
    h = 1e-6
    fd = (reverse.fun(x + h * v) - reverse.fun(x - h * v)) / (2 * h)
    np.testing.assert_allclose(J @ v, fd, rtol=1e-5, atol=1e-8)
