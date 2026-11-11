"""Small jitted forward and gradient checks, run on the GPU runner with `pytest -m gpu`.

They are cheap enough to also pass on CPU, where they run with the rest of the suite.
"""
import jax
import jax.numpy as jnp
import pytest

from essos.coils import Coils, Curves
from essos.dynamics import Tracing
from essos.fields import BiotSavart

pytestmark = pytest.mark.gpu

RADIUS, CURRENT, Z = 1.3, 2.5e5, 0.4


def _loop(current=CURRENT):
    dofs = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(RADIUS).at[0, 1, 1].set(RADIUS)
    return Coils(Curves(dofs, n_segments=64, nfp=1, stellsym=False), jnp.array([current]))


def _axis_bz(current=CURRENT):
    # mu_0 I R^2 / (2 (R^2 + z^2)^{3/2}) for an ideal loop
    return 2.0e-7 * jnp.pi * current * RADIUS**2 / (RADIUS**2 + Z**2) ** 1.5


def test_biot_savart_jit_matches_analytic_loop_on_axis():
    field = BiotSavart(_loop())
    B = jax.jit(field.B)(jnp.array([0.0, 0.0, Z]))
    assert jnp.allclose(B, jnp.array([0.0, 0.0, _axis_bz()]), rtol=1e-4, atol=1e-4 * _axis_bz())


def test_biot_savart_batched_points_are_finite():
    field = BiotSavart(_loop())
    points = jnp.stack([jnp.array([0.1 * i, 0.0, Z]) for i in range(8)])
    B = jax.jit(jax.vmap(field.B))(points)
    assert B.shape == (8, 3) and bool(jnp.all(jnp.isfinite(B)))


def test_biot_savart_jacobian_is_traceless():
    field = BiotSavart(_loop())
    J = jax.jit(jax.jacfwd(field.B))(jnp.array([0.2, 0.1, Z]))
    scale = jnp.max(jnp.abs(J))
    assert abs(float(jnp.trace(J))) < 1e-4 * float(scale)


def test_gradient_of_field_with_respect_to_current_is_linear_response():
    point = jnp.array([0.0, 0.0, Z])

    def bz(current):
        return BiotSavart(_loop(current)).B(point)[2]

    g = jax.jit(jax.grad(bz))(CURRENT)
    assert jnp.allclose(g, _axis_bz(1.0), rtol=1e-4)


def test_field_line_tracing_in_a_coil_field_is_differentiable():
    field = BiotSavart(_loop())

    def end_z(x0):
        tracing = Tracing(field=field, model="FieldLine", initial_conditions=x0,
                          maxtime=1e-2, timestep=1e-3, times_to_trace=3)
        return tracing.trajectories[0, -1, 2]

    x0 = jnp.array([[0.1, 0.0, Z]])
    assert bool(jnp.isfinite(end_z(x0)))
    assert bool(jnp.all(jnp.isfinite(jax.grad(end_z)(x0))))
