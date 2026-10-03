import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.rotational_transform import field_period_map, magnetic_axis, rotational_transform

JSON = os.path.join(os.path.dirname(__file__), '..', 'examples', 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils.from_json(JSON)
field = BiotSavart(coils)


def test_axis_is_fixed_point():
    axis = magnetic_axis(field, jnp.array([field.r_axis, 0.0]), coils.nfp)
    np.testing.assert_allclose(field_period_map(field, axis, coils.nfp), axis, atol=1e-10)


def test_iota_matches_equilibrium_and_converges():
    axis = magnetic_axis(field, jnp.array([field.r_axis, 0.0]), coils.nfp)
    R = axis[0] + jnp.array([0.02, 0.06])
    i64 = rotational_transform(field, R, 0.0, coils.nfp, axis, n_periods=64)
    i256 = rotational_transform(field, R, 0.0, coils.nfp, axis, n_periods=256)
    np.testing.assert_allclose(i64, i256, atol=2e-4)
    assert np.all((i256 > 0.41) & (i256 < 0.43))  # VMEC target 0.417-0.423


def test_gradient_wrt_coil_dofs_matches_finite_difference():
    R0 = 1.25

    def iota(dofs):
        return rotational_transform(BiotSavart(coils.with_dofs(dofs)), R0, 0.0, coils.nfp, n_periods=32, steps=16)

    dofs = coils.dofs
    g = jax.grad(iota)(dofs)
    v = jax.random.normal(jax.random.PRNGKey(0), dofs.shape)
    eps = 1e-6
    fd = (iota(dofs + eps * v) - iota(dofs - eps * v)) / (2 * eps)
    np.testing.assert_allclose(jnp.vdot(g, v), fd, rtol=1e-4)
