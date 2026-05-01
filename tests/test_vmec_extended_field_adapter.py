import numpy as np
import pytest
import jax.numpy as jnp

from essos.vmec_extender import VmecExtendedField, _wrap_field_method, _wrap_grad_method, build_vmec_extended_field


class LinearField:
    def B_xyz(self, points):
        return 2.0 * jnp.asarray(points)

    def gradB_xyz(self, points):
        pts = jnp.asarray(points)
        eye = 2.0 * jnp.eye(3, dtype=pts.dtype)
        if pts.ndim == 1:
            return eye
        return jnp.broadcast_to(eye, pts.shape[:-1] + (3, 3))


def test_vmec_extended_field_single_point_methods():
    field = VmecExtendedField(LinearField())
    point = jnp.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(field.B(point), jnp.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(field.B_covariant(point), field.B(point))
    np.testing.assert_allclose(field.B_contravariant(point), field.B(point))
    assert np.isclose(field.AbsB(point), np.sqrt(56.0))
    np.testing.assert_allclose(field.dB_by_dX(point), 2.0 * jnp.eye(3))
    np.testing.assert_allclose(field.curl_B(point), jnp.zeros(3))
    np.testing.assert_allclose(field.to_xyz(point), point)


def test_wrap_field_method_supports_aos_and_soa_batches():
    def single(point):
        return point + 1.0

    wrapped = _wrap_field_method(single)
    aos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    soa = aos.T

    np.testing.assert_allclose(wrapped(aos), aos + 1.0)
    np.testing.assert_allclose(wrapped(soa), soa + 1.0)


def test_wrap_grad_method_supports_aos_batches():
    def single(_point):
        return 3.0 * jnp.eye(3)

    wrapped = _wrap_grad_method(single)
    aos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = wrapped(aos)

    assert got.shape == (2, 3, 3)
    np.testing.assert_allclose(got[0], 3.0 * jnp.eye(3))


def test_builder_requires_vmec_state_static_for_now():
    with pytest.raises(ValueError, match="vmec_state and vmec_static are required"):
        build_vmec_extended_field()

    with pytest.raises(NotImplementedError, match="wout object construction"):
        build_vmec_extended_field(wout=object())
