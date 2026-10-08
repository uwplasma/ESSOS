import jax
import jax.numpy as jnp
import pytest
from essos.surfaces import (surfacerzfourier_from_boundary, B_on_surface,
                            BdotN, BdotN_over_B, SquaredFlux)


@jax.tree_util.register_pytree_node_class
class _StubField:
    """Field with B(x) = fn(x). fn is static (aux data); no array leaves."""
    def __init__(self, fn):
        self.fn = fn
    def B(self, x):
        return self.fn(x)
    def tree_flatten(self):
        return (), self.fn
    @classmethod
    def tree_unflatten(cls, fn, children):
        return cls(fn)


def _uniform_z():
    return _StubField(lambda x: jnp.array([0., 0., 1.]))

def _position():
    return _StubField(lambda x: x)


def _small_surface():
    rbc = jnp.zeros((5, 3)).at[2, 0].set(1.0).at[2, 1].set(0.2)
    zbs = jnp.zeros((5, 3)).at[2, 1].set(0.2)
    return surfacerzfourier_from_boundary(rbc, zbs, 2, nphi=8, ntheta=10)


def test_B_on_surface_shape():
    s = _small_surface()
    assert B_on_surface(s, _uniform_z()).shape == s.gamma.shape


def test_B_on_surface_preserves_point_ordering():
    s = _small_surface()
    assert jnp.allclose(B_on_surface(s, _position()), s.gamma)


def test_BdotN_uniform_field_is_normal_z_component():
    s = _small_surface()
    assert jnp.allclose(BdotN(s, _uniform_z()), s.unitnormal[..., 2])


def test_BdotN_over_B_is_bounded_by_one():
    s = _small_surface()
    ratio = BdotN_over_B(s, _position())
    assert ratio.shape == s.gamma.shape[:-1]
    assert jnp.all(jnp.abs(ratio) <= 1 + 1e-5)  # float32 unless x64 is enabled


@pytest.mark.parametrize("definition", ("local", "quadratic flux", "normalized"))
def test_SquaredFlux_finite_and_nonnegative(definition):
    value = SquaredFlux(_small_surface(), _position(), definition=definition)
    assert jnp.isfinite(value) and value >= 0


def test_SquaredFlux_rejects_unknown_definition():
    with pytest.raises(ValueError, match="Unknown definition"):
        SquaredFlux(_small_surface(), _uniform_z(), definition="bogus")
