import jax
import jax.numpy as jnp
import pytest
from essos.surfaces import (surfacerzfourier_from_boundary, B_on_surface,
                            BdotN, BdotN_over_B, SquaredFlux, PointCloudSurface)


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


def _small_cloud():
    s = _small_surface()
    return s, PointCloudSurface(s.gamma.reshape(-1, 3), s.unitnormal.reshape(-1, 3),
                          s.area_element.reshape(-1))


def test_point_cloud_B_on_surface_shape_and_values():
    s, cloud = _small_cloud()
    out = B_on_surface(cloud, _position())
    assert out.shape == cloud.gamma.shape == (80, 3)
    assert jnp.allclose(out, cloud.gamma)


def test_point_cloud_BdotN_matches_grid_surface():
    s, cloud = _small_cloud()
    field = _position()
    result = BdotN(cloud, field)
    assert result.shape == (80,)
    assert jnp.allclose(result, BdotN(s, field).reshape(-1))


@pytest.mark.parametrize("definition", ("local", "quadratic flux", "normalized"))
def test_point_cloud_SquaredFlux_matches_grid_surface(definition):
    s, cloud = _small_cloud()
    field = _position()
    assert jnp.allclose(SquaredFlux(cloud, field, definition=definition),
                        SquaredFlux(s, field, definition=definition))


def test_point_cloud_default_weights_are_uniform():
    s, cloud = _small_cloud()
    uniform = PointCloudSurface(cloud.gamma, cloud.unitnormal)
    assert jnp.array_equal(uniform.area_element, jnp.ones(80))
    assert uniform.npoints == 80


def test_point_cloud_validates_shapes():
    with pytest.raises(ValueError, match="shape"):
        PointCloudSurface(jnp.zeros((5, 2)), jnp.zeros((5, 2)))
    with pytest.raises(ValueError, match="shape"):
        PointCloudSurface(jnp.zeros((5, 3)), jnp.zeros((4, 3)))
    with pytest.raises(ValueError, match="area_element"):
        PointCloudSurface(jnp.zeros((5, 3)), jnp.zeros((5, 3)), jnp.ones(4))


def test_point_cloud_from_csv_roundtrip(tmp_path):
    import numpy as np
    s, cloud = _small_cloud()
    np.savetxt(tmp_path / "p.csv", np.asarray(cloud.gamma), delimiter=",", header="x,y,z")
    np.savetxt(tmp_path / "n.csv", np.asarray(cloud.unitnormal), delimiter=",", header="nx,ny,nz")
    np.savetxt(tmp_path / "w.csv", np.asarray(cloud.area_element), delimiter=",", header="w")
    loaded = PointCloudSurface.from_csv(tmp_path / "p.csv", tmp_path / "n.csv", tmp_path / "w.csv")
    assert jnp.allclose(loaded.gamma, cloud.gamma)
    assert jnp.allclose(loaded.unitnormal, cloud.unitnormal)
    assert jnp.allclose(loaded.area_element, cloud.area_element)
    assert jnp.allclose(SquaredFlux(loaded, _position()), SquaredFlux(cloud, _position()))
    no_weights = PointCloudSurface.from_csv(tmp_path / "p.csv", tmp_path / "n.csv")
    assert jnp.array_equal(no_weights.area_element, jnp.ones(80))
