import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coils import Coils
from essos.fields import FilamentaryBiotSavart
from essos.planar_coils import (
    PlanarCoils,
    PlanarXYCurves,
    load_coils_json,
    load_simsopt_coils_json,
)


def _circle_xy_dofs(radius):
    # Local ordering is [sin(1), cos(1)]: X=r*cos(theta), Y=r*sin(theta).
    return jnp.asarray([[[0.0, radius], [radius, 0.0]]])


def test_as_xyz_dofs_uses_center_as_the_only_dc_term():
    centers = jnp.asarray([[1.0, 2.0, 3.0]])
    quaternions = jnp.asarray([[2.0, 0.0, 0.0, 0.0]])
    xy_dofs = jnp.asarray([[[0.1, 2.0, -0.2, 0.3], [1.5, 0.4, 0.6, -0.7]]])
    planar = PlanarXYCurves(
        centers, quaternions, xy_dofs, n_segments=12, stellsym=False
    )

    expected = np.zeros((1, 3, 5))
    expected[:, :, 0] = centers
    expected[:, :2, 1:] = xy_dofs
    np.testing.assert_allclose(planar.quaternions, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(planar.as_xyz_dofs(), expected)
    np.testing.assert_allclose(planar.as_curves().dofs, expected)
    assert planar.order == 2
    assert planar.n_base_curves == 1


def test_quaternion_places_the_local_xy_curve_in_an_arbitrary_plane():
    center = jnp.asarray([[1.25, -0.5, 0.75]])
    half_angle = jnp.pi / 4
    # +pi/2 about global Y maps the local +Z plane normal to global +X.
    quaternion = jnp.asarray([[jnp.cos(half_angle), 0.0, jnp.sin(half_angle), 0.0]])
    planar = PlanarXYCurves(
        center,
        quaternion,
        _circle_xy_dofs(0.4),
        n_segments=24,
        stellsym=False,
    )

    np.testing.assert_allclose(planar.normals, [[1.0, 0.0, 0.0]], atol=2.0e-15)
    np.testing.assert_allclose(planar.gamma[0, :, 0], center[0, 0], atol=2.0e-15)


def test_zero_quaternion_falls_back_and_tiny_finite_quaternion_normalizes():
    centers = jnp.asarray([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
    quaternions = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0e-300, -2.0e-300, 3.0e-300, -4.0e-300],
        ]
    )
    xy_dofs = jnp.broadcast_to(_circle_xy_dofs(0.5), (2, 2, 2))
    planar = PlanarXYCurves(
        centers, quaternions, xy_dofs, n_segments=12, stellsym=False
    )

    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            np.asarray([1.0, -2.0, 3.0, -4.0]) / np.sqrt(30.0),
        ]
    )
    np.testing.assert_allclose(planar.quaternions, expected, rtol=2.0e-15)
    assert np.all(np.isfinite(np.asarray(planar.as_xyz_dofs())))
    assert np.all(np.isfinite(np.asarray(planar.normals)))


def test_geometry_properties_delegate_to_curves_with_essos_symmetries():
    radius = 0.5
    planar = PlanarXYCurves(
        centers=jnp.asarray([[2.0, 0.0, 0.0]]),
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        xy_dofs=_circle_xy_dofs(radius),
        n_segments=32,
        nfp=2,
        stellsym=True,
    )

    assert planar.gamma.shape == (4, 32, 3)
    assert planar.tangent.shape == (4, 32, 3)
    np.testing.assert_array_equal(planar.gamma_dash, planar.tangent)
    np.testing.assert_allclose(planar.curvature, 1 / radius, atol=2.0e-14)
    np.testing.assert_allclose(planar.length, 2 * np.pi * radius, atol=2.0e-14)


def test_as_xyz_dofs_and_gamma_are_jittable_and_differentiable():
    planar = PlanarXYCurves(
        centers=jnp.asarray([[1.0, 0.0, 0.0]]),
        quaternions=jnp.asarray([[1.0, 0.2, -0.1, 0.3]]),
        xy_dofs=_circle_xy_dofs(0.3),
        n_segments=16,
        stellsym=False,
    )
    eager = planar.as_xyz_dofs()
    compiled = jax.jit(lambda curves: curves.as_xyz_dofs())(planar)
    np.testing.assert_allclose(compiled, eager, rtol=2.0e-15, atol=2.0e-15)

    def sampled_gamma(centers, xy_dofs):
        return PlanarXYCurves(
            centers,
            planar.quaternions,
            xy_dofs,
            n_segments=planar.n_segments,
            stellsym=False,
        ).gamma

    direction_centers = jnp.asarray([[0.4, -0.2, 0.1]])
    direction_xy = jnp.zeros_like(planar.xy_dofs).at[0, 0, 1].set(0.7)
    primal, tangent = jax.jit(
        lambda centers, xy_dofs, dcenters, dxy: jax.jvp(
            sampled_gamma, (centers, xy_dofs), (dcenters, dxy)
        )
    )(
        planar.centers,
        planar.xy_dofs,
        direction_centers,
        direction_xy,
    )
    assert primal.shape == (1, 16, 3)
    assert tangent.shape == primal.shape
    assert np.all(np.isfinite(np.asarray(tangent)))
    assert np.linalg.norm(np.asarray(tangent)) > 0.0


def test_quaternion_pytree_jvp_preserves_scale_invariance():
    planar = PlanarXYCurves(
        centers=jnp.asarray([[1.0, -0.2, 0.1]]),
        quaternions=jnp.asarray([[1.0, 0.2, -0.1, 0.3]]),
        xy_dofs=_circle_xy_dofs(0.3),
        n_segments=16,
        stellsym=False,
    )
    leaves, definition = jax.tree_util.tree_flatten(planar)
    radial_direction = jax.tree_util.tree_unflatten(
        definition,
        (
            jnp.zeros_like(leaves[0]),
            leaves[1],
            jnp.zeros_like(leaves[2]),
        ),
    )

    _gamma, gamma_tangent = jax.jvp(
        lambda curves: curves.gamma,
        (planar,),
        (radial_direction,),
    )
    point = jnp.asarray([0.75, 0.08, -0.04])
    _field, field_tangent = jax.jvp(
        lambda curves: PlanarCoils(curves, jnp.asarray([1.2e5])).to_field().B(point),
        (planar,),
        (radial_direction,),
    )

    np.testing.assert_allclose(gamma_tangent, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(field_tangent, 0.0, atol=2.0e-14)


def test_quaternion_pytree_jvp_matches_normalized_finite_difference():
    planar = PlanarXYCurves(
        centers=jnp.asarray([[1.0, -0.2, 0.1]]),
        quaternions=jnp.asarray([[1.0, 0.2, -0.1, 0.3]]),
        xy_dofs=_circle_xy_dofs(0.3),
        n_segments=16,
        stellsym=False,
    )
    quaternion_direction = jnp.asarray([[0.2, -0.3, 0.4, 0.1]])
    leaves, definition = jax.tree_util.tree_flatten(planar)
    pytree_direction = jax.tree_util.tree_unflatten(
        definition,
        (
            jnp.zeros_like(leaves[0]),
            quaternion_direction,
            jnp.zeros_like(leaves[2]),
        ),
    )
    _gamma, tangent = jax.jvp(
        lambda curves: curves.gamma,
        (planar,),
        (pytree_direction,),
    )

    def rebuilt_gamma(step):
        return PlanarXYCurves(
            planar.centers,
            planar.quaternions + step * quaternion_direction,
            planar.xy_dofs,
            n_segments=planar.n_segments,
            nfp=planar.nfp,
            stellsym=planar.stellsym,
        ).gamma

    step = 1.0e-6
    finite_difference = (rebuilt_gamma(step) - rebuilt_gamma(-step)) / (2 * step)
    np.testing.assert_allclose(
        tangent,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_from_polar_radius_is_an_exact_xy_fourier_conversion():
    polar_center = jnp.asarray([[1.0, -1.0, 0.5]])
    radius_dofs = jnp.asarray([[2.0, 0.4, 0.6]])
    planar = PlanarXYCurves.from_polar_radius(
        centers=polar_center,
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        radius_dofs=radius_dofs,
        n_segments=40,
        stellsym=False,
    )

    assert planar.order == 2
    np.testing.assert_allclose(planar.centers, [[1.3, -0.8, 0.5]], atol=2.0e-15)
    expected_xy_dofs = np.asarray([[[0.0, 2.0, 0.2, 0.3], [2.0, 0.0, 0.3, -0.2]]])
    np.testing.assert_allclose(planar.xy_dofs, expected_xy_dofs, atol=2.0e-15)

    theta = 2 * np.pi * np.arange(planar.n_segments) / planar.n_segments
    radius = 2.0 + 0.4 * np.sin(theta) + 0.6 * np.cos(theta)
    expected_gamma = np.stack(
        (
            polar_center[0, 0] + radius * np.cos(theta),
            polar_center[0, 1] + radius * np.sin(theta),
            np.full_like(theta, polar_center[0, 2]),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(planar.gamma[0], expected_gamma, atol=5.0e-15)


@pytest.mark.parametrize(
    ("centers", "quaternions", "xy_dofs", "error"),
    [
        (np.zeros((1, 2)), np.zeros((1, 4)), np.zeros((1, 2, 2)), ValueError),
        (np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 2, 2)), ValueError),
        (np.zeros((1, 3)), np.zeros((1, 4)), np.zeros((1, 2, 3)), ValueError),
        (np.zeros((2, 3)), np.zeros((1, 4)), np.zeros((1, 2, 2)), ValueError),
    ],
)
def test_invalid_component_shapes_are_rejected(centers, quaternions, xy_dofs, error):
    with pytest.raises(error):
        PlanarXYCurves(centers, quaternions, xy_dofs)


def test_invalid_static_metadata_is_rejected():
    args = (
        np.zeros((1, 3)),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        np.zeros((1, 2, 2)),
    )
    with pytest.raises(ValueError, match="n_segments"):
        PlanarXYCurves(*args, n_segments=2)
    with pytest.raises(ValueError, match="nfp"):
        PlanarXYCurves(*args, nfp=0)
    with pytest.raises(TypeError, match="stellsym"):
        PlanarXYCurves(*args, stellsym=1)


def test_planar_coils_field_and_versioned_json_roundtrip(tmp_path):
    planar = PlanarXYCurves(
        centers=jnp.asarray([[1.0, 0.0, 0.2], [0.0, 1.1, -0.1]]),
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.2, -0.1, 0.3]]),
        xy_dofs=jnp.broadcast_to(_circle_xy_dofs(0.25), (2, 2, 2)),
        n_segments=20,
        nfp=2,
        stellsym=True,
    )
    coils = PlanarCoils(planar, jnp.asarray([1.2e5, -8.0e4]))
    ordinary = coils.as_coils()
    np.testing.assert_allclose(coils.gamma, ordinary.gamma)
    np.testing.assert_allclose(coils.currents, ordinary.currents)
    points = jnp.asarray([[0.7, 0.1, 0.05], [0.9, -0.2, -0.03]])
    np.testing.assert_allclose(
        coils.to_field().B(points),
        FilamentaryBiotSavart.from_coils(ordinary).B(points),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    transformed_field = jax.jit(lambda value: value.to_field().B(points))(coils)
    np.testing.assert_allclose(
        transformed_field,
        FilamentaryBiotSavart.from_coils(ordinary).B(points),
        rtol=2.0e-15,
        atol=2.0e-15,
    )

    path = tmp_path / "planar-coils.json"
    coils.to_json(path)
    rebuilt = load_coils_json(path)
    assert isinstance(rebuilt, PlanarCoils)
    np.testing.assert_allclose(rebuilt.centers, coils.centers)
    np.testing.assert_allclose(rebuilt.quaternions, coils.quaternions)
    np.testing.assert_allclose(rebuilt.xy_dofs, coils.xy_dofs)
    np.testing.assert_allclose(rebuilt.base_currents, coils.base_currents)
    np.testing.assert_allclose(rebuilt.gamma, coils.gamma)

    legacy_path = tmp_path / "legacy-xyz.json"
    ordinary.to_json(legacy_path)
    legacy = load_coils_json(legacy_path)
    assert isinstance(legacy, Coils)
    np.testing.assert_allclose(legacy.gamma, ordinary.gamma)
    np.testing.assert_allclose(legacy.base_currents, ordinary.base_currents)


def test_planar_coils_constructor_is_traceable_and_preserves_current_dtype():
    planar = PlanarXYCurves(
        centers=jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
        xy_dofs=_circle_xy_dofs(0.3).astype(jnp.float32),
        n_segments=20,
        stellsym=False,
    )
    currents = jnp.asarray([1.234567890123e5], dtype=jnp.float64)
    coils = PlanarCoils(planar, currents)
    assert coils.base_currents.dtype == jnp.float64

    point = jnp.asarray([0.7, 0.1, 0.05])

    def response(values):
        return PlanarCoils(planar, values).to_field().B(point)[0]

    compiled = jax.jit(response)(currents)
    gradient = jax.grad(response)(currents)
    np.testing.assert_allclose(compiled, response(currents))
    assert gradient.shape == currents.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_planar_json_rejects_inconsistent_redundant_currents(tmp_path):
    coils = PlanarCoils(
        PlanarXYCurves(
            jnp.asarray([[1.0, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
            _circle_xy_dofs(0.3),
            stellsym=False,
        ),
        jnp.asarray([1.0e5]),
    )
    path = tmp_path / "inconsistent.json"
    coils.to_json(path)
    import json

    data = json.loads(path.read_text())
    data["base_currents"] = [2.0e5]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inconsistent current"):
        PlanarCoils.from_json(path)


class _FakeSimsoptPlanarCurve:
    order = 1
    local_full_dof_names = (
        "rc(0)",
        "rc(1)",
        "rs(1)",
        "q0",
        "qi",
        "qj",
        "qk",
        "X",
        "Y",
        "Z",
    )
    quadpoints = np.linspace(0.0, 1.0, 24, endpoint=False)

    def __init__(self):
        self.local_full_x = np.asarray(
            [0.4, 0.03, -0.02, 2.0, 0.1, -0.2, 0.3, 1.0, -0.5, 0.2]
        )
        self.x = np.asarray(
            [999.0]
        )  # Must never be used: it represents free DOFs only.


class _FakeSimsoptPlanarCurveOrderTwo:
    order = 2
    local_full_dof_names = (
        "rc(0)",
        "rc(1)",
        "rc(2)",
        "rs(1)",
        "rs(2)",
        "q0",
        "qi",
        "qj",
        "qk",
        "X",
        "Y",
        "Z",
    )
    quadpoints = np.linspace(0.0, 1.0, 24, endpoint=False)

    def __init__(self):
        self.local_full_x = np.asarray(
            [
                0.5,
                -0.04,
                0.01,
                0.02,
                -0.03,
                1.0,
                0.0,
                0.0,
                0.0,
                -0.3,
                0.4,
                0.1,
            ]
        )


class _FakeSimsoptXYZCurve:
    order = 1
    local_full_dof_names = (
        "xc(0)",
        "xs(1)",
        "xc(1)",
        "yc(0)",
        "ys(1)",
        "yc(1)",
        "zc(0)",
        "zs(1)",
        "zc(1)",
    )
    quadpoints = np.linspace(0.0, 1.0, 24, endpoint=False)

    def __init__(self):
        self.local_full_x = np.asarray([1.0, 0.0, 0.3, -0.2, 0.25, 0.0, 0.1, 0.0, 0.05])
        self.x = np.asarray([999.0])


class RotatedCurve:
    def __init__(self, curve, period, flip, nfp):
        self.curve = curve
        angle = 2 * np.pi * period / nfp
        self.rotmat = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        ).T
        if flip:
            self.rotmat = self.rotmat @ np.diag((1.0, -1.0, -1.0))


class _FakeCurrent:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


class _FakeSimsoptCoil:
    def __init__(self, value, curve=None):
        self.curve = _FakeSimsoptPlanarCurve() if curve is None else curve
        self.current = _FakeCurrent(value)


def _expanded_fake_coils(base_coils, *, nfp, stellsym):
    result = []
    for period in range(nfp):
        for flip in [False, True] if stellsym else [False]:
            for base_coil in base_coils:
                curve = (
                    base_coil.curve
                    if period == 0 and not flip
                    else RotatedCurve(base_coil.curve, period, flip, nfp)
                )
                current = base_coil.current.get_value()
                result.append(_FakeSimsoptCoil(-current if flip else current, curve))
    return result


def _install_fake_simsopt_load(monkeypatch, loaded):
    calls = []
    module = types.ModuleType("simsopt")

    def load(path):
        calls.append(path)
        return loaded

    module.load = load
    monkeypatch.setitem(sys.modules, "simsopt", module)
    return calls


def test_simsopt_planar_import_uses_full_dofs_and_preserves_currents():
    imported = PlanarCoils.from_simsopt_planar([_FakeSimsoptCoil(1.3e5)])
    expected = PlanarXYCurves.from_polar_radius(
        centers=jnp.asarray([[1.0, -0.5, 0.2]]),
        quaternions=jnp.asarray([[2.0, 0.1, -0.2, 0.3]]),
        radius_dofs=jnp.asarray([[0.4, -0.02, 0.03]]),
        n_segments=24,
        stellsym=False,
    )
    np.testing.assert_allclose(
        imported.planar_curves.as_xyz_dofs(), expected.as_xyz_dofs()
    )
    np.testing.assert_allclose(imported.base_currents, [1.3e5])
    assert imported.stellsym is False
    assert imported.gamma.shape[0] == 1

    broken = _FakeSimsoptPlanarCurve()
    broken.local_full_dof_names = ("unexpected",)
    with pytest.raises(ValueError, match="unsupported SIMSOPT"):
        PlanarXYCurves.from_simsopt_planar([broken], stellsym=False)


def test_simsopt_planar_import_guards_quaternion_and_quadrature_semantics():
    tiny = _FakeSimsoptPlanarCurve()
    tiny.local_full_x[3:7] = [1.0e-12, 1.0e-12, 0.0, 0.0]
    with pytest.raises(ValueError, match="quaternion norm below 1e-8"):
        PlanarXYCurves.from_simsopt_planar([tiny])

    shifted = _FakeSimsoptPlanarCurve()
    shifted.quadpoints = shifted.quadpoints + 0.01
    with pytest.raises(ValueError, match="noncanonical quadpoints"):
        PlanarXYCurves.from_simsopt_planar([shifted])
    resampled = PlanarXYCurves.from_simsopt_planar([shifted], n_segments=18)
    assert resampled.n_segments == 18


def test_simsopt_planar_import_zero_pads_mixed_radial_orders_exactly():
    first = _FakeSimsoptPlanarCurve()
    second = _FakeSimsoptPlanarCurveOrderTwo()
    imported = PlanarXYCurves.from_simsopt_planar([first, second])
    assert imported.order == 3

    first_alone = PlanarXYCurves.from_simsopt_planar([first])
    np.testing.assert_allclose(
        imported.gamma[0], first_alone.gamma[0], rtol=0.0, atol=3.0e-15
    )


def test_simsopt_json_loader_preserves_planar_geometry_and_symmetry(
    monkeypatch,
):
    base = (_FakeSimsoptCoil(1.3e5),)
    loaded = types.SimpleNamespace(
        coils=_expanded_fake_coils(base, nfp=2, stellsym=True)
    )
    calls = _install_fake_simsopt_load(monkeypatch, loaded)

    imported = load_simsopt_coils_json("planar-simsopt.json", nfp=2, stellsym=True)
    expected = PlanarCoils.from_simsopt_planar(base, nfp=2, stellsym=True)
    assert isinstance(imported, PlanarCoils)
    assert calls == ["planar-simsopt.json"]
    np.testing.assert_allclose(imported.dofs_curves, expected.dofs_curves)
    np.testing.assert_allclose(imported.base_currents, expected.base_currents)
    np.testing.assert_allclose(imported.currents, expected.currents)

    via_class = PlanarCoils.from_simsopt_json(
        "planar-simsopt.json", nfp=2, stellsym=True
    )
    np.testing.assert_allclose(via_class.gamma, imported.gamma)


def test_simsopt_json_loader_uses_full_xyz_dofs(monkeypatch):
    base = _FakeSimsoptCoil(8.0e4, _FakeSimsoptXYZCurve())
    _install_fake_simsopt_load(monkeypatch, [base])
    imported = load_simsopt_coils_json(
        "xyz-simsopt.json",
        nfp=1,
        stellsym=False,
        n_segments=18,
    )

    assert type(imported) is Coils
    np.testing.assert_allclose(
        imported.dofs_curves,
        base.curve.local_full_x.reshape((1, 3, 3)),
    )
    np.testing.assert_allclose(imported.base_currents, [8.0e4])
    assert imported.n_segments == 18
    with pytest.raises(TypeError, match="non-planar"):
        PlanarCoils.from_simsopt_json("xyz-simsopt.json", nfp=1, stellsym=False)


def test_simsopt_json_loader_rejects_bad_expansion_and_mixed_curves(
    monkeypatch,
):
    base = (_FakeSimsoptCoil(1.3e5),)
    _install_fake_simsopt_load(monkeypatch, types.SimpleNamespace(coils=base * 3))
    with pytest.raises(ValueError, match="not divisible"):
        load_simsopt_coils_json("bad.json", nfp=2, stellsym=True)

    wrong_sign = _expanded_fake_coils(base, nfp=2, stellsym=True)
    wrong_sign[1].current = _FakeCurrent(base[0].current.get_value())
    _install_fake_simsopt_load(
        monkeypatch,
        types.SimpleNamespace(coils=wrong_sign),
    )
    with pytest.raises(ValueError, match="current signs"):
        load_simsopt_coils_json("wrong-sign.json", nfp=2, stellsym=True)

    wrong_transform = _expanded_fake_coils(base, nfp=2, stellsym=True)
    wrong_transform[1].curve.rotmat = wrong_transform[1].curve.rotmat.copy()
    wrong_transform[1].curve.rotmat[0, 0] += 0.1
    _install_fake_simsopt_load(
        monkeypatch,
        types.SimpleNamespace(coils=wrong_transform),
    )
    with pytest.raises(ValueError, match="transform"):
        load_simsopt_coils_json("wrong-transform.json", nfp=2, stellsym=True)

    mixed = [base[0], _FakeSimsoptCoil(8.0e4, _FakeSimsoptXYZCurve())]
    _install_fake_simsopt_load(monkeypatch, mixed)
    with pytest.raises(TypeError, match="mixes planar"):
        load_simsopt_coils_json(
            "mixed.json",
            nfp=1,
            stellsym=False,
            source_is_expanded=False,
        )


def test_simsopt_xyz_export_contains_only_independent_base_objects():
    pytest.importorskip("simsopt")
    from simsopt.field import Coil
    from simsopt.geo import CurveXYZFourier

    planar = PlanarXYCurves(
        centers=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.1, 0.2]]),
        quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.2, -0.1, 0.3]]),
        xy_dofs=jnp.broadcast_to(_circle_xy_dofs(0.25), (2, 2, 2)),
        n_segments=20,
        nfp=3,
        stellsym=True,
    )
    curves = planar.to_simsopt_xyz()
    assert len(curves) == planar.n_base_curves
    assert all(isinstance(curve, CurveXYZFourier) for curve in curves)
    np.testing.assert_allclose(curves[0].gamma(), planar.gamma[0])

    coils = PlanarCoils(planar, jnp.asarray([1.2e5, -8.0e4]))
    exported = coils.to_simsopt_xyz()
    assert len(exported) == planar.n_base_curves
    assert all(isinstance(coil, Coil) for coil in exported)
    np.testing.assert_allclose(
        [coil.current.get_value() for coil in exported], coils.base_currents
    )
