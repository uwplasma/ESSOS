import numpy as np
import pytest
import jax
from essos.coils import Coils, Curves
from essos.surfaces import surfacerzfourier_from_boundary
import jax.numpy as jnp
import random

def test_curves_initialization():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert curves.dofs.shape == (2, 3, 5)
    assert curves.n_segments == 100
    assert curves.nfp == 1
    assert curves.stellsym == True
    assert curves.order == 2
    assert curves.curves.shape == (4, 3, 5)
    assert curves.gamma.shape == (4, 100, 3)
    assert curves.gamma_dash.shape == (4, 100, 3)

def test_curve_and_coil_dof_names_follow_flattened_dofs():
    curves = Curves(jnp.zeros((2, 3, 5)), stellsym=False)
    assert curves.dof_names[:7] == (
        "coil[0].x0", "coil[0].xs(1)", "coil[0].xc(1)",
        "coil[0].xs(2)", "coil[0].xc(2)", "coil[0].y0", "coil[0].ys(1)")
    assert len(curves.dof_names) == curves.dofs.size
    coils = Coils(curves, jnp.array([1.0, 2.0]))
    assert coils.dof_names[-2:] == ("coil[0].current", "coil[1].current")
    assert len(coils.dof_names) == coils.dofs.size
    updated = coils.with_dofs(coils.dofs + 1.0)
    assert jnp.allclose(updated.dofs, coils.dofs + 1.0)
    assert not jnp.allclose(coils.dofs, updated.dofs)
    gradient = jax.grad(lambda dofs: jnp.sum(coils.with_dofs(dofs).gamma))(coils.dofs)
    assert gradient.shape == coils.dofs.shape and jnp.all(jnp.isfinite(gradient))
    updated_curves = curves.with_dofs(curves.dofs + 2.0)
    assert jnp.allclose(updated_curves.dofs, curves.dofs + 2.0)
    assert not jnp.allclose(curves.dofs, updated_curves.dofs)

def test_surface_from_vmec_boundary_preserves_modes_and_is_differentiable():
    rbc = jnp.arange(15.0).reshape(5, 3); zbs = -rbc
    surface = surfacerzfourier_from_boundary(
        rbc, zbs, nfp=2, nphi=8, ntheta=10)
    expected_r = jnp.concatenate((rbc[2:, 0], rbc[:, 1:].T.ravel()))
    expected_z = jnp.concatenate((zbs[2:, 0], zbs[:, 1:].T.ravel()))
    assert surface.mpol == 2 and surface.ntor == 2
    assert jnp.array_equal(surface.rc, expected_r)
    assert jnp.array_equal(surface.zs, expected_z)
    gradient = jax.grad(lambda values: jnp.sum(
        surfacerzfourier_from_boundary(values, zbs, 2, nphi=8, ntheta=10).gamma))(rbc)
    assert jnp.all(jnp.isfinite(gradient))

    with pytest.raises(ValueError, match="equal shape"):
        surfacerzfourier_from_boundary(jnp.zeros((4, 3)), jnp.zeros((4, 3)), 2)

def _small_surface():
    rbc = jnp.zeros((5, 3)); zbs = jnp.zeros((5, 3))
    rbc = rbc.at[2, 0].set(1.0).at[2, 1].set(0.2)
    zbs = zbs.at[2, 1].set(0.2)
    return surfacerzfourier_from_boundary(rbc, zbs, 2, nphi=8, ntheta=10)

@pytest.mark.parametrize("name, caches", (
    ("theta2d", ("_theta2d", "_phi2d")),
    ("phi2d", ("_theta2d", "_phi2d")),
    ("angles", ("_angles",)),
    ("gamma", ("_gamma", "_gammadash_theta", "_gammadash_phi")),
    ("gammadash_theta", ("_gamma", "_gammadash_theta", "_gammadash_phi")),
    ("gammadash_phi", ("_gamma", "_gammadash_theta", "_gammadash_phi")),
    ("normal", ("_normal", "_unitnormal", "_area_element")),
    ("unitnormal", ("_normal", "_unitnormal", "_area_element")),
    ("area_element", ("_normal", "_unitnormal", "_area_element")),
))
def test_surface_cache_is_concrete_and_does_not_retain_tracers(name, caches):
    surface = _small_surface()
    value = jax.jit(lambda scale: scale * jnp.sum(getattr(surface, name)))(1.0)
    assert jnp.isfinite(value)
    # Access after the transform must recompute concrete values, not retrieve a
    # DynamicJaxprTracer that escaped from the compiled objective.
    assert jnp.all(jnp.isfinite(getattr(surface, name)))
    cached = [getattr(surface, cache) for cache in caches]
    assert all(item is not None and not isinstance(item, jax.core.Tracer) for item in cached)

def test_curves_initialization_with_params():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, n_segments=50, nfp=2, stellsym=False)
    assert curves.dofs.shape == (2, 3, 5)
    assert curves.n_segments == 50
    assert curves.nfp == 2
    assert curves.stellsym == False
    assert curves.order == 2
    assert curves.curves.shape == (4, 3, 5)
    assert curves.gamma.shape == (4, 50, 3)
    assert curves.gamma_dash.shape == (4, 50, 3)

def test_curves_computed_attributes():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert curves.gamma.shape == (4, 100, 3)
    assert curves.gamma_dash.shape == (4, 100, 3)
    assert curves.length.shape == (4,)

def test_curves_property_setters():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    new_dofs = jnp.ones((2, 3, 5))
    curves.dofs = new_dofs
    assert jnp.allclose(curves.dofs, new_dofs)
    curves.n_segments = 50
    assert curves.n_segments == 50
    curves.nfp = 2
    assert curves.nfp == 2
    curves.stellsym = False
    assert curves.stellsym == False

def test_curves_pytree_preserves_scaling_metadata():
    dofs = jnp.ones((2, 3, 5))
    curves = Curves(dofs, scaling_type=2, scaling_factor=0.3, scale_fixed=7.0)
    curves_copy = jax.tree_util.tree_map(lambda x: x, curves)

    assert curves_copy.scaling_type == curves.scaling_type
    assert curves_copy.scaling_factor == curves.scaling_factor
    assert curves_copy.scale_fixed == curves.scale_fixed

def _fake_simsopt(monkeypatch):
    import sys
    import types
    import numpy as np

    class CurveXYZFourier:
        def __init__(self, n_segments, order):
            self.x = np.zeros(3 * (2 * order + 1))

    geo = types.SimpleNamespace(CurveXYZFourier=CurveXYZFourier)
    field = types.SimpleNamespace(
        Current=lambda value: value,
        coils_via_symmetries=lambda curves, currents, nfp, stellsym: [
            types.SimpleNamespace(curve=c, current=i) for c, i in zip(curves, currents)])
    monkeypatch.setitem(sys.modules, "simsopt", types.ModuleType("simsopt"))
    monkeypatch.setitem(sys.modules, "simsopt.geo", geo)
    monkeypatch.setitem(sys.modules, "simsopt.field", field)


def test_scaled_curves_convert_with_physical_coefficients(monkeypatch):
    from essos.coils import DiscretizedCoils
    _fake_simsopt(monkeypatch)
    dofs = jnp.zeros((1, 3, 5)).at[0, 0, 2].set(2.0).at[0, 1, 1].set(2.0).at[0, 2, 3].set(0.3)
    curves = Curves(dofs, 32, 1, False, scaling_factor=0.3, scale_fixed=7.0)
    coils = Coils(curves, jnp.array([2.0e5]), currents_scale=1.0e5)
    assert not jnp.allclose(curves.dofs, dofs)  # public dofs are scaled

    assert jnp.allclose(curves.to_simsopt()[0].x, dofs[0].ravel())
    simsopt_coil = coils.to_simsopt()[0]
    assert jnp.allclose(simsopt_coil.curve.x, dofs[0].ravel())
    assert jnp.isclose(simsopt_coil.current, 2.0e5)
    assert jnp.allclose(DiscretizedCoils.from_Coils(coils).gamma, coils.gamma, atol=1e-12)


def test_curves_str_repr():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert isinstance(str(curves), str)
    assert isinstance(repr(curves), str)

def test_curves_save_curves(tmp_path):
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    filename = tmp_path / "curves.txt"
    curves.save_curves(filename)
    with open(filename, "r") as file:
        content = file.read()
    assert "nfp stellsym order" in content

def test_curves_plot():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    curves.plot(show=False)

def test_curves_len():
    dofs = jnp.zeros((2, 3, 5))
    nfp = random.randint(1, 10)
    curves = Curves(dofs, nfp=nfp)
    assert len(curves) == 2*2*nfp

def test_curves_getitem():
    dofs = jnp.ones((2, 3, 5))
    nfp = random.randint(4, 10)
    curves = Curves(dofs, nfp=nfp)
    assert curves[0].curves.shape == (1, 3, 5)
    assert curves[1].curves.shape == (1, 3, 5)
    assert curves[2].curves.shape == (1, 3, 5)
    assert curves[3].curves.shape == (1, 3, 5)
    assert curves[1:3].curves.shape == (2, 3, 5)
    assert curves[jnp.array([0, 3])].curves.shape == (2, 3, 5)    

def test_curves_add():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=False)
    curves3 = curves1 + curves2
    assert curves3.dofs.shape == (4, 3, 5)

def test_curves_contains():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, stellsym=False)
    assert curves[0] in curves
    assert curves[1] in curves

def test_curves_eq():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=False)
    assert curves1 == curves2

def test_curves_ne():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=True)
    assert curves1 != curves2

def test_curves_iter():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, stellsym=False)
    for curve in curves:
        assert curve.curves.shape == (1, 3, 5)

if __name__ == "__main__":
    pytest.main()


def test_two_coil_fields_share_jit_caches():
    """Coils carried its current scale as an array in pytree metadata, so the
    second BiotSavart traced under jit failed comparing it with the first."""
    from pathlib import Path
    from essos.fields import BiotSavart

    path = str(Path(__file__).resolve().parents[1] / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json")
    first, second = BiotSavart(Coils.from_json(path)), BiotSavart(Coils.from_json(path))
    x = jnp.array([1.0, 0.1, 0.05])
    assert jnp.allclose(jax.jit(lambda p: first.AbsB(p))(x), jax.jit(lambda p: second.AbsB(p))(x))
    assert isinstance(first.coils._tree_flatten()[1]["currents_scale"], float)


def test_curves_pytree_round_trips_placeholder_leaves():
    """diffrax's implicit solvers unflatten the field with jax.ShapeDtypeStruct
    leaves; unflatten used to divide them by the mode scaling and failed."""
    from pathlib import Path

    path = str(Path(__file__).resolve().parents[1] / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json")
    coils = Coils.from_json(path)
    leaves, treedef = jax.tree_util.tree_flatten(coils)
    structs = [jax.ShapeDtypeStruct(jnp.shape(leaf), jnp.result_type(leaf)) for leaf in leaves]
    assert jax.tree_util.tree_leaves(jax.tree_util.tree_unflatten(treedef, structs)) == structs
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(rebuilt.dofs_curves, coils.dofs_curves)
    assert jnp.allclose(rebuilt.gamma, coils.gamma)


def test_coil_tracer_unflatten_preserves_physical_scales():
    curves = Curves(jnp.arange(18.).reshape(2, 3, 3), stellsym=False,
                    scale_fixed=3., scaling_factor=0.4)
    coils = Coils(curves, jnp.array([3., 6.]), currents_scale=3.)
    leaves, tree = jax.tree_util.tree_flatten(coils)
    weights = jnp.arange(1., 19.).reshape(2, 3, 3)
    def physical_value(coefficients, currents):
        rebuilt = jax.tree_util.tree_unflatten(tree, [coefficients, currents])
        return jnp.sum(rebuilt.curves._dofs * weights) + jnp.sum(rebuilt.dofs_currents_raw**2)
    value, (geometry_grad, current_grad) = jax.jit(jax.value_and_grad(
        physical_value, argnums=(0, 1)))(*leaves)
    assert jnp.allclose(value, jnp.sum(curves._dofs * weights) + 45.)
    assert jnp.allclose(geometry_grad, weights / curves.scaling[None, None, :])
    assert jnp.allclose(current_grad, jnp.array([18., 36.]))


def test_coils_indexing_addition_and_exclusion():
    from essos.coils import CreateEquallySpacedCurves
    coils = Coils(CreateEquallySpacedCurves(2, 2, 1.0, 0.3, 32, 2, True), jnp.array([1e5, 2e5]))
    one, part = coils[3], coils[1:5]
    np.testing.assert_allclose(one.gamma[0], coils.gamma[3], atol=1e-14)
    np.testing.assert_allclose(one.currents, coils.currents[3:4])
    np.testing.assert_allclose(part.currents, coils.currents[1:5])
    both = coils + coils
    np.testing.assert_allclose(both.gamma, jnp.concatenate([coils.gamma] * 2), atol=1e-14)
    fewer = coils.__exclude_coil__(2)
    np.testing.assert_allclose(fewer.gamma, jnp.delete(coils.gamma, 2, axis=0), atol=1e-14)


def test_discretized_coils_copy_and_json_round_trip(tmp_path):
    from essos.coils import DiscretizedCoils
    t = jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False)
    base = jnp.stack([jnp.stack([3 + .5 * jnp.cos(t), .2 * k + 0 * t, .5 * jnp.sin(t)], axis=1) for k in range(4)])
    coils = DiscretizedCoils(base, jnp.arange(1., 5.) * 1e5, nfp=2, stellsym=True)
    np.testing.assert_allclose(coils.copy().gamma, coils.gamma, atol=1e-12)
    coils.to_json(str(tmp_path / "c.json"))
    loaded = DiscretizedCoils.from_json(str(tmp_path / "c.json"))
    np.testing.assert_allclose(loaded.gamma, coils.gamma, atol=1e-12)


def test_zero_currents_give_finite_dofs():
    from essos.coils import CreateEquallySpacedCurves
    curves = CreateEquallySpacedCurves(n_curves=2, order=2, R=1.7, r=0.6, nfp=2, stellsym=True)
    coils = Coils(curves=curves, currents=[0.0, 0.0])
    assert jnp.all(jnp.isfinite(coils.dofs)) and jnp.all(jnp.isfinite(coils.currents))


def _circle_dofs(radius=1.5, z=0.0):
    """Fourier dofs of x = R cos 2 pi t, y = R sin 2 pi t, z = const: layout [a0, sin1, cos1] per axis."""
    d = np.zeros((1, 3, 3))
    d[0, 0, 2] = radius
    d[0, 1, 1] = radius
    d[0, 2, 0] = z
    return jnp.asarray(d)


def test_curves_circle_length_curvature_and_scaling_options(tmp_path):
    """Circle of radius R: length 2 pi R, curvature 1/R, gamma'' = -(2 pi)^2 (gamma - centre)."""
    R = 1.5
    c = Curves(_circle_dofs(R), n_segments=64, nfp=1, stellsym=False)
    np.testing.assert_allclose(c.length, 2 * np.pi * R, rtol=1e-12)
    np.testing.assert_allclose(c.curvature, 1 / R, rtol=1e-10)
    np.testing.assert_allclose(c.gamma_dashdash, -(2 * np.pi) ** 2 * c.gamma, atol=1e-10)
    np.testing.assert_allclose(c.gamma_dashdash, c.gamma_dashdash)  # cached second access

    # scaling_type spellings are equivalent for the 1D mode measure; unknown ones raise
    for spelling in ("L1", 1, "L2", 2, "Linfty", -1, jnp.inf):
        assert Curves(_circle_dofs(), 16, 1, False, scaling_type=spelling).scaling.shape == (3,)
    with pytest.raises(ValueError, match="Unknown scaling_type"):
        Curves(_circle_dofs(), 16, 1, False, scaling_type="L3")
    with pytest.raises(AssertionError):
        Curves(jnp.zeros((1, 3, 4)))  # even number of Fourier slots

    # exp(scaling_factor * order) * scale_fixed, applied mode by mode; setters invalidate the cache
    c.scaling_type = "L1"
    c.scaling_factor = 0.5
    c.scale_fixed = 2.0
    assert c.scaling_type == 1
    np.testing.assert_allclose(c.scaling, 2.0 * np.exp(0.5 * np.array([0, 1, 1])), rtol=1e-12)
    np.testing.assert_allclose(c.dofs[0, 0, 2] * c.scaling[2], c.dofs[0, 0, 2] * c.scaling[2])
    np.testing.assert_allclose(c.copy().scaling, c.scaling)

    # order setter pads with zeros (higher) or truncates (lower) without touching the circle
    gamma0 = c.gamma
    c.order = 3
    assert c.order == 3 and c._dofs.shape == (1, 3, 7)
    np.testing.assert_allclose(c.gamma, gamma0, atol=1e-12)
    c.order = 1
    assert c._dofs.shape == (1, 3, 3)
    np.testing.assert_allclose(c.gamma, gamma0, atol=1e-12)


def test_curves_container_protocol_errors_and_exports(tmp_path):
    a = Curves(_circle_dofs(1.0), 32, 1, False)
    b = Curves(_circle_dofs(2.0), 32, 1, False)
    ab = a + b
    assert len(ab) == 2 and (a == a) and not (a == b) and (a != b) and not (a != a)
    assert (ab[0:1] == a) and (ab[jnp.array([1])] == b)
    assert a in ab and b in ab
    assert not (a == ab)  # different shape
    for bad in (lambda: a["x"], lambda: a + 1, lambda: 1 in a, lambda: a == 1):
        with pytest.raises(TypeError):
            bad()
    # a close=True plot adds the first point again; derivative arrows and a non-3d axis are accepted
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection="3d")
    ab.plot(ax=ax, show=False, plot_derivative=True, close=True, label="c")
    assert len(ax.lines) == 2
    ab.plot(ax=None, show=False, axis_equal=False)
    plt.close("all")
    # VTK: closed polylines carry one extra point per coil, open ones do not
    ab.to_vtk(str(tmp_path / "closed"), extra_data={"w": np.zeros(66)})
    ab.to_vtk(str(tmp_path / "open"), close=False, extra_data={"w": np.zeros(64)})
    assert (tmp_path / "closed.vtu").exists() or (tmp_path / "closed.vtp").exists()
    assert not np.isnan(a.length).any()


def test_coils_container_protocol_json_and_dof_setters(tmp_path):
    import json
    from essos.coils import CreateEquallySpacedCurves
    coils = Coils(CreateEquallySpacedCurves(2, 2, 1.0, 0.3, 32, 2, True), jnp.array([1e5, 2e5]))
    assert len(coils) == 8 and "nfp" in str(coils) and "Currents scaling factor" in repr(coils)
    assert coils == coils.copy() and coils in coils and coils.curvature.shape == (8, 32) and coils.length.shape == (8,)
    assert coils.gamma_dashdash.shape == (8, 32, 3) and coils.n_segments == 32 and coils.order == 2
    for bad in (lambda: coils["x"], lambda: coils + 1, lambda: 1 in coils, lambda: coils == 1):
        with pytest.raises(TypeError):
            bad()
    assert not (coils == coils[0:3])  # different dof count

    # dof setters round trip: raw currents, normalised currents, the scale and the stacked vector
    cs = float(coils.currents_scale)
    c2 = coils.with_dofs(coils.dofs * 1.0)
    c2.dofs_currents = jnp.array([2.0, 4.0])
    np.testing.assert_allclose(c2.dofs_currents_raw, jnp.array([2.0, 4.0]) * cs)
    np.testing.assert_allclose(c2.currents[:2], jnp.array([2.0, 4.0]) * cs)
    c2.currents_scale = 2 * cs  # the normalised dofs are held, so the physical currents follow the scale
    np.testing.assert_allclose(c2.dofs_currents_raw, jnp.array([2.0, 4.0]) * 2 * cs)
    c2.dofs_currents_raw = 3.0  # scalar -> atleast_1d
    assert c2.dofs_currents_raw.shape == (1,)
    c3 = coils.copy()
    c3.x = coils.x
    np.testing.assert_allclose(c3.dofs, coils.dofs)
    c3.dofs_curves = coils.dofs_curves * 1.0
    assert c3.dofs_curves.shape == coils.dofs_curves.shape

    # to_json / from_json: current format and the legacy key names
    coils.to_json(str(tmp_path / "new.json"))
    back = Coils.from_json(str(tmp_path / "new.json"))
    np.testing.assert_allclose(back.gamma, coils.gamma, atol=1e-13)
    np.testing.assert_allclose(back.currents, coils.currents)
    data = json.load(open(tmp_path / "new.json"))
    legacy = {k: data[k] for k in ("nfp", "stellsym", "n_segments")}
    legacy["dofs_curves"] = data["dofs_curves_raw"]
    legacy["dofs_currents"] = data["dofs_currents_raw"]
    json.dump(legacy, open(tmp_path / "old.json", "w"))
    old = Coils.from_json(str(tmp_path / "old.json"))
    np.testing.assert_allclose(old.gamma, coils.gamma, atol=1e-13)

    # save_coils appends a readable text record including the extra note
    path = tmp_path / "coils.txt"
    coils.save_coils(str(path), text="note")
    txt = path.read_text()
    assert "Curves scaling type" in txt and txt.rstrip().endswith("note")
    # plot/to_vtk delegate to the curves
    import matplotlib
    matplotlib.use("Agg")
    coils.plot(show=False)
    coils.to_vtk(str(tmp_path / "coils"))
    import matplotlib.pyplot as plt
    plt.close("all")


def test_fit_dofs_from_coils_recovers_circle_for_any_parametrisation():
    """Arclength resampling makes the fit independent of the input parametrisation."""
    from essos.coils import fit_dofs_from_coils
    R, M = 1.5, 50
    s = jnp.linspace(0, 1, M, endpoint=False)
    t = s + 0.08 * jnp.sin(2 * jnp.pi * s)  # monotone non-uniform parametrisation
    circle = jnp.stack([R * jnp.cos(2 * jnp.pi * t), R * jnp.sin(2 * jnp.pi * t), 0 * t], axis=1)[None]
    dofs, gam = fit_dofs_from_coils(circle, order=3, n_segments=64)  # arclength path
    assert dofs.shape == (1, 3, 7) and gam.shape == (1, 64, 3)
    np.testing.assert_allclose(jnp.linalg.norm(gam[0, :, :2], axis=1), R, rtol=5e-3)
    np.testing.assert_allclose([dofs[0, 0, 2], dofs[0, 1, 1]], R, rtol=5e-3)  # cos1 of x, sin1 of y
    # uniform fast path: equal M (identity) and index-space subsampling
    u = jnp.stack([R * jnp.cos(2 * jnp.pi * jnp.arange(48) / 48), R * jnp.sin(2 * jnp.pi * jnp.arange(48) / 48),
                   jnp.zeros(48)], axis=1)[None]
    d_same, g_same = fit_dofs_from_coils(u, 2, 48, assume_uniform=True)
    np.testing.assert_allclose(g_same, u)
    np.testing.assert_allclose(d_same[0, 0, 2], R, rtol=1e-12)
    np.testing.assert_allclose(d_same[0, 1, 1], R, rtol=1e-12)
    d_sub, g_sub = fit_dofs_from_coils(u, 2, 24, assume_uniform=True)
    assert g_sub.shape == (1, 24, 3)
    np.testing.assert_allclose(d_sub[0, 0, 2], R, rtol=1e-12)
    # order above Nyquist is zero padded
    d_pad, _ = fit_dofs_from_coils(u[:, ::6], order=6, n_segments=8, assume_uniform=True)
    assert d_pad.shape == (1, 3, 13) and jnp.all(d_pad[..., 9:] == 0)


def test_discretized_coils_properties_symmetry_and_roundtrips(tmp_path):
    """Finite-difference derivatives of a circle, symmetry expansion, setters, conversions and errors."""
    import json
    from essos.coils import DiscretizedCoils
    n, R = 128, 2.0
    th = 2 * jnp.pi * jnp.arange(n) / n
    ring = jnp.stack([R * jnp.cos(th), R * jnp.sin(th), 0.3 + 0 * th], axis=1)
    dc = DiscretizedCoils(ring[None], 2.0e5)  # scalar current is broadcast
    assert dc.gamma.shape == (1, n, 3) and dc.n_base_curves == 1 and len(dc) == 1
    np.testing.assert_allclose(dc.length, 2 * np.pi * R, rtol=1e-3)  # central-difference error O(1/n^2)
    np.testing.assert_allclose(dc.curvature, 1 / R, rtol=1e-3)
    np.testing.assert_allclose(dc.gamma_dash, dc.gamma_dash)  # cached
    np.testing.assert_allclose(dc.curvature, dc.curvature)
    assert "DiscretizedCoils" in str(dc) and "DiscretizedCoils" in repr(dc)
    np.testing.assert_allclose(dc.currents, [2.0e5])
    assert dc.x.shape == dc.dofs.shape

    # an already expanded stellarator-symmetric set is folded back to its base curves
    full = DiscretizedCoils(ring[None], 1e5, nfp=2, stellsym=True)
    assert full.gamma.shape == (4, n, 3) and full.currents.shape == (4,)
    refolded = DiscretizedCoils(full.gamma, full.currents, nfp=2, stellsym=True)
    assert refolded.n_base_curves == 1
    np.testing.assert_allclose(refolded.gamma, full.gamma, atol=1e-12)
    # setters: expanded gamma is accepted, wrong counts rejected; dofs round trip
    full.gamma = full.gamma
    full.gamma = ring[None] * 1.0
    with pytest.raises(AssertionError):
        full.gamma = jnp.zeros((3, n, 3))
    with pytest.raises(AssertionError):
        full.gamma = jnp.zeros((1, n))
    with pytest.raises(AssertionError):
        full.dofs_gamma = jnp.zeros((1, n))
    with pytest.raises(AssertionError):
        full.dofs_currents_raw = jnp.ones(3)
    full.dofs = full.dofs * 1.0
    full.dofs_gamma = full.dofs_gamma * 1.0
    full.scale_fixed = 2.0 * full.scale_fixed  # normalised dofs are held, so gamma follows the scale
    np.testing.assert_allclose(full.gamma[0], 2 * ring, atol=1e-12)
    s = float(full.currents_scale)
    full.currents_scale = 2 * s
    np.testing.assert_allclose(full.dofs_currents_raw, [2e5])
    full.dofs_currents = jnp.array([3.0])
    np.testing.assert_allclose(full.dofs_currents_raw, jnp.array([3.0]) * full.currents_scale)

    # container protocol
    two = DiscretizedCoils(jnp.stack([ring, ring + jnp.array([0, 0, 1.0])]), jnp.array([1.0, 2.0]) * 1e5)
    assert two[0].gamma.shape == (1, n, 3) and two[0:2].gamma.shape == (2, n, 3)
    assert two[jnp.array([1])].gamma.shape == (1, n, 3)
    assert (two + two).gamma.shape == (4, n, 3)
    assert two in two and two == two.copy() and not (two != two) and two != two[0:1]
    assert [c.gamma.shape for c in two] == [(1, n, 3)] * 2
    for bad in (lambda: two["x"], lambda: two + 1, lambda: 1 in two, lambda: two == 1):
        with pytest.raises(TypeError):
            bad()
    with pytest.raises(AssertionError):
        DiscretizedCoils(ring, 1.0)  # not 3D
    with pytest.raises(AssertionError):
        DiscretizedCoils(ring[None, :, :2], 1.0)
    with pytest.raises(AssertionError):
        DiscretizedCoils(ring[None], jnp.ones(2))

    # pytree round trip and jit
    leaves, tree = jax.tree_util.tree_flatten(two)
    again = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_allclose(again.gamma, two.gamma, atol=1e-12)
    np.testing.assert_allclose(jax.jit(lambda c: c.gamma.sum())(two), two.gamma.sum(), rtol=1e-12)

    # text / json (all legacy key spellings) / plot / vtk
    two.save_coils(str(tmp_path / "d.txt"), text="hi")
    assert "Base gamma dofs" in (tmp_path / "d.txt").read_text()
    two.to_json(str(tmp_path / "d.json"))
    d = json.load(open(tmp_path / "d.json"))
    for key, name in (("dofs_gamma", "a"), ("gamma", "b")):
        legacy = {"dofs_currents": d["dofs_currents"], "currents_scale": d["currents_scale"], "nfp": 1, "stellsym": False,
                  "fixed_scale": d["scale_fixed"], key: (jnp.asarray(d["dofs_gamma_raw"]) / d["scale_fixed"]).tolist()}
        json.dump(legacy, open(tmp_path / f"{name}.json", "w"))
        loaded = DiscretizedCoils.from_json(str(tmp_path / f"{name}.json"))
        np.testing.assert_allclose(loaded.gamma, two.gamma, atol=1e-10)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    two.plot(show=False, plot_derivative=True, close=True, label="x")
    two.plot(show=False, axis_equal=False)
    plt.close("all")
    two.to_vtk(str(tmp_path / "dv"))
    two.to_vtk(str(tmp_path / "dv_open"), close=False, extra_data={"w": np.zeros(2 * n)})

    # conversion to Fourier Coils and back keeps the geometry of a smooth curve
    cc = DiscretizedCoils(ring[None], 1e5, nfp=3, stellsym=True)
    fourier = cc.to_Coils(order=4)
    np.testing.assert_allclose(fourier.gamma, cc.gamma, atol=1e-10)
    default_order = cc.to_Coils()
    assert default_order.order == n // 2 - 1
    back = DiscretizedCoils.from_Coils(fourier)
    np.testing.assert_allclose(back.gamma, cc.gamma, atol=1e-10)
