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


def test_axis_coils_recover_circular_initializer_and_custom_axis():
    from essos.coils import CreateCoilsAroundAxis, CreateEquallySpacedCurves
    for nfp, stellsym in ((1, False), (3, True)):
        surface = surfacerzfourier_from_boundary(jnp.array([[3., .4]]), jnp.array([[0., .5]]), nfp)
        options = dict(n_segments=16, stellsym=stellsym, scaling_factor=.2, scale_fixed=3.)
        actual = CreateCoilsAroundAxis(3, 2, .8, surface=surface, **options)
        expected = CreateEquallySpacedCurves(3, 2, 3., .8, nfp=nfp, **options)
        np.testing.assert_allclose(actual.dofs, expected.dofs, atol=3e-14)
        np.testing.assert_allclose(actual.gamma, expected.gamma, atol=3e-14)
        custom = CreateCoilsAroundAxis(3, 2, .8, axis_function=surface.centerline, nfp=nfp, **options)
        np.testing.assert_allclose(custom.gamma, actual.gamma, atol=3e-14)
    for change in (dict(n_coils=0), dict(order=0), dict(n_axis_samples=1), dict(coil_radius=-1.),
                   dict(axis_function=surface.centerline), dict(surface=None)):
        with pytest.raises(ValueError):
            CreateCoilsAroundAxis(**(dict(n_coils=3, order=2, coil_radius=.8, surface=surface) | change))


def test_axis_coils_follow_3d_boundary_centerline_and_tangent():
    from essos.coils import CreateCoilsAroundAxis
    from essos.surfaces import SurfaceRZFourier
    surface = SurfaceRZFourier(jnp.array([3., .2, .02, .4, .01]), jnp.array([0., .15, .01, .5, .04]),
                              2, 1, 1, rs=jnp.array([0., .03, .01, 0., .02]),
                              zc=jnp.array([.04, .02, 0., 0., .02]))
    radii = np.linspace(.7, 1., 5)
    curves = CreateCoilsAroundAxis(5, 3, radii, surface=surface, n_segments=16)
    centers = np.asarray(curves.curves[:, :, 0]); phi = np.arctan2(centers[:5, 1], centers[:5, 0])
    R = 3 + .2 * np.cos(2 * phi) - .03 * np.sin(2 * phi)
    Z = .04 - .15 * np.sin(2 * phi) + .02 * np.cos(2 * phi)
    np.testing.assert_allclose(centers[:5], np.stack((R*np.cos(phi), R*np.sin(phi), Z), axis=1), atol=2e-14)
    np.testing.assert_allclose(surface.centerline(phi), centers[:5], atol=2e-14)
    np.testing.assert_allclose(surface.centerline(phi + np.pi), centers[5:], atol=2e-14)
    dR = -.4 * np.sin(2 * phi) - .06 * np.cos(2 * phi)
    dZ = -.3 * np.cos(2 * phi) - .04 * np.sin(2 * phi)
    tangent = np.stack((dR*np.cos(phi)-R*np.sin(phi), dR*np.sin(phi)+R*np.cos(phi), dZ), axis=1)
    offsets = np.asarray(curves.gamma[:5]) - centers[:5, None, :]
    np.testing.assert_allclose(np.sum(offsets * tangent[:, None, :], axis=2), 0., atol=3e-14)
    np.testing.assert_allclose(np.linalg.norm(offsets, axis=2), np.broadcast_to(radii[:, None], (5, 16)), atol=2e-14)
    # Independent quadrature checks arclength spacing rather than equal toroidal angles.
    grid = np.linspace(0., np.pi, 8193)
    speed = np.sqrt((3+.2*np.cos(2*grid)-.03*np.sin(2*grid))**2
                    + (-.4*np.sin(2*grid)-.06*np.cos(2*grid))**2
                    + (-.3*np.cos(2*grid)-.04*np.sin(2*grid))**2)
    arc = np.r_[0., np.cumsum((speed[1:] + speed[:-1]) * np.diff(grid) / 2)]
    np.testing.assert_allclose(np.interp(phi, grid, arc) / arc[-1], (np.arange(5)+.5)/5, atol=2e-5)


def test_axis_coil_initializer_gradients_match_finite_differences():
    from essos.coils import CreateCoilsAroundAxis
    from essos.surfaces import SurfaceRZFourier
    def objective(parameters):
        shape, radius = parameters
        surface = SurfaceRZFourier(jnp.array([3., shape, 0., .4, 0.]),
                                  jnp.array([0., .15, 0., .5, 0.]), 2, 1, 1)
        curves = CreateCoilsAroundAxis(3, 2, radius, surface=surface, n_segments=8, n_axis_samples=32)
        return jnp.sum(curves.dofs * jnp.arange(curves.dofs.size).reshape(curves.dofs.shape))
    parameters = jnp.array([.2, .8])
    value, gradient = jax.jit(jax.value_and_grad(objective))(parameters)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    evaluate = jax.jit(objective); h = 1e-5
    finite_difference = [(evaluate(parameters.at[i].add(h)) - evaluate(parameters.at[i].add(-h))) / (2*h)
                         for i in range(2)]
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-7, atol=2e-7)


def test_axis_coil_viewer_builds_closed_full_torus_without_changing_target(monkeypatch):
    import sys
    from types import SimpleNamespace
    from essos.coils import CreateCoilsAroundAxis
    from examples.coil_optimization.optimize_axis_coils_alm import plot_pyvista
    meshes = []
    def lines(points, close=False):
        return np.concatenate((points, points[:1])) if close else points
    monkeypatch.setitem(sys.modules, 'pyvista', SimpleNamespace(
        StructuredGrid=lambda x, y, z: np.stack((x, y, z), axis=-1), lines_from_points=lines))
    scene = SimpleNamespace(add_mesh=lambda mesh, **kwargs: meshes.append((mesh, kwargs)),
                            add_text=lambda *a, **k: None, add_axes=lambda: None,
                            add_legend=lambda: None, view_isometric=lambda: None)
    surface = _small_surface(); surface.range_torus = 'half period'
    final = Coils(CreateCoilsAroundAxis(1, 1, .6, surface=surface, stellsym=True, n_segments=8), [1.])
    orbits = np.arange(180., dtype=float).reshape(10, 6, 3)
    labels = [f'wout-gc: {"trapped" if i < 5 else "passing"} {i+1}' for i in range(10)]
    plot_pyvista(surface, final, scene, 'QA', orbits, labels)
    assert surface.range_torus == 'half period' and not surface.close
    assert surface.nphi == 8 and surface.ntheta == 10
    boundary = meshes[0][0]
    np.testing.assert_allclose(boundary[0], boundary[-1], atol=1e-14)
    np.testing.assert_allclose(boundary[:, 0], boundary[:, -1], atol=1e-14)
    assert meshes[0][1]['opacity'] == .30
    assert len(meshes) == 11 + len(final)
    for (mesh, style), expected in zip(meshes[1:-10], final.gamma):
        np.testing.assert_array_equal(mesh[:-1], expected)
        np.testing.assert_array_equal(mesh[0], mesh[-1])
        assert style['color'] == 'gold'
    for (mesh, style), expected, label in zip(meshes[-10:], orbits, labels):
        np.testing.assert_array_equal(mesh, expected)  # Particle paths must not be closed like coils.
        assert style['label'] == label
    assert len({style['color'] for _, style in meshes[-10:]}) == 10
    meshes.clear()
    chosen = [0, 1, 5, 6]
    plot_pyvista(surface, final, scene, 'QA', orbits[chosen], [labels[i] for i in chosen])
    assert len(meshes) == 5 + len(final)
    assert [style['color'] for _, style in meshes[-4:]] == ['tomato', 'orangered', 'cyan', 'limegreen']
    meshes.clear()
    plot_pyvista(surface, final, scene, 'QA', orbits[chosen], [labels[i] for i in chosen], trapped_fraction=.25)
    assert [style['color'] for _, style in meshes[-4:]] == ['tomato', 'cyan', 'limegreen', 'dodgerblue']


def test_orbit_overlay_classifies_both_models_and_rejects_failed_traces(monkeypatch):
    from types import SimpleNamespace
    import essos.dynamics as dynamics
    from examples.coil_optimization import optimize_axis_coils_alm as example
    equilibrium = SimpleNamespace(nfp=2, AbsB=lambda x: 2. + .1*jnp.cos(x[1]),
        to_xyz=lambda x: jnp.array([3*jnp.cos(x[2]), 3*jnp.sin(x[2]), x[0]]), boundary_distance=lambda x: jnp.array(1.))
    uniform = SimpleNamespace(B=lambda x: jnp.array([0., 0., 5.86]),
                              B_contravariant=lambda x: jnp.array([0., 0., 5.86]), AbsB=lambda x: jnp.array(5.86))
    monkeypatch.setattr(example, 'BiotSavart', lambda c: uniform)
    states = np.zeros((10, 64, 4)); states[..., 0] = .3
    states[..., 2] = np.linspace(0., 4*np.pi, 64)
    states[:5, :, 3] = 1e6*np.cos(np.linspace(0., 4*np.pi, 64)); states[5:, :, 3] = 1e7
    xyz = np.stack((np.cos(states[..., 2]), np.sin(states[..., 2]), states[..., 0]), axis=-1)
    def tracing(**kwargs):
        assert kwargs['solver'].__class__.__name__ == 'Dopri8'
        p = kwargs['particles']; trapped = int(jnp.sum(jnp.abs(p.initial_vparallel_over_v) < .5))
        passing = p.nparticles-trapped
        indices = np.r_[np.arange(trapped) % 5, 5+np.arange(passing) % 5]
        current, positions = states[indices], xyz[indices]
        np.testing.assert_allclose(p.initial_vparallel_over_v, [.1]*trapped + [.95*(-1)**i for i in range(passing)])
        if kwargs['model'] == 'FullOrbitAdaptative':
            np.testing.assert_allclose(np.linalg.norm(p.initial_vxvyvz, axis=1), p.total_speed, rtol=1e-13)
            np.testing.assert_allclose(p.initial_vxvyvz[:, 2]/p.total_speed, p.initial_vparallel_over_v)
            omega = p.charge*5.86/p.mass
            np.testing.assert_allclose(p.initial_xyz_fullorbit - jnp.cross(jnp.array([0., 0., 1.]), p.initial_vxvyvz)/omega,
                                       p.initial_xyz, atol=1e-14)
            assert kwargs['timestep'] <= 2*np.pi/omega/16
            full = np.zeros((p.nparticles, 64, 6)); full[..., :3] = positions; full[..., 5] = current[..., 3]
            return SimpleNamespace(trajectories=full, trajectories_xyz=positions)
        return SimpleNamespace(trajectories=current, trajectories_xyz=positions, status=np.zeros(p.nparticles))
    monkeypatch.setattr(dynamics, 'Tracing', tracing)
    for model in ('wout-gc', 'coils-lorentz'):
        for count, fraction, trapped in ((4, .5, 2), (10, .5, 5), (4, .25, 1), (4, .75, 3),
                                          (4, 0., 0), (4, 1., 4), (3, .5, 2), (12, .25, 3)):
            actual, labels = example.trace_orbits(equilibrium, None, model, nparticles=count, trapped_fraction=fraction)
            indices = np.r_[np.arange(trapped) % 5, 5+np.arange(count-trapped) % 5]
            np.testing.assert_array_equal(actual, xyz[indices])
            assert sum(': trapped ' in label for label in labels) == trapped
            assert sum(': passing ' in label for label in labels) == count-trapped
            assert len(example.orbit_colors(count, fraction)) == count
    for invalid in (0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            example.trace_orbits(equilibrium, None, nparticles=invalid)
    for fraction in (-.1, 1.1, np.nan):
        with pytest.raises(ValueError, match='trapped fraction'):
            example.trace_orbits(equilibrium, None, trapped_fraction=fraction)
    for change in ('nonfinite', 'failed'):
        with monkeypatch.context() as patch:
            bad = states.copy()
            if change == 'nonfinite': bad[0, -1, 3] = np.nan
            patch.setattr(dynamics, 'Tracing', lambda **k: SimpleNamespace(
                trajectories=bad, trajectories_xyz=xyz, status=np.full(10, 4 if change == 'failed' else 0)))
            with pytest.raises(RuntimeError, match='Incomplete orbit trace'):
                example.trace_orbits(equilibrium, None)
    states[:5, :, 3] = 1e7
    _, labels = example.trace_orbits(equilibrium, None)
    assert all(': passing ' in label for label in labels)  # Do not preserve assumed seed labels.
    states[..., 2] = 0.
    _, labels = example.trace_orbits(equilibrium, None)
    assert all(': unresolved ' in label for label in labels)
    stopped, stopped_xyz = states.copy(), xyz.copy()
    stopped[0, 20:], stopped_xyz[0, 20:] = np.inf, np.nan
    monkeypatch.setattr(dynamics, 'Tracing', lambda **k: SimpleNamespace(
        trajectories=stopped, trajectories_xyz=stopped_xyz, status=np.r_[1, np.zeros(9)],
        lcfs_states=states[:, 19], lcfs_positions=xyz[:, 19]))
    actual, labels = example.trace_orbits(equilibrium, None)
    assert np.isfinite(actual).all() and '(LCFS exit)' in labels[0]
    np.testing.assert_array_equal(actual[0, -1], xyz[0, 19])


def test_axis_coil_metrics_report_absolute_field_error_and_physical_geometry(monkeypatch, capsys):
    from scipy.special import ellipe
    from essos.coils import CreateEquallySpacedCurves
    from examples.coil_optimization import optimize_axis_coils_alm as example
    circle = CreateEquallySpacedCurves(1, 1, 3., .6, n_segments=128)
    ellipse = Curves(circle.dofs.at[:, 2, 1].set(-.3), n_segments=128, stellsym=False)
    monkeypatch.setattr(example, 'BdotN_over_B', lambda *a: jnp.array([[-.4, .1]]))
    example.print_metrics('Final', _small_surface(), Coils(ellipse, [1.]))
    output = capsys.readouterr().out
    assert 'Final: max |B.n|/|B|=4.000000e-01' in output
    row = np.asarray(output.splitlines()[-1].split(), dtype=float)
    length = 4 * .6 * ellipe(1 - (.3/.6)**2)
    np.testing.assert_allclose(row, [1., length, .6/.3**2, 2*np.pi/length], atol=1e-6, rtol=0)


def test_axis_coil_metrics_evaluate_real_field_without_mocks(capsys):
    from essos.coils import CreateCoilsAroundAxis
    from examples.coil_optimization.optimize_axis_coils_alm import print_metrics
    surface = _small_surface()
    coils = Coils(CreateCoilsAroundAxis(2, 2, .4, surface=surface, n_segments=16), [1., 1.])
    print_metrics('Initial', surface, coils)
    output = capsys.readouterr().out
    assert 'computing boundary-field diagnostics' in output and 'computing coil geometry' in output
    summary = next(line for line in output.splitlines() if line.startswith('Initial: max'))
    values = [float(item.split('=')[1]) for item in summary.split(';')]
    assert np.isfinite(values).all() and 0 <= values[1] <= values[0] <= 1 + 1e-12


def test_axis_alm_reports_after_autodiff_keep_cache_metadata_static(monkeypatch, capsys):
    from pathlib import Path
    from types import SimpleNamespace
    from examples.coil_optimization import optimize_axis_coils_alm as example
    surface = _small_surface(); surface.nphi = surface.ntheta = 4
    monkeypatch.setattr(example.SurfaceRZFourier, 'from_vmec', lambda *a, **k: surface)
    def checked_coils(*args, **kwargs):
        coils = Coils(*args, **kwargs)
        assert isinstance(coils._tree_flatten()[1]['currents_scale'], float), 'tracer in static JIT cache metadata'
        return coils
    monkeypatch.setattr(example, 'Coils', checked_coils)
    options = SimpleNamespace(grid=4, coils=1, order=1, segments=8, radius_factor=1.3,
                              length_factor=1.3, curvature_factor=3., surface_gap=.1, coil_gap=.05,
                              inner_steps=1, outer_steps=1)
    _, final, history = example.optimize(Path('analytic_boundary.nc'), options)
    output = capsys.readouterr().out
    assert 'Initial: max' in output and 'Final: max' in output
    assert final.curves.n_base_curves == 1 and len(history) == 2 and np.isfinite(history).all()


def test_reactor_scaling_preserves_vmec_geometry_and_field_units_and_normalizes_coils(monkeypatch):
    from essos.fields import Vmec, BiotSavart, VMEC_WOUT_PARTNERS
    from essos.coils import CreateCoilsAroundAxis
    from examples.coil_optimization import optimize_axis_coils_alm as example
    s = jnp.linspace(0., 1., 5); rh = jnp.sqrt(jnp.r_[0., s[1:]-.125])
    table = lambda a, b: jnp.stack((jnp.full_like(s, a), b), axis=-1)
    arrays = dict(nfp=1, ns=5, xm=jnp.array([0, 1]), xn=jnp.array([0, 0]),
                  xm_nyq=jnp.array([0, 1]), xn_nyq=jnp.array([0, 0]), Aminor_p=1.,
                  rmnc=table(3., jnp.sqrt(s)), zmns=table(0., jnp.sqrt(s)),
                  bmnc=table(3., rh), gmnc=table(1.5, .5*rh), bsubsmns=table(0., .02*s),
                  bsubumnc=table(.1, .02*rh), bsubvmnc=table(9., rh),
                  bsupumnc=table(.01, .001*rh), bsupvmnc=table(1., .01*rh))
    arrays.update({partner: .03*arrays[name].at[:, 0].set(0.) for name, partner in VMEC_WOUT_PARTNERS.items()})
    source = Vmec.from_arrays(**arrays, ntheta=3, nphi=3)
    class Reader:
        def __new__(cls, *a, **k): return source
        from_arrays = staticmethod(Vmec.from_arrays)
    monkeypatch.setattr(example, 'Vmec', Reader)
    scaled = example.reactor_equilibrium('in_memory', 1.7, 5.86)
    point = jnp.array([.43, 1.2, .7]); L = 1.7; B = float(scaled.bmnc[2, 0]/source.bmnc[2, 0])
    np.testing.assert_allclose(scaled.Aminor_p, 1.7)
    for method, factor in (('to_xyz', L), ('B', B), ('AbsB', B), ('B_covariant', B*L),
                           ('B_contravariant', B/L), ('sqrtg', L**3), ('curl_B', B/L**2)):
        np.testing.assert_allclose(getattr(scaled, method)(point), getattr(source, method)(point)*factor, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(source.rmnc, arrays['rmnc'])
    # Independent dense midpoint integration, including asymmetric Fourier partners.
    rr, tt, pp = jnp.meshgrid((jnp.arange(12)+.5)/12, jnp.arange(24)*2*jnp.pi/24,
                               jnp.arange(24)*2*jnp.pi/24, indexing='ij')
    grid = jnp.stack((rr, tt, pp), axis=-1).reshape(-1, 3)
    weight = np.asarray(jax.lax.map(lambda q: jnp.abs(scaled.sqrtg(q)), grid, batch_size=32))
    norm = np.asarray(jax.lax.map(scaled.AbsB, grid, batch_size=32))
    np.testing.assert_allclose(np.average(norm, weights=weight), 5.86, rtol=3e-3)
    coils = Coils(CreateCoilsAroundAxis(2, 1, 2.8, surface=scaled.surface, stellsym=True, n_segments=32), [-1., -1.], currents_scale=1.)
    final = example.reactor_coils(scaled, coils)
    norm = np.asarray(jax.lax.map(lambda q: BiotSavart(final).AbsB(scaled.to_xyz(q)), grid, batch_size=32))
    np.testing.assert_allclose(np.average(norm, weights=weight), 5.86, rtol=3e-3)
    np.testing.assert_array_equal(final.gamma, coils.gamma)
