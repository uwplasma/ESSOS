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

def test_surface_cache_does_not_retain_outer_jit_tracers():
    rbc = jnp.zeros((5, 3)); zbs = jnp.zeros((5, 3))
    rbc = rbc.at[2, 0].set(1.0).at[3, 0].set(0.2)
    zbs = zbs.at[3, 0].set(0.2)
    surface = surfacerzfourier_from_boundary(rbc, zbs, 2, nphi=8, ntheta=10)
    value = jax.jit(lambda scale: scale * jnp.sum(surface.gamma))(1.0)
    assert jnp.isfinite(value)
    # Access after the transform must recompute concrete values, not retrieve a
    # DynamicJaxprTracer that escaped from the compiled objective.
    assert jnp.all(jnp.isfinite(surface.gamma))

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
