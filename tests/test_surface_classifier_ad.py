"""Classifier angular periodicity and derivatives, including the 0/2pi join."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.interpolate import RegularGridInterpolator

from essos.surfaces import SurfaceClassifier, SurfaceRZFourier

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def interpolated_classifier():
    # Independent analytic periodic data isolate interpolation from surface
    # sampling. Retain the production table's single trailing component.
    grids=(jnp.linspace(.8,1.5,5),jnp.linspace(0.,2*jnp.pi,33),jnp.linspace(-.2,.2,5))
    r,p,z=jnp.meshgrid(*grids,indexing="ij")
    values=(r+.2*z+.4*jnp.cos(p)+.3*jnp.sin(p))[...,None]
    classifier=SurfaceClassifier.__new__(SurfaceClassifier)
    classifier.angular_interpolation="periodic_cubic"
    classifier.dist=RegularGridInterpolator(grids,values,fill_value=-1.)
    return classifier


def test_periodic_cubic_preserves_sample_values(interpolated_classifier):
    classifier=interpolated_classifier
    r,p,z=jnp.meshgrid(*classifier.dist.grid,indexing="ij")
    points=jnp.stack([r.ravel(),p.ravel(),z.ravel()],axis=-1)
    actual=jax.jit(jax.vmap(classifier.evaluate_rphiz))(points)
    np.testing.assert_allclose(actual,classifier.dist.values.ravel(),atol=3e-14,rtol=0.)


def test_classifier_values_and_gradients_are_periodic(interpolated_classifier):
    function=jax.jit(jax.value_and_grad(interpolated_classifier.evaluate_rphiz))
    for phi in (-4*np.pi,-1e-7,0.,.13,2*np.pi-1e-7,2*np.pi,12*np.pi+.4):
        point=jnp.array([1.23,phi,.013])
        value,gradient=function(point)
        other,other_gradient=function(point+jnp.array([0.,4*jnp.pi,0.]))
        np.testing.assert_allclose(value,other,atol=2e-14,rtol=0.)
        np.testing.assert_allclose(gradient,other_gradient,atol=2e-14,rtol=0.)


@pytest.mark.parametrize("cartesian",[False,True])
def test_seam_gradient_matches_centered_finite_difference(interpolated_classifier,cartesian):
    classifier=interpolated_classifier
    evaluate=jax.jit(classifier.evaluate_xyz if cartesian else classifier.evaluate_rphiz)
    point=jnp.array([1.23,0.,.013])
    gradient=jax.jit(jax.grad(evaluate))(point)
    eps=1e-6
    finite=np.array([(evaluate(point+d)-evaluate(point-d))/(2*eps) for d in np.eye(3)*eps])
    np.testing.assert_allclose(gradient,finite,rtol=2e-7,atol=2e-9)
    np.testing.assert_allclose(gradient[jnp.array([0,2])], [1.,.2],atol=2e-14,rtol=0.)


def test_first_derivative_is_continuous_at_all_angular_knots(interpolated_classifier):
    derivative=jax.jit(jax.vmap(jax.grad(interpolated_classifier.evaluate_rphiz)))
    angles=interpolated_classifier.dist.grid[1]
    points=jnp.column_stack([jnp.full_like(angles,1.23),angles,jnp.full_like(angles,.013)])
    delta=jnp.array([0.,1e-8,0.])
    np.testing.assert_allclose(derivative(points-delta),derivative(points+delta),atol=3e-8,rtol=0.)


def test_periodic_cubic_retains_radial_and_vertical_bounds(interpolated_classifier):
    classifier=interpolated_classifier
    for point in ([.7,0.,0.],[1.6,0.,0.],[1.2,-.1,-.3],[1.2,7.,.3]):
        assert classifier.evaluate_rphiz(jnp.asarray(point)) == -1.
    legacy=SurfaceClassifier.__new__(SurfaceClassifier)
    legacy.angular_interpolation="linear"
    legacy.dist=classifier.dist
    point=jnp.array([1.23,.17,.013])
    np.testing.assert_array_equal(legacy.evaluate_rphiz(point),legacy.dist(point)[0,0])


def test_surface_constructor_supports_periodic_cubic():
    surface=SurfaceRZFourier(jnp.array([1.,.2]),jnp.array([0.,.2]),
        nfp=1,mpol=1,ntor=0,ntheta=16,nphi=16,close=False)
    classifier=SurfaceClassifier(surface,h=.1,padding=.4,angular_interpolation="periodic_cubic")
    assert classifier.evaluate_xyz(jnp.array([1.,0.,0.]))>0
    assert classifier.evaluate_xyz(jnp.array([1.5,0.,0.]))<0
    point=jnp.array([1.15,0.,.03])
    eps=1e-6
    derivative=jax.grad(classifier.evaluate_xyz)(point)[1]
    finite=(classifier.evaluate_xyz(point+jnp.array([0.,eps,0.]))
        -classifier.evaluate_xyz(point-jnp.array([0.,eps,0.])))/(2*eps)
    np.testing.assert_allclose(derivative,finite,atol=2e-8,rtol=0.)
    with pytest.raises(ValueError,match="angular_interpolation"):
        SurfaceClassifier(surface,angular_interpolation="unknown")
