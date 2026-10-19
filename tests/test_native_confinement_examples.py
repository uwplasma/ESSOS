"""Physical setup fidelity and the corrected reference radial diagnostic."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.constants import FUSION_ALPHA_PARTICLE_ENERGY, ONE_EV
from essos.dynamics import Particles
from essos.particle_losses import classifier_loss_fraction
from examples.coil_optimization import native_particle_optimization as native
from examples.coil_optimization import optimize_coils_particle_confinement_fullorbit_native as full
from examples.coil_optimization import optimize_coils_particle_confinement_guidingcenter_native as gc


@pytest.mark.parametrize("example,factor", [(full,0.), (gc,1.2)])
def test_reference_case_births_and_planar_circles(example, factor):
    settings = dict(example.PARAMETERS, wout=native.WOUT, bdotn_weight=0.)
    base, theta, axis, xyz, pitches, _ = native.build_objective(SimpleNamespace(**settings), "final_radius")
    # Geometric checks on actual physical curves, independent of Fourier scaling.
    gamma = np.asarray(base.gamma)
    centers = gamma.mean(axis=1)
    np.testing.assert_allclose(np.linalg.norm(centers[:,:2], axis=1), 7.74, rtol=2e-14)
    np.testing.assert_allclose(centers[:,2], 0., atol=2e-15)
    displacement = gamma-centers[:,None,:]
    np.testing.assert_allclose(np.linalg.norm(displacement,axis=-1), 4.5, rtol=2e-14)
    normals = np.column_stack((-centers[:,1], centers[:,0], np.zeros(len(centers))))
    np.testing.assert_allclose(np.einsum("csi,ci->cs", displacement, normals), 0., atol=5e-14)
    assert gamma.shape == (12,60,3) and theta.shape == (120,)
    assert base.curves.scaling_factor == factor
    np.testing.assert_allclose(base.dofs_currents_raw, 1.84e7)
    np.testing.assert_allclose(axis, [7.74,0.], atol=2e-14)
    phi = jnp.linspace(0.,2*jnp.pi,10)
    reference_xyz = jnp.column_stack((7.74*jnp.cos(phi),7.74*jnp.sin(phi),jnp.zeros(10)))
    reference_birth = Particles(reference_xyz)
    np.testing.assert_allclose(xyz,reference_xyz,atol=2e-14)
    np.testing.assert_array_equal(pitches,reference_birth.initial_vparallel_over_v)
    assert settings["energy_ev"]*ONE_EV == FUSION_ALPHA_PARTICLE_ENERGY


def test_radial_increment_uses_every_saved_time_and_correct_cartesian_axes():
    radii=jnp.array([[1.,2.,4.],[3.,2.,1.]])
    states=jnp.zeros((2,3,6)).at[...,0].set(7.74+radii)
    # Means of the two radial increments are 1.5 and -1: squares sum to 3.25.
    np.testing.assert_allclose(native.radial_increment_cost(states,7.74,0.),3.25,atol=2e-12)
    gradient=jax.grad(native.radial_increment_cost)(states,7.74,0.)
    np.testing.assert_allclose(gradient[...,0],[[-1.5,0.,1.5],[1.,0.,-1.]],atol=2e-12)
    np.testing.assert_array_equal(gradient[...,3:],0.)


def test_reference_surface_scales_to_circular_case_major_radius():
    small=native.fixed_surface(native.WOUT, ntheta=8,nphi=8)
    large=native.fixed_surface(native.WOUT, ntheta=8,nphi=8,major_radius=7.74)
    np.testing.assert_allclose(large.rc,7.74*small.rc,rtol=2e-14,atol=1e-16)
    np.testing.assert_allclose(large.zs,7.74*small.zs,rtol=2e-14,atol=1e-16)
    np.testing.assert_allclose(large.rc[0],7.74,rtol=2e-14)


def test_circular_target_has_requested_geometry_without_reading_vmec(tmp_path):
    surface = native.fixed_surface(tmp_path / "not_needed.nc", shape="torus",
        major_radius=7.74, minor_radius=3.5, nfp=2, ntheta=32, nphi=24)
    points = np.asarray(surface.gamma)
    radius = np.hypot(np.hypot(points[...,0], points[...,1])-7.74, points[...,2])
    np.testing.assert_allclose(radius, 3.5, rtol=2e-14, atol=2e-14)
    assert surface.nfp == 2
    assert full.PARAMETERS["target_minor_radius"] < full.PARAMETERS["coil_minor_radius"]
    assert gc.PARAMETERS["target_surface"] == full.PARAMETERS["target_surface"] == "torus"


@pytest.fixture(scope="module")
def torus_classifier():
    return native.fixed_classifier(None, shape="torus", major_radius=7.74,
        minor_radius=3.5, nfp=2)


def test_torus_classifier_and_loss_gradient_have_correct_sign(torus_classifier):
    points = jnp.array([[7.74+2.5, 0., 0.], [7.74+3.5, 0., 0.], [7.74+3.6, 0., 0.]])
    distance = jax.jit(jax.vmap(torus_classifier.evaluate_xyz))(points)
    assert distance.shape == (3,)
    assert distance[0] > .8 and distance[2] < -.05
    assert abs(float(distance[1])) < .04
    states = jnp.zeros((1,2,4)).at[0,:,:3].set(points[jnp.array([0,2])])
    loss = lambda states: classifier_loss_fraction(states, torus_classifier, softness=3.)
    value, gradient = jax.jit(jax.value_and_grad(loss))(states)
    np.testing.assert_allclose(value, jax.nn.sigmoid(.3), atol=.03, rtol=0.)
    assert np.isfinite(gradient).all()
    assert gradient[0,-1,0] > 0.  # Moving an outside endpoint further out raises loss.


@pytest.mark.parametrize("example", [gc, full])
def test_torus_loss_objective_runs_native_reverse_ad(example):
    settings = dict(example.PARAMETERS, particles=2, horizon=2e-8, saves=4,
        length_weight=0., curvature_weight=0., bdotn_weight=0., axis_field_weight=0.)
    _, theta, _, _, _, objective = native.build_objective(SimpleNamespace(**settings), "loss_fraction")
    (value, (components, states)), gradient = jax.block_until_ready(
        jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.zeros_like(theta)))
    assert 0 <= value <= 1
    np.testing.assert_array_equal(value, components["particle_metric"])
    assert gradient.shape == theta.shape
    assert np.isfinite(gradient).all() and np.isfinite(states).all()


def test_target_must_be_smaller_than_initial_circular_coils(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        full.main(["--target-minor-radius", "4.5", "--output", str(output)])
    assert not output.exists()
