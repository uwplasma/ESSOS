"""Physical setup fidelity and the corrected reference radial diagnostic."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.constants import FUSION_ALPHA_PARTICLE_ENERGY, ONE_EV
from essos.dynamics import Particles, Tracing
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
@pytest.mark.parametrize("accumulation", ["step", "segment"])
def test_torus_loss_objective_runs_native_reverse_ad(example, accumulation):
    settings = dict(example.PARAMETERS, particles=2, horizon=2e-8, saves=4,
        parameter_accumulation=accumulation,
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


@pytest.mark.parametrize("radius", [3.5, 4.5, -1., float("nan")])
def test_safety_surface_must_lie_between_target_and_coils(tmp_path, radius):
    output = tmp_path / "invalid_safety"
    with pytest.raises(SystemExit):
        full.main(["--safety-minor-radius", str(radius), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("model", ["GuidingCenterAdaptative", "FullOrbitAdaptative"])
@pytest.mark.parametrize("method", ["essos_rk45", "essos_dopri8"])
def test_safety_stops_escaping_orbits_and_retains_accepted_step_ad(model, method):
    from tests.test_native_tracing import UniformField

    stop = native.toroidal_safety_stop(7.74, .3)

    def run(speed):
        xyz = jnp.array([[7.74, 0., 0.]])
        particles = Particles(xyz, initial_vparallel_over_v=jnp.array([speed]),
            mass=1., charge=1., energy=.5, initial_xyz_fullorbit=xyz,
            initial_vxvyvz=jnp.array([[0., 0., speed]]))
        trace = Tracing(field=UniformField(), particles=particles, model=model,
            integrator=method, native_adjoint="accepted",
            native_adjoint_max_accepted_steps=32, native_adjoint_checkpoint_steps=4,
            maxtime=2., timestep=.07, times_to_trace=21, progress=False,
            stopping_criteria=stop)
        return trace.trajectories[0,-1,2], (trace.trajectories, trace.boundary_hits)

    (value, (states, hits)), gradient = jax.jit(jax.value_and_grad(run, has_aux=True))(.53)
    np.testing.assert_array_equal(hits, [True])
    assert np.isfinite(states).all() and 0 < value < .3
    np.testing.assert_allclose(states[0,-1], states[0,-2], atol=0.)
    assert float(jnp.max(jnp.abs(states[0,:,2]))) < .3
    assert np.isfinite(gradient) and gradient > 0
    # This constant-velocity orbit keeps the same accepted/saved-time masks;
    # finite differences therefore also check the frozen accepted-path VJP.
    epsilon = 1e-5
    forward = jax.jit(run)
    fd = (forward(.53+epsilon)[0]-forward(.53-epsilon)[0])/(2*epsilon)
    np.testing.assert_allclose(gradient, fd, rtol=2e-8, atol=1e-10)


@pytest.mark.parametrize("example", [gc, full])
def test_real_coil_objective_applies_separate_safety_stop(example):
    # A small safety radius makes the actual reference-coil particle leave in
    # a short run, so this tests wiring through build_objective and the loss.
    settings = dict(example.PARAMETERS, particles=1, horizon=1e-7, saves=40,
        target_minor_radius=.001, safety_minor_radius=.002,
        length_weight=0., curvature_weight=0., bdotn_weight=0., axis_field_weight=0.)
    _, theta, axis, _, _, objective = native.build_objective(SimpleNamespace(**settings), "final_radius")
    (value, (components, states)), gradient = jax.block_until_ready(
        jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.zeros_like(theta)))
    assert components["safety_stop_fraction"] == 1.
    assert np.isfinite(value) and np.isfinite(gradient).all() and np.isfinite(states).all()
    radius = np.asarray(native.minor_radius(states, *axis))
    # Full-orbit births include the Larmor-radius displacement: these can be
    # initially outside this deliberately tiny safety torus and stop at t=0.
    if example is gc:
        assert np.max(radius) < settings["safety_minor_radius"]
    np.testing.assert_array_equal(states[0,-1], states[0,-2])


@pytest.mark.parametrize("example", [gc, full])
@pytest.mark.parametrize("objective_name", ["final_radius", "loss_fraction"])
def test_split_optimization_uses_production_forward_and_full_reverse_ad(example, objective_name, monkeypatch):
    from essos.background_species import BackgroundSpecies
    from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q
    from essos.fields import BiotSavart
    from essos.split import trace_split, SplitStatus

    def forbidden(*args, **kwargs):
        raise AssertionError("collision optimization used the collisionless tracer")
    monkeypatch.setattr(native, "Tracing", forbidden)
    settings = dict(example.PARAMETERS, particles=2, horizon=2e-8, saves=4,
        collisions=True, collision_step=5e-9, collision_dtmax=1e-8,
        collision_checkpoint_steps=2, checkpoint_steps=2,
        collision_accepted_capacity=16, accepted_capacity=128,
        max_steps=32, max_orbit_steps=32,
        length_weight=0., curvature_weight=0., bdotn_weight=0., axis_field_weight=0.)
    options = SimpleNamespace(**settings)
    base, theta, _, xyz, pitch, objective = native.build_objective(options, objective_name)
    (value, (components, states)), gradient = jax.block_until_ready(
        jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.zeros_like(theta)))
    assert np.isfinite(value) and np.isfinite(gradient).all() and np.isfinite(states).all()
    assert np.linalg.norm(gradient) > 0 and components["mean_collision_accepted"] > 0
    assert components["safety_stop_fraction"] == 0
    assert states.shape == (2, 4, 5 if example is gc else 6)

    # Compare against the ordinary production forward API, not an alternate
    # collision formula or a benchmark-specific Gaussian sampler.
    field = BiotSavart(base)
    birth = Particles(xyz, initial_vparallel_over_v=pitch, energy=options.energy_ev*ONE_EV,
        mass=M, charge=Q, field=field if example is full else None,
        phase_angle_full_orbit=options.gyrophase)
    background = BackgroundSpecies(1, jnp.array([options.background_mass_proton]),
        jnp.array([options.background_charge_e]), jnp.array([options.background_density]),
        jnp.array([options.background_temperature_ev]), coulomb_log_model=options.coulomb_log_model)
    reference = trace_split(field, birth, background,
        model="GuidingCenter" if example is gc else "FullOrbit", tmax=options.horizon,
        timestep=options.collision_step, collision_dtmax=options.collision_dtmax,
        orbit_dt0=options.initial_step, n_save=options.saves,
        rtol=options.tolerance, atol=options.tolerance,
        collision_tolerance=options.collision_tolerance,
        event=native.toroidal_safety_stop(7.74, 4.4),
        max_steps=options.max_steps, max_orbit_steps=options.max_orbit_steps)
    np.testing.assert_array_equal(reference.status, [SplitStatus.SUCCESS]*2)
    # Avoid comparing quantities with SI velocity and magnetic-moment scales
    # against one shared absolute error.
    energy = options.energy_ev*ONE_EV
    speed = np.sqrt(2*energy/M)
    scales = np.array([1., 1., 1., speed, energy] if example is gc else [1.]*3+[speed]*3)
    assert np.max(np.abs(np.asarray(states-reference.ys))/scales) < 3e-13


@pytest.mark.parametrize("args", [
    ["--background-density", "-1"], ["--collision-step", "0"],
    ["--collision-spatial-diffusion"], ["--collision-accepted-capacity", "0"]])
def test_invalid_collision_settings_fail_before_output_creation(tmp_path, args):
    output = tmp_path / "invalid_collision"
    with pytest.raises(SystemExit):
        full.main(["--collisions", "--output", str(output), *args])
    assert not output.exists()
