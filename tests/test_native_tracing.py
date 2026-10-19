import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS
from essos.dynamics import Particles, Tracing


class UniformField:
    def B_contravariant(self, point):
        return jnp.array([0., 0., 1.])

    B_covariant = B_contravariant

    def AbsB(self, point):
        return 1.

    def dAbsB_by_dX(self, point):
        return jnp.zeros(3)

    curl_b = kappa = dAbsB_by_dX

    def sqrtg(self, point):
        return 1.

    def to_xyz(self, point):
        return point


@pytest.mark.parametrize("integrator, tolerance", [("diffrax", 1e-7), ("essos_rk45", 1e-9), ("essos_dopri8", 1e-9)])
def test_native_defaults_to_i_at_one_e_minus_nine(integrator, tolerance):
    particles = Particles(jnp.zeros((1, 3)), initial_vparallel_over_v=[.2], mass=1., charge=1., energy=.5)
    trace = Tracing(field=UniformField(), particles=particles, model="GuidingCenterAdaptative",
                    maxtime=.2, timestep=.1, times_to_trace=3, integrator=integrator, progress=False)
    assert trace.native_controller == "i" and trace.rtol == trace.atol == tolerance
    assert trace.native_dense_output == "standard"
    np.testing.assert_allclose(trace.trajectories[0, :, 2], .2 * trace.times, atol=1e-14)


@pytest.mark.parametrize("model", ["GuidingCenter", "GuidingCenterAdaptative", "FullOrbit", "FullOrbitAdaptative"])
@pytest.mark.parametrize("integrator", ["essos_rk45", "essos_dopri8"])
@pytest.mark.parametrize("controller", ["i", "pi"])
def test_uniform_orbits_match_analytic_and_diffrax(model, integrator, controller):
    particles = Particles(initial_xyz=jnp.zeros((2, 3)), mass=1., charge=1., energy=.5,
                          initial_vparallel_over_v=[.2, -.4], initial_xyz_fullorbit=jnp.zeros((2, 3)),
                          initial_vxvyvz=jnp.array([[.3, 0., .2], [.3, 0., -.4]]))
    options = dict(field=UniformField(), particles=particles, model=model, maxtime=10*np.pi,
                   timestep=.04, times_to_trace=43, atol=1e-11, rtol=1e-11, progress=False)
    native = Tracing(**options, integrator=integrator, native_controller=controller)
    reference = Tracing(**options, solver=diffrax.Dopri5() if integrator == "essos_rk45" else diffrax.Dopri8())
    t = np.asarray(native.times)
    expected = np.zeros(native.trajectories.shape)
    expected[:, :, 2] = np.array([.2, -.4])[:, None]*t
    if model.startswith("FullOrbit"):
        expected[:, :, 0] = .3*np.sin(t)
        expected[:, :, 1] = .3*(np.cos(t) - 1)
        expected[:, :, 3] = .3*np.cos(t)
        expected[:, :, 4] = -.3*np.sin(t)
        expected[:, :, 5] = np.array([.2, -.4])[:, None]
    else:
        expected[:, :, 3] = np.array([.2, -.4])[:, None]
    np.testing.assert_allclose(native.trajectories, expected, atol=4e-9, rtol=0.)
    np.testing.assert_allclose(native.trajectories, reference.trajectories, atol=4e-9, rtol=0.)
    assert native.integrator == integrator and reference.integrator == "diffrax"
    assert native.native_error_control == "physics"
    assert native.native_controller == controller
    children, metadata = native._tree_flatten()
    restored = Tracing._tree_unflatten(metadata, children)
    assert (restored.native_controller, restored.native_pcoeff, restored.native_icoeff) == (controller, .4, .3)


@pytest.mark.parametrize("model", ["GuidingCenter", "GuidingCenterAdaptative", "FullOrbit", "FullOrbitAdaptative"])
@pytest.mark.parametrize("integrator", ["essos_rk45", "essos_dopri8"])
@pytest.mark.parametrize("dense_output", ["standard", "optimized", "higher_order"])
def test_coil_field_matches_diffrax_dopri5_and_dopri8(model, integrator, dense_output):
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart

    field = BiotSavart(Coils(CreateEquallySpacedCurves(2, 1, 1., .4, 16, 2, True), [1e6, 1e6]))
    particles = Particles(initial_xyz=jnp.array([[.95, 0., 0.], [1.05, 0., .01]]),
                          initial_vparallel_over_v=[.3, -.6], mass=PROTON_MASS,
                          charge=ELEMENTARY_CHARGE, energy=5000*ONE_EV, field=field)
    options = dict(field=field, particles=particles, model=model, maxtime=2e-7,
                   timestep=2e-10 if model == "FullOrbit" else 1e-9,
                   times_to_trace=17, atol=1e-10, rtol=1e-10, progress=False)
    native = Tracing(**options, integrator=integrator, native_dense_output=dense_output)
    children, metadata = native._tree_flatten()
    assert Tracing._tree_unflatten(metadata, children).native_dense_output == dense_output
    assert jnp.isfinite(native.trajectories).all()
    for solver in (diffrax.Dopri5(), diffrax.Dopri8()):
        reference = Tracing(**options, solver=solver)
        np.testing.assert_allclose(native.trajectories[:, :, :3], reference.trajectories[:, :, :3],
                                   atol=2e-8, rtol=0., err_msg=type(solver).__name__)
        np.testing.assert_allclose(native.trajectories[:, :, 3:] / particles.total_speed,
                                   reference.trajectories[:, :, 3:] / particles.total_speed,
                                   atol=2e-8, rtol=0., err_msg=type(solver).__name__)


@pytest.mark.parametrize("integrator", ["essos_rk45", "essos_dopri8"])
def test_native_selection_bypasses_diffrax_solve_and_preserves_batched_events(monkeypatch, integrator):
    import essos.dynamics as dynamics

    def forbidden(*args, **kwargs):
        raise AssertionError("native selection called diffeqsolve")
    monkeypatch.setattr(dynamics, "diffeqsolve", forbidden)
    particles = Particles(jnp.zeros((3, 3)), initial_vparallel_over_v=[.2, .5, -.3], mass=1., charge=1., energy=.5)
    options = dict(field=UniformField(), particles=particles, model="GuidingCenterAdaptative",
                   maxtime=2., timestep=.07, times_to_trace=21, integrator=integrator,
                   stopping_criteria=lambda t, y, args, **kwargs: .3-y[2])
    whole = Tracing(**options)
    batched = Tracing(**options, particle_batch_size=2)
    np.testing.assert_array_equal(whole.boundary_hits, [True, True, False])
    np.testing.assert_array_equal(batched.boundary_hits, whole.boundary_hits)
    np.testing.assert_allclose(batched.trajectories, whole.trajectories, atol=1e-14)
    assert jnp.isfinite(whole.trajectories).all()
    assert jnp.max(whole.trajectories[:2, :, 2]) < .3


@pytest.mark.parametrize("options, message", [
    (dict(integrator="unknown"), "integrator"),
    (dict(native_controller="pid"), "native_controller"),
    (dict(native_dense_output="unknown"), "native_dense_output"),
    (dict(integrator="essos_rk45", model="FullOrbitCollisions"), "collisionless"),
    (dict(integrator="essos_rk45", model="FullOrbit_Boris"), "collisionless"),
    (dict(integrator="essos_rk45", model="FullOrbitAdaptative", solver=diffrax.Dopri5()), "solver"),
    (dict(integrator="essos_dopri8", model="GuidingCenterCollisions"), "collisionless"),
    (dict(integrator="essos_dopri8", model="GuidingCenterAdaptative", solver=diffrax.Dopri8()), "solver"),
])
def test_unsupported_integrator_configurations_raise(options, message):
    with pytest.raises(ValueError, match=message):
        Tracing(**options)


def test_invalid_native_error_policy_raises():
    with pytest.raises(ValueError, match="native_error_control"):
        Tracing(native_error_control="unknown")


@pytest.mark.parametrize("integrator", ["essos_rk45", "essos_dopri8"])
def test_standard_error_policy_remains_available(integrator):
    particles = Particles(jnp.zeros((1, 3)), initial_vparallel_over_v=[.2], mass=1., charge=1., energy=.5)
    trace = Tracing(field=UniformField(), particles=particles, model="GuidingCenterAdaptative",
                    maxtime=1., timestep=.1, times_to_trace=5, integrator=integrator,
                    native_error_control="standard")
    np.testing.assert_allclose(trace.trajectories[0, :, 2], .2 * trace.times, atol=1e-14)
    assert trace._native_error_norm() is None
