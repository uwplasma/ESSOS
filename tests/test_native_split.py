"""Forward split physics and bounded-control contracts; CPU-only CI."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from essos import split as s
from essos.background_species import BackgroundSpecies, nu_D_ab, nu_par_ab, nu_s_ab, coulomb_logarithm
from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q, EPSILON_0, HBAR, ONE_EV
from essos.dynamics import Particles
from essos.electric_field import Electric_field_zero
from tests.test_native_orbit import UniformField, coil_field, tokamak


def bath(**kwargs):
    return BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([kwargs.pop('density', 1e20)]), jnp.array([1000.]), **kwargs)


def zero(t, y, args):
    return jnp.zeros_like(y)


def solve(kick, **kwargs):
    options = dict(t1=1., dt0=.5, ts=jnp.array([0., 1.]), key=jax.random.PRNGKey(3),
                   noise_dim=1, method='dopri5', max_steps=64) | kwargs
    return s.solve_split(zero, kick, jnp.zeros(1), **options)


def test_brownian_conditioning_preserves_samples_and_covariance():
    key = jax.random.PRNGKey(9)
    memory = s.brownian_memory(key, 2, capacity=8)
    parent, _, dw, full = s.brownian_propose(memory, 1.)
    assert not full
    repeated, _, same, _ = s.brownian_propose(parent, 1.)
    np.testing.assert_array_equal(same, dw)
    np.testing.assert_array_equal(repeated.key, parent.key)
    child, h, left, full = s.brownian_propose(parent, .3, t_end=.3)
    assert not full and h == .3
    tail = s.brownian_accept(child)
    _, h, right, _ = s.brownian_propose(tail, 2.)
    assert h == pytest.approx(.7)
    np.testing.assert_allclose(left+right, dw, atol=2e-16)
    z = jnp.array([.2, -.3])
    conditioned, left, full = s.brownian_sample(parent, .4, normal=z)
    assert not full
    np.testing.assert_allclose(left, .4*dw+np.sqrt(.24)*z, atol=2e-16)
    jac = jax.jacfwd(lambda z: s.brownian_sample(parent, .4, normal=z)[1])(z)
    np.testing.assert_allclose(jac@jac.T, .24*np.eye(2), atol=2e-16)
    conditioned, whole, _ = s.brownian_sample(conditioned, 1.)
    np.testing.assert_allclose(whole, dw, atol=2e-16)
    tail = s.brownian_consume(conditioned, .4)
    _, right, _ = s.brownian_sample(tail, 1.)
    np.testing.assert_allclose(left+right, dw, atol=2e-16)


def test_brownian_capacity_and_cumulative_sampling_are_bounded():
    for change in (dict(capacity=0), dict(capacity=True), dict(noise_dim=0), dict(noise_dim=True)):
        options = dict(noise_dim=1, capacity=2) | change
        with pytest.raises(ValueError): s.brownian_memory(jax.random.PRNGKey(1), **options)
    memory, _, _, _ = s.brownian_propose(s.brownian_memory(jax.random.PRNGKey(1), 1, capacity=1), 1.)
    after, _, _, full = s.brownian_propose(memory, .5)
    assert full
    for a, b in zip(memory, after): np.testing.assert_array_equal(a, b)
    after, _, full = s.brownian_sample(memory, .5)
    assert full
    np.testing.assert_array_equal(after.increments, memory.increments)
    memory = s.brownian_memory(jax.random.PRNGKey(2), 1, capacity=4)
    memory, first, _ = s.brownian_sample(memory, .2)
    with pytest.raises(ValueError): s.brownian_sample(memory,.1,normal=jnp.zeros(2))
    memory, whole, _ = s.brownian_sample(memory, .7)
    consumed = s.brownian_consume(memory, .2)
    _, tail, _ = s.brownian_sample(consumed, .7)
    np.testing.assert_allclose(first+tail, whole, atol=2e-16)


def test_physical_coefficients_match_diffusion_divergence_and_zero_flux():
    background, point = bath(coulomb_log_model='impact'), jnp.zeros(3)
    for speed in (2e4, 2e6):
        velocity = jnp.array([.3, -.4, np.sqrt(.75)])*speed
        def diffusion(v):
            r = jnp.linalg.norm(v); direction = v/r
            dp = .5*r*r*nu_par_ab(M, Q, 0, r, point, background)
            dt = .5*r*r*nu_D_ab(M, Q, 0, r, point, background)
            return dt*jnp.eye(3)+(dp-dt)*jnp.outer(direction, direction)
        divergence = jnp.einsum('ijj->i', jax.jacfwd(diffusion)(velocity))
        expected = -nu_s_ab(M, Q, 0, speed, point, background)*velocity+divergence
        c = s.collision_coefficients(background, M, Q, jnp.asarray(speed), point, full_orbit=True)
        drift = c.friction+c.parallel_derivative+2*(c.parallel-c.perpendicular)/speed
        np.testing.assert_allclose(drift*velocity/speed, expected, rtol=2e-10)
        np.testing.assert_allclose(c.friction, -M*speed*c.parallel/(1000*Q), rtol=3e-14)
        basis = s.velocity_basis(velocity)
        noise = basis@jnp.diag(jnp.sqrt(2*jnp.array([c.perpendicular, c.perpendicular, c.parallel])))
        np.testing.assert_allclose(noise@noise.T/2, diffusion(velocity), rtol=2e-13)
    assert s.collision_coefficients(background, M, Q, jnp.asarray(1e5), point).particle_drift_derivative == 0


def test_collision_maps_reflections_and_physical_noise_covariance():
    c = s.CollisionCoefficients(.8, .4, .12, -.3, .07, .02)
    v, xi, h, tol = 2., -.4, .03, .004
    dw = jnp.array([.1, -.2])
    out = s.guiding_center_collision_step(v, xi, h, dw, c, tolerance=tol, return_error_model=True)
    nu = 2*c.perpendicular/v**2
    expected = [v+(-.3+.12+.8)*h+np.sqrt(1.6)*dw[0]+.06*(dw[0]**2-h),
        xi-xi*nu*h+np.sqrt((1-xi*xi)*nu)*dw[1]-.5*xi*nu*(dw[1]**2-h)]
    np.testing.assert_allclose(out.y, expected, rtol=2e-14)
    np.testing.assert_allclose(s.reflect_pitch(jnp.array([-5.2, -1.2, 1.2, 7.3])), [-.8, -.8, .8, -.7], atol=2e-15)
    for map_fn, initial, noise in ((s.guiding_center_collision_step, (2., .3), jnp.zeros(2)),
                                   (s.particle_collision_step, (jnp.array([1.2, 0., 1.6]),), jnp.zeros(3))):
        vacuum = s.CollisionCoefficients(*(0.,)*6)
        result = map_fn(*initial, h, noise, vacuum)
        np.testing.assert_allclose(result.y, jnp.asarray(initial if len(initial)==2 else initial[0]))
        assert result.drift_error == result.diffusion_error == 0
    basis = s.velocity_basis(jnp.array([1., 2., 3.]))
    np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=3e-16)
    point = jnp.array([0., 0., 1.]); w = jnp.array([.2, -.4, 3.])
    dx = s.guiding_center_spatial_diffusion(xi, c, 7.)
    np.testing.assert_allclose(s.guiding_center_spatial_increment(xi, w, c, point, 7.), np.sqrt(2*dx)*w*jnp.array([1., 1., 0.]))
    for value in (0., 1., np.nan):
        result = s._diffusion_root(jnp.asarray(value))
        assert np.isnan(result) if np.isnan(value) else result == np.sqrt(value)
    assert jax.grad(s._diffusion_root)(0.) == 0


def test_error_models_match_actual_estimates_and_candidate_bounds():
    for gc in (False, True):
        for vacuum in (False, True):
            c = s.CollisionCoefficients(*((0.,)*6 if vacuum else (.8, .4, .12, -.3, .07, .02)))
            dw = jnp.array([.1, -.2] if gc else [.1, -.2, .3])
            out = (s.guiding_center_collision_step(2., -.4, .03, dw, c, tolerance=.004, return_error_model=True)
                   if gc else s.particle_collision_step(jnp.array([1.2, 0., 1.6]), .03, dw, c, tolerance=.004, return_error_model=True))
            np.testing.assert_allclose(s.collision_error_estimates(out.error_model, .03, dw), out[1:3], rtol=3e-14)
            error = float(max(out.drift_error, out.diffusion_error))
            for target in (.9, 1., error/2+1e-14, 2*error+1e-14):
                assert bool(s.collision_error_passes(out.error_model, .03, dw, target=target)) == (error <= target)
            h = s._fit_error_step(out.error_model, 1., 0., jnp.ones_like(dw), .9)
            estimates = s.collision_error_estimates(out.error_model, h, jnp.ones_like(dw)*jnp.sqrt(h))
            assert max(estimates) <= .9+1e-12
    model = s.CollisionErrorModel(.1, 0., 0., .1, 0., 0., .004, jnp.array(0))
    assert s.collision_error_passes(model, .03, jnp.zeros(1))
    assert not s.collision_error_passes(model, .03, jnp.ones(1))


def test_controller_lookahead_retains_probes_and_respects_boundaries():
    memory = s.brownian_memory(jax.random.PRNGKey(8), 1, capacity=8)
    for error, expected in ((.25, 1.5), (4., .45)):
        after, h, full = s.paper_collision_step_size(memory, 1., jnp.array([2.]), error,
            1e-18, jnp.asarray(error <= 1), dtmin=0., dtmax=10., barrier=10.)
        assert not full and h == pytest.approx(expected) and after.size > 0
        _, h, full = s.paper_collision_step_size(memory, 1., jnp.zeros(1), error,
            0., jnp.asarray(error <= 1), dtmin=.4, dtmax=10., barrier=.2)
        assert not full and h == .2
    for drift in (0., 1.):
        model = s.CollisionErrorModel(0., .5, drift, 0., 0., 0., 1., jnp.array(0))
        after, h, full = s.candidate_collision_step_size(memory, .5, model, jnp.array(True), dtmin=0., dtmax=2., barrier=1.)
        assert not full and after.size == 0 and abs(h-(.9**(2/3) if drift else 1.)) < 1/256
    model = s.CollisionErrorModel(0., .5, 0., .3, 0., 0., .2, jnp.array(0))
    parent, _, _, _ = s.brownian_propose(s.brownian_memory(jax.random.PRNGKey(3), 1, capacity=1), 1.)
    after, _, full = s.candidate_collision_step_size(parent, 1., model, jnp.array(False), dtmin=0., dtmax=1., barrier=1.)
    assert full
    np.testing.assert_array_equal(after.increments, parent.increments)


def test_impact_cutoffs_screening_and_vacuum_are_explicit():
    from essos.background_species import debye_inverse_squared, impact_parameters, impact_coulomb_logarithm
    for ma, qa, mb, qb in ((M, Q, M, Q), (4*M, 2*Q, 12*M, 6*Q)):
        v = 1e5; reduced = ma*mb/(ma+mb)
        bmin, bmax = impact_parameters(ma, qa, mb, qb, v, 0., 1e-4)
        np.testing.assert_allclose(bmin, np.hypot(abs(qa*qb)/(4*np.pi*EPSILON_0*reduced*v*v), HBAR/(2*reduced*v)), rtol=3e-14)
        assert bmax == 1e-4
    inverse = debye_inverse_squared(jnp.array([1e20, 0.]), jnp.array([1000., 0.]), jnp.array([Q, -Q]))
    assert inverse == pytest.approx(1e20*Q/(EPSILON_0*1000.))
    assert np.isnan(debye_inverse_squared(jnp.array([-1.]), jnp.ones(1), jnp.ones(1)))
    with pytest.raises(ValueError): bath(coulomb_log_model='bad')
    with pytest.raises(ValueError): bath(screening_species=bath())
    screen = bath(coulomb_log_model='impact')
    physical = bath(coulomb_log_model='impact', screening_species=screen)
    log = lambda v: coulomb_logarithm(M, Q, 0, v, jnp.zeros(3), physical)
    assert np.isfinite(log(0.)) and jax.grad(log)(0.) == 0
    vacuum = bath(density=0., coulomb_log_model='impact')
    assert coulomb_logarithm(M, Q, 0, 1e5, jnp.zeros(3), vacuum) == 0
    np.testing.assert_array_equal(s.collision_coefficients(vacuum, M, Q, jnp.asarray(1e5), jnp.zeros(3)), np.zeros(6))
    assert np.isnan(impact_coulomb_logarithm(M, Q, M, Q, 100., 1000., jnp.array([1e40]), jnp.array([1000.]), jnp.array([Q])))


def test_fixed_split_clocks_saves_and_multiple_orbit_steps():
    collision = lambda t, y, h, dw, a: s.CollisionStep(y+h, 0., 0.)
    for saves in (jnp.array([]), jnp.array([0., .1, .1, .4, 1.])):
        out = jax.jit(lambda: s.solve_split(lambda t,y,a:jnp.ones_like(y), collision, jnp.zeros(1),
            t1=1., dt0=.25, orbit_dt0=.05, adaptive=False, orbit_adaptive=False, ts=saves,
            key=jax.random.PRNGKey(1), noise_dim=1, record_steps=True))()
        assert out.status == 0 and out.stats.orbit_steps > 3*out.stats.accepted
        np.testing.assert_allclose(out.y, [2.], atol=2e-14)
        np.testing.assert_allclose(out.ys[:, 0], 2*saves, atol=2e-14)
        assert out.step_ts[0] == 0 and out.step_ts[out.stats.accepted] == 1


def test_adaptive_rejection_keeps_the_original_wiener_path():
    model = s.CollisionErrorModel(0., .5, .5, .3, 0., 0., .2, jnp.array(0))
    for controller in ('conservative', 'paper', 'candidate'):
        def one(key):
            def kick(t, y, h, dw, a):
                ed, es = s.collision_error_estimates(model, h, dw)
                return s.CollisionStep(y+dw, ed, es, model)
            out = s.solve_split(zero, kick, jnp.zeros(1), t1=1., dt0=1., dtmax=1., ts=jnp.array([1.]),
                key=key, noise_dim=1, method='dopri5', collision_controller=controller, max_steps=256)
            _, draw = jax.random.split(key)
            return out, jax.random.normal(draw, (1,), dtype=jnp.float64)
        out, parent = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(25), 8))
        assert np.all(out.status == 0) and np.all(out.stats.rejected > 0)
        assert np.max(out.stats.max_accepted_error) <= 1
        np.testing.assert_allclose(out.y, parent, atol=1e-12, rtol=0)


def test_orbit_events_skip_kicks_and_refinement_retains_fixed_anchors():
    def bad_kick(t, y, h, dw, a): return s.CollisionStep(jnp.full_like(y, jnp.nan), 0., 0.)
    out = s.solve_split(lambda t,y,a:jnp.ones_like(y), bad_kick, jnp.zeros(1),
        t1=1., dt0=.5, ts=jnp.array([0., .1, 1.]), key=jax.random.PRNGKey(1), noise_dim=1,
        event=(lambda t,y,a:.05-y[0], lambda t,y,a:.2-y[0]), method='dopri5')
    assert out.status == 1 and out.stats.accepted == 0
    np.testing.assert_allclose(out.t, .05, atol=2e-14)
    np.testing.assert_allclose(out.ys[:, 0], [0., .05, .05], atol=2e-14)
    kick = lambda t,y,h,w,a:s.CollisionStep(y+h, 0., 0.)
    out = solve(kick, adaptive=False, orbit_adaptive=False, event=lambda t,y,a:.31-y[0],
        step_limit=lambda t,y,a:jnp.where(y[0] < .2, .2, .1), event_dtmax=.025, record_steps=True)
    assert out.status == 1 and .31 <= float(out.y[0]) <= .335
    assert out.stats.rejected > 0


def test_input_validation_and_failure_statuses_keep_last_finite_state():
    kick = lambda t,y,h,w,a:s.CollisionStep(y+h, 0., 0.)
    for change in (dict(max_steps=0), dict(max_orbit_steps=True), dict(collision_controller='bad'),
                   dict(orbit_stage_execution='bad'), dict(dtmax=-1.), dict(orbit_dt0=0.),
                   dict(dtmin=1.), dict(event_dtmax=.1), dict(t0=jnp.ones(2))):
        with pytest.raises(ValueError): solve(kick, **change)
    for bad in (lambda t,y,h,w,a:s.CollisionStep(jnp.zeros(2), 0., 0.),
                lambda t,y,h,w,a:s.CollisionStep(y, jnp.zeros(2), 0.)):
        with pytest.raises(ValueError): solve(bad)
    with pytest.raises(ValueError): solve(kick, event=lambda t,y,a:jnp.zeros(2))
    with pytest.raises(ValueError): solve(kick, collision_controller='candidate')
    model = s.CollisionErrorModel(0., 0., 0., 0., 0., 0., jnp.ones(2), jnp.array(0))
    with pytest.raises(ValueError):
        solve(lambda t,y,h,w,a:s.CollisionStep(y, 0., 0., model), collision_controller='candidate')
    bad = solve(lambda t,y,h,w,a:s.CollisionStep(jnp.full_like(y, jnp.nan), 0., 0.))
    assert bad.status == s.SplitStatus.NONFINITE
    np.testing.assert_array_equal(bad.y, [0.])
    minimum = solve(lambda t,y,h,w,a:s.CollisionStep(y+1., 4., 0.), dtmin=.5)
    assert minimum.status == s.SplitStatus.MIN_STEP and minimum.stats.accepted == 0
    limited = s.solve_split(zero, kick, jnp.zeros(1), t1=1., dt0=.1, ts=jnp.array([1.]),
        key=jax.random.PRNGKey(1), noise_dim=1, adaptive=False, max_steps=1)
    assert limited.status == s.SplitStatus.MAX_STEPS
    invalid = solve(kick, step_limit=lambda t,y,a:jnp.asarray(-1.))
    assert invalid.status == s.SplitStatus.INVALID_INPUT
    for limit in (lambda t,y,a:jnp.ones(2),):
        with pytest.raises(ValueError): solve(kick, step_limit=limit)
    for times in (jnp.ones((1, 1)),):
        with pytest.raises(ValueError): s.solve_split(zero, kick, jnp.zeros(1), t1=1., dt0=.1,
            ts=times, key=jax.random.PRNGKey(1), noise_dim=1)
    for initial in (jnp.zeros((1,1)), jnp.array([],dtype=float), jnp.array([1+1j])):
        with pytest.raises(ValueError): s.solve_split(zero, kick, initial, t1=1., dt0=.1,
            ts=jnp.array([1.]), key=jax.random.PRNGKey(1), noise_dim=1)


def test_cartesian_gc_and_full_orbit_use_actual_collision_maps():
    field, background = UniformField(), bath(density=0.)
    birth = Particles(jnp.array([[1., 0., 0.]]), mass=M, charge=Q, energy=10*ONE_EV,
        initial_vparallel_over_v=jnp.array([.4]), field=field, phase_angle_full_orbit=0.)
    for model in ('GuidingCenter', 'FullOrbit'):
        for integrator in ('essos_rk45', 'essos_dopri8'):
            out = s.trace_split(field, birth, background, model=model, integrator=integrator,
                tmax=1e-10, timestep=5e-11, orbit_dt0=1e-11, n_save=2,
                adaptive=False, orbit_adaptive=False, speed_floor=1., spatial_diffusion=model=='GuidingCenter')
            assert out.status[0] == 0 and out.stats.orbit_steps[0] > out.stats.accepted[0]
            assert np.isfinite(out.ys).all()
        out = s.trace_split(field, birth, bath(), model=model, tmax=1e-10,
            timestep=1e-10, n_save=2, orbit=False)
        assert out.status[0] == 0
        for controller in ('paper', 'candidate'):
            out = s.trace_split(field, birth, bath(), model=model, tmax=1e-10,
                timestep=5e-11, n_save=3, orbit=False, collision_controller=controller)
            assert out.status[0] == 0
    for change in (dict(integrator='bad'), dict(model='bad'), dict(collision_tolerance=0.),
                   dict(n_save=1), dict(execution='bad'), dict(spatial_diffusion=True, model='FullOrbit'),
                   dict(speed_floor=-1.), dict(times=jnp.zeros((1,1)))):
        with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-10, timestep=1e-10, **change)
    with pytest.raises(ValueError): s.trace_split(field, birth, None, tmax=1e-10, timestep=1e-10)
    with pytest.raises(ValueError): s.trace_split(field, birth, SimpleNamespace(number_species=0), tmax=1e-10, timestep=1e-10)
    for parameter, value in (('mass', -1.), ('charge', 0.), ('initial_vparallel_over_v', jnp.array([2.]))):
        original = getattr(birth, parameter); setattr(birth, parameter, value)
        with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-10, timestep=1e-10)
        setattr(birth, parameter, original)
    birth.initial_xyz_fullorbit = None
    with pytest.raises(ValueError): s.trace_split(field, birth, bath(), model='FullOrbit', tmax=1e-10, timestep=1e-10)


def test_coordinate_gc_charts_preserve_mu_and_enforce_supported_physics():
    from pathlib import Path
    from essos.fields import Vmec
    vmec = Vmec(str(Path(__file__).resolve().parents[1]/'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'), ntheta=8, nphi=8)
    for field in (vmec, tokamak()):
        birth = Particles(jnp.array([[.3, .2, .1]]), mass=M, charge=Q, energy=10*ONE_EV,
            initial_vparallel_over_v=jnp.array([.4]))
        out = s.trace_split(field, birth, bath(density=0.), tmax=1e-11,
            timestep=1e-11, n_save=2, event=lambda t,y,a:1.-y[0])
        assert out.status[0] == 0
        np.testing.assert_allclose(out.ys[0, 0, 4], out.y[0, 4], rtol=2e-14)
        for change in (dict(model='FullOrbit'), dict(spatial_diffusion=True)):
            with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-11, timestep=1e-11, **change)
    birth.initial_xyz = jnp.array([[-.1, .2, .1]])
    with pytest.raises(ValueError): s.trace_split(tokamak(), birth, bath(), tmax=1e-11, timestep=1e-11)
    with pytest.raises(ValueError): s.trace_split(tokamak(), birth, bath(), tmax=1e-11, timestep=1e-11, electric_field=object())


def test_independent_execution_validates_capabilities_without_fallback(monkeypatch):
    field = coil_field()
    birth = Particles(jnp.array([[1., .03, .01]]), mass=s.M, charge=s.Q, energy=4000*ONE_EV,
        initial_vparallel_over_v=jnp.array([.5]))
    options = dict(model='GuidingCenter', integrator='essos_dopri8', adaptive=True,
        orbit_adaptive=True, controller='i', collision_controller='conservative',
        electric_field=Electric_field_zero(), spatial_diffusion=False, orbit=True,
        record_steps=False, reuse_orbit_step=True, event=None, times=jnp.array([0.,1.]), tmax=1.)
    with pytest.raises(ValueError): s._validate_independent(field, birth, bath(), **options)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    s._validate_independent(field, birth, bath(), **options)
    for change in (dict(integrator='essos_rk45'), dict(controller='pi'), dict(collision_controller='paper'),
                   dict(orbit_adaptive=False), dict(reuse_orbit_step=False), dict(spatial_diffusion=True),
                   dict(orbit=False), dict(electric_field=object()), dict(record_steps=True),
                   dict(times=jnp.array([])), dict(event=lambda t,y,a:True),
                   dict(event=lambda t,y,a:jnp.ones(2)), dict(event=lambda t,y,a:1.-t)):
        with pytest.raises(ValueError): s._validate_independent(field, birth, bath(), **(options|change))
    for field_, species in ((UniformField(), bath()), (field, bath(coulomb_log_model='impact')),
                             (field, SimpleNamespace(number_species=2))):
        with pytest.raises(ValueError): s._validate_independent(field_, birth, species, **options)
    birth.mass = M
    with pytest.raises(ValueError): s._validate_independent(field, birth, bath(), **options)
    birth.mass = s.M
    old = birth.random_keys; birth.random_keys = jax.random.key(0, impl='rbg')[None]
    with pytest.raises(ValueError): s._validate_independent(field, birth, bath(), **options)
    birth.random_keys = old
    s._validate_independent(field, birth, bath(), **(options|dict(event=lambda t,y,a:1.)))
    with pytest.raises(ValueError):
        jax.eval_shape(lambda mass:s._validate_independent(field,
            SimpleNamespace(mass=mass,charge=s.Q,random_keys=old),bath(),**options),jnp.asarray(s.M))
    original = s._prepare_split_kernel
    monkeypatch.setattr(s,'_prepare_split_kernel',lambda *a,**k:original(*a,**(k|dict(interpret=True))))
    monkeypatch.setattr(s,'_validate_independent',lambda *a,**k:None)
    out = s.trace_split(field,birth,bath(),tmax=1e-12,timestep=1e-12,n_save=2,
        adaptive=False,orbit_adaptive=False,execution='independent',max_steps=4,max_orbit_steps=4)
    assert out.status[0] == 0


def kernel_problem(*, adaptive=False, event=False, fail=False, limit=32, pending=8):
    def factory(refs, y0):
        z = s.scalar(y0, 7)
        def rhs(y): return jnp.where(jnp.arange(8)==0, z+1., z)
        def kick(y,h,w):
            result = y+jnp.where(jnp.arange(8)==3, s.scalar(w, 0), z)
            return result, (h/.04)**2 if adaptive else z, z
        norm = lambda y,e,v,b:z
        level = (lambda y:.05-s.scalar(y,0)) if event else lambda y:z+1.
        return rhs,kick,norm,level
    return s._prepare_split_kernel((),factory,jnp.array([0.,.1,.2]),adaptive=adaptive,
        dt0=.1,dtmax=.1,speed=1.,orbit_dt0=.025,max_steps=limit,max_orbit_steps=16,
        max_pending=pending,interpret=True)


def test_independent_kernel_interpretation_keeps_noise_and_inner_counts():
    initial, key = jnp.zeros(8), jax.random.PRNGKey(3)
    for adaptive in (False, True):
        out = jax.jit(kernel_problem(adaptive=adaptive))(initial,key)
        ys,y,t,stats,status = out
        assert status == 0 and t == .2 and stats[3] > stats[1]
        np.testing.assert_allclose(ys[:,0], [0.,.1,.2], atol=3e-15)
        if not adaptive:
            assert stats[0] == stats[1] == 2 and stats[3] == 8
        else:
            assert stats[2] > 0 and stats[6] <= 1
    _,y,t,stats,status = jax.jit(kernel_problem(event=True))(initial,key)
    assert status == 1 and stats[1] == 0
    np.testing.assert_allclose(t,.05,atol=2e-14)
    _,_,_,_,status = jax.jit(kernel_problem(limit=1))(initial,key)
    assert status == 2
    _,_,_,_,status = jax.jit(kernel_problem(adaptive=True,pending=1))(initial,key)
    assert status == 6


def test_private_coil_field_and_collision_maps_match_public_functions():
    from essos.dynamics import _guiding_center_velocity
    field, background = coil_field(), bath()
    def check_kick(kick, initial, magnitude, gc=True):
        h = jnp.asarray(1e-9)
        dw = jnp.asarray([.2,-.3] if gc else [.2,-.3,.4])*jnp.sqrt(h)
        actual,ed,es = jax.jit(kick)(initial,h,dw)
        speed = jnp.sqrt(initial[3]**2+2*initial[4]*magnitude/s.M) if gc else jnp.linalg.norm(initial[3:6])
        c = s.collision_coefficients(background,s.M,s.Q,speed,jnp.zeros(3),full_orbit=not gc)
        if gc:
            step = s.guiding_center_collision_step(speed,initial[3]/speed,h,dw,c,tolerance=.003,speed_floor=1.)
            v,xi = step.y
            expected = initial.at[3].set(v*xi).at[4].set(s.M*v*v*(1-xi*xi)/(2*magnitude))
        else:
            step = s.particle_collision_step(initial[3:6],h,dw,c,tolerance=.003)
            expected = initial.at[3:6].set(step.y)
        np.testing.assert_allclose(actual,expected,rtol=3e-10,atol=1e-25)
        np.testing.assert_allclose([ed,es],[step.drift_error,step.diffusion_error],rtol=3e-9,atol=1e-25)
    position = jnp.array([1., .03, .01]); vpar = 1e5; moment = s.M*2e10/(2*field.AbsB(position))
    for gc in (True, False):
        initial = jnp.r_[position,jnp.array([vpar,moment,0.,0.,0.])] if gc else jnp.r_[position,jnp.array([1e5,2e5,1e5,0.,0.])]
        arrays,factory = s._prepare_split_field(field,background,gc=gc,collision_tolerance=.003,speed_floor=1.)
        rhs,kick,norm,level = factory(arrays,initial)
        value = jax.jit(rhs)(initial)
        if gc:
            dx,dv = _guiding_center_velocity(field,Electric_field_zero(),s.Q,s.M,position,vpar,mu=moment)
            expected = jnp.r_[dx,jnp.array([dv,0.,0.,0.,0.])]
        else:
            expected = jnp.r_[initial[3:6],s.Q/s.M*jnp.cross(initial[3:6],field.B_contravariant(position)),jnp.zeros(2)]
        np.testing.assert_allclose(value,expected,rtol=3e-10,atol=1e-7)
        changed,ed,es = jax.jit(kick)(initial,jnp.asarray(1e-9),jnp.zeros(2 if gc else 3))
        assert np.isfinite(changed).all() and ed >= 0 and es >= 0
        check_kick(kick,initial,field.AbsB(position),gc)
        assert level(initial) > 0 and norm(initial,jnp.ones(8),4e5,1e-9) > 0
    with pytest.raises(ValueError): s._split_field_data(SimpleNamespace(table=True,sine_coef=None))
    from pathlib import Path
    from essos.fields import Vmec
    from essos.dynamics import _to_axis_regular, _axis_regular
    from essos.boozer import guiding_center_rhs
    vmec = Vmec(str(Path(__file__).resolve().parents[1]/'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),ntheta=3,nphi=3)
    for field in (vmec,tokamak()):
        point = jnp.array([.3,.2,.1])
        B = field.AbsB(point) if isinstance(field,Vmec) else field.modB(*point)
        moment = s.M*2e10/(2*B)
        physical = jnp.r_[point,jnp.array([1e5,moment])]
        initial = (_to_axis_regular(physical) if isinstance(field,Vmec) else
            physical.at[0].set(jnp.sqrt(point[0])*jnp.cos(point[1])).at[1].set(jnp.sqrt(point[0])*jnp.sin(point[1])))
        nstate = initial.size; initial = jnp.pad(initial,(0,8-nstate))
        event = lambda t,y,a:4.-jnp.sum((y[:2]-jnp.array([.1,.2]))**2)
        arrays,factory = s._prepare_split_field(field,background,gc=True,collision_tolerance=.003,
            speed_floor=1.,state_size=nstate,event=event)
        rhs,kick,norm,level = factory(arrays,initial)
        value = jax.jit(rhs)(initial)
        if isinstance(field,Vmec):
            def prepare_inside_jit(y):
                arrays,factory = s._prepare_split_field(field,background,gc=True,
                    collision_tolerance=.003,speed_floor=1.,state_size=nstate)
                return factory(arrays,y)[0](y)
            np.testing.assert_allclose(jax.jit(prepare_inside_jit)(initial),value,rtol=3e-12,atol=1e-6)
            def public_rhs(t,y,a):
                dx,dv = _guiding_center_velocity(field,Electric_field_zero(),s.Q,s.M,y[:3],y[3],mu=y[4])
                return jnp.r_[dx,jnp.array([dv,0.])]
            expected = jnp.pad(_axis_regular(public_rhs)(0.,initial[:nstate],None),(0,8-nstate))
        else:
            expected = jnp.pad(guiding_center_rhs(field,initial[:4],moment/s.M,s.M,s.Q),(0,4))
        np.testing.assert_allclose(value,expected,rtol=3e-10,atol=1e-6)
        changed,ed,es = jax.jit(kick)(initial,jnp.asarray(1e-9),jnp.zeros(2))
        assert np.isfinite(changed).all() and ed >= 0 and es >= 0
        check_kick(kick,initial,B)
        assert np.isfinite(norm(initial,jnp.ones(8)*1e-12,4e5,1e-9))
        np.testing.assert_allclose(level(initial),event(0.,initial,None),atol=2e-15)
    matrix = jnp.eye(2)
    with pytest.raises(ValueError):
        s._prepare_split_field(coil_field(),background,gc=True,collision_tolerance=.003,
            speed_floor=1.,event=lambda t,y,a:jnp.sum(matrix@y[:2]))


def test_gpu_sampler_uses_the_production_threefry_normal_stream():
    key = jax.random.PRNGKey(19)
    for dim in (2,3):
        next_key, normal = jax.jit(lambda key:s._draw_normal(key,dim))(key)
        expected_key, draw = jax.random.split(key)
        expected = jax.random.normal(draw,(1<<(dim-1).bit_length(),),dtype=jnp.float64)
        np.testing.assert_array_equal(next_key,expected_key)
        np.testing.assert_allclose(normal,expected,rtol=2e-14,atol=2e-14)
    original = jax.config.jax_threefry_partitionable
    jax.config.update('jax_threefry_partitionable',False)
    try:
        with pytest.raises(ValueError): s._draw_normal(key,2)
    finally:
        jax.config.update('jax_threefry_partitionable',original)


def test_scalar_lowering_preserves_constants_slices_and_surface_values():
    x = jnp.array([.2,-.3,.4,.5,0.,0.,0.,0.])
    functions = (lambda x:x[1:2],lambda x:jax.lax.dynamic_slice(x,(x[0].astype(int),),(1,)),
        lambda x:jnp.sin(x)+jnp.cos(x),lambda x:jax.lax.erf(x),
        lambda x:jnp.array([jnp.sum(x),jnp.max(x),jnp.min(x)]),
        lambda x:jnp.stack([x]),lambda x:x[jnp.array([2])],
        lambda x:(jnp.any(x>0),jnp.all(x<1)),lambda x:tuple(x),
        lambda x:tuple(jnp.split(x,len(x))))
    for function in functions:
        graph = s._flatten(function,x)
        actual = jax.jit(lambda x:s._evaluate_kernel(graph.jaxpr,graph.consts,x))(x)
        for a,b in zip(actual,jax.tree.leaves(function(x))): np.testing.assert_allclose(a,b,atol=2e-15)
    for function in (lambda x:jax.lax.dynamic_update_slice(x,x+1,(0,)),lambda x:jnp.pad(x,(1,0))):
        small=jnp.array([.3]); graph=s._flatten(function,small)
        actual=s._evaluate_kernel(graph.jaxpr,graph.consts,small,scalarize=True)[0]
        np.testing.assert_allclose(actual,function(small),atol=2e-15)
    from jax._src.lax.lax import _convert_element_type
    graph=s._flatten(lambda x:_convert_element_type(x,np.dtype('float64'),weak_type=True),x)
    np.testing.assert_allclose(jax.jit(lambda x:s._evaluate_kernel(graph.jaxpr,graph.consts,x)[0])(x),x)
    for function in (lambda y:y[1:3],lambda y:jax.lax.dynamic_slice(y,(y[0].astype(int),),(2,)),
                     lambda y:y[jnp.array([1,2])],lambda y:jnp.concatenate((y,y,y))):
        graph=s._flatten(function,x)
        with pytest.raises(ValueError): s._evaluate_kernel(graph.jaxpr,graph.consts,x)
    for function in (lambda y:1-jnp.sum(y[:3]**2),lambda y:jnp.dot(y[:3],y[:3])-.1,
                     lambda y:jnp.max(y[:3])-jnp.min(y[:3]),lambda y:y[jnp.array([1])][0],
                     lambda y:sum(y[:3]),lambda y:jnp.sum(jnp.array([y[0],y[1]])),
                     lambda y:y[:1].reshape(())):
        graph = s._flatten(function,x)
        np.testing.assert_allclose(s._evaluate_surface(graph.jaxpr,graph.consts,x),function(x),atol=2e-15)
    flag = s._split_select(jnp.array([True,False]),1,0)
    assert not flag[0]


def test_independent_driver_rejects_invalid_work_limits():
    for name,value in (('max_steps',0),('max_orbit_steps',False),('max_pending',0)):
        options = dict(max_steps=8,max_orbit_steps=8,max_pending=8) | {name:value}
        with pytest.raises(ValueError):
            s._solve_independent(None,None,jnp.zeros(5),field=None,species=None,gc=True,
                birth_speed=1.,speed_floor=.1,collision_tolerance=.003,**options)
