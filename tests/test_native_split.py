"""Split orbit/collision solver: physics, numerics and execution-equivalence contracts.

Collision maps are checked against the Coulomb Fokker-Planck generator (Lorentz
pitch-angle moments, Maxwellian stationarity, Einstein relation), the Brownian
bridge against its conditional law, and the Lie splitting against its exact
first-order weak/strong errors. The persistent GPU kernel is checked on CPU through
the Pallas interpreter against shared JAX execution. CPU-only CI.
"""
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import trapezoid
from essos import split as s
from essos.background_species import BackgroundSpecies, nu_D_ab, nu_par_ab, nu_s_ab, coulomb_logarithm
from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q, EPSILON_0, HBAR, ONE_EV
from essos.dynamics import Particles
from essos.electric_field import Electric_field_zero
from tests.test_native_orbit import UniformField, coil_field, tokamak

VMEC_FILE = Path(__file__).resolve().parents[1]/'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'


def bath(**kwargs):
    return BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([kwargs.pop('density', 1e20)]), jnp.array([1000.]), **kwargs)


def zero(t, y, args):
    return jnp.zeros_like(y)


def solve(kick, **kwargs):
    options = dict(t1=1., dt0=.5, ts=jnp.array([0., 1.]), key=jax.random.PRNGKey(3),
                   noise_dim=1, method='dopri5', max_steps=64) | kwargs
    return s.solve_split(zero, kick, jnp.zeros(1), **options)


def expectation(function, h, dim, order=8):
    """E[function(dW)] for dW ~ N(0, h I) by Gauss-Hermite (exact for low-order polynomials)."""
    x, w = np.polynomial.hermite_e.hermegauss(order)
    w = w/np.sqrt(2*np.pi)
    nodes = np.stack(np.meshgrid(*[x]*dim, indexing='ij'), -1).reshape(-1, dim)
    weights = np.prod(np.stack(np.meshgrid(*[w]*dim, indexing='ij'), -1).reshape(-1, dim), axis=1)
    values = jax.vmap(lambda z: function(jnp.sqrt(h)*z))(nodes)
    return jnp.tensordot(weights, values, axes=1)


# Brownian memory and collision coefficients

def test_brownian_bridge_follows_its_conditional_law_with_bounded_memory():
    memory = s.brownian_memory(jax.random.PRNGKey(9), 2, capacity=8)
    parent, _, dw, full = s.brownian_propose(memory, 1.)
    repeated, _, same, _ = s.brownian_propose(parent, 1.)
    assert not full
    np.testing.assert_array_equal(same, dw)                    # a retry keeps its parent increment
    np.testing.assert_array_equal(repeated.key, parent.key)
    child, h, left, _ = s.brownian_propose(parent, .3, t_end=.3)
    assert h == .3
    _, h, right, _ = s.brownian_propose(s.brownian_accept(child), 2.)
    assert h == pytest.approx(.7)
    np.testing.assert_allclose(left+right, dw, atol=2e-16)    # sub-increments always sum to the parent

    def split(key):
        parent = s.brownian_propose(s.brownian_memory(key, 1, capacity=4), 1.)[0]
        child, _, left, _ = s.brownian_propose(parent, .3, t_end=.3)
        right = s.brownian_propose(s.brownian_accept(child), 1.)[2]
        return left[0], right[0]
    left, right = jax.jit(jax.vmap(split))(jax.random.split(jax.random.PRNGKey(1), 40000))
    tolerance = 5*np.sqrt(2/40000)
    assert abs(np.var(left)/.3-1) < tolerance and abs(np.var(right)/.7-1) < tolerance
    assert abs(np.corrcoef(left, right)[0, 1]) < 5/np.sqrt(40000)       # disjoint increments are independent

    memory = s.brownian_memory(jax.random.PRNGKey(1), 1, capacity=1)
    memory, _, _, _ = s.brownian_propose(memory, 1.)
    after, _, _, full = s.brownian_propose(memory, .5)         # no room to split: reported, nothing lost
    assert full
    for a, b in zip(memory, after):
        np.testing.assert_array_equal(a, b)
    for change in (dict(capacity=0), dict(capacity=True), dict(noise_dim=0), dict(noise_dim=True)):
        with pytest.raises(ValueError):
            s.brownian_memory(jax.random.PRNGKey(1), **(dict(noise_dim=1, capacity=2) | change))


def test_coefficients_obey_einstein_relation_and_divergence_form():
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
        # Ito drift of the Cartesian velocity is friction plus the divergence of D.
        drift = c.friction+c.parallel_derivative+2*(c.parallel-c.perpendicular)/speed
        np.testing.assert_allclose(drift*velocity/speed, expected, rtol=2e-10)
        np.testing.assert_allclose(c.friction, -M*speed*c.parallel/(1000*Q), rtol=3e-14)   # fluctuation-dissipation
        noise = s.velocity_basis(velocity)@jnp.diag(jnp.sqrt(2*jnp.array([c.perpendicular, c.perpendicular, c.parallel])))
        np.testing.assert_allclose(noise@noise.T/2, diffusion(velocity), rtol=1e-12)
    assert s.collision_coefficients(background, M, Q, jnp.asarray(1e5), point).particle_drift_derivative == 0


def test_maxwellian_is_stationary_for_the_speed_generator():
    """E_Maxwell[L g] = 0 for L = A d/dv + D_par d^2/dv^2 (zero flux of the equilibrium)."""
    background = bath(coulomb_log_model='impact')
    sigma = np.sqrt(1000*Q/M)
    r = np.linspace(1e-3, 10., 20001); v = r*sigma
    c = jax.vmap(lambda u: s.collision_coefficients(background, M, Q, u, jnp.zeros(3)))(jnp.asarray(v))
    drift = np.asarray(c.friction+c.parallel_derivative+2*c.parallel/v); D = np.asarray(c.parallel)
    weight = r*r*np.exp(-r*r/2)
    for g1, g2 in ((2*v, 2.+0*v), (4*v**3, 12*v**2), (np.exp(-r)*(-1/sigma), np.exp(-r)/sigma**2)):
        scale = trapezoid(weight*(abs(drift*g1)+abs(D*g2)), r)
        assert abs(trapezoid(weight*(drift*g1+D*g2), r)) < 2e-5*scale


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


# Collision maps against the Fokker-Planck generator

def test_guiding_center_map_has_lorentz_and_speed_generator_moments():
    c = s.CollisionCoefficients(.8, .4, .12, -.3, .07, .02)
    v, xi = 2., -.4
    nu = 2*c.perpendicular/v**2                                  # pitch-angle scattering rate
    drift = c.friction+c.parallel_derivative+2*c.parallel/v      # Ito speed drift
    kick = lambda h, p: lambda dw: (lambda y: jnp.array([y[0], y[1], y[0]**2, y[1]**2]))(
        s.guiding_center_collision_step(v, p, h, dw, c).y)
    residual = []
    for h in (.02, .01):
        mean = expectation(kick(h, xi), h, 2)
        np.testing.assert_allclose(mean[0]-v, drift*h, rtol=1e-11)                 # E[dv] = A h
        np.testing.assert_allclose(mean[1]-xi, -nu*xi*h, rtol=1e-11)               # Lorentz: E[dxi] = -nu xi h
        # E[dxi^2] -> nu(1-3xi^2) h is the l=2 Legendre eigenvalue, 3 nu; errors are O(h^2).
        residual.append([mean[3]-xi**2-nu*(1-3*xi**2)*h, mean[2]-v**2-(2*v*drift+2*c.parallel)*h])
    np.testing.assert_allclose(np.asarray(residual[0])/np.asarray(residual[1]), 4., rtol=2e-2)
    # Pure pitch scattering through the split driver: the exact Legendre relaxation
    # <xi> = xi0 exp(-nu t), <xi^2> = 1/3 + (xi0^2 - 1/3) exp(-3 nu t) up to O(h) and sampling error.
    scatter = s.CollisionCoefficients(0., .5, 0., 0., 0., 0.)           # v = 1, nu = 1
    horizon, steps = .5, 50

    def relax(key):
        return s.solve_split(zero, lambda t, y, h, dw, a: s.guiding_center_collision_step(y[0], y[1], h, dw, scatter),
            jnp.array([1., xi]), t1=horizon, dt0=horizon/steps, adaptive=False, orbit_adaptive=False, ts=jnp.array([horizon]),
            key=key, noise_dim=2, method='dopri5', max_steps=64).y[1]
    final = np.asarray(jax.jit(jax.vmap(relax))(jax.random.split(jax.random.PRNGKey(2), 40000)))
    assert abs(final.mean()-xi*np.exp(-horizon)) < 6*final.std()/np.sqrt(len(final))+3e-3
    assert abs((final**2).mean()-(1/3+(xi**2-1/3)*np.exp(-3*horizon))) < 6*(final**2).std()/np.sqrt(len(final))+3e-3
    np.testing.assert_allclose(s.reflect_pitch(jnp.array([-5.2, -1.2, 1.2, 7.3])), [-.8, -.8, .8, -.7], atol=2e-15)
    # Vacuum is a fixed point with zero error, differentiably (no 0*inf from the noise root).
    vacuum = s.CollisionCoefficients(*(0.,)*6)
    for fixed in (lambda x: s.guiding_center_collision_step(x, .3, .03, jnp.zeros(2), vacuum),
                  lambda x: s.particle_collision_step(jnp.array([1.2, 0., 1.6])*x, .03, jnp.zeros(3), vacuum)):
        result = fixed(2.)
        assert result.drift_error == result.diffusion_error == 0
        assert np.isfinite(jax.jacfwd(lambda x: fixed(x).y)(2.)).all()
    basis = s.velocity_basis(jnp.array([1., 2., 3.]))
    np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=3e-16)


def test_particle_map_and_spatial_kick_have_the_diffusion_tensor_covariance():
    c = s.CollisionCoefficients(.8, .4, .12, -.3, .07, .02)
    velocity = jnp.array([1.2, 0., 1.6]); speed = jnp.linalg.norm(velocity); direction = velocity/speed
    drift = c.friction+c.parallel_derivative+2*(c.parallel-c.perpendicular)/speed
    h = .002
    delta = lambda dw: s.particle_collision_step(velocity, h, dw, c).y-velocity
    mean = expectation(delta, h, 3, order=6)
    cov = expectation(lambda dw: jnp.outer(delta(dw), delta(dw)), h, 3, order=6)
    np.testing.assert_allclose(mean, drift*h*direction, rtol=1e-11, atol=1e-17)
    np.testing.assert_allclose(cov, 2*h*(c.perpendicular*jnp.eye(3)+(c.parallel-c.perpendicular)*jnp.outer(direction, direction)),
                               rtol=5*h, atol=1e-9)
    b = jnp.array([0., 0., 1.]); omega = 7.
    dx = s.guiding_center_spatial_diffusion(-.4, c, omega)
    spatial = lambda dw: s.guiding_center_spatial_increment(-.4, dw, c, b, omega)
    np.testing.assert_allclose(expectation(lambda dw: jnp.outer(spatial(dw), spatial(dw)), h, 3),
                               2*dx*h*jnp.diag(jnp.array([1., 1., 0.])), rtol=1e-11, atol=1e-18)


# Splitting and the adaptive driver

def test_lie_splitting_is_first_order_weakly_and_strongly():
    # Deterministic: y' = a y (orbit) and y += b h (kick) have exact solution; Lie error ~ h.
    a, b, T = -.7, 1.3, 2.
    exact = np.exp(a*T)*.5+b*(np.exp(a*T)-1)/a
    errors = []
    for n in (8, 16, 32):
        out = s.solve_split(lambda t, y, args: a*y, lambda t, y, h, dw, args: s.CollisionStep(y+b*h, 0., 0.),
            jnp.array([.5]), t1=T, dt0=T/n, adaptive=False, orbit_adaptive=False, orbit_dt0=T/n/4,
            method='dopri8', ts=jnp.array([T]), key=jax.random.PRNGKey(0), noise_dim=1, max_steps=64)
        errors.append(abs(float(out.y[0])-exact))
    np.testing.assert_allclose(np.array(errors[:-1])/np.array(errors[1:]), 2., rtol=.1)
    # Stochastic Brownian flight (dx = v dt, dv = dW): the split variance is exactly
    # T^3/3 (1 - 3/(2N) + 1/(2N^2)), a weak error of order h.
    def flight(key, n):
        out = s.solve_split(lambda t, y, args: jnp.array([y[1], 0.]),
            lambda t, y, h, dw, args: s.CollisionStep(y.at[1].add(dw[0]), 0., 0.), jnp.zeros(2),
            t1=1., dt0=1./n, adaptive=False, orbit_adaptive=False, method='dopri5', ts=jnp.array([1.]),
            key=key, noise_dim=1, max_steps=64)
        return out.y
    keys = jax.random.split(jax.random.PRNGKey(5), 20000)
    for n in (4, 16):
        y = np.asarray(jax.jit(jax.vmap(lambda k: flight(k, n)))(keys))
        np.testing.assert_allclose(np.var(y[:, 0])*3, 1-1.5/n+.5/n**2, atol=6*np.sqrt(2/20000)/1.0)
        np.testing.assert_allclose(np.var(y[:, 1]), 1., atol=6*np.sqrt(2/20000))


def test_adaptive_rejection_preserves_the_wiener_measure():
    # Rejecting large increments and refining them through Brownian bridges must not bias
    # the path law: the final value stays N(0, T).
    def one(key):
        def kick(t, y, h, dw, a):
            return s.CollisionStep(y+dw, (jnp.abs(dw[0])/.5)**3, 0.)
        return s.solve_split(zero, kick, jnp.zeros(1), t1=1., dt0=1., dtmax=1., ts=jnp.array([1.]),
            key=key, noise_dim=1, method='dopri5', max_steps=256)
    out = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(25), 8000))
    assert np.all(out.status == 0) and np.mean(out.stats.rejected) > 1 and np.max(out.stats.max_accepted_error) <= 1
    y = np.asarray(out.y[:, 0])
    assert abs(y.mean()) < 5/np.sqrt(8000) and abs(y.var()-1) < 5*np.sqrt(2/8000)
    assert np.max(out.stats.peak_pending) >= 2


def test_events_steps_and_failures_keep_the_last_finite_state():
    kick = lambda t, y, h, w, a: s.CollisionStep(y+h, 0., 0.)
    out = s.solve_split(lambda t, y, a: jnp.ones_like(y), lambda t, y, h, dw, a: s.CollisionStep(jnp.full_like(y, jnp.nan), 0., 0.),
        jnp.zeros(1), t1=1., dt0=.5, ts=jnp.array([0., .1, 1.]), key=jax.random.PRNGKey(1), noise_dim=1,
        event=(lambda t, y, a: .05-y[0], lambda t, y, a: .2-y[0]), method='dopri5')
    assert out.status == s.SplitStatus.EVENT and out.stats.accepted == 0      # orbit crossing is root-found
    np.testing.assert_allclose(out.t, .05, atol=2e-14)
    np.testing.assert_allclose(out.ys[:, 0], [0., .05, .05], atol=2e-14)
    out = solve(kick, adaptive=False, orbit_adaptive=False, event=lambda t, y, a: .31-y[0],
        step_limit=lambda t, y, a: jnp.where(y[0] < .2, .2, .1), event_dtmax=.025, record_steps=True)
    assert out.status == s.SplitStatus.EVENT and .31 <= float(out.y[0]) <= .335 and out.stats.rejected > 0
    nan = solve(lambda t, y, h, w, a: s.CollisionStep(jnp.full_like(y, jnp.nan), 0., 0.))
    assert nan.status == s.SplitStatus.NONFINITE and nan.y[0] == 0
    stuck = solve(lambda t, y, h, w, a: s.CollisionStep(y+1., 4., 0.), dtmin=.5)
    assert stuck.status == s.SplitStatus.MIN_STEP and stuck.stats.accepted == 0
    assert solve(kick, max_steps=1, dt0=.1, adaptive=False).status == s.SplitStatus.MAX_STEPS
    assert solve(kick, step_limit=lambda t, y, a: jnp.asarray(-1.)).status == s.SplitStatus.INVALID_INPUT
    # A full Brownian memory stops the run instead of dropping a sampled increment.
    out = solve(lambda t, y, h, w, a: s.CollisionStep(y+w, (h/.15)**2, 0.), max_pending=1)
    assert out.status == s.SplitStatus.BROWNIAN_CAPACITY and out.y[0] == 0
    # Saves are exact boundaries and the orbit takes several RK steps per kick.
    for saves in (jnp.array([]), jnp.array([0., .1, .1, .4, 1.])):
        out = jax.jit(lambda: s.solve_split(lambda t, y, a: jnp.ones_like(y), lambda t, y, h, w, a: s.CollisionStep(y+h, 0., 0.),
            jnp.zeros(1), t1=1., dt0=.25, orbit_dt0=.05, adaptive=False, orbit_adaptive=False, ts=saves,
            key=jax.random.PRNGKey(1), noise_dim=1, record_steps=True))()
        assert out.status == 0 and out.stats.orbit_steps > 3*out.stats.accepted
        np.testing.assert_allclose(out.y, [2.], atol=2e-14)
        np.testing.assert_allclose(out.ys[:, 0], 2*saves, atol=2e-14)
        assert out.step_ts[0] == 0 and out.step_ts[out.stats.accepted] == 1
    # Endpoint alignment avoids a tiny trailing RK interval without changing the result.
    options = dict(t1=.5, dt0=.105, dtmax=.105, orbit_dt0=.1, ts=jnp.array([0., .5]), key=jax.random.PRNGKey(42),
                   noise_dim=1, step_limit=lambda t, y, args: jnp.asarray(.105), max_steps=20, max_orbit_steps=20)
    run = lambda **kw: s.solve_split(lambda t, y, a: jnp.ones_like(y), lambda t, y, h, w, a: s.CollisionStep(y, 0., 0.),
                                     jnp.zeros(1), **options, **kw)
    plain, aligned = run(), run(align_orbit=True)
    np.testing.assert_allclose(aligned.ys, plain.ys, atol=2e-14)
    assert aligned.status == plain.status == 0 and aligned.stats.orbit_steps < plain.stats.orbit_steps


def test_solver_arguments_are_validated():
    kick = lambda t, y, h, w, a: s.CollisionStep(y+h, 0., 0.)
    for change in (dict(max_steps=0), dict(max_orbit_steps=True), dict(orbit_stage_execution='bad'), dict(dtmax=-1.),
                   dict(orbit_dt0=0.), dict(dtmin=1.), dict(event_dtmax=.1), dict(t0=jnp.ones(2)),
                   dict(align_orbit=True, adaptive=False), dict(ts=jnp.ones((1, 1))),
                   dict(event=lambda t, y, a: jnp.zeros(2)), dict(step_limit=lambda t, y, a: jnp.ones(2))):
        with pytest.raises(ValueError): solve(kick, **change)
    for bad in (lambda t, y, h, w, a: s.CollisionStep(jnp.zeros(2), 0., 0.),
                lambda t, y, h, w, a: s.CollisionStep(y, jnp.zeros(2), 0.)):
        with pytest.raises(ValueError): solve(bad)
    for initial in (jnp.zeros((1, 1)), jnp.array([], dtype=float), jnp.array([1+1j])):
        with pytest.raises(ValueError):
            s.solve_split(zero, kick, initial, t1=1., dt0=.1, ts=jnp.array([1.]), key=jax.random.PRNGKey(1), noise_dim=1)


# trace_split

def test_cartesian_gc_and_full_orbit_trace_with_the_actual_maps():
    field, background = UniformField(), bath(density=0.)
    birth = Particles(jnp.array([[1., 0., 0.]]), mass=M, charge=Q, energy=10*ONE_EV,
        initial_vparallel_over_v=jnp.array([.4]), field=field, phase_angle_full_orbit=0.)
    for model, integrator in (('GuidingCenter', 'essos_rk45'), ('GuidingCenter', 'essos_dopri8'), ('FullOrbit', 'essos_dopri8')):
        out = s.trace_split(field, birth, background, model=model, integrator=integrator,
            tmax=1e-10, timestep=5e-11, orbit_dt0=1e-11, n_save=2, adaptive=False, orbit_adaptive=False,
            speed_floor=1., spatial_diffusion=model == 'GuidingCenter')
        assert out.status[0] == 0 and out.stats.orbit_steps[0] > out.stats.accepted[0] and np.isfinite(out.ys).all()
    for model in ('GuidingCenter', 'FullOrbit'):
        out = s.trace_split(field, birth, bath(), model=model, tmax=1e-10, timestep=5e-11, n_save=3, orbit=False)
        assert out.status[0] == 0 and out.stats.accepted[0] >= 2          # adaptive moments controller
        limited = s.trace_split(field, birth, background, model=model, tmax=1e-10, timestep=1e-10,
            n_save=2, step_limit=lambda t, y, a: 2.5e-11)
        assert limited.status[0] == 0 and limited.stats.accepted[0] == 4
    assert s.trace_split(field, birth, bath(), tmax=1e-10, timestep=5e-11, n_save=3, orbit=False,
                         collision_controller='conservative').status[0] == 0
    for change in (dict(integrator='bad'), dict(model='bad'), dict(collision_controller='paper'), dict(collision_tolerance=0.),
                   dict(n_save=1), dict(execution='bad'), dict(spatial_diffusion=True, model='FullOrbit'),
                   dict(speed_floor=-1.), dict(times=jnp.zeros((1, 1)))):
        with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-10, timestep=1e-10, **change)
    for change in (dict(step_limit=lambda t, y, a: 1., execution='independent'),):
        with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-10, timestep=1e-10, **change)
    with pytest.raises(ValueError): s.trace_split(field, birth, None, tmax=1e-10, timestep=1e-10)
    with pytest.raises(ValueError): s.trace_split(field, birth, SimpleNamespace(number_species=0), tmax=1e-10, timestep=1e-10)
    for parameter, value in (('mass', -1.), ('charge', 0.), ('initial_vparallel_over_v', jnp.array([2.]))):
        original = getattr(birth, parameter); setattr(birth, parameter, value)
        with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-10, timestep=1e-10)
        setattr(birth, parameter, original)
    birth.initial_xyz_fullorbit = None
    with pytest.raises(ValueError): s.trace_split(field, birth, bath(), model='FullOrbit', tmax=1e-10, timestep=1e-10)


def test_coordinate_charts_invert_the_cartesian_kick_and_check_geometry():
    point = jnp.array([.4, .3, .2, 1., 2.]); dx = jnp.array([.02, -.03, .01])
    position = lambda y: jnp.array([y[0]+.2*y[1]**2, y[1], 2*y[2]])
    moved = jax.jit(lambda y, d: s.move_in_cartesian(y, d, position))(point, dx)
    np.testing.assert_allclose(position(moved), position(point)+dx, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(moved[3:], point[3:])
    np.testing.assert_allclose(jax.jacfwd(lambda d: s.move_in_cartesian(point, d, position)[:3])(dx),
        jnp.linalg.inv(jax.jacfwd(lambda q: position(moved.at[:3].set(q)))(moved[:3])), rtol=2e-13)
    np.testing.assert_array_equal(s.move_in_cartesian(point, jnp.zeros(3), position), point)
    noise = jnp.concatenate((jnp.eye(3), -jnp.eye(3)))*.01      # Ito drift of the chart: E[dx] = -.2 E[dy^2]
    ensemble = jax.vmap(lambda d: s.move_in_cartesian(point, d, position))(noise)
    np.testing.assert_allclose(ensemble[:, 0].mean()-point[0], -.2*jnp.mean(noise[:, 1]**2), atol=2e-14)
    assert np.isnan(s.move_in_cartesian(point, dx, lambda y: jnp.ones(3))[:3]).all()
    # Circular torus and toroidal-shift sign; the axis chart stays differentiable.
    radial = np.array([.1, .3, .5]); radius = .2*np.sqrt(radial)
    export = SimpleNamespace(s_b=radial, xm_b=np.array([0, 1]), xn_b=np.zeros(2), rmnc_b=np.stack((np.full(3, 2.), radius)),
        zmns_b=np.stack((np.zeros(3), radius)), numns_b=np.zeros((2, 3)))
    geometry = s.BoozerGeometry.from_booz_xform(export)
    q = jnp.array([.3, .4, .2]); R = 2+.2*jnp.sqrt(q[0])*jnp.cos(q[1])
    np.testing.assert_allclose(geometry(q), [R*jnp.cos(q[2]), R*jnp.sin(q[2]), .2*jnp.sqrt(q[0])*jnp.sin(q[1])], atol=1e-14)
    assert np.isfinite(jax.jacfwd(geometry.regular_xyz)(jnp.array([0., 0., .2]))).all()
    shifted = SimpleNamespace(**vars(export)); shifted.numns_b = np.stack((np.zeros(3), .05*np.sqrt(radial)))
    phi = q[2]-.05*jnp.sqrt(q[0])*jnp.sin(q[1])
    np.testing.assert_allclose(s.BoozerGeometry.from_booz_xform(shifted)(q),
        [R*jnp.cos(phi), R*jnp.sin(phi), .2*jnp.sqrt(q[0])*jnp.sin(q[1])], atol=1e-14)
    with pytest.raises(ValueError): s.BoozerGeometry.from_booz_xform(SimpleNamespace(s_b=radial, xm_b=np.array([0, 1])))


def boozer_geometry():
    """Circular torus (R=5, a=.2) exported like booz_xform's geometry spectra."""
    surfaces = np.array([.1, .3, .5, .9])
    return s.BoozerGeometry.from_booz_xform(SimpleNamespace(s_b=surfaces, xm_b=np.array([0, 1]), xn_b=np.zeros(2),
        rmnc_b=np.stack((np.full(4, 5.), .2*np.sqrt(surfaces))), zmns_b=np.stack((np.zeros(4), .2*np.sqrt(surfaces))),
        numns_b=np.zeros((2, 4))))


def test_vmec_and_boozer_spatial_kicks_equal_the_cartesian_formula():
    from essos.fields import Vmec
    vmec = Vmec(str(VMEC_FILE), ntheta=8, nphi=8)
    geometry = boozer_geometry()
    for field, embedding in ((vmec, None), (tokamak(), geometry)):
        birth = Particles(jnp.array([[.3, .2, .1]]), mass=M, charge=Q, energy=10*ONE_EV, initial_vparallel_over_v=jnp.array([.4]))
        out = s.trace_split(field, birth, bath(density=0.), tmax=1e-11, timestep=1e-11, n_save=2,
                            event=lambda t, y, a: 1.-y[0], spatial_diffusion=False, collision_controller='conservative')
        assert out.status[0] == 0
        np.testing.assert_allclose(out.ys[0, 0, 4], out.y[0, 4], rtol=2e-14)       # mu is preserved
        if field is not vmec:
            with pytest.raises(ValueError, match='spatial_geometry'):
                s.trace_split(field, birth, bath(), tmax=1e-11, timestep=1e-11)
        for change in (dict(model='FullOrbit'), dict(execution='independent')):
            with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-11, timestep=1e-11, **change)
        assert out.stats.accepted[0] > 0
        output = s.trace_split(field, birth, bath(), tmax=1e-11, timestep=1e-11, n_save=2, spatial_geometry=embedding,
                               orbit=False, collision_controller='conservative')
        q = birth.initial_xyz[0]; mapping = field.to_xyz if field is vmec else embedding
        B = field.AbsB(q) if field is vmec else field.modB(*q)
        speed = birth.initial_vparallel[0]/.4; c = s.collision_coefficients(bath(), M, Q, speed, q)
        _, _, dw, _ = s.brownian_propose(s.brownian_memory(birth.random_keys[0], 5), 1e-11)
        tangent = (jax.jacfwd(mapping)(q)@field.B_contravariant(q)) if field is vmec else (
            jax.jacfwd(mapping)(q)@jnp.array([0., field.profiles(q[0])[0][0], 1.]))
        dx = s.guiding_center_spatial_increment(.4, dw[:3], c, tangent/jnp.linalg.norm(tangent), Q*B/M)
        np.testing.assert_allclose(mapping(output.y[0, :3]), mapping(q)+dx, rtol=0, atol=3e-13)
        expected = s.guiding_center_collision_step(speed, .4, 1e-11, dw[-2:], c, speed_floor=.05*np.sqrt(2*1000*Q/M)).y
        newB = field.AbsB(output.y[0, :3]) if field is vmec else field.modB(*output.y[0, :3])
        actual = jnp.sqrt(output.y[0, 3]**2+2*output.y[0, 4]*newB/M)
        np.testing.assert_allclose([actual, output.y[0, 3]/actual], expected, rtol=2e-13)
    with pytest.raises(ValueError): s.trace_split(tokamak(), birth, bath(), tmax=1e-11, timestep=1e-11, spatial_geometry=geometry, electric_field=object())
    birth.initial_xyz = jnp.array([[-.1, .2, .1]])
    with pytest.raises(ValueError): s.trace_split(tokamak(), birth, bath(), tmax=1e-11, timestep=1e-11)


# Moment predictor

def dense_budget(velocity, jet, vel, tau):
    """Independent dense-matrix form of expanded_moment_budget (mean/covariance defects)."""
    f0, F, H = velocity; a, g, aj, ah, qj, qh, xa, xq = jet; q = g*g
    n = f0.size; J = jnp.zeros((n, n)).at[:, -2:].set(F)
    G = jnp.zeros((n, 2)).at[-2:, :].set(jnp.diag(g)); Q_ = G@G.T
    mean = -F@a-.5*(H@q)
    mean = mean.at[-2:].add(aj@f0[-2:]+xa)
    cov = jnp.zeros((n, n)).at[-2:, -2:].set(jnp.diag(qj@f0[-2:]+xq))-J@Q_-Q_@J.T
    mean = mean.at[-2:].add(-(aj@a+.5*jnp.einsum('ijj,j->i', ah, q)))
    lq = qj@a+.5*jnp.einsum('ijj,j->i', qh, q)
    cov = cov.at[-2:, -2:].add(jnp.diag(.25*jnp.diag(qj)**2-lq)-aj@jnp.diag(q)-jnp.diag(q)@aj.T)
    B = (J@G).at[-2:, :].set(0.)
    cubic = -B@B.T/3
    b = mean+.5*tau*(J@mean)
    transport = lambda A: A+.5*tau*(J@A+A@J.T)+(tau*tau/3)*(J@A@J.T)
    C, D = transport(cov), transport(cubic)
    P = jnp.concatenate((jnp.eye(n), jnp.zeros((1, n)).at[0, -2:].set(vel[::-1])), axis=0)
    bp = (P@b).at[-1].add(C[-2, -1])
    Gp, Bp = P@G, P@(J@G)
    spread = tau*jnp.sum((Gp+.5*tau*Bp)**2, axis=1)+tau**3/12*jnp.sum(Bp*Bp, axis=1)
    scale = jnp.sqrt(jnp.maximum(spread, 1e-28))
    pair = lambda A: jnp.abs(A)/(scale[:, None]*scale[None, :])
    return (jnp.maximum(jnp.max(.5*tau*jnp.abs(bp)/scale), jnp.max(.5*tau*pair(P@C@P.T))),
            jnp.max(tau*pair(P@D@P.T)))


def test_moment_budget_reproduces_the_brownian_flight_splitting_defect():
    # dx = v dt, dv = g dW: Lie splitting underestimates Var(x) by 3h/(2T) (see the
    # splitting test) and the budget's linear coefficient is that rate, 1.5/tau.
    zero2 = jnp.zeros(2)
    jet = s.CollisionJet(zero2, jnp.array([2., 0.]), jnp.zeros((2, 2)), jnp.zeros((2, 2, 2)),
                         jnp.zeros((2, 2)), jnp.zeros((2, 2, 2)), zero2, zero2)
    vel = jnp.array([2., .5])
    velocity = (jnp.array([2., 0, 0, 0, 0]), jnp.zeros((5, 2)).at[0, 0].set(1.), jnp.zeros((5, 2)))
    linear, quadratic = s.expanded_moment_budget(velocity, jet, vel, .005)
    np.testing.assert_allclose([linear, quadratic], [1.5/.005, 1/.005**2], rtol=2e-14)
    h = s.interval_from_budget(linear, quadratic, .01)
    assert linear*h+quadratic*h*h == pytest.approx(.008)
    assert s.interval_from_budget(0., 0., .01) == .0025
    # Frozen angular diffusion has exact Legendre moments: local mean/variance defects.
    rate, pitch = 2.3, .6
    ang = s.collision_jet(lambda r, x: jnp.array([0., 0., jnp.sqrt(rate)]), jnp.array(3.), jnp.array(pitch), jnp.zeros(3), jnp.zeros(3))
    np.testing.assert_allclose(ang.drift, [0., -pitch*rate])
    np.testing.assert_allclose(ang.noise[1], np.sqrt((1-pitch**2)*rate))
    # The expanded algebra equals the dense matrix form, including zero/tiny noise and both pitch endpoints.
    rng = np.random.default_rng(42)
    noises = jnp.concatenate((jnp.array([0., 1e-25, 1.])[:, None]*jnp.array([.4, .3]), jnp.asarray(rng.uniform(.1, 1., (8, 2)))))
    jacobians = jnp.asarray(rng.normal(size=(11, 5, 2))); hessians = jnp.asarray(rng.normal(size=(11, 5, 2)))
    velocities = jnp.stack((jnp.full(11, 2.), jnp.linspace(-1., 1., 11)), axis=1)
    rich = s.CollisionJet(jnp.array([.3, -.2]), zero2, jnp.eye(2)*.1, jnp.ones((2, 2, 2))*.01, jnp.ones((2, 2))*.02,
                          jnp.ones((2, 2, 2))*.002, jnp.array([.001, .002]), jnp.array([.003, .004]))

    def compare(g, F, Hh, v):
        model = rich._replace(noise=g); jets = (jnp.array([0., 0, 1.8, 0, 0]), F, Hh)
        return jnp.asarray(dense_budget(jets, model, v, .005)), jnp.asarray(s.expanded_moment_budget(jets, model, v, .005))
    before, after = jax.jit(jax.vmap(compare))(noises, jacobians, hessians, velocities)
    np.testing.assert_allclose(before, after, rtol=3e-12, atol=3e-12)


def test_analytic_velocity_jets_equal_automatic_differentiation():
    from essos.fields import Vmec
    from essos.dynamics import _axis_regular, _from_axis_regular, _guiding_center_velocity
    background = bath()
    vmec = Vmec(str(VMEC_FILE), ntheta=3, nphi=3)
    point = jnp.array([.3, .2, .1]); moment = M*2e10/(2*vmec.AbsB(point))
    state = jnp.r_[point, jnp.array([1e5, moment])]

    def public_rhs(t, y, a):
        dx, dv = _guiding_center_velocity(vmec, Electric_field_zero(), Q, M, y[:3], y[3], mu=y[4])
        return jnp.r_[dx, jnp.array([dv, 0.])]
    from essos.dynamics import _to_axis_regular
    regular = _to_axis_regular(state)
    settings = dict(gc=True, coordinate=True, tmax=.01, dtmax=.01, tolerance=.005)

    def predictor(electric):
        return s.native_step_limit(vmec, background, M, Q, regular, _axis_regular(public_rhs), _from_axis_regular,
            lambda y: vmec.to_xyz(_from_axis_regular(y)[:3]), vmec.AbsB, electric_field=electric, **settings)
    # The built-in zero field takes the exact quotient derivatives; a custom zero-E object
    # follows the generic autodiff route. Both give the same interval.
    fast, reference = predictor(Electric_field_zero()), predictor(SimpleNamespace(E_covariant=lambda point: jnp.zeros(3)))
    limits = jax.jit(lambda y: jnp.array([fast(0., y, None), reference(0., y, None)]))(regular)
    assert 0 < float(limits[0]) < settings['dtmax']
    np.testing.assert_allclose(limits[0], limits[1], rtol=2e-10)
    # Same check for a Cartesian field with spatially varying profiles (convective collision jet).
    mirror = _Mirror(2., .03)
    varying = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([[.2e20, 1.8e20]]), jnp.array([[500., 1500.]]),
        radial_grid=jnp.array([-.1, .1]), coulomb_log_model='impact')
    y0 = jnp.array([.004, -.003, .02, 3e5, M*4e10/(2*mirror.AbsB(jnp.array([.004, -.003, .02])))])

    def rhs(t, y, a):
        dx, dv = _guiding_center_velocity(mirror, Electric_field_zero(), Q, M, y[:3], y[3], mu=y[4])
        return jnp.r_[dx, dv, 0.]
    cartesian = lambda electric: s.native_step_limit(mirror, varying, M, Q, y0, rhs, lambda y: y, lambda y: y[:3], mirror.AbsB,
        gc=True, electric_field=electric, tmax=1e-3, dtmax=1e-3)
    a = cartesian(Electric_field_zero())(1e-4, y0, None)
    b = cartesian(SimpleNamespace(E_covariant=lambda point: jnp.zeros(3)))(1e-4, y0, None)
    assert 0 < float(a) <= 1e-3
    np.testing.assert_allclose(a, b, rtol=2e-10)


class _Mirror:
    """B/B0=(-a x z, -a y z, 1+a z^2), a=1/L^2; div B=0."""
    def __init__(self, magnitude, length):
        self.magnitude, self.a = magnitude, 1/length**2

    def B_contravariant(self, p):
        return self.magnitude*jnp.array([-self.a*p[0]*p[2], -self.a*p[1]*p[2], 1+self.a*p[2]**2])
    B_covariant = B_contravariant

    def AbsB(self, p): return jnp.linalg.norm(self.B_contravariant(p))
    def sqrtg(self, p): return 1.

    def gc_quantities(self, p):
        B = self.B_contravariant(p); value = jnp.linalg.norm(B); b = B/value
        jac = jax.jacfwd(self.B_contravariant)(p)
        gradient = jac.T@b; db = (jac-b[:, None]*gradient[None, :])/value
        curl = jnp.array([db[2, 1]-db[1, 2], db[0, 2]-db[2, 0], db[1, 0]-db[0, 1]])
        return B, B, value, gradient, curl, db@b, 1.

    def dAbsB_by_dX(self, p): return self.gc_quantities(p)[3]
    def curl_b(self, p): return self.gc_quantities(p)[4]
    def kappa(self, p): return self.gc_quantities(p)[5]


# Independent GPU kernel, interpreted on CPU, against shared JAX execution

def run_pair(field, background, model='GuidingCenter', **kwargs):
    options = dict(model=model, tmax=2e-8, timestep=1e-8, n_save=3, max_steps=64, max_orbit_steps=64) | kwargs
    if hasattr(field, 'coils'):
        birth = Particles(jnp.array([[1., .03, .01], [1., .02, .0]]), mass=M, charge=Q, energy=4000*ONE_EV,
                          initial_vparallel_over_v=jnp.array([.5, .3]), field=field, phase_angle_full_orbit=0.)
    else:
        birth = Particles(jnp.array([[.3, .2, .1]]), mass=M, charge=Q, energy=4000*ONE_EV, initial_vparallel_over_v=jnp.array([.4]))
    shared = s.trace_split(field, birth, background, **options)
    independent = s.trace_split(field, birth, background, execution='independent', interpret=True, **options)
    return shared, independent


def assert_equivalent(shared, independent):
    assert np.all(shared.status == independent.status) and np.all(shared.status == 0)
    np.testing.assert_allclose(independent.ys, shared.ys, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(independent.y, shared.y, rtol=1e-11, atol=1e-14)
    for name in ('steps', 'accepted', 'rejected', 'orbit_steps', 'accepted_orbit_steps'):
        np.testing.assert_array_equal(getattr(independent.stats, name), getattr(shared.stats, name))


@pytest.mark.parametrize('model,adaptive', [('GuidingCenter', False), ('GuidingCenter', True), ('FullOrbit', True)])
def test_independent_kernel_matches_shared_execution_in_coil_fields(model, adaptive):
    options = dict(adaptive=adaptive, orbit_adaptive=adaptive, spatial_diffusion=model == 'GuidingCenter')
    assert_equivalent(*run_pair(coil_field(), bath(), model, **options))


def test_independent_kernel_matches_shared_execution_in_vmec_and_boozer_fields():
    from essos.fields import Vmec
    vmec = Vmec(str(VMEC_FILE), ntheta=3, nphi=3)
    assert_equivalent(*run_pair(vmec, bath(), adaptive=False, orbit_adaptive=False, spatial_diffusion=False))
    # Adaptive Boozer with the moment predictor and nonlinear Cartesian spatial kicks.
    assert_equivalent(*run_pair(tokamak(), bath(), adaptive=True, orbit_adaptive=True, spatial_diffusion=True,
        spatial_geometry=boozer_geometry()))


def test_independent_kernel_handles_radial_profiles_events_and_failures():
    varying = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([[.5e20, 1.5e20]]), jnp.array([[800., 1200.]]),
        radial_grid=jnp.array([.0, 2.]), coulomb_log_model='impact')
    field = coil_field()
    assert_equivalent(*run_pair(field, varying, adaptive=True, orbit_adaptive=True, spatial_diffusion=False))
    # A stopping surface ends the kernel run exactly like the shared solver.
    shared, independent = run_pair(field, bath(), adaptive=False, orbit_adaptive=False, tmax=1e-6, timestep=1e-7, n_save=3,
                                   event=lambda t, y, a: .03-y[2], max_steps=200, max_orbit_steps=200)
    np.testing.assert_array_equal(shared.status, independent.status)
    np.testing.assert_allclose(independent.t, shared.t, rtol=1e-9)
    # A tiny step budget reports MAX_STEPS rather than a wrong answer.
    shared, independent = run_pair(field, bath(), adaptive=False, orbit_adaptive=False, max_steps=1)
    assert np.all(independent.status == shared.status) and np.all(independent.status == s.SplitStatus.MAX_STEPS)


def test_scalar_lowering_reproduces_plain_jax_results():
    """The kernel's jaxpr interpreters (scalar-only, no small arrays) match direct evaluation."""
    x = jnp.array([.2, -.3, .4, .5, 0., 0., 0., 0.])
    functions = (lambda x: x[1:2], lambda x: jax.lax.dynamic_slice(x, (x[0].astype(int),), (1,)),
        lambda x: jnp.sin(x)+jnp.cos(x), jax.lax.erf, lambda x: jnp.array([jnp.sum(x), jnp.max(x), jnp.min(x)]),
        lambda x: jnp.stack([x]), lambda x: x[jnp.array([2])], lambda x: (jnp.any(x > 0), jnp.all(x < 1)),
        tuple, lambda x: tuple(jnp.split(x, len(x))), lambda x: x**-2+x**3+x**6)
    for function in functions:
        graph = s._flatten(function, x)
        actual = jax.jit(lambda x: s._evaluate_kernel(graph.jaxpr, graph.consts, x))(x)
        for a, b in zip(actual, jax.tree.leaves(function(x))):
            np.testing.assert_allclose(a, b, atol=2e-15, rtol=1e-14)
    for function in (lambda y: 1-jnp.sum(y[:3]**2), lambda y: jnp.dot(y[:3], y[:3])-.1, lambda y: jnp.max(y[:3])-jnp.min(y[:3]),
                     lambda y: y[jnp.array([1])][0], lambda y: sum(y[:3])):
        graph = s._flatten(function, x)
        np.testing.assert_allclose(s._evaluate_surface(graph.jaxpr, graph.consts, x), function(x), atol=2e-15)

    def scanned(point, reverse):
        carry, ys = jax.lax.scan(lambda a, b: (a+point[0]*b, a), point[1], point[2:5], reverse=reverse)
        return jnp.r_[carry, ys]
    interpolation = lambda point: jnp.interp(point[0], jnp.array([-1., 0., 1.]), point[1:4])
    loop = lambda p: jax.lax.while_loop(lambda c: c[0] < 3+p[0], lambda c: (c[0]+1, c[1]+p[1:3]), (jnp.asarray(0), jnp.ones(2)))[1]
    for fn in (lambda p: scanned(p, False), lambda p: scanned(p, True), interpolation, jax.jacfwd(interpolation), loop):
        flat = lambda p: tuple(jnp.ravel(fn(p)))
        graph = s._flatten(flat, x)
        actual = jax.jit(lambda p: jnp.asarray(s.scalar_graph(graph.jaxpr, graph.consts, p)))(x)
        np.testing.assert_allclose(actual, jnp.asarray(flat(x)), rtol=3e-13, atol=3e-14)


def test_independent_execution_validates_its_capabilities():
    field = coil_field()
    birth = Particles(jnp.array([[1., .03, .01]]), mass=M, charge=Q, energy=4000*ONE_EV, initial_vparallel_over_v=jnp.array([.5]))
    base = dict(tmax=1e-8, timestep=1e-8, n_save=2)
    with pytest.raises(ValueError, match='GPU'):                  # no silent fallback on CPU
        s.trace_split(field, birth, bath(), execution='independent', **base)
    pinned = dict(execution='independent', interpret=True, **base)
    for change in (dict(integrator='essos_rk45'), dict(controller='pi'), dict(orbit_adaptive=False), dict(reuse_orbit_step=False),
                   dict(electric_field=object()), dict(record_steps=True), dict(times=jnp.array([])),
                   dict(event=lambda t, y, a: True), dict(event=lambda t, y, a: jnp.ones(2)), dict(event=lambda t, y, a: 1.-t),
                   dict(max_steps=0), dict(max_orbit_steps=False), dict(max_pending=0)):
        with pytest.raises(ValueError): s.trace_split(field, birth, bath(), **(pinned | change))
    old = birth.random_keys; birth.random_keys = jax.random.key(0, impl='rbg')[None]
    with pytest.raises(ValueError): s.trace_split(field, birth, bath(), **pinned)
    birth.random_keys = old
