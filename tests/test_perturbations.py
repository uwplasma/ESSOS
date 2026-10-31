"""Boozer-coordinate shear-Alfven perturbations: E_parallel = 0, exact conservation laws, time stepping."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.boozer import (BoozerField, _axis_intersection, _embedded_step, _rk_step,
                          guiding_center_rhs, trace_boozer)
from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE, ONE_EV
from essos.perturbations import (AlfvenSpectrum, AnalyticalAlfvenMode, BoozerPerturbation,
                                 poloidal_flux, wave_frame_invariant)

SPEED = np.sqrt(2*3.5e6*ONE_EV/MASS)
MU = .5*SPEED**2/(2*5.)
POINT = jnp.array([.45, .27, -.3, .7*SPEED])
OMEGA, N = 9e5, 2
KW = dict(speed=SPEED, mass=MASS, charge=CHARGE, progress=False, devices=jax.devices()[:1])


def equilibrium(helicity=0, uniform=False):
    """Analytic fixture with nonconstant iota, G and I (or uniform B0 with G=40, I=iota=0)."""
    s = np.linspace(.005, .995, 32)
    if uniform:
        return BoozerField.from_booz(s, np.array([5.+0*s]), [0], [0], 0*s, 40.+0*s, 0*s, -40., 1, angle_grid=None)
    return BoozerField.from_booz(s, np.stack([5.+0*s, -np.sqrt(s)]), [0, 1], [0, helicity],
                                 .4+.1*s, 40.+.8*s, .8*s, -40., max(1, abs(helicity)), angle_grid=None)


def mode(**kwargs):
    params = dict(m=3, n=N, omega=OMEGA, amplitude=2e4, phase=.3, radial_center=.4, radial_width=.5)
    return AnalyticalAlfvenMode(**{**params, **kwargs})


def singular(**kwargs):
    return AnalyticalAlfvenMode(1, N, OMEGA, kwargs.pop("amplitude", 2000.),
                                regularize_axis=False, **kwargs)


def spectrum(relative_phase):
    """Adjacent m, common n and omega: one toroidally rotating pattern."""
    return AlfvenSpectrum(mode(m=3, phase=.3), mode(m=4, phase=.3+relative_phase))


class UniformInduction(BoozerPerturbation):
    """Exact nonideal calibration: uniform alpha(t), Phi=0, constant B0."""

    amplitude: float = .003
    omega: float = 7.5e5
    phase: float = .4

    def potentials(self, field, xyz, t):
        alpha = self.amplitude*jnp.sin(self.omega*t+self.phase)
        return jnp.stack([alpha, jnp.zeros_like(alpha)])

    def exact(self, times, v0, zeta0, *, B=5., G=40.):
        k = CHARGE*B/MASS*self.amplitude
        v = v0-k*(np.sin(self.omega*times+self.phase)-np.sin(self.phase))
        zeta = zeta0+B/G*((v0+k*np.sin(self.phase))*times
                         +k/self.omega*(np.cos(self.omega*times+self.phase)-np.cos(self.phase)))
        return v, zeta


def modB(field, y):
    return field.modB(y[0]**2+y[1]**2, jnp.arctan2(y[1], y[0]), y[2])


def chart(states):
    """Saved (s, theta, zeta, v_par, v) states to the regular chart (u, w, zeta, v_par)."""
    s, theta, zeta, vpar, _ = np.moveaxis(states, -1, 0)
    return jnp.stack([np.sqrt(s)*np.cos(theta), np.sqrt(s)*np.sin(theta), zeta, vpar], -1)


def energy_momentum(field, wave, y, mu, t):
    """GC energy H and canonical toroidal momentum P = G (M v/B + q alpha) - q psi_P."""
    B = modB(field, y)
    (_, G, _), _ = field.profiles(y[0]**2+y[1]**2)
    alpha, phi = wave.potentials(field, y[:3], t)
    return jnp.stack([MASS*(.5*y[3]**2+mu*B)+CHARGE*phi,
                      G*(MASS*y[3]/B+CHARGE*alpha)-CHARGE*poloidal_flux(field, y[0]**2+y[1]**2)])


def phase_space_density(field, wave, y, t):
    """GC phase-space density in (u, w, zeta, v_par) up to the constant 2 M q psi0.

    From gamma_theta = q psi0 s + k I, gamma_zeta = -q psi_P + k G, k = M v/B + q alpha:
    [G + iota I + (M v/(q B) + alpha)(G I' - I G')/psi0]/B. It is time dependent.
    """
    s = y[0]**2+y[1]**2
    (iota, G, I), (_, Gs, Is) = field.profiles(s)
    alpha = wave.potentials(field, y[:3], t)[0]
    return (G+iota*I+(MASS*y[3]/(CHARGE*modB(field, y))+alpha)*(G*Is-I*Gs)/field.psi0)/modB(field, y)


def conservation_residuals(field, wave, y, t):
    """Normalized rotating-frame invariant and Liouville residuals of the production RHS.

    The last two repeat both laws with the inductive acceleration removed; they must
    not vanish, otherwise the laws could not detect a missing term.
    """
    invariant = lambda y, t: energy_momentum(field, wave, y, MU, t)@jnp.array([1., -OMEGA/N])
    density = lambda y, t: phase_space_density(field, wave, y, t)
    rhs = lambda y: guiding_center_rhs(field, y, MU, MASS, CHARGE, t=t, perturbation=wave)
    gradient = jax.grad(invariant, 0)(y, t)
    rate = gradient@rhs(y)+jax.grad(invariant, 1)(y, t)
    source = jax.grad(density, 1)(y, t)
    terms = jnp.diag(jax.jacfwd(lambda y: density(y, t)*rhs(y))(y))
    volume = source+jnp.sum(terms)
    volume_scale = jnp.maximum(abs(source)+jnp.sum(abs(terms)), 1e-30)
    energy_scale = abs(energy_momentum(field, wave, y, MU, t)[0])*OMEGA
    removed = CHARGE*modB(field, y)/MASS*wave.derivatives(field, y[:3], t)[1][0, 3]
    return jnp.array([rate/energy_scale, volume/volume_scale,
                      (rate+gradient[3]*removed)/energy_scale,
                      (volume+jax.grad(density, 0)(y, t)[3]*removed)/volume_scale])


def test_poloidal_flux_integrates_iota_with_the_vmec_flux_sign():
    field = equilibrium()
    s = jnp.array([0., .13, .5, 1.])
    np.testing.assert_allclose(jax.vmap(lambda x: poloidal_flux(field, x))(s),
                               field.psi0*(.4*s+.05*s*s), atol=1e-13)


@pytest.mark.parametrize("helicity", [0, 4])
def test_potentials_satisfy_ideal_parallel_electric_field(helicity):
    field, wave, t = equilibrium(helicity), mode(), 2.3e-6
    (_, phi), derivative = jax.jit(lambda x: wave.derivatives(field, x, t))(POINT[:3])
    (iota, G, I), _ = field.profiles(jnp.sum(POINT[:2]**2))
    phi_theta = -POINT[1]*derivative[1, 0]+POINT[0]*derivative[1, 1]
    # B0 . (-d_t(alpha B0) - grad Phi) = 0
    assert abs(float(phi)) > 1.
    assert abs(float(derivative[0, 3]+(iota*phi_theta+derivative[1, 2])/(G+iota*I))) < 1e-10


def test_zero_amplitude_recovers_existing_gc_equations():
    field = equilibrium()
    actual = guiding_center_rhs(field, POINT, MU, MASS, CHARGE, t=1e-6, perturbation=mode(amplitude=0.))
    np.testing.assert_allclose(actual, guiding_center_rhs(field, POINT, MU, MASS, CHARGE), rtol=2e-13)


@pytest.mark.parametrize("helicity", [0, 4])
def test_single_mode_wave_frame_invariant_is_a_differential_identity(helicity):
    field, wave, t = equilibrium(helicity), mode(), 2.3e-6
    invariant = lambda y, t: wave_frame_invariant(field, wave, y, MU, MASS, CHARGE, t, helicity=helicity)
    dy = guiding_center_rhs(field, POINT, MU, MASS, CHARGE, t=t, perturbation=wave)
    rate = jax.grad(invariant, 0)(POINT, t)@dy+jax.grad(invariant, 1)(POINT, t)
    assert abs(float(rate))/(abs(float(invariant(POINT, t)))*OMEGA) < 1e-12
    # Energy exchange is physical even while this combined invariant is conserved.
    kinetic = lambda y: MASS*(.5*y[3]**2+MU*modB(field, y))
    assert abs(float(jax.grad(kinetic)(POINT)@dy)) > 1e-12


@pytest.mark.parametrize("phase", np.arange(4)*np.pi/2)
def test_two_harmonic_wave_preserves_energy_momentum_and_phase_space(phase):
    field, wave = equilibrium(), spectrum(phase)
    points = jnp.array([[.45, .27, -.3, .7*SPEED], [.3, -.4, .7, -.65*SPEED],
                        [-.5, .2, 1.2, .5*SPEED], [.2, .6, -.8, -.8*SPEED]])
    times = jnp.array([0., 1.3e-6, 4.7e-6, 9e-6])
    r = np.asarray(jax.jit(jax.vmap(lambda y, t: conservation_residuals(field, wave, y, t)))(points, times))
    assert np.max(abs(r[:, :2])) < 1e-12
    assert np.max(abs(r[:, 2])) > 1e-6 and np.max(abs(r[:, 3])) > 1e-10
    # The time-dependent density stays finite and positive towards the axis (u, w chart).
    near = [float(phase_space_density(field, wave, POINT.at[:2].set(jnp.array([x, 0.])), 0.)) for x in (.01, .001)]
    assert min(near) > 1. and near[0] == pytest.approx(near[1], rel=.01)


def test_uniform_induction_has_the_exact_sign_and_normalization():
    wave, t = UniformInduction(), 1.3e-6
    dy = guiding_center_rhs(equilibrium(uniform=True), POINT, 1e12, MASS, CHARGE, t=t, perturbation=wave)
    expected = -CHARGE*5./MASS*wave.amplitude*wave.omega*np.cos(wave.omega*t+wave.phase)
    np.testing.assert_allclose(dy[:2], 0., atol=1e-12)
    assert float(dy[2]) == pytest.approx(float(5./40.*POINT[3]), rel=1e-13)
    assert float(dy[3]) == pytest.approx(float(expected), rel=1e-13)


@pytest.mark.parametrize("m", [0, 1, 3, 15, 30])
def test_regular_potential_and_gc_rhs_are_finite_on_axis(m):
    field, wave = equilibrium(), mode(m=m)
    y = jnp.array([0., 0., .2, .7*SPEED])
    value, derivative = wave.derivatives(field, y[:3], 1e-6)
    assert np.isfinite(value).all() and np.isfinite(derivative).all()
    assert np.isfinite(guiding_center_rhs(field, y, 1e12, MASS, CHARGE, t=1e-6, perturbation=wave)).all()


@pytest.mark.parametrize("method", ["rk4", "dopri5", "dopri8", "adaptive", "adaptive8"])
def test_runge_kutta_uses_physical_stage_times(method):
    rhs = lambda t, y: jnp.full_like(y, t**3)
    t, dt, y = .7, .2, jnp.array([1.])
    if method.startswith("adaptive"):
        actual, _ = _embedded_step(rhs, y, dt, jnp.ones_like(y), method, t=t)
    else:
        actual = _rk_step(rhs, y, dt, method, t=t)
    np.testing.assert_allclose(actual, y+((t+dt)**4-t**4)/4, rtol=1e-14)


@pytest.mark.parametrize("method", ["rk4", "dopri5", "dopri8", "adaptive", "adaptive8"])
def test_perturbed_trace_conserves_wave_invariant_and_reports_physical_work(method):
    field, wave = equilibrium(), mode()
    out = trace_boozer(field, [.3, .35], [.2, .7], [0., .3], [.65, -.7], tmax=8e-6, timestep=2e-8,
                       n_save=15, method=method, tolerance=1e-11, perturbation=wave, compact=False, **KW)
    assert not out.failed.any() and not out.lost.any() and out.energy_error is None
    assert np.max(out.kinetic_energy_change) > 1e-6
    np.testing.assert_array_equal(out.loss_fractions(), 0.)
    for states in out.states:
        y = chart(states)
        mu = SPEED**2*(1-(float(y[0, 3])/SPEED)**2)/(2*float(modB(field, y[0])))
        values = jax.vmap(lambda y, t: wave_frame_invariant(field, wave, y, mu, MASS, CHARGE, t))(y, out.times)
        assert np.max(np.abs(values/values[0]-1)) < 2e-8


def test_adaptive_traces_match_exact_induction_and_two_harmonic_invariant():
    wave = UniformInduction()
    out = trace_boozer(equilibrium(uniform=True), [.3], [.2], [.3], [.65], tmax=1.5e-5, timestep=2e-7,
                       n_save=17, method="adaptive8", tolerance=1e-10, perturbation=wave, compact=False, **KW)
    v, zeta = wave.exact(out.times, .65*SPEED, .3)
    assert np.max(abs(out.states[0, :, 3]-v))/SPEED < 1e-8
    assert np.max(abs(out.states[0, :, 2]-zeta)) < 1e-7
    np.testing.assert_allclose(out.states[0, :, :2], [[.3, .2]]*17, atol=1e-13)
    field, wave = equilibrium(), spectrum(np.pi/2)
    out = trace_boozer(field, [.3, .43], [.2, 1.4], [0., .3], [.65, -.7], tmax=1.5e-5, timestep=2e-7,
                       n_save=17, method="adaptive8", tolerance=1e-10, perturbation=wave, compact=False, **KW)
    for states in out.states:
        y = chart(states)
        mu = SPEED**2*(1-(float(y[0, 3])/SPEED)**2)/(2*float(modB(field, y[0])))
        H, P = np.asarray(jax.vmap(lambda y, t: energy_momentum(field, wave, y, mu, t))(y, out.times)).T
        K = H-OMEGA/N*P
        assert np.max(abs(K-K[0]))/abs(H[0]) < 1e-8
        assert np.max(abs(H/H[0]-1)) > 1e-5  # wave work is real, K is the conserved combination


def test_amplitude_gradient_matches_finite_difference_through_gc_step():
    field, wave = equilibrium(), mode()

    def objective(amplitude):
        changed = eqx.tree_at(lambda p: p.amplitude, wave, amplitude)
        rhs = lambda t, y: guiding_center_rhs(field, y, 1e12, MASS, CHARGE, t=t, perturbation=changed)
        end = _rk_step(rhs, POINT, 1e-8, "rk4", t=2e-6)
        return end[0]**2+end[1]**2

    ad = float(jax.jit(jax.grad(objective))(2e4))
    assert abs(ad) > 1e-12
    assert ad == pytest.approx(float((objective(2e4+1.)-objective(2e4-1.))/2), rel=2e-5, abs=1e-13)
    # A mode may also be constructed inside the differentiated function.
    xyz = jnp.array([.4, .2, .3])
    evaluate = lambda a: mode(amplitude=a).potentials(field, xyz, 1e-6)[1]
    np.testing.assert_allclose(jax.jit(jax.grad(evaluate))(2e4), evaluate(1.), rtol=1e-13)


@pytest.mark.parametrize("kwargs", [dict(omega=0.), dict(m=1.5), dict(n=True), dict(radial_width=-1.),
                                    dict(axis_stop_s=0.), dict(axis_stop_s=np.nan), dict(axis_stop_s=1.)])
def test_invalid_mode_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        mode(**kwargs)


def test_per_marker_speeds_zero_amplitude_and_collision_rules():
    field = equilibrium()
    speed = [.8*SPEED, SPEED]
    kwargs = {**KW, "speed": speed, "tmax": 2e-7, "timestep": 1e-8, "n_save": 4, "method": "rk4", "compact": False}
    births = ([.3, .4], [.2, .3], [0., .1], [.7, -.7])
    plain = trace_boozer(field, *births, **kwargs)
    zero = trace_boozer(field, *births, perturbation=mode(amplitude=0.), **kwargs)
    np.testing.assert_allclose(plain.states, zero.states, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(plain.states[:, 0, 4], speed)
    with pytest.raises(ValueError, match="fixed RK"):
        trace_boozer(field, [.3], [0.], [0.], [.7], tmax=1e-6, perturbation=mode(), species=object(),
                     method="adaptive8", **KW)


@pytest.mark.parametrize("method", ["rk4", "adaptive8"])
def test_wave_time_is_preserved_across_progress_chunks_and_compaction(method):
    field, wave = equilibrium(), mode()
    s = np.array([.9997, .3, .4, .45, .5, .55, .6])
    kwargs = dict(tmax=2e-6, timestep=2e-8, n_save=11, method=method, tolerance=1e-10, perturbation=wave)
    births = (s, np.full(s.size, np.pi/2), np.zeros(s.size), np.full(s.size, .7))
    reference = trace_boozer(field, *births, compact=False, **kwargs, **KW)
    reports = []
    compact = trace_boozer(field, *births, **kwargs, **{**KW, "progress": lambda done, total: reports.append(done)})
    assert reference.lost.any() and (~reference.lost).any()
    assert reports[-1] == 10
    np.testing.assert_allclose(compact.states, reference.states, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(compact.loss_times, reference.loss_times, atol=1e-14)


# --- singular (radially uniform) modes and their inner boundary ------------------------------

def test_only_nonzero_singular_angular_modes_request_axis_stops():
    assert float(singular().axis_stop_threshold()) == 1e-8
    assert float(singular(amplitude=0.).axis_stop_threshold()) == 0.
    assert mode().axis_stop_threshold() is None
    assert singular(axis_stop_s=None).axis_stop_threshold() is None
    assert AnalyticalAlfvenMode(0, N, OMEGA, 2000., regularize_axis=False).axis_stop_threshold() is None
    assert float(AlfvenSpectrum(singular(), singular(axis_stop_s=1e-7)).axis_stop_threshold()) == 1e-7


def test_axis_segment_intersection_detects_entry_and_pass_through():
    hit, fraction = _axis_intersection(jnp.array([-.2, 0., 0., 1.]), jnp.array([.2, 0., 0., 1.]), .01)
    assert bool(hit) and float(fraction) == pytest.approx(.25)
    # A tangentially displaced chord misses the disk; a disabled cutoff or nonfinite end never hits.
    assert not bool(_axis_intersection(jnp.array([-.2, .2]), jnp.array([.2, .2]), .01)[0])
    assert not bool(_axis_intersection(jnp.array([-.2, 0.]), jnp.array([.2, 0.]), 0.)[0])
    assert not bool(_axis_intersection(jnp.array([.2, 0.]), jnp.array([jnp.nan, 0.]), .01)[0])


@pytest.mark.parametrize("compact", [False, True])
def test_births_inside_axis_boundary_stop_and_survive_compaction(compact):
    out = trace_boozer(equilibrium(), [0., 5e-9, .3], [0., 0., .2], [0., 0., 0.], [1., 1., 1.],
                       tmax=1e-10, timestep=1e-11, n_save=4, method="rk4", perturbation=singular(),
                       compact=compact, **KW)
    np.testing.assert_array_equal(out.axis_stopped, [True, True, False])
    np.testing.assert_allclose(out.axis_stop_times[:2], 0.)
    assert not out.failed.any() and not out.lost.any() and np.isfinite(out.states).all()
    np.testing.assert_allclose(out.states[:2], np.broadcast_to(out.states[:2, :1], out.states[:2].shape))
    np.testing.assert_allclose(out.loss_fractions(), 0.)
    np.testing.assert_allclose(out.loss_fractions(include_axis=True), 2/3)


@pytest.mark.parametrize("method", ["rk4", "adaptive8"])
def test_incoming_orbit_stops_at_the_configured_boundary_with_a_finite_frozen_state(method):
    cutoff = 1e-8
    out = trace_boozer(equilibrium(), [4*cutoff], [np.pi], [0.], [0.], tmax=2e-9, timestep=2e-11, n_save=6,
                       method=method, tolerance=1e-10, perturbation=singular(axis_stop_s=cutoff),
                       compact=False, **KW)
    assert out.axis_stopped[0] and not out.lost[0] and not out.failed[0]
    assert 0 < out.axis_stop_times[0] < out.times[-1]
    assert out.states[0, -1, 0] == pytest.approx(cutoff, rel=1e-10)
    frozen = out.states[0, out.times >= out.axis_stop_times[0]]
    np.testing.assert_allclose(frozen, np.broadcast_to(frozen[0], frozen.shape))
    assert out.loss_fractions()[-1] == 0. and out.loss_fractions(include_axis=True)[-1] == 1.


def test_zero_amplitude_singular_mode_neither_stops_nor_fails_on_axis():
    out = trace_boozer(equilibrium(), [0., 1e-10], [0., .3], [0., 0.], [1., 1.], tmax=1e-10, timestep=1e-11,
                       n_save=3, method="rk4", perturbation=singular(amplitude=0.), compact=False, **KW)
    assert not out.axis_stopped.any() and not out.failed.any() and np.isfinite(out.states).all()
