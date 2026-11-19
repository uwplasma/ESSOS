"""Ideal electric fields, GC invariants and nonautonomous stepping checks."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.boozer import BoozerField, _embedded_step, _rk_step, guiding_center_rhs, trace_boozer
from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE, ONE_EV
from essos.perturbations import (AnalyticalAlfvenMode, poloidal_flux,
                                 wave_frame_invariant)


SPEED = np.sqrt(2*3.5e6*ONE_EV/MASS)


def equilibrium(helicity=0):
    s = np.linspace(.005, .995, 32)
    return BoozerField.from_booz(s, np.stack([5.+0*s, -np.sqrt(s)]), [0, 1], [0, helicity],
                                 .4+.1*s, 40.+.1*s, .2*s, -40., max(1, abs(helicity)),
                                 angle_grid=None)


def mode(**kwargs):
    params = dict(m=3, n=2, omega=9.e5, amplitude=2.e4, phase=.3,
                  radial_center=.4, radial_width=.5)
    params.update(kwargs)
    return AnalyticalAlfvenMode(**params)


def test_poloidal_flux_integrates_iota_with_the_vmec_flux_sign():
    field = equilibrium()
    s = jnp.array([0., .13, .5, 1.])
    np.testing.assert_allclose(jax.vmap(lambda x: poloidal_flux(field, x))(s),
                               field.psi0*(.4*s+.05*s*s), atol=1e-13)


@pytest.mark.parametrize("helicity", [0, 4])
def test_potentials_satisfy_ideal_parallel_electric_field(helicity):
    field, wave = equilibrium(helicity), mode()
    xyz, t = jnp.array([.45, .27, -.3]), 2.3e-6
    (_, phi), derivative = jax.jit(lambda x: wave.derivatives(field, x, t))(xyz)
    (iota, G, I), _ = field.profiles(jnp.sum(xyz[:2]**2))
    phi_theta = -xyz[1]*derivative[1, 0]+xyz[0]*derivative[1, 1]
    residual = derivative[0, 3]+(iota*phi_theta+derivative[1, 2])/(G+iota*I)
    assert abs(float(phi)) > 1.
    assert abs(float(residual)) < 1e-10


def test_zero_amplitude_recovers_existing_gc_equations():
    field, wave = equilibrium(), mode(amplitude=0.)
    y, mu = jnp.array([.45, .27, -.3, .7*SPEED]), .5*SPEED**2/(2*5.)
    actual = guiding_center_rhs(field, y, mu, MASS, CHARGE, t=1e-6, perturbation=wave)
    np.testing.assert_allclose(actual, guiding_center_rhs(field, y, mu, MASS, CHARGE), rtol=2e-13)


@pytest.mark.parametrize("helicity", [0, 4])
def test_single_mode_wave_frame_invariant_is_a_differential_identity(helicity):
    field, wave = equilibrium(helicity), mode()
    y, mu, t = jnp.array([.45, .27, -.3, .7*SPEED]), .5*SPEED**2/(2*5.), 2.3e-6
    invariant = lambda y, t: wave_frame_invariant(field, wave, y, mu, MASS, CHARGE,
                                                  t, helicity=helicity)
    dy = guiding_center_rhs(field, y, mu, MASS, CHARGE, t=t, perturbation=wave)
    rate = jax.grad(invariant, 0)(y, t)@dy+jax.grad(invariant, 1)(y, t)
    scale = abs(float(invariant(y, t)))*abs(float(wave.omega))
    assert abs(float(rate))/scale < 1e-12
    # Energy exchange is physical even while this combined invariant is conserved.
    kinetic = lambda y: MASS*(.5*y[3]**2+mu*field.modB(y[0]**2+y[1]**2,
                                                     jnp.arctan2(y[1], y[0]), y[2]))
    assert abs(float(jax.grad(kinetic)(y)@dy)) > 1e-12


@pytest.mark.parametrize("m", [0, 1, 2, 3, 15, 30])
def test_regular_potential_and_gc_rhs_are_finite_on_axis(m):
    field, wave = equilibrium(), mode(m=m)
    y = jnp.array([0., 0., .2, .7*SPEED])
    value, derivative = wave.derivatives(field, y[:3], 1e-6)
    assert np.isfinite(value).all() and np.isfinite(derivative).all()
    assert np.isfinite(guiding_center_rhs(field, y, 1e12, MASS, CHARGE,
                                         t=1e-6, perturbation=wave)).all()


@pytest.mark.parametrize("method", ["rk4", "dopri5", "dopri8", "adaptive", "adaptive8"])
def test_runge_kutta_uses_physical_stage_times(method):
    rhs = lambda t, y: jnp.full_like(y, t**3)
    t, dt, y = .7, .2, jnp.array([1.])
    expected = y+((t+dt)**4-t**4)/4
    if method.startswith("adaptive"):
        actual, _ = _embedded_step(rhs, y, dt, jnp.ones_like(y), method, t=t)
    else:
        actual = _rk_step(rhs, y, dt, method, t=t)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


@pytest.mark.parametrize("method", ["rk4", "dopri5", "dopri8", "adaptive", "adaptive8"])
def test_perturbed_trace_conserves_wave_invariant_and_reports_physical_work(method):
    field, wave = equilibrium(), mode()
    out = trace_boozer(field, [.3, .35], [.2, .7], [0., .3], [.65, -.7],
                        speed=SPEED, mass=MASS, charge=CHARGE, tmax=8e-6,
                        timestep=2e-8, n_save=15, method=method, tolerance=1e-11,
                        perturbation=wave, progress=False, compact=False, devices=jax.devices()[:1])
    assert not out.failed.any() and not out.lost.any()
    assert out.energy_error is None
    assert np.max(out.kinetic_energy_change) > 1e-6
    np.testing.assert_array_equal(out.loss_fractions(), 0.)
    for trajectory in out.states:
        s, th, ze, vp, speed = trajectory.T
        y = jnp.stack([jnp.sqrt(s)*jnp.cos(th), jnp.sqrt(s)*jnp.sin(th), ze, vp], axis=1)
        mu = SPEED**2*(1-(float(vp[0])/SPEED)**2)/(2*field.modB(s[0], th[0], ze[0]))
        values = jax.vmap(lambda y, t: wave_frame_invariant(field, wave, y, mu, MASS,
                                                            CHARGE, t))(y, out.times)
        assert np.max(np.abs(values/values[0]-1)) < 2e-8


def test_mode_amplitude_derivative_matches_finite_difference_through_gc_step():
    field, wave = equilibrium(), mode()
    y = jnp.array([.45, .27, -.3, .7*SPEED])

    def objective(amplitude):
        changed = eqx.tree_at(lambda p: p.amplitude, wave, amplitude)
        rhs = lambda t, y: guiding_center_rhs(field, y, 1e12, MASS, CHARGE, t=t, perturbation=changed)
        end = _rk_step(rhs, y, 1e-8, "rk4", t=2e-6)
        return end[0]**2+end[1]**2

    amplitude, h = float(wave.amplitude), 1.
    ad = float(jax.jit(jax.grad(objective))(amplitude))
    fd = float((objective(amplitude+h)-objective(amplitude-h))/(2*h))
    assert abs(ad) > 1e-12
    assert ad == pytest.approx(fd, rel=2e-5, abs=1e-13)


def test_unregularized_paper_harmonic_is_flagged_on_axis():
    field, wave = equilibrium(), mode(regularize_axis=False, axis_stop_s=None)
    out = trace_boozer(field, [0.], [0.], [0.], [.7], speed=SPEED, mass=MASS,
                        charge=CHARGE, tmax=1e-7, timestep=1e-8, n_save=2,
                        perturbation=wave, method="rk4", progress=False, devices=jax.devices()[:1])
    assert out.failed[0] and not out.lost[0]
    with pytest.raises(RuntimeError, match="trajectories fail"):
        out.loss_fractions()


@pytest.mark.parametrize("kwargs", [dict(omega=0.), dict(m=1.5), dict(n=True), dict(radial_width=-1.)])
def test_invalid_mode_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        mode(**kwargs)


def test_per_marker_birth_speeds_and_zero_mode_trace():
    field = equilibrium()
    kwargs = dict(speed=[.8*SPEED, SPEED], mass=MASS, charge=CHARGE, tmax=2e-7,
                  timestep=1e-8, n_save=4, method="rk4", compact=False, progress=False,
                  devices=jax.devices()[:1])
    plain = trace_boozer(field, [.3, .4], [.2, .3], [0., .1], [.7, -.7], **kwargs)
    zero = trace_boozer(field, [.3, .4], [.2, .3], [0., .1], [.7, -.7],
                         perturbation=mode(amplitude=0.), **kwargs)
    np.testing.assert_allclose(plain.states, zero.states, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(plain.states[:, 0, 4], kwargs["speed"])


@pytest.mark.parametrize("method", ["rk4", "adaptive8"])
def test_wave_time_is_preserved_across_progress_chunks_and_compaction(method):
    field, wave = equilibrium(), mode()
    s = np.array([.9997, .3, .4, .45, .5, .55, .6])
    kwargs = dict(speed=SPEED, mass=MASS, charge=CHARGE, tmax=2e-6,
                  timestep=2e-8, n_save=11, method=method, tolerance=1e-10,
                  perturbation=wave, devices=jax.devices()[:1])
    births = (s, np.full(s.size, np.pi/2), np.zeros(s.size), np.full(s.size, .7))
    reference = trace_boozer(field, *births, progress=False, compact=False, **kwargs)
    reports = []
    compact = trace_boozer(field, *births, progress=lambda done, total: reports.append(done),
                            compact=True, **kwargs)
    assert reference.lost.any() and (~reference.lost).any()
    assert reports[-1] == 10
    np.testing.assert_allclose(compact.states, reference.states, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(compact.loss_times, reference.loss_times, atol=1e-14)


def test_mode_can_be_constructed_inside_an_amplitude_gradient():
    field = equilibrium()
    xyz = jnp.array([.4, .2, .3])
    evaluate = lambda amplitude: mode(amplitude=amplitude).potentials(field, xyz, 1e-6)[1]
    np.testing.assert_allclose(jax.jit(jax.grad(evaluate))(1.e4), evaluate(1.), rtol=1e-13)


def test_combined_collisions_require_fixed_steps():
    with pytest.raises(ValueError, match="fixed RK"):
        trace_boozer(equilibrium(), [.3], [0.], [0.], [.7], speed=SPEED, mass=MASS,
                     charge=CHARGE, tmax=1e-6, perturbation=mode(), species=object(), method="adaptive8")
