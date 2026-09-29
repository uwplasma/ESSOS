"""Boozer-coordinate guiding-centre tracing and Monte Carlo collisions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from essos.background_species import BackgroundSpecies, coulomb_logarithm, nu_D_ab
from essos.boozer import BoozerField, collision_kick, trace_boozer
from essos.constants import (ALPHA_PARTICLE_CHARGE as Q, ALPHA_PARTICLE_MASS as M, ELECTRON_MASS,
                             ELEMENTARY_CHARGE, EPSILON_0, FUSION_ALPHA_PARTICLE_ENERGY, PROTON_MASS)

B0, EPS, IOTA, G, PSI0 = 5.0, 0.25, 0.4, 40.0, 40.0
V0 = float(np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / M))


def tokamak(nfp=1):
    """|B| = B0 (1 - EPS r cos theta), constant iota and G, I = 0 (axisymmetric)."""
    s = np.linspace(0.005, 0.995, 50)
    bmnc = np.stack([np.full_like(s, B0), -B0 * EPS * np.sqrt(s)])
    return BoozerField.from_booz(s, bmnc, [0, 1], [0, 0], np.full_like(s, IOTA),
                                 np.full_like(s, G), np.zeros_like(s), PSI0, nfp)


def test_field_reproduces_the_spectrum_and_is_regular_on_the_axis():
    field = tokamak()
    for s, theta in ((0.3, 0.7), (0.9, 2.5), (1e-12, 1.0)):
        assert float(field.modB(s, theta, 0.3)) == pytest.approx(B0 * (1 - EPS * np.sqrt(s) * np.cos(theta)), rel=1e-10)
    iota, G_s, I_s = (float(x) for x in field.profiles(0.0)[0])
    assert (iota, G_s, I_s) == pytest.approx((IOTA, G, 0.0))


def test_orbits_conserve_energy_and_toroidal_canonical_momentum():
    """Axisymmetry: E and P_zeta = m v_par G / B - q iota psi0 s are invariants (White 2014)."""
    field = tokamak()
    n = 16
    pitch = jnp.linspace(-0.95, 0.95, n)
    theta = jnp.linspace(0, 2 * np.pi, n, endpoint=False)
    out = trace_boozer(field, jnp.full(n, 0.3), theta, jnp.zeros(n), pitch, speed=V0, mass=M,
                       charge=Q, tmax=2e-4, timestep=2e-8, n_save=5, devices=jax.devices()[:1])
    assert not out.lost.any() and np.all(out.energy_error < 1e-8)
    s, th, ze, vpar, v = np.moveaxis(out.states, -1, 0)
    B = B0 * (1 - EPS * np.sqrt(s) * np.cos(th))
    p_zeta = M * vpar * G / B - Q * IOTA * PSI0 * s
    np.testing.assert_allclose(p_zeta - p_zeta[:, :1], 0.0, atol=1e-7 * np.abs(p_zeta).max())


def test_nonfinite_step_is_reported_as_failed_not_confined():
    """The last finite state alone cannot reveal a failed RK step."""
    singular = eqx.tree_at(lambda f: f.psi0, tokamak(), 0.0)
    out = trace_boozer(singular, [0.3], [0.0], [0.0], [0.2], speed=V0,
                       mass=M, charge=Q, tmax=1e-6, timestep=1e-7, n_save=3)
    assert out.failed[0] and not out.lost[0]
    assert out.failed_times[0] == pytest.approx(1e-7)
    assert np.isfinite(out.states).all()  # frozen at the previous, finite state
    assert np.isinf(out.energy_error[0])
    with pytest.raises(RuntimeError, match="trajectories fail"):
        out.loss_fractions()


def test_invalid_births_and_steps_are_rejected_before_tracing():
    kwargs = dict(speed=V0, mass=M, charge=Q, tmax=1e-6, timestep=1e-7, n_save=3)
    for args, override in (
        (([], [], [], []), {}),
        (([1.0], [0.0], [0.0], [0.0]), {}),
        (([0.3], [np.nan], [0.0], [0.0]), {}),
        (([0.3], [0.0], [0.0], [1.2]), {}),
        (([[0.3]], [[0.0]], [[0.0]], [[0.0]]), {}),
        (([0.3], [0.0], [0.0], [0.0]), {"timestep": 0.0}),
        (([0.3], [0.0], [0.0], [0.0]), {"n_save": 1}),
        (([0.3], [0.0], [0.0], [0.0]), {"speed": 0.0}),
        (([0.3], [0.0], [0.0], [0.0]), {"mass": 0.0}),
        (([0.3], [0.0], [0.0], [0.0]), {"charge": 0.0}),
    ):
        with pytest.raises(ValueError):
            trace_boozer(tokamak(), *args, **(kwargs | override))


def test_repeated_traces_use_current_field_coefficients():
    """A compiled trace must not retain coefficients from the previous field."""
    s = np.linspace(0.005, 0.995, 50)
    stronger_ripple = BoozerField.from_booz(
        s, np.stack([np.full_like(s, B0), -2 * B0 * EPS * np.sqrt(s)]),
        [0, 1], [0, 0], np.full_like(s, IOTA), np.full_like(s, G),
        np.zeros_like(s), PSI0, 1)
    args = (jnp.full(4, 0.3), jnp.linspace(0, 3, 4),
            jnp.zeros(4), jnp.linspace(-0.6, 0.6, 4))
    kwargs = dict(speed=V0, mass=M, charge=Q, tmax=1e-5, timestep=1e-7, n_save=3)
    original = trace_boozer(tokamak(), *args, **kwargs)
    changed = trace_boozer(stronger_ripple, *args, **kwargs)
    repeated = trace_boozer(tokamak(), *args, **kwargs)
    assert np.max(np.abs(original.states - changed.states)) > 1e-4
    np.testing.assert_array_equal(original.states, repeated.states)


def electron_background(n=1e20, T=1.0e4):
    return BackgroundSpecies(1, jnp.array([ELECTRON_MASS / PROTON_MASS]), jnp.array([-1.0]),
                             jnp.array([n]), jnp.array([T]))


def kicks(species, v, pitch, t, steps, seed=0):
    dt = t / steps
    point = jnp.array([0.5, 0.0, 0.0])

    def step(carry, k):
        v, lam = carry
        noise = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(seed), k), (v.size, 2))
        return jax.vmap(lambda v, l, x: collision_kick(species, M, Q, v, l, point, dt, x))(v, lam, noise), None

    return jax.lax.scan(step, (v, pitch), jnp.arange(steps))[0]


def stix_rate(species, n_e=1e20, T_e=1.0e4 * ELEMENTARY_CHARGE):
    """Electron drag rate on a fast alpha: 1/tau_se (Stix 1972) times the exact G(x)/G_small(x)."""
    from scipy.special import erf

    lnL = float(coulomb_logarithm(M, Q, 0, V0, jnp.zeros(3), species))
    tau = 3 * (2 * np.pi) ** 1.5 * EPSILON_0**2 * M * T_e**1.5 / (n_e * Q**2 * ELEMENTARY_CHARGE**2 * ELECTRON_MASS**0.5 * lnL)
    x = V0 / np.sqrt(2 * T_e / ELECTRON_MASS)
    chandrasekhar = (erf(x) - 2 * x / np.sqrt(np.pi) * np.exp(-x * x)) / (2 * x * x)
    return chandrasekhar / (2 * x / (3 * np.sqrt(np.pi))) / tau


def test_collisional_orbits_slow_down_at_the_drag_rate():
    """Drift plus collisions in one trace: the mean speed of confined alphas decays at the drag rate."""
    species = electron_background()
    rate = stix_rate(species)
    n, t = 64, 0.05 / rate
    out = trace_boozer(tokamak(), jnp.full(n, 0.3), jnp.linspace(0, 6, n), jnp.zeros(n),
                       jnp.linspace(-0.9, 0.9, n), speed=V0, mass=M, charge=Q, tmax=t,
                       timestep=t / 20000, n_save=3, species=species, devices=jax.devices()[:1])
    assert not out.lost.any() and not (out.thermalized_times >= 0).any()
    assert np.mean(out.states[:, -1, 4]) / V0 == pytest.approx(np.exp(-0.05), rel=2e-3)


def test_slowing_down_on_electrons_follows_the_spitzer_time():
    """<v>(t) = v0 exp(-t / tau_se), tau_se = 3 (2 pi)^1.5 eps0^2 m_a T_e^1.5 / (n_e Z^2 e^4 m_e^0.5 lnL)
    (Stix 1972), the x = v / v_te -> 0 limit; at x = 0.22 the exact G(x) lowers the rate by 3 %."""
    species = electron_background()
    rate = stix_rate(species)
    n = 2000
    v, _ = kicks(species, jnp.full(n, V0), jnp.zeros(n), 0.2 / rate, 200)
    assert float(jnp.mean(v)) / V0 == pytest.approx(np.exp(-0.2), rel=2e-3)


def test_pitch_angle_scattering_decays_the_mean_pitch_at_nu_D():
    """Lorentz operator: <lambda>(t) = lambda0 exp(-nu_D t) at fixed speed."""
    species = BackgroundSpecies(1, jnp.array([1e4]), jnp.array([1.0]), jnp.array([1e20]), jnp.array([1.0e4]))
    nu = float(nu_D_ab(M, Q, 0, V0, jnp.zeros(3), species))
    n, t = 20000, 0.5 / nu
    v, lam = kicks(species, jnp.full(n, V0), jnp.full(n, 0.6), t, 400)
    assert float(jnp.mean(v)) == pytest.approx(V0, rel=5e-2)
    assert float(jnp.mean(lam)) == pytest.approx(0.6 * np.exp(-0.5), abs=4 * 0.8 / np.sqrt(n) + 1e-2)


def test_progress_chunks_reproduce_the_unchunked_trace():
    """Host-side chunks for progress carry the whole state: the trace is bit-identical."""
    field, n = tokamak(), 6
    kwargs = dict(speed=V0, mass=M, charge=Q, tmax=1e-4, timestep=2e-8, n_save=23,
                  species=electron_background(), seed=3)
    args = (jnp.full(n, 0.5), jnp.linspace(0, 6, n), jnp.zeros(n), jnp.linspace(-0.9, 0.9, n))
    calls = []
    chunked = trace_boozer(field, *args, **kwargs, progress=lambda d, t: calls.append((d, t)))
    whole = trace_boozer(field, *args, **kwargs)
    assert calls == [(k, 22) for k in (3, 6, 9, 12, 15, 18, 21, 22)]
    for name in ("states", "loss_times", "thermalized_times", "energy_error"):
        np.testing.assert_array_equal(getattr(chunked, name), getattr(whole, name))
