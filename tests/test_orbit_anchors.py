"""Closed-form anchors for Boozer guiding-centre orbits and pitch-angle scattering.

Guiding-centre Lagrangian integrated by ``essos.boozer.guiding_center_rhs`` (White, The Theory of
Toroidally Confined Plasmas, 3rd ed., 2014, eq. 4.25 in Boozer coordinates), with toroidal flux
psi = psi0 s and poloidal flux psi_p = iota psi0 s for constant iota:

    L = (q psi + m v_par I / B) theta' + (m v_par G / B - q psi_p) zeta' - (m v_par^2 / 2 + m mu B),

so P_theta = q psi0 s + m v_par I / B and P_zeta = m v_par G / B - q iota psi0 s. The rate
equations of ``guiding_center_rhs`` (C = -q iota + ..., F = q + ...) are the Euler-Lagrange
equations of this L. mu = v_perp^2 / (2 B) is a parameter of the equations, constant by
construction, so it needs no test. If |B| depends on the angles only through m theta - k zeta
(``xn = k``), L is invariant under theta -> theta + k e, zeta -> zeta + m e, and Noether's
theorem gives the helical momentum J = k P_theta + m P_zeta.
"""
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre
from scipy.special import ellipk

from essos.background_species import BackgroundSpecies
from essos.boozer import BoozerField, _collision_rates, collision_kick, trace_boozer
from essos.constants import ALPHA_PARTICLE_CHARGE as Q
from essos.constants import ALPHA_PARTICLE_MASS as M
from essos.constants import FUSION_ALPHA_PARTICLE_ENERGY

jax.config.update("jax_enable_x64", True)

B0, G, PSI0, IOTA, S0 = 5.0, 40.0, 40.0, 0.4, 0.3
V0 = float(np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / M))


def field(modes, mode_tolerance=1e-6):
    """|B| = B0 (1 + sum_j a_j r^m_j cos(m_j theta - k_j zeta)), constant iota and G, I = 0."""
    s = np.linspace(0.005, 0.995, 50)
    bmnc = np.stack([np.full_like(s, B0)] + [B0 * a * np.sqrt(s)**m for m, _, a in modes])
    return BoozerField.from_booz(s, bmnc, [0] + [m for m, _, _ in modes], [0] + [k for _, k, _ in modes],
                                 np.full_like(s, IOTA), np.full_like(s, G), np.zeros_like(s), PSI0, 1,
                                 mode_tolerance, angle_grid=None)


def trace(f, theta, pitch, tmax, dt, n_save, method="dopri8", speed=V0):
    n = len(pitch)
    return trace_boozer(f, jnp.full(n, S0), jnp.asarray(theta, float), jnp.zeros(n), jnp.asarray(pitch, float),
                        speed=speed, mass=M, charge=Q, tmax=tmax, timestep=dt, n_save=n_save,
                        devices=jax.devices()[:1], method=method, progress=False)


def momenta(f, states):
    s, th, ze, vpar = (states[..., i] for i in range(4))
    B = np.asarray(jax.vmap(f.modB)(s.ravel(), th.ravel(), ze.ravel())).reshape(s.shape)
    return Q * PSI0 * s, M * vpar * G / B - Q * IOTA * PSI0 * s


def drift(x):
    return np.abs(x - x[:, :1]).max(axis=1) / (Q * PSI0 * S0)


def test_toroidal_momentum_drift_converges_at_the_order_of_rk4_for_passing_and_trapped_orbits():
    """Axisymmetric |B| = B0 (1 - 0.25 r cos theta) at s = 0.3: a passing (pitch 0.9) and a trapped
    (pitch 0.2 < sqrt(2 eps r / (1 + eps r)) = 0.49) alpha, 20 poloidal transits. RK4 has global order
    4, so halving the step must divide the drift of P_zeta and of the energy by at least 2^4 up to
    the next term of the expansion: measured order >= 3.8. Measured 5.0 (passing) and 4.1 (trapped)."""
    f = field([(1, 0, -0.25)])
    errors = []
    for dt in (1e-7, 5e-8):
        out = trace(f, [0.0, 0.0], [0.9, 0.2], 2e-4, dt, 41, "rk4")
        vpar = out.states[..., 3]
        assert not out.lost.any()
        assert vpar[0].min() > 0 and vpar[1].min() < 0 < vpar[1].max()  # passing, trapped
        errors.append((drift(momenta(f, out.states)[1]), out.energy_error))
    for coarse, fine in zip(*errors):
        assert np.all(np.log2(coarse / fine) >= 3.8), np.log2(coarse / fine)
    assert errors[1][0].max() < 1e-7 and errors[1][1].max() < 1e-9


def test_bounce_period_of_a_trapped_orbit_is_the_pendulum_period():
    """|B| = B0 (1 - eps r cos theta), I = 0: with theta' = iota v_par B / G and
    v_par' = -iota mu B dB/dtheta / G (``guiding_center_rhs``) the bounce motion is a pendulum of
    frequency w0 = iota B0 sqrt(mu B0 eps r) / G and amplitude sin^2(theta_b / 2) = v_par0^2 / (4 mu B0 eps r),
    period T = 4 K(sin^2(theta_b / 2)) / w0. Neglected: B / B0 - 1 = O(eps r) in theta' and the radial
    excursion, which a speed of V0 / 20 makes small, so |T / T_pendulum - 1| <= eps r. Births at theta = 0
    with theta_b = 1 rad for eps = 0.04 and 0.01."""
    r, speed = np.sqrt(S0), V0 / 20
    for eps in (0.04, 0.01):
        er = eps * r
        a = 2 * er * np.sin(0.5)**2 / (1 - er)
        pitch = np.sqrt(a / (1 + a))
        mu = speed**2 * (1 - pitch**2) / (2 * B0 * (1 - er))
        k2 = (pitch * speed)**2 / (4 * mu * B0 * er)
        period = 4 * ellipk(k2) / (IOTA * B0 * np.sqrt(mu * B0 * er) / G)
        out = trace(field([(1, 0, -eps)]), [0.0], [pitch], 3 * period, period / 2000, 601, speed=speed)
        t, v = out.times, out.states[0, :, 3]
        i = np.flatnonzero(np.sign(v[1:]) != np.sign(v[:-1]))
        crossings = t[i] - v[i] * (t[i + 1] - t[i]) / (v[i + 1] - v[i])
        assert crossings.size >= 5
        assert abs(2 * np.mean(np.diff(crossings)) / period - 1) < er, (eps, np.diff(crossings) / period)


def test_helical_momentum_is_exact_for_one_helicity_and_broken_linearly_by_a_second():
    """|B| = B0 (1 - 0.1 r cos(theta - 2 zeta)): J = 2 P_theta + P_zeta is conserved to the Dopri5
    error at dt = 4e-8 s (bound 1e-9 of q psi0 s, measured 3.6e-10; Dopri8 gives 2e-14) while P_zeta alone moves by 2-6 %. Adding
    delta r cos(theta - zeta) breaks the symmetry; dJ/dt = -dH/d(angle) is first order in delta, so the
    drift of J over the same time scales as delta^1: exponent from delta = 1e-3 and 4e-3 in [0.9, 1.1]."""
    theta, pitch = [0.0, 1.0, 2.0, 3.0], [0.9, 0.3, -0.5, 0.1]
    drifts = []
    for delta in (0.0, 1e-3, 4e-3):
        f = field([(1, 2, -0.1), (1, 1, delta)], mode_tolerance=-1)  # keep the zero mode: one compilation
        out = trace(f, theta, pitch, 1e-4, 4e-8, 21, "dopri5")
        assert not out.lost.any()
        p_theta, p_zeta = momenta(f, out.states)
        drifts.append(drift(2 * p_theta + p_zeta))
        if not delta:
            assert drifts[0].max() < 1e-9 and drift(p_zeta).min() > 1e-2
    exponent = np.log(drifts[2] / drifts[1]) / np.log(4.0)
    assert np.all((exponent > 0.9) & (exponent < 1.1)), exponent


SCATTERING = BackgroundSpecies(1, jnp.array([1e4]), jnp.array([1.0]), jnp.array([1e20]), jnp.array([1.0e4]))
POINT = jnp.array([0.5, 0.0, 0.0])


def scatter(n, nu_t, steps, lam0, seed=0):
    """Pitch kicks of ``collision_kick`` at fixed speed V0 (its speed update is discarded)."""
    nu = float(_collision_rates(SCATTERING, M, Q, V0, POINT)[0])
    dt = nu_t / nu / steps

    def step(lam, k):
        noise = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(seed), k), (n, 2))
        return jax.vmap(lambda x, xi: collision_kick(SCATTERING, M, Q, V0, x, POINT, dt, xi)[1])(lam, noise), None

    return np.asarray(jax.jit(lambda: jax.lax.scan(step, jnp.full(n, lam0), jnp.arange(steps))[0])())


def test_legendre_moments_of_pitch_scattering_decay_at_l_l_plus_1_nu_over_2():
    """``collision_kick`` is d lambda = -nu_D lambda dt + sqrt((1 - lambda^2) nu_D dt) xi, whose generator
    (nu_D / 2) d/dlambda (1 - lambda^2) d/dlambda has the Legendre polynomials as eigenfunctions:
    <P_l(lambda)>(t) = P_l(lambda0) exp(-l (l + 1) nu_D t / 2), with nu_D the deflection rate ESSOS
    computes. N = 1e5 markers, nu_D t = 0.5, lambda0 = 0.6, l = 1, 2, 3, tolerance k = 4 standard errors.
    Time step: the scheme is weak order 1, so its bias shrinks with nu_D dt. With 25 steps (nu_D dt = 0.02)
    the l = 3 bias is resolved (measured -0.011, 9 standard errors); with 400 steps all three agree."""
    lam0, nu_t, n = 0.6, 0.5, 100_000
    for steps, seed in ((400, 1), (25, 0)):
        lam = scatter(n, nu_t, steps, lam0, seed)
        for l in (1, 2, 3):
            basis = [0] * l + [1]
            p = legendre.legval(lam, basis)
            bias = p.mean() - legendre.legval(lam0, basis) * np.exp(-l * (l + 1) * nu_t / 2)
            if steps == 400:
                assert abs(bias) < 4 * p.std() / np.sqrt(n), (l, bias)
    assert abs(bias) > 4 * p.std() / np.sqrt(n), bias
