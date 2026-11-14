"""FaxEigenmode: a FAX global eigenmode (committed CSV output) as a Boozer-coordinate perturbation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.boozer import guiding_center_rhs, trace_boozer
from essos.perturbations import FaxEigenmode
from tests.test_perturbations import CHARGE, KW, MASS, MU, POINT, SPEED, equilibrium

FAX = Path(__file__).resolve().parents[1]/"examples/input_files/fax_mode"
AMPLITUDE = 2e4


def load(**kwargs):
    return FaxEigenmode.from_fax(FAX, **{"amplitude": AMPLITUDE, **kwargs})


def test_fax_csv_round_trip_reproduces_the_stored_structure():
    wave = load(mode=0, phase=.3)
    rows = np.genfromtxt(FAX/"global_mode_structures.csv", delimiter=",", names=True)
    assert wave.m.tolist() == [3, 4, 5, 6] and wave.n.tolist() == [2]*4
    assert float(wave.omega) == pytest.approx(2*np.pi*np.genfromtxt(FAX/"global_eigenmodes.csv", delimiter=",",
                                                                    names=True)["frequency_hz"], rel=1e-12)
    field, t, (u, w, zeta) = equilibrium(), 2.3e-6, POINT[:3]
    s = float(u*u + w*w)
    # Independent evaluation: np.interp of each harmonic, explicit angle, sin form (complex amplitude).
    phi = 0.
    for m, n in zip(wave.m, wave.n):
        h = rows[(rows["m"] == m) & (rows["n"] == n)]
        a = np.interp(s, h["s"], h["amplitude_real"]) + 1j*np.interp(s, h["s"], h["amplitude_imag"])
        phi += np.imag(a*np.exp(1j*(m*np.arctan2(w, u) - n*zeta + float(wave.omega)*t + .3)))
    phi *= AMPLITUDE/np.max(np.hypot(rows["amplitude_real"], rows["amplitude_imag"]))
    assert abs(phi) > 1. and float(wave.potentials(field, POINT[:3], t)[1]) == pytest.approx(phi, rel=1e-10)
    # Dirichlet ends of the FAX radial grid: no potential outside it.
    assert float(jnp.max(abs(wave.potentials(field, jnp.array([.05, 0., .1]), t)))) == 0.
    assert float(wave.s[0]) > 0 and float(abs(wave.profiles[0]).max()) == 0.
    with pytest.raises(ValueError):
        FaxEigenmode(wave.s, wave.profiles, wave.m, wave.n, 0., 1.)


def test_fax_mode_satisfies_the_ideal_parallel_electric_field_and_zero_amplitude_is_identity():
    field, wave, t = equilibrium(), load(), 2.3e-6
    (_, phi), derivative = jax.jit(lambda x: wave.derivatives(field, x, t))(POINT[:3])
    (iota, G, I), _ = field.profiles(jnp.sum(POINT[:2]**2))
    phi_theta = -POINT[1]*derivative[1, 0] + POINT[0]*derivative[1, 1]
    # B0 . (-d_t(alpha B0) - grad Phi) = 0, harmonic by harmonic
    assert abs(float(phi)) > 1.
    assert abs(float(derivative[0, 3] + (iota*phi_theta + derivative[1, 2])/(G + iota*I))) < 1e-9
    np.testing.assert_allclose(guiding_center_rhs(field, POINT, MU, MASS, CHARGE, t=t, perturbation=load(amplitude=0.)),
                               guiding_center_rhs(field, POINT, MU, MASS, CHARGE), rtol=2e-13)


def test_energy_change_is_the_work_of_the_wave_electric_field_and_mu_is_conserved():
    # White, Theory of Toroidally Confined Plasmas: dH/dt = q (dPhi/dt - d_t alpha B0 . xdot) with H = KE + q Phi: the energy is the work of E . xdot.
    wave, field = load(amplitude=2e3), equilibrium(uniform=True)
    v_res = float(wave.omega)*40./(2*5.)  # detuned from the Doppler resonance, where the ideal wave does no work
    out = trace_boozer(field, [.5, .5], [.2, .2], [0., 0.], [.3*v_res/SPEED, -.6*v_res/SPEED], tmax=2e-5,
                       timestep=2e-7, n_save=41, method="rk4", perturbation=wave, compact=False, **KW)
    t, states = out.times, out.states
    s, theta, zeta, vpar, speed = (states[:, :, i] for i in range(5))
    y = jnp.stack([jnp.sqrt(s)*jnp.cos(theta), jnp.sqrt(s)*jnp.sin(theta), zeta, vpar], -1)
    mu = (speed**2 - vpar**2)/(2*5.)

    def rates(y, mu, t):
        d = wave.derivatives(field, y[:3], t)
        return d[0][1], CHARGE*(d[1][1, 3] - d[1][0, 3]*40.*guiding_center_rhs(field, y, mu, MASS, CHARGE,
                                                                               t=t, perturbation=wave)[2])

    phi, power = (np.asarray(x) for x in jax.vmap(jax.vmap(rates, (0, 0, None)), (1, 1, 0), 1)(y, mu, t))
    np.testing.assert_allclose(mu, mu[:, :1]*np.ones(t.size), rtol=1e-10)
    work = np.trapezoid(power, t, axis=1)
    gain = .5*MASS*(vpar[:, -1]**2 - vpar[:, 0]**2) + CHARGE*(phi[:, -1] - phi[:, 0])
    np.testing.assert_allclose(gain, work, atol=2e-3*abs(work).max())
    assert abs(work).min() > 1e-17
