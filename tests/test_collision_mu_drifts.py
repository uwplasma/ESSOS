"""Independent checks of the (vpar, mu) guiding-centre collision SDEs.

In a uniform field the guiding-centre phase-space Jacobian is constant in
(vpar, mu), so the Fokker-Planck operator in Landau form
``d_t f = d_i(-K_i f + D_ij d_j f)`` with friction ``K = -nu_s (vpar, 2 mu)``
has Ito drift ``A_i = K_i + d_j D_ij``.  ``D`` is the velocity-space tensor
``(D_par vhat vhat + D_perp (1 - vhat vhat)) / m**2`` mapped to (vpar, mu).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from essos.background_species import JOULE_PER_EV, BackgroundSpecies, nu_D_ab, nu_par_ab, nu_s_ab
from essos.constants import ELEMENTARY_CHARGE, PROTON_MASS, SPEED_OF_LIGHT
from essos.dynamics import (GuidingCenterCollisionsDiffusionMu, GuidingCenterCollisionsDriftMuIto,
                            GuidingCenterCollisionsDriftMuStratonovich)

B0 = 2.0
M = PROTON_MASS
Q = ELEMENTARY_CHARGE


class _UniformField:
    def AbsB(self, x):
        return B0

    def B_contravariant(self, x):
        return jnp.array([0.0, 0.0, B0])

    B_covariant = B_contravariant

    def dAbsB_by_dX(self, x):
        return jnp.zeros(3)

    kappa = curl_b = dAbsB_by_dX

    def sqrtg(self, x):
        return 1.0


class _NoE:
    def E_covariant(self, x):
        return jnp.zeros(3)


class _Particle:
    mass = M
    charge = Q


FIELD, PARTICLE, EFIELD = _UniformField(), _Particle(), _NoE()
X0 = jnp.array([1.0, 0.0, 0.0])


def _species(T_eV):
    return BackgroundSpecies(1, jnp.array([1.0]), jnp.array([1.0]), jnp.array([1e20]), jnp.array([T_eV]))


def _rates(z, species):
    vpar, mu = z
    v = jnp.sqrt(vpar**2 + 2 * mu * B0 / M)
    return [f(M, Q, 0, v, X0, species) for f in (nu_s_ab, nu_D_ab, nu_par_ab)]


def _D_reference(z, species):
    """Diffusion tensor in physical (vpar, mu), built from the velocity-space tensor."""
    vpar, mu = z
    vperp = jnp.sqrt(2 * mu * B0 / M)
    vec = jnp.array([vperp, 0.0, vpar])
    v = jnp.linalg.norm(vec)
    _, nu_D, nu_par = _rates(z, species)
    vhat = vec / v
    P = jnp.outer(vhat, vhat)
    Dv = (v**2 * nu_par / 2 * P + v**2 * nu_D / 2 * (jnp.eye(3) - P))
    J = jnp.array([[0.0, 0.0, 1.0], [M * vperp / B0, 0.0, 0.0]])  # d(vpar, mu)/d(vx, vy, vz)
    return J @ Dv @ J.T


def _ito_reference(z, species):
    nu_s = _rates(z, species)[0]
    K = -nu_s * jnp.array([z[0], 2 * z[1]])
    divD = jnp.einsum('ijj->i', jax.jacfwd(_D_reference)(z, species))
    return K + divD


_SCALE = jnp.array([SPEED_OF_LIGHT, SPEED_OF_LIGHT**2 * M])


def _code(fn, z, species):
    state = jnp.concatenate([X0, z / _SCALE])
    out = fn(0.0, state, (FIELD, PARTICLE, EFIELD, species, 1.0))
    return out[3:] * _SCALE


def _code_sigma(z, species):
    state = jnp.concatenate([X0, z / _SCALE])
    out = GuidingCenterCollisionsDiffusionMu(0.0, state, (FIELD, PARTICLE, EFIELD, species, 1.0))
    return out[3:, 3:] * _SCALE[:, None]


POINTS = [(v, xi, T) for v in (3e5, 2e6) for xi in (-0.7, 0.3) for T in (100.0, 2000.0)]


def _z(v, xi):
    return jnp.array([xi * v, M * v**2 * (1 - xi**2) / (2 * B0)])


@pytest.mark.parametrize("v,xi,T", POINTS)
def test_diffusion_matrix_matches_velocity_space_tensor(v, xi, T):
    species, z = _species(T), _z(v, xi)
    sigma = _code_sigma(z, species)
    np.testing.assert_allclose(sigma @ sigma.T / 2, _D_reference(z, species), rtol=1e-9,
                               atol=1e-12 * np.abs(_D_reference(z, species)).max())


@pytest.mark.parametrize("v,xi,T", POINTS)
def test_ito_drift_matches_divergence_of_diffusion_tensor(v, xi, T):
    species, z = _species(T), _z(v, xi)
    expected = _ito_reference(z, species)
    np.testing.assert_allclose(_code(GuidingCenterCollisionsDriftMuIto, z, species), expected,
                               rtol=1e-8, atol=1e-10 * np.abs(expected).max())


@pytest.mark.parametrize("v,xi,T", POINTS)
def test_stratonovich_drift_is_ito_minus_noise_induced_drift(v, xi, T):
    species, z = _species(T), _z(v, xi)
    sigma = _code_sigma(z, species)
    dsigma = jax.jacfwd(_code_sigma)(z, species)  # [i, k, j] = d sigma_ik / d z_j
    expected = _ito_reference(z, species) - 0.5 * jnp.einsum('jk,ikj->i', sigma, dsigma)
    np.testing.assert_allclose(_code(GuidingCenterCollisionsDriftMuStratonovich, z, species), expected,
                               rtol=1e-7, atol=1e-9 * np.abs(expected).max())


@pytest.mark.parametrize("v,xi", [(3e5, -0.7), (6e5, 0.3), (2e6, 0.5)])
def test_maxwellian_at_background_temperature_has_zero_flux(v, xi):
    T = 1000.0
    species, z = _species(T), _z(v, xi)
    T_J = T * JOULE_PER_EV

    def f(z):
        return jnp.exp(-(M * z[0]**2 / 2 + z[1] * B0) / T_J)

    def Df(z):
        sigma = _code_sigma(z, species)
        return sigma @ sigma.T / 2 * f(z)

    advective = _code(GuidingCenterCollisionsDriftMuIto, z, species) * f(z)
    flux = advective - jnp.einsum('ijj->i', jax.jacfwd(Df)(z))
    np.testing.assert_allclose(flux, 0.0, atol=1e-8 * np.abs(advective).max())


def test_cold_background_speed_drift_is_slowing_down():
    # T_b -> 0: energy diffusion vanishes and pitch scattering conserves speed,
    # so the Ito drift of v = |v| reduces to -nu_s v.
    species, z = _species(1e-2), _z(2e6, 0.4)
    A = _code(GuidingCenterCollisionsDriftMuIto, z, species)
    sigma = _code_sigma(z, species)

    def speed(z):
        return jnp.sqrt(z[0]**2 + 2 * z[1] * B0 / M)

    ito_v = jax.grad(speed)(z) @ A + 0.5 * jnp.sum(jax.hessian(speed)(z) * (sigma @ sigma.T))
    nu_s = _rates(z, species)[0]
    np.testing.assert_allclose(ito_v, -nu_s * speed(z), rtol=1e-6)
