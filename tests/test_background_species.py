import jax
import jax.numpy as jnp
import numpy as np
import pytest
from essos.background_species import BackgroundSpecies, coulomb_logarithm, d_nu_D_ab, nu_D_ab, nu_s_ab
from essos.constants import ELECTRON_MASS, PROTON_MASS, ELEMENTARY_CHARGE, EPSILON_0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_chandrasekhar_matches_positive_integral_at_zero_and_small_speed(dtype):
    """G(x)=2x/sqrt(pi) int_0^1 t² exp(-x²t²) dt has a regular origin."""
    from scipy.integrate import quad
    from essos.background_species import chandrasekhar, d_chandrasekhar

    tolerance = 3e-5 if dtype == jnp.float32 else 2e-11
    for value in (0., 1e-10, 1e-8, 1e-6, .0999, .1001, .2499, .2501, 1., 10., -1e-8, -.2501):
        x = jnp.asarray(value, dtype=dtype)
        value = float(x)
        reference = 2*value/np.sqrt(np.pi)*quad(lambda t: t*t*np.exp(-value*value*t*t), 0, 1)[0]
        derivative = 2/np.sqrt(np.pi)*quad(lambda t: t*t*(1-2*value*value*t*t)*np.exp(-value*value*t*t), 0, 1)[0]
        assert float(chandrasekhar(x)) == pytest.approx(reference, rel=tolerance, abs=1e-30)
        assert float(d_chandrasekhar(x)) == pytest.approx(derivative, rel=tolerance)
        assert float(jax.grad(chandrasekhar)(x)) == pytest.approx(derivative, rel=tolerance)
    for value in (100., 1e4, 1e8):
        x = jnp.asarray(value, dtype=dtype)
        assert float(chandrasekhar(x)) == pytest.approx(1/(2*value**2), rel=tolerance)
        assert float(d_chandrasekhar(x)) == pytest.approx(-1/value**3, rel=tolerance, abs=1e-30)
        assert float(jax.grad(chandrasekhar)(x)) == pytest.approx(-1/value**3, rel=tolerance, abs=1e-30)

MASS = jnp.array([ELECTRON_MASS / PROTON_MASS, 1.0])
CHARGE = jnp.array([-1.0, 1.0])
S = jnp.linspace(0, 1, 11)
N = jnp.stack([2e19 * (1 - 0.9 * S**2), 1.8e19 * (1 - 0.9 * S**2)])
T = jnp.stack([300 * (1 - 0.95 * S), 150 * (1 - 0.9 * S)])


def local(point):
    """Constant-profile species with the values of the profiles at ``point``."""
    return BackgroundSpecies(2, MASS, CHARGE, jnp.interp(point[0], S, N[0]) * jnp.array([1.0, 0.9]),
                             jnp.array([jnp.interp(point[0], S, T[0]), jnp.interp(point[0], S, T[1])]))


def test_profiles_are_interpolated_in_the_first_coordinate():
    species = BackgroundSpecies(2, MASS, CHARGE, N, T, radial_grid=S)
    point = jnp.array([0.37, 1.0, 2.0])
    assert species.get_density(1, point) == pytest.approx(float(jnp.interp(0.37, S, N[1])))
    assert species.get_temperature(0, point) == pytest.approx(float(jnp.interp(0.37, S, T[0])))
    assert species.get_v_thermal(1, point) == pytest.approx(float(local(point).get_v_thermal(1, point)))
    assert species.get_temperature(0, jnp.array([1.2, 0.0, 0.0])) == pytest.approx(float(T[0, -1]))


def test_collision_frequencies_follow_the_local_profiles():
    species = BackgroundSpecies(2, MASS, CHARGE, N, T, radial_grid=S)
    v = 1.9e6  # 20 keV proton
    for s in (0.05, 0.6):
        point = jnp.array([s, 0.3, 0.1])
        for b in (0, 1):
            for nu in (nu_s_ab, nu_D_ab):
                assert nu(PROTON_MASS, ELEMENTARY_CHARGE, b, v, point, species) == pytest.approx(
                    float(nu(PROTON_MASS, ELEMENTARY_CHARGE, b, v, point, local(point))), rel=1e-12)
    core, edge = (nu_s_ab(PROTON_MASS, ELEMENTARY_CHARGE, 0, v, jnp.array([s, 0.0, 0.0]), species) for s in (0.05, 0.6))
    assert core != pytest.approx(float(edge), rel=0.1)


def test_constant_species_ignore_the_position():
    species = BackgroundSpecies(2, MASS, CHARGE, jnp.array([1e20, 1e20]), jnp.array([1e3, 2e3]))
    for point in (jnp.array([0.1, 0.0, 0.0]), jnp.array([0.9, 1.0, 2.0])):
        assert species.get_density(0, point) == 1e20
        assert species.get_temperature(1, point) == 2e3


@pytest.mark.parametrize("mass,charge", [(ELECTRON_MASS, -1.), (PROTON_MASS, 1.), (2*PROTON_MASS, 2.)])
def test_deflection_rate_matches_rutherford_pitch_decay(mass, charge):
    """Integrate nv 2pi b db (1-cos chi), chi=2 arctan(b90/b), for cold heavy ions."""
    species = BackgroundSpecies(2, MASS.at[1].set(1e6), CHARGE,
                                jnp.array([1e19, 1e19]), jnp.array([100., 1e-8]))
    point = jnp.zeros(3)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    for speed in (2e5, 1e6, 4e6):
        ln = float(coulomb_logarithm(mass, charge*ELEMENTARY_CHARGE, 1, speed, point, species))
        b90 = abs(charge)*ELEMENTARY_CHARGE**2/(4*np.pi*EPSILON_0*mass*speed**2)
        b = 100*b90*np.exp(ln*(nodes+1)/2)
        rate = 1e19*speed*4*np.pi*b90**2*np.sum(weights*ln/2*b*b/(b*b+b90*b90))
        actual = float(nu_D_ab(mass, charge*ELEMENTARY_CHARGE, 1, speed, point, species))
        assert actual == pytest.approx(rate, rel=5e-6)


def test_deflection_rate_matches_maxwellian_flow_decay():
    """NRL Lorentz flow rate, 4 sqrt(2pi)/3 n e^4 lnL / [(4pi eps0)^2 sqrt(m) T^1.5]."""
    from scipy.special import roots_genlaguerre
    species = BackgroundSpecies(2, MASS.at[1].set(1e6), CHARGE,
                                jnp.array([1e19, 1e19]), jnp.array([100., 1e-8]))
    point = jnp.zeros(3)
    ln = float(coulomb_logarithm(ELECTRON_MASS, -ELEMENTARY_CHARGE, 1, 1., point, species))
    temperature = 100*ELEMENTARY_CHARGE
    u, weights = roots_genlaguerre(64, 0.)
    speeds = jnp.asarray(np.sqrt(2*temperature/ELECTRON_MASS*u))
    rates = np.asarray(jax.vmap(lambda v: nu_D_ab(ELECTRON_MASS, -ELEMENTARY_CHARGE, 1, v, point, species))(speeds))
    flow = 4/(3*np.sqrt(np.pi))*np.sum(weights*u**1.5*rates)
    reference = 4*np.sqrt(2*np.pi)/3*1e19*ELEMENTARY_CHARGE**4*ln/((4*np.pi*EPSILON_0)**2*np.sqrt(ELECTRON_MASS)*temperature**1.5)
    assert flow == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize("b", [0, 1])
def test_deflection_derivative_matches_autodiff_and_finite_difference(b):
    species = BackgroundSpecies(2, MASS, CHARGE, jnp.array([1e19, 1e19]), jnp.array([100., 150.]))
    point = jnp.zeros(3)
    rate = lambda v: nu_D_ab(PROTON_MASS, ELEMENTARY_CHARGE, b, v, point, species)
    for speed in (1e4, 2e5, 2e6, 1e7):
        derivative = float(d_nu_D_ab(PROTON_MASS, ELEMENTARY_CHARGE, b, speed, point, species))
        assert derivative == pytest.approx(float(jax.grad(rate)(speed)), rel=1e-8)
        h = speed*1e-4
        finite = float((rate(speed+h)-rate(speed-h))/(2*h))
        assert derivative == pytest.approx(finite, rel=1e-6)


@pytest.mark.parametrize("mass,z,background,T,n", [
    (ELECTRON_MASS, -1., 0, 2., 1e16),
    (PROTON_MASS, 1., 1, 100., 1e19),
    (2*PROTON_MASS, 1., 0, 1e4, 1e21),
    (4*PROTON_MASS, 2., 2, 1e3, 1e18),
])
def test_deflection_matches_maxwellian_integral_across_species_and_speeds(mass, z, background, T, n):
    """Helander et al. (2017): integrate H(x)=2x/sqrt(pi) int_0^1 (1-t²) exp(-x²t²) dt.

    This positive integral independently checks both the warm and cold limits,
    avoiding the implementation's subtraction of erf and Chandrasekhar terms.
    """
    from scipy.integrate import quad
    from essos.background_species import JOULE_PER_EV

    masses = np.array([ELECTRON_MASS, PROTON_MASS, 12*PROTON_MASS])
    charges = np.array([-1., 1., 6.])
    species = BackgroundSpecies(3, jnp.asarray(masses/PROTON_MASS), jnp.asarray(charges),
                                jnp.full(3, n), jnp.full(3, T))
    point = jnp.zeros(3)
    vth = np.sqrt(2*T*JOULE_PER_EV/masses[background])
    ln = float(coulomb_logarithm(mass, z*ELEMENTARY_CHARGE, background, vth, point, species))
    prefactor = n*(z*charges[background]*ELEMENTARY_CHARGE**2)**2*ln/(4*np.pi*EPSILON_0**2*mass**2)
    for x in (1e-3, .02, .5, 2., 8., 100.):
        h = 2*x/np.sqrt(np.pi)*quad(lambda t: (1-t*t)*np.exp(-x*x*t*t), 0, 1, epsabs=1e-13)[0]
        dh = 2/np.sqrt(np.pi)*quad(lambda t: (1-t*t)*(1-2*x*x*t*t)*np.exp(-x*x*t*t),
                                 0, 1, epsabs=1e-13)[0]
        v = x*vth
        rate = nu_D_ab(mass, z*ELEMENTARY_CHARGE, background, v, point, species)
        derivative = d_nu_D_ab(mass, z*ELEMENTARY_CHARGE, background, v, point, species)
        assert rate == pytest.approx(prefactor*h/v**3, rel=1e-9)
        assert derivative == pytest.approx(prefactor*(x*dh-3*h)/v**4, rel=1e-9)
        assert nu_D_ab(mass, -z*ELEMENTARY_CHARGE, background, v, point, species) == rate


@pytest.mark.parametrize("densities", [(0., 0.), (1e19, 0.), (0., 1e19)])
def test_absent_background_species_have_zero_deflection_and_derivative(densities):
    species = BackgroundSpecies(2, MASS, CHARGE, jnp.asarray(densities), jnp.array([100., 150.]))
    for b, density in enumerate(densities):
        if density == 0:
            assert nu_D_ab(PROTON_MASS, ELEMENTARY_CHARGE, b, 1e6, jnp.zeros(3), species) == 0
            assert d_nu_D_ab(PROTON_MASS, ELEMENTARY_CHARGE, b, 1e6, jnp.zeros(3), species) == 0
