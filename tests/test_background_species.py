import jax
import jax.numpy as jnp
import numpy as np
import pytest
from essos.background_species import BackgroundSpecies, coulomb_logarithm, d_nu_D_ab, nu_D_ab, nu_s_ab
from essos.constants import ELECTRON_MASS, PROTON_MASS, ELEMENTARY_CHARGE, EPSILON_0

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
