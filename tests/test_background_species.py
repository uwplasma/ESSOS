import jax.numpy as jnp
import pytest
from essos.background_species import BackgroundSpecies, nu_D_ab, nu_s_ab
from essos.constants import ELECTRON_MASS, PROTON_MASS, ELEMENTARY_CHARGE

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
