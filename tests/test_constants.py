import pytest
from essos.constants import PROTON_MASS, NEUTRON_MASS, ELEMENTARY_CHARGE, ONE_EV, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY, ELECTRON_MASS

def test_proton_mass():
    assert PROTON_MASS == 1.67262192369e-27

def test_electron_mass():
    assert ELECTRON_MASS == 9.1093837139e-31

def test_neutron_mass():
    assert NEUTRON_MASS == 1.67492749804e-27

def test_elementary_charge():
    assert ELEMENTARY_CHARGE == 1.602176634e-19

def test_one_ev():
    assert ONE_EV == 1.602176634e-19

def test_alpha_particle_mass():
    expected_mass = 2 * PROTON_MASS + 2 * NEUTRON_MASS
    assert ALPHA_PARTICLE_MASS == expected_mass

def test_alpha_particle_charge():
    expected_charge = 2 * ELEMENTARY_CHARGE
    assert ALPHA_PARTICLE_CHARGE == expected_charge

def test_fusion_alpha_particle_energy():
    expected_energy = 3.52e6 * ONE_EV
    assert FUSION_ALPHA_PARTICLE_ENERGY == expected_energy

if __name__ == "__main__":
    pytest.main()
