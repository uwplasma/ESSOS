import pytest
import jax.numpy as jnp
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ELECTRON_MASS,PROTON_MASS
from essos.dynamics import Particles, GuidingCenter, Lorentz, FieldLine, Tracing
from essos.background_species import BackgroundSpecies

def test_particles_initialization_all_params():
    nparticles = 100
    initial_xyz = jnp.array([[1.0, 0.0, 0.0]] * nparticles)
    initial_vparallel_over_v = jnp.linspace(-1, 1, nparticles)
    charge = ALPHA_PARTICLE_CHARGE
    mass = ALPHA_PARTICLE_MASS
    energy = FUSION_ALPHA_PARTICLE_ENERGY

    particles = Particles(initial_xyz, initial_vparallel_over_v, charge, mass, energy)

    assert particles.nparticles == nparticles
    assert particles.charge == charge
    assert particles.mass == mass
    assert particles.energy == energy
    assert jnp.allclose(particles.initial_xyz, initial_xyz)
    assert jnp.allclose(particles.initial_vparallel_over_v, initial_vparallel_over_v)

def test_particles_initialization_default_params():
    nparticles = 100
    particles = Particles(jnp.array([[1.0, 0.0, 0.0]] * nparticles))

    assert particles.nparticles == nparticles
    assert particles.charge == ALPHA_PARTICLE_CHARGE
    assert particles.mass == ALPHA_PARTICLE_MASS
    assert particles.energy == FUSION_ALPHA_PARTICLE_ENERGY
    assert particles.initial_xyz.shape == (nparticles, 3)
    assert particles.initial_vparallel_over_v.shape == (nparticles,)

def test_particles_initialization_with_initial_conditions():
    initial_xyz = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    initial_vparallel_over_v = jnp.array([0.5, -0.5])
    particles = Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=initial_vparallel_over_v)

    assert particles.nparticles == 2
    assert jnp.allclose(particles.initial_xyz, initial_xyz)
    assert jnp.allclose(particles.initial_vparallel_over_v, initial_vparallel_over_v)

def test_particles_computed_attributes():
    nparticles = 100
    particles = Particles(jnp.array([[1.0, 0.0, 0.0]] * nparticles))
    v = jnp.sqrt(2 * particles.energy / particles.mass)
    expected_vparallel = v * particles.initial_vparallel_over_v
    expected_vperpendicular = jnp.sqrt(v**2 - expected_vparallel**2)

    assert jnp.allclose(particles.initial_vparallel, expected_vparallel)
    assert jnp.allclose(particles.initial_vperpendicular, expected_vperpendicular)

class MockField:
    def B_covariant(self, points):
        return jnp.array([1.0, 0.0, 0.0])
    
    def B_contravariant(self, points):
        return jnp.array([1.0, 0.0, 0.0])
    
    def sqrtg(self,points):
        return 1.0
    
    def AbsB(self, points):
        return 1.0
    
    def dAbsB_by_dX(self, points):
        return jnp.array([0.0, 0.0, 1.0])
    
    
    def grad_B_covariant(self, points):
        return jnp.array([0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0])   
 

    def curl_B(self, points):
        return jnp.array([0.0,0.0,0.0])
    
    
    def curl_b(self, points):
        return jnp.array([0.0,0.0,0.0])

    def kappa(self, points):
        return jnp.array([0.0,0.0,0.0])


    def to_xyz(self, points):
        return points
    
class MockElectricField:
    def E_covariant(self, points):
        return jnp.array([0.0, 0.0, 0.0])
    

@pytest.fixture
def particles():
    return Particles(jnp.array([[1.0, 0.0, 0.0]] * 10))

@pytest.fixture
def field():
    return MockField()

@pytest.fixture
def electric_field():
    return MockElectricField()

def test_particles_initialization(particles):
    assert particles.nparticles == 10
    assert particles.charge == ALPHA_PARTICLE_CHARGE
    assert particles.mass == ALPHA_PARTICLE_MASS
    assert particles.energy == FUSION_ALPHA_PARTICLE_ENERGY
    assert particles.initial_xyz.shape == (10, 3)
    assert particles.initial_vparallel.shape == (10,)
    assert particles.initial_vperpendicular.shape == (10,)

def test_guiding_center(field, particles,electric_field):
    initial_conditions = jnp.array([1.0, 0.0, 0.0, 1])
    t = 0.0
    result = GuidingCenter(t, initial_conditions, (field, particles,electric_field))
    assert result.shape == (4,)

def test_lorentz(field, particles):
    initial_condition = jnp.array([1.0, 0.0, 0.0, 0.1, 0.1, 0.1])
    t = 0.0
    result = Lorentz(t, initial_condition, (field, particles))
    assert result.shape == (6,)

def test_field_line(field):
    initial_condition = jnp.array([1.0, 0.0, 0.0])
    t = 0.0
    result = FieldLine(t, initial_condition, field)
    assert result.shape == (3,)

def test_tracing_initialization(field, particles,electric_field):
    x = jnp.linspace(1, 2, particles.nparticles)
    y = jnp.zeros(particles.nparticles)
    z = jnp.zeros(particles.nparticles)
    initial_conditions =jnp.array([x, y, z]).T
    tracing = Tracing(initial_conditions=initial_conditions, field=field,electric_field=electric_field, model='GuidingCenter', particles=particles, times_to_trace=200)
    assert tracing.field == field
    assert tracing.model == 'GuidingCenter'
    assert tracing.initial_conditions.shape == (particles.nparticles, 4)
    assert tracing.times.shape == (200,)

def test_tracing_trace(field, particles,electric_field):
    x = jnp.linspace(1, 2, particles.nparticles)
    y = jnp.zeros(particles.nparticles)
    z = jnp.zeros(particles.nparticles)
    initial_conditions =jnp.array([x, y, z]).T
    tracing = Tracing(initial_conditions=initial_conditions, field=field,electric_field=electric_field, model='GuidingCenter', particles=particles, times_to_trace=200)
    trajectories = tracing.trace()
    assert trajectories.shape == (particles.nparticles, 200, 4)

def test_tracing_trace_adaptative(field, particles,electric_field):
    x = jnp.linspace(1, 2, particles.nparticles)
    y = jnp.zeros(particles.nparticles)
    z = jnp.zeros(particles.nparticles)
    initial_conditions =jnp.array([x, y, z]).T
    tracing = Tracing(initial_conditions=initial_conditions, field=field,electric_field=electric_field, model='GuidingCenterAdaptative', particles=particles, times_to_trace=200)
    trajectories = tracing.trace()
    assert trajectories.shape == (particles.nparticles, 200, 4)


def test_tracing_trace_collisions_fixed(field, particles,electric_field):
    x = jnp.linspace(1, 2, particles.nparticles)
    y = jnp.zeros(particles.nparticles)
    z = jnp.zeros(particles.nparticles)
    initial_conditions =jnp.array([x, y, z]).T
    #Initialize background species
    number_species=1  #(electrons,deuterium)
    mass_array=jnp.array([1.,ELECTRON_MASS/PROTON_MASS])    #mass_over_mproton
    charge_array=jnp.array([1.,-1])    #mass_over_mproton
    T0=1.e+3  #eV
    n0=1e+20  #m^-3
    n_array=jnp.array([n0,n0])
    T_array=jnp.array([T0,T0])
    species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)
    tracing = Tracing(initial_conditions=initial_conditions, field=field,electric_field=electric_field, model='GuidingCenterCollisionsMuFixed', particles=particles, times_to_trace=200,maxtime=1.e-6,species=species)
    trajectories = tracing.trace()
    assert species.mass.shape == (2,)
    assert species.charge.shape == (2,)
    assert trajectories.shape == (particles.nparticles, 200, 5)

def test_tracing_trace_collisions_ito(field, particles,electric_field):
    x = jnp.linspace(1, 2, particles.nparticles)
    y = jnp.zeros(particles.nparticles)
    z = jnp.zeros(particles.nparticles)
    initial_conditions =jnp.array([x, y, z]).T
    #Initialize background species
    number_species=1  #(electrons,deuterium)
    mass_array=jnp.array([1.,ELECTRON_MASS/PROTON_MASS])    #mass_over_mproton
    charge_array=jnp.array([1.,-1])    #mass_over_mproton
    T0=1.e+3  #eV
    n0=1e+20  #m^-3
    n_array=jnp.array([n0,n0])
    T_array=jnp.array([T0,T0])
    species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)
    tracing = Tracing(initial_conditions=initial_conditions, field=field,electric_field=electric_field, model='GuidingCenterCollisionsMuIto', particles=particles, times_to_trace=200,maxtime=1.e-6,species=species)
    trajectories = tracing.trace()
    assert species.mass.shape == (2,)
    assert species.charge.shape == (2,)
    assert trajectories.shape == (particles.nparticles, 200, 5)

def test_tracing_trace_collisions_adaptative(field, particles,electric_field):
    x = jnp.linspace(1, 2, particles.nparticles)
    y = jnp.zeros(particles.nparticles)
    z = jnp.zeros(particles.nparticles)
    initial_conditions =jnp.array([x, y, z]).T
    #Initialize background species
    number_species=1  #(electrons,deuterium)
    mass_array=jnp.array([1.,ELECTRON_MASS/PROTON_MASS])    #mass_over_mproton
    charge_array=jnp.array([1.,-1])    #mass_over_mproton
    T0=1.e+3  #eV
    n0=1e+20  #m^-3
    n_array=jnp.array([n0,n0])
    T_array=jnp.array([T0,T0])
    species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)
    tracing = Tracing(initial_conditions=initial_conditions, field=field,electric_field=electric_field, model='GuidingCenterCollisionsMuAdaptative', particles=particles, times_to_trace=200,maxtime=1.e-6,species=species)
    trajectories = tracing.trace()
    assert species.mass.shape == (2,)
    assert species.charge.shape == (2,)
    assert trajectories.shape == (particles.nparticles, 200, 5)

if __name__ == "__main__":
    pytest.main()