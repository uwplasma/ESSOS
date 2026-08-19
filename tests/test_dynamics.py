import os
import pytest
import jax
import jax.numpy as jnp
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ELECTRON_MASS,PROTON_MASS
from essos.dynamics import (
    FieldLine,
    GuidingCenter,
    Lorentz,
    Particles,
    Tracing,
    _fill_terminated_trajectories,
    _vmec_radial_events,
    _VMEC_GUIDING_CENTER_MODELS,
)
from essos.background_species import BackgroundSpecies
from essos.fields import Vmec, VMEC_WOUT_ARRAYS

WOUT_FILE = os.path.join(os.path.dirname(__file__), "..", "examples", "input_files",
                         "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")

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


class MockVmec(MockField, Vmec):
    def __init__(self):
        pass

    def B_contravariant(self, points):
        return jnp.array([-1.0, 0.0, 0.0])

    def dAbsB_by_dX(self, points):
        return jnp.zeros(3)
    

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


def test_fill_terminated_trajectories():
    trajectories = jnp.array(
        [
            [[0.2, 1.0], [0.3, 2.0], [jnp.inf, jnp.inf]],
            [[0.4, 3.0], [0.5, 4.0], [0.6, 5.0]],
        ]
    )
    filled = _fill_terminated_trajectories(trajectories)
    assert jnp.allclose(filled[0, 2], trajectories[0, 1])
    assert jnp.allclose(filled[1], trajectories[1])


def test_fill_terminated_trajectories_rejects_below_axis_states():
    trajectories = jnp.array([[[0.2, 1.0], [0.1, 2.0], [-0.1, 3.0]]])
    filled = _fill_terminated_trajectories(trajectories, axis_threshold=0.0)
    assert jnp.allclose(filled[0, 2], trajectories[0, 1])


def test_vmec_radial_events_support_all_guiding_center_state_sizes():
    reached_axis, reached_boundary = _vmec_radial_events(1e-6)

    for state_size in (4, 5):
        state = jnp.zeros(state_size).at[0].set(0.5)
        assert not reached_axis(0.0, state, None)
        assert not reached_boundary(0.0, state, None)
        assert reached_axis(0.0, state.at[0].set(1e-6), None)
        assert reached_boundary(0.0, state.at[0].set(1.0), None)


def test_vmec_axis_events_cover_every_guiding_center_stepper():
    assert _VMEC_GUIDING_CENTER_MODELS == {
        "GuidingCenter",
        "GuidingCenterAdaptative",
        "GuidingCenterCollisions",
        "GuidingCenterCollisionsMuIto",
        "GuidingCenterCollisionsMuFixed",
        "GuidingCenterCollisionsMuAdaptative",
    }
    assert "FullOrbit" not in _VMEC_GUIDING_CENTER_MODELS
    assert "FullOrbitAdaptative" not in _VMEC_GUIDING_CENTER_MODELS
    assert "FullOrbit_Boris" not in _VMEC_GUIDING_CENTER_MODELS


@pytest.mark.parametrize("model", ["GuidingCenter", "GuidingCenterAdaptative"])
def test_vmec_axis_event_terminates_deterministic_steppers(model):
    particles = Particles(
        initial_xyz=jnp.array([[1e-2, 0.0, 0.0]]),
        initial_vparallel_over_v=jnp.array([1.0]),
    )
    tracing = Tracing(
        field=MockVmec(),
        model=model,
        particles=particles,
        maxtime=1e-9,
        timestep=1e-10,
        times_to_trace=20,
        axis_threshold=1e-6,
    )

    assert tracing.axis_hits.tolist() == [True]
    assert tracing.boundary_hits.tolist() == [False]
    assert tracing.total_particles_unresolved == 1
    assert tracing.total_particles_lost == 0
    assert jnp.isfinite(tracing.trajectories).all()
    assert jnp.all(tracing.trajectories[:, :, 0] > tracing.axis_threshold)


def radial_tracing(peaks, times=jnp.linspace(0.0, 1.0, 40)):
    """Tracing holding prescribed radial excursions, bypassing the ODE solve."""
    trajectories_r = 0.5 + (peaks[:, None] - 0.5) * jnp.sin(jnp.pi * times)[None, :]
    tracing = Tracing.__new__(Tracing)
    tracing.trajectories = jnp.stack([trajectories_r, jnp.zeros_like(trajectories_r), jnp.zeros_like(trajectories_r)], axis=-1)
    tracing.times = times
    return tracing

def test_soft_loss_fraction_converges_to_loss_fraction():
    tracing = radial_tracing(jnp.array([0.70, 0.93, 0.86, 0.99]))
    exact = tracing.loss_fraction(r_max=0.9)[0][-1]
    assert exact == 0.5

    errors = [abs(float(tracing.soft_loss_fraction(r_max=0.9, width=width) - exact)) for width in (0.02, 0.01, 0.005, 0.002)]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-3

def test_soft_loss_fraction_gradient_is_nonzero_where_loss_fraction_is_flat():
    peaks = jnp.array([0.70, 0.93, 0.86, 0.99])
    exact_gradient = jax.grad(lambda p: radial_tracing(p).loss_fraction(r_max=0.9)[0][-1])(peaks)
    soft_gradient = jax.grad(lambda p: radial_tracing(p).soft_loss_fraction(r_max=0.9, width=0.01))(peaks)

    assert jnp.all(exact_gradient == 0.0)
    assert jnp.all(soft_gradient[1:] > 0.0)

def vmec_alpha_tracing(field, nparticles=4, maxtime=4e-6, times_to_trace=10):
    theta = jnp.linspace(0, 2*jnp.pi, nparticles)
    phi = jnp.linspace(0, 2*jnp.pi/field.nfp, nparticles)
    particles = Particles(initial_xyz=jnp.array([0.85*jnp.ones(nparticles), theta, phi]).T, mass=ALPHA_PARTICLE_MASS,
                          charge=ALPHA_PARTICLE_CHARGE, energy=FUSION_ALPHA_PARTICLE_ENERGY, field=field)
    return Tracing(field=field, model='GuidingCenterAdaptative', particles=particles, maxtime=maxtime,
                   timestep=1e-8, times_to_trace=times_to_trace, atol=1e-5, rtol=1e-5)

def test_vmec_from_arrays_traces_identically():
    vmec = Vmec(WOUT_FILE)
    rebuilt = Vmec.from_arrays(nfp=vmec.nfp, ns=vmec.ns, **{name: getattr(vmec, name) for name in VMEC_WOUT_ARRAYS})

    assert jnp.array_equal(vmec_alpha_tracing(rebuilt).trajectories, vmec_alpha_tracing(vmec).trajectories)

def test_soft_loss_fraction_differentiates_vmec_coefficients():
    vmec = Vmec(WOUT_FILE)
    arrays = {name: getattr(vmec, name) for name in VMEC_WOUT_ARRAYS}
    scaled = ('bmnc', 'bsubsmns', 'bsubumnc', 'bsubvmnc', 'bsupumnc', 'bsupvmnc')

    def soft_loss_of_field_scale(scale):
        field = Vmec.from_arrays(nfp=vmec.nfp, ns=vmec.ns,
                                 **{**arrays, **{name: arrays[name]*scale for name in scaled}})
        return vmec_alpha_tracing(field).soft_loss_fraction(r_max=0.88, width=0.01)

    assert jax.grad(soft_loss_of_field_scale)(1.0) != 0.0


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
