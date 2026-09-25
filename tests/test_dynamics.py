import pytest
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ELECTRON_MASS,PROTON_MASS
from essos.dynamics import (
    FieldLine,
    FieldLineArclength,
    FieldLineToroidal,
    GuidingCenter,
    Lorentz,
    Particles,
    Tracing,
    LevelsetStoppingCriterion,
    trace_field_lines,
    _axis_regular,
    _from_axis_regular,
    _to_axis_regular,
    _vmec_boundary_event,
    _VMEC_GUIDING_CENTER_MODELS,
)
from essos.background_species import BackgroundSpecies
from essos.fields import Vmec
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier

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


def test_arclength_fieldline_has_unit_speed_without_changing_direction():
    class ScaledField(MockField):
        def B_contravariant(self, points):
            return jnp.array([3.0, 4.0, 0.0])

    derivative = FieldLineArclength(0.0, jnp.zeros(3), ScaledField())
    assert jnp.allclose(derivative, jnp.array([0.6, 0.8, 0.0]))

    tracing = Tracing(
        field=ScaledField(), model="FieldLineArclength",
        initial_conditions=jnp.zeros((1, 3)), maxtime=2.0,
        timestep=0.1, times_to_trace=11)
    assert jnp.allclose(
        tracing.trajectories[0, -1], jnp.array([1.2, 1.6, 0.0]))


def test_toroidal_fieldline_uses_third_coordinate_as_parameter():
    class FluxField(MockField):
        def B_contravariant(self, points):
            return jnp.array([0.0, 2.0, 4.0])

        def toroidal_angle_batch(self, points):
            return points[:, 2]

    derivative = FieldLineToroidal(0.0, jnp.zeros(3), FluxField())
    assert jnp.allclose(derivative, jnp.array([0.0, 0.5, 1.0]))

    tracing = Tracing(
        field=FluxField(), model="FieldLineToroidal",
        initial_conditions=jnp.zeros((1, 3)), maxtime=2.0,
        timestep=0.1, times_to_trace=11)
    assert jnp.allclose(tracing.trajectories[0, -1], jnp.array([0.0, 1.0, 2.0]))
    assert jnp.allclose(tracing.toroidal_angles[0], tracing.trajectories[0, :, 2])


def test_trace_field_lines_selects_clear_physical_parameterizations(capsys):
    class FluxField(MockField):
        def B_contravariant(self, points):
            return jnp.array([0.0, 2.0, 4.0])

        def toroidal_angle_batch(self, points):
            return points[:, 2]

    arclength = trace_field_lines(
        MockField(), jnp.zeros((1, 3)), length=2.0, samples=11,
        tolerance=1.0e-8, progress=False, label="Cartesian test")
    toroidal = trace_field_lines(
        FluxField(), jnp.zeros((1, 3)), toroidal_turns=0.5, samples=11,
        tolerance=1.0e-8, progress=False, label=None)

    assert arclength.model == "FieldLineArclength"
    assert toroidal.model == "FieldLineToroidal"
    assert float(arclength.maxtime) == pytest.approx(2.0)
    assert float(toroidal.maxtime) == pytest.approx(float(jnp.pi))
    assert "Tracing Cartesian test" in capsys.readouterr().out


def test_trace_field_lines_reports_stops_and_uses_batched_coordinates(capsys):
    class BatchedField(MockField):
        def to_xyz_batch(self, points):
            return points

    class PlaneClassifier:
        def evaluate_xyz(self, xyz):
            return 0.2 - xyz[0]

    result = trace_field_lines(
        BatchedField(), jnp.zeros((1, 3)), length=1.0, samples=11,
        stopping_criteria=LevelsetStoppingCriterion(PlaneClassifier()),
        progress=False, label="bounded test")
    assert result.boundary_hits.tolist() == [True]
    assert "1/1 lines reached a stopping event" in capsys.readouterr().out


@pytest.mark.parametrize("kwargs", ({}, {"length": 1.0, "toroidal_turns": 1.0}))
def test_trace_field_lines_requires_one_extent(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        trace_field_lines(MockField(), jnp.zeros((1, 3)), progress=False, **kwargs)


@pytest.mark.parametrize("kwargs, message", (
    ({"length": 1.0, "samples": 1}, "samples"),
    ({"length": 0.0}, "length"),
    ({"toroidal_turns": 0.0}, "toroidal_turns"),
))
def test_trace_field_lines_validates_positive_extent_and_samples(kwargs, message):
    with pytest.raises(ValueError, match=message):
        trace_field_lines(MockField(), jnp.zeros((1, 3)), progress=False, **kwargs)


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


def test_axis_regular_chart_round_trip_and_boundary_event():
    state = jnp.array([0.25, 2.0, 0.3, 1.0, 0.5])
    regular = _to_axis_regular(state)
    assert regular.shape == (6,)
    assert jnp.allclose(regular[:2], 0.5 * jnp.array([jnp.cos(2.0), jnp.sin(2.0)]))
    assert jnp.allclose(_from_axis_regular(regular), state)
    assert jnp.allclose(_from_axis_regular(regular.at[-1].set(1.0))[1], 3.0)
    assert jnp.isinf(_from_axis_regular(jnp.full(6, jnp.inf))).all()
    for state_size in (5, 6):
        inside = jnp.zeros(state_size).at[:2].set(jnp.array([0.6, -0.7]))
        assert not _vmec_boundary_event(0.0, inside, None)
        assert _vmec_boundary_event(0.0, inside.at[1].set(-0.8), None)


def test_axis_regular_vector_field_maps_back_to_the_flux_field():
    def flux_field(t, y, args):
        s, theta = y[0], y[1]
        drift = jnp.array([s * jnp.sin(theta) + 0.3 * jnp.sqrt(s), 1.0 + jnp.cos(theta), 0.2, -0.1])
        return jnp.stack([drift, 2 * drift]).T  # a diffusion-like matrix

    for y in (jnp.array([0.3, -0.2, 0.1, 0.5, 0.7]), jnp.array([0.01, 0.02, 0.1, 0.5, -2.0])):
        regular = _axis_regular(flux_field)(0.0, y, None)
        pushed = jnp.stack([jax.jvp(_from_axis_regular, (y,), (column,))[1] for column in regular.T]).T
        assert jnp.allclose(pushed, flux_field(0.0, _from_axis_regular(y), None))
    on_axis = _axis_regular(flux_field)(0.0, jnp.array([0.0, 0.0, 0.1, 0.5, 0.0]), None)
    assert jnp.isfinite(on_axis).all()


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


def test_levelset_stopping_criterion_stops_and_fills_field_line():
    class PlaneClassifier:
        def evaluate_xyz(self, xyz):
            return 1.5 - xyz[0]

    criterion = LevelsetStoppingCriterion(PlaneClassifier(), maximum_distance=0.2)
    tracing = Tracing(
        field=MockField(), model="FieldLineAdaptative",
        initial_conditions=jnp.array([[1.0, 0.0, 0.0]]),
        maxtime=1.0, timestep=0.01, times_to_trace=21,
        stopping_criteria=criterion,
    )

    assert tracing.boundary_hits.tolist() == [True]
    assert tracing.progress is False
    assert jnp.isfinite(tracing.trajectories).all()
    assert jnp.max(tracing.trajectories[0, :, 0]) <= 1.7 + 1e-8
    assert jnp.allclose(tracing.trajectories[0, -1], tracing.trajectories[0, -2])


def test_poincare_plot_unwraps_toroidal_crossings_and_accepts_line_colors():
    phase = jnp.linspace(0.0, 4.0 * jnp.pi, 101)
    first = jnp.stack((jnp.cos(phase), jnp.sin(phase), 0.1 * jnp.sin(phase)), axis=1)
    second = first.at[:, :2].multiply(1.1)
    tracing = Tracing.__new__(Tracing)
    tracing.times = phase
    tracing.trajectories_xyz = jnp.stack((first, second))

    figure, axis = plt.subplots()
    sections = tracing.poincare_plot(
        shifts=[0.0], ax=axis, show=False, color=["tab:blue", "tab:orange"])
    plt.close(figure)

    assert len(sections) == 2
    assert all(len(section[0]) == 2 for section in sections)
    assert all(jnp.allclose(section[1], 0.0, atol=1e-12) for section in sections)

    figure, axis = plt.subplots()
    z_sections = tracing.poincare_plot(
        shifts=[0.0], orientation="z", ax=axis, show=False, color="time")
    plt.close(figure)
    assert any(len(section[0]) > 0 for section in z_sections)

    with pytest.raises(ValueError, match="orientation"):
        tracing.poincare_plot(shifts=[0.0], orientation="x", show=False)


def test_poincare_plot_prefers_continuous_native_toroidal_angle():
    phase = jnp.linspace(0.0, 4.0 * jnp.pi, 101)
    # Deliberately give Cartesian points an unrelated azimuth: native flux
    # coordinates must define the section for a VMEC tracing adapter.
    trace = jnp.stack((1.0 + 0.001 * phase, jnp.zeros_like(phase),
                       0.001 * phase + 0.1 * jnp.sin(phase)), axis=1)
    tracing = Tracing.__new__(Tracing)
    tracing.times = phase
    tracing.trajectories_xyz = trace[None]
    tracing.toroidal_angles = phase[None]

    figure, axis = plt.subplots()
    sections = tracing.poincare_plot(shifts=[0.0], ax=axis, show=False)
    plt.close(figure)
    assert len(sections[0][0]) == 2


def test_levelset_stopping_criterion_validates_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        LevelsetStoppingCriterion(MockField(), maximum_distance=-0.1)
    with pytest.raises(TypeError, match="evaluate_xyz"):
        LevelsetStoppingCriterion(object())
    with pytest.raises(ValueError, match="condition or stopping_criteria"):
        Tracing(field=MockField(), model="FieldLine", initial_conditions=jnp.ones((1, 3)),
                condition=lambda *args: False, stopping_criteria=lambda *args: False)
    with pytest.raises(ValueError, match="callable criteria"):
        Tracing(field=MockField(), model="FieldLine", initial_conditions=jnp.ones((1, 3)),
                stopping_criteria=[])
    with pytest.raises(ValueError, match="at least one"):
        Tracing(field=MockField(), model="FieldLine", initial_conditions=jnp.ones((1, 3)),
                devices=[])


def test_surface_classifier_signed_distance_for_circular_torus():
    surface = SurfaceRZFourier(
        rc=jnp.array([1.0, 0.2]), zs=jnp.array([0.0, 0.2]),
        nfp=1, mpol=1, ntor=0, ntheta=16, nphi=16, close=False)
    classifier = SurfaceClassifier(surface, h=0.1, padding=0.4)
    assert classifier.evaluate_xyz(jnp.array([1.0, 0.0, 0.0])) > 0.0
    assert classifier.evaluate_xyz(jnp.array([1.5, 0.0, 0.0])) < 0.0
    with pytest.raises(ValueError, match="padding"):
        SurfaceClassifier(surface, h=0.1, padding=0.0)


def test_vmec_fieldline_uses_the_lcfs_event():
    tracing = Tracing(
        field=MockVmec(), model="FieldLineArclength",
        initial_conditions=jnp.array([[0.5, 0.0, 0.0]]),
        maxtime=0.01, timestep=0.001, times_to_trace=3)
    assert callable(tracing.condition)


WOUT_QA = str(Path(__file__).resolve().parents[1] / "examples" / "input_files"
              / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")


def _near_axis_particles():
    # 3.5 MeV alphas: four born at s = 0.01 whose orbits pass within s ~ 1e-4 of the
    # axis, and one born at s = 1e-7, next to it.
    theta = jnp.array([3, 4, 5, 0, 1]) * jnp.pi / 4
    xyz = jnp.stack([jnp.array([0.01, 0.01, 0.01, 0.01, 1e-7]), theta, jnp.zeros(5)], axis=1)
    return Particles(initial_xyz=xyz, initial_vparallel_over_v=jnp.array([0.9, 0.9, 0.9, -0.9, 0.9]))


def test_vmec_guiding_centers_cross_the_magnetic_axis():
    """The orbit born at s = 1e-7 used to stop at once, at s <= axis_threshold = 1e-6."""
    particles = _near_axis_particles()
    tracing = Tracing(field=Vmec(WOUT_QA, ntheta=8, nphi=8), model="GuidingCenterAdaptative",
                      particles=particles, maxtime=2e-5, timestep=1e-9, times_to_trace=200,
                      atol=1e-9, rtol=1e-9)
    s = tracing.trajectories[:, :, 0]
    assert not tracing.axis_hits.any() and not tracing.boundary_hits.any()
    assert jnp.isfinite(tracing.trajectories).all()
    assert (s.min(axis=1) < 1e-3).all()
    assert (s[:, -1] > 1.5 * s.min(axis=1)).all()
    assert jnp.all((tracing.trajectories[:, :, 1] >= 0) & (tracing.trajectories[:, :, 1] < 2 * jnp.pi))
    assert jnp.abs(tracing.energy() / particles.energy - 1).max() < 1e-5


def test_vmec_guiding_center_condition_sees_flux_coordinates():
    def below(t, y, args, **kwargs):
        return y[0] < 5e-3

    tracing = Tracing(field=Vmec(WOUT_QA, ntheta=8, nphi=8), model="GuidingCenterAdaptative",
                      particles=_near_axis_particles(), maxtime=2e-5, timestep=1e-9, times_to_trace=200,
                      atol=1e-9, rtol=1e-9, condition=below)
    s = tracing.trajectories[:, :, 0]
    stopped = ~jnp.isfinite(s[:, -1])
    assert stopped.all()
    assert (jnp.where(jnp.isfinite(s), s, 1.0).min(axis=1) > 4e-3).all()


@pytest.mark.parametrize("model", ["GuidingCenterCollisions", "GuidingCenterCollisionsMuFixed"])
def test_vmec_collisional_guiding_centers_cross_the_magnetic_axis(model):
    species = BackgroundSpecies(number_species=2, mass_array=jnp.array([ELECTRON_MASS / PROTON_MASS, 2.0]),
                                charge_array=jnp.array([-1.0, 1.0]), n_array=jnp.array([1e20, 1e20]),
                                T_array=jnp.array([1e4, 1e4]))
    tracing = Tracing(field=Vmec(WOUT_QA, ntheta=8, nphi=8), model=model, particles=_near_axis_particles(),
                      maxtime=2e-5, timestep=1e-8, times_to_trace=100, species=species)
    s = tracing.trajectories[:, :, 0]
    assert not tracing.axis_hits.any()
    assert jnp.isfinite(tracing.trajectories).all()
    assert (s.min(axis=1) < 2e-3).all()
    assert (s[:, -1] > 1.5 * s.min(axis=1)).all()


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


def test_tracing_max_steps_is_configurable_and_bounded():
    """The Diffrax ceiling used to be 1e10, so a trace that could not finish ran
    until the process was killed rather than returning."""
    import inspect

    from essos.dynamics import Tracing

    default = inspect.signature(Tracing).parameters["max_steps"].default
    assert default == 1_000_000

    source = inspect.getsource(Tracing)
    assert "max_steps=10000000000" not in source
    assert source.count("max_steps=self.max_steps") == 9

