import os
import pytest
import numpy as np
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
    connection_length,
    _axis_regular,
    _from_axis_regular,
    _to_axis_regular,
    _vmec_boundary_event,
    _VMEC_GUIDING_CENTER_MODELS,
)
from essos.background_species import BackgroundSpecies
from essos.fields import Vmec, VMEC_WOUT_ARRAYS
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier

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


def test_tracing_is_differentiable_in_its_initial_conditions():
    """Tracing inside jax.grad, as in the coil-optimization losses, must not move
    traced initial conditions to the host."""
    class ScaledField(MockField):
        def B_contravariant(self, points):
            return jnp.array([3.0, 4.0, 0.0])

    def end_x(x0):
        tracing = Tracing(field=ScaledField(), model="FieldLineArclength", initial_conditions=x0,
                          maxtime=2.0, timestep=0.1, times_to_trace=11)
        return tracing.trajectories[0, -1, 0]

    assert jnp.allclose(jax.grad(end_x)(jnp.zeros((1, 3))), jnp.array([[1.0, 0.0, 0.0]]))


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


def radial_tracing(peaks, times=jnp.linspace(0.0, 1.0, 40)):
    """Tracing holding prescribed radial excursions, bypassing the ODE solve."""
    trajectories_r = 0.5 + (peaks[:, None] - 0.5) * jnp.sin(jnp.pi * times)[None, :]
    tracing = Tracing.__new__(Tracing)
    tracing.trajectories = jnp.stack([trajectories_r, jnp.zeros_like(trajectories_r), jnp.zeros_like(trajectories_r)], axis=-1)
    tracing.times = times
    tracing.field = Vmec.__new__(Vmec)
    tracing.model = "GuidingCenterAdaptative"
    tracing._has_boundary_event = True
    tracing.boundary_hits = jnp.zeros(len(peaks), dtype=bool)
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

@pytest.mark.parametrize("width", [0.0, -0.02, float("nan"), float("inf")])
def test_soft_loss_rejects_invalid_width(width):
    with pytest.raises(ValueError, match="width"):
        radial_tracing(jnp.array([0.8])).soft_loss_fraction(width=width)


def test_soft_loss_distinguishes_boundary_stop_from_failure():
    trace = radial_tracing(jnp.array([0.8]))
    trace.trajectories = trace.trajectories.at[0, -1, 1].set(jnp.nan)
    assert jnp.isnan(trace.soft_loss_fraction())
    trace.boundary_hits = jnp.array([True])
    assert jnp.isfinite(trace.soft_loss_fraction())
    assert trace.soft_loss_fraction(width=0.001) > 0.99


@pytest.mark.parametrize("model, field", [("Lorentz", Vmec.__new__(Vmec)), ("GuidingCenterAdaptative", object())])
def test_soft_loss_rejects_nonflux_trajectories(model, field):
    trace = radial_tracing(jnp.array([0.8]))
    trace.model, trace.field = model, field
    with pytest.raises(ValueError, match="VMEC guiding"):
        trace.soft_loss_fraction()


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

    evaluate = jax.jit(jax.value_and_grad(soft_loss_of_field_scale))
    for scale in (1.0, 1.01):
        value, gradient = evaluate(scale)
        assert jnp.isfinite(value) and jnp.isfinite(gradient) and gradient != 0.0


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



class _UniformField:
    def __init__(self, direction, magnitude=2.5):
        direction = jnp.asarray(direction, dtype=float)
        self.vector = magnitude * direction / jnp.linalg.norm(direction)

    def B_contravariant(self, xyz):
        return self.vector


def _legacy_positive_charge_start(field, xyz, vpar, total_speed, mass, charge, phase):
    """The pre-fix construction, correct for a positive charge and B not along z."""
    b = field.B_contravariant(xyz) / jnp.linalg.norm(field.B_contravariant(xyz))
    p2 = jnp.array([0.0, 0.0, 1.0])
    p3 = -jnp.cross(b, p2)
    p3 /= jnp.linalg.norm(p3)
    q2 = p2 - jnp.dot(b, p2) * b
    q2 /= jnp.linalg.norm(q2)
    q3 = p3 - jnp.dot(b, p3) * b - jnp.dot(q2, p3) * q2
    q3 /= jnp.linalg.norm(q3)
    speed_perp = jnp.sqrt(total_speed**2 - vpar**2)
    rg = mass * speed_perp / (abs(charge) * jnp.linalg.norm(field.B_contravariant(xyz)))
    return xyz + rg * (jnp.sin(phase) * q2 + jnp.cos(phase) * q3)


@pytest.mark.parametrize("charge_sign", [1.0, -1.0])
@pytest.mark.parametrize("direction", [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.3, -0.2, 0.9), (1.0, 0.0, 0.0)])
@pytest.mark.parametrize("phase", [0.0, 1.1, 2.9])
def test_gc_to_fullorbit_recovers_the_guiding_center(charge_sign, direction, phase):
    from essos.dynamics import gc_to_fullorbit

    field = _UniformField(direction)
    mass, charge, total_speed = 6.64e-27, charge_sign * 3.2e-19, 1.3e7
    centers = jnp.asarray([[1.0, 0.2, -0.3], [0.5, -1.0, 0.25]])
    vpar = jnp.asarray([0.4e7, -0.9e7])
    positions, velocities = gc_to_fullorbit(field, centers, vpar, total_speed, mass, charge, phase)
    b = field.vector / jnp.linalg.norm(field.vector)
    omega = charge * jnp.linalg.norm(field.vector) / mass
    implied = positions - jnp.cross(b, velocities) / omega
    assert jnp.all(jnp.isfinite(positions)) and jnp.all(jnp.isfinite(velocities))
    np.testing.assert_allclose(implied, centers, rtol=0, atol=1e-12)
    np.testing.assert_allclose(velocities @ b, vpar, rtol=1e-12)
    np.testing.assert_allclose(jnp.linalg.norm(velocities, axis=1), total_speed, rtol=1e-12)
    if charge_sign > 0 and abs(b[2]) < 0.9:
        legacy = jnp.stack([
            _legacy_positive_charge_start(field, x, v, total_speed, mass, charge, phase)
            for x, v in zip(centers, vpar)
        ])
        np.testing.assert_allclose(positions, legacy, rtol=0, atol=1e-13)


def test_particles_honours_the_full_orbit_phase_argument():
    from essos.dynamics import Particles

    particles = Particles(initial_xyz=jnp.zeros((2, 3)), phase_angle_full_orbit=0.7)
    assert particles.phase_angle_full_orbit == 0.7


class HelicalSlabField:
    """B = (-y, x, pitch): helices of constant radius between walls z = +-1."""

    def __init__(self, pitch):
        self.pitch = pitch

    def B_contravariant(self, xyz):
        return jnp.array([-xyz[1], xyz[0], self.pitch])


def test_connection_length_matches_helical_slab_closed_form():
    pitch, max_length = 0.5, 20.0
    seeds = jnp.array([[1.0, 0.0, 0.2], [0.0, 0.5, -0.6], [12.0, 0.0, 0.0]])
    result = connection_length(HelicalSlabField(pitch), seeds,
                               lambda xyz: 1.0 - xyz[2] ** 2, max_length=max_length)
    r = jnp.hypot(seeds[:, 0], seeds[:, 1])
    speed = jnp.hypot(r, pitch)
    for direction, sign in enumerate((1.0, -1.0)):
        s = (1.0 - sign * seeds[:, 2]) * speed / pitch
        angle = jnp.arctan2(seeds[:, 1], seeds[:, 0]) + sign * s / speed
        expected = jnp.stack([r * jnp.cos(angle), r * jnp.sin(angle),
                              jnp.full_like(r, sign)], axis=1)
        capped = s > max_length
        assert jnp.allclose(result["hit"][:, direction], ~capped)
        assert jnp.allclose(result["lengths"][:, direction],
                            jnp.minimum(s, max_length), rtol=1e-7)
        assert jnp.allclose(result["strike_points"][~capped, direction],
                            expected[~capped], atol=1e-7)
    assert jnp.allclose(result["connection_length"], result["lengths"].sum(axis=1))


def test_connection_length_requires_positive_cap():
    with pytest.raises(ValueError, match="max_length"):
        connection_length(HelicalSlabField(1.0), jnp.zeros((1, 3)), lambda x: 1.0, max_length=0.0)


def test_connection_length_flags_outside_seeds_and_step_exhaustion():
    wall = lambda xyz: 1.0 - xyz[2] ** 2  # noqa: E731
    result = connection_length(HelicalSlabField(0.5), jnp.array([[1.0, 0.0, 1.5]]), wall, max_length=20.0)
    assert jnp.all(result["lengths"] == 0.0) and jnp.all(result["hit"])
    result = connection_length(HelicalSlabField(0.5), jnp.array([[1.0, 0.0, 0.2]]), wall,
                               max_length=20.0, max_steps=3)
    assert jnp.all(jnp.isnan(result["lengths"])) and not jnp.any(result["hit"])


class _UniformCartesianField:
    def __init__(self, direction, magnitude=1.0):
        direction = jnp.asarray(direction, dtype=float)
        self.vector = magnitude * direction / jnp.linalg.norm(direction)

    def B_contravariant(self, xyz):
        return self.vector

    def to_xyz(self, point):
        return point

    def AbsB(self, point):
        return jnp.linalg.norm(self.vector)


def _boris(maxtime, timestep, times_to_trace=None, stopping_criteria=None):
    from essos.dynamics import Particles, Tracing

    field = _UniformCartesianField((0.0, 0.0, 1.0), magnitude=1.0)
    mass, charge = 1.6726e-27, 1.602e-19
    v_par, v_perp = 2.0e5, 3.0e5
    particles = Particles(
        initial_xyz=jnp.zeros((1, 3)), mass=mass, charge=charge,
        initial_xyz_fullorbit=jnp.zeros((1, 3)),
        initial_vxvyvz=jnp.asarray([[v_perp, 0.0, v_par]]),
    )
    tracing = Tracing(
        field=field, particles=particles, model="FullOrbit_Boris", maxtime=maxtime,
        timestep=timestep, times_to_trace=times_to_trace, stopping_criteria=stopping_criteria,
    )
    period = 2 * np.pi * mass / (charge * 1.0)
    return tracing, v_par, v_perp, period


def test_boris_integrates_the_whole_requested_time_span():
    period = 2 * np.pi * 1.6726e-27 / 1.602e-19
    tracing, v_par, v_perp, _ = _boris(maxtime=200 * period, timestep=period / 50, times_to_trace=11)
    trajectory = tracing.trajectories[0]
    assert trajectory.shape[0] == 11
    np.testing.assert_allclose(trajectory[-1, 2], v_par * 200 * period, rtol=1e-10)
    speeds = jnp.linalg.norm(trajectory[:, 3:], axis=1)
    np.testing.assert_allclose(speeds, np.hypot(v_par, v_perp), rtol=1e-12)
    # The perpendicular motion stays on the gyro-circle (diameter 2 rho).
    rho = v_perp * period / (2 * np.pi)
    excursion = jnp.linalg.norm(trajectory[:, :2] - trajectory[:1, :2], axis=1)
    assert float(jnp.max(excursion)) <= 2.0 * rho * (1 + 1e-6)


def test_boris_honours_stopping_criteria_with_an_event_mask():
    period = 2 * np.pi * 1.6726e-27 / 1.602e-19
    ceiling = 0.5 * 2.0e5 * 100 * period

    def below_ceiling(t, y, args, **kwargs):
        return ceiling - y[2]

    tracing, *_ = _boris(maxtime=100 * period, timestep=period / 40, times_to_trace=21,
                         stopping_criteria=below_ceiling)
    trajectory = tracing.trajectories[0]
    assert bool(tracing.boundary_hits[0])
    assert float(jnp.max(trajectory[:, 2])) < ceiling
    np.testing.assert_allclose(trajectory[-1], trajectory[-2], rtol=0, atol=0)


def _edge_tracing(model, s0, times_to_trace, **kwargs):
    from pathlib import Path
    wout = Path(__file__).resolve().parents[1] / "examples" / "input_files" / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
    n = 16
    particles = Particles(initial_xyz=jnp.stack([jnp.full(n, s0), jnp.linspace(0, 6, n), jnp.zeros(n)], axis=1),
                          initial_vparallel_over_v=jnp.linspace(-0.9, 0.9, n))
    return Tracing(field=Vmec(str(wout), ntheta=8, nphi=8), model=model, particles=particles, maxtime=1e-5,
                   timestep=1e-8, times_to_trace=times_to_trace, **kwargs)


def test_vmec_losses_are_counted_at_the_lcfs():
    """Orbits born at s = 0.993 used to count as lost at t = 0 (r_max = 0.99)."""
    tracing = _edge_tracing("GuidingCenterAdaptative", 0.993, 50, atol=1e-9, rtol=1e-9)
    assert 0 < tracing.boundary_hits.sum() < 16
    assert tracing.total_particles_lost == tracing.boundary_hits.sum()
    assert jnp.all((tracing.lost_times > 0) == tracing.boundary_hits)


def test_vmec_lost_energies_come_from_the_last_finite_state():
    """With coarse saves the first sample after a loss is infinite, and the lost
    energies and positions used to be read there."""
    species = BackgroundSpecies(number_species=2, mass_array=jnp.array([ELECTRON_MASS / PROTON_MASS, 2.0]),
                                charge_array=jnp.array([-1.0, 1.0]), n_array=jnp.array([1e20, 1e20]),
                                T_array=jnp.array([1e4, 1e4]))
    tracing = _edge_tracing("GuidingCenterCollisionsMuFixed", 0.975, 4, species=species)
    lost = tracing.boundary_hits
    assert lost.any()
    assert jnp.isfinite(tracing.lost_energies).all() and jnp.isfinite(tracing.lost_positions).all()
    assert jnp.allclose(tracing.lost_energies[lost], tracing.particles.energy, rtol=1e-2)
    assert jnp.all(tracing.lost_positions[lost, 0] < 1)


def test_custom_loss_grad_through_adaptive_guiding_center_matches_finite_difference():
    # custom_loss jits its value and gradient, so Tracing.trace sees tracer
    # initial conditions and must not pull them back to the host.
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    from essos.losses import custom_loss

    curves = CreateEquallySpacedCurves(n_curves=2, order=1, R=1.0, r=0.4, n_segments=24, nfp=2, stellsym=True)
    coils = Coils(curves=curves, currents=jnp.array([1e6, 1e6]))
    R0 = jnp.linspace(0.95, 1.05, 2)
    particles = Particles(initial_xyz=jnp.array([R0, 0 * R0, 0 * R0]).T)

    def final_position(field, particles):
        tracing = Tracing(field=field, model="GuidingCenterAdaptative", particles=particles,
                          maxtime=1e-7, times_to_trace=4, atol=1e-10, rtol=1e-10)
        xyz = tracing.trajectories[:, -1, :3]
        return jnp.sum(jnp.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)) + jnp.sum(xyz[:, 2])

    loss = custom_loss(final_position, "field", particles=particles)
    loss.dependencies = {"field": BiotSavart(coils)}
    dofs = loss.starting_dofs

    gradient = loss.grad(dofs)
    direction = jax.random.normal(jax.random.key(0), dofs.shape)
    step = 1e-4 * jnp.linalg.norm(dofs) / jnp.linalg.norm(direction)
    finite_difference = (loss(dofs + step * direction) - loss(dofs - step * direction)) / (2 * step)

    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(gradient @ direction, finite_difference, rtol=1e-6)


def test_vmec_guiding_centers_seeded_on_the_axis_leave_it():
    """Seeds at s = 0 used to stay there: the VMEC Jacobian vanishes on the axis."""
    from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE

    wout = str(Path(__file__).resolve().parents[1] / "examples" / "input_files"
               / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
    vmec = Vmec(wout, ntheta=8, nphi=8)
    particles = Particles(initial_xyz=jnp.array([[0.0, 0.0, 0.3], [0.0, 1.0, 1.0]]),
                          initial_vparallel_over_v=jnp.array([0.9, -0.5]), mass=PROTON_MASS,
                          charge=ELEMENTARY_CHARGE, energy=5e3 * ELEMENTARY_CHARGE)
    tracing = Tracing(field=vmec, model="GuidingCenterAdaptative", particles=particles,
                      maxtime=2e-5, timestep=1e-8, times_to_trace=5)
    s = tracing.trajectories[:, :, 0]
    assert jnp.all(jnp.isfinite(s)) and jnp.all(s[:, -1] > 1e-8)
