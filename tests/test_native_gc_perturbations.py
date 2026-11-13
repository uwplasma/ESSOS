"""Full-geometry MHD corrections, including nonorthogonal radial components."""

from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE, ONE_EV
from essos.dynamics import (GuidingCenterMu, Particles, Tracing, _axis_regular,
                             _guiding_center_velocity, _to_axis_regular)
from essos.electric_field import Electric_field_zero
from essos.fields import Vmec
from essos.gc_perturbations import (AnalyticalVmecAlfvenMode, GuidingCenterPerturbation,
                                    MappedBoozerPerturbation)


SPEED = np.sqrt(2*3.5e6*ONE_EV/MASS)
WOUT = Path(__file__).resolve().parents[1]/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"


class ShearedField:
    """Divergence-free Cartesian helix in a nonorthogonal coordinate chart.

    Physical position is (x1,x2,x3+shear*x1*sin(x2)); det(dx/dq)=1.
    B=(-twist*Y,twist*X,5) has curvature and a nonzero radial covariant
    component. All geometry and field derivatives are analytic/JAX exact.
    """

    def __init__(self, shear=.35, twist=2.):
        self.shear, self.twist = shear, twist

    def to_xyz(self, x):
        return x.at[2].add(self.shear*x[0]*jnp.sin(x[1]))

    def B_covariant(self, x):
        return jnp.array([-self.twist*x[1]+5*self.shear*jnp.sin(x[1]),
                          self.twist*x[0]+5*self.shear*x[0]*jnp.cos(x[1]), 5.])

    def B_contravariant(self, x):
        return jnp.array([-self.twist*x[1], self.twist*x[0],
                          5.+self.shear*self.twist*x[1]*jnp.sin(x[1])
                          -self.shear*self.twist*x[0]**2*jnp.cos(x[1])])

    def AbsB(self, x):
        return jnp.sqrt(25.+self.twist**2*(x[0]**2+x[1]**2))

    def sqrtg(self, x):
        return jnp.ones(())

    def dAbsB_by_dX(self, x):
        return jax.grad(self.AbsB)(x)

    def curl_B(self, x):
        d = jax.jacfwd(self.B_covariant)(x)
        return jnp.array([d[2, 1]-d[1, 2], d[0, 2]-d[2, 0], d[1, 0]-d[0, 1]])

    def curl_b(self, x):
        return (self.curl_B(x)/self.AbsB(x)
                +jnp.cross(self.B_covariant(x), self.dAbsB_by_dX(x))/self.AbsB(x)**2)

    def kappa(self, x):
        return -jnp.cross(self.B_contravariant(x), self.curl_b(x))/self.AbsB(x)


class Wave(GuidingCenterPerturbation):
    def potentials(self, field, point, t):
        xyz = field.to_xyz(point)
        phase = .5*xyz[0]+.7*xyz[1]-xyz[2]+7e5*t
        return jnp.array([.003*jnp.cos(phase), 2e4*jnp.sin(phase+.4)])


class Zero(GuidingCenterPerturbation):
    def potentials(self, field, point, t):
        return jnp.zeros(2)


class Induction(GuidingCenterPerturbation):
    def potentials(self, field, point, t):
        return jnp.array([.003*jnp.sin(7.5e5*t+.4), 0.])


def velocity(field, wave, point, vpar=.7*SPEED, mu=1e-14, t=1.3e-6):
    return _guiding_center_velocity(field, Electric_field_zero(), CHARGE, MASS,
                                    point, vpar, mu=mu, t=t, perturbation=wave)


def test_full_geometry_corrections_match_effective_field_equations():
    """Compare to ASCOT's vector equations, without any matrix solve."""
    field, wave, x, v, mu, t = ShearedField(), Wave(), jnp.array([.4, .7, .2]), .7*SPEED, 1e-14, 1.3e-6
    (alpha, _), derivative = wave.derivatives(field, x, t)
    Bcov, B, gradB = field.B_covariant(x), field.AbsB(x), field.dAbsB_by_dX(x)
    Bstar = (field.B_contravariant(x)+MASS*v/CHARGE*field.curl_b(x)
             +jnp.cross(derivative[0, :3], Bcov)+alpha*field.curl_B(x))
    Estar = -derivative[1, :3]-derivative[0, 3]*Bcov-mu/CHARGE*gradB
    denominator = jnp.dot(Bcov, Bstar)/B
    expected_x = (v*Bstar+jnp.cross(Estar, Bcov/B))/denominator
    expected_v = CHARGE/MASS*jnp.dot(Bstar, Estar)/denominator
    actual_x, actual_v = velocity(field, wave, x, v, mu, t)
    assert abs(float(Bcov[0])) > .1
    assert abs(float(jnp.dot(Bstar, field.kappa(x)))) > 1e-7
    np.testing.assert_allclose(actual_x, expected_x, rtol=2e-13, atol=1e-9)
    assert float(actual_v) == pytest.approx(float(expected_v), rel=2e-13)


def test_perturbed_gc_is_covariant_under_a_nonorthogonal_coordinate_change():
    field, cartesian, wave = ShearedField(), ShearedField(shear=0.), Wave()
    x = jnp.array([.4, .7, .2])
    dx, dv = velocity(field, wave, x)
    physical_dx = dx.at[2].add(field.shear*jnp.sin(x[1])*dx[0]
                              +field.shear*x[0]*jnp.cos(x[1])*dx[1])
    expected_dx, expected_dv = velocity(cartesian, wave, field.to_xyz(x))
    np.testing.assert_allclose(physical_dx, expected_dx, rtol=2e-13, atol=1e-9)
    assert float(dv) == pytest.approx(float(expected_dv), rel=2e-13)


def test_zero_potential_preserves_existing_gc_rhs():
    field, x = ShearedField(), jnp.array([.4, .7, .2])
    actual, expected = velocity(field, Zero(), x), velocity(field, None, x)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-14)
    assert float(actual[1]) == pytest.approx(float(expected[1]), rel=1e-14, abs=1e-8)


def test_full_geometry_gc_obeys_the_time_dependent_action_work_identity():
    field, wave, x, mu, v, t = ShearedField(), Wave(), jnp.array([.4, .7, .2]), 1e-14, .7*SPEED, 1.3e-6
    state = jnp.append(x, v)
    H = lambda y, t: .5*MASS*y[3]**2+mu*field.AbsB(y[:3])+CHARGE*wave.potentials(field, y[:3], t)[1]
    dx, dv = velocity(field, wave, x, v, mu, t)
    measured = jax.grad(H, 0)(state, t)@jnp.append(dx, dv)+jax.grad(H, 1)(state, t)
    d = wave.derivatives(field, x, t)[1]
    expected = CHARGE*(d[1, 3]-d[0, 3]*jnp.dot(field.B_covariant(x), dx))
    assert abs(float(measured)) > 1e-8
    assert abs(float(measured-expected))/abs(float(H(state, t)*7e5)) < 1e-13


def test_vmec_ideal_wave_and_radial_covariant_field_are_retained():
    field = Vmec(str(WOUT), ntheta=8, nphi=8, mode_tolerance=.01)
    wave, point = AnalyticalVmecAlfvenMode(3, 2, 9e5, 2e4), jnp.array([.35, .7, .3])
    derivative = wave.derivatives(field, point, 1.3e-6)[1]
    Bcov, Bcontra, B = field.B_covariant(point), field.B_contravariant(point), field.AbsB(point)
    residual = jnp.dot(Bcontra, -derivative[1, :3]-derivative[0, 3]*Bcov)
    scale = abs(jnp.dot(Bcontra, derivative[1, :3]))+abs(derivative[0, 3])*B**2
    # VMEC interpolates Bcov, Bcontra and |B| separately; their contraction
    # equals B^2 only to interpolation accuracy, unlike an analytic metric.
    assert abs(float(residual/scale)) < 1e-3
    assert abs(float(Bcov[0])) > 1e-6
    grad_cov = field.grad_B_covariant(point)
    contribution = jnp.array([0., grad_cov[0, 2], -grad_cov[0, 1]])/field.sqrtg(point)
    assert np.linalg.norm(contribution) > 1e-6
    dx, dv = velocity(field, wave, point)
    assert np.isfinite(dx).all() and np.isfinite(dv)
    # The existing regular VMEC chart must also accept the new full-geometry
    # force near the axis, where individual native-coordinate terms are large.
    particles = Particles(initial_xyz=point[None, :], initial_vparallel_over_v=jnp.array([.7]))
    rhs = _axis_regular(partial(GuidingCenterMu, perturbation=wave))
    near_axis = jnp.array([1e-7, .7, .3, .7*SPEED, 1e-14])
    assert np.isfinite(rhs(1.3e-6, _to_axis_regular(near_axis),
                           (field, particles, Electric_field_zero()))).all()


def test_boozer_adapter_includes_coordinate_map_derivatives():
    from examples.benchmarks.alfven_gc_verification import equilibrium, spectrum
    background, wave = equilibrium(), spectrum(np.pi/2)
    mapping = lambda x: jnp.array([x[0], x[1]+.2*jnp.sin(x[2]), x[2]+.1*x[0]*jnp.sin(x[1])])
    adapter = MappedBoozerPerturbation(background, wave, mapping)
    x, t = jnp.array([.35, .7, .3]), 1.3e-6
    _, derivative = adapter.derivatives(None, x, t)
    z, steps = np.append(x, t), [1e-6, 1e-6, 1e-6, 1e-11]
    columns = []
    for i, step in enumerate(steps):
        dz = np.eye(4)[i]*step
        plus, minus = z+dz, z-dz
        columns.append((adapter.potentials(None, plus[:3], plus[3])
                        -adapter.potentials(None, minus[:3], minus[3]))/(2*step))
    np.testing.assert_allclose(derivative, np.stack(columns, axis=1), rtol=1e-6, atol=1e-7)


def test_adaptive_trace_uses_constant_mu_while_inductive_field_changes_energy():
    field = ShearedField(twist=0.)
    particles = Particles(initial_xyz=jnp.array([[.3, .2, .1], [.4, .5, .2]]),
                          initial_vparallel_over_v=jnp.array([.65, -.4]))
    trace = Tracing(field=field, model="GuidingCenterAdaptative", particles=particles,
                    perturbation=Induction(), maxtime=1.5e-5, timestep=2e-8,
                    times_to_trace=17, atol=1e-10, rtol=1e-10, max_steps=4096,
                    progress=False, devices=jax.devices()[:1])
    states, times = np.asarray(trace.trajectories), np.asarray(trace.times)
    k = CHARGE*5./MASS*.003
    expected_v = np.asarray(particles.initial_vparallel)[:, None]-k*(np.sin(7.5e5*times+.4)-np.sin(.4))
    expected_z = (np.asarray(particles.initial_xyz)[:, 2:3]
                  +(np.asarray(particles.initial_vparallel)[:, None]+k*np.sin(.4))*times
                  +k/7.5e5*(np.cos(7.5e5*times+.4)-np.cos(.4)))
    assert states.shape == (2, 17, 5)
    assert np.max(abs(states[:, :, 3]-expected_v))/SPEED < 1e-8
    np.testing.assert_allclose(states[:, :, 2], expected_z, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(states[:, :, 4], np.broadcast_to(states[:, :1, 4], states[:, :, 4].shape), rtol=1e-14)
    assert np.max(abs(np.asarray(trace.energy())/float(particles.energy)-1)) > .01


def test_native_mode_amplitude_gradient_passes_through_the_existing_rhs():
    field, point = ShearedField(twist=0.), jnp.array([.4, .7, .2])

    def objective(amplitude):
        wave = AnalyticalVmecAlfvenMode(3, 2, 9e5, amplitude)
        return velocity(field, wave, point)[0][0]

    ad = float(jax.grad(objective)(2e4))
    fd = float((objective(2e4+1.)-objective(2e4-1.))/2)
    assert abs(ad) > 1e-8
    assert ad == pytest.approx(fd, rel=1e-6)


def test_unsupported_models_and_fixed_energy_shortcut_are_rejected():
    with pytest.raises(ValueError, match="conserved magnetic moment"):
        _guiding_center_velocity(ShearedField(), Electric_field_zero(), CHARGE, MASS,
                                  jnp.array([.4, .7, .2]), .7*SPEED, energy=3.5e6*ONE_EV,
                                  perturbation=Wave())
    with pytest.raises(ValueError, match="collisionless guiding centers"):
        Tracing(model="GuidingCenterCollisions", perturbation=Wave())
    with pytest.raises(ValueError, match="exterior wave model"):
        Tracing(model="GuidingCenterAdaptative", perturbation=Wave(), exterior_field=ShearedField())
