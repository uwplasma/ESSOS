"""Prescribed-potential corrections through the shared native timestepper."""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.boozer import guiding_center_rhs, trace_boozer
from essos.dynamics import GuidingCenterMu, Particles, Tracing, _axis_regular, _to_axis_regular, _guiding_center_velocity
from essos.electric_field import Electric_field_zero
from essos.gc_perturbations import AnalyticalVmecAlfvenMode, GuidingCenterPerturbation, MappedBoozerPerturbation, VmecBoozerMap
from essos.perturbations import AnalyticalAlfvenMode, AlfvenSpectrum, BoozerPerturbation, poloidal_flux, wave_frame_invariant
from essos.rk import dopri8_step
from examples.benchmarks.alfven_gc_verification import (
    MASS, CHARGE, SPEED, UniformInduction, differential_checks, equilibrium, spectrum,
)


def mode(**kwargs):
    return AnalyticalAlfvenMode(**dict(dict(m=3, n=2, omega=9e5, amplitude=2e4, phase=.3), **kwargs))


def trace(field, wave, *, integrator="essos_dopri8", **kwargs):
    options = dict(speed=SPEED, mass=MASS, charge=CHARGE, tmax=1e-6, timestep=2e-8,
                   n_save=5, rtol=1e-10, atol=1e-10, progress=False, devices=jax.devices()[:1])
    options.update(kwargs)
    return trace_boozer(field, [.3], [.2], [.3], [.65], integrator=integrator, perturbation=wave, **options)


def test_ideal_potential_flux_and_single_wave_invariant():
    field, wave = equilibrium(), mode()
    y, mu, t = jnp.array([.45, .27, -.3, .7*SPEED]), 1e12, 2.3e-6
    _, derivative = wave.derivatives(field, y[:3], t)
    (iota, G, I), _ = field.profiles(jnp.sum(y[:2]**2))
    phi_theta = -y[1]*derivative[1, 0]+y[0]*derivative[1, 1]
    assert abs(derivative[0, 3]+(iota*phi_theta+derivative[1, 2])/(G+iota*I)) < 1e-10
    s = jnp.array([.13, .5, 1.])
    np.testing.assert_allclose(jax.vmap(lambda x: poloidal_flux(field, x))(s), field.psi0*(.4*s+.05*s*s), atol=1e-13)
    invariant = lambda y, t: wave_frame_invariant(field, wave, y, mu, MASS, CHARGE, t)
    dy = guiding_center_rhs(field, y, mu, MASS, CHARGE, t=t, perturbation=wave)
    rate = jax.grad(invariant, 0)(y, t)@dy+jax.grad(invariant, 1)(y, t)
    assert abs(rate)/abs(invariant(y, t)*wave.omega) < 1e-12


def test_axis_regular_potentials_and_singular_cutoff_contract():
    field, y = equilibrium(), jnp.array([0., 0., .2, .7*SPEED])
    for m in (1, 15):
        wave = mode(m=m)
        assert all(np.isfinite(x).all() for x in wave.derivatives(field, y[:3], 1e-6))
        assert np.isfinite(guiding_center_rhs(field, y, 1e12, MASS, CHARGE, t=1e-6, perturbation=wave)).all()
    singular = mode(regularize_axis=False)
    assert singular.axis_stop_threshold() == 1e-8
    assert mode(amplitude=1, regularize_axis=False).axis_stop_threshold() == 1e-8
    assert mode().axis_stop_threshold() is None
    assert mode(amplitude=0., regularize_axis=False).axis_stop_threshold() == 0.
    assert AlfvenSpectrum(singular, mode(regularize_axis=False, axis_stop_s=1e-7)).axis_stop_threshold() == 1e-7


def test_native_step_amplitude_gradient():
    field, y = equilibrium(), jnp.array([.45, .27, -.3, .7*SPEED])
    def objective(amplitude):
        wave = mode(amplitude=amplitude)
        rhs = lambda t, y, args: guiding_center_rhs(field, y, 1e12, MASS, CHARGE, t=t, perturbation=wave)
        end = dopri8_step(rhs, 2e-6, y, 1e-8).y
        return end[0]**2+end[1]**2
    ad = jax.jit(jax.grad(objective))(2e4)
    fd = (objective(2e4+1)-objective(2e4-1))/2
    assert abs(ad) > 1e-12
    assert ad == pytest.approx(fd, rel=2e-5, abs=1e-13)


def test_two_harmonic_conservation_detects_missing_induction():
    field, wave = equilibrium(), spectrum(np.pi/2)
    points = jnp.array([[.45, .27, -.3, .7*SPEED], [.3, -.4, .7, -.65*SPEED]])
    checks = np.asarray(jax.vmap(lambda y: differential_checks(field, wave, y, 1e12, 1.3e-6))(points))
    assert np.max(abs(checks[:, :2])) < 1e-12
    assert np.max(abs(checks[:, 2])) > 1e-6
    assert np.max(abs(checks[:, 3])) > 1e-10


def test_native_uniform_induction_exact_trajectory():
    wave = UniformInduction()
    for integrator in ("essos_rk45", "essos_dopri8"):
        out = trace(equilibrium(uniform=True), wave, integrator=integrator, tmax=1.5e-5, n_save=9)
        expected_v, expected_z = wave.exact(out.times, .65*SPEED, .3)
        assert not out.failed.any() and not out.lost.any()
        assert out.step_stats[0, 1] > 0 and out.energy_error is None
        assert np.max(abs(out.states[0, :, 3]-expected_v))/SPEED < 1e-8
        np.testing.assert_allclose(out.states[0, :, 2], expected_z, rtol=1e-9, atol=1e-8)
        assert out.kinetic_energy_change[0] > .01


def test_native_wave_invariant_and_physical_work():
    field, wave = equilibrium(), mode()
    first = trace(field, wave)
    s, th, ze, vp, _ = first.states[0].T
    ys = jnp.stack([np.sqrt(s)*np.cos(th), np.sqrt(s)*np.sin(th), ze, vp], axis=1)
    mu = SPEED**2*(1-.65**2)/(2*field.modB(.3, .2, .3))
    values = jax.vmap(lambda y, t: wave_frame_invariant(field, wave, y, mu, MASS, CHARGE, t))(ys, first.times)
    assert np.max(abs(values/values[0]-1)) < 2e-8
    assert first.kinetic_energy_change[0] > 1e-6


def test_zero_wave_preserves_existing_rhs():
    field = equilibrium()
    y = jnp.array([.45, .27, -.3, .7*SPEED])
    plain = guiding_center_rhs(field, y, 1e12, MASS, CHARGE)
    zero = guiding_center_rhs(field, y, 1e12, MASS, CHARGE, t=1e-6, perturbation=mode(amplitude=0.))
    np.testing.assert_allclose(plain, zero, rtol=2e-13)


def test_native_axis_stops_are_distinct_from_lcfs_exits_and_failures():
    field, cutoff = equilibrium(), 1e-8
    wave = mode(m=1, amplitude=2000., regularize_axis=False, axis_stop_s=cutoff)
    births = ([0., 5e-9, 4*cutoff], [0., 0., np.pi], [0., 0., 0.], [1., 1., 0.])
    options = dict(speed=SPEED, mass=MASS, charge=CHARGE, tmax=2e-9, timestep=2e-11, n_save=6,
                   rtol=1e-10, atol=1e-10, progress=False, devices=jax.devices()[:1])
    out = trace_boozer(field, *births, perturbation=wave, **options)
    assert out.axis_stopped.all() and not out.failed.any() and not out.lost.any()
    np.testing.assert_array_equal(out.axis_stop_times[:2], 0.)
    assert 0 < out.axis_stop_times[2] < out.times[-1]
    assert out.states[2, -1, 0] == pytest.approx(cutoff, rel=1e-8)
    assert out.loss_fractions()[-1] == 0 and out.loss_fractions(include_axis=True)[-1] == 1
    zero = trace_boozer(field, *births, perturbation=mode(m=1, amplitude=0., regularize_axis=False), **options)
    assert not zero.axis_stopped.any() and not zero.failed.any()
    failed = trace(field, mode(), max_steps=1, tmax=1e-3)
    with pytest.raises(RuntimeError, match="trajectories fail"):
        failed.loss_fractions()
    from essos.background_species import BackgroundSpecies
    from essos.split import SplitStatus, trace_split
    particles = Particles(initial_xyz=jnp.array([[cutoff/2,0.,0.]]),initial_vparallel_over_v=jnp.array([.65]))
    bath = BackgroundSpecies(1,jnp.array([2.]),jnp.array([1.]),jnp.array([0.]),jnp.array([17800.]))
    stopped = trace_split(field,particles,bath,tmax=2e-9,timestep=1e-9,n_save=3,
        perturbation=wave,spatial_diffusion=False,adaptive=False,orbit_adaptive=False,
        max_steps=4,max_orbit_steps=4)
    assert stopped.status[0] == SplitStatus.EVENT and stopped.t[0] == 0.
    np.testing.assert_array_equal(stopped.ys[0,:,0],cutoff/2)


class HelicalField:
    """Analytic Cartesian helix in x,y,z+shear*x*sin(y); retains radial B_cov."""
    def __init__(self, shear=.35, twist=2.):
        self.shear, self.twist = shear, twist
    def to_xyz(self, x):
        return x.at[2].add(self.shear*x[0]*jnp.sin(x[1]))
    def B_covariant(self, x):
        return jnp.array([-self.twist*x[1]+5*self.shear*jnp.sin(x[1]),
            self.twist*x[0]+5*self.shear*x[0]*jnp.cos(x[1]), 5.])
    def B_contravariant(self, x):
        return jnp.array([-self.twist*x[1], self.twist*x[0],
            5+self.shear*self.twist*x[1]*jnp.sin(x[1])-self.shear*self.twist*x[0]**2*jnp.cos(x[1])])
    def AbsB(self, x):
        return jnp.sqrt(25+self.twist**2*(x[0]**2+x[1]**2))
    def sqrtg(self, x):
        return jnp.ones(())
    def dAbsB_by_dX(self, x):
        return jax.grad(self.AbsB)(x)
    def curl_B(self, x):
        d = jax.jacfwd(self.B_covariant)(x)
        return jnp.array([d[2, 1]-d[1, 2], d[0, 2]-d[2, 0], d[1, 0]-d[0, 1]])
    def curl_b(self, x):
        return self.curl_B(x)/self.AbsB(x)+jnp.cross(self.B_covariant(x), self.dAbsB_by_dX(x))/self.AbsB(x)**2
    def kappa(self, x):
        return -jnp.cross(self.B_contravariant(x), self.curl_b(x))/self.AbsB(x)


class Wave(GuidingCenterPerturbation):
    def potentials(self, field, point, t):
        x = field.to_xyz(point)
        phase = .5*x[0]+.7*x[1]-x[2]+7e5*t
        return jnp.array([.003*jnp.cos(phase), 2e4*jnp.sin(phase+.4)])


class Induction(GuidingCenterPerturbation):
    def potentials(self, field, point, t):
        return jnp.array([.003*jnp.sin(7.5e5*t+.4), 0.])


def velocity(field, wave, x, v=.7*SPEED, mu=1e-14, t=1.3e-6):
    return _guiding_center_velocity(field, Electric_field_zero(), CHARGE, MASS, x, v, mu=mu, t=t, perturbation=wave)


def test_full_geometry_effective_fields_and_coordinate_covariance():
    field, wave, x, v, mu, t = HelicalField(), Wave(), jnp.array([.4, .7, .2]), .7*SPEED, 1e-14, 1.3e-6
    (alpha, _), d = wave.derivatives(field, x, t)
    bcov, B = field.B_covariant(x), field.AbsB(x)
    bstar = field.B_contravariant(x)+MASS*v/CHARGE*field.curl_b(x)+jnp.cross(d[0, :3], bcov)+alpha*field.curl_B(x)
    estar = -d[1, :3]-d[0, 3]*bcov-mu/CHARGE*field.dAbsB_by_dX(x)
    dx, dv = velocity(field, wave, x, v, mu, t)
    assert abs(bcov[0]) > .1 and abs(jnp.dot(bstar, field.kappa(x))) > 1e-7
    np.testing.assert_allclose(dx, (v*bstar+jnp.cross(estar, bcov/B))/(jnp.dot(bcov, bstar)/B), rtol=2e-13, atol=1e-9)
    assert dv == pytest.approx(CHARGE/MASS*jnp.dot(bstar, estar)/(jnp.dot(bcov, bstar)/B), rel=2e-13)
    cart_dx, cart_dv = velocity(HelicalField(shear=0.), wave, field.to_xyz(x))
    np.testing.assert_allclose(dx.at[2].add(field.shear*jnp.sin(x[1])*dx[0]+field.shear*x[0]*jnp.cos(x[1])*dx[1]), cart_dx, rtol=2e-13, atol=1e-9)
    assert dv == pytest.approx(cart_dv, rel=2e-13)


def test_full_geometry_time_dependent_action_work():
    field, wave, x, mu, v, t = HelicalField(), Wave(), jnp.array([.4, .7, .2]), 1e-14, .7*SPEED, 1.3e-6
    H = lambda y, t: .5*MASS*y[3]**2+mu*field.AbsB(y[:3])+CHARGE*wave.potentials(field, y[:3], t)[1]
    y, (dx, dv) = jnp.append(x, v), velocity(field, wave, x, v, mu, t)
    measured = jax.grad(H, 0)(y, t)@jnp.append(dx, dv)+jax.grad(H, 1)(y, t)
    d = wave.derivatives(field, x, t)[1]
    expected = CHARGE*(d[1, 3]-d[0, 3]*jnp.dot(field.B_covariant(x), dx))
    assert abs(measured) > 1e-8 and abs(measured-expected)/abs(H(y, t)*7e5) < 1e-13


def test_native_general_gc_induction_conserves_mu():
    particles = Particles(initial_xyz=jnp.array([[.3, .2, .1]]), initial_vparallel_over_v=jnp.array([.65]))
    result = Tracing(field=HelicalField(twist=0.), model="GuidingCenterAdaptative", particles=particles,
        perturbation=Induction(), integrator="essos_dopri8", maxtime=1.5e-5, timestep=2e-8,
        times_to_trace=9, rtol=1e-10, atol=1e-10, progress=False, devices=jax.devices()[:1], max_steps=4096)
    y, t = np.asarray(result.trajectories)[0], np.asarray(result.times)
    k, v0 = CHARGE*5/MASS*.003, float(particles.initial_vparallel[0])
    exact_v = v0-k*(np.sin(7.5e5*t+.4)-np.sin(.4))
    exact_z = .1+(v0+k*np.sin(.4))*t+k/7.5e5*(np.cos(7.5e5*t+.4)-np.cos(.4))
    assert np.max(abs(y[:, 3]-exact_v))/float(particles.total_speed) < 1e-8
    np.testing.assert_allclose(y[:, 2], exact_z, rtol=1e-9, atol=1e-8)
    np.testing.assert_array_equal(y[:, 4], np.full(len(t), y[0, 4]))


def test_vmec_boozer_map_and_potential_chain_rule():
    s = np.linspace(.1, .9, 8)
    zero = 0*s
    w, lam = .02*np.sqrt(s), .03*np.sqrt(s)
    booz = SimpleNamespace(asym=False, s_in=s, xm=np.array([0, 1]), xn=np.array([0, 2]),
        xm_nyq=np.array([0, 1]), xn_nyq=np.array([0, 2]), bsubumnc=np.array([zero+.2, w]),
        bsubvmnc=np.array([zero+40, -2*w]), lmns=np.array([zero, lam]), iota=zero+.4)
    mapping, x = VmecBoozerMap.from_booz_xform(booz), jnp.array([.35, .7, .3])
    phase = x[1]-2*x[2]
    nu = (.02-.2*.03)*jnp.sqrt(x[0])*jnp.sin(phase)/(40+.4*.2)
    expected = jnp.array([x[0], x[1]+.03*jnp.sqrt(x[0])*jnp.sin(phase)+.4*nu, x[2]+nu])
    np.testing.assert_allclose(mapping(x), expected, atol=1e-13)
    np.testing.assert_allclose(mapping.inverse(expected), x, atol=1e-13)
    adapter = MappedBoozerPerturbation(equilibrium(), mode(), mapping)
    z = jnp.append(x, 1e-6)
    derivative = adapter.derivatives(None, x, z[3])[1]
    steps = np.array([1e-6, 1e-6, 1e-6, 1e-11])
    columns = [(adapter.potentials(None, (z+dz)[:3], (z+dz)[3])-adapter.potentials(None, (z-dz)[:3], (z-dz)[3]))/(2*h)
               for dz, h in zip(np.eye(4)*steps[:, None], steps)]
    np.testing.assert_allclose(derivative, np.stack(columns, axis=1), rtol=2e-6, atol=1e-6)


def test_vmec_wave_rhs_retains_radial_field_and_constant_mu():
    from essos.fields import Vmec
    root = Path(__file__).resolve().parents[1]
    field = Vmec(str(root/'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'), ntheta=8, nphi=8, mode_tolerance=.01)
    particles = Particles(initial_xyz=jnp.array([[.35, .7, .3]]), initial_vparallel_over_v=jnp.array([.65]))
    x = particles.initial_xyz[0]
    mu = particles.mass*particles.initial_vperpendicular[0]**2/(2*field.AbsB(x))
    y = jnp.append(x, jnp.array([particles.initial_vparallel[0], mu]))
    from functools import partial
    rhs = _axis_regular(partial(GuidingCenterMu, perturbation=AnalyticalVmecAlfvenMode(3, 2, 9e5, 2e4)))
    value = rhs(1e-6, _to_axis_regular(y), (field, particles, Electric_field_zero()))
    assert np.isfinite(value).all() and value[4] == 0.  # Last component tracks the unwrapped angle.
    assert abs(field.B_covariant(x)[0]) > 1e-8
    from essos.rk import _prepare_vmec_data, _KernelVmecField, _vmec_mu_rhs, _lower_kernel_rhs, prepare_solver
    data = _prepare_vmec_data(field)
    model = _KernelVmecField(data.arrays, data.tables)
    wave = AnalyticalVmecAlfvenMode(3, 2, 9e5, 2e4)
    expected = wave.derivatives(field, tuple(x), 1e-6)
    actual = wave.derivatives(model, tuple(x), 1e-6)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,rtol=3e-12,atol=1e-7)
    for m, width in ((0, None), (1, .5), (15, .5)):
        other = AnalyticalVmecAlfvenMode(m, -2, 9e5, 2e4, radial_width=width)
        for s in (1e-8, float(x[0])):
            point = x.at[0].set(s)
            for a,b in zip(other.derivatives(model,tuple(point),1e-6),
                           BoozerPerturbation.derivatives(other,field,point,1e-6)):
                np.testing.assert_allclose(a,b,rtol=5e-12,atol=1e-7)
    z = jnp.concatenate((x,jnp.array([1e-6])))
    np.testing.assert_allclose(jax.jacrev(lambda q: wave.potentials(model,tuple(q[:3]),q[3]))(z),
                              expected[1],rtol=3e-12,atol=1e-10)
    def parameter_value(a, background):
        return AnalyticalVmecAlfvenMode(3, 2, a[0], a[1], phase=a[2],
            radial_center=a[3], radial_width=.5).derivatives(background, tuple(x), a[4])
    parameters = jnp.array([9e5,2e4,.2,.4,1e-6])
    expected_parameters = jax.jacfwd(lambda a: parameter_value(a, field))(parameters)
    for differentiate in (jax.jacfwd, jax.jacrev):
        for a,b in zip(differentiate(lambda a: parameter_value(a, model))(parameters), expected_parameters):
            np.testing.assert_allclose(a,b,rtol=3e-12,atol=1e-7)
    def field_probe(background, point):
        return jnp.sum(background.B_covariant(point))+background.sqrtg(point)
    direction = jnp.array([.2,-.3,.4])
    np.testing.assert_allclose(jax.jvp(lambda q: field_probe(model,q),(x,),(direction,)),
                              jax.jvp(lambda q: field_probe(field,q),(x,),(direction,)),rtol=3e-12,atol=1e-10)
    regular = _to_axis_regular(y)
    padded = jnp.concatenate([regular[:4], regular[-1:], jnp.zeros(3)])
    for wave in (None, AnalyticalVmecAlfvenMode(3, 2, 9e5, 2e4)):
        reference = _axis_regular(partial(GuidingCenterMu, perturbation=wave))(1e-6, regular,
            (field, particles, Electric_field_zero()))
        evaluate = _lower_kernel_rhs(lambda t, state: _vmec_mu_rhs(model, state, mu,
            particles.mass, particles.charge, t, wave), jnp.asarray(0.), padded)
        actual = jax.jit(evaluate)(1e-6, padded)
        np.testing.assert_allclose(np.asarray(actual)[[0,1,2,3,4]],
            np.asarray(reference)[[0,1,2,3,5]], rtol=3e-12, atol=1e-7)
    controls = dict(mass=particles.mass,charge=particles.charge,energy=particles.energy,
                    times=np.array([0.,1e-8]),dt0=1e-8,model='gc_mu',perturbation=wave,max_steps=8)
    native = jax.jit(prepare_solver(field,execution='cuda',interpret=True,**controls))(regular)
    portable = jax.jit(prepare_solver(field,**controls))(regular)
    assert native.status == portable.status == 0
    np.testing.assert_allclose(native.y,portable.y,rtol=3e-12,atol=1e-10)
    with pytest.raises(ValueError, match='gc_mu'):
        prepare_solver(field, mass=particles.mass, charge=particles.charge, energy=particles.energy,
            times=np.array([0., 1e-8]), dt0=1e-8, perturbation=wave)


def test_paul_shared_births_and_native_job(tmp_path):
    from examples.benchmarks.paul_conventions import convert_births, load_essos_births, particle_parameters
    from examples.benchmarks.paul_boozer_vmec_benchmark import commands
    from examples.benchmarks.paul_boozer_vmec_loss_comparison import compare
    births = convert_births([.3, .4], [.1, .2], [.4, .5], [.7, -.2], source="simsopt_2023")
    archive = tmp_path/'births.npz'
    np.savez(archive, **dict(zip(('birth_s','birth_theta','birth_zeta','birth_pitch'), births)),
             metadata_json=json.dumps(dict(source_convention='simsopt_2023', birth_coordinate_convention='essos')))
    for actual, expected in zip(load_essos_births(archive, 2, source='simsopt_2023'), births):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='source convention'):
        load_essos_births(archive, 2, source='essos')
    particle = particle_parameters('simsopt_2023', measured_mass=MASS, energy_joule=3.52e6*1.602176634e-19)
    assert .5*particle['mass_kg']*particle['birth_speed_m_s']**2 == pytest.approx(particle['energy_joule'])
    steps = commands('python', 100, 'cpu', 120, tmp_path, 8, 8, births=archive)
    assert steps[0][steps[0].index('--births')+1] == str(archive)
    assert steps[1][steps[1].index('--births')+1] == str(tmp_path/'boozer/trajectories.npz')
    boozer, vmec = tmp_path/'boozer', tmp_path/'vmec'
    boozer.mkdir()
    vmec.mkdir()
    common = dict(particles_per_case=3, numerical_checks_passed=True, all_orbits_resolved=True, mass_kg=MASS)
    (boozer/'summary.json').write_text(json.dumps(common))
    (vmec/'summary.json').write_text(json.dumps(dict(common, mass_kg=2*MASS)))
    with pytest.raises(ValueError, match='Different particle parameter mass_kg'):
        compare(boozer, vmec, tmp_path/'comparison', particles=3)


def test_invalid_modes_and_unsupported_wave_paths():
    for kwargs in (dict(omega=0.), dict(m=1.5), dict(n=True), dict(radial_width=-1.), dict(axis_stop_s=0.)):
        with pytest.raises(ValueError):
            mode(**kwargs)
    with pytest.raises(ValueError, match='legacy Boozer stepper'):
        trace(equilibrium(), mode(), method='rk4')
    with pytest.raises(ValueError, match='collisionless'):
        trace(equilibrium(), mode(), species=object())
    with pytest.raises(ValueError, match='conserved magnetic moment'):
        _guiding_center_velocity(HelicalField(), Electric_field_zero(), CHARGE, MASS, jnp.ones(3), SPEED,
                                energy=1e-13, perturbation=Wave())


def test_real_analytic_derivatives_match_autodiff_potentials():
    field, point, t = equilibrium(), jnp.array([.4, .2, .3]), 1.3e-6
    waves = [mode(m=0), mode(m=1, radial_width=.3), mode(m=15, regularize_axis=False)]
    waves.append(AlfvenSpectrum(*waves))
    for wave in waves:
        actual = wave.derivatives(field, point, t)
        expected = BoozerPerturbation.derivatives(wave, field, point, t)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=3e-12, atol=1e-7)


def test_fused_stage_times_match_exact_nonautonomous_solution_on_cpu():
    from essos.rk import _prepare_gpu_solver
    def factory(refs, point, invariant):
        return (lambda t, y: jnp.full_like(y, t**3),
                lambda y, error, speed: jnp.max(jnp.abs(error)), lambda y: jnp.asarray(1.))
    times = np.array([0., .1, .2])
    for method in ('dopri5', 'dopri8'):
        solve = _prepare_gpu_solver((), factory, 1., times, .2, 1e-9, 1e-9, 4,
            method=method, adaptive=False, interpret=True, time_dependent=True)
        out = jax.jit(solve)(jnp.zeros(4), jnp.asarray(0.))
        np.testing.assert_allclose(out.ys, np.repeat((times**4/4)[:, None], 4, axis=1), atol=2e-15)
        assert out.status == 0 and out.stats.accepted == 1
    from essos.split import _prepare_split_kernel
    def split_factory(refs, point):
        return (lambda t,y:jnp.full_like(y,t**3), lambda y,h,dw:(y,0.,0.),
                lambda y,error,speed,budget:jnp.max(jnp.abs(error)), lambda y:jnp.asarray(1.))
    solve = _prepare_split_kernel((), split_factory, times, adaptive=False, dt0=.1, dtmax=.1,
        speed=1., max_steps=4, max_orbit_steps=4, orbit_dt0=.1, interpret=True, time_dependent=True)
    values, _, _, _, status = jax.jit(solve)(jnp.zeros(8), jax.random.PRNGKey(0))
    assert status == 0
    np.testing.assert_allclose(values, np.repeat((times**4/4)[:, None], 8, axis=1), atol=2e-15)


def test_fused_wave_orbit_matches_portable_and_stops_at_axis():
    from essos.rk import _prepare_boozer_gpu, _boozer_batch
    class ArbitraryPotentials(BoozerPerturbation):
        def potentials(self, field, xyz, t):
            u, w, z = xyz
            (iota, G, I), _ = field.profiles(u*u+w*w)
            phase = 2*u+3*w-2*z+9e5*t
            return jnp.array([.002*jnp.sin(phase), 2e4*jnp.cos(phase)/(G+iota*I)])
    field = equilibrium()
    interpret = 'cuda' not in jax.devices()[0].client.platform_version.lower()
    wave = AlfvenSpectrum(mode(), mode(m=4, n=1, omega=1.1e6, phase=.8))
    y, mu = jnp.array([[.45, .27, -.3, .7*SPEED]]), jnp.array([1e12])
    times, dt = jnp.array([0., 1e-8, 2e-8]), 1e-8
    args = (field, y, mu, MASS, CHARGE, SPEED, times, dt, 1e-9, 1e-9, 'dopri8', False, 'i', .4, .3, 8)
    for perturbation in (wave, ArbitraryPotentials()):
        solve = _prepare_boozer_gpu(field, MASS, CHARGE, SPEED, times, dt, 1e-9, 1e-9, 8,
            adaptive=False, interpret=interpret, perturbation=perturbation)
        portable = _boozer_batch(*args, perturbation=perturbation)
        fused = _boozer_batch(*args, orbit_solver=solve, perturbation=perturbation)
        np.testing.assert_allclose(fused[0], portable[0], rtol=2e-12, atol=1e-9)
        np.testing.assert_array_equal(fused[4], portable[4])
    singular = mode(m=1, regularize_axis=False, axis_stop_s=.01)
    stop = _prepare_boozer_gpu(field, MASS, CHARGE, SPEED, times, dt, 1e-9, 1e-9, 8,
        adaptive=False, interpret=interpret, perturbation=singular)
    stopped = _boozer_batch(field, y.at[0, :2].set(0.), *args[2:], orbit_solver=stop, perturbation=singular)
    assert stopped[6][0] == 0. and stopped[1][0] < 0 and stopped[2][0] < 0
    from essos.split import CollisionStep, _prepare_split_kernel, _split_field_data, prepare_physics, solve_split
    initial = jnp.append(y[0], MASS*mu[0])
    def build(background):
        rhs = lambda t,state,args:jnp.append(guiding_center_rhs(background,state[:4],state[4]/MASS,
            MASS,CHARGE,t=t,perturbation=wave),0.)
        kick = lambda t,state,h,dw,args:CollisionStep(state,jnp.asarray(0.),jnp.asarray(0.))
        return rhs,kick,lambda a,b,e:jnp.max(jnp.abs(e)),lambda t,state,args:jnp.asarray(1.),lambda t,state,args:jnp.asarray(jnp.inf)
    arrays, factory = prepare_physics(field,_split_field_data(field),build,initial,2,time_dependent=True)
    solve = _prepare_split_kernel(arrays,factory,times,adaptive=False,dt0=dt,dtmax=dt,speed=SPEED,
        orbit_dt0=dt,max_steps=4,max_orbit_steps=4,state_size=5,interpret=True,time_dependent=True)
    values, _, _, _, status = jax.jit(solve)(jnp.pad(initial,(0,3)),jax.random.PRNGKey(0))
    rhs,kick,*_ = build(field)
    plain = jax.jit(lambda y:solve_split(rhs,kick,y,t1=times[-1],dt0=dt,ts=times,
        key=jax.random.PRNGKey(0),noise_dim=2,adaptive=False,orbit_adaptive=False,orbit_dt0=dt,
        max_steps=4,max_orbit_steps=4))(initial)
    assert status == plain.status == 0
    np.testing.assert_allclose(values[:,:5],plain.ys,rtol=2e-12,atol=1e-9)


def test_native_split_wave_and_accepted_gradient():
    from essos.background_species import BackgroundSpecies
    from essos.split import trace_split

    class AmplitudeInduction(GuidingCenterPerturbation):
        amplitude: jax.Array
        def potentials(self, field, point, t):
            return jnp.array([self.amplitude*jnp.sin(7.5e5*t+.4), 0.])

    field = HelicalField(shear=0., twist=0.)
    particles = Particles(initial_xyz=jnp.array([[.3,.2,.1]]), initial_vparallel_over_v=jnp.array([.65]))
    def run(amplitude, density, *, adjoint='none', adaptive=False):
        bath = BackgroundSpecies(1,jnp.array([2.]),jnp.array([1.]),jnp.array([density]),jnp.array([17800.]))
        return trace_split(field,particles,bath,tmax=4e-8,timestep=2e-8,n_save=3,
            perturbation=AmplitudeInduction(jnp.asarray(amplitude)),spatial_diffusion=False,
            adaptive=adaptive,orbit_adaptive=adaptive,orbit_dt0=2e-8,collision_dtmax=2e-8,
            max_steps=4,max_orbit_steps=4,adjoint=adjoint,
            adjoint_checkpoint_steps=2,adjoint_orbit_checkpoint_steps=2,
            adjoint_max_accepted_steps=4,adjoint_max_orbit_steps=8)
    vacuum = run(.003,0.)
    exact = particles.initial_vparallel[0]-CHARGE*5/MASS*.003*(jnp.sin(7.5e5*vacuum.ts+.4)-jnp.sin(.4))
    assert int(vacuum.status[0]) == 0
    np.testing.assert_allclose(vacuum.ys[0,:,3],exact,rtol=2e-12,atol=1e-8)
    np.testing.assert_allclose(vacuum.ys[0,:,4],vacuum.ys[0,0,4],rtol=2e-14,atol=0.)
    adaptive = run(.003,5e19,adaptive=True)
    assert int(adaptive.status[0]) == 0 and np.isfinite(adaptive.ys).all()
    objective = lambda a:run(a,5e19,adjoint='accepted').y[0,3]/SPEED
    gradient = jax.grad(objective)(jnp.asarray(.003))
    finite = (run(.00301,5e19).y[0,3]-run(.00299,5e19).y[0,3])/(.00002*SPEED)
    assert abs(gradient) > 1e-4
    np.testing.assert_allclose(gradient,finite,rtol=2e-5,atol=1e-8)
