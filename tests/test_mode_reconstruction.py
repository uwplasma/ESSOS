"""Analytic recovery, coordinate units and Maxwell identities for mode fields."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.gc_perturbations import GuidingCenterPerturbation
from essos.mode_reconstruction import (SpectralAlfvenMode, perturbed_fields,
                                       physical_components, radial_field_rms)
from essos.perturbations import AnalyticalAlfvenMode, AlfvenSpectrum
from examples.benchmarks.alfven_gc_verification import equilibrium


def test_complex_spectrum_recovers_two_analytical_modes_and_derivatives():
    from essos.rk import _select_one
    selected = lambda q:_select_one(jnp.array([q,-2*q,q*q]),2,0,False)
    assert jax.grad(selected)(.7) == pytest.approx(1.4)
    assert jax.grad(jax.grad(selected))(.7) == pytest.approx(2.)
    field = equilibrium()
    s = np.linspace(0, 1, 9)
    phases, amplitudes = [.3, -.7], [2e4, -1e4]
    modes = [[3, 2], [4, 3]]
    coefficients = np.tile(-1j*np.array(amplitudes)*np.exp(1j*np.array(phases)), (len(s), 1))
    computed = SpectralAlfvenMode(s, modes, coefficients, 9e5)
    analytic = AlfvenSpectrum(*[AnalyticalAlfvenMode(m, n, 9e5, a, phase=p)
        for (m, n), a, p in zip(modes, amplitudes, phases)])
    for point in (jnp.array([.4, .3, .7]), jnp.array([0., 0., .4])):
        a, da = computed.derivatives(field, point, 1.3e-6)
        b, db = analytic.derivatives(field, point, 1.3e-6)
        np.testing.assert_allclose(a, b, rtol=2e-13, atol=1e-12)
        np.testing.assert_allclose(da, db, rtol=2e-13, atol=1e-6)
    from essos.boozer import guiding_center_rhs
    from essos.mode_reconstruction import PhysicalPotentialMode
    from essos.split import CollisionStep, _prepare_split_kernel, _split_field_data, prepare_physics, solve_split
    from examples.benchmarks.alfven_gc_verification import MASS, CHARGE, SPEED
    physical_modes=modes+[[2,1]]
    physical_coefficients=np.column_stack((coefficients*s[:,None]**(np.array(modes)[:,0]/2),3e3j*s))
    physical = PhysicalPotentialMode(s,physical_modes,physical_coefficients,9e5)
    initial = jnp.array([.4,.3,.7,.7*SPEED,1e-14])
    times = jnp.array([0.,1e-8,2e-8])
    def build(background):
        rhs = lambda t,y,a:jnp.append(guiding_center_rhs(background,y[:4],y[4]/MASS,
            MASS,CHARGE,t=t,perturbation=physical),0.)
        kick = lambda t,y,h,dw,a:CollisionStep(y,jnp.asarray(0.),jnp.asarray(0.))
        return rhs,kick,lambda a,b,e:jnp.max(jnp.abs(e)),lambda t,y,a:jnp.asarray(1.),lambda t,y,a:jnp.asarray(jnp.inf)
    arrays,factory = prepare_physics(field,_split_field_data(field),build,initial,2,time_dependent=True)
    solve = _prepare_split_kernel(arrays,factory,times,adaptive=False,dt0=1e-8,dtmax=1e-8,
        speed=SPEED,orbit_dt0=1e-8,max_steps=4,max_orbit_steps=4,state_size=5,interpret=True,time_dependent=True)
    values,_,_,_,status = jax.jit(solve)(jnp.pad(initial,(0,3)),jax.random.PRNGKey(0))
    rhs,kick,*_ = build(field)
    portable = jax.jit(lambda y:solve_split(rhs,kick,y,t1=times[-1],dt0=1e-8,ts=times,
        key=jax.random.PRNGKey(0),noise_dim=2,adaptive=False,orbit_adaptive=False,
        orbit_dt0=1e-8,max_steps=4,max_orbit_steps=4))(initial)
    assert status == portable.status == 0
    np.testing.assert_allclose(values[:,:5],portable.ys,rtol=2e-12,atol=1e-9)


def test_negative_m_and_full_torus_n_satisfy_ideal_relation_and_periodicity():
    field = equilibrium()
    wave = SpectralAlfvenMode(np.linspace(0, 1, 5), [[-3, 7]], np.ones((5, 1))*1e4j, -7e5)
    point = jnp.array([.4, -.2, .3])
    (_, _), d = wave.derivatives(field, point, 1e-6)
    s = float(point[:2]@point[:2])
    (iota, G, I), _ = field.profiles(s)
    # theta derivative in the regular (u,w) chart.
    phi_theta = -point[1]*d[1, 0]+point[0]*d[1, 1]
    residual = d[0, 3]+(iota*phi_theta+d[1, 2])/(G+iota*I)
    assert abs(float(residual)) < 1e-10
    np.testing.assert_allclose(wave.potentials(field, point, 0.),
        wave.potentials(field, point.at[2].add(2*np.pi), 0.), rtol=1e-13)


def test_annular_extension_is_c2_and_finite_on_axis():
    field = equilibrium()
    wave = SpectralAlfvenMode(np.linspace(.1, .9, 7), [[3, 2]], np.ones((7, 1)), 7e5, taper_width=.1)
    for s in (0., .05, .1, .9, .95):
        x = jnp.array([np.sqrt(s), 0., .3])
        values, d = wave.derivatives(field, x, 1e-6)
        np.testing.assert_allclose(values, 0., atol=1e-15)
        np.testing.assert_allclose(d, 0., atol=1e-14)
        second = jax.jacfwd(lambda z: wave.derivatives(field, z, 1e-6)[1])(x)
        np.testing.assert_allclose(second, 0., atol=1e-10)


def test_amplitude_gradient_passes_through_spectral_potentials():
    field = equilibrium()
    mode = SpectralAlfvenMode(np.linspace(0, 1, 5), [[2, 3]], np.ones((5, 1))*1j, 7e5)
    x = jnp.array([.4, .2, .3])
    objective = lambda a: eqx.tree_at(lambda p: p.amplitude, mode, a).potentials(field, x, 1e-6)[1]
    assert float(jax.grad(objective)(2.)) == pytest.approx(float(objective(1.)), rel=1e-12)
    np.testing.assert_allclose(jax.jit(lambda p: p.potentials(field, x, 1e-6))(mode),
                               mode.potentials(field, x, 1e-6), rtol=1e-12)
    from essos.background_species import BackgroundSpecies
    from essos.dynamics import Particles
    from essos.split import trace_split
    from examples.benchmarks.alfven_gc_verification import SPEED
    particles = Particles(initial_xyz=jnp.array([[.3,.2,.3]]),initial_vparallel_over_v=jnp.array([.65]))
    bath = BackgroundSpecies(1,jnp.array([2.]),jnp.array([1.]),jnp.array([5e19]),jnp.array([17800.]))
    physical = eqx.tree_at(lambda w:w.coefficients,mode,mode.coefficients*1e4)
    def run(a,adjoint):
        return trace_split(field,particles,bath,tmax=2e-8,timestep=1e-8,n_save=3,
            perturbation=eqx.tree_at(lambda p:p.amplitude,physical,a),spatial_diffusion=False,
            adaptive=False,orbit_adaptive=False,orbit_dt0=1e-8,max_steps=4,max_orbit_steps=4,
            adjoint=adjoint,adjoint_checkpoint_steps=2,adjoint_orbit_checkpoint_steps=2,
            adjoint_max_accepted_steps=4,adjoint_max_orbit_steps=8).y[0,3]/SPEED
    actual = jax.grad(lambda a:run(a,'accepted'))(jnp.asarray(2.))
    finite = (run(2.001,'none')-run(1.999,'none'))/.002
    assert abs(actual) > 1e-8
    np.testing.assert_allclose(actual,finite,rtol=2e-5,atol=1e-9)


class UniformField:
    def B_covariant(self, x): return jnp.array([0., 0., 5.])
    def curl_B(self, x): return jnp.zeros(3)
    def sqrtg(self, x): return jnp.ones(())
    def AbsB(self, x): return jnp.asarray(5.)
    def to_xyz(self, x): return x


class InductiveWave(GuidingCenterPerturbation):
    def potentials(self, field, point, t):
        return jnp.array([.003*jnp.cos(point[1])*jnp.sin(7e5*t), 100*jnp.sin(point[0]-7e5*t)])


def test_physical_components_respect_a_nonorthogonal_embedding():
    class Chart:
        def to_xyz(self, q):
            return q.at[2].add(.3*q[0]*jnp.sin(q[1]))

    q = jnp.array([.4, .7, .3])
    B, E = np.array([.5, 1.2, -.7]), np.array([3., -2., 5.])
    contravariant = jnp.array([B[0], B[1], B[2]-.3*jnp.sin(q[1])*B[0]-.3*q[0]*jnp.cos(q[1])*B[1]])
    covariant = jnp.array([E[0]+.3*jnp.sin(q[1])*E[2], E[1]+.3*q[0]*jnp.cos(q[1])*E[2], E[2]])
    b, e = physical_components(Chart(), q, contravariant, covariant)
    np.testing.assert_allclose(b, B, rtol=1e-13)
    np.testing.assert_allclose(e, E, rtol=1e-13)


def test_fields_match_literal_cartesian_formula_and_finite_difference_maxwell():
    field, wave = UniformField(), InductiveWave()
    x, t = jnp.array([.4, .2, .3]), 1.3e-6
    B, E = perturbed_fields(field, wave, x, t)
    np.testing.assert_allclose(B, [-.015*np.sin(x[1])*np.sin(7e5*t), 0., 0.], atol=1e-14)
    np.testing.assert_allclose(E, [-100*np.cos(x[0]-7e5*t), 0.,
        -.015*7e5*np.cos(x[1])*np.cos(7e5*t)], rtol=1e-13)
    bx, ex = physical_components(field, x, B, E)
    np.testing.assert_allclose(bx, B)
    np.testing.assert_allclose(ex, E)
    h, dt = 1e-5, 1e-10
    dB, dE = [], []
    for delta in np.eye(3)*h:
        plus = perturbed_fields(field, wave, x+delta, t)
        minus = perturbed_fields(field, wave, x-delta, t)
        dB.append((plus[0]-minus[0])/(2*h))
        dE.append((plus[1]-minus[1])/(2*h))
    dB, dE = np.array(dB).T, np.array(dE).T
    assert abs(np.trace(dB)) < 1e-10
    curl_E = np.array([dE[2, 1]-dE[1, 2], dE[0, 2]-dE[2, 0], dE[1, 0]-dE[0, 1]])
    time_B = (perturbed_fields(field, wave, x, t+dt)[0]-perturbed_fields(field, wave, x, t-dt)[0])/(2*dt)
    np.testing.assert_allclose(curl_E+time_B, 0., atol=2e-5)
    # Exact area/time RMS of .003*sin(theta)*sin(omega*t).
    assert radial_field_rms(field, wave, .4, 7e5, angular_points=8) == pytest.approx(.003/2, rel=1e-13)


def test_invalid_spectral_data_are_rejected():
    for kwargs in (dict(omega=0),dict(harmonics=[[1.5,2]]),dict(coefficients=np.ones((4,1))),
                   dict(s=[0,.3,.2,.8,1]),dict(taper_width=-.1)):
        args = dict(s=np.linspace(0,1,5),harmonics=[[2,3]],coefficients=np.ones((5,1)),omega=7e5)
        with pytest.raises(ValueError):
            SpectralAlfvenMode(**dict(args,**kwargs))
