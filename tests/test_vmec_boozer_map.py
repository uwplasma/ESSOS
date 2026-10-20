"""Check the prescribed wave against booz_xform's independent output map."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.dynamics import GuidingCenterMu, Particles, Tracing
from essos.electric_field import Electric_field_zero
from essos.fields import Vmec
from essos.gc_perturbations import MappedBoozerPerturbation, VmecBoozerMap
from essos.perturbations import AnalyticalAlfvenMode
from essos.boozer import BoozerField, psi0_from_vmec


WOUT = Path(__file__).resolve().parents[1]/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"


@pytest.fixture(scope="module")
def transformed():
    from booz_xform_jax import Booz_xform
    from netCDF4 import Dataset
    booz = Booz_xform(verbose=0, mboz=16, nboz=16)
    booz.read_wout(str(WOUT), flux=False)
    booz.run()
    with Dataset(WOUT) as nc:
        psi = psi0_from_vmec(nc.variables["phi"][-1])
    return booz, VmecBoozerMap.from_booz_xform(booz), BoozerField.from_booz_xform(booz, psi, angle_grid=None)


def test_map_agrees_with_booz_xform_inverse_output_and_chain_rule(transformed):
    booz, mapping, background = transformed
    j = len(booz.s_b)//2
    point = jnp.array([booz.s_b[j], .7, .3])
    mapped = mapping(point)
    nu = np.sin(float(mapped[1])*np.asarray(booz.xm_b)
                -float(mapped[2])*np.asarray(booz.xn_b))@np.asarray(booz.numns_b)[:, j]
    assert float(mapped[2]-point[2]) == pytest.approx(nu, abs=2e-6)
    np.testing.assert_allclose(mapping.inverse(mapped), point, atol=2e-13)
    adapter = MappedBoozerPerturbation(background, AnalyticalAlfvenMode(15, 7, 399000., 1e4,
                                      regularize_axis=False), mapping)
    assert adapter.axis_stop_threshold() == 1e-8
    t = 1e-6
    derivative = jax.jit(lambda x, t: adapter.derivatives(None, x, t))(point, t)[1]
    z = np.append(point, t)
    columns = []
    for i, step in enumerate([1e-6, 1e-6, 1e-6, 1e-11]):
        plus, minus = z+np.eye(4)[i]*step, z-np.eye(4)[i]*step
        columns.append((adapter.potentials(None, plus[:3], plus[3])
                        -adapter.potentials(None, minus[:3], minus[3]))/(2*step))
    np.testing.assert_allclose(derivative, np.stack(columns, axis=1), rtol=2e-6, atol=1e-6)


def test_vmec_wave_inner_cutoff_is_separate_from_lcfs_loss_and_failure(transformed):
    _, mapping, background = transformed
    field = Vmec(str(WOUT), ntheta=8, nphi=8, mode_tolerance=.01)
    # Enlarge the cutoff to exercise both t=0 stopping and a root-found crossing
    # without a long transport run or evaluating the potential on the axis.
    wave = AnalyticalAlfvenMode(3, 2, 9e5, 1e3, regularize_axis=False, axis_stop_s=.36)
    adapter = MappedBoozerPerturbation(background, wave, mapping)
    particles = Particles(initial_xyz=jnp.array([[.35, .7, .3]]),
                          initial_vparallel_over_v=jnp.array([.65]))
    speed = particles.total_speed

    def radial_speed(theta):
        point = jnp.array([.36, theta, .3])
        mu = particles.mass*speed**2*(1-.65**2)/(2*field.AbsB(point))
        state = jnp.concatenate([point, jnp.array([.65*speed, mu])])
        return GuidingCenterMu(0., state, (field, particles, Electric_field_zero()), perturbation=adapter)[0]

    angles = jnp.linspace(0., 2*jnp.pi, 9)[:-1]
    speeds = jax.jit(jax.vmap(radial_speed))(angles)
    inward = int(jnp.argmin(speeds))
    assert speeds[inward] < 0
    moving = jnp.array([.36-float(speeds[inward])*5e-9, angles[inward], .3])
    particles = Particles(initial_xyz=jnp.vstack([particles.initial_xyz, moving]),
                          initial_vparallel_over_v=jnp.array([.65, .65]))
    trace = Tracing(field=field, particles=particles, model="GuidingCenterAdaptative",
                    perturbation=adapter, maxtime=2e-8, timestep=1e-9, times_to_trace=3,
                    max_steps=32, progress=False, devices=jax.devices()[:1])
    assert trace.status[0] == 6
    assert trace.axis_stopped[0] and trace.axis_stop_times[0] == 0.
    assert trace.status[1] == 6 and 0 < trace.axis_stop_times[1] < 2e-8
    assert trace.axis_stop_states[1, 0] == pytest.approx(.36, abs=1e-9)
    assert not trace.failed.any() and not trace.boundary_hits.any()
    np.testing.assert_allclose(trace.axis_stop_states[0, :3], particles.initial_xyz[0])
