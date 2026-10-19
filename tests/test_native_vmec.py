"""Native orbit solvers with the unchanged, file-backed VMEC field."""
from pathlib import Path

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
from essos.dynamics import Particles, Tracing
from essos.fields import Vmec


@pytest.fixture(scope="module")
def vmec_reference():
    path = Path(__file__).resolve().parents[1]/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
    field = Vmec(str(path), ntheta=8, nphi=8)
    particles = Particles(jnp.array([[.3, .4, .2], [.2, 1.5, .1]]), initial_vparallel_over_v=[.7, -.5],
                          mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, energy=5e3*ELEMENTARY_CHARGE)
    options = dict(field=field, particles=particles, maxtime=1e-6, times_to_trace=9, progress=False)
    reference = Tracing(**options, model="GuidingCenterAdaptative", timestep=1e-9,
                        solver=diffrax.Dopri8(), rtol=1e-12, atol=1e-12)
    return options, reference


@pytest.mark.parametrize("integrator", ["essos_rk45", "essos_dopri8"])
@pytest.mark.parametrize("model,controller", [("GuidingCenter", "i"),
    ("GuidingCenterAdaptative", "i"), ("GuidingCenterAdaptative", "pi")])
def test_native_vmec_matches_tight_diffrax_and_defaults_to_standard(vmec_reference, integrator, model, controller, monkeypatch):
    import essos.dynamics as dynamics
    def forbidden(*args, **kwargs):
        raise AssertionError("native VMEC path called diffeqsolve")
    monkeypatch.setattr(dynamics, "diffeqsolve", forbidden)
    options, reference = vmec_reference
    trace = Tracing(**options, model=model, timestep=1e-8, integrator=integrator, native_controller=controller)
    assert trace.native_dense_output == "standard" and trace.rtol == trace.atol == 1e-9
    assert not np.asarray(trace.failed).any() and not np.asarray(trace.boundary_hits).any()
    np.testing.assert_allclose(trace.trajectories_xyz, reference.trajectories_xyz, atol=3e-8, rtol=0.)
    np.testing.assert_allclose(trace.trajectories[..., 3]/trace.particles.total_speed,
                               reference.trajectories[..., 3]/trace.particles.total_speed, atol=3e-8, rtol=0.)
    assert np.isfinite(trace.trajectories).all()
