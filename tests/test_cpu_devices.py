import os
import subprocess
import sys

import pytest

CODE = """
import os
os.sched_getaffinity = lambda _pid: {0, 1, 2}
os.cpu_count = lambda: 64
import essos, jax
print(jax.config.jax_num_cpu_devices)
"""


@pytest.mark.parametrize("env, expected", [({}, 3), ({"SLURM_CPUS_PER_TASK": "2"}, 2), ({"ESSOS_CPU_DEVICES": "5"}, 5)])
def test_cpu_devices_follow_the_cores_granted(env, expected):
    clean = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "SLURM_CPUS_PER_TASK", "ESSOS_CPU_DEVICES")}
    out = subprocess.run([sys.executable, "-c", CODE], env=clean | env, capture_output=True, text=True, check=True)
    assert int(out.stdout.split()[-1]) == expected


TRACE_COMMITTED = """
import essos, jax, jax.numpy as jnp, numpy as np
from essos.dynamics import Particles, Tracing
from essos.fields import Vmec, VMEC_WOUT_ARRAYS
assert len(jax.devices()) == 2
vmec = Vmec("examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc", ntheta=8, nphi=8)
committed = {n: jax.device_put(getattr(vmec, n), jax.devices()[0]) for n in VMEC_WOUT_ARRAYS}
particles = Particles(initial_xyz=jnp.array([[0.5, 0.3, 0.2], [0.4, 1.0, 1.0], [0.6, 2.0, 1.0], [0.3, 3.0, 2.0]]),
                      initial_vparallel_over_v=jnp.array([0.3, 0.5, -0.5, 0.2]))

def end_s(scale):
    field = Vmec.from_arrays(nfp=vmec.nfp, ns=vmec.ns, ntheta=8, nphi=8, **{**committed, "bmnc": committed["bmnc"] * scale})
    return Tracing(field=field, model="GuidingCenter", particles=particles, maxtime=1e-6, timestep=1e-8,
                   times_to_trace=3).trajectories[:, -1, 0]

reference = end_s(1.0)
assert reference.shape == (4,) and np.all(np.isfinite(reference))
assert np.isfinite(jax.grad(lambda x: jnp.sum(end_s(x)))(1.0))
print("ok")
"""


def test_tracing_on_several_devices_accepts_field_arrays_committed_to_one_device():
    env = os.environ | {"ESSOS_CPU_DEVICES": "2"}
    env.pop("XLA_FLAGS", None)
    out = subprocess.run([sys.executable, "-c", TRACE_COMMITTED], env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0 and out.stdout.split()[-1] == "ok", out.stderr[-2000:]
