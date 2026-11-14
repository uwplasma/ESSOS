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
