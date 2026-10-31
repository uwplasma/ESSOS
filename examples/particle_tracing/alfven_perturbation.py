"""Guiding centres in a VMEC equilibrium with and without a prescribed shear-Alfven wave.

The wave Phi = 2e4 V s^(3/2) sin(3 theta - 2 phi + omega t) with alpha = (n B^phi - m B^theta) Phi / (omega B^2)
has E_parallel = 0, constant magnetic moment and physical energy exchange (see docs/analytical_perturbations.rst).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from essos.dynamics import Particles, Tracing
from essos.fields import Vmec
from essos.gc_perturbations import AnalyticalVmecAlfvenMode

field = Vmec(str(Path(__file__).resolve().parents[1]/"input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"),
             ntheta=8, nphi=8, mode_tolerance=.01)
particles = Particles(initial_xyz=jnp.array([[.35, .7, .3], [.4, 1.2, .5]]),
                      initial_vparallel_over_v=jnp.array([.65, -.7]))
wave = AnalyticalVmecAlfvenMode(3, 2, 9e5, 2e4, radial_center=.4, radial_width=.5)

for name, perturbation in (("equilibrium", None), ("wave", wave)):
    trace = Tracing(field=field, particles=particles, model="GuidingCenterAdaptative", perturbation=perturbation,
                    maxtime=2e-7, timestep=1e-8, times_to_trace=9, rtol=1e-9, atol=1e-10, max_steps=1024,
                    progress=False, devices=jax.devices()[:1])
    energy = np.asarray(trace.energy())/float(particles.energy) - 1
    print(f"{name:12s} failed {int(trace.failed.sum())}  max |dE/E| {np.max(abs(energy)):.2e}"
          + ("" if perturbation is None else
             f"  max |dmu/mu| {np.max(abs(np.asarray(trace.trajectories)[:, :, 4]/trace.trajectories[:, :1, 4]-1)):.1e}"))
