"""Field lines and alpha particles in an MRX field, compared with VMEC.

MRX (https://github.com/ToBlick/mrx) computes magnetic fields as spline
2-forms on a map of the plasma domain. Here it computes the vacuum field of
the Landreman-Paul QA domain read from a VMEC wout file, and ESSOS traces field
lines and guiding centers in it and in the VMEC field of the same file.
With an MRX checkpoint, ``python trace_particles_mrx.py GEOMETRY CHECKPOINT``
traces the relaxed state stored there instead, for example a state with
magnetic islands from MRX's ``scripts/tutorials/4_li383_island_seed.py``.
Needs ``pip install mrx`` (Python >= 3.11).
"""
import os
import sys
from time import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from essos.fields import MRXField, Vmec
from essos.dynamics import Tracing, Particles

input_files = os.path.join(os.path.dirname(__file__), "..", "input_files")
geometry = sys.argv[1] if len(sys.argv) > 1 else os.path.join(input_files, "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
checkpoint = sys.argv[2] if len(sys.argv) > 2 else None
periods, n_lines, n_particles, tmax = 300, 16, 8, 1e-4

time0 = time()
mrx = MRXField.from_mrx(geometry, checkpoint, resolution=(12, 16, 16))
vmec = Vmec(geometry)
print(f"MRX field built in {time() - time0:.1f} s")
# MRX fixes the L2 norm of B, not its strength: scale it to the VMEC field on the axis.
mrx = float(vmec.AbsB(jnp.array([1e-6, 0., 0.])) / mrx.AbsB(jnp.array([1e-6, 0., 0.]))) * mrx

# Compare the two fields at the same physical points, s in [0.1, 0.9].
u = jax.random.uniform(jax.random.PRNGKey(0), (500, 3))
points = jnp.stack([0.1 + 0.8 * u[:, 0], 2 * np.pi * u[:, 1], 2 * np.pi * u[:, 2]], 1)
difference = mrx.compare(vmec, points)
print(f"|B_MRX - B_VMEC| / |B_MRX|: mean {jnp.mean(difference):.2e}, max {jnp.max(difference):.2e}")

# Field lines, sampled once per field period on the phi = 0 plane.
s0 = jnp.linspace(0.02, 0.95, n_lines)
x0 = jnp.stack([s0, 0 * s0, 0 * s0], 1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (name, field) in zip(axes, (("VMEC", vmec), ("MRX", mrx))):
    time0 = time()
    lines = Tracing(field=field, model="FieldLineToroidal", initial_conditions=x0,
                    maxtime=2 * np.pi * periods / field.nfp, times_to_trace=periods + 1, atol=1e-10, rtol=1e-10)
    xyz = np.asarray(lines.trajectories_xyz)
    trajectories = np.asarray(lines.trajectories)
    iota = (trajectories[:, -1, 1] - trajectories[:, 0, 1]) / (trajectories[:, -1, 2] - trajectories[:, 0, 2])
    print(f"{name}: {n_lines} field lines over {periods} periods in {time() - time0:.1f} s, "
          f"iota from {iota[1]:+.4f} to {iota[-1]:+.4f}")
    ax.scatter(np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2], s=0.3, c="k")
    ax.set_title(f"{name}, $\\phi = 0$"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]"); ax.set_aspect("equal")

# Fusion alphas, started on s = 0.25 at different poloidal angles.
theta = jnp.linspace(0, 2 * np.pi, n_particles, endpoint=False)
initial = jnp.stack([0.25 + 0 * theta, theta, 0 * theta], 1)
for name, field in (("VMEC", vmec), ("MRX", mrx)):
    particles = Particles(initial_xyz=initial, field=field)
    time0 = time()
    tracing = Tracing(field=field, model="GuidingCenterAdaptative", particles=particles, maxtime=tmax,
                      times_to_trace=500, atol=1e-8, rtol=1e-8)
    s = np.asarray(tracing.trajectories[..., 0])
    print(f"{name}: {n_particles} guiding centers over {tmax:.0e} s in {time() - time0:.1f} s, "
          f"loss fraction {float(tracing.loss_fractions[-1]):.2f}")
    axes[2].plot(np.asarray(tracing.times) * 1e3, s.T, "-" if name == "MRX" else "--", lw=0.8,
                 color="C1" if name == "MRX" else "C0")
axes[2].set_xlabel("t [ms]"); axes[2].set_ylabel("s"); axes[2].set_title("alpha particles: MRX (solid), VMEC (dashed)")
plt.tight_layout()
plt.show()
