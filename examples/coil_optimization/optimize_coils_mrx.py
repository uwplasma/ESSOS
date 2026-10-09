"""Coils that reproduce an MRX field.

MRX (https://github.com/ToBlick/mrx) computes the vacuum field of the
Landreman-Paul QA domain. In vacuum the field inside is fixed by its values
on the boundary, so the coils are optimized to match the full MRX vector field
(B.n = 0 and the tangential part) on the boundary, with length and curvature
penalties. The result is checked inside the volume and with field lines.
Needs ``pip install mrx`` (Python >= 3.11).
"""
import os
from time import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart, MRXField
from essos.losses import custom_loss
from essos.dynamics import Tracing

wout = os.path.join(os.path.dirname(__file__), "..", "input_files", "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
N_COILS, ORDER, R0, R1 = 4, 4, 10.0, 4.5
LENGTH_TARGET, CURVATURE_TARGET = 35.0, 0.5
MAX_EVALUATIONS = 1000

mrx = MRXField.from_mrx(wout, resolution=(12, 16, 16))
mrx = float(5.7 / mrx.AbsB(jnp.array([1e-6, 0., 0.]))) * mrx  # 5.7 T on the axis

# Target: B on 24 x 48 boundary points of one field period; stellarator symmetry gives the rest.
theta, phi = [a.ravel() for a in jnp.meshgrid(jnp.linspace(0, 2 * np.pi, 24, endpoint=False),
                                              jnp.linspace(0, 2 * np.pi / mrx.nfp, 48, endpoint=False))]
boundary = jnp.stack([jnp.ones_like(theta), theta, phi], 1)
xyz, B_target = jax.vmap(mrx.to_xyz)(boundary), jax.vmap(mrx.B)(boundary)


def mismatch(field):
    return jnp.mean(jnp.sum((jax.vmap(field.B)(xyz) - B_target)**2, 1) / jnp.sum(B_target**2, 1))


curves = CreateEquallySpacedCurves(N_COILS, ORDER, R0, R1, n_segments=60, nfp=mrx.nfp, stellsym=True)
loss = (custom_loss(mismatch, "field")
        + 1e-4 * custom_loss(lambda f: jnp.mean(jnp.maximum(0, f.coils.length - LENGTH_TARGET)**2), "field")
        + 1e-1 * custom_loss(lambda f: jnp.mean(jnp.maximum(0, f.coils.curvature - CURVATURE_TARGET)**2), "field"))
loss.dependencies = {"field": BiotSavart(Coils(curves=curves, currents=[1e7] * N_COILS))}
time0 = time()
result = least_squares(loss, loss.starting_dofs, loss.grad, ftol=1e-10, gtol=1e-10, xtol=1e-14, max_nfev=MAX_EVALUATIONS)
field = loss.dofs_to_pytree(result.x)["field"]
print(f"Optimization: {result.nfev} evaluations in {time() - time0:.1f} s, "
      f"boundary mismatch {jnp.sqrt(mismatch(field)):.2e} (rms of |dB|/|B|)")

# Inside the volume, at the same physical points.
u = jax.random.uniform(jax.random.PRNGKey(0), (2000, 3))
points = jnp.stack([0.02 + 0.96 * u[:, 0], 2 * np.pi * u[:, 1], 2 * np.pi * u[:, 2]], 1)
difference = mrx.compare(field, points)
print(f"Volume: |B_coils - B_MRX| / |B_MRX| mean {jnp.mean(difference):.2e}, max {jnp.max(difference):.2e}")

# Field lines of both fields from the same points of the phi = 0 plane.
s0 = jnp.linspace(0.05, 0.85, 8)
x0 = jnp.stack([s0, 0 * s0, 0 * s0], 1)
periods = 100
in_mrx = Tracing(field=mrx, model="FieldLineToroidal", initial_conditions=x0, maxtime=2 * np.pi * periods / mrx.nfp,
                 times_to_trace=periods + 1, atol=1e-10, rtol=1e-10)
in_coils = Tracing(field=field, model="FieldLineArclength", initial_conditions=jax.vmap(mrx.to_xyz)(x0),
                   maxtime=periods * 2 * np.pi * R0 / mrx.nfp, times_to_trace=40 * periods, atol=1e-10, rtol=1e-10)

fig = plt.figure(figsize=(11, 4.5))
ax = fig.add_subplot(121, projection="3d")
field.coils.plot(ax=ax, show=False)
ax.scatter(*np.asarray(xyz).T, s=0.3, c="gray")
ax = fig.add_subplot(122)
ax.set_title(r"field lines on $\phi = 0$: MRX (orange), coils (blue)")
# Crossings of the coil field lines with the phi = 0 planes of every field period, linearly interpolated.
x = np.asarray(in_coils.trajectories)[..., :3]
period = np.unwrap(np.arctan2(x[..., 1], x[..., 0]), axis=1) * mrx.nfp / (2 * np.pi)
i, j = np.nonzero(np.diff(np.floor(period), axis=1))
w = ((np.floor(period[i, j + 1]) - period[i, j]) / (period[i, j + 1] - period[i, j]))[:, None]
crossing = (1 - w) * x[i, j] + w * x[i, j + 1]
ax.scatter(np.hypot(crossing[:, 0], crossing[:, 1]), crossing[:, 2], s=0.5, c="C0")
y = np.asarray(in_mrx.trajectories_xyz)
ax.scatter(np.hypot(y[..., 0], y[..., 1]), y[..., 2], s=0.5, c="C1")
ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
plt.tight_layout()
plt.show()
