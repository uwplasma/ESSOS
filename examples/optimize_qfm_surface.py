import os
number_of_processors_to_use = 3  # Parallelization
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time
from jax import block_until_ready

from essos.dynamics import Tracing
from essos.surfaces import BdotN_over_B, toroidal_flux
from essos.surfaces import SurfaceRZFourier
from essos.qfm import QfmSurface 
from essos.fields import Vmec, BiotSavart

# Load initial guess surface
ntheta=35
nphi=36
vmec = os.path.join('input_files','input.rotating_ellipse')
surf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True)
surf.change_resolution(6,6)

initialsurf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True)

# Load target VMEC surface
truevmec = Vmec(os.path.join(os.path.dirname(__file__), 'input_files', 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
                ntheta=ntheta, nphi=nphi, range_torus='half period', close=True)

# Load coils and construct field
from essos.coils import Coils_from_json
coils = Coils_from_json("input_files/stellarator_coils.json") # from optimize_coils_vmec_surface.py
field = BiotSavart(coils)

# QFM optimization setup
method = 'lbfgs'
label = 'toroidal_flux'
initial_label = toroidal_flux(surf, field)
targetlabel = toroidal_flux(truevmec.surface, field)
tol = 1e-6
constraint_weight = 1e10
maxiter = 1000

BdotN_over_B_initial = BdotN_over_B(surf, BiotSavart(coils))

# Initialize QFM optimizer
qfm = QfmSurface(field=field, surface=surf, label=label, targetlabel=targetlabel)

print("Degrees of Freedom:", qfm.surface.x.shape[0])
result = qfm.run(tol=tol, maxiter=maxiter, method=method, constraint_weight=constraint_weight)

# Evaluate final objective and constraint
x_opt = result["s"].x
qfm_loss = float(jnp.asarray(qfm.objective(x_opt)))
c_loss = float(jnp.asarray(qfm.constraint(x_opt)))

BdotN_over_B_optimized = BdotN_over_B(result['s'], BiotSavart(coils))
print("Optimization method:", method)
print("Optimization label:", label)
print("Optimization success:", result['success'])
print(f"final qfm objective = {qfm_loss:.3e}, final constraint objective = {c_loss:.3e}")
print("Iterations:", result['iter'])
print(f"initial label: {initial_label}, target label: {targetlabel}, final label: {toroidal_flux(result['s'], field)}")
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")

# Plot surfaces
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# coils.plot(ax=ax1, show=False)
# coils.plot(ax=ax2, show=False)
# coils.plot(ax=ax3, show=False)


initialsurf.plot(ax=ax1, show=False)
truevmec.surface.plot(ax=ax2, show=False)
result['s'].plot(ax=ax3, show=False)

ax1.set_title("Initial Surface")
ax2.set_title("True VMEC Surface")
ax3.set_title("Final Surface")

plt.tight_layout()
plt.show()

# Field line tracing
tmax = 10000000000
nfieldlines_per_core = 3
nfieldlines = nfieldlines_per_core * number_of_processors_to_use
R0 = jnp.linspace(12.2, 13.5, nfieldlines)
trace_tolerance = 1e-7
num_steps = 60000

Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T

time0 = time()
tracing = block_until_ready(Tracing(
    field=field,
    model='FieldLineAdaptative',
    initial_conditions=initial_xyz,
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance
))
print(f"ESSOS tracing took {time() - time0:.2f} seconds")

trajectories = tracing.trajectories
traj = trajectories[0]
R, phi, Z = traj[:, 0], traj[:, 1], traj[:, 2]

phi_u = jnp.unwrap(phi)
phi0_cross = jnp.where((phi_u[:-1] < 0) & (phi_u[1:] >= 0))[0]
phi90_cross = jnp.where((phi_u[:-1] < jnp.pi / 2) & (phi_u[1:] >= jnp.pi / 2))[0]

theta = jnp.linspace(0, 2 * jnp.pi, 200)

def compute_rz_on_phi(surface, theta, phi=0.0):
    angles = jnp.outer(theta, surface.xm) - phi * surface.xn
    R = jnp.sum(surface.rmnc_interp * jnp.cos(angles), axis=1)
    Z = jnp.sum(surface.zmns_interp * jnp.sin(angles), axis=1)
    return R, Z

# Contours from optimized surface
R0_opt, Z0_opt = compute_rz_on_phi(result['s'], theta, phi=0.0)
R90_opt, Z90_opt = compute_rz_on_phi(result['s'], theta, phi=jnp.pi/2)

# Contours from true VMEC surface
R0_true, Z0_true = compute_rz_on_phi(truevmec.surface, theta, phi=0.0)
R90_true, Z90_true = compute_rz_on_phi(truevmec.surface, theta, phi=jnp.pi/2)

fig, ax = plt.subplots(figsize=(6, 6))

tracing.poincare_plot(ax=ax, show=False, shifts=[0, jnp.pi / 2])
ax.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 0$")
ax.plot(R90_opt, Z90_opt, color='black', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2$")
ax.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
ax.plot(R90_true, Z90_true, color='blue', linestyle='--', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")

ax.set_xlabel("R")
ax.set_ylabel("Z")
ax.set_title("Poincaré + Surfaces Comparison @ φ = 0 and π/2")
ax.legend()
ax.axis("equal")
plt.tight_layout()
plt.show()
