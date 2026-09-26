import os
number_of_processors_to_use = 4 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import dataclasses
from time import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import vmex
from essos.coils import Coils
from essos.fields import ExternalField
from essos.surfaces import SurfaceRZFourier, SurfaceClassifier
from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from essos.dynamics import Tracing, Particles

# Guiding centers from inside a VMEX equilibrium, through the LCFS, to a wall.
# VMEX solves the free-boundary vacuum equilibrium of the Landreman-Paul QA
# coils, and gives the field outside the LCFS (VmecExtender on the tabulated
# coil field). Inside, ESSOS traces in VMEC coordinates; an orbit that crosses
# the LCFS continues in Cartesian coordinates until it strikes the wall or
# comes back inside. Requires VMEX (pip install vmex); about a minute on a laptop.

# Input parameters
tmax = 4e-5
timestep = 1.e-9
times_to_trace = 2000
nparticles_per_core = 8
nparticles = number_of_processors_to_use*nparticles_per_core
energy = 3e3*ONE_EV
wall_gap = 0.03 # wall distance outside the LCFS [m]
atol = 1e-8
rtol = 1e-8

# Solve the free-boundary equilibrium of the coils with VMEX
input_dir = os.path.join(os.path.dirname(__file__), "..", "input_files")
coils = Coils.from_json(os.path.join(input_dir, "ESSOS_biot_savart_LandremanPaulQA.json"))
coil_grid = vmex.MgridField.from_coils(coils, rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6, ir=96, jz=96, kp=32)
inp = vmex.VmecInput.from_file(os.path.join(input_dir, "input.LandremanPaul2021_QA_lowres")).change_resolution(mpol=5, ntor=5)
inp = dataclasses.replace(inp, ns_array=[16], niter_array=[4000], ftol_array=[1e-10], lfreeb=True,
                          mgrid_file="essos_coils(direct)", nzeta=16, phiedge=-0.025)
time0 = time()
result = vmex.solve_free_boundary_multigrid(inp, external_field=coil_grid, raise_on_max_iterations=False)
wout = vmex.wout_from_state(inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                            fsql=float(result.fsql), niter=int(result.iterations), converged=bool(result.converged),
                            vacuum_output=result.vacuum)
print(f"VMEX free-boundary solve took {time()-time0:.1f} seconds, converged: {bool(result.converged)}")

# Fields: VMEC inside; outside, the tricubic (C1) coil table through VmecExtender
vmec = vmex.essos_vmec_field(wout)
outside = ExternalField(vmex.VmecExtender.from_wout(wout, external_field=dataclasses.replace(coil_grid, order=3),
                                                    plasma="vacuum"))

# Wall: the LCFS with its m = 1 modes grown by wall_gap
lcfs = SurfaceRZFourier.from_vmec(vmec, ntheta=64, nphi=128)
m1 = (lcfs.xm == 1) & (lcfs.xn == 0)
wall = SurfaceRZFourier(lcfs.rc + wall_gap * m1, lcfs.zs + wall_gap * jnp.sign(lcfs.zs[m1][0]) * m1, lcfs.nfp,
                        lcfs.mpol, lcfs.ntor, ntheta=64, nphi=128)
wall._xm, wall._xn = lcfs.xm, lcfs.xn
time0 = time()
wall_classifier = SurfaceClassifier(wall, h=0.02, padding=0.05)
print(f"Wall classifier took {time()-time0:.1f} seconds")

# Initialize particles inside the LCFS
key_s, key_theta, key_phi, key_pitch = jax.random.split(jax.random.key(0), 4)
initial_xyz = jnp.stack([jax.random.uniform(key_s, (nparticles,), minval=0.5, maxval=0.95),
                         jax.random.uniform(key_theta, (nparticles,), maxval=2*jnp.pi),
                         jax.random.uniform(key_phi, (nparticles,), maxval=2*jnp.pi)], axis=1)
particles = Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=jax.random.uniform(key_pitch, (nparticles,), minval=-1, maxval=1),
                      mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, energy=energy)

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=vmec, model='GuidingCenterAdaptative', particles=particles, maxtime=tmax, timestep=timestep,
                  times_to_trace=times_to_trace, atol=atol, rtol=rtol, exterior_field=outside, wall=wall_classifier)
print(f"ESSOS tracing of {nparticles} particles during {tmax}s took {time()-time0:.2f} seconds")
crossed = np.isfinite(tracing.lcfs_times)
print(f"Stayed inside the LCFS:        {np.mean(~crossed)*100:.1f}%")
print(f"Crossed the LCFS and returned: {np.mean(crossed & (tracing.returns > 0) & ~tracing.wall_hits)*100:.1f}%")
print(f"Struck the wall:               {np.mean(tracing.wall_hits)*100:.1f}%")
print(f"Outside the LCFS at tmax:      {np.mean(tracing.status == 2)*100:.1f}%")
print(f"Failed:                        {np.mean(tracing.failed)*100:.1f}%")
if tracing.wall_hits.any():
    print(f"Largest energy error at the wall: {np.max(np.abs(tracing.wall_energies[tracing.wall_hits]/energy - 1)):.1e}")

# Plot the LCFS, the wall and the trajectories
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
lcfs.plot(ax=ax1, show=False, alpha=0.3)
xyz = np.asarray(tracing.trajectories_xyz)
for i in range(nparticles):
    ax1.plot(xyz[i, :, 0], xyz[i, :, 1], xyz[i, :, 2], lw=0.7)
ax1.scatter(*tracing.wall_positions[tracing.wall_hits].T, color='k', s=10, label='wall strikes')
ax1.legend()
for surface, label in ((lcfs, 'LCFS'), (wall, 'wall')):
    gamma = np.asarray(surface.gamma)[0]
    ax2.plot(np.hypot(gamma[:, 0], gamma[:, 1]), gamma[:, 2], label=label)
R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
period = 2*np.pi/vmec.nfp
phi = np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]) + period/2, period) - period/2
near = np.abs(phi) < 0.02 # points near the phi = 0 plane
inside = near & (tracing.region == 0)
ax2.scatter(R[inside], Z[inside], s=2, color='tab:blue', label='inside the LCFS')
ax2.scatter(R[near & (tracing.region == 1)], Z[near & (tracing.region == 1)], s=2, color='tab:red', label='outside the LCFS')
ax2.set_xlabel('R [m]')
ax2.set_ylabel('Z [m]')
ax2.set_aspect('equal')
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()
