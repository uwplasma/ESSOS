import os
from functools import partial
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import jit, grad, block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from essos.coils import Coils_from_json
from essos.constants import PROTON_MASS, ONE_EV, ELEMENTARY_CHARGE
from essos.fields import BiotSavart
from essos.dynamics import Tracing, Particles


# Input parameters
tmax_fl = 50000
tmax_gc = 1e-3
tmax_fo = 1e-3

nparticles = number_of_processors_to_use*1
nfieldlines = number_of_processors_to_use*8
s = 0.25 # s-coordinate: flux surface label
trace_tolerance = 1e-15
dt_fo = 1e-9
dt_gc = 1e-7
timesteps_gc = int(tmax_gc/dt_gc)
timesteps_fo = int(tmax_fo/dt_fo)
mass = PROTON_MASS
energy = 4000*ONE_EV
print("cyclotron period:", 1/(ELEMENTARY_CHARGE*0.3/mass))

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), '../examples/input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

R0_fieldlines = jnp.linspace(1.21, 1.41, nfieldlines)
R0_particles= jnp.linspace(1.21, 1.41, nparticles)
Z0_fieldlines = jnp.zeros(nfieldlines)
Z0_particles = jnp.zeros(nparticles)
phi0_fieldlines = jnp.zeros(nfieldlines)
phi0_particles = jnp.zeros(nparticles)

initial_xyz_fieldlines=jnp.array([R0_fieldlines*jnp.cos(phi0_fieldlines), R0_fieldlines*jnp.sin(phi0_fieldlines), Z0_fieldlines]).T
initial_xyz_particles=jnp.array([R0_particles*jnp.cos(phi0_particles), R0_particles*jnp.sin(phi0_particles), Z0_particles]).T

particles = Particles(initial_xyz=initial_xyz_particles, mass=mass, energy=energy, field=field, min_vparallel_over_v=0.8)

# Trace in ESSOS
# time0 = time()
# tracing_fl = Tracing(field=field, model='FieldLine', initial_conditions=initial_xyz_fieldlines,
#                      maxtime=tmax_fl, timesteps=tmax_fl*10, tol_step_size=trace_tolerance)
# block_until_ready(tracing_fl)
# print(f"ESSOS tracing of {nfieldlines} field lines took {time()-time0:.2f} seconds")

time0 = time()
tracing_fo = Tracing(field=field, model='FullOrbit', particles=particles, maxtime=tmax_fo,
                     timesteps=timesteps_fo, tol_step_size=trace_tolerance)
# tracing_fo.trajectories = tracing_fo.trajectories[:, 0::100, :]
# tracing_fo.times = tracing_fo.times[0::100]
# tracing_fo.energy = tracing_fo.energy[:, 0::100]
block_until_ready(tracing_fo)
print(f"ESSOS tracing of {nparticles} particles with FO for {tmax_fo:.1e}s took {time()-time0:.2f} seconds")

time0 = time()
tracing_gc = Tracing(field=field, model='GuidingCenter', particles=particles, maxtime=tmax_gc,
                     timesteps=timesteps_gc, tol_step_size=trace_tolerance)
block_until_ready(tracing_gc)
print(f"ESSOS tracing of {nparticles} particles with GC for {tmax_gc:.1e}s took {time()-time0:.2f} seconds")

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(projection='3d')
# coils.plot(ax=ax, show=False)
# tracing_fl.plot(ax=ax, show=False)
# plt.tight_layout()

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(projection='3d')
# coils.plot(ax=ax, show=False)
# tracing_fo.plot(ax=ax, show=False)
# plt.tight_layout()

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(projection='3d')
# coils.plot(ax=ax, show=False)
# tracing_gc.plot(ax=ax, show=False)
# plt.tight_layout()

# fig, ax = plt.subplots(figsize=(9, 6))
# time0 = time()
# tracing_fl.poincare_plot(ax=ax, shifts=[jnp.pi/2], show=False, s=0.5)
# print(f"ESSOS Poincare plot of {nfieldlines} field lines took {time()-time0:.2f} seconds")
# plt.xlabel('R (m)')
# plt.ylabel('Z (m)')
# ax.set_xlim(0.3, 1.3)
# ax.set_ylim(-0.3, 0.3)
# plt.grid(visible=False)
# plt.tight_layout()
# plt.savefig(os.path.join(os.path.dirname(__file__), 'poincare_plot_fl.png'), dpi=300)
# plt.savefig(os.path.join(os.path.dirname(__file__), "../../../../UW/article/figures/" , 'poincare_plot_fl.png'), dpi=300)


# fig, ax = plt.subplots(figsize=(9, 6))
# time0 = time()
# tracing_fo.poincare_plot(ax=ax, shifts=[jnp.pi/2], show=False)
# print(f"ESSOS Poincare plot of {nparticles} particles took {time()-time0:.2f} seconds")
# plt.xlabel('R (m)')
# plt.ylabel('Z (m)')
# plt.xlim(0.3, 1.3)
# plt.ylim(-0.3, 0.3)
# plt.grid(visible=False)
# plt.tight_layout()
# plt.savefig(os.path.join(os.path.dirname(__file__), 'poincare_plot_fo.png'), dpi=300)
# plt.savefig(os.path.join(os.path.dirname(__file__), "../../../../UW/article/figures/" , 'poincare_plot_fo.png'), dpi=300)


# fig, ax = plt.subplots(figsize=(9, 6))
# time0 = time()
# tracing_gc.poincare_plot(ax=ax, shifts=[jnp.pi/2], show=False)
# print(f"ESSOS Poincare plot of {nparticles} particles took {time()-time0:.2f} seconds")
# plt.xlabel('R (m)')
# plt.ylabel('Z (m)')
# ax.set_xlim(0.3, 1.3)
# ax.set_ylim(-0.3, 0.3)
# plt.grid(visible=False)
# plt.tight_layout()
# plt.savefig(os.path.join(os.path.dirname(__file__), 'poincare_plot_gc.png'), dpi=300)
# plt.savefig(os.path.join(os.path.dirname(__file__), "../../../../UW/article/figures/" , 'poincare_plot_gc.png'), dpi=300)

# plt.show()