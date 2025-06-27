import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from jax import vmap
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.coils import Coils
from essos.constants import PROTON_MASS, ONE_EV, ELEMENTARY_CHARGE
from essos.dynamics import Tracing, Particles
from jax import block_until_ready

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), '../examples/input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils.from_json(json_file)
field = BiotSavart(coils)

# Particle parameters
nparticles = number_of_processors_to_use
mass=PROTON_MASS
energy=5000*ONE_EV
cyclotron_frequency = ELEMENTARY_CHARGE*0.3/mass
print("cyclotron period:", 1/cyclotron_frequency)

# Particles initialization
initial_xyz=jnp.array([[1.23, 0, 0]])

particles_passing = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, initial_vparallel_over_v=[0.1], phase_angle_full_orbit=0)
particles_traped = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, initial_vparallel_over_v=[0.9], phase_angle_full_orbit=0)
particles = particles_passing.join(particles_traped, field=field)

# Tracing parameters
tmax = 1e-3
trace_tolerance = 1e-14
dt_gc = 1e-7
dt_fo = 1e-9
num_steps_gc = int(tmax/dt_gc)
num_steps_fo = int(tmax/dt_fo)

# Trace in ESSOS
time0 = time()
tracing_gc = Tracing(field=field, model='GuidingCenter', particles=particles,
                  maxtime=tmax, timesteps=num_steps_gc, tol_step_size=trace_tolerance)
trajectories_guidingcenter = block_until_ready(tracing_gc.trajectories)
print(f"ESSOS guiding center tracing took {time()-time0:.2f} seconds")

time0 = time()
tracing_fo = Tracing(field=field, model='FullOrbit', particles=particles, maxtime=tmax,
                     timesteps=num_steps_fo, tol_step_size=trace_tolerance)
block_until_ready(tracing_fo.trajectories)
print(f"ESSOS full orbit tracing took {time()-time0:.2f} seconds")

# Plot trajectories, velocity parallel to the magnetic field, and energy error
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(projection='3d')
coils.plot(ax=ax, show=False)
tracing_gc.plot(ax=ax, show=False, color='black', linewidth=2)
tracing_fo.plot(ax=ax, show=False)
plt.tight_layout()

plt.figure(figsize=(9, 6))
plt.plot(tracing_gc.times*1000, jnp.abs(tracing_gc.energy()[0]/particles.energy-1), label='Guiding Center', color='red')
plt.plot(tracing_fo.times*1000, jnp.abs(tracing_fo.energy()[0]/particles.energy-1), label='Full Orbit', color='blue')
plt.xlabel('Time (ms)')
plt.ylabel('Relative Energy Error')
plt.xlim(0, tmax*1000)
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energies.png'), dpi=300)
plt.savefig(os.path.join(os.path.dirname(__file__), "../../../../UW/article/figures/" ,'energies.png'), dpi=300)

plt.show()

## Save results in vtk format to analyze in Paraview
# tracing_gc.to_vtk(os.path.join(output_dir, 'trajectories_gc'))
# tracing_fo.to_vtk(os.path.join(output_dir, 'trajectories_fo'))
# coils.to_vtk(os.path.join(output_dir, 'coils'))