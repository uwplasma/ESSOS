import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from jax import vmap
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.constants import PROTON_MASS, ONE_EV
from essos.dynamics import Tracing, Particles

# Input parameters
tmax = 1e-4
dt=1.e-8
nparticles_per_core=2
nparticles = number_of_processors_to_use*nparticles_per_core
R0 = jnp.linspace(1.23, 1.27, nparticles)
trace_tolerance = 1e-7
num_steps = 3000
mass=PROTON_MASS
energy=4000*ONE_EV

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, field=field)

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=field, model='FullOrbit_Boris', particles=particles,
                  maxtime=tmax, times_to_trace=num_steps,timestep=dt, atol=trace_tolerance,rtol=trace_tolerance)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
trajectories = tracing.trajectories

# Plot trajectories, velocity parallel to the magnetic field, and energy error
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

coils.plot(ax=ax1, show=False)
tracing.plot(ax=ax1, show=False)

for i, trajectory in enumerate(trajectories):
    ax2.plot(tracing.times, jnp.abs(tracing.energy[i]-particles.energy)/particles.energy, label=f'Particle {i+1}', linewidth=0.2)
    def compute_v_parallel(trajectory_t):
        magnetic_field_unit_vector = field.B(trajectory_t[:3]) / field.AbsB(trajectory_t[:3])
        return jnp.dot(trajectory_t[3:], magnetic_field_unit_vector)
    v_parallel = vmap(compute_v_parallel)(trajectory)
    ax3.plot(tracing.times, v_parallel / particles.total_speed, label=f'Particle {i+1}')
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}', linewidth=0.2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Relative Energy Error')
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax2.legend()
ax3.set_xlabel('Time (s)')
ax3.legend()
ax4.set_xlabel('R (m)')
ax4.set_ylabel('Z (m)')
ax4.legend()
plt.tight_layout()
plt.show()

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')