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
from jax import block_until_ready

# Input parameters
tmax = 1.e-4
dt_fo=1.e-9
nparticles_per_core=2
nparticles = number_of_processors_to_use*nparticles_per_core
R0 = jnp.linspace(1.23, 1.27, nparticles)
trace_tolerance = 1e-5
num_steps_gc = 5000
num_steps_fo = int(tmax/dt_fo)
mass=PROTON_MASS
energy=5000*ONE_EV

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
initial_vparallel_over_v = jnp.linspace(-0.1, 0.1, nparticles)
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, field=field, initial_vparallel_over_v=initial_vparallel_over_v)

# Trace in ESSOS
time0 = time()
tracing_guidingcenter = Tracing(field=field, model='GuidingCenterAdaptative', particles=particles,
                  maxtime=tmax,times_to_trace=num_steps_gc, atol=trace_tolerance,rtol=trace_tolerance)
trajectories_guidingcenter = block_until_ready(tracing_guidingcenter.trajectories)
print(f"ESSOS guiding center tracing took {time()-time0:.2f} seconds")

time0 = time()
tracing_fullorbit = Tracing(field=field, model='FullOrbit_Boris', particles=particles,
                  maxtime=tmax, times_to_trace=num_steps_fo,timestep=dt_fo)
trajectories_fullorbit = block_until_ready(tracing_fullorbit.trajectories)
print(f"ESSOS full orbit tracing took {time()-time0:.2f} seconds")

# Plot trajectories, velocity parallel to the magnetic field, and energy error
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

coils.plot(ax=ax1, show=False)
tracing_guidingcenter.plot(ax=ax1, show=False)
tracing_fullorbit.plot(ax=ax1, show=False)

for i, (trajectory_gc, trajectory_fo) in enumerate(zip(trajectories_guidingcenter, trajectories_fullorbit)):
    ax2.plot(tracing_guidingcenter.times, jnp.abs(tracing_guidingcenter.energy[i]-particles.energy)/particles.energy, '-', label=f'Particle {i+1} GC', linewidth=1.0, alpha=0.7)
    ax2.plot(tracing_fullorbit.times, jnp.abs(tracing_fullorbit.energy[i]-particles.energy)/particles.energy, '--', label=f'Particle {i+1} FO', linewidth=1.0, markersize=0.5, alpha=0.7)
    def compute_v_parallel(trajectory_t):
        magnetic_field_unit_vector = field.B(trajectory_t[:3]) / field.AbsB(trajectory_t[:3])
        return jnp.dot(trajectory_t[3:], magnetic_field_unit_vector)
    v_parallel_fo = vmap(compute_v_parallel)(trajectory_fo)
    ax3.plot(tracing_guidingcenter.times, trajectory_gc[:, 3] / particles.total_speed, '-', label=f'Particle {i+1} GC', linewidth=1.1, alpha=0.95)
    ax3.plot(tracing_fullorbit.times, v_parallel_fo / particles.total_speed, '--', label=f'Particle {i+1} FO', linewidth=0.5, markersize=0.5, alpha=0.2)
    # ax4.plot(jnp.sqrt(trajectory_gc[:,0]**2+trajectory_gc[:,1]**2), trajectory_gc[:, 2], '-', label=f'Particle {i+1} GC', linewidth=1.5, alpha=0.3)
    # ax4.plot(jnp.sqrt(trajectory_fo[:,0]**2+trajectory_fo[:,1]**2), trajectory_fo[:, 2], '--', label=f'Particle {i+1} FO', linewidth=1.5, markersize=0.5, alpha=0.2)
tracing_guidingcenter.poincare_plot(ax=ax4, show=False, color='k', label=f'GC', shifts=[jnp.pi/2])#, 0])
tracing_fullorbit.poincare_plot(    ax=ax4, show=False, color='r', label=f'FO', shifts=[jnp.pi/2])#, 0])

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Relative Energy Error')
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax2.legend(loc='upper right')
ax3.set_xlabel('Time (s)')
ax3.legend(loc='upper right')
ax4.set_xlabel('R (m)')
ax4.set_ylabel('Z (m)')
ax4.legend(loc='upper right')
plt.tight_layout()
plt.show()


## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')