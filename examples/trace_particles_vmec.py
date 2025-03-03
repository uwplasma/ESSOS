import os
from jax import vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import Vmec
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.dynamics import Tracing, Particles

# Input parameters
tmax = 1e-4
nparticles = 5
s = 0.1 # s-coordinate: flux surface label
theta = jnp.linspace(0, 2*jnp.pi, nparticles)
phi = jnp.linspace(0, 2*jnp.pi/2/3, nparticles)
trace_tolerance = 1e-7
num_steps = 1500

# Load coils and field
wout_file = os.path.join(os.path.dirname(__file__), '..', 'tests', 'input_files', "wout_n3are_R7.75B5.7.nc")
field = Vmec(wout_file)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([[s]*nparticles, theta, phi]).T
particles = Particles(initial_xyz=initial_xyz, mass=ALPHA_PARTICLE_MASS,
                      charge=ALPHA_PARTICLE_CHARGE, energy=FUSION_ALPHA_PARTICLE_ENERGY, field=field)

# Trace in ESSOS
tracing = Tracing(field=field, model='GuidingCenter', particles=particles, maxtime=tmax,
                  timesteps=num_steps, tol_step_size=trace_tolerance)
trajectories_ESSOS = tracing.trajectories

# Plot trajectories, velocity parallel to the magnetic field, and energy error
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

tracing.plot(ax=ax1, show=False)

for i, trajectory in enumerate(trajectories_ESSOS):
    ax2.plot(tracing.times, jnp.abs(tracing.energy[i]-particles.energy)/particles.energy, label=f'Particle {i+1}')
    ax3.plot(tracing.times, trajectory[:, 3]/particles.total_speed, label=f'Particle {i+1}')
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
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