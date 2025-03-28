import os
number_of_processors_to_use = 100 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import Vmec
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.dynamics import Tracing, Particles
import numpy as np

# Input parameters
tmax = 1e-3
nparticles = number_of_processors_to_use
s = 0.25 # s-coordinate: flux surface label
theta = jnp.linspace(0, 2*jnp.pi, nparticles)
phi = jnp.linspace(0, 2*jnp.pi/2/3, nparticles)
trace_tolerance = 1e-4
num_steps_to_plot = 250
energy=FUSION_ALPHA_PARTICLE_ENERGY/10

# Load coils and field
wout_file = os.path.join(os.path.dirname(__file__), 'input_files', "wout_n3are_R7.75B5.7.nc")
vmec = Vmec(wout_file)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([[s]*nparticles, theta, phi]).T
particles = Particles(initial_xyz=initial_xyz, mass=ALPHA_PARTICLE_MASS,
                      charge=ALPHA_PARTICLE_CHARGE, energy=energy, field=vmec)

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=vmec, model='GuidingCenter', particles=particles, maxtime=tmax,
                  timesteps=num_steps_to_plot, tol_step_size=trace_tolerance)
print(f"ESSOS tracing of {nparticles} particles during {tmax}s took {time()-time0:.2f} seconds")
print(f"Final loss fraction: {tracing.loss_fractions[-1]*100:.2f}%")
trajectories = tracing.trajectories

# Plot trajectories, velocity parallel to the magnetic field, loss fractions and/or energy error
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# Plot 5 random particles
## Plot trajectories in 3D
vmec.surface.plot(ax=ax1, show=False, alpha=0.4)
tracing.plot(ax=ax1, show=False, n_trajectories_plot=5)
for i in np.random.choice(nparticles, 5, replace=False):
    trajectory = trajectories[i]
    ## Plot energy error
    ax2.plot(tracing.times[2:], jnp.abs(tracing.energy[i][2:]-particles.energy)/particles.energy, label=f'Particle {i+1}')
    ## Plot velocity parallel to the magnetic field
    ax3.plot(tracing.times, trajectory[:, 3]/particles.total_speed, label=f'Particle {i+1}')
    ## Plot s-coordinate
    # ax4.plot(tracing.times, trajectory[:,0], label=f'Particle {i+1}')
    # ax4.set_ylabel(r'$s=\psi/\psi_b$')
## Plot loss fractions
ax4.plot(tracing.times, tracing.loss_fractions)
ax4.set_ylabel('Loss Fraction');ax4.set_ylim(0, 1);ax4.set_xscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel('Relative Energy Error')
ax2.set_xlabel('Time (s)')
ax3.set_ylim(-1, 1)
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax3.set_xlabel('Time (s)')
ax4.set_xlabel('Time (s)')
plt.tight_layout()
plt.show()

# # Save results in vtk format to analyze in Paraview
# vmec.surface.to_vtk('surface')
# tracing.to_vtk('trajectories')