import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import near_axis_test
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ONE_EV
from essos.dynamics import Tracing, Particles
import numpy as np
#To read from UWplasma stellarator database
import requests

# Load nearaxis configuration from UWplasma stellarator database
url="https://stellarator.physics.wisc.edu/backend/api/configs"


#choose a configuration id from the database
config_id=100
config = next(cfg for cfg in requests.get(url).json()["configs"] if cfg["id"] == config_id)
rc=[1, config["rc1"], config["rc2"], config["rc3"]]
zs=[0, config["zs1"], config["zs2"], config["zs3"]]
nfp=config["nfp"] 
etabar=config["etabar"]
    

# Initialize Near-Axis field QH nfp=4
#rc=jnp.array([1.0,-0.5997117])
#zs=jnp.array([0.0,-0.66064245])
#etabar=1.3164784
#nfp=4


# Initialize Near-Axis field QA nfp=3
#rc=jnp.array([1, 0.045])
#zs=jnp.array([0,-0.045])
#etabar=0.9
#nfp=3

field = near_axis_test(rc=rc, zs=zs, etabar=etabar, nfp=nfp)


# Input parameters
timestep = 1.e-8
times_to_trace=1000
nparticles_per_core=50
nparticles = number_of_processors_to_use*nparticles_per_core
n_particles_to_plot = min(4, nparticles)
atol = 1e-8
rtol = 1e-8
energy=energy=4000*ONE_EV


# Trace fieldlines
tmax = 1.e-4

r = jnp.ones(nparticles)*0.05
phi0=jnp.zeros(nparticles)
initial_xyz=jnp.array([r, phi0, phi0]).T


#particles = Particles(initial_xyz=initial_xyz, mass=ALPHA_PARTICLE_MASS,charge=ALPHA_PARTICLE_CHARGE, energy=energy, field=vmec)
particles = Particles(initial_xyz=initial_xyz, mass=ALPHA_PARTICLE_MASS,charge=ALPHA_PARTICLE_CHARGE, energy=energy, field=field)

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=field, model='GuidingCenterAdaptative', particles=particles, maxtime=tmax,
                  timestep=timestep,times_to_trace=times_to_trace, atol=atol,rtol=rtol,r_max=1.8)
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
field.plot(ax=ax1, show=False,r=0.2,alpha=0.3,cmap='jet')
tracing.plot(ax=ax1, show=False, n_trajectories_plot=nparticles)
for i in np.random.choice(nparticles, size=n_particles_to_plot, replace=False):
    trajectory = trajectories[i]
    ## Plot energy error
    ax2.plot(tracing.times[2:], jnp.abs(tracing.energy[i][2:]-particles.energy)/particles.energy, label=f'Particle {i+1}')
    ## Plot velocity parallel to the magnetic field
    ax3.plot(tracing.times, trajectory[:, 3]/particles.total_speed, label=f'Particle {i+1}')
    ## Plot s-coordinate
    ax4.plot(tracing.times, trajectory[:,0], label=f'Particle {i+1}')
    # ax4.set_ylabel(r'$s=\psi/\psi_b$')
## Plot loss fractions
#ax4.plot(tracing.times, tracing.loss_fractions)
#ax4.set_ylabel('Loss Fraction');ax4.set_ylim(0, 1);ax4.set_xscale('log')
ax2.set_ylabel('Relative Energy Error')
ax2.set_xlabel('Time (s)')
ax3.set_ylim(-1, 1)
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax3.set_xlabel('Time (s)')
ax4.set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('trace_particles_nearaxis.png', dpi=300)

# # Save results in vtk format to analyze in Paraview
# vmec.surface.to_vtk('surface')
# tracing.to_vtk('trajectories')
