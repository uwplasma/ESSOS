import os
number_of_processors_to_use = 5 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.constants import PROTON_MASS, ONE_EV,ELECTRON_MASS
from essos.dynamics import Tracing, Particles
from essos.background_species import BackgroundSpecies

# Input parameters
tmax =1e-5
dt=1.e-9
nparticles = number_of_processors_to_use
R0 = jnp.linspace(1.23, 1.27, nparticles)
trace_tolerance = 1e-7
num_steps = int(tmax/dt)
mass=PROTON_MASS
mass_e=ELECTRON_MASS
energy=4000*ONE_EV

# Load coils and field
json_file = os.path.join(os.path.dirname(__name__), 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy)


#Initialize background species
number_species=2  #(electrons,deuterium)
mass_array=jnp.array([ELECTRON_MASS/PROTON_MASS,2])    #mass_over_mproton
charge_array=jnp.array([-1.,1.])    #mass_over_mproton
T0=1.e+3  #eV
n0=1.e+20  #m^-3
#n_array=[lambda x,y,z: n0,lambda x,y,z: n0 ]
#T_array=[lambda x,y,z: T0,lambda x,y,z: T0 ]
n_array=jnp.array([n0, n0])
T_array=jnp.array([T0, T0])
species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)

import jax
import jax.numpy as jnp
from essos.dynamics import GuidingCenterCollisionsDrift as GCCD
from essos.dynamics import GuidingCenterCollisionsDiffusion as GCCDiff
from essos.background_species import nu_s_ab,nu_D_ab,nu_par_ab, d_nu_par_ab
total_speed_temp=particles.total_speed*jnp.ones(particles.nparticles)
initial_conditions = jnp.concatenate([particles.initial_xyz,total_speed_temp[:, None], particles.initial_vparallel_over_v[:, None]], axis=1)
args = (field, particles,species)
GCCD(0,initial_conditions[0],args)
GCCDiff(0,initial_conditions[0],args)
#initial_condition = jnp.concatenate([particles.initial_xyz,total_speed_temp[:, None], particles.initial_vparallel_over_v[:, None]], axis=1)[0]

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=field, model='GuidingCenterCollisions', particles=particles,
                  maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance,species=species)
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
    ax2.plot(tracing.times, (tracing.energy[i]-tracing.energy[i][0])/tracing.energy[i][0], label=f'Particle {i+1}')
    ax3.plot(tracing.times, trajectory[:, 4], label=f'Particle {i+1}')
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Normalized energy variation')
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