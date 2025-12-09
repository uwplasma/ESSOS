import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart,Vmec
from essos.surfaces import SurfaceClassifier
from essos.coils import Coils_from_json,Coils_from_simsopt
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ONE_EV,ELECTRON_MASS,PROTON_MASS,SPEED_OF_LIGHT
from essos.dynamics import Tracing, Particles
from essos.background_species import BackgroundSpecies


# Input parameters
tmax = 1e-4
timestep=1.e-8
times_to_trace=1000
nparticles_per_core=2
nparticles = number_of_processors_to_use*nparticles_per_core
R0 = 17.0
atol= 1e-5
rtol=1.e-5
energy=FUSION_ALPHA_PARTICLE_ENERGY


#Initialize background species
#number_species=2  #(electrons,deuterium)
#mass_array=jnp.array([ELECTRON_MASS/PROTON_MASS,2])    #mass_over_mproton
number_species=2  #(electrons,deuterium)
mass_array=jnp.array([ELECTRON_MASS/PROTON_MASS,2])    #mass_over_mproton
charge_array=jnp.array([-1.,1.])    #mass_over_mproton
T0=17.8e+3  #eV
n0=4.21e+20  #m^-3
#n_array=[lambda x,y,z: n0,lambda x,y,z: n0 ]
#T_array=[lambda x,y,z: T0,lambda x,y,z: T0 ]
n_array=jnp.array([n0,n0])
T_array=jnp.array([T0,T0])
#n_array=jnp.array([n0, n0])
#T_array=jnp.array([T0, T0])
species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)

# Load coils and field
#json_file = os.path.join(os.path.dirname(__name__), 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
#coils = Coils_from_json(json_file)
#field = BiotSavart(coils)

# Load coils and field
json_file = os.path.join(os.path.dirname(__name__), 'input_files', 'QH_simple_scaled.json')#'SIMSOPT_biot_savart_LandremanPaulQA.json')
coils = Coils_from_simsopt(json_file,nfp=4)
field = BiotSavart(coils)


# Load coils and field
wout_file = os.path.join(os.path.dirname(__name__), 'input_files','wout_QH_simple_scaled.nc')
vmec = Vmec(wout_file)

timeI=time()
boundary=SurfaceClassifier(vmec.surface,h=0.1)
print(f"ESSOS boundary took {time()-timeI:.2f} seconds")

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
particles = Particles(initial_xyz=initial_xyz, mass=ALPHA_PARTICLE_MASS,charge=ALPHA_PARTICLE_CHARGE, energy=energy)

print(f"Initialization performed")
# Trace in ESSOS
time0 = time()
tracing = Tracing(field=field, model='GuidingCenterCollisionsMuFixed', particles=particles,
                  maxtime=tmax, timestep=timestep,times_to_trace=times_to_trace, atol=atol,rtol=rtol,tag_gc=1.,species=species,boundary=boundary)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
print(f"Final loss fraction: {tracing.loss_fractions[-1]*100:.2f}%")
print(f"Particle 1 lost {tracing.lost_energies[1]} Joules while hitting (s,$\\theta$,$\\phi$)=({tracing.lost_positions[1,0]:.2f},{tracing.lost_positions[1,1]:.2f},{tracing.lost_positions[1,2]:.2f})")
trajectories = tracing.trajectories

# Plot trajectories, velocity parallel to the magnetic field, and energy error
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

vmec.surface.plot(ax=ax1, show=False, alpha=0.4)
coils.plot(ax=ax1, show=False)
tracing.plot(ax=ax1, show=False, n_trajectories_plot=nparticles)

for i, trajectory in enumerate(trajectories):
    #ax2.plot(tracing.times, jnp.abs(tracing.energy[i]-particles.energy)/particles.energy, label=f'Particle {i+1}')
    ax2.plot(tracing.times, (tracing.energy[i]-tracing.energy[i][0])/particles.energy, label=f'Particle {i+1}')    
    ax3.plot(tracing.times, trajectory[:, 3]*SPEED_OF_LIGHT/particles.total_speed, label=f'Particle {i+1}')
    #ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Energy variation')
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
