import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.constants import PROTON_MASS, ONE_EV,ELECTRON_MASS,SPEED_OF_LIGHT
from essos.dynamics import Tracing, Particles
from essos.background_species import BackgroundSpecies,gamma_ab
import numpy as np
import jax 

from jax import config
# to use higher precision
config.update("jax_enable_x64", True)

# Input parameters
tmax = 1.e-5
dt=1.e-8
times_to_trace=100
nparticles_per_core=10
nparticles = number_of_processors_to_use*nparticles_per_core
R0 = 1.25
num_steps = jnp.round(tmax/dt)
mass=PROTON_MASS
mass_e=ELECTRON_MASS
T_test=3000.
energy=T_test*ONE_EV


# Load coils and field
json_file = os.path.join(os.path.dirname(__name__), 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
particles = Particles(initial_xyz=initial_xyz,initial_vparallel_over_v=1.0*jnp.ones(nparticles), mass=mass, energy=energy)


#Initialize background species
number_species=1  #(electrons,deuterium)
mass_array=jnp.array([1.])    #mass_over_mproton
charge_array=jnp.array([1.])    #mass_over_mproton
T0=1.e+3  #eV
n0=1e+20  #m^-3
n_array=jnp.array([n0])
T_array=jnp.array([T0])
species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)
vth_c=jnp.sqrt(T0*ONE_EV/PROTON_MASS)/SPEED_OF_LIGHT
vpar_mean=0.
vpar_sigma=vth_c
v_mean=vth_c*jnp.sqrt(8./jnp.pi)
v_sigma=vth_c*jnp.sqrt((3.*jnp.pi-8.)/jnp.pi)
vperp_mean=vth_c*jnp.sqrt(jnp.pi/2.)
vperp_sigma=vth_c*jnp.sqrt(2.-jnp.pi/2.)
pitch_mean=0.
pitch_sigma=jnp.sqrt(2.**2/12)


# Trace in ESSOS
time0 = time()
tracing = Tracing(field=field, model='GuidingCenterCollisionsMuFixed', particles=particles,
                  maxtime=tmax, timestep=dt,times_to_trace=times_to_trace,species=species,tag_gc=0.)
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
    ax2.plot(tracing.times, (tracing.energy[i]-tracing.energy[i,0])/tracing.energy[i,0], label=f'Particle {i+1}')     
    ax3.plot(tracing.times, 299792458*trajectory[:, 3]/jnp.sqrt(tracing.energy[i]/mass*2.), label=f'Particle {i+1}')    
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')




ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Normalized energy variation')
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax3.set_xlabel('Time (s)')
ax4.set_xlabel('R (m)')
ax4.set_ylabel('Z (m)')
plt.tight_layout()
plt.savefig('traj.pdf')




v=jnp.sqrt(tracing.energy*2./particles.mass)
vpar=trajectories[:,:,3]
vperp=tracing.vperp_final
pitch=vpar/v

# Plot distribution in velocities initial t and final 
fig3 = plt.figure(figsize=(9, 8))
ax13 = fig3.add_subplot(241)
ax23 = fig3.add_subplot(242)
ax33 = fig3.add_subplot(243)
ax43 = fig3.add_subplot(244)
ax53 = fig3.add_subplot(245)
ax63 = fig3.add_subplot(246)
ax73 = fig3.add_subplot(247)
ax83 = fig3.add_subplot(248)
ax13.plot(tracing.times,jnp.nanmean(v/SPEED_OF_LIGHT,axis=0))
ax13.axhline(y=v_mean, color='r', linestyle='--')
ax23.plot(tracing.times,jnp.nanstd(v/SPEED_OF_LIGHT,axis=0))
ax23.axhline(y=v_sigma, color='r', linestyle='--')
ax33.plot(tracing.times,jnp.nanmean(pitch,axis=0))
ax33.axhline(y=pitch_mean, color='r', linestyle='--')
ax43.plot(tracing.times,jnp.nanstd(pitch,axis=0))
ax43.axhline(y=pitch_sigma, color='r', linestyle='--')
ax53.plot(tracing.times,jnp.nanmean(vpar/SPEED_OF_LIGHT,axis=0))
ax53.axhline(y=vpar_mean, color='r', linestyle='--')
ax63.plot(tracing.times,jnp.nanstd(vpar/SPEED_OF_LIGHT,axis=0))
ax63.axhline(y=vpar_sigma, color='r', linestyle='--')
ax73.plot(tracing.times,jnp.nanmean(vperp/SPEED_OF_LIGHT,axis=0))
ax73.axhline(y=vperp_mean, color='r', linestyle='--')
ax83.plot(tracing.times,jnp.nanstd(vperp/SPEED_OF_LIGHT,axis=0))
ax83.axhline(y=vperp_sigma, color='r', linestyle='--')
ax13.set_title('Mean energy')
ax13.set_xlabel('time')
ax13.set_ylabel('Energy')
ax23.set_title('sigma energy')
ax23.set_xlabel('time')
ax23.set_ylabel('Energy')
ax33.set_title('Mean pitch')
ax33.set_xlabel('time')
ax33.set_ylabel('pitch')
ax43.set_title('sigma pitch')
ax43.set_xlabel('time')
ax43.set_ylabel('pitch')
ax53.set_title('Mean vpar')
ax53.set_xlabel('time')
ax53.set_ylabel('vpar')
ax63.set_title('sigma vpar')
ax63.set_xlabel('time')
ax63.set_ylabel('vpar')
ax73.set_title('Mean vperp')
ax73.set_xlabel('time')
ax73.set_ylabel('vperp')
ax83.set_title('sigma vperp')
ax83.set_xlabel('time')
ax83.set_ylabel('vperp')
plt.tight_layout()
plt.savefig('statistics.pdf')



# Plot distribution in velocities initial t and final 
fig2 = plt.figure(figsize=(9, 8))
ax12 = fig2.add_subplot(251)
ax22 = fig2.add_subplot(252)
ax32 = fig2.add_subplot(253)
ax42 = fig2.add_subplot(254)
ax52 = fig2.add_subplot(255)
ax62 = fig2.add_subplot(256)
ax72 = fig2.add_subplot(257)
ax82 = fig2.add_subplot(258)
nbins=64

v0=jnp.sqrt(tracing.energy[:,0]*2./particles.mass)
vfinal=jnp.sqrt(tracing.energy[:,-1]*2./particles.mass)
vperp0=tracing.vperp_final[:,0]
vperpfinal=tracing.vperp_final[:,-1]
vpar0=trajectories[:,0,3]
vparfinal=trajectories[:,-1,3]
pitch0=vpar0/v0
pitch_final=vparfinal/vfinal





bad_indices_v0 = jnp.isnan(v0) 
bad_indices_vfinal = jnp.isnan(vfinal) 
bad_indices_pitch0 = jnp.isnan(pitch0) 
bad_indices_pitch_final = jnp.isnan(pitch_final) 
bad_indices_vperp0 = jnp.isnan(vperp0) 
bad_indices_vperp_final = jnp.isnan(vperpfinal)
bad_indices_vpar0 = jnp.isnan(vpar0) 
bad_indices_vpar_final = jnp.isnan(vparfinal) 
good_indices_v0 = ~bad_indices_v0
good_indices_vfinal = ~bad_indices_vfinal
good_indices_pitch0 = ~bad_indices_pitch0
good_indices_pitch_final = ~bad_indices_pitch_final
good_indices_vpar0 = ~bad_indices_vpar0
good_indices_vpar_final = ~bad_indices_vpar_final
good_indices_vperp0 = ~bad_indices_vperp0
good_indices_vperp_final = ~bad_indices_vperp_final
good_v0 = v0[good_indices_v0]
good_vfinal = vfinal[good_indices_vfinal]
good_pitch0 = pitch0[good_indices_pitch0]
good_pitch_final = pitch_final[good_indices_pitch_final]


good_vpar0 = vpar0[good_indices_vpar0]
good_vpar_final = vparfinal[good_indices_vpar_final]
good_vperp0 = vperp0[good_indices_vperp0]
good_vperp_final = vperpfinal[good_indices_vperp_final]


v0_counts,v0_bins=jnp.histogram(good_v0,bins=nbins)
vfinal_counts,vfinal_bins=jnp.histogram(good_vfinal,bins=nbins)

pitch_t0_counts,pitch_t0_bins=jnp.histogram(good_pitch0,bins=nbins)
pitch_tfinal_counts,pitch_tfinal_bins=jnp.histogram(good_pitch_final,bins=nbins)

vpar_t0_counts,vpar_t0_bins=jnp.histogram(good_vpar0,bins=nbins)
vpar_tfinal_counts,vpar_tfinal_bins=jnp.histogram(good_vpar_final,bins=nbins)


vperp_t0_counts,vperp_t0_bins=jnp.histogram(good_vperp0,bins=nbins)
vperp_tfinal_counts,vperp_tfinal_bins=jnp.histogram(good_vperp_final,bins=nbins)


ax12.stairs(v0_counts,v0_bins)
ax22.stairs(vfinal_counts,vfinal_bins)
ax32.stairs(vpar_t0_counts,vpar_t0_bins)
ax42.stairs(vpar_tfinal_counts,vpar_tfinal_bins)
ax52.stairs(pitch_t0_counts,pitch_t0_bins)
ax62.stairs(pitch_tfinal_counts,pitch_tfinal_bins)
ax72.stairs(vperp_t0_counts,vperp_t0_bins)
ax82.stairs(vperp_tfinal_counts,vperp_tfinal_bins)

ax12.set_title('t=0')
ax12.set_xlabel('v')
ax12.set_ylabel('Counts')
ax22.set_title('t=t_final')
ax22.set_xlabel('v')
ax22.set_ylabel('Counts')
ax32.set_title('t=0')
ax32.set_ylabel('Counts')
ax32.set_xlabel(r'$v_{\parallel}$')
ax42.set_title('t=t_final')
ax42.set_xlabel(r'$v_{parallel}$')
ax42.set_ylabel('Counts')
ax52.set_title('t=0')
ax52.set_xlabel(r'$v_{\parallel}/v$')
ax52.set_ylabel('Counts')
ax62.set_title('t=t_final')
ax62.set_xlabel(r'$v_{\parallel}/v$')
ax62.set_ylabel('Counts')
ax72.set_title('t=0')
ax72.set_ylabel('Counts')
ax72.set_xlabel(r'$v_{\perp}$')
ax82.set_title('t=t_final')
ax82.set_ylabel('Counts')
ax82.set_xlabel(r'$v_{\perp}$')


plt.tight_layout()
plt.savefig('dist.pdf')
