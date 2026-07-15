import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors
from essos.fields import BiotSavart,Vmec
from essos.constants import PROTON_MASS, ONE_EV,ELECTRON_MASS,SPEED_OF_LIGHT,ELEMENTARY_CHARGE
from essos.dynamics import Tracing, Particles
from essos.background_species import BackgroundSpecies,gamma_ab
import numpy as np
import jax 

from jax import config
# to use higher precision
config.update("jax_enable_x64", True)

# Input parameters
tmax = 1e-4
dt=1.e-14
times_to_trace=1000
nparticles_per_core=10
nparticles = number_of_processors_to_use*nparticles_per_core
s=0.25
num_steps = jnp.round(tmax/dt)
mass=PROTON_MASS
mass_e=ELECTRON_MASS
T_test=3000.
energy=T_test*ONE_EV

# # Load coils and field
# json_file = os.path.join(os.path.dirname(__file__), '../input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
# coils = Coils_from_json(json_file)
plt.rcParams.update({'font.size': 16})
# field = BiotSavart(coils)

# # Initialize particles
# Z0 = jnp.zeros(nparticles)
# phi0 = jnp.zeros(nparticles)
# initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
# particles = Particles(initial_xyz=initial_xyz,initial_vparallel_over_v=1.0*jnp.ones(nparticles), mass=mass, energy=energy)


# Load coils and field
wout_file = os.path.join(os.path.dirname(__file__), '..', 'input_files', "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
vmec = Vmec(wout_file, ntheta=60, nphi=60, range_torus='half period', close=True)

theta = jnp.zeros(nparticles)
phi = jnp.zeros(nparticles)

initial_xyz=jnp.array([[s]*nparticles, theta, phi]).T
particles = Particles(initial_xyz=initial_xyz, mass=mass,
                      charge=ELEMENTARY_CHARGE, energy=energy, field=vmec,initial_vparallel_over_v=1.0*jnp.ones(nparticles))

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


time0 = time()
tracing = Tracing(field=vmec, model='GuidingCenterCollisionsMuAdaptative', particles=particles,
                  maxtime=tmax, timestep=dt,times_to_trace=times_to_trace,species=species,tag_gc=0.)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
trajectories = tracing.trajectories


# Plot trajectories, velocity parallel to the magnetic field, and energy error
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221)#, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

#vmec.plot(ax=ax1, show=False)
#tracing.plot(ax=ax1, show=False)

# Plot only a random subset of 10 particles in 3D
subset_size = 10
import numpy as np
subset_indices = np.random.choice(len(trajectories), subset_size, replace=False)

for i in subset_indices:
    trajectory = trajectories[i]
    ax1.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label=f'Particle {i+1}')
    ax2.plot(tracing.times, (tracing.energy()[i]-tracing.energy()[i,0])/tracing.energy()[i,0], label=f'Particle {i+1}')     
    ax3.plot(tracing.times, 299792458*trajectory[:, 3]/jnp.sqrt(tracing.energy()[i]/mass*2.), label=f'Particle {i+1}')    
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')


# Set bold font for all axes and tick labels
for ax in [ax1, ax2, ax3, ax4]:
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.title.set_fontweight('bold')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')

ax2.set_xlabel(r'$t~[\mathrm{s}]$', fontweight='bold')
ax2.set_ylabel(r'$\frac{E-E_0}{E_0}$', fontweight='bold')
ax3.set_ylabel(r'$v_{\parallel}/v$', fontweight='bold')
ax3.set_xlabel(r'$t~[\mathrm{s}]$', fontweight='bold')
ax4.set_xlabel(r'$R~[\mathrm{m}]$', fontweight='bold')
ax4.set_ylabel(r'$Z~[\mathrm{m}]$', fontweight='bold')
plt.tight_layout()
plt.savefig('traj.pdf')


v=jnp.sqrt(tracing.energy()*2./particles.mass)
vpar=trajectories[:,:,3]*SPEED_OF_LIGHT
vpar=jnp.where(jnp.isfinite(vpar), vpar, jnp.nan)
vperp=tracing.v_perp()
pitch=vpar/v


# Improve font size for all plots
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})

# 1. v
fig_v = plt.figure(figsize=(7, 5))
ax_v_mean = fig_v.add_subplot(211)
ax_v_std = fig_v.add_subplot(212)
for ax in [ax_v_mean, ax_v_std]:
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
ax_v_mean.plot(tracing.times, jnp.nanmean(v/SPEED_OF_LIGHT, axis=0), linewidth=5)
ax_v_mean.axhline(y=v_mean, color='r', linestyle='--', linewidth=5)
ax_v_mean.set_title(r'$\langle v \rangle$', fontweight='bold')
ax_v_mean.set_xlabel('time', fontweight='bold')
ax_v_mean.set_ylabel(r'$v/c$', fontweight='bold')
ax_v_std.plot(tracing.times, jnp.nanstd(v/SPEED_OF_LIGHT, axis=0), linewidth=5)
ax_v_std.axhline(y=v_sigma, color='r', linestyle='--', linewidth=5)
ax_v_std.set_title(r'$\sigma(v)$', fontweight='bold')
ax_v_std.set_xlabel('time', fontweight='bold')
ax_v_std.set_ylabel(r'$v/c$', fontweight='bold')
plt.tight_layout()
fig_v.savefig('statistics_v.pdf', dpi=300)

# 2. pitch
fig_pitch = plt.figure(figsize=(7, 5))
ax_pitch_mean = fig_pitch.add_subplot(211)
ax_pitch_std = fig_pitch.add_subplot(212)
for ax in [ax_pitch_mean, ax_pitch_std]:
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
ax_pitch_mean.plot(tracing.times, jnp.nanmean(pitch, axis=0), linewidth=5)
ax_pitch_mean.axhline(y=pitch_mean, color='r', linestyle='--', linewidth=5)
ax_pitch_mean.set_title(r'$\langle \text{pitch} \rangle$', fontweight='bold')
ax_pitch_mean.set_xlabel('time', fontweight='bold')
ax_pitch_mean.set_ylabel('pitch', fontweight='bold')
ax_pitch_std.plot(tracing.times, jnp.nanstd(pitch, axis=0), linewidth=5)
ax_pitch_std.axhline(y=pitch_sigma, color='r', linestyle='--', linewidth=5)
ax_pitch_std.set_title(r'$\sigma(\text{pitch})$', fontweight='bold')
ax_pitch_std.set_xlabel('time', fontweight='bold')
ax_pitch_std.set_ylabel('pitch', fontweight='bold')
plt.tight_layout()
fig_pitch.savefig('statistics_pitch.pdf', dpi=300)

# 3. v_parallel/c
fig_vpar = plt.figure(figsize=(7, 5))
ax_vpar_mean = fig_vpar.add_subplot(211)
ax_vpar_std = fig_vpar.add_subplot(212)
for ax in [ax_vpar_mean, ax_vpar_std]:
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
ax_vpar_mean.plot(tracing.times, jnp.nanmean(vpar/SPEED_OF_LIGHT, axis=0), linewidth=5)
ax_vpar_mean.axhline(y=vpar_mean, color='r', linestyle='--', linewidth=5)
ax_vpar_mean.set_title(r'$\langle v_{\parallel}/c \rangle$', fontweight='bold')
ax_vpar_mean.set_xlabel(r'$t~[\mathrm{s}]$', fontweight='bold')
ax_vpar_mean.set_ylabel(r'$v_{\parallel}/c$', fontweight='bold')
ax_vpar_std.plot(tracing.times, jnp.nanstd(vpar/SPEED_OF_LIGHT, axis=0), linewidth=5)
ax_vpar_std.axhline(y=vpar_sigma, color='r', linestyle='--', linewidth=5)
ax_vpar_std.set_title(r'$\sigma(v_{\parallel}/c)$', fontweight='bold')
ax_vpar_std.set_xlabel(r'$t~[\mathrm{s}]$', fontweight='bold')
ax_vpar_std.set_ylabel(r'$\sigma_{v_{\parallel}/c}$', fontweight='bold')
plt.tight_layout()
fig_vpar.savefig('statistics_vpar.pdf', dpi=300)

# 4. v_perp/c
fig_vperp = plt.figure(figsize=(7, 5))
ax_vperp_mean = fig_vperp.add_subplot(211)
ax_vperp_std = fig_vperp.add_subplot(212)
for ax in [ax_vperp_mean, ax_vperp_std]:
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
ax_vperp_mean.plot(tracing.times, jnp.nanmean(vperp/SPEED_OF_LIGHT, axis=0), linewidth=5)
ax_vperp_mean.axhline(y=vperp_mean, color='r', linestyle='--', linewidth=5)
ax_vperp_mean.set_title(r'$\langle v_{\perp}/c \rangle$', fontweight='bold')
ax_vperp_mean.set_xlabel(r'$t~[\mathrm{s}]$', fontweight='bold')
ax_vperp_mean.set_ylabel(r'$v_{\perp}/c$', fontweight='bold')
ax_vperp_std.plot(tracing.times, jnp.nanstd(vperp/SPEED_OF_LIGHT, axis=0), linewidth=5)
ax_vperp_std.axhline(y=vperp_sigma, color='r', linestyle='--', linewidth=5)
ax_vperp_std.set_title(r'$\sigma(v_{\perp}/c)$', fontweight='bold')
ax_vperp_std.set_xlabel(r'$t~[\mathrm{s}]$', fontweight='bold')
ax_vperp_std.set_ylabel(r'$\sigma_{v_{\perp}/c}$', fontweight='bold')
plt.tight_layout()
fig_vperp.savefig('statistics_vperp.pdf', dpi=300)


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

v0=jnp.sqrt(tracing.energy()[:,0]*2./particles.mass)/SPEED_OF_LIGHT
vfinal=jnp.sqrt(tracing.energy()[:,-1]*2./particles.mass)/SPEED_OF_LIGHT
vperp0=tracing.v_perp()[:,0]/SPEED_OF_LIGHT
vperpfinal=tracing.v_perp()[:,-1]/SPEED_OF_LIGHT
vpar0=vpar[:,0]/SPEED_OF_LIGHT
vparfinal=vpar[:,-1]/SPEED_OF_LIGHT
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
plt.figure(figsize=(7, 5))
plt.hist(good_vfinal, bins=nbins, color='b', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(good_v0), color='r', linestyle='--', linewidth=3, label='Initial Mean')
plt.title(r'$v/c$ Distribution', fontweight='bold')
plt.xlabel(r'$v/c$', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('dist_v.pdf', dpi=300)

plt.figure(figsize=(7, 5))
plt.hist(good_pitch_final, bins=nbins, color='g', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(good_pitch0), color='r', linestyle='--', linewidth=3, label='Initial Mean')
plt.title(r'Pitch Distribution', fontweight='bold')
plt.xlabel(r'Pitch', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('dist_pitch.pdf', dpi=300)

plt.figure(figsize=(7, 5))
plt.hist(good_vpar_final, bins=nbins, color='#FA7000', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(good_vpar0), color='b', linestyle='--', linewidth=3, label='Initial Mean')
plt.title(r'$v_{\parallel}/c$ Distribution', fontweight='bold')
plt.xlabel(r'$v_{\parallel}/c$', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('dist_vpar.pdf', dpi=300)

plt.figure(figsize=(7, 5))
plt.hist(good_vperp_final, bins=nbins, color='m', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(good_vperp0), color='b', linestyle='--', linewidth=3, label='Initial Mean')
plt.title(r'$v_{\perp}/c$ Distribution', fontweight='bold')
plt.xlabel(r'$v_{\perp}/c$', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('dist_vperp.pdf', dpi=300)


plt.figure(figsize=(7, 5))
plt.hist(good_vperp_final, bins=nbins, color='#FA7000', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(good_vperp0), color='b', linestyle='--', linewidth=3, label='Initial Mean')
plt.title(r'$v_{\perp}/c$ Distribution', fontweight='bold')
plt.xlabel(r'$v_{\perp}/c$', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('dist_vperp_color.pdf', dpi=300)
