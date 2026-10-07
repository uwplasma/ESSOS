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

# Input parameters
light_speed=SPEED_OF_LIGHT
tmax = 1.e-4
dt=1.e-8
nparticles_per_core=100
nparticles = number_of_processors_to_use*nparticles_per_core
s=0.25
trace_tolerance = 1e-7
times_to_trace=1000
mass=PROTON_MASS
mass_a=4.*mass
mass_e=ELECTRON_MASS
vth_c_b2=1.0657889247888946e-06   #T=1000 eV 
vth_c_a2=1.e-2#4.263155699155578e-06
#vth_c_a2=4.263155699155578e-06
vth_c_e2=0.1


T_a=vth_c_a2*light_speed**2*mass_a/ONE_EV

T_e=20.e+3*ONE_EV
vth_c_e2=T_e/mass_e*2./light_speed**2

energy=3.5e+6*ONE_EV#0.5*mass_a*vth_c_a2*light_speed**2
vth_c_a2=energy*2./mass_a/light_speed**2



# Load coils and field
wout_file = os.path.join(os.path.dirname(__file__), '..', 'input_files', "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
vmec = Vmec(wout_file, ntheta=60, nphi=60, range_torus='half period', close=True)

theta = jnp.zeros(nparticles)
phi = jnp.zeros(nparticles)

initial_xyz=jnp.array([[s]*nparticles, theta, phi]).T
particles = Particles(initial_xyz=initial_xyz, mass=mass,
                      charge=ELEMENTARY_CHARGE, energy=energy, field=vmec,initial_vparallel_over_v=1.0*jnp.ones(nparticles))


#Initialize background species
#number_species=2  #(electrons,deuterium)
mass_array=jnp.array([ELECTRON_MASS/PROTON_MASS])    #mass_over_mproton
number_species=1  #(electrons,deuterium)
#mass_array=jnp.array([1.])    #mass_over_mproton
charge_array=jnp.array([-1.])    #mass_over_mproton
T_b=vth_c_e2*light_speed**2*mass_e/ONE_EV/2.  #eV
n_0=100000e+20  #m^-3
#n_array=[lambda x,y,z: n0,lambda x,y,z: n0 ]
#T_array=[lambda x,y,z: T0,lambda x,y,z: T0 ]
n_array=jnp.array([n_0])
T_array=jnp.array([T_b])
#n_array=jnp.array([n0, n0])
#T_array=jnp.array([T0, T0])


species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)
#t_s=1./gamma_ab(particles.mass, particles.charge, 0,vth_c_a2, points, species: BackgroundSpecies) 

vth_c=jnp.sqrt(T_b*ONE_EV/mass_e)/light_speed
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
tracing = Tracing(field=vmec, model='GuidingCenterCollisionsMuFixed', particles=particles,
                  maxtime=tmax, timestep=dt,times_to_trace=times_to_trace,species=species,tag_gc=0.)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
trajectories = tracing.trajectories
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

#vmec.plot(ax=ax1, show=False)
#tracing.plot(ax=ax1, show=False)

# Plot only a random subset of 10 particles in 3D
subset_size = 10
subset_indices = np.random.choice(len(trajectories), subset_size, replace=False)

for i in subset_indices:
    trajectory = trajectories[i]
    ax1.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label=f'Particle {i+1}')
    ax2.plot(tracing.times, (tracing.energy()[i]-tracing.energy()[i,0])/tracing.energy()[i,0], label=f'Particle {i+1}')     
    ax3.plot(tracing.times, 299792458*trajectory[:, 3]/jnp.sqrt(tracing.energy()[i]/mass*2.), label=f'Particle {i+1}')    
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')


ax2.set_xlabel(r'$t~[\mathrm{s}]$')
ax2.set_ylabel('Normalized energy variation')
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax3.set_xlabel(r'$t~[\mathrm{s}]$')
ax4.set_xlabel(r'$R~[\mathrm{m}]$')
ax4.set_ylabel(r'$Z~[\mathrm{m}]$')
plt.tight_layout()
plt.savefig('traj_time.pdf')


v=jnp.sqrt(tracing.energy()*2./particles.mass)
vpar=trajectories[:,:,3]*SPEED_OF_LIGHT
vpar=jnp.where(jnp.isfinite(vpar), vpar, jnp.nan)
vperp=tracing.v_perp()
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
ax13.plot(tracing.times,jnp.nanmean(v/light_speed,axis=0))
ax13.axhline(y=v_mean, color='r', linestyle='--')
ax23.plot(tracing.times,jnp.nanstd(v/light_speed,axis=0))
ax23.axhline(y=v_sigma, color='r', linestyle='--')
ax33.plot(tracing.times,jnp.nanmean(pitch,axis=0))
ax33.axhline(y=pitch_mean, color='r', linestyle='--')
ax43.plot(tracing.times,jnp.nanstd(pitch,axis=0))
ax43.axhline(y=pitch_sigma, color='r', linestyle='--')
ax53.plot(tracing.times,jnp.nanmean(vpar/light_speed,axis=0))
ax53.axhline(y=vpar_mean, color='r', linestyle='--')
ax63.plot(tracing.times,jnp.nanstd(vpar/light_speed,axis=0))
ax63.axhline(y=vpar_sigma, color='r', linestyle='--')
ax73.plot(tracing.times,jnp.nanmean(vperp/light_speed,axis=0))
ax73.axhline(y=vperp_mean, color='r', linestyle='--')
ax83.plot(tracing.times,jnp.nanstd(vperp/light_speed,axis=0))
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
ax92 = fig2.add_subplot(259)
nbins=64

v0=jnp.sqrt(tracing.energy()[:,0]*2./particles.mass)/SPEED_OF_LIGHT
vfinal=jnp.sqrt(tracing.energy()[:,-1]*2./particles.mass)/SPEED_OF_LIGHT
vperp0=tracing.v_perp()[:,0]/SPEED_OF_LIGHT
vperpfinal=tracing.v_perp()[:,-1]/SPEED_OF_LIGHT
vpar0=vpar[:,0]/SPEED_OF_LIGHT
vparfinal=vpar[:,-1]/SPEED_OF_LIGHT
pitch0=vpar0/v0
pitch_final=vparfinal/vfinal


#def find_first_less_than_numpy(arr, value):
#    indices = jnp.where(arr < value)[0]
#    if indices.size > 0:
#        result=indices[0]
#    else:
#    	result=jnp.nan
#    return result

#t_final=jax.vmap(find_first_less_than_numpy,in_axes=(0,None),out_axes=0)(v/light_speed,0.7*jnp.sqrt(vth_c_a2))


def find_first_less_than_numpy(arr, value):
    result=np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        indices = np.where(arr[i] < value)[0]
        if indices.size > 0:
            result[i]=tracing.times[indices[0]]
        else:
    	    result[i]=np.nan
    return result

t_final=find_first_less_than_numpy(np.log(v/v[0,0]),np.log(0.6))


print('Mean slowing down time is :', jnp.nanmean(t_final,axis=0))
print('Standard deviation of slowing down time is :', jnp.nanstd(t_final,axis=0))


bad_indices_v0 = jnp.isnan(v0) 
bad_indices_vfinal = jnp.isnan(vfinal) 
bad_indices_pitch0 = jnp.isnan(pitch0) 
bad_indices_pitch_final = jnp.isnan(pitch_final) 
bad_indices_vperp0 = jnp.isnan(vperp0) 
bad_indices_vperp_final = jnp.isnan(vperpfinal)
bad_indices_vpar0 = jnp.isnan(vpar0) 
bad_indices_vpar_final = jnp.isnan(vparfinal) 
bad_indices_t_final = jnp.isnan(t_final) 
good_indices_v0 = ~bad_indices_v0
good_indices_vfinal = ~bad_indices_vfinal
good_indices_pitch0 = ~bad_indices_pitch0
good_indices_pitch_final = ~bad_indices_pitch_final
good_indices_vpar0 = ~bad_indices_vpar0
good_indices_vpar_final = ~bad_indices_vpar_final
good_indices_vperp0 = ~bad_indices_vperp0
good_indices_vperp_final = ~bad_indices_vperp_final
good_indices_t_final = ~bad_indices_t_final
good_v0 = v0[good_indices_v0]
good_vfinal = vfinal[good_indices_vfinal]
good_pitch0 = pitch0[good_indices_pitch0]
good_pitch_final = pitch_final[good_indices_pitch_final]
good_t_final = t_final[good_indices_t_final]

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

tfinal_counts,tfinal_bins=jnp.histogram(good_t_final,bins=nbins)

ax12.stairs(v0_counts,v0_bins)
ax22.stairs(vfinal_counts,vfinal_bins)
#ax32.hist2d(vpar0/v0,vperp0/v0,bins=nbins)
#ax42.hist2d(vparfinal/vfinal,vperpfinal/vfinal,bins=nbins)
ax32.stairs(vpar_t0_counts,vpar_t0_bins)
ax42.stairs(vpar_tfinal_counts,vpar_tfinal_bins)
ax52.stairs(pitch_t0_counts,pitch_t0_bins)
ax62.stairs(pitch_tfinal_counts,pitch_tfinal_bins)
#ax72.hist2d(v0,pitch0,bins=nbins)
#ax82.hist2d(vfinal,pitch_final,bins=nbins)
ax72.stairs(vperp_t0_counts,vperp_t0_bins)
ax82.stairs(vperp_tfinal_counts,vperp_tfinal_bins)
ax92.stairs(tfinal_counts,tfinal_bins)

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

ax92.set_title('t=t_final')
ax92.set_ylabel('Counts')
ax92.set_xlabel(r'$t_{final}$')

plt.tight_layout()

# Improved time distribution plot
plt.figure(figsize=(7, 5))
plt.hist(good_t_final, bins=nbins, color='c', edgecolor='black', alpha=0.7)
plt.title(r'$t_{final}$ Distribution', fontweight='bold')
plt.xlabel(r'$t_{final}$', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.tight_layout()
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
plt.savefig('dist_time.pdf', dpi=300)

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')
