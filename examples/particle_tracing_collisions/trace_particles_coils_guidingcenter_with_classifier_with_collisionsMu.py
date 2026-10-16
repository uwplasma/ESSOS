import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import jax
print(jax.devices())
from time import time
from jax import block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart,SurfaceRZFourier
from essos.surfaces import SurfaceClassifier, B_on_surface
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ONE_EV,ELECTRON_MASS,PROTON_MASS,SPEED_OF_LIGHT
from essos.dynamics import Tracing, Particles
from essos.objective_functions import normB_axis
import matplotlib.pyplot as plt
from essos.background_species import BackgroundSpecies

# Input parameters
tmax = 2.e-4
timestep=1.e-8
times_to_trace=2000
nparticles_per_core=10
nparticles = number_of_processors_to_use*nparticles_per_core
atol=1.e-7
rtol=1.e-7
energy=FUSION_ALPHA_PARTICLE_ENERGY


def compute_rz_on_phi(surface, theta, phi=0.0):
    angles = jnp.outer(theta, surface.xm) - phi * surface.xn
    R = jnp.sum(surface.rc * jnp.cos(angles), axis=1)
    Z = jnp.sum(surface.zs * jnp.sin(angles), axis=1)
    return R, Z

    
# Load coils and field
json_file = os.path.join(os.path.dirname(__name__), '../input_files', 'QI_opt_coils_essos_precise.json')
coils = Coils.from_json(json_file)
field = BiotSavart(coils)

# Renormalize coils to have B_target=5.7 on axis
# Load surface first to extract magnetic axis
wout_file = os.path.join(os.path.dirname(__name__), '../input_files','wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc')
vmec = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=60, nphi=60, range_torus='full torus')

# Load the ACTUAL magnetic axis at s=0 (not from s=1 surface)
vmec_axis = SurfaceRZFourier.from_wout_file(wout_file, s=0, ntheta=60, nphi=60, range_torus='full period')

# Extract magnetic axis from s=0 surface (m=0 modes give the axis in cylindrical coordinates)
m0_mask_axis = vmec_axis.xm == 0
rc_axis = vmec_axis.rc[m0_mask_axis]
zs_axis = vmec_axis.zs[m0_mask_axis]
xn_axis = vmec_axis.xn[m0_mask_axis]

# Function to compute axis position at given phi
def compute_axis_point(phi):
    """Compute magnetic axis position at toroidal angle phi using s=0 surface"""
    angles = xn_axis * phi
    R_val = jnp.sum(rc_axis * jnp.cos(angles))
    Z = -jnp.sum(zs_axis * jnp.sin(angles))
    return R_val, Z

# Evaluate B on magnetic axis and scale
phi_array = jnp.linspace(0, 2 * jnp.pi, 200)
B_axis_old = jnp.array([field.AbsB(jnp.array([R_val * jnp.cos(phi), R_val * jnp.sin(phi), Z])) 
                         for phi in phi_array 
                         for R_val, Z in [compute_axis_point(phi)]])
B_target = 5.7
scale_factor = B_target / jnp.mean(B_axis_old)
#scale_factor = 1/(4.*jnp.pi*1.e-7) # No scaling for now, to compare with previous results
coils.dofs_currents = coils.dofs_currents * scale_factor
field_scaled = BiotSavart(coils)


#Initialize background species
number_species=2  #(electrons,deuterium)
mass_array=jnp.array([ELECTRON_MASS/PROTON_MASS,2])    #mass_over_mproton
charge_array=jnp.array([-1.,1.])    #mass_over_mproton
T0=6.6e+3  #eV
n0=4.e+20  #m^-3
n_array=jnp.array([n0,n0])
T_array=jnp.array([T0,T0])
species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)


timeI=time()
boundary=SurfaceClassifier(vmec,h=0.1)
print(f"ESSOS boundary took {time()-timeI:.2f} seconds")
# Initialize particles from surface axis
particles = Particles.InitializeParticlesAroundSurfaceAxis(
    surface=vmec_axis,                 # axis-defining surface
    boundary_surface=vmec,             # boundary/LCFS-like surface
    n_particles=nparticles,
    distance_from_axis=0.7,            # 60% of local axis->boundary distance
    distance_mode='fraction_to_boundary',
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    energy=energy,
)



time0 = time()
tracing = Tracing(field=field_scaled, model='GuidingCenterCollisionsMuFixed', particles=particles, maxtime=tmax,
                  timestep=timestep, times_to_trace=times_to_trace, species=species,tag_gc=1.)
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
#subset_indices = np.random.choice(len(trajectories), subset_size, replace=False)
subset_indices = jnp.array([4,7]) # For reproducibility, plot particles 5 and 6
for i in subset_indices:
    trajectory = trajectories[i]
    ax1.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label=f'Particle {i+1}')
    ax2.plot(tracing.times, (tracing.energy()[i]-tracing.energy()[i,0])/tracing.energy()[i,0], label=f'Particle {i+1}')     
    ax3.plot(tracing.times, SPEED_OF_LIGHT*trajectory[:, 3]/jnp.sqrt(tracing.energy()[i]/particles.mass*2.), label=f'Particle {i+1}')    
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')




ax2.set_xlabel(r'$t~[\mathrm{s}]$')
ax2.set_ylabel(r'$\frac{E-E_0}{E_0}$')
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax3.set_xlabel(r'$t~[\mathrm{s}]$')
ax4.set_xlabel(r'$R~[\mathrm{m}]$')
ax4.set_ylabel(r'$Z~[\mathrm{m}]$')
plt.tight_layout()
plt.savefig('tracing_collisions.pdf')


## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')


# Improve font and resolution settings
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titlesize': 20})

# Plot only a random subset of 2 particles in 3D
subset_indices = jnp.array([4,7]) # For reproducibility, plot particles 5 and 6

# 3D Trajectory plot
fig1 = plt.figure(figsize=(8, 7), dpi=300)
ax1 = fig1.add_subplot(111, projection='3d')
for i in subset_indices:
    trajectory = trajectories[i]
    ax1.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label=f'Particle {i+1}', linewidth=2)
ax1.set_xlabel('X [m]', fontweight='bold')
ax1.set_ylabel('Y [m]', fontweight='bold')
ax1.set_zlabel('Z [m]', fontweight='bold')
ax1.set_title('3D Trajectories', fontweight='bold')
ax1.legend(fontsize=14)
plt.tight_layout()
plt.savefig('tracing_collisions_3d.pdf', dpi=300)
plt.close(fig1)


# Energy variation plot
fig2 = plt.figure(figsize=(8, 7), dpi=300)
ax2 = fig2.add_subplot(111)
for i in subset_indices:
    energy = tracing.energy()[i]
    ax2.plot(tracing.times, (energy-energy[0])/energy[0], label=f'Particle {i+1}', linewidth=2)
ax2.set_xlabel(r'$t~[\mathrm{s}]$', fontweight='bold')
ax2.set_ylabel('Normalized energy variation', fontweight='bold')
ax2.set_title('Energy Variation', fontweight='bold')
ax2.legend(fontsize=14)
plt.tight_layout()
plt.savefig('tracing_collisions_energy.pdf', dpi=300)
plt.close(fig2)

# Parallel velocity plot
fig3 = plt.figure(figsize=(8, 7), dpi=300)
ax3 = fig3.add_subplot(111)
for i in subset_indices:
    trajectory = trajectories[i]
    energy = tracing.energy()[i]
    ax3.plot(tracing.times, SPEED_OF_LIGHT*trajectory[:, 3]/jnp.sqrt(energy/particles.mass*2.), label=f'Particle {i+1}', linewidth=2)
ax3.set_xlabel(r'$t~[\mathrm{s}]$', fontweight='bold')
ax3.set_ylabel(r'$v_{\parallel}/v$', fontweight='bold')
ax3.set_title('Parallel Velocity', fontweight='bold')
ax3.legend(fontsize=14)
plt.tight_layout()
plt.savefig('tracing_collisions_vpar.pdf', dpi=300)
plt.close(fig3)

# R-Z plot
fig4 = plt.figure(figsize=(8, 7), dpi=300)
ax4 = fig4.add_subplot(111)
for i in subset_indices:
    trajectory = trajectories[i]
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}', linewidth=2)
ax4.set_xlabel(r'$R~[\mathrm{m}]$', fontweight='bold')
ax4.set_ylabel(r'$Z~[\mathrm{m}]$', fontweight='bold')
ax4.set_title('R-Z Trajectories', fontweight='bold')
ax4.legend(fontsize=14)
plt.tight_layout()
plt.savefig('tracing_collisions_rz.pdf', dpi=300)
plt.close(fig4)



# Cross section GIF animation for particles 4 and 7 only
import imageio
import io
selected_particles = [4]
phi_movie_angles = jnp.linspace(0, 2*jnp.pi, 1000, endpoint=False)
theta = jnp.linspace(0, 2 * jnp.pi, 200)
cross_sections = [compute_rz_on_phi(vmec, theta, phi=phi_val) for phi_val in phi_movie_angles]

for i in selected_particles:
    frames = []
    trajectory = trajectories[i]
    for frame_num in range(len(trajectory)):
        x, y = trajectory[frame_num, 0], trajectory[frame_num, 1]
        phi = jnp.arctan2(y, x)
        phi = phi if phi >= 0 else phi + 2*jnp.pi
        phi_diffs = jnp.abs(phi_movie_angles - phi)
        phi_diffs_wrapped = jnp.abs(phi_movie_angles - (phi + 2*jnp.pi))
        phi_diffs = jnp.minimum(phi_diffs, phi_diffs_wrapped)
        nearest_idx = int(jnp.argmin(phi_diffs))
        R_sec, Z_sec = cross_sections[nearest_idx]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(R_sec, Z_sec, 'k-', alpha=0.7, linewidth=2, zorder=1)
        R_traj = jnp.sqrt(trajectory[:frame_num+1, 0]**2 + trajectory[:frame_num+1, 1]**2)
        Z_traj = trajectory[:frame_num+1, 2]
        ax.plot(R_traj, Z_traj, color='blue' if i == 4 else 'orange', alpha=0.8, linewidth=2.5, zorder=2)
        R = jnp.sqrt(x**2 + y**2)
        ax.scatter([R], [trajectory[frame_num, 2]], c=['blue' if i == 4 else 'orange'], s=120, marker='o', zorder=3, edgecolors='black', linewidths=1.5)
        ax.set_xlabel('R (m)', fontsize=15, fontweight='bold')
        ax.set_ylabel('Z (m)', fontsize=15, fontweight='bold')
        ax.set_title(f'Particle {i+1} Cross Section', fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        frame_data = imageio.imread(buf)
        frames.append(frame_data)
        plt.close(fig)
        buf.close()
    gif_name = f'particle_cross_section_{i+1}.gif'
    imageio.mimsave(gif_name, frames, fps=20, loop=0)
    print(f"✓ Cross section GIF saved as: {gif_name}")