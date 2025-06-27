import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready, random
from simsopt import load
from simsopt.field import (particles_to_vtk, trace_particles, plot_poincare_data)
from essos.coils import Coils
from essos.constants import PROTON_MASS, ONE_EV
from essos.dynamics import Tracing, Particles
from essos.fields import BiotSavart as BiotSavart_essos
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 18})

tmax_gc = 5e-4
nparticles = 5
axis_shft=0.02
R0 = jnp.linspace(1.2125346+axis_shft, 1.295-axis_shft, nparticles)
trace_tolerance_array = [1e-5, 1e-7, 1e-9, 1e-11, 1e-13]
trace_tolerance_ESSOS = 1e-9
mass=PROTON_MASS
energy=5000*ONE_EV

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nfp=2
LandremanPaulQA_json_file = os.path.join(os.path.dirname(__file__), '../examples', 'input_files', 'SIMSOPT_biot_savart_LandremanPaulQA.json')
field_simsopt = load(LandremanPaulQA_json_file)
field_essos = BiotSavart_essos(Coils.from_simsopt(LandremanPaulQA_json_file, nfp))

Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
initial_vparallel_over_v = random.uniform(random.PRNGKey(42), (nparticles,), minval=-1, maxval=1)

phis_poincare = [(i/4)*(2*jnp.pi/nfp) for i in range(4)]

particles = Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=initial_vparallel_over_v, mass=mass, energy=energy)

# Trace in SIMSOPT
runtime_SIMSOPT_array = []
trajectories_SIMSOPT_array = []
avg_steps_SIMSOPT_array = []
relative_energy_error_SIMSOPT_array = []
print(f'Output being saved to {output_dir}')
print(f'SIMSOPT LandremanPaulQA json file location: {LandremanPaulQA_json_file}\n')
for trace_tolerance_SIMSOPT in trace_tolerance_array:
    print(f'Tracing SIMSOPT guiding center with tolerance={trace_tolerance_SIMSOPT}')
    t1 = time()
    trajectories_SIMSOPT, trajectories_SIMSOPT_phi_hits = block_until_ready(trace_particles(
                    field=field_simsopt, xyz_inits=particles.initial_xyz, mass=particles.mass,
                    parallel_speeds=particles.initial_vparallel, tmax=tmax_gc, mode='gc_vac',
                    charge=particles.charge, Ekin=particles.energy, tol=trace_tolerance_SIMSOPT))
    runtime_SIMSOPT = time() - t1
    runtime_SIMSOPT_array.append(runtime_SIMSOPT)
    avg_steps_SIMSOPT = sum([len(l) for l in trajectories_SIMSOPT]) // nparticles
    avg_steps_SIMSOPT_array.append(avg_steps_SIMSOPT)
    # print(trajectories_SIMSOPT_this_tolerance[0].shape)
    print(f"Time for SIMSOPT tracing={runtime_SIMSOPT:.3f}s. Avg num steps={avg_steps_SIMSOPT}\n")
    trajectories_SIMSOPT_array.append(trajectories_SIMSOPT)
    
    relative_energy_SIMSOPT = []
    for i, trajectory in enumerate(trajectories_SIMSOPT):
        xyz = jnp.asarray(trajectory[:, 1:4])
        vpar = trajectory[:, 4]
        field_simsopt.set_points(xyz)
        AbsB = field_simsopt.AbsB()[:,0]
        mu = (particles.energy - particles.mass*vpar[0]**2/2)/AbsB[0]
        relative_energy_SIMSOPT.append(jnp.abs(particles.mass*vpar**2/2+mu*AbsB-particles.energy)/particles.energy)
    relative_energy_error_SIMSOPT_array.append(relative_energy_SIMSOPT)

    # particles_to_vtk(trajectories_SIMSOPT_this_tolerance, os.path.join(output_dir,f'guiding_center_SIMSOPT'))

# Trace in ESSOS
runtime_ESSOS_array = []
times_essos_array = []
trajectories_ESSOS_array = []
relative_energy_error_ESSOS_array = []

# Creating a tracing object for compilation
compile_tracing = Tracing('GuidingCenter', field_essos, tmax_gc, timesteps=100, method='Dopri5',
                  stepsize='adaptive', tol_step_size=trace_tolerance_array[0], particles=particles)
block_until_ready(compile_tracing.trajectories)

for index, trace_tolerance_ESSOS in enumerate(trace_tolerance_array):
    num_steps_essos = avg_steps_SIMSOPT_array[index]
    print(f'Tracing ESSOS guiding center with tolerance={trace_tolerance_ESSOS}')
    start_time = time()
    tracing = Tracing('GuidingCenter', field_essos, tmax_gc, timesteps=num_steps_essos, method='Dopri5',
                    stepsize='adaptive', tol_step_size=trace_tolerance_ESSOS, particles=particles)
    block_until_ready(tracing.trajectories)
    runtime_ESSOS = time() - start_time
    runtime_ESSOS_array.append(runtime_ESSOS)
    times_essos_array.append(tracing.times)
    trajectories_ESSOS_array.append(tracing.trajectories)
    # print(tracing.trajectories.shape)

    trajectories_ESSOS = tracing.trajectories
    print(f"Time for ESSOS tracing={runtime_ESSOS:.3f}s. Num steps={len(trajectories_ESSOS[0])}\n")

    relative_energy_error_ESSOS = jnp.abs(tracing.energy()-particles.energy)/particles.energy
    relative_energy_error_ESSOS_array.append(relative_energy_error_ESSOS)
    # tracing.to_vtk(os.path.join(output_dir,f'guiding_center_ESSOS'))

print('Plotting the results to output directory...')
plt.figure(figsize=(9, 6))
colors = ['blue', 'orange', 'green', 'red', 'purple']

SIMSOPT_energy_interp = []

for tolerance_idx in range(len(trace_tolerance_array)):
    interpolation = jnp.stack([
        jnp.interp(times_essos_array[tolerance_idx], trajectories_SIMSOPT_array[tolerance_idx][particle_idx][:, 0], relative_energy_error_SIMSOPT_array[tolerance_idx][particle_idx])
        for particle_idx in range(nparticles)
    ]) # This will have shape (nparticles, len(times_essos_array[tolerance_idx]))
    SIMSOPT_energy_interp.append(interpolation)

    plt.plot(times_essos_array[tolerance_idx]*1000, jnp.mean(interpolation, axis=0), '--', color=colors[tolerance_idx])
    plt.plot(times_essos_array[tolerance_idx]*1000, jnp.mean(relative_energy_error_ESSOS_array[tolerance_idx], axis=0), '-', color=colors[tolerance_idx])

legend_elements = [Line2D([0], [0], color=colors[tolerance_idx], linestyle='-', label=fr"tol=$10^{{{int(jnp.log10(trace_tolerance_array[tolerance_idx])-1e-3)}}}$")
                   for tolerance_idx in range(len(trace_tolerance_array))]

plt.legend(handles=legend_elements, loc='lower right', title='ESSOS (─), SIMSOPT (--)', fontsize=14, title_fontsize=14)
plt.yscale('log')
plt.xlabel('Time (ms)')
plt.ylabel('Average Relative Energy Error')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'relative_energy_error_gc_SIMSOPT_vs_ESSOS.pdf'), dpi=150)

# Plot time comparison in a bar chart

quantities = [(fr"tol=$10^{{{int(jnp.log10(trace_tolerance_array[tolerance_idx])-1e-3)}}}$", runtime_ESSOS_array[tolerance_idx], runtime_SIMSOPT_array[tolerance_idx]) 
              for tolerance_idx in range(len(trace_tolerance_array))]

labels = [q[0] for q in quantities]
essos_vals = [q[1] for q in quantities]
simsopt_vals = [q[2] for q in quantities]

X_axis = jnp.arange(len(labels))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(X_axis - bar_width/2, essos_vals, bar_width, label="ESSOS", color="red", edgecolor="black")
ax.bar(X_axis + bar_width/2, simsopt_vals, bar_width, label="SIMSOPT", color="blue", edgecolor="black")

ax.set_xticks(X_axis)
ax.set_xticklabels(labels)
ax.set_ylabel("Computation time (s)")
ax.set_yscale('log')
ax.set_ylim(1e0, 1e2)
ax.grid(axis='y', which='both', linestyle='--', linewidth=0.6)
ax.legend(fontsize=14)
plt.savefig(os.path.join(output_dir, 'times_gc_SIMSOPT_vs_ESSOS.pdf'), dpi=150)

##################################

def interpolate_SIMSOPT_to_ESSOS(trajectory_SIMSOPT, time_ESSOS):
    time_simsopt = trajectory_SIMSOPT[:, 0]  # Time values from SIMSOPT trajectory

    interp_x = jnp.interp(time_ESSOS, time_simsopt, trajectory_SIMSOPT[:, 1])
    interp_y = jnp.interp(time_ESSOS, time_simsopt, trajectory_SIMSOPT[:, 2])
    interp_z = jnp.interp(time_ESSOS, time_simsopt, trajectory_SIMSOPT[:, 3])
    interp_v = jnp.interp(time_ESSOS, time_simsopt, trajectory_SIMSOPT[:, 4])

    coords_SIMSOPT_interp = jnp.column_stack([interp_x, interp_y, interp_z, interp_v])
    
    return coords_SIMSOPT_interp

xyz_error_fig, xyz_error_ax = plt.subplots(figsize=(9, 6))
vpar_error_fig, vpar_error_ax = plt.subplots(figsize=(9, 6))

avg_relative_xyz_error_array = []
avg_relative_v_error_array = []
for tolerance_idx in range(len(trace_tolerance_array)):
    this_trajectory_SIMSOPT = jnp.stack([interpolate_SIMSOPT_to_ESSOS(
                                            trajectories_SIMSOPT_array[tolerance_idx][particle_idx], times_essos_array[tolerance_idx]
                                        ) for particle_idx in range(nparticles)])

    this_trajectory_ESSOS = trajectories_ESSOS_array[tolerance_idx]

    relative_xyz_errors = jnp.linalg.norm(this_trajectory_ESSOS[:, :, :3] - this_trajectory_SIMSOPT[:, :, :3], axis=2) / (jnp.linalg.norm(this_trajectory_SIMSOPT[:, :, :3], axis=2) + 1e-12)
    relative_v_errors = jnp.abs(this_trajectory_SIMSOPT[:, :, 3] - this_trajectory_ESSOS[:, :, 3]) / (jnp.abs(this_trajectory_SIMSOPT[:, :, 3]) + 1e-12)
        
    avg_relative_xyz_errors = jnp.mean(relative_xyz_errors, axis=0)
    avg_relative_v_errors = jnp.mean(relative_v_errors, axis=0)
    avg_relative_xyz_error_array.append(jnp.mean(avg_relative_xyz_errors))
    avg_relative_v_error_array.append(jnp.mean(avg_relative_v_errors))

    xyz_error_ax.plot(times_essos_array[tolerance_idx]*1000, avg_relative_xyz_errors, label=rf'tol=$10^{{{int(jnp.log10(trace_tolerance_array[tolerance_idx])-1e-3)}}}$')
    vpar_error_ax.plot(times_essos_array[tolerance_idx]*1000, avg_relative_v_errors, label=rf'tol=$10^{{{int(jnp.log10(trace_tolerance_array[tolerance_idx])-1e-3)}}}$')

for ax, fig in zip([xyz_error_ax, vpar_error_ax], [xyz_error_fig, vpar_error_fig]):
    ax.legend()
    ax.set_xlabel('Time (ms)')
    ax.set_yscale('log')

xyz_error_ax.set_ylabel(r'Relative $x,y,z$ Error')
vpar_error_ax.set_ylabel(r'Relative $v_\parallel$ Error')
xyz_error_fig.savefig(os.path.join(output_dir, f'relative_xyz_error_gc_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
vpar_error_fig.savefig(os.path.join(output_dir, f'relative_vpar_error_gc_SIMSOPT_vs_ESSOS.pdf'), dpi=150)

quantities = [(fr"tol=$10^{{{int(jnp.log10(trace_tolerance_array[tolerance_idx])-1e-3)}}}$", avg_relative_xyz_error_array[tolerance_idx], avg_relative_v_error_array[tolerance_idx]) 
              for tolerance_idx in range(len(trace_tolerance_array))]

labels = [q[0] for q in quantities]
xyz_vals = [q[1] for q in quantities]
vpar_vals = [q[2] for q in quantities]

X_axis = jnp.arange(len(labels))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(X_axis - bar_width/2, xyz_vals, bar_width, label=r"x,y,z", color="red", edgecolor="black")
ax.bar(X_axis + bar_width/2, vpar_vals, bar_width, label=r"$v_\parallel$", color="blue", edgecolor="black")

ax.set_xticks(X_axis)
ax.set_xticklabels(labels)
ax.set_ylabel("Time Averaged Relative Error")
ax.set_yscale('log')
ax.set_ylim(1e-6, 1e-1)
ax.grid(axis='y', which='both', linestyle='--', linewidth=0.6)
ax.legend(fontsize=14)
plt.savefig(os.path.join(output_dir, 'relative_errors_gc_SIMSOPT_vs_ESSOS.pdf'), dpi=150)

plt.show()