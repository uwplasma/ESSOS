import os
import time
import jax.numpy as jnp
from jax import block_until_ready, random
from simsopt import load
from simsopt.field import (particles_to_vtk, trace_particles, plot_poincare_data)
from essos.coils import Coils_from_simsopt
from essos.constants import PROTON_MASS, ONE_EV
from essos.dynamics import Tracing, Particles
from essos.fields import BiotSavart as BiotSavart_essos
import matplotlib.pyplot as plt

tmax_gc = 1e-4
nparticles = 5
axis_shft=0.02
R0 = jnp.linspace(1.2125346+axis_shft, 1.295-axis_shft, nparticles)
trace_tolerance_SIMSOPT_array = [1e-5, 1e-7, 1e-9, 1e-11]
trace_tolerance_ESSOS = 1e-7
mass=PROTON_MASS
energy=5000*ONE_EV

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nfp=2
LandremanPaulQA_json_file = os.path.join(os.path.dirname(__file__), '..', 'input_files', 'SIMSOPT_biot_savart_LandremanPaulQA.json')
field_simsopt = load(LandremanPaulQA_json_file)
field_essos = BiotSavart_essos(Coils_from_simsopt(LandremanPaulQA_json_file, nfp))

Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
initial_vparallel_over_v = random.uniform(random.PRNGKey(42), (nparticles,), minval=-1, maxval=1)

phis_poincare = [(i/4)*(2*jnp.pi/nfp) for i in range(4)]

particles = Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=initial_vparallel_over_v, mass=mass, energy=energy)

# Trace in SIMSOPT
time_SIMSOPT_array = []
trajectories_SIMSOPT_array = []
avg_steps_SIMSOPT = 0
relative_energy_error_SIMSOPT_array = []
print(f'Output being saved to {output_dir}')
print(f'SIMSOPT LandremanPaulQA json file location: {LandremanPaulQA_json_file}')
for trace_tolerance_SIMSOPT in trace_tolerance_SIMSOPT_array:
    print(f'Tracing SIMSOPT guiding center with tolerance={trace_tolerance_SIMSOPT}')
    t1 = time.time()
    trajectories_SIMSOPT_this_tolerance, trajectories_SIMSOPT_phi_hits = block_until_ready(trace_particles(
                    field=field_simsopt, xyz_inits=particles.initial_xyz, mass=particles.mass,
                    parallel_speeds=particles.initial_vparallel, tmax=tmax_gc, mode='gc_vac',
                    charge=particles.charge, Ekin=particles.energy, tol=trace_tolerance_SIMSOPT))
    time_SIMSOPT_array.append(time.time()-t1)
    avg_steps_SIMSOPT += sum([len(l) for l in trajectories_SIMSOPT_this_tolerance])//nparticles
    print(f"  Time for SIMSOPT tracing={time.time()-t1:.3f}s. Avg num steps={avg_steps_SIMSOPT}")
    trajectories_SIMSOPT_array.append(trajectories_SIMSOPT_this_tolerance)
    
    relative_energy_SIMSOPT = []
    for i, trajectory in enumerate(trajectories_SIMSOPT_this_tolerance):
        xyz = jnp.asarray(trajectory[:, 1:4])
        vpar = trajectory[:, 4]
        field_simsopt.set_points(xyz)
        AbsB = field_simsopt.AbsB()[:,0]
        mu = (particles.energy - particles.mass*vpar[0]**2/2)/AbsB[0]
        relative_energy_SIMSOPT.append(jnp.abs(particles.mass*vpar**2/2+mu*AbsB-particles.energy)/particles.energy)
    relative_energy_error_SIMSOPT_array.append(relative_energy_SIMSOPT)

particles_to_vtk(trajectories_SIMSOPT_this_tolerance, os.path.join(output_dir,f'guiding_center_SIMSOPT'))

# Trace in ESSOS
num_steps_essos = int(jnp.mean(jnp.array([len(trajectories_SIMSOPT[0]) for trajectories_SIMSOPT in trajectories_SIMSOPT_array])))
time_essos = jnp.linspace(0, tmax_gc, num_steps_essos)

print(f'Tracing ESSOS guiding center with tolerance={trace_tolerance_ESSOS}')
t1 = time.time()
tracing = block_until_ready(Tracing(field=field_essos, model='GuidingCenter', particles=particles,
                                    maxtime=tmax_gc, timesteps=num_steps_essos, tol_step_size=trace_tolerance_ESSOS))
trajectories_ESSOS = tracing.trajectories
time_ESSOS = time.time()-t1
print(f"  Time for ESSOS tracing={time.time()-t1:.3f}s. Num steps={len(trajectories_ESSOS[0])}")
tracing.to_vtk(os.path.join(output_dir,f'guiding_center_ESSOS'))

relative_energy_error_ESSOS = jnp.abs(tracing.energy-particles.energy)/particles.energy

print('Plotting the results to output directory...')
plt.figure()
SIMSOPT_energy_interp_this_particle = jnp.zeros((len(trace_tolerance_SIMSOPT_array), nparticles, len(trajectories_SIMSOPT_array[-1][-1][:,0])))
for j in range(nparticles):
    for i, relative_energy_error_SIMSOPT in enumerate(relative_energy_error_SIMSOPT_array):
        SIMSOPT_energy_interp_this_particle = SIMSOPT_energy_interp_this_particle.at[i,j].set(jnp.interp(trajectories_SIMSOPT_array[-1][-1][:,0], trajectories_SIMSOPT_array[i][j][:,0], relative_energy_error_SIMSOPT[j][:]))
plt.plot(time_essos[2:], jnp.mean(relative_energy_error_ESSOS, axis=0)[2:], '-', label=f'ESSOS Tol={trace_tolerance_ESSOS}')
for i, SIMSOPT_energy_interp in enumerate(SIMSOPT_energy_interp_this_particle):
    plt.plot(trajectories_SIMSOPT_array[-1][-1][4:,0], jnp.mean(SIMSOPT_energy_interp, axis=0)[4:], '--', label=f'SIMSOPT Tol={trace_tolerance_SIMSOPT_array[i]}')
plt.legend()
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Average Relative Energy Error')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'relative_energy_error_guiding_center_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

# Plot time comparison in a bar chart
labels = [f'SIMSOPT\nTol={tol}' for tol in trace_tolerance_SIMSOPT_array] + [f'ESSOS\nTol={trace_tolerance_ESSOS}']
times = time_SIMSOPT_array + [time_ESSOS]
plt.figure()
bars = plt.bar(labels, times, color=['blue']*len(trace_tolerance_SIMSOPT_array) + ['red'], edgecolor=['black']*len(trace_tolerance_SIMSOPT_array) + ['black'], hatch=['//']*len(trace_tolerance_SIMSOPT_array) + ['|'])
plt.xlabel('Tracing Tolerance of SIMSOPT')
plt.ylabel('Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
blue_patch = plt.Line2D([0], [0], color='blue', lw=4, label='SIMSOPT', linestyle='--')
orange_patch = plt.Line2D([0], [0], color='red', lw=4, label=f'ESSOS', linestyle='-')
plt.legend(handles=[blue_patch, orange_patch])
plt.savefig(os.path.join(output_dir, 'times_guiding_center_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

def interpolate_ESSOS_to_SIMSOPT(trajectory_SIMSOPT, trajectory_ESSOS):
    time_SIMSOPT = jnp.array(trajectory_SIMSOPT)[:, 0]  # Time values from guiding center SIMSOPT
    # coords_SIMSOPT = jnp.array(trajectory_SIMSOPT)[:, 1:]  # Coordinates (x, y, z) from guiding center SIMSOPT
    coords_ESSOS = jnp.array(trajectory_ESSOS)

    interp_x = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 0])
    interp_y = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 1])
    interp_z = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 2])
    interp_v = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 3])

    coords_ESSOS_interp = jnp.column_stack([ interp_x, interp_y, interp_z, interp_v])
    
    return coords_ESSOS_interp

relative_error_array = []
for i, trajectories_SIMSOPT in enumerate(trajectories_SIMSOPT_array):
    trajectories_ESSOS_interp = [interpolate_ESSOS_to_SIMSOPT(trajectories_SIMSOPT[i], trajectories_ESSOS[i]) for i in range(nparticles)]
    tracing.trajectories = trajectories_ESSOS_interp
    if i==len(trace_tolerance_SIMSOPT_array)-1: tracing.to_vtk(os.path.join(output_dir,f'guiding_center_ESSOS_interp'))

    relative_error_trajectories_SIMSOPT_vs_ESSOS = []
    plt.figure()
    for j in range(nparticles):
        this_trajectory_SIMSOPT = jnp.array(trajectories_SIMSOPT[j])[:,1:]
        this_trajectory_ESSOS = trajectories_ESSOS_interp[j]
        average_relative_error = []
        for trajectory_SIMSOPT_t, trajectory_ESSOS_t in zip(this_trajectory_SIMSOPT, this_trajectory_ESSOS):
            relative_error_x = jnp.abs(trajectory_SIMSOPT_t[0] - trajectory_ESSOS_t[0])/(jnp.abs(trajectory_SIMSOPT_t[0])+1e-12)
            relative_error_y = jnp.abs(trajectory_SIMSOPT_t[1] - trajectory_ESSOS_t[1])/(jnp.abs(trajectory_SIMSOPT_t[1])+1e-12)
            relative_error_z = jnp.abs(trajectory_SIMSOPT_t[2] - trajectory_ESSOS_t[2])/(jnp.abs(trajectory_SIMSOPT_t[2])+1e-12)
            relative_error_v = jnp.abs(trajectory_SIMSOPT_t[3] - trajectory_ESSOS_t[3])/(jnp.abs(trajectory_SIMSOPT_t[3])+1e-12)
            average_relative_error.append((relative_error_x + relative_error_y + relative_error_z + relative_error_v)/4)
        average_relative_error = jnp.array(average_relative_error)
        relative_error_trajectories_SIMSOPT_vs_ESSOS.append(average_relative_error)
        plt.plot(jnp.linspace(0, tmax_gc, len(average_relative_error))[1:], average_relative_error[1:], label=f'Particle {1+j}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'relative_error_guiding_center_SIMSOPT_vs_ESSOS_tolerance{trace_tolerance_SIMSOPT_array[i]}.pdf'), dpi=150)
    plt.close()
    
    relative_error_array.append(relative_error_trajectories_SIMSOPT_vs_ESSOS)
    
    plt.figure()
    for j in range(nparticles):
        R_SIMSOPT   = jnp.sqrt(trajectories_SIMSOPT[j][:,1]**2+trajectories_SIMSOPT[j][:,2]**2)
        phi_SIMSOPT = jnp.arctan2(trajectories_SIMSOPT[j][:,2], trajectories_SIMSOPT[j][:,1])
        Z_SIMSOPT   = trajectories_SIMSOPT[j][:,3]

        R_ESSOS  = jnp.sqrt(trajectories_ESSOS_interp[j][:,0]**2+trajectories_ESSOS_interp[j][:,1]**2)
        phi_ESSOS = jnp.arctan2(trajectories_ESSOS_interp[j][:,1], trajectories_ESSOS_interp[j][:,0])
        Z_ESSOS  = trajectories_ESSOS_interp[j][:,2]

        plt.plot(R_SIMSOPT, Z_SIMSOPT, '-', linewidth=2.5, label=f'SIMSOPT {1+j}')
        plt.plot(R_ESSOS, Z_ESSOS, '--', linewidth=2.5, label=f'ESSOS {1+j}')
    plt.legend()
    plt.xlabel('R')
    plt.ylabel('Z')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f'guiding_center_RZ_SIMSOPT_vs_ESSOS_tolerance{trace_tolerance_SIMSOPT_array[i]}.pdf'), dpi=150)
    plt.close()
    
    plt.figure()
    for j in range(nparticles):
        time_SIMSOPT = jnp.array(trajectories_SIMSOPT[j][:,0])
        vpar_SIMSOPT = jnp.array(trajectories_SIMSOPT[j][:,4])
        vpar_ESSOS = jnp.array(trajectories_ESSOS_interp[j][:,3])
        # plt.plot(time_SIMSOPT, jnp.abs((vpar_SIMSOPT-vpar_ESSOS)/vpar_SIMSOPT), '-', linewidth=2.5, label=f'Particle {1+j}')
        plt.plot(time_SIMSOPT, vpar_SIMSOPT, '-', linewidth=2.5, label=f'SIMSOPT {1+j}')
        plt.plot(time_SIMSOPT, vpar_ESSOS, '--', linewidth=2.5, label=f'ESSOS {1+j}')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel(r'$v_{\parallel}/v$')
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f'guiding_center_vpar_SIMSOPT_vs_ESSOS_tolerance{trace_tolerance_SIMSOPT_array[i]}.pdf'), dpi=150)
    plt.close()

# Calculate RMS error for each tolerance
rms_error_array = jnp.array([[jnp.sqrt(jnp.mean(jnp.square(jnp.array(error)))) for error in relative_error] for relative_error in relative_error_array])

# Plot RMS error in a bar chart
plt.figure()
bar_width = 0.15
x = jnp.arange(len(trace_tolerance_SIMSOPT_array))
for i in range(rms_error_array.shape[1]):
    plt.bar(x + i * bar_width, rms_error_array[:, i], bar_width, label=f'Particle {1+i}')
plt.xlabel('Tracing Tolerance of SIMSOPT')
plt.ylabel('RMS Error')
plt.yscale('log')
plt.xticks(x + bar_width * (rms_error_array.shape[1] - 1) / 2, [f'Tol={tol}' for tol in trace_tolerance_SIMSOPT_array], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rms_error_guiding_center_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

# Calculate maximum error for each tolerance
max_error_array = jnp.array([[jnp.max(jnp.array(error)) for error in relative_error] for relative_error in relative_error_array])
# Plot maximum error in a bar chart
plt.figure()
bar_width = 0.15
x = jnp.arange(len(trace_tolerance_SIMSOPT_array))
for i in range(max_error_array.shape[1]):
    plt.bar(x + i * bar_width, max_error_array[:, i], bar_width, label=f'Particle {1+i}')
plt.xlabel('Tracing Tolerance of SIMSOPT')
plt.ylabel('Maximum Error')
plt.yscale('log')
plt.xticks(x + bar_width * (max_error_array.shape[1] - 1) / 2, [f'Tol={tol}' for tol in trace_tolerance_SIMSOPT_array], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'max_error_guiding_center_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

# Calculate mean error for each tolerance
mean_error_array = jnp.array([[jnp.mean(jnp.array(error)) for error in relative_error] for relative_error in relative_error_array])
# Plot mean error in a bar chart
plt.figure()
bar_width = 0.15
x = jnp.arange(len(trace_tolerance_SIMSOPT_array))
for i in range(mean_error_array.shape[1]):
    plt.bar(x + i * bar_width, mean_error_array[:, i], bar_width, label=f'Particle {1+i}')
plt.xlabel('Tracing Tolerance of SIMSOPT')
plt.ylabel('Mean Error')
plt.yscale('log')
plt.xticks(x + bar_width * (mean_error_array.shape[1] - 1) / 2, [f'Tol={tol}' for tol in trace_tolerance_SIMSOPT_array], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_error_guiding_center_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()