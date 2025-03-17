import os
import time
import jax.numpy as jnp
from jax import block_until_ready
from simsopt import load
from simsopt.field import (particles_to_vtk, compute_fieldlines, plot_poincare_data)
from essos.coils import Coils_from_simsopt
from essos.dynamics import Tracing
from essos.fields import BiotSavart as BiotSavart_essos
import matplotlib.pyplot as plt

tmax_fl = 150
nfieldlines = 3
axis_shft=0.02
R0 = jnp.linspace(1.2125346+axis_shft, 1.295-axis_shft, nfieldlines)
nfp = 2
trace_tolerance_SIMSOPT_array = [1e-5, 1e-7, 1e-9, 1e-11, 1e-13]
trace_tolerance_ESSOS = 1e-7

Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)

phis_poincare = [(i/4)*(2*jnp.pi/nfp) for i in range(4)]

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
LandremanPaulQA_json_file = os.path.join(os.path.dirname(__file__), '..', 'input_files', 'SIMSOPT_biot_savart_LandremanPaulQA.json')
field_simsopt = load(LandremanPaulQA_json_file)
field_essos = BiotSavart_essos(Coils_from_simsopt(LandremanPaulQA_json_file, nfp))

fieldlines_SIMSOPT_array = []
time_SIMSOPT_array = []
avg_steps_SIMSOPT = 0

print(f'Output being saved to {output_dir}')
print(f'SIMSOPT LandremanPaulQA json file location: {LandremanPaulQA_json_file}')
for trace_tolerance_SIMSOPT in trace_tolerance_SIMSOPT_array:
    print(f' Tracing SIMSOPT fieldlines with tolerance={trace_tolerance_SIMSOPT}')
    t1 = time.time()
    fieldlines_SIMSOPT_this_tolerance, fieldlines_SIMSOPT_phi_hits = block_until_ready(compute_fieldlines(field_simsopt, R0, Z0, tmax=tmax_fl, tol=trace_tolerance_SIMSOPT, phis=phis_poincare))
    time_SIMSOPT_array.append(time.time()-t1)
    avg_steps_SIMSOPT += sum([len(l) for l in fieldlines_SIMSOPT_this_tolerance])//nfieldlines
    print(f"  Time for SIMSOPT tracing={time.time()-t1:.3f}s. Avg num steps={avg_steps_SIMSOPT}")
    fieldlines_SIMSOPT_array.append(fieldlines_SIMSOPT_this_tolerance)

particles_to_vtk(fieldlines_SIMSOPT_this_tolerance, os.path.join(output_dir,f'fieldlines_SIMSOPT'))
# plot_poincare_data(fieldlines_phi_hits, phis_poincare, os.path.join(output_dir,f'poincare_fieldline_SIMSOPT.pdf'), dpi=150)

# Trace in ESSOS
num_steps_essos = int(jnp.mean(jnp.array([len(fieldlines_SIMSOPT[0]) for fieldlines_SIMSOPT in fieldlines_SIMSOPT_array])))
time_essos = jnp.linspace(0, tmax_fl, num_steps_essos)

print(f'Tracing ESSOS fieldlines with tolerance={trace_tolerance_ESSOS}')
t1 = time.time()
tracing = block_until_ready(Tracing(field=field_essos, model='FieldLine', initial_conditions=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T,
                                    maxtime=tmax_fl, timesteps=num_steps_essos, tol_step_size=trace_tolerance_ESSOS))
fieldlines_ESSOS = tracing.trajectories
time_ESSOS = time.time()-t1
print(f"  Time for ESSOS tracing={time.time()-t1:.3f}s. Num steps={len(fieldlines_ESSOS[0])}")

tracing.to_vtk(os.path.join(output_dir,f'fieldlines_ESSOS'))
# tracing.poincare_plot(phis_poincare, show=False)

print('Plotting the results to output directory...')
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
plt.savefig(os.path.join(output_dir, 'times_fieldlines_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

def interpolate_ESSOS_to_SIMSOPT(fieldine_SIMSOPT, fieldline_ESSOS):
    time_SIMSOPT = jnp.array(fieldine_SIMSOPT)[:, 0]  # Time values from fieldlines_SIMSOPT
    # coords_SIMSOPT = jnp.array(fieldine_SIMSOPT)[:, 1:]  # Coordinates (x, y, z) from fieldlines_SIMSOPT
    coords_ESSOS = jnp.array(fieldline_ESSOS)

    interp_x = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 0])
    interp_y = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 1])
    interp_z = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 2])

    coords_ESSOS_interp = jnp.column_stack([ interp_x, interp_y, interp_z])
    
    return coords_ESSOS_interp

relative_error_array = []
for i, fieldlines_SIMSOPT in enumerate(fieldlines_SIMSOPT_array):
    fieldlines_ESSOS_interp = [interpolate_ESSOS_to_SIMSOPT(fieldlines_SIMSOPT[i], fieldlines_ESSOS[i]) for i in range(nfieldlines)]
    tracing.trajectories = fieldlines_ESSOS_interp
    if i==len(trace_tolerance_SIMSOPT_array)-1: tracing.to_vtk(os.path.join(output_dir,f'fieldlines_ESSOS_interp'))

    relative_error_fieldlines_SIMSOPT_vs_ESSOS = []
    plt.figure()
    for j in range(nfieldlines):
        this_fieldline_SIMSOPT = jnp.array(fieldlines_SIMSOPT[j])[:,1:]
        this_fieldlines_ESSOS = fieldlines_ESSOS_interp[j]
        average_relative_error = []
        for fieldline_SIMSOPT_t, fieldline_ESSOS_t in zip(this_fieldline_SIMSOPT, this_fieldlines_ESSOS):
            relative_error_x = jnp.abs(fieldline_SIMSOPT_t[0] - fieldline_ESSOS_t[0])/(jnp.abs(fieldline_SIMSOPT_t[0])+1e-12)
            relative_error_y = jnp.abs(fieldline_SIMSOPT_t[1] - fieldline_ESSOS_t[1])/(jnp.abs(fieldline_SIMSOPT_t[1])+1e-12)
            relative_error_z = jnp.abs(fieldline_SIMSOPT_t[2] - fieldline_ESSOS_t[2])/(jnp.abs(fieldline_SIMSOPT_t[2])+1e-12)
            average_relative_error.append((relative_error_x + relative_error_y + relative_error_z)/3)
        average_relative_error = jnp.array(average_relative_error)
        relative_error_fieldlines_SIMSOPT_vs_ESSOS.append(average_relative_error)
        plt.plot(jnp.linspace(0, tmax_fl, len(average_relative_error))[1:], average_relative_error[1:], label=f'Fieldline {j}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'relative_error_fieldlines_SIMSOPT_vs_ESSOS_tolerance{trace_tolerance_SIMSOPT_array[i]}.pdf'), dpi=150)
    plt.close()
    
    # relative_error_fieldlines_SIMSOPT_vs_ESSOS = jnp.array(relative_error_fieldlines_SIMSOPT_vs_ESSOS)
    # print(f"Relative difference between SIMSOPT and ESSOS fieldlines={relative_error_fieldlines_SIMSOPT_vs_ESSOS}")
    relative_error_array.append(relative_error_fieldlines_SIMSOPT_vs_ESSOS)
    
    plt.figure()
    for j in range(nfieldlines):
        R_SIMSOPT   = jnp.sqrt(fieldlines_SIMSOPT[j][:,1]**2+fieldlines_SIMSOPT[j][:,2]**2)
        phi_SIMSOPT = jnp.arctan2(fieldlines_SIMSOPT[j][:,2], fieldlines_SIMSOPT[j][:,1])
        Z_SIMSOPT   = fieldlines_SIMSOPT[j][:,3]

        R_ESSOS  = jnp.sqrt(fieldlines_ESSOS_interp[j][:,0]**2+fieldlines_ESSOS_interp[j][:,1]**2)
        phi_ESSOS = jnp.arctan2(fieldlines_ESSOS_interp[j][:,1], fieldlines_ESSOS_interp[j][:,0])
        Z_ESSOS  = fieldlines_ESSOS_interp[j][:,2]

        plt.plot(R_SIMSOPT, Z_SIMSOPT, '-', linewidth=2.5, label=f'SIMSOPT {j}')
        plt.plot(R_ESSOS, Z_ESSOS, '--', linewidth=2.5, label=f'ESSOS {j}')
    plt.legend()
    plt.xlabel('R')
    plt.ylabel('Z')
    plt.savefig(os.path.join(output_dir,f'fieldlines_SIMSOPT_vs_ESSOS_tolerance{trace_tolerance_SIMSOPT_array[i]}.pdf'), dpi=150)
    plt.close()

# Calculate RMS error for each tolerance
rms_error_array = jnp.array([[jnp.sqrt(jnp.mean(jnp.square(jnp.array(error)))) for error in relative_error] for relative_error in relative_error_array])

# Plot RMS error in a bar chart
plt.figure()
bar_width = 0.15
x = jnp.arange(len(trace_tolerance_SIMSOPT_array))
for i in range(rms_error_array.shape[1]):
    plt.bar(x + i * bar_width, rms_error_array[:, i], bar_width, label=f'Fieldline {i}')
plt.xlabel('Tracing Tolerance of SIMSOPT')
plt.ylabel('RMS Error')
plt.yscale('log')
plt.xticks(x + bar_width * (rms_error_array.shape[1] - 1) / 2, [f'Tol={tol}' for tol in trace_tolerance_SIMSOPT_array], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rms_error_fieldlines_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

# Calculate maximum error for each tolerance
max_error_array = jnp.array([[jnp.max(jnp.array(error)) for error in relative_error] for relative_error in relative_error_array])
# Plot maximum error in a bar chart
plt.figure()
bar_width = 0.15
x = jnp.arange(len(trace_tolerance_SIMSOPT_array))
for i in range(max_error_array.shape[1]):
    plt.bar(x + i * bar_width, max_error_array[:, i], bar_width, label=f'Fieldline {i}')
plt.xlabel('Tracing Tolerance of SIMSOPT')
plt.ylabel('Maximum Error')
plt.yscale('log')
plt.xticks(x + bar_width * (max_error_array.shape[1] - 1) / 2, [f'Tol={tol}' for tol in trace_tolerance_SIMSOPT_array], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'max_error_fieldlines_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

# Calculate mean error for each tolerance
mean_error_array = jnp.array([[jnp.mean(jnp.array(error)) for error in relative_error] for relative_error in relative_error_array])
# Plot mean error in a bar chart
plt.figure()
bar_width = 0.15
x = jnp.arange(len(trace_tolerance_SIMSOPT_array))
for i in range(mean_error_array.shape[1]):
    plt.bar(x + i * bar_width, mean_error_array[:, i], bar_width, label=f'Fieldline {i}')
plt.xlabel('Tracing Tolerance of SIMSOPT')
plt.ylabel('Mean Error')
plt.yscale('log')
plt.xticks(x + bar_width * (mean_error_array.shape[1] - 1) / 2, [f'Tol={tol}' for tol in trace_tolerance_SIMSOPT_array], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_error_fieldlines_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()
