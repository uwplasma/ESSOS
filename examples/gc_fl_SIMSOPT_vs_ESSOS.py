import os
import time
import jax.numpy as jnp
from jax import block_until_ready
from simsopt import load
from simsopt.field import (particles_to_vtk, compute_fieldlines, plot_poincare_data)
from essos.coils import Coils_from_simsopt
from essos.dynamics import Tracing
from essos.fields import BiotSavart as BiotSavart_essos

tmax_fl = 80
nfieldlines = 3
axis_shft=0.02
R0 = jnp.linspace(1.2125346+axis_shft, 1.295-axis_shft, nfieldlines)
nfp = 2

Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)

phis_poincare = [(i/4)*(2*jnp.pi/nfp) for i in range(4)]

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
json_file = os.path.join(os.path.dirname(__file__), 'input', 'biot_savart_opt.json')
field_simsopt = load(json_file)
field_essos = BiotSavart_essos(Coils_from_simsopt(json_file, nfp))

# Trace in SIMSOPT
t1 = time.time()
fieldlines_SIMSOPT, fieldlines_SIMSOPT_phi_hits = compute_fieldlines(field_simsopt, R0, Z0, tmax=tmax_fl, tol=1e-11, phis=phis_poincare)
t2 = time.time()
avg_steps = sum([len(l) for l in fieldlines_SIMSOPT])//nfieldlines
print(f"Time for SIMSOPT fieldline tracing={t2-t1:.3f}s. Avg num steps={avg_steps}")

# Trace in ESSOS
t1 = time.time()
tracing = Tracing(field=field_essos, model='FieldLine',
                  initial_conditions=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T, maxtime=tmax_fl, timesteps=avg_steps)
fieldlines_ESSOS = tracing.trajectories
t2 = time.time()
print(f"Time for ESSOS fieldline tracing={t2-t1:.3f}s. Num steps={avg_steps}")

# tracing.poincare_plot(phis_poincare, show=False)
# plot_poincare_data(fieldlines_phi_hits, phis_poincare, os.path.join(output_dir,f'poincare_fieldline_SIMSOPT.pdf'), dpi=150)

particles_to_vtk(fieldlines_SIMSOPT, os.path.join(output_dir,f'fieldlines_SIMSOPT'))
tracing.to_vtk(os.path.join(output_dir,f'fieldlines_ESSOS'))

time_essos = jnp.linspace(0, tmax_fl, avg_steps)
def interpolate_ESSOS_to_SIMSOPT(fieldine_SIMSOPT, fieldline_ESSOS):
    time_SIMSOPT = jnp.array(fieldine_SIMSOPT)[:, 0]  # Time values from fieldlines_SIMSOPT
    # coords_SIMSOPT = jnp.array(fieldine_SIMSOPT)[:, 1:]  # Coordinates (x, y, z) from fieldlines_SIMSOPT
    coords_ESSOS = jnp.array(fieldline_ESSOS)

    interp_x = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 0])
    interp_y = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 1])
    interp_z = jnp.interp(time_SIMSOPT, time_essos, coords_ESSOS[:, 2])

    coords_ESSOS_interp = jnp.column_stack([ interp_x, interp_y, interp_z])
    
    return coords_ESSOS_interp
fieldlines_ESSOS_interp = [interpolate_ESSOS_to_SIMSOPT(fieldlines_SIMSOPT[i], fieldlines_ESSOS[i]) for i in range(nfieldlines)]
tracing.trajectories = fieldlines_ESSOS_interp
tracing.to_vtk(os.path.join(output_dir,f'fieldlines_ESSOS_interp'))

relative_error_fieldlines_SIMSOPT_vs_ESSOS = []
for i in range(nfieldlines):
    fieldline_SIMSOPT = jnp.array(fieldlines_SIMSOPT[i])[:,1:]
    fieldline_ESSOS = fieldlines_ESSOS_interp[i]
    average_relative_error = 0
    for fieldline_SIMSOPT, fieldline_ESSOS in zip(fieldline_SIMSOPT, fieldline_ESSOS):
        relative_error_x = jnp.abs(fieldline_SIMSOPT[0] - fieldline_ESSOS[0])/(jnp.abs(fieldline_SIMSOPT[0])+1e-12)
        relative_error_y = jnp.abs(fieldline_SIMSOPT[1] - fieldline_ESSOS[1])/(jnp.abs(fieldline_SIMSOPT[1])+1e-12)
        relative_error_z = jnp.abs(fieldline_SIMSOPT[2] - fieldline_ESSOS[2])/(jnp.abs(fieldline_SIMSOPT[2])+1e-12)
        average_relative_error+=(relative_error_x + relative_error_y + relative_error_z)/3
    average_relative_error/=len(fieldline_SIMSOPT)
    relative_error_fieldlines_SIMSOPT_vs_ESSOS.append(average_relative_error)
relative_error_fieldlines_SIMSOPT_vs_ESSOS = jnp.array(relative_error_fieldlines_SIMSOPT_vs_ESSOS)
print(f"Relative difference between SIMSOPT and ESSOS fieldlines={relative_error_fieldlines_SIMSOPT_vs_ESSOS}")

import matplotlib.pyplot as plt
plt.figure()

for i in range(nfieldlines):

    R_SIMSOPT   = jnp.sqrt(fieldlines_SIMSOPT[i][:,1]**2+fieldlines_SIMSOPT[i][:,2]**2)
    phi_SIMSOPT = jnp.arctan2(fieldlines_SIMSOPT[i][:,2], fieldlines_SIMSOPT[i][:,1])
    Z_SIMSOPT   = fieldlines_SIMSOPT[i][:,3]

    R_ESSOS  = jnp.sqrt(fieldlines_ESSOS_interp[i][:,0]**2+fieldlines_ESSOS_interp[i][:,1]**2)
    phi_ESSOS = jnp.arctan2(fieldlines_ESSOS_interp[i][:,1], fieldlines_ESSOS_interp[i][:,0])
    Z_ESSOS  = fieldlines_ESSOS_interp[i][:,2]

    plt.plot(R_SIMSOPT, Z_SIMSOPT, '-', linewidth=4.0, label=f'SIMSOPT {i}')
    plt.plot(R_ESSOS, Z_ESSOS, '--', linewidth=4.0, label=f'ESSOS {i}')
plt.legend()
plt.xlabel('R')
plt.ylabel('Z')
plt.title('Fieldline')
plt.savefig(os.path.join(output_dir,f'fieldlines_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

plt.figure()

for i in range(nfieldlines):
    fieldline_SIMSOPT = jnp.array(fieldlines_SIMSOPT[i])[:, 1:]
    fieldline_ESSOS = fieldlines_ESSOS_interp[i]
    relative_errors = []

    for fieldline_SIMSOPT, fieldline_ESSOS in zip(fieldline_SIMSOPT, fieldline_ESSOS):
        relative_error_x = jnp.abs(fieldline_SIMSOPT[0] - fieldline_ESSOS[0]) / (jnp.abs(fieldline_SIMSOPT[0]) + 1e-12)
        relative_error_y = jnp.abs(fieldline_SIMSOPT[1] - fieldline_ESSOS[1]) / (jnp.abs(fieldline_SIMSOPT[1]) + 1e-12)
        relative_error_z = jnp.abs(fieldline_SIMSOPT[2] - fieldline_ESSOS[2]) / (jnp.abs(fieldline_SIMSOPT[2]) + 1e-12)
        average_relative_error = (relative_error_x + relative_error_y + relative_error_z) / 3
        relative_errors.append(average_relative_error)

    plt.plot(jnp.linspace(0, tmax_fl, len(relative_errors)), relative_errors, label=f'Fieldline {i}')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Relative Error')
plt.title('Relative Error between SIMSOPT and ESSOS Fieldlines')
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'relative_error_fieldlines_SIMSOPT_vs_ESSOS.pdf'), dpi=150)
plt.close()

# error_fieldline_SIMSOPT_vs_ESSOS = jnp.array([jnp.linalg.norm(jnp.array(fieldlines_SIMSOPT[i])[:,1:] - fieldlines_ESSOS_interp[i]) for i in range(nfieldlines)])
# print(f"Max difference between SIMSOPT and ESSOS fieldlines={jnp.max(error_fieldline_SIMSOPT_vs_ESSOS)}")