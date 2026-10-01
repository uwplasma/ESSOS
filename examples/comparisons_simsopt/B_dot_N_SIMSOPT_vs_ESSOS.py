import os
import sys
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from simsopt.field import Coil, BiotSavart, Current
from simsopt.geo import CurveXYZFourier
from simsopt.field.magneticfieldclasses import DipoleField as SimsoptDipoleField
import numpy as np
import gc
import jax
from essos.util import read_famus_dipoles


jax.config.update('jax_enable_x64', True)


script_dir = os.path.dirname(__file__)
essos_path = os.path.join(script_dir, '..', '..') 
sys.path.insert(0, essos_path)
from essos.custom_dipole_field import compare_dipole_fields

output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Create and save BiotSavart field
curve = CurveXYZFourier(quadpoints=20, order=4)
n_coeffs = curve.order + 1
total_dofs = curve.full_dof_size
dofs = jnp.zeros(total_dofs)
dofs = dofs.at[0].set(1.0)
dofs = dofs.at[n_coeffs].set(1.0)
curve.set_dofs(dofs)
current = Current(1e5)
coil = Coil(curve, current)
bs = BiotSavart([coil])
temp_json = os.path.join(script_dir, '..', 'test_files', 'temp_saved_field.json')
os.makedirs(os.path.dirname(temp_json), exist_ok=True)
bs.save(temp_json)

# Dipole field comparison
famus_file = os.path.join(script_dir, '..', 'input_files', 'zot80.focus')
surface_file = os.path.join(script_dir, '..', 'input_files', 'input.muse')
grid_sizes = [2, 4, 8, 16, 32]
essos_times = []
simsopt_times = []

data = read_famus_dipoles(famus_file)
for n in grid_sizes:
 print(f"Processing grid size {n}x{n}")
 try:
     result = compare_dipole_fields(
         surface_file, famus_file,data, output_dir=output_dir, plot=(n == 16), nphi=n, ntheta=n
     )
     if len(result) == 6:
         field_essos, s_plot, gamma, unitnormal, essos_time, simsopt_time = result
         essos_times.append(essos_time)
         simsopt_times.append(simsopt_time)
         del field_essos, s_plot, gamma, unitnormal
         gc.collect()
 except Exception as e:
     print(f"Error in compare_dipole_fields for grid size {n}x{n}: {e}")
     raise


timing_data = {
 'grid_sizes': grid_sizes,
 'essos_times': essos_times,
 'simsopt_times': simsopt_times
}
print("Timing data:", timing_data)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(grid_sizes, essos_times, label='ESSOS', marker='o')
plt.plot(grid_sizes, simsopt_times, label='SimSOPT', marker='o')
plt.xlabel('Grid Size (n x n)')
plt.ylabel('Time (s)')
plt.title('Time to Compute B·n')
plt.legend()
plt.grid(True)
plt.xticks(grid_sizes)
plt.ylim(bottom=0)
plt.show()
plt.savefig(os.path.join(output_dir, 'timing_plot.png'), bbox_inches='tight')
