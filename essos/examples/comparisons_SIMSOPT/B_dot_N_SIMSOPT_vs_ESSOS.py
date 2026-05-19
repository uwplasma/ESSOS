"""
B·N Computation: SimSOPT vs ESSOS Timing Breakdown

This script computes the magnetic field normal component (B·n) on a toroidal surface for various grid sizes using both SimSOPT and ESSOS dipole field implementations. It measures timing for each step (reading FAMUS dipoles, filtering, reading surface, field computation, B·n calculation, and plotting) and generates a breakdown plot showing how time scales with grid size.

Dependencies:
- simsopt: For surface and field computations.
- jax: For high-performance array operations.
- matplotlib: For plotting.
- essos: Custom dipole field module.

Usage:
- Run the script to generate timings and save the plot to 'output/timing_breakdown_plot.png'.
- Adjust `grid_sizes`, `famus_file`, or `surface_file` as needed.
"""

import os
import sys
import time
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simsopt.field import Coil, BiotSavart, Current
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier
from simsopt.field.magneticfieldclasses import DipoleField as SimsoptDipoleField
import numpy as np
import gc
import jax
from jax import jit, vmap
from functools import partial
import numpy as np
from essos.util import read_famus_dipoles
from essos.fields import DipoleField


# JAX configuration
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_captured_constants_warn_bytes', -1)

# Set up module path
script_dir = os.path.dirname(__file__)
essos_path = os.path.join(script_dir, '..', '..')
sys.path.insert(0, essos_path)
from essos.custom_dipole_field import compare_dipole_fields

# Create output directory
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Create and save BiotSavart field
start = time.time()
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
biot_savart_time = time.time() - start
print(f"BiotSavart creation time: {biot_savart_time} seconds")

# Dipole field comparison
famus_file = os.path.join(script_dir, '..', 'input_files', 'zot80.focus')
surface_file = os.path.join(script_dir, '..', 'input_files', 'input.muse')
grid_sizes = [2, 4, 8, 16, 32]
timing_data = {
    'grid_sizes': grid_sizes,
    'read_famus_times': [],
    'filter_dipoles_times': [],
    'read_surface_times': [],
    'simsopt_field_times': [],
    'essos_field_times': [],
    'bn_calc_times': [],
    'plot_times': [],
    'total_times': []
}
for n in grid_sizes:
    print(f"Processing grid size {n}x{n}")
    try:
        start = time.time()
        result = compare_dipole_fields(
            surface_file, famus_file, output_dir=output_dir, plot=True, nphi=n, ntheta=n
        )
        total_time = time.time() - start
        timing_data['total_times'].append(total_time)
        print(f"Total time for {n}x{n}: {total_time} seconds")
        if len(result) == 5:
            field_essos, s_plot, gamma, unitnormal, timings = result
            timing_data['read_famus_times'].append(timings['read_famus'])
            timing_data['filter_dipoles_times'].append(timings['filter_dipoles'])
            timing_data['read_surface_times'].append(timings['read_surface'])
            timing_data['simsopt_field_times'].append(timings['simsopt_field'])
            timing_data['essos_field_times'].append(timings['essos_field'])
            timing_data['bn_calc_times'].append(timings['bn_calc'])
            timing_data['plot_times'].append(timings['plot'])
            del field_essos, s_plot, gamma, unitnormal
            gc.collect()
    except Exception as e:
        print(f"Error in compare_dipole_fields for grid size {n}x{n}: {e}")
        raise

print("Timing data:", timing_data)

# Plot timing results
plt.figure(figsize=(10, 6))
plt.plot(grid_sizes, timing_data['read_famus_times'], label='Read FAMUS', marker='o')
plt.plot(grid_sizes, timing_data['filter_dipoles_times'], label='Filter Dipoles', marker='o')
plt.plot(grid_sizes, timing_data['read_surface_times'], label='Read Surface', marker='o')
plt.plot(grid_sizes, timing_data['simsopt_field_times'], label='SimSOPT Field', marker='o')
plt.plot(grid_sizes, timing_data['essos_field_times'], label='ESSOS Field', marker='o')
plt.plot(grid_sizes, timing_data['bn_calc_times'], label='B·n Calc', marker='o')
plt.plot(grid_sizes, timing_data['plot_times'], label='Plot', marker='o')
plt.plot(grid_sizes, timing_data['total_times'], label='Total', marker='o')
plt.xlabel('Grid Size (n x n)')
plt.ylabel('Time (s)')
plt.title('Time Breakdown for B·n Computation')
plt.legend()
plt.grid(True)
plt.xticks(grid_sizes)
plt.ylim(bottom=0)
plt.savefig(os.path.join(output_dir, 'timing_breakdown_plot.png'), bbox_inches='tight')
plt.close()



def compare_dipole_fields(surface_file, famus_file,data, output_dir="output", plot=False, nphi=16, ntheta=16):
    """Compare SIMSOPT and custom dipole field calculations."""
    positions, moments, Ic, pho = read_famus_file(
    mask = (Ic == 1)
    positions = positions[mask]
    moments = moments[mask]
    pho = pho[mask]
    s_plot = SurfaceRZFourier.from_focus(surface_file, quadpoints_phi=jnp.linspace(0, 1, nphi), quadpoints_theta=jnp.linspace(0, 1, ntheta))
    gamma = s_plot.gamma().reshape((-1, 3))
    unitnormal = s_plot.unitnormal().reshape((-1, 3))
    if positions.size == 0:
        print("No dipoles found in famus_file")
        field_simsopt = None
    else:
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        if moments.ndim == 1:
            moments = moments.reshape(-1, 3)
        field_simsopt = SimsoptDipoleField(positions, moments, stellsym=True, nfp=s_plot.nfp)
    if field_simsopt is not None:
        field_simsopt.set_points(gamma)
        start = time.time()
        B_simsopt = field_simsopt.B().reshape((-1, 3))
        simsopt_time = time.time() - start
    else:
        B_simsopt = None
        simsopt_time = 0.0
    field_essos = DipoleField(positions, moments, pho, stellsym=True, nfp=s_plot.nfp)
    start = time.time()
    B_essos = field_essos.B(gamma)
    essos_time = time.time() - start
    Bnormal_simsopt = jnp.sum(B_simsopt * unitnormal, axis=1).reshape((nphi, ntheta)) if field_simsopt is not None else jnp.zeros((nphi, ntheta))
    Bnormal_essos = jnp.sum(B_essos * unitnormal, axis=1).reshape((nphi, ntheta))
    diff_Bn = Bnormal_essos - Bnormal_simsopt
    max_diff_Bn = jnp.max(jnp.abs(diff_Bn))
    mean_diff_Bn = jnp.mean(jnp.abs(diff_Bn)) 
    print(f"Max |ΔB·n|: {max_diff_Bn}")
    print(f"Mean |ΔB·n|: {mean_diff_Bn}")
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['SIMSOPT B·n', 'ESSOS B·n', 'ΔB·n']
        data_sets = [Bnormal_simsopt, Bnormal_essos, diff_Bn]
        phi_grid = jnp.linspace(0, 1, nphi)
        theta_grid = jnp.linspace(0, 1, ntheta)
        vmin = min(d.min() for d in data_sets if d.size > 0)
        vmax = max(d.max() for d in data_sets if d.size > 0)
        ims = []
        for ax, data, title in zip(axes, data_sets, titles):
            im = ax.contourf(phi_grid, theta_grid, data.T, levels=20, vmin=vmin, vmax=vmax, cmap='viridis')
            ax.set_xlabel('Phi')
            ax.set_ylabel('Theta')
            ax.set_title(title)
            ims.append(im)
            #ax.colorbar(im,orientation='horizontal', label='B·n (T)')
            fig.colorbar(im, ax=ax, orientation='horizontal',
                         fraction=0.046, pad=0.25, label='B·n (T)')


        #plt.subplots_adjust(bottom=0.35, wspace=0.3, left=0.05, right=0.95)
        #cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05]) 
        #cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal', label='B·n (T)')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'b_n_plot.png'), bbox_inches='tight', dpi=150)
        plt.close(fig) 
    return field_essos, s_plot, gamma, unitnormal, essos_time, simsopt_time