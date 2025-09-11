import os
import jax.numpy as jnp
from simsopt.geo import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import DipoleField as SimsoptDipoleField
from jax import jit, vmap
from functools import partial
import time
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from .util import read_famus_dipoles
from .fields import DipoleField
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_captured_constants_warn_bytes', -1)

def compare_dipole_fields(surface_file, famus_file, output_dir="output", plot=False, nphi=16, ntheta=16):
    """Compare SIMSOPT and custom dipole field calculations."""
    positions, moments, Ic, pho = read_famus_dipoles(famus_file)
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
        plt.subplots_adjust(bottom=0.35, wspace=0.3, left=0.05, right=0.95)
        cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05]) 
        cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal', label='B·n (T)')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'b_n_plot.png'), bbox_inches='tight', dpi=150)
        plt.close(fig) 
    return field_essos, s_plot, gamma, unitnormal, essos_time, simsopt_time