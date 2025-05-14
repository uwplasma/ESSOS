import os
from functools import partial
number_of_processors_to_use = 8 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import jit, grad, block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec
from essos.objective_functions import loss_BdotN

# Optimization parameters
max_coil_length = 40
max_coil_curvature = 0.5
order_Fourier_series_coils = 6
number_coil_points = order_Fourier_series_coils*10
maximum_function_evaluations = 300
number_coils_per_half_field_period = 4
tolerance_optimization = 1e-5
ntheta=32
nphi=32

# Initialize VMEC field
vmec = Vmec(os.path.join(os.path.dirname(__file__), '../examples/input_files',
            'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
            ntheta=ntheta, nphi=nphi, range_torus='half period')

# Initialize coils
current_on_each_coil = 1
number_of_field_periods = vmec.nfp
major_radius_coils = vmec.r_axis
minor_radius_coils = vmec.r_axis/1.5
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)

coils = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)


loss_partial = partial(loss_BdotN, dofs_curves_shape=coils.dofs_curves.shape, currents_scale=coils.currents_scale, 
                       nfp=coils.nfp, n_segments=coils.n_segments, stellsym=coils.stellsym,
                       vmec=vmec, max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature)

grad_loss_partial = jit(grad(loss_partial))

time0 = time()
loss = loss_partial(coils.x)
block_until_ready(loss)
print(f"Loss took {time()-time0:.4f} seconds. Gradient would take {(time()-time0)*(coils.x.size +1):.4f} seconds")

time0 = time()
loss_comp = loss_partial(coils.x)
block_until_ready(loss_comp)
print(f"Compiled loss took {time()-time0:.4f} seconds. Gradient would take {(time()-time0)*(coils.x.size +1):.4f} seconds")

time0 = time()
grad_loss = grad_loss_partial(coils.x)
block_until_ready(grad_loss)
print(f"Gradient took {time()-time0:.4f} seconds")

time0 = time()
grad_loss_comp = grad_loss_partial(coils.x)
block_until_ready(grad_loss_comp)
print(f"Compiled gradient took {time()-time0:.4f} seconds")

# Parameter to perturb
param = 42

# Set the possible perturbations
h_list = jnp.arange(-9, -0.9, 1/3)
h_list = 10.**h_list

# Number of orders for finite differences
fd_loss = jnp.zeros(4)

# Array to store the relative difference
fd_diff = jnp.zeros((fd_loss.size, h_list.size))

# Compute finite differences
for index, h in enumerate(h_list):
    delta = jnp.zeros(coils.x.shape)
    delta = delta.at[param].set(h)

    # 1st order finite differences
    fd_loss = fd_loss.at[0].set((loss_partial(coils.x+delta)-loss_partial(coils.x))/h)
    # 2nd order finite differences
    fd_loss = fd_loss.at[1].set((loss_partial(coils.x+delta)-loss_partial(coils.x-delta))/(2*h))
    # 4th order finite differences
    fd_loss = fd_loss.at[2].set((loss_partial(coils.x-2*delta)-8*loss_partial(coils.x-delta)+8*loss_partial(coils.x+delta)-loss_partial(coils.x+2*delta))/(12*h))
    # 6th order finite differences
    fd_loss = fd_loss.at[3].set((loss_partial(coils.x+3*delta)-9*loss_partial(coils.x+2*delta)+45*loss_partial(coils.x+delta)-45*loss_partial(coils.x-delta)+9*loss_partial(coils.x-2*delta)-loss_partial(coils.x-3*delta))/(60*h))
    
    fd_diff_h = jnp.abs((grad_loss[param]-fd_loss)/grad_loss[param])
    fd_diff = fd_diff.at[:, index].set(fd_diff_h)
    

# plot relative difference
plt.figure(figsize=(9, 6))
plt.plot(h_list, fd_diff[0], "o-", label=f'1st order', clip_on=False, linewidth=2.5)
plt.plot(h_list, fd_diff[1], "^-", label=f'2nd order', clip_on=False, linewidth=2.5)
plt.plot(h_list, fd_diff[2], "*-", label=f'4th order', clip_on=False, linewidth=2.5)
plt.plot(h_list, fd_diff[3], "s-", label=f'6th order', clip_on=False, linewidth=2.5)
plt.legend()
plt.xlabel('Finite differences stepsize h')
plt.ylabel('Relative difference')
plt.xscale('log')
plt.yscale('log')
plt.xlim(jnp.min(h_list), jnp.max(h_list))
plt.grid(which='both', axis='x')
plt.grid(which='major', axis='y')
for spine in plt.gca().spines.values():
    spine.set_zorder(0)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'gradients.pdf'))
plt.savefig(os.path.join(os.path.dirname(__file__), "../../../../UW/article/figures/" ,'gradients.pdf'))
plt.show()