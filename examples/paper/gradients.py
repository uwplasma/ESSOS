import os
import jax.numpy as jnp
from jax import block_until_ready
from time import perf_counter as timer
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec
from essos.fields import BiotSavart
from essos.surfaces import BdotN
from essos.losses import custom_loss

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize VMEC field
input_filepath = os.path.join(os.path.dirname(__file__), "..", "input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
vmec = Vmec(vmec_input, ntheta=32, nphi=32, range_torus='half period')

# Initialize coils
FOURIER_ORDER = 6
N_SEGMENTS = FOURIER_ORDER*10
N_COILS = 4
COIL_CURRENT = 1.
NFP = vmec.nfp
STELLSYM = True
LARGE_R = vmec.r_axis
SMALL_R = vmec.r_axis/1.5

curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils = Coils(curves=curves, currents=[COIL_CURRENT]*N_COILS)
field = BiotSavart(coils)

""" Creating the loss functions """
def loss(field, surface):
    return jnp.sum(jnp.abs(BdotN(surface, field)))

Loss = custom_loss(loss, "field", surface=vmec.surface)
Loss.dependencies = {"field": field}
dofs = Loss.starting_dofs

loss_value = Loss(dofs)
grad_loss = Loss.grad(dofs)
print("Loss value:", loss_value)
print("Gradient:", grad_loss)

t_start = timer()
block_until_ready(Loss(dofs))
t_end = timer()
print(f"Loss took {t_end - t_start:.4f} seconds. Gradient would take {(t_end - t_start)*(coils.x.size +1):.4f} seconds")

t_start = timer()
block_until_ready(Loss.grad(dofs))
t_end = timer()
print(f"Gradient took {t_end - t_start:.4f} seconds")

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
    delta = jnp.zeros_like(dofs)
    delta = delta.at[param].set(h)

    # 1st order finite differences
    fd_loss = fd_loss.at[0].set((Loss(dofs+delta)-Loss(dofs))/h)
    # 2nd order finite differences
    fd_loss = fd_loss.at[1].set((Loss(dofs+delta)-Loss(dofs-delta))/(2*h))
    # 4th order finite differences
    fd_loss = fd_loss.at[2].set((Loss(dofs-2*delta)-8*Loss(dofs-delta)+8*Loss(dofs+delta)-Loss(dofs+2*delta))/(12*h))
    # 6th order finite differences
    fd_loss = fd_loss.at[3].set((Loss(dofs+3*delta)-9*Loss(dofs+2*delta)+45*Loss(dofs+delta)-45*Loss(dofs-delta)+9*Loss(dofs-2*delta)-Loss(dofs-3*delta))/(60*h))
    
    fd_diff_h = jnp.abs((grad_loss[param]-fd_loss)/grad_loss[param])
    fd_diff = fd_diff.at[:, index].set(fd_diff_h)
    

# plot relative difference
plt.figure(figsize=(9, 6))
plt.plot(h_list, fd_diff[0], "o-", label=f'1st order', clip_on=False, linewidth=2.5)
plt.plot(h_list, fd_diff[1], "^-", label=f'2nd order', clip_on=False, linewidth=2.5)
plt.plot(h_list, fd_diff[2], "*-", label=f'4th order', clip_on=False, linewidth=2.5)
plt.plot(h_list, fd_diff[3], "s-", label=f'6th order', clip_on=False, linewidth=2.5)
plt.legend(fontsize=15)
plt.xlabel('Finite differences stepsize h')
plt.ylabel('Relative error')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-13, 1e-1)
plt.xlim(jnp.min(h_list), jnp.max(h_list))
plt.grid(which='both', axis='x', linestyle='--', linewidth=0.6)
plt.grid(which='major', axis='y', linestyle='--', linewidth=0.6)
for spine in plt.gca().spines.values():
    spine.set_zorder(0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gradients.pdf'))
plt.show()