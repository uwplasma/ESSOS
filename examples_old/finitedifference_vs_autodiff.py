import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/comparison") if not os.path.exists("images/comparison") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32'

import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad

# Show on which platform JAX is running.
print("JAX running on", len(jax.devices()), jax.devices()[0].platform.upper())

from time import time
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, loss

n_curves=2
order=3

A = 3 # Aspect ratio
R = 6
r = R/A
r_init = r/4

maxtime = 1.e-5

timesteps = 100#int(maxtime/2.0e-8)

particles = Particles(len(jax.devices()))

dofs = jnp.reshape(jnp.array(
    [[[5.964341187863394, 0.2116310603026459, 1.9702076256052303, 0.0719811983399396, -0.08561554280557947, 0.13637116070005728, 0.10941944189589492], [1.1811449317739924, 0.044807464113727616, 0.4394278611081422, -0.14570755776403574, 0.18288598358274613, 0.09601367195896943, 0.2216667807193357], [0.13480068371825368, -1.939188982511522, -0.22574365146389608, -0.10549268881871819, -0.11789881277415921, -0.14460737436080312, 0.34391858857149904]], [[4.855603072401319, -0.0069180916629228975, 1.4324319351358707, -0.07841048471606478, 0.11580711464829999, -0.07637893962918636, -0.2050763997637452], [3.508683543003509, 0.008986198913345473, 1.2712964882625748, -0.01109479040848278, -0.14145995939639752, 0.08952406052658744, 0.0197011243513614], [0.07079705102950629, -1.927105855662876, -0.04369408308600709, -0.09026801272307805, -0.0005634160450492046, -0.09573212442759443, -0.04184592788639541]]]
), (n_curves, 3, 2*order+1))

curves = Curves(dofs, nfp=4, stellsym=True)

# curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
# dofs = curves.dofs
stel = Coils(curves, jnp.array([7e6]*n_curves))

initial_values = stel.initial_conditions(particles, R, r_init, model="Guiding Center")
initial_vperp = initial_values[4, :]

time0 = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")

############################################################################################################

start = time()
loss_0 = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps)
print(f"Loss function initial value: {loss_0:.8f}, took: {time()-start:.2f} seconds")


start = time()
grad_loss_0 = grad(loss, argnums=0)(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps)
print(f"Grad value:\n{jnp.ravel(grad_loss_0)}")
print(f"Grad shape: {grad_loss_0.shape}, took: {time()-start:.2f} seconds")
print("\n" + "#"*100 + "\n")

############################################################################################################

rel_error_list_1order = []
rel_error_list_2order = []
rel_error_list_4order = []
rel_error_list_6order = []
start_h = 4
end_h = 13
for up in range(start_h, end_h):
    h = 10**-up
    start = time()
    grad_1order = []
    grad_2order = []
    grad_4order = []
    grad_6order = []
    for i in range(jnp.ravel(dofs).shape[0]):
        dofs_h = jnp.reshape(jnp.ravel(stel.dofs).at[i].set(jnp.ravel(dofs)[i] + h), (n_curves, 3, 2*order+1))
        loss_h = loss(dofs_h, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
        dofs_2h = jnp.reshape(jnp.ravel(stel.dofs).at[i].set(jnp.ravel(dofs)[i] + 2*h), (n_curves, 3, 2*order+1))
        loss_2h = loss(dofs_2h, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
        dofs_3h = jnp.reshape(jnp.ravel(stel.dofs).at[i].set(jnp.ravel(dofs)[i] + 3*h), (n_curves, 3, 2*order+1))
        loss_3h = loss(dofs_3h, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
        dofs_4h = jnp.reshape(jnp.ravel(stel.dofs).at[i].set(jnp.ravel(dofs)[i] + 4*h), (n_curves, 3, 2*order+1))
        loss_4h = loss(dofs_4h, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
        dofs_5h = jnp.reshape(jnp.ravel(stel.dofs).at[i].set(jnp.ravel(dofs)[i] + 5*h), (n_curves, 3, 2*order+1))
        loss_5h = loss(dofs_5h, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
        dofs_6h = jnp.reshape(jnp.ravel(stel.dofs).at[i].set(jnp.ravel(dofs)[i] + 6*h), (n_curves, 3, 2*order+1))
        loss_6h = loss(dofs_6h, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)

        grad_1order += [(loss_h - loss_0)/h]
        grad_2order += [(2*loss_h - 3/2*loss_0-1/2*loss_2h)/h]
        grad_4order += [(4*loss_h - 25/12*loss_0-3*loss_2h+4/3*loss_3h-1/4*loss_4h)/h]
        grad_6order += [(6*loss_h - 49/20*loss_0-15/2*loss_2h+20/3*loss_3h-15/4*loss_4h+6/5*loss_5h-1/6*loss_6h)/h]


    grad_1order = jnp.reshape(jnp.array(grad_1order), (n_curves, 3, 2*order+1))
    grad_2order = jnp.reshape(jnp.array(grad_2order), (n_curves, 3, 2*order+1))
    grad_4order = jnp.reshape(jnp.array(grad_4order), (n_curves, 3, 2*order+1))
    grad_6order = jnp.reshape(jnp.array(grad_6order), (n_curves, 3, 2*order+1))

# ############################################################################################################

    rel_error_1order = jnp.linalg.norm(grad_loss_0-grad_1order)/jnp.linalg.norm(grad_loss_0)
    rel_error_list_1order += [rel_error_1order]

    rel_error_2order = jnp.linalg.norm(grad_loss_0-grad_2order)/jnp.linalg.norm(grad_loss_0)
    rel_error_list_2order += [rel_error_2order]

    rel_error_4order = jnp.linalg.norm(grad_loss_0-grad_4order)/jnp.linalg.norm(grad_loss_0)
    rel_error_list_4order += [rel_error_4order]

    rel_error_6order = jnp.linalg.norm(grad_loss_0-grad_6order)/jnp.linalg.norm(grad_loss_0)
    rel_error_list_6order += [rel_error_6order]

############################################################################################################

plt.figure()
x_axis = [10**-index for index in range(start_h, end_h)]
print(x_axis)   
print(rel_error_list_1order)
print(rel_error_list_2order)
print(rel_error_list_4order)
print(rel_error_list_6order)
plt.plot(x_axis, rel_error_list_1order, marker = 'o', color='black', label="1st order")
plt.plot(x_axis, rel_error_list_2order, marker = 's', color='red', label="2nd order")
plt.plot(x_axis, rel_error_list_4order, marker = 'D', color='blue', label="4th order")
plt.plot(x_axis, rel_error_list_6order, marker = '^', color='green', label="6th order")

plt.xlabel(r"Stepsize $h$")
plt.ylabel("Relative error")
plt.xscale("log")
plt.yscale("log")
plt.xticks(x_axis)
plt.grid(True, which="both", ls = "-", axis="x")
plt.grid(True, which="major", ls = "-", axis="y")
plt.legend(loc='upper left')
plt.savefig("images/comparison/gradients.pdf", transparent=True)


stel.plot(trajectories=trajectories, title="Stellator", show=True)
