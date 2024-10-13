import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/comparison") if not os.path.exists("images/comparison") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad

# Show on which platform JAX is running.
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])

from time import time
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, projection2D, projection2D_top, loss
from MagneticField import B_norm, B

n_curves=2
order=3

A = 3 # Aspect ratio
R = 6
r = R/A
r_init = r/4

maxtime = 2.e-5

timesteps = 200#int(maxtime/2.0e-8)

particles = Particles(len(jax.devices()))

dofs = jnp.reshape(jnp.array(
    [[[5.870205908602173, 0.14303169100115984, 1.8268465327091103, 0.019793549261208666, -0.05241282052884366, 0.2615207277313168, 0.347252923605286], [1.3666625234635976, 0.07497001636679158, 0.6004793826362339, -0.15752843271387096, 0.17740493391440987, 0.15485304684048565, 0.1965183985157585], [0.24349125662022789, -1.7530617461967655, -0.3547240494886801, 0.08664851921282211, -0.4262830117935991, 0.0011698516341633902, 0.5395437927813146]], [[4.933641825627229, -0.057218148747632716, 1.3599251792893692, 0.10567415750715889, 0.765339681417863, -0.055749384221982036, -0.08164910186802331], [3.6154032495722275, 0.004132238191516624, 1.1457666700981353, 0.03368168962846941, -0.19275120307619145, -0.037498437312045836, 0.25097988320588355], [0.03242416650877201, -1.745815916613291, -0.006036974699157093, -0.0005666470524547741, -0.07966014025956372, 0.39844234091259334, 0.16420062956518963]]]
), (n_curves, 3, 2*order+1))

curves = Curves(dofs, nfp=4, stellsym=True)

# curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
# dofs = curves.dofs
stel = Coils(curves, jnp.array([7e6]*n_curves))

initial_values = stel.initial_conditions(particles, R, r_init, model="Guiding Center")
initial_vperp = initial_values[4, :]

time0 = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")

############################################################################################################

start = time()
loss_0 = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
print(f"Loss function initial value: {loss_0:.8f}, took: {time()-start:.2f} seconds")


start = time()
grad_loss_0 = grad(loss, argnums=0)(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
print(f"Grad value:\n{jnp.ravel(grad_loss_0)}")
print(f"Grad shape: {grad_loss_0.shape}, took: {time()-start:.2f} seconds")
print("\n" + "#"*100 + "\n")

############################################################################################################

rel_error_list_1order = []
rel_error_list_2order = []
rel_error_list_4order = []
rel_error_list_6order = []
start_h = 3
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
#     print(f"Discrete 1st order gradient:\n{jnp.ravel(grad_1order)}")
#     print(f"Grad shape: {grad_1order.shape}, took: {time()-start:.2f} seconds\n")

    grad_2order = jnp.reshape(jnp.array(grad_2order), (n_curves, 3, 2*order+1))
#     print(f"Discrete 2nd order gradient:\n{jnp.ravel(grad_2order)}")
#     print(f"Grad shape: {grad_2order.shape}, took: {time()-start:.2f} seconds\n")

    grad_4order = jnp.reshape(jnp.array(grad_4order), (n_curves, 3, 2*order+1))
#     print(f"Discrete 4th order gradient:\n{jnp.ravel(grad_4order)}")
#     print(f"Grad shape: {grad_4order.shape}, took: {time()-start:.2f} seconds\n")

    grad_6order = jnp.reshape(jnp.array(grad_6order), (n_curves, 3, 2*order+1))
#     print(f"Discrete 6th order gradient:\n{jnp.ravel(grad_6order)}")
#     print(f"Grad shape: {grad_6order.shape}, took: {time()-start:.2f} seconds\n")

# ############################################################################################################

    rel_error_1order = jnp.linalg.norm(grad_loss_0-grad_1order)/jnp.linalg.norm(grad_loss_0)
    rel_error_list_1order += [rel_error_1order]
#     print(f"Relative error of 1st order: {rel_error_1order:.3e}\n")

    rel_error_2order = jnp.linalg.norm(grad_loss_0-grad_2order)/jnp.linalg.norm(grad_loss_0)
    rel_error_list_2order += [rel_error_2order]
#     print(f"Relative error of 2nd order: {rel_error_2order:.3e}\n")

    rel_error_4order = jnp.linalg.norm(grad_loss_0-grad_4order)/jnp.linalg.norm(grad_loss_0)
    rel_error_list_4order += [rel_error_4order]
#     print(f"Relative error of 4th order: {rel_error_4order:.3e}\n")

    rel_error_6order = jnp.linalg.norm(grad_loss_0-grad_6order)/jnp.linalg.norm(grad_loss_0)
    rel_error_list_6order += [rel_error_6order]
#     print(f"Relative error of 6th order: {rel_error_6order:.3e}\n")

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

plt.xlabel("h")
plt.ylabel("Relative error")
plt.xscale("log")
plt.yscale("log")
plt.xticks(x_axis)
plt.grid(True, which="both", ls = "-", axis="x")
plt.grid(True, which="major", ls = "-", axis="y")
plt.legend(loc='upper left')
plt.savefig("images/comparison/gradients.pdf", transparent=True)


stel.plot(trajectories=trajectories, title="Stellator", show=True)
