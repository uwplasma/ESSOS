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

plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 11, 7

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, projection2D, projection2D_top, loss
from MagneticField import B_norm, B

n_curves=2
order=3

A = 4.5 # Aspect ratio
R = 7.75
r = R/A
r_init = r/4

maxtime = 1.0e-5

timesteps = 100#int(maxtime/2.0e-8)

particles = Particles(len(jax.devices())*5)

dofs = jnp.reshape(jnp.array(
    [[[7.598977262977724, -0.06204381206588949, 1.960221489212042, 0.08087152956300107, 0.0824068756276288, -0.21685516830513563, 0.3358566890272415], [1.643112294525154, 0.0912255777501672, 0.4457002547274044, -0.23749612023036357, 0.12921698758506556, -0.16179225016869853, 0.04368567029061642], [0.017466078442610288, -1.74050056132967, 0.11873499033931252, 0.00562604695481728, 0.06847095039356701, -0.2542865018166823, -0.29519101366213796]], [[6.507785380593552, 0.041043785605086446, 1.3362598919646926, 0.05966847653297775, -0.10388002720663599, 0.16216731032336307, 0.19986238498578562], [4.353566352639947, 0.048370947976885036, 0.9705972658743076, -0.1043723970417417, 0.4309250671302714, 0.3159408392783848, 0.15054974401945123], [0.15817477627995258, -1.5667258867991969, -0.25808048023312474, -0.048571112771940525, 0.13705497165272865, 0.22698993288243674, 0.2160013309861985]]]
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
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
print(f"Loss function initial value: {loss_value:.8f}, took: {time()-start:.2f} seconds")


start = time()
grad_loss_value = grad(loss, argnums=0)(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
print(f"Grad value:\n{jnp.ravel(grad_loss_value)}")
print(f"Grad shape: {grad_loss_value.shape}, took: {time()-start:.2f} seconds")
print("\n" + "#"*100 + "\n")

############################################################################################################

rel_error_list = []
start_h = 2
end_h = 16
for up in range(start_h, end_h):
    h = 10**-up
    start = time()
    grad_discrete = []
    for i in range(jnp.ravel(dofs).shape[0]):
        shifted_dofs = jnp.reshape(jnp.ravel(stel.dofs).at[i].set(jnp.ravel(dofs)[i] + h), (n_curves, 3, 2*order+1))
        shifted_loss = loss(shifted_dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)

        grad_discrete += [(shifted_loss - loss_value)/h]


    grad_discrete = jnp.reshape(jnp.array(grad_discrete), (n_curves, 3, 2*order+1))

    print(f"Discrete grad value:\n{jnp.ravel(grad_discrete)}")
    print(f"Grad shape: {grad_discrete.shape}, took: {time()-start:.2f} seconds")

############################################################################################################

    rel_error = jnp.linalg.norm(grad_loss_value-grad_discrete)/jnp.linalg.norm(grad_loss_value)
    rel_error_list += [rel_error]
    print(f"Relative error: {rel_error:.3e}")
    print(f"Diff between gradients:\n{jnp.ravel(grad_discrete)}")
    print("\n" + "#"*100 + "\n")

############################################################################################################

plt.figure()
for index, value in enumerate(rel_error_list):
    plt.plot(10**-(index+start_h), value, 'o', color='black')

plt.title("Comparison between JAX and discrete gradients")
plt.xlabel("h")
plt.ylabel("Relative error")
plt.xscale("log")
plt.yscale("log")
plt.gca().invert_xaxis()
plt.savefig("images/comparison/gradients.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Stellator", show=True)
