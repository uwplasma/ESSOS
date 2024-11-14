import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/optimization") if not os.path.exists("images/optimization") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=26'

import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad

# Show on which platform JAX is running.
print("JAX running on", len(jax.devices()), jax.devices()[0].platform.upper())

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, optimize, loss, loss_discrete, projection2D, projection2D_top
from MagneticField import norm_B

import matplotlib.pyplot as plt
from time import time

n_curves=2
order=5

A = 1.7 # Aspect ratio
R = 6
r = R/A
r_init = r/3
n_total_optimizations = 6
n_iterations_adam   = [5, 20, 30, 10]
learning_rates_adam = [0.01, 0.005, 0.001, 0.0005]
n_iteration_least_squares = 30
ftol_least_squares = 1e-3
advance_factor_each_optimization = 1.3

model = "Guiding Center"

maxtime = 2.3e-5 # seconds
timesteps = int(maxtime*1e7)

particles = Particles(len(jax.devices()))

# dofs = jnp.reshape(jnp.array(
#     [[[5.89060390004571, -0.015150958534461115, 3.4591263438392446, 0.009699411716427936, 0.005537886410328967, 0.01955236062484789, -0.0021303528735225332, 0.0001238418385942917, 0.0011086450745253376, -0.002420205928959896, -0.0006756278019258931], [1.1793007949496566, 0.08004018778961734, 0.7029351106867904, -0.07094041261590181, 0.0020962549285922, -0.09580795350852979, 0.002692152068876774, -0.001971671169262141, 0.0015050178986789498, 0.011716003007586445, -0.0002192853422018339], [0.00925104968020525, -3.5337315253521644, -0.0006108384827500621, -0.004366106990217004, -0.005030258098696509, 0.001104679532083246, 9.417702560043752e-05, -0.000986852560245771, -0.00013184236797089718, 0.001498170896074649, -8.605289805770677e-05]], [[4.966928866203091, -0.020499263364279284, 2.9215033402400006, 0.030097757994063147, -0.001658376055921632, 0.05565323764723816, 0.002707143955919183, 0.001902227900558677, -0.0006014732428081339, -0.006951086409465891, -0.0015487952197048328], [3.3391470376526553, 0.03817293358747562, 1.9710511196219191, -0.0657504681460036, -0.008659372879836676, -0.07895161230412942, 0.002366128646258934, -0.0012328464807348071, 0.00120388490548888, 0.008955092767896359, -0.0008852913774149211], [0.02850918894375921, -3.5196447485566495, -0.003713297221886397, 0.005876325060884672, -0.01669711926998968, -0.0037687087970454496, 0.0036648895183505935, 0.00048733839516404377, 0.001556413876890159, 0.0023240807293555795, -0.0008712806280707237]]]
# ), (n_curves, 3, 2*order+1))
# curves = Curves(dofs, nfp=4, stellsym=True)
# stel = Coils(curves, jnp.array([1.0, 1.202074057360835]))

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
stel = Coils(curves, jnp.array([1.0]*n_curves))

initial_values = stel.initial_conditions(particles, R, r_init, model=model, more_trapped_particles=True)
initial_vperp = initial_values[4, :]

time0 = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps)
print("Trajectories shape:", trajectories.shape)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")

projection2D(R, r, trajectories, show=False, save_as="images/optimization/init_pol_trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="images/optimization/init_tor_trajectories.pdf")

plt.figure()
for i in range(len(trajectories)):
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
plt.title("Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
y_limit = max(jnp.abs(jnp.max(trajectories[:, :, 3])), jnp.abs(jnp.min(trajectories[:, :, 3])))
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("images/optimization/init_vpar.pdf", transparent=True)

normB = jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], stel.gamma, stel.gamma_dash, stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

y_limit = 0
plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(norm_B, 1, trajectories[i, :, :3], stel.gamma, stel.gamma_dash, stel.currents)
    normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy-1
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))
plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("images/optimization/init_energy.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Initial Stellator", save_as="images/optimization/init_stellator.pdf", show=False)
plt.close()

############################################################################################################

start = time()
dofs_with_currents = jnp.array(jnp.concatenate((jnp.ravel(stel.dofs), stel.dofs_currents[1:])))
# loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
loss_value = loss(dofs_with_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
print(f"Loss function initial value: {loss_value:.8f}, took: {time()-start:.2f} seconds")


start = time()
# grad_loss_value = grad(loss, argnums=0)(stel.dofs, stel.dofs_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
grad_loss_value = grad(loss)(dofs_with_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
print(f"Grad loss function initial value:\n{jnp.ravel(grad_loss_value)}")
print(f"Grad shape: {grad_loss_value.shape}, took: {time()-start:.2f} seconds")

start = time()
for i in range(n_total_optimizations):
    ## USING ADAM AND OPTAX
    # for n_it, lr in zip(n_iterations_adam, learning_rates_adam):
    #     print(f"Optimization {i+1} of {n_total_optimizations} with learning rate {lr}, {n_it} iterations, maxtime={maxtime}, timesteps={timesteps}")
    #     res=optimize(stel, particles, R, r, initial_values, maxtime=maxtime, timesteps=timesteps, method={"method": "OPTAX adam", "learning_rate": lr, "iterations": n_it})
    ## USING SCIPY AND LEAST SQUARES
    print(f"Optimization {i+1} of {n_total_optimizations} with {n_iteration_least_squares} iterations, ftol={ftol_least_squares}, maxtime={maxtime}, timesteps={timesteps}")
    res=optimize(stel, particles, R, r, initial_values, maxtime=maxtime, timesteps=timesteps, method={"method": "least_squares", "ftol": ftol_least_squares, "max_nfev": n_iteration_least_squares})
    
    stel.save_coils(f"Optimization_{i+1}.txt", text=f"loss={res}, maxtime={maxtime}, timesteps={timesteps}")
    maxtime *= advance_factor_each_optimization
    timesteps = int(timesteps*advance_factor_each_optimization)

print(f"Optimization took: {time()-start:.1f} seconds") 

stel.save_coils("Optimizations.txt")

trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps)

projection2D(R, r, trajectories, show=False, save_as="images/optimization/opt_pol_trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="images/optimization/opt_tor_trajectories.pdf")

plt.figure()
for i in range(len(trajectories)):
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
plt.title("Optimized Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
y_limit = max(jnp.abs(jnp.max(trajectories[:, :, 3])), jnp.abs(jnp.min(trajectories[:, :, 3])))
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("images/optimization/opt_vpar.pdf", transparent=True)

normB = jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], stel.gamma, stel.gamma_dash, stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

y_limit = 0
plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(norm_B, 1, trajectories[i, :, :3], stel.gamma, stel.gamma_dash, stel.currents)
    normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy-1
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))
plt.title("Optimized Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("images/optimization/opt_energy.pdf", transparent=True)

stel.plot(show=True, trajectories=trajectories, title="Optimized Stellator", save_as="images/optimization/opt_stellator.pdf")
