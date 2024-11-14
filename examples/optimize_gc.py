import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/optimization") if not os.path.exists("images/optimization") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=36'

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
n_total_optimizations = 12
n_iterations_adam   = [5, 20, 30, 10]
learning_rates_adam = [0.01, 0.005, 0.001, 0.0005]
n_iteration_least_squares = 50
ftol_least_squares = 1e-4

model = "Guiding Center"

# maxtime = 2.6e-5
# timesteps = 250

maxtime = 2.5e-5
timesteps = 250

particles = Particles(len(jax.devices()))

# dofs = jnp.reshape(jnp.array(
#     [[[5.8996555013465795, -0.2977598353229329, 3.257862151626881, -0.01657816896998508, 0.00037025842543959966, 0.2310143373702469, 0.00885456106750108, 0.005026823623697239, 0.003412834231525588], [1.1565919684150672, 1.5069351024361222, 0.6952471865609638, -0.09857388422698929, 0.04501014357091313, -1.1539969880602563, 0.024629594108078613, -0.022734008897607128, 7.641954258735538e-05], [0.07929694538720682, -3.285107121440028, -0.0023259050545656737, -0.00219408265825304, -0.043159001393953164, -0.03096884884765943, 0.0032430921123970803, -0.0031235540180576658, 0.005668618042057361]], [[4.92816751938445, -0.7606469863266081, 2.7489567702844564, 0.022675085771569146, -0.01947800025719468, 0.6574538200748608, -0.015013391053226398, 0.014944079152503295, 0.0012258586649540938], [3.2101257538027084, 1.1516976717059042, 1.8741913878823666, -0.08530499216147946, 0.007602413624491156, -0.9729574624465114, 0.006261493077716787, -0.021894939627232317, -0.006616694627150778], [0.055254281569103295, -3.3602698461799316, -0.006818303537683689, 0.011026429216855838, -0.025233932138129047, 0.02164798609485814, 0.006692643931948487, -0.001480515689466872, -0.00680694679890677]]]
# ), (n_curves, 3, 2*order+1))
# curves = Curves(dofs, nfp=4, stellsym=True)
# stel = Coils(curves, jnp.array([1.0, 1.5134984710023578]))

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
    maxtime *= 1.3
    timesteps = int(timesteps*1.3)

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
