import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/optimization") if not os.path.exists("images/optimization") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32'

import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad

# Show on which platform JAX is running.
print("JAX running on", len(jax.devices()), jax.devices()[0].platform.upper())

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, optimize, loss, loss_discrete, projection2D, projection2D_top
from MagneticField import B_norm

import matplotlib.pyplot as plt
from time import time




n_curves=2
order=3

A = 3 # Aspect ratio
R = 6
r = R/A
r_init = r/4

maxtime = 1.0e-5
model = "Guiding Center"

timesteps = 100#int(maxtime/2.0e-8)

particles = Particles(len(jax.devices()))

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
stel = Coils(curves, jnp.array([7e6]*n_curves))

initial_values = stel.initial_conditions(particles, R, r_init, model=model)
initial_vperp = initial_values[4, :]

time0 = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
#print("Trajectories shape:", trajectories.shape)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")

projection2D(R, r, r_init, trajectories, show=False, save_as="images/optimization/init_pol_trajectories.pdf")
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

normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], stel.gamma(), stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

y_limit = 0
plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(B_norm, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
    normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy-1
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))
plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("images/optimization/init_energy.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Initial Stellator", save_as="images/optimization/init_stellator.pdf", show=True)

############################################################################################################

start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
print(f"Loss function initial value: {loss_value:.8f}, took: {time()-start:.2f} seconds")


start = time()
grad_loss_value = grad(loss, argnums=0)(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
print(f"Grad loss function initial value:\n{jnp.ravel(grad_loss_value)}")
print(f"Grad shape: {grad_loss_value.shape}, took: {time()-start:.2f} seconds\n")

start = time()
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 250}, print_loss=False)
print(stel)
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 250}, print_loss=False)
print(stel)
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 250}, print_loss=False)
print(stel)
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 250}, print_loss=False)
print(stel)
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 250}, print_loss=False)
print(stel)
maxtime = 1.1e-5
timesteps = 110
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 250}, print_loss=False)
print(stel)
maxtime = 1.15e-5
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 250}, print_loss=False)
print(stel)
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 250}, print_loss=False)
print(stel)
maxtime = 2.e-5
timesteps = 200
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100, method={"method": "OPTAX adam", "iterations": 10}, print_loss=False)
print(stel)

print(f"Optimization took: {time()-start:.1f} seconds") 

stel.save_coils("Optimizations.txt")

curves_segments = stel.gamma()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)

projection2D(R, r, r_init, trajectories, show=False, save_as="images/optimization/opt_pol_trajectories.pdf")
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

normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], curves_segments, stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

y_limit = 0
plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(B_norm, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
    normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy-1
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))
plt.title("Optimized Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("images/optimization/opt_energy.pdf", transparent=True)

stel.plot(show=True, trajectories=trajectories, title="Optimized Stellator", save_as="images/optimization/opt_stellator.pdf")
