import jax
import jax.numpy as jnp
from jax import grad

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

# Show on which platform JAX is running.
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])

from ESSOS import CreateEquallySpacedCurves, Coils, Particles, optimize, loss, optimize_adam, projection2D, projection2D_top
from MagneticField import B_norm

import matplotlib.pyplot as plt
from time import time

n_curves=3
order=4
r = 1.7
A = 6. # Aspect ratio
R = A*r

r_init = r/5
maxtime = 2e-5
timesteps=4000

particles = Particles(len(jax.devices())*3)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
stel = Coils(curves, jnp.array([1e7, 1e7, 1e7]))

initial_values = stel.initial_conditions(particles, R, r_init)
initial_vperp = initial_values[4, :]

#trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
time0 = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
print(trajectories)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")
#start = time()
#stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
#end = time()
#print(f"Compiled took: {end-start:.2f} seconds")

stel.order = 5

projection2D(R, r, r_init, trajectories, show=False, save_as="examples/trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="examples/trajectories_top.pdf")

plt.figure()
for i in range(len(trajectories)):
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
plt.title("Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
plt.savefig("examples/non_opt_v_par.pdf", transparent=True)

normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], stel.gamma(), stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(B_norm, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy)
plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E}{E_\alpha}$")
plt.savefig("examples/non_opt_energy.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Initial Stellator", save_as="examples/non_opt_stellator.pdf", show=False)

#plt.show()
#raise SystemExit
############################################################################################################

start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
end = time()
print(f"Loss function initial value: {loss_value:.8f}")
print(f"Took: {end-start:.2f} seconds")

start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
end = time()
print(f"Compiled took: {end-start:.2f} seconds")


start = time()
grad_loss_value = grad(loss, argnums=0)(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
end = time()
print(f"Grad loss function initial value: {grad_loss_value}")
print(f"Took: {end-start:.2f} seconds")

start = time()
grad_loss_value = grad(loss, argnums=0)(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, 100)
end = time()
print(f"Compiled took: {end-start:.2f} seconds")

start = time()
optimize(stel, particles, R, r_init, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
#optimize_adam(stel, particles, R, r_init, initial_values, maxtime=maxtime*10, timesteps=timesteps, n_segments=100)
end = time()

print(f"Optimization took: {end-start:.1f} seconds") 

stel.save_coils("Optimizations.txt")

curves_segments = stel.gamma()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)

projection2D(R, r, r_init, trajectories, show=False, save_as="examples/opt_trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="examples/opt_trajectories_top.pdf")

plt.figure()
for i in range(len(trajectories)):
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
plt.title("Optimized Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
plt.savefig("examples/opt_v_par.pdf", transparent=True)

normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], curves_segments, stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(B_norm, 1, trajectories[i, :, :3], curves_segments, stel.currents)
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy)
plt.title("Optimized Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E}{E_\alpha}$")
plt.savefig("examples/opt_energy.pdf", transparent=True)

stel.plot(show=True, trajectories=trajectories, title="Optimized Stellator", save_as="examples/opt_stellator.pdf")
