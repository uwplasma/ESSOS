import jax
import sys
import os
from time import time
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
# Show on which platform JAX is running.
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])

try:
    os.mkdir("examples/images")
except FileExistsError:
    pass

sys.path.insert(1, os.getcwd())

from ESSOS import CreateEquallySpacedCurves, Coils, Particles, optimize, loss, optimize_adam, projection2D, projection2D_top
from MagneticField import B_norm

n_curves=3
order=3
r = 1.7
A = 6. # Aspect ratio
R = A*r

r_init = r/5
maxtime = 1e-7
timesteps=100

particles = Particles(len(jax.devices())*2)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
stel = Coils(curves, jnp.array([1e7, 1e7, 1e7]))

initial_values = stel.initial_conditions(particles, R, r_init)
initial_vperp = initial_values[4, :]

########################################
# Tracing trajectories

start = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
end = time()
print(f"Time to trace trajectories: {end-start:.2f} seconds")

start = time()
stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
end = time()
print(f"Compiled took: {end-start:.2f} seconds")

########################################

########################################
# Tracing trajectories vec

start = time()
trajectories_vec = stel.trace_trajectories_vec(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
end = time()
print(f"Time to trace trajectories: {end-start:.2f} seconds")

start = time()
stel.trace_trajectories_vec(particles, initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=100)
end = time()
print(f"Compiled took: {end-start:.2f} seconds")

########################################
print(jnp.allclose(trajectories, trajectories_vec))
print(trajectories.shape)
print(trajectories_vec.shape)

projection2D(R, r, r_init, trajectories, show=False, save_as="examples/images/trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="examples/images/top_trajectories.pdf")

plt.figure()
for i in range(len(trajectories)):
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
plt.title("Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
plt.savefig("examples/images/v_par_non_opt.pdf", transparent=True)

normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], stel.gamma(), stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(B_norm, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy)
plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E}{E_\alpha}$")
plt.savefig("examples/images/energy_non_opt.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Stellator", save_as="examples/images/stellator_non_opt.pdf", show=True)
