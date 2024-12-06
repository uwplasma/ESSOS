import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/tracing") if not os.path.exists("images/tracing") else None
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

# jax.config.update("jax_traceback_filtering", "on")





from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, projection2D, projection2D_top, loss
from MagneticField import norm_B, B

n_curves=2
order=3
coil_current=7e6
n_segments=100

A = 3 # Aspect ratio
R = 6
r = R/A
r_init = r/4

maxtime = 2.e-5
timesteps = 2000 #int(maxtime/5.0e-10)

n_particles = len(jax.devices())
particles = Particles(n_particles)

dofs = jnp.reshape(jnp.array(
   [[[5.985254982928303, 0.26581044356679034, 1.8754815286562527, -0.002343217483557303, -0.13669786810588033, 0.274432599186244, 0.4160476648834756], [1.2574374960252934, 0.10979241260668832, 0.6942665166154958, -0.19713961137146607, 0.35333705125214127, 9.050297716582071e-05, 0.1744407039128007], [0.11723510813683916, -1.8482324850625271, -0.39166704965539123, 0.04617678339000649, -0.1448873074461018, 0.276807890403535, 0.5368756261731553]], [[4.873073395434523, 0.06763789699865053, 1.3084718949379763, -0.08748655977036536, 0.47029037378652094, 0.004441953215590499, -0.09071496636936334], [3.546339051540446, -0.08897437925830792, 1.374811798081262, -0.07208518124148725, -0.23604204056271447, 0.04266176781494097, 0.10802499314056253], [0.03854159493794037, -1.8433393610727444, 0.044055864631221346, 0.13116427637178907, -0.10285257651452304, 0.30317131125725527, -0.038550567788503076]]]
), (n_curves, 3, 2*order+1))
curves = Curves(dofs, nfp=4, stellsym=True)

# curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))


x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')

B_norm = jnp.apply_along_axis(norm_B, 0, jnp.array([x0, y0, z0]), stel.gamma, stel.gamma_dash, stel.currents)
μ = particles.mass*vperp0**2/(2*B_norm)

########################################
# Tracing trajectories

start = time()
trajectories = stel.trace_trajectories(particles, initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime=maxtime, timesteps=timesteps)
print(f"Time to trace trajectories: {time()-start:.2f} seconds")

########################################
# Calculating the loss

start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r, jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime, timesteps)
print(f"Loss function value: {loss_value:.8f}, took: {time()-start:.2f} seconds")

########################################

projection2D(R, r, trajectories, show=False, save_as="images/tracing/pol_trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="images/tracing/tor_trajectories.pdf")

plt.figure()
for i in range(n_particles):
    v_par = trajectories[i, :, 3]
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, v_par)

plt.title("Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
y_limit = max(jnp.abs(jnp.max(trajectories[:, :, 3])), jnp.abs(jnp.min(trajectories[:, :, 3])))
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("images/tracing/v_par.pdf", transparent=True)

y_limit = 0
plt.figure()
for i in range(n_particles):
    normB = jnp.apply_along_axis(norm_B, 1, trajectories[i, :, :3], stel.gamma, stel.gamma_dash, stel.currents)
    normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy-1
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))

plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-y_limit, y_limit)
plt.savefig("images/tracing/energy.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Stellator", save_as="images/tracing/stellator.pdf", show=True)