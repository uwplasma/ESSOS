import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/tracing") if not os.path.exists("images/tracing") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=1'

import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad

# Show on which platform JAX is running.
print("JAX running on", len(jax.devices()), jax.devices()[0].platform.upper())

from time import time
import matplotlib.pyplot as plt





from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, projection2D, projection2D_top
from MagneticField import B_norm, B

model = "Lorentz" # "Guiding Center" or "Lorentz"

n_curves=1
order=3
coil_current=7e6
n_segments=100

R = 6
A = 3 # Aspect ratio
r = R/A

r_init = r/4
maxtime = 1e-6
timesteps = int(maxtime/5.0e-10)

n_particles = len(jax.devices()*1)
particles = Particles(n_particles)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=1, stellsym=False)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
x0, y0, z0 = jnp.array([-6]), jnp.array([0]), jnp.array([0])
v0 = jnp.empty((3, n_particles))
norm_B0 = jnp.zeros((n_particles,))
for i in range(n_particles):
    B0 = B(jnp.array([x0[i],y0[i],z0[i]]), curve_segments=stel.gamma(n_segments), currents=stel.currents)
    b0 = B0/jnp.linalg.norm(B0)
    perp_vector_1 = jnp.array([0, b0[2], -b0[1]])
    perp_vector_1 /= jnp.linalg.norm(perp_vector_1)
    perp_vector_2 = jnp.cross(b0, perp_vector_1)
    v0 = v0.at[:, i].set(vpar0[i]*b0 + vperp0[i]*(perp_vector_1+perp_vector_2)/jnp.sqrt(2))
    norm_B0 = norm_B0.at[i].set(B_norm(jnp.array([x0[i],y0[i],z0[i]]), stel.gamma(n_segments), stel.currents))
μ = particles.mass*vperp0**2/(2*norm_B0)

########################################
# Tracing trajectories

start = time()
if model == "Guiding Center":
    trajectories = stel.trace_trajectories(particles, initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime=maxtime, timesteps=timesteps, n_segments=100)
elif model == "Lorentz":
    trajectories = stel.trace_trajectories_lorentz(particles, initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime=maxtime, timesteps=timesteps, n_segments=100)
else:
    raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
end = time()
print(f"Time to trace trajectories: {end-start:.2f} seconds")

########################################

projection2D(R, r, r_init, trajectories, show=False, save_as="images/optimization/init_pol_trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="images/optimization/init_tor_trajectories.pdf")

plt.figure()
for i in range(len(trajectories)):
    if model == "Guiding Center":
        v_par = trajectories[i, :, 3]
    else:
        B_field = jnp.apply_along_axis(B, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
        v_par = jnp.sum(trajectories[i, :, 3:]*B_field, axis=1)/jnp.linalg.norm(B_field, axis=1)
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, v_par)

plt.title("Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
y_limit = max(jnp.abs(jnp.max(trajectories[:, :, 3])), jnp.abs(jnp.min(trajectories[:, :, 3])))
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("images/tracing/v_par.pdf", transparent=True)

y_limit = 0
plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(B_norm, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
    if model == "Guiding Center":
        normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy-1
    else:
        normalized_energy = 0.5*particles.mass*(jnp.sum(trajectories[i, :, 3:]**2, axis=1))/particles.energy-1
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))

plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-y_limit, y_limit)
plt.savefig("images/tracing/energy_non_opt.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Stellator", save_as="images/tracing/stellator.pdf", show=True)
