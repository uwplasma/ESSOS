import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/comparison") if not os.path.exists("images/comparison") else None
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
plt.rcParams['font.size'] = 16

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, projection2D, projection2D_top
from MagneticField import norm_B, B

n_curves=1
order=3
coil_current=7e6
n_segments=100

R = 3
A = 4 # Aspect ratio
r = R/A

r_init = r/4
maxtime = 4e-7
timesteps = int(maxtime/5.0e-10)

n_particles = len(jax.devices()*1)
particles = Particles(n_particles)
m = particles.mass
q = particles.charge

curves = CreateEquallySpacedCurves(n_curves, order, R, r, n_segments=n_segments, nfp=1, stellsym=False)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

_, _, _, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
x0, y0, z0 = jnp.array([-R]), jnp.array([0]), jnp.array([0])
v0 = jnp.empty((3, n_particles))
B_norm = jnp.apply_along_axis(norm_B, 0, jnp.array([x0, y0, z0]), stel.gamma, stel.gamma_dash, stel.currents)
μ = m*vperp0**2/(2*B_norm)
for i in range(n_particles):
    B0 = B(jnp.array([x0[i],y0[i],z0[i]]), stel.gamma, stel.gamma_dash, stel.currents)
    b0 = B0/jnp.linalg.norm(B0)
    perp_vector_1 = jnp.array([0, b0[2], -b0[1]])
    perp_versor_1 = perp_vector_1/jnp.linalg.norm(perp_vector_1)
    perp_versor_2 = jnp.cross(b0, perp_versor_1)
    v0 = v0.at[:, i].set(vpar0[i]*b0 + vperp0[i]*(perp_versor_1+perp_versor_2)/jnp.sqrt(2))
gyroangle = -jnp.pi/4
x0_lorentz, y0_lorentz, z0_lorentz = jnp.array([x0, y0, z0]) + jnp.array([perp_versor_1*jnp.sin(gyroangle)+perp_versor_2*jnp.cos(gyroangle)]).T * m/q*vperp0/B_norm


########################################
# Tracing trajectories

start = time()
trajectories_gc = stel.trace_trajectories(particles, initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime=maxtime, timesteps=timesteps)
trajectories_lorentz = stel.trace_trajectories_lorentz(particles, initial_values=jnp.array([x0_lorentz, y0_lorentz, z0_lorentz, v0[0], v0[1], v0[2]]), maxtime=maxtime, timesteps=timesteps)

end = time()
print(f"Time to trace trajectories: {end-start:.2f} seconds")

########################################

y_limit = 0
plt.figure()
for i in range(n_particles):
    normB = jnp.apply_along_axis(norm_B, 1, trajectories_gc[i, :, :3], stel.gamma, stel.gamma_dash, stel.currents)
    normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories_gc[i, :, 3]**2)/particles.energy-1
    plt.plot(jnp.arange(timesteps)*maxtime/timesteps, normalized_energy, color="blue", linewidth=1.7)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-8e-7, 8e-7)
plt.grid(axis="y", linestyle="--") 
plt.tight_layout()
plt.savefig("images/comparison/energy_gc.pdf", transparent=True)

y_limit = 0
plt.figure()
for i in range(n_particles):
    normB = jnp.apply_along_axis(norm_B, 1, trajectories_lorentz[i, :, :3], stel.gamma, stel.gamma_dash, stel.currents)
    normalized_energy = 0.5*particles.mass*(jnp.sum(trajectories_lorentz[i, :, 3:]**2, axis=1))/particles.energy-1
    plt.plot(jnp.arange(timesteps)*maxtime/timesteps, normalized_energy, color="blue", linewidth=1.7)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))

plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-3e-15, 3e-15)
plt.grid(axis="y", linestyle="--") 
plt.tight_layout()
plt.savefig("images/comparison/energy_lorentz.pdf", transparent=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

gamma = stel.gamma
xlims = []
ylims = []
zlims = []
for i in range(n_curves):
    color = "lightgrey"
    ax.plot3D(gamma[i, :, 0], gamma[i, :,  1], gamma[i, :, 2], color=color, zorder=10)
    if i == 0:
        xlims = [jnp.min(gamma[i, :, 0]), jnp.max(gamma[i, :, 0])]
        ylims = [jnp.min(gamma[i, :, 1]), jnp.max(gamma[i, :, 1])]
        zlims = [jnp.min(gamma[i, :, 2]), jnp.max(gamma[i, :, 2])]
    else:
        xlims = [min(xlims[0], jnp.min(gamma[i, :, 0])), max(xlims[1], jnp.max(gamma[i, :, 0]))]
        ylims = [min(ylims[0], jnp.min(gamma[i, :, 1])), max(ylims[1], jnp.max(gamma[i, :, 1]))]
        zlims = [min(zlims[0], jnp.min(gamma[i, :, 2])), max(zlims[1], jnp.max(gamma[i, :, 2]))]


for i in range(jnp.size(trajectories_gc, 0)):
    ax.plot3D(trajectories_gc[i, :, 0], trajectories_gc[i, :, 1], trajectories_gc[i, :, 2], zorder=0, color= "red")
for i in range(jnp.size(trajectories_lorentz, 0)):
    ax.plot3D(trajectories_lorentz[i, :, 0], trajectories_lorentz[i, :, 1], trajectories_lorentz[i, :, 2], zorder=0, color= "blue")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.set_zlim(zlims)

ax.set_aspect('equal')
ax.locator_params(axis='z', nbins=3)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.axis('off')
ax.grid(False)
fig.tight_layout()

plt.savefig("images/comparison/trajectories.pdf", transparent=True)
plt.show()