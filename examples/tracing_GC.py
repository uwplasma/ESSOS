import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/tracing") if not os.path.exists("images/tracing") else None
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

# jax.config.update("jax_traceback_filtering", "on")





from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, projection2D, projection2D_top, loss
from MagneticField import norm_B, B

n_curves=2
order=3
coil_current=7e6
n_segments=100

A = 4.5 # Aspect ratio
R = 7.75
r = R/A

r_init = r/4
maxtime = 1.5e-5
timesteps = 100#int(maxtime/5.0e-10)

n_particles = len(jax.devices())
particles = Particles(n_particles)

#dofs = jnp.reshape(jnp.array(
#    [[[7.659175987226048, -0.04041399248053201, 1.9000419191055233, 0.09984825794723749, 0.12141277775953799, -0.1802613655260083, 0.31285398985391266], [1.6347473551588028, 0.10374265613747131, 0.5484620451498052, -0.2467564733908836, -0.012948835384266454, -0.2428340210279131, 0.0309345404359862], [-0.0016659721547670797, -1.722966622710651, 0.15334408422046225, -0.017781357517937517, 0.053890811586643765, -0.16312627482765366, -0.30200232777424296]], [[6.509310482871267, 0.04243770546820939, 1.3554276067256248, 0.09364112697926023, -0.06182844844102781, 0.13594928751305274, 0.1503271683030747], [4.345736292458465, 0.03856793707689319, 0.9649080561894142, -0.13168653236561864, 0.4565006970796698, 0.34350580255001223, 0.17158207991691418], [0.22993997525979984, -1.554508114311937, -0.25129611645158545, -0.07479651705652306, 0.1756537051472456, 0.25196472260154, 0.18388052641444086]]]
#), (n_curves, 3, 2*order+1))
#curves = Curves(dofs, nfp=4, stellsym=True)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
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

projection2D(R, r, trajectories, show=False, save_as="images/optimization/init_pol_trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="images/optimization/init_tor_trajectories.pdf")

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
plt.savefig("images/tracing/energy_non_opt.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Stellator", save_as="images/tracing/stellator.pdf", show=True)