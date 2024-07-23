import jax
import sys
import os
from time import time
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'
# Show on which platform JAX is running.
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])

os.mkdir("images") if not os.path.exists("images") else None


sys.path.insert(1, os.path.dirname(os.getcwd()))

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, optimize, loss, optimize_adam, projection2D, projection2D_top
from MagneticField import B_norm, B

model = "Lorentz" # "Guiding Center" or "Lorentz"

n_curves=2
order=3
coil_current=7e6
n_segments=100

r = 3
A = 2. # Aspect ratio
R = A*r

r_init = r/4
maxtime = 3e-5
timesteps=15000

n_particles = len(jax.devices())
particles = Particles(n_particles)

dofs = jnp.reshape(jnp.array(
    [5.890267840284325, 0.023015696788602182, 3.0157059371274517, -0.046787514040739324, 0.030158677016322175, 0.023729563724613666, 0.010555318529532919, 1.2003653254255195, 0.021229108610829883, 0.7054071640246989, -0.03393562054298491, -0.09097561376688454, 0.04491308553952268, -0.09660039344228535, 0.03346565357388103, -3.011851311084344, -0.0014506467708994709, 0.014675300276992076, 0.012068127915754798, 0.046991298948393295, 0.03500700615731799, 4.956795617354576, -0.05711561438660614, 2.4294629222993422, -0.026804562686284138, -0.04148356880312705, 0.0067459924660292, -0.00027898376776209653, 3.376085535692114, 0.6991845688079809, 1.719176069256245, -0.04299883638572471, -0.08076762907715977, -0.06997710316728854, 0.0037054411220158933, 0.001713667571414534, -3.0191643465872406, -0.01674095593209053, -0.008726490715081067, -0.014843704032555879, -0.045924289908639615, 0.03541769998164553]
), (n_curves, 3, 2*order+1))

#curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
curves = Curves(dofs, nfp=4, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
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

projection2D(R, r, r_init, trajectories, show=False, save_as="images/trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="images/top_trajectories.pdf")

plt.figure()
for i in range(len(trajectories)):
    if model == "Guiding Center":
        v_par = trajectories[i, :, 3]
    elif model == "Lorentz":
        B_field = jnp.apply_along_axis(B, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
        v_par = jnp.sum(trajectories[i, :, 3:]*B_field, axis=1)/jnp.linalg.norm(B_field, axis=1)
    else:
        raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
    
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, v_par)
plt.title("Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
plt.savefig("images/v_par.pdf", transparent=True)

plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(B_norm, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
    if model == "Guiding Center":
        normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy
    elif model == "Lorentz":
        normalized_energy = 0.5*particles.mass*(jnp.sum(trajectories[i, :, 3:]**2, axis=1))/particles.energy
    else:
        raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E}{E_\alpha}$")
y_limit = max(jnp.max(normalized_energy)-1, 1-jnp.min(normalized_energy))
order = jnp.floor(jnp.log10(y_limit))
plt.ylim(1-10**(order+1), 1+10**(order+1))
plt.yticks(jnp.linspace(1-10**(order+1), 1+10**(order+1), 11))
plt.savefig("images/energy_non_opt.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Stellator", save_as="images/stellator_non_opt.pdf", show=True)
