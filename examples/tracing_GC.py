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





from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, projection2D, projection2D_top
from MagneticField import B_norm, B

model = "Guiding Center" # "Guiding Center" or "Lorentz"

n_curves=2
order=3
coil_current=7e6
n_segments=100

r = 3
A = 2. # Aspect ratio
R = A*r

r_init = r/4
maxtime = 3e-5
timesteps = int(maxtime/5.0e-10)

n_particles = len(jax.devices()*2)
particles = Particles(n_particles)

dofs = jnp.reshape(jnp.array(
    [[[6.092399894957858, -0.024470340574007914, 3.0719746921121005, -0.028368927672627452, 0.0209969184912135, 0.047894031060180094, 0.25637378199180805], [0.8716042523881473, 0.05766079617231436, 0.5590378572065909, -0.18902743059120447, 0.15146435631733493, 0.03685179096003223, 0.557422958102676], [0.5045795016525154, -2.492316864969798, -0.3112614640391156, -0.012557962312413287, -0.4874554808123133, 0.25782973452019947, 0.4904250515493357]], [[4.699634400701741, 0.05433444026266678, 2.1102361473807227, -0.025550352547440258, -0.14693310973186055, 0.05046299854266632, 0.3816227741533424], [3.521371748567928, 0.014105103236513228, 1.5537123508542654, 0.0715558471890779, 0.20395300747959985, -0.023761103913984536, 0.35927656131445906], [0.06408331708629529, -2.5656461690839008, -0.4639042571658612, 0.5090810923281613, -0.10814277419357944, 0.5600433107883754, 0.5620951717018253]]]
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
