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
from MagneticField import B_norm

model = "Lorentz" # "Guiding Center" or "Lorentz"

n_curves=2
order=2
coil_current=7e6

r = 1.7
A = 6. # Aspect ratio
R = A*r

r_init = r/5
maxtime = 1e-7
timesteps=100

particles = Particles(len(jax.devices())*2)

dofs = jnp.reshape(jnp.array(
    [5.8816768669949235, -0.0001091781773779678, 2.9426538084165696, -0.0006121931796838491, 0.0005488831652268538, 1.1700514303927547, -3.846204946440075e-05, 0.5854790792369962, 0.0007283374762158154, -0.00032084656581056047, 0.0001646041074676944, -3.0003310202122244, 0.0006047324370085148, -0.00042555013540153966, -0.0006396618232154825, 4.988660226798091, -6.1443615346557e-05, 2.4948909560523433, 6.478701619722714e-05, -0.0006480203964514419, 3.2928787356556644, -0.00011788627795246733, 1.6669849337476699, 7.960341750094052e-05, 0.0010048939352689755, 0.0014223007560808846, -2.9993493557197306, 0.0001873509988254814, -0.0006047079448731891, -0.0005231007977422194]
), (n_curves, 3, 2*order+1))

#curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)
curves = Curves(dofs, nfp=4, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

initial_values = stel.initial_conditions(particles, R, r_init, model=model)
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

projection2D(R, r, r_init, trajectories, show=False, save_as="images/trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="images/top_trajectories.pdf")

plt.figure()
for i in range(len(trajectories)):
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
plt.title("Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
plt.savefig("images/v_par_non_opt.pdf", transparent=True)

normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], stel.gamma(), stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(B_norm, 1, trajectories[i, :, :3], stel.gamma(), stel.currents)
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy)
plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E}{E_\alpha}$")
plt.savefig("images/energy_non_opt.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Stellator", save_as="images/stellator_non_opt.pdf", show=True)
