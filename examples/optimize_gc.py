import os
os.mkdir("output") if not os.path.exists("output") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=42'

import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad

# Show on which platform JAX is running.
print("JAX running on", len(jax.devices()), jax.devices()[0].platform.upper())

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, optimize, loss, loss_discrete, projection2D, projection2D_top
from MagneticField import norm_B

import matplotlib.pyplot as plt
from time import time

from simsopt.geo import CurveXYZFourier, curves_to_vtk
from simsopt.field import particles_to_vtk

n_curves=2
order=5
nfp = 4

A = 1.6 # Aspect ratio
R = 6
r = R/A
r_init = r/3
n_total_optimizations = 14
n_iterations_adam   = [5, 20, 30, 10]
learning_rates_adam = [0.01, 0.005, 0.001, 0.0005]
# n_iteration_least_squares = [20, 20, 20]
# diff_step_least_squares = [1e-2, 1e-3, 1e-4]
# jax_grad_least_squares = [False, False, True]
n_iteration_least_squares = [50]*7
diff_step_least_squares = [None, 1e-2, 1e-2,  1e-3,  1e-4,  1e-5, None]
jax_grad_least_squares =  [True, False, False, False, False, False, True]
ftol_least_squares = 1e-10

model = "Guiding Center"

maxtime = 2.0e-5 # seconds
timesteps = int(maxtime*1.2e7)
advance_factor_each_optimization = 1.2

optimize_least_squares = True
optimize_adam = False

particles = Particles(len(jax.devices()))

# dofs = jnp.reshape(jnp.array(
#     [[[5.87861514784579, -0.28501866663349484, 3.4605189781659873, -0.12015624299307263, 0.021570557809795016, 0.011180132886233103, -0.005914068747621175, 0.034911113393685356, -0.2251164586817591, -0.48021484259221686, 0.0017654158055129938], [1.0658542864260812, 1.596040619988665, 0.565536458298842, 0.008660268749333927, -0.055656894617905996, -0.5943288470635641, -0.007504236106932222, 0.015126688042198883, 0.03629734685444535, 0.1406524187674465, -0.010631452790200351], [-0.12442634754918139, -3.4938686453340564, 0.0398564066737513, 0.0063662297101601956, 0.01397593917706845, 0.10356710656582224, -0.03663673656744568, -0.1302055966545474, -0.009540145756700466, -0.06783184015770215, -0.04386722025275589]], [[4.928538865238128, -0.8889703375010589, 2.9498631330089893, 0.004118912063285094, -0.0027167488153265534, 0.31069225476189133, -0.12072964005061287, 0.07013635477010735, -0.08818514845400854, -0.10438658509356558, 0.001015675716562636], [3.348184787512189, 1.3038766710921765, 1.891757654652235, -0.012860090424311877, 0.019435714929617435, -0.6940406707925618, -0.1362967646810659, 0.3497993853357873, -0.04575370296007795, -0.08763635202967397, -0.07427452292886186], [-0.015370703753422317, -3.525501559957153, 0.040276889036711075, 0.16315769641365546, -0.033099327566270216, 0.05300123362473358, 0.024936527119119065, -0.004079761449087372, -0.06293940543249064, -0.004823080906574666, -0.014828452800671684]]]
# ), (n_curves, 3, 2*order+1))
# curves = Curves(dofs, nfp=nfp, stellsym=True)
# stel = Coils(curves, jnp.array([1.0, 1.6315314795574953]))

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)

stel = Coils(curves, jnp.array([1.0]*n_curves))

initial_values = stel.initial_conditions(particles, R, r_init, model=model, more_trapped_particles=True)
initial_vperp = initial_values[4, :]

time0 = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps)
print("Trajectories shape:", trajectories.shape)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")

projection2D(R, r, trajectories, show=False, save_as="output/init_pol_trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="output/init_tor_trajectories.pdf")

plt.figure()
for i in range(len(trajectories)):
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
plt.title("Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
y_limit = max(jnp.abs(jnp.max(trajectories[:, :, 3])), jnp.abs(jnp.min(trajectories[:, :, 3])))
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("output/init_vpar.pdf", transparent=True)

normB = jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], stel.gamma, stel.gamma_dash, stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

y_limit = 0
plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(norm_B, 1, trajectories[i, :, :3], stel.gamma, stel.gamma_dash, stel.currents)
    normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy-1
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))
plt.title("Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("output/init_energy.pdf", transparent=True)

stel.plot(trajectories=trajectories, title="Initial Stellator", save_as="output/init_stellator.pdf", show=False)
plt.close()

def create_simsopt_curves(curves):
    curves_simsopt = []
    for i, curve in enumerate(curves):
        curves_simsopt.append( CurveXYZFourier(100, order) )
        curves_simsopt[i].x = jnp.ravel(curve)
    return curves_simsopt

curves_to_vtk(create_simsopt_curves(stel._curves), "output/curves_init", close=True)
particles_to_vtk(res_tys=jnp.concatenate([trajectories[:, :, 3:4], trajectories[:, :, :3]], axis=2), filename="output/particles_init")
############################################################################################################

start = time()
dofs_with_currents = jnp.array(jnp.concatenate((jnp.ravel(stel.dofs), stel.dofs_currents[1:])))
# loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
loss_value = loss(dofs_with_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
print(f"Loss function initial value: {loss_value:.8f}, took: {time()-start:.2f} seconds")


start = time()
# grad_loss_value = grad(loss, argnums=0)(stel.dofs, stel.dofs_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
grad_loss_value = grad(loss)(dofs_with_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
# print(f"Grad loss function initial value:\n{jnp.ravel(grad_loss_value)}")
print(f"Grad shape: {grad_loss_value.shape}, took: {time()-start:.2f} seconds")

start = time()
for i in range(n_total_optimizations):
    ## USING SCIPY AND LEAST SQUARES
    if optimize_least_squares:
        for j, n_it in enumerate(n_iteration_least_squares):
            print(f"Optimization {i+1}/{n_total_optimizations}: Iteration {j+1} of {len(n_iteration_least_squares)} with {n_it} iterations, ftol={ftol_least_squares}, maxtime={maxtime}, timesteps={timesteps}, diff_step={diff_step_least_squares[j]}, jax_grad={jax_grad_least_squares[j]}")
            start_time=time();loss_value = loss(dofs_with_currents, stel, particles, R, r, initial_values, maxtime, timesteps);print(f"Loss function value: {loss_value:.8f}, took: {time()-start_time:.2f} seconds")
            res=optimize(stel, particles, R, r, initial_values, maxtime=maxtime, timesteps=timesteps, method={"method": "least_squares", "ftol": ftol_least_squares, "max_nfev": n_iteration_least_squares[j], "diff_step": diff_step_least_squares[j], "jax_grad": jax_grad_least_squares[j]})
            start_time=time();loss_value = loss(dofs_with_currents, stel, particles, R, r, initial_values, maxtime, timesteps);print(f"Loss function value: {loss_value:.8f}, took: {time()-start_time:.2f} seconds")
    # USING ADAM AND OPTAX
    if optimize_adam:
        for n_it, lr in zip(n_iterations_adam, learning_rates_adam):
            print(f"Optimization {i+1}/{n_total_optimizations}: Iteration {j+1} of {len(n_iterations_adam)} with learning rate {lr}, {n_it} iterations, maxtime={maxtime}, timesteps={timesteps}")
            res=optimize(stel, particles, R, r, initial_values, maxtime=maxtime, timesteps=timesteps, method={"method": "OPTAX adam", "learning_rate": lr, "iterations": n_it})
    
    stel.save_coils(f"output/Optimization_{i+1}.txt", text=f"loss={res}, maxtime={maxtime}, timesteps={timesteps}, lengths={stel.length[:n_curves]}")
    curves_to_vtk(create_simsopt_curves(stel._curves), f"output/curves_{i+1}", close=True)
    trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps)
    particles_to_vtk(res_tys=jnp.concatenate([trajectories[:, :, 3:4], trajectories[:, :, :3]], axis=2), filename=f"output/particles_{i+1}")
    
    maxtime *= advance_factor_each_optimization
    timesteps = int(timesteps*advance_factor_each_optimization)

print(f"Optimization took: {time()-start:.1f} seconds") 

stel.save_coils("Optimizations.txt")

trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps)

projection2D(R, r, trajectories, show=False, save_as="output/opt_pol_trajectories.pdf")
projection2D_top(R, r, trajectories, show=False, save_as="output/opt_tor_trajectories.pdf")

plt.figure()
for i in range(len(trajectories)):
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
plt.title("Optimized Parallel Velocity")
plt.xlabel("time [s]")
plt.ylabel(r"parallel velocity [ms$^{-1}$]")
y_limit = max(jnp.abs(jnp.max(trajectories[:, :, 3])), jnp.abs(jnp.min(trajectories[:, :, 3])))
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("output/opt_vpar.pdf", transparent=True)

normB = jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], stel.gamma, stel.gamma_dash, stel.currents)
μ = particles.mass*initial_vperp**2/(2*normB)

y_limit = 0
plt.figure()
for i in range(len(trajectories)):
    normB = jnp.apply_along_axis(norm_B, 1, trajectories[i, :, :3], stel.gamma, stel.gamma_dash, stel.currents)
    normalized_energy = (μ[i]*normB + 0.5*particles.mass*trajectories[i, :, 3]**2)/particles.energy-1
    plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, normalized_energy)
    y_limit = max(y_limit, jnp.abs(jnp.max(normalized_energy)), jnp.abs(jnp.min(normalized_energy)))
plt.title("Optimized Energy Conservation")
plt.xlabel("time [s]")
plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
plt.ylim(-1.2*y_limit, 1.2*y_limit)
plt.savefig("output/opt_energy.pdf", transparent=True)

stel.plot(show=True, trajectories=trajectories, title="Optimized Stellator", save_as="output/opt_stellator.pdf")

curves_to_vtk(create_simsopt_curves(stel._curves), f"output/curves_opt", close=True)
particles_to_vtk(res_tys=jnp.concatenate([trajectories[:, :, 3:4], trajectories[:, :, :3]], axis=2), filename="output/particles_opt")