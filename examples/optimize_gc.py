import os
os.mkdir("output") if not os.path.exists("output") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=13'

import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad

# Show on which platform JAX is running.
print("JAX running on", len(jax.devices()), jax.devices()[0].platform.upper())

from ESSOS import CreateEquallySpacedCurves, Curves, Coils, Particles, optimize, loss, loss_discrete, projection2D, projection2D_top
from MagneticField import norm_B, B, BdotGradPhi, BdotGradTheta, BdotGradr

import matplotlib.pyplot as plt
from time import time

from simsopt.geo import CurveXYZFourier, curves_to_vtk
from simsopt.field import particles_to_vtk

from pyevtk.hl import polyLinesToVTK
import numpy as np

n_curves=2
order=5
nfp = 3

A = 1.6 # Aspect ratio
R = 7.75 # Major Radius
r = R/A
r_init = r/3
n_total_optimizations = 24

optimize_adam = False
n_iterations_adam   = [10, 10, 10, 10]
learning_rates_adam = [1e-2, 1e-3, 1e-4, 1e-5]

optimize_scipy_minimize = False
n_iteration_scipy_minimize = [150]*1#[150]*6
diff_step_scipy_minimize =   [None]*1#[None, 1e-2,  1e-3,  1e-4,  1e-5,  None]
jax_grad_scipy_minimize =    [True]*1#[True, False, False, False, False, True]
ftol_scipy_minimize = 1e-7

optimize_least_squares = True
n_iteration_least_squares = [30]*1
diff_step_least_squares =   [None]*1
jax_grad_least_squares =    [True]*1
# n_iteration_least_squares = [50]*7 #[150] + [50]*5 + [150]
# diff_step_least_squares =   [None, 1e-2,  1e-3,  1e-4,  1e-5, 1e-6,  None]
# jax_grad_least_squares =    [True, False, False, False, False, False, True]
ftol_break_opt=5e-2
ftol_least_squares = 1e-7

model = "Guiding Center"

maxtime = 2.0e-5 # seconds
timesteps = int(maxtime*1.0e7)
advance_factor_each_optimization = 1.15
current_on_axis = 5.7 # Tesla

n_segments = int(order*9)

particles = Particles(len(jax.devices()))

# dofs = jnp.reshape(jnp.array(
#     [[[5.884646365152857, -0.006966333266796913, 3.9232541897919755, -0.0010264920955900733, -6.067892763040939e-06, 0.0018722717253900393, -1.8236185511322842e-05, -0.00026234739026347894, -5.3896539420907005e-06, -0.00020557093647479178, 1.914205305631024e-05, -5.7085273431708875e-05, -6.7150177142536006e-06], [1.170750894459735, 0.0350464847261024, 0.7803773655254461, 0.004294172560803225, 4.181280897514952e-06, -0.00939485233801154, -1.6192074744714683e-05, 0.0013545887856467088, -3.906021902689677e-06, 0.0010241093063451988, 4.128938422449541e-06, 0.0002932983193410755, -1.0942373043147827e-06], [0.00035146985887338493, -4.000086199108211, -4.792998877239866e-06, 1.403980449008426e-05, -0.00018167008189412715, 1.1420551667236116e-05, 6.321750009217482e-06, -6.545698466671377e-06, 5.013406431260768e-06, -1.3162282903677286e-05, -9.447949286193992e-07, 6.251865938105334e-06, 8.311336236438015e-07]], [[4.988623020260489, -0.019395970135114664, 3.3257630130074816, -0.0025539994504142987, 1.5026220936024754e-05, 0.0053169222873849405, -8.088241806055982e-06, -0.0007596858459847748, 1.0834688311771722e-06, -0.0005837377828511828, 1.5494833378778795e-05, -0.00016416368074109974, -5.800328496401407e-06], [3.3337018351146024, 0.029065495166206545, 2.222234761832781, 0.003559344963302599, -3.1807167499988598e-06, -0.007926241618039386, -2.102172277983952e-05, 0.001151097068369355, 9.027071608931016e-07, 0.0008664368096924012, 1.0129819613287715e-05, 0.00024747672157938934, -3.7114376565492085e-06], [0.0003092885875381999, -3.99984653672012, -2.0779262231424474e-05, 1.283580579885897e-05, -0.0001619928142811235, 7.888203837649906e-06, 2.4634411135559413e-05, -1.4026108970183353e-05, 6.976868494072207e-06, -1.2262192546230949e-05, -3.2674641862872915e-06, 6.117638310954991e-06, 1.367103375155425e-07]]]
# ), (n_curves, 3, 2*order+1))
# curves = Curves(dofs, nfp=nfp, stellsym=True)
# stel = Coils(curves, jnp.array([1.0, 1.0017654275184882]))

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True, n_segments=n_segments)
stel = Coils(curves, jnp.array([5.7*current_on_axis/len(curves._curves)]*n_curves))
key = jax.random.PRNGKey(42)
stel.dofs += jax.random.normal(key, stel.dofs.shape)*r*1e-2

# print(stel.gamma.shape)
# print(jnp.mean(stel.gamma[2, :, 2]))
# exit()

initial_values = stel.initial_conditions(particles, R, r_init, model=model, more_trapped_particles=False)#True, trapped_fraction_more=0.4)
initial_vperp = initial_values[4, :]

time0 = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps)

# nr = 4;nphi = 4;nz = 4;
# r_max = r/5;phi_min = 1e-3;phi_max = 1e-4#2*jnp.pi-1e-3
# r_0 = jnp.linspace(start=-r_max, stop=r_max, num=nr)
# phi_array = jnp.linspace(start=phi_min, stop=phi_max, num=nphi)
# z_array = jnp.linspace(start=-r_max, stop=r_max, num=nz)
# r_0_grid, phi_grid, z_grid = jnp.meshgrid(r_0, phi_array, z_array)
# xyz_array = jnp.stack([(r_0_grid + R) * jnp.cos(phi_grid),
#                             (r_0_grid + R) * jnp.sin(phi_grid),
#                             z_grid], axis=-1).reshape(-1, 3)
# # B_z = jax.vmap(lambda rphi: B(rphi, coils.gamma, coils.gamma_dash, coils.currents, R)[2])(r_phi_array_plus)
# B_iota_fieldlines  = jax.vmap(lambda xyz: 
#                                         BdotGradTheta(xyz, stel.gamma, stel.gamma_dash, stel.currents, R)
#                                         /BdotGradPhi( xyz, stel.gamma, stel.gamma_dash, stel.currents, R)
#                                         )(xyz_array)
# B_r_fieldlines  = jax.vmap(lambda rphi: BdotGradr(rphi, stel.gamma, stel.gamma_dash, stel.currents, R))(xyz_array)/len(xyz_array)
# print(f"xyz array: {xyz_array}")
# print(f"Biota: {B_iota_fieldlines}")
# print(f"Br: {B_r_fieldlines}")
# exit()

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
print(f"normB for a particle at t=0: {normB[0]:2f} T")
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

def create_field_lines(stel, maxtime, timesteps, n_segments, filename):
    def particles_to_vtk_fl(res_tys, filename):
        x = np.concatenate([xyz[:, 0] for xyz in res_tys])
        y = np.concatenate([xyz[:, 1] for xyz in res_tys])
        z = np.concatenate([xyz[:, 2] for xyz in res_tys])
        ppl = np.asarray([xyz.shape[0] for xyz in res_tys])
        data = np.concatenate([i*np.ones((res_tys[i].shape[0], )) for i in range(len(res_tys))])
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})
    # r_init = r/3
    n_fieldlines = len(trajectories)
    angle = 0
    r_ = jnp.linspace(start=-r_init, stop=r_init, num=n_fieldlines)
    ϕ = jnp.ones(n_fieldlines)*angle
    x_fl = (r_+R)*jnp.cos(ϕ)
    y_fl = (r_+R)*jnp.sin(ϕ)
    z_fl = jnp.zeros(n_fieldlines)
    trajectories_fieldlines = stel.trace_fieldlines(jnp.array([x_fl, y_fl, z_fl]), maxtime/50, timesteps, n_segments)
    particles_to_vtk_fl(res_tys=trajectories_fieldlines, filename=filename)

def create_simsopt_curves(curves):
    curves_simsopt = []
    for i, curve in enumerate(curves):
        curves_simsopt.append( CurveXYZFourier(100, order) )
        curves_simsopt[i].x = jnp.ravel(curve)
    return curves_simsopt

curves_to_vtk(create_simsopt_curves(stel._curves), "output/curves_init", close=True)
particles_to_vtk(res_tys=jnp.concatenate([trajectories[:, :, 3:4], trajectories[:, :, :3]], axis=2), filename="output/particles_init")
create_field_lines(stel, maxtime, timesteps, n_segments, filename=f"output/field_lines_init")
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

def save_all(stel,res,maxtime,timesteps,i):
    stel.save_coils(f"output/Optimization_{i+1}.txt", text=f"loss={res}, maxtime={maxtime}, timesteps={timesteps}, lengths={stel.length[:n_curves]}")
    curves_to_vtk(create_simsopt_curves(stel._curves), f"output/curves_{i+1}", close=True)
    trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps)
    particles_to_vtk(res_tys=jnp.concatenate([trajectories[:, :, 3:4], trajectories[:, :, :3]], axis=2), filename=f"output/particles_{i+1}")
    create_field_lines(stel, maxtime, timesteps, n_segments, filename=f"output/field_lines_{i+1}")

start = time()
for i in range(n_total_optimizations):
    ## USING SCIPY AND LEAST SQUARES
    if optimize_least_squares:
        for j, n_it in enumerate(n_iteration_least_squares):
            new_loss_value = 0
            while jnp.abs(1-jnp.abs(new_loss_value/loss_value)) > ftol_break_opt:
                start_time=time();loss_value = new_loss_value
                print(f"Optimization {i+1}/{n_total_optimizations}: Iteration {j+1} of {len(n_iteration_least_squares)} with {n_it} iterations, ftol={ftol_least_squares}, maxtime={maxtime}, timesteps={timesteps}, diff_step={diff_step_least_squares[j]}, jax_grad={jax_grad_least_squares[j]}")
                res=optimize(stel, particles, R, r, initial_values, maxtime=maxtime, timesteps=timesteps, method={"method": "least_squares", "ftol": ftol_least_squares, "max_nfev": n_iteration_least_squares[j], "diff_step": diff_step_least_squares[j], "jax_grad": jax_grad_least_squares[j]})
                dofs_with_currents = jnp.array(jnp.concatenate((jnp.ravel(stel.dofs), stel.dofs_currents[1:])))
                new_loss_value = loss(dofs_with_currents, stel, particles, R, r, initial_values, maxtime, timesteps)
                save_all(stel,res,maxtime,timesteps,i)            
                print(f"Loss function value: {new_loss_value:.8f}, took: {time()-start_time:.2f} seconds and is {jnp.abs(1-jnp.abs(new_loss_value/loss_value)):.2f} smaller than previous")
            loss_value = new_loss_value
    if optimize_scipy_minimize:
        for j, n_it in enumerate(n_iteration_scipy_minimize):
            print(f"Optimization {i+1}/{n_total_optimizations}: Iteration {j+1} of {len(n_iteration_scipy_minimize)} with {n_it} iterations, ftol={ftol_scipy_minimize}, maxtime={maxtime}, timesteps={timesteps}, diff_step={diff_step_scipy_minimize[j]}, jax_grad={jax_grad_scipy_minimize[j]}")
            res=optimize(stel, particles, R, r, initial_values, maxtime=maxtime, timesteps=timesteps, method={"method": "scipy_minimize", "ftol": ftol_scipy_minimize, "max_nfev": n_iteration_scipy_minimize[j], "diff_step": diff_step_scipy_minimize[j], "jax_grad": jax_grad_scipy_minimize[j]})
            dofs_with_currents = jnp.array(jnp.concatenate((jnp.ravel(stel.dofs), stel.dofs_currents[1:])))
            start_time=time();loss_value = loss(dofs_with_currents, stel, particles, R, r, initial_values, maxtime, timesteps);print(f"Loss function value: {loss_value:.8f}, took: {time()-start_time:.2f} seconds")
            save_all(stel,res,maxtime,timesteps,i)
    # USING ADAM AND OPTAX
    if optimize_adam:
        for j, n_it in enumerate(n_iterations_adam):
            print(f"Optimization {i+1}/{n_total_optimizations}: Iteration {j+1} of {len(n_iterations_adam)} with learning rate {learning_rates_adam[j]}, {n_it} iterations, maxtime={maxtime}, timesteps={timesteps}")
            res=optimize(stel, particles, R, r, initial_values, maxtime=maxtime, timesteps=timesteps, method={"method": "OPTAX adam", "learning_rate": learning_rates_adam[j], "iterations": n_it})
            save_all(stel,res,maxtime,timesteps,i)

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