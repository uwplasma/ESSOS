import os
import jax
import sys
import pybobyqa # type: ignore
from jax import jit
from time import time
import jax.numpy as jnp
from matplotlib import cm
from functools import partial
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy.optimize import least_squares, minimize
#### INPUT PARAMETERS START HERE ####
number_of_cores = 14
number_of_particles_per_core = 1
#### Some other imports
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.append("..")
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm
#### Input parameters continue here
n_curves=2
nfp=5
order=3
r = 2
A = 3. # Aspect ratio
R = A*r
r_init = r/4
maxtime = 1e-5
timesteps=1000
nparticles = len(jax.devices())*number_of_particles_per_core
n_segments=100
change_currents = False
model = 'Guiding Center' # 'Guiding Center' or 'Lorentz'
method = 'least_squares' # 'Bayesian', 'BOBYQA', 'least_squares' or one of scipy.optimize.minimize methods such as 'BFGS'
max_function_evaluations = 30
min_val = -11 # minimum coil dof value
max_val =  15 # maximum coil dof value
max_function_evaluations_BOBYQA = 150
coil_current = 1e7
##### Input parameters stop here
particles = Particles(nparticles)
curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

times = jnp.linspace(0, maxtime, timesteps)
x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
v0 = jnp.zeros((3, nparticles))
for i in range(nparticles):
    B0 = B(jnp.array([x0[i],y0[i],z0[i]]), curve_segments=curves.gamma(n_segments), currents=stel.currents)
    b0 = B0/jnp.linalg.norm(B0)
    perp_vector_1 = jnp.array([0, b0[2], -b0[1]])
    perp_vector_1_normalized = perp_vector_1/jnp.linalg.norm(perp_vector_1)
    perp_vector_2 = jnp.cross(b0, perp_vector_1)
    v0 = v0.at[:,i].set(vpar0[i]*b0 + vperp0[i]*(perp_vector_1_normalized/jnp.sqrt(2)+perp_vector_2/jnp.sqrt(2)))
normB0 = jnp.apply_along_axis(B_norm, 0, jnp.array([x0, y0, z0]), stel.gamma(), stel.currents)
μ = particles.mass*vperp0**2/(2*normB0)

start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime, timesteps, n_segments, model=model)
print(f"Loss function initial value: {loss_value:.8f}")
end = time()
print(f"Took: {end-start:.2f} seconds")

loss_partial = partial(loss, old_coils=stel,
                       particles=particles, R=R, r_init=r_init,
                       initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]) if model=='Lorentz' else jnp.array([x0, y0, z0, vpar0, vperp0]),
                       maxtime=maxtime, timesteps=timesteps, n_segments=n_segments, model=model)

len_dofs = len(jnp.ravel(stel.dofs))

@jit
def loss_partial_dofs_max(*args, **kwargs):
    x = jnp.array(list(kwargs.values()))
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    return -loss_partial(dofs, currents)

@jit
def loss_partial_dofs_min(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    return loss_partial(dofs, currents)

if change_currents:
    all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
else:
    all_dofs = jnp.ravel(stel.dofs)
print(f'Number of dofs: {len(all_dofs)}')

if method == 'Bayesian':
    initial_points = 20
    pbounds = {}
    for i in range(1, len(all_dofs) + 1):
        param = f'x{i}'
        if i<=len(jnp.ravel(stel.dofs)):
            pbounds[param] = (min_val, max_val)
        else:
            pbounds[param] = (1e5, 1e8)
    optimizer = BayesianOptimization(f=loss_partial_dofs_max,pbounds=pbounds,random_state=1)
    optimizer.maximize(init_points=initial_points,n_iter=max_function_evaluations)
    print(optimizer.max)
    x = jnp.array(list(optimizer.max['params'].values()))
else:
    if method == 'least_squares':
        res = least_squares(loss_partial_dofs_min, x0=all_dofs, verbose=2, ftol=1e-5, max_nfev=max_function_evaluations)
    else:
        if method == 'BOBYQA':
            max_function_evaluations = max_function_evaluations_BOBYQA
            if change_currents:
                lower = jnp.concatenate((jnp.array([min_val]*len_dofs), jnp.array([1e5]*n_curves)))
                upper = jnp.concatenate((jnp.array([max_val]*len_dofs), jnp.array([1e8]*n_curves)))
                res = pybobyqa.solve(loss_partial_dofs_min, x0=all_dofs, print_progress=True, objfun_has_noise=False, seek_global_minimum=False, rhoend=1e-5, maxfun=max_function_evaluations, bounds=(lower,upper))
            else:
                lower = jnp.array([min_val]*len_dofs)
                upper = jnp.array([max_val]*len_dofs)
                res = pybobyqa.solve(loss_partial_dofs_min, x0=all_dofs, print_progress=True, objfun_has_noise=False, seek_global_minimum=False, rhoend=1e-5, maxfun=max_function_evaluations)#, bounds=(lower,upper))
        else:            
            res = minimize(loss_partial_dofs_min, x0=all_dofs, method=method, options={'disp': True, 'maxiter':3, 'gtol':1e-5, 'xrtol':1e-5})
    x = jnp.array(res.x)
    
print(f'Resulting dofs: {repr(x.tolist())}')

time0 = time()
if model=='Lorentz':
    trajectories_initial = stel.trace_trajectories_lorentz(particles, initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
else:
    trajectories_initial = stel.trace_trajectories(particles, initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")

len_dofs = len(jnp.ravel(stel.dofs))
dofs = jnp.reshape(jnp.array(x)[:len_dofs], shape=stel.dofs.shape)
stel.dofs = dofs
if change_currents:
    currents = jnp.array(x)[len_dofs:]
    stel.currents = currents

time0 = time()
if model=='Lorentz':
    trajectories_final = stel.trace_trajectories_lorentz(particles, initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
else:
    trajectories_final = stel.trace_trajectories(particles, initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")

plt.figure()
colors = cm.viridis(jnp.linspace(0, 1, nparticles))
for i in range(nparticles):
    if model=='Lorentz':
        x_i, y_i, z_i, vx_i, vy_i, vz_i = trajectories_initial[i, :, :].transpose()
        x_f, y_f, z_f, vx_f, vy_f, vz_f = trajectories_final[i, :, :].transpose()
    else:
        x_i, y_i, z_i, vpar_i = trajectories_initial[i, :, :].transpose()
        x_f, y_f, z_f, vpar_f = trajectories_final[i, :, :].transpose()
    r_i = jnp.sqrt(x_i**2 + y_i**2)
    r_f = jnp.sqrt(x_f**2 + y_f**2)
    plt.plot(r_i, z_i, label='Initial Trajectories' if i==0 else '_no_legend_', color=colors[i], linestyle='-')
    plt.plot(r_f, z_f, label='Final Trajectories' if i==0 else '_no_legend_', color=colors[i], linestyle='--')
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.legend()

plt.show()

# # time0 = time()
# # trajectories_lorentz = stel.trace_trajectories_lorentz(particles, initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
# # print(f"Time to trace trajectories Lorentz: {time()-time0:.2f} seconds")

