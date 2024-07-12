import os
import jax
import sys
from tqdm import tqdm
from jax import jit
from time import time
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize
from matplotlib import cm
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from bayes_opt import BayesianOptimization
number_of_cores = 5
number_of_particles_per_core = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.insert(1, os.path.dirname(os.getcwd()))
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm

n_curves=2
nfp=4
order=2
r = 3
A = 2 # Aspect ratio
R = A*r
r_init = r/4
maxtime = 4e-6
timesteps_guiding_center=max(1000,int(maxtime/1.0e-8))
timesteps_lorentz=int(maxtime/1.0e-10)
nparticles = len(jax.devices())*1
n_segments=100
coil_current = 7e6
max_function_evaluations = 25
method = 'least_squares' # 'L-BFGS-B','least_squares','Bayesian' or one of scipy.optimize.minimize methods such as 'BFGS'

particles = Particles(nparticles)
curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
v0 = jnp.zeros((3, nparticles))
for i in range(nparticles):
    B0 = B(jnp.array([x0[i],y0[i],z0[i]]), curve_segments=stel.gamma(n_segments), currents=stel.currents)
    b0 = B0/jnp.linalg.norm(B0)
    perp_vector_1 = jnp.array([0, b0[2], -b0[1]])
    perp_vector_1_normalized = perp_vector_1/jnp.linalg.norm(perp_vector_1)
    perp_vector_2 = jnp.cross(b0, perp_vector_1_normalized)
    v0 = v0.at[:,i].set(vpar0[i]*b0 + vperp0[i]*(perp_vector_1_normalized/jnp.sqrt(2)+perp_vector_2/jnp.sqrt(2)))
normB0 = jnp.apply_along_axis(B_norm, 0, jnp.array([x0, y0, z0]), stel.gamma(n_segments), stel.currents)
μ = particles.mass*vperp0**2/(2*normB0)

start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime, timesteps_guiding_center, n_segments, model='Guiding Center')
print(f"Loss function initial value with Guiding Center: {loss_value:.8f} took: {time()-start:.2f} seconds")

start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime, timesteps_lorentz, n_segments, model='Lorentz')
print(f"Loss function initial value with Lorentz: {loss_value:.8f} took: {time()-start:.2f} seconds")
##############
loss_partial_gc = partial(loss, old_coils=stel,
                       particles=particles, R=R, r_init=r_init,
                       initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]),
                       maxtime=maxtime, timesteps=timesteps_guiding_center, n_segments=n_segments, model='Guiding Center')

change_currents = False

@jit
def loss_partial_gc_x0_max(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (stel.dofs, stel.currents.at[0].set(x)[:n_curves]),
        lambda _: (stel.dofs.at[0, 0, 2].set(x), stel.currents[:n_curves]),
        operand=None
    )
    return -loss_partial_gc(dofs, currents)

@jit
def loss_partial_gc_x0_min(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (stel.dofs, stel.currents.at[0].set(x[0])[:n_curves]),
        lambda _: (stel.dofs.at[0, 0, 2].set(x[0]), stel.currents[:n_curves]),
        operand=None
    )
    return loss_partial_gc(dofs, currents)

##############

loss_partial_lorentz = partial(loss, old_coils=stel,
                       particles=particles, R=R, r_init=r_init,
                       initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]),
                       maxtime=maxtime, timesteps=timesteps_lorentz, n_segments=n_segments, model='Lorentz')
@jit
def loss_partial_lorentz_x0_max(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (stel.dofs, stel.currents.at[0].set(x)[:n_curves]),
        lambda _: (stel.dofs.at[0, 0, 2].set(x), stel.currents[:n_curves]),
        operand=None
    )
    return -loss_partial_lorentz(dofs, currents)

@jit
def loss_partial_lorentz_x0_min(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (stel.dofs, stel.currents.at[0].set(x[0])[:n_curves]),
        lambda _: (stel.dofs.at[0, 0, 2].set(x[0]), stel.currents[:n_curves]),
        operand=None
    )
    return loss_partial_lorentz(dofs, currents)

##############

if change_currents:
    x0 = stel.currents[0]
    xmin = x0*5
    xmax = x0/5
else:
    xmin = 1.5
    xmax = 8
    x0 = stel.dofs[0, 0, 2]*0.7
print(f'Initial guess: {x0} with bounds: {xmin} and {xmax}')

pbounds = {'x': (xmin, xmax)}

if method=='Bayesian':
    print('Guiding Center Optimization')
    time0_gc = time()
    optimizer_gc = BayesianOptimization(f=loss_partial_gc_x0_max,pbounds=pbounds,random_state=1)
    optimizer_gc.maximize(init_points=5,n_iter=max_function_evaluations)
    sol_gc = optimizer_gc.max['params']['x']
    print('Lorentz Optimization')
    time0_l = time()
    optimizer_lorentz = BayesianOptimization(f=loss_partial_lorentz_x0_max,pbounds=pbounds,random_state=1)
    optimizer_lorentz.maximize(init_points=5,n_iter=max_function_evaluations)
    sol_lorentz = optimizer_lorentz.max['params']['x']
else:
    print('Guiding Center Optimization')
    time0_gc = time()
    if method == 'least_squares':
        res_gc = least_squares(loss_partial_gc_x0_min, x0=x0, verbose=2, ftol=1e-5, max_nfev=max_function_evaluations)#, diff_step=1e-4)
        print('Lorentz Optimization');time0_l = time()
        res_lorentz = least_squares(loss_partial_lorentz_x0_min, x0=x0, verbose=2, ftol=1e-5, max_nfev=max_function_evaluations)#, diff_step=1e-4)
    else:
        res_gc = minimize(loss_partial_gc_x0_min, x0=x0, method=method, options={'disp': True, 'maxiter':10, 'maxfun':max_function_evaluations, 'gtol':1e-5})
        print('Lorentz Optimization');time0_l = time()
        res_lorentz = minimize(loss_partial_lorentz_x0_min, x0=x0, method=method, options={'disp': True, 'maxiter':10, 'maxfun':max_function_evaluations, 'gtol':1e-5})
    sol_gc = res_gc.x
    sol_lorentz = res_lorentz.x
print(f'  Time to optimize Guiding Center with {method} optimization: {time0_l-time0_gc:.2f} seconds with x={sol_gc}')
print(f'  Time to optimize Lorentz with {method} optimization: {time()-time0_l:.2f} seconds with x={sol_lorentz}')

dofs_array = jnp.linspace(min(min(sol_lorentz[0]*0.95,xmin),sol_gc[0]*0.95),max(max(sol_lorentz[0]*1.03,xmax),sol_gc[0]*1.03),20)
plt.figure()
plt.axvline(x=x0, linestyle='--', color='k', linewidth=2, label='Initial Guess')
plt.plot(dofs_array, [loss_partial_lorentz_x0_min([x]) for x in tqdm(dofs_array)], color='r', label='Lorentz')
plt.axvline(x=sol_lorentz, linestyle='--', color='r', linewidth=2, label='Lorentz Optimum')
plt.plot(dofs_array, [loss_partial_gc_x0_min([x]) for x in tqdm(dofs_array)], color='b', label='Guiding Center')
plt.axvline(x=sol_gc, linestyle='--', color='b', linewidth=2, label='Guiding Center Optimum')
plt.xlabel("dof")
plt.ylabel("Loss function")
plt.legend()
plt.show()
