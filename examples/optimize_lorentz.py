import os
import jax
import sys
from jax import jit
from time import time
import jax.numpy as jnp
from matplotlib import cm
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from bayes_opt import BayesianOptimization
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=14'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.append("..")
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm

n_curves=2
nfp=5
order=1
r = 1.7
A = 6. # Aspect ratio
R = A*r

r_init = r/5
maxtime = 1e-5
timesteps=1000
nparticles = len(jax.devices())*1
n_segments=100

particles = Particles(nparticles)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([3e6]*n_curves))

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

model = 'Lorentz'

start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime, timesteps, n_segments, model=model)
print(f"Loss function initial value: {loss_value:.8f}")
end = time()
print(f"Took: {end-start:.2f} seconds")


loss_partial = partial(loss, dofs_currents=stel.dofs_currents, old_coils=stel,
                       particles=particles, R=R, r_init=r_init,
                       initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]) if model=='Lorentz' else jnp.array([x0, y0, z0, vpar0, vperp0]),
                       maxtime=maxtime, timesteps=timesteps, n_segments=n_segments, model=model)

@jit
def loss_partial_dofs_max(*args, **kwargs):
    x = jnp.array(list(kwargs.values()))
    dofs = jnp.reshape(x, shape=stel.dofs.shape)
    return -loss_partial(dofs)

@jit
def loss_partial_dofs_min(x):
    dofs = jnp.reshape(x, shape=stel.dofs.shape)
    return loss_partial(dofs)

method = 'BFGS'#'Bayesian'# 'BFGS'

all_dofs = jnp.ravel(stel.dofs)
print(f'Number of dofs: {len(all_dofs)}')
if method == 'Bayesian':
    min_val = -3
    max_val = 15
    initial_points = 20
    n_iterations = 20
    pbounds = {}
    for i in range(1, len(all_dofs) + 1):
        param = f'x{i}'
        pbounds[param] = (min_val, max_val)
    optimizer = BayesianOptimization(f=loss_partial_dofs_max,pbounds=pbounds,random_state=1)
    optimizer.maximize(init_points=initial_points,n_iter=n_iterations)
    print(optimizer.max)
    x = jnp.array(list(optimizer.max['params'].values()))
else:
    res_gc = least_squares(loss_partial_dofs_min, x0=all_dofs, verbose=2, ftol=1e-6, max_nfev=30)
    x = jnp.array(res_gc.x)
    
print(f'Resulting dofs: {repr(x.tolist())}')

time0 = time()
if model=='Lorentz':
    trajectories_initial = stel.trace_trajectories_lorentz(particles, initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
else:
    trajectories_initial = stel.trace_trajectories(particles, initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")

dofs = jnp.reshape(x, shape=stel.dofs.shape)
stel.dofs = dofs

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

