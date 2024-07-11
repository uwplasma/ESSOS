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
number_of_particles_per_core = 2
#### Some other imports
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.append("..")
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm
#### Input parameters continue here
n_curves=3
nfp=4
order=3
r = 2
A = 3. # Aspect ratio
R = A*r
r_init = r/4
maxtime = 1.0e-5
timesteps=1000
nparticles = len(jax.devices())*number_of_particles_per_core
n_segments=100
coil_current = 1e7
change_currents = False
model = 'Guiding Center' # 'Guiding Center' or 'Lorentz'
method = 'least_squares' # 'Bayesian', 'BOBYQA', 'least_squares' or one of scipy.optimize.minimize methods such as 'BFGS'
max_function_evaluations = 30
min_val = -11 # minimum coil dof value
max_val =  15 # maximum coil dof value
max_function_evaluations_BOBYQA = 150
##### Input parameters stop here
particles = Particles(nparticles)
curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

# If there is a previous optimization, use the following x to set the dofs and currents
# x = [5.761005018206357, -0.03217874387454478, 2.1523026880117087, -0.056394290858425, -0.02670514231469476, 0.015223740575546827, 0.016962232236878434, 0.011142162350601857, 0.03220289079562492, 1.0679336076895585, 0.3346908609647495, -0.05220035728901279, -0.0901512362478669, -0.18614321908151527, -0.024297725459829677, 0.027173903654063306, -0.01860025439798505, 0.03669747690774177, 0.1879019266794462, -2.091736886412198, -0.05392483502340483, 0.08912911428267228, -0.10111848598983585, 0.04629800524586173, 0.010678637591110996, 0.06204574085058528, 0.0044765973154110615, 5.74960658184721, -0.054461086556134575, 1.8727504160944906, -0.06794331153376382, 0.11602416578531412, 0.0038426400254395043, 0.06703670114570646, 0.03590987831645892, -0.001442262509199517, 2.242105275434312, 0.06102291717623049, 0.3333577991752048, -0.06426005433019982, 0.020420214260041033, 0.004396710432917432, 0.04773346847156976, 0.003605034838076655, 0.018344412877616446, 0.18006068264684005, -1.7205651824068793, -0.02865963818533225, -0.08910711847676438, -0.11783143645339124, -0.08206090969971559, -0.012962330585414388, 0.03714291494689949, 0.03280914892389674, 5.2190719441478794, -0.30416794091004506, 1.5010103887299697, -0.11391829312212226, 0.087148139491984, -0.038617224358240804, 0.010769947460525826, 0.02889572569755693, 0.03930168791321329, 3.6790278529819664, 0.39266418087374577, 1.615032428279713, 0.23486110051072231, 0.09094227716074844, 0.11533185302232794, -0.001191251536658603, -0.0026942433010710472, 0.06803584521313358, -0.02154419964486222, -2.2717082398949286, 0.08685187920776022, -0.09525254803135513, 0.19037270526124922, 0.01939872750376672, 0.08289485318647245, -0.017833863082218595, 0.03517967586686895]
# len_dofs = len(jnp.ravel(stel.dofs))
# dofs = jnp.reshape(jnp.array(x)[:len_dofs], shape=stel.dofs.shape)
# stel.dofs = dofs
# if len(x)>len_dofs:
#     print("Setting currents")
#     currents = jnp.array(x)[len_dofs:]
#     stel.currents = currents

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
print(f"Time to trace initial trajectories: {time()-time0:.2f} seconds")

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
print(f"Time to trace final trajectories: {time()-time0:.2f} seconds")

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

