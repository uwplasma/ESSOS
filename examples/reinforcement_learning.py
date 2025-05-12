
import os
number_of_processors_to_use = 6 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.linen as nn
from typing import Sequence
from optax import adam, apply_updates
from time import time
from jax import jit, grad
from functools import partial
from scipy.optimize import minimize
from jax import random, jit, value_and_grad
from essos.dynamics import Particles, Tracing
from essos.coils import Curves, Coils, CreateEquallySpacedCurves
from essos.objective_functions import loss_optimize_coils_for_particle_confinement

# Optimization parameters
target_B_on_axis = 5.7
max_coil_length = 31
max_coil_curvature = 0.4
nparticles = number_of_processors_to_use
order_Fourier_series_coils = 4
number_coil_points = 80
maximum_function_evaluations = 29
maxtime_tracing = 1e-5
number_coils_per_half_field_period = 3
number_of_field_periods = 2
model = 'GuidingCenter'
tolerance_optimization=1e-4
num_steps = 500
trace_tolerance = 1e-5
method='L-BFGS-B'

# Initialize coils
current_on_each_coil = 1.84e7
major_radius_coils = 7.75
minor_radius_coils = 4.5
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Initialize particles
phi_array = jnp.linspace(0, 2*jnp.pi, nparticles)
initial_xyz=jnp.array([major_radius_coils*jnp.cos(phi_array), major_radius_coils*jnp.sin(phi_array), 0*phi_array]).T
particles = Particles(initial_xyz=initial_xyz)
tracing_initial = Tracing(field=coils_initial, particles=particles, maxtime=maxtime_tracing, model=model)

# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations and maxtime_tracing={maxtime_tracing}')
time0 = time()

initial_dofs=coils_initial.x
coils=coils_initial
len_dofs_curves = len(jnp.ravel(coils.dofs_curves))
nfp = coils.nfp
stellsym = coils.stellsym
n_segments = coils.n_segments
dofs_curves_shape = coils.dofs_curves.shape
currents_scale = coils.currents_scale
func = partial(loss_optimize_coils_for_particle_confinement, max_coil_curvature=max_coil_curvature, particles=particles,
                                                 n_segments=n_segments, stellsym=stellsym, target_B_on_axis=target_B_on_axis,
                                                 max_coil_length=max_coil_length, num_steps=num_steps, trace_tolerance=trace_tolerance, model=model)
loss_partial = partial(func, dofs_curves=coils.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym)

# fraction = step / num_steps
# new_maxtime = maxtime_tracing * (1 + 9 * fraction)  # From 1x to 10x

f = partial(loss_partial, maxtime=maxtime_tracing)


# --- Policy network ---
class Policy(nn.Module):
    action_dim: int
    hidden_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        return mean, log_std

# --- Sample action from policy ---
def sample_action(params, key, model):
    mean, log_std = model.apply(params, initial_dofs)
    std = jnp.exp(log_std)
    action = mean + std * random.normal(key, mean.shape)
    return action

# --- Reward function ---
def compute_reward(x):
    fx = f(x)
    if fx.ndim == 1:
        loss = jnp.sum(fx ** 2)
    else:
        loss = fx  # assume already scalar
    reward = -loss
    return reward

# --- Policy gradient loss ---
def loss_fn(params, key, model):
    key, subkey = random.split(key)
    action = sample_action(params, subkey, model)
    reward = compute_reward(action)
    log_prob = -0.5 * jnp.sum((action - model.apply(params, subkey)[0]) ** 2)  # Gaussian log-prob approx
    return -(reward * log_prob)  # Negative because we minimize loss

# --- Training step ---
@jit
def train_step(opt_state, params, key, model, optimizer):
    loss, grads = value_and_grad(loss_fn)(params, key, model)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = apply_updates(params, updates)
    return params, opt_state, loss

# --- Main training loop ---
def train(seed=0, num_steps=1000, lr=1e-3, action_dim=5):
    key = random.PRNGKey(seed)
    model = Policy(action_dim)
    params = model.init(key, initial_dofs)
    optimizer = adam(lr)
    opt_state = optimizer.init(params)

    for step in range(num_steps):
        key, subkey = random.split(key)
        params, opt_state, loss = train_step(opt_state, params, subkey, model, optimizer)
        if step % 100 == 0:
            print(f"Step {step}, Loss {loss:.6f}")

    return params, model

trained_params, policy_model = train(action_dim=5)
key = random.PRNGKey(42)
final_action = sample_action(trained_params, key, policy_model)
final_x = final_action


# jac_loss_partial = jit(grad(loss_partial))
# result = minimize(loss_partial, x0=initial_dofs, jac=jac_loss_partial, method=method,
#                     tol=tolerance_optimization, options={'maxiter': maximum_function_evaluations, 'disp': True, 'gtol': 1e-14, 'ftol': 1e-14})
# final_x = result.x

dofs_curves = jnp.reshape(final_x[:len_dofs_curves], (dofs_curves_shape))
dofs_currents = final_x[len_dofs_curves:]
curves = Curves(dofs_curves, n_segments, nfp, stellsym)
new_coils = Coils(curves=curves, currents=dofs_currents*coils.currents_scale)
coils_optimized = new_coils

print(f"  Optimization took {time()-time0:.2f} seconds")
tracing_optimized = Tracing(field=coils_optimized, particles=particles, maxtime=maxtime_tracing, model=model)

# Plot trajectories, before and after optimization
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

coils_initial.plot(ax=ax1, show=False)
tracing_initial.plot(ax=ax1, show=False)
for i, trajectory in enumerate(tracing_initial.trajectories):
    ax3.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
ax3.set_xlabel('R (m)');ax3.set_ylabel('Z (m)');#ax3.legend()
coils_optimized.plot(ax=ax2, show=False)
tracing_optimized.plot(ax=ax2, show=False)
for i, trajectory in enumerate(tracing_optimized.trajectories):
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
ax4.set_xlabel('R (m)');ax4.set_ylabel('Z (m)');#ax4.legend()
plt.tight_layout()
plt.show()

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# tracing_initial.to_vtk('trajectories_initial')
# tracing_optimized.to_vtk('trajectories_final')
# coils_initial.to_vtk('coils_initial')
# coils_optimized.to_vtk('coils_optimized')