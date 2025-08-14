
import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from jax import jit, value_and_grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.dynamics import Particles, Tracing
from essos.coils import Coils, CreateEquallySpacedCurves,Curves
from essos.objective_functions import loss_particle_r_cross_max
from essos.objective_functions import loss_coil_curvature,loss_coil_length, loss_normB_axis_average
from functools import partial
import optax


# Optimization parameters
target_B_on_axis = 5.7
max_coil_length = 31
max_coil_curvature = 0.4
nparticles = number_of_processors_to_use*1
order_Fourier_series_coils = 4
number_coil_points = 80
maximum_function_evaluations = 3
maxtimes = [2.e-5]
num_steps=100
number_coils_per_half_field_period = 3
number_of_field_periods = 2
model = 'GuidingCenterAdaptative'

# Initialize coils
current_on_each_coil = 1.84e7
major_radius_coils = 7.75
minor_radius_coils = 4.45
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

len_dofs_curves = len(jnp.ravel(coils_initial.dofs_curves))
nfp = coils_initial.nfp
stellsym = coils_initial.stellsym
n_segments = coils_initial.n_segments
dofs_curves_shape = coils_initial.dofs_curves.shape
currents_scale = coils_initial.currents_scale

# Initialize particles
phi_array = jnp.linspace(0, 2*jnp.pi, nparticles)
initial_xyz=jnp.array([major_radius_coils*jnp.cos(phi_array), major_radius_coils*jnp.sin(phi_array), 0*phi_array]).T
particles = Particles(initial_xyz=initial_xyz)

t=maxtimes[0]

curvature_partial=partial(loss_coil_curvature, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,max_coil_curvature=max_coil_curvature)
length_partial=partial(loss_coil_length, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,max_coil_length=max_coil_length)
Baxis_average_partial=partial(loss_normB_axis_average,dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,npoints=15,target_B_on_axis=target_B_on_axis)
r_max_partial = partial(loss_particle_r_cross_max, particles=particles,dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,maxtime=t,model = model,num_steps=num_steps)
def total_loss(params):
    return jnp.linalg.norm(curvature_partial(params)+length_partial(params)+Baxis_average_partial(params))**2

params=coils_initial.x
optimizer=optax.lbfgs()
opt_state=optimizer.init(params)

@jit
def update(params,opt_state):
    value, grad = value_and_grad(total_loss)(params)        
    updates, opt_state =optimizer.update(grad, opt_state, params, value=value, grad=grad, value_fn=total_loss)
    params = optax.apply_updates(params, updates)
    return params,opt_state

for i in range(maximum_function_evaluations):
    params,opt_state=update(params,opt_state)
    if i % 3 == 0:
        print('Objective function at iteration {:d}: {:.2E}'.format(i, total_loss(params)))

dofs_curves = jnp.reshape(params[:len_dofs_curves], (dofs_curves_shape))
dofs_currents = params[len_dofs_curves:]
curves = Curves(dofs_curves, n_segments, nfp, stellsym)
new_coils = Coils(curves=curves, currents=dofs_currents*coils_initial.currents_scale)
params=new_coils.x
tracing_initial = Tracing(field=coils_initial, particles=particles, maxtime=t, model=model
                ,times_to_trace=200,timestep=1.e-8,atol=1.e-5,rtol=1.e-5)
tracing_optimized = Tracing(field=new_coils, particles=particles, maxtime=t, model=model,times_to_trace=200,timestep=1.e-8,atol=1.e-5,rtol=1.e-5)


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

ax3.set_xlabel('R (m)')
ax3.set_ylabel('Z (m)')
#ax3.legend()
new_coils.plot(ax=ax2, show=False)
tracing_optimized.plot(ax=ax2, show=False)
for i, trajectory in enumerate(tracing_optimized.trajectories):
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
ax4.set_xlabel('R (m)')
ax4.set_ylabel('Z (m)')#ax4.legend()
plt.tight_layout()
# plt.savefig(f'opt_lbfgs.pdf')
plt.show()

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# tracing_initial.to_vtk('trajectories_initial')
#tracing_optimized.to_vtk('trajectories_final')
#coils_initial.to_vtk('coils_initial')
#new_coils.to_vtk('coils_optimized')


