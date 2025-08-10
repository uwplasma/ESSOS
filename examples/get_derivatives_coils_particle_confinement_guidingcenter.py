
import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax
print(jax.devices())
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.dynamics import Particles, Tracing
from essos.coils import Coils, CreateEquallySpacedCurves,Curves
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_particle_r_cross_final_new,loss_particle_r_cross_max,loss_particle_radial_drift,loss_particle_gamma_c
from essos.objective_functions import loss_coil_curvature_new,loss_coil_length_new,loss_normB_axis_new,loss_normB_axis_average
from functools import partial
import optax


# Optimization parameters
target_B_on_axis = 5.7
max_coil_length = 31
max_coil_curvature = 0.4
nparticles = number_of_processors_to_use*10
order_Fourier_series_coils = 4
number_coil_points = 80
maximum_function_evaluations = 301
maxtimes = [2.e-5]
num_steps=100
number_coils_per_half_field_period = 3
number_of_field_periods = 2
model = 'GuidingCenter'

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

curvature_partial=partial(loss_coil_curvature_new, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,max_coil_curvature=max_coil_curvature)
length_partial=partial(loss_coil_length_new, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,max_coil_length=max_coil_length)
Baxis_average_partial=partial(loss_normB_axis_average,dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,npoints=15,target_B_on_axis=target_B_on_axis)
r_max_partial = partial(loss_particle_r_cross_max, particles=particles,dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,maxtime=t,num_steps=num_steps)



def total_loss(params):
    return jnp.sum(jnp.square(r_max_partial(params)+curvature_partial(params)+length_partial(params)+Baxis_average_partial(params)))


params=coils_initial.x
loss=total_loss(params)
gradients=jax.grad(total_loss)(params)

print('Objective function: {:.2E}'.format(loss))
print('Gradients: ',gradients)
