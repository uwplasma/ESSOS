
import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_optimize_coils_for_particle_confinement

# Optimization parameters
target_B_on_axis = 5.7
max_coil_length = 31
max_coil_curvature = 0.4
nparticles = number_of_processors_to_use*1
order_Fourier_series_coils = 4
number_coil_points = 80
maximum_function_evaluations = 10
maxtime_tracing = 1e-6
number_coils_per_half_field_period = 3
number_of_field_periods = 2
model = 'FullOrbit_Boris'
timesteps = 3000#int(3*maxtime_tracing/1e-8)

nparticles_plot = number_of_processors_to_use*2
model_plot = 'GuidingCenterAdaptative'
timesteps_plot = 10000
maxtime_tracing_plot = 3e-5

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
particles.to_full_orbit(BiotSavart(coils_initial))
tracing_initial = Tracing(field=coils_initial, particles=particles, maxtime=maxtime_tracing, model=model, times_to_trace=timesteps)

# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations and maxtime_tracing={maxtime_tracing}')
time0 = time()
coils_optimized = optimize_loss_function(loss_optimize_coils_for_particle_confinement, initial_dofs=coils_initial.x,
                           coils=coils_initial, tolerance_optimization=1e-4, particles=particles,
                           maximum_function_evaluations=maximum_function_evaluations, max_coil_curvature=max_coil_curvature,
                           target_B_on_axis=target_B_on_axis, max_coil_length=max_coil_length, model=model,
                           maxtime=maxtime_tracing, num_steps=timesteps)
print(f"  Optimization took {time()-time0:.2f} seconds")
particles.to_full_orbit(BiotSavart(coils_optimized))

phi_array_plot = jnp.linspace(0, 2*jnp.pi, nparticles_plot)
initial_xyz_plot=jnp.array([major_radius_coils*jnp.cos(phi_array_plot), major_radius_coils*jnp.sin(phi_array_plot), 0*phi_array_plot]).T
particles_plot = Particles(initial_xyz=initial_xyz_plot)
particles.to_full_orbit(BiotSavart(coils_optimized))
tracing_optimized = Tracing(field=coils_optimized, particles=particles, maxtime=maxtime_tracing_plot, model=model_plot, times_to_trace=timesteps_plot)

# Plot trajectories, before and after optimization
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

coils_initial.plot(ax=ax1, show=False)
tracing_initial.plot(ax=ax1, show=False)
for i, trajectory in enumerate(tracing_initial.trajectories):
    ax3.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}', linewidth=0.2)
ax3.set_xlabel('R (m)');ax3.set_ylabel('Z (m)');#ax3.legend()
coils_optimized.plot(ax=ax2, show=False)
tracing_optimized.plot(ax=ax2, show=False)
# for i, trajectory in enumerate(tracing_optimized.trajectories):
#     ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}', linewidth=0.2)
# ax4.set_xlabel('R (m)');ax4.set_ylabel('Z (m)');#ax4.legend()
plotting_data = tracing_optimized.poincare_plot(ax=ax4, shifts = [jnp.pi/4, jnp.pi/2, 3*jnp.pi/4], show=False)
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