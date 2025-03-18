
import os
number_of_processors_to_use = 8 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.dynamics import Particles, Tracing
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.optimization import optimize_coils_for_particle_confinement

# Optimization parameters
target_B_on_axis = 5.7
max_coil_length = 31
max_coil_curvature = 0.4
nparticles = 8
order_Fourier_series_coils = 4
number_coil_points = 80
maximum_function_evaluations = 29
maxtime_tracing = 2e-5
number_coils_per_half_field_period = 3
number_of_field_periods = 2
model = 'GuidingCenter'

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
coils_optimized = optimize_coils_for_particle_confinement(coils_initial, particles, target_B_on_axis=target_B_on_axis, maxtime=maxtime_tracing, model=model,
                                        max_coil_length=max_coil_length, maximum_function_evaluations=maximum_function_evaluations, max_coil_curvature=max_coil_curvature)
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