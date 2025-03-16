from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.dynamics import Tracing
from essos.optimization import optimize_coils_and_nearaxis, optimize_coils_for_nearaxis

# Optimization parameters
max_coil_length = 5.0
max_coil_curvature = 4
order_Fourier_series_coils = 5
number_coil_points = order_Fourier_series_coils*10
maximum_function_evaluations = 100
number_coils_per_half_field_period = 3
tolerance_optimization = 1e-8

# Initialize Near-Axis field
rc=jnp.array([1, 0.045])
zs=jnp.array([0,-0.045])
etabar=-0.9
nfp=3
field_nearaxis_initial = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=nfp)

# Initialize coils
current_on_each_coil = 17e5*field_nearaxis_initial.B0/nfp/2
number_of_field_periods = nfp
major_radius_coils = field_nearaxis_initial.R0[0]
minor_radius_coils = major_radius_coils/1.5
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Optimize coils
print(f'Optimizing coils for initial near=axis with {maximum_function_evaluations} function evaluations.')
time0 = time()
coils_optimized_initial_nearaxis = optimize_coils_for_nearaxis(field_nearaxis_initial, coils_initial, maximum_function_evaluations=maximum_function_evaluations,
                                                    max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,
                                                    tolerance_optimization=tolerance_optimization)
print(f"Optimization took {time()-time0:.2f} seconds")

# Optimize coils
print(f'Optimizing coils and near-axis with {maximum_function_evaluations} function evaluations.')
time0 = time()
coils_optimized, field_nearaxis_optimized = optimize_coils_and_nearaxis(field_nearaxis_initial, coils_initial, maximum_function_evaluations=maximum_function_evaluations,
                                                    max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,
                                                    tolerance_optimization=tolerance_optimization)
print(f"Optimization took {time()-time0:.2f} seconds")

print(f'############################################')
print(f'Iota for initial near-axis: {field_nearaxis_initial.iota}')
print(f'Iota for optimized near-axis: {field_nearaxis_optimized.iota}')
print(f'Maximum elongation for initial near-axis: {max(field_nearaxis_initial.elongation)}')
print(f'Maximum elongation for optimized near-axis: {max(field_nearaxis_optimized.elongation)}')

# Trace fieldlines
nfieldlines = 6
num_steps = 1000
tmax = 150
trace_tolerance = 1e-7

R0_initial   = jnp.linspace(field_nearaxis_initial.R0[0],   1.05*field_nearaxis_initial.R0[0],   nfieldlines)
R0_optimized = jnp.linspace(field_nearaxis_optimized.R0[0], 1.05*field_nearaxis_optimized.R0[0], nfieldlines)
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz_initial   = jnp.array([R0_initial*jnp.cos(phi0),   R0_initial*jnp.sin(phi0),   Z0]).T
initial_xyz_optimized = jnp.array([R0_optimized*jnp.cos(phi0), R0_optimized*jnp.sin(phi0), Z0]).T

time0 = time()
tracing_initial = Tracing(field=BiotSavart(coils_optimized_initial_nearaxis), model='FieldLine', initial_conditions=initial_xyz_initial,
                  maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
tracing_optimized = Tracing(field=BiotSavart(coils_optimized), model='FieldLine', initial_conditions=initial_xyz_optimized,
                  maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
print(f"Tracing took {time()-time0:.2f} seconds")

# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils_optimized_initial_nearaxis.plot(ax=ax1, show=False)
field_nearaxis_initial.plot(ax=ax1, show=False, alpha=0.1)
tracing_initial.plot(ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False)
field_nearaxis_optimized.plot(ax=ax2, show=False, alpha=0.1)
tracing_optimized.plot(ax=ax2, show=False)
plt.show()

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# coils_optimized_initial_nearaxis.to_vtk('coils_initial')
# coils_optimized.to_vtk('coils_optimized')
# tracing_initial.to_vtk('trajectories_initial')
# tracing_optimized.to_vtk('trajectories_final')