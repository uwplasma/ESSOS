import os
number_of_processors_to_use = 8 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.dynamics import Tracing
from essos.optimization import optimize_coils_for_nearaxis

# Optimization parameters
max_coil_length = 3.5
max_coil_curvature = 3
order_Fourier_series_coils = 6
number_coil_points = order_Fourier_series_coils*10
maximum_function_evaluations = 10
number_coils_per_half_field_period = 3
tolerance_optimization = 1e-8

# Initialize Near-Axis field
rc=jnp.array([1, 0.1])
zs=jnp.array([0, 0.1])
etabar=1.0
nfp=2
start_time = time()
field = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=nfp)
print(f"Initialization of the field took {time()-start_time:.2f} seconds")

# Initialize coils
current_on_each_coil = 1e5
number_of_field_periods = nfp
major_radius_coils = rc[0]
minor_radius_coils = major_radius_coils/2
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Optimize coils
print('');print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
time0 = time()
coils_optimized = optimize_coils_for_nearaxis(field, coils_initial, maximum_function_evaluations=maximum_function_evaluations,
                                                    max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,
                                                    tolerance_optimization=tolerance_optimization)
print(f"Optimization took {time()-time0:.2f} seconds")

# Trace fieldlines
nfieldlines = 8
num_steps = 1500
tmax = 150
trace_tolerance = 1e-7

R0 = jnp.linspace(rc[0], rc[0]+2*rc[1], nfieldlines)
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T

time0 = time()
tracing_initial = Tracing(field=BiotSavart(coils_initial), model='FieldLine', initial_conditions=initial_xyz,
                  maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
tracing_optimized = Tracing(field=BiotSavart(coils_optimized), model='FieldLine', initial_conditions=initial_xyz,
                  maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
print(f"Tracing took {time()-time0:.2f} seconds")

# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils_initial.plot(ax=ax1, show=False)
field.plot(ax=ax1, show=False)
tracing_initial.plot(ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False)
field.plot(ax=ax2, show=False)
tracing_optimized.plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# coils_initial.to_vtk('coils_initial')
# coils_optimized.to_vtk('coils_optimized')