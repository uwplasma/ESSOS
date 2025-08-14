import os
number_of_processors_to_use = 5 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.surfaces import BdotN_over_B
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec, BiotSavart
from essos.objective_functions import loss_BdotN
from essos.optimization import optimize_loss_function

# Optimization parameters
max_coil_length = 10
max_coil_curvature = 1.0
order_Fourier_series_coils = 3
number_coil_points = order_Fourier_series_coils*15
maximum_function_evaluations = 50
number_coils_per_half_field_period = 3
tolerance_optimization = 1e-5
ntheta=35
nphi=35

# Initialize VMEC field
vmec = Vmec(os.path.join(os.path.dirname(__file__), 'input_files',
             'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
            ntheta=ntheta, nphi=nphi, range_torus='half period')

# Initialize coils
current_on_each_coil = 1
number_of_field_periods = vmec.nfp
major_radius_coils = vmec.r_axis
minor_radius_coils = vmec.r_axis/1.8
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
time0 = time()
coils_optimized = optimize_loss_function(loss_BdotN, initial_dofs=coils_initial.x, coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, vmec=vmec,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)
print(f"Optimization took {time()-time0:.2f} seconds")

BdotN_over_B_initial = BdotN_over_B(vmec.surface, BiotSavart(coils_initial))
BdotN_over_B_optimized = BdotN_over_B(vmec.surface, BiotSavart(coils_optimized))
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")

# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils_initial.plot(ax=ax1, show=False)
vmec.surface.plot(ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False)
vmec.surface.plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# from essos.fields import BiotSavart
# vmec.surface.to_vtk('surface_initial', field=BiotSavart(coils_initial))
# vmec.surface.to_vtk('surface_final',   field=BiotSavart(coils_optimized))
# coils_initial.to_vtk('coils_initial')
# coils_optimized.to_vtk('coils_optimized')