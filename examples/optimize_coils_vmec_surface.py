import os
from time import time
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec
from essos.optimization import optimize_coils_for_vmec_surface

# Optimization parameters
target_coil_length = 42
target_coil_curvature = 0.1
order_Fourier_series_coils = 5
number_coil_points = 50
maximum_function_evaluations = 100
number_coils_per_half_field_period = 3
ntheta=30
nphi=30

# Initialize VMEC field
vmec = Vmec(os.path.join(os.path.dirname(__file__), 'input_files',
             'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'),
            ntheta=ntheta, nphi=nphi, range='half period')

# Initialize coils
current_on_each_coil = 1
number_of_field_periods = vmec.nfp
major_radius_coils = vmec.r_axis
minor_radius_coils = vmec.r_axis/1.5
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Optimize coils
print('');print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
time0 = time()
coils_optimized = optimize_coils_for_vmec_surface(vmec, coils_initial, maximum_function_evaluations=maximum_function_evaluations,
                                                    target_coil_length=target_coil_length, target_coil_curvature=target_coil_curvature)
print(f"  Optimization took {time()-time0:.2f} seconds")

## Save results in vtk format to analyze in Paraview
# vmec.surface.to_vtk('surface_initial', field=BiotSavart(coils_initial))
# vmec.surface.to_vtk('surface_final',   field=BiotSavart(coils_optimized))
# coils_initial.to_vtk('coils_initial')
# coils_optimized.to_vtk('coils_optimized')

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