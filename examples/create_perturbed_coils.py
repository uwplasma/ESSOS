
import os
number_of_processors_to_use = 8 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves,Curves
from functools import partial
from essos.coil_perturbation import GaussianSampler
from essos.coil_perturbation import perturb_curves_statistic,perturb_curves_systematic




# Coils parameters
order_Fourier_series_coils = 4
number_coil_points = 80
number_coils_per_half_field_period = 3
number_of_field_periods = 2

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



g=GaussianSampler(coils_initial.quadpoints,sigma=0.2,length_scale=0.1,n_derivs=2)

#Split the key for reproducibility  
key=0
split_keys=jax.random.split(jax.random.key(key), num=2)
#Add systematic error
coils_sys = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)
perturb_curves_systematic(coils_sys, g, key=split_keys[0])
# Add statistical error
coils_stat = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)
perturb_curves_statistic(coils_stat, g, key=split_keys[1])
# Add both systematic and statistical errors
coils_perturbed = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)
perturb_curves_systematic(coils_perturbed, g, key=split_keys[0])
perturb_curves_statistic(coils_perturbed, g, key=split_keys[1])


fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(111, projection='3d')
coils_initial.plot(ax=ax1, show=False,color='brown',linewidth=1,label='Initial coils')
coils_sys.plot(ax=ax1, show=False,color='blue',linewidth=1,label='Systematic perturbation')
coils_stat.plot(ax=ax1, show=False,color='green',linewidth=1,label='Statistical perturbation')
coils_perturbed.plot(ax=ax1, show=False,color='magenta',linewidth=1,label='Perturbed coils')
plt.legend()
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
