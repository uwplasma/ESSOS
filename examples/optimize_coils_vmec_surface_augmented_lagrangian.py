import os
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.surfaces import BdotN_over_B
from essos.coils import Coils, CreateEquallySpacedCurves,Curves
from essos.fields import Vmec, BiotSavart
from essos.objective_functions import loss_BdotN_only_constraint,loss_coil_curvature_new,loss_coil_length_new,loss_BdotN_only
from essos.objective_functions import loss_coil_curvature,loss_coil_length
from essos.objective_functions import loss_BdotN
from essos.optimization import optimize_loss_function

import essos.augmented_lagrangian as alm
from functools import partial

# Optimization parameters
maximum_function_evaluations=10
max_coil_length = 40
max_coil_curvature = 0.5
bdotn_tol=1.e-6
order_Fourier_series_coils = 6
number_coil_points = order_Fourier_series_coils*10
number_coils_per_half_field_period = 4
ntheta=32
nphi=32
#Tolerance for no normal (no ALM) optimization
tolerance_optimization = 1e-5

# Initialize VMEC field
vmec = Vmec(os.path.join(os.path.dirname(__name__), 'input_files',
             'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
            ntheta=ntheta, nphi=nphi, range_torus='half period')

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

len_dofs_curves = len(jnp.ravel(coils_initial.dofs_curves))
nfp = coils_initial.nfp
stellsym = coils_initial.stellsym
n_segments = coils_initial.n_segments
dofs_curves = coils_initial.dofs_curves
currents_scale = coils_initial.currents_scale
dofs_curves_shape = coils_initial.dofs_curves.shape




# Create the constraints
penalty = 0.1 #Intial penalty values
multiplier=0.5 #Initial lagrange multiplier values
sq_grad=0.0   #Initial square gradient parameter value for Mu adaptative
model_lagrangian='Standard'  #Use standard augmented lagragian suitable for bounded optimizers 
#Since we are using LBFGS-B from jaxopt, model_mu will be updated with tolerances so we do not need to difinte the model


curvature_partial=partial(loss_coil_curvature, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,max_coil_curvature=max_coil_curvature)
length_partial=partial(loss_coil_length, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,max_coil_length=max_coil_length)
bdotn_partial=partial(loss_BdotN_only_constraint, vmec=vmec, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp,n_segments=n_segments, stellsym=stellsym,target_tol=bdotn_tol)
bdotn_only_partial=partial(loss_BdotN_only, vmec=vmec, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp,n_segments=n_segments, stellsym=stellsym)

#Construct constraints
constraints = alm.combine(
alm.eq(curvature_partial,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),
alm.eq(length_partial,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),
alm.eq(bdotn_partial,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad)
)



beta=2.                                     #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-7    #desired contraint tolerance, associated with variation of contraints



#If loss=cost_function(x) is not prescribed, f(x)=0 is considered
ALM=alm.ALM_model_jaxopt_lbfgsb(constraints,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)

#Initializing lagrange multipliers
lagrange_params=constraints.init(coils_initial.x)
#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
params = coils_initial.x, lagrange_params
#This is just to initialize an empty state for the lagrange multiplier update and get some information
lag_state,grad,info=ALM.init(params)

#Initializing first tolerances for the inner minimisation loop iteration
mu_average=alm.penalty_average(lagrange_params)
omega=1./mu_average
eta=1./mu_average**0.1





# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations no ALM.')
time0 = time()
coils_optimized = optimize_loss_function(loss_BdotN, initial_dofs=coils_initial.x, coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, vmec=vmec,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)
print(f"Optimization took {time()-time0:.2f} seconds")





# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations using ALM.')
time0 = time()


i=0
while i<=maximum_function_evaluations and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info,eta,omega = ALM.update(params,lag_state,grad,info,eta,omega)    
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
    print('lagrange',params[1])
    i=i+1



dofs_curves = jnp.reshape(params[0][:len_dofs_curves], (dofs_curves_shape))
dofs_currents = params[0][len_dofs_curves:]
curves = Curves(dofs_curves, n_segments, nfp, stellsym)
coils_optimized_alm = Coils(curves=curves, currents=dofs_currents*coils_initial.currents_scale)

print(f"Optimization took {time()-time0:.2f} seconds")


BdotN_over_B_initial = BdotN_over_B(vmec.surface, BiotSavart(coils_initial))
BdotN_over_B_optimized = BdotN_over_B(vmec.surface, BiotSavart(coils_optimized))
curvature=jnp.mean(BiotSavart(coils_optimized).coils.curvature, axis=1)
length=jnp.max(jnp.ravel(BiotSavart(coils_optimized).coils.length))
BdotN_over_B_optimized_alm = BdotN_over_B(vmec.surface, BiotSavart(coils_optimized_alm))
curvature_alm=jnp.mean(BiotSavart(coils_optimized_alm).coils.curvature, axis=1)
length_alm=jnp.max(jnp.ravel(BiotSavart(coils_optimized_alm).coils.length))


print(f"Maximum allowed curvature target: ",max_coil_curvature)
print(f"Maximum allowed length target: ",max_coil_length)
print(f"Mean curvature without ALM: ",curvature)
print(f"Length withou ALM:", length)
print(f"Mean curvature with ALM: ",curvature_alm)
print(f"Length with ALM:", length_alm)
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization without ALM: {jnp.max(BdotN_over_B_optimized):.2e}")
print(f"Maximum BdotN/B after optimization with ALM: {jnp.max(BdotN_over_B_optimized_alm):.2e}")
# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils_initial.plot(ax=ax1, show=False)
vmec.surface.plot(ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False, label='Optimized no ALM')
coils_optimized_alm.plot(ax=ax2, show=False,color='orange', label='Optimized with ALM')
vmec.surface.plot(ax=ax2, show=False)
plt.legend()
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