import os
from time import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.optimization import optimize_loss_function
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B
from essos.losses import custom_loss, base_loss

import essos.augmented_lagrangian as alm
from functools import partial
#  In this exmple, `scipy.optimize.least_squares` is used for the normal optimization, but any other optimizer, e.g. from
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

# Optimization parameters
maximum_function_evaluations=100

input_filepath = os.path.join(os.path.dirname(__file__), "..", "input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=32, nphi=32, range_torus='half period')


""" Creating starting coils and surface """
N_COILS = 4
FOURIER_ORDER = 3
LARGE_R = 10.
SMALL_R = 5.7
NFP = 2
N_SEGMENTS = 30
STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)


""" Setting the losses weights and targets """
LENGTH_WEIGHT = 1.
LENGTH_TARGET = 40.
CURVATURE_WEIGHT = 1.
CURVATURE_TARGET = 0.5
NORMAL_FIELD_WEIGHT = 1.
BdotN_Target_tol=1.e-6

EXPORT = False

init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)
init_surface=surface


""" Creating the loss functions """
def loss(field, surface):
    return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))

def BdotN_constraint(field,surface,target_tol=1.e-6):
    bdotn_over_b = BdotN_over_B(surface, field)
    bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0)))
    return bdotn_over_b_loss

def loss_length_constraint(field):
    return jnp.maximum(0, field.coils.length - LENGTH_TARGET)

def loss_curvature_contraint(field):
    return jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET)
    #return jnp.mean(jnp.maximum(0, (field.coils.curvature - CURVATURE_TARGET)/CURVATURE_TARGET))

def loss_length(field):
    return jnp.mean(jnp.maximum(0, field.coils.length - LENGTH_TARGET))

def loss_curvature(field):
    return jnp.mean(jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET))


""" Defining custom losses """
L_normal_field = custom_loss(loss, "field", surface=surface)
L_normal_field_constraint = custom_loss(BdotN_constraint, "field", surface=surface)
L_length_constraint = custom_loss(loss_length_constraint, "field")
L_curvature_constraint = custom_loss(loss_curvature_contraint, "field")
L_length = custom_loss(loss_length, "field")
L_curvature = custom_loss(loss_curvature, "field")

""" Defining total loss + setting dependencies """
L_normal_field.dependencies = {"field": init_field}
L_length_constraint.dependencies = {"field": init_field}
L_curvature_constraint.dependencies = {"field": init_field}
L_normal_field_constraint.dependencies = {"field": init_field}
L_length.dependencies = {"field": init_field}
L_curvature.dependencies = {"field": init_field}


# Create the constraints
penalty = 1.0 #Intial penalty values
multiplier=0. #Initial lagrange multiplier values
#Initializing first tolerances for the inner minimisation loop iteration
omega=1./penalty
eta=1./penalty**0.1
sq_grad=0.0   #Initial square gradient parameter value for Mu adaptative
model_lagrangian='Standard'  #Use standard augmented lagragian suitable for bounded optimizers
#Since we are using LBFGS-B from jaxopt, model_mu will be updated with tolerances so we do not need to difinte the model
model_mu='Tolerance'


beta=2.                                     #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-7    #desired contraint tolerance, associated with variation of contraints

#curvature_constraint=alm.ScaledConstraint(alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad))
#length_constraint=alm.ScaledConstraint(alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad))
#field_constraint=alm.ScaledConstraint(alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad))
curvature_constraint=alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad)
length_constraint=alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad)
field_constraint=alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad)


C_normal_field_constraint = alm.SelectiveConstraint(field_constraint, "field", surface=surface, target_tol=BdotN_Target_tol)
C_length_constraint = alm.SelectiveConstraint(length_constraint, "field")
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field")



C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint)


C_normal_field_constraint.dependencies = {"field": init_field}
C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_Total_constraint.dependencies = {"field": init_field}


#If loss=cost_function(x) is not prescribed, f(x)=0 is considered, uncomment second line to use B dot N as a loss and not a constraint
ALM=alm.ALM_model_jaxopt_lbfgsb(constraints=C_Total_constraint,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)

#Initializing lagrange multipliers
lagrange_params=C_Total_constraint.init(init_field.dofs)
#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
params = init_field.dofs, lagrange_params
#This is just to initialize an empty state for the lagrange multiplier update and get some information
lag_state,grad,info=ALM.init(params)


""" Optimizing  with alm"""
t_start = time()

i=0
while i<=maximum_function_evaluations and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info = ALM.update(params,lag_state,grad,info)
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
    #print('lagrange',params[1])
    i=i+1

t_end = time()


opt_field_alm = C_normal_field_constraint.dofs_to_pytree(params[0])[0]
opt_coils_alm = opt_field_alm.coils


""" Defining total loss for nornmal optimization"""
L_total = NORMAL_FIELD_WEIGHT*L_normal_field+ LENGTH_WEIGHT*L_length + CURVATURE_WEIGHT*L_curvature
L_total.dependencies = {"field": init_field}

""" Optimizing the total loss """
t_start = time()
res = least_squares(L_total, L_total.starting_dofs, L_total.grad, verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=maximum_function_evaluations)
t_end = time()

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", L_total(L_total.starting_dofs))
print("Loss after optimization:", L_total(res.x))

opt_field = L_total.dofs_to_pytree(res.x)["field"]
opt_coils = opt_field.coils


print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial B dot N:", jnp.max(BdotN_over_B(surface, init_field)))
print("B dot N after optimization:", jnp.max(BdotN_over_B(surface, opt_field)))
print("B dot N after optimization alm:", jnp.max(BdotN_over_B(surface, opt_field_alm)))
print("Initial curvature :", jnp.average(init_field.coils.curvature,axis=0))
print("Curvature after optimization:",jnp.average(opt_field.coils.curvature,axis=0))
print("Curvature after optimization alm:",jnp.average(opt_field_alm.coils.curvature,axis=0))
print("Curvature target:",CURVATURE_TARGET)
print("Initial length :", init_field.coils.length)
print("Length after optimization:",opt_field.coils.length)
print("Length after optimization alm:",opt_field_alm.coils.length)
print("Length target:",LENGTH_TARGET)



fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(131, projection='3d')
init_coils.plot(ax=ax1, show=False,label='Initial coils')
surface.plot(ax=ax1, show=False)
ax2 = fig.add_subplot(132, projection='3d')
opt_coils.plot(ax=ax2, show=False,label='Standard optimized coils')
surface.plot(ax=ax2, show=False)
ax3 = fig.add_subplot(133, projection='3d')
opt_coils_alm.plot(ax=ax3, show=False,label='ALM optimized coils')
surface.plot(ax=ax3, show=False)
plt.legend()
plt.tight_layout()
plt.show()

if EXPORT:
    output_filepath = os.path.join(os.path.dirname(__file__), "output")

    """ Save the coils to a json file """
    init_coils.to_json(os.path.join(output_filepath, "init_coils_vmec_surface.json"))
    opt_coils.to_json(os.path.join(output_filepath, "opt_coils_vmec_surface.json"))

    """ Save results in vtk format to analyze in Paraview """
    surface.to_vtk(os.path.join(output_filepath, "init_surface_vmec_surface.json"), field=init_field)
    surface.to_vtk(os.path.join(output_filepath, "final_surface_vmec_surface.json"), field=opt_field)
    init_coils.to_vtk(os.path.join(output_filepath, "init_coils_vmec_surface.json"))
    opt_coils.to_vtk(os.path.join(output_filepath, "opt_coils_vmec_surface.json"))
    opt_coils_alm.to_vtk(os.path.join(output_filepath, "opt_coils_alm_vmec_surface.json"))
