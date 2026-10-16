import os
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.optimization import optimize_loss_function
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B, toroidal_flux, B_on_surface
from essos.losses import custom_loss, base_loss

import essos.augmented_lagrangian as alm
from functools import partial
#  In this exmple, `scipy.optimize.least_squares` is used for the normal optimization, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

# Optimization parameters
maximum_function_evaluations=100


input_filepath = os.path.join(os.path.dirname(__name__), "../input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=60, nphi=60, range_torus='half period')


""" Creating starting coils and surface """
N_COILS = 4; FOURIER_ORDER = 6; LARGE_R = 10.; SMALL_R = 5.7; NFP = 2; N_SEGMENTS = 60; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)
init_surface=surface

""" Setting the losses weights and targets """
LENGTH_WEIGHT = 1.; LENGTH_TARGET = 40.
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 0.5
NORMAL_FIELD_WEIGHT = 1.
BdotN_Target_tol=1.e-3

""" Creating the loss functions """
def loss(field, surface):
    return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))

def BdotN_constraint_test(field,surface,target_tol=1.e-6):
    bdotn_over_b = BdotN_over_B(surface, field)
    #bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0))/target_tol)
    #bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0)))
    bdotn_over_b_loss = jnp.abs(bdotn_over_b)-target_tol 
    #bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum((jnp.square(bdotn_over_b)-target_tol)/target_tol,0.0)))
    return bdotn_over_b_loss

def BdotN_constraint(field,surface,target_tol=1.e-6,P=50.):
    bdotn_over_b = BdotN_over_B(surface, field)
    g_local =jnp.abs(bdotn_over_b) - target_tol
    # Ajustamos P para a escala física (ex: se target é 1e-4, P será 200.000)
    P = 20.0 / jnp.maximum(1e-8, target_tol) 
    # LogSumExp estável
    #g_max = (1.0 / P) * jax.scipy.special.logsumexp(P * g_local)
    g_max=(1.0 / P) *  jax.scipy.special.logsumexp(P * g_local) - (jnp.log(bdotn_over_b.size) / P)
    return g_max

def loss_length_constraint(field,N_coils=4):
    #return jnp.maximum(0, field.coils.length - LENGTH_TARGET)
    ##return jnp.maximum(0, (field.coils.length[:N_coils] - LENGTH_TARGET))
    return (field.coils.length[:N_coils] - LENGTH_TARGET)

def loss_curvature_contraint(field,N_coils=4):
    ###return jnp.maximum(0, field.coils.curvature[:N_coils] - CURVATURE_TARGET)
    return  field.coils.curvature[:N_coils] - CURVATURE_TARGET    
    #return jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET)
    #return jnp.mean(jnp.maximum(0, (field.coils.curvature - CURVATURE_TARGET)/CURVATURE_TARGET))

def loss_length(field):
    return jnp.mean(jnp.maximum(0, field.coils.length - LENGTH_TARGET))/LENGTH_TARGET

def loss_curvature(field):
    return jnp.mean(jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET))/CURVATURE_TARGET


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
model_mu='Mu_Adaptative_2'




beta=1.01                                     #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-5    #desired constraint tolerance (base, for internal ALM penalty scheduling)
decrease_tol=0.99

# Per-constraint tolerances: scale eta_tol by target magnitude for consistent meaning
# So "1e-5" means: 1% relative error for each constraint's natural scale
# Example: BdotN target=1e-6, so freeze when violation < 1e-5*1e-6 = 1e-11
#          Length target=40, so freeze when violation < 1e-5*40 = 4e-4
eta_tol_bdotn = eta_tol * BdotN_Target_tol
eta_tol_length = eta_tol * LENGTH_TARGET
eta_tol_curvature = eta_tol * CURVATURE_TARGET

#curvature_constraint=alm.ScaledConstraint(alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad))
#length_constraint=alm.ScaledConstraint(alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad))
#field_constraint=alm.ScaledConstraint(alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,sq_grad=sq_grad))
#curvature_constraint=alm.ScaledConstraint(alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta_tol=eta_tol_curvature),auto_scale=False)
#length_constraint=alm.ScaledConstraint(alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta_tol=eta_tol_length),auto_scale=False)
#field_constraint=alm.ScaledConstraint(alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta_tol=eta_tol_bdotn),auto_scale=False)

curvature_constraint=alm.ScaledConstraint(alm.ineq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta_tol=eta_tol_curvature,use_analytical_penalty=True),auto_scale=False)
length_constraint=alm.ScaledConstraint(alm.ineq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta_tol=eta_tol_length,use_analytical_penalty=True),auto_scale=False)
field_constraint=alm.ScaledConstraint(alm.ineq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta_tol=eta_tol_bdotn,use_analytical_penalty=True),auto_scale=False)








C_normal_field_constraint = alm.SelectiveConstraint(field_constraint, "field", surface=surface, target_tol=BdotN_Target_tol)
C_length_constraint = alm.SelectiveConstraint(length_constraint, "field",N_coils=N_COILS)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field",N_coils=N_COILS)



C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint)






C_normal_field_constraint.dependencies = {"field": init_field}
C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_Total_constraint.dependencies = {"field": init_field} 



    # use_per_constraint_weights=False,  #If False: no weights, 'global': one weight for all constraints, 'constraint': one weight per constraint (fast), 'element': element-wise weights (slow)
    # weight_g_epsilon=1.e-4,          #Epsilon for per-constraint weight computation
    # use_perturbation_weights=False,  #If True: evaluate gradient norms at perturbed parameters to avoid huge weights for satisfied constraints
    # perturbation_magnitude=0.01,     #Relative magnitude of perturbation (e.g., 0.01 = 1%)
    # perturbation_mode='relative',    #Type of perturbation: 'relative' (default) or 'absolute'

#If loss=cost_function(x) is not prescribed, f(x)=0 is considered, uncomment second line to use B dot N as a loss and not a constraint
#ALM=alm.ALM_model_jaxopt_lbfgsb(constraints,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)

ALM=alm.ALM_model_jaxopt_lbfgsb(constraints=C_Total_constraint,model_lagrangian=model_lagrangian,model_mu=model_mu,
beta=beta,mu_max=mu_max,alpha=alpha,
gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,
use_per_constraint_weights=True,weight_g_epsilon=1.e-4,
use_perturbation_weights=True,perturbation_magnitude=0.01,perturbation_mode='relative',
perturbation_seed=30)

#Initializing lagrange multipliers
lagrange_params=C_Total_constraint.init(init_field.dofs)
#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
params = init_field.dofs, lagrange_params
#This is just to initialize an empty state for the lagrange multiplier update and get some information
params,lag_state,grad,info=ALM.init(params)

# Print auto-scaling factors for each constraint
def print_weights():
    print("\n" + "="*70)
    print("AUTO-SCALING FACTORS FOR CONSTRAINTS")
    print("="*70)
    constraint_names = [
        "Coil Length",
        "Coil Curvature",
        "Normal Field (BdotN)",         
    ]
    selective_constraints = [
        C_length_constraint,
        C_curvature_constraint,
        C_normal_field_constraint,
    ]
    for name, sel_constraint in zip(constraint_names, selective_constraints):
        # Access the underlying ScaledConstraint
        scaled_constraint = sel_constraint.constraint
        #print(f"\n{name}:")
        #print(f"  auto_scale: {scaled_constraint.auto_scale}")
        #print(f"  max_scale_factor: {scaled_constraint.max_scale_factor}")
        if hasattr(scaled_constraint, '_initial_scale_scalar'):
            if scaled_constraint._initial_scale_scalar is not None:
                scale = float(scaled_constraint._initial_scale_scalar)
                print(f"{name}:  → scale factor = {scale:.6e} (auto-scaled)")
            else:
                print(f"{name}:  → scale factor = NOT COMPUTED (auto_scale=True but _initial_scale_scalar=None)")
                print(f"{name}:      This usually means auto-scaling evaluation failed!")
        elif hasattr(scaled_constraint, 'scale_factor'):
            print(f"{name}:  → scale factor = {scaled_constraint.scale_factor:.6e} (manual fallback)")
        else:
            print(f"{name}:  → no scaling info available")
    print("="*70)

print_weights()


C_Total_constraint.loss(params[1],params[0])


""" Optimizing with alm"""
t_start = time()

print("\nTarget tolerance scales (for stopping criterion):")
print(f"  Length:    < {eta_tol_length:.2e}")
print(f"  Curvature: < {eta_tol_curvature:.2e}")
print(f"  BdotN:     < {eta_tol_bdotn:.2e}\n")

i=0
while i<=1000 and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info = ALM.update(params,lag_state,grad,info)    
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
    #print('lagrange',params[1])
    i=i+1



t_end = time()


opt_field_alm = L_normal_field_constraint.dofs_to_pytree(params[0])[0]
opt_coils_alm = opt_field_alm.coils





print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("\n=== OPTIMIZATION RESULTS ===")
print("Initial B dot N:", jnp.max(BdotN_over_B(surface, init_field)))    
print("B dot N after optimization:", jnp.max(BdotN_over_B(surface, opt_field_alm)))
print("\nInitial curvature :", jnp.average(init_field.coils.curvature,axis=0))    
print("Curvature after optimization:",jnp.average(opt_field_alm.coils.curvature,axis=0))
print("Curvature target:",CURVATURE_TARGET)
print("\nInitial length :", init_field.coils.length)    
print("Length after optimization:",opt_field_alm.coils.length)
print("Length target:",LENGTH_TARGET)


# Plot coils and surface
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(121, projection='3d')
init_coils.plot(ax=ax1, show=False,label='Initial coils')
surface.plot(ax=ax1, show=False)
ax1.set_title('Initial Configuration')

ax2 = fig.add_subplot(122, projection='3d')
opt_coils_alm.plot(ax=ax2, show=False,label='Optimized coils')
surface.plot(ax=ax2, show=False)
ax2.set_title('ALM Optimized')

plt.legend()
plt.tight_layout()
plt.savefig("coil_optimization_comparison.png", dpi=300)


# Plot |B| on surface
print("\n=== COMPUTING |B| ON SURFACE ===")
B_final = B_on_surface(surface, opt_field_alm)
modB = jnp.linalg.norm(B_final, axis=-1)

print(f"|B| statistics:")
print(f"  min: {float(jnp.min(modB)):.6e} T")
print(f"  max: {float(jnp.max(modB)):.6e} T")
print(f"  mean: {float(jnp.mean(modB)):.6e} T")
print(f"  ripple: {float((jnp.max(modB)-jnp.min(modB))/jnp.mean(modB)*100):.2f}%")

# Create coordinate grids
phi_grid = surface._quadpoints_phi
theta_grid = surface._quadpoints_theta
PHI, THETA = jnp.meshgrid(phi_grid, theta_grid, indexing='ij')

# Plot |B| contours
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Version 1: Using 1D quadpoints with transposed modB
ax = axes[0]
cs1 = ax.contourf(phi_grid, theta_grid, modB.T, levels=30, cmap='viridis')
plt.colorbar(cs1, ax=ax, label='|B| [T]')
ax.set_title('|B| on surface (corrected)')
ax.set_xlabel('φ (toroidal)')
ax.set_ylabel('θ (poloidal)')

# Version 2: Using 2D meshgrid
ax = axes[1]
cs2 = ax.contourf(PHI, THETA, modB, levels=30, cmap='viridis')
plt.colorbar(cs2, ax=ax, label='|B| [T]')
ax.set_title('|B| (meshgrid version)')
ax.set_xlabel('φ')
ax.set_ylabel('θ')

plt.tight_layout()
plt.savefig('modB_optimized.png', dpi=300)
print("✓ Saved: modB_optimized.png")



# ========== SAVE COILS IN MULTIPLE FORMATS ==========
print("\n" + "="*70)
print("SAVING OPTIMIZED COILS")
print("="*70)

# 1. ESSOS JSON format (lightweight, ESSOS-specific)
opt_coils_alm.to_json("opt_coils_essos.json")
print("✓ ESSOS format: opt_coils_essos.json")

# 2. SIMSOPT format (can be loaded directly in SIMSOPT)
try:
    from simsopt._core import save
    from simsopt.field import BiotSavart as BiotSavart_SIMSOPT
    
    simsopt_coils = opt_coils_alm.to_simsopt()
    bs_simsopt = BiotSavart_SIMSOPT(simsopt_coils)
    save(bs_simsopt, 'optimized_coils_simsopt.json')
    print("✓ SIMSOPT format: optimized_coils_simsopt.json")
    print("\n  To use in SIMSOPT:")
    print("    from simsopt import load")
    print("    bs = load('optimized_coils_simsopt.json')")
    print("    coils = bs.coils")
except ImportError:
    print("⚠ SIMSOPT not installed, skipping SIMSOPT format")

# 3. VTK format for visualization (ParaView, VisIt)
opt_coils_alm.to_vtk('coils_optimized')
surface.to_vtk('surface_optimized', field=opt_field_alm)
print("✓ VTK format: coils_optimized.vtu, surface_optimized.vts")

print("="*70)