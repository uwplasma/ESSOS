import os
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.optimization import optimize_loss_function
from essos.coils import Coils, CreateEquallySpacedCurves,CreateCoilsAroundAxis
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B,B_on_surface
from essos.losses import custom_loss, base_loss

import essos.augmented_lagrangian as alm
from functools import partial
#  In this exmple, `scipy.optimize.least_squares` is used for the normal optimization, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

# Optimization parameters
maximum_function_evaluations=100
ntheta=60
nphi=60

input_filepath = os.path.join(os.path.dirname(__name__), "../input_files")
vmec_input = os.path.join(input_filepath, 'wout_QH_test.nc')
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='half period')


""" Creating starting coils and surface """
N_COILS = 3; FOURIER_ORDER = 6;  SMALL_R = 0.3; NFP = 4; N_SEGMENTS = 64; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

init_curves = CreateCoilsAroundAxis(n_coils=N_COILS, order=FOURIER_ORDER, coil_radius=SMALL_R, surface=surface, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM,scaling_factor=1.2,scaling_type=jnp.inf,scale_fixed=1.)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)
init_surface=surface

""" Setting the losses weights and targets """
LENGTH_WEIGHT = 1.; LENGTH_TARGET = 3.4
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 5.0
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

ALM = alm.ALM_model_jaxopt_lbfgsb(
    constraints=C_Total_constraint,
    model_lagrangian=model_lagrangian,
    model_mu=model_mu,
    beta=beta,
    mu_max=mu_max,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon,
    eta_tol=eta_tol,
    omega_tol=omega_tol,
    decrease_tol=decrease_tol,
    use_per_constraint_weights='element',
    weight_g_epsilon=1.e-4,
    use_perturbation_weights=True,
    perturbation_magnitude=0.01,
    perturbation_mode='relative',
    perturbation_seed=30,
    use_scaled_main_params=True
)




# Get starting DOFs - this will respect frozen DOFs because init_surface_2 is already frozen
# The starting_dofs combines coils.dofs and surface.dofs (which excludes frozen DOFs)
#starting_dofs = C_Total_constraint.starting_dofs
print("DEBUG: Getting starting_dofs...")
starting_dofs = C_Total_constraint.starting_dofs
print(f"DEBUG: Starting DOFs shape: {starting_dofs.shape}")
print(f"DEBUG: Starting DOFs total size: {starting_dofs.size}")
print(f"DEBUG: Starting DOFs dtype: {starting_dofs.dtype}")

# Break down DOF structure
print("\nDEBUG: DOF Component Breakdown:")
print(f"  Coil DOFs: {init_coils.dofs.shape if hasattr(init_coils, 'dofs') else 'N/A'}")
print(f"  Surface 2 DOFs: {init_surface_2.dofs.shape if hasattr(init_surface_2, 'dofs') else 'N/A'}")
print(f"  Surface 1 DOFs: {init_surface_1.dofs.shape if hasattr(init_surface_1, 'dofs') else 'N/A'}")

print("DEBUG: Getting scaled DOFs...")
starting_dofs_scaled = alm._get_constraints_scaled_dofs(C_Total_constraint)
print(f"DEBUG: Starting scaled DOFs shape: {starting_dofs_scaled.shape}")
print(f"DEBUG: Starting scaled DOFs size: {starting_dofs_scaled.size}")
print(f"DEBUG: Scaled DOF max abs value: {jnp.max(jnp.abs(starting_dofs_scaled))}")
print(f"DEBUG: Scaled DOF min abs value: {jnp.min(jnp.abs(starting_dofs_scaled[starting_dofs_scaled != 0]))}")

#Initializing lagrange multipliers
print("DEBUG: Initializing lagrange multipliers...")
lagrange_params = C_Total_constraint.init(starting_dofs)
print(f"DEBUG: Lagrange params initialized, type: {type(lagrange_params)}")
print(f"DEBUG: Lagrange params length: {len(lagrange_params) if hasattr(lagrange_params, '__len__') else 'N/A'}")
print("DEBUG: About to call print_weights()...")


#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
##params = starting_dofs, lagrange_params
params = starting_dofs_scaled, lagrange_params
#This is just to initialize an empty state for the lagrange multiplier update and get some information
params, lag_state, grad, info = ALM.init(params)
scale = alm._get_constraints_scaling_factors(C_Total_constraint)
C_Total_constraint.loss(params[1],params[0] / jnp.maximum(scale, 1e-12))




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
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2],params[1]):g}')
    #print('lagrange',params[1])
    i=i+1



t_end = time()


filename = f"alm_QH_mn2_ncoils6_op9.pkl"
params, lag_state, grad, eta, omega, info = alm.load_checkpoint_for_reconstruction(
    filename,
    C_Total_constraint  # Your freshly created constraint object
)


filename = f"alm_QH_coils_precise_final.pkl"

alm.save_checkpoint_with_constraints(
    filename, params, lag_state, grad, eta, omega, info, 
    C_Total_constraint, 
    save_constraint_object=False  # Autoscales still saved!
)


opt_field_alm = L_normal_field_constraint.dofs_to_pytree(params[0] / jnp.maximum(scale, 1e-12))[0]
opt_coils_alm = opt_field_alm.coils





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
plt.savefig("coil_optimization_comparison_QH.png", dpi=300)






B_final= B_on_surface(surface, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(surface.phi2d,surface.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('target_modB_QH.png', dpi=300)

surface_final = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full torus')

#init_coils.to_json("init_coils_vmec_surface.json")
opt_coils_alm.to_json( "opt_coils_QH_precise.json")

init_field.coils.to_vtk('coils_initial_QH_precise')
opt_coils_alm.to_vtk('coils_QH_precise')
surface_final.to_vtk('surface_QH_precise',field=opt_field_alm)
surface_final.to_vtk('surface_initial_QH_precise',field=init_field)


import pyvista as pv
coils_init_file='coils_initial_QH_precise.vtu'
coils_opt_file='coils_QH_precise.vtu'

surf_final_file='surface_QH_precise.vts'
surf_initial_file='surface_initial_QH_precise.vts'

# Load the coil and field files
#surf1_mesh = pv.read(surf_1_file)
surf_initial_mesh = pv.read(surf_initial_file)
surf_mesh = pv.read(surf_final_file)
coils_mesh = pv.read(coils_opt_file)
coils_init_mesh = pv.read(coils_init_file)
#Function to use with pyvista plotter to change rcamera rotation based on aximuthal angle and elevation angle
def camera_rotation(pl,azim_angle=0,elev_angle=0):
    cam = pl.camera
    # Load current camera state
    pos = jnp.array(cam.position)
    f = jnp.array(cam.focal_point)
    up = jnp.array(cam.up)
    # Camera vector (from focal point to camera)
    v = pos - f
    r = jnp.linalg.norm(v)
    # Convert to spherical coordinates
    x, y, z = v
    theta = jnp.arctan2(y, x)        # azimuth angle
    phi   = jnp.arccos(z / r)        # polar angle
    # Apply changes (in radians)
    theta += jnp.radians(azim_angle)
    phi   += jnp.radians(elev_angle)
    # Clamp phi to avoid flipping over the poles
    phi = jnp.clip(phi, 1e-3, jnp.pi - 1e-3)
    # Convert back to Cartesian, maintaining SAME radius (zoom)
    vx = r * jnp.sin(phi) * jnp.cos(theta)
    vy = r * jnp.sin(phi) * jnp.sin(theta)
    vz = r * jnp.cos(phi)
    # Update camera
    cam.position = f + jnp.array([vx, vy, vz])
    cam.up = up  # preserve the original up vector




# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf_mesh,scalars='B_BiotSavart',show_scalar_bar=False,cmap='jet')
pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color='silver', line_width=5)
camera_rotation(pl,-100.,elev_angle=110)
pl.render()
#pl.show()
pl.screenshot('coils_QH_opt.png')

# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf_initial_mesh,scalars='B_BiotSavart',show_scalar_bar=False,cmap='jet')
pl.add_mesh(coils_init_mesh,style='wireframe',render_lines_as_tubes=True, color='silver', line_width=5)
camera_rotation(pl,-100.,elev_angle=110)
pl.render()
#pl.show()
pl.screenshot('coils_QH_init.png')


# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf_mesh,scalars='B_BiotSavart',show_scalar_bar=False,cmap='jet')
pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color='silver', line_width=5)
camera_rotation(pl,-100.,elev_angle=12)
pl.render()
#pl.show()
pl.screenshot('coils_QH_opt_view_2.png')

# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf_initial_mesh,scalars='B_BiotSavart',show_scalar_bar=False,cmap='jet')
pl.add_mesh(coils_init_mesh,style='wireframe',render_lines_as_tubes=True, color='silver', line_width=5)
camera_rotation(pl,-100.,elev_angle=12)
pl.render()
#pl.show()
pl.screenshot('coils_QH_init_view_2.png')