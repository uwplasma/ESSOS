
import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.surfaces import SurfaceRZFourier, SurfaceClassifier
from essos.dynamics import Particles, Tracing
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from functools import partial
import essos.augmented_lagrangian as alm
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ONE_EV




""" Creating starting coils and surface """
N_COILS = 4; 
FOURIER_ORDER = 6; 
LARGE_R = 10.; 
SMALL_R = 5.7; 
NFP = 2; 
N_SEGMENTS = 60; 
STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.84e7  # Amperes (optimization does not depend on current magnitude)



# Optimization parameters
target_B_on_axis = 5.7
max_coil_length = 45
max_coil_curvature = 1.0
nparticles = number_of_processors_to_use*10

maximum_function_evaluations = 100

model = 'GuidingCenterAdaptative'

# Load coils and field
json_file = os.path.join(os.path.dirname(__name__), '../input_files', 'QI_opt_coils_essos_5.json')
coils = Coils.from_json(json_file)
init_field = BiotSavart(coils)

# Renormalize coils to have B_target=5.7 on axis
# Load surface first to extract magnetic axis
wout_file = os.path.join(os.path.dirname(__name__), '../input_files','wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc')
vmec = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=60, nphi=60, range_torus='full torus')

# Load the ACTUAL magnetic axis at s=0 (not from s=1 surface)
vmec_axis = SurfaceRZFourier.from_wout_file(wout_file, s=0, ntheta=60, nphi=60, range_torus='full period')

# Extract magnetic axis from s=0 surface (m=0 modes give the axis in cylindrical coordinates)
m0_mask_axis = vmec_axis.xm == 0
rc_axis = vmec_axis.rc[m0_mask_axis]
zs_axis = vmec_axis.zs[m0_mask_axis]
xn_axis = vmec_axis.xn[m0_mask_axis]


# Function to compute axis position at given phi
def compute_axis_point(phi):
    """Compute magnetic axis position at toroidal angle phi using s=0 surface"""
    angles = xn_axis * phi
    R_val = jnp.sum(rc_axis * jnp.cos(angles))
    Z = -jnp.sum(zs_axis * jnp.sin(angles))
    return R_val, Z

# Evaluate B on magnetic axis and scale
phi_array = jnp.linspace(0, 2 * jnp.pi, 200)
B_axis_old = jnp.array([init_field.AbsB(jnp.array([R_val * jnp.cos(phi), R_val * jnp.sin(phi), Z])) 
                         for phi in phi_array 
                         for R_val, Z in [compute_axis_point(phi)]])


B_target = 5.7
scale_factor = B_target / jnp.mean(B_axis_old)
#scale_factor = 1/(4.*jnp.pi*1.e-7) # No scaling for now, to compare with previous results
coils.dofs_currents = coils.dofs_currents * scale_factor
init_field = BiotSavart(coils)


timeI=time()
boundary=SurfaceClassifier(vmec,h=0.1)
print(f"ESSOS boundary took {time()-timeI:.2f} seconds")
# Initialize particles from surface axis


# Input parameters
tmax = 1.e-4
timestep=1.e-8
times_to_trace=10
nparticles_per_core=20
nparticles = number_of_processors_to_use*nparticles_per_core
atol=1.e-7
rtol=1.e-7
energy=FUSION_ALPHA_PARTICLE_ENERGY


particles = Particles.InitializeParticlesAroundSurfaceAxis(
    surface=vmec_axis,                 # axis-defining surface
    boundary_surface=vmec,             # boundary/LCFS-like surface
    n_particles=nparticles,
    distance_from_axis=0.7,            # 60% of local axis->boundary distance
    distance_mode='fraction_to_boundary',
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    energy=energy,
)



#Targets
LENGTH_TARGET = max_coil_length
CURVATURE_TARGET = max_coil_curvature
Baxis_TARGET = target_B_on_axis
weight_length=1.
weight_curvature=1.
weight_Baxis=1.
weight_loss_fraction=1.




# @partial(jit, static_argnums=(0, 1))
def loss_normB_axis_average(field,target_B_on_axis=5.7):
    R_axis=jnp.mean(jnp.sqrt(jax.vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(field.coils.dofs_curves)))
    phi_array = jnp.linspace(0, 2 * jnp.pi, 100)
    B_axis = jax.vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return jnp.average(B_axis)-target_B_on_axis

def loss_length_constraint(field,target_length=40.0,N_coils=4):
    #return jnp.maximum(0, field.coils.length - target_length)
    #This was divided by target_length to make it scale invariant, but it seems to cause issues with the optimization, so I am trying without it for now
    #return jnp.maximum(0, (field.coils.length - target_length)/target_length)
    return field.coils.length[0:N_coils] - target_length

def loss_curvature_contraint(field,target_curvature=0.5,N_coils=4):
    #return jnp.maximum(0, field.coils.curvature - target_curvature)
    #This was divided by target_curvature to make it scale invariant, but it seems to cause issues with the optimization, so I am trying without it for now
    #return jnp.maximum(0, (field.coils.curvature - target_curvature)/target_curvature)
    return field.coils.curvature[0:N_coils] - target_curvature


# Use existing differentiable method from Tracing class for memory efficiency
# loss_fraction_classifier_differentiable() has been optimized with ~80% memory reduction

def loss_lost_fraction(field, particles,boundary, t=1.e-3, times_to_trace=100, timestep=1.e-8, model='GuidingCenterAdaptative'):
    tracing = Tracing(field=field, particles=particles, maxtime=t, model=model,times_to_trace=times_to_trace,timestep=timestep,boundary=boundary)
    loss_fraction = tracing.loss_fraction_classifier_differentiable(
        boundary=boundary,
        softness=30.0,
        stride=1,
        final_timestep_only=False,
    )
    return loss_fraction


# Create the constraints
penalty = 1. #Intial penalty values
multiplier=0.0 #Initial lagrange multiplier values
#Initializing first tolerances for the inner minimisation loop iteration
omega=1./penalty
eta=1./penalty**0.1
model_lagrangian='Standard'  #Use standard augmented lagragian suitable for bounded optimizers 
#Since we are using LBFGS-B from jaxopt, model_mu will be updated with tolerances so we do not need to difinte the model
model_mu='Mu_Adaptative_2'




beta=1.01                                     #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-5   #desired contraint tolerance, associated with variation of contraints
decrease_tol=0.99

""" Defining constraints"""
eta_tol_field_axis=1.e-5*Baxis_TARGET
eta_tol_length=1.e-5*LENGTH_TARGET
eta_tol_curvature=1.e-5*CURVATURE_TARGET
eta_tol_loss_fraction=1.e-5


length_constraint=alm.ScaledConstraint(alm.ineq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_length,eta_tol=eta_tol_length,use_analytical_penalty=True),auto_scale=False)
curvature_constraint=alm.ScaledConstraint(alm.ineq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_curvature,eta_tol=eta_tol_curvature,use_analytical_penalty=True),auto_scale=False)
C_Baxis_average_constraint=alm.ScaledConstraint(alm.eq(loss_normB_axis_average,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_Baxis,eta_tol=eta_tol_field_axis),auto_scale=False)
C_loss_constraint=alm.ScaledConstraint(alm.eq(loss_lost_fraction,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_loss_fraction,eta_tol=eta_tol_loss_fraction),auto_scale=False)


C_length_constraint = alm.SelectiveConstraint(length_constraint, "field", target_length=LENGTH_TARGET,N_coils=N_COILS)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field", target_curvature=CURVATURE_TARGET,N_coils=N_COILS)
C_Baxis_average_constraint = alm.SelectiveConstraint(C_Baxis_average_constraint, "field", target_B_on_axis=Baxis_TARGET)
C_loss_constraint = alm.SelectiveConstraint(C_loss_constraint, "field",particles=particles,boundary=boundary, t=tmax,times_to_trace=times_to_trace,timestep=timestep,model=model)
##TEST WITHOUT QS
C_Total_constraint = alm.combine(C_Baxis_average_constraint, C_length_constraint, C_curvature_constraint, C_loss_constraint)




C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_Baxis_average_constraint.dependencies = {"field": init_field}
C_loss_constraint.dependencies = {"field": init_field}

C_Total_constraint.dependencies = {"field": init_field}  

# Per-constraint weighting options:
#   False: no weighting (default, fastest)
#   'constraint': one weight per constraint (fast, ~seconds, low memory)
#   'element': element-wise weights (slow, ~minutes, higher memory, most precise)
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
    use_per_constraint_weights='constraint',
    weight_g_epsilon=1.e-4,
    use_perturbation_weights='True',
    perturbation_magnitude=0.01,
    perturbation_mode='relative',
    perturbation_seed=42,
    use_scaled_main_params=True,
)




# Get starting DOFs - this will respect frozen DOFs because init_surface_2 is already frozen
# The starting_dofs combines coils.dofs and surface.dofs (which excludes frozen DOFs)
#starting_dofs = C_Total_constraint.starting_dofs
starting_dofs = C_Total_constraint.starting_dofs
starting_dofs_scaled = alm._get_constraints_scaled_dofs(C_Total_constraint)
print(f"Starting DOFs shape: {starting_dofs_scaled.shape}")

#Initializing lagrange multipliers
lagrange_params = C_Total_constraint.init(starting_dofs)



#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
#params = starting_dofs, lagrange_params
params = starting_dofs_scaled, lagrange_params
#This is just to initialize an empty state for the lagrange multiplier update and get some information
params, lag_state, grad, info = ALM.init(params)
scale = alm._get_constraints_scaling_factors(C_Total_constraint)
C_Total_constraint.loss(params[1],params[0] / jnp.maximum(scale, 1e-12))
#C_Total_constraint.loss(params[1],params[0] )
print(f'grad_coils: {jnp.linalg.norm(grad[0]):g}')




""" Optimizing  with alm"""
t_start = time()

i=0
while i<=60000 and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info= ALM.update(params,lag_state,grad,info)    
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2],params[1]):g}, grad_coils: {jnp.linalg.norm(grad[0][0:160]):g}, grad_surface: {jnp.linalg.norm(grad[0][160:186]):g}, grad_surface axis: {jnp.linalg.norm(grad[0][186:]):g}')
    i=i+1

t_end = time()





opt_field_alm = C_Total_constraint.dofs_to_pytree(params[0] / jnp.maximum(scale, 1e-12))[0]
opt_coils_alm = opt_field_alm.coils




filename = f"alm_loss_fraction.pkl"
params, lag_state, grad, eta, omega, info = alm.load_checkpoint_for_reconstruction(
    filename,
    C_Total_constraint  # Your freshly created constraint object
)


filename = f"alm_loss_fraction.pkl"

alm.save_checkpoint_with_constraints(
    filename, params, lag_state, grad, eta, omega, info, 
    C_Total_constraint, 
    save_constraint_object=False  # Autoscales still saved!
)








tracing_initial = Tracing(field=init_field, particles=particles, maxtime=tmax, model=model,times_to_trace=times_to_trace,timestep=timestep,boundary=boundary)
tracing_initial.loss_fractions
tracing_optimized = Tracing(field=opt_field_alm,particles=particles, maxtime=tmax, model=model,times_to_trace=times_to_trace,timestep=timestep,boundary=boundary)
tracing_optimized.loss_fractions
#print('Final params',params)
#print(info[1])
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

ax3.set_xlabel('R (m)')
ax3.set_ylabel('Z (m)')
#ax3.legend()
new_coils.plot(ax=ax2, show=False)
tracing_optimized.plot(ax=ax2, show=False)
for i, trajectory in enumerate(tracing_optimized.trajectories):
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
ax4.set_xlabel('R (m)')
ax4.set_ylabel('Z (m)')#ax4.legend()
plt.tight_layout()
plt.show()

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# tracing_initial.to_vtk('trajectories_initial')
#tracing_optimized.to_vtk('trajectories_final')
coils.to_vtk('coils')
#new_coils.to_vtk('coils_optimized')
