
import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.surfaces import SurfaceRZFourier, SurfaceClassifier, BdotN_over_B,B_on_surface
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
vmec_period = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=60, nphi=60, range_torus='full period')

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
BdotN_Target=1.e-3
weight_length=1.
weight_curvature=1.
weight_Baxis=1.
weight_loss_fraction=1.
weight_BdotN=1.



def BdotN_constraint(field,surface,target_tol=1.e-6,P=50.):
    bdotn_over_b = BdotN_over_B(surface, field)
    g_local =jnp.abs(bdotn_over_b) - target_tol
    # Ajustamos P para a escala física (ex: se target é 1e-4, P será 200.000)
    P = 20.0 / jnp.maximum(1e-8, target_tol) 
    # LogSumExp estável
    #g_max = (1.0 / P) * jax.scipy.special.logsumexp(P * g_local)
    g_max=(1.0 / P) *  jax.scipy.special.logsumexp(P * g_local) - (jnp.log(bdotn_over_b.size) / P)
    return g_max


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
eta_tol_BdotN=1.e-5*BdotN_Target

bdotn_constraint = alm.ScaledConstraint(alm.ineq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_BdotN,eta_tol=eta_tol_BdotN,use_analytical_penalty=True),auto_scale=False)
length_constraint=alm.ScaledConstraint(alm.ineq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_length,eta_tol=eta_tol_length,use_analytical_penalty=True),auto_scale=False)
curvature_constraint=alm.ScaledConstraint(alm.ineq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_curvature,eta_tol=eta_tol_curvature,use_analytical_penalty=True),auto_scale=False)
Baxis_average_constraint=alm.ScaledConstraint(alm.eq(loss_normB_axis_average,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_Baxis,eta_tol=eta_tol_field_axis),auto_scale=False)
loss_constraint=alm.ScaledConstraint(alm.eq(loss_lost_fraction,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_loss_fraction,eta_tol=eta_tol_loss_fraction),auto_scale=False)


C_BdotN_constraint = alm.SelectiveConstraint(bdotn_constraint, "field", surface=vmec_period, target_tol=BdotN_Target)
C_length_constraint = alm.SelectiveConstraint(length_constraint, "field", target_length=LENGTH_TARGET,N_coils=N_COILS)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field", target_curvature=CURVATURE_TARGET,N_coils=N_COILS)
C_Baxis_average_constraint = alm.SelectiveConstraint(Baxis_average_constraint, "field", target_B_on_axis=Baxis_TARGET)
C_loss_constraint = alm.SelectiveConstraint(loss_constraint, "field",particles=particles,boundary=boundary, t=tmax,times_to_trace=times_to_trace,timestep=timestep,model=model)
##TEST WITHOUT QS
C_Total_constraint = alm.combine(C_BdotN_constraint, C_Baxis_average_constraint, C_length_constraint, C_curvature_constraint, C_loss_constraint)



C_BdotN_constraint.dependencies = {"field": init_field}
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


B_old=jnp.mean(jnp.array([opt_field_alm.AbsB(jnp.array([R_val * jnp.cos(phi), R_val * jnp.sin(phi), Z])) 
                         for phi in phi_array 
                         for R_val, Z in [compute_axis_point(phi)]]))

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("\n=== OPTIMIZATION RESULTS ===")
print("Initial B dot N:", jnp.max(BdotN_over_B(vmec, init_field)))    
print("B dot N after optimization:", jnp.max(BdotN_over_B(vmec, opt_field_alm)))
print("\nInitial curvature :", jnp.average(init_field.coils.curvature,axis=0))    
print("Curvature after optimization:",jnp.average(opt_field_alm.coils.curvature,axis=0))
print("Curvature target:",CURVATURE_TARGET)
print("\nInitial length :", init_field.coils.length)    
print("Length after optimization:",opt_field_alm.coils.length)
print("Length target:",LENGTH_TARGET)


filename = f"alm_loss_fraction_bdotn_4.pkl"
params, lag_state, grad, eta, omega, info = alm.load_checkpoint_for_reconstruction(
    filename,
    C_Total_constraint  # Your freshly created constraint object
)


filename = f"alm_loss_fraction_bdotn_5.pkl"

alm.save_checkpoint_with_constraints(
    filename, params, lag_state, grad, eta, omega, info, 
    C_Total_constraint, 
    save_constraint_object=False  # Autoscales still saved!
)




scale_factor = B_target / B_old
#scale_factor = 1/(4.*jnp.pi*1.e-7) # No scaling for now, to compare with previous results
opt_coils_alm.dofs_currents = coils.dofs_currents * scale_factor
opt_field_alm_scaled = BiotSavart(opt_coils_alm)



tracing_initial = Tracing(field=init_field, particles=particles, maxtime=tmax, model=model,times_to_trace=times_to_trace,timestep=timestep,boundary=boundary)
tracing_initial.loss_fractions
tracing_optimized = Tracing(field=opt_field_alm,particles=particles, maxtime=tmax, model=model,times_to_trace=times_to_trace,timestep=timestep,boundary=boundary)
tracing_optimized.loss_fractions

# Plot loss fraction comparison
import matplotlib.ticker as mticker
fig_lf, ax_lf = plt.subplots(figsize=(7, 5))
ax_lf.plot(tracing_initial.times, tracing_initial.loss_fractions, label='Initial', color='b', linewidth=3)
ax_lf.plot(tracing_optimized.times, tracing_optimized.loss_fractions, label='Optimized', color='gold', linewidth=3)
ax_lf.set_title('Loss Fraction vs Time', fontweight='bold')
ax_lf.set_xlabel('Time [s]', fontweight='bold')
ax_lf.set_ylabel('Loss Fraction', fontweight='bold')
ax_lf.legend(fontsize=14)
ax_lf.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# Alternatively, for more control, use ScalarFormatter:
# ax_lf.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
# ax_lf.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax_lf.tick_params(axis='both', labelsize=16)
fig_lf.tight_layout()
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
fig_lf.savefig('loss_fraction_comparison.pdf', dpi=300)
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





import pyvista as pv
coils_init_file='coils.vtu'
coils_opt_file ='coils_opt.vtu'
opt_coils_alm.to_vtk('coils_opt')
coils.to_vtk('coils_init')

# Load the coil and field files
#surf1_mesh = pv.read(surf_1_file)
coils_init_mesh = pv.read(coils_init_file)
coils_opt_mesh = pv.read(coils_opt_file)
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

# Exclude zero values from tracing mesh


pl = pv.Plotter(off_screen=True)
pl.add_mesh(coils_opt_mesh, style='wireframe', render_lines_as_tubes=True, color='gold', line_width=5)
pl.add_mesh(coils_init_mesh, style='wireframe', render_lines_as_tubes=True, color='blue', line_width=5)
pl.add_text('Initial', position='upper_left', font_size=18, color='blue')
pl.add_text('Optimized', position='upper_right', font_size=18, color='gold')
camera_rotation(pl, -100., elev_angle=12)
pl.render()
pl.screenshot('coils_paraviw_style_compare_loss_fraction.png')





# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing


phi_poincare=jnp.pi / 2
#tmax = 50000000000
tmax = 10000
nfieldlines_per_core = 40
nfieldlines = nfieldlines_per_core * number_of_processors_to_use

trace_tolerance = 1e-7
num_steps = 60000
R0=jnp.linspace(16.5, 17.35, nfieldlines) # Starting points for field line tracing, evenly spaced in R
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T
time0 = time()

fieldline_tracing = block_until_ready(Tracing(
    field=opt_field_alm_scaled,
    model='FieldLineAdaptative',
    initial_conditions=initial_xyz,
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance
))

print(f"ESSOS tracing took {time() - time0:.2f} seconds")

fieldline_trajectories = fieldline_tracing.trajectories
traj = fieldline_trajectories[0]
R, phi, Z = traj[:, 0], traj[:, 1], traj[:, 2]

phi_u = jnp.unwrap(phi)
phi0_cross = jnp.where((phi_u[:-1] < 0) & (phi_u[1:] >= 0))[0]
phi90_cross = jnp.where((phi_u[:-1] < phi_poincare) & (phi_u[1:] >= phi_poincare))[0]

theta = jnp.linspace(0, 2 * jnp.pi, 200)

def compute_rz_on_phi(surface, theta, phi=0.0):
    angles = jnp.outer(theta, surface.xm) - phi * surface.xn
    R = jnp.sum(surface.rc * jnp.cos(angles), axis=1)
    Z = jnp.sum(surface.zs * jnp.sin(angles), axis=1)
    return R, Z


# # Contours from true VMEC surface
phi_poincare_45 = jnp.pi / 4
phi_poincare_135 = 3 * jnp.pi / 4

R0_true, Z0_true = compute_rz_on_phi(vmec, theta, phi=0.0)
R45_true, Z45_true = compute_rz_on_phi(vmec, theta, phi=phi_poincare_45)
R90_true, Z90_true = compute_rz_on_phi(vmec, theta, phi=phi_poincare)
R135_true, Z135_true = compute_rz_on_phi(vmec, theta, phi=phi_poincare_135)



# Compute optimized surface contours for each phi
R0_opt, Z0_opt = compute_rz_on_phi(opt_field_alm.surface, theta, phi=0.0)
R45_opt, Z45_opt = compute_rz_on_phi(opt_field_alm.surface, theta, phi=phi_poincare_45)
R90_opt, Z90_opt = compute_rz_on_phi(opt_field_alm.surface, theta, phi=phi_poincare)


plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 22,
    'axes.titleweight': 'bold',
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

# φ = 0
fig0, ax0 = plt.subplots(figsize=(7, 6))
fieldline_tracing.poincare_plot(ax=ax0, show=False, shifts=[0])
ax0.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized at $\phi = 0$")
ax0.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC at $\phi = 0$")
ax0.set_xlabel("R")
ax0.set_ylabel("Z")
ax0.set_title("Poincaré at φ = 0")
ax0.axis("equal")
ax0.title.set_fontsize(22)
ax0.xaxis.label.set_size(18)
ax0.yaxis.label.set_size(18)
ax0.grid(False)
ax0.tick_params(axis='both', labelsize=16)
legend0 = ax0.legend()
if legend0:
    for text in legend0.get_texts():
        text.set_fontsize(16)


fig0.tight_layout()
fig0.savefig('poincare_phi0.png', dpi=300)

# φ = π/4
fig45, ax45 = plt.subplots(figsize=(7, 6))
fieldline_tracing.poincare_plot(ax=ax45, show=False, shifts=[phi_poincare_45])
ax45.plot(R45_opt, Z45_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = \pi/4$")
ax45.plot(R45_true, Z45_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/4$")
ax45.set_xlabel("R")
ax45.set_ylabel("Z")
ax45.set_title("Poincaré + Surfaces @ φ = π/4")
ax45.axis("equal")
ax45.title.set_fontsize(22)
ax45.xaxis.label.set_size(18)
ax45.yaxis.label.set_size(18)
ax45.grid(False)
ax45.tick_params(axis='both', labelsize=16)
legend45 = ax45.legend()
if legend45:
    for text in legend45.get_texts():
        text.set_fontsize(16)


fig45.tight_layout()
fig45.savefig('poincare_phi45.png', dpi=300)

# φ = π/2

fig90, ax90 = plt.subplots(figsize=(7, 6))
fieldline_tracing.poincare_plot(ax=ax90, show=False, shifts=[phi_poincare])
ax90.plot(R90_opt, Z90_opt, color='black', linewidth=2.5, label=r"Optimized at $\phi = \pi/2$")
ax90.plot(R90_true, Z90_true, color='blue', linewidth=2, label=r"True VMEC at $\phi = \pi/2$")
ax90.set_xlabel("R", fontweight='bold')
ax90.set_ylabel("Z", fontweight='bold')
ax90.set_title("Poincaré at φ = π/2", fontweight='bold')
ax90.axis("equal")
ax90.grid(False)
ax90.tick_params(axis='both', labelsize=18, width=2)
legend90 = ax90.legend()
if legend90:
    for text in legend90.get_texts():
        text.set_fontsize(18)
        text.set_fontweight('bold')


fig90.tight_layout()
fig90.savefig('poincare_phi90.png', dpi=300)

# # φ = 3π/4 (uncomment to enable)
# fig135, ax135 = plt.subplots(figsize=(7, 6))
# fieldline_tracing.poincare_plot(ax=ax135, show=False, shifts=[phi_poincare_135])
# ax135.plot(R135_opt, Z135_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 3\pi/4$")
# ax135.plot(R135_true, Z135_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 3\pi/4$")
# ax135.set_xlabel("R")
# ax135.set_ylabel("Z")
# ax135.set_title("Poincaré + Surfaces @ φ = 3π/4")
# ax135.axis("equal")
# ax135.title.set_fontsize(22)
# ax135.xaxis.label.set_size(18)
# ax135.yaxis.label.set_size(18)
# ax135.tick_params(axis='both', labelsize=16)
# legend135 = ax135.legend()
# if legend135:
#     for text in legend135.get_texts():
#         text.set_fontsize(16)
# fig135.tight_layout()
# fig135.savefig('poincare_phi135.png', dpi=300)


B_final= B_on_surface(vmec, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)

fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(vmec.phi2d[0:30,:], vmec.theta2d[0:30,:], modB[0:30,:], levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB_loss_fraction.png', dpi=300)