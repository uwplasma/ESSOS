
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
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ONE_EV, ELECTRON_MASS,PROTON_MASS,SPEED_OF_LIGHT
from essos.background_species import BackgroundSpecies



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

maximum_function_evaluations = 100

model = 'GuidingCenterCollisionsMuFixed'

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


#SPECIES
number_species=2  #(electrons,deuterium)
mass_array=jnp.array([ELECTRON_MASS/PROTON_MASS,2])    #mass_over_mproton
charge_array=jnp.array([-1.,1.])    #mass_over_mproton
T0=6.6e+3  #eV
n0=4.e+20  #m^-3
n_array=jnp.array([n0,n0])
T_array=jnp.array([T0,T0])
species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)


    

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
weight_heat_flux=1.







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


def boundary_curve_xyz_at_phi(surface, phi0, ntheta=400):
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    angles = jnp.outer(theta, surface.xm) - phi0 * surface.xn
    r = jnp.sum(surface.rc[None, :] * jnp.cos(angles), axis=1)
    z = jnp.sum(surface.zs[None, :] * jnp.sin(angles), axis=1)
    x = r * jnp.cos(phi0)
    y = r * jnp.sin(phi0)
    return jnp.stack([x, y, z], axis=1), theta


def boundary_curve_xyz_at_theta(surface, theta0, nphi=400):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nphi, endpoint=False)
    angles = theta0 * surface.xm[None, :] - jnp.outer(phi, surface.xn)
    r = jnp.sum(surface.rc[None, :] * jnp.cos(angles), axis=1)
    z = jnp.sum(surface.zs[None, :] * jnp.sin(angles), axis=1)
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    return jnp.stack([x, y, z], axis=1), phi


def build_line_from_boundary_phi(surface, phi0, theta_ref=0.0, ntheta=400):
    curve_xyz, theta = boundary_curve_xyz_at_phi(surface, phi0, ntheta=ntheta)
    # Choose the boundary point nearest theta_ref and use local tangent there.
    idx = jnp.argmin(jnp.abs(theta - theta_ref))
    idx_prev = (idx - 1) % ntheta
    idx_next = (idx + 1) % ntheta
    line_point = curve_xyz[idx]
    tangent = curve_xyz[idx_next] - curve_xyz[idx_prev]
    line_direction = tangent / (jnp.linalg.norm(tangent) + 1e-12)
    return line_point, line_direction, curve_xyz


def build_line_from_boundary_theta(surface, theta0, phi_ref=0.0, nphi=400):
    curve_xyz, phi = boundary_curve_xyz_at_theta(surface, theta0, nphi=nphi)
    # Choose the boundary point nearest phi_ref and use local tangent there.
    idx = jnp.argmin(jnp.abs(phi - phi_ref))
    idx_prev = (idx - 1) % nphi
    idx_next = (idx + 1) % nphi
    line_point = curve_xyz[idx]
    tangent = curve_xyz[idx_next] - curve_xyz[idx_prev]
    line_direction = tangent / (jnp.linalg.norm(tangent) + 1e-12)
    return line_point, line_direction, curve_xyz




def _heat_flux_from_fixed_trajectories(field, trajectories, particles, boundary,softness=30.0, final_timestep_only=False,aggregation_mode='mean', smoothmax_beta=10.0, max_heat_flux_target=0.0):
    trajectories_xyz = trajectories[:, :, :3]
    if final_timestep_only:
        trajectories_xyz = trajectories_xyz[:, -1:, :]
        trajectories_sampled = trajectories[:, -1:, :]
    else:
        trajectories_xyz = trajectories_xyz[:, ::1, :]
        trajectories_sampled = trajectories[:, ::1, :]
    nparticles, ntimesteps_sampled = trajectories_xyz.shape[:2]
    finite_xyz = jnp.all(jnp.isfinite(trajectories_xyz), axis=2)
    trajectories_xyz_safe = jnp.where(finite_xyz[:, :, None], trajectories_xyz, 0.0)
    distances_flat = jax.vmap(boundary.evaluate_xyz)(trajectories_xyz_safe.reshape(-1, 3))
    distances = distances_flat.reshape(nparticles, ntimesteps_sampled)
    outside_prob = jax.nn.sigmoid(-softness * distances)
    outside_prob = jnp.where(finite_xyz, outside_prob, 0.0)
    cumulative_loss_prob = jax.lax.associative_scan(jnp.maximum, outside_prob, axis=1)
    first_loss_prob = jnp.concatenate(
        [cumulative_loss_prob[:, :1], cumulative_loss_prob[:, 1:] - cumulative_loss_prob[:, :-1]],
        axis=1,
    )
    first_loss_prob = jnp.maximum(first_loss_prob, 0.0)
    mass = particles.mass
    vpar = SPEED_OF_LIGHT * trajectories_sampled[:, :, 3]
    mu = trajectories_sampled[:, :, 4] * (SPEED_OF_LIGHT ** 2 * mass)
    finite_vpar = jnp.isfinite(vpar)
    finite_mu = jnp.isfinite(mu)
    valid_mask = finite_xyz & finite_vpar & finite_mu
    vpar_safe = jnp.where(finite_vpar, vpar, 0.0)
    mu_safe = jnp.where(finite_mu, mu, 0.0)
    abs_b_flat = jax.vmap(field.AbsB)(trajectories_xyz_safe.reshape(-1, 3))
    abs_b = abs_b_flat.reshape(nparticles, ntimesteps_sampled)
    abs_b_safe = jnp.where(finite_xyz & jnp.isfinite(abs_b), abs_b, 1e-12)
    v_perp_sq = 2.0 * mu_safe * abs_b_safe / mass
    v_perp = jnp.sqrt(jnp.maximum(v_perp_sq, 0.0))
    e_total_joules = 0.5 * mass * (vpar_safe ** 2 + v_perp ** 2)
    q_perp_joules = e_total_joules * v_perp
    normalization_factor = particles.energy * particles.total_speed
    q_perp_scaled = q_perp_joules / (normalization_factor + 1e-12)
    q_perp_scaled = jnp.where(valid_mask, q_perp_scaled, 0.0)
    first_loss_prob_safe = jnp.where(valid_mask, first_loss_prob, 0.0)
    if aggregation_mode == 'max':
        # Compute maximum heat flux over all particles at their loss events.
        # For each particle: aggregate heat flux during its loss (weighted mean by first_loss_prob).
        # Then take smooth-max of these per-particle values.
        loss_weight_per_particle = jnp.sum(first_loss_prob_safe, axis=1)
        q_per_particle = jnp.sum(first_loss_prob_safe * q_perp_scaled, axis=1) / (loss_weight_per_particle + 1e-8)
        has_loss_particle = loss_weight_per_particle > 1e-8
        q_per_particle_masked = jnp.where(has_loss_particle, q_per_particle, -1e3)
        smooth_max = jax.scipy.special.logsumexp(smoothmax_beta * q_per_particle_masked) / smoothmax_beta
        return jnp.where(jnp.any(has_loss_particle), smooth_max - max_heat_flux_target, 0.0)
    weighted_q_perp = jnp.sum(first_loss_prob_safe * q_perp_scaled)
    total_loss_weight = jnp.sum(first_loss_prob_safe)
    return weighted_q_perp / (total_loss_weight + 1e-8)


def make_loss_perpendicular_heat_flux(particles, boundary, t=1.e-4, times_to_trace=10,
                                      timestep=1.e-8, model='GuidingCenterCollisionsMuFixed',
                                      species=species, final_timestep_only=False,
                                      aggregation_mode='mean', smoothmax_beta=10.0,
                                      max_heat_flux_target=0.0):
    """Build a custom-VJP heat-flux loss where only `field` is differentiable."""
    t_fixed = t
    times_to_trace_fixed = int(times_to_trace)
    timestep_fixed = timestep
    model_fixed = model
    species_fixed = species
    boundary_fixed = boundary
    particles_fixed = particles
    final_only_fixed = final_timestep_only
    aggregation_mode_fixed = aggregation_mode
    smoothmax_beta_fixed = smoothmax_beta
    max_heat_flux_target_fixed = max_heat_flux_target
    @jax.custom_vjp
    def loss_perpendicular_heat_flux(field):
        tracing = Tracing(
            field=field,
            particles=particles_fixed,
            maxtime=t_fixed,
            model=model_fixed,
            times_to_trace=times_to_trace_fixed,
            timestep=timestep_fixed,
            species=species_fixed,
            boundary=boundary_fixed,
        )
        return _heat_flux_from_fixed_trajectories(
            field,
            tracing.trajectories,
            particles_fixed,
            boundary_fixed,
            softness=30.0,
            final_timestep_only=final_only_fixed,
            aggregation_mode=aggregation_mode_fixed,
            smoothmax_beta=smoothmax_beta_fixed,
            max_heat_flux_target=max_heat_flux_target_fixed,
        )
    def _loss_perpendicular_heat_flux_fwd(field):
        tracing = Tracing(
            field=field,
            particles=particles_fixed,
            maxtime=t_fixed,
            model=model_fixed,
            times_to_trace=times_to_trace_fixed,
            timestep=timestep_fixed,
            species=species_fixed,
            boundary=boundary_fixed,
        )
        loss = _heat_flux_from_fixed_trajectories(
            field,
            tracing.trajectories,
            particles_fixed,
            boundary_fixed,
            softness=30.0,
            final_timestep_only=final_only_fixed,
            aggregation_mode=aggregation_mode_fixed,
            smoothmax_beta=smoothmax_beta_fixed,
            max_heat_flux_target=max_heat_flux_target_fixed,
        )
        return loss, (tracing.trajectories, field)
    def _loss_perpendicular_heat_flux_bwd(residuals, g):
        trajectories, field = residuals
        def proxy(field_local):
            return _heat_flux_from_fixed_trajectories(
                field_local,
                trajectories,
                particles_fixed,
                boundary_fixed,
                softness=30.0,
                final_timestep_only=final_only_fixed,
                aggregation_mode=aggregation_mode_fixed,
                smoothmax_beta=smoothmax_beta_fixed,
                max_heat_flux_target=max_heat_flux_target_fixed,
            )
        grad_field = jax.grad(proxy)(field)
        return (jax.tree_util.tree_map(lambda x: g * x, grad_field),)
    loss_perpendicular_heat_flux.defvjp(
        _loss_perpendicular_heat_flux_fwd,
        _loss_perpendicular_heat_flux_bwd,
    )
    return loss_perpendicular_heat_flux





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
eta_tol_heat_flux=1.e-5  # Tolerance for perpendicular heat flux in Joules
initial_max_metric = loss_max_no_target(init_field)
# 3) Set target = half of initial max
MAX_QPERP_TARGET = 0.5 * float(initial_max_metric)
eta_tol_heat_flux = 1e-5 * MAX_QPERP_TARGET


bdotn_constraint = alm.ScaledConstraint(alm.ineq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_BdotN,eta_tol=eta_tol_BdotN,use_analytical_penalty=True),auto_scale=False)
length_constraint=alm.ScaledConstraint(alm.ineq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_length,eta_tol=eta_tol_length,use_analytical_penalty=True),auto_scale=False)
curvature_constraint=alm.ScaledConstraint(alm.ineq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_curvature,eta_tol=eta_tol_curvature,use_analytical_penalty=True),auto_scale=False)
Baxis_average_constraint=alm.ScaledConstraint(alm.eq(loss_normB_axis_average,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_Baxis,eta_tol=eta_tol_field_axis),auto_scale=False)
loss_perpendicular_heat_flux = make_loss_perpendicular_heat_flux(
    particles=particles,
    boundary=boundary,
    t=tmax,
    times_to_trace=times_to_trace,
    timestep=timestep,
    model=model,
    species=species,
    final_timestep_only=False,
    aggregation_mode='max',
    smoothmax_beta=10.0,
    max_heat_flux_target=MAX_QPERP_TARGET,
)

# initial_heat_flux_loss = loss_perpendicular_heat_flux(init_field)
# print(f"Initial differentiable heat-flux loss: {float(initial_heat_flux_loss):.6e}")

#heat_flux_constraint=alm.ScaledConstraint(alm.eq(loss_perpendicular_heat_flux,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_heat_flux,eta_tol=eta_tol_heat_flux),auto_scale=False)
heat_flux_constraint=alm.ScaledConstraint(alm.ineq(loss_perpendicular_heat_flux,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_heat_flux,eta_tol=eta_tol_heat_flux,use_analytical_penalty=True),auto_scale=False)

C_length_constraint = alm.SelectiveConstraint(length_constraint, "field", target_length=LENGTH_TARGET,N_coils=N_COILS)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field", target_curvature=CURVATURE_TARGET,N_coils=N_COILS)
C_Baxis_average_constraint = alm.SelectiveConstraint(Baxis_average_constraint, "field", target_B_on_axis=Baxis_TARGET)
C_BdotN_constraint = alm.SelectiveConstraint(bdotn_constraint, "field", surface=vmec_period, target_tol=BdotN_Target)
C_heat_flux_constraint = alm.SelectiveConstraint(
    heat_flux_constraint,
    "field",
)


##TEST
C_Total_constraint = alm.combine(C_Baxis_average_constraint,C_BdotN_constraint, C_length_constraint, C_curvature_constraint, C_heat_flux_constraint)




C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_Baxis_average_constraint.dependencies = {"field": init_field}
C_BdotN_constraint.dependencies = {"field": init_field}
C_heat_flux_constraint.dependencies = {"field": init_field}

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
    use_perturbation_weights=True,
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


filename = f"alm_loss_fraction_heat_vjp.pkl"
params, lag_state, grad, eta, omega, info = alm.load_checkpoint_for_reconstruction(
    filename,
    C_Total_constraint  # Your freshly created constraint object
)


filename = f"alm_loss_fraction_heat_vjp_2.pkl"

alm.save_checkpoint_with_constraints(
    filename, params, lag_state, grad, eta, omega, info, 
    C_Total_constraint, 
    save_constraint_object=False  # Autoscales still saved!
)








tracing_initial = Tracing(field=init_field, particles=particles, maxtime=tmax, model=model,times_to_trace=times_to_trace,timestep=timestep,species=species,boundary=boundary)
tracing_initial.loss_fractions
tracing_optimized = Tracing(field=opt_field_alm,particles=particles, maxtime=tmax, model=model,species=species,boundary=boundary)
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

# Plot perpendicular heat flux at lost positions: initial vs optimized
lost_mask_initial = tracing_initial.lost_times >= 0
lost_mask_opt = tracing_optimized.lost_times >= 0

lost_positions_initial = tracing_initial.lost_positions[lost_mask_initial]
lost_positions_opt = tracing_optimized.lost_positions[lost_mask_opt]

lost_q_perp_initial = tracing_initial.lost_q_perp[lost_mask_initial]
lost_q_perp_opt = tracing_optimized.lost_q_perp[lost_mask_opt]

print(f"\nLost particles (initial): {jnp.sum(lost_mask_initial)}")
print(f"Lost particles (optimized): {jnp.sum(lost_mask_opt)}")

if lost_positions_initial.shape[0] > 0 or lost_positions_opt.shape[0] > 0:
    fig_heat_flux = plt.figure(figsize=(16, 6))
    surface_xyz = vmec.gamma
    
    # 3D plot: Initial lost positions with heat flux
    ax1_hf = fig_heat_flux.add_subplot(131, projection='3d')
    ax1_hf.plot_surface(
        surface_xyz[:, :, 0],
        surface_xyz[:, :, 1],
        surface_xyz[:, :, 2],
        alpha=0.12,
        color='gray',
        edgecolor='lightgray',
        linewidth=0.2,
        rstride=2,
        cstride=2,
    )
    if lost_positions_initial.shape[0] > 0:
        scatter1 = ax1_hf.scatter(
            lost_positions_initial[:, 0],
            lost_positions_initial[:, 1],
            lost_positions_initial[:, 2],
            c=lost_q_perp_initial,
            cmap='hot',
            edgecolors='black',
            s=80,
            vmin=0,
            vmax=max(jnp.max(lost_q_perp_initial) if lost_positions_initial.shape[0] > 0 else 1,
                     jnp.max(lost_q_perp_opt) if lost_positions_opt.shape[0] > 0 else 1),
            alpha=0.8,
        )
        plt.colorbar(scatter1, ax=ax1_hf, label='q⊥ (normalized)', shrink=0.6)
        ax1_hf.set_title(f'Initial: {lost_positions_initial.shape[0]} lost particles')
    else:
        ax1_hf.text(0.5, 0.5, 0.5, 'No lost particles', ha='center', va='center', transform=ax1_hf.transAxes)
        ax1_hf.set_title('Initial: No losses')
    ax1_hf.set_xlabel('X (m)')
    ax1_hf.set_ylabel('Y (m)')
    ax1_hf.set_zlabel('Z (m)')
    ax1_hf.set_box_aspect([1, 1, 1])
    
    # 3D plot: Optimized lost positions with heat flux
    ax2_hf = fig_heat_flux.add_subplot(132, projection='3d')
    ax2_hf.plot_surface(
        surface_xyz[:, :, 0],
        surface_xyz[:, :, 1],
        surface_xyz[:, :, 2],
        alpha=0.12,
        color='gray',
        edgecolor='lightgray',
        linewidth=0.2,
        rstride=2,
        cstride=2,
    )
    if lost_positions_opt.shape[0] > 0:
        scatter2 = ax2_hf.scatter(
            lost_positions_opt[:, 0],
            lost_positions_opt[:, 1],
            lost_positions_opt[:, 2],
            c=lost_q_perp_opt,
            cmap='hot',
            edgecolors='black',
            s=80,
            vmin=0,
            vmax=max(jnp.max(lost_q_perp_initial) if lost_positions_initial.shape[0] > 0 else 1,
                     jnp.max(lost_q_perp_opt) if lost_positions_opt.shape[0] > 0 else 1),
            alpha=0.8,
        )
        plt.colorbar(scatter2, ax=ax2_hf, label='q⊥ (normalized)', shrink=0.6)
        ax2_hf.set_title(f'Optimized: {lost_positions_opt.shape[0]} lost particles')
    else:
        ax2_hf.text(0.5, 0.5, 0.5, 'No lost particles', ha='center', va='center', transform=ax2_hf.transAxes)
        ax2_hf.set_title('Optimized: No losses')
    ax2_hf.set_xlabel('X (m)')
    ax2_hf.set_ylabel('Y (m)')
    ax2_hf.set_zlabel('Z (m)')
    ax2_hf.set_box_aspect([1, 1, 1])
    
    # Histogram comparison
    ax3_hf = fig_heat_flux.add_subplot(133)
    if lost_positions_initial.shape[0] > 0:
        ax3_hf.hist(lost_q_perp_initial, bins=20, alpha=0.6, label='Initial', color='blue', edgecolor='black')
        print(f"Initial mean q⊥: {jnp.mean(lost_q_perp_initial):.6f}, max: {jnp.max(lost_q_perp_initial):.6f}")
    if lost_positions_opt.shape[0] > 0:
        ax3_hf.hist(lost_q_perp_opt, bins=20, alpha=0.6, label='Optimized', color='red', edgecolor='black')
        print(f"Optimized mean q⊥: {jnp.mean(lost_q_perp_opt):.6f}, max: {jnp.max(lost_q_perp_opt):.6f}")
    ax3_hf.set_xlabel('Perpendicular Heat Flux q⊥ (normalized)')
    ax3_hf.set_ylabel('Count')
    ax3_hf.set_title('Heat Flux Distribution at Lost Positions')
    ax3_hf.legend()
    ax3_hf.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_flux_comparison_at_lost_positions.png', dpi=260, bbox_inches='tight')
    print('Saved heat flux comparison plot: heat_flux_comparison_at_lost_positions.png')
else:
    print('No lost particles detected in either configuration; skipping heat flux plot.')





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
R0=jnp.linspace(16.5, 17.4, nfieldlines) # Starting points for field line tracing, evenly spaced in R
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T
time0 = time()

fieldline_tracing = block_until_ready(Tracing(
    field=opt_field_alm,
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

fig, ((ax0, ax45), (ax90, ax135)) = plt.subplots(2, 2, figsize=(14, 12))

# φ = 0 subplot
fieldline_tracing.poincare_plot(ax=ax0, show=False, shifts=[0])
ax0.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 0$")
ax0.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
ax0.set_xlabel("R")
ax0.set_ylabel("Z")
ax0.set_title("Poincaré + Surfaces @ φ = 0")
ax0.legend()
ax0.axis("equal")

# φ = π/4 subplot
fieldline_tracing.poincare_plot(ax=ax45, show=False, shifts=[phi_poincare_45])
ax45.plot(R45_opt, Z45_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = \pi/4$")
ax45.plot(R45_true, Z45_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/4$")
ax45.set_xlabel("R")
ax45.set_ylabel("Z")
ax45.set_title("Poincaré + Surfaces @ φ = π/4")
ax45.legend()
ax45.axis("equal")

# φ = π/2 subplot
fieldline_tracing.poincare_plot(ax=ax90, show=False, shifts=[phi_poincare])
ax90.plot(R90_opt, Z90_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2$")
ax90.plot(R90_true, Z90_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")
ax90.set_xlabel("R")
ax90.set_ylabel("Z")
ax90.set_title("Poincaré + Surfaces @ φ = π/2")
ax90.legend()
ax90.axis("equal")

# # φ = 3π/4 subplot
# fieldline_tracing.poincare_plot(ax=ax135, show=False, shifts=[phi_poincare_135])
# ax135.plot(R135_opt, Z135_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 3\pi/4$")
# ax135.plot(R135_true, Z135_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 3\pi/4$")
# ax135.set_xlabel("R")
# ax135.set_ylabel("Z")
# ax135.set_title("Poincaré + Surfaces @ φ = 3π/4")
# ax135.legend()
# ax135.axis("equal")

plt.tight_layout()
plt.savefig('optimize_qfm_surface_poincare_loss_fraction.png', dpi=300)


B_final= B_on_surface(vmec, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)

fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(vmec.phi2d[0:30,:], vmec.theta2d[0:30,:], modB[0:30,:], levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB_loss_fraction.png', dpi=300)