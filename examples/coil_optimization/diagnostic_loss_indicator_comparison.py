import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import jax.numpy as jnp
from essos.dynamics import Particles, Tracing
from essos.surfaces import SurfaceRZFourier, SurfaceClassifier
from essos.fields import BiotSavart
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY, SPEED_OF_LIGHT, ELECTRON_MASS, PROTON_MASS
from essos.background_species import BackgroundSpecies

# --- Load surface and coils ---
json_file = '/home/exouser/ESSOS/examples/input_files/QI_opt_coils_essos_5.json'
wout_file = '/home/exouser/ESSOS/examples/input_files/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc'
coils = Coils.from_json(json_file)
init_field = BiotSavart(coils)
# Reduce ntheta/nphi for memory
vmec = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=20, nphi=20, range_torus='full torus')
vmec_axis = SurfaceRZFourier.from_wout_file(wout_file, s=0, ntheta=20, nphi=20, range_torus='full period')
boundary = SurfaceClassifier(vmec, h=0.1)


    

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



# --- Initialize species ---
number_species = 2  # electrons, deuterium
mass_array = jnp.array([ELECTRON_MASS/PROTON_MASS, 2])
charge_array = jnp.array([-1., 1.])
T0 = 6.6e+3  # eV
n0 = 4.e+20  # m^-3
n_array = jnp.array([n0, n0])
T_array = jnp.array([T0, T0])
species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)

# --- Initialize 20 particles ---
nparticles = 20
tmax = 1e-4
timestep = 1e-8
times_to_trace = 100
energy = FUSION_ALPHA_PARTICLE_ENERGY
particles = Particles.InitializeParticlesAroundSurfaceAxis(
    surface=vmec_axis,
    boundary_surface=vmec,
    n_particles=nparticles,
    distance_from_axis=0.7,
    distance_mode='fraction_to_boundary',
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    energy=energy,
)

# --- Trace particles ---
tracing = Tracing(
    field=init_field,
    particles=particles,
    maxtime=tmax,
    model='GuidingCenterCollisionsMuFixed',
    times_to_trace=times_to_trace,
    timestep=timestep,
    species=species,
    boundary=boundary,
)
trajectories = tracing.trajectories

# --- Compute boundary distances ---
trajectories_xyz = trajectories[:, :, :3]
finite_xyz = jnp.all(jnp.isfinite(trajectories_xyz), axis=2)
trajectories_xyz_safe = jnp.where(finite_xyz[:, :, None], trajectories_xyz, 0.0)
distances_flat = jax.vmap(boundary.evaluate_xyz)(trajectories_xyz_safe.reshape(-1, 3))
distances = distances_flat.reshape(nparticles, times_to_trace)

# --- Sigmoid-based indicator ---
softness = 50.0
loss_indicator = jax.nn.sigmoid(-softness * distances)
cumulative_loss_prob = jax.lax.associative_scan(jnp.maximum, loss_indicator, axis=1)

# --- True lost mask ---
lost_mask_true = jnp.any(distances < 0, axis=1)
loss_indicator_final = cumulative_loss_prob[:, -1]

# --- Compare lost indices ---
# True lost index: first time step where distance < 0
true_lost_indices = jnp.argmax(distances < 0, axis=1)
true_lost_any = jnp.any(distances < 0, axis=1)
true_lost_indices = jnp.where(true_lost_any, true_lost_indices, -1)

# Sigmoid lost index: first time step where cumulative_loss_prob > 0.5
sigmoid_lost_indices = jnp.argmax(cumulative_loss_prob > 0.5, axis=1)
sigmoid_lost_any = jnp.any(cumulative_loss_prob > 0.5, axis=1)
sigmoid_lost_indices = jnp.where(sigmoid_lost_any, sigmoid_lost_indices, -1)

# --- Check lost index match ---
lost_index_match = true_lost_indices == sigmoid_lost_indices
num_match = jnp.sum(lost_index_match)
print(f"\nLost index match count: {num_match} / {nparticles}")
if num_match == nparticles:
    print("All particles lost at the same index by both methods.")
else:
    print("Particles with mismatched lost index:")
    for i in range(nparticles):
        if not lost_index_match[i]:
            print(f"Particle {i:2d}: True lost at {true_lost_indices[i]}, Sigmoid lost at {sigmoid_lost_indices[i]}")
print("===========================================\n")

print("\n=== Diagnostic: Sigmoid vs True Lost Mask ===")
print("True lost mask (sum):", jnp.sum(lost_mask_true))
print("Sigmoid indicator (mean):", jnp.mean(loss_indicator_final))
print("Sigmoid indicator (min/max):", jnp.min(loss_indicator_final), jnp.max(loss_indicator_final))
print("Overlap (sigmoid > 0.5 & true lost):", jnp.sum((loss_indicator_final > 0.5) & lost_mask_true))
print("False positives (sigmoid > 0.5 & not lost):", jnp.sum((loss_indicator_final > 0.5) & (~lost_mask_true)))
print("False negatives (sigmoid <= 0.5 & true lost):", jnp.sum((loss_indicator_final <= 0.5) & lost_mask_true))
print("False negatives (sigmoid <= 0.5 & not lost):", jnp.sum((loss_indicator_final <= 0.5) & (~lost_mask_true)))
print("===========================================\n")

print("\n=== Lost Index Comparison ===")
for i in range(nparticles):
    print(f"Particle {i:2d}: True lost at {true_lost_indices[i]}, Sigmoid lost at {sigmoid_lost_indices[i]}")
print("===========================================\n")

# --- Compute Q_perp for each particle and timestep ---
# For GuidingCenterCollisionsMuFixed: vpar = trajectory[:, :, 3], mu = trajectory[:, :, 4]
mass = particles.mass
vpar = SPEED_OF_LIGHT * trajectories[:, :, 3]
mu = trajectories[:, :, 4] * (SPEED_OF_LIGHT**2 * mass)
abs_b_flat = jax.vmap(init_field.AbsB)(trajectories_xyz_safe.reshape(-1, 3))
abs_b = abs_b_flat.reshape(nparticles, times_to_trace)
v_perp_sq = 2.0 * mu * abs_b / mass
v_perp = jnp.sqrt(jnp.maximum(v_perp_sq, 0.0))
e_total_joules = 0.5 * mass * (vpar**2 + v_perp**2)
q_perp_joules = e_total_joules * v_perp

# --- Finite mask for q_perp_joules ---
finite_mask = jnp.isfinite(q_perp_joules)

# --- Q_perp calculation matching Tracing.lost_q_perp ---
normalization_factor = particles.energy * particles.total_speed
q_perp_scaled = q_perp_joules / (normalization_factor + 1e-12)

# Hard mask: lost at any timestep, finite q_perp
hard_mask = (distances < 0) & finite_mask
q_perp_lost_masked = jnp.where(hard_mask, q_perp_joules, 0.0)
q_perp_lost_sum_masked = jnp.sum(q_perp_lost_masked, axis=1)

# Sigmoid weighting: soft indicator, finite q_perp
sigmoid_weights = loss_indicator #* finite_mask
q_perp_lost_weighted = jnp.sum(sigmoid_weights * q_perp_lost_masked, axis=1)

# --- Differentiable softmax-based q_perp_lost ---
# Differentiable soft-first-crossing: use cumulative survival probability
softness_cross = .0
loss_indicator_cross = jax.nn.sigmoid(-softness_cross * distances)
# Compute soft-first-crossing weights: for lost particles, weight at first crossing timestep is 1, else softmax
true_lost_indices = jnp.argmax(distances < 0, axis=1)
true_lost_any = jnp.any(distances < 0, axis=1)
true_lost_indices = jnp.where(true_lost_any, true_lost_indices, -1)
ntimesteps = distances.shape[1]
particle_indices = jnp.arange(distances.shape[0])
# For lost particles: one-hot at first crossing
one_hot_first_cross = jnp.zeros_like(distances)
one_hot_first_cross = one_hot_first_cross.at[particle_indices, true_lost_indices].set(1.0)
# For not-lost particles: softmax over loss_indicator_cross
softmax_weights = jax.nn.softmax(loss_indicator_cross, axis=1)
# Find last valid finite timestep before loss for lost particles
xyz_finite = jnp.all(jnp.isfinite(trajectories_xyz), axis=2)
energy_finite = jnp.isfinite(e_total_joules)
valid_state_mask = xyz_finite & energy_finite
timestep_ids = jnp.arange(ntimesteps)[None, :]
pre_event_mask = valid_state_mask & (timestep_ids < true_lost_indices[:, None])
pre_event_indices = jnp.where(pre_event_mask, timestep_ids, -1)
last_finite_pre_event = jnp.max(pre_event_indices, axis=1)
record_indices = jnp.where(last_finite_pre_event >= 0, last_finite_pre_event, jnp.clip(true_lost_indices, 0, ntimesteps - 1))
# One-hot assignment at record_indices
one_hot_first_cross = jnp.zeros_like(distances)
one_hot_first_cross = one_hot_first_cross.at[particle_indices, record_indices].set(1.0)
one_hot_first_cross = jnp.where(true_lost_any[:, None], one_hot_first_cross, 0.0)
# Combine: lost particles get one-hot, not-lost get softmax
first_cross_weights = jnp.where(true_lost_any[:, None], one_hot_first_cross, softmax_weights)
# Apply finite mask to q_perp_joules
finite_mask = jnp.isfinite(q_perp_joules)
q_perp_joules_safe = jnp.where(finite_mask, q_perp_joules, 0.0)
# Normalize weights (should already sum to 1)
first_cross_weights = first_cross_weights / (jnp.sum(first_cross_weights, axis=1, keepdims=True) + 1e-12)
# q_perp_lost_soft_first is differentiable for not-lost particles, exact for lost
q_perp_lost_soft_first = jnp.sum(first_cross_weights * q_perp_joules_safe, axis=1)#/ (normalization_factor + 1e-12)

# First lost index (matches Tracing.lost_q_perp)
first_lost_indices = true_lost_indices
has_lost = first_lost_indices != -1
particle_indices = jnp.arange(nparticles)

# --- Q_perp_lost calculation: match Tracing.lost_q_perp exactly ---
# Use Tracing's lost mask and record_indices logic
lost_indices = jnp.argmax(distances < 0, axis=1)
lost_indices = jnp.where(jnp.any(distances < 0, axis=1), lost_indices, -1)
ntimesteps = times_to_trace
safe_indices = jnp.clip(lost_indices, 0, ntimesteps - 1)
particle_indices = jnp.arange(nparticles)
xyz_finite = jnp.all(jnp.isfinite(trajectories_xyz), axis=2)
energy_finite = jnp.isfinite(e_total_joules)
valid_state_mask = xyz_finite & energy_finite
timestep_ids = jnp.arange(ntimesteps)[None, :]
pre_event_mask = valid_state_mask & (timestep_ids < safe_indices[:, None])
pre_event_indices = jnp.where(pre_event_mask, timestep_ids, -1)
last_finite_pre_event = jnp.max(pre_event_indices, axis=1)
record_indices = jnp.where(last_finite_pre_event >= 0, last_finite_pre_event, safe_indices)
e_total_at_loss = e_total_joules[particle_indices, record_indices]
v_perp_at_loss = v_perp[particle_indices, record_indices]
q_perp_at_loss_raw = e_total_at_loss * v_perp_at_loss
has_lost = record_indices != -1
finite_at_loss = jnp.isfinite(q_perp_at_loss_raw)
q_perp_at_loss = jnp.where(has_lost & finite_at_loss, q_perp_at_loss_raw, 0.0)

print("\n=== Q_perp_lost Comparison (normalized) ===")
for i in range(nparticles):
    print(f"Particle {i:2d}: Hard mask sum = {q_perp_lost_sum_masked[i]:.6e}, First lost (e_total*v_perp) = {q_perp_at_loss[i]:.6e}, Sigmoid-weighted sum = {q_perp_lost_weighted[i]:.6e}, Soft-first-cross (e_total*v_perp) = {q_perp_lost_soft_first[i]:.6e}, Tracing.lost_q_perp = {tracing.lost_q_perp[i]:.6e}")
print("===========================================\n")
