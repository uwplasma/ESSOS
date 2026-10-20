import os
import jax
import jax.numpy as jnp
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier
from essos.dynamics import Particles, Tracing
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY, ELECTRON_MASS, PROTON_MASS
from essos.background_species import BackgroundSpecies

# --- Copy of _escape_line_from_fixed_trajectories ---
def _escape_line_from_fixed_trajectories(field, trajectories, boundary, line_point, 
                                        line_direction, softness=30.0, location_softness=10.0, mode='mean', target_max_distance=0.0):
    """Probability-weighted mean distance to line at lost time for each particle, averaged over all particles."""
    trajectories_xyz = trajectories[:, :, :3]
    nparticles, ntimesteps = trajectories_xyz.shape[:2]
    finite_mask = jnp.all(jnp.isfinite(trajectories_xyz), axis=2)
    trajectories_xyz_safe = jnp.where(finite_mask[:, :, None], trajectories_xyz, 0.0)
    trajectories_flat = trajectories_xyz_safe.reshape(-1, 3)
    distances_flat = jax.vmap(boundary.evaluate_xyz)(trajectories_flat)
    distances = distances_flat.reshape(nparticles, ntimesteps)
    loss_indicator = 1.0 / (1.0 + jnp.exp(-softness * (-distances)))
    loss_indicator = jnp.where(finite_mask, loss_indicator, 0.0)
    line_dir_norm = line_direction / (jnp.linalg.norm(line_direction) + 1e-12)
    vec_to_point = trajectories_xyz_safe - line_point[None, None, :]
    projections = jnp.sum(vec_to_point * line_dir_norm[None, None, :], axis=2, keepdims=True) * line_dir_norm[None, None, :]
    perp_vectors = vec_to_point - projections
    distances_to_line = jnp.linalg.norm(perp_vectors, axis=2)  # shape (nparticles, ntimesteps)
    weighted_dist_per_particle = jnp.sum(loss_indicator * distances_to_line, axis=1) / (jnp.sum(loss_indicator, axis=1) + 1e-8)
    if mode == 'max':
        g_local = weighted_dist_per_particle - target_max_distance
        P = 20.0 / jnp.maximum(1e-8, target_max_distance)
        g_max = (1.0 / P) * jax.scipy.special.logsumexp(P * g_local) - (jnp.log(weighted_dist_per_particle.size) / P)
        return g_max
    return jnp.mean(weighted_dist_per_particle)

# --- Setup (mimics optimization script) ---
N_COILS = 4
FOURIER_ORDER = 6
number_of_processors_to_use = 1
nparticles_per_core = 2  # Small for test
nparticles = number_of_processors_to_use * nparticles_per_core

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), '../input_files', 'QI_opt_coils_essos_5.json')
coils = Coils.from_json(json_file)
init_field = BiotSavart(coils)

# Load surface and boundary
wout_file = os.path.join(os.path.dirname(__file__), '../input_files', 'wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc')
vmec = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=60, nphi=60, range_torus='full torus')
boundary = SurfaceClassifier(vmec, h=0.1)

# Species
number_species = 2
mass_array = jnp.array([ELECTRON_MASS / PROTON_MASS, 2])
charge_array = jnp.array([-1., 1.])
T0 = 6.6e+3
n0 = 4.e+20
n_array = jnp.array([n0, n0])
T_array = jnp.array([T0, T0])
species = BackgroundSpecies(number_species=number_species, mass_array=mass_array, charge_array=charge_array, n_array=n_array, T_array=T_array)

# Particles
vmec_axis = SurfaceRZFourier.from_wout_file(wout_file, s=0, ntheta=60, nphi=60, range_torus='full period')
particles = Particles.InitializeParticlesAroundSurfaceAxis(
    surface=vmec_axis,
    boundary_surface=vmec,
    n_particles=nparticles,
    distance_from_axis=0.7,
    distance_mode='fraction_to_boundary',
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    energy=FUSION_ALPHA_PARTICLE_ENERGY,
)

# Build a dummy line (as in optimization script)
line_point = jnp.array([vmec.rc[0], vmec.zs[0], 0.0])
line_direction = jnp.array([1.0, 0.0, 0.0])

# Get trajectories for test
tracing = Tracing(
    field=init_field,
    particles=particles,
    maxtime=1e-4,
    model='GuidingCenterCollisionsMuFixed',
    times_to_trace=3,
    timestep=1e-8,
    species=species,
    boundary=boundary,
)
trajectories = tracing.trajectories

# --- Gradient test ---
def loss_fn(field):
    return _escape_line_from_fixed_trajectories(
        field,
        trajectories,
        boundary,
        line_point,
        line_direction,
        softness=30.0,
        location_softness=10.0,
        mode='mean',
    )
grad_fn = jax.grad(loss_fn)
grad_val = grad_fn(init_field)
print('Gradient norm:', jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x) if hasattr(x, 'shape') else x, grad_val))
print('Gradient:', grad_val)
