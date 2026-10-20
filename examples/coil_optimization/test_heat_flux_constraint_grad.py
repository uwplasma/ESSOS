import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
import jax.numpy as jnp
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, SurfaceClassifier
from essos.dynamics import Particles
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.background_species import BackgroundSpecies
import essos.augmented_lagrangian as alm

# Load input files
json_file = '/home/exouser/ESSOS/examples/input_files/QI_opt_coils_essos_5.json'
wout_file = '/home/exouser/ESSOS/examples/input_files/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc'
coils = Coils.from_json(json_file)
init_field = BiotSavart(coils)
vmec = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=20, nphi=20, range_torus='full torus')
boundary = SurfaceClassifier(vmec, h=0.1)

# Initialize particles
nparticles = 20
tmax = 1e-4
timestep = 1e-8
times_to_trace = 100
energy = FUSION_ALPHA_PARTICLE_ENERGY
particles = Particles.InitializeParticlesAroundSurfaceAxis(
    surface=vmec,
    boundary_surface=vmec,
    n_particles=nparticles,
    distance_from_axis=0.7,
    distance_mode='fraction_to_boundary',
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    energy=energy,
)
species = BackgroundSpecies(number_species=2, mass_array=jnp.array([9.10938356e-31/1.6726219e-27, 2]), charge_array=jnp.array([-1., 1.]), n_array=jnp.array([4.e+20, 4.e+20]), T_array=jnp.array([6.6e+3, 6.6e+3]))

# Import constraint from main script
from optimize_coils_particle_confinement_loss_fraction_augmented_lagrangian_coils_heat_flux_vpj import C_heat_flux_constraint

# Test gradient with respect to field (coils.dofs_currents)
def test_heat_flux_constraint_grad():
    field = BiotSavart(coils)
    def constraint_loss(currents):
        coils.dofs_currents = currents
        field = BiotSavart(coils)
        return C_heat_flux_constraint.loss(None, field)
    grad_fn = jax.grad(constraint_loss)
    grad = grad_fn(coils.dofs_currents)
    print("Gradient shape:", grad.shape)
    print("Gradient (sample):", grad[:10])

if __name__ == "__main__":
    test_heat_flux_constraint_grad()
