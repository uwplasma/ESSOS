import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import jax
print(jax.devices())
from time import time
from jax import block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart,SurfaceRZFourier
from essos.surfaces import SurfaceClassifier, B_on_surface
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ONE_EV
from essos.dynamics import Tracing, Particles
from essos.objective_functions import normB_axis
import matplotlib.pyplot as plt


# Input files
# Input files
json_file = '/home/exouser/ESSOS/examples/input_files/QI_opt_coils_essos_5.json'
wout_file = '/home/exouser/ESSOS/examples/input_files/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc'

# Parameters
number_of_processors_to_use = 1
nparticles_per_core = 100
nparticles = number_of_processors_to_use * nparticles_per_core
energy = FUSION_ALPHA_PARTICLE_ENERGY
tmax = 1.e-4
timestep = 1.e-8
times_to_trace = 1000
atol = 1.e-7
rtol = 1.e-7

# Load coils and surfaces
coils = Coils.from_json(json_file)
vmec = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=60, nphi=60, range_torus='full torus')
vmec_axis = SurfaceRZFourier.from_wout_file(wout_file, s=0, ntheta=60, nphi=60, range_torus='full period')
boundary = SurfaceClassifier(vmec, h=0.1)


# Function to compute axis position at given phi
def compute_axis_point(phi):
    """Compute magnetic axis position at toroidal angle phi using s=0 surface"""
    angles = xn_axis * phi
    R_val = jnp.sum(rc_axis * jnp.cos(angles))
    Z = -jnp.sum(zs_axis * jnp.sin(angles))
    return R_val, Z

# Extract magnetic axis from s=0 surface (m=0 modes give the axis in cylindrical coordinates)
m0_mask_axis = vmec_axis.xm == 0
rc_axis = vmec_axis.rc[m0_mask_axis]
zs_axis = vmec_axis.zs[m0_mask_axis]
xn_axis = vmec_axis.xn[m0_mask_axis]


field_aux = BiotSavart(coils, use_interpolator=False)
phi_array = jnp.linspace(0, 2 * jnp.pi, 200)
B_axis_old = jnp.array([field_aux.AbsB(jnp.array([R_val * jnp.cos(phi), R_val * jnp.sin(phi), Z])) 
                        for phi in phi_array 
                        for R_val, Z in [compute_axis_point(phi)]])
B_target = 5.7
scale_factor = B_target / jnp.mean(B_axis_old)
#scale_factor = 1/(4.*jnp.pi*1.e-7) # No scaling for now, to compare with previous results
coils.dofs_currents = coils.dofs_currents * scale_factor   




# Initialize particles
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

# Test configurations
interp_methods = ['direct', 'trilinear', 'spline']
resolutions = [(10,10,10), (20,20,20)]
results = {}

for method in interp_methods:
    for res in resolutions:
        print(f"\nTesting BiotSavart: method={method}, resolution={res}")
        if method == 'direct':
            field = BiotSavart(coils, use_interpolator=False)
            field.B(jnp.array([[1.0, 0.0, 0.0]]))  # Warm up call to trigger JIT compilation
        else:
            t0 = time()
            field = BiotSavart(coils, use_interpolator=True, interp_method=method, interp_resolution=res)
            field.B(jnp.array([[1.0, 0.0, 0.0]]))  # Warm up call to trigger JIT compilation
            t1 = time()
            print(f"Field interpolator build time: {t1-t0:.2f} s (resolution: {res})")
        # Trace
        time0 = time()
        tracing = Tracing(field=field, model='GuidingCenterAdaptative', particles=particles,
                          maxtime=tmax, timestep=timestep, times_to_trace=times_to_trace,
                          atol=atol, rtol=rtol, boundary=boundary)
        elapsed = time() - time0
        print(f"Tracing time: {elapsed:.2f} s")
        # Store results
        results[(method, res)] = {
            'time': elapsed,
            'trajectories': tracing.trajectories,
            'loss_fraction': tracing.loss_fractions[-1]
        }

# Compare results
for key, val in results.items():
    method, res = key
    print(f"Method: {method}, Resolution: {res}, Time: {val['time']:.2f} s, Loss Fraction: {val['loss_fraction']*100:.2f}%")

# Plot example trajectories

# Plot trajectories for multiple particles in subplots with different styles

# Plot trajectories for multiple particles in subplots with unique point styles
n_particles_to_plot = 4
marker_styles = ['o', 's', '^', 'x', 'D', 'v', '>', '<', 'p', '*', 'h', 'H', '+', 'd', '|', '_']
fig, axs = plt.subplots(1, n_particles_to_plot, figsize=(16, 4), sharey=True)
for i in range(n_particles_to_plot):
    for idx, (key, val) in enumerate(results.items()):
        traj = val['trajectories'][i]
        marker = marker_styles[(i * len(results) + idx) % len(marker_styles)]
        axs[i].scatter(jnp.sqrt(traj[:,0]**2 + traj[:,1]**2), traj[:,2], marker=marker, s=10, label=f"{key[0]}-{key[1]}")
    axs[i].set_xlabel('R (m)')
    axs[i].set_title(f'Particle {i+1}')
axs[0].set_ylabel('Z (m)')
for ax in axs:
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('biot_savart_interpolation_comparison_subplots_points.png', dpi=150)
print("Trajectory comparison plot saved as biot_savart_interpolation_comparison_subplots_points.png")
