
import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.dynamics import Particles, Tracing
from essos.coils import Coils, CreateEquallySpacedCurves
from jax import vmap, jit
#  In this exmple, `scipy.optimize.least_squares` is used, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares
from essos.losses import custom_loss
from essos.fields import BiotSavart



# Particle optimization parameters
# Optimization parameters
NPARTICLES = number_of_processors_to_use*10
MAXTIME_TRACING = 1e-4
NUMBER_COILS_PER_HALF_FIELD_PERIOD = 3
NUMBER_OF_FIELD_PERIODS = 2
MODEL = 'FullOrbit_Boris'
TIMESTEP=1.e-14
TRACE_TOLERANCE=1e-8
NUM_STEPS=1000


NPARTICLES_PLOT = number_of_processors_to_use*10
MAXTIME_TRACING_PLOT = 1e-4





""" Creating starting coils and surface """
N_COILS = 3
FOURIER_ORDER = 6
LARGE_R = 7.74
SMALL_R = 4.5
NFP = 2
N_SEGMENTS = 60
STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.84e7  # Amperes (optimization does not depend on current magnitude)
MAXIMUM_FUNCTION_EVALUATIONS =200

# Initialize coils
init_curves = CreateEquallySpacedCurves(n_curves=N_COILS,
                                   order=FOURIER_ORDER,
                                   R=LARGE_R, r=SMALL_R,
                                   n_segments=N_SEGMENTS,
                                   nfp=NFP, stellsym=STELLSYM)
init_coils = Coils(curves=init_curves, currents=jnp.array([COIL_CURRENT]*N_COILS))
init_field = BiotSavart(init_coils)

""" Setting the losses weights and targets """
LENGTH_WEIGHT = 1.; LENGTH_TARGET = 31.
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 0.4
BAXIS_WEIGHT = 1.; BAXIS_TARGET = 5.7
RADIAL_DRIFT_WEIGHT = 1.

def loss_particle_radial_drift(field, particles, timestep=1.e-8, maxtime=1e-5, num_steps=300, trace_tolerance=1e-5, model='GuidingCenterAdaptative',boundary=None):
    particles.to_full_orbit(field)
    tracing = Tracing(field=field, model=model, particles=particles, maxtime=maxtime,
                      timestep=timestep,times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance,boundary=boundary)
    xyz = tracing.trajectories[:,:, :3]
    R_axis=field.r_axis
    Z_axis=field.z_axis
    #Ideally here one would differentiate in time through diffrax !TODO
    r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(xyz[:,0])+jnp.square(xyz[:,1]))-R_axis+1.e-12)+jnp.square(xyz[:,2]-Z_axis+1.e-12))
    v_r_cross=jnp.diff(r_cross,axis=1)#/tracing.times_to_trace*tracing.maxtime     
    return (jnp.sum(jnp.square(jnp.average(v_r_cross,axis=1))))

def normB_axis(field, npoints=15):
    R_axis=field.r_axis
    phi_array = jnp.linspace(0, 2 * jnp.pi, npoints)
    B_axis = vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return B_axis

def loss_normB_axis_average(field,npoints=15, target_B=BAXIS_TARGET):
    B_axis = normB_axis(field, npoints)
    return jnp.abs(jnp.average(B_axis)-target_B)

def loss_length(field,length_target=LENGTH_TARGET):
    return jnp.mean(jnp.maximum(0, field.coils.length - length_target))

def loss_curvature(field,curvature_target=CURVATURE_TARGET):
    return jnp.mean(jnp.maximum(0, field.coils.curvature - curvature_target))



# Initialize particles
phi_array = jnp.linspace(0, 2*jnp.pi, NPARTICLES)
initial_xyz=jnp.array([LARGE_R*jnp.cos(phi_array), LARGE_R*jnp.sin(phi_array), 0*phi_array]).T
particles = Particles(initial_xyz=initial_xyz)








""" Defining custom losses """
L_radial_drift= custom_loss(loss_particle_radial_drift, "field", particles=particles, timestep=TIMESTEP, maxtime=MAXTIME_TRACING, num_steps=NUM_STEPS, trace_tolerance=TRACE_TOLERANCE, model=MODEL) 
L_B_axis= custom_loss(loss_normB_axis_average, "field")
L_length = custom_loss(loss_length, "field")
L_curvature = custom_loss(loss_curvature, "field")
""" Defining total loss + setting dependencies """
L_total = RADIAL_DRIFT_WEIGHT*L_radial_drift + L_B_axis + LENGTH_WEIGHT*L_length + CURVATURE_WEIGHT*L_curvature


L_total.dependencies = {"field": init_field}

""" Optimizing the total loss """
t_start = time()
res = least_squares(L_total, L_total.starting_dofs, L_total.grad, verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=MAXIMUM_FUNCTION_EVALUATIONS)
t_end = time()

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", L_total(L_total.starting_dofs))    
print("Loss after optimization:", L_total(res.x))

opt_field = L_total.dofs_to_pytree(res.x)["field"]
opt_coils = opt_field.coils

""" Plotting results """
phi_array_plot = jnp.linspace(0, 2*jnp.pi, NPARTICLES_PLOT)
initial_xyz_plot=jnp.array([LARGE_R*jnp.cos(phi_array_plot), LARGE_R*jnp.sin(phi_array_plot), 0*phi_array_plot]).T
particles_plot = Particles(initial_xyz=initial_xyz_plot)
particles_plot.to_full_orbit(opt_field)
tracing_initial = Tracing(field=init_field, particles=particles_plot, maxtime=MAXTIME_TRACING_PLOT, model=MODEL, times_to_trace=NUM_STEPS, timestep=TIMESTEP, atol=TRACE_TOLERANCE, rtol=TRACE_TOLERANCE)
tracing_optimized = Tracing(field=opt_field, particles=particles_plot, maxtime=MAXTIME_TRACING_PLOT, model=MODEL, times_to_trace=NUM_STEPS, timestep=TIMESTEP, atol=TRACE_TOLERANCE, rtol=TRACE_TOLERANCE)

# Plot trajectories, before and after optimization
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

init_coils.plot(ax=ax1, show=False)
tracing_initial.plot(ax=ax1, show=False)
for i, trajectory in enumerate(tracing_initial.trajectories):
    ax3.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')

ax3.set_xlabel('R (m)');ax3.set_ylabel('Z (m)');#ax3.legend()
opt_coils.plot(ax=ax2, show=False)
tracing_optimized.plot(ax=ax2, show=False)
for i, trajectory in enumerate(tracing_optimized.trajectories):
    ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')

ax4.set_xlabel('R (m)');ax4.set_ylabel('Z (m)');#ax4.legend()
plt.tight_layout()
plt.show()

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils
# coils = Coils.from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# tracing_initial.to_vtk('trajectories_initial')
# tracing_optimized.to_vtk('trajectories_final')
# coils_initial.to_vtk('coils_initial')
# coils_optimized.to_vtk('coils_optimized')