import os
number_of_processors_to_use = 2 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.surfaces import BdotN_over_B
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec, BiotSavart
from essos.objective_functions import loss_BdotN
from essos.optimization import optimize_loss_function

# Optimization parameters
max_coil_length = 40
max_coil_curvature = 0.5
order_Fourier_series_coils = 3
number_coil_points = order_Fourier_series_coils*15
maximum_function_evaluations = 1000
number_coils_per_half_field_period = 3
tolerance_optimization = 1e-7
ntheta=35
nphi=35

# Initialize VMEC field
vmec = Vmec(os.path.join(os.path.dirname(__name__), 'input_files',
             'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
            ntheta=ntheta, nphi=nphi, range_torus='half period')

# Initialize coils
current_on_each_coil = 1
number_of_field_periods = vmec.nfp
major_radius_coils = vmec.r_axis
minor_radius_coils = vmec.r_axis/1.8
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
time0 = time()
coils_optimized = optimize_loss_function(loss_BdotN, initial_dofs=coils_initial.x, coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, vmec=vmec,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)
print(f"Optimization took {time()-time0:.2f} seconds")


BdotN_over_B_initial = BdotN_over_B(vmec.surface, BiotSavart(coils_initial))
BdotN_over_B_optimized = BdotN_over_B(vmec.surface, BiotSavart(coils_optimized))
curvature=jnp.mean(BiotSavart(coils_optimized).coils.curvature, axis=1)
length=jnp.max(jnp.ravel(BiotSavart(coils_optimized).coils.length))
print(f"Mean curvature: ",curvature)
print(f"Length:", length)
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")

# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils_initial.plot(ax=ax1, show=False)
vmec.surface.plot(ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False)
vmec.surface.plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()

# # Save the coils to a json file
coils_optimized.to_json("stellarator_coils_normal.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# from essos.fields import BiotSavart
# vmec.surface.to_vtk('surface_initial', field=BiotSavart(coils_initial))
# vmec.surface.to_vtk('surface_final',   field=BiotSavart(coils_optimized))
# coils_initial.to_vtk('coils_initial')
# coils_optimized.to_vtk('coils_optimized')



# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing


field_optimized=BiotSavart(coils_optimized)
tmax = 100000000000
nfieldlines_per_core = 13
nfieldlines = nfieldlines_per_core * number_of_processors_to_use
R0 = jnp.linspace(12.2, 13.5, nfieldlines)
trace_tolerance = 1e-7
num_steps = 60000

Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T

time0 = time()
tracing = block_until_ready(Tracing(
    field=field_optimized,
    model='FieldLineAdaptative',
    initial_conditions=initial_xyz,
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance
))
print(f"ESSOS tracing took {time() - time0:.2f} seconds")


def compute_rz_on_phi(surface, theta, phi=0.0):
    angles = jnp.outer(theta, surface.xm) - phi * surface.xn
    R = jnp.sum(surface.rmnc_interp * jnp.cos(angles), axis=1)
    Z = jnp.sum(surface.zmns_interp * jnp.sin(angles), axis=1)
    return R, Z


theta = jnp.linspace(0, 2 * jnp.pi, 200)

# # Contours from true VMEC surface
R0_true, Z0_true = compute_rz_on_phi(vmec.surface, theta, phi=0.0)
R90_true, Z90_true = compute_rz_on_phi(vmec.surface, theta, phi=jnp.pi/2)

fig, ax = plt.subplots(figsize=(6, 6))

tracing.poincare_plot(ax=ax, show=False, shifts=[0, jnp.pi / 2])
ax.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
ax.plot(R90_true, Z90_true, color='blue', linestyle='--', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")

ax.set_xlabel("R")
ax.set_ylabel("Z")
ax.set_title("Poincaré + Surfaces Comparison @ φ = 0 and π/2")
ax.legend()
ax.axis("equal")
plt.tight_layout()
plt.savefig('poincare_coils.png', dpi=300)
