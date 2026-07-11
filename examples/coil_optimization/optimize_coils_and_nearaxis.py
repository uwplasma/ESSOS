import os
number_of_processors_to_use = 1 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from pyqsc_jax.near_axis import near_axis
from essos.dynamics import Tracing
from essos.optimization import optimize_loss_function
from jax import vmap, jit
#  In this exmple, `scipy.optimize.least_squares` is used, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares
from essos.losses import custom_loss







""" Creating starting coils and surface """
N_COILS = 3; FOURIER_ORDER = 6; LARGE_R = 10; SMALL_R = 5.6; NFP = 3; N_SEGMENTS = 60; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)
tolerance_optimization = 1e-8
maximum_function_evaluations = 200



# Initialize Near-Axis field
rc=jnp.array([1.,0.01])
zs=jnp.array([0.,0.01])
etabar=-0.9
field_nearaxis_initial = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=NFP,order='r2')

# Initialize coils
current_on_each_coil = 17.e5*field_nearaxis_initial.B0/NFP/2.
number_of_field_periods = NFP
major_radius_coils = field_nearaxis_initial.R0[0]
minor_radius_coils = major_radius_coils/2.0
init_curves = CreateEquallySpacedCurves(n_curves=N_COILS,
                                   order=FOURIER_ORDER,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=N_SEGMENTS,
                                   nfp=number_of_field_periods, stellsym=STELLSYM)
init_coils = Coils(curves=init_curves, currents=jnp.array([current_on_each_coil]*N_COILS))
init_field = BiotSavart(init_coils)


""" Setting the losses weights and targets """
LENGTH_WEIGHT = 1.; LENGTH_TARGET = 4.
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 6.
B_DIFFERENCE_WEIGHT = 1.
GRADB_DIFFERENCE_WEIGHT = 1.
IOTA_TARGET = 0.41
IOTA_WEIGHT = 10.
R0_TARGET = field_nearaxis_initial.R0[0]
R0_WEIGHT = 10.



""" Creating the loss functions """
def near_axis_field_quantities(field_nearaxis):
    Raxis = field_nearaxis.R0
    Zaxis = field_nearaxis.Z0
    phi = field_nearaxis.phi
    Xaxis = Raxis*jnp.cos(phi)
    Yaxis = Raxis*jnp.sin(phi)
    points = jnp.array([Xaxis, Yaxis, Zaxis])
    B_nearaxis = field_nearaxis.B_axis.T
    gradB_nearaxis = field_nearaxis.grad_B_axis.T
    return points, B_nearaxis, gradB_nearaxis



def loss_B_difference_coils_near_axis(field, field_nearaxis):
    points, B_nearaxis, _ = near_axis_field_quantities(field_nearaxis)
    B_coils = vmap(field.B)(points.T)
    B_difference_loss = jnp.sum(jnp.abs(jnp.array(B_coils)-jnp.array(B_nearaxis)))
    return B_difference_loss

def loss_gradB_difference_coils_near_axis(field, field_nearaxis):
    points, _, gradB_nearaxis = near_axis_field_quantities(field_nearaxis)
    gradB_coils = vmap(field.dB_by_dX)(points.T)
    gradB_difference_loss = jnp.sum(jnp.abs(jnp.array(gradB_coils)-jnp.array(gradB_nearaxis)))
    return gradB_difference_loss

def loss_iota_near_axis(field_nearaxis,iota_target=IOTA_TARGET):
    return jnp.abs((field_nearaxis.iota - iota_target))

def loss_r0_near_axis(field_nearaxis, r0_target=R0_TARGET):
    return jnp.abs((field_nearaxis.R0[0] - r0_target))

def loss_length(field,length_target=LENGTH_TARGET):
    return jnp.mean(jnp.maximum(0, field.coils.length - length_target))

def loss_curvature(field,curvature_target=CURVATURE_TARGET):
    return jnp.mean(jnp.maximum(0, field.coils.curvature - curvature_target))


""" Defining custom losses """
L_B_difference = custom_loss(loss_B_difference_coils_near_axis, "field", "field_nearaxis")
L_gradB_difference = custom_loss(loss_gradB_difference_coils_near_axis, "field", "field_nearaxis")
L_length = custom_loss(loss_length, "field")
L_curvature = custom_loss(loss_curvature, "field")
L_iota = custom_loss(loss_iota_near_axis, "field_nearaxis")
L_r0 = custom_loss(loss_r0_near_axis, "field_nearaxis")


""" Defining total loss + setting dependencies """
L_total = B_DIFFERENCE_WEIGHT*L_B_difference + GRADB_DIFFERENCE_WEIGHT*L_gradB_difference + LENGTH_WEIGHT*L_length + CURVATURE_WEIGHT*L_curvature + IOTA_WEIGHT*L_iota+ R0_WEIGHT*L_r0


L_total.dependencies = {"field": init_field, "field_nearaxis": field_nearaxis_initial}

""" Optimizing the total loss """
t_start = time()
res = least_squares(L_total, L_total.starting_dofs, L_total.grad, verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=maximum_function_evaluations)
t_end = time()

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", L_total(L_total.starting_dofs))    
print("Loss after optimization:", L_total(res.x))

opt_field = L_total.dofs_to_pytree(res.x)["field"]
opt_coils = opt_field.coils

opt_field_nearaxis = L_total.dofs_to_pytree(res.x)["field_nearaxis"]




B_difference_initial = loss_B_difference_coils_near_axis(init_field, field_nearaxis_initial)
gradB_difference_initial = loss_gradB_difference_coils_near_axis(init_field, field_nearaxis_initial)

B_difference_optimized = loss_B_difference_coils_near_axis(opt_field, opt_field_nearaxis)
gradB_difference_optimized = loss_gradB_difference_coils_near_axis(opt_field, opt_field_nearaxis)


print(f'############################################')
print(f'Iota for initial near-axis: {field_nearaxis_initial.iota}')
print(f'Iota for optimized near-axis: {opt_field_nearaxis.iota}')
print(f'Maximum elongation for initial near-axis: {max(field_nearaxis_initial.elongation)}')
print(f'Maximum elongation for optimized near-axis: {max(opt_field_nearaxis.elongation)}')
print(f'Loss of B difference for initial near-axis: {B_difference_initial}')
print(f'Loss of B difference for optimized near-axis: {B_difference_optimized}')
print(f'Loss of gradB difference for initial near-axis: {gradB_difference_initial}')
print(f'Loss of gradB difference for optimized near-axis: {gradB_difference_optimized}')
print(f'Loss of R0 difference for initial near-axis: {loss_r0_near_axis(field_nearaxis_initial)}')
print(f'Loss of R0 difference for optimized near-axis: {loss_r0_near_axis(opt_field_nearaxis)}')


# Trace fieldlines
nfieldlines = 6
num_steps = 1000
tmax = 1.e-6
trace_tolerance = 1e-7

R0_initial   = jnp.linspace(field_nearaxis_initial.R0[0],   1.05*field_nearaxis_initial.R0[0],   nfieldlines)
R0_optimized = jnp.linspace(opt_field_nearaxis.R0[0], 1.05*opt_field_nearaxis.R0[0], nfieldlines)
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz_initial   = jnp.array([R0_initial*jnp.cos(phi0),   R0_initial*jnp.sin(phi0),   Z0]).T
initial_xyz_optimized = jnp.array([R0_optimized*jnp.cos(phi0), R0_optimized*jnp.sin(phi0), Z0]).T

time0 = time()
tracing_initial = Tracing(field=init_field, model='FieldLineAdaptative', initial_conditions=initial_xyz_initial,
                  maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance)
tracing_optimized = Tracing(field=opt_field, model='FieldLineAdaptative', initial_conditions=initial_xyz_optimized,
                  maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance)
print(f"Tracing took {time()-time0:.2f} seconds")

# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
init_coils.plot(ax=ax1, show=False)
field_nearaxis_initial.plot(ax=ax1, show=False, alpha=0.35)
tracing_initial.plot(ax=ax1, show=False)
opt_coils.plot(ax=ax2, show=False)
opt_field_nearaxis.plot(ax=ax2, show=False, alpha=0.35)
tracing_optimized.plot(ax=ax2, show=False)
plt.show()

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils
# coils = Coils.from_json("stellarator_coils.json")

# # Save results in vtk format to analyze in Paraview
# init_coils.to_vtk('coils_initial')
# opt_coils.to_vtk('coils_optimized')
# tracing_initial.to_vtk('trajectories_initial')
# tracing_optimized.to_vtk('trajectories_final')