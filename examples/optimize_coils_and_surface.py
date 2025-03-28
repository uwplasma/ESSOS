import os
number_of_processors_to_use = 12 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from essos.fields import BiotSavart, near_axis
from essos.surfaces import BdotN_over_B, SurfaceRZFourier, B_on_surface
from essos.coils import Coils, CreateEquallySpacedCurves, Curves
from essos.optimization import optimize_loss_function, new_nearaxis_from_x_and_old_nearaxis
from essos.objective_functions import (loss_coil_curvature, difference_B_gradB_onaxis, loss_coil_length)
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, devices, device_put, grad, debug
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from time import time
import matplotlib.pyplot as plt

mesh = Mesh(devices(), ("dev",))
sharding = NamedSharding(mesh, PartitionSpec("dev", None))

ntheta=24
nphi=24
input = os.path.join('input_files','input.rotating_ellipse')
surface_initial = SurfaceRZFourier(input, ntheta=ntheta, nphi=nphi, range_torus='half period')

# Optimization parameters
max_coil_length = 40
max_coil_curvature = 0.4
order_Fourier_series_coils = 4
number_coil_points = max(70,order_Fourier_series_coils*10)
maximum_function_evaluations = 25
number_coils_per_half_field_period = 3
tolerance_optimization = 1e-8
target_B_on_axis = 5.7
target_iota = 0.41

# Initialize coils
number_of_field_periods = surface_initial.nfp
current_on_each_coil = 1.3e8/(number_coils_per_half_field_period*number_of_field_periods)
major_radius_coils = surface_initial.dofs[0]
minor_radius_coils = major_radius_coils/1.8
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Initialize near-axis
rc=jnp.array([1, 0.1, 1e-3])*major_radius_coils
zs=jnp.array([0,-0.1, 1e-3])*major_radius_coils
etabar=-0.8/major_radius_coils
field_nearaxis_initial = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=number_of_field_periods, B0=target_B_on_axis)

# # Find etabar that leads to max iota and min elongation
# etabar_array = jnp.linspace(0.01,0.15, 15)
# iota_array = []
# elongation_array = []
# for etabar in etabar_array:
#     field_nearaxis_initial = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=number_of_field_periods, B0=target_B_on_axis)
#     iota_array.append(field_nearaxis_initial.iota)
#     elongation_array.append(jnp.max(field_nearaxis_initial.elongation))
# plt.plot(etabar_array, iota_array, label='iota');plt.plot(etabar_array, jnp.array(elongation_array)/max(elongation_array), label='elongation');plt.xlabel('etabar');plt.legend();plt.show();exit()

print(f"Initial iota near-axis: {field_nearaxis_initial.iota} and max elongation: {max(field_nearaxis_initial.elongation)} at etabar={etabar}")
print(f"Mean Magnetic field on surface: {jnp.mean(jnp.linalg.norm(B_on_surface(surface_initial, BiotSavart(coils_initial)), axis=2))}")

def grad_AbsB_on_surface(surface, field):
    gamma_sharded = device_put(surface.gamma.reshape(surface.nphi * surface.ntheta, 3), sharding)
    dAbsB_by_dX_on_surface = jit(vmap(field.dAbsB_by_dX), in_shardings=sharding, out_shardings=sharding)(gamma_sharded)
    dAbsB_by_dX_on_surface = dAbsB_by_dX_on_surface.reshape(surface.nphi, surface.ntheta, 3)
    return dAbsB_by_dX_on_surface
def B_dot_GradAbsB(points, field):
    return jnp.sum(field.B(points) * field.dAbsB_by_dX(points), axis=-1)
def grad_B_dot_GradAbsB(points, field):
    return grad(B_dot_GradAbsB, argnums=0)(points, field)
def grad_B_dot_GradAbsB_on_surface(surface, field):
    gamma_sharded = device_put(surface.gamma.reshape(surface.nphi * surface.ntheta, 3), sharding)
    partial_grad_B_dot_GradAbsB = partial(grad_B_dot_GradAbsB, field=field)
    grad_B_dot_GradAbsB_on_surface = jit(vmap(partial_grad_B_dot_GradAbsB), in_shardings=sharding, out_shardings=sharding)(gamma_sharded)
    grad_B_dot_GradAbsB_on_surface = grad_B_dot_GradAbsB_on_surface.reshape(surface.nphi, surface.ntheta, 3)
    return grad_B_dot_GradAbsB_on_surface
def loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface, field):
    gradAbsB_surface = grad_AbsB_on_surface(surface, field)
    grad_B_dot_GradB_surface = grad_B_dot_GradAbsB_on_surface(surface, field)
    normal_cross_GradB_surface = jnp.cross(surface.normal, gradAbsB_surface, axisa=-1, axisb=-1)
    normal_cross_GradB_dot_grad_B_dot_GradB_surface = jnp.sum(normal_cross_GradB_surface * grad_B_dot_GradB_surface, axis=-1)
    return normal_cross_GradB_dot_grad_B_dot_GradB_surface

@partial(jit, static_argnums=(1, 5, 6, 7, 8, 9, 10))
def loss_coils_and_surface(x, surface_all, field_nearaxis, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.5, target_iota=0.41):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_currents = x[len_dofs_curves_ravelled:-len(surface_all.x)-len(field_nearaxis.x)]
    new_dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], (dofs_curves.shape))
    field = BiotSavart(Coils(curves=Curves(new_dofs_curves, n_segments, nfp, stellsym), currents=dofs_currents*currents_scale))
    surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
    surface.dofs = x[-len(surface_all.x)-len(field_nearaxis.x):-len(field_nearaxis.x)]
    field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(x[-len(field_nearaxis.x):], field_nearaxis)
    B_difference, gradB_difference = difference_B_gradB_onaxis(field_nearaxis, field)
    B_difference_loss              = jnp.ravel(B_difference)
    gradB_difference_loss          = jnp.ravel(gradB_difference)
    bdotn_over_b_loss              = jnp.ravel(BdotN_over_B(surface, field))
    mean_cross_sectional_area_loss = jnp.array([surface.mean_cross_sectional_area()-surface_all.mean_cross_sectional_area()])
    # axis_loss                      = jnp.array([jnp.abs(jnp.max(jnp.array([jnp.abs(field_nearaxis.rc[0] - surface.dofs[0]) - 0.1 * field_nearaxis.rc[0], 0])))])
    iota_loss                      = 1e1*jnp.array([field_nearaxis.iota-target_iota])
    coil_length_loss               = 1e-1*jnp.array([jnp.max(jnp.concatenate([loss_coil_length(field)-max_coil_length,jnp.array([0])]))])
    coil_curvature_loss            = jnp.array([jnp.max(jnp.concatenate([loss_coil_curvature(field)-max_coil_curvature,jnp.array([0])]))])
    # debug.print("######################")
    # debug.print("bdotn_over_b_loss=             {}", jnp.sum(bdotn_over_b_loss**2))
    # debug.print("B_difference_loss=             {}", jnp.sum(B_difference_loss**2))
    # debug.print("gradB_difference_loss=         {}", jnp.sum(gradB_difference_loss**2))
    # debug.print("iota_loss=                     {}", jnp.sum(iota_loss**2))
    # debug.print("axis_loss=                     {}", jnp.sum(axis_loss**2))
    # debug.print("mean_cross_sectional_area_loss={}", jnp.sum(mean_cross_sectional_area_loss**2))
    # debug.print('coil_length_loss=              {}', jnp.sum(coil_length_loss**2))
    # debug.print('coil_curvature_loss=           {}', jnp.sum(coil_curvature_loss**2))
    return jnp.concatenate([
        B_difference_loss, gradB_difference_loss, iota_loss,# axis_loss,
        coil_length_loss, coil_curvature_loss, mean_cross_sectional_area_loss, bdotn_over_b_loss
        ])

@partial(jit, static_argnums=(1, 5, 6, 7, 8, 9, 10))
def loss_coils_and_surface_qs(x, surface_all, field_nearaxis, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.5, target_iota=0.41):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_currents = x[len_dofs_curves_ravelled:-len(surface_all.x)-len(field_nearaxis.x)]
    new_dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], (dofs_curves.shape))
    field = BiotSavart(Coils(curves=Curves(new_dofs_curves, n_segments, nfp, stellsym), currents=dofs_currents*currents_scale))
    surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
    surface.dofs = x[-len(surface_all.x)-len(field_nearaxis.x):-len(field_nearaxis.x)]
    field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(x[-len(field_nearaxis.x):], field_nearaxis)
    B_difference, gradB_difference = difference_B_gradB_onaxis(field_nearaxis, field)
    normal_cross_GradB_dot_grad_B_dot_GradB_surface = 1e-4 * jnp.ravel(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface, field))
    B_difference_loss              = jnp.ravel(B_difference)
    gradB_difference_loss          = jnp.ravel(gradB_difference)
    bdotn_over_b_loss              = jnp.ravel(BdotN_over_B(surface, field))
    mean_cross_sectional_area_loss = jnp.array([surface.mean_cross_sectional_area()-surface_all.mean_cross_sectional_area()])
    # axis_loss                      = jnp.array([jnp.abs(jnp.max(jnp.array([jnp.abs(field_nearaxis.rc[0] - surface.dofs[0]) - 0.1 * field_nearaxis.rc[0], 0])))])
    iota_loss                      = 1e1*jnp.array([field_nearaxis.iota-target_iota])
    coil_length_loss               = 1e-1*jnp.array([jnp.max(jnp.concatenate([loss_coil_length(field)-max_coil_length,jnp.array([0])]))])
    coil_curvature_loss            = jnp.array([jnp.max(jnp.concatenate([loss_coil_curvature(field)-max_coil_curvature,jnp.array([0])]))])
    # debug.print('normal_cross_GradB_dot_grad_B_dot_GradB_surface= {}', jnp.sum(normal_cross_GradB_dot_grad_B_dot_GradB_surface**2))
    return jnp.concatenate([
        B_difference_loss, gradB_difference_loss, iota_loss,# axis_loss,
        coil_length_loss, coil_curvature_loss, mean_cross_sectional_area_loss, bdotn_over_b_loss,
        normal_cross_GradB_dot_grad_B_dot_GradB_surface
        ])

# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
time0 = time()
initial_dofs = jnp.concatenate((coils_initial.x, surface_initial.x, field_nearaxis_initial.x))
coils_optimized, surface_optimized, field_nearaxis_optimized = optimize_loss_function(loss_coils_and_surface, initial_dofs=initial_dofs, coils=coils_initial,
                                tolerance_optimization=tolerance_optimization, maximum_function_evaluations=maximum_function_evaluations, target_iota=target_iota,
                                surface_all=surface_initial, field_nearaxis=field_nearaxis_initial, max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature)
initial_dofs = jnp.concatenate((coils_optimized.x, surface_optimized.x, field_nearaxis_optimized.x))
coils_optimized, surface_optimized, field_nearaxis_optimized = optimize_loss_function(loss_coils_and_surface_qs, initial_dofs=initial_dofs, coils=coils_optimized,
                                tolerance_optimization=tolerance_optimization, maximum_function_evaluations=maximum_function_evaluations, target_iota=target_iota,
                                surface_all=surface_optimized, field_nearaxis=field_nearaxis_optimized, max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature)
print(f"Optimization took {time()-time0:.2f} seconds")
print(f'############################################')
print(f"Mean B on surface:                  {jnp.mean(jnp.linalg.norm(B_on_surface(surface_optimized, BiotSavart(coils_optimized)), axis=2))}")
print(f"Initial   max(BdotN/B):             {jnp.max(BdotN_over_B(surface_initial, BiotSavart(coils_initial))):.2e}")
print(f"Optimized max(BdotN/B):             {jnp.max(BdotN_over_B(surface_optimized, BiotSavart(coils_optimized))):.2e}")
print(f'Initial   iota on-axis:             {field_nearaxis_initial.iota}')
print(f'Optimized iota on-axis:             {field_nearaxis_optimized.iota}')
print(f'Initial   max(elongation):          {max(field_nearaxis_initial.elongation)}')
print(f'Optimized max(elongation):          {max(field_nearaxis_optimized.elongation)}')
print(f"Initial   coils length:             {coils_initial.length[:number_coils_per_half_field_period]}")
print(f"Optimized coils length:             {coils_optimized.length[:number_coils_per_half_field_period]}")
print(f"Initial   coils curvature:          {jnp.mean(coils_initial.curvature, axis=1)[:number_coils_per_half_field_period]}")
print(f"Optimized coils curvature:          {jnp.mean(coils_optimized.curvature, axis=1)[:number_coils_per_half_field_period]}")
B_difference_initial, gradB_difference_initial = difference_B_gradB_onaxis(field_nearaxis_initial, BiotSavart(coils_initial))
B_difference_optimized, gradB_difference_optimized = difference_B_gradB_onaxis(field_nearaxis_optimized, BiotSavart(coils_optimized))
print(f'Initial   B on axis difference:     {jnp.sum(jnp.abs(B_difference_initial))}')
print(f'Optimized B on axis difference:     {jnp.sum(jnp.abs(B_difference_optimized))}')
print(f'Initial   gradB on axis difference: {jnp.sum(jnp.abs(gradB_difference_initial))}')
print(f'Optimized gradB on axis difference: {jnp.sum(jnp.abs(gradB_difference_optimized))}')
print(f'Initial cross sectional area:       {surface_initial.mean_cross_sectional_area()}')
print(f'Optimized cross sectional area:     {surface_optimized.mean_cross_sectional_area()}')
print(f'Initial QS metric:                  {jnp.sum(jnp.abs(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface_initial, BiotSavart(coils_initial))))}')
print(f'Optimized QS metric:                {jnp.sum(jnp.abs(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface_optimized, BiotSavart(coils_optimized))))}')
# print(f'initial axis loss: {jnp.abs(jnp.max(jnp.array([jnp.abs(field_nearaxis_initial.rc[0] - surface_initial.dofs[0]) - 0.2 * field_nearaxis_initial.rc[0], 0])))}')
# print(f'Optimized axis loss: {jnp.abs(jnp.max(jnp.array([jnp.abs(field_nearaxis_optimized.rc[0] - surface_optimized.dofs[0]) - 0.2 * field_nearaxis_optimized.rc[0], 0])))}')
print(f'Resulting near-axis rc={field_nearaxis_optimized.rc}, zs={field_nearaxis_optimized.zs}, etabar={field_nearaxis_optimized.etabar}')

# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils_initial.plot(ax=ax1, show=False)
surface_initial.plot(ax=ax1, show=False)
field_nearaxis_initial.plot(r=major_radius_coils/12, ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False)
surface_optimized.plot(ax=ax2, show=False)
field_nearaxis_optimized.plot(r=major_radius_coils/12, ax=ax2, show=False)
plt.tight_layout()
plt.show()

# Save the surface to a VMEC file
surface_optimized.to_vmec('input.optimized')

# Save results in vtk format to analyze in Paraview
surface_initial.to_vtk('initial_surface', field=BiotSavart(coils_initial))
coils_initial.to_vtk('initial_coils')
field_nearaxis_initial.to_vtk('initial_field_nearaxis', r=major_radius_coils/12, field=BiotSavart(coils_initial))
surface_optimized.to_vtk('optimized_surface', field=BiotSavart(coils_optimized))
coils_optimized.to_vtk('optimized_coils')
field_nearaxis_optimized.to_vtk('optimized_field_nearaxis', r=major_radius_coils/12, field=BiotSavart(coils_optimized))

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")