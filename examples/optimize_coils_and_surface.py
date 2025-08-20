import os
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from essos.fields import BiotSavart, near_axis
from essos.dynamics import Particles, Tracing
from essos.surfaces import BdotN_over_B, SurfaceRZFourier, B_on_surface
from essos.coils import Coils, CreateEquallySpacedCurves, Curves
from essos.optimization import optimize_loss_function, new_nearaxis_from_x_and_old_nearaxis
from essos.objective_functions import (loss_coil_curvature, difference_B_gradB_onaxis,
                                       loss_coil_length,field_from_dofs)
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, devices, device_put, grad, debug
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from time import time
import matplotlib.pyplot as plt

mesh = Mesh(devices(), ("dev",))
sharding = NamedSharding(mesh, PartitionSpec("dev", None))

ntheta=30
nphi=30
input = os.path.join('input_files','input.rotating_ellipse')
surface_initial = SurfaceRZFourier(input, ntheta=ntheta, nphi=nphi, range_torus='half period')

# Optimization parameters
max_coil_length = 38
max_coil_curvature = 0.3
order_Fourier_series_coils = 5
number_coil_points = order_Fourier_series_coils*10
maximum_function_evaluations = 20#600
number_coils_per_half_field_period = 4
tolerance_optimization = 1e-7
target_B_on_axis = 5.7

nparticles = number_of_processors_to_use
maxtime_tracing = 4e-5
num_steps=300
trace_tolerance=1e-5
model = 'GuidingCenterAdaptative'

# Initialize coils
current_on_each_coil = 1.714e7
number_of_field_periods = surface_initial.nfp
major_radius_coils = surface_initial.dofs[0]
minor_radius_coils = major_radius_coils/1.3
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Initialize near-axis
rc=jnp.array([1, 0.045])*major_radius_coils
zs=jnp.array([0,-0.045])*major_radius_coils
etabar=-0.9/major_radius_coils
field_nearaxis_initial = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=number_of_field_periods, B0=target_B_on_axis)

print(f"Mean Magnetic field on surface: {jnp.mean(jnp.linalg.norm(B_on_surface(surface_initial, BiotSavart(coils_initial)), axis=2))}")

# Initialize particles
# Xaxis = field_nearaxis_initial.R0*jnp.cos(field_nearaxis_initial.phi)
# Yaxis = field_nearaxis_initial.R0*jnp.sin(field_nearaxis_initial.phi)
# initial_xyz = jnp.array([Xaxis, Yaxis, field_nearaxis_initial.Z0]).T[:nparticles]
# particles = Particles(initial_xyz=initial_xyz, field=BiotSavart(coils_initial))
# tracing_initial = Tracing(field=coils_initial, particles=particles, maxtime=maxtime_tracing, model=model, timesteps=num_steps)

# # Plot initial state
# fig = plt.figure(figsize=(9, 8))
# ax = fig.add_subplot(111, projection='3d')
# tracing_initial.plot(ax=ax, show=False)
# field_nearaxis_initial.plot(r=major_radius_coils/12, ax=ax, show=False)
# coils_initial.plot(ax=ax, show=False)
# surface_initial.plot(ax=ax, show=False)
# plt.show()

# @partial(jit, static_argnames=['surface','field'])
def grad_AbsB_on_surface(surface, field):
    ntheta = surface.ntheta
    nphi = surface.nphi
    gamma = surface.gamma
    gamma_reshaped = gamma.reshape(nphi * ntheta, 3)
    gamma_sharded = device_put(gamma_reshaped, sharding)
    dAbsB_by_dX_on_surface = jit(vmap(field.dAbsB_by_dX), in_shardings=sharding, out_shardings=sharding)(gamma_sharded)
    dAbsB_by_dX_on_surface = dAbsB_by_dX_on_surface.reshape(nphi, ntheta, 3)
    return dAbsB_by_dX_on_surface

# @partial(jit, static_argnames=['field'])
def B_dot_GradAbsB(points, field):
    B = field.B(points)
    GradAbsB = field.dAbsB_by_dX(points)
    B_dot_GradAbsB = jnp.sum(B * GradAbsB, axis=-1)
    return B_dot_GradAbsB

# @partial(jit, static_argnames=['field'])
def grad_B_dot_GradAbsB(points, field):
    return grad(B_dot_GradAbsB, argnums=0)(points, field)

# @partial(jit, static_argnames=['surface','field'])
def grad_B_dot_GradAbsB_on_surface(surface, field):
    ntheta = surface.ntheta
    nphi = surface.nphi
    gamma = surface.gamma
    gamma_reshaped = gamma.reshape(nphi * ntheta, 3)
    gamma_sharded = device_put(gamma_reshaped, sharding)
    partial_grad_B_dot_GradAbsB = partial(grad_B_dot_GradAbsB, field=field)
    grad_B_dot_GradAbsB_on_surface = jit(vmap(partial_grad_B_dot_GradAbsB), in_shardings=sharding, out_shardings=sharding)(gamma_sharded)
    grad_B_dot_GradAbsB_on_surface = grad_B_dot_GradAbsB_on_surface.reshape(nphi, ntheta, 3)
    return grad_B_dot_GradAbsB_on_surface

# @partial(jit, static_argnames=['surface','field'])
def loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface, field):
    gradAbsB_surface = grad_AbsB_on_surface(surface, field)
    grad_B_dot_GradB_surface = grad_B_dot_GradAbsB_on_surface(surface, field)
    normal_cross_GradB_surface = jnp.cross(surface.normal, gradAbsB_surface, axisa=-1, axisb=-1)
    normal_cross_GradB_dot_grad_B_dot_GradB_surface = jnp.sum(normal_cross_GradB_surface * grad_B_dot_GradB_surface, axis=-1)
    return normal_cross_GradB_dot_grad_B_dot_GradB_surface

@partial(jit, static_argnums=(1, 5, 6, 7, 8, 9, 10))
def loss_coils_and_surface(x, surface_all, field_nearaxis, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.5, target_B_on_surface=5.7):
    
    field=field_from_dofs(x[:-len(surface_all.x)-len(field_nearaxis.x)] ,dofs_curves=dofs_curves, currents_scale=currents_scale, nfp=nfp,n_segments=n_segments, stellsym=stellsym)     
    surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
    surface.dofs = x[-len(surface_all.x)-len(field_nearaxis.x):-len(field_nearaxis.x)]
    
    field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(x[-len(field_nearaxis.x):], field_nearaxis)
    
    coil_length = loss_coil_length(x[:-len(surface_all.x)-len(field_nearaxis.x)],dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_length=max_coil_length)
    coil_curvature = loss_coil_curvature(x[:-len(surface_all.x)-len(field_nearaxis.x)],dofs_curves=dofs_curves,currents_scale=currents_scale,nfp=nfp,n_segments=n_segments,stellsym=stellsym,max_coil_curvature=max_coil_curvature)
    
    
    coil_length_loss    = 1e3*jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))
    coil_curvature_loss = 1e3*jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))
    
    normal_cross_GradB_dot_grad_B_dot_GradB_surface = jnp.sum(jnp.abs(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface, field)))
    
    bdotn_over_b = BdotN_over_B(surface, field)
    bdotn_over_b_loss = 10*jnp.sum(jnp.abs(bdotn_over_b))
    
    mean_cross_sectional_area_loss = 100*jnp.abs(surface.mean_cross_sectional_area()-surface_all.mean_cross_sectional_area())

    AbsB_on_surface = jnp.linalg.norm(B_on_surface(surface, field), axis=2)
    AbsB_surface_loss = jnp.abs(jnp.mean(AbsB_on_surface)-target_B_on_surface)
    
    B_difference, gradB_difference = difference_B_gradB_onaxis(field_nearaxis, field)
    B_difference_loss = 30*jnp.sum(jnp.abs(B_difference))
    gradB_difference_loss = 30*jnp.sum(jnp.abs(gradB_difference))
    
    elongation = field_nearaxis.elongation
    iota = field_nearaxis.iota
    elongation_loss = jnp.sum(jnp.abs(elongation))
    iota_loss = 50/jnp.abs(iota)
    
    axis_surface = surface.dofs[0]
    axis_nearaxis = field_nearaxis.rc[0]
    axis_loss = jnp.abs(jnp.min(jnp.array([axis_nearaxis-axis_surface,0])))
    
    # debug.print("######################")
    # debug.print("normal_cross_GradB_dot_grad_B_dot_GradB_surface={}", normal_cross_GradB_dot_grad_B_dot_GradB_surface)
    # debug.print("bdotn_over_b_loss={}", bdotn_over_b_loss)
    # debug.print("mean_cross_sectional_area_loss={}", mean_cross_sectional_area_loss)
    # debug.print("B_difference_loss={}", B_difference_loss)
    # debug.print("gradB_difference_loss={}", gradB_difference_loss)
    # debug.print("iota_loss={}", iota_loss)
    # debug.print("axis_loss={}", axis_loss)
    
    # Xaxis = field_nearaxis.R0*jnp.cos(field_nearaxis.phi)
    # Yaxis = field_nearaxis.R0*jnp.sin(field_nearaxis.phi)
    # initial_xyz = jnp.array([Xaxis, Yaxis, field_nearaxis.Z0]).T[:nparticles]
    # particles = Particles(initial_xyz=initial_xyz, field=field)
    # particles_drift_loss = jnp.sum(loss_particle_drift(field, particles, maxtime_tracing, num_steps, trace_tolerance, model=model))/num_steps/nparticles
    
    return (
        coil_length_loss+coil_curvature_loss
    #    +normal_cross_GradB_dot_grad_B_dot_GradB_surface
       +bdotn_over_b_loss
       +mean_cross_sectional_area_loss
    #    +AbsB_surface_loss
       +B_difference_loss
       +gradB_difference_loss
       +elongation_loss
       +iota_loss
    #    +axis_loss
    #    +particles_drift_loss
    )

# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
time0 = time()
initial_dofs = jnp.concatenate((coils_initial.x, surface_initial.x, field_nearaxis_initial.x))
coils_optimized, surface_optimized, field_nearaxis_optimized = optimize_loss_function(loss_coils_and_surface, initial_dofs=initial_dofs, coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, surface_all=surface_initial, field_nearaxis=field_nearaxis_initial,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature, target_B_on_surface=target_B_on_axis)
print(f"Optimization took {time()-time0:.2f} seconds")
# Xaxis = field_nearaxis_optimized.R0*jnp.cos(field_nearaxis_optimized.phi)
# Yaxis = field_nearaxis_optimized.R0*jnp.sin(field_nearaxis_optimized.phi)
# initial_xyz = jnp.array([Xaxis, Yaxis, field_nearaxis_optimized.Z0]).T[:nparticles]
# particles = Particles(initial_xyz=initial_xyz, field=BiotSavart(coils_optimized))
# tracing_optimized = Tracing(field=coils_optimized, particles=particles, maxtime=maxtime_tracing, model=model, timesteps=num_steps)

print(f'############################################')
print(f"Mean Magnetic field on surface: {jnp.mean(jnp.linalg.norm(B_on_surface(surface_optimized, BiotSavart(coils_optimized)), axis=2))}")
print(f"Initial   max(BdotN/B): {jnp.max(BdotN_over_B(surface_initial, BiotSavart(coils_initial))):.2e}")
print(f"Optimized max(BdotN/B): {jnp.max(BdotN_over_B(surface_optimized, BiotSavart(coils_optimized))):.2e}")
print(f'Initial   iota on-axis: {field_nearaxis_initial.iota}')
print(f'Optimized iota on-axis: {field_nearaxis_optimized.iota}')
print(f'Initial   max(elongation): {max(field_nearaxis_initial.elongation)}')
print(f'Optimized max(elongation): {max(field_nearaxis_optimized.elongation)}')
print(f"Initial   coils length: {coils_initial.length[:number_coils_per_half_field_period]}")
print(f"Optimized coils length: {coils_optimized.length[:number_coils_per_half_field_period]}")
print(f"Initial   coils curvature: {jnp.mean(coils_initial.curvature, axis=1)[:number_coils_per_half_field_period]}")
print(f"Optimized coils curvature: {jnp.mean(coils_optimized.curvature, axis=1)[:number_coils_per_half_field_period]}")
B_difference_initial, gradB_difference_initial = difference_B_gradB_onaxis(field_nearaxis_initial, BiotSavart(coils_initial))
B_difference_optimized, gradB_difference_optimized = difference_B_gradB_onaxis(field_nearaxis_optimized, BiotSavart(coils_optimized))
print(f'Initial   B on axis difference: {jnp.sum(jnp.abs(B_difference_initial))}')
print(f'Optimized B on axis difference: {jnp.sum(jnp.abs(B_difference_optimized))}')
print(f'Initial   gradB on axis difference: {jnp.sum(jnp.abs(gradB_difference_initial))}')
print(f'Optimized gradB on axis difference: {jnp.sum(jnp.abs(gradB_difference_optimized))}')

# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils_initial.plot(ax=ax1, show=False)
surface_initial.plot(ax=ax1, show=False)
field_nearaxis_initial.plot(r=major_radius_coils/12, ax=ax1, show=False)
# tracing_initial.plot(ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False)
surface_optimized.plot(ax=ax2, show=False)
field_nearaxis_optimized.plot(r=major_radius_coils/12, ax=ax2, show=False)
# tracing_optimized.plot(ax=ax2, show=False)
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
# tracing_initial.to_vtk('initial_tracing')
# tracing_optimized.to_vtk('optimized_tracing')

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")