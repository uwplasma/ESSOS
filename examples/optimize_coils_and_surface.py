import os
number_of_processors_to_use = 12 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from essos.fields import BiotSavart
from essos.surfaces import BdotN_over_B, SurfaceRZFourier, B_on_surface
from essos.coils import Coils, CreateEquallySpacedCurves, Curves
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_coil_curvature, loss_coil_length
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, devices, device_put, grad
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
max_coil_length = 40
max_coil_curvature = 0.5
order_Fourier_series_coils = 6
number_coil_points = order_Fourier_series_coils*10
maximum_function_evaluations = 300
number_coils_per_half_field_period = 4
tolerance_optimization = 1e-7
target_B_on_surface = 0.5

# Initialize coils
current_on_each_coil = 1e6
number_of_field_periods = surface_initial.nfp
major_radius_coils = surface_initial.dofs[0]
minor_radius_coils = major_radius_coils/1.5
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

@partial(jit, static_argnames=['surface','field'])
def grad_AbsB_on_surface(surface, field):
    ntheta = surface.ntheta
    nphi = surface.nphi
    gamma = surface.gamma
    gamma_reshaped = gamma.reshape(nphi * ntheta, 3)
    gamma_sharded = device_put(gamma_reshaped, sharding)
    dAbsB_by_dX_on_surface = jit(vmap(field.dAbsB_by_dX), in_shardings=sharding, out_shardings=sharding)(gamma_sharded)
    dAbsB_by_dX_on_surface = dAbsB_by_dX_on_surface.reshape(nphi, ntheta, 3)
    return dAbsB_by_dX_on_surface

@partial(jit, static_argnames=['field'])
def B_dot_GradAbsB(points, field):
    B = field.B(points)
    B_dot_GradAbsB = field.dAbsB_by_dX(points)
    B_dot_GradAbsB = jnp.sum(B_dot_GradAbsB * B, axis=-1)
    return B_dot_GradAbsB

@partial(jit, static_argnames=['field'])
def grad_B_dot_GradAbsB(points, field):
    return grad(B_dot_GradAbsB, argnums=0)(points, field)

@partial(jit, static_argnames=['surface','field'])
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

@partial(jit, static_argnames=['surface','field'])
def loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface, field):
    gradAbsB_surface = grad_AbsB_on_surface(surface, field)
    grad_B_dot_GradB_surface = grad_B_dot_GradAbsB_on_surface(surface, field)
    normal_cross_GradB_surface = jnp.cross(surface.normal, gradAbsB_surface, axisa=-1, axisb=-1)
    normal_cross_GradB_dot_grad_B_dot_GradB_surface = jnp.sum(normal_cross_GradB_surface * grad_B_dot_GradB_surface, axis=-1)
    return normal_cross_GradB_dot_grad_B_dot_GradB_surface

@partial(jit, static_argnums=(1, 4, 5, 6, 7, 8, 9))
def loss_coils_and_surface(x, surface_all, dofs_curves, currents_scale, nfp, max_coil_length=42,
               n_segments=60, stellsym=True, max_coil_curvature=0.5, target_B_on_surface=1):
    len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
    dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], (dofs_curves.shape))
    len_dofs_surface = len(surface_all.x)
    dofs_currents = x[len_dofs_curves_ravelled:-len_dofs_surface]
    
    curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    
    surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, ntheta=surface_all.ntheta, nphi=surface_all.nphi, range_torus=surface_all.range_torus)
    surface.dofs = x[-len_dofs_surface:]
    
    coil_length = loss_coil_length(field)
    coil_curvature = loss_coil_curvature(field)
    
    coil_length_loss    = 1e3*jnp.max(jnp.concatenate([coil_length-max_coil_length,jnp.array([0])]))
    coil_curvature_loss = 1e3*jnp.max(jnp.concatenate([coil_curvature-max_coil_curvature,jnp.array([0])]))
    
    normal_cross_GradB_dot_grad_B_dot_GradB_surface = jnp.sum(jnp.abs(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface, field)))
    
    bdotn_over_b = BdotN_over_B(surface, field)
    bdotn_over_b_loss = jnp.sum(jnp.abs(bdotn_over_b))
    
    mean_cross_sectional_area_loss = jnp.abs(surface.mean_cross_sectional_area()-surface_all.mean_cross_sectional_area())

    AbsB_on_surface = jnp.linalg.norm(B_on_surface(surface, field), axis=2)
    B_surface_loss = jnp.abs(jnp.mean(AbsB_on_surface)-target_B_on_surface)
    
    ## Add near-axis so that the coils are optimized for the near-axis field and surface quasisymmetry?
    
    return coil_length_loss+coil_curvature_loss+10*normal_cross_GradB_dot_grad_B_dot_GradB_surface\
          +bdotn_over_b_loss+30*mean_cross_sectional_area_loss+30*B_surface_loss

# Optimize coils
print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
time0 = time()
initial_dofs = jnp.concatenate((coils_initial.x, surface_initial.x))
coils_optimized, surface_optimized = optimize_loss_function(loss_coils_and_surface, initial_dofs=initial_dofs, coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, surface_all=surface_initial,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature, target_B_on_surface=target_B_on_surface)
print(f"Optimization took {time()-time0:.2f} seconds")

BdotN_over_B_initial = BdotN_over_B(surface_initial, BiotSavart(coils_initial))
BdotN_over_B_optimized = BdotN_over_B(surface_optimized, BiotSavart(coils_optimized))
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")

# Plot coils, before and after optimization
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils_initial.plot(ax=ax1, show=False)
surface_initial.plot(ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False)
surface_optimized.plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()


# # Save results in vtk format to analyze in Paraview
# surface_initial.to_vtk('initial_surface', field=BiotSavart(coils_initial))
# surface_optimized.to_vtk('optimized_surface', field=BiotSavart(coils_optimized))
# coils_initial.to_vtk('initial_coils')
# coils_optimized.to_vtk('optimized_coils')

# # Save the surface to a VMEC file
# surface_optimized.to_vmec('input.optimized')

# # Save the coils to a json file
# coils_optimized.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")