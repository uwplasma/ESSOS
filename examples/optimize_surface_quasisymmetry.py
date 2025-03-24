import os
number_of_processors_to_use = 12 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from essos.fields import BiotSavart, near_axis
from essos.dynamics import Particles, Tracing
from essos.surfaces import BdotN_over_B, SurfaceRZFourier, B_on_surface
from essos.coils import Coils, CreateEquallySpacedCurves, Curves
from essos.optimization import optimize_loss_function, new_nearaxis_from_x_and_old_nearaxis
from essos.objective_functions import (loss_coil_curvature, difference_B_gradB_onaxis,
                                       loss_coil_length, loss_particle_drift)
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
maximum_function_evaluations = 600
number_coils_per_half_field_period = 4
tolerance_optimization = 1e-7

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
    # loss = jnp.abs(normal_cross_GradB_dot_grad_B_dot_GradB_surface)
    # return loss
    return normal_cross_GradB_dot_grad_B_dot_GradB_surface

@partial(jit, static_argnames=['surface_initial', 'n_segments'])
def qs_loss(surface_dofs, dofs_curves, dofs_currents, surface_initial, currents_scale=1, n_segments=100):
    surface = SurfaceRZFourier(rc=surface_initial.rc, zs=surface_initial.zs, nfp=surface_initial.nfp, range_torus=surface_initial.range_torus, nphi=surface_initial.nphi, ntheta=surface_initial.ntheta)
    surface.dofs = surface_dofs
    curves = Curves(dofs_curves, n_segments, surface_initial.nfp)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    field = BiotSavart(coils)
    
    loss = jnp.sum(jnp.abs(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface, field)))
    return loss

print(f'############################################')
print(surface_initial.ntor)
print(surface_initial.mpol)
print(surface_initial.rmnc_interp)
print(surface_initial.zmns_interp)
surface_initial.to_vtk('initial_surface')
# print(qs_loss(surface_initial.dofs, coils_initial.dofs_curves, coils_initial.dofs_currents, surface_initial, currents_scale=coils_initial.currents_scale, n_segments=number_coil_points))

print(f'############################################')
print(f"Mean Magnetic field on surface: {jnp.mean(jnp.linalg.norm(B_on_surface(surface_initial, BiotSavart(coils_initial)), axis=2))}")
print(f"Initial max(BdotN/B): {jnp.max(BdotN_over_B(surface_initial, BiotSavart(coils_initial))):.2e}")
print(f"Initial coils length: {coils_initial.length[:number_coils_per_half_field_period]}")
print(f"Initial coils curvature: {jnp.mean(coils_initial.curvature, axis=1)[:number_coils_per_half_field_period]}")
print(f"Initial loss qs surface: {jnp.sum(jnp.abs(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface_initial, BiotSavart(coils_initial)))):.2e}")
