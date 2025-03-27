import os
number_of_processors_to_use = 12 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from essos.fields import BiotSavart, near_axis
from essos.dynamics import Particles, Tracing
from essos.surfaces import BdotN_over_B, SurfaceRZFourier, B_on_surface
from essos.coils import Coils, CreateEquallySpacedCurves, Curves
from essos.optimization import optimize_loss_function, new_nearaxis_from_x_and_old_nearaxis
from essos.objective_functions import (loss_coil_curvature, difference_B_gradB_onaxis,
                                       loss_coil_length, loss_particle_drift, loss_BdotN)
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, devices, device_put, grad, debug
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from time import time
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual

mesh = Mesh(devices(), ("dev",))
sharding = NamedSharding(mesh, PartitionSpec("dev", None))

ntheta=30
nphi=30
input = os.path.join('input_files','input.rotating_ellipse')
surface_initial = SurfaceRZFourier(input, ntheta=ntheta, nphi=nphi, range_torus='half period')

# Optimization parameters
max_coil_length = 40
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
coils_initial = optimize_loss_function(loss_BdotN, initial_dofs=coils_initial.x, coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                maximum_function_evaluations=maximum_function_evaluations, surface=surface_initial,
                                max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,)

def B_on_surface(surface, field):
    ntheta = surface.ntheta
    nphi = surface.nphi
    gamma = surface.gamma
    gamma_reshaped = gamma.reshape(nphi * ntheta, 3)
    gamma_sharded = device_put(gamma_reshaped, sharding)
    B_on_surface = jit(vmap(field.B), in_shardings=sharding, out_shardings=sharding)(gamma_sharded)
    B_on_surface = B_on_surface.reshape(nphi, ntheta, 3)
    return B_on_surface

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
    B_surface = B_on_surface(surface, field)
    grad_B_dot_GradB_surface = grad_B_dot_GradAbsB_on_surface(surface, field)
    normal_cross_GradB_surface = jnp.cross(surface.normal, gradAbsB_surface, axisa=-1, axisb=-1)
    normal_cross_GradB_dot_grad_B_dot_GradB_surface = jnp.sum(normal_cross_GradB_surface * grad_B_dot_GradB_surface, axis=-1)
    B_cross_GradB = jnp.cross(B_surface, gradAbsB_surface, axisa=-1, axisb=-1)
    # B_cross_GradB_dot_grad_B_dot_GradB_surface = jnp.sum(B_cross_GradB * grad_B_dot_GradB_surface, axis=-1)
    # debug.print("normal_cross_GradB_dot_grad_B_dot_GradB_surface: {}", jnp.sum(jnp.abs(normal_cross_GradB_dot_grad_B_dot_GradB_surface)))
    # debug.print("B_cross_GradB_dot_grad_B_dot_GradB_surface: {}", jnp.sum(jnp.abs(B_cross_GradB_dot_grad_B_dot_GradB_surface)))
    return normal_cross_GradB_dot_grad_B_dot_GradB_surface#, normal_cross_GradB_dot_grad_B_dot_GradB_surface + B_cross_GradB_dot_grad_B_dot_GradB_surface

def vmec_qs_from_surface(filename):
    vmec = Vmec(filename, verbose=False)
    qs = QuasisymmetryRatioResidual(vmec, surfaces=[1], helicity_m=1, helicity_n=0)
    return jnp.sum(jnp.abs(qs.residuals()))

# @partial(jit, static_argnames=['surface_initial', 'n_segments'])
def qs_loss(surface_dofs, dofs_curves, dofs_currents, surface_initial, currents_scale=1, n_segments=100):
    surface = SurfaceRZFourier(rc=surface_initial.rc, zs=surface_initial.zs, nfp=surface_initial.nfp, range_torus=surface_initial.range_torus, nphi=surface_initial.nphi, ntheta=surface_initial.ntheta)
    surface.dofs = surface_dofs
    curves = Curves(dofs_curves, n_segments, surface_initial.nfp)
    coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
    print("##############################")
    print(f"initial max(BdotN/B): {jnp.max(BdotN_over_B(surface, BiotSavart(coils))):.2e}")
    # field = BiotSavart(coils)
    coils = optimize_loss_function(loss_BdotN, initial_dofs=coils.x, coils=coils, tolerance_optimization=tolerance_optimization,
                                    maximum_function_evaluations=maximum_function_evaluations, surface=surface,
                                    max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature,disp=False)
    field = BiotSavart(coils)
    print(f"Final max(BdotN/B): {jnp.max(BdotN_over_B(surface, field)):.2e}")
    loss = jnp.sum(jnp.abs(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface, field)))
    return loss, coils

print(f'############################################')
dofs_old = surface_initial.dofs
new_dof_array = jnp.linspace(-1, 1, 10)
qs_ESSOS_loss_array = []
qs_VMEC_loss_array = []
coils = coils_initial
for dof in new_dof_array:
    print(f'dof: {dof}')
    dofs = dofs_old.at[2].set(dof)
    loss, coils = qs_loss(dofs, coils.dofs_curves, coils.dofs_currents, surface_initial, currents_scale=coils.currents_scale, n_segments=number_coil_points)
    qs_ESSOS_loss_array.append(loss)
    
    filename = 'input.rotating_ellipse_dof'
    new_surface = SurfaceRZFourier(rc=surface_initial.rc, zs=surface_initial.zs, nfp=surface_initial.nfp, range_torus=surface_initial.range_torus, nphi=surface_initial.nphi, ntheta=surface_initial.ntheta)
    new_surface.dofs = dofs
    new_surface.to_vmec(filename)
    new_surface.to_vtk('surface_dof')
    qs = vmec_qs_from_surface(filename)
    qs_VMEC_loss_array.append(qs)
qs_ESSOS_loss_array = jnp.array(qs_ESSOS_loss_array)
qs_VMEC_loss_array = jnp.array(qs_VMEC_loss_array)
plt.plot(new_dof_array, qs_ESSOS_loss_array/jnp.max(qs_ESSOS_loss_array), label='ESSOS')
plt.plot(new_dof_array, qs_VMEC_loss_array/jnp.max(qs_VMEC_loss_array), label='VMEC')
plt.legend()
plt.xlabel('Dof')
plt.ylabel('Loss')
plt.show()

print(f'############################################')
print(f"Mean Magnetic field on surface: {jnp.mean(jnp.linalg.norm(B_on_surface(surface_initial, BiotSavart(coils_initial)), axis=2))}")
print(f"Initial max(BdotN/B): {jnp.max(BdotN_over_B(surface_initial, BiotSavart(coils_initial))):.2e}")
print(f"Initial coils length: {coils_initial.length[:number_coils_per_half_field_period]}")
print(f"Initial coils curvature: {jnp.mean(coils_initial.curvature, axis=1)[:number_coils_per_half_field_period]}")
print(f"Initial loss qs surface: {jnp.sum(jnp.abs(loss_normal_cross_GradB_dot_grad_B_dot_GradB_surface(surface_initial, BiotSavart(coils_initial)))):.2e}")
