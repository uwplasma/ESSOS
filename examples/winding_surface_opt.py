"""Optimize a winding surface using the corrected float64 surrogate."""

import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy.optimize import minimize

from essos.surfaces import SurfaceRZFourier

MU0 = 4 * jnp.pi * 1e-7
INTEGRATION_FACTOR = 4 * jnp.pi ** 2

SVD_WEIGHT = 1.0
SINGULAR_STRENGTH_WEIGHT = 1.0
VOLUME_WEIGHT = 0.02
SPECTRAL_WEIGHT = 0.02
DISTANCE_WEIGHT = 10.0
SELF_INTERSECTION_WEIGHT = 100.0
MINIMUM_DISTANCE = 0.06
MINIMUM_SELF_RADIUS = 0.05
DISTANCE_WALL_SCALE = 0.01
SELF_RADIUS_WALL_SCALE = 0.01
SELF_NEIGHBOR_RADIUS = 2
SHARPNESS = 300.0
MAXITER = 100
ACTIVE_MPOL = 2
ACTIVE_NTOR = 2
COEFFICIENT_STEP_BOUND = 0.04

plasma_ntheta = 24
plasma_nphi = 24
winding_ntheta = 24
winding_nphi = 24
VARIANT = "final"
OBJECTIVE_NAMES = ("inverse_singular_entropy", "singular_strength", "volume_m3",
                   "spectral_penalty", "plasma_distance_penalty", "local_area_penalty",
                   "smooth_self_radius_m", "self_intersection_penalty")

def mean_minor_radius(surface):
    return jnp.sqrt(INTEGRATION_FACTOR * surface.mean_cross_sectional_area() / jnp.pi)

def extend_via_normal_dofs(dofs, xm, xn, offset, *, ntheta=24, nphi=24,
                           scaling=None):
    """Normal-offset an ESSOS RZ Fourier surface and return refit dofs."""
    dofs = jnp.asarray(dofs)
    xm = jnp.asarray(xm)
    xn = jnp.asarray(xn)
    nmodes = xm.size
    if dofs.size != 2 * nmodes:
        raise ValueError(f"Expected {2 * nmodes} ESSOS dofs, got {dofs.size}.")

    scaling = jnp.ones(nmodes, dtype=dofs.dtype) if scaling is None else jnp.asarray(scaling)
    rc = dofs[:nmodes] / scaling
    zs = dofs[nmodes:] / scaling

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nphi, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    angle = xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * phi2d[None, :, :]
    sin_angle = jnp.sin(angle)
    cos_angle = jnp.cos(angle)

    R = jnp.einsum("i,ijk->jk", rc, cos_angle)
    Z = jnp.einsum("i,ijk->jk", zs, sin_angle)
    cos_phi = jnp.cos(phi2d)
    sin_phi = jnp.sin(phi2d)
    gamma = jnp.stack((R * cos_phi, R * sin_phi, Z), axis=-1)

    dR_dtheta = -jnp.einsum("i,ijk->jk", xm * rc, sin_angle)
    dZ_dtheta = jnp.einsum("i,ijk->jk", xm * zs, cos_angle)
    gammadash_theta = jnp.stack(
        (dR_dtheta * cos_phi, dR_dtheta * sin_phi, dZ_dtheta), axis=-1)

    dR_dphi = jnp.einsum("i,ijk->jk", xn * rc, sin_angle)
    dZ_dphi = -jnp.einsum("i,ijk->jk", xn * zs, cos_angle)
    gammadash_phi = jnp.stack((
        dR_dphi * cos_phi - R * sin_phi,
        dR_dphi * sin_phi + R * cos_phi,
        dZ_dphi), axis=-1)

    normal = jnp.cross(gammadash_phi, gammadash_theta, axis=2)
    unitnormal = normal / jnp.linalg.norm(normal, axis=2, keepdims=True)
    offset_points = gamma + offset * unitnormal
    R_offset = jnp.sqrt(offset_points[:, :, 0] ** 2 + offset_points[:, :, 1] ** 2)
    Z_offset = offset_points[:, :, 2]
    phi_offset = jnp.arctan2(offset_points[:, :, 1], offset_points[:, :, 0])

    fit_angle = (
        xm[:, None, None] * theta2d[None, :, :]
        - xn[:, None, None] * phi_offset[None, :, :])
    cos_basis = jnp.cos(fit_angle).reshape(nmodes, -1).T
    sin_basis = jnp.sin(fit_angle).reshape(nmodes, -1).T
    rc_offset = jnp.linalg.lstsq(cos_basis, R_offset.reshape(-1), rcond=None)[0]
    zs_offset = jnp.linalg.lstsq(sin_basis, Z_offset.reshape(-1), rcond=None)[0]
    return jnp.concatenate((rc_offset * scaling, zs_offset * scaling))

def surface_quadrature_weights(surface):
    """Periodic trapezoidal weights, including the surface Jacobian."""
    ntheta_cells = surface.ntheta - int(surface.close)
    nphi_cells = surface.nphi - int(surface.close)
    phi_period = 2 * jnp.pi if surface.range_torus == "full torus" else jnp.pi / surface.nfp
    endpoint_theta = jnp.ones(surface.ntheta).at[jnp.array([0, -1])].set(0.5) if surface.close else jnp.ones(surface.ntheta)
    endpoint_phi = jnp.ones(surface.nphi).at[jnp.array([0, -1])].set(0.5) if surface.close else jnp.ones(surface.nphi)
    return (surface.area_element * endpoint_phi[:, None] * endpoint_theta[None, :]
            * (2 * jnp.pi / ntheta_cells) * (phi_period / nphi_cells)).reshape(-1)

def reduced_memory_induction_matrix(winding_points, plasma_points,
                                    dipole_normals, plasma_normals,
                                    winding_weights, plasma_weights):
    difference = winding_points[None, :, :] - plasma_points[:, None, :]
    distance_squared = jnp.sum(difference ** 2, axis=2)
    diff_dot_dipole = jnp.einsum("ijk,jk->ij", difference, dipole_normals)
    diff_dot_plasma = jnp.einsum("ijk,ik->ij", difference, plasma_normals)
    dipole_dot_plasma = jnp.einsum("jk,ik->ij", dipole_normals, plasma_normals)
    kernel = (MU0 / (4 * jnp.pi)) * (
        3 * diff_dot_dipole * diff_dot_plasma
        - distance_squared * dipole_dot_plasma) / distance_squared ** 2.5
    return jnp.sqrt(plasma_weights[:, None]) * kernel * jnp.sqrt(winding_weights[None, :])

def singular_value_objective(singular_values):
    probabilities = singular_values / jnp.sum(singular_values)
    singular_entropy = -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-300)))
    return 1 / jnp.maximum(singular_entropy, 1e-16)

def spectral_objective(surface):
    rc_obj = jnp.sum(jnp.abs(surface.xm * surface.rc)**2)
    zs_obj = jnp.sum(jnp.abs(surface.xm * surface.zs)**2)
    return rc_obj + zs_obj

def smooth_minimum_distance(surface1, surface2, sharpness):
    points1 = surface1.gamma
    points1 = points1.reshape(-1, 3)
    points2 = surface2.gamma
    points2 = points2.reshape(-1, 3)
    distance = jnp.linalg.norm(points1[:, None, :] - points2[None, :, :], axis=2)
    weights = jax.nn.softmax(-sharpness * distance.reshape(-1))
    return jnp.sum(weights * distance.reshape(-1))

def smooth_minimum_tangent_radius(surface, sharpness, neighbor_radius):
    """Nonlocal surface thickness; nearby parameter-grid points are excluded."""
    points = surface.gamma.reshape(-1, 3)
    normals = surface.unitnormal.reshape(-1, 3)
    difference = points[None, :, :] - points[:, None, :]
    distance_squared = jnp.sum(difference ** 2, axis=2)
    tangent_radius = distance_squared / (
        2 * jnp.abs(jnp.einsum("ijk,ik->ij", difference, normals)) + 1e-14)

    iphi, itheta = jnp.meshgrid(jnp.arange(surface.nphi),
                                jnp.arange(surface.ntheta), indexing="ij")
    dphi = jnp.abs(iphi.reshape(-1, 1) - iphi.reshape(1, -1))
    dtheta = jnp.abs(itheta.reshape(-1, 1) - itheta.reshape(1, -1))
    dphi = jnp.minimum(dphi, surface.nphi - dphi)
    dtheta = jnp.minimum(dtheta, surface.ntheta - dtheta)
    nonlocal_pair = (dphi > neighbor_radius) | (dtheta > neighbor_radius)
    tangent_radius = jnp.where(nonlocal_pair, tangent_radius, 1e6)
    weights = jax.nn.softmax(-sharpness * tangent_radius.reshape(-1))
    return jnp.sum(weights * tangent_radius.reshape(-1))

input_filepath = os.path.join(os.path.dirname(__file__), ".", "input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')

plasma_surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=plasma_ntheta, nphi=plasma_nphi, close=False, range_torus='full torus')

#plasma_surface.to_vtk('plasma_surface')

minor_radius_plasma = mean_minor_radius(plasma_surface)
plasma_points = plasma_surface.gamma
plasma_points = plasma_points.reshape(-1, 3)
plasma_unitnormals = plasma_surface.unitnormal
plasma_unitnormals = plasma_unitnormals.reshape(-1, 3)
plasma_weights = surface_quadrature_weights(plasma_surface)

winding_surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=winding_ntheta, nphi=winding_nphi, close=False, range_torus='full torus')

winding_dofs = extend_via_normal_dofs(winding_surface.dofs, winding_surface.xm, winding_surface.xn, minor_radius_plasma, ntheta = winding_surface.ntheta, nphi = winding_surface.nphi, scaling = winding_surface.scaling)
winding_surface.dofs = winding_dofs
#winding_surface.to_vtk('winding_surface_init')

def individual_objectives(winding_surface):
    winding_points = winding_surface.gamma
    winding_points = winding_points.reshape(-1, 3)
    winding_unitnormals = winding_surface.unitnormal
    winding_unitnormals = winding_unitnormals.reshape(-1, 3)
    winding_area_elements = winding_surface.area_element
    winding_area_elements = winding_area_elements.reshape(-1)

    winding_weights = surface_quadrature_weights(winding_surface)
    induction_matrix = reduced_memory_induction_matrix(
        winding_points, plasma_points, winding_unitnormals, plasma_unitnormals,
        winding_weights, plasma_weights)
    singular_values = jnp.linalg.svd(induction_matrix, compute_uv=False)

    svd_objective = singular_value_objective(singular_values)
    singular_strength = jnp.sum(singular_values)
    volume = INTEGRATION_FACTOR * jnp.abs(winding_surface.volume)
    spectral = spectral_objective(winding_surface)
    distance = smooth_minimum_distance(plasma_surface, winding_surface, sharpness=SHARPNESS)
    self_radius = smooth_minimum_tangent_radius(
        winding_surface, sharpness=SHARPNESS, neighbor_radius=SELF_NEIGHBOR_RADIUS)
    minimum_area_element = jnp.min(winding_area_elements)

    distance_objective = 1 + jnp.tanh((MINIMUM_DISTANCE - distance) / DISTANCE_WALL_SCALE)
    self_intersection_objective = 1 + jnp.tanh(
        (MINIMUM_SELF_RADIUS - self_radius) / SELF_RADIUS_WALL_SCALE)
    minimum_area_element_objective = jnp.square(jnp.maximum(1e-6 - minimum_area_element, 0.0)) * 1e12

    return (svd_objective, singular_strength, volume, spectral,
            distance_objective, minimum_area_element_objective,
            self_radius, self_intersection_objective)

scales = individual_objectives(winding_surface)

x0_full = winding_surface.dofs
nmodes = winding_surface.xm.size
active_mode_indices = jnp.where((winding_surface.xm <= ACTIVE_MPOL)
                                & (jnp.abs(winding_surface.xn / winding_surface.nfp) <= ACTIVE_NTOR))[0]
active_rc_indices = active_mode_indices[active_mode_indices != 0]
active_zs_indices = active_mode_indices + nmodes
active_indices = jnp.concatenate((active_rc_indices, active_zs_indices))
x0 = x0_full[active_indices]
bounds = [(float(value - COEFFICIENT_STEP_BOUND), float(value + COEFFICIENT_STEP_BOUND))
          for value in x0]

def objective_function(active_dofs):
    winding_surface.dofs = x0_full.at[active_indices].set(active_dofs)
    objectives = individual_objectives(winding_surface)

    return (SVD_WEIGHT * objectives[0] / scales[0] # Flatten out singular values
            + SINGULAR_STRENGTH_WEIGHT * jnp.square(jnp.maximum(1 - objectives[1] / scales[1], 0)) # Prevent global operator weakening
            - VOLUME_WEIGHT * objectives[2] / scales[2] # Maximize volume
            + SPECTRAL_WEIGHT * objectives[3] / jnp.maximum(scales[3], 1e-16) # Minimize poloidal spectral modes
            + DISTANCE_WEIGHT * objectives[4] # Keep winding surface away from plasma surface
            + 100 * objectives[5] # Prevent local surface collapse
            + SELF_INTERSECTION_WEIGHT * objectives[7]) # Prevent nonlocal self-intersection

value_and_grad = jax.jit(jax.value_and_grad(objective_function))
evaluation_count = 0

def fun(x):
    global evaluation_count
    value, grad = value_and_grad(x)
    evaluation_count += 1
    print(f"evaluation {evaluation_count:03d}: objective = {float(value):.9e}")
    return value, grad

res = minimize(fun, x0, method='L-BFGS-B', jac=True, bounds=bounds, options={"maxiter": MAXITER})

res_full = x0_full.at[active_indices].set(jnp.asarray(res.x))
winding_surface.dofs = res_full
#winding_surface.to_vtk('winding_surface_opt')
final_objectives = individual_objectives(winding_surface)

print(f"\nvariant {VARIANT}: {__doc__}")
print(f"optimizer: success={res.success}, iterations={res.nit}, evaluations={res.nfev}, objective={res.fun:.9e}")
for name, initial, final in zip(OBJECTIVE_NAMES, scales, final_objectives):
    print(f"{name}: {float(initial):.9e} -> {float(final):.9e}")
print(f"smooth_plasma_distance_m: {float(smooth_minimum_distance(plasma_surface, winding_surface, SHARPNESS)):.9e}")
print(f"active_dof_step_norm: {float(jnp.linalg.norm(x0 - res.x)):.9e}")
