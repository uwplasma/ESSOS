"""Optimize a winding surface using a differentiable current-potential solve."""

import os
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset
from scipy.optimize import minimize

from essos.surfaces import SurfaceRZFourier


MU0 = 4 * jnp.pi * 1e-7
INTEGRATION_FACTOR = 4 * jnp.pi ** 2
TARGET_MAX_CURRENTS = jnp.asarray((6.7e6, 7e6, 7.5e6))
MAX_CURRENT_SMOOTHING = 1e-3
POTENTIAL_MPOL = 6
POTENTIAL_NTOR = 6

VOLUME_WEIGHT = 100.0
SPECTRAL_WEIGHT = 0.02
DISTANCE_WEIGHT = 1e6
SELF_INTERSECTION_WEIGHT = 100.0
MINIMUM_SELF_RADIUS = 0.05
SHARPNESS = 300.0
MAXITER = 100
ACTIVE_MPOL = 2
ACTIVE_NTOR = 2
COEFFICIENT_STEP_BOUND = 0.04

plasma_ntheta = 32
plasma_nphi = 64
winding_ntheta = 32
winding_nphi = 64


def quadrature_weights(surface):
    """Periodic trapezoidal weights including the surface Jacobian."""
    ntheta = surface.ntheta - int(surface.close)
    nphi = surface.nphi - int(surface.close)
    phi_period = (2 * jnp.pi if surface.range_torus == "full torus"
                  else jnp.pi / surface.nfp)
    theta_weights = jnp.ones(surface.ntheta)
    phi_weights = jnp.ones(surface.nphi)
    if surface.close:
        theta_weights = theta_weights.at[jnp.array([0, -1])].set(0.5)
        phi_weights = phi_weights.at[jnp.array([0, -1])].set(0.5)
    return (surface.area_element * phi_weights[:, None] * theta_weights[None, :]
            * (2 * jnp.pi / ntheta) * (phi_period / nphi)).reshape(-1)


def mean_minor_radius(surface):
    return jnp.sqrt(
        INTEGRATION_FACTOR * surface.mean_cross_sectional_area() / jnp.pi)


def normal_offset_dofs(surface, offset):
    """Offset grid points along the outward normal and refit the Fourier surface."""
    nmodes = surface.xm.size
    # SurfaceRZFourier's unit normal points inward for this parameterization.
    points = surface.gamma - offset * surface.unitnormal
    radius = jnp.linalg.norm(points[:, :, :2], axis=2)
    phi = jnp.arctan2(points[:, :, 1], points[:, :, 0])
    angle = (surface.xm[:, None, None] * surface.theta2d[None, :, :]
             - surface.xn[:, None, None] * phi[None, :, :])
    rc = jnp.linalg.lstsq(jnp.cos(angle).reshape(nmodes, -1).T,
                          radius.reshape(-1), rcond=None)[0]
    zs = jnp.linalg.lstsq(jnp.sin(angle).reshape(nmodes, -1).T,
                          points[:, :, 2].reshape(-1), rcond=None)[0]
    return jnp.concatenate((rc * surface.scaling, zs * surface.scaling))


def potential_modes(mpol, ntor, nfp):
    """REGCOIL's stellarator-symmetric, non-constant Fourier modes."""
    xm = [0] * ntor
    xn = list(range(1, ntor + 1))
    for m in range(1, mpol + 1):
        for n in range(-ntor, ntor + 1):
            xm.append(m)
            xn.append(n)
    return jnp.asarray(xm), nfp * jnp.asarray(xn)


def current_numerators(surface, net_poloidal_current):
    """Return J*K for each current-potential mode and the secular current."""
    xm, xn = potential_modes(POTENTIAL_MPOL, POTENTIAL_NTOR, surface.nfp)
    theta = surface.theta2d.reshape(-1)
    phi = surface.phi2d.reshape(-1)
    rtheta = surface.gammadash_theta.reshape(-1, 3)
    rphi = surface.gammadash_phi.reshape(-1, 3)
    cosine = jnp.cos(xm[:, None] * theta - xn[:, None] * phi)
    basis = cosine[:, :, None] * (
        xn[:, None, None] * rtheta[None, :, :]
        + xm[:, None, None] * rphi[None, :, :])
    fixed = net_poloidal_current * rtheta / (2 * jnp.pi)
    return xm, xn, basis, fixed


def dipole_normal_field(plasma, winding, potential):
    """Landreman-Boozer dipole-layer map from Phi to B dot n."""
    plasma_points = plasma.gamma.reshape(-1, 3)
    plasma_normals = plasma.unitnormal.reshape(-1, 3)
    winding_points = winding.gamma.reshape(-1, 3)
    winding_normals = winding.unitnormal.reshape(-1, 3)
    difference = plasma_points[:, None, :] - winding_points[None, :, :]
    distance_squared = jnp.sum(difference ** 2, axis=2)
    kernel = MU0 / (4 * jnp.pi) * (
        jnp.einsum("pk,qk->pq", plasma_normals, winding_normals)
        - 3 * jnp.einsum("pqk,pk->pq", difference, plasma_normals)
        * jnp.einsum("pqk,qk->pq", difference, winding_normals)
        / distance_squared) / distance_squared ** 1.5
    return kernel @ (quadrature_weights(winding)[:, None] * potential)


def fixed_current_normal_field(plasma, winding, current_numerator):
    """Biot-Savart normal field from the secular net-current term."""
    difference = (plasma.gamma.reshape(-1, 1, 3)
                  - winding.gamma.reshape(1, -1, 3))
    inverse_distance_cubed = jnp.sum(difference ** 2, axis=2) ** -1.5
    cross = jnp.cross(current_numerator[None, :, :], difference, axis=2)
    dtheta = 2 * jnp.pi / (winding.ntheta - int(winding.close))
    dphi = 2 * jnp.pi / (winding.nphi - int(winding.close))
    return (MU0 / (4 * jnp.pi) * dtheta * dphi
            * jnp.einsum("pqk,pq,pk->p", cross, inverse_distance_cubed,
                         plasma.unitnormal.reshape(-1, 3)))


def build_operators(plasma, winding, net_poloidal_current):
    """Build weighted physical operators for f_B and f_K."""
    xm, xn, current_basis_numerator, fixed_current_numerator = \
        current_numerators(winding, net_poloidal_current)
    potential = jnp.sin(
        winding.theta2d.reshape(-1, 1) * xm[None, :]
        - winding.phi2d.reshape(-1, 1) * xn[None, :])
    field_matrix = dipole_normal_field(plasma, winding, potential)
    # This sign aligns ESSOS's normal convention with the dipole-layer map.
    fixed_field = -fixed_current_normal_field(
        plasma, winding, fixed_current_numerator)

    plasma_sqrt_weight = jnp.sqrt(quadrature_weights(plasma))
    winding_sqrt_weight = jnp.sqrt(quadrature_weights(winding))
    jacobian = winding.area_element.reshape(-1)
    point_current_matrix = (-jnp.transpose(
        current_basis_numerator, (1, 2, 0)) / jacobian[:, None, None])
    point_fixed_current = fixed_current_numerator / jacobian[:, None]
    current_matrix = (
        winding_sqrt_weight[:, None, None] * point_current_matrix).reshape(
            -1, current_basis_numerator.shape[0])
    fixed_current = (
        winding_sqrt_weight[:, None] * point_fixed_current).reshape(-1)
    return (plasma_sqrt_weight[:, None] * field_matrix,
            plasma_sqrt_weight * fixed_field, current_matrix, fixed_current,
            point_current_matrix.reshape(-1, current_basis_numerator.shape[0]),
            point_fixed_current.reshape(-1))


def solve_current(operators, regularization):
    """Minimize f_B + regularization*f_K for one winding surface."""
    field_matrix, fixed_field, current_matrix, fixed_current, _, _ = operators
    gram_field = field_matrix.T @ field_matrix
    cross_field = field_matrix.T @ fixed_field
    gram_current = current_matrix.T @ current_matrix
    cross_current = current_matrix.T @ fixed_current
    coefficients = jnp.linalg.solve(
        gram_field + regularization * gram_current,
        -(cross_field + regularization * cross_current))
    field = field_matrix @ coefficients + fixed_field
    current = current_matrix @ coefficients + fixed_current
    return coefficients, jnp.vdot(field, field), jnp.vdot(current, current)


def solve_for_max_current(operators, target_max_current):
    """Choose lambda so a smooth upper bound on max |K| reaches its limit."""
    _, _, current_matrix, fixed_current, point_matrix, point_fixed = operators
    gram_current = current_matrix.T @ current_matrix
    cross_current = current_matrix.T @ fixed_current
    lower = jnp.log(1e-20)
    upper = jnp.log(1e-8)

    def smooth_maximum(coefficients):
        magnitude = jnp.linalg.norm(
            (point_matrix @ coefficients + point_fixed).reshape(-1, 3), axis=1)
        return (MAX_CURRENT_SMOOTHING * target_max_current
                * jax.scipy.special.logsumexp(
                    magnitude / (MAX_CURRENT_SMOOTHING * target_max_current)))

    low_maximum = smooth_maximum(solve_current(operators, jnp.exp(lower))[0])
    high_maximum = smooth_maximum(solve_current(operators, jnp.exp(upper))[0])
    for _ in range(22):
        log_regularization = 0.5 * (lower + upper)
        regularization = jnp.exp(log_regularization)
        coefficients = solve_current(operators, regularization)[0]
        maximum = smooth_maximum(coefficients)
        lower = jnp.where(maximum > target_max_current,
                          log_regularization, lower)
        upper = jnp.where(maximum > target_max_current,
                          upper, log_regularization)

    # The stopped bisection locates the root; one Newton step supplies its
    # implicit derivative with respect to the winding-surface coefficients.
    log_regularization = jax.lax.stop_gradient(0.5 * (lower + upper))
    regularization = jnp.exp(log_regularization)
    coefficients = solve_current(operators, regularization)[0]
    lhs = (operators[0].T @ operators[0]
           + regularization * gram_current)
    coefficient_derivative = jnp.linalg.solve(
        lhs, -cross_current - gram_current @ coefficients)
    point_current = (point_matrix @ coefficients + point_fixed).reshape(-1, 3)
    magnitude = jnp.linalg.norm(point_current, axis=1)
    magnitude_derivative = regularization * jnp.sum(
        point_current * (point_matrix @ coefficient_derivative).reshape(-1, 3),
        axis=1) / magnitude
    derivative = jnp.vdot(
        jax.nn.softmax(magnitude / (
            MAX_CURRENT_SMOOTHING * target_max_current)),
        magnitude_derivative)
    derivative = jnp.where(jnp.abs(derivative) > 1e-30, derivative, -1e-30)
    newton_root = jnp.clip(
        log_regularization
        - (smooth_maximum(coefficients) - target_max_current) / derivative,
        jnp.log(1e-20), jnp.log(1e-8))
    log_regularization = jnp.where(
        low_maximum <= target_max_current, jnp.log(1e-20),
        jnp.where(high_maximum > target_max_current, jnp.log(1e-8),
                  newton_root))
    coefficients, f_b, _ = solve_current(
        operators, jnp.exp(log_regularization))
    max_current = jnp.max(jnp.linalg.norm(
        (point_matrix @ coefficients + point_fixed).reshape(-1, 3), axis=1))
    return coefficients, f_b, max_current, jnp.exp(log_regularization)


def spectral_objective(surface):
    return (jnp.sum(jnp.abs(surface.xm * surface.rc) ** 2)
            + jnp.sum(jnp.abs(surface.xm * surface.zs) ** 2))


def smooth_minimum_distance(surface1, surface2):
    distance = jnp.linalg.norm(
        surface1.gamma.reshape(-1, 1, 3)
        - surface2.gamma.reshape(1, -1, 3), axis=2).reshape(-1)
    return jnp.vdot(jax.nn.softmax(-SHARPNESS * distance), distance)


def smooth_minimum_tangent_radius(surface, neighbor_radius=2):
    """Nonlocal tangent-point thickness; nearby grid points are excluded."""
    points = surface.gamma.reshape(-1, 3)
    normals = surface.unitnormal.reshape(-1, 3)
    difference = points[None, :, :] - points[:, None, :]
    radius = jnp.sum(difference ** 2, axis=2) / (
        2 * jnp.abs(jnp.einsum("ijk,ik->ij", difference, normals)) + 1e-14)
    iphi, itheta = jnp.meshgrid(jnp.arange(surface.nphi),
                                jnp.arange(surface.ntheta), indexing="ij")
    dphi = jnp.abs(iphi.reshape(-1, 1) - iphi.reshape(1, -1))
    dtheta = jnp.abs(itheta.reshape(-1, 1) - itheta.reshape(1, -1))
    dphi = jnp.minimum(dphi, surface.nphi - dphi)
    dtheta = jnp.minimum(dtheta, surface.ntheta - dtheta)
    radius = jnp.where((dphi > neighbor_radius) | (dtheta > neighbor_radius),
                       radius, 1e6)
    return jnp.vdot(jax.nn.softmax(-SHARPNESS * radius.reshape(-1)),
                    radius.reshape(-1))


input_filepath = os.path.join(os.path.dirname(__file__), "input_files")
vmec_input = os.path.join(
    input_filepath, "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
with Dataset(vmec_input) as dataset:
    bvco = dataset.variables["bvco"][:]
net_poloidal_current = float(
    2 * jnp.pi / MU0 * (1.5 * bvco[-1] - 0.5 * bvco[-2]))

plasma_surface = SurfaceRZFourier.from_wout_file(
    vmec_input, s=1, ntheta=plasma_ntheta, nphi=plasma_nphi,
    close=False, range_torus="full torus")
winding_surface = SurfaceRZFourier.from_wout_file(
    vmec_input, s=1, ntheta=winding_ntheta, nphi=winding_nphi,
    close=False, range_torus="full torus")
winding_surface.dofs = normal_offset_dofs(
    winding_surface, mean_minor_radius(plasma_surface))


def individual_objectives(surface):
    operators = build_operators(
        plasma_surface, surface, net_poloidal_current)
    solutions = [solve_for_max_current(operators, target)
                 for target in TARGET_MAX_CURRENTS]
    f_b = jnp.asarray([solution[1] for solution in solutions])
    max_current = jnp.asarray([solution[2] for solution in solutions])
    regularization = jnp.asarray([solution[3] for solution in solutions])
    volume = INTEGRATION_FACTOR * jnp.abs(surface.volume)
    spectral = spectral_objective(surface)
    distance = smooth_minimum_distance(plasma_surface, surface)
    self_radius = smooth_minimum_tangent_radius(surface)
    minimum_jacobian = jnp.min(surface.area_element)
    self_penalty = 1 + jnp.tanh((MINIMUM_SELF_RADIUS - self_radius) / 0.01)
    jacobian_penalty = jnp.square(
        jnp.maximum(1e-6 - minimum_jacobian, 0.0)) * 1e12
    return (f_b, max_current, regularization, volume, spectral, distance,
            minimum_jacobian, self_radius, jacobian_penalty, self_penalty)


initial = individual_objectives(winding_surface)
x0_full = winding_surface.dofs
nmodes = winding_surface.xm.size
active_modes = jnp.where(
    (winding_surface.xm <= ACTIVE_MPOL)
    & (jnp.abs(winding_surface.xn / winding_surface.nfp) <= ACTIVE_NTOR))[0]
nonconstant_modes = active_modes[active_modes != 0]
active_indices = jnp.concatenate((nonconstant_modes,
                                  nonconstant_modes + nmodes))
x0 = x0_full[active_indices]
step_bound = min(COEFFICIENT_STEP_BOUND,
                 0.02 * float(mean_minor_radius(plasma_surface)))
bounds = [(float(x - step_bound), float(x + step_bound)) for x in x0]


def objective_function(active_dofs):
    winding_surface.dofs = x0_full.at[active_indices].set(active_dofs)
    values = individual_objectives(winding_surface)
    return (jnp.mean(values[0] / initial[0])
            + 100 * jnp.mean(jnp.maximum(
                values[1] / TARGET_MAX_CURRENTS - 1, 0) ** 2)
            + VOLUME_WEIGHT * (values[3] / initial[3] - 1) ** 2
            + SPECTRAL_WEIGHT * values[4] / initial[4]
            + DISTANCE_WEIGHT * jnp.maximum(
                1 - values[5] / (0.9 * initial[5]), 0) ** 2
            + 100 * values[8]
            + SELF_INTERSECTION_WEIGHT * values[9])


value_and_grad = jax.jit(jax.value_and_grad(objective_function))
evaluation_count = 0


def fun(x):
    global evaluation_count
    value, gradient = value_and_grad(jnp.asarray(x))
    evaluation_count += 1
    if evaluation_count == 1 or evaluation_count % 10 == 0:
        print(f"evaluation {evaluation_count:03d}: objective = {float(value):.9e}")
    return float(value), np.asarray(gradient)


start_time = time.perf_counter()
result = minimize(fun, np.asarray(x0), method="L-BFGS-B", jac=True,
                  bounds=bounds, options={"maxiter": MAXITER})
elapsed_time = time.perf_counter() - start_time
winding_surface.dofs = x0_full.at[active_indices].set(jnp.asarray(result.x))
final = individual_objectives(winding_surface)

print(f"\noptimizer: success={result.success}, iterations={result.nit}, "
      f"evaluations={result.nfev}, elapsed={elapsed_time:.1f} s, "
      f"objective={result.fun:.9e}")
for target, before, after in zip(TARGET_MAX_CURRENTS, initial[0], final[0]):
    print(f"normal_field_error_T2_m2 at {float(target) / 1e6:g} MA/m: "
          f"{float(before):.9e} -> {float(after):.9e}")
for name, before, after in zip(
        ("volume_m3", "spectral_penalty", "plasma_distance_m",
         "minimum_surface_jacobian", "tangent_point_radius_m"),
        initial[3:8], final[3:8]):
    print(f"{name}: {float(before):.9e} -> {float(after):.9e}")

operators = build_operators(plasma_surface, winding_surface,
                            net_poloidal_current)
coefficients = solve_for_max_current(
    operators, TARGET_MAX_CURRENTS[1])[0]
potential_xm, potential_xn = potential_modes(
    POTENTIAL_MPOL, POTENTIAL_NTOR, winding_surface.nfp)
output_file = "winding_surface_opt.npz"
np.savez(output_file, dofs=np.asarray(winding_surface.dofs),
         coefficients=np.asarray(coefficients),
         potential_xm=np.asarray(potential_xm),
         potential_xn=np.asarray(potential_xn),
         net_poloidal_current=net_poloidal_current,
         target_max_current=np.asarray(TARGET_MAX_CURRENTS[1]),
         vmec_input=vmec_input)
print(f"saved surface and 7 MA/m current potential to {output_file}")
