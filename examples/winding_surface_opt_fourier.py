"""Optimize a winding surface using a Fourier-coefficient SVD/entropy surrogate.

Same idea as winding_surface_opt.py's dipole/grid SVD surrogate, but the
winding-surface current potential Phi is represented as a truncated Fourier
series (the same sin(m*theta - n*nfp*phi) basis winding_surface_opt_2.py uses)
instead of as raw dipole strengths at grid points. The induction matrix's
columns are then "set Fourier coefficient k to 1," not "put a unit dipole at
grid point j." No current-potential solve is performed; the entropy objective
is computed straight from that matrix's singular values, exactly as before.
"""

import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from scipy.optimize import minimize

from essos.surfaces import SurfaceRZFourier

MU0 = 4 * jnp.pi * 1e-7
INTEGRATION_FACTOR = 4 * jnp.pi ** 2

SVD_WEIGHT = 10.0
SINGULAR_STRENGTH_WEIGHT = 0.0
VOLUME_WEIGHT = 0.02
SPECTRAL_WEIGHT = 0.02
DISTANCE_WEIGHT = 5.0
SELF_INTERSECTION_WEIGHT = 100.0
SELF_NEIGHBOR_RADIUS = 2
SHARPNESS = 300.0
MAXITER = 20
ACTIVE_MPOL = 2
ACTIVE_NTOR = 2

# Fourier truncation for the current potential Phi (not the winding surface's
# own R,Z shape). Small on purpose: this is a smoke test of the pipeline, not
# a tuned production basis (the comparison study uses mpol=ntor=6).
POTENTIAL_MPOL = 3
POTENTIAL_NTOR = 3

# Smoke-test grids. winding_ntheta/nphi is a quadrature grid for the induction
# integral (>= ~4x the highest retained Fourier harmonic), not a set of
# independent degrees of freedom.
plasma_ntheta = 16
plasma_nphi = 16
winding_ntheta = 24
winding_nphi = 32

OBJECTIVE_NAMES = ("inverse_singular_entropy", "singular_strength", "volume_m3",
                   "spectral_penalty", "plasma_distance_penalty", "local_area_penalty",
                   "smooth_self_radius_m", "self_intersection_penalty")


# ---------------------------------------------------------------------------
# Copied verbatim from winding_surface_opt.py: geometry / objective machinery
# that does not depend on how the current potential is represented.
# ---------------------------------------------------------------------------

def mean_minor_radius(surface):
    return jnp.sqrt(INTEGRATION_FACTOR * surface.mean_cross_sectional_area() / jnp.pi)

def normal_offset_dofs(surface, offset):
    """Offset grid points along the outward normal and refit the Fourier surface.

    Copied from winding_surface_opt_2.py rather than winding_surface_opt.py's
    extend_via_normal_dofs: the latter adds offset*unitnormal and was verified
    (empirically, on this file's QA equilibrium) to place the winding surface
    only ~0.15 m from the plasma instead of the intended ~1.7 m offset, i.e.
    it offsets in the wrong direction. Subtracting reproduces the intended
    offset (checked: resulting min distance equals `offset` exactly for a
    surface that starts coincident with the plasma).
    """
    nmodes = surface.xm.size
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

def surface_quadrature_weights(surface):
    """Periodic trapezoidal weights, including the surface Jacobian."""
    ntheta_cells = surface.ntheta - int(surface.close)
    nphi_cells = surface.nphi - int(surface.close)
    phi_period = 2 * jnp.pi if surface.range_torus == "full torus" else jnp.pi / surface.nfp
    endpoint_theta = jnp.ones(surface.ntheta).at[jnp.array([0, -1])].set(0.5) if surface.close else jnp.ones(surface.ntheta)
    endpoint_phi = jnp.ones(surface.nphi).at[jnp.array([0, -1])].set(0.5) if surface.close else jnp.ones(surface.nphi)
    return (surface.area_element * endpoint_phi[:, None] * endpoint_theta[None, :]
            * (2 * jnp.pi / ntheta_cells) * (phi_period / nphi_cells)).reshape(-1)

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


# ---------------------------------------------------------------------------
# Copied verbatim from winding_surface_opt_2.py: Fourier current-potential
# machinery (REGCOIL-style single-valued Phi basis and its induced fields).
# ---------------------------------------------------------------------------

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
    return kernel @ (surface_quadrature_weights(winding)[:, None] * potential)

def fixed_current_normal_field(plasma, winding, current_numerator):
    """Biot-Savart normal field from a J*K surface-current numerator field."""
    difference = (plasma.gamma.reshape(-1, 1, 3)
                  - winding.gamma.reshape(1, -1, 3))
    inverse_distance_cubed = jnp.sum(difference ** 2, axis=2) ** -1.5
    cross = jnp.cross(current_numerator[None, :, :], difference, axis=2)
    dtheta = 2 * jnp.pi / (winding.ntheta - int(winding.close))
    dphi = 2 * jnp.pi / (winding.nphi - int(winding.close))
    return (MU0 / (4 * jnp.pi) * dtheta * dphi
            * jnp.einsum("pqk,pq,pk->p", cross, inverse_distance_cubed,
                         plasma.unitnormal.reshape(-1, 3)))


# ---------------------------------------------------------------------------
# New: build the induction matrix from Fourier coefficients of Phi instead of
# from dipole strengths at grid points.
# ---------------------------------------------------------------------------

def fourier_basis_grid(winding_surface, xm, xn):
    """sin(m*theta - n*phi) basis functions evaluated at each winding grid point."""
    theta = winding_surface.theta2d.reshape(-1, 1)
    phi = winding_surface.phi2d.reshape(-1, 1)
    return jnp.sin(xm[None, :] * theta - xn[None, :] * phi)  # (n_winding_grid, n_modes)

def fourier_induction_matrix(plasma_surface, winding_surface, xm, xn):
    """Row-weighted map from Phi's Fourier coefficients to B_normal on the plasma surface."""
    potential_basis = fourier_basis_grid(winding_surface, xm, xn)
    physical_field = dipole_normal_field(plasma_surface, winding_surface, potential_basis)
    plasma_weights = surface_quadrature_weights(plasma_surface)
    return jnp.sqrt(plasma_weights)[:, None] * physical_field


# ---------------------------------------------------------------------------
# Load surfaces.
# ---------------------------------------------------------------------------

input_filepath = os.path.join(os.path.dirname(__file__), "input_files")
vmec_input = os.path.join(
    input_filepath, "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")

def make_plasma_and_winding(plasma_ntheta, plasma_nphi, winding_ntheta, winding_nphi):
    """Fresh plasma/winding surface pair at a given grid resolution."""
    plasma = SurfaceRZFourier.from_wout_file(
        vmec_input, s=1, ntheta=plasma_ntheta, nphi=plasma_nphi,
        close=False, range_torus="full torus")
    winding = SurfaceRZFourier.from_wout_file(
        vmec_input, s=1, ntheta=winding_ntheta, nphi=winding_nphi,
        close=False, range_torus="full torus")
    offset = mean_minor_radius(plasma)
    winding.dofs = normal_offset_dofs(winding, offset)
    return plasma, winding, offset

plasma_surface, winding_surface, minor_radius_plasma = make_plasma_and_winding(
    plasma_ntheta, plasma_nphi, winding_ntheta, winding_nphi)

potential_xm, potential_xn = potential_modes(
    POTENTIAL_MPOL, POTENTIAL_NTOR, winding_surface.nfp)
n_potential_modes = potential_xm.size

# Geometric penalty scales, relative to the plasma's minor radius rather than
# hardcoded meters, so they stay sensible regardless of which equilibrium is
# loaded.
MINIMUM_DISTANCE = 0.3 * float(minor_radius_plasma)
DISTANCE_WALL_SCALE = 2.5 * MINIMUM_DISTANCE
MINIMUM_SELF_RADIUS = 0.3 * float(minor_radius_plasma)
SELF_RADIUS_WALL_SCALE = 0.2 * MINIMUM_SELF_RADIUS
COEFFICIENT_STEP_BOUND = min(1.0, 0.1 * float(minor_radius_plasma))


# ---------------------------------------------------------------------------
# Benchmark: does fourier_induction_matrix compute the correct field?
# ---------------------------------------------------------------------------

def biot_savart_field_from_coefficients(plasma_surface, winding_surface, coefficients):
    """Independent Biot-Savart field of K = n x grad(Phi_sv), Phi_sv only (no secular term)."""
    _, _, basis, _ = current_numerators(winding_surface, net_poloidal_current=0.0)
    current_numerator = jnp.einsum("m,mgc->gc", coefficients, basis)
    return fixed_current_normal_field(plasma_surface, winding_surface, current_numerator)

def benchmark_induction_matrix(plasma_surface, winding_surface, label):
    rng = np.random.default_rng(0)
    test_vectors = [jnp.zeros(n_potential_modes).at[i].set(1.0)
                     for i in np.linspace(0, n_potential_modes - 1, 5).astype(int)]
    test_vectors += [jnp.asarray(rng.normal(size=n_potential_modes)) for _ in range(2)]

    potential_basis = fourier_basis_grid(winding_surface, potential_xm, potential_xn)
    plasma_weights = surface_quadrature_weights(plasma_surface)

    flux_errors = []
    sign_errors = []
    for coefficients in test_vectors:
        B_dipole = dipole_normal_field(plasma_surface, winding_surface, potential_basis) @ coefficients
        scale = jnp.max(jnp.abs(B_dipole))

        flux = jnp.sum(plasma_weights * B_dipole)
        flux_errors.append(float(jnp.abs(flux) / (scale * jnp.sum(plasma_weights))))

        B_biot_savart = biot_savart_field_from_coefficients(
            plasma_surface, winding_surface, coefficients)
        err_plus = jnp.max(jnp.abs(B_dipole - B_biot_savart)) / scale
        err_minus = jnp.max(jnp.abs(B_dipole + B_biot_savart)) / scale
        sign_errors.append((float(err_plus), float(err_minus)))

    best_sign = "+" if sum(e[0] for e in sign_errors) < sum(e[1] for e in sign_errors) else "-"
    best_errors = [e[0] if best_sign == "+" else e[1] for e in sign_errors]

    print(f"\nbenchmark [{label}]: "
          f"grid=({winding_surface.ntheta}x{winding_surface.nphi} winding, "
          f"{plasma_surface.ntheta}x{plasma_surface.nphi} plasma), "
          f"n_modes={n_potential_modes}")
    print(f"  flux-conservation relative error: max={max(flux_errors):.3e}, "
          f"mean={np.mean(flux_errors):.3e}")
    print(f"  dipole-layer vs Biot-Savart relative error (sign='{best_sign}'): "
          f"max={max(best_errors):.3e}, mean={np.mean(best_errors):.3e}")
    return max(flux_errors), max(best_errors)

# The Biot-Savart cross-check needs a finer quadrature grid than the
# optimization loop does to converge to a small residual (checked separately:
# the naive ~4x-oversampling rule undersamples the near-field falloff of the
# Biot-Savart kernel specifically). Use dedicated, higher-resolution surfaces
# for this check only; the optimization loop keeps its own cheaper grid.
print("Benchmarking fourier_induction_matrix against independent field calculations...")
benchmark_plasma, benchmark_winding, _ = make_plasma_and_winding(32, 32, 48, 64)
smoke_flux_error, smoke_sign_error = benchmark_induction_matrix(
    benchmark_plasma, benchmark_winding, "benchmark resolution")

refined_plasma, refined_winding, _ = make_plasma_and_winding(64, 64, 96, 128)
fine_flux_error, fine_sign_error = benchmark_induction_matrix(
    refined_plasma, refined_winding, "2x benchmark resolution")

print(f"\nconvergence check: flux error {smoke_flux_error:.3e} -> {fine_flux_error:.3e}, "
      f"Biot-Savart error {smoke_sign_error:.3e} -> {fine_sign_error:.3e} "
      f"({'shrank' if fine_flux_error < smoke_flux_error and fine_sign_error < smoke_sign_error else 'DID NOT SHRINK -- investigate'})")


# ---------------------------------------------------------------------------
# Surrogate objective and optimization (mirrors winding_surface_opt.py, with
# fourier_induction_matrix in place of reduced_memory_induction_matrix).
# ---------------------------------------------------------------------------

def individual_objectives(winding_surface):
    winding_area_elements = winding_surface.area_element.reshape(-1)

    induction_matrix = fourier_induction_matrix(
        plasma_surface, winding_surface, potential_xm, potential_xn)
    _, singular_values, _ = jnp.linalg.svd(induction_matrix, compute_uv=True, full_matrices=False)

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


# ---------------------------------------------------------------------------
# Gradient check: finite differences vs. jax.grad on a handful of active DOFs.
# ---------------------------------------------------------------------------

def finite_difference_gradient_check(x0, n_check=5, eps=1e-6):
    rng = np.random.default_rng(0)
    indices = rng.choice(x0.size, size=min(n_check, x0.size), replace=False)
    _, analytic_grad = value_and_grad(x0)
    errors = []
    for i in indices:
        perturbation = jnp.zeros_like(x0).at[i].set(eps)
        value_plus, _ = value_and_grad(x0 + perturbation)
        value_minus, _ = value_and_grad(x0 - perturbation)
        finite_difference = (value_plus - value_minus) / (2 * eps)
        relative_error = jnp.abs(finite_difference - analytic_grad[i]) / jnp.maximum(
            jnp.abs(analytic_grad[i]), 1e-12)
        errors.append(float(relative_error))
    return indices, errors

print("\nChecking gradients against central finite differences...")
checked_indices, gradient_errors = finite_difference_gradient_check(x0)
for index, error in zip(checked_indices, gradient_errors):
    print(f"  active_dof[{index}]: relative error = {error:.3e}")
print(f"  max relative error = {max(gradient_errors):.3e}")


# ---------------------------------------------------------------------------
# Optimize.
# ---------------------------------------------------------------------------

evaluation_count = 0

def fun(x):
    global evaluation_count
    value, grad = value_and_grad(x)
    evaluation_count += 1
    print(f"evaluation {evaluation_count:03d}: objective = {float(value):.9e}")
    return value, grad

initial_objective = float(objective_function(x0))
res = minimize(fun, x0, method='L-BFGS-B', jac=True, bounds=bounds, options={"maxiter": MAXITER})

res_full = x0_full.at[active_indices].set(jnp.asarray(res.x))
winding_surface.dofs = res_full
final_objectives = individual_objectives(winding_surface)

print(f"\n{__doc__}")
print(f"optimizer: success={res.success}, iterations={res.nit}, evaluations={res.nfev}, "
      f"objective={initial_objective:.9e} -> {res.fun:.9e}")
for name, initial, final in zip(OBJECTIVE_NAMES, scales, final_objectives):
    print(f"{name}: {float(initial):.9e} -> {float(final):.9e}")
print(f"smooth_plasma_distance_m: {float(smooth_minimum_distance(plasma_surface, winding_surface, SHARPNESS)):.9e}")
print(f"active_dof_step_norm: {float(jnp.linalg.norm(x0 - res.x)):.9e}")
