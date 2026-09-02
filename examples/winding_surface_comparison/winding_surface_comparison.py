"""Compare winding-surface optimizers and current-potential solvers.

Install the comparison codes with ``pip install regcoil==0.1.3`` and
``pip install git+https://github.com/lankef/quadcoil.git@368f4aa``.
Set ``REGCOIL_ADJOINT`` to the legacy REGCOIL executable. The study compares
normal offsets, two ESSOS objectives, and REGCOIL's adjoint surface optimizer.
The current solvers use the same surface, Fourier basis, grids, Kmax, cutter,
and independent sheet and filament evaluations. Run this file with no arguments;
set ``RESUME_COMPARISON=1`` to reuse completed runs.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import psutil
from contourpy import LineType, contour_generator
from netCDF4 import Dataset
from scipy.optimize import minimize

from essos.surfaces import SurfaceRZFourier


MU0 = 4 * np.pi * 1e-7
INTEGRATION_FACTOR = 4 * np.pi ** 2
KMAX = 10e6
KMAX_TARGETS = jnp.asarray((0.95, 1.0, 1.05)) * KMAX
POTENTIAL_MPOL = 6
POTENTIAL_NTOR = 6
COILS_PER_HALF_PERIOD = 10

OPT_NTHETA = 12
OPT_NPHI_PER_PERIOD = 12
PARETO_RESOLUTION = 32
CURRENT_NTHETA = 48
CURRENT_NPHI_PER_PERIOD = 48
CUT_RESOLUTION = 96
FIELD_RESOLUTION = 48
MAXITER = 80
ADJOINT_MAXITER = 12
ACTIVE_MPOL = 2
ACTIVE_NTOR = 2
COEFFICIENT_STEP_BOUND = 0.04
MAX_CURRENT_SMOOTHING = 1e-3

VOLUME_WEIGHT = 100.0
SPECTRAL_WEIGHT = 0.02
DISTANCE_WEIGHT = 1e6
SELF_INTERSECTION_WEIGHT = 100.0
MINIMUM_SELF_RADIUS = 0.05
SHARPNESS = 300.0

ROOT = os.path.dirname(__file__)
EXAMPLES = os.path.dirname(ROOT)
OUTPUT = os.path.join(ROOT, "output")
METHODS = ("ESSOS", "REGCOIL", "QUADCOIL")
OBJECTIVES = ("entropy", "three-point Pareto")
SURFACE_METHODS = ("normal offset", "ESSOS entropy", "ESSOS Pareto",
                   "REGCOIL adjoint")


def make_surface(wout, ntheta, nphi_per_period, dofs=None):
    with Dataset(wout) as dataset:
        nfp = int(dataset.variables["nfp"][0])
    surface = SurfaceRZFourier.from_wout_file(
        wout, s=1, ntheta=ntheta, nphi=nphi_per_period * nfp,
        close=False, range_torus="full torus")
    if dofs is not None:
        surface.dofs = jnp.asarray(dofs)
    return surface


def quadrature_weights(surface, one_period=False, full_integral=False):
    weights = surface.area_element * (2 * jnp.pi / surface.ntheta) * (
        2 * jnp.pi / surface.nphi)
    if one_period:
        weights = weights[:surface.nphi // surface.nfp]
        if full_integral:
            weights = surface.nfp * weights
    return weights.reshape(-1)


def mean_minor_radius(surface):
    return jnp.sqrt(
        INTEGRATION_FACTOR * surface.mean_cross_sectional_area() / jnp.pi)


def normal_offset_dofs(surface, offset):
    points = surface.gamma - offset * surface.unitnormal
    radius = jnp.linalg.norm(points[:, :, :2], axis=2)
    phi = jnp.arctan2(points[:, :, 1], points[:, :, 0])
    angle = (surface.xm[:, None, None] * surface.theta2d[None]
             - surface.xn[:, None, None] * phi[None])
    nmodes = surface.xm.size
    rc = jnp.linalg.lstsq(jnp.cos(angle).reshape(nmodes, -1).T,
                          radius.reshape(-1), rcond=None)[0]
    zs = jnp.linalg.lstsq(jnp.sin(angle).reshape(nmodes, -1).T,
                          points[:, :, 2].reshape(-1), rcond=None)[0]
    return jnp.concatenate((rc * surface.scaling, zs * surface.scaling))


def potential_modes(nfp):
    xm = [0] * POTENTIAL_NTOR
    xn = list(range(1, POTENTIAL_NTOR + 1))
    for m in range(1, POTENTIAL_MPOL + 1):
        for n in range(-POTENTIAL_NTOR, POTENTIAL_NTOR + 1):
            xm.append(m)
            xn.append(n)
    return jnp.asarray(xm), nfp * jnp.asarray(xn)


def dipole_kernel(plasma_points, plasma_normals, winding_points,
                  winding_normals):
    difference = plasma_points[:, None, :] - winding_points[None, :, :]
    distance_squared = jnp.sum(difference ** 2, axis=2)
    return MU0 / (4 * jnp.pi) * (
        jnp.einsum("pk,qk->pq", plasma_normals, winding_normals)
        - 3 * jnp.einsum("pqk,pk->pq", difference, plasma_normals)
        * jnp.einsum("pqk,qk->pq", difference, winding_normals)
        / distance_squared) / distance_squared ** 1.5


def induction_singular_values(plasma, winding):
    """Exact block-Fourier reduction of the full-torus induction matrix."""
    nphi_period = plasma.nphi // plasma.nfp
    plasma_points = plasma.gamma[:nphi_period].reshape(-1, 3)
    plasma_normals = plasma.unitnormal[:nphi_period].reshape(-1, 3)
    plasma_weights = quadrature_weights(plasma, one_period=True)
    winding_weights = quadrature_weights(winding).reshape(
        winding.nphi, winding.ntheta)
    blocks = []
    for period in range(winding.nfp):
        section = slice(period * nphi_period, (period + 1) * nphi_period)
        kernel = dipole_kernel(
            plasma_points, plasma_normals,
            winding.gamma[section].reshape(-1, 3),
            winding.unitnormal[section].reshape(-1, 3))
        blocks.append(jnp.sqrt(plasma_weights[:, None]) * kernel
                      * jnp.sqrt(winding_weights[section].reshape(-1)[None, :]))
    singular_values = []
    for mode in range(winding.nfp):
        matrix = sum(jnp.exp(-2j * jnp.pi * mode * period / winding.nfp)
                     * block for period, block in enumerate(blocks))
        singular_values.append(jnp.linalg.svd(matrix, compute_uv=False))
    return jnp.concatenate(singular_values)


def build_operators(plasma, winding, net_current, plasma_bnormal=None):
    nphi_period = plasma.nphi // plasma.nfp
    plasma_points = plasma.gamma[:nphi_period].reshape(-1, 3)
    plasma_normals = plasma.unitnormal[:nphi_period].reshape(-1, 3)
    winding_points = winding.gamma.reshape(-1, 3)
    winding_normals = winding.unitnormal.reshape(-1, 3)
    xm, xn = potential_modes(winding.nfp)

    theta = winding.theta2d.reshape(-1)
    phi = winding.phi2d.reshape(-1)
    potential = jnp.sin(theta[:, None] * xm[None, :]
                        - phi[:, None] * xn[None, :])
    field_matrix = dipole_kernel(
        plasma_points, plasma_normals, winding_points, winding_normals)
    field_matrix = field_matrix @ (
        quadrature_weights(winding)[:, None] * potential)

    rtheta = winding.gammadash_theta.reshape(-1, 3)
    rphi = winding.gammadash_phi.reshape(-1, 3)
    current_basis = jnp.cos(
        xm[:, None] * theta - xn[:, None] * phi)[:, :, None] * (
            xn[:, None, None] * rtheta[None]
            + xm[:, None, None] * rphi[None])
    fixed_numerator = net_current * rtheta / (2 * jnp.pi)
    difference = plasma_points[:, None, :] - winding_points[None, :, :]
    fixed_field = -MU0 / (4 * jnp.pi) * (
        2 * jnp.pi / winding.ntheta) * (2 * jnp.pi / winding.nphi) * jnp.einsum(
            "pqk,pq,pk->p", jnp.cross(fixed_numerator[None], difference),
            jnp.sum(difference ** 2, axis=2) ** -1.5, plasma_normals)
    if plasma_bnormal is not None:
        fixed_field = fixed_field + jnp.asarray(plasma_bnormal).reshape(-1)

    nbase = nphi_period * winding.ntheta
    jacobian = winding.area_element[:nphi_period].reshape(-1)
    point_matrix = (-jnp.transpose(current_basis[:, :nbase], (1, 2, 0))
                    / jacobian[:, None, None])
    point_fixed = fixed_numerator[:nbase] / jacobian[:, None]
    plasma_sqrt_weight = jnp.sqrt(
        quadrature_weights(plasma, one_period=True, full_integral=True))
    winding_sqrt_weight = jnp.sqrt(
        quadrature_weights(winding, one_period=True, full_integral=True))
    current_matrix = (
        winding_sqrt_weight[:, None, None] * point_matrix).reshape(-1, len(xm))
    fixed_current = (winding_sqrt_weight[:, None] * point_fixed).reshape(-1)
    return (plasma_sqrt_weight[:, None] * field_matrix,
            plasma_sqrt_weight * fixed_field, current_matrix, fixed_current,
            point_matrix.reshape(-1, len(xm)), point_fixed.reshape(-1),
            plasma_sqrt_weight)


def solve_current(operators, regularization):
    field_matrix, fixed_field, current_matrix, fixed_current = operators[:4]
    coefficients = jnp.linalg.solve(
        field_matrix.T @ field_matrix
        + regularization * current_matrix.T @ current_matrix,
        -(field_matrix.T @ fixed_field
          + regularization * current_matrix.T @ fixed_current))
    field = field_matrix @ coefficients + fixed_field
    current = current_matrix @ coefficients + fixed_current
    return coefficients, jnp.vdot(field, field), jnp.vdot(current, current)


def solve_for_max_current(operators, target):
    point_matrix, point_fixed = operators[4:6]

    def smooth_maximum(coefficients):
        magnitude = jnp.linalg.norm(
            (point_matrix @ coefficients + point_fixed).reshape(-1, 3), axis=1)
        scale = MAX_CURRENT_SMOOTHING * target
        return scale * jax.scipy.special.logsumexp(magnitude / scale)

    lower, upper = jnp.log(1e-20), jnp.log(1e-8)
    low_max = smooth_maximum(solve_current(operators, jnp.exp(lower))[0])
    high_max = smooth_maximum(solve_current(operators, jnp.exp(upper))[0])
    for _ in range(22):
        middle = 0.5 * (lower + upper)
        maximum = smooth_maximum(solve_current(operators, jnp.exp(middle))[0])
        lower = jnp.where(maximum > target, middle, lower)
        upper = jnp.where(maximum > target, upper, middle)
    log_lambda = jax.lax.stop_gradient(0.5 * (lower + upper))
    regularization = jnp.exp(log_lambda)
    coefficients = solve_current(operators, regularization)[0]
    lhs = (operators[0].T @ operators[0]
           + regularization * operators[2].T @ operators[2])
    derivative_coefficients = jnp.linalg.solve(
        lhs, -(operators[2].T @ operators[3]
               + operators[2].T @ operators[2] @ coefficients))
    current = (point_matrix @ coefficients + point_fixed).reshape(-1, 3)
    magnitude = jnp.linalg.norm(current, axis=1)
    magnitude_derivative = regularization * jnp.sum(
        current * (point_matrix @ derivative_coefficients).reshape(-1, 3),
        axis=1) / magnitude
    derivative = jnp.vdot(jax.nn.softmax(
        magnitude / (MAX_CURRENT_SMOOTHING * target)), magnitude_derivative)
    root = jnp.clip(log_lambda - (smooth_maximum(coefficients) - target)
                    / jnp.where(jnp.abs(derivative) > 1e-30, derivative, -1e-30),
                    jnp.log(1e-20), jnp.log(1e-8))
    log_lambda = jnp.where(low_max <= target, jnp.log(1e-20),
                           jnp.where(high_max > target, jnp.log(1e-8), root))
    coefficients, f_b, _ = solve_current(operators, jnp.exp(log_lambda))
    maximum = jnp.max(jnp.linalg.norm(
        (point_matrix @ coefficients + point_fixed).reshape(-1, 3), axis=1))
    return coefficients, f_b, maximum


def spectral_objective(surface):
    return (jnp.sum(jnp.abs(surface.xm * surface.rc) ** 2)
            + jnp.sum(jnp.abs(surface.xm * surface.zs) ** 2))


def smooth_minimum_distance(plasma, winding):
    nphi_period = winding.nphi // winding.nfp
    distance = jnp.linalg.norm(
        winding.gamma[:nphi_period].reshape(-1, 1, 3)
        - plasma.gamma.reshape(1, -1, 3), axis=2).reshape(-1)
    return jnp.vdot(jax.nn.softmax(-SHARPNESS * distance), distance)


def smooth_tangent_radius(surface, neighbor_radius=2):
    nphi_period = surface.nphi // surface.nfp
    points = surface.gamma.reshape(-1, 3)
    base_points = surface.gamma[:nphi_period].reshape(-1, 3)
    base_normals = surface.unitnormal[:nphi_period].reshape(-1, 3)
    difference = points[None] - base_points[:, None]
    radius = jnp.sum(difference ** 2, axis=2) / (
        2 * jnp.abs(jnp.einsum("ijk,ik->ij", difference, base_normals)) + 1e-14)
    base_phi, base_theta = jnp.meshgrid(
        jnp.arange(nphi_period), jnp.arange(surface.ntheta), indexing="ij")
    all_phi, all_theta = jnp.meshgrid(
        jnp.arange(surface.nphi), jnp.arange(surface.ntheta), indexing="ij")
    dphi = jnp.abs(base_phi.reshape(-1, 1) - all_phi.reshape(1, -1))
    dtheta = jnp.abs(base_theta.reshape(-1, 1) - all_theta.reshape(1, -1))
    dphi = jnp.minimum(dphi, surface.nphi - dphi)
    dtheta = jnp.minimum(dtheta, surface.ntheta - dtheta)
    radius = jnp.where((dphi > neighbor_radius) | (dtheta > neighbor_radius),
                       radius, 1e6).reshape(-1)
    return jnp.vdot(jax.nn.softmax(-SHARPNESS * radius), radius)


def geometry_metrics(plasma, surface):
    volume = INTEGRATION_FACTOR * jnp.abs(surface.volume)
    distance = smooth_minimum_distance(plasma, surface)
    self_radius = smooth_tangent_radius(surface)
    minimum_jacobian = jnp.min(surface.area_element)
    jacobian_penalty = jnp.maximum(1e-6 - minimum_jacobian, 0) ** 2 * 1e12
    self_penalty = 1 + jnp.tanh((MINIMUM_SELF_RADIUS - self_radius) / 0.01)
    return (volume, spectral_objective(surface), distance, self_radius,
            minimum_jacobian, jacobian_penalty, self_penalty)


def net_poloidal_current(wout):
    with Dataset(wout) as dataset:
        bvco = dataset.variables["bvco"][:]
    return float(2 * np.pi / MU0 * (1.5 * bvco[-1] - 0.5 * bvco[-2]))


def load_plasma_bnormal(wout, source, ntheta, nphi_per_period):
    if source is None:
        return None
    import regcoil
    plasma = regcoil.PlasmaSurface.from_wout(
        wout, ntheta=ntheta, nzeta=nphi_per_period)
    if str(source).endswith(".nc"):
        plasma.set_bnormal_from_virtual_casing(source)
    else:
        plasma.set_bnormal_from_bnorm_file(source)
    return np.asarray(plasma.Bnormal_from_plasma_current).T


def optimize_surface(wout, objective, bnormal_source=None):
    resolution = PARETO_RESOLUTION if objective == "pareto" else OPT_NTHETA
    plasma = make_surface(wout, resolution, resolution)
    winding = make_surface(wout, resolution, resolution)
    winding.dofs = baseline_surface(wout)
    net_current = net_poloidal_current(wout)
    plasma_bnormal = load_plasma_bnormal(
        wout, bnormal_source, resolution, resolution)

    def values(surface):
        geometry = geometry_metrics(plasma, surface)
        if objective == "entropy":
            singular_values = induction_singular_values(plasma, surface)
            probabilities = singular_values / jnp.sum(singular_values)
            entropy = -jnp.sum(probabilities * jnp.log(
                jnp.maximum(probabilities, 1e-300)))
            return (1 / entropy, jnp.sum(singular_values)) + geometry
        operators = build_operators(
            plasma, surface, net_current, plasma_bnormal)
        solutions = [solve_for_max_current(operators, target)
                     for target in KMAX_TARGETS]
        return (jnp.asarray([solution[1] for solution in solutions]),
                jnp.asarray([solution[2] for solution in solutions])) + geometry

    initial = values(winding)
    jax.block_until_ready(initial)
    initial_dofs = winding.dofs
    nmodes = winding.xm.size
    active_modes = jnp.where(
        (winding.xm <= ACTIVE_MPOL)
        & (jnp.abs(winding.xn / winding.nfp) <= ACTIVE_NTOR))[0]
    active = jnp.concatenate((active_modes[active_modes != 0],
                              active_modes[active_modes != 0] + nmodes))
    x0 = initial_dofs[active]
    step_bound = min(COEFFICIENT_STEP_BOUND,
                     0.02 * float(mean_minor_radius(plasma)))
    bounds = [(float(x - step_bound), float(x + step_bound)) for x in x0]

    def objective_function(active_dofs):
        winding.dofs = initial_dofs.at[active].set(active_dofs)
        current = values(winding)
        if objective == "entropy":
            physics = (current[0] / initial[0]
                       + jnp.maximum(1 - current[1] / initial[1], 0) ** 2)
            geometry, reference = current[2:], initial[2:]
            volume_term = -0.02 * geometry[0] / reference[0]
        else:
            physics = (jnp.mean(current[0] / initial[0])
                       + 100 * jnp.mean(jnp.maximum(
                           current[1] / KMAX_TARGETS - 1, 0) ** 2))
            geometry, reference = current[2:], initial[2:]
            volume_term = VOLUME_WEIGHT * (geometry[0] / reference[0] - 1) ** 2
        return (physics + volume_term
                + SPECTRAL_WEIGHT * geometry[1] / reference[1]
                + DISTANCE_WEIGHT * jnp.maximum(
                    1 - geometry[2] / (0.9 * reference[2]), 0) ** 2
                + 100 * geometry[5]
                + SELF_INTERSECTION_WEIGHT * geometry[6])

    value_and_grad = jax.jit(jax.value_and_grad(objective_function))
    evaluations = 0

    def fun(x):
        nonlocal evaluations
        value, gradient = value_and_grad(jnp.asarray(x))
        evaluations += 1
        if evaluations == 1 or evaluations % 10 == 0:
            print(f"{os.path.basename(wout)} {objective} evaluation "
                  f"{evaluations:03d}: {float(value):.8e}")
        return float(value), np.asarray(gradient)

    start = time.perf_counter()
    result = minimize(fun, np.asarray(x0), method="L-BFGS-B", jac=True,
                      bounds=bounds, options={"maxiter": MAXITER})
    elapsed = time.perf_counter() - start
    winding.dofs = initial_dofs.at[active].set(jnp.asarray(result.x))
    final_geometry = geometry_metrics(plasma, winding)
    print(f"{objective}: success={result.success}, iterations={result.nit}, "
          f"runtime={elapsed:.2f} s, distance={float(final_geometry[2]):.4f} m")
    return np.asarray(winding.dofs), elapsed, int(result.nit), bool(result.success)


def baseline_surface(wout):
    plasma = make_surface(wout, CURRENT_NTHETA, CURRENT_NPHI_PER_PERIOD)
    winding = make_surface(wout, CURRENT_NTHETA, CURRENT_NPHI_PER_PERIOD)
    winding.dofs = normal_offset_dofs(winding, mean_minor_radius(plasma))
    return np.asarray(winding.dofs)


def active_surface_dofs(surface):
    nmodes = surface.xm.size
    modes = jnp.where((surface.xm <= ACTIVE_MPOL)
                      & (jnp.abs(surface.xn / surface.nfp) <= ACTIVE_NTOR))[0]
    return jnp.concatenate((modes[modes != 0],
                            modes[modes != 0] + nmodes))


def geometry_control(plasma, surface, reference):
    geometry = geometry_metrics(plasma, surface)
    return (VOLUME_WEIGHT * (geometry[0] / reference[0] - 1) ** 2
            + SPECTRAL_WEIGHT * geometry[1] / reference[1]
            + DISTANCE_WEIGHT * jnp.maximum(
                1 - geometry[2] / (0.9 * reference[2]), 0) ** 2
            + 100 * geometry[5]
            + SELF_INTERSECTION_WEIGHT * geometry[6])


def write_nescin(surface, filename):
    """Write the stellarator-symmetric winding surface used by old REGCOIL."""
    with open(filename, "w") as stream:
        stream.write("------ Current Surface\n\n")
        stream.write(f"{surface.xm.size}\n\n\n")
        for m, n, rc, zs in zip(surface.xm, surface.xn,
                                surface.rc, surface.zs):
            stream.write(f"{int(m)} {-int(n) // surface.nfp} "
                         f"{float(rc):.17e} {float(zs):.17e} 0 0\n")


def regcoil_adjoint_value_gradient(wout, dofs, binary, bnorm=None,
                                   target="max_K_lse", target_value=KMAX):
    """Return fixed-Kmax REGCOIL fB and its adjoint surface gradient."""
    surface = make_surface(wout, 16, 16, dofs)
    with tempfile.TemporaryDirectory(prefix="regcoil-adjoint-") as directory:
        nescin = os.path.join(directory, "nescin.surface")
        input_file = os.path.join(directory, "regcoil_in.surface")
        write_nescin(surface, nescin)
        load_bnorm = bnorm is not None
        with open(input_file, "w") as stream:
            stream.write(f"""&regcoil_nml
general_option=5
symmetry_option=1
nlambda=40
target_option='{target}'
target_value={target_value:.17e}
target_option_p={2 if target == 'lp_norm_K' else 1e3}
ntheta_plasma=16
ntheta_coil=16
nzeta_plasma=16
nzeta_coil=16
mpol_potential={POTENTIAL_MPOL}
ntor_potential={POTENTIAL_NTOR}
geometry_option_plasma=2
wout_filename='{os.path.abspath(wout)}'
geometry_option_coil=3
nescin_filename='{nescin}'
load_bnorm={'.true.' if load_bnorm else '.false.'}
bnorm_filename='{os.path.abspath(bnorm) if load_bnorm else ''}'
sensitivity_option=3
fixed_norm_sensitivity_option=.true.
nmax_sensitivity={ACTIVE_NTOR}
mmax_sensitivity={ACTIVE_MPOL}
/
""")
        run = subprocess.run([binary, os.path.basename(input_file)],
                             cwd=directory, capture_output=True, text=True,
                             check=True)
        output_file = os.path.join(directory, "regcoil_out.surface.nc")
        if not os.path.exists(output_file):
            raise RuntimeError(run.stdout.strip() or "REGCOIL produced no output")
        with Dataset(output_file) as data:
            if int(data.variables["exit_code"][:]) != 0:
                raise RuntimeError("REGCOIL could not reach the requested Kmax")
            value = float(data.variables["chi2_B"][-1])
            derivative = np.asarray(data.variables["dchi2Bdomega"][-1])
            diagnostics = {
                "max_K": float(data.variables["max_K"][-1]),
                "rms_K": float(jnp.sqrt(
                    data.variables["chi2_K"][-1]
                    / data.variables["area_coil"][()])),
            }
            xm = np.asarray(data.variables["xm_sensitivity"][:], int)
            xn = np.asarray(data.variables["xn_sensitivity"][:], int)

    nmodes = surface.xm.size
    gradient = np.zeros(2 * nmodes)
    for mode, (m, n) in enumerate(zip(np.asarray(surface.xm, int),
                                      -np.asarray(surface.xn, int) // surface.nfp)):
        sign = 1
        if m == 0 and n < 0:
            n, sign = -n, -1
        indices = np.where((xm == m) & (xn == n))[0]
        if len(indices) == 2:
            gradient[mode] = derivative[indices[0]]
            gradient[mode + nmodes] = sign * derivative[indices[1]]
    scaling = np.tile(np.asarray(surface.scaling), 2)
    return value, gradient / scaling, diagnostics


def optimize_regcoil_surface(wout, binary, bnorm=None):
    import regcoil
    plasma = make_surface(wout, OPT_NTHETA, OPT_NPHI_PER_PERIOD)
    winding = make_surface(wout, OPT_NTHETA, OPT_NPHI_PER_PERIOD,
                           baseline_surface(wout))
    initial_dofs = winding.dofs
    active = active_surface_dofs(winding)
    x0 = np.asarray(initial_dofs[active])
    reference_geometry = geometry_metrics(plasma, winding)
    reference_surface = make_surface(wout, 16, 16, initial_dofs)
    reference_plasma = regcoil.PlasmaSurface.from_wout(
        wout, ntheta=16, nzeta=16)
    if bnorm is not None:
        reference_plasma.set_bnormal_from_bnorm_file(bnorm)
    reference_coil = regcoil.CoilSurface(
        np.asarray(reference_surface.xm, int),
        np.asarray(reference_surface.xn, int),
        np.asarray(reference_surface.rc), np.asarray(reference_surface.zs),
        nfp=reference_surface.nfp, ntheta=16, nzeta=16)
    reference_solution = regcoil.Regcoil(
        reference_plasma, reference_coil,
        mpol_potential=POTENTIAL_MPOL,
        ntor_potential=POTENTIAL_NTOR).solve_for_target("max_K", KMAX)
    target_current_norm = np.sqrt(
        reference_solution.f_K / reference_coil.area)
    initial_f_b, _, _ = regcoil_adjoint_value_gradient(
        wout, initial_dofs, binary, bnorm, target="lp_norm_K",
        target_value=target_current_norm)

    def geometry(active_dofs):
        winding.dofs = initial_dofs.at[active].set(active_dofs)
        return geometry_control(plasma, winding, reference_geometry)

    geometry_value_gradient = jax.jit(jax.value_and_grad(geometry))

    def fun(x):
        dofs = initial_dofs.at[active].set(jnp.asarray(x))
        try:
            f_b, f_b_gradient, _ = regcoil_adjoint_value_gradient(
                wout, dofs, binary, bnorm, target="lp_norm_K",
                target_value=target_current_norm)
        except RuntimeError:
            displacement = np.asarray(x) - x0
            return 1e3 + np.vdot(displacement, displacement), 2 * displacement
        g_value, g_gradient = geometry_value_gradient(jnp.asarray(x))
        return (f_b / initial_f_b + float(g_value),
                f_b_gradient[np.asarray(active)] / initial_f_b
                + np.asarray(g_gradient))

    bounds = [(value - 0.01, value + 0.01) for value in x0]
    start = time.perf_counter()
    result = minimize(fun, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": ADJOINT_MAXITER, "maxls": 100})
    elapsed = time.perf_counter() - start
    return (np.asarray(initial_dofs.at[active].set(jnp.asarray(result.x))),
            elapsed, int(result.nit), bool(result.success))


def quadcoil_adjoint_value_gradient(wout, dofs, bnormal=None,
                                    current_norm=None):
    """Return fixed-Kmax QUADCOIL fB and its KKT-adjoint surface gradient."""
    from quadcoil import quadcoil
    resolution = 16
    surface = make_surface(wout, resolution, resolution, dofs)
    plasma = make_surface(wout, resolution, resolution)
    qphi = jnp.linspace(0, 1 / surface.nfp, resolution, endpoint=False)
    qtheta = jnp.linspace(0, 1, resolution, endpoint=False)
    full_phi = jnp.linspace(0, 1, resolution * surface.nfp, endpoint=False)
    constraint_name = "f_max_K2" if current_norm is None else "f_K"
    constraint_value = KMAX ** 2 if current_norm is None else current_norm
    output, _, _, result = quadcoil(
        nfp=surface.nfp, stellsym=True,
        mpol=POTENTIAL_MPOL, ntor=POTENTIAL_NTOR,
        plasma_dofs=to_quadcoil_dofs(plasma),
        plasma_mpol=plasma.mpol, plasma_ntor=plasma.ntor,
        net_poloidal_current_amperes=net_poloidal_current(wout),
        net_toroidal_current_amperes=0.0,
        plasma_quadpoints_phi=qphi, plasma_quadpoints_theta=qtheta,
        Bnormal_plasma=bnormal,
        winding_dofs=to_quadcoil_dofs(surface),
        winding_mpol=surface.mpol, winding_ntor=surface.ntor,
        winding_quadpoints_phi=full_phi, winding_quadpoints_theta=qtheta,
        quadpoints_phi=qphi, quadpoints_theta=qtheta,
        objective_name="f_B", constraint_name=(constraint_name,),
        constraint_type=("<=",), constraint_unit=(constraint_value,),
        constraint_value=jnp.asarray((constraint_value,)),
        metric_name=("f_B", "f_K"), value_only=False, smoothing="approx",
        smoothing_params={"lse_epsilon": 1e-3}, precond="svd",
        solver="slsqp", solver_options={"atol": 1e-8, "rtol": 1e-8},
        convex=True, maxiter=1000, verbose=0)
    if not bool(result["converged"]):
        print("QUADCOIL inner solve did not converge: "
              f"iterations={int(result['niter'])}, "
              f"max constraint={float(jnp.max(result['fin_g'])):.3e}")
    value = float(output["f_B"]["value"])
    qgradient = np.asarray(
        output["f_B"]["grad"]["df_dwinding_dofs"])
    nmodes = surface.xm.size
    gradient = np.concatenate((qgradient[:nmodes], [0], qgradient[nmodes:]))
    scaling = np.tile(np.asarray(surface.scaling), 2)
    return value, gradient / scaling, {
        "f_K": float(output["f_K"]["value"]),
        "converged": bool(result["converged"]),
        "max_constraint": float(jnp.max(result["fin_g"])),
    }


def reference_quadcoil_f_k(wout, dofs, bnormal_source=None):
    import regcoil
    surface = make_surface(wout, 16, 16, dofs)
    plasma = regcoil.PlasmaSurface.from_wout(wout, ntheta=16, nzeta=16)
    if bnormal_source is not None:
        if str(bnormal_source).endswith(".nc"):
            plasma.set_bnormal_from_virtual_casing(bnormal_source)
        else:
            plasma.set_bnormal_from_bnorm_file(bnormal_source)
    coil = regcoil.CoilSurface(
        np.asarray(surface.xm, int), np.asarray(surface.xn, int),
        np.asarray(surface.rc), np.asarray(surface.zs), nfp=surface.nfp,
        ntheta=16, nzeta=16)
    solution = regcoil.Regcoil(
        plasma, coil, mpol_potential=POTENTIAL_MPOL,
        ntor_potential=POTENTIAL_NTOR).solve_for_target("max_K", KMAX)
    return 0.5 * solution.f_K


def optimize_quadcoil_surface(wout, bnormal=None, current_norm=None):
    plasma = make_surface(wout, OPT_NTHETA, OPT_NPHI_PER_PERIOD)
    winding = make_surface(wout, OPT_NTHETA, OPT_NPHI_PER_PERIOD,
                           baseline_surface(wout))
    initial_dofs = winding.dofs
    active = active_surface_dofs(winding)
    x0 = np.asarray(initial_dofs[active])
    reference_geometry = geometry_metrics(plasma, winding)
    if current_norm is None:
        _, _, initial_diagnostics = quadcoil_adjoint_value_gradient(
            wout, initial_dofs, bnormal)
        current_norm = initial_diagnostics["f_K"]
    initial_f_b, _, _ = quadcoil_adjoint_value_gradient(
        wout, initial_dofs, bnormal, current_norm=current_norm)

    def geometry(active_dofs):
        winding.dofs = initial_dofs.at[active].set(active_dofs)
        return geometry_control(plasma, winding, reference_geometry)

    geometry_value_gradient = jax.jit(jax.value_and_grad(geometry))

    def fun(x):
        dofs = initial_dofs.at[active].set(jnp.asarray(x))
        f_b, f_b_gradient, diagnostics = quadcoil_adjoint_value_gradient(
            wout, dofs, bnormal, current_norm=current_norm)
        if diagnostics["max_constraint"] > 5e-3:
            raise RuntimeError("QUADCOIL inner solve must converge for its adjoint")
        g_value, g_gradient = geometry_value_gradient(jnp.asarray(x))
        return (f_b / initial_f_b + float(g_value),
                f_b_gradient[np.asarray(active)] / initial_f_b
                + np.asarray(g_gradient))

    bounds = [(value - COEFFICIENT_STEP_BOUND,
               value + COEFFICIENT_STEP_BOUND) for value in x0]
    start = time.perf_counter()
    result = minimize(fun, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": ADJOINT_MAXITER})
    elapsed = time.perf_counter() - start
    return (np.asarray(initial_dofs.at[active].set(jnp.asarray(result.x))),
            elapsed, int(result.nit), bool(result.success))


def to_quadcoil_dofs(surface):
    return jnp.concatenate((surface.rc, surface.zs[1:]))


def solve_with_method(method, wout, surface_dofs, bnormal_source=None):
    surface = make_surface(
        wout, CURRENT_NTHETA, CURRENT_NPHI_PER_PERIOD, surface_dofs)
    plasma = make_surface(wout, CURRENT_NTHETA, CURRENT_NPHI_PER_PERIOD)
    net_current = net_poloidal_current(wout)
    plasma_bnormal = load_plasma_bnormal(
        wout, bnormal_source, CURRENT_NTHETA, CURRENT_NPHI_PER_PERIOD)
    if method == "ESSOS":
        return np.asarray(solve_for_max_current(
            build_operators(plasma, surface, net_current, plasma_bnormal),
            KMAX)[0]), {"converged": True}
    if method == "REGCOIL":
        import regcoil
        plasma_regcoil = regcoil.PlasmaSurface.from_wout(
            wout, ntheta=CURRENT_NTHETA, nzeta=CURRENT_NPHI_PER_PERIOD)
        if bnormal_source is not None:
            if str(bnormal_source).endswith(".nc"):
                plasma_regcoil.set_bnormal_from_virtual_casing(bnormal_source)
            else:
                plasma_regcoil.set_bnormal_from_bnorm_file(bnormal_source)
        coil = regcoil.CoilSurface(
            np.asarray(surface.xm, int), np.asarray(surface.xn, int),
            np.asarray(surface.rc), np.asarray(surface.zs), nfp=surface.nfp,
            ntheta=CURRENT_NTHETA, nzeta=CURRENT_NPHI_PER_PERIOD)
        problem = regcoil.Regcoil(
            plasma_regcoil, coil, mpol_potential=POTENTIAL_MPOL,
            ntor_potential=POTENTIAL_NTOR)
        try:
            solution = problem.solve_for_target("max_K", KMAX)
        except ValueError as error:
            if "target max_K" not in str(error):
                raise
            solution = problem.solve(lam=0)
            if solution.max_K > KMAX:
                raise
        return solution.solution, {"converged": True}
    from quadcoil import quadcoil
    qphi = jnp.linspace(0, 1 / surface.nfp, CURRENT_NPHI_PER_PERIOD,
                        endpoint=False)
    qtheta = jnp.linspace(0, 1, CURRENT_NTHETA, endpoint=False)
    full_phi = jnp.linspace(0, 1, CURRENT_NPHI_PER_PERIOD * surface.nfp,
                            endpoint=False)
    _, _, dofs, result = quadcoil(
        nfp=surface.nfp, stellsym=True,
        mpol=POTENTIAL_MPOL, ntor=POTENTIAL_NTOR,
        plasma_dofs=to_quadcoil_dofs(plasma),
        plasma_mpol=plasma.mpol, plasma_ntor=plasma.ntor,
        net_poloidal_current_amperes=net_current,
        net_toroidal_current_amperes=0.0,
        plasma_quadpoints_phi=qphi, plasma_quadpoints_theta=qtheta,
        Bnormal_plasma=plasma_bnormal,
        winding_dofs=to_quadcoil_dofs(surface),
        winding_mpol=surface.mpol, winding_ntor=surface.ntor,
        winding_quadpoints_phi=full_phi, winding_quadpoints_theta=qtheta,
        quadpoints_phi=qphi, quadpoints_theta=qtheta,
        objective_name="f_B", constraint_name=("f_max_K2",),
        constraint_type=("<=",), constraint_unit=(KMAX ** 2,),
        constraint_value=jnp.asarray((KMAX ** 2,)),
        metric_name=(), value_only=True, smoothing="approx",
        smoothing_params={"lse_epsilon": 1e-3}, precond="svd",
        solver="slsqp", solver_options={"atol": 1e-8, "rtol": 1e-8},
        convex=True, maxiter=500, verbose=0)
    if not bool(result["converged"]):
        print("QUADCOIL reached its iteration limit; the shared evaluator "
              "will report its achieved Kmax.")
    return np.asarray(dofs["phi"]), {
        "converged": bool(result["converged"]),
        "iterations": int(result["niter"]),
        "max_constraint": float(jnp.max(result["fin_g"])),
    }


class MemoryMonitor:
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.running = True

        def sample():
            while self.running:
                self.peak = max(self.peak, self.process.memory_info().rss)
                time.sleep(0.005)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.running = False
        self.thread.join()
        self.peak_mb = max(0, self.peak - self.baseline) / 1024 ** 2


def evaluate_surface(surface, theta, phi):
    angle = (np.asarray(surface.xm)[:, None] * np.ravel(theta)
             - np.asarray(surface.xn)[:, None] * np.ravel(phi))
    radius = np.asarray(surface.rc) @ np.cos(angle)
    z = np.asarray(surface.zs) @ np.sin(angle)
    xyz = np.column_stack((radius * np.cos(np.ravel(phi)),
                           radius * np.sin(np.ravel(phi)), z))
    return xyz.reshape(theta.shape + (3,))


def potential_grid(coefficients, nfp, net_current, resolution):
    theta, phi = np.meshgrid(
        np.linspace(0, 2 * np.pi, resolution, endpoint=False),
        np.linspace(0, 2 * np.pi / nfp, resolution, endpoint=False),
        indexing="ij")
    xm, xn = potential_modes(nfp)
    angle = np.asarray(xm)[:, None] * theta.ravel() \
        - np.asarray(xn)[:, None] * phi.ravel()
    potential = (coefficients @ np.sin(angle)).reshape(theta.shape)
    potential += net_current * phi / (2 * np.pi)
    return theta, phi, potential / net_current * nfp


def cut_coils(surface, potential):
    theta0 = np.linspace(0, 2 * np.pi, potential.shape[0], endpoint=False)
    phi = np.linspace(0, 2 * np.pi / surface.nfp,
                      potential.shape[1], endpoint=False)
    levels = (np.arange(2 * COILS_PER_HALF_PERIOD) + 0.5) / (
        2 * COILS_PER_HALF_PERIOD)
    period = 2 * np.pi / surface.nfp
    phi3 = np.concatenate((phi - period, phi, phi + period))
    for shift in range(potential.shape[0]):
        theta = theta0[(-shift) % len(theta0)] + theta0
        shifted = np.roll(potential, shift, axis=0)
        potential3 = np.concatenate((shifted - 1, shifted, shifted + 1), axis=1)
        generator = contour_generator(x=phi3, y=theta, z=potential3,
                                      line_type=LineType.Separate)
        contours = [generator.lines(float(level)) for level in levels]
        if all(len(contour) == 1 for contour in contours):
            curves = []
            for contour in contours:
                for field_period in range(surface.nfp):
                    phi_curve = contour[0][:, 0] + field_period * period
                    theta_curve = contour[0][:, 1]
                    curve = evaluate_surface(surface, theta_curve, phi_curve)
                    curves.append(np.vstack((curve, curve[0])))
            return curves, levels
    raise RuntimeError("Could not find one closed contour per coil level")


def filament_field(points, curves, current):
    field = np.zeros_like(points)
    for start in range(0, len(points), 256):
        evaluation_points = points[start:start + 256]
        for curve in curves:
            r1 = evaluation_points[:, None] - curve[None, :-1]
            r2 = evaluation_points[:, None] - curve[None, 1:]
            norm1 = np.linalg.norm(r1, axis=2)
            norm2 = np.linalg.norm(r2, axis=2)
            denominator = norm1 * norm2 + np.sum(r1 * r2, axis=2)
            field[start:start + 256] += MU0 * current / (4 * np.pi) * np.sum(
                np.cross(r1, r2) / denominator[:, :, None]
                * (1 / norm1 + 1 / norm2)[:, :, None], axis=1)
    return field


def vmec_mod_b(wout, theta, phi):
    with Dataset(wout) as dataset:
        bmnc = 1.5 * dataset.variables["bmnc"][-1] - 0.5 * dataset.variables["bmnc"][-2]
        bmns = (1.5 * dataset.variables["bmns"][-1]
                - 0.5 * dataset.variables["bmns"][-2]
                if "bmns" in dataset.variables else np.zeros_like(bmnc))
        xm = dataset.variables["xm_nyq"][:]
        xn = dataset.variables["xn_nyq"][:]
    angle = xm[:, None] * np.ravel(theta) - xn[:, None] * np.ravel(phi)
    return np.sum(bmnc[:, None] * np.cos(angle)
                  + bmns[:, None] * np.sin(angle), axis=0)


def worker(spec_file):
    with open(spec_file) as stream:
        spec = json.load(stream)
    surface_info = np.load(spec["surface_file"])
    dofs = surface_info["dofs"]
    with MemoryMonitor() as memory:
        start = time.perf_counter()
        coefficients, solve_info = solve_with_method(
            spec["method"], spec["wout"], dofs, spec.get("bnormal"))
        cold_runtime = time.perf_counter() - start
        if solve_info["converged"]:
            start = time.perf_counter()
            coefficients, solve_info = solve_with_method(
                spec["method"], spec["wout"], dofs, spec.get("bnormal"))
            steady_runtime = time.perf_counter() - start
        else:
            steady_runtime = np.nan

    surface = make_surface(
        spec["wout"], CURRENT_NTHETA, CURRENT_NPHI_PER_PERIOD, dofs)
    plasma = make_surface(spec["wout"], CURRENT_NTHETA, CURRENT_NPHI_PER_PERIOD)
    operators = build_operators(
        plasma, surface, net_poloidal_current(spec["wout"]),
        load_plasma_bnormal(spec["wout"], spec.get("bnormal"),
                            CURRENT_NTHETA, CURRENT_NPHI_PER_PERIOD))
    weighted_bnormal = np.asarray(operators[0] @ coefficients + operators[1])
    sheet_f_b = float(np.vdot(weighted_bnormal, weighted_bnormal))
    sheet_bnormal = weighted_bnormal / np.asarray(operators[6])
    nphi_period = plasma.nphi // plasma.nfp
    mod_b = vmec_mod_b(
        spec["wout"], plasma.theta2d[:nphi_period], plasma.phi2d[:nphi_period])
    sheet_max_ratio = float(np.max(np.abs(sheet_bnormal) / np.abs(mod_b)))
    current = np.asarray(
        operators[4] @ coefficients + operators[5]).reshape(-1, 3)
    achieved_kmax = float(np.max(np.linalg.norm(current, axis=1)))

    cut_surface = make_surface(
        spec["wout"], CUT_RESOLUTION, CUT_RESOLUTION, dofs)
    theta, phi, potential = potential_grid(
        coefficients, cut_surface.nfp, net_poloidal_current(spec["wout"]),
        CUT_RESOLUTION)
    curves, levels = cut_coils(cut_surface, potential)
    plasma_eval = make_surface(
        spec["wout"], FIELD_RESOLUTION, FIELD_RESOLUTION)
    nphi_eval = plasma_eval.nphi // plasma_eval.nfp
    points = np.asarray(plasma_eval.gamma[:nphi_eval]).reshape(-1, 3)
    normals = np.asarray(plasma_eval.unitnormal[:nphi_eval]).reshape(-1, 3)
    field = filament_field(
        points, curves, net_poloidal_current(spec["wout"]) / len(curves))
    bnormal = np.sum(field * normals, axis=1)
    plasma_bnormal = load_plasma_bnormal(
        spec["wout"], spec.get("bnormal"),
        FIELD_RESOLUTION, FIELD_RESOLUTION)
    if plasma_bnormal is not None:
        bnormal = bnormal + plasma_bnormal.reshape(-1)
    weights = np.asarray(quadrature_weights(
        plasma_eval, one_period=True, full_integral=True))
    mod_b = vmec_mod_b(
        spec["wout"], plasma_eval.theta2d[:nphi_eval],
        plasma_eval.phi2d[:nphi_eval])
    geometry = geometry_metrics(plasma, surface)
    metrics = dict(method=spec["method"], configuration=spec["configuration"],
                   surface_method=spec["surface_method"],
                   surface_runtime_s=float(surface_info["runtime"]),
                   surface_iterations=int(surface_info["iterations"]),
                   surface_converged=bool(surface_info["success"]),
                   solver_converged=solve_info["converged"],
                   cold_runtime_s=cold_runtime,
                   steady_runtime_s=steady_runtime, peak_memory_mb=memory.peak_mb,
                   sheet_f_B_T2_m2=sheet_f_b,
                   sheet_max_abs_Bn_over_B=sheet_max_ratio,
                   achieved_Kmax_A_per_m=achieved_kmax,
                   filament_f_B_T2_m2=float(np.sum(weights * bnormal ** 2)),
                   filament_max_abs_Bn_over_B=float(
                       np.max(np.abs(bnormal) / np.abs(mod_b))),
                   minimum_distance_m=float(geometry[2]),
                   self_intersection_radius_m=float(geometry[3]),
                   minimum_jacobian=float(geometry[4]))
    metrics.update({f"solver_{key}": value for key, value in solve_info.items()
                    if key != "converged"})
    np.savez(spec["result_file"], metrics=json.dumps(metrics), potential=potential,
             theta=theta[:, 0], phi=phi[0], levels=levels,
             surface_dofs=dofs, coefficients=coefficients,
             **{f"coil_{index}": curve for index, curve in enumerate(curves)})
    print(json.dumps(metrics, indent=2))


def load_result(path):
    result = np.load(path)
    curves = [result[key] for key in sorted(
        (key for key in result.files if key.startswith("coil_")),
        key=lambda key: int(key.split("_")[1]))]
    return result, curves, json.loads(str(result["metrics"]))


def sheet_resolution_study(results, cases):
    """Re-evaluate fixed solutions without refitting on successively finer grids."""
    resolutions = (48, 56, 64)
    rows = []
    selections = [
        (case, surface, method)
        for case in cases for surface in SURFACE_METHODS
        for method in (("REGCOIL",) if surface not in (
            "normal offset", "ESSOS Pareto") else METHODS)]
    for case, surface_method, method in selections:
        result, _, metrics = results[(case, surface_method, method)]
        if not metrics["solver_converged"] or "coefficients" not in result.files:
            continue
        coefficients = result["coefficients"]
        for resolution in resolutions:
            surface = make_surface(cases[case]["wout"], resolution, resolution,
                                   result["surface_dofs"])
            plasma = make_surface(cases[case]["wout"], resolution, resolution)
            operators = build_operators(
                plasma, surface, net_poloidal_current(cases[case]["wout"]),
                load_plasma_bnormal(cases[case]["wout"],
                                    cases[case].get("bnormal"),
                                    resolution, resolution))
            weighted_bnormal = np.asarray(
                operators[0] @ coefficients + operators[1])
            current = np.asarray(
                operators[4] @ coefficients + operators[5]).reshape(-1, 3)
            rows.append(dict(configuration=case, surface_method=surface_method,
                             method=method, resolution=resolution,
                             sheet_f_B_T2_m2=float(np.vdot(
                                 weighted_bnormal, weighted_bnormal)),
                             achieved_Kmax_A_per_m=float(np.max(
                                 np.linalg.norm(current, axis=1)))))
            jax.clear_caches()

    with open(os.path.join(OUTPUT, "sheet_resolution_convergence.csv"), "w",
              newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(len(cases), 2, figsize=(12, 7),
                                constrained_layout=True, sharex=True)
    colors = dict(ESSOS="#0072B2", REGCOIL="#D55E00", QUADCOIL="#009E73")
    surface_colors = dict(zip(SURFACE_METHODS,
                              ("#777777", "#56B4E9", "#0072B2", "#D55E00")))
    for row_index, case in enumerate(cases):
        axis = axes[row_index, 0]
        for surface_method in SURFACE_METHODS:
            subset = [row for row in rows if row["configuration"] == case
                      and row["surface_method"] == surface_method
                      and row["method"] == "REGCOIL"]
            axis.plot([row["resolution"] for row in subset],
                      [row["sheet_f_B_T2_m2"] for row in subset], "o-",
                      color=surface_colors[surface_method], label=surface_method)
        axis.set_title(f"{case}: winding surfaces")
        axis.set_ylabel(r"Sheet $f_B$ [$T^2m^2$]")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)

        axis = axes[row_index, 1]
        for surface_method, linestyle in (("normal offset", "-"),
                                           ("ESSOS Pareto", "--")):
            for method in METHODS:
                subset = [row for row in rows if row["configuration"] == case
                          and row["surface_method"] == surface_method
                          and row["method"] == method]
                if subset:
                    axis.plot([row["resolution"] for row in subset],
                              [row["sheet_f_B_T2_m2"] for row in subset], "o",
                              linestyle=linestyle, color=colors[method],
                              label=f"{method}, {surface_method}")
        axis.set_title(f"{case}: current solvers")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("points per angle per field period")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    figure.savefig(os.path.join(OUTPUT, "sheet_resolution_convergence.png"),
                   dpi=180)
    plt.close(figure)


def validate_surfaces(surface_data, cases, resolution=96):
    """Apply one resolved REGCOIL current solve to every candidate surface."""
    import regcoil
    rows = []
    validated = {}
    for case, data in cases.items():
        for surface_method in SURFACE_METHODS:
            surface_info = np.load(surface_data[(case, surface_method)])
            dofs = surface_info["dofs"]
            surface = make_surface(data["wout"], resolution, resolution, dofs)
            plasma = regcoil.PlasmaSurface.from_wout(
                data["wout"], ntheta=resolution, nzeta=resolution)
            if data.get("bnormal"):
                plasma.set_bnormal_from_virtual_casing(data["bnormal"])
            coil = regcoil.CoilSurface(
                np.asarray(surface.xm, int), np.asarray(surface.xn, int),
                np.asarray(surface.rc), np.asarray(surface.zs), nfp=surface.nfp,
                ntheta=resolution, nzeta=resolution)
            problem = regcoil.Regcoil(
                plasma, coil, mpol_potential=POTENTIAL_MPOL,
                ntor_potential=POTENTIAL_NTOR)
            start = time.perf_counter()
            feasible = True
            try:
                solution = problem.solve_for_target("max_K", KMAX)
            except ValueError:
                solution = problem.solve(lam=0)
                if solution.max_K > KMAX:
                    feasible = False
                    solution = problem.solve(lam=1e20)
            if feasible:
                theta, phi, potential = potential_grid(
                    solution.solution, surface.nfp,
                    net_poloidal_current(data["wout"]), CUT_RESOLUTION)
                curves, levels = cut_coils(surface, potential)
                plasma_eval = make_surface(
                    data["wout"], FIELD_RESOLUTION, FIELD_RESOLUTION)
                nphi_eval = plasma_eval.nphi // plasma_eval.nfp
                points = np.asarray(
                    plasma_eval.gamma[:nphi_eval]).reshape(-1, 3)
                normals = np.asarray(
                    plasma_eval.unitnormal[:nphi_eval]).reshape(-1, 3)
                field = filament_field(
                    points, curves,
                    net_poloidal_current(data["wout"]) / len(curves))
                filament_bnormal = np.sum(field * normals, axis=1)
                plasma_bnormal = load_plasma_bnormal(
                    data["wout"], data.get("bnormal"),
                    FIELD_RESOLUTION, FIELD_RESOLUTION)
                if plasma_bnormal is not None:
                    filament_bnormal += plasma_bnormal.reshape(-1)
                weights = np.asarray(quadrature_weights(
                    plasma_eval, one_period=True, full_integral=True))
                mod_b = vmec_mod_b(
                    data["wout"], plasma_eval.theta2d[:nphi_eval],
                    plasma_eval.phi2d[:nphi_eval])
                filament_f_b = float(np.sum(weights * filament_bnormal ** 2))
                filament_max_ratio = float(np.max(
                    np.abs(filament_bnormal) / np.abs(mod_b)))
                validated[(case, surface_method)] = (
                    surface, theta, phi, potential, levels, curves)
            else:
                filament_f_b = filament_max_ratio = np.nan
            rows.append(dict(
                configuration=case, surface_method=surface_method,
                resolution=resolution, feasible_at_Kmax=feasible,
                sheet_f_B_T2_m2=solution.f_B if feasible else np.nan,
                sheet_max_abs_Bn_over_B=(solution.max_Bnormal_over_B
                                         if feasible else np.nan),
                filament_f_B_T2_m2=filament_f_b,
                filament_max_abs_Bn_over_B=filament_max_ratio,
                achieved_or_minimum_Kmax_A_per_m=solution.max_K,
                surface_runtime_s=float(surface_info["runtime"]),
                validation_runtime_s=time.perf_counter() - start))

    with open(os.path.join(OUTPUT, "surface_validation_96.csv"), "w",
              newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    colors = dict(zip(SURFACE_METHODS,
                      ("#777777", "#56B4E9", "#0072B2", "#D55E00")))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = (("sheet_f_B_T2_m2", r"Resolved sheet $f_B$ [$T^2m^2$]", True),
               ("sheet_max_abs_Bn_over_B", r"Resolved max $|B_n|/B$", True),
               ("filament_f_B_T2_m2", r"Filament $f_B$ [$T^2m^2$]", True),
               ("filament_max_abs_Bn_over_B", r"Filament max $|B_n|/B$", True),
               ("achieved_or_minimum_Kmax_A_per_m",
                r"Achieved/minimum $K_{max}$ [MA/m]", False),
               ("surface_runtime_s", "Surface optimization runtime [s]", True))
    x = np.arange(len(cases)); width = 0.18
    for axis, (metric, title, logarithmic) in zip(axes.flat, metrics):
        for index, surface_method in enumerate(SURFACE_METHODS):
            entries = [next(row for row in rows
                            if row["configuration"] == case
                            and row["surface_method"] == surface_method)
                       for case in cases]
            values = np.asarray([entry[metric] for entry in entries], float)
            if "Kmax" in metric:
                values /= 1e6
            positions = x + (index - 1.5) * width
            axis.bar(positions, values, width, label=surface_method,
                     color=colors[surface_method])
            for position, entry in zip(positions, entries):
                if not entry["feasible_at_Kmax"]:
                    axis.text(position, 0.03, "infeasible", rotation=90,
                              color="#AA0000", ha="center", va="bottom",
                              transform=axis.get_xaxis_transform(), fontsize=7)
        axis.set_xticks(x, cases)
        axis.set_title(title)
        if logarithmic:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.savefig(os.path.join(OUTPUT, "surface_validation_96.png"), dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(cases), len(SURFACE_METHODS),
                                figsize=(15, 7), constrained_layout=True,
                                sharex=True, sharey=True)
    values = [entry[3] for entry in validated.values()]
    lower, upper = min(map(np.min, values)), max(map(np.max, values))
    for row_index, case in enumerate(cases):
        for column, surface_method in enumerate(SURFACE_METHODS):
            _, theta, phi, potential, levels, _ = validated[(case, surface_method)]
            axis = axes[row_index, column]
            image = axis.contourf(phi * cases[case]["nfp"], theta, potential,
                                  40, vmin=lower, vmax=upper, cmap="viridis")
            axis.contour(phi * cases[case]["nfp"], theta, potential,
                         levels=levels, colors="black", linewidths=0.4)
            axis.set_title(f"{case}\n{surface_method}")
            axis.set_xlabel(r"$N_{fp}\phi$")
            axis.set_ylabel(r"$\theta$")
        figure.colorbar(image, ax=axes[row_index],
                        label=r"$N_{fp}\Phi/I_{pol}$")
    figure.savefig(os.path.join(OUTPUT,
                                "validated_surface_current_potential.png"),
                   dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    for row_index, case in enumerate(cases):
        for column, surface_method in enumerate(SURFACE_METHODS):
            surface, _, _, _, _, curves = validated[(case, surface_method)]
            axis = figure.add_subplot(
                len(cases), len(SURFACE_METHODS),
                row_index * len(SURFACE_METHODS) + column + 1,
                projection="3d")
            xyz = np.asarray(surface.gamma)
            axis.plot_surface(*np.moveaxis(xyz, 2, 0), color="#BBBBBB",
                              alpha=0.18, shade=False, linewidth=0)
            for curve in curves:
                axis.plot(*curve.T, color=colors[surface_method], linewidth=0.55)
            axis.set_box_aspect(np.ptp(xyz.reshape(-1, 3), axis=0))
            axis.set_axis_off()
            axis.view_init(elev=25, azim=35)
            axis.set_title(f"{case}\n{surface_method}")
    figure.savefig(os.path.join(OUTPUT, "validated_surface_and_coils.png"),
                   dpi=180)
    plt.close(figure)
    return rows


def plot_results(results, cases):
    surface_results = {(case, surface): results[(case, surface, "REGCOIL")]
                       for case in cases for surface in SURFACE_METHODS}
    colors = dict(ESSOS="#0072B2", REGCOIL="#D55E00", QUADCOIL="#009E73")
    surface_colors = dict(zip(SURFACE_METHODS,
                              ("#777777", "#56B4E9", "#0072B2", "#D55E00")))

    figure, axes = plt.subplots(len(cases), len(SURFACE_METHODS),
                                figsize=(15, 7), constrained_layout=True,
                                sharex=True, sharey=True)
    for row, case in enumerate(cases):
        values = [surface_results[(case, method)][0]["potential"]
                  for method in SURFACE_METHODS]
        lower, upper = min(map(np.min, values)), max(map(np.max, values))
        for column, surface_method in enumerate(SURFACE_METHODS):
            result = surface_results[(case, surface_method)][0]
            axis = axes[row, column]
            image = axis.contourf(result["phi"] * cases[case]["nfp"],
                                  result["theta"], result["potential"], 40,
                                  vmin=lower, vmax=upper, cmap="viridis")
            axis.contour(result["phi"] * cases[case]["nfp"], result["theta"],
                         result["potential"], levels=result["levels"],
                         colors="black", linewidths=0.4)
            axis.set_title(f"{case}\n{surface_method}")
            axis.set_xlabel(r"$N_{fp}\phi$")
            axis.set_ylabel(r"$\theta$")
        figure.colorbar(image, ax=axes[row], label=r"$N_{fp}\Phi/I_{pol}$")
    figure.savefig(os.path.join(OUTPUT, "surface_current_potential.png"), dpi=180)
    plt.close(figure)

    for filename, show_surface in (("surface_coils.png", False),
                                   ("surface_and_coils.png", True)):
        figure = plt.figure(figsize=(15, 8), constrained_layout=True)
        for row, case in enumerate(cases):
            for column, surface_method in enumerate(SURFACE_METHODS):
                result, curves, _ = surface_results[(case, surface_method)]
                axis = figure.add_subplot(len(cases), len(SURFACE_METHODS),
                    row * len(SURFACE_METHODS) + column + 1, projection="3d")
                if show_surface:
                    surface = make_surface(cases[case]["wout"],
                                           FIELD_RESOLUTION, FIELD_RESOLUTION,
                                           result["surface_dofs"])
                    xyz = np.asarray(surface.gamma)
                    axis.plot_surface(*np.moveaxis(xyz, 2, 0), color="#BBBBBB",
                                      alpha=0.18, shade=False, linewidth=0)
                    axis.set_box_aspect(np.ptp(xyz.reshape(-1, 3), axis=0))
                for curve in curves:
                    axis.plot(*curve.T, color=surface_colors[surface_method],
                              linewidth=0.55)
                axis.set_axis_off()
                axis.view_init(elev=25, azim=35)
                axis.set_title(f"{case}\n{surface_method}")
        figure.savefig(os.path.join(OUTPUT, filename), dpi=180)
        plt.close(figure)

    metrics = (("sheet_f_B_T2_m2", r"Sheet $f_B$ [$T^2m^2$]", True),
               ("filament_f_B_T2_m2", r"Filament $f_B$ [$T^2m^2$]", True),
               ("filament_max_abs_Bn_over_B", r"Filament max $|B_n|/B$", True),
               ("minimum_distance_m", "Minimum plasma distance [m]", False),
               ("surface_runtime_s", "Surface optimization runtime [s]", True),
               ("achieved_Kmax_A_per_m", r"Achieved $K_{max}$ [MA/m]", False))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = np.arange(len(cases)); width = 0.18
    for axis, (metric, title, logarithmic) in zip(axes.flat, metrics):
        for offset, surface_method in enumerate(SURFACE_METHODS):
            values = [surface_results[(case, surface_method)][2][metric]
                      for case in cases]
            if metric == "achieved_Kmax_A_per_m": values = np.asarray(values) / 1e6
            axis.bar(x + (offset - 1.5) * width, values, width,
                     label=surface_method, color=surface_colors[surface_method])
        axis.set_xticks(x, cases)
        axis.set_title(title)
        if logarithmic and np.all(np.asarray(values) > 0): axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.savefig(os.path.join(OUTPUT, "surface_metrics.png"), dpi=180)
    plt.close(figure)

    rows = [(case, surface) for case in cases
            for surface in ("normal offset", "ESSOS Pareto")]
    solver_metrics = (("steady_runtime_s", "Current solve runtime [s]", True),
                      ("peak_memory_mb", "Incremental peak RSS [MB]", True),
                      ("sheet_f_B_T2_m2", r"Sheet $f_B$ [$T^2m^2$]", True),
                      ("filament_f_B_T2_m2", r"Filament $f_B$ [$T^2m^2$]", True),
                      ("filament_max_abs_Bn_over_B", r"Filament max $|B_n|/B$", True),
                      ("achieved_Kmax_A_per_m", r"Achieved $K_{max}$ [MA/m]", False))
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    x = np.arange(len(rows)); width = 0.25
    for axis, (metric, title, logarithmic) in zip(axes.flat, solver_metrics):
        for offset, solver in enumerate(METHODS):
            entries = [results[(case, surface, solver)][2]
                       for case, surface in rows]
            valid = np.asarray([entry["solver_converged"] for entry in entries])
            values = np.asarray([entry[metric] for entry in entries], float)
            if metric == "achieved_Kmax_A_per_m": values = np.asarray(values) / 1e6
            values[~valid] = np.nan
            axis.bar(x + (offset - 1) * width, values, width,
                     label=solver, color=colors[solver])
            for position in x[~valid] + (offset - 1) * width:
                axis.text(position, 0.03, "failed", rotation=90, color="#AA0000",
                          ha="center", va="bottom",
                          transform=axis.get_xaxis_transform(), fontsize=7)
        axis.set_xticks(x, [f"{case}\n{surface}" for case, surface in rows],
                        rotation=12)
        axis.set_title(title)
        if logarithmic: axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(ncol=3, fontsize=9)
    figure.savefig(os.path.join(OUTPUT, "solver_metrics.png"), dpi=180)
    plt.close(figure)


def run_comparison():
    import regcoil
    os.makedirs(OUTPUT, exist_ok=True)
    resume = os.environ.get("RESUME_COMPARISON") == "1"
    cases = {
        "Landreman-Paul QA": {
            "wout": os.path.join(
                EXAMPLES, "input_files", "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")},
        "W7-X": {"wout": str(regcoil.examples("W7-X").wout),
                 "bnormal": str(regcoil.examples("W7-X").vcasing),
                 "bnorm": str(regcoil.examples("W7-X").bnorm)},
    }
    for case in cases.values():
        with Dataset(case["wout"]) as dataset:
            case["nfp"] = int(dataset.variables["nfp"][0])

    binary = os.environ.get("REGCOIL_ADJOINT")
    if not binary or not os.path.isfile(binary):
        raise RuntimeError("Set REGCOIL_ADJOINT to the legacy REGCOIL executable")
    surface_data = {}
    for case, data in cases.items():
        optimizers = {
            "normal offset": lambda: (baseline_surface(data["wout"]), 0, 0, True),
            "ESSOS entropy": lambda: optimize_surface(
                data["wout"], "entropy", data.get("bnormal")),
            "ESSOS Pareto": lambda: optimize_surface(
                data["wout"], "pareto", data.get("bnormal")),
            "REGCOIL adjoint": lambda: optimize_regcoil_surface(
                data["wout"], binary, data.get("bnorm")),
        }
        for surface_method, optimizer in optimizers.items():
            stem = f"surface_{case.replace(' ', '_')}_{surface_method.replace(' ', '_')}"
            surface_file = os.path.join(OUTPUT, stem + ".npz")
            if not (resume and os.path.exists(surface_file)):
                dofs, runtime, iterations, success = optimizer()
                np.savez(surface_file, dofs=dofs, runtime=runtime,
                         iterations=iterations, success=success)
            surface_data[(case, surface_method)] = surface_file
            jax.clear_caches()

    results = {}
    for case, data in cases.items():
        for surface_method in SURFACE_METHODS:
            solvers = METHODS if surface_method in (
                "normal offset", "ESSOS Pareto") else ("REGCOIL",)
            surface_info = np.load(surface_data[(case, surface_method)])
            for method in solvers:
                stem = (f"k10_{case.replace(' ', '_')}_"
                        f"{surface_method.replace(' ', '_')}_{method}")
                spec_file = os.path.join(OUTPUT, stem + ".json")
                result_file = os.path.join(OUTPUT, stem + ".npz")
                spec = dict(configuration=case, surface_method=surface_method,
                            method=method, wout=data["wout"],
                            bnormal=data.get("bnormal"),
                            surface_runtime_s=float(surface_info["runtime"]),
                            surface_iterations=int(surface_info["iterations"]),
                            surface_converged=bool(surface_info["success"]),
                            surface_file=surface_data[(case, surface_method)],
                            result_file=result_file)
                with open(spec_file, "w") as stream:
                    json.dump(spec, stream)
                if not (resume and os.path.exists(result_file)):
                    subprocess.run(
                        [sys.executable, __file__, "--worker", spec_file], check=True)
                results[(case, surface_method, method)] = load_result(result_file)

    validate_surfaces(surface_data, cases)
    with open(os.path.join(OUTPUT, "comparison_metrics.csv"), "w", newline="") as stream:
        rows = [results[key][2] for key in results]
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    sheet_resolution_study(results, cases)
    plot_results(results, cases)
    print(f"Saved comparison data and figures to {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    arguments = parser.parse_args()
    worker(arguments.worker) if arguments.worker else run_comparison()
