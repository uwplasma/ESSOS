"""Optimize a winding surface for a small, specified number of cut coils."""

import os
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from winding_surface_comparison import winding_surface_comparison as ws


COILS_PER_HALF_PERIOD = 4
RESOLUTION = 24
MAXITER = 100

vmec_input = os.path.join(
    os.path.dirname(__file__), "input_files",
    "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
plasma = ws.make_surface(vmec_input, RESOLUTION, RESOLUTION)
winding = ws.make_surface(
    vmec_input, RESOLUTION, RESOLUTION, ws.baseline_surface(vmec_input))
net_current = ws.net_poloidal_current(vmec_input)
initial_dofs = winding.dofs
active = ws.active_surface_dofs(winding)
reference_geometry = ws.geometry_metrics(plasma, winding)


def filtered_coefficients(operators, sigma_cutoff):
    """Current-normalized SVD filter, evaluated with a stable SPD solve."""
    field, fixed_field, current, fixed_current = operators[:4]
    gram = current.T @ current
    minimum_current = jnp.linalg.solve(gram, -current.T @ fixed_current)
    target = field @ minimum_current + fixed_field
    cholesky = jnp.linalg.cholesky(gram)
    normalized_field = jnp.linalg.solve(cholesky, field.T).T
    # If normalized_field = U diag(sigma) V.T, this solve applies
    # V diag(sigma / (sigma**2 + sigma_cutoff**2)) U.T.
    whitened = jnp.linalg.solve(
        normalized_field.T @ normalized_field
        + sigma_cutoff ** 2 * jnp.eye(normalized_field.shape[1]),
        -normalized_field.T @ target)
    return minimum_current + jnp.linalg.solve(cholesky.T, whitened)


initial_operators = ws.build_operators(plasma, winding, net_current)
initial_kmax_solution = ws.solve_for_max_current(
    initial_operators, ws.KMAX)[0]
initial_current = (initial_operators[2] @ initial_kmax_solution
                   + initial_operators[3])
current_budget = jnp.vdot(initial_current, initial_current)
field, fixed_field, current, fixed_current = initial_operators[:4]
minimum_current = jnp.linalg.solve(
    current.T @ current, -current.T @ fixed_current)
remaining_budget = jnp.maximum(
    current_budget
    - jnp.vdot(current @ minimum_current + fixed_current,
               current @ minimum_current + fixed_current),
    0.01 * current_budget)
sigma_cutoff = jnp.sqrt(
    jnp.vdot(field @ minimum_current + fixed_field,
             field @ minimum_current + fixed_field) / remaining_budget)


def finite_coil_error(surface, coefficients):
    """Leading phase-averaged field error from equally spaced contours."""
    nphi_period = plasma.nphi // plasma.nfp
    kernel = ws.dipole_kernel(
        plasma.gamma[:nphi_period].reshape(-1, 3),
        plasma.unitnormal[:nphi_period].reshape(-1, 3),
        surface.gamma.reshape(-1, 3), surface.unitnormal.reshape(-1, 3))
    transfer = (jnp.sqrt(ws.quadrature_weights(
        plasma, one_period=True, full_integral=True))[:, None] * kernel
        * ws.quadrature_weights(surface)[None, :])
    xm, xn = ws.potential_modes(surface.nfp)
    potential = (
        jnp.sin(surface.theta2d.reshape(-1, 1) * xm
                - surface.phi2d.reshape(-1, 1) * xn) @ coefficients
        + net_current * surface.phi2d.reshape(-1) / (2 * jnp.pi))
    levels = 2 * COILS_PER_HALF_PERIOD
    delta_phi = net_current / (surface.nfp * levels)
    phase = 2 * jnp.pi * surface.nfp * levels * potential / net_current
    cosine_field = transfer @ (delta_phi / jnp.pi * jnp.sin(phase))
    sine_field = transfer @ (-delta_phi / jnp.pi * jnp.cos(phase))
    return 0.5 * (jnp.vdot(cosine_field, cosine_field)
                  + jnp.vdot(sine_field, sine_field))


def physics(surface):
    operators = ws.build_operators(plasma, surface, net_current)
    coefficients = filtered_coefficients(operators, sigma_cutoff)
    field_error = operators[0] @ coefficients + operators[1]
    return jnp.vdot(field_error, field_error) + finite_coil_error(
        surface, coefficients)


initial_physics = physics(winding)


def objective(active_dofs):
    winding.dofs = initial_dofs.at[active].set(active_dofs)
    return (physics(winding) / initial_physics
            + ws.geometry_control(plasma, winding, reference_geometry))


value_and_grad = jax.jit(jax.value_and_grad(objective))
evaluations = 0


def fun(x):
    global evaluations
    value, gradient = value_and_grad(jnp.asarray(x))
    evaluations += 1
    if evaluations == 1 or evaluations % 10 == 0:
        print(f"evaluation {evaluations:03d}: objective = {float(value):.8e}")
    return float(value), np.asarray(gradient)


x0 = np.asarray(initial_dofs[active])
step = min(ws.COEFFICIENT_STEP_BOUND,
           0.02 * float(ws.mean_minor_radius(plasma)))
start = time.perf_counter()
result = minimize(fun, x0, method="L-BFGS-B", jac=True,
                  bounds=[(x - step, x + step) for x in x0],
                  options={"maxiter": MAXITER, "maxls": 100})
runtime = time.perf_counter() - start
winding.dofs = initial_dofs.at[active].set(jnp.asarray(result.x))

# Use the usual fixed-Kmax current solve for the reported and cut coils.
current_plasma = ws.make_surface(vmec_input, 48, 48)
current_surface = ws.make_surface(vmec_input, 48, 48, winding.dofs)
operators = ws.build_operators(current_plasma, current_surface, net_current)
coefficients, sheet_f_b, achieved_kmax = ws.solve_for_max_current(
    operators, ws.KMAX)
cut_surface = ws.make_surface(vmec_input, 96, 96, winding.dofs)
theta, phi, potential = ws.potential_grid(
    np.asarray(coefficients), cut_surface.nfp, net_current, 96)
ws.COILS_PER_HALF_PERIOD = COILS_PER_HALF_PERIOD
curves, levels = ws.cut_coils(cut_surface, potential)
plasma_eval = ws.make_surface(vmec_input, 48, 48)
nphi = plasma_eval.nphi // plasma_eval.nfp
field = ws.filament_field(
    np.asarray(plasma_eval.gamma[:nphi]).reshape(-1, 3), curves,
    net_current / len(curves))
bnormal = np.sum(
    field * np.asarray(plasma_eval.unitnormal[:nphi]).reshape(-1, 3), axis=1)
mod_b = ws.vmec_mod_b(
    vmec_input, plasma_eval.theta2d[:nphi], plasma_eval.phi2d[:nphi])
weights = np.asarray(ws.quadrature_weights(
    plasma_eval, one_period=True, full_integral=True))
filament_f_b = np.sum(weights * bnormal ** 2)
maximum_ratio = np.max(np.abs(bnormal) / np.abs(mod_b))

print(f"optimizer: success={result.success}, evaluations={evaluations}, "
      f"runtime={runtime:.2f} s")
print(f"sheet f_B: {float(sheet_f_b):.6e} T^2 m^2")
print(f"achieved Kmax: {float(achieved_kmax) / 1e6:.3f} MA/m")
print(f"filament f_B: {filament_f_b:.6e} T^2 m^2")
print(f"filament max |B.n|/B: {maximum_ratio:.6e}")

np.savez("winding_surface_opt_3.npz", dofs=np.asarray(winding.dofs),
         coefficients=np.asarray(coefficients),
         curves=np.asarray(curves, dtype=object),
         filament_f_B=filament_f_b, filament_max_Bn_over_B=maximum_ratio)

figure = plt.figure(figsize=(11, 4.5), constrained_layout=True)
axis = figure.add_subplot(121)
image = axis.contourf(
    phi * cut_surface.nfp, theta, potential, 40, cmap="viridis")
axis.contour(phi * cut_surface.nfp, theta, potential, levels=levels,
             colors="black", linewidths=0.6)
axis.set(xlabel=r"$N_{fp}\phi$", ylabel=r"$\theta$",
         title="Current potential and four coil contours")
figure.colorbar(image, ax=axis, label=r"$N_{fp}\Phi/I_{pol}$")
axis = figure.add_subplot(122, projection="3d")
for curve in curves:
    axis.plot(*curve.T, color="#0072B2", linewidth=0.8)
axis.set_box_aspect(np.ptp(np.concatenate(curves), axis=0))
axis.set_axis_off()
axis.set_title("Cut filament coils")
figure.savefig("winding_surface_opt_3.png", dpi=180)
plt.close(figure)
print("saved winding_surface_opt_3.npz and winding_surface_opt_3.png")
