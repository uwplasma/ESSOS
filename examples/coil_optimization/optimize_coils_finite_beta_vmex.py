#!/usr/bin/env python
"""Optimize ESSOS coils for a prescribed low-beta VMEX equilibrium."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import virtual_casing as vc

from essos.coils import Coils
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import surfacerzfourier_from_boundary

NPHI, NTHETA, VC_DIGITS = 24, 24, 4
MAXITER = 120
NORMAL_FIELD_WEIGHT, PRESSURE_BALANCE_WEIGHT = 2.0e4, 5.0e3
LENGTH_TARGET, LENGTH_WEIGHT = 5.0, 0.2
CURVATURE_LIMIT, CURVATURE_WEIGHT = 5.0, 1.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.08, 1.0e3
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = 0.20, 1.0e3
SHAPE_SCALE, CURRENT_SCALE = 0.02, 0.5
# This independent vacuum seed needs more than one scaled shape step to reach
# sub-percent peak B.n/B; [-1, 1] stalls at 1.7% for the same objective.
PARAMETER_BOUND = 3.0

ci_smoke = os.environ.get("ESSOS_EXAMPLES_CI") == "1"
if ci_smoke:
    NPHI, NTHETA, VC_DIGITS, MAXITER = 8, 8, 3, 1

DATA = Path(__file__).resolve().parents[1] / "input_files"
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_beta0p5_bootstrap")

print("Solving the fixed-boundary beta=0.5% QA target with bootstrap current...")
equilibrium = opt.solve_equilibrium(inp)
surface_data = vc.surface_field_data_from_state(
    inp, equilibrium.state, runtime=equilibrium.runtime, nphi=NPHI, ntheta=NTHETA)
precision = vc.plan_vc_precision(surface_data, digits=VC_DIGITS)
interface = vc.PlasmaVacuumInterface.from_surface_data(
    surface_data, digits=VC_DIGITS, precision=precision)
surface = surfacerzfourier_from_boundary(
    inp.rbc, inp.zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)

# Start from the matched vacuum coil shapes and rescale their common current so
# their boundary-field RMS is appropriate for this finite-beta equilibrium.
coils0 = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
n_shape, n_current = coils0.curves.dofs.size, coils0.dofs_currents.size
x0 = np.array(coils0.dofs, copy=True)

def coil_field(coils):
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)

B_reference = jnp.sqrt(jnp.sum(interface.weights * jnp.sum(surface_data.B_total**2, axis=0)))
B_coils0 = interface.external_B(coil_field(coils0))
current_factor = B_reference / jnp.sqrt(jnp.sum(interface.weights * jnp.sum(B_coils0**2, axis=0)))
# Squared interface residuals can admit a globally reversed near-solution.
# Select the current orientation aligned with the VMEX total field.
alignment = jnp.sum(interface.weights * jnp.sum(B_coils0 * surface_data.B_total, axis=0))
current_factor *= jnp.where(alignment < 0.0, -1.0, 1.0)
x0[-n_current:] *= float(current_factor)
scales = np.r_[np.full(n_shape, SHAPE_SCALE), np.full(n_current, CURRENT_SCALE)]

def coils_from_u(u):
    return coils0.with_dofs(jnp.asarray(x0) + jnp.asarray(scales) * u)

def objective(u):
    coils = coils_from_u(u); external_field = coil_field(coils)
    normal = jnp.sqrt(interface.weights) * interface.bnormal_residual(external_field) / B_reference
    pressure = jnp.sqrt(interface.weights) * interface.pressure_balance_residual(external_field) / B_reference**2
    lengths = coils.length[:n_current] - LENGTH_TARGET
    curvature = jnp.maximum(coils.curvature - CURVATURE_LIMIT, 0.0)
    costs = jnp.asarray([
        0.5 * NORMAL_FIELD_WEIGHT * jnp.vdot(normal, normal),
        0.5 * PRESSURE_BALANCE_WEIGHT * jnp.vdot(pressure, pressure),
        0.5 * LENGTH_WEIGHT * jnp.vdot(lengths, lengths),
        0.5 * CURVATURE_WEIGHT * jnp.mean(curvature**2),
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_LIMIT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_LIMIT, block_size=32),
    ])
    return jnp.sum(costs), costs

term_names = ("normal field", "pressure balance", "length", "curvature",
              "coil separation", "coil-surface separation")
monitor = opt.OptimizationMonitor(); value_and_grad_jax = jax.jit(
    jax.value_and_grad(objective, has_aux=True))

# This explicit adapter is all SciPy needs: a scalar, its exact JAX gradient,
# and optional per-term values for the progress table/plot.
def value_and_grad(u):
    (value, costs), gradient = value_and_grad_jax(jnp.asarray(u))
    return monitor.cache_evaluation(
        u, value, gradient, dict(zip(term_names, map(float, np.asarray(costs)))))

problem = vj.FunctionProblem.from_functions(
    np.zeros_like(x0), value_and_grad=value_and_grad, names=coils0.dof_names)

print(f"Optimizing {x0.size} ESSOS shape/current variables with exact reverse-mode gradients")
print(f"dof_names = {problem.dof_names}")
problem.compile_value_and_gradient(report_interval=10.0)
result = minimize(problem.value_and_grad, problem.x0, jac=True, method="L-BFGS-B",
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size, callback=monitor,
    options={"maxiter": MAXITER, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 20})

coils = coils_from_u(result.x); external_field = coil_field(coils)
Bmag = jnp.linalg.norm(interface.total_B_out(external_field), axis=0)
Bn_over_B = interface.bnormal_residual(external_field) / Bmag
pressure_error = interface.pressure_balance_residual(external_field) / B_reference**2
print(f"B.n/B mean = {100 * float(jnp.sum(interface.weights * jnp.abs(Bn_over_B))):.3f}%, "
      f"RMS = {100 * float(jnp.sqrt(jnp.sum(interface.weights * Bn_over_B**2))):.3f}%, "
      f"max = {100 * float(jnp.max(jnp.abs(Bn_over_B))):.3f}%")
print("Normalized total-pressure jump RMS = "
      f"{float(jnp.sqrt(jnp.sum(interface.weights * pressure_error**2))):.3e}")
print(f"Coil lengths = {np.asarray(coils.length[:n_current])}")
print(f"Maximum curvature = {float(jnp.max(coils.curvature)):.3f} 1/m")

# Save and plot results
coils.to_json("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
surface.to_vtk("surface_LandremanPaulQA_beta0p5_bootstrap", extra_data={
    "B_dot_n_over_B": np.asarray(Bn_over_B)[None], "B": np.asarray(Bmag)[None],
    "pressure_balance_error": np.asarray(pressure_error)[None]})
coils.to_vtk("coils_LandremanPaulQA_beta0p5_bootstrap")
monitor.save("finite_beta_coil_objectives.csv")
monitor.plot("finite_beta_coil_objectives.png", title="Finite-beta coil objective terms")
print("Wrote finite-beta coil JSON, VTK, CSV, and objective plot")
