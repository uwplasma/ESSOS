"""Single-stage finite-beta optimization of a near-axis target and its coils."""

import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyqsc_jax as qsc
from jax.flatten_util import ravel_pytree

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.field_jet import (
    field_jet_loss_terms,
    loss_field_jet_coils_near_axis,
    near_axis_field_jet_target,
)
from essos.fields import BiotSavart

# Near-axis parameters and bounded single-stage variables:
# (rc[1], zs[1], etabar, B2c). I2 remains exactly zero.
CONFIGURATION = "plasma_stellarator"
BASE_RC = jnp.asarray([1.0, -0.5415884, 0.029195854, 0.0048646266])
BASE_ZS = jnp.asarray([0.0, -0.57113713, 0.029922731, 0.0041398546])
INITIAL_NEAR_AXIS_VARIABLES = jnp.asarray(
    [-0.5415884, -0.57113713, 1.1396117, -0.050057083]
)
LOWER_NEAR_AXIS_BOUNDS = jnp.asarray([-0.57, -0.60, 0.95, -0.20])
UPPER_NEAR_AXIS_BOUNDS = jnp.asarray([-0.51, -0.54, 1.30, 0.10])
NFP = 4
I2 = 0.0
P2 = -28248.188
NPHI = 15
FORMAL_RADIUS = 0.15
IOTA_TARGET = -2.8

# Coil and objective parameters
N_BASE_COILS = 2
COIL_ORDER = 2
COIL_SEGMENTS = 24
INITIAL_CURRENT = 1.5e6
FIELD_JET_WEIGHTS = jnp.asarray([1.0, 1.0, 0.25])
IOTA_WEIGHT = 0.05
B20_WEIGHT = 1.0e-4
LENGTH_WEIGHT = 1.0e-3
CURVATURE_WEIGHT = 1.0e-4
CURRENT_WEIGHT = 1.0e-5
MAXIMUM_COIL_LENGTH = 5.0
MAXIMUM_COIL_CURVATURE = 10.0
OPTIMIZATION_STEPS = 3
INITIAL_STEP_SIZE = 0.03
BACKTRACKING_STEPS = 8

# Output controls
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT_DIRECTORY = Path("examples/output_files/external_nearaxis_single_stage")

initial_curves = CreateEquallySpacedCurves(
    n_curves=N_BASE_COILS,
    order=COIL_ORDER,
    R=1.0,
    r=0.5,
    n_segments=COIL_SEGMENTS,
    nfp=NFP,
    stellsym=True,
)
initial_coils = Coils(
    curves=initial_curves,
    currents=jnp.full(N_BASE_COILS, INITIAL_CURRENT),
)
initial_coil_vector, unravel_coils = ravel_pytree(initial_coils)
number_coil_variables = initial_coil_vector.size
initial_vector = jnp.concatenate((initial_coil_vector, INITIAL_NEAR_AXIS_VARIABLES))
reference_configuration = qsc.get_configuration(CONFIGURATION)
print("source:", reference_configuration.source_url)


def solve_near_axis(variables):
    rc1, zs1, etabar, B2c = variables
    rc = BASE_RC.at[1].set(rc1)
    zs = BASE_ZS.at[1].set(zs1)
    return qsc.Qsc(
        rc=rc,
        zs=zs,
        nfp=NFP,
        etabar=etabar,
        I2=I2,
        p2=P2,
        B2c=B2c,
        nphi=NPHI,
        order="r2",
    )


def engineering_loss(coils):
    length_excess = jnp.maximum(coils.length / MAXIMUM_COIL_LENGTH - 1, 0)
    curvature_excess = jnp.maximum(
        coils.curvature / MAXIMUM_COIL_CURVATURE - 1,
        0,
    )
    return (
        LENGTH_WEIGHT * jnp.mean(jnp.square(length_excess))
        + CURVATURE_WEIGHT * jnp.mean(jnp.square(curvature_excess))
        + CURRENT_WEIGHT * jnp.mean(jnp.square(coils.dofs_currents))
    )


def objective(vector):
    coils = unravel_coils(vector[:number_coil_variables])
    solution = solve_near_axis(vector[number_coil_variables:])
    target = near_axis_field_jet_target(
        solution,
        formal_radius=FORMAL_RADIUS,
        angular_resolution=32,
    )
    field_jet_loss = loss_field_jet_coils_near_axis(
        BiotSavart(coils),
        target,
        weights=FIELD_JET_WEIGHTS,
    )
    iota_loss = IOTA_WEIGHT * jnp.square((solution.iota - IOTA_TARGET) / IOTA_TARGET)
    b20_loss = B20_WEIGHT * jnp.square(solution.B20_residual)
    return field_jet_loss + iota_loss + b20_loss + engineering_loss(coils)


value_and_gradient = jax.jit(jax.value_and_grad(objective))
objective_compiled = jax.jit(objective)
optimization_vector = initial_vector
initial_loss = float(objective_compiled(optimization_vector))
initial_solution = solve_near_axis(INITIAL_NEAR_AXIS_VARIABLES)
initial_target = near_axis_field_jet_target(
    initial_solution,
    formal_radius=FORMAL_RADIUS,
    angular_resolution=32,
)
print("initial objective:", initial_loss)
print("initial iota:", float(initial_solution.iota))
print("initial B20 residual:", float(initial_solution.B20_residual))
assert initial_solution.inputs.I2 == 0
assert initial_solution.inputs.p2 != 0

start_time = perf_counter()
for iteration in range(OPTIMIZATION_STEPS):
    value, gradient = value_and_gradient(optimization_vector)
    coil_gradient = gradient[:number_coil_variables]
    near_axis_gradient = gradient[number_coil_variables:]
    direction = jnp.concatenate(
        (
            -coil_gradient / jnp.maximum(jnp.linalg.norm(coil_gradient), 1.0e-14),
            -near_axis_gradient
            / jnp.maximum(jnp.linalg.norm(near_axis_gradient), 1.0e-14),
        )
    )
    step_size = INITIAL_STEP_SIZE
    accepted = False
    for _ in range(BACKTRACKING_STEPS):
        candidate = optimization_vector + step_size * direction
        bounded_near_axis = jnp.clip(
            candidate[number_coil_variables:],
            LOWER_NEAR_AXIS_BOUNDS,
            UPPER_NEAR_AXIS_BOUNDS,
        )
        candidate = candidate.at[number_coil_variables:].set(bounded_near_axis)
        candidate_value = objective_compiled(candidate)
        if float(candidate_value) < float(value):
            optimization_vector = candidate
            accepted = True
            break
        step_size *= 0.5
    print(
        f"iteration {iteration + 1}: loss={float(value):.6e}, "
        f"step={step_size:.3e}, accepted={accepted}"
    )
    if not accepted:
        break

optimized_coils = unravel_coils(optimization_vector[:number_coil_variables])
optimized_near_axis_variables = optimization_vector[number_coil_variables:]
optimized_solution = solve_near_axis(optimized_near_axis_variables)
optimized_target = near_axis_field_jet_target(
    optimized_solution,
    formal_radius=FORMAL_RADIUS,
    angular_resolution=32,
)
optimized_field = BiotSavart(optimized_coils)
final_loss = float(objective_compiled(optimization_vector))
initial_terms = np.asarray(
    field_jet_loss_terms(BiotSavart(initial_coils), initial_target)
)
final_terms = np.asarray(field_jet_loss_terms(optimized_field, optimized_target))
print("optimization seconds:", perf_counter() - start_time)
print("final objective:", final_loss)
print(
    "final near-axis variables (rc1, zs1, etabar, B2c):",
    np.asarray(optimized_near_axis_variables),
)
print("final iota:", float(optimized_solution.iota))
print("final B20 residual:", float(optimized_solution.B20_residual))
optimized_torsion_rms = jnp.sqrt(
    jnp.sum(optimized_solution.torsion**2 * optimized_solution.geometry.d_l_d_phi)
    / jnp.sum(optimized_solution.geometry.d_l_d_phi)
)
assert optimized_solution.inputs.I2 == 0
assert jnp.abs(optimized_solution.iota) > 0.4
assert optimized_torsion_rms > 0.5
print(
    "final I2, p2:",
    float(optimized_solution.inputs.I2),
    float(optimized_solution.inputs.p2),
)
print("final RMS axis torsion:", float(optimized_torsion_rms))
print("initial (B, grad-B, Hessian) terms:", initial_terms)
print("final (B, grad-B, Hessian) terms:", final_terms)

figure = plt.figure(figsize=(10, 4))
axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
axis_3d.plot(
    np.asarray(initial_target.points[:, 0]),
    np.asarray(initial_target.points[:, 1]),
    np.asarray(initial_target.points[:, 2]),
    color="tab:gray",
    linewidth=2,
    label="initial axis",
)
axis_3d.plot(
    np.asarray(optimized_target.points[:, 0]),
    np.asarray(optimized_target.points[:, 1]),
    np.asarray(optimized_target.points[:, 2]),
    color="black",
    linewidth=2,
    label="optimized axis",
)
for index, curve in enumerate(np.asarray(optimized_coils.gamma)):
    axis_3d.plot(
        curve[:, 0],
        curve[:, 1],
        curve[:, 2],
        color="tab:blue",
        label="optimized coils" if index == 0 else None,
    )
axis_3d.set_box_aspect((1, 1, 1))
axis_3d.legend()

axis_loss = figure.add_subplot(1, 2, 2)
locations = np.arange(3)
axis_loss.bar(locations - 0.18, initial_terms, width=0.36, label="initial")
axis_loss.bar(locations + 0.18, final_terms, width=0.36, label="optimized")
axis_loss.set_xticks(locations, ("B", "grad B", "Hessian"))
axis_loss.set_yscale("log")
axis_loss.set_ylabel("mean-square normalized residual")
axis_loss.legend()
figure.tight_layout()

if SAVE_OUTPUT:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    optimized_coils.to_json(OUTPUT_DIRECTORY / "optimized_coils.json")
    optimized_rc = BASE_RC.at[1].set(optimized_near_axis_variables[0])
    optimized_zs = BASE_ZS.at[1].set(optimized_near_axis_variables[1])
    near_axis_output = {
        "rc": np.asarray(optimized_rc).tolist(),
        "zs": np.asarray(optimized_zs).tolist(),
        "etabar": float(optimized_near_axis_variables[2]),
        "B2c": float(optimized_near_axis_variables[3]),
        "I2": I2,
        "p2": P2,
        "iota": float(optimized_solution.iota),
        "torsion_rms": float(optimized_torsion_rms),
        "formal_radius": FORMAL_RADIUS,
        "source_database_id": reference_configuration.source_database_id,
        "source_url": reference_configuration.source_url,
    }
    (OUTPUT_DIRECTORY / "optimized_near_axis.json").write_text(
        json.dumps(near_axis_output, indent=2) + "\n"
    )
    figure.savefig(
        OUTPUT_DIRECTORY / "single_stage_external_field_jet.png",
        dpi=180,
        bbox_inches="tight",
    )
    print("saved outputs to:", OUTPUT_DIRECTORY)

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)
