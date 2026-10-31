"""Stage-two finite-beta coil optimization against a surface-free field jet."""

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

# Near-axis parameters
CONFIGURATION = "plasma_stellarator"
NPHI = 15
FORMAL_RADIUS = 0.15

# Coil and optimization parameters
N_BASE_COILS = 2
COIL_ORDER = 2
COIL_SEGMENTS = 24
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
INITIAL_CURRENT = 1.5e6
FIELD_JET_WEIGHTS = jnp.asarray([1.0, 1.0, 0.25])
LENGTH_WEIGHT = 1.0e-3
CURVATURE_WEIGHT = 1.0e-4
CURRENT_WEIGHT = 1.0e-5
MAXIMUM_COIL_LENGTH = 5.0
MAXIMUM_COIL_CURVATURE = 10.0
OPTIMIZATION_STEPS = 4
INITIAL_STEP_SIZE = 0.05
BACKTRACKING_STEPS = 8

# Output controls
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT_DIRECTORY = Path("examples/output_files/external_nearaxis_stage_two")

print("Solving the fixed pressure-only stellarator target...")
configuration = qsc.get_configuration(CONFIGURATION)
solution = configuration.solve(nphi=NPHI, order="r2")
NFP = configuration.nfp
target = near_axis_field_jet_target(
    solution,
    formal_radius=FORMAL_RADIUS,
    angular_resolution=64,
)
torsion_rms = jnp.sqrt(
    jnp.sum(solution.torsion**2 * solution.geometry.d_l_d_phi)
    / jnp.sum(solution.geometry.d_l_d_phi)
)
assert solution.inputs.I2 == 0
assert solution.inputs.p2 != 0
assert jnp.abs(solution.iota) > 0.4
assert torsion_rms > 0.5
print("source:", configuration.source_url)
print("iota:", float(solution.iota))
print("I2, p2:", float(solution.inputs.I2), float(solution.inputs.p2))
print("RMS axis torsion:", float(torsion_rms))
print(
    "external target shapes:",
    target.field.shape,
    target.gradient_independent.shape,
    target.hessian_independent.shape,
)

initial_curves = CreateEquallySpacedCurves(
    n_curves=N_BASE_COILS,
    order=COIL_ORDER,
    R=COIL_MAJOR_RADIUS,
    r=COIL_MINOR_RADIUS,
    n_segments=COIL_SEGMENTS,
    nfp=NFP,
    stellsym=True,
)
initial_coils = Coils(
    curves=initial_curves,
    currents=jnp.full(N_BASE_COILS, INITIAL_CURRENT),
)
initial_vector, unravel_coils = ravel_pytree(initial_coils)


def engineering_loss(coils):
    length_excess = jnp.maximum(coils.length / MAXIMUM_COIL_LENGTH - 1, 0)
    curvature_excess = jnp.maximum(
        coils.curvature / MAXIMUM_COIL_CURVATURE - 1,
        0,
    )
    normalized_current = coils.dofs_currents
    return (
        LENGTH_WEIGHT * jnp.mean(jnp.square(length_excess))
        + CURVATURE_WEIGHT * jnp.mean(jnp.square(curvature_excess))
        + CURRENT_WEIGHT * jnp.mean(jnp.square(normalized_current))
    )


def objective(coil_vector):
    coils = unravel_coils(coil_vector)
    coil_field = BiotSavart(coils)
    return loss_field_jet_coils_near_axis(
        coil_field,
        target,
        weights=FIELD_JET_WEIGHTS,
    ) + engineering_loss(coils)


value_and_gradient = jax.jit(jax.value_and_grad(objective))
objective_compiled = jax.jit(objective)
coil_vector = initial_vector
initial_loss = float(objective_compiled(coil_vector))
print("initial normalized objective:", initial_loss)

start_time = perf_counter()
for iteration in range(OPTIMIZATION_STEPS):
    value, gradient = value_and_gradient(coil_vector)
    direction = -gradient / jnp.maximum(jnp.linalg.norm(gradient), 1.0e-14)
    step_size = INITIAL_STEP_SIZE
    accepted = False
    for _ in range(BACKTRACKING_STEPS):
        candidate = coil_vector + step_size * direction
        candidate_value = objective_compiled(candidate)
        if float(candidate_value) < float(value):
            coil_vector = candidate
            accepted = True
            break
        step_size *= 0.5
    print(
        f"iteration {iteration + 1}: loss={float(value):.6e}, "
        f"step={step_size:.3e}, accepted={accepted}"
    )
    if not accepted:
        break

optimized_coils = unravel_coils(coil_vector)
optimized_field = BiotSavart(optimized_coils)
final_loss = float(objective_compiled(coil_vector))
initial_terms = np.asarray(field_jet_loss_terms(BiotSavart(initial_coils), target))
final_terms = np.asarray(field_jet_loss_terms(optimized_field, target))
print("optimization seconds:", perf_counter() - start_time)
print("final normalized objective:", final_loss)
print("initial (B, grad-B, Hessian) terms:", initial_terms)
print("final (B, grad-B, Hessian) terms:", final_terms)

figure = plt.figure(figsize=(10, 4))
axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
axis_3d.plot(
    np.asarray(target.points[:, 0]),
    np.asarray(target.points[:, 1]),
    np.asarray(target.points[:, 2]),
    color="black",
    linewidth=2,
    label="magnetic axis",
)
for index, curve in enumerate(np.asarray(initial_coils.gamma)):
    axis_3d.plot(
        curve[:, 0],
        curve[:, 1],
        curve[:, 2],
        color="tab:gray",
        alpha=0.35,
        label="initial coils" if index == 0 else None,
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
    figure.savefig(
        OUTPUT_DIRECTORY / "stage_two_external_field_jet.png",
        dpi=180,
        bbox_inches="tight",
    )
    print("saved outputs to:", OUTPUT_DIRECTORY)

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)
