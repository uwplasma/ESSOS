import jax
import jax.numpy as jnp
from essos.dynamics import Tracing
from essos.fields import BiotSavart

# Import or define these from your main script or workspace
# from optimize_coils_particle_confinement_loss_fraction_augmented_lagrangian_coils_escape_line_customvjp import (
#     particles, tmax, times_to_trace, timestep, species, boundary, line_point_xyz, line_direction_xyz, init_field
# )

# Dummy objects for test (replace with actual imports/objects)
class DummyField:
    def __init__(self):
        pass

init_field = DummyField()
particles = None
boundary = None
species = None
line_point_xyz = jnp.array([0.0, 0.0, 0.0])
line_direction_xyz = jnp.array([1.0, 0.0, 0.0])
tmax = 1e-4
times_to_trace = 10
timestep = 1e-8

# Dummy Tracing for test (replace with actual Tracing)
class DummyTracing:
    def __init__(self, field, particles, maxtime, model, times_to_trace, timestep, species, boundary):
        self.lost_times = jnp.array([0, 1, 2])
        self.lost_positions = jnp.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 3.0]])

Tracing = DummyTracing

# Loss function
from jax import grad

def loss_mean_distance_to_line(field):
    tracing = Tracing(
        field=field,
        particles=particles,
        maxtime=tmax,
        model='GuidingCenterAdaptative',
        times_to_trace=times_to_trace,
        timestep=timestep,
        species=species,
        boundary=boundary,
    )
    lost_mask = tracing.lost_times >= 0
    lost_positions = tracing.lost_positions[lost_mask]
    if lost_positions.shape[0] == 0:
        return 0.0
    line_dir_norm = line_direction_xyz / (jnp.linalg.norm(line_direction_xyz) + 1e-12)
    vec_to_point = lost_positions - line_point_xyz[None, :]
    projections = jnp.sum(vec_to_point * line_dir_norm[None, :], axis=1, keepdims=True) * line_dir_norm[None, :]
    perp_vectors = vec_to_point - projections
    distances = jnp.linalg.norm(perp_vectors, axis=1)
    return jnp.mean(distances)

# Test differentiability
try:
    grad_loss = grad(loss_mean_distance_to_line)(init_field)
    print("Gradient shape:", jax.tree_util.tree_map(lambda x: x.shape, grad_loss))
    print("Differentiable test PASSED: loss_mean_distance_to_line is differentiable.")
except Exception as e:
    print("Differentiable test FAILED:", e)
