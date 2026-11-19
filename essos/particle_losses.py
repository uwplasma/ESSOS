"""Array objectives for Cartesian GC and Lorentz trajectories: final minor radius and soft classifier loss fraction."""
import jax
import jax.numpy as jnp
from jax import lax


def minor_radius(trajectories, axis_r, axis_z):
    xyz = trajectories[..., :3]
    return jnp.sqrt((jnp.sqrt(xyz[..., 0]**2+xyz[..., 1]**2)-axis_r+1e-12)**2
        +(xyz[..., 2]-axis_z+1e-12)**2)


def mean_final_radius(trajectories, axis_r, axis_z):
    return jnp.mean(minor_radius(trajectories, axis_r, axis_z)[:, -1])


def _loss_probabilities(trajectories, boundary, softness):
    xyz = trajectories[..., :3]
    finite = jnp.isfinite(xyz).all(axis=-1)
    safe = jnp.where(finite[..., None], xyz, 0.)
    distances = jax.vmap(boundary.evaluate_xyz)(safe.reshape((-1, 3)))
    if distances.shape != (safe.shape[0]*safe.shape[1],):
        raise ValueError("classifier must return one scalar distance per point")
    probability = jnp.where(finite, jax.nn.sigmoid(-softness*distances.reshape(finite.shape)), 0.)
    cumulative = lax.associative_scan(jnp.maximum, probability, axis=1)
    weights = jnp.concatenate([cumulative[:, :1], jnp.diff(cumulative, axis=1)], axis=1)
    return safe, probability, jnp.maximum(weights, 0.)


def classifier_loss_fraction(trajectories, boundary, softness=10.):
    _, probability, _ = _loss_probabilities(trajectories, boundary, softness)
    return jnp.mean(jnp.max(probability, axis=1))
