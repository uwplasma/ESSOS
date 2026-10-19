"""Array objectives for Cartesian GC and Lorentz trajectories.

Classifier loss fraction and first-loss weights follow jax_9/jax_10 Tracing.
The heat proxy follows jax_10; perpendicular energy is separately exposed in
joules (jax_9's heat method) and with jax_10's original normalization. State
reconstruction is adapted explicitly to collisionless SI states, not the
normalized five-component Mu-collision states used by those branches. Native
split GC states instead supply evolving physical mu [J/T] as component five.
"""
import jax
import jax.numpy as jnp
from jax import lax


def minor_radius(trajectories, axis_r, axis_z):
    xyz = trajectories[..., :3]
    return jnp.sqrt((jnp.sqrt(xyz[..., 0]**2+xyz[..., 1]**2)-axis_r+1e-12)**2
        +(xyz[..., 2]-axis_z+1e-12)**2)


def mean_final_radius(trajectories, axis_r, axis_z):
    return jnp.mean(minor_radius(trajectories, axis_r, axis_z)[:, -1])


def legacy_rcross_cost(trajectories, axis_r, axis_z):
    """Exact main loss_particle_rcross_final reduction: time mean then norm."""
    return jnp.linalg.norm(jnp.mean(minor_radius(trajectories, axis_r, axis_z), axis=1))


def _sample(trajectories, stride, final_timestep_only):
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if trajectories.ndim != 3 or trajectories.shape[1] == 0:
        raise ValueError("trajectories must have shape (particles, nonempty saves, state)")
    return trajectories[:, -1:, :] if final_timestep_only else trajectories[:, ::stride, :]


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


def classifier_loss_fraction(trajectories, boundary, softness=10., stride=1, final_timestep_only=False):
    sampled = _sample(trajectories, stride, final_timestep_only)
    _, probability, _ = _loss_probabilities(sampled, boundary, softness)
    return jnp.mean(jnp.max(probability, axis=1))


def classifier_loss_location(trajectories, target_position, boundary, softness=10.,
                             stride=1, final_timestep_only=False):
    sampled = _sample(trajectories, stride, final_timestep_only)
    xyz, _, weights = _loss_probabilities(sampled, boundary, softness)
    distance = jnp.linalg.norm(xyz-jnp.asarray(target_position)[None, None, :], axis=-1)
    return jnp.sum(weights*distance)/(jnp.sum(weights)+1e-8)


def kinetic_components(trajectories, field, mass, birth_energy, model):
    """Total/perpendicular energy in J and perpendicular speed in m/s."""
    xyz = trajectories[..., :3]
    if model in ("GuidingCenter", "GuidingCenterAdaptative") and trajectories.shape[-1] in (4, 5):
        magnitude = jax.vmap(field.AbsB)(xyz.reshape((-1, 3))).reshape(xyz.shape[:2])
        if trajectories.shape[-1] == 5:
            perpendicular = jnp.maximum(trajectories[..., 4]*magnitude, 0.)
        else:
            initial_mu = (birth_energy-.5*mass*trajectories[:, 0, 3]**2)/magnitude[:, 0]
            perpendicular = jnp.maximum(initial_mu[:, None]*magnitude, 0.)
        total = .5*mass*trajectories[..., 3]**2+perpendicular
    elif model in ("FullOrbit", "FullOrbitAdaptative") and trajectories.shape[-1] == 6:
        magnetic = jax.vmap(field.B_contravariant)(xyz.reshape((-1, 3))).reshape(xyz.shape)
        direction = magnetic/jnp.linalg.norm(magnetic, axis=-1, keepdims=True)
        velocity = trajectories[..., 3:6]
        parallel = jnp.sum(velocity*direction, axis=-1)
        total = .5*mass*jnp.sum(velocity**2, axis=-1)
        perpendicular = jnp.maximum(total-.5*mass*parallel**2, 0.)
    else:
        raise ValueError("kinetic objectives require Cartesian GC (4 or physical-mu 5) or Lorentz (6) SI states")
    return total, perpendicular, jnp.sqrt(2*perpendicular/mass)


def classifier_energy_cost(trajectories, field, mass, birth_energy, model, boundary,
                           softness=10., stride=1, final_timestep_only=False, kind="heat_jax10"):
    if kind not in ("heat_jax10", "perpendicular_jax9", "perpendicular_jax10"):
        raise ValueError("unknown classifier energy objective")
    sampled = _sample(trajectories, stride, final_timestep_only)
    _, _, weights = _loss_probabilities(sampled, boundary, softness)
    # Reconstruct mu from the original birth save before temporal subsampling.
    valid = jnp.isfinite(trajectories).all(axis=-1)
    safe = jnp.where(valid[..., None], trajectories, 0.)
    total, perpendicular, speed_perp = kinetic_components(safe, field, mass, birth_energy, model)
    if kind == "heat_jax10":
        value = total*speed_perp/(birth_energy*jnp.sqrt(2*birth_energy/mass)+1e-12)
    elif kind == "perpendicular_jax10":
        value = perpendicular/(birth_energy+1e-12)
    else:
        value = perpendicular
    selected = value[:, -1:] if final_timestep_only else value[:, ::stride]
    mask = valid[:, -1:] if final_timestep_only else valid[:, ::stride]
    weights = jnp.where(mask, weights, 0.)
    return jnp.sum(weights*jnp.where(mask, selected, 0.))/(jnp.sum(weights)+1e-8)
