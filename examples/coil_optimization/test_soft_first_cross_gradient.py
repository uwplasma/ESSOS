import jax
import jax.numpy as jnp

# Dummy data for testing (replace with real data as needed)
nparticles = 20
ntimesteps = 100
distances = jnp.linspace(1.0, -1.0, ntimesteps)[None, :].repeat(nparticles, axis=0)
q_perp_joules = jnp.abs(jnp.sin(jnp.linspace(0, 2 * jnp.pi, ntimesteps)))[None, :].repeat(nparticles, axis=0)

# Differentiable soft-first-crossing weights
softness_cross = 1.0
loss_indicator_cross = jax.nn.sigmoid(-softness_cross * distances)
softmax_weights = jax.nn.softmax(loss_indicator_cross, axis=1)

# True lost indices
true_lost_indices = jnp.argmax(distances < 0, axis=1)
true_lost_any = jnp.any(distances < 0, axis=1)
true_lost_indices = jnp.where(true_lost_any, true_lost_indices, -1)
particle_indices = jnp.arange(nparticles)

# One-hot at first crossing for lost particles
one_hot_first_cross = jnp.zeros_like(distances)
one_hot_first_cross = one_hot_first_cross.at[particle_indices, true_lost_indices].set(1.0)
one_hot_first_cross = jnp.where(true_lost_any[:, None], one_hot_first_cross, 0.0)

# Combine weights
first_cross_weights = jnp.where(true_lost_any[:, None], one_hot_first_cross, softmax_weights)
first_cross_weights = first_cross_weights / (jnp.sum(first_cross_weights, axis=1, keepdims=True) + 1e-12)

# Q_perp_lost_soft_first
q_perp_lost_soft_first = jnp.sum(first_cross_weights * q_perp_joules, axis=1)

# Define a simple loss function for gradient test
def loss_fn(distances, q_perp_joules):
    softness_cross = 1.0
    loss_indicator_cross = jax.nn.sigmoid(-softness_cross * distances)
    softmax_weights = jax.nn.softmax(loss_indicator_cross, axis=1)
    true_lost_indices = jnp.argmax(distances < 0, axis=1)
    true_lost_any = jnp.any(distances < 0, axis=1)
    true_lost_indices = jnp.where(true_lost_any, true_lost_indices, -1)
    particle_indices = jnp.arange(distances.shape[0])
    one_hot_first_cross = jnp.zeros_like(distances)
    one_hot_first_cross = one_hot_first_cross.at[particle_indices, true_lost_indices].set(1.0)
    one_hot_first_cross = jnp.where(true_lost_any[:, None], one_hot_first_cross, 0.0)
    first_cross_weights = jnp.where(true_lost_any[:, None], one_hot_first_cross, softmax_weights)
    first_cross_weights = first_cross_weights / (jnp.sum(first_cross_weights, axis=1, keepdims=True) + 1e-12)
    q_perp_lost_soft_first = jnp.sum(first_cross_weights * q_perp_joules, axis=1)
    return jnp.sum(q_perp_lost_soft_first)

# Compute gradient with respect to distances
grad_fn = jax.grad(loss_fn, argnums=0)
grad_distances = grad_fn(distances, q_perp_joules)

print("Gradient shape:", grad_distances.shape)
print("Gradient (sample):", grad_distances[0, :10])
