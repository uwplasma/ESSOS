"""Test automatic differentiation for tangent."""

import jax.numpy as jnp
from jax import jacfwd, vmap

def helical_axis(phi):
    """Helical axis that closes on itself."""
    R = 1.0
    n_periods = 2
    r_mod = R + 0.3 * jnp.sin(n_periods * phi)
    x = r_mod * jnp.cos(phi)
    y = r_mod * jnp.sin(phi)
    z = 0.4 * jnp.sin(n_periods * phi)
    return jnp.stack([x, y, z], axis=-1)

phi_test = jnp.array([0.0, jnp.pi/4, jnp.pi/2, jnp.pi])

print("Testing JAX automatic differentiation...")
print(f"phi_test: {phi_test}")

# Method 1: jacfwd directly
print("\nMethod 1: jacfwd(axis_function)")
tangent_func = jacfwd(helical_axis)
try:
    tangent = vmap(tangent_func)(phi_test)
    print(f"tangent shape: {tangent.shape}")
    print(f"tangent:\n{tangent}")
    print(f"Has NaN: {jnp.any(jnp.isnan(tangent))}")
except Exception as e:
    print(f"Error: {e}")

# Method 2: Define scalar function and vmap
print("\nMethod 2: Define scalar->vector function")
def axis_single(phi_scalar):
    return helical_axis(phi_scalar)

tangent_func2 = jacfwd(axis_single)
try:
    tangent2 = vmap(tangent_func2)(phi_test)
    print(f"tangent2 shape: {tangent2.shape}")
    print(f"tangent2:\n{tangent2}")
    print(f"Has NaN: {jnp.any(jnp.isnan(tangent2))}")
except Exception as e:
    print(f"Error: {e}")

# Method 3: Compute derivatives manually for each point
print("\nMethod 3: Loop over points")
tangents = []
for phi_val in phi_test:
    tang_func = jacfwd(helical_axis)
    tang = tang_func(phi_val)
    tangents.append(tang)
    print(f"phi={phi_val:.4f}, tangent={tang}")

tangent3 = jnp.stack(tangents, axis=0)
print(f"\ntangent3 shape: {tangent3.shape}")
print(f"tangent3:\n{tangent3}")
print(f"Has NaN: {jnp.any(jnp.isnan(tangent3))}")
