"""Debug tangent computation issue."""

import numpy as np
import jax.numpy as jnp

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
eps = 1e-8

print("phi_test:", phi_test)
print("phi_test + eps:", phi_test + eps)
print("phi_test - eps:", phi_test - eps)

axis_at = helical_axis(phi_test)
axis_plus = helical_axis(phi_test + eps)
axis_minus = helical_axis(phi_test - eps)

print("\naxis at phi:")
print(axis_at)
print("\naxis at phi+eps:")
print(axis_plus)
print("\naxis at phi-eps:")
print(axis_minus)

print("\nDifference (plus - minus):")
print(axis_plus - axis_minus)

print("\nTangent ((plus - minus) / (2*eps)):")
tangent = (axis_plus - axis_minus) / (2 * eps)
print(tangent)

# Try analytical derivative
print("\n\nAnalytical derivative:")
n_periods = 2
R = 1.0
r_mod = R + 0.3 * jnp.sin(n_periods * phi_test)
dr_dphi = 0.3 * n_periods * jnp.cos(n_periods * phi_test)

dx_dphi = dr_dphi * jnp.cos(phi_test) - r_mod * jnp.sin(phi_test)
dy_dphi = dr_dphi * jnp.sin(phi_test) + r_mod * jnp.cos(phi_test)
dz_dphi = 0.4 * n_periods * jnp.cos(n_periods * phi_test)

tangent_analytical = jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=-1)
print(tangent_analytical)
