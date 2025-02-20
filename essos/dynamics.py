import jax.numpy as jnp
from jax import jit, lax, random

q = 2*1.602176565e-19
m = 4*1.660538921e-27

@jit
def GuidingCenter(t:              float,
                  inital_values:  jnp.ndarray,
                  args:           tuple) -> jnp.ndarray:
    field, mu = args

    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (4,), "initial values must have shape (4,) with x, y, z, vpar"
    assert inital_values.dtype == float, "initial values must be a float"

    x, y, z, vpar = inital_values
    condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    def dxdt_dvdt(_):
        points = jnp.array([x, y, z])

        B_covariant = field.B_covariant(points)
        B_contravariant = field.B_contravariant(points)
        AbsB = field.AbsB(points)
        gradB = field.dAbsB_by_dX(points)

        omega = q*AbsB/m

        dxdt = vpar*B_contravariant/AbsB + (vpar**2/omega+mu/q)*jnp.cross(B_covariant, gradB)/AbsB/AbsB
        dvdt = -mu/m*jnp.dot(B_contravariant,gradB)/AbsB

        return jnp.append(dxdt,dvdt)

    def zero_derivatives(_):
        return jnp.zeros(4, dtype=float)

    return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@jit
def Lorentz(inital_values: jnp.ndarray,
            t:             float,
            field:          tuple) -> jnp.ndarray:
    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (6,), "initial values must have shape (6,) with x, y, z, vx, vy, vz"
    assert inital_values.dtype == float, "initial values must be a float"

    x, y, z, vx, vy, vz = inital_values
    condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    def dxdt_dvdt(_):
        points = jnp.array([x, y, z])
        B_contravariant = field.B_contravariant(points)

        dxdt = jnp.array([vx, vy, vz])
        dvdt = q / m * jnp.cross(dxdt, B_contravariant)

        return jnp.concatenate((dxdt, dvdt))

    def zero_derivatives(_):
        return jnp.zeros(6, dtype=float)

    return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@jit
def FieldLine(t:             float,
              inital_values: jnp.ndarray,
              field) -> jnp.ndarray:

    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (3,), "initial values must have shape (3,) with x, y, z"
    assert inital_values.dtype == float, "initial values must be a float"

    x, y, z = inital_values
    key = random.PRNGKey(42)
    random_value = random.choice(key, jnp.array([-1.0, 1.0]))
    vpar = random_value*299792458 # speed of light
    condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    def compute_derivatives(_):
        position = jnp.array([x, y, z])
        B_contravariant = field.B_contravariant(position)
        AbsB = field.AbsB(position)
        dxdt = vpar*B_contravariant/AbsB
        return dxdt

    def zero_derivatives(_):
        return jnp.zeros(3, dtype=float)

    return lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)
