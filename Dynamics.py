import jax
jax.config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp
from MagneticField import B, grad_B

@jit
def GuidingCenter(InitialValues:  jnp.ndarray,
                  t:              float,
                  currents:       jnp.ndarray,
                  curve_segments: jnp.ndarray,
                  μ:              float) -> jnp.ndarray:
    
    """ Calculates the motion derivatives with the Guiding Center aproximation
        Attributes:
    InitialValues: jnp.ndarray: Point in phase space where we want to calculate the derivatives - shape (4,)
    t: float: Time when the Guiding Center is calculated
    currents: jnp.ndarray: Currents of the coils - shape (n_coils,)
    curve_segments: jnp.ndarray: Points of the coils - shape (n_coils, n_segments, 3)
    μ: float: Magnetic moment, the 1st adiabatic constant
        Returns:
    Dx, Dvpar: jnp.ndarray: Derivatives of position and parallel velocity at time t due to the given coils
    """

    assert isinstance(InitialValues, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert InitialValues.shape == (4,), "initial values must have shape (4,) with x, y, z, vpar"
    assert InitialValues.dtype == float, "initial values must be a float"
    #assert isinstance(t, float), f"time must be a float, not a {type(t)}"
    #assert t >= 0, "time must be positive"
    assert isinstance(currents, jnp.ndarray), "currents must be a jnp.ndarray"
    assert currents.ndim == 1, "currents must be a 1D array"
    assert currents.dtype == float, "currents must be a float"
    assert isinstance(curve_segments, jnp.ndarray), "curve segments must be a jnp.ndarray"
    assert curve_segments.ndim == 3, "curve segments must be a 3D array"
    assert curve_segments.shape[0] == currents.size, "number of coils must match number of currents"
    assert curve_segments.shape[2] == 3, "curve segments must have shape (n_coils, n_segments, 3)"
    assert curve_segments.dtype == float, "curve segments must be a float"
    #assert isinstance(μ, float), f"μ must be a float, not a {type(μ)}"

    # Charge and mass for alpha particles in SI units
    q = 2*1.602176565e-19
    m = 4*1.660538921e-27

    # Calculationg the magentic field
    x, y, z, vpar = InitialValues
    
   # Condition to check if any of x, y, z is greater than 10
    condition = (jnp.sqrt(x**2 + y**2) > 100) | (jnp.abs(z) > 20)

    def compute_derivatives(_):
        r = jnp.array([x, y, z])
        
        B_field = B(r, curve_segments, currents)
        normB = jnp.linalg.norm(B_field)
        b = B_field/normB
        
        # Gyrofrequency
        Ω = q*normB/m
        
        # Gradient of the magnetic field
        gradB = grad_B(r, curve_segments, currents)

        # Position derivative of the particle
        Dx = vpar*b + (vpar**2/Ω+μ/q)*jnp.cross(b, gradB)/normB
        # Parallel velocity derivative of the particle
        Dvpar = -μ/m*jnp.dot(b,gradB)

        return jnp.append(Dx,Dvpar)

    def zero_derivatives(_):
        return jnp.zeros(4, dtype=float)

    return jax.lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)

@jit
def Lorentz(InitialValues: jnp.ndarray,
            t: float,
            currents: jnp.ndarray,
            curve_segments: jnp.ndarray) -> jnp.ndarray:
    
    """ Calculates the motion derivatives with the full gyromotion aproximation
        Attributes:
    InitialValues: jnp.ndarray: Point in phase space where we want to calculate the derivatives - shape (6,)
    t: float: Time when the full gyromotion is calculated
    currents: jnp.ndarray: Currents of the coils - shape (n_coils,)
    curve_segments: jnp.ndarray: Points of the coils - shape (n_coils, n_segments, 3)
        Returns:
    Dx, Dv: jnp.ndarray: Derivatives of position and parallel velocity at time t due to the given coils
    """

    assert isinstance(InitialValues, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert InitialValues.shape == (6,), "initial values must have shape (6,) with x, y, z, vx, vy, vz"
    assert InitialValues.dtype == float, "initial values must be a float"
    #assert isinstance(t, float), f"time must be a float, not a {type(t)}"
    #assert t >= 0, "time must be positive"
    assert isinstance(currents, jnp.ndarray), "currents must be a jnp.ndarray"
    assert currents.ndim == 1, "currents must be a 1D array"
    assert currents.dtype == float, "currents must be a float"
    assert isinstance(curve_segments, jnp.ndarray), "curve segments must be a jnp.ndarray"
    assert curve_segments.ndim == 3, "curve segments must be a 3D array"
    assert curve_segments.shape[0] == currents.size, "number of coils must match number of currents"
    assert curve_segments.shape[2] == 3, "curve segments must have shape (n_coils, n_segments, 3)"
    assert curve_segments.dtype == float, "curve segments must be a float"

    # Charge and mass for alpha particles in SI units
    q = 2*1.602176565e-19
    m = 4*1.660538921e-27

    # Calculationg the magentic field
    x, y, z, vx, vy, vz = InitialValues
    
   # Condition to check if x, y, z is greater than a threshold
    condition = (jnp.sqrt(x**2 + y**2) > 50) | (jnp.abs(z) > 20)

    def compute_derivatives(_):
        r = jnp.array([x, y, z])
        B_field = B(r, curve_segments, currents)

        # Position derivative of the particle
        Dx = jnp.array([vx, vy, vz])
        # Parallel velocity derivative of the particle
        Dv = q / m * jnp.cross(Dx, B_field)

        return jnp.concatenate((Dx, Dv))

    def zero_derivatives(_):
        return jnp.zeros(6, dtype=float)

    return jax.lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)
