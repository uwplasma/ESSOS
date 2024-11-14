import jax
jax.config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp
from MagneticField import B, grad_B
from time import time

@jit
def GuidingCenter(t:              float,
                  inital_values:  jnp.ndarray,
                  args:           tuple
                #   gamma:          jnp.ndarray,
                #   gamma_dash:     jnp.ndarray,
                #   currents:       jnp.ndarray,
                #   μ:              float
                  ) -> jnp.ndarray:
    gamma, gamma_dash, currents, μ = args
    """Calculates the motion derivatives with the Guiding Center aproximation
    Attributes:
        inital_values (jnp.array - shape (4,)): Point in phase space where we want to calculate the derivatives
        t (float): Time when the Guiding Center is calculated
        gamma (jnp.array - shape (n_coils, n_segments, 3)): Points of the coils
        gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Points of the coils
        currents (jnp.array - shape (n_coils,)): Currents of the coils
        μ (float): Magnetic moment, the 1st adiabatic constant
    Returns:
        Dx, Dvpar: Derivatives of the position and parallel velocity at time t due to the given coils
    """

    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (4,), "initial values must have shape (4,) with x, y, z, vpar"
    assert inital_values.dtype == float, "initial values must be a float"
    # assert isinstance(t, float), f"time must be a float, not a {type(t)}"
    # assert t >= 0, "time must be positive"
    assert isinstance(gamma, jnp.ndarray), "gamma must be a jnp.ndarray"
    assert gamma.ndim == 3, "gamma must be a 3D array"
    assert gamma.shape[0] == currents.size, "number of coils must match number of currents"
    assert gamma.shape[2] == 3, "gamma must have shape (n_coils, n_segments, 3)"
    assert gamma.dtype == float, "gamma must be a float"
    assert isinstance(gamma_dash, jnp.ndarray), "gamma_dash must be a jnp.ndarray"
    assert gamma_dash.ndim == 3, "gamma_dash must be a 3D array"
    assert gamma_dash.shape[0] == currents.size, "number of coils must match number of currents"
    assert gamma_dash.shape[2] == 3, "gamma_dash must have shape (n_coils, n_segments, 3)"
    assert gamma_dash.dtype == float, "gamma_dash must be a float"
    assert isinstance(currents, jnp.ndarray), "currents must be a jnp.ndarray"
    assert currents.ndim == 1, "currents must be a 1D array"
    assert currents.dtype == float, "currents must be a float"
    # assert isinstance(μ, float), f"μ must be a float, not a {type(μ)}"

    # Charge and mass for alpha particles in SI units
    q = 2*1.602176565e-19
    m = 4*1.660538921e-27

    # Calculationg the magentic field
    x, y, z, vpar = inital_values
    
   # Condition to check if any of x, y, z is greater than 10
    condition = (jnp.sqrt(x**2 + y**2) > 100) | (jnp.abs(z) > 20)

    def compute_derivatives(_):
        r = jnp.array([x, y, z])
        
        # start_time = time()
        B_field = B(r, gamma, gamma_dash, currents)
        # print(f"Time to calculate B: {time()-start_time:.2f} seconds")
        normB = jnp.linalg.norm(B_field)
        b = B_field/normB
        
        # Gyrofrequency
        Ω = q*normB/m
        
        # Gradient of the magnetic field
        # start_time = time()
        gradB = grad_B(r, gamma, gamma_dash, currents)
        # print(f"Time to calculate gradB: {time()-start_time:.2f} seconds")

        # Position derivative of the particle
        Dx = vpar*b + (vpar**2/Ω+μ/q)*jnp.cross(b, gradB)/normB
        # Parallel velocity derivative of the particle
        Dvpar = -μ/m*jnp.dot(b,gradB)

        return jnp.append(Dx,Dvpar)

    def zero_derivatives(_):
        return jnp.zeros(4, dtype=float)

    return jax.lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)

@jit
def Lorentz(inital_values: jnp.ndarray,
            t:             float,
            gamma:         jnp.ndarray,
            gamma_dash:    jnp.ndarray,
            currents:      jnp.ndarray) -> jnp.ndarray:
    
    """Calculates the motion derivatives following the full gyromotion
    Attributes:
        inital_values (jnp.array - shape (6,)): Point in phase space where we want to calculate the derivatives
        t (float): Time when the Guiding Center is calculated
        gamma (jnp.array - shape (n_coils, n_segments, 3)): Points of the coils
        gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Points of the coils
        currents (jnp.array - shape (n_coils,)): Currents of the coils
    Returns:
        Dx, Dvpar: Derivatives of the position and parallel velocity at time t due to the given coils
    """

    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (6,), "initial values must have shape (6,) with x, y, z, vx, vy, vz"
    assert inital_values.dtype == float, "initial values must be a float"
    #assert isinstance(t, float), f"time must be a float, not a {type(t)}"
    #assert t >= 0, "time must be positive"
    assert isinstance(gamma, jnp.ndarray), "gamma must be a jnp.ndarray"
    assert gamma.ndim == 3, "gamma must be a 3D array"
    assert gamma.shape[0] == currents.size, "number of coils must match number of currents"
    assert gamma.shape[2] == 3, "gamma must have shape (n_coils, n_segments, 3)"
    assert gamma.dtype == float, "gamma must be a float"
    assert isinstance(gamma_dash, jnp.ndarray), "gamma_dash must be a jnp.ndarray"
    assert gamma_dash.ndim == 3, "gamma_dash must be a 3D array"
    assert gamma_dash.shape[0] == currents.size, "number of coils must match number of currents"
    assert gamma_dash.shape[2] == 3, "gamma_dash must have shape (n_coils, n_segments, 3)"
    assert gamma_dash.dtype == float, "gamma_dash must be a float"
    assert isinstance(currents, jnp.ndarray), "currents must be a jnp.ndarray"
    assert currents.ndim == 1, "currents must be a 1D array"
    assert currents.dtype == float, "currents must be a float"

    # Charge and mass for alpha particles in SI units
    q = 2*1.602176565e-19
    m = 4*1.660538921e-27

    # Calculationg the magentic field
    x, y, z, vx, vy, vz = inital_values
    
   # Condition to check if x, y, z is greater than a threshold
    condition = (jnp.sqrt(x**2 + y**2) > 50) | (jnp.abs(z) > 20)

    def compute_derivatives(_):
        r = jnp.array([x, y, z])
        B_field = B(r, gamma, gamma_dash, currents)

        # Position derivative of the particle
        Dx = jnp.array([vx, vy, vz])
        # Parallel velocity derivative of the particle
        Dv = q / m * jnp.cross(Dx, B_field)

        return jnp.concatenate((Dx, Dv))

    def zero_derivatives(_):
        return jnp.zeros(6, dtype=float)

    return jax.lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)

@jit
def FieldLine(inital_values: jnp.ndarray,
              t:             float,
              gamma:         jnp.ndarray,
              gamma_dash:    jnp.ndarray,
              currents:      jnp.ndarray) -> jnp.ndarray:
    
    """ Calculates the motion derivatives for a certain field line 
        Attributes:
    inital_values: Point in phase space where we want to calculate the derivatives - shape (4,)
    t: Time when the field line is calculated
    currents: Currents of the coils - shape (n_coils,)
    gamma: Points of the coils - shape (n_coils, n_segments, 3)
        Returns:
    Dx, Dvpar: Derivatives of the position and parallel velocity at time t due to the given coils
    """

    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (3,), "initial values must have shape (3,) with x, y, z"
    assert inital_values.dtype == float, "initial values must be a float"
    #assert isinstance(t, float), f"time must be a float, not a {type(t)}"
    #assert t >= 0, "time must be positive"
    assert isinstance(gamma, jnp.ndarray), "gamma must be a jnp.ndarray"
    assert gamma.ndim == 3, "gamma must be a 3D array"
    assert gamma.shape[0] == currents.size, "number of coils must match number of currents"
    assert gamma.shape[2] == 3, "gamma must have shape (n_coils, n_segments, 3)"
    assert gamma.dtype == float, "gamma must be a float"
    assert isinstance(gamma_dash, jnp.ndarray), "gamma_dash must be a jnp.ndarray"
    assert gamma_dash.ndim == 3, "gamma_dash must be a 3D array"
    assert gamma_dash.shape[0] == currents.size, "number of coils must match number of currents"
    assert gamma_dash.shape[2] == 3, "gamma_dash must have shape (n_coils, n_segments, 3)"
    assert gamma_dash.dtype == float, "gamma_dash must be a float"
    assert isinstance(currents, jnp.ndarray), "currents must be a jnp.ndarray"
    assert currents.ndim == 1, "currents must be a 1D array"
    assert currents.dtype == float, "currents must be a float"

    # Calculationg the magentic field
    x, y, z = inital_values
    vpar = 299792458 # speed of light
    
   # Condition to check if any of x, y, z is greater than 10
    condition = (jnp.sqrt(x**2 + y**2) > 100) | (jnp.abs(z) > 20)

    def compute_derivatives(_):
        r = jnp.array([x, y, z])
        
        B_field = B(r, gamma, gamma_dash, currents)
        normB = jnp.linalg.norm(B_field)
        b = B_field/normB
        
        # Position derivative of the particle
        Dx = vpar*b

        return Dx

    def zero_derivatives(_):
        return jnp.zeros(3, dtype=float)

    return jax.lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)
