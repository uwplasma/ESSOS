import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad, jacfwd

@jit # TODO: calculate for multiple positions
def B(R: jnp.array, gamma: jnp.array, gamma_dash: jnp.array, currents:jnp.array) -> jnp.array:

    """Calculates the magnetic field at a point R from linearized coils with Biot-Savart
        
    Args:
        R (jnp.array - shape (3,)): Point where B is calculated
        gamma (jnp.array - shape (n_coils, n_segments, 3)): Discretized curve
        gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Discretized derivative of the curve
        currents (jnp.array - shape (n_coils,)): Currents of the coils
    Returns:
        B (jnp.array - shape (3,)): Magnetic field at point R
    """

    dif_R = (R-gamma).T
    dB = jnp.cross(gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0)/jnp.linalg.norm(dif_R, axis=0)**3
    dB_sum = jnp.einsum("i,bai", currents*1e-7, dB, optimize="greedy")
    return jnp.mean(dB_sum, axis=0)

@jit
def norm_B(R: jnp.array, gamma: jnp.array, gamma_dash:jnp.array, currents: jnp.array) -> float:
    """Calculates the magnetic field norm at a point R from linearized coils with Biot-Savart
    
    Args:
        R (jnp.array - shape (3,)): Point where B is calculated
        gamma (jnp.array - shape (n_coils, n_segments, 3)): Discretized curve
        gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Discretized derivative of the curve
        currents (jnp.array - shape (n_coils,)): Currents of the coils
    Returns:
        B (jnp.array - shape (3,)): Magnetic field at point R
    """
    return jnp.linalg.norm(B(R, gamma, gamma_dash, currents))

@jit
def grad_B(R: jnp.array, gamma: jnp.array, gamma_dash: jnp.array, currents:jnp.array) -> jnp.array:
    """Calculates the magnetic field gradient at a point R from linearized coils with Biot-Savart
    
    Args:
        R (jnp.array - shape (3,)): Point where B is calculated
        gamma (jnp.array - shape (n_coils, n_segments, 3)): Discretized curve
        gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Discretized derivative of the curve
        currents (jnp.array - shape (n_coils,)): Currents of the coils
    Returns:
        B (jnp.array - shape (3,)): Magnetic field at point R
    """
    return grad(norm_B)(R, gamma, gamma_dash, currents)

@jit
def grad_B_vector(R: jnp.array, gamma: jnp.array, gamma_dash: jnp.array, currents:jnp.array) -> jnp.array:
    """Calculates the magnetic field gradient at a point R from linearized coils with Biot-Savart
    
    Args:
        R (jnp.array - shape (3,)): Point where B is calculated
        gamma (jnp.array - shape (n_coils, n_segments, 3)): Discretized curve
        gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Discretized derivative of the curve
        currents (jnp.array - shape (n_coils,)): Currents of the coils
    Returns:
        B (jnp.array - shape (3,)): Magnetic field at point R
    """
    return jacfwd(B)(R, gamma, gamma_dash, currents)