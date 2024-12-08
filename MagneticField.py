import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad, jacfwd

@jit # TODO: calculate for multiple positions
def B(R: jnp.array, gamma: jnp.array, gamma_dash: jnp.array, currents:jnp.array, overal_factor=7e6) -> jnp.array:

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
    dB_sum = jnp.sum((currents[None, None, :] * dB) * 1e-7, axis=2).T * overal_factor
    return jnp.mean(dB_sum, axis=0)

@jit
def BdotGradr(r, gamma: jnp.array, gamma_dash: jnp.array, currents:jnp.array, R0=6):
    x, y, z = r
    sqrtx2y2 = jnp.sqrt(x**2 + y**2)
    minor_radius = jnp.sqrt(jnp.square(sqrtx2y2-R0)+z**2)
    num = -R0 + sqrtx2y2
    gradr = jnp.array([x*num/sqrtx2y2/minor_radius, y*num/sqrtx2y2/minor_radius, z/minor_radius])
    B_field = B(r, gamma, gamma_dash, currents)
    return jnp.dot(B_field, gradr)

@jit
def BdotGradTheta(r, gamma: jnp.array, gamma_dash: jnp.array, currents:jnp.array, R0=6):
    x, y, z = r
    sqrtx2y2 = jnp.sqrt(x**2 + y**2)
    denom = (sqrtx2y2-R0)**2+z**2
    minor_radius = jnp.sqrt(jnp.square(sqrtx2y2-R0)+z**2)
    gradtheta = jnp.array([-x*z/sqrtx2y2/denom,-y*z/sqrtx2y2/denom,1/(-R0+sqrtx2y2)/(1+z**2/(R0-sqrtx2y2)**2)])#*minor_radius
    B_field = B(r, gamma, gamma_dash, currents)
    return jnp.dot(B_field, gradtheta)

@jit
def BdotGradPhi(r, gamma: jnp.array, gamma_dash: jnp.array, currents:jnp.array, R0=6):
    x, y, z = r
    major_radius_squared = x**2+y**2
    gradphi = jnp.array([-y/major_radius_squared, x/major_radius_squared, 0])#*jnp.sqrt(major_radius_squared)
    B_field = B(r, gamma, gamma_dash, currents)
    return jnp.dot(B_field, gradphi)

@jit
def BcrossGradBdotGradTheta(r, gamma: jnp.array, gamma_dash: jnp.array, currents:jnp.array, R0=6):
    B_field = B(r, gamma, gamma_dash, currents)
    gradB = grad_B(r, gamma, gamma_dash, currents)
    x, y, z = r
    sqrtx2y2 = jnp.sqrt(x**2 + y**2)
    minor_radius = jnp.sqrt(jnp.square(sqrtx2y2-R0)+z**2)
    denom = (sqrtx2y2-R0)**2+z**2
    gradtheta = jnp.array([-x*z/sqrtx2y2/denom,-y*z/sqrtx2y2/denom,1/(-R0+sqrtx2y2)/(1+z**2/(R0-sqrtx2y2)**2)])#*minor_radius
    return jnp.dot(jnp.cross(B_field, gradB),gradtheta)

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