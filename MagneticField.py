import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad

@jit
def B_old(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.array:
    """Calculates the magnetic field at a point R from linearized coils with Biot-Savart
        Attributes:
    R: jnp.array: Point where B is calculated - shape (3,)
    curve_points: jnp.array: Coil segments vectors - shape (N_coils, N_CurvePoints, 3)
    currents: jnp.array: Currents of the coils - shape (N_coils,)
        Returns:
    B: jnp.array: Magnetic field at point R - shape (3,)
    """
    directions = jnp.diff(curve_points, axis=1)
    Rprime = jsp.signal.convolve(curve_points, jnp.array([[[0.5],[0.5]]]), mode='valid')
    dB = jnp.divide(jnp.cross(directions,R-Rprime), jnp.reshape(jnp.repeat(jnp.linalg.norm(R-Rprime, axis=2)**3, 3), (len(curve_points),len(Rprime[0]),3)))
    dB_sum = jnp.einsum("a,abc", currents*1e-7, dB)
    return jsp.integrate.trapezoid(dB_sum, axis=0)

@jit # TODO: calculate for multiple positions
def B(R: jnp.array, curve_segments: jnp.array, currents:jnp.array) -> jnp.array:

    """Calculates the magnetic field at a point R from linearized coils with Biot-Savart
        Attributes:
    R: Point where B is calculated - shape (3,)
    curve_segments: Coil segments vectors - shape (n_coils, n_segments, 3)
    currents: Currents of the coils - shape (n_coils,)
        Returns:
    B: Magnetic field at point R - shape (3,)
    """

    directions = jnp.diff(curve_segments, axis=1)
    Rprime = jsp.signal.convolve(curve_segments, jnp.array([[[0.5],[0.5]]]), mode='valid')
    dif_R = (R-Rprime).T
    dB = jnp.cross(directions.T, dif_R, axisa=0, axisb=0, axisc=0)/jnp.linalg.norm(dif_R, axis=0)**3
    dB_sum = jnp.einsum("i,bai", currents*1e-7, dB)
    return jsp.integrate.trapezoid(dB_sum, axis=0)

@jit
def B_norm(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> float:
    """Calculates the magnetic field norm at a point R from linearized coils with Biot-Savart
        Attributes:
    R: jnp.array: Point where B is calculated - shape (3,)
    curve_points: jnp.array: Coil segments vectors - shape (N_coils, N_CurvePoints, 3)
    currents: jnp.array: Currents of the coils - shape (N_coils,)
        Returns:
    B: jnp.float32: Magnetic field Norm at point R
    """
    return jnp.linalg.norm(B(R, curve_points, currents))

@jit
def grad_B(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.array:
    """Calculates the magnetic field gradient at a point R from linearized coils with Biot-Savart
        Attributes:
    R: jnp.array: Point where B is calculated - shape (3,)
    curve_points: jnp.array: Coil segments vectors - shape (N_coils, N_CurvePoints, 3)
    currents: jnp.array: Currents of the coils - shape (N_coils,)
        Returns:
    B: jnp.array: Magnetic field gradient at point R - shape (3,)
    """
    return grad(B_norm)(R, curve_points, currents)
