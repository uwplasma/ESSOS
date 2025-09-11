from jax import grad, lax, vmap, jit
import jax.numpy as jnp
from functools import partial
from scipy.interpolate import  splrep, PPoly

# @jit
def newton(f, x0):
  """Newton's method for root-finding."""
  initial_state = (0, x0)  # (iteration, x)

  def cond(state):
    it, x = state
    # We fix 25 iterations for simplicity, this is plenty for convergence in our tests.
    return (it < 25)

  def body(state):
    it, x = state
    fx, dfx = f(x), grad(f)(x)
    step = fx / dfx
    new_state = it + 1, x - step
    return new_state

  return lax.while_loop(
    cond,
    body,
    initial_state,
  )[1]

@jit
def roots(x, y, shift=0):
    """
    Outputs all unique roots of an interpolated function y(x) = y, where y is a scalar
    and x is a vector. Removes repeated trailing values.

    Args:
      x (array-like): A vector of x values.
      y (array-like): A vector of y values corresponding to x.
      shift (float, optional): A value to shift the y values by. Defaults to 0.

    Returns:
      jnp.ndarray: An array of unique roots where the interpolated function crosses the shifted y value.
    """
    sign_changes = jnp.nonzero(jnp.diff(jnp.sign(y - shift-1e-2)), size=len(y))[0]
    def interpolated_array_at_point(x0):
        return jnp.interp(jnp.array([x0]), x, y, left=0, right=0)[0] - shift
    def find_root(idx):
        return lax.custom_root(interpolated_array_at_point, x[idx], newton, lambda g, y: y / g(1.0))
    roots_array = vmap(find_root)(sign_changes)
    return roots_array

def roots_scipy(x,y, shift = 0):         
    """
    Finds roots using scipy.interpolate

    Args:
      x (array-like): 1D array of independent values, must be strictly increasing (e.g., Time).
      y (array-like): 1D array of dependent values (e.g., X, Y, Z, B, or V).
      shift (float, optional): Value to shift y to find roots at a non-zero value. Default is 0.

    Returns:
      array-like: Array of root values.
    """
    interp = splrep(x, (y - shift), k=3)
    roots = PPoly.from_spline(interp)
    x_values = roots.roots(extrapolate=False)
    return x_values

def read_famus_dipoles(path, debug=False):
    positions, moments, IC_list, pho_list = [], [], [], []
    first_ic_one = -1

    with open(path, "r") as f:
        for line_num, line in enumerate(f):
            if line.strip().startswith("#") or len(line.strip()) == 0:
                continue

            parts = line.split(",")
            if debug and line_num < 10:
                print(f"Line {line_num}: {parts}")

            if len(parts) >= 12:
                try:
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    Ic = float(parts[6])
                    M = float(parts[7])
                    pho = float(parts[8])
                    phi = float(parts[10])
                    theta = float(parts[11])

                    if Ic == 1 and first_ic_one == -1:
                        first_ic_one = line_num

                    positions.append([x, y, z])
                    mx = M * jnp.sin(phi) * jnp.sin(theta)
                    my = -M * jnp.cos(phi) * jnp.sin(theta)
                    mz = M * jnp.cos(theta)
                    moments.append([mx, my, mz])
                    IC_list.append(Ic)
                    pho_list.append(pho)

                except Exception as e:
                    if debug:
                        print(f"Error parsing line {line_num}: {e}")
                    continue

    positions = jnp.array(positions)
    moments = jnp.array(moments)
    return positions, moments, jnp.array(IC_list), jnp.array(pho_list)
