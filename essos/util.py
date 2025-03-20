from jax import grad, lax, vmap, jit
import jax.numpy as jnp
from functools import partial

# imports for scipy 
from scipy.interpolate import  splrep, splev, PPoly
import jax.numpy as jnp

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

@partial(jit, static_argnums=(3))
def roots(x, y, shift=0, size=10):
    """Outputs all roots of an interpolated
    function y(x) = y, where y is a scalar
    and x is a vector. The function is interpolated
    and the roots are found using Newton's method.
    """
    sign_changes = jnp.nonzero(jnp.diff(jnp.sign(y - shift)), size=size)[0]
    def interpolated_array_at_point(x0):
        return jnp.interp(jnp.array([x0]), x, y)[0] - shift
    def find_root(idx):
        return lax.custom_root(interpolated_array_at_point, x[idx], newton, lambda g, y: y / g(1.0))
    roots_list = vmap(find_root)(sign_changes)
    roots_array = jnp.array(roots_list)
    return roots_array

def roots_scipy(x,y, shift = 0):
              
    '''
    Finds roots using scipy 
    
    x: 1D array of independent values, must be strictly increasing -- such as Time
    y: 1D array of dependent values -- such as X, Y, Z, B, or V 
    shift: option to shift y to find roots at a non-zero value 
    '''         
    interp = splrep(x, (y - shift), k=3)
              
    roots = PPoly.from_spline(interp)
              
    x_values = roots.roots(extrapolate=False)
             
    return x_values