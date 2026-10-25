"""Private fixed-field CUDA views and compatibility lowering for the native solver."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Literal

from .boozer import BoozerField, guiding_center_rhs, _chart



def _padded_spline(knots, coefficients):
    """Power-of-two kernel view of the same piecewise cubics.

    End extrapolation is expressed about the last knot. Recentring and a
    different reduction order can introduce ordinary floating-point changes.
    This setup is outside jit and intended for fixed, nondifferentiated fields.
    """
    knots, coefficients = np.asarray(knots), np.asarray(coefficients)
    count = len(knots)
    size = 1 << (count-1).bit_length()
    width = 1 << (coefficients.shape[1]-1).bit_length()
    grid = np.full(size, np.inf)
    grid[:count] = knots
    values = np.zeros((size, width, 4))
    values[:count-1, :coefficients.shape[1]] = coefficients
    a, b, c, d = np.moveaxis(coefficients[-1], -1, 0)
    h = knots[-1]-knots[-2]
    shifted = np.stack((a, b+3*a*h, c+2*b*h+3*a*h*h, ((a*h+b)*h+c)*h+d), axis=-1)
    values[count-1:, :coefficients.shape[1]] = shifted
    return jnp.asarray(grid), jnp.asarray(values)


def _select_one(value, index, axis):
    """Bit-preserving scalar/slice selection on a kernel-local array."""
    shape = [1]*value.ndim
    shape[axis] = value.shape[axis]
    mask = jnp.arange(value.shape[axis]).reshape(shape) == index
    dtype = {8: jnp.uint64, 4: jnp.uint32, 2: jnp.uint16, 1: jnp.uint8}[value.dtype.itemsize]
    bits = jax.lax.bitcast_convert_type(value, dtype)
    selected = jnp.max(jnp.where(mask, bits, jnp.zeros_like(bits)), axis=axis, keepdims=True)
    return jax.lax.bitcast_convert_type(selected, value.dtype)


def _kernel_is_finite(value):
    """Exact IEEE finite test using primitives supported by older Pallas.

    JAX 0.7.2's Triton backend cannot lower lax.is_finite. Inspecting the
    exponent preserves NaN/infinity rejection without changing the RHS guard.
    Integer bit operations also avoid assumptions about fast float comparisons.
    """
    if value.dtype == jnp.float64:
        unsigned, exponent = jnp.uint64, 0x7ff0000000000000
    elif value.dtype == jnp.float32:
        unsigned, exponent = jnp.uint32, 0x7f800000
    else:
        raise TypeError("The fused trial finite guard supports float32/float64")
    bits = jax.lax.bitcast_convert_type(value, unsigned)
    mask = jnp.asarray(exponent, dtype=unsigned)
    return jax.lax.bitwise_and(bits, mask) != mask


class _KernelField:
    """Kernel-local Ref adapter; evaluation uses the original field methods."""
    table = sine_coef = None
    modB_derivatives = BoozerField.modB_derivatives

    def __init__(self, refs, psi0):
        radial, b, flux, profiles, xm, xn = refs
        self.r_knots, self.b_coef = radial[...], b
        self.s_knots, self.profile_coef = flux[...], profiles
        self.xm, self.xn, self.psi0 = xm[...], xn[...], psi0

    def profiles(self, s):
        value, slope = BoozerField.profiles(self, s)
        return tuple(value[i] for i in range(3)), tuple(slope[i] for i in range(3))


def _kernel_rhs(model, y0, moment, mass, charge):
    """Lower the original checked RHS; adapt unsupported local-array indexing."""
    def physical(y):
        value = guiding_center_rhs(model, y, moment, mass, charge)
        _, r, theta, *_ = _chart(y)
        B = model.modB_derivatives(r, theta, y[2])[0]
        return jnp.where(jnp.isfinite(B) & (B > 0), value, jnp.nan)

    original = jax.make_jaxpr(physical)(y0)
    flat = jax.make_jaxpr(lambda y: _evaluate_inline(original.jaxpr, original.consts, y)[0])(y0)

    def evaluate(y):
        graph = flat.jaxpr
        env = dict(zip(graph.constvars, flat.consts))
        env.update(zip(graph.invars, (y,)))
        def read(v):
            return v.val if isinstance(v, Literal) else env[v]
        for equation in graph.eqns:
            values = [read(v) for v in equation.invars]
            name = equation.primitive.name
            if name == "slice":
                value = values[0]
                for axis, (start, end, size) in enumerate(zip(equation.params["start_indices"],
                        equation.params["limit_indices"], value.shape)):
                    if start != 0 or end != size:
                        if end-start != 1:
                            raise ValueError("Unsupported kernel-local RHS slice")
                        value = _select_one(value, start, axis)
                result = [value]
            elif name == "dynamic_slice":
                value = values[0]
                for axis, (index, size, wanted) in enumerate(zip(values[1:], value.shape,
                        equation.params["slice_sizes"])):
                    if size != wanted:
                        if wanted != 1:
                            raise ValueError("Unsupported kernel-local dynamic RHS slice")
                        value = _select_one(value, jnp.clip(index, 0, size-1), axis)
                result = [value]
            elif name == "concatenate" and len(values) > 2:
                value = jnp.zeros((len(values),), values[0].dtype)
                for i, v in enumerate(values):
                    value = jnp.where(jnp.arange(len(values)) == i, jnp.reshape(v, ()), value)
                result = [value]
            elif name == "le_to":
                # Positive finite knots followed by +inf; NaN queries sort last.
                result = [jnp.less_equal(values[0], values[1]) | jnp.isnan(values[1])]
            elif name == "is_finite":
                result = [_kernel_is_finite(values[0])]
            else:
                result = _bind_primitive(equation.primitive, values, equation.params)
                if not equation.primitive.multiple_results:
                    result = [result]
            env.update(zip(equation.outvars, result))
        return read(graph.outvars[0])
    return evaluate



def _bind_primitive(primitive, values, parameters):
    """Bind an equation using either supported JAX get_bind_params ABI.

    Older JAX returns (subfunctions, keyword_parameters); newer JAX returns
    only keyword_parameters. Keep legacy subfunctions for opaque AD/call
    primitives rather than discarding them while adapting ordinary equations.
    This Python dispatch runs during graph preparation, not each RK stage.
    """
    binding = primitive.get_bind_params(parameters)
    if isinstance(binding, tuple):
        subfunctions, binding = binding
        return primitive.bind(*subfunctions, *values, **binding)
    return primitive.bind(*values, **binding)


def _evaluate_inline(graph, constants, *arguments):
    """Expose jit boundaries without entering custom AD/control-flow calls."""
    from jax.extend.core import Literal
    environment = dict(zip(graph.constvars, constants))
    environment.update(zip(graph.invars, arguments))

    def read(var):
        return var.val if isinstance(var, Literal) else environment[var]

    for equation in graph.eqns:
        values = [read(var) for var in equation.invars]
        if equation.primitive.name == "jit":
            nested = equation.params["jaxpr"]
            result = _evaluate_inline(nested.jaxpr, nested.consts, *values)
        else:
            result = _bind_primitive(equation.primitive, values, equation.params)
            if not equation.primitive.multiple_results:
                result = [result]
        environment.update(zip(equation.outvars, result))
    return [read(var) for var in graph.outvars]

