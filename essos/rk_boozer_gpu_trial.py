"""Experimental forward-only CUDA Dopri8 trial for fixed Boozer fields.

The adaptive driver, error norm, controller, dense output and event handling
remain in rk.py. This module places the same RK stages and physical RHS in
one Pallas kernel. It is opt-in while target-GPU performance is validated.
It does not implement a reverse rule or change the field object's storage.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.extend.core import Literal

from .boozer import BoozerField, guiding_center_rhs, _chart
from .rk import _A8, _ERROR8, _RKTrial, _evaluate_inline, _bind_primitive


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
            else:
                result = _bind_primitive(equation.primitive, values, equation.params)
                if not equation.primitive.multiple_results:
                    result = [result]
            env.update(zip(equation.outvars, result))
        return read(graph.outvars[0])
    return evaluate


def prepare_trial_factory(field, mass, charge):
    """Prepare an opt-in trial factory outside jit for a fixed Fourier field."""
    if jax.default_backend() != "gpu" or "cuda" not in jax.devices()[0].client.platform_version.lower():
        raise ValueError("The experimental fused trial requires CUDA")
    if field.table is not None or field.sine_coef is not None:
        raise ValueError("The experimental fused trial currently requires a cosine Fourier Boozer field")
    radial, b = _padded_spline(field.r_knots, field.b_coef)
    flux, profiles = _padded_spline(field.s_knots, field.profile_coef)
    modes = b.shape[1]
    xm = jnp.pad(field.xm, (0, modes-len(field.xm)))
    xn = jnp.pad(field.xn, (0, modes-len(field.xn)))
    arrays = (radial, b, flux, profiles, xm, xn)
    psi0 = float(field.psi0)
    matrix = np.zeros((16, 16))
    for i, row in enumerate(_A8, 1):
        matrix[i, :len(row)] = row
    coefficients = jnp.asarray(matrix)
    error_weights = jnp.asarray(tuple(_ERROR8)+(0., 0.))

    def factory(moment):
        def step(rhs, t, y, h, args, f0, nodes, rows, weights):
            del rhs, t, args, nodes, rows
            if len(weights) != 14:
                raise ValueError("The fused trial implements Dopri8 only")
            def kernel(yref, fref, href, mref, *refs):
                *field_refs, aref, eref, yout, eout, kout, fout = refs
                model = _KernelField(field_refs, psi0)
                point, h = yref[...], href[...]
                evaluate = _kernel_rhs(model, point, mref[...], mass, charge)
                first = fref[...]
                index = jnp.arange(16)[:, None]
                bank = jnp.where(index == 0, h*first[None, :], jnp.zeros((16, 4), point.dtype))
                def stage(i, carry):
                    bank, _, _ = carry
                    point_i = point+jnp.sum(aref[i, :][:, None]*bank, axis=0)
                    final = evaluate(point_i)
                    bank = jnp.where(index == i, h*final[None, :], bank)
                    return bank, point_i, final
                bank, endpoint, final = jax.lax.fori_loop(1, 14, stage, (bank, point, first))
                error = jnp.sum(eref[...][:, None]*bank, axis=0)
                yout[...], eout[...], fout[...] = endpoint, error, final
                for i in range(14):
                    kout[i, :] = _select_one(bank, i, 0).reshape(4)
            shapes = (jax.ShapeDtypeStruct((4,), y.dtype), jax.ShapeDtypeStruct((4,), y.dtype),
                jax.ShapeDtypeStruct((14, 4), y.dtype), jax.ShapeDtypeStruct((4,), y.dtype))
            values = pl.pallas_call(kernel, out_shape=shapes, grid=(),
                compiler_params=plt.CompilerParams(num_warps=1))(
                    y, f0, h, moment, *arrays, coefficients, error_weights)
            return _RKTrial(*values)
        return step
    return factory
