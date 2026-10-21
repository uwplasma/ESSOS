"""Small forward linearizations inside the accepted-map collision adjoint.

GC velocity collisions have only two kinetic inputs: speed and pitch. Form
their scalar JVPs before transposing the update, rather than building a large
reverse graph through the coefficient calculation. Other physical inputs
remain differentiable, including captured background/profile parameters.
"""
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero


def speed_pitch_linearization(function, speed, pitch):
    """Evaluate the unchanged ``function(speed, pitch)`` with a small AD rule.

    Closure conversion makes active captured parameters explicit. Their JVP
    stays ordinary JAX AD; only the two kinetic partials use scalar forward
    linearizations. No field/coil Jacobian, physical approximation or frozen
    coefficient derivative is introduced. Reflection and zero-diffusion
    conventions come directly from the original production map.
    """
    # Trace the production map through its two-component kinetic state before
    # taking scalar partials. Keeping that shared state graph avoids the much
    # slower scalar-only reverse graph in the production GC kernel.
    def kinetic_state(state):
        return function(state[0], state[1])

    def kinetic(v, xi):
        return kinetic_state(jnp.array([v, xi]))

    # Keep JAX's perturbation-aware treatment of floating captures. Only
    # integer/bool jit tracers need a separate explicit argument: they have
    # trivial tangents but cannot remain in a custom-rule closure.
    graph = jax.make_jaxpr(kinetic)(speed, pitch)
    selectors_mask = tuple(isinstance(value, jax.core.Tracer)
        and not jnp.issubdtype(value.dtype, jnp.inexact) for value in graph.consts)
    selectors = tuple(value for value, flag in zip(graph.consts, selectors_mask) if flag)
    if selectors:
        def bind_selectors(v, xi, values):
            iterator = iter(values)
            constants = [next(iterator) if flag else value
                         for value, flag in zip(graph.consts, selectors_mask)]
            return jax.core.eval_jaxpr(graph.jaxpr, constants, v, xi)[0]
        converted, parameters = jax.closure_convert(bind_selectors, speed, pitch, selectors)
    else:
        original, parameters = jax.closure_convert(kinetic, speed, pitch)
        def converted(v, xi, unused, *values):
            return original(v, xi, *values)

    @jax.custom_jvp
    def evaluate(v, xi, selectors, parameters):
        return converted(v, xi, selectors, *parameters)

    def tangent(primals, tangents):
        v, xi, selectors, parameters = primals
        dv, dxi, _, dp = tangents
        value = evaluate(v, xi, selectors, parameters)
        derivative = jnp.zeros_like(value)
        if not isinstance(dv, SymbolicZero):
            slope = jax.jacfwd(lambda u: converted(u, xi, selectors, *parameters))(v)
            derivative += slope*dv
        if not isinstance(dxi, SymbolicZero):
            slope = jax.jacfwd(lambda u: converted(v, u, selectors, *parameters))(xi)
            derivative += slope*dxi
        if any(not isinstance(d, SymbolicZero) for d in jax.tree.leaves(dp)):
            def materialize(p, d):
                if not isinstance(d, SymbolicZero):
                    return d
                dtype = p.dtype if jnp.issubdtype(p.dtype, jnp.inexact) else jax.dtypes.float0
                return jnp.zeros(p.shape, dtype)
            dp = jax.tree.map(materialize, parameters, dp)
            _, change = jax.jvp(lambda p: converted(v, xi, selectors, *p), (parameters,), (dp,))
            derivative += change
        return value, derivative

    evaluate.defjvp(tangent, symbolic_zeros=True)
    return evaluate(speed, pitch, selectors, parameters)
