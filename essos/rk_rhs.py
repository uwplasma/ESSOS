"""Forward RHS preparation without redundant views of parameter arrays.

This changes the representation of an existing JAX computation, not the RHS
or its derivatives. Custom derivative and control-flow bodies stay opaque.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal


def coalesced_rhs(graph, t0, y0):
    """Return a prepared RHS, or None when no stored views can be removed.

    Only remove a view when its ancestor is already another setup output.
    Recreate those cheap views at their use site, so the compiler can choose
    their layouts together. All parameter values remain differentiable.
    """
    from .rk import _bind_primitive, _evaluate_inline, _partition_graph

    flat = jax.make_jaxpr(lambda t, y: _evaluate_inline(
        graph.jaxpr, graph.consts, t, y)[0])(t0, y0)
    setup, stage = _partition_graph(flat, [True, True])
    outputs = {v for v in setup.jaxpr.outvars if not isinstance(v, Literal)}
    producers = {v: equation for equation in setup.jaxpr.eqns for v in equation.outvars}
    views = {"transpose", "reshape", "slice", "squeeze", "broadcast_in_dim"}
    aliases = {}
    for output in setup.jaxpr.outvars:
        if isinstance(output, Literal):
            continue
        current, path = output, []
        while current in producers:
            equation = producers[current]
            if (equation.primitive.name not in views or len(equation.invars) != 1
                    or len(equation.outvars) != 1):
                break
            path.append(equation)
            current = equation.invars[0]
            if isinstance(current, Literal):
                break
            if current in outputs:
                aliases[output] = current, tuple(reversed(path))
                break
    if not aliases:
        return None

    bases = [v for v in setup.jaxpr.outvars if not isinstance(v, Literal) and v not in aliases]
    reduced = setup.jaxpr.replace(outvars=bases)
    residuals = jax.core.eval_jaxpr(reduced, setup.consts)

    def evaluate(t, y, unused_args):
        cache = dict(zip(bases, residuals))

        def read(variable):
            if isinstance(variable, Literal):
                return variable.val
            if variable not in cache:
                ancestor, path = aliases[variable]
                value = read(ancestor)
                for equation in path:
                    with equation.ctx.manager:
                        value = _bind_primitive(equation.primitive, (value,), equation.params)
                        if equation.primitive.multiple_results:
                            value = value[0]
                cache[variable] = value
            return cache[variable]

        prepared = [read(v) for v in setup.jaxpr.outvars]
        return jax.core.eval_jaxpr(stage.jaxpr, stage.consts, *prepared, t, y)[0]

    return evaluate


def grouped_reductions_rhs(rhs, t0, y0, args):
    """Group independent small floating-point sums before RK stage expansion.

    The prepared physical RHS is unchanged. Stack independent operands so the
    GPU compiler can evaluate their sums together. This changes layout, not
    the arithmetic count, and remains experimental after a large-batch timing
    regression. Reduction depth prevents the regrouping from
    introducing dependencies between groups. Custom AD rules and effects stay
    opaque; all parameter values retain their ordinary JAX derivatives.
    """
    from .rk import _bind_primitive, _contains_custom_rule, _evaluate_inline

    original = jax.make_jaxpr(lambda t, y: rhs(t, y, args))(t0, y0)
    if original.jaxpr.effects or _contains_custom_rule(original.jaxpr):
        return rhs
    flat = jax.make_jaxpr(lambda t, y: _evaluate_inline(
        original.jaxpr, original.consts, t, y)[0])(t0, y0)
    graph = flat.jaxpr
    producers, depth, candidates = {}, {}, {}
    for equation in graph.eqns:
        level = max((depth.get(v, 0) for v in equation.invars
                     if not isinstance(v, Literal)), default=0)
        if equation.primitive.name.startswith("reduce_"):
            level += 1
        for variable in equation.outvars:
            producers[variable], depth[variable] = equation, level
        if equation.primitive.name == "reduce_sum":
            value = equation.invars[0]
            if (0 < np.prod(value.aval.shape) <= 64 and
                    np.issubdtype(value.aval.dtype, np.floating)):
                key = (value.aval.shape, tuple(equation.params["axes"]),
                       value.aval.dtype, level)
                candidates.setdefault(key, []).append(equation)
    groups = {equation.outvars[0]: (key[1], equations)
              for key, equations in candidates.items() if len(equations) > 1
              for equation in equations}
    if not groups:
        return rhs

    def evaluate(t, y, unused_args):
        environment = dict(zip(graph.constvars, flat.consts))
        environment.update(zip(graph.invars, (t, y)))

        def read(variable):
            if isinstance(variable, Literal):
                return variable.val
            if variable in environment:
                return environment[variable]
            if variable in groups:
                axes, equations = groups[variable]
                operands = jnp.stack(tuple(read(e.invars[0]) for e in equations))
                values = jnp.sum(operands, axis=tuple(axis + 1 for axis in axes))
                environment.update((e.outvars[0], value) for e, value in zip(equations, values))
            else:
                equation = producers[variable]
                values = [read(v) for v in equation.invars]
                with equation.ctx.manager:
                    result = _bind_primitive(equation.primitive, values, equation.params)
                environment.update(zip(equation.outvars,
                    result if equation.primitive.multiple_results else (result,)))
            return environment[variable]

        return read(graph.outvars[0])

    return evaluate
