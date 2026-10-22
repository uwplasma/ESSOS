"""Forward RHS preparation without redundant views of parameter arrays.

This changes the representation of an existing JAX computation, not the RHS
or its derivatives. Custom derivative and control-flow bodies stay opaque.
"""
import jax
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
