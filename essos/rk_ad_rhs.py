"""Compact RHS residuals for accepted-step reverse RK reconstruction.

Keep small results of large reductions; rematerialize field-sized arrays.
Parameter-only pullback data is prepared once and remains shared by every
stage. Stage-dependent residuals are packed by dtype into a few small buffers.
Physical RHS values and derivatives are preserved. Recognized stock Biot--Savart
calls use an exact native geometry linearization during reverse replay.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.interpreters import partial_eval

from .rk import _contains_custom_rule, _evaluate_inline


def prepare_reverse_rhs(prepare, evaluate, t, y, raw, *, specialize_biot_savart=True):
    """Repartition the same RHS after exposing jit boundaries to reverse AD.

    This is a reverse-only representation: the adaptive forward driver keeps
    its original prepared RHS. Field objects and their own AD rules are unchanged.
    Stock prepared Biot--Savart B and B/J calls use exact geometry transposes.
    RHSs containing custom derivative primitives retain their original
    preparation, including rules nested in jit or control-flow calls.
    """
    original = jax.make_jaxpr(lambda t,y,p: evaluate(t,y,prepare(p)))(t,y,raw)
    if _contains_custom_rule(original.jaxpr):
        return prepare, evaluate
    if specialize_biot_savart:
        from .rk_ad_biot_savart import has_prepared_biot_savart, prepare_biot_savart_rhs
        if has_prepared_biot_savart(original.jaxpr):
            specialized = prepare_biot_savart_rhs(prepare, evaluate, t, y, raw)
            if specialized is not None:
                return specialized
    if not any(eq.primitive.name == "jit" for eq in original.jaxpr.eqns):
        return prepare, evaluate
    graph = jax.make_jaxpr(lambda t,y,p: _evaluate_inline(original.jaxpr, original.consts,
        t,y,*tree_util.tree_leaves(p))[0])(t,y,raw)
    setup, stage, _, _ = partial_eval.partial_eval_jaxpr_nounits(
        graph, [True, True]+[False]*len(tree_util.tree_leaves(raw)), instantiate=True)

    def prepare_values(raw):
        return tuple(jax.core.eval_jaxpr(setup.jaxpr, setup.consts, *tree_util.tree_leaves(raw)))

    def rhs(t,y,parameters):
        return jax.core.eval_jaxpr(stage.jaxpr, stage.consts, *parameters,t,y)[0]

    return prepare_values, rhs


def prepare_rhs_linearization(rhs, t, y, parameters, *, cache_field_outputs=True):
    """Return ``(evaluate, pullback, initial_cache)`` or None for cheap RHSs.

    ``evaluate(t, y)`` returns the original RHS and compact residuals.
    ``pullback(cache, seed)`` returns state and prepared-parameter cotangents.
    The initial cache is used only for the first accepted step's initial FSAL
    derivative. Other steps reuse the preceding endpoint derivative in AD.
    Named small Biot--Savart outputs remain visible even when their reductions
    are hidden behind a custom JVP. Large geometry arrays are rematerialized.
    ``cache_field_outputs=False`` isolates this retention in internal benchmarks.
    """
    selected = []

    def save_compact_value(primitive, *avals, **params):
        if (cache_field_outputs and primitive.name == "name"
                and params.get("name") == "native_biot_savart_output"):
            keep = len(avals) == 1 and np.prod(avals[0].shape) <= 12
            if keep:
                selected.append(True)
            return bool(keep)
        if primitive.name != "reduce_sum":
            return False
        shape = avals[0].shape
        output_size = np.prod([size for axis, size in enumerate(shape) if axis not in params["axes"]])
        keep = np.prod(shape) > 9 and output_size <= 9
        if keep:
            selected.append(True)
        return bool(keep)

    remat_rhs = jax.checkpoint(rhs, policy=save_compact_value, prevent_cse=False)
    transpose_graph = []

    def linearize(t, y, values):
        value, pullback = jax.vjp(lambda y, p: remat_rhs(t, y, p), y, values)
        graph, shape = jax.make_jaxpr(pullback, return_shape=True)(jnp.zeros_like(value))
        transpose_graph[:] = [(graph.jaxpr, tree_util.tree_structure(shape))]
        return value, tuple(graph.consts)

    graph = jax.make_jaxpr(linearize)(t, y, parameters)
    if not selected:
        return None
    # This is value dependence for storing residuals, not a stop-gradient on
    # physical parameters: the transpose still returns every parameter bar.
    setup, stage, unknown, _ = partial_eval.partial_eval_jaxpr_nounits(
        graph, [True, True]+[False]*len(tree_util.tree_leaves(parameters)),
        instantiate=[True]+[False]*(len(graph.out_avals)-1))
    values = jax.core.eval_jaxpr(setup.jaxpr, setup.consts, *tree_util.tree_leaves(parameters))
    fixed_count = sum(not flag for flag in unknown)
    constants, bindings = values[:fixed_count], values[fixed_count:]

    descriptors, groups = [], {}
    for aval in stage.out_avals[1:]:
        dtype = aval.dtype
        size = int(np.prod(aval.shape))
        group = len(groups) if dtype not in groups else list(groups).index(dtype)
        offset = groups.get(dtype, 0)
        groups[dtype] = offset+size
        descriptors.append((group, offset, size, aval.shape))

    def evaluate(t, y):
        out = jax.core.eval_jaxpr(stage.jaxpr, stage.consts, *bindings, t, y)
        buffers = [[] for _ in groups]
        for value, (group, _, _, _) in zip(out[1:], descriptors):
            buffers[group].append(jnp.ravel(value))
        return out[0], tuple(jnp.concatenate(values) for values in buffers)

    transpose, structure = transpose_graph[0]

    def pullback(packed, seed):
        residuals = [packed[group][offset:offset+size].reshape(shape)
                     for group, offset, size, shape in descriptors]
        fixed, variable = iter(constants), iter(residuals)
        values = [next(variable) if flag else next(fixed) for flag in unknown[1:]]
        out = jax.core.eval_jaxpr(transpose, values, seed)
        return tree_util.tree_unflatten(structure, out)

    return evaluate, pullback, evaluate(t, y)[1]
