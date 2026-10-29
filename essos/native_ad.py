"""Checkpointed reverse AD of accepted native orbit and collision maps.

Controllers, rejected trials, event fractions and recorded Wiener increments
are frozen. Stage physics, collision coefficients and spatial kicks remain
differentiable. Only one checkpoint segment's stage residuals is live.
"""
from typing import NamedTuple
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util, ad_checkpoint
from jax._src.interpreters import partial_eval
from jax.extend.core import Literal
from jax.custom_derivatives import SymbolicZero, custom_vjp_primal_tree_values
from .rk import (solve_rk, _trial_step, _trial_unrolled, _C, _A, _ERROR,
    _C8, _A8, _ERROR8, _rk45_dense, _dopri8_dense, _RKTrial, RKStatus,
    _evaluate_inline, _bind_primitive)
from .split import solve_split, SplitStatus

def gpu_jit_options():
    """Compile guarded GPU reverse without CUDA command-buffer capture.

    Apply to the outer JIT; the solver and its replay remain on the GPU.
    """
    return ({'compiler_options':{'xla_gpu_enable_command_buffer':''}}
            if jax.default_backend()=='gpu' else {})

def _select_trial_step(execution):
    return _trial_unrolled if execution == 'unrolled' else _trial_step

def _contains_custom_rule(graph):
    """Find opaque derivative rules, including calls nested in control flow."""
    for equation in graph.eqns:
        if equation.primitive.name.startswith("custom_"):
            return True
        for value in tree_util.tree_leaves(equation.params):
            nested = getattr(value, "jaxpr", value)
            if hasattr(nested, "eqns") and _contains_custom_rule(nested):
                return True
    return False


def _partition_graph(graph, unknowns):
    # Partial evaluation can replace a nested custom-JVP call by its primal
    # graph. In that case only partition whole primitive calls by dependency;
    # never transform their bodies. Parameter-only calls still move to setup,
    # avoiding pullbacks of fixed discretization data inside the stage VJP.
    if _contains_custom_rule(graph.jaxpr):
        from jax.extend.core import ClosedJaxpr, Literal
        if not all(unknowns):
            raise ValueError("opaque RHS partition expects state/time inputs")
        examples = [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype) for var in graph.jaxpr.invars]
        original = graph
        graph = jax.make_jaxpr(lambda *values: _evaluate_inline(
            original.jaxpr, original.consts, *values))(*examples)
        body = graph.jaxpr
        unknown = set(body.invars)
        setup_equations, stage_equations = [], []
        for equation in body.eqns:
            dependent = bool(equation.effects) or any(
                not isinstance(var, Literal) and var in unknown for var in equation.invars)
            (stage_equations if dependent else setup_equations).append(equation)
            if dependent:
                unknown.update(equation.outvars)
        residuals, seen = [], set()
        for var in [var for equation in stage_equations for var in equation.invars]+list(body.outvars):
            if not isinstance(var, Literal) and var not in unknown and var not in seen:
                residuals.append(var)
                seen.add(var)
        setup = body.replace(invars=[], outvars=residuals, eqns=setup_equations,
            effects=set().union(*(eq.effects for eq in setup_equations)))
        stage = body.replace(constvars=[], invars=residuals+list(body.invars),
            eqns=stage_equations, effects=set().union(*(eq.effects for eq in stage_equations)))
        return ClosedJaxpr(setup, graph.consts), ClosedJaxpr(stage, ())
    setup, stage, _, _ = partial_eval.partial_eval_jaxpr_nounits(
        graph, unknowns, instantiate=True)
    return setup, stage


def _partition_rhs(rhs, t0, y0, args):
    """Split solve-invariant setup from time/state-dependent stage arithmetic."""
    graph = jax.make_jaxpr(lambda t, y: rhs(t, y, args))(t0, y0)
    return _partition_graph(graph, [True, True])


class _RKCheckpointTape(NamedTuple):
    """Accepted step sizes and segment starts; no per-step state/save tape."""
    step_sizes: jax.Array
    times: jax.Array
    states: jax.Array
    derivatives: jax.Array
    fraction: jax.Array
    final_time: jax.Array
    initial_saves: jax.Array
    count: jax.Array
    status: jax.Array


class _RKSchedule(NamedTuple):
    """Accepted schedule and optional sparse checkpoints for a containing split solve."""
    times: jax.Array
    fraction: jax.Array
    count: jax.Array
    states: object = None
    derivatives: object = None


def _geometry_pullback(values,seed):
    """Transpose every prepared input, including independent tangent aliases.

    The original setup pullback combines the tangent views and handles coil
    symmetry/current scaling. Diagonal placeholders retain their own bars.
    Coordinate-wise matrix products fuse with the geometry operations here.
    """
    point,gamma,ly,lz,lx,weights,z0,tangent,z1,z2 = values
    u,matrix = seed
    r = (point-gamma).T
    cross_tangent = jnp.stack([lx,ly,lz])
    radius = jnp.linalg.norm(r,axis=0)
    inverse3 = 1./radius**3
    inverse5 = inverse3/radius**2
    inverse7 = inverse5/radius**2
    cross = jnp.cross(cross_tangent,r,axisa=0,axisb=0,axisc=0)
    mr = jnp.stack([matrix[i,0]*r[0]+matrix[i,1]*r[1]+matrix[i,2]*r[2] for i in range(3)])
    mtc = jnp.stack([matrix[0,i]*cross[0]+matrix[1,i]*cross[1]+matrix[2,i]*cross[2] for i in range(3)])
    skew = jnp.array([matrix[2,1]-matrix[1,2],matrix[0,2]-matrix[2,0],matrix[1,0]-matrix[0,1]])[:,None,None]
    u = u[:,None,None]
    k = jnp.sum(u*cross+skew*tangent,axis=0)
    d = jnp.sum(cross*mr,axis=0)
    dr = (jnp.cross(u,cross_tangent,axisa=0,axisb=0,axisc=0)*inverse3
          -3*(k*r+jnp.cross(mr,cross_tangent,axisa=0,axisb=0,axisc=0)+mtc)*inverse5
          +15*d*r*inverse7)
    dl = (jnp.cross(r,u,axisa=0,axisb=0,axisc=0)*inverse3
          -3*jnp.cross(r,mr,axisa=0,axisb=0,axisc=0)*inverse5)
    factor = weights/gamma.shape[1]
    dr,dl = dr*factor,dl*factor
    bars = (jnp.sum(dr,axis=(1,2)),-dr.T,dl[1],dl[2],dl[0],
        jnp.sum(k*inverse3-3*d*inverse5,axis=0)/gamma.shape[1],
        matrix[0,0].reshape((1,)),skew*inverse3*factor,
        matrix[1,1].reshape((1,)),matrix[2,2].reshape((1,)))
    return tuple(bar.astype(value.dtype) for bar,value in zip(bars,values))


def _supported_field(closed, jacobian):
    from .fields import BiotSavart
    body = closed.jaxpr
    name = "B_and_dB_by_dX" if jacobian else "B"
    original = getattr(getattr(BiotSavart, name, None), "__wrapped__", None)
    if original is None:
        return False
    source = original.__code__
    expected = f"{name} at {source.co_filename}:{source.co_firstlineno}"
    if (getattr(getattr(body,"debug_info",None),"func_src_info",None) != expected
            or body.effects or len(body.invars) != (10 if jacobian else 6)):
        return False
    shapes = [var.aval.shape for var in body.invars]
    if len(shapes[1]) != 3 or shapes[1][-1] != 3:
        return False
    coils,nodes,_ = shapes[1]
    if coils < 1 or nodes < 1:
        return False
    expected_shapes = [(3,),(coils,nodes,3),(nodes,coils),(nodes,coils),(nodes,coils),(coils,)]
    if jacobian:
        expected_shapes += [(1,),(3,nodes,coils),(1,),(1,)]
    if shapes != expected_shapes or [var.aval.shape for var in body.outvars] != ([(3,),(3,3)] if jacobian else [(3,)]):
        return False
    if any(var.aval.dtype != jnp.dtype('float64') for var in body.invars):
        return False
    cross = [eq for eq in body.eqns if eq.primitive.name == 'jit' and eq.params.get('name') == 'cross']
    return len(cross) == 1 and list(cross[0].invars[1:]) == list(body.invars[2:5])


def _B_pullback(values, seed):
    """Exact field-only transpose, retaining all six independent input bars.

    Coordinate products expose the small cross products to fusion. The setup
    pullback still combines tangent views and maps geometry/current weights to
    the original coil DOFs; no physical inputs are frozen here.
    """
    point, gamma, ly, lz, lx, weights = values
    r = (point-gamma).T
    inverse3 = 1./jnp.linalg.norm(r, axis=0)**3
    inverse5 = inverse3/jnp.sum(r*r, axis=0)
    factor = weights/gamma.shape[1]
    u = seed[0][:, None, None]
    c0, c1, c2 = ly*r[2]-lz*r[1], lz*r[0]-lx*r[2], lx*r[1]-ly*r[0]
    k = u[0]*c0+u[1]*c1+u[2]*c2
    dr = jnp.stack((
        factor*((u[1]*lz-u[2]*ly)*inverse3-3*k*r[0]*inverse5),
        factor*((u[2]*lx-u[0]*lz)*inverse3-3*k*r[1]*inverse5),
        factor*((u[0]*ly-u[1]*lx)*inverse3-3*k*r[2]*inverse5)))
    dl = jnp.stack((r[1]*u[2]-r[2]*u[1], r[2]*u[0]-r[0]*u[2],
                    r[0]*u[1]-r[1]*u[0]))*inverse3*factor
    bars = (jnp.sum(dr, axis=(1, 2)), -dr.T, dl[1], dl[2], dl[0],
            jnp.sum(k*inverse3, axis=0)/gamma.shape[1])
    return tuple(bar.astype(value.dtype) for bar, value in zip(bars, values))


def has_prepared_biot_savart(graph):
    """Recognize ordinary JIT calls without entering custom/control-flow bodies."""
    for equation in graph.eqns:
        if equation.primitive.name == "jit":
            nested = equation.params["jaxpr"]
            name = equation.params.get("name")
            if name in ("B", "B_and_dB_by_dX") and _supported_field(nested, name != "B"):
                return True
            if has_prepared_biot_savart(nested.jaxpr):
                return True
    return False


def _field_call(closed, values, jacobian=True):
    """Keep original values and use the exact transpose to define a JVP.

    Transposing the linear pullback supplies the tangent without freezing its
    geometry coefficients. Ordinary JAX AD still provides higher derivatives.
    """
    pullback = _geometry_pullback if jacobian else _B_pullback
    @jax.custom_jvp
    def evaluate(*values):
        return tuple(jax.core.eval_jaxpr(closed.jaxpr,closed.consts,*values))
    @evaluate.defjvp
    def tangent(primals,tangents):
        value = evaluate(*primals)
        seed_shape = tree_util.tree_map(jnp.zeros_like,value)
        transpose = jax.linear_transpose(lambda seed:pullback(primals,seed),seed_shape)
        return value,transpose(tangents)[0]
    # Expose only the small field outputs to the outer checkpoint policy.
    # A custom JVP otherwise hides its internal field reductions from it.
    return tuple(ad_checkpoint.checkpoint_name(value, "native_biot_savart_output")
                 for value in evaluate(*values))


def _rewrite(graph,constants,*arguments):
    env = dict(zip(graph.constvars,constants))
    env.update(zip(graph.invars,arguments))
    def read(var):
        return var.val if isinstance(var,Literal) else env[var]
    for eq in graph.eqns:
        values = [read(var) for var in eq.invars]
        with eq.ctx.manager:
            if eq.primitive.name == 'jit':
                closed = eq.params['jaxpr']
                name = eq.params.get('name')
                if name in ('B', 'B_and_dB_by_dX') and _supported_field(closed, name != 'B'):
                    result = _field_call(closed, values, name != 'B')
                else:
                    result = _rewrite(closed.jaxpr,closed.consts,*values)
            else:
                result = _bind_primitive(eq.primitive, values, eq.params)
                if not eq.primitive.multiple_results:
                    result = [result]
        env.update(zip(eq.outvars,result))
    return [read(var) for var in graph.outvars]


def prepare_biot_savart_rhs(prepare,evaluate,t,y,raw):
    """Specialize only the prepared reverse RHS; preserve the original setup."""
    parameters = prepare(raw)
    graph = jax.make_jaxpr(evaluate)(t,y,parameters)
    if not has_prepared_biot_savart(graph.jaxpr):
        return None
    def rhs(t,y,parameters):
        return _rewrite(graph.jaxpr,graph.consts,t,y,*tree_util.tree_leaves(parameters))[0]
    # Materialize the transformed graph now so the selection is deterministic.
    changed = jax.make_jaxpr(rhs)(t,y,parameters)
    def prepared_rhs(t,y,parameters):
        return jax.core.eval_jaxpr(changed.jaxpr,changed.consts,t,y,*tree_util.tree_leaves(parameters))[0]
    prepared_rhs._native_biot_savart_transpose = True
    return prepare,prepared_rhs


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


def _zeros(tree):
    def zero(x):
        x = jnp.asarray(x)
        dtype = x.dtype if jnp.issubdtype(x.dtype, jnp.inexact) else jax.dtypes.float0
        return jnp.zeros(x.shape, dtype=dtype)
    return tree_util.tree_map(zero, tree)


def _add(left, right):
    return tree_util.tree_map(lambda a, b: a if a.dtype == jax.dtypes.float0 else a+b, left, right)


class _ParameterActivity(eqx.Module):
    """Static JAX activity flags, distinct from the frozen time schedule."""
    flags: tuple = eqx.field(static=True)


def _convert_all(function, *examples):
    """Expose every graph constant, including outer jit tracers, as an input."""
    graph, shape = jax.make_jaxpr(function, return_shape=True)(*examples)
    structure = tree_util.tree_structure(shape)
    def evaluate(arguments, constants):
        flat = jax.core.eval_jaxpr(graph.jaxpr, constants, *tree_util.tree_leaves(arguments))
        return tree_util.tree_unflatten(structure, flat)
    return evaluate, graph.consts


def _prepared_activity(prepare, active):
    """Exclude fixed tables from parameter transposes before tracing stages."""
    graph=jax.make_jaxpr(prepare)(active)
    live=set(graph.jaxpr.invars)
    for equation in graph.jaxpr.eqns:
        if equation.primitive.name!='stop_gradient' and any(
                not isinstance(v,Literal) and v in live for v in equation.invars):
            live.update(equation.outvars)
    return tuple(not isinstance(v,Literal) and v in live for v in graph.jaxpr.outvars)


def _freeze_fixed(values, flags):
    leaves,structure=tree_util.tree_flatten(values)
    return tree_util.tree_unflatten(structure,
        [value if active else lax.stop_gradient(value) for value,active in zip(leaves,flags)])


def _step_pullback(rhs, y, t, h, endpoint, increments, parameters,
                   fraction, sample_fractions, owned, by, bf, saved_bar, method,
                   initial_fsal=False, cached_pullback=None):
    """Transpose the RK recurrence from reconstructed primal stage values.

    Segment reconstruction supplies the primal stage values. Linear RK/dense
    algebra is transposed separately; each RHS VJP retains the physical
    derivatives at the original stage time/state. Stock prepared Biot--Savart
    calls can use the exact native field linearization. Only one RHS transpose's temporary arrays are
    live at a time, rather than every stage's field intermediates. The reverse
    recurrence includes all stage dependencies; stages are not frozen in AD.
    """
    nodes, rows, _ = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    dense = _rk45_dense if method == "dopri5" else _dopri8_dense
    def outputs(y, endpoint, increments, parameters):
        terminal = dense(y, endpoint, increments, fraction)
        samples = jax.vmap(lambda s: dense(y, endpoint, increments, s))(sample_fractions)
        return terminal, samples
    _, output_vjp = jax.vjp(outputs, y, endpoint, increments, parameters)
    dy, dy1, dk, dense_parameters = output_vjp((by, jnp.where(owned[:, None], saved_bar, 0.)))
    dp = dense_parameters

    dy = dy+dy1
    bank = increments[:-1]
    dk = dk.at[:-1].add(jnp.asarray(rows[-1], y.dtype)[:, None]*dy1)
    coefficients = np.zeros((len(increments), len(bank)))
    for i, row in enumerate(rows, start=1):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, y.dtype)
    times = jnp.asarray((0.,)+nodes, y.dtype)

    def stage(reverse_index, carry):
        dy, dk, dp = carry
        i = len(increments)-1-reverse_index
        weights = coefficients[i]
        point = y+jnp.tensordot(weights, bank, axes=1)
        point = jnp.where(i == len(increments)-1, endpoint, jnp.where(i == 0, y, point))
        # The endpoint FSAL derivative is unscaled; its RK increment also
        # contributes to dense output. Stage zero is visited only on the first
        # accepted step, where the initial FSAL value is an actual RHS call.
        seed = h*dk[i]+jnp.where(i == len(increments)-1, bf, 0.)
        if cached_pullback is None:
            _, rhs_vjp = jax.vjp(lambda state,p: rhs(t+times[i]*h,state,p), point, parameters)
            dpoint, dparams = rhs_vjp(seed)
        else:
            dpoint, dparams = cached_pullback(i, seed)
        return dy+dpoint, dk.at[:-1].add(weights[:, None]*dpoint), _add(dp, dparams)
    bound = len(increments)-1+jnp.asarray(initial_fsal, jnp.int32)
    dy, dk, dp = lax.fori_loop(jnp.asarray(0, jnp.int32), bound, stage, (dy, dk, dp))
    return dy, jnp.where(initial_fsal, 0., h*dk[0]), dp


def _trial_with_residuals(linearize, t, y, h, f0, constants, initial_cache):
    """Replay the unchanged RK stages and keep compact RHS residuals.

    Used only in backward reconstruction; no embedded error is needed here.
    Stage zero's cache belongs to the initial trajectory RHS. It is consumed
    only when ``initial_fsal`` is true in the first accepted step's pullback.
    """
    nodes, rows, error_weights = constants
    n = len(error_weights)
    coefficients = np.zeros((n-2, n-1))
    for i, row in enumerate(rows[:-1]):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, y.dtype)
    times = jnp.asarray(nodes[:-1], y.dtype)
    bank = jnp.concatenate([(h*f0)[None], jnp.zeros((n-2,)+y.shape, y.dtype)])
    cache = tree_util.tree_map(lambda r: jnp.zeros((n,)+r.shape, r.dtype).at[0].set(r), initial_cache)

    def stage(i, values):
        bank, cache = values
        point = y+jnp.tensordot(coefficients[i-1], bank, axes=1)
        f, residuals = linearize(t+times[i-1]*h, point)
        return lax.dynamic_update_index_in_dim(bank, h*f, i, axis=0), tree_util.tree_map(
            lambda buffer, r: lax.dynamic_update_index_in_dim(buffer, r, i, axis=0), cache, residuals)

    bank, cache = lax.fori_loop(1, n-1, stage, (bank, cache))
    endpoint = y+jnp.tensordot(jnp.asarray(rows[-1], y.dtype), bank, axes=1)
    final, residuals = linearize(t+h, endpoint)
    cache = tree_util.tree_map(lambda buffer, r: buffer.at[-1].set(r), cache, residuals)
    trial = _RKTrial(endpoint, jnp.zeros_like(y), jnp.concatenate([bank, (h*final)[None]]), final)
    return trial, cache


def solve_accepted(rhs, y0, args=None, **options):
    """Return the original forward result with an accepted-step reverse rule.

    Closure conversion exposes RHS field/coil parameters as explicit VJP inputs.
    A second conversion exposes forward-only controls (including event/norm
    closures) separately: their cotangents are zero under this AD contract.
    """
    if options.get("event") is not None:
        # The converted driver uses prepared RHS parameters internally; event
        # callbacks must still receive their original public args object.
        options["event"] = tree_util.tree_map(
            lambda function:lambda t,y,unused:function(t,y,args), options["event"])
    y0 = jnp.asarray(y0, dtype=jnp.result_type(y0, float))
    t0 = lax.stop_gradient(jnp.asarray(options.get("t0", 0.), dtype=y0.dtype))
    ts = lax.stop_gradient(jnp.asarray(options.get("ts", ()), dtype=y0.dtype))
    _validate_ad_sizes(adjoint_checkpoint_steps=options.get('adjoint_checkpoint_steps',16),
        adjoint_max_accepted_steps=options.get('adjoint_max_accepted_steps'))
    checkpoint_steps = int(options.pop("adjoint_checkpoint_steps", 16))
    parameter_accumulation = options.pop("adjoint_parameter_accumulation", "segment")
    if parameter_accumulation not in ("step", "segment"):
        raise ValueError("adjoint_parameter_accumulation must be 'step' or 'segment'")
    # Bind arbitrary ESSOS args before conversion. Particles intentionally is
    # an ordinary object, and need not become a new field/particle PyTree just
    # to cross the custom VJP boundary. Its actually used traced attributes,
    # and those of the field, are exposed by the converted numerical graph.
    setup, stage = _partition_rhs(rhs, t0, y0, args)
    raw_parameters = (setup.consts, stage.consts)

    def prepare(raw):
        return jax.core.eval_jaxpr(setup.jaxpr, raw[0]), raw[1]

    def evaluate(t, y, parameters):
        prepared, stage_constants = parameters
        return jax.core.eval_jaxpr(stage.jaxpr, stage_constants, *prepared, t, y)[0]

    def forward(initial, parameters):
        return solve_rk(evaluate, initial, parameters, **options,
            _recorder=_OrbitRecorder(options.get("adjoint_max_accepted_steps") or options.get("max_steps", 1_000_000), checkpoint_steps))

    converted_forward, driver_constants = _convert_all(forward, y0, prepare(raw_parameters))
    controls = (driver_constants, t0, ts)
    constants = (_C, _A, _ERROR) if options.get("method", "dopri5") == "dopri5" else (_C8, _A8, _ERROR8)
    method = options.get("method", "dopri5")

    @jax.custom_vjp
    def integrate(initial, parameters, controls):
        return converted_forward((initial, prepare(parameters)), controls[0])[0]

    def fwd(initial, parameters, controls):
        activity = _ParameterActivity(tuple(leaf.perturbed for leaf in tree_util.tree_leaves(parameters)))
        initial, raw, controls = custom_vjp_primal_tree_values((initial, parameters, controls))
        parameters = prepare(raw)
        out, tape = converted_forward((initial, parameters), controls[0])
        return out, (initial, raw, controls, tape, activity)

    def bwd(residual, bar):
        initial, raw, controls, tape, activity = residual
        tape = eqx.error_if(tape, tape.count > tape.step_sizes.shape[0],
            "accepted-step AD capacity exceeded; increase adjoint_max_accepted_steps")
        raw_leaves, raw_structure = tree_util.tree_flatten(raw)
        active = tuple(value for value, flag in zip(raw_leaves, activity.flags) if flag)
        backward_prepare, backward_rhs = prepare_reverse_rhs(prepare, evaluate, controls[1], initial, raw)
        def prepare_active(values):
            iterator = iter(values)
            full = tree_util.tree_unflatten(raw_structure,
                [next(iterator) if flag else value for value, flag in zip(raw_leaves, activity.flags)])
            return backward_prepare(full)
        parameters, setup_pullback = jax.vjp(prepare_active, active)
        parameter_flags=_prepared_activity(prepare_active,active)
        original_rhs=backward_rhs
        backward_rhs=lambda t,y,p:original_rhs(t,y,_freeze_fixed(p,parameter_flags))
        _, t0, ts = controls
        linearization = prepare_rhs_linearization(backward_rhs, t0, initial, parameters)
        if linearization is None:
            initial_cache = ()
        else:
            linearize, rhs_pullback, initial_cache = linearization
        sample_indices = jnp.arange(len(ts))
        terminal_bar = jnp.zeros_like(initial) if isinstance(bar.y, SymbolicZero) else bar.y
        saved_bar = jnp.zeros((len(ts), initial.size), initial.dtype) if isinstance(bar.ys, SymbolicZero) else bar.ys

        def replay_trial(t, y, h, f):
            if linearization is None:
                return _trial_step(backward_rhs, t, y, h, parameters, f, *constants), ()
            return _trial_with_residuals(linearize, t, y, h, f, constants, initial_cache)

        def reverse_step(carry, y, t, endpoint, increments, cache):
            i, by, bf, bp = carry
            i = i-1
            h = tape.step_sizes[i]
            fraction = jnp.where(i == tape.count-1, tape.fraction, 1.)
            end_time = jnp.where(i == tape.count-1, tape.final_time, t+h)
            owned = (ts > t) & (ts <= end_time)
            sample_fractions = jnp.where(owned, (ts-t)/h, 0.)
            def cached_pullback(index, seed):
                return rhs_pullback(tree_util.tree_map(lambda buffer: buffer[index], cache), seed)
            dy, df, dp = _step_pullback(backward_rhs, y, t, h, endpoint, increments, parameters,
                fraction, sample_fractions, owned, by, bf, saved_bar, method,
                initial_fsal=(i == 0), cached_pullback=None if linearization is None else cached_pullback)
            contribution = setup_pullback(dp)[0] if parameter_accumulation == "step" else dp
            return i, dy, df, _add(bp, contribution)

        def reverse_segment(carry):
            end = carry[0]
            segment = jnp.maximum((end-1) // checkpoint_steps, 0)
            start = segment*checkpoint_steps
            y, f = tape.states[segment], tape.derivatives[segment]
            t = tape.times[segment]
            if checkpoint_steps == 1:
                trial, cache = replay_trial(t, y, tape.step_sizes[start], f)
                return reverse_step(carry, y, t, trial.y, trial.increments, cache)

            # Reconstruct the segment once. Cache RK stage vectors and compact
            # RHS reductions; field-sized arrays are rematerialized.
            # Rejections and adaptation are absent from this RK reverse replay.
            states = jnp.zeros((checkpoint_steps, initial.size), initial.dtype)
            endpoints = jnp.zeros_like(states)
            increments = jnp.zeros((checkpoint_steps, len(constants[2]), initial.size), initial.dtype)
            times = jnp.zeros((checkpoint_steps,), initial.dtype)
            cache = tree_util.tree_map(lambda r: jnp.zeros(
                (checkpoint_steps, len(constants[2]))+r.shape, r.dtype), initial_cache)
            def reconstruct(local):
                i, t, y, f, states, endpoints, increments, times, cache = local
                slot = i-start
                states = states.at[slot].set(y)
                times = times.at[slot].set(t)
                h = tape.step_sizes[i]
                trial, residuals = replay_trial(t, y, h, f)
                cache = tree_util.tree_map(lambda buffer, r: buffer.at[slot].set(r), cache, residuals)
                endpoints = endpoints.at[slot].set(trial.y)
                increments = increments.at[slot].set(trial.increments)
                # An event can only terminate the last accepted step. Its
                # full endpoint is harmless here: no later slot consumes it.
                return i+1, t+h, trial.y, trial.f1, states, endpoints, increments, times, cache
            _, _, _, _, states, endpoints, increments, times, cache = lax.while_loop(
                lambda local: local[0] < end, reconstruct,
                (start, t, y, f, states, endpoints, increments, times, cache))
            def local_reverse(local):
                index = local[0]-1-start
                return reverse_step(local, states[index], times[index], endpoints[index], increments[index],
                                    tree_util.tree_map(lambda buffer: buffer[index], cache))
            return lax.while_loop(lambda local: local[0] > start, local_reverse, carry)

        def reverse(carry):
            if parameter_accumulation == "step":
                return reverse_segment(carry)
            # Preparation's pullback is linear in these cotangents. Sum them
            # locally before applying it; the outer carry still stores only
            # active raw-parameter bars, never a full-trajectory field tape.
            i, by, bf, bp = carry
            i, dy, df, dp = reverse_segment((i, by, bf, _zeros(parameters)))
            return i, dy, df, _add(bp, setup_pullback(dp)[0])

        _, by, bf, bp = lax.while_loop(lambda carry: carry[0] > 0, reverse,
            (tape.count, terminal_bar, jnp.zeros_like(initial), _zeros(active)))
        by = by+jnp.sum(jnp.where((sample_indices < tape.initial_saves)[:, None], saved_bar, 0.), axis=0)
        iterator = iter(bp)
        raw_bar = tree_util.tree_unflatten(raw_structure,
            [next(iterator) if flag else _zeros(value) for value, flag in zip(raw_leaves, activity.flags)])
        derivatives = (by, raw_bar, _zeros(controls))
        return eqx.error_if(derivatives, tape.status > RKStatus.EVENT,
            "accepted-step AD requires a successful/event solve; inspect status and max_steps")

    integrate.defvjp(fwd, bwd, symbolic_zeros=True)
    return integrate(y0, raw_parameters, controls)


def replay_accepted(rhs, y0, args, tape, ts, *, method='dopri8', stage_execution='loop'):
    """Independent JVP/FD oracle: propagate new inputs on the frozen grid."""
    tape = tree_util.tree_map(lax.stop_gradient, tape)
    dense = _rk45_dense if method == 'dopri5' else _dopri8_dense
    constants = (_C, _A, _ERROR) if method == 'dopri5' else (_C8, _A8, _ERROR8)
    trial_step = _select_trial_step(stage_execution)
    ts = lax.stop_gradient(jnp.asarray(ts))
    y0 = jnp.asarray(y0)
    t0 = tape.times[0]
    saved = jnp.where((jnp.arange(len(ts)) < tape.initial_saves)[:,None],y0,jnp.inf)
    def step(c):
        i,t,y,f,saved=c
        h=tape.step_sizes[i]
        trial=trial_step(rhs,t,y,h,args,f,*constants)
        end=jnp.where(i==tape.count-1,tape.final_time,t+h)
        fraction=jnp.where(i==tape.count-1,tape.fraction,1.)
        owned=(ts>t)&(ts<=end)
        fractions=jnp.where(owned,(ts-t)/h,0.)
        values=jax.vmap(lambda s:dense(y,trial.y,trial.increments,s))(fractions)
        return i+1,t+h,dense(y,trial.y,trial.increments,fraction),trial.f1,jnp.where(owned[:,None],values,saved)
    _,_,y,_,saved=lax.while_loop(lambda c:c[0]<tape.count,step,(jnp.array(0),t0,y0,rhs(t0,y0,args),saved))
    return saved,y

class _SplitCheckpointTape(NamedTuple):
    """Accepted schedules/noise and a flat bank of sparse inner checkpoints."""
    times: jax.Array
    increments: jax.Array
    kicks: jax.Array
    offsets: jax.Array
    orbit_times: jax.Array
    fractions: jax.Array
    states: jax.Array
    count: jax.Array
    orbit_count: jax.Array
    initial_saves: jax.Array
    saved_count: jax.Array
    status: jax.Array
    checkpoint_offsets: jax.Array
    orbit_states: jax.Array
    orbit_derivatives: jax.Array
    orbit_endpoints: jax.Array
    checkpoint_count: jax.Array


def _checked_tape(tape, *, compact=False):
    # A checked count gates replay without routing large banks through branches.
    checked = tape.count if compact else tape
    for condition, message in (
        (tape.status > SplitStatus.EVENT,
         "accepted split AD requires a successful/event solve; inspect status and work limits"),
        (tape.count > tape.times.shape[0],
         "accepted split AD capacity exceeded; increase adjoint_max_accepted_steps"),
        (tape.orbit_count > tape.orbit_times.shape[0],
         "accepted orbit AD capacity exceeded; increase adjoint_max_orbit_steps")):
        checked = eqx.error_if(checked, condition, message)
    return tape._replace(count=checked) if compact else checked


def _partition(function, *examples):
    """Expose closure parameters and hoist parameter-only kernel preparation."""
    graph, shape = jax.make_jaxpr(function, return_shape=True)(*examples)
    structure = tree_util.tree_structure(shape)
    setup, stage = _partition_graph(graph, [True] * len(examples))

    def prepare(raw):
        return jax.core.eval_jaxpr(setup.jaxpr, raw[0]), raw[1]

    def evaluate(arguments, parameters):
        residuals, constants = parameters
        out = jax.core.eval_jaxpr(stage.jaxpr, constants, *residuals, *arguments)
        return tree_util.tree_unflatten(structure, out)

    return (setup.consts, stage.consts), prepare, evaluate


def _state_projection(prepare, evaluate, raw, *examples):
    """Prune collision error-only inputs before transposing parameter setup.

    The forward driver still evaluates the complete production CollisionStep.
    Reverse replay uses its state result only. Selecting live prepared inputs
    here avoids differentiating error-only preparation, including expressions
    whose derivatives can be singular even when the physical kick is regular.
    """
    parameters = prepare(raw)
    leaves = tree_util.tree_leaves(parameters)
    graph = jax.make_jaxpr(lambda arguments,p:evaluate(arguments,p).y)(examples,parameters)
    nargs = len(tree_util.tree_leaves(examples))
    state_graph, used = partial_eval.dce_jaxpr(graph.jaxpr,[True],
        instantiate=[True]*nargs+[False]*len(leaves))
    live = used[nargs:]

    def prepare_state(values):
        return tuple(value for value,flag in zip(tree_util.tree_leaves(prepare(values)),live) if flag)

    def state(arguments,values):
        return jax.core.eval_jaxpr(state_graph,graph.consts,*tree_util.tree_leaves(arguments),*values)[0]

    return prepare_state,state


def _orbit_replay(rhs, initial, parameters, tape, index, method, stage_execution="loop"):
    """Forward-only oracle; propagate new physical inputs on the frozen grid."""
    trial_step = _select_trial_step(stage_execution)
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    dense = _rk45_dense if method == "dopri5" else _dopri8_dense
    start, end = tape.offsets[index], tape.offsets[index+1]
    f = rhs(tape.orbit_times[start, 0], initial, parameters)

    def step(local):
        i, y, f = local
        t, h = tape.orbit_times[i]
        trial = trial_step(rhs, t, y, h, parameters, f, *constants)
        fraction = jnp.where(i == end-1, tape.fractions[index], 1.)
        terminal = dense(y, trial.y, trial.increments, fraction)
        return i+1, terminal, trial.f1

    _, final, _ = lax.while_loop(
        lambda local: local[0] < end, step, (start, initial, f))
    return final


def replay_accepted_split(rhs, collision, initial, args, tape, ts, *, method="dopri8", stage_execution="loop"):
    """Forward-only frozen-schedule/noise oracle for finite-difference checks.

    New parameters propagate through every orbit and collision map. Recorded
    checkpoint states are deliberately ignored. No adaptation or noise drawing
    occurs. Returns (saved_states, terminal_state); ordinary JVPs are supported.
    """
    tape = _checked_tape(tree_util.tree_map(lax.stop_gradient, tape))
    ts = lax.stop_gradient(jnp.asarray(ts))
    rhs_bound = lambda t, y, unused: rhs(t, y, args)

    def advance(local):
        index, y, saved = local
        y = _orbit_replay(rhs_bound, y, None, tape, index, method, stage_execution)
        t, end = tape.times[index]
        y = lax.cond(tape.kicks[index],
            lambda y: collision(end, y, end-t, tape.increments[index], args).y,
            lambda y: y, y)
        owned = (ts > t) & (ts <= end)
        return index+1, y, jnp.where(owned[:, None], y, saved)

    _, final, saved = lax.while_loop(lambda local: local[0] < tape.count, advance,
        (jnp.array(0), initial, jnp.broadcast_to(initial, (len(ts), initial.size))))
    saved = jnp.where((jnp.arange(len(ts)) >= tape.saved_count)[:, None], final, saved)
    return saved, final


def solve_accepted_split(rhs, collision, y0, args=None, **options):
    """Attach a discrete reverse rule to the unchanged production split solve."""
    forward_override = options.pop('_forward_override',None)
    # Kernels receive prepared numerical parameters internally. Control
    # callbacks retain the user's original args ABI, while their numerical
    # decisions are frozen by the outer custom VJP.
    if options.get("event") is not None:
        options["event"] = tree_util.tree_map(
            lambda function: lambda t,y,unused,**kw: function(t,y,args,**kw), options["event"])
    if options.get("step_limit") is not None:
        step_limit = options["step_limit"]
        options["step_limit"] = lambda t,y,unused: step_limit(t,y,args)
    y0 = jnp.asarray(y0, dtype=jnp.result_type(y0, float))
    t0 = lax.stop_gradient(jnp.asarray(options.get("t0", 0.), y0.dtype))
    ts = lax.stop_gradient(jnp.asarray(options["ts"], y0.dtype))
    h0 = jnp.asarray(options["dt0"], y0.dtype)
    noise = jnp.zeros((options["noise_dim"],), y0.dtype)
    raw_rhs, prepare_rhs, evaluate_rhs = _partition(lambda t, y: rhs(t, y, args), t0, y0)
    raw_collision, prepare_collision, evaluate_collision = _partition(
        lambda t, y, h, dw: collision(t, y, h, dw, args), t0, y0, h0, noise)
    prepare_collision_state, collision_state = _state_projection(
        prepare_collision,evaluate_collision,raw_collision,t0,y0,h0,noise)

    def prepare(raw):
        return prepare_rhs(raw[0]), prepare_collision(raw[1])

    def orbit_rhs(t, y, parameters):
        return evaluate_rhs((t, y), parameters)

    # Components omitted from a stored endpoint must be copied unchanged by
    # the collision map. This metadata selects compact full-orbit storage;
    # generic maps and GC retain their previous layout.
    collision_indices = options.get("_collision_state_indices")
    forward_options = dict(options)
    if collision_indices is not None:
        # _partition already hoisted all parameter-only RHS preparation.
        forward_options["_orbit_rhs_prepared"] = True

    def forward(initial, parameters):
        if forward_override is not None:
            # The independent recorder supplies the same accepted tape ABI.
            # Its numerical inputs are closure-converted into frozen controls;
            # physical bars still flow through raw_rhs/raw_collision below.
            return forward_override(initial,parameters,record=True)
        return solve_split(lambda t, y, p: orbit_rhs(t, y, p[0]),
            lambda t, y, h, dw, p: evaluate_collision((t, y, h, dw), p[1]),
            initial, parameters, **forward_options, _recorder=_SplitRecorder.from_options(initial, forward_options))

    raw = (raw_rhs, raw_collision)
    converted_forward, driver_constants = _convert_all(forward, y0, prepare(raw))
    controls = (driver_constants, ts)
    outer_segment = int(options.get("adjoint_checkpoint_steps", 16))
    inner_segment = int(options.get("adjoint_orbit_checkpoint_steps", 16))
    workspace = min(options.get("max_orbit_steps", 10_000),
                    options.get("adjoint_max_orbit_steps") or options.get("max_steps", 100_000))
    method = options.get("method", "dopri8")
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)

    @jax.custom_vjp
    def integrate(initial, raw, controls):
        return converted_forward((initial, prepare(raw)), controls[0])[0]

    def fwd(initial, raw, controls):
        activity = _ParameterActivity(tuple(leaf.perturbed for leaf in tree_util.tree_leaves(raw)))
        initial, raw, controls = custom_vjp_primal_tree_values((initial, raw, controls))
        out, tape = converted_forward((initial, prepare(raw)), controls[0])
        saved_controls = (None,controls[1]) if forward_override is not None else controls
        return out, (initial, raw, saved_controls, tape, activity,
                     out.y if collision_indices is not None else None)

    def bwd(residual, bar):
        initial, raw, controls, tape, activity, final = residual
        tape = _checked_tape(tape, compact=collision_indices is not None)
        raw_leaves, raw_structure = tree_util.tree_flatten(raw)
        active = tuple(value for value, flag in zip(raw_leaves, activity.flags) if flag)

        def active_raw(values):
            iterator = iter(values)
            return tree_util.tree_unflatten(raw_structure,
                [next(iterator) if flag else value for value, flag in zip(raw_leaves, activity.flags)])

        backward_prepare, backward_rhs = prepare_reverse_rhs(
            prepare_rhs, orbit_rhs, t0, initial, raw[0])
        rp, rhs_setup_pullback = jax.vjp(
            lambda values: backward_prepare(active_raw(values)[0]), active)
        rhs_flags=_prepared_activity(lambda values:backward_prepare(active_raw(values)[0]),active)
        original_rhs=backward_rhs
        backward_rhs=lambda t,y,p:original_rhs(t,y,_freeze_fixed(p,rhs_flags))
        linearization = prepare_rhs_linearization(backward_rhs, t0, initial, rp)
        if linearization is not None:
            linearize, rhs_pullback, initial_cache = linearization
        else:
            initial_cache = ()
        csp, collision_setup_pullback = jax.vjp(
            lambda values:prepare_collision_state(active_raw(values)[1]),active)
        collision_flags=_prepared_activity(lambda values:prepare_collision_state(active_raw(values)[1]),active)
        backward_collision_state=lambda arguments,p:collision_state(arguments,_freeze_fixed(p,collision_flags))
        ts = controls[1]
        saved_bar = jnp.zeros((len(ts), initial.size), initial.dtype) if isinstance(bar.ys, SymbolicZero) else bar.ys
        terminal_bar = jnp.zeros_like(initial) if isinstance(bar.y, SymbolicZero) else bar.y
        terminal_bar += jnp.sum(jnp.where((jnp.arange(len(ts)) >= tape.saved_count)[:, None], saved_bar, 0.), axis=0)

        def reverse_map(carry, initial):
            index, by, bp, post = carry
            index -= 1
            offset = tape.checkpoint_offsets[index]
            # The next map's initial state is this map's post-collision state.
            # Restore only changed components to obtain the pre-collision state.
            orbit_end = (tape.orbit_endpoints[index] if collision_indices is None else
                post.at[jnp.asarray(collision_indices)].set(tape.orbit_endpoints[index]))
            count = (workspace+inner_segment-1)//inner_segment
            rows = offset+jnp.arange(count)
            # Padded rows are never read by the bounded inner reverse loop.
            if collision_indices is None:
                checkpoints = (tape.orbit_states[rows],tape.orbit_derivatives[rows])
            else:
                # One first derivative per preceding interval is omitted.
                derivative_rows = offset-index+jnp.arange(max(count-1,0))
                checkpoints = (tape.orbit_states[rows],tape.orbit_derivatives[derivative_rows])
            t, end = tape.times[index]
            by += jnp.sum(jnp.where(((ts > t) & (ts <= end))[:, None], saved_bar, 0.), axis=0)

            def kick_reverse(values):
                by, bp = values
                _, pullback = jax.vjp(lambda y, values: backward_collision_state(
                    (end, y, end-t, tape.increments[index]),values),orbit_end,csp)
                dy, dp = pullback(by)
                return dy, (bp[0],_add(bp[1],dp))

            by, bp = lax.cond(tape.kicks[index], kick_reverse, lambda values: values, (by, bp))
            orbit_start, orbit_stop = tape.offsets[index], tape.offsets[index+1]
            states, derivatives = checkpoints
            # Each collision starts a new orbit: its initial FSAL cache must
            # use this map's state, not the trajectory's original birth state.
            if collision_indices is None:
                map_initial_cache = (linearize(tape.orbit_times[orbit_start, 0], initial)[1]
                                     if linearization is not None else ())
            elif linearization is not None:
                map_initial_value, map_initial_cache = linearize(tape.orbit_times[orbit_start, 0], initial)
            else:
                map_initial_value = backward_rhs(tape.orbit_times[orbit_start, 0], initial, rp)
                map_initial_cache = ()

            def reverse_orbit_segment(local):
                stop, by, bf, bp = local
                segment = jnp.maximum((stop-orbit_start-1) // inner_segment, 0)
                start = orbit_start + segment*inner_segment
                y = states[segment]
                if collision_indices is None:
                    f = derivatives[segment]
                else:
                    f = (jnp.where(segment == 0, map_initial_value, derivatives[jnp.maximum(segment-1,0)])
                         if derivatives.shape[0] else map_initial_value)
                state_bank = jnp.zeros((inner_segment, initial.size), initial.dtype)
                endpoint_bank = jnp.zeros_like(state_bank)
                increment_bank = jnp.zeros((inner_segment, len(constants[2]), initial.size), initial.dtype)
                cache_bank = tree_util.tree_map(lambda value: jnp.zeros(
                    (inner_segment, len(constants[2]))+value.shape, value.dtype), initial_cache)

                def reconstruct(values):
                    i, y, f, state_bank, endpoint_bank, increment_bank, cache_bank = values
                    slot = i-start
                    t, h = tape.orbit_times[i]
                    if linearization is None:
                        trial = _trial_step(backward_rhs, t, y, h, rp, f, *constants)
                        residuals = ()
                    else:
                        trial, residuals = _trial_with_residuals(
                            linearize, t, y, h, f, constants, map_initial_cache)
                    cache_bank = tree_util.tree_map(
                        lambda buffer,value: buffer.at[slot].set(value), cache_bank, residuals)
                    return (i+1, trial.y, trial.f1, state_bank.at[slot].set(y),
                        endpoint_bank.at[slot].set(trial.y), increment_bank.at[slot].set(trial.increments),
                        cache_bank)

                _, _, _, state_bank, endpoint_bank, increment_bank, cache_bank = lax.while_loop(
                    lambda values: values[0] < stop, reconstruct,
                    (start, y, f, state_bank, endpoint_bank, increment_bank, cache_bank))

                def reverse_step(values):
                    i, by, bf, dp = values
                    i -= 1
                    slot = i-start
                    t, h = tape.orbit_times[i]
                    fraction = jnp.where(i == orbit_stop-1, tape.fractions[index], 1.)
                    def cached_pullback(stage, seed):
                        return rhs_pullback(tree_util.tree_map(
                            lambda buffer: buffer[slot, stage], cache_bank), seed)
                    dy, df, contribution = _step_pullback(backward_rhs, state_bank[slot], t, h,
                        endpoint_bank[slot], increment_bank[slot], rp, fraction,
                        jnp.empty((0,), initial.dtype), jnp.empty((0,), bool),
                        by, bf, jnp.empty((0, initial.size), initial.dtype), method,
                        initial_fsal=(i == orbit_start),
                        cached_pullback=cached_pullback if linearization is not None else None)
                    return i, dy, df, _add(dp, contribution)

                # Share the orbit-only segment policy: transpose parameter
                # preparation once after summing the segment's prepared bars.
                i, by, bf, dp = lax.while_loop(lambda values: values[0] > start, reverse_step,
                    (stop, by, bf, _zeros(rp)))
                return i, by, bf, (_add(bp[0],rhs_setup_pullback(dp)[0]),bp[1])

            _, by, _, bp = lax.while_loop(lambda values: values[0] > orbit_start,
                reverse_orbit_segment, (orbit_stop, by, jnp.zeros_like(initial), bp))
            return index, by, bp, initial if collision_indices is not None else None

        def reverse(carry):
            stop = carry[0]
            segment = jnp.maximum((stop-1) // outer_segment, 0)
            start = segment*outer_segment
            def reverse_recorded(values):
                index = values[0]-1
                initial = (jnp.where(tape.offsets[index+1]>tape.offsets[index],
                    tape.orbit_states[tape.checkpoint_offsets[index]],tape.orbit_endpoints[index])
                    if collision_indices is None else tape.orbit_states[tape.checkpoint_offsets[index]])
                return reverse_map(values,initial)
            return lax.while_loop(lambda values: values[0]>start,reverse_recorded,carry)

        _, by, bp, _ = lax.while_loop(lambda values: values[0] > 0, reverse,
            (tape.count, terminal_bar, (_zeros(active),_zeros(csp)), final))
        has_kick = jnp.any((jnp.arange(tape.kicks.size) < tape.count) & tape.kicks)
        collision_bar = lax.cond(has_kick,lambda _:collision_setup_pullback(bp[1])[0],
            lambda _:_zeros(active),None)
        bp = _add(bp[0],collision_bar)
        by += jnp.sum(jnp.where((jnp.arange(len(ts)) < tape.initial_saves)[:, None], saved_bar, 0.), axis=0)
        iterator = iter(bp)
        raw_bar = tree_util.tree_unflatten(raw_structure,
            [next(iterator) if flag else _zeros(value) for value, flag in zip(raw_leaves, activity.flags)])
        # Independent GPU preparation is frozen controller/forward data.
        # A symbolic zero prunes its entire setup transpose. Materialized
        # zero arrays can otherwise retain a second coil-geometry pullback.
        return by, raw_bar, None if forward_override is not None else _zeros(controls)

    integrate.defvjp(fwd, bwd, symbolic_zeros=True)
    return integrate(y0, raw, controls)


class _OrbitRecorder:
    """Sparse states/FSAL plus accepted time sizes; never record a rejection."""
    def __init__(self, capacity, segment, schedule=False):
        self.capacity, self.segment, self.schedule = capacity, segment, schedule
        _validate_ad_sizes(capacity=capacity, segment=segment)

    def initialize(self, state):
        y = state['y']; nodes = (self.capacity+self.segment-1)//self.segment
        state.update(ad_states=jnp.zeros((nodes,y.size),y.dtype),
            ad_derivatives=jnp.zeros((nodes,y.size),y.dtype),
            ad_times=jnp.zeros((nodes,),y.dtype), ad_steps=jnp.zeros((self.capacity,),y.dtype),
            ad_fraction=jnp.ones((),y.dtype))
        if self.schedule:
            state['ad_schedule']=jnp.zeros((self.capacity,2),y.dtype)
        return state

    def record(self, previous, updated, t, y, h, fraction, accept):
        i=previous['accepted']
        def store(state):
            def checkpoint(state):
                slot=i//self.segment
                return dict(state,ad_states=state['ad_states'].at[slot].set(y),
                    ad_derivatives=state['ad_derivatives'].at[slot].set(previous['f']),
                    ad_times=state['ad_times'].at[slot].set(t))
            state=lax.cond(i%self.segment==0,checkpoint,lambda s:s,state)
            state=dict(state,ad_steps=state['ad_steps'].at[i].set(h),ad_fraction=fraction)
            if self.schedule:
                state['ad_schedule']=state['ad_schedule'].at[i].set(jnp.stack([t,h]))
            return state
        return lax.cond(accept&(i<self.capacity),store,lambda s:s,updated)

    def finish(self, state, solution, initial_saves):
        if self.schedule:
            return solution,_RKSchedule(state['ad_schedule'],state['ad_fraction'],
                state['accepted'],state['ad_states'],state['ad_derivatives'])
        return solution,_RKCheckpointTape(state['ad_steps'],state['ad_times'],
            state['ad_states'],state['ad_derivatives'],state['ad_fraction'],
            state['t'],initial_saves,state['accepted'],solution.status)

def _validate_ad_sizes(**values):
    for name,value in values.items():
        if value is not None and (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<=0):
            raise ValueError(name+' must be a positive integer or None')

class _SplitRecorder:
    """Flat accepted inner schedules, independent retained noise and sparse nodes."""
    @classmethod
    def from_options(cls, initial, options):
        result=cls()
        result.options=options
        result.y0=initial
        _validate_ad_sizes(**{k:v for k,v in options.items() if k.startswith('adjoint_')})
        return result

    def initialize(self,s):
        y0=self.y0; o=self.options
        max_steps=o.get('max_steps',100000); noise_dim=o['noise_dim']
        adjoint_max_accepted_steps=o.get('adjoint_max_accepted_steps')
        adjoint_max_orbit_steps=o.get('adjoint_max_orbit_steps')
        adjoint_checkpoint_steps=o.get('adjoint_checkpoint_steps',16)
        adjoint_orbit_checkpoint_steps=o.get('adjoint_orbit_checkpoint_steps',16)
        _collision_state_indices=o.get('_collision_state_indices')
        outer_capacity = min(max_steps, adjoint_max_accepted_steps or max_steps)
        inner_capacity = adjoint_max_orbit_steps or max_steps
        checkpoints = (outer_capacity + adjoint_checkpoint_steps - 1) // adjoint_checkpoint_steps
        inner_segment = adjoint_orbit_checkpoint_steps
        # Sum ceil(n_map / inner_segment) <= ceil(total / inner_segment) + maps - 1.
        # Only committed maps consume this flat bank; rejected outer trials
        # retain their original Brownian bookkeeping and discard trial nodes.
        if _collision_state_indices is None:
            node_capacity = (inner_capacity+inner_segment-1)//inner_segment+outer_capacity-1
        else:
            # A nonempty map needs at least one inner step. This bounds the
            # total checkpoints without reserving impossible map/step pairs.
            nonempty_maps = min(outer_capacity, inner_capacity)
            node_capacity = nonempty_maps+(inner_capacity-nonempty_maps)//inner_segment
        endpoint_size = y0.size if _collision_state_indices is None else len(_collision_state_indices)
        derivative_capacity = (node_capacity if _collision_state_indices is None else
                               (inner_capacity-1)//inner_segment)
        s.update(ad_checkpoint_offsets=jnp.zeros((outer_capacity+1,), jnp.int32),
            ad_orbit_states=jnp.zeros((node_capacity, y0.size), y0.dtype),
            ad_orbit_derivatives=jnp.zeros((derivative_capacity, y0.size), y0.dtype),
            ad_orbit_endpoints=jnp.zeros((outer_capacity, endpoint_size), y0.dtype),
            ad_checkpoint_count=jnp.array(0, jnp.int32),
            ad_times=jnp.zeros((outer_capacity, 2), y0.dtype),
            ad_dw=jnp.zeros((outer_capacity, noise_dim), y0.dtype),
            ad_kicks=jnp.zeros((outer_capacity,), bool),
            ad_offsets=jnp.zeros((outer_capacity+1,), jnp.int32),
            ad_orbit_times=jnp.zeros((inner_capacity, 2), y0.dtype),
            ad_fractions=jnp.ones((outer_capacity,), y0.dtype),
            ad_states=jnp.zeros((checkpoints, y0.size), y0.dtype),
            ad_count=jnp.array(0), ad_orbit_count=jnp.array(0))
        return s

    def record(self,s,state,accepted,orbit_event,orbit,schedule,dw):
        o=self.options
        outer_capacity=min(o.get('max_steps',100000),o.get('adjoint_max_accepted_steps') or o.get('max_steps',100000))
        inner_capacity=o.get('adjoint_max_orbit_steps') or o.get('max_steps',100000)
        inner_segment=o.get('adjoint_orbit_checkpoint_steps',16)
        adjoint_checkpoint_steps=o.get('adjoint_checkpoint_steps',16)
        _collision_state_indices=o.get('_collision_state_indices')
        committed = accepted | orbit_event
        i, k = s["ad_count"], s["ad_orbit_count"]
        def record(state):
            def checkpoint(states):
                return states.at[i // adjoint_checkpoint_steps].set(s["y"])
            states = lax.cond(i % adjoint_checkpoint_steps == 0,
                checkpoint, lambda states: states, state["ad_states"])
            orbit_times = lax.fori_loop(0, schedule.count,
                lambda j, rows: rows.at[k+j].set(schedule.times[j]), state["ad_orbit_times"])
            node_start = s["ad_checkpoint_count"]
            node_count = ((schedule.count+inner_segment-1)//inner_segment).astype(node_start.dtype)
            if _collision_state_indices is None:
                def node(j, buffers):
                    states, derivatives = buffers
                    return (states.at[node_start+j].set(schedule.states[j]),
                        derivatives.at[node_start+j].set(schedule.derivatives[j]))
                nodes, derivatives = lax.fori_loop(0,node_count,node,
                    (state["ad_orbit_states"],state["ad_orbit_derivatives"]))
            else:
                nodes = lax.fori_loop(0,node_count,
                    lambda j, bank:bank.at[node_start+j].set(schedule.states[j]),
                    state["ad_orbit_states"])
                # Every committed interval contains at least one inner
                # step. Removing its first derivative leaves this flat
                # offset; no second offsets array is needed.
                derivative_start = node_start-i
                derivative_count = jnp.maximum(node_count-1,0)
                derivatives = state["ad_orbit_derivatives"]
                if derivatives.shape[0]:
                    derivatives = lax.fori_loop(0,derivative_count,
                        lambda j, bank:bank.at[derivative_start+j].set(schedule.derivatives[j+1]),
                        derivatives)
            return dict(state,
                ad_orbit_states=nodes, ad_orbit_derivatives=derivatives,
                ad_checkpoint_offsets=state["ad_checkpoint_offsets"].at[i+1].set(node_start+node_count),
                ad_checkpoint_count=node_start+node_count,
                ad_orbit_endpoints=state["ad_orbit_endpoints"].at[i].set(
                    orbit.y if _collision_state_indices is None else
                    orbit.y[jnp.asarray(_collision_state_indices)]),
                ad_times=state["ad_times"].at[i].set(jnp.stack([s["t"], state["t"]])),
                ad_dw=state["ad_dw"].at[i].set(dw),
                ad_kicks=state["ad_kicks"].at[i].set(accepted),
                ad_offsets=state["ad_offsets"].at[i+1].set((k+schedule.count).astype(state["ad_offsets"].dtype)),
                ad_orbit_times=orbit_times,
                ad_fractions=state["ad_fractions"].at[i].set(schedule.fraction),
                ad_states=states)
        can_record = committed & (i < outer_capacity) & (k+schedule.count <= inner_capacity)
        state = lax.cond(can_record, record, lambda state: state, state)
        state = dict(state, ad_count=i+committed.astype(i.dtype),
            ad_orbit_count=k+jnp.where(committed, schedule.count, 0))
        return state

    def finish(self,s,solution,initial_saves):
        return solution,_SplitCheckpointTape(s['ad_times'],s['ad_dw'],s['ad_kicks'],
            s['ad_offsets'],s['ad_orbit_times'],s['ad_fractions'],s['ad_states'],
            s['ad_count'],s['ad_orbit_count'],initial_saves,s['index'],solution.status,
            s['ad_checkpoint_offsets'],s['ad_orbit_states'],s['ad_orbit_derivatives'],
            s['ad_orbit_endpoints'],s['ad_checkpoint_count'])


class _TapeView:
    """Static scalar/row view into one packed device tape allocation."""
    def __init__(self,ref,offset,shape):
        self.ref,self.offset,self.shape=ref,offset,shape

    def __setitem__(self,index,value):
        if index is Ellipsis:
            positions=self.offset+jnp.arange(int(np.prod(self.shape))).reshape(self.shape) if self.shape else self.offset
        elif isinstance(index,tuple):
            row,column=index
            positions=self.offset+row*self.shape[1]+(jnp.arange(self.shape[1]) if isinstance(column,slice) else column)
        else:
            positions=self.offset+index
        self.ref[positions]=jnp.asarray(value,self.ref.dtype)


# Independent direct-coil reverse: exact field transposes, compact stage caches.
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from .constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q
from .rk import _DENSE8, _cross as cross, _dot as dot, _coil_field_values as field_values, _pack as pack
from .split import (_evaluate_kernel, scalar,
    _coefficient_graph as coefficient_graph, CollisionCoefficients,
    guiding_center_collision_step, _diffusion_root,_register_kernel_placement)
from .split import guiding_center_spatial_diffusion


def flatten(function, *examples):
    graph=jax.make_jaxpr(function)(*examples)
    return jax.make_jaxpr(lambda *args:_evaluate_inline(graph.jaxpr,graph.consts,*args))(*examples)

def evaluate(graph, constants, *arguments, **options):
    values=_evaluate_kernel(graph,constants,*arguments,**options)
    # DCE can expose a constant zero geometry bar (Lorentz collision kicks
    # have no direct field dependence). Materialize it within the kernel,
    # rather than allowing a NumPy array to become a captured device constant.
    result=[]
    for value in values:
        if not isinstance(value,jax.core.Tracer) and np.ndim(value):
            data=np.asarray(value); flat=data.reshape(-1)
            value=jnp.full(data.shape,flat[0].item(),data.dtype)
            indices=jnp.arange(data.size).reshape(data.shape)
            for i,item in enumerate(flat):
                if item!=flat[0]:value=jnp.where(indices==i,item.item(),value)
        result.append(value)
    return tuple(result)

def _inverse_powers(radius2):
    """Share one float64 reciprocal radius across the inverse powers."""
    inverse=jax.lax.rsqrt(radius2)
    inverse2=inverse*inverse
    inverse3=inverse*inverse2
    inverse5=inverse3*inverse2
    return inverse3,inverse5,inverse5*inverse2


def _reverse_field_values(values,point,nc,nq,jacobian):
    """Same direct quadrature, sharing inverse powers in backward replay."""
    gamma,tangent=tuple(ref[...] for ref in values[:3]),tuple(ref[...] for ref in values[3:])
    pc=1<<(nc-1).bit_length()
    index=jnp.arange(gamma[0].size)
    mask=(index//pc<nq)&(index%pc<nc)
    r=tuple(point[i]-gamma[i] for i in range(3))
    radius2=jnp.where(mask,dot(r,r),1.)
    inverse3,inverse5,_=_inverse_powers(radius2)
    c=cross(tangent,r)
    mean=lambda value:jnp.sum(jnp.where(mask,value,0.))/nq
    B=tuple(mean(value*inverse3) for value in c)
    if not jacobian:
        return B,None
    zero=jnp.zeros_like(inverse3)
    tx,ty,tz=tangent
    skew=((zero,-tz,ty),(tz,zero,-tx),(-ty,tx,zero))
    J=tuple(tuple(mean(skew[i][j]*inverse3-3*c[i]*r[j]*inverse5) for j in range(3)) for i in range(3))
    return B,J



def _rhs_from_field(y, B, J, gc):
    """Small physical algebra, separated from the field-sized reduction."""
    zero = jnp.zeros_like(y[0])
    if not gc:
        return y[3:6]+tuple(Q/M*v for v in cross(y[3:6], B))+(zero, zero)
    magnitude = jnp.sqrt(dot(B, B))
    gradient = tuple(sum(J[i][j]*B[i]/magnitude for i in range(3)) for j in range(3))
    curl = J[2][1]-J[1][2], J[0][2]-J[2][0], J[1][0]-J[0][1]
    bg = cross(B, gradient)
    curl_b = tuple(curl[i]/magnitude+bg[i]/magnitude**2 for i in range(3))
    curvature = tuple(-v/magnitude for v in cross(B, curl_b))
    vpar, moment = y[3:5]
    star = tuple(B[i]+vpar*M/Q*curl_b[i] for i in range(3))
    force = tuple(moment*gradient[i]+M*vpar**2*curvature[i] for i in range(3))
    denominator = dot(B, star)
    perpendicular = cross(B, force)
    velocity = tuple(vpar*B[i]/magnitude+perpendicular[i]/denominator/Q for i in range(3))
    acceleration = -dot(star, force)/denominator*magnitude/M
    return velocity+(acceleration, zero, zero, zero, zero)


def _field_transpose(geometry, point, seed_B, seed_J, nc, nq, gc):
    """Exact position/geometry transpose, including the GC Jacobian seed."""
    gamma, tangent = geometry[:3], geometry[3:]
    pc = 1 << (nc-1).bit_length()
    index = jnp.arange(gamma[0].size)
    mask = (index//pc<nq)&(index%pc<nc)
    r = tuple(point[i]-gamma[i] for i in range(3))
    radius2 = jnp.where(mask, dot(r,r), 1.)
    inverse3,inverse5,inverse7 = _inverse_powers(radius2)
    c = cross(tangent,r)
    u = tuple(value/nq for value in seed_B)
    k = dot(u,c)
    dr = tuple(value*inverse3-3*k*r[i]*inverse5 for i,value in enumerate(cross(u,tangent)))
    dt = tuple(value*inverse3 for value in cross(r,u))
    if gc:
        U = tuple(tuple(value/nq for value in row) for row in seed_J)
        skew = U[2][1]-U[1][2],U[0][2]-U[2][0],U[1][0]-U[0][1]
        ur = tuple(sum(U[i][j]*r[j] for j in range(3)) for i in range(3))
        utc = tuple(sum(U[j][i]*c[j] for j in range(3)) for i in range(3))
        d = dot(c,ur)
        sk = dot(skew,tangent)
        extra = cross(ur,tangent)
        dl = cross(r,ur)
        dr = tuple(dr[i]-3*(sk*r[i]+extra[i]+utc[i])*inverse5+15*d*r[i]*inverse7 for i in range(3))
        dt = tuple(dt[i]+skew[i]*inverse3-3*dl[i]*inverse5 for i in range(3))
    dr = tuple(jnp.where(mask,value,0.) for value in dr)
    dt = tuple(jnp.where(mask,value,0.) for value in dt)
    return tuple(jnp.sum(value) for value in dr),tuple(-value for value in dr)+dt


def _kick(y, geometry, constants, floor, h, dw, graph, nc, nq, gc, spatial_diffusion=False):
    field = field_values(geometry, y[:3], nc, nq, False)[0] if gc else None
    B = jnp.sqrt(dot(field, field)) if gc else jnp.ones_like(y[0])
    speed = jnp.sqrt(y[3]**2+2*y[4]*B/M) if gc else jnp.sqrt(dot(y[3:6], y[3:6]))
    coefficients = tuple(jax.core.eval_jaxpr(graph.jaxpr, constants, speed))
    if gc:
        result = guiding_center_collision_step(speed, y[3]/speed, h, jnp.stack(dw[-2:]),
            CollisionCoefficients(*coefficients), speed_floor=floor).y
        v, xi = result[0], result[1]
        point=y[:3]
        if spatial_diffusion:
            pitch=y[3]/speed
            dx=guiding_center_spatial_diffusion(pitch,CollisionCoefficients(*coefficients),Q*B/M)
            unit=tuple(v/B for v in field)
            parallel=sum(a*b for a,b in zip(unit,dw[:3]))
            delta=tuple(_diffusion_root(2*dx)*(dw[i]-unit[i]*parallel) for i in range(3))
            point=tuple(point[i]+delta[i] for i in range(3))
            newfield=field_values(geometry,point,nc,nq,False)[0]
            B=jnp.sqrt(dot(newfield,newfield))
        return point+(v*xi, M*v*v*(1-xi*xi)/(2*B))+y[5:]
    dp, dt, ddp, drag = coefficients[:4]
    drift = drag+ddp+2*(dp-dt)/speed
    delta = (_diffusion_root(2*dt)*dw[0], _diffusion_root(2*dt)*dw[1],
        drift*h+_diffusion_root(2*dp)*dw[2]+.5*ddp*(dw[2]**2-h))
    direction = tuple(v/speed for v in y[3:6])
    a = tuple(jnp.abs(v) for v in direction)
    i0 = (a[0]<=a[1]) & (a[0]<=a[2])
    i1 = ~i0 & (a[1]<=a[2])
    ref = tuple(v.astype(speed.dtype) for v in (i0, i1, ~i0 & ~i1))
    first = cross(direction, ref)
    length = jnp.sqrt(dot(first, first))
    first = tuple(v/length for v in first)
    second = cross(direction, first)
    velocity = tuple(y[3+i]+first[i]*delta[0]+second[i]*delta[1]+direction[i]*delta[2] for i in range(3))
    return y[:3]+velocity+y[6:]


def physical_graphs(geometry, species, floor, nc, nq, gc, *, active_coefficients=True, active_floor=True,
                    spatial_diffusion=False, coefficient_data=None):
    """Expose the exact primal and transpose graphs before kernel lowering."""
    if coefficient_data is None:
        coefficients = coefficient_graph(species, not gc)
    else:
        from jax.extend.core import ClosedJaxpr
        coefficients = ClosedJaxpr(*coefficient_data)
    constants = tuple(coefficients.consts)
    y = tuple(jnp.asarray(v, jnp.float64) for v in
        ((1.23, 0., 0., 2e5, 1e-15, 0., 0., 0.) if gc else
         (1.23, 0., 0., 1e5, 2e5, 3e5, 0., 0.)))
    seed = tuple(jnp.ones_like(v) for v in y)
    def rhs(y,values):
        B,J = _reverse_field_values(values,y[:3],nc,nq,gc)
        cache = B+tuple(value for row in J for value in row) if gc else B
        return _rhs_from_field(y,B,J,gc)+cache
    def trace(function, *examples):
        from jax._src.interpreters.partial_eval import dce_jaxpr
        from jax.extend.core import ClosedJaxpr
        leaves, structure = tree_util.tree_flatten(examples)
        graph = flatten(lambda *args: function(*tree_util.tree_unflatten(structure,args)), *leaves)
        pruned, _ = dce_jaxpr(graph.jaxpr, [True]*len(graph.out_avals), instantiate=True)
        return ClosedJaxpr(pruned, graph.consts)
    rhs_graph = trace(rhs, y, geometry)
    cache = tuple(jnp.asarray(1.) for _ in range(12 if gc else 3))
    def rhs_transpose(y,values,bar,cache):
        if not gc:
            B=cache[:3]
            # For acceleration k*(v x B), bB=k*(ba x v) and
            # bv=bx+k*(B x ba). Field sensitivities then use the
            # same exact point/geometry transpose as GC.
            u=tuple(Q/M*v for v in cross(bar[3:6],y[3:6]))
            point_bar,geometry_bar=_field_transpose(values,y[:3],u,None,nc,nq,False)
            velocity_bar=tuple(bar[i]+Q/M*v for i,v in enumerate(cross(B,bar[3:6])))
            zero=jnp.zeros_like(y[0])
            return point_bar+velocity_bar+(zero,zero),geometry_bar
        B = cache[:3]
        J = tuple(tuple(cache[3+3*i+j] for j in range(3)) for i in range(3))
        by,(bB,bJ) = jax.vjp(lambda y,field:_rhs_from_field(y,*field,gc),y,(B,J))[1](bar)
        point_bar,geometry_bar = _field_transpose(values,y[:3],bB,bJ,nc,nq,gc)
        return tuple(by[i]+point_bar[i] if i<3 else by[i] for i in range(8)),geometry_bar
    rhs_pullback = trace(rhs_transpose, y, geometry, seed, cache)
    kick = lambda y, values, c, f, h, dw: _kick(y, values, c, f, h, dw, coefficients, nc, nq, gc,spatial_diffusion)
    noise = tuple(jnp.asarray(.0001) for _ in range((5 if spatial_diffusion else 2) if gc else 3))
    def kick_transpose(y, values, c, f, h, dw, bar):
        by, bg, bc, bf = jax.vjp(lambda y, values, c, f: kick(y, values, c, f, h, dw),
                                y, values, c, f)[1](bar)
        # Hoisted physical branch flags have exact zero (float0) cotangents;
        # use numeric scratch slots, restoring float0 at the custom-VJP boundary.
        bc = tuple(jnp.zeros(value.shape,jnp.float64) if b.dtype==jax.dtypes.float0 else b
                   for value,b in zip(c,bc))
        return by, bg, bc if active_coefficients else tree_util.tree_map(jnp.zeros_like,bc), bf if active_floor else jnp.zeros_like(f)
    kick_pullback = trace(kick_transpose, y, geometry, constants, floor,
                            jnp.asarray(1e-8), noise, seed)
    return rhs_graph, rhs_pullback, kick_pullback, constants


def reverse_kernel(terminal_state, parameters, tape, saved_bar, terminal_bar, times,
                   graphs, *, gc, segment_steps=16, warps=None, spatial_diffusion=False, interpret=False):
    """Replay and transpose one particle entirely inside a persistent kernel."""
    if not interpret:_register_kernel_placement()
    rhs_graph, rhs_pb, kick_pb = graphs
    warps = (2 if gc else 4) if warps is None else warps
    geometry, coefficients, floor = parameters
    nstate, noise_dim = (5, 5 if spatial_diffusion else 2) if gc else (6, 3)
    nsave, ncoeff = len(times), len(coefficients)
    rows = np.zeros((16, 16))
    for i, row in enumerate(_A8, 1):
        rows[i, :len(row)] = row
    dense = np.zeros((len(_DENSE8[0]), 16))
    dense[:, :len(_ERROR8)] = np.asarray(_DENSE8).T
    nstage = len(_ERROR8)
    cache_size, cache_width = (12,16) if gc else (3,4)
    scratch_rows = 1 << (segment_steps-1).bit_length()
    field_size = geometry[0].size
    scalar_shape = lambda shape: jax.ShapeDtypeStruct(shape, jnp.float64)
    nfield=len(geometry)
    out_shapes = (scalar_shape((8,)),scalar_shape((nfield,field_size)),
                  scalar_shape((1 << (ncoeff-1).bit_length(),)),scalar_shape(()),
                  scalar_shape((scratch_rows,8)),scalar_shape((scratch_rows,16,8)),
                  scalar_shape((scratch_rows,16,cache_width)))
    graph_constants = tuple(rhs_graph.consts)+tuple(rhs_pb.consts)+tuple(kick_pb.consts)
    const_counts = len(rhs_graph.consts), len(rhs_pb.consts), len(kick_pb.consts)
    # Only these tape fields are needed by reverse; no noise is generated.
    fields = ('times', 'increments', 'kicks', 'offsets', 'orbit_times', 'fractions',
              'orbit_states', 'orbit_endpoints', 'checkpoint_offsets',
              'count', 'saved_count', 'initial_saves')
    tape_values = tuple(getattr(tape, name) for name in fields)

    def kernel(*refs):
        geometries = refs[:nfield]
        coeffrefs = refs[nfield:nfield+ncoeff]
        floorref, yref, byref, bref, tsref, aref, dref = refs[nfield+ncoeff:nfield+ncoeff+7]
        start = nfield+ncoeff+7
        tref, dwref, kicks, offsets, otime, fractions, states, endpoints, nodes, countref, saved_count, initial_saves = refs[start:start+12]
        start += 12
        c0, c1, c2 = const_counts
        rhs_constants = tuple(ref[...] for ref in refs[start:start+c0])
        rhs_pb_constants = tuple(ref[...] for ref in refs[start+c0:start+c0+c1])
        kick_constants = tuple(ref[...] for ref in refs[start+c0+c1:start+c0+c1+c2])
        bout, gpout, cpout, fpout, state_scratch, stage_scratch, field_scratch = refs[-7:]
        geom = tuple(ref[...] for ref in geometries)
        cvalues = tuple(ref[...].reshape(()) for ref in coeffrefs)
        floor_value = floorref[...]
        zero = jnp.asarray(0., jnp.float64)
        zero8 = jnp.zeros((8,), jnp.float64)
        at = lambda y: tuple(scalar(y, i) for i in range(8))
        def rhs(y):
            values = evaluate(rhs_graph.jaxpr, rhs_constants, *at(y), *geom)
            return pack(values[:8]),pack(values[8:]+(zero,)*(cache_width-cache_size))
        def rhs_transpose(y, seed, cache):
            values = evaluate(rhs_pb.jaxpr, rhs_pb_constants, *at(y), *geom, *at(seed),
                              *(scalar(cache,i) for i in range(cache_size)))
            return pack(values[:8]), tuple(values[8:])
        def read_state(ref, row):
            return pack(tuple(ref[row, i] for i in range(nstate))+(zero,)*(8-nstate))
        def saved(i):
            return pack(tuple(byref[i, j] for j in range(nstate))+(zero,)*(8-nstate))
        by = pack(tuple(bref[j] for j in range(nstate))+(zero,)*(8-nstate))
        for i in range(nsave):
            by += jnp.where(i>=saved_count[...], saved(i), zero8)
        gp = tuple(jnp.zeros_like(value) for value in geom)
        cp = jnp.zeros((1 << (ncoeff-1).bit_length(),), jnp.float64)
        fp = zero
        row_indices = jnp.arange(16)[:, None]
        def reverse_map(carry):
            index, by, gp, cp, fp, post = carry
            index -= 1
            t, end = tref[index, 0], tref[index, 1]
            for i in range(nsave):
                by += jnp.where((tsref[i]>t)&(tsref[i]<=end), saved(i), zero8)
            offset = nodes[index]
            map_initial = read_state(states, offset)
            endpoint = (read_state(endpoints, index) if gc else
                pack(tuple(scalar(post, i) for i in range(3))+
                     tuple(endpoints[index, i] for i in range(3))+(zero, zero)))
            def kick_reverse(values):
                by, gp, cp, fp = values
                dw = tuple(dwref[index, j] for j in range(noise_dim))
                result = evaluate(kick_pb.jaxpr, kick_constants, *at(endpoint), *geom,
                                  *cvalues, floor_value, end-t, *dw, *at(by), scalarize=True)
                by = pack(result[:8])
                gp = tuple(a+b for a, b in zip(gp, result[8:8+nfield]))
                first_constant=8+nfield
                cp += pack(tuple(jnp.asarray(v).reshape(()) for v in result[first_constant:first_constant+ncoeff])
                           +(zero,)*(cp.shape[0]-ncoeff))
                fp += result[first_constant+ncoeff].reshape(())
                return by, gp, cp, fp
            by, gp, cp, fp = jax.lax.cond(kicks[index], kick_reverse, lambda values: values, (by, gp, cp, fp))
            orbit_start, orbit_stop = offsets[index], offsets[index+1]
            def reverse_segment(local):
                stop, by, bf, gp = local
                segment = (stop-orbit_start-1)//segment_steps
                start = orbit_start+segment*segment_steps
                y = read_state(states, offset+segment)
                f, cache = rhs(y)
                def reconstruct(values):
                    i, y, f, cache = values
                    h = otime[i, 1]
                    bank = jnp.where(row_indices==0, h*f[None, :], jnp.zeros((16, 8), jnp.float64))
                    cache_bank = jnp.where(row_indices==0, cache[None,:], jnp.zeros((16,cache_width),jnp.float64))
                    def stage(j, values):
                        bank, f, cache_bank, cache = values
                        point = y+jnp.sum(jnp.where(row_indices<j, aref[j, :][:, None]*bank, 0.), axis=0)
                        # The final FSAL derivative has no consumer at an
                        # ordinary outer endpoint. Event dense output still
                        # requires it. Drop only that provably unused call.
                        needed = (j!=nstage-1)|(i!=orbit_stop-1)|(fractions[index]!=1)
                        f, cache = jax.lax.cond(needed, rhs,
                            lambda _: (zero8,jnp.zeros((cache_width,),jnp.float64)),point)
                        bank = jnp.where(row_indices==j, (h*f)[None, :], bank)
                        cache_bank = jnp.where(row_indices==j, cache[None,:], cache_bank)
                        return bank, f, cache_bank, cache
                    bank, f, cache_bank, cache = jax.lax.fori_loop(1, nstage, stage, (bank, f, cache_bank, cache))
                    endpoint = y+jnp.sum(jnp.where(row_indices<nstage-1, aref[nstage-1, :][:, None]*bank, 0.), axis=0)
                    state_scratch[i-start, :] = y
                    stage_scratch[i-start, :, :] = bank
                    field_scratch[i-start,:,:] = cache_bank
                    return i+1, endpoint, f, cache
                jax.lax.while_loop(lambda values: values[0]<stop, reconstruct, (start, y, f, cache))
                def reverse_step(values):
                    i, by, bf, gp = values
                    i -= 1
                    y, bank = state_scratch[i-start, :], stage_scratch[i-start, :, :]
                    h = otime[i, 1]
                    fraction = jnp.where(i==orbit_stop-1, fractions[index], 1.)
                    weights = jnp.zeros((16,), jnp.float64)
                    for power in range(dense.shape[0]):
                        weights = weights*fraction+dref[power, :]
                    weights *= fraction
                    weights = jnp.where(fraction==1, aref[nstage-1, :], jnp.where(fraction==0, 0., weights))
                    dk, dy = weights[:, None]*by[None, :], by
                    last_stage = nstage-1-((i==orbit_stop-1)&(fraction==1)).astype(jnp.int32)
                    def transpose_stage(reverse_index, values):
                        dy, dk, gp = values
                        j = last_stage-reverse_index
                        point = y+jnp.sum(jnp.where(row_indices<j, aref[j, :][:, None]*bank, 0.), axis=0)
                        seed = h*jnp.sum(jnp.where(row_indices==j, dk, 0.), axis=0)+jnp.where(j==nstage-1, bf, zero8)
                        cache = field_scratch[i-start,j,:]
                        dpoint, delta = rhs_transpose(point, seed, cache)
                        return dy+dpoint, dk+aref[j, :][:, None]*dpoint[None, :], tuple(a+b for a, b in zip(gp, delta))
                    dy, dk, gp = jax.lax.fori_loop(0, last_stage, transpose_stage, (dy, dk, gp))
                    df = h*jnp.sum(jnp.where(row_indices==0, dk, 0.), axis=0)
                    def first(values):
                        dy, gp = values
                        cache = field_scratch[i-start,0,:]
                        delta_y, delta = rhs_transpose(y, df, cache)
                        return dy+delta_y, tuple(a+b for a, b in zip(gp, delta))
                    dy, gp = jax.lax.cond(i==orbit_start, first, lambda values: values, (dy, gp))
                    return i, dy, jnp.where(i==orbit_start, zero8, df), gp
                return jax.lax.while_loop(lambda values: values[0]>start, reverse_step, (stop, by, bf, gp))
            _, by, _, gp = jax.lax.while_loop(lambda values: values[0]>orbit_start,
                reverse_segment, (orbit_stop, by, zero8, gp))
            return index, by, gp, cp, fp, map_initial
        post = yref[...]
        _, by, gp, cp, fp, _ = jax.lax.while_loop(lambda values: values[0]>0, reverse_map,
            (countref[...], by, gp, cp, fp, post))
        for i in range(nsave):
            by += jnp.where(i<initial_saves[...], saved(i), zero8)
        bout[...], cpout[...], fpout[...] = by, cp, fp
        for i, value in enumerate(gp):
            gpout[i, :] = value

    final_padding=(0,8-terminal_state.size)
    inputs=(*geometry,*coefficients,floor,jnp.pad(terminal_state,final_padding),
            saved_bar, terminal_bar, times, jnp.asarray(rows), jnp.asarray(dense),
            *tape_values, *graph_constants)
    result = pl.pallas_call(kernel, out_shape=out_shapes, grid=(),
        compiler_params=plt.CompilerParams(num_warps=warps),interpret=interpret)(*inputs)
    return result[0][:nstate], tuple(result[1][i] for i in range(6)), tuple(
        result[2][i].reshape(value.shape).astype(value.dtype) if jnp.issubdtype(value.dtype,jnp.inexact)
        else jnp.zeros(value.shape,jax.dtypes.float0) for i, value in enumerate(coefficients)), result[3]


def solve_gpu_accepted(forward, initial, field, species, *, gc, speed_floor, options, arrays, spatial_diffusion):
    """Keep the current production forward and replay its physical accepted maps."""
    coefficient_model=coefficient_graph(species,not gc)
    coefficients=tuple(coefficient_model.consts)
    coefficient_body=coefficient_model.jaxpr
    nc,nq,_=field.coils.gamma.shape
    parameters=(tuple(arrays[:6]),coefficients,speed_floor)
    converted,driver_constants=_convert_all(lambda y,p:forward(y,record=True),initial,parameters)
    controls=driver_constants,jnp.asarray(options['ts'])
    @jax.custom_vjp
    def integrate(y,p,controls):
        return converted((y,p),controls[0])[0]
    def fwd(y,p,controls):
        activity=_ParameterActivity(tuple(v.perturbed for v in tree_util.tree_leaves(p)))
        y,p,controls=custom_vjp_primal_tree_values((y,p,controls))
        out,tape=converted((y,p),controls[0])
        return out,(p,tape,out.y,activity,controls[1])
    def bwd(residual,bar):
        p,tape,final,activity,times=residual
        tape=_checked_tape(tape,compact=True)
        saved=jnp.zeros((len(times),final.size),final.dtype) if isinstance(bar.ys,SymbolicZero) else bar.ys
        terminal=jnp.zeros_like(final) if isinstance(bar.y,SymbolicZero) else bar.y
        graphs=physical_graphs(p[0],species,p[2],nc,nq,gc,
            active_coefficients=any(activity.flags[6:-1]),active_floor=activity.flags[-1],
            spatial_diffusion=spatial_diffusion,coefficient_data=(coefficient_body,p[1]))
        by,bg,bc,bf=reverse_kernel(final,p,tape,saved,terminal,times,graphs[:3],gc=gc,
            segment_steps=options.get('adjoint_orbit_checkpoint_steps',16),spatial_diffusion=spatial_diffusion)
        return by,(bg,bc,bf),None
    integrate.defvjp(fwd,bwd,symbolic_zeros=True)
    return integrate(initial,parameters,controls)
