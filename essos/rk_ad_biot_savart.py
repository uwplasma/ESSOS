"""Native reverse-only transposes for stock prepared Biot-Savart B and B/J calls.

The original field call computes every primal value. Its transpose is an exact
algebraic derivative in position, geometry, current weights and all prepared
inputs. Other signatures/fields keep ordinary JAX AD. No field class changes.
"""
import jax
import jax.numpy as jnp
from jax import tree_util, ad_checkpoint
from jax.extend.core import Literal

from .rk import _bind_primitive



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


def _supported(closed):
    from .fields import BiotSavart
    body = closed.jaxpr
    original = getattr(BiotSavart.B_and_dB_by_dX, "__wrapped__", None)
    if original is None:
        return False
    source = original.__code__
    expected = f"B_and_dB_by_dX at {source.co_filename}:{source.co_firstlineno}"
    if getattr(getattr(body,"debug_info",None),"func_src_info",None) != expected or body.effects or len(body.invars) != 10:
        return False
    shapes = [var.aval.shape for var in body.invars]
    if len(shapes[1]) != 3 or shapes[1][-1] != 3:
        return False
    coils,nodes,_ = shapes[1]
    if coils < 1 or nodes < 1:
        return False
    expected_shapes = [(3,),(coils,nodes,3),(nodes,coils),(nodes,coils),(nodes,coils),
                       (coils,),(1,),(3,nodes,coils),(1,),(1,)]
    if shapes != expected_shapes or [var.aval.shape for var in body.outvars] != [(3,),(3,3)]:
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


def _supported_B(closed):
    from .fields import BiotSavart
    body = closed.jaxpr
    original = getattr(BiotSavart.B, "__wrapped__", None)
    if original is None:
        return False
    source = original.__code__
    expected = f"B at {source.co_filename}:{source.co_firstlineno}"
    if (getattr(getattr(body, "debug_info", None), "func_src_info", None) != expected
            or body.effects or len(body.invars) != 6):
        return False
    shapes = [var.aval.shape for var in body.invars]
    if len(shapes[1]) != 3 or shapes[1][-1] != 3:
        return False
    coils, nodes, _ = shapes[1]
    if coils < 1 or nodes < 1:
        return False
    if shapes != [(3,), (coils, nodes, 3), (nodes, coils), (nodes, coils), (nodes, coils), (coils,)]:
        return False
    if any(var.aval.dtype != jnp.dtype('float64') for var in body.invars):
        return False
    cross = [eq for eq in body.eqns if eq.primitive.name == 'jit' and eq.params.get('name') == 'cross']
    return (len(cross) == 1 and list(cross[0].invars[1:]) == list(body.invars[2:5])
            and [var.aval.shape for var in body.outvars] == [(3,)])


def has_prepared_biot_savart(graph):
    """Recognize ordinary JIT calls without entering custom/control-flow bodies."""
    for equation in graph.eqns:
        if equation.primitive.name == "jit":
            nested = equation.params["jaxpr"]
            if equation.params.get("name") == "B_and_dB_by_dX" and _supported(nested):
                return True
            if equation.params.get("name") == "B" and _supported_B(nested):
                return True
            if has_prepared_biot_savart(nested.jaxpr):
                return True
    return False


def _field_call(closed,values):
    """Keep original values and use the exact transpose to define a JVP.

    Transposing the linear pullback supplies the tangent without freezing its
    geometry coefficients. Ordinary JAX AD still provides higher derivatives.
    """
    @jax.custom_jvp
    def evaluate(*values):
        return tuple(jax.core.eval_jaxpr(closed.jaxpr,closed.consts,*values))
    @evaluate.defjvp
    def tangent(primals,tangents):
        value = evaluate(*primals)
        seed_shape = tree_util.tree_map(jnp.zeros_like,value)
        transpose = jax.linear_transpose(lambda seed:_geometry_pullback(primals,seed),seed_shape)
        return value,transpose(tangents)[0]
    # Expose only the small field outputs to the outer checkpoint policy.
    # A custom JVP otherwise hides its internal field reductions from it.
    return tuple(ad_checkpoint.checkpoint_name(value, "native_biot_savart_output")
                 for value in evaluate(*values))


def _B_call(closed, values):
    """Preserve original field values and obtain its JVP from the exact transpose."""
    @jax.custom_jvp
    def evaluate(*values):
        return tuple(jax.core.eval_jaxpr(closed.jaxpr, closed.consts, *values))
    @evaluate.defjvp
    def tangent(primals, tangents):
        value = evaluate(*primals)
        transpose = jax.linear_transpose(lambda seed: _B_pullback(primals, seed),
            tree_util.tree_map(jnp.zeros_like, value))
        return value, transpose(tangents)[0]
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
                if eq.params.get('name') == 'B_and_dB_by_dX' and _supported(closed):
                    result = _field_call(closed,values)
                elif eq.params.get('name') == 'B' and _supported_B(closed):
                    result = _B_call(closed, values)
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
