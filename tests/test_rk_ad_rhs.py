"""Compact RHS residuals preserve the original field map and its derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import solve_rk
from essos.rk_ad import replay_accepted
from essos.rk_ad_rhs import prepare_rhs_linearization, prepare_reverse_rhs

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("size,name,retained", [
    (3, "native_biot_savart_output", True),
    (24, "native_biot_savart_output", False),
    (3, "unrelated_output", False),
])
def test_named_field_cache_is_small_selective_and_preserves_derivatives(size, name, retained):
    from jax.ad_checkpoint import checkpoint_name
    def rhs(t, y, p):
        value = checkpoint_name(jnp.sin(y+p), name)
        return value**2
    initial, parameter = jnp.linspace(.1, .3, size), jnp.array(.4)
    cached = prepare_rhs_linearization(rhs, jnp.array(0.), initial, parameter)
    if not retained:
        assert cached is None
        return
    assert cached is not None
    assert prepare_rhs_linearization(rhs, jnp.array(0.), initial, parameter,
        cache_field_outputs=False) is None
    linearize, cached_pullback, initial_cache = cached
    assert sum(value.size for value in jax.tree.leaves(initial_cache)) <= 4*size
    seed = jnp.linspace(-.2, .7, size)
    for state in (initial, initial+.02):
        value, cache = linearize(jnp.array(0.), state)
        expected, pullback = jax.vjp(lambda y,p:rhs(jnp.array(0.), y, p), state, parameter)
        np.testing.assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
        for a, b in zip(cached_pullback(cache, seed), pullback(seed)):
            np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-13)


def test_small_reduction_cache_keeps_mixed_dtypes_and_parameter_cotangents():
    nodes32 = jnp.linspace(.1, 1., 64, dtype=jnp.float32)
    nodes64 = nodes32.astype(jnp.float64)

    def rhs(t, y, p):
        a = jnp.sum(jnp.sin(nodes32*y[0].astype(jnp.float32)+p[0]))
        b = jnp.sum(jnp.cos(nodes64*y[1]+p[1]*t))
        return jnp.array([a*a/4096+y[1], b*b/4096-y[0]])

    parameters = (jnp.float32(.3), jnp.float64(.7))
    initial = jnp.array([.7, -.2])
    evaluate, pullback, initial_cache = prepare_rhs_linearization(rhs, .2, initial, parameters)
    assert {value.dtype for value in initial_cache} == {jnp.dtype("float32"), jnp.dtype("float64")}
    assert sum(value.size for value in initial_cache) < 32
    seed = jnp.array([.3, -.4])
    for t, y in ((.2, initial), (.37, initial+.1)):
        value, cache = jax.jit(evaluate)(t, y)
        expected, reference = jax.vjp(lambda y, p: rhs(t, y, p), y, parameters)
        np.testing.assert_array_equal(value, expected)
        found = jax.jit(pullback)(cache, seed)
        for actual, desired in zip(jax.tree.leaves(found), jax.tree.leaves(reference(seed))):
            np.testing.assert_allclose(actual, desired, rtol=2e-6, atol=2e-7)


def test_cheap_rhs_keeps_existing_reconstruction():
    assert prepare_rhs_linearization(lambda t, y, p: p*jnp.sin(y+t),
                                     .2, jnp.ones(2), .7) is None


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("mode", ["standard", "optimized", "higher_order"])
@pytest.mark.parametrize("checkpoint_steps", [1, 2])
@pytest.mark.parametrize("accumulation", ["step", "segment"])
def test_cached_rhs_dense_event_and_partial_segment_match_frozen_jvp(method, mode, checkpoint_steps, accumulation):
    nodes = jnp.linspace(.1, 1., 64)

    def rhs(t, y, p):
        average = jnp.mean(jnp.sin(nodes*y[0]+p[0]*t)+p[1]*jnp.cos(nodes*y[1]))
        return jnp.array([y[1]+p[0]*average, -y[0]+p[1]*average*average])

    options = dict(t1=.7, dt0=.15, adaptive=False, max_steps=10, method=method,
        dense_output=mode, ts=jnp.array([0., .07, .18, .41]), event=lambda t, y, p: .43-t)
    x = jnp.array([.7, -.2, .6, .8])
    plain, tape = solve_rk(rhs, x[:2], x[2:], **options, _record_ad=True)
    assert int(plain.stats.accepted) == 3

    def loss(ys, final):
        return jnp.sum(ys**2)+.3*jnp.sum(final**2)

    def cost(x):
        out = solve_rk(rhs, x[:2], x[2:], **options, adjoint="accepted",
                       adjoint_checkpoint_steps=checkpoint_steps, adjoint_parameter_accumulation=accumulation)
        return loss(out.ys, out.y)

    def frozen(x):
        ys, final = replay_accepted(rhs, x[:2], x[2:], tape, options["ts"],
                                    method=method, dense_output=mode)
        return loss(ys, final)

    value, gradient = jax.jit(jax.value_and_grad(cost))(x)
    direction = jnp.array([.2, -.3, .4, -.1])
    reference_value, reference_tangent = jax.jit(lambda x, d: jax.jvp(frozen, (x,), (d,)))(x, direction)
    np.testing.assert_allclose(value, reference_value, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(gradient@direction, reference_tangent, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("accumulation", ["step", "segment"])
def test_cached_batched_empty_path_with_unused_singular_rhs(method, accumulation):
    nodes = jnp.linspace(.1, 1., 64)

    def one(parameter, horizon):
        def rhs(t, y, p):
            average = jnp.mean(jnp.sin(nodes*y[0]))
            return jnp.sqrt(p)*average*y
        return solve_rk(rhs, jnp.ones(1), parameter, t1=horizon, dt0=.03,
                        adaptive=False, max_steps=10, method=method,
                        adjoint="accepted", adjoint_checkpoint_steps=3,
                        adjoint_parameter_accumulation=accumulation).y[0]

    parameters, horizons = jnp.array([-1., .7]), jnp.array([0., .13])
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(jax.vmap(one)(p, horizons))))(parameters)
    expected = jax.jit(jax.grad(one))(.7, .13)
    np.testing.assert_array_equal(gradient[0], 0.)
    np.testing.assert_allclose(gradient[1], expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_segment_parameter_accumulation_handles_nonlinear_setup_and_rejections(method):
    from essos.rk import RKStatus

    def objective(x, accumulation):
        # Parameter-only preparation is nonlinear. It is differentiated after
        # summing cotangents, rather than evaluating preparation at a new point.
        weights = jnp.sin(x[2:])+x[2:]**2
        def rhs(t, y, unused):
            return weights*jnp.array([y[1]+jnp.sin(t), -y[0]])
        out = solve_rk(rhs, x[:2], t1=1., dt0=1., ts=jnp.array([0., .13, .43, .87, 1.]),
            method=method, rtol=1e-11, atol=1e-12, max_steps=100,
            adjoint="accepted", adjoint_checkpoint_steps=3,
            adjoint_parameter_accumulation=accumulation)
        return jnp.sum(out.ys**2)+.3*jnp.sum(out.y**2), (out.stats, out.status)

    x = jnp.array([.7, -.2, .6, .8])
    ((value, (stats, status)), reference) = jax.jit(jax.value_and_grad(
        lambda x: objective(x, "step"), has_aux=True))(x)
    ((found, (other_stats, other_status)), gradient) = jax.jit(jax.value_and_grad(
        lambda x: objective(x, "segment"), has_aux=True))(x)
    assert status == other_status == RKStatus.SUCCESS and stats.rejected > 0
    np.testing.assert_array_equal(stats, other_stats)
    np.testing.assert_allclose(found, value, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gradient, reference, rtol=2e-10, atol=2e-11)


def test_invalid_parameter_accumulation_is_rejected():
    with pytest.raises(ValueError, match="adjoint_parameter_accumulation"):
        solve_rk(lambda t,y,p: y, jnp.ones(1), t1=1., dt0=.1,
                 adjoint_parameter_accumulation="all")


@pytest.mark.parametrize("rule", ["jvp", "vjp"])
@pytest.mark.parametrize("mode", [0, 1])
def test_reverse_rhs_inlining_preserves_custom_rules_and_control_flow(rule, mode):
    @jax.custom_jvp
    def marked_jvp(x):
        return jnp.sin(x)

    @marked_jvp.defjvp
    def marked_jvp_rule(primals, tangents):
        x, = primals
        dx, = tangents
        return marked_jvp(x), 2*jnp.cos(x)*dx

    @jax.custom_vjp
    def marked_vjp(x):
        return jnp.sin(x)

    def forward(x):
        return marked_vjp(x), x

    def backward(x, seed):
        return (3*jnp.cos(x)*seed,)

    marked_vjp.defvjp(forward, backward)
    marked = marked_jvp if rule == "jvp" else marked_vjp

    @jax.jit
    def prepare(raw):
        return {"weights": marked(raw["weights"])+raw["weights"]**2, "mode": raw["mode"]}

    @jax.jit
    def evaluate(t,y,p):
        value = jax.lax.cond(p["mode"] > 0, lambda z: marked(z), lambda z: z**2, y+t)
        return p["weights"]*value

    t, y = .2, jnp.array([.7, -.3])
    raw = {"weights": jnp.array([.4, .8]), "mode": jnp.int32(mode)}
    build, rhs = prepare_reverse_rhs(prepare, evaluate, t,y,raw)
    found, pullback = jax.vjp(lambda y,p:rhs(t,y,build(p)), y,raw)
    expected, reference = jax.vjp(lambda y,p:evaluate(t,y,prepare(p)), y,raw)
    seed = jnp.array([.3, -.7])
    np.testing.assert_allclose(found, expected, rtol=2e-13, atol=2e-13)
    for a, b in zip(jax.tree.leaves(pullback(seed)), jax.tree.leaves(reference(seed))):
        assert a.dtype == b.dtype
        if a.dtype != jax.dtypes.float0:
            np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12)

    # The original solver preparation must preserve the rules too, before
    # reverse-only preparation ever sees the graph. Cover both partition users.
    from essos.rk import _partition_rhs
    from essos.split_ad import _partition

    def orbit_partition(y, p):
        setup, stage = _partition_rhs(lambda t,y,args:evaluate(t,y,prepare(args)), t,y,p)
        residuals = jax.core.eval_jaxpr(setup.jaxpr, setup.consts)
        return jax.core.eval_jaxpr(stage.jaxpr, stage.consts, *residuals,t,y)[0]

    def split_partition(y, p):
        raw, build, rhs = _partition(lambda t,y:evaluate(t,y,prepare(p)), t,y)
        return rhs((t,y), build(raw))

    for function in (orbit_partition, split_partition):
        value, pullback = jax.vjp(function, y,raw)
        np.testing.assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
        for a,b in zip(jax.tree.leaves(pullback(seed)), jax.tree.leaves(reference(seed))):
            assert a.dtype == b.dtype
            if a.dtype != jax.dtypes.float0:
                np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("accumulation", ["step", "segment"])
def test_solver_preserves_nested_custom_jvp_on_frozen_grid(method, mode, accumulation):
    @jax.custom_jvp
    def marked(x):
        return jnp.sin(x)

    @marked.defjvp
    def rule(primals,tangents):
        x, = primals
        dx, = tangents
        return marked(x), 2*jnp.cos(x)*dx

    nodes = jnp.linspace(.1,1.,64)

    @jax.jit
    def rhs(t,y,p):
        z = nodes*y[0]+p["weights"][0]*t
        values = jax.lax.cond(p["mode"] > 0, marked, jnp.sin, z)
        mean = jnp.mean(values)
        return jnp.array([y[1]+mean*p["weights"][0], -y[0]+mean*p["weights"][1]])

    options = dict(t1=.35,dt0=.1,adaptive=False,max_steps=5,method=method,ts=jnp.array([0.,.11,.24,.35]))
    x = jnp.array([.4,.7,.3,.7])
    parameters = lambda x: {"weights": x[2:], "mode": jnp.int32(mode)}
    _,tape = solve_rk(rhs,x[:2],parameters(x),**options,_record_ad=True)

    def cost(x):
        result = solve_rk(rhs,x[:2],parameters(x),**options,adjoint="accepted",adjoint_checkpoint_steps=3,
                          adjoint_parameter_accumulation=accumulation)
        return jnp.sum(result.ys**2)+.2*jnp.sum(result.y**2)

    def frozen(x):
        ys,y = replay_accepted(rhs,x[:2],parameters(x),tape,options["ts"],method=method)
        return jnp.sum(ys**2)+.2*jnp.sum(y**2)

    value,gradient = jax.jit(jax.value_and_grad(cost))(x)
    direction = jnp.array([.3,-.2,.7,.4])
    expected,tangent = jax.jit(lambda x,d:jax.jvp(frozen,(x,),(d,)))(x,direction)
    np.testing.assert_allclose(value,expected,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(gradient@direction,tangent,rtol=2e-10,atol=2e-11)
