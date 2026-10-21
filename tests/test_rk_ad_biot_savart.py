"""Native field linearization against the unchanged field and frozen RK map."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import GuidingCenter, Particles
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart
from essos.orbit_error import OrbitErrorNorm
from essos.rk import solve_rk, _partition_rhs
from essos.rk_ad import replay_accepted
from essos.rk_ad_biot_savart import _supported, _field_call, has_prepared_biot_savart
from essos.rk_ad_rhs import prepare_reverse_rhs, prepare_rhs_linearization

ROOT = Path(__file__).resolve().parents[1]
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def base():
    return Coils.from_json(str(ROOT/"examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"))


def _initialize(base, x):
    n = len(base.dofs)
    field = BiotSavart(base.with_dofs(x[:n]))
    point = jnp.array([1.23, 0., 0.])+.01*x[n:n+3]
    energy = 4000*ONE_EV*jnp.exp(.05*x[n+3])
    particle = Particles(point[None], mass=M, charge=Q, energy=energy,
        initial_vparallel_over_v=jnp.array([.35+.05*x[n+4]]))
    initial = jnp.concatenate([point, jnp.ravel(particle.initial_vparallel)[:1]])
    rhs = lambda t,y,args: GuidingCenter(t,y,(field,particle,Electric_field_zero()))
    return initial, rhs


@pytest.fixture(scope="module")
def prepared_call(base):
    x = jnp.concatenate([base.dofs, jnp.zeros(5)])
    initial, rhs = _initialize(base, x)
    _, stage = _partition_rhs(rhs, jnp.array(0.), initial, None)

    def find(graph):
        for equation in graph.eqns:
            if equation.primitive.name == "jit":
                nested = equation.params["jaxpr"]
                if _supported(nested):
                    return nested
                result = find(nested.jaxpr)
                if result is not None:
                    return result

    if not has_prepared_biot_savart(stage.jaxpr):
        pytest.skip("JAX does not expose the supported prepared stock field signature")
    closed = find(stage.jaxpr)
    assert closed is not None
    random = np.random.default_rng(21)
    values = tuple(jnp.asarray(random.normal(size=var.aval.shape)) for var in closed.jaxpr.invars)
    values = (jnp.array([.2, -.1, .3]), values[1]+jnp.array([3., 2., -1.]), *values[2:])
    return closed, values


@pytest.mark.parametrize("kind", ["both", "B", "trace"])
def test_prepared_values_and_all_input_cotangents_match_original(prepared_call, kind):
    closed, values = prepared_call
    reference = lambda *values: tuple(jax.core.eval_jaxpr(closed.jaxpr, closed.consts, *values))
    expected, pullback = jax.vjp(reference, *values)
    found, custom = jax.vjp(lambda *values: _field_call(closed, values), *values)
    seed = (jnp.array([.7, -.2, .5]), jnp.arange(9, dtype=float).reshape(3, 3)/7-.3)
    if kind == "B":
        seed = seed[0], jnp.zeros((3, 3))
    elif kind == "trace":
        seed = jnp.zeros(3), jnp.eye(3)
    for a, b in zip(found, expected):
        np.testing.assert_array_equal(a, b)
    # Tangent views deliberately differ from the full tangent tensor. Check
    # every independent input, rather than only a merged coil gradient.
    for a, b in zip(jax.jit(custom)(seed), jax.jit(pullback)(seed)):
        np.testing.assert_allclose(a, b, rtol=4e-12, atol=2e-15)


def test_prepared_jvp_and_hessian_vector_preserve_geometry_derivatives(prepared_call):
    closed, values = prepared_call
    reference = lambda *values: tuple(jax.core.eval_jaxpr(closed.jaxpr, closed.consts, *values))
    candidate = lambda *values: _field_call(closed, values)
    random = np.random.default_rng(11)
    directions = tuple(jnp.asarray(random.normal(size=value.shape))/10 for value in values)
    actual = jax.jit(lambda p,d: jax.jvp(lambda p:candidate(*p), (p,), (d,)))(values, directions)
    expected = jax.jit(lambda p,d: jax.jvp(lambda p:reference(*p), (p,), (d,)))(values, directions)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=4e-12, atol=2e-14)
    seed = (jnp.array([.7, -.2, .5]), jnp.arange(9, dtype=float).reshape(3, 3)/7-.3)

    def hessian_vector(function):
        def cost(point):
            field, jacobian = function(point, *values[1:])
            return jnp.vdot(field, seed[0])+jnp.vdot(jacobian, seed[1])
        return jax.jit(lambda p,d: jax.jvp(jax.grad(cost), (p,), (d,)))(values[0], directions[0])

    for a, b in zip(hessian_vector(candidate), hessian_vector(reference)):
        np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-13)


def test_source_check_rejects_a_same_named_user_function(prepared_call):
    _, values = prepared_call

    @jax.jit
    def B_and_dB_by_dX(*values):
        point = values[0]
        return point**2, jnp.outer(point, point)

    graph = jax.make_jaxpr(B_and_dB_by_dX)(*values)
    assert not has_prepared_biot_savart(graph.jaxpr)


def test_reverse_factory_selects_stock_call_without_changing_setup(base, prepared_call):
    initial, rhs = _initialize(base, jnp.concatenate([base.dofs, jnp.zeros(5)]))
    setup, stage = _partition_rhs(rhs, jnp.array(0.), initial, None)
    raw = setup.consts, stage.consts

    def prepare(raw):
        return jax.core.eval_jaxpr(setup.jaxpr, raw[0]), raw[1]

    def evaluate(t, y, parameters):
        values, constants = parameters
        return jax.core.eval_jaxpr(stage.jaxpr, constants, *values, t, y)[0]

    selected, function = prepare_reverse_rhs(prepare, evaluate, jnp.array(0.), initial, raw)
    assert selected is prepare and function._native_biot_savart_transpose
    _, ordinary = prepare_reverse_rhs(prepare, evaluate, jnp.array(0.), initial, raw, specialize_biot_savart=False)
    assert not getattr(ordinary, "_native_biot_savart_transpose", False)

    parameters = selected(raw)
    cached = prepare_rhs_linearization(function, jnp.array(0.), initial, parameters)
    assert cached is not None
    linearize, cached_pullback, initial_cache = cached
    assert sum(value.size for value in jax.tree.leaves(initial_cache)) <= 32
    shift = .002*jnp.maximum(jnp.abs(initial), 1.)*jnp.sin(jnp.arange(initial.size)+.3)
    seed = jnp.linspace(-.3, .4, initial.size)
    for point in (initial, initial+shift):
        value, cache = linearize(jnp.array(0.), point)
        expected, pullback = jax.vjp(lambda y,p:function(jnp.array(0.), y, p), point, parameters)
        np.testing.assert_allclose(value, expected, rtol=2e-13, atol=2e-12)
        for a, b in zip(jax.tree.leaves(cached_pullback(cache, seed)), jax.tree.leaves(pullback(seed))):
            if b.dtype == jax.dtypes.float0:
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("mode", ["standard", "higher_order"])
@pytest.mark.parametrize("accumulation", ["step", "segment"])
def test_complete_rk_coil_and_birth_gradients_match_frozen_jvp(base, method, mode, accumulation):
    x = jnp.concatenate([base.dofs, jnp.zeros(5)])
    speed = float(np.sqrt(2*4000*ONE_EV/M))
    scale = jnp.array([1., 1., 1., speed])
    times = jnp.array([0., .14, .45, .87, 2.])*1e-6
    options = dict(t1=2e-6, dt0=1e-6, ts=times, method=method, dense_output=mode,
        rtol=1e-9, atol=1e-9, error_norm=OrbitErrorNorm(1e-9, 1e-9, speed, velocity_components=1),
        max_steps=128, adjoint_max_accepted_steps=64)
    initial, rhs = _initialize(base, x)
    out, tape = jax.jit(lambda: solve_rk(rhs, initial, **options, _record_ad=True))()
    assert out.status == 0 and tape.count > 1

    def metric(saved, final):
        return jnp.sum((saved/scale)**2)+.3*jnp.sum((final/scale)**2)

    def cost(x):
        initial, rhs = _initialize(base, x)
        result = solve_rk(rhs, initial, **options, adjoint="accepted", adjoint_checkpoint_steps=3,
            adjoint_parameter_accumulation=accumulation)
        return metric(result.ys, result.y)

    def frozen(x):
        initial, rhs = _initialize(base, x)
        saved, final = replay_accepted(rhs, initial, None, tape, times, method=method, dense_output=mode)
        return metric(saved, final)

    value, gradient = jax.jit(jax.value_and_grad(cost))(x)
    assert jnp.isfinite(gradient).all()
    for birth in (False, True):
        direction = jnp.sin(jnp.arange(len(x))+.5)/np.sqrt(len(x))
        direction = direction.at[:len(base.dofs)].set(0.) if birth else direction.at[len(base.dofs):].set(0.)
        expected, tangent = jax.jit(lambda x,d: jax.jvp(frozen, (x,), (d,)))(x, direction)
        np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(gradient@direction, tangent, rtol=2e-9, atol=2e-10)
