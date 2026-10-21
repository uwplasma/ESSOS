"""Native B-only transpose against the original field and frozen Lorentz map."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import Lorentz, Particles
from essos.fields import BiotSavart
from essos.orbit_error import OrbitErrorNorm
from essos.rk import solve_rk, _partition_rhs
from essos.rk_ad import replay_accepted
from essos.rk_ad_biot_savart import _supported_B, _B_call, has_prepared_biot_savart
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
    particle = Particles(point[None], mass=M, charge=Q,
        energy=4000*ONE_EV*jnp.exp(.05*x[n+3]),
        initial_vparallel_over_v=jnp.array([.35+.05*x[n+4]]), field=field,
        phase_angle_full_orbit=jnp.pi+.1*x[n+5])
    initial = jnp.concatenate((particle.initial_xyz_fullorbit[0], particle.initial_vxvyvz[0]))
    return initial, lambda t,y,args: Lorentz(t, y, (field, particle))


@pytest.fixture(scope="module")
def prepared_call(base):
    initial, rhs = _initialize(base, jnp.concatenate((base.dofs, jnp.zeros(6))))
    _, stage = _partition_rhs(rhs, jnp.array(0.), initial, None)

    def find(graph):
        for equation in graph.eqns:
            if equation.primitive.name == "jit":
                nested = equation.params["jaxpr"]
                if _supported_B(nested):
                    return nested
                found = find(nested.jaxpr)
                if found is not None:
                    return found

    closed = find(stage.jaxpr)
    if closed is None:
        pytest.skip("JAX does not expose the supported prepared stock B signature")
    random = np.random.default_rng(32)
    values = tuple(jnp.asarray(random.normal(size=v.aval.shape)) for v in closed.jaxpr.invars)
    values = (jnp.array([.2, -.1, .3]), values[1]+jnp.array([3., 2., -1.]), *values[2:])
    return closed, values


@pytest.mark.parametrize("seed", [3, 8, 21])
def test_field_values_and_all_six_input_cotangents(prepared_call, seed):
    closed, values = prepared_call
    reference = lambda *p: tuple(jax.core.eval_jaxpr(closed.jaxpr, closed.consts, *p))
    expected, ordinary = jax.vjp(reference, *values)
    actual, candidate = jax.vjp(lambda *p: _B_call(closed, p), *values)
    np.testing.assert_array_equal(actual[0], expected[0])
    bar = (jnp.asarray(np.random.default_rng(seed).normal(size=3)),)
    for a, b in zip(jax.jit(candidate)(bar), jax.jit(ordinary)(bar)):
        np.testing.assert_allclose(a, b, rtol=4e-12, atol=2e-15)


def test_field_jvp_and_mixed_hessian_vector_preserve_all_inputs(prepared_call):
    closed, values = prepared_call
    reference = lambda p: tuple(jax.core.eval_jaxpr(closed.jaxpr, closed.consts, *p))
    candidate = lambda p: _B_call(closed, p)
    random = np.random.default_rng(19)
    direction = tuple(jnp.asarray(random.normal(size=v.shape))/10 for v in values)
    a = jax.jit(lambda p,d: jax.jvp(candidate, (p,), (d,)))(values, direction)
    b = jax.jit(lambda p,d: jax.jvp(reference, (p,), (d,)))(values, direction)
    for actual, expected in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=2e-14)

    def hessian_vector(function):
        metric = lambda p: jnp.dot(function(p)[0], jnp.array([.2, -.3, .7]))
        return jax.jit(lambda p,d: jax.jvp(jax.grad(metric), (p,), (d,))[1])(values, direction)
    for actual, expected in zip(hessian_vector(candidate), hessian_vector(reference)):
        np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=2e-14)


def test_same_named_user_field_is_not_specialized(prepared_call):
    _, values = prepared_call
    @jax.jit
    def B(*values):
        return values[0]**2
    assert not has_prepared_biot_savart(jax.make_jaxpr(B)(*values).jaxpr)


def test_factory_selects_stock_B_without_changing_parameter_preparation(base, prepared_call):
    initial, rhs = _initialize(base, jnp.concatenate((base.dofs, jnp.zeros(6))))
    setup, stage = _partition_rhs(rhs, jnp.array(0.), initial, None)
    raw = setup.consts, stage.consts
    def prepare(raw):
        return jax.core.eval_jaxpr(setup.jaxpr, raw[0]), raw[1]
    def evaluate(t, y, parameters):
        values, constants = parameters
        return jax.core.eval_jaxpr(stage.jaxpr, constants, *values, t, y)[0]
    selected, function = prepare_reverse_rhs(prepare, evaluate, jnp.array(0.), initial, raw)
    assert selected is prepare and function._native_biot_savart_transpose
    _, ordinary = prepare_reverse_rhs(prepare, evaluate, jnp.array(0.), initial, raw,
        specialize_biot_savart=False)
    assert not getattr(ordinary, "_native_biot_savart_transpose", False)

    parameters = selected(raw)
    cached = prepare_rhs_linearization(function, jnp.array(0.), initial, parameters)
    assert cached is not None
    linearize, cached_pullback, initial_cache = cached
    assert sum(value.size for value in jax.tree.leaves(initial_cache)) <= 24
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
def test_lorentz_coil_and_birth_gradients_match_original_frozen_jvp(base, method, mode, accumulation):
    x = jnp.concatenate((base.dofs, jnp.zeros(6)))
    speed = float(np.sqrt(2*4000*ONE_EV/M))
    scale = jnp.array([1., 1., 1., speed, speed, speed])
    times = jnp.array([0., .014, .045, .087, .2])*1e-6
    options = dict(t1=2e-7, dt0=1e-7, ts=times, method=method, dense_output=mode,
        rtol=1e-9, atol=1e-9, error_norm=OrbitErrorNorm(1e-9, 1e-9, speed, velocity_components=3),
        max_steps=128, adjoint_max_accepted_steps=64)
    initial, rhs = _initialize(base, x)
    out, tape = jax.jit(lambda: solve_rk(rhs, initial, **options, _record_ad=True))()
    assert out.status == 0 and tape.count > 1
    metric = lambda saved,final: jnp.sum((saved/scale)**2)+.3*jnp.sum((final/scale)**2)
    def cost(x):
        initial, rhs = _initialize(base, x)
        result = solve_rk(rhs, initial, **options, adjoint="accepted",
            adjoint_checkpoint_steps=3, adjoint_parameter_accumulation=accumulation)
        return metric(result.ys, result.y)
    def frozen(x):
        initial, rhs = _initialize(base, x)
        saved, final = replay_accepted(rhs, initial, None, tape, times,
            method=method, dense_output=mode)
        return metric(saved, final)
    value, gradient = jax.jit(jax.value_and_grad(cost))(x)
    assert jnp.isfinite(gradient).all()
    for birth in (False, True):
        direction = jnp.sin(jnp.arange(len(x))+.5)/np.sqrt(len(x))
        direction = direction.at[:len(base.dofs)].set(0.) if birth else direction.at[len(base.dofs):].set(0.)
        expected, tangent = jax.jit(lambda x,d: jax.jvp(frozen, (x,), (d,)))(x, direction)
        np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(gradient@direction, tangent, rtol=2e-9, atol=2e-10)
