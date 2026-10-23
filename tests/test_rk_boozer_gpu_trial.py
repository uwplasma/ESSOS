"""Opt-in GPU trial preserves the physical RK map, including edge charts."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_fused_trial_matches_original_dopri8_at_axis_and_spline_edges():
    from essos.boozer import boozer_compiler_options, guiding_center_rhs
    from essos.rk import _trial_step, _C8, _A8, _ERROR8
    from essos.rk_boozer_gpu_trial import prepare_trial_factory
    from examples.benchmarks.benchmark_native_boozer_alpha import PARAMETERS, ROOT, setup, MASS, CHARGE
    if not boozer_compiler_options():
        pytest.skip("CUDA-only kernel")
    settings = SimpleNamespace(**(dict(PARAMETERS) | dict(particles=64,
        field_cache=ROOT/"examples/benchmarks/results/gpu_boozer_fixed_20261022/coordinate_inputs.npz")))
    field, _, _, speed, _, _ = setup(settings)
    s = jnp.asarray(np.resize([0., 1e-14, .005, .3, .95, .995, .999999],64))
    theta = jnp.linspace(-jnp.pi,jnp.pi,64)
    zeta = jnp.linspace(-1.1,5.2,64)
    pitch = jnp.linspace(-.9,.9,64)
    y = jnp.stack((jnp.sqrt(s)*jnp.cos(theta),jnp.sqrt(s)*jnp.sin(theta),zeta,speed*pitch),axis=1)
    mu = speed**2*(1-pitch**2)/(2*jax.vmap(field.modB)(s,theta,zeta))
    factory = prepare_trial_factory(field,MASS,CHARGE)
    h = jnp.asarray(1e-10)
    def one(point,moment,fused):
        rhs = lambda t,y,a:guiding_center_rhs(field,y,moment,MASS,CHARGE)
        f0 = rhs(0.,point,None)
        step = factory(moment) if fused else _trial_step
        return step(rhs,0.,point,h,None,f0,_C8,_A8,_ERROR8)
    ordinary = jax.jit(jax.vmap(lambda y,m:one(y,m,False)))(y,mu)
    fused = jax.jit(jax.vmap(lambda y,m:one(y,m,True)))(y,mu)
    scale = jnp.array([1.,1.,1.,speed])
    for actual,expected in zip(fused[:3],ordinary[:3]):
        np.testing.assert_allclose(actual/scale,expected/scale,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(fused.f1/(scale/h),ordinary.f1/(scale/h),rtol=3e-12,atol=3e-12)


def test_experimental_trial_override_cannot_enter_reverse_replay():
    from essos.rk import solve_rk
    with pytest.raises(ValueError,match="forward-only"):
        solve_rk(lambda t,y,a:y,jnp.ones(1),t1=1.,dt0=.1,
            adjoint="accepted",_trial_override=lambda *args:None)
