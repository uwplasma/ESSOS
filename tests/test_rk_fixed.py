"""Aligned fixed-grid execution must preserve event and failure semantics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.rk import solve_rk, RKStatus
from essos.rk_fixed import aligned_substeps, solve_fixed_grid

jax.config.update("jax_enable_x64", True)

def test_grid_selection_preserves_timestep_and_work_limit():
    assert aligned_substeps(2e-5, 1.25e-7, 9, 160) == 20
    assert aligned_substeps(2e-5, 1.25e-7, 8, 160) is None
    assert aligned_substeps(2e-5, 1.25e-7, 9, 159) is None
    assert aligned_substeps(1., 2., 2, 100) is None
    assert aligned_substeps(1., .5, 3, 0) is None


@pytest.mark.parametrize("layout", ["loop", "unrolled"])
@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_mixed_lanes_keep_root_found_events_and_unreached_saves(layout, method):
    times = jnp.linspace(0., 1., 5)
    def rhs(t, y, a):
        return jnp.stack((1 + .2*jnp.sin(t), .3*y[1]))
    event = lambda t,y,a:1-y[0]
    births = jnp.array([[-2., 1.], [.1, 1.], [.7, 1.], [1., 1.]])
    options = dict(ts=times, dt0=.125, event=event, stage_execution=layout, method=method)
    fast = jax.jit(jax.vmap(lambda y:solve_fixed_grid(rhs,y,substeps=2,**options)))(births)
    normal = jax.jit(jax.vmap(lambda y:solve_rk(rhs,y,t1=1.,adaptive=False,
        max_steps=8,**options)))(births)
    for name in ("y", "t", "ys", "ts"):
        np.testing.assert_allclose(getattr(fast,name),getattr(normal,name),rtol=3e-14,atol=3e-14)
    np.testing.assert_array_equal(fast.status,normal.status)
    np.testing.assert_array_equal(fast.stats,normal.stats)
    assert fast.status[0] == RKStatus.SUCCESS
    assert fast.status[-1] == RKStatus.EVENT
    np.testing.assert_allclose(fast.y[1:3,0],1.,rtol=0.,atol=2e-14)


@pytest.mark.parametrize("bad_root", [False, True])
@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_nonfinite_trial_and_root_roll_back_without_an_extra_accepted_step(bad_root, method):
    times = jnp.linspace(0., 1., 5)
    def rhs(t,y,a):
        return jnp.where((t>.3) & (not bad_root),jnp.nan,jnp.ones_like(y))
    def event(t,y,a):
        return jnp.where(bad_root & (t>.26) & (t<.30),jnp.nan,.28-y[0])
    options = dict(ts=times,dt0=.125,event=event,stage_execution="unrolled",method=method)
    birth = jnp.array([0.])
    fast = jax.jit(lambda y:solve_fixed_grid(rhs,y,substeps=2,**options))(birth)
    normal = jax.jit(lambda y:solve_rk(rhs,y,t1=1.,adaptive=False,max_steps=8,**options))(birth)
    for name in ("y", "t", "ys", "ts", "status", "stats"):
        np.testing.assert_allclose(getattr(fast,name),getattr(normal,name),rtol=3e-14,atol=3e-14)
    assert fast.status == RKStatus.NONFINITE


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_boozer_production_kernel_keeps_lcfs_root_energy_and_held_states(method):
    from essos.boozer import _native_orbits, BoozerField
    from essos.constants import (ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q,
                                 FUSION_ALPHA_PARTICLE_ENERGY)
    V0 = float(np.sqrt(2*FUSION_ALPHA_PARTICLE_ENERGY/M))
    s = np.linspace(.005,.995,50)
    field = BoozerField.from_booz(s,np.stack((np.full_like(s,5.),-1.25*np.sqrt(s))),
        [0,1],[0,0],np.full_like(s,.4),np.full_like(s,40.),np.zeros_like(s),-40.,1,angle_grid=None)
    birth = jnp.array([[0.,jnp.sqrt(.9999),0.,0.],[jnp.sqrt(.2),0.,0.,.5*V0]])
    B = jax.vmap(field.modB)(jnp.array([.9999,.2]),jnp.array([jnp.pi/2,0.]),jnp.zeros(2))
    mu = (V0**2-birth[:,3]**2)/(2*B)
    times = jnp.linspace(0.,1e-6,5)
    args = (field,birth,mu,M,Q,V0,times,1.25e-7,1e-9,1e-9,method,False,"i",.4,.3,"standard",8,"unrolled")
    fast = _native_orbits(*args,fixed_substeps=2)
    normal = _native_orbits(*args)
    for a,b in zip(fast,normal):
        np.testing.assert_allclose(a,b,rtol=5e-13,atol=1e-10)
    assert fast[1][0]>0
    assert fast[-1][0] == RKStatus.EVENT


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_component_recurrence_retains_state_and_parameter_tangents(method):
    from essos.rk_fixed import _component_step
    from essos.rk import _trial_step_unrolled, _C, _A, _ERROR, _C8, _A8, _ERROR8

    tableau = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)

    def endpoint(values, componentwise):
        y, coefficient = values[:2], values[2]
        def rhs(t, y, unused):
            return jnp.stack((coefficient*jnp.sin(y[1])+t, y[0]*y[1]-.2*y[1]))
        f = rhs(.1, y, None)
        if componentwise:
            return jnp.stack(_component_step(rhs,.1,tuple(y),.03,None,tuple(f),*tableau).y)
        return _trial_step_unrolled(rhs,.1,y,.03,None,f,*tableau).y

    values, direction = jnp.array([.3,-.5,1.2]), jnp.array([.2,.4,-.1])
    actual = jax.jvp(lambda x:endpoint(x,True),(values,),(direction,))
    reference = jax.jvp(lambda x:endpoint(x,False),(values,),(direction,))
    np.testing.assert_allclose(actual,reference,rtol=2e-14,atol=2e-14)
