"""Physical first-hit gradients on frozen accepted trajectories."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.collisions import CollisionStep
from essos.first_loss import first_loss_event, first_loss_time, loss_time_penalty, mean_loss_time_cost
from essos.rk import solve_rk
from essos.split import solve_split


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_analytic_crossing_flow_birth_and_moving_boundary(method):
    p = jnp.array([2., .6, .1, .2])
    def boundary(t,y,p):
        return p[1]+p[3]*t-y[0]
    def run(p, adjoint):
        return solve_rk(lambda t,y,p:jnp.ones_like(y)*p[0], p[2:3], p,
            t1=1.,dt0=.8,event=first_loss_event(boundary),method=method,
            max_steps=32,adjoint=adjoint,record_event_tangent=True)
    def cost(p):
        return mean_loss_time_cost(run(p,"accepted"),boundary,1.,p)
    value,grad = jax.jit(jax.value_and_grad(cost))(p)
    analytic = lambda p:loss_time_penalty((p[1]-p[2])/(p[0]-p[3]),1.)
    np.testing.assert_allclose(value,analytic(p),rtol=1e-12)
    np.testing.assert_allclose(grad,jax.grad(analytic)(p),rtol=2e-11,atol=1e-12)
    np.testing.assert_allclose(run(p,"none").event_tangent,[p[0]],rtol=1e-13)
    primal = jax.jit(lambda p:mean_loss_time_cost(run(p,"none"),boundary,1.,p))
    eps=1e-6
    finite=np.array([(primal(p+eps*e)-primal(p-eps*e))/(2*eps) for e in np.eye(4)])
    np.testing.assert_allclose(grad,finite,rtol=3e-8,atol=1e-10)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_adaptive_rejections_and_nonlinear_crossing(method):
    def cost(p):
        boundary=lambda t,y,a:p[1]-y[0]
        out=solve_rk(lambda t,y,a:p[0]*y,jnp.array([.4]),t1=1.,dt0=1.,
            event=boundary,method=method,rtol=1e-11,atol=1e-11,max_steps=128,
            adjoint="accepted",record_event_tangent=True)
        return mean_loss_time_cost(out,boundary,1.),out.stats.rejected
    p=jnp.array([3.,1.7])
    (value,rejected),grad=jax.jit(jax.value_and_grad(cost,has_aux=True))(p)
    analytic=lambda p:loss_time_penalty(jnp.log(p[1]/.4)/p[0],1.)
    assert rejected>0
    np.testing.assert_allclose(value,analytic(p),rtol=1e-8)
    np.testing.assert_allclose(grad,jax.grad(analytic)(p),rtol=5e-8,atol=1e-10)


def test_coarse_step_uses_dense_polynomial_slope():
    boundary=lambda t,y,p:1.4-y[0]
    def run(p,adjoint):
        return solve_rk(lambda t,y,p:p[0]*y*y,jnp.ones(1),p,t1=.25,dt0=.25,
            adaptive=False,event=boundary,max_steps=4,method="dopri5",
            adjoint=adjoint,record_event_tangent=True)
    p=jnp.array([1.5])
    derivative=jax.jit(jax.grad(lambda p:first_loss_time(run(p,"accepted"),boundary)))(p)
    primal=jax.jit(lambda p:run(p,"none").t)
    eps=1e-6
    finite=(primal(p+eps)-primal(p-eps))/(2*eps)
    np.testing.assert_allclose(derivative[0],finite,rtol=2e-8,atol=1e-10)
    out=run(p,"none")
    assert abs(float(out.event_tangent[0]-p[0]*out.y[0]**2))>1e-6


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_split_noise_and_drag_influence_first_hit_gradient(method):
    boundary=lambda t,y,p:.7-y[0]
    def run(p,adjoint,record):
        def rhs(t,y,p):
            return jnp.array([y[1],0.])
        def kick(t,y,h,dw,p):
            return CollisionStep(y.at[1].add(-p[1]*y[1]*h+p[2]*dw[0]),jnp.array(0.),jnp.array(0.))
        return solve_split(rhs,kick,jnp.array([0.,p[0]]),p,t1=1.,dt0=.2,ts=jnp.array([0.,1.]),
            key=jax.random.key(71),noise_dim=1,adaptive=False,method=method,
            event=boundary,max_steps=32,max_orbit_steps=32,orbit_dt0=.1,
            adjoint=adjoint,adjoint_max_orbit_steps=128,record_event_tangent=record)
    p=jnp.array([1.,.15,.1])
    value,grad=jax.jit(jax.value_and_grad(lambda p:mean_loss_time_cost(run(p,"accepted",True),boundary,1.,p)))(p)
    primal=jax.jit(lambda p:mean_loss_time_cost(run(p,"none",True),boundary,1.,p))
    eps=1e-6
    finite=np.array([(primal(p+eps*e)-primal(p-eps*e))/(2*eps) for e in np.eye(3)])
    np.testing.assert_allclose(grad,finite,rtol=3e-7,atol=1e-10)
    assert np.isfinite(value) and np.all(np.abs(grad)>1e-5)
    baseline=run(p,"none",False)
    recorded=run(p,"none",True)
    for a,b in zip(baseline[:-1],recorded[:-1]):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_array_equal(x,y)


def test_batched_retained_and_initially_lost_states():
    boundary=lambda t,y,p:.5-y[0]
    def cost(initial):
        solutions=jax.vmap(lambda y:solve_rk(lambda t,y,p:jnp.ones_like(y)*.1,y,
            t1=1.,dt0=.2,event=first_loss_event(boundary),max_steps=16,
            adjoint="accepted",record_event_tangent=True))(initial)
        times=first_loss_time(solutions,boundary)
        return jnp.sum(loss_time_penalty(times,1.)),times
    (value,times),gradient=jax.jit(jax.value_and_grad(cost,has_aux=True))(jnp.array([[0.],[.8]]))
    np.testing.assert_allclose(times,[1.,0.],atol=1e-13)
    np.testing.assert_allclose(value,1.)
    np.testing.assert_array_equal(gradient,np.zeros((2,1)))


def test_grazing_and_unrefined_event_derivatives_are_rejected():
    boundary=lambda t,y,a:1.-y[0]
    def loss(p,tangent):
        out=SimpleNamespace(t=jnp.array(.5),y=p,event_tangent=tangent,status=jnp.array(1))
        return first_loss_time(out,boundary)
    np.testing.assert_allclose(loss(jnp.ones(1),jnp.zeros(1)),.5)
    for tangent in (jnp.zeros(1),jnp.full((1,),jnp.nan)):
        with pytest.raises(Exception,match="transverse outward continuous crossing"):
            jax.jit(jax.grad(lambda p:loss(p,tangent)))(jnp.ones(1)).block_until_ready()


def test_wrong_boundary_and_incomplete_solve_are_not_confinement():
    boundary=lambda t,y,a:1.-y[0]
    out=SimpleNamespace(t=jnp.array(.5),y=jnp.array([.5]),event_tangent=jnp.ones(1),status=jnp.array(1))
    with pytest.raises(Exception,match="not on the requested"):
        first_loss_time(out,boundary)
    out.status=jnp.array(2)
    with pytest.raises(Exception,match="requires successful or event"):
        first_loss_time(out,boundary)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_event_tangent_recording_keeps_rk_forward_values(method):
    options=dict(t1=1.,dt0=.2,event=lambda t,y,p:.4-y[0],ts=jnp.linspace(0.,1.,6),
        method=method,max_steps=16)
    rhs=lambda t,y,p:jnp.ones_like(y)
    baseline=solve_rk(rhs,jnp.zeros(1),**options)
    observed=solve_rk(rhs,jnp.zeros(1),**options,record_event_tangent=True)
    for a,b in zip(baseline[:-1],observed[:-1]):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_allclose(x,y,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(observed.event_tangent,[1.],rtol=1e-13)


def test_instantaneous_spatial_kick_has_no_continuous_hit_derivative():
    boundary=lambda t,y,p:.1-y[0]
    def cost(p):
        out=solve_split(lambda t,y,p:jnp.zeros_like(y),
            lambda t,y,h,dw,p:CollisionStep(y+p*h,jnp.array(0.),jnp.array(0.)),
            jnp.zeros(1),p,t1=1.,dt0=.5,ts=jnp.array([0.,1.]),key=jax.random.key(4),
            noise_dim=1,adaptive=False,event=boundary,max_steps=8,max_orbit_steps=8,
            adjoint="accepted",record_event_tangent=True)
        return mean_loss_time_cost(out,boundary,1.)
    assert np.isfinite(cost(jnp.array(.2)))
    with pytest.raises(Exception,match="transverse outward continuous crossing"):
        jax.jit(jax.grad(cost))(jnp.array(.2)).block_until_ready()
