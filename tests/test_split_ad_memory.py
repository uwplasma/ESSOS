"""Regression cases for bounded checkpoints and partial endpoint storage."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from essos.collisions import CollisionStep
from essos.split import solve_split,SplitStatus
from essos.split_ad import replay_accepted_split

@pytest.mark.parametrize('segment',[1,3,16])
def test_compact_checkpoint_slots_are_bounded_by_inner_capacity(segment):
    def rhs(t,y,p):return p[0]*y
    def collision(t,y,h,dw,p):
        return CollisionStep(y+p[1]*jnp.tanh(y)*dw,jnp.array(0.),jnp.array(0.))
    options=dict(t1=.4,dt0=.1,ts=jnp.array([0.,.4]),noise_dim=1,key=jax.random.key(84),
        adaptive=False,orbit_adaptive=False,method='dopri5',max_steps=64,max_orbit_steps=8,
        adjoint_max_accepted_steps=64,adjoint_max_orbit_steps=8,
        adjoint_orbit_checkpoint_steps=segment,_collision_state_indices=(0,))
    p=jnp.array([.3,.05,1.2])
    out,tape=solve_split(rhs,collision,p[2:],p[:2],**options,_record_ad=True)
    assert out.status==SplitStatus.SUCCESS and int(tape.count)==4
    assert tape.orbit_states.shape==(8,1)
    assert tape.orbit_derivatives.shape==((8-1)//segment,1)
    def metric(saved,final):return jnp.sum(final**2)+.3*jnp.sum(saved**2)
    def cost(p):
        solution=solve_split(rhs,collision,p[2:],p[:2],**options,adjoint='accepted')
        return metric(solution.ys,solution.y)
    def frozen(p):
        saved,final=replay_accepted_split(rhs,collision,p[2:],p[:2],tape,options['ts'],method='dopri5')
        return metric(saved,final)
    direction=jnp.array([.2,-.3,.4])
    value,gradient=jax.jit(jax.value_and_grad(cost))(p)
    _,tangent=jax.jvp(frozen,(p,),(direction,))
    np.testing.assert_allclose(value,metric(out.ys,out.y),rtol=2e-13)
    np.testing.assert_allclose(jnp.vdot(gradient,direction),tangent,rtol=2e-11,atol=1e-12)

@pytest.mark.parametrize('adaptive',[False,True])
def test_partial_endpoint_reverse_matches_full_with_unequal_events(adaptive):
    def rhs(t,y,p):return jnp.stack([y[1],-p[0]*y[0]])
    def collision(t,y,h,dw,p):
        return CollisionStep(y.at[1].add(-.1*h*y[1]+p[1]*(1+y[0])*dw[0]),
            (h/.12)**2,jnp.array(0.))
    times=jnp.array([0.,.2,.45,.7])
    keys=jax.random.split(jax.random.key(85),2)
    base=jnp.array([[.1,.4],[.6,.9]])
    options=dict(t1=.7,dt0=.4,ts=times,noise_dim=1,method='dopri5',adaptive=adaptive,
        event=lambda t,y,p:.8-y[0],max_steps=64,max_orbit_steps=32,
        adjoint_max_accepted_steps=64,adjoint_max_orbit_steps=128,
        adjoint_checkpoint_steps=3,adjoint_orbit_checkpoint_steps=3)
    def run(p,packed,adjoint='none',record=False):
        initial=base.at[:,0].add(p[2])
        return jax.vmap(lambda y,key:solve_split(rhs,collision,y,p[:2],**options,key=key,
            adjoint=adjoint,_record_ad=record,_collision_state_indices=(1,) if packed else None))(initial,keys)
    def metric(saved,final):return jnp.sum(saved**2)+jnp.sum(final)
    def cost(p,packed):
        out=run(p,packed,'accepted')
        return metric(out.ys,out.y)
    p=jnp.array([.3,.08,.01])
    baseline,gradient=jax.jit(jax.value_and_grad(lambda p:cost(p,False)))(p)
    packed,packed_gradient=jax.jit(jax.value_and_grad(lambda p:cost(p,True)))(p)
    np.testing.assert_allclose(packed,baseline,rtol=2e-13)
    np.testing.assert_allclose(packed_gradient,gradient,rtol=2e-11,atol=1e-12)
    out,tapes=jax.jit(lambda p:run(p,True,record=True))(p)
    assert int(out.status[0])==SplitStatus.SUCCESS and int(out.status[1])==SplitStatus.EVENT
    assert int(tapes.count[0])!=int(tapes.count[1])
    assert tapes.orbit_endpoints.shape[-1]==1
    def frozen(p):
        initial=base.at[:,0].add(p[2])
        saved,final=jax.vmap(lambda y,tape:replay_accepted_split(rhs,collision,y,p[:2],
            tape,times,method='dopri5'))(initial,tapes)
        return metric(saved,final)
    direction=jnp.array([.3,-.2,.4])
    _,tangent=jax.jvp(frozen,(p,),(direction,))
    np.testing.assert_allclose(jnp.vdot(packed_gradient,direction),tangent,rtol=2e-10,atol=2e-12)

@pytest.mark.parametrize('initial_event',[False,True])
def test_partial_endpoint_empty_path_preserves_initial_gradients(initial_event):
    def cost(y):
        out=solve_split(lambda t,y,p:y,
            lambda t,y,h,dw,p:CollisionStep(y.at[1].add(dw[0]),jnp.array(0.),jnp.array(0.)),
            y,t1=.5 if initial_event else 0.,dt0=.1,
            ts=jnp.array([0.,.2,.5]) if initial_event else jnp.zeros(3),key=jax.random.key(86),
            noise_dim=1,event=(lambda t,y,p:jnp.array(-1.)) if initial_event else None,
            max_steps=8,max_orbit_steps=8,adjoint='accepted',_collision_state_indices=(1,))
        return jnp.sum(out.y)+jnp.sum(out.ys)
    value,gradient=jax.jit(jax.value_and_grad(cost))(jnp.array([.3,.4]))
    np.testing.assert_allclose(value,2.8,atol=1e-13)
    np.testing.assert_array_equal(gradient,jnp.array([4.,4.]))


@pytest.mark.parametrize('segment',[1,3,16])
@pytest.mark.parametrize('cached',[False,True])
def test_compact_derivative_offsets_cover_unequal_inner_paths(segment,cached):
    """Every map omits its first derivative, including maps between checkpoints."""
    def rhs(t,y,p):
        if cached:
            phases=jnp.arange(40).reshape(20,2)*.01
            return p[0]*jnp.mean(jnp.sin(y[None,:]+t+phases),axis=0)
        return p[0]*jnp.sin(y+.2*t)
    def collision(t,y,h,dw,p):
        return CollisionStep(y.at[1].add(p[1]*(1+y[0])*dw[0]),jnp.array(0.),jnp.array(0.))
    times=jnp.array([0.,.23,.5])
    keys=jax.random.split(jax.random.key(87),2)
    inner_steps=jnp.array([.009,.023])
    base=jnp.array([[.2,.3],[.1,.4]])
    opts=dict(t1=.5,dt0=.1,ts=times,noise_dim=1,adaptive=False,orbit_adaptive=False,
        method='dopri8',max_steps=16,max_orbit_steps=32,adjoint_max_accepted_steps=16,
        adjoint_max_orbit_steps=128,adjoint_orbit_checkpoint_steps=segment,
        _collision_state_indices=(1,))
    def run(p,adjoint='none',record=False):
        return jax.vmap(lambda y,h,key:solve_split(rhs,collision,y,p[:2],**opts,
            orbit_dt0=h,key=key,adjoint=adjoint,_record_ad=record))(
                base.at[:,0].add(p[2]),inner_steps,keys)
    def metric(saved,final):return jnp.sum(saved**2)+jnp.sum(final**2)
    def cost(p):
        out=run(p,'accepted')
        return metric(out.ys,out.y)
    p=jnp.array([.4,.03,.1])
    out,tapes=jax.jit(lambda p:run(p,record=True))(p)
    for lane in range(2):
        count=int(tapes.count[lane])
        offsets=np.asarray(tapes.checkpoint_offsets[lane])
        assert int(tapes.checkpoint_count[lane])-count<=tapes.orbit_derivatives.shape[1]
        for index in range(count):
            slots=np.arange(offsets[index]+1,offsets[index+1])
            steps=int(tapes.offsets[lane,index])+segment*np.arange(1,len(slots)+1)
            values=tapes.orbit_states[lane,slots]
            node_times=tapes.orbit_times[lane,steps,0]
            expected=jax.vmap(lambda t,y:rhs(t,y,p[:2]))(node_times,values)
            stored=tapes.orbit_derivatives[lane,slots-index-1]
            np.testing.assert_allclose(stored,expected,rtol=2e-13,atol=2e-14)
    def frozen(p):
        saved,final=jax.vmap(lambda y,tape:replay_accepted_split(rhs,collision,y,p[:2],
            tape,times,method='dopri8'))(base.at[:,0].add(p[2]),tapes)
        return metric(saved,final)
    value,gradient=jax.jit(jax.value_and_grad(cost))(p)
    direction=jnp.array([.2,-.3,.4])
    _,tangent=jax.jvp(frozen,(p,),(direction,))
    np.testing.assert_allclose(value,metric(out.ys,out.y),rtol=2e-13)
    np.testing.assert_allclose(gradient@direction,tangent,rtol=2e-10,atol=2e-12)


def test_compact_single_inner_step_needs_no_derivative_bank():
    def rhs(t,y,p):return p[0]*y
    def collision(t,y,h,dw,p):
        return CollisionStep(y+p[1]*dw,jnp.array(0.),jnp.array(0.))
    opts=dict(t1=.1,dt0=.1,ts=jnp.array([0.,.1]),noise_dim=1,key=jax.random.key(88),
        adaptive=False,orbit_adaptive=False,max_steps=2,max_orbit_steps=1,
        adjoint_max_accepted_steps=2,adjoint_max_orbit_steps=1,
        adjoint_orbit_checkpoint_steps=1,_collision_state_indices=(0,))
    p=jnp.array([.3,.02,1.2])
    out,tape=solve_split(rhs,collision,p[2:],p[:2],**opts,_record_ad=True)
    assert tape.orbit_derivatives.shape==(0,1)
    def cost(p):
        return jnp.sum(solve_split(rhs,collision,p[2:],p[:2],**opts,adjoint='accepted').y**2)
    def frozen(p):
        return jnp.sum(replay_accepted_split(rhs,collision,p[2:],p[:2],tape,opts['ts'])[1]**2)
    value,gradient=jax.jit(jax.value_and_grad(cost))(p)
    direction=jnp.array([.2,-.3,.4])
    tangent=jax.jvp(frozen,(p,),(direction,))[1]
    np.testing.assert_allclose(value,jnp.sum(out.y**2),rtol=2e-13)
    np.testing.assert_allclose(gradient@direction,tangent,rtol=2e-11,atol=1e-12)




