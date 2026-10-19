"""Pathwise accepted-schedule AD: production noise, nested grids and physics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.collisions import CollisionStep
from essos.split import solve_split, trace_split, SplitStatus
from essos.split_ad import replay_accepted_split


def kernels():
    def rhs(t, y, p):
        return p[0]*jnp.sin(y+12*t)
    def collision(t, y, h, dw, p):
        state = y-p[2]*h*y+p[1]*(.7+.1*y)*dw
        # p[3] influences acceptance only, and must receive zero cotangent.
        return CollisionStep(state, (h/(.14*jnp.exp(p[3])))**2, jnp.array(0.))
    return rhs, collision


def settings(method="dopri5", controller="conservative", segment=3):
    return dict(t1=1., dt0=.8, ts=jnp.array([0., 0., .31, .73, 1.]),
        key=jax.random.key(71), noise_dim=1, method=method,
        collision_controller=controller, rtol=1e-10, atol=1e-10,
        max_steps=100, max_orbit_steps=64, adjoint_checkpoint_steps=segment,
        adjoint_orbit_checkpoint_steps=3, adjoint_max_accepted_steps=64,
        adjoint_max_orbit_steps=256)


def objective(out):
    return jnp.sum(out.y**2)+jnp.sum(out.ys*jnp.linspace(.1, .7, len(out.ts))[:, None])


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_recording_keeps_production_forward_and_noise_exact(method):
    rhs, collision = kernels()
    p, y0 = jnp.array([.4, .2, .1, 0.]), jnp.array([1.2])
    opts = settings(method)
    normal = jax.jit(lambda: solve_split(rhs, collision, y0, p, **opts))()
    recorded, tape = jax.jit(lambda: solve_split(rhs, collision, y0, p, **opts, _record_ad=True))()
    accepted = jax.jit(lambda: solve_split(rhs, collision, y0, p, **opts, adjoint="accepted"))()
    for actual in (recorded, accepted):
        for left, right in zip(jax.tree.leaves(normal), jax.tree.leaves(actual)):
            np.testing.assert_array_equal(left, right)
    assert recorded.status == SplitStatus.SUCCESS and recorded.stats.rejected > 0
    assert tape.orbit_count >= tape.count and tape.count == recorded.stats.accepted
    saved, final = jax.jit(lambda: replay_accepted_split(rhs, collision, y0, p, tape, opts['ts'], method=method))()
    np.testing.assert_allclose(saved, recorded.ys, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(final, recorded.y, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
@pytest.mark.parametrize("controller", ["conservative", "paper"])
@pytest.mark.parametrize("segment", [1, 3])
def test_nested_reverse_matches_frozen_grid_noise_finite_differences(method, controller, segment):
    rhs, collision = kernels()
    opts = settings(method, controller, segment)
    p = jnp.array([.4, .2, .1, 0., 1.2])
    def cost(p):
        return objective(solve_split(rhs, collision, p[4:5], p[:4], **opts, adjoint="accepted"))
    value, gradient = jax.jit(jax.value_and_grad(cost))(p)
    out, tape = jax.jit(lambda p: solve_split(rhs, collision, p[4:5], p[:4], **opts, _record_ad=True))(p)
    def frozen(p):
        saved, final = replay_accepted_split(rhs, collision, p[4:5], p[:4], tape, opts['ts'], method=method)
        return jnp.sum(final**2)+jnp.sum(saved*jnp.linspace(.1, .7, len(opts['ts']))[:, None])
    frozen = jax.jit(frozen)
    eps = 1e-5
    finite = np.array([(frozen(p+row*eps)-frozen(p-row*eps))/(2*eps) for row in np.eye(len(p))])
    np.testing.assert_allclose(value, objective(out), rtol=2e-14)
    np.testing.assert_allclose(gradient, finite, rtol=2e-7, atol=2e-8)
    assert gradient[3] == 0 and np.all(np.abs(np.asarray(gradient)[[0,1,2,4]]) > 1e-5)
    assert out.stats.rejected > 0 and out.stats.orbit_rejected > 0


def test_orbit_event_skips_kick_and_freezes_root_time():
    def cost(p):
        out = solve_split(lambda t,y,a: jnp.ones_like(y)*p[0],
            lambda t,y,h,dw,a: CollisionStep(y+p[1], jnp.array(0.), jnp.array(0.)),
            jnp.zeros(1), t1=1., dt0=1., ts=jnp.array([0., .8, 1.]),
            key=jax.random.key(2), noise_dim=1, event=lambda t,y,a: .5+p[2]-y[0],
            adjoint="accepted", max_steps=8, max_orbit_steps=16,
            adjoint_checkpoint_steps=2, adjoint_orbit_checkpoint_steps=2)
        return out.y[0]+jnp.sum(out.ys)
    value, gradient = jax.jit(jax.value_and_grad(cost))(jnp.array([2., .3, 0.]))
    np.testing.assert_allclose(value, 1.5, atol=2e-13)
    np.testing.assert_allclose(gradient, [.75, 0., 0.], atol=2e-12)


def test_kick_event_and_initial_event_hold_tail_saves():
    def cost(p, boundary):
        out = solve_split(lambda t,y,a: jnp.zeros_like(y),
            lambda t,y,h,dw,a: CollisionStep(y+p*h, jnp.array(0.), jnp.array(0.)),
            jnp.zeros(1), t1=1., dt0=.5, ts=jnp.array([0., .8, 1.]),
            key=jax.random.key(2), noise_dim=1, event=lambda t,y,a: boundary-y[0],
            adaptive=False, adjoint="accepted", max_steps=8, max_orbit_steps=16)
        return out.y[0]+jnp.sum(out.ys)
    value, gradient = jax.jit(jax.value_and_grad(lambda p: cost(p, .3)))(.7)
    np.testing.assert_allclose([value, gradient], [1.05, 1.5], atol=2e-13)
    initial = lambda y: jnp.sum(solve_split(lambda t,y,a:y,
        lambda t,y,h,dw,a: CollisionStep(y,jnp.array(0.),jnp.array(0.)),
        y,t1=1.,dt0=.1,ts=jnp.array([0.,.4,1.]),key=jax.random.key(0),noise_dim=1,
        event=lambda t,y,a:-1.,adjoint="accepted",max_steps=8,max_orbit_steps=8).ys)
    np.testing.assert_array_equal(jax.jit(jax.grad(initial))(jnp.array([2.])), [3.])


def test_event_and_interval_limit_keep_original_args_and_zero_control_gradients():
    def cost(p):
        out=solve_split(lambda t,y,a:jnp.ones_like(y)*a[0],
            lambda t,y,h,dw,a:CollisionStep(y+a[3]*h,jnp.array(0.),jnp.array(0.)),
            jnp.zeros(1),p,t1=1.,dt0=1.,ts=jnp.array([0.,.8,1.]),
            key=jax.random.key(2),noise_dim=1,adaptive=False,max_steps=8,max_orbit_steps=16,
            event=lambda t,y,a:a[1]-y[0],step_limit=lambda t,y,a:a[2],adjoint="accepted")
        return out.y[0]+jnp.sum(out.ys)
    value,gradient=jax.jit(jax.value_and_grad(cost))(jnp.array([2.,.5,.2,.1]))
    np.testing.assert_allclose(value,1.5,atol=2e-13)
    np.testing.assert_allclose(gradient,[.72,0.,0.,.6],atol=2e-12)


def test_singular_error_only_expression_cannot_poison_physical_gradient():
    def cost(p):
        out=solve_split(lambda t,y,a:a[0]*y,
            lambda t,y,h,dw,a:CollisionStep(y+a[1]*dw,jnp.sqrt(a[2])*h,jnp.array(0.)),
            jnp.ones(1),p,t1=.3,dt0=.1,ts=jnp.array([.3]),key=jax.random.key(4),
            noise_dim=1,max_steps=8,max_orbit_steps=16,adjoint="accepted")
        return jnp.sum(out.y)
    gradient=jax.jit(jax.grad(cost))(jnp.array([.4,.2,0.]))
    assert np.isfinite(gradient).all()
    assert gradient[2] == 0 and np.all(np.abs(gradient[:2])>1e-5)


@pytest.mark.parametrize("model", ["gc","lorentz"])
def test_zero_diffusion_maps_have_identity_state_gradient(model):
    from essos.collisions import (CollisionCoefficients, guiding_center_collision_step,
        guiding_center_spatial_increment, particle_collision_step)
    state=jnp.array([2.,.3]) if model == "gc" else jnp.array([2.,.3,.5])
    weights=jnp.arange(1,len(state)+1,dtype=state.dtype)
    def cost(state):
        # Coefficients vanish identically, as in a vacuum, but their original
        # rate formulas still contain speed/position-dependent expressions.
        zero=0*jnp.sum(state)
        c=CollisionCoefficients(*([zero]*6))
        if model == "gc":
            result=guiding_center_collision_step(state[0],state[1],.1,jnp.array([.2,-.1]),c).y
            spatial=guiding_center_spatial_increment(state[1],jnp.array([.1,.2,.3]),c,
                jnp.array([0.,0.,1.]),2.)
            return jnp.dot(result,weights)+jnp.sum(spatial)
        return jnp.dot(particle_collision_step(state,.1,jnp.array([.2,-.1,.1]),c).y,weights)
    value,gradient=jax.jit(jax.value_and_grad(cost))(state)
    np.testing.assert_allclose(value,jnp.dot(state,weights),atol=1e-14)
    np.testing.assert_allclose(gradient,weights,atol=1e-14)


@pytest.mark.parametrize("outer,inner,message", [(1,256,'accepted split AD capacity'), (64,1,'accepted orbit AD capacity')])
def test_capacity_overflow_preserves_forward_and_rejects_gradient(outer, inner, message):
    rhs, collision = kernels()
    p = jnp.array([.4,.2,.1,0.])
    opts = settings()
    expected = solve_split(rhs,collision,jnp.array([1.2]),p,**opts)
    opts.update(adjoint_max_accepted_steps=outer, adjoint_max_orbit_steps=inner)
    def run(p):
        return solve_split(rhs,collision,jnp.array([1.2]),p,**opts,adjoint="accepted")
    out = jax.jit(run)(p)
    for a,b in zip(jax.tree.leaves(out),jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(Exception,match=message):
        jax.jit(jax.grad(lambda p:objective(run(p))))(p).block_until_ready()


def test_failure_and_zero_span_contract():
    def bad(p):
        return solve_split(lambda t,y,a:p*y,
            lambda t,y,h,dw,a:CollisionStep(y,jnp.array(0.),jnp.array(0.)),
            jnp.ones(1),t1=1.,dt0=.1,ts=jnp.array([1.]),key=jax.random.key(0),
            noise_dim=1,max_steps=1,max_orbit_steps=8,adaptive=False,adjoint="accepted").y[0]
    with pytest.raises(Exception,match="successful/event"):
        jax.jit(jax.grad(bad))(.3).block_until_ready()
    empty = lambda y: jnp.sum(solve_split(lambda t,y,a:y,
        lambda t,y,h,dw,a:CollisionStep(y,jnp.array(0.),jnp.array(0.)),
        y,t1=0.,dt0=.1,ts=jnp.array([0.,0.]),key=jax.random.key(0),noise_dim=1,
        max_steps=1,max_orbit_steps=2,adjoint="accepted").ys)
    np.testing.assert_array_equal(jax.jit(jax.grad(empty))(jnp.array([2.])), [2.])


def test_vmap_heterogeneous_and_zero_length_paths():
    def cost(p,horizon,key,adjoint):
        # Inactive vmapped lanes can see zero-filled checkpoint workspaces.
        # sqrt has a singular derivative there; their cotangents must still
        # remain zero rather than contaminating active particles.
        return solve_split(lambda t,y,a:p*jnp.sqrt(y),
            lambda t,y,h,dw,a:CollisionStep(y+.03*jnp.sqrt(y)*dw,jnp.array(0.),jnp.array(0.)),
            jnp.ones(1),t1=horizon,dt0=.1,ts=jnp.empty(0),key=key,noise_dim=1,
            adaptive=False,adjoint=adjoint,max_steps=32,max_orbit_steps=16,
            adjoint_checkpoint_steps=3,adjoint_orbit_checkpoint_steps=2,
            adjoint_max_orbit_steps=64).y[0]
    p,times=jnp.array([.2,.4,.6]),jnp.array([0.,.3,.7])
    keys=jax.random.split(jax.random.key(24),3)
    evaluate=jax.jit(jax.vmap(lambda p,t,k:cost(p,t,k,"none")))
    gradient=jax.jit(jax.vmap(jax.grad(lambda p,t,k:cost(p,t,k,"accepted"))))(p,times,keys)
    eps=1e-5
    np.testing.assert_allclose(gradient,(evaluate(p+eps,times,keys)-evaluate(p-eps,times,keys))/(2*eps),
        rtol=2e-6,atol=1e-9)
    assert gradient[0] == 0


def test_schedule_allocation_is_flat_and_independently_bounded():
    rhs,collision=kernels()
    opts=settings()
    opts.update(adjoint_max_accepted_steps=12,adjoint_max_orbit_steps=256)
    out,tape=solve_split(rhs,collision,jnp.array([1.2]),jnp.array([.4,.2,.1,0.]),**opts,_record_ad=True)
    assert out.status == SplitStatus.SUCCESS
    assert tape.times.shape == (12,2) and tape.orbit_times.shape == (256,2)
    assert tape.states.shape == (4,1) and tape.increments.shape == (12,1)


@pytest.mark.parametrize("name,value", [("adjoint","bad"),("adjoint_checkpoint_steps",True),
    ("adjoint_orbit_checkpoint_steps",0),("adjoint_max_accepted_steps",0),
    ("adjoint_max_orbit_steps",1.2)])
def test_invalid_ad_configuration(name,value):
    opts=settings()
    opts[name]=value
    rhs,collision=kernels()
    with pytest.raises(ValueError,match=name):
        solve_split(rhs,collision,jnp.array([1.2]),jnp.array([.4,.2,.1,0.]),**opts)


@pytest.mark.parametrize("controller", ["paper", "candidate"])
@pytest.mark.parametrize("method", ["dopri5", "dopri8"])
def test_native_gc_milstein_metadata_and_lookahead_gradients(controller, method):
    from essos.collisions import CollisionCoefficients, guiding_center_collision_step
    def rhs(t,y,p):
        return p[0]*jnp.array([jnp.sin(7*t), .1*jnp.cos(t)])
    def collision(t,y,h,dw,p):
        c = CollisionCoefficients(.2*p[1], .3*p[1], .03*p[1], -.1*p[1], -.01*p[1], 0.)
        return guiding_center_collision_step(y[0],y[1],h,dw,c,tolerance=.001,
            speed_floor=.05,return_error_model=(controller == "candidate"))
    opts = dict(t1=.5,dt0=.5,dtmax=.5,ts=jnp.array([0.,.2,.5]),key=jax.random.key(14),
        noise_dim=2,method=method,collision_controller=controller,
        max_steps=128,max_orbit_steps=64,adjoint_max_accepted_steps=128,
        adjoint_max_orbit_steps=512,adjoint_checkpoint_steps=3,adjoint_orbit_checkpoint_steps=2)
    p, y0 = jnp.array([.2,1.]),jnp.array([2.,.3])
    def cost(p):
        return objective(solve_split(rhs,collision,y0,p,**opts,adjoint="accepted"))
    value,gradient=jax.jit(jax.value_and_grad(cost))(p)
    normal,tape=jax.jit(lambda p:solve_split(rhs,collision,y0,p,**opts,_record_ad=True))(p)
    def frozen(p):
        saved,final=replay_accepted_split(rhs,collision,y0,p,tape,opts['ts'],method=method)
        return jnp.sum(final**2)+jnp.sum(saved*jnp.linspace(.1,.7,3)[:,None])
    frozen=jax.jit(frozen)
    eps=1e-5
    finite=np.array([(frozen(p+row*eps)-frozen(p-row*eps))/(2*eps) for row in np.eye(2)])
    assert normal.status == SplitStatus.SUCCESS
    np.testing.assert_allclose(value,objective(normal),rtol=2e-13)
    np.testing.assert_allclose(gradient,finite,rtol=2e-6,atol=2e-8)
    assert np.all(np.abs(gradient)>1e-5)


@pytest.mark.parametrize("model,spatial", [("GuidingCenter",False),("GuidingCenter",True),("FullOrbit",False)])
def test_production_trace_field_density_temperature_and_birth_gradients(model, spatial):
    from essos.background_species import BackgroundSpecies
    from essos.dynamics import Particles
    from essos.fields import ExternalField
    from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q
    tmax=1e-6 if model == "GuidingCenter" else 2e-8
    def cost(p,adjoint):
        field=ExternalField(lambda point:jnp.stack([0*point[...,0],0*point[...,0],
            2*p[0]*(1+.05*point[...,0])],axis=-1))
        birth=Particles(jnp.array([[.2,0.,.1],[.25,.03,.11]]),mass=M,charge=Q,
            energy=2000*Q*jnp.exp(p[3]),initial_vparallel_over_v=jnp.array([p[4],.4]),
            field=field if model == "FullOrbit" else None,phase_angle_full_orbit=.7)
        species=BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),
            jnp.array([1e20*jnp.exp(p[1])]),jnp.array([1000*jnp.exp(p[2])]))
        out=trace_split(field,birth,species,model=model,tmax=tmax,timestep=tmax/4,
            times=jnp.linspace(0.,tmax,4),adaptive=False,orbit_adaptive=False,
            spatial_diffusion=spatial,max_steps=32,max_orbit_steps=8,
            adjoint=adjoint,adjoint_checkpoint_steps=2,adjoint_orbit_checkpoint_steps=2,
            adjoint_max_accepted_steps=32,adjoint_max_orbit_steps=64)
        xyz=out.ys[...,:3]
        velocity=out.ys[...,3]/1e6
        metric=jnp.mean(xyz**2)+jnp.mean(.3*velocity+velocity**2)
        if model == "GuidingCenter":
            metric+=jnp.mean(out.ys[...,4]/(2000*Q))*.2
        else:
            metric+=jnp.mean(out.ys[...,4:6]**2)/1e12
        return metric
    p=jnp.array([1.,0.,0.,0.,.6])
    value,gradient=jax.jit(jax.value_and_grad(lambda p:cost(p,"accepted")))(p)
    # With both grids fixed, the same production Gaussian/bridge samples are
    # used by every evaluation: this FD changes physical parameters only.
    primal=jax.jit(lambda p:cost(p,"none"))
    eps=1e-5
    finite=np.array([(primal(p+row*eps)-primal(p-row*eps))/(2*eps) for row in np.eye(5)])
    np.testing.assert_allclose(value,primal(p),rtol=2e-13)
    np.testing.assert_allclose(gradient,finite,rtol=3e-5,atol=3e-9)
    assert np.isfinite(gradient).all() and np.all(np.abs(gradient)>1e-8)


@pytest.mark.parametrize("model", ["GuidingCenter", "FullOrbit"])
def test_production_adaptive_trace_record_replay_and_jvp(model):
    from essos.background_species import BackgroundSpecies
    from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q
    from essos.dynamics import Particles
    from essos.fields import ExternalField

    horizon = 1e-6 if model == "GuidingCenter" else 2e-8
    options = dict(model=model, tmax=horizon, timestep=horizon/3,
        collision_dtmax=horizon/2, n_save=5, max_steps=32, max_orbit_steps=32,
        adjoint_max_accepted_steps=32, adjoint_max_orbit_steps=128)
    def run(p, **mode):
        field = ExternalField(lambda x:jnp.stack([0*x[...,0],0*x[...,0],
            2*p[0]*(1+.05*x[...,0])],axis=-1))
        birth = Particles(jnp.array([[.2,0.,.1]]),mass=M,charge=Q,energy=2000*Q,
            initial_vparallel_over_v=jnp.array([.6]),field=field if model == "FullOrbit" else None,
            phase_angle_full_orbit=.7)
        background = BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),
            jnp.array([1e20*jnp.exp(p[1])]),jnp.array([1000.]))
        return trace_split(field,birth,background,**options,**mode)
    def metric(states):
        return jnp.mean(states[...,:3]**2)+jnp.mean((states[...,3]/1e6)**2)
    p = jnp.array([1.,0.])
    normal,tapes = jax.jit(lambda p:run(p,_record_ad=True))(p)
    def frozen(p):
        states,final = run(p,_replay_tapes=tapes)
        return metric(states), (states,final)
    frozen = jax.jit(frozen)
    _,(states,final) = frozen(p)
    np.testing.assert_allclose(states,normal.ys,rtol=3e-13,atol=1e-14)
    np.testing.assert_allclose(final,normal.y,rtol=3e-13,atol=1e-14)
    gradient = jax.jit(jax.grad(lambda p:metric(run(p,adjoint="accepted").ys)))(p)
    direction = jnp.array([.3,-.2])
    tangent = jax.jit(lambda p:jax.jvp(lambda x:frozen(x)[0],(p,),(direction,))[1])(p)
    eps = 1e-5
    finite = (frozen(p+eps*direction)[0]-frozen(p-eps*direction)[0])/(2*eps)
    np.testing.assert_allclose(gradient@direction,tangent,rtol=3e-8,atol=1e-11)
    np.testing.assert_allclose(gradient@direction,finite,rtol=3e-5,atol=1e-10)
