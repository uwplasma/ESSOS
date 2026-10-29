"""Small accepted-path oracles; no long tracing or Monte Carlo runs in CI."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from essos.rk import solve_rk
from essos.split import solve_split, CollisionStep, move_in_cartesian
from essos.native_ad import replay_accepted_split, _OrbitRecorder, replay_accepted


def kernels():
    def rhs(t,y,p):
        return p[0]*jnp.sin(y+12*t)
    def kick(t,y,h,dw,p):
        return CollisionStep(y-p[2]*h*y+p[1]*(.7+.1*y)*dw,
            (h/(.04*jnp.exp(p[3])))**2,jnp.array(0.))
    return rhs,kick


def settings(method='dopri8', segment=3):
    return dict(t1=.4,dt0=.3,ts=jnp.array([0.,0.,.13,.4]),
        key=jax.random.key(71),noise_dim=1,method=method,rtol=1e-8,atol=1e-8,
        max_steps=64,max_orbit_steps=32,adjoint_checkpoint_steps=segment,
        adjoint_orbit_checkpoint_steps=segment,adjoint_max_accepted_steps=32,
        adjoint_max_orbit_steps=128)


def cost(out):
    return jnp.sum(out.ys**2)+.3*jnp.sum(out.y**2)


@pytest.mark.parametrize('method',['dopri5','dopri8'])
def test_split_reverse_matches_frozen_jvp_with_rejections(method):
    rhs,kick=kernels();p=jnp.array([.8,.1,.2,0.]);y=jnp.array([.7]);o=settings(method)
    out,tape=jax.jit(lambda p:solve_split(rhs,kick,y,p,_record_ad=True,**o))(p)
    assert out.status==0 and out.stats.rejected>0 and out.stats.orbit_rejected>0
    def frozen(p):
        saved,terminal=replay_accepted_split(rhs,kick,y,p,tape,o['ts'],method=method)
        return jnp.sum(saved**2)+.3*jnp.sum(terminal**2)
    value,gradient=jax.jit(jax.value_and_grad(lambda p:cost(solve_split(rhs,kick,y,p,adjoint='accepted',**o))))(p)
    direction=jnp.array([.2,-.3,.4,.5])
    expected,tangent=jax.jit(lambda p,d:jax.jvp(frozen,(p,),(d,)))(p,direction)
    np.testing.assert_allclose(value,expected,rtol=3e-12)
    np.testing.assert_allclose(jnp.dot(gradient,direction),tangent,rtol=3e-10,atol=1e-11)
    assert gradient[3]==0 # acceptance-only parameter is frozen


@pytest.mark.parametrize('method',['dopri5','dopri8'])
def test_orbit_dense_output_and_checkpoint_sizes(method):
    rhs=lambda t,y,p:p*jnp.sin(y+t)
    y=jnp.array([.7]);p=jnp.array(.3)
    o=dict(t1=.4,dt0=.12,ts=jnp.array([0.,.031,.17,.4]),method=method,max_steps=32)
    out,tape=jax.jit(lambda p:solve_rk(rhs,y,p,_recorder=_OrbitRecorder(32,3),**o))(p)
    def frozen(p):
        saved,terminal=replay_accepted(rhs,y,p,tape,o['ts'],method=method)
        return jnp.sum(saved**2)+.3*jnp.sum(terminal**2)
    expected,tangent=jax.jvp(frozen,(p,),(jnp.ones_like(p),))
    for segment in (1,3,8):
        value,gradient=jax.jit(jax.value_and_grad(lambda p:cost(solve_rk(rhs,y,p,
            adjoint='accepted',adjoint_checkpoint_steps=segment,**o))))(p)
        np.testing.assert_allclose(value,cost(out),rtol=3e-13)
        np.testing.assert_allclose(value,expected,rtol=3e-13)
        np.testing.assert_allclose(gradient,tangent,rtol=2e-10)


def test_chart_kick_has_exact_inverse_derivative_including_gauge():
    def position(y):
        u,w=y[0]*jnp.cos(y[5])-y[1]*jnp.sin(y[5]),y[1]*jnp.cos(y[5])+y[0]*jnp.sin(y[5])
        return jnp.array([(3+u)*jnp.cos(y[2]),(3+u)*jnp.sin(y[2]),w])
    y=jnp.array([.2,.1,.3,2.,.4,.2]);delta=jnp.array([.03,-.02,.04])
    def fn(y,delta):return position(move_in_cartesian(y,delta,position,implicit=True))
    by,bd=jax.jit(jax.jacrev(fn,argnums=(0,1)))(y,delta)
    np.testing.assert_allclose(bd,jnp.eye(3),atol=5e-14)
    np.testing.assert_allclose(by,jax.jacfwd(position)(y),atol=8e-14)
    np.testing.assert_array_equal(move_in_cartesian(y,delta,position)[3:],y[3:])


def test_split_events_hold_saves_and_skip_collision_derivatives():
    rhs=lambda t,y,p:jnp.ones_like(y)*p[0]
    kick=lambda t,y,h,dw,p:CollisionStep(y+p[1]*dw,jnp.array(0.),jnp.array(0.))
    o=dict(t1=1.,dt0=.5,ts=jnp.array([0.,.7,1.]),key=jax.random.key(1),noise_dim=1,
        event=lambda t,y,p:.2-y[0],adaptive=False,orbit_adaptive=False,max_steps=8,max_orbit_steps=8)
    value,g=jax.jit(jax.value_and_grad(lambda p:cost(solve_split(rhs,kick,jnp.zeros(1),p,
        adjoint='accepted',**o))))(jnp.array([1.,.3]))
    assert np.isfinite(value) and np.isfinite(g).all() and g[1]==0
    # Event fraction is frozen: endpoint sensitivity is the recorded crossing time.
    np.testing.assert_allclose(g[0],2*.2*.2*2.3,atol=2e-13)


def test_failed_or_capacity_limited_reverse_raises():
    rhs,kick=kernels();o=settings();o.update(adaptive=False,orbit_adaptive=False,dt0=.1,orbit_dt0=.01,
        adjoint_max_orbit_steps=1)
    f=jax.jit(jax.grad(lambda p:cost(solve_split(rhs,kick,jnp.array([.7]),p,adjoint='accepted',**o))))
    with pytest.raises(Exception,match='capacity exceeded'):
        jax.block_until_ready(f(jnp.array([.8,.1,.2,0.])))


def test_closure_parameters_initial_state_and_custom_jvp_are_preserved():
    @jax.custom_jvp
    def nonlinear(y):return jnp.sin(y)
    @nonlinear.defjvp
    def tangent(primals,tangents):
        y,=primals;dy,=tangents
        return nonlinear(y),3*jnp.cos(y)*dy
    o=dict(t1=.2,dt0=.1,ts=jnp.array([0.,.13,.2]),adaptive=False,max_steps=4)
    def run(p,record=False):
        rhs=lambda t,y,a:p[0]*nonlinear(y)
        return solve_rk(rhs,p[1:],adjoint='none' if record else 'accepted',
            _recorder=_OrbitRecorder(4,2) if record else None,**o)
    p=jnp.array([.3,.7]);out,tape=jax.jit(lambda p:run(p,True))(p)
    def frozen(p):
        saved,final=replay_accepted(lambda t,y,a:p[0]*nonlinear(y),p[1:],None,tape,o['ts'],method='dopri5')
        return jnp.sum(saved**2)+.3*jnp.sum(final**2)
    gradient=jax.jit(jax.grad(lambda p:cost(run(p))))(p)
    _,expected=jax.jvp(frozen,(p,),(jnp.array([.2,.4]),))
    np.testing.assert_allclose(gradient@jnp.array([.2,.4]),expected,rtol=3e-11)


def test_gpu_recorder_interpretation_keeps_primal_noise_and_sparse_nodes():
    from essos.split import _prepare_split_kernel, scalar
    def factory(refs,y0):
        zero=scalar(y0,7)
        def rhs(y):return jnp.where(jnp.arange(8)==0,zero+1,zero)
        def kick(y,h,w):return y+jnp.where(jnp.arange(8)==3,scalar(w,0),zero),zero,zero
        return rhs,kick,lambda y,e,v,b:zero,lambda y:zero+1
    options=dict(adaptive=False,dt0=.1,dtmax=.1,speed=1.,orbit_dt0=.025,
        max_steps=8,max_orbit_steps=8,interpret=True,adjoint_max_orbit_steps=16,
        adjoint_orbit_checkpoint_steps=2,adjoint_max_accepted_steps=8)
    times=jnp.array([0.,.1,.2]);y=jnp.zeros(8);key=jax.random.PRNGKey(3)
    ordinary=jax.jit(_prepare_split_kernel((),factory,times,**options))(y,key)
    recorded,tape=jax.jit(_prepare_split_kernel((),factory,times,record=True,**options))(y,key)
    for left,right in zip(ordinary,recorded):np.testing.assert_array_equal(left,right)
    assert tape.count==2 and tape.orbit_count==8 and tape.checkpoint_count==4
    assert tape.initial_saves==1
    np.testing.assert_array_equal(tape.offsets[:3],[0,4,8])
    np.testing.assert_allclose(tape.orbit_states[:4,0],[0.,.05,.1,.15],atol=2e-15)
    rhs=lambda t,y,a:jnp.where(jnp.arange(8)==0,1.,0.)
    kick=lambda t,y,h,dw,a:CollisionStep(y.at[3].add(dw[0]),jnp.array(0.),jnp.array(0.))
    saved,final=jax.jit(lambda:replay_accepted_split(rhs,kick,y,None,tape,times))()
    np.testing.assert_allclose(saved,recorded[0],atol=2e-15)
    np.testing.assert_allclose(final,recorded[1],atol=2e-15)


def test_exact_coil_field_transpose_includes_jacobian_sensitivities():
    from tests.test_native_orbit import coil_field
    from essos.rk import _coil_geometry
    from essos.native_ad import _field_transpose, _reverse_field_values
    field=coil_field();arrays,nc,nq=_coil_geometry(field)
    x=tuple(jnp.asarray(v) for v in (1.1,.03,.02))
    b=tuple(jnp.asarray(v) for v in (.2,-.3,.4))
    bj=tuple(tuple(jnp.asarray(v) for v in row) for row in ((.2,.4,.3),(-.1,.5,.2),(.4,.3,-.2)))
    for gc in (False,True):
        seed=(b,bj if gc else None)
        expected=jax.vjp(lambda p,g:_reverse_field_values(g,p,nc,nq,gc),x,arrays)[1](seed)
        actual=_field_transpose(arrays,x,b,seed[1],nc,nq,gc)
        for left,right in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
            np.testing.assert_allclose(left,right,rtol=5e-12,atol=3e-12)


def test_gc_spatial_kick_gradient_uses_new_position_and_magnetic_moment():
    from tests.test_native_orbit import coil_field
    from tests.test_native_split import bath
    from essos.constants import ALPHA_PARTICLE_MASS as mass, ALPHA_PARTICLE_CHARGE as charge
    from essos.rk import _coil_geometry
    from essos.native_ad import _kick, coefficient_graph
    from essos.split import (guiding_center_collision_step, collision_coefficients,
        guiding_center_spatial_increment)
    field=coil_field();arrays,nc,nq=_coil_geometry(field);background=bath()
    graph=coefficient_graph(background,False);constants=tuple(graph.consts)
    y=jnp.array([1.1,.03,.02,2e5,1e-15,0.,0.,0.]);h=1e-9
    dw=jnp.array([.2,-.3,.4,-.5,.3])*jnp.sqrt(h)
    def production(y):
        B=field.AbsB(y[:3]);speed=jnp.sqrt(y[3]**2+2*y[4]*B/mass)
        c=collision_coefficients(background,mass,charge,speed,y[:3]);pitch=y[3]/speed
        step=guiding_center_collision_step(speed,pitch,h,dw[-2:],c,speed_floor=1.)
        dx=guiding_center_spatial_increment(pitch,dw[:3],c,field.B(y[:3])/B,charge*B/mass)
        point=y[:3]+dx;v,xi=step.y
        return y.at[:3].set(point).at[3].set(v*xi).at[4].set(mass*v*v*(1-xi*xi)/(2*field.AbsB(point)))
    def native(y):
        return jnp.asarray(_kick(tuple(y),arrays,constants,jnp.asarray(1.),h,tuple(dw),graph,nc,nq,True,spatial_diffusion=True))
    scale=jnp.array([1.,1.,1.,1e5,1e-15,1.,1.,1.]);direction=scale*jnp.array([.1,.2,-.1,.3,.2,0.,0.,0.])
    left,dl=jax.jvp(native,(y,),(direction,));right,dr=jax.jvp(production,(y,),(direction,))
    np.testing.assert_allclose(left/scale,right/scale,rtol=5e-12,atol=2e-13)
    np.testing.assert_allclose(dl/scale,dr/scale,rtol=5e-11,atol=2e-12)


def test_ad_configuration_rejects_invalid_modes_and_capacities():
    from essos.split import trace_split
    for value in (0,-1,True,1.5):
        with pytest.raises(ValueError):_OrbitRecorder(8,value)
    for options in (dict(adjoint='bad'),dict(reverse_execution='bad'),
            dict(adjoint='accepted',adjoint_orbit_checkpoint_steps=0),
            dict(adjoint='accepted',reverse_execution='independent'),
            dict(adjoint='accepted',_replay_tapes=object())):
        with pytest.raises(ValueError):trace_split(None,None,None,tmax=.1,timestep=.1,**options)


def test_native_tracing_api_and_array_objectives_receive_physical_gradients():
    from types import SimpleNamespace
    from tests.test_native_orbit import UniformField
    from essos.dynamics import Particles,Tracing
    from essos.particle_losses import (mean_final_radius,classifier_loss_fraction,
        classifier_loss_location,classifier_energy_cost)
    def objective(v):
        p=Particles(jnp.zeros((1,3)),initial_vparallel_over_v=[.2],mass=1.,charge=1.,energy=.5,
            initial_xyz_fullorbit=jnp.zeros((1,3)),initial_vxvyvz=jnp.array([[v,0.,.2]]))
        out=Tracing(field=UniformField(),particles=p,model='FullOrbitAdaptative',integrator='essos_dopri8',
            native_adjoint='accepted',maxtime=.2,timestep=.1,times_to_trace=3,max_steps=16,progress=False)
        return jnp.sum(out.trajectories[0,-1,:3]**2)
    gradient=jax.jit(jax.grad(objective))(jnp.array(.3))
    np.testing.assert_allclose(gradient,8*.3*np.sin(.1)**2,rtol=2e-10)
    states=jnp.array([[[0.,0.,0.,.2,.3,.4],[.2,0.,0.,.2,.3,.4]],
                      [[0.,0.,0.,.2,.3,.4],[.3,0.,0.,.2,.3,.4]]])
    boundary=SimpleNamespace(evaluate_xyz=lambda x:.5-x[0])
    np.testing.assert_allclose(mean_final_radius(states,0.,0.),.25,atol=2e-12)
    np.testing.assert_allclose(classifier_loss_fraction(states,boundary,softness=4.),
        jnp.mean(jax.nn.sigmoid(jnp.array([-1.2,-.8]))),rtol=1e-14)
    distance=classifier_loss_location(states,jnp.array([1.,0.,0.]),boundary)
    assert .7<distance<1.
    energy=classifier_energy_cost(states,UniformField(),1.,1.,'FullOrbit',boundary,kind='perpendicular_jax9')
    np.testing.assert_allclose(energy,.065,rtol=2e-7)
    derivative=jax.grad(lambda y:classifier_energy_cost(y,UniformField(),1.,1.,'FullOrbit',boundary))(states)
    assert np.isfinite(derivative).all() and np.any(np.asarray(derivative[...,3:])!=0)


def test_profile_and_collision_coefficient_derivatives_remain_active():
    from essos.background_species import BackgroundSpecies
    from essos.constants import ALPHA_PARTICLE_MASS as mass, ALPHA_PARTICLE_CHARGE as charge
    from essos.split import collision_coefficients,guiding_center_collision_step
    o=dict(t1=2e-7,dt0=1e-7,ts=jnp.array([0.,2e-7]),key=jax.random.key(33),
        noise_dim=2,adaptive=False,orbit_adaptive=False,orbit_dt0=5e-8,
        method='dopri5',max_steps=8,max_orbit_steps=8,adjoint_max_orbit_steps=16)
    def problem(p):
        bath=BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),
            jnp.exp(p[0])*jnp.array([[1e20,2e20]]),jnp.array([[1000.,1200.]]),
            radial_grid=jnp.array([0.,1.]),coulomb_log_model='impact')
        def rhs(t,y,a):return jnp.array([.02*y[1]*y[2]/4e5,0.,0.])
        def kick(t,y,h,dw,a):
            coefficients=collision_coefficients(bath,mass,charge,y[1],jnp.array([y[0],0.,0.]))
            step=guiding_center_collision_step(y[1],y[2],h,dw,coefficients,speed_floor=1.)
            return step._replace(y=jnp.concatenate((y[:1],step.y)))
        return rhs,kick,jnp.array([.3+p[1],4e5*jnp.exp(p[2]),.4])
    p=jnp.array([.1,.02,.03]);rhs,kick,y=problem(p)
    _,tape=jax.jit(lambda:solve_split(rhs,kick,y,_record_ad=True,**o))()
    metric=lambda y:y[0]+y[1]/4e5+y[2]
    def cost(p):
        rhs,kick,y=problem(p)
        return metric(solve_split(rhs,kick,y,adjoint='accepted',**o).y)
    def frozen(p):
        rhs,kick,y=problem(p)
        return metric(replay_accepted_split(rhs,kick,y,None,tape,o['ts'],method='dopri5')[1])
    gradient=jax.jit(jax.grad(cost))(p);direction=jnp.array([.1,.2,.1])
    _,expected=jax.jit(lambda p:jax.jvp(frozen,(p,),(direction,)))(p)
    np.testing.assert_allclose(gradient@direction,expected,rtol=2e-10,atol=1e-12)
    assert abs(float(gradient[0]))>1e-8


@pytest.mark.parametrize('gc',[False,True])
def test_persistent_reverse_kernel_matches_complete_frozen_map_on_cpu(gc):
    from tests.test_native_orbit import coil_field
    from tests.test_native_split import bath
    from essos.rk import _coil_geometry
    from essos.native_ad import (_rhs_from_field,_reverse_field_values,_kick,physical_graphs,reverse_kernel)
    field=coil_field();geometry,nc,nq=_coil_geometry(field);background=bath();floor=jnp.array(1.)
    # Rate derivatives are checked above. Constant coefficients keep this
    # complete RK/kick transpose fixture small enough for CPU-only CI.
    diffusion=jnp.array(1e10)
    flag=jnp.array(True)
    coefficients=jax.make_jaxpr(lambda v:(jnp.where(flag,diffusion,0.),2*diffusion,jnp.array(0.),
        jnp.array(-1e8),jnp.array(0.),jnp.array(0.)))(jnp.array(4e5))
    graphs=physical_graphs(geometry,background,floor,nc,nq,gc,
        active_coefficients=False,active_floor=False,spatial_diffusion=gc,
        coefficient_data=(coefficients.jaxpr,coefficients.consts))
    n=5 if gc else 6
    initial=jnp.array([1.1,.03,.02,2e5,1e-15] if gc else [1.1,.03,.02,1e5,2e5,3e5])
    def rhs(t,y,p):
        state=tuple(jnp.pad(y,(0,8-n)))
        return jnp.asarray(_rhs_from_field(state,*_reverse_field_values(p,state[:3],nc,nq,gc),gc))[:n]
    def kick(t,y,h,dw,p):
        state=_kick(tuple(jnp.pad(y,(0,8-n))),p,tuple(coefficients.consts),floor,
            h,tuple(dw),coefficients,nc,nq,gc,spatial_diffusion=gc)
        return CollisionStep(jnp.asarray(state)[:n],jnp.array(0.),jnp.array(0.))
    times=jnp.array([0.,1e-9]);options=dict(t1=1e-9,dt0=1e-9,ts=times,key=jax.random.key(3),
        noise_dim=5 if gc else 3,adaptive=False,orbit_adaptive=False,max_steps=4,max_orbit_steps=4,
        adjoint_orbit_checkpoint_steps=1,_collision_state_indices=None if gc else (3,4,5))
    out,tape=jax.jit(lambda:solve_split(rhs,kick,initial,geometry,_record_ad=True,**options))()
    assert tape.count==1 and tape.orbit_count==1
    scale=jnp.array([1.,1.,1.,1e5,1e-15] if gc else [1.,1.,1.,1e5,1e5,1e5])
    saved_bar=jnp.ones((2,n))/scale;terminal_bar=.3/scale
    def frozen(y,p):
        from essos.rk import dopri8_step
        endpoint=dopri8_step(rhs,0.,y,1e-9,p).y
        terminal=kick(1e-9,endpoint,1e-9,tape.increments[0],p).y
        saved=jnp.stack((y,terminal))
        return jnp.sum(saved*saved_bar)+jnp.sum(terminal*terminal_bar)
    expected=jax.jit(lambda:jax.vjp(frozen,initial,geometry)[1](jnp.array(1.)))()
    actual=jax.jit(lambda:reverse_kernel(out.y,(geometry,tuple(coefficients.consts),floor),tape,
        saved_bar,terminal_bar,times,graphs[:3],gc=gc,segment_steps=1,spatial_diffusion=gc,interpret=True))()
    for left,right in zip(jax.tree.leaves(actual[:2]),jax.tree.leaves(expected)):
        np.testing.assert_allclose(left,right,rtol=2e-9,atol=2e-11)
    for value,bar in zip(coefficients.consts,actual[2]):
        if value.dtype==jnp.bool_:assert bar.dtype==jax.dtypes.float0


def test_qi_alm_line_heat_and_upper_bound_constraints():
    from types import SimpleNamespace
    from essos.particle_losses import classifier_line_loss, classifier_energy_cost
    from examples.coil_optimization.native_gc_alm import make_constraints, boundary_line
    from essos.surfaces import SurfaceRZFourier
    from tests.test_native_orbit import UniformField
    boundary=SimpleNamespace(evaluate_xyz=lambda x:-x[0])
    states=jnp.array([[[1.,.2,0.,.2,.3,.4],[2.,.4,0.,.2,.3,.4]],
                      [[.5,.3,0.,.2,.3,.4],[1.5,.7,0.,.2,.3,.4]]])
    probability=jax.nn.sigmoid(30*states[...,0]);total=probability.sum(axis=0)
    centroids=(states[...,:3]*probability[...,None]).sum(axis=0)/(total[:,None]+1e-8)
    expected=jnp.sum(jnp.linalg.norm(centroids[:,1:],axis=1)*(total/2)/(1+10*total/2))
    np.testing.assert_allclose(classifier_line_loss(states,jnp.zeros(3),jnp.array([1.,0.,0.]),boundary),expected,rtol=1e-12)
    energy=classifier_energy_cost(states,UniformField(),1.,1.,'FullOrbit',boundary)
    assert energy>0 and np.isfinite(jax.grad(lambda y:classifier_line_loss(y,jnp.zeros(3),
        jnp.array([1.,0.,0.]),boundary))(states)).all()
    torus=SurfaceRZFourier(jnp.array([3.,1.]),jnp.array([0.,1.]),1,1,0)
    point,direction=boundary_line(torus,0.,0.,'fixed_theta')
    np.testing.assert_allclose(point,[4.,0.,0.],atol=1e-14)
    np.testing.assert_allclose(direction,[0.,1.,0.],atol=1e-14)
    geometry=lambda x:dict(axis=x[0],length=jnp.array([2.])-x[1],curvature=jnp.array([1.])-x[1])
    options=SimpleNamespace(penalty=1.,bdotn_weight=0.)
    constraint=make_constraints(geometry,lambda x:x[0]**2,options)
    feasible=jnp.zeros(2);bad=jnp.array([0.,3.]);lp=constraint.init(feasible)
    assert float(constraint.loss(lp,feasible)[0])==0
    assert float(constraint.loss(lp,bad)[0])>0


def test_specialized_multispecies_rates_and_classifier_grid():
    from types import SimpleNamespace
    from jax.scipy.interpolate import RegularGridInterpolator
    from jax.experimental import pallas as pl
    from essos.background_species import BackgroundSpecies
    from essos.constants import (ELECTRON_MASS,PROTON_MASS,
        ALPHA_PARTICLE_MASS as mass,ALPHA_PARTICLE_CHARGE as charge)
    from essos.split import ClassifierStop,_coefficient_graph,collision_coefficients
    from essos.rk import _coil_geometry
    def buffers(p,direct):
        gamma=jnp.arange(45,dtype=float).reshape(3,5,3)*p[0]
        tangent=jnp.sin(gamma)*p[1];currents=jnp.array([1.,2.,3.])*p[2]
        if direct:return jnp.stack(_coil_geometry(SimpleNamespace(coils=SimpleNamespace(
            gamma=gamma,gamma_dash=tangent,currents=currents)))[0])
        return jnp.stack([jnp.pad(a[:,:,i].T,((0,3),(0,1))).reshape(-1)
            for a in (gamma,tangent*(currents*1e-7)[:,None,None]) for i in range(3)])
    for transform in (jax.jit,lambda f:jax.jit(jax.jacfwd(f))):
        np.testing.assert_allclose(transform(lambda p:buffers(p,True))(jnp.array([.1,.3,1e7])),
            transform(lambda p:buffers(p,False))(jnp.array([.1,.3,1e7])),rtol=1e-14,atol=1e-14)
    for n in (2,3):
        def rates(p,prepared):
            bath=BackgroundSpecies(n,jnp.array([ELECTRON_MASS/PROTON_MASS,2.,3.][:n]),
                jnp.array([-1.,1.,1.][:n]),jnp.full(n,4e20)*jnp.exp(p[0]),jnp.full(n,17800.))
            v=1.3e7*jnp.exp(p[1])
            if not prepared:return jnp.array(collision_coefficients(bath,mass,charge,v,jnp.zeros(3),full_orbit=True))
            graph=_coefficient_graph(bath,True)
            assert all(c.shape==() for c in graph.consts)
            return jnp.array(jax.core.eval_jaxpr(graph.jaxpr,graph.consts,v))
        p=jnp.array([.1,.02])
        for transform in (lambda f:jax.jit(f),lambda f:jax.jit(jax.jacfwd(f))):
            np.testing.assert_allclose(transform(lambda p:rates(p,True))(p),
                transform(lambda p:rates(p,False))(p),rtol=3e-11,atol=1e-5)
    for radii in (jnp.linspace(.8,2.,7),jnp.array([.8,.9,1.2,1.5,1.6,1.9,2.])):
        grid=(radii,jnp.linspace(0.,2*jnp.pi,9),jnp.linspace(-.8,.8,6))
        r,phi,z=jnp.meshgrid(*grid,indexing='ij')
        interpolation=RegularGridInterpolator(grid,(r+2*jnp.cos(phi)+.3*z*z)[...,None],fill_value=-1.)
        def evaluate(x):return interpolation(jnp.array([jnp.hypot(x[0],x[1]),
            jnp.mod(jnp.arctan2(x[1],x[0]),2*jnp.pi),x[2]]))[0,0]
        stop=ClassifierStop(SimpleNamespace(dist=interpolation,evaluate_xyz=evaluate),.2)
        data,level=stop._kernel_view()
        def sample(point):
            def kernel(y,*refs):refs[-1][()]=level(refs[:-1],y[...])
            return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((),jnp.float64),interpret=True)(point,*data)
        points=jnp.array([[1.2,0.,.1],[0.,1.4,-.3],[-1.2,-.2,.5],[.2,0.,0.],
                         [1.2,-1e-12,0.],[1.2,1e-12,0.]])
        padded=jnp.pad(points,((0,0),(0,5)))
        np.testing.assert_allclose(jax.jit(jax.vmap(sample))(padded),
            jax.vmap(lambda y:jnp.maximum(0.,evaluate(y[:3])+.2))(padded),rtol=1e-14,atol=2e-14)
        np.testing.assert_allclose(jax.vmap(stop.evaluate_xyz)(points),
            jax.vmap(evaluate)(points),rtol=1e-14,atol=2e-14)
        np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(stop.evaluate_xyz)))(points),
            jax.jit(jax.vmap(jax.grad(evaluate)))(points),rtol=2e-12,atol=2e-12)
    from tests.test_native_orbit import coil_field
    from essos.split import _prepare_split_field,guiding_center_collision_step
    field=coil_field();y=jnp.array([1.,.03,.01,-2e5,1e-16,0.,0.,0.])
    bath=BackgroundSpecies(2,jnp.array([ELECTRON_MASS/PROTON_MASS,2.]),
        jnp.array([-1.,1.]),jnp.full(2,4e20),jnp.full(2,17800.))
    def apply(floor):
        data,factory=_prepare_split_field(field,bath,gc=True,collision_tolerance=.001,
            speed_floor=floor,event=stop,state_size=5)
        def kernel(yr,*refs):
            _,kick,_,level=factory(refs[:-2],yr[...])
            refs[-2][...]=kick(yr[...],jnp.asarray(1e-9),jnp.zeros(8))[0]
            refs[-1][()]=level(yr[...])
        return pl.pallas_call(kernel,out_shape=(jax.ShapeDtypeStruct((8,),y.dtype),
            jax.ShapeDtypeStruct((),y.dtype)),interpret=True)(y,*data)
    B=field.AbsB(y[:3]);v=jnp.sqrt(y[3]**2+2*y[4]*B/mass)
    c=collision_coefficients(bath,mass,charge,v,y[:3])
    changed,level=jax.jit(apply)(jnp.asarray(1e4))
    speed,pitch=guiding_center_collision_step(v,y[3]/v,1e-9,jnp.zeros(2),c,speed_floor=1e4).y
    expected=y.at[3].set(speed*pitch).at[4].set(mass*speed**2*(1-pitch**2)/(2*B))
    np.testing.assert_allclose(changed,expected,rtol=3e-12,atol=1e-20)
    np.testing.assert_allclose(level,stop(0.,y),rtol=1e-14)
