"""Small forward contracts; GPU algebra is exercised on CPU via Pallas interpretation."""
from pathlib import Path
from types import SimpleNamespace
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from essos import rk
from essos.rk import OrbitErrorNorm

jax.config.update('jax_enable_x64', True)


def oscillator(t,y,args):
    return jnp.array([y[1],-y[0]])


@pytest.mark.parametrize('method', ['dopri5','dopri8'])
def test_tableau_fsal_and_standard_dense_match_diffrax(method):
    step_fn,interpolate,solver=((rk.rk45_step,rk.rk45_interpolate,diffrax.Dopri5()) if method=='dopri5'
                               else (rk.dopri8_step,rk.dopri8_interpolate,diffrax.Dopri8()))
    y=jnp.array([1.,.2]); h=.1
    step=step_fn(oscillator,0.,y,h)
    term=diffrax.ODETerm(oscillator)
    yn,error,dense,_,_=solver.step(term,0.,h,y,None,solver.init(term,0.,h,y,None),False)
    np.testing.assert_allclose(step.y,yn,rtol=0,atol=2e-15)
    np.testing.assert_allclose(step.error,error,rtol=0,atol=2e-15)
    reference=solver.interpolation_cls(t0=0.,t1=h,**dense)
    for fraction in (0.,.2,.7,1.):
        np.testing.assert_allclose(interpolate(y,h,step,fraction),reference.evaluate(fraction*h),atol=2e-14)
    cached=step_fn(oscillator,h,step.y,h,f0=step.stages[-1])
    for a,b in zip(cached,step_fn(oscillator,h,step.y,h)):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('method', ['dopri5','dopri8'])
def test_adaptive_control_rejects_and_is_independent_of_saves(method):
    for controller in ('i','pi'):
        opts=dict(t1=2.,dt0=2.,rtol=1e-10,atol=1e-10,method=method,controller=controller,
                  max_steps=128,record_steps=True)
        run=jax.jit(lambda saves:rk.solve_rk(oscillator,jnp.array([1.,0.]),ts=saves,**opts))
        a,b=run(jnp.array([0.,2.])),run(jnp.linspace(0,2,9))
        assert a.status==b.status==rk.RKStatus.SUCCESS and a.stats.rejected>0
        np.testing.assert_array_equal(a.stats,b.stats)
        np.testing.assert_array_equal(a.step_ts,b.step_ts)
        np.testing.assert_allclose(b.ys,jnp.stack((jnp.cos(b.ts),-jnp.sin(b.ts)),axis=1),atol=3e-8)
        loop=jax.jit(lambda:rk.solve_rk(oscillator,jnp.array([1.,0.]),ts=jnp.array([0.,2.]),
                                       stage_execution='loop',**opts))()
        np.testing.assert_allclose(a.y,loop.y,atol=2e-12)


def test_fixed_clock_empty_duplicate_saves_and_zero_span():
    for solve in (rk.solve_rk45,rk.solve_dopri8):
        a=jax.jit(lambda:solve(lambda t,y,a:jnp.ones_like(y),jnp.zeros(1),t0=.2,t1=1.2,
            dt0=.1,adaptive=False,ts=jnp.array([.2,.2,.65,1.2]),max_steps=10))()
        assert a.stats.accepted==10 and a.status==0
        np.testing.assert_allclose(a.ys[:,0],a.ts-.2,atol=2e-15)
        z=solve(oscillator,jnp.array([1.,0.]),t1=0.,dt0=.1,ts=[0.,0.])
        assert z.stats.steps==0
        np.testing.assert_array_equal(z.ys,[[1.,0.],[1.,0.]])
        assert solve(oscillator,jnp.array([1.,0.]),t1=.1,dt0=.1).ys.shape==(0,2)


def test_levelset_boolean_initial_and_simultaneous_events():
    rhs=lambda t,y,a:jnp.ones_like(y)
    for event,root,expected in (({'later':lambda t,y,a:.7-y[0],'first':lambda t,y,a:.3-y[0]},True,.3),
                               (lambda t,y,a:y[0]>.2,True,.4),
                               (lambda t,y,a:.3-y[0],False,.4),
                               (lambda t,y,a:y[0],True,0.)):
        out=rk.solve_rk(rhs,jnp.zeros(1),t1=1.,dt0=.4,adaptive=False,ts=[0.,.1,.5,1.],event=event,root_find=root)
        assert out.status==rk.RKStatus.EVENT
        np.testing.assert_allclose(out.t,expected,atol=2e-15)
        assert sum(map(bool,jax.tree_util.tree_leaves(out.event_mask)))==1
        if expected==0:
            assert np.isinf(out.ys).all()
    both=(lambda t,y,a:.3-y[0],)*2
    np.testing.assert_array_equal(rk.solve_rk(rhs,jnp.zeros(1),t1=1.,dt0=.4,event=both).event_mask,[True,False])


def test_vmapped_particles_stop_independently():
    def one(v):
        return rk.solve_rk(lambda t,y,a:jnp.ones_like(y)*v,jnp.zeros(1),t1=1.,dt0=.1,
                           event=lambda t,y,a:.3-y[0],max_steps=32)
    out=jax.jit(jax.vmap(one))(jnp.array([1.,.1]))
    np.testing.assert_array_equal(out.status,[1,0])
    np.testing.assert_allclose(out.t,[.3,1.],atol=2e-15)


def test_work_limits_nonfinite_and_recovery():
    for adaptive in (False,True):
        out=rk.solve_rk(lambda t,y,a:jnp.full_like(y,jnp.nan),jnp.ones(1),t1=1.,dt0=.1,
                        adaptive=adaptive,dtmin=.1,max_steps=4)
        assert out.status==rk.RKStatus.NONFINITE and out.t==0
    assert rk.solve_rk(oscillator,jnp.array([1.,0.]),t1=10.,dt0=.1,max_steps=1).status==rk.RKStatus.MAX_STEPS
    assert rk.solve_rk(oscillator,jnp.array([1.,0.]),t1=10.,dt0=1.,dtmin=1.,atol=1e-15,rtol=0.).status==rk.RKStatus.MIN_STEP
    rhs=lambda t,y,a:jnp.where(t>.3,jnp.nan,jnp.ones_like(y))
    out=rk.solve_rk(rhs,jnp.zeros(1),t1=.2,dt0=.2,max_steps=128)
    assert out.status==0
    bad=rk.solve_rk(oscillator,jnp.array([1.,0.]),t1=1.,dt0=.1,event=lambda t,y,a:jnp.nan)
    assert bad.status==rk.RKStatus.NONFINITE


def test_static_input_and_event_contracts():
    base=dict(t1=1.,dt0=.1)
    invalid=[dict(method='rk4'),dict(controller='pid'),dict(stage_execution='x'),dict(max_steps=0),
             dict(max_steps=True),dict(dt0=0),dict(dt0=np.inf),dict(t0=np.nan),dict(t1=-1),
             dict(dtmin=-1),dict(dtmax=-1),dict(dtmin=.2),dict(dtmin=.1,dtmax=.01),
             dict(atol=0),dict(rtol=-1),dict(ts=[.7,.2]),dict(ts=[2.]),dict(ts=[[0.]]),
             dict(t0=[0.]),dict(controller='pi',pcoeff=-1),dict(controller='pi',icoeff=0),
             dict(controller='pi',pcoeff=[.4]),dict(event=lambda t,y,a:jnp.ones(2)),
             dict(event=lambda t,y,a:jnp.asarray(1)),dict(error_norm=lambda *a:jnp.ones(2))]
    for change in invalid:
        with pytest.raises(ValueError):
            rk.solve_rk(oscillator,jnp.array([1.,0.]),**(base|change))
    for y in ([],[[1.]],jnp.array([1j])):
        with pytest.raises(ValueError):
            rk.solve_rk(oscillator,y,**base)


def test_dynamic_validation_and_physical_error_blocks():
    run=jax.jit(lambda dt,atol,p:rk.solve_rk(oscillator,jnp.array([1.,0.]),t1=1.,dt0=dt,
                atol=atol,controller='pi',pcoeff=p,max_steps=16))
    for dt,atol,p in ((-1.,1e-9,.4),(.1,0.,.4),(.1,1e-9,-.4)):
        assert run(dt,atol,p).status==rk.RKStatus.INVALID_INPUT
    for components in (1,3):
        norm=OrbitErrorNorm(1e-9,1e-9,10.,2.,components)
        state=jnp.zeros(6); error=jnp.array([2e-9,0,0,1e-8,0,0])
        np.testing.assert_allclose(norm(state,state,error),.5)
        mapped=OrbitErrorNorm(1e-9,1e-9,10.,2.,components,lambda y:2*y[:3])
        np.testing.assert_allclose(mapped(state,state,error),1.)
        assert np.isinf(norm(state,state,jnp.full(6,jnp.nan)))
    for options in (dict(velocity_components=2),dict(speed_scale=0.),dict(rtol=-1.),dict(length_scale=[1.])):
        with pytest.raises(ValueError):
            OrbitErrorNorm(**(dict(rtol=1e-9,atol=1e-9,speed_scale=1.)|options))


class UniformField:
    def B_contravariant(self,x): return jnp.array([0.,0.,1.])
    B_covariant=B_contravariant
    def AbsB(self,x): return 1.
    def dAbsB_by_dX(self,x): return jnp.zeros(3)
    curl_b=kappa=dAbsB_by_dX
    def sqrtg(self,x): return 1.
    def to_xyz(self,x): return x


@pytest.mark.parametrize('model',['GuidingCenterAdaptative','FullOrbitAdaptative'])
def test_tracing_native_dispatch_matches_analytic_and_diffrax(model):
    from essos.dynamics import Particles,Tracing
    p=Particles(jnp.zeros((1,3)),initial_vparallel_over_v=[.2],mass=1.,charge=1.,energy=.5,
                initial_xyz_fullorbit=jnp.zeros((1,3)),initial_vxvyvz=jnp.array([[.3,0.,.2]]))
    opts=dict(field=UniformField(),particles=p,model=model,maxtime=.3,timestep=.1,times_to_trace=4,progress=False)
    ref=Tracing(**opts)
    for integrator in ('essos_rk45','essos_dopri8'):
        out=Tracing(**opts,integrator=integrator)
        assert out.rtol==out.atol==1e-9
        np.testing.assert_allclose(out.trajectories,ref.trajectories,atol=2e-9)
    for options in (dict(integrator='x'),dict(integrator='essos_dopri8',solver=diffrax.Dopri8()),
                    dict(integrator='essos_dopri8',model='FullOrbitCollisions'),
                    dict(native_error_control='x'),dict(native_controller='x')):
        with pytest.raises(ValueError):
            Tracing(**(opts|options))
    standard=Tracing(**opts,integrator='essos_rk45',native_error_control='standard')
    from essos.rk import _trace_error_norm as error_norm
    assert error_norm(standard) is None
    end=out._solve(*out._terms(None),0.,out.initial_conditions[0],out.args,diffrax.Event(lambda t,y,a:False),throw=True)
    assert end.result==diffrax.RESULTS.successful


def coil_field():
    from essos.coils import Coils,CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    return BiotSavart(Coils(CreateEquallySpacedCurves(1,1,1.,.4,7,1,False),[1e6]))


def test_private_coil_field_and_rhs_match_public_equations():
    from essos.rk import _coil_geometry as geometry, _coil_field_values as field_values, _coil_rhs_values as rhs_values
    from essos.dynamics import Particles,GuidingCenter,Lorentz
    from essos.electric_field import Electric_field_zero
    field=coil_field(); arrays,nc,nq=geometry(field); x=jnp.array([1.,.03,.01])
    B,J=field_values(arrays,tuple(x),nc,nq,True)
    np.testing.assert_allclose(B,field.B(x),atol=3e-15)
    np.testing.assert_allclose(J,jax.jacfwd(field.B)(x),atol=3e-14)
    p=Particles(x[None],initial_vparallel_over_v=[.3],mass=1.,charge=1.,energy=.5)
    for model,y,rhs,args in (('gc',jnp.append(x,.3),GuidingCenter,(field,p,Electric_field_zero())),
                              ('lorentz',jnp.concatenate((x,jnp.array([.2,.1,.3]),jnp.zeros(2))),Lorentz,(field,p))):
        actual=rhs_values(arrays,y,nc,nq,model,.5,1.,1.)
        np.testing.assert_allclose(actual[:4 if model=='gc' else 6],rhs(0.,y[:4 if model=='gc' else 6],args),atol=3e-14)


def tokamak():
    from essos.boozer import BoozerField
    s=np.linspace(.005,.995,5)
    return BoozerField.from_booz(s,np.stack((s*0+5.,-.2*np.sqrt(s))),[0,1],[0,0],
                                s*0+.4,s*0+40.,s*0,40.,1,angle_grid=None)


def test_boozer_public_native_outputs_and_failures(monkeypatch):
    from essos.boozer import trace_boozer
    from essos.rk import trace_boozer_orbits as native_orbits
    field=tokamak()
    for method in ('essos_rk45','essos_dopri8'):
        out=trace_boozer(field,[.3],[.4],[0.],[.2],speed=1.,mass=1.,charge=1.,tmax=.01,
                          timestep=.003,n_save=3,integrator=method,progress=False)
        assert not out.failed.any() and not out.lost.any() and (out.step_stats[:,1]>0).all()
        assert np.isfinite(out.states).all()
    with pytest.raises(ValueError,match='native_execution'):
        native_orbits(field,None,None,1.,1.,1.,None,.1,1e-9,1e-9,'dopri8',True,'i',.4,.3,8,execution='bad')
    kwargs=dict(speed=1.,mass=1.,charge=1.,tmax=.001,timestep=.001,n_save=2,integrator='essos_dopri8',progress=False)
    legacy=trace_boozer(field,[.3],[.4],[0.],[.2],**(kwargs|dict(integrator=None,method='rk4')))
    assert np.isfinite(legacy.states).all() and legacy.step_stats is None
    for change in (dict(integrator='x'),dict(method='rk4'),dict(species=object()),dict(adaptive='yes'),dict(rtol=-1.),dict(atol=0.)):
        with pytest.raises(ValueError): trace_boozer(field,[.3],[.4],[0.],[.2],**(kwargs|change))
    from essos import rk as module
    original=module._prepare_boozer_gpu
    monkeypatch.setattr(module,'_prepare_boozer_gpu',lambda *a,**k:original(*a,**k,interpret=True))
    out=trace_boozer(field,[.3],[.4],[0.],[.2],**(kwargs|dict(native_execution='cuda',progress=lambda *a:None)))
    assert not out.failed.any()
    with pytest.raises(ValueError,match='I control'):
        trace_boozer(field,[.3],[.4],[0.],[.2],**(kwargs|dict(native_execution='cuda',native_controller='pi')))
    with pytest.raises(ValueError,match='cosine'):
        original(SimpleNamespace(table=1.,sine_coef=None),1.,1.,1.,[0.,.1],.1,1e-9,1e-9,8,interpret=True)


def test_vmec_private_values_match_current_main():
    from essos.fields import Vmec
    from essos.rk import _prepare_vmec_data as prepare, _vmec_quantities as quantities, _vmec_rhs_values as rhs_values, _vmec_xyz_values as xyz_values
    from essos.dynamics import GuidingCenter,Particles,_axis_regular,_to_axis_regular
    from essos.electric_field import Electric_field_zero
    filename=Path(__file__).resolve().parents[1]/'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'
    field=Vmec(str(filename),ntheta=3,nphi=3)
    prepared=prepare(field); x=jnp.array([.3,.4,.2])
    actual=quantities(prepared.arrays,tuple(x),prepared.tables)
    expected=(field.B_covariant(x),field.B_contravariant(x),field.AbsB(x),field.dAbsB_by_dX(x),field.curl_b(x),field.kappa(x),field.sqrtg(x))
    for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12)
    p=Particles(x[None],initial_vparallel_over_v=[.3],mass=1.,charge=1.,energy=.5)
    y=jnp.pad(_to_axis_regular(jnp.append(x,.3)),(0,3))
    np.testing.assert_allclose(rhs_values(prepared.arrays,y,prepared.tables,1.,1.,.5)[:5],
        _axis_regular(GuidingCenter)(0.,y[:5],(field,p,Electric_field_zero())),atol=3e-12)
    np.testing.assert_allclose(xyz_values(prepared.arrays,y,prepared.tables),field.to_xyz(x),atol=3e-12)
    from essos.rk import prepare_solver
    args=dict(mass=1.,charge=1.,energy=.5,times=np.array([0.,.001]),dt0=.001,max_steps=8)
    reference=jax.jit(prepare_solver(field,**args))(y[:5])
    gpu=jax.jit(prepare_solver(field,execution='cuda',interpret=True,**args))(y[:5])
    np.testing.assert_allclose(gpu.y,reference.y,atol=3e-12)
    for change in (dict(model='lorentz'),dict(execution='x'),dict(energy=-1),dict(execution='cuda',controller='pi')):
        with pytest.raises(ValueError): prepare_solver(field,**(args|change))
    # Exercise segment dispatch on an analytic cylinder, avoiding compilation
    # of the full equilibrium inverse map in this small CI integration test.
    class Cylinder(UniformField,Vmec):
        def __init__(self): self.Aminor_p=1.
        def to_xyz(self,x):
            r=jnp.sqrt(jnp.maximum(x[0],0.))
            return jnp.array([r*jnp.cos(x[1]),r*jnp.sin(x[1]),x[2]])
        def boundary_distance(self,x): return 1.-jnp.sqrt(x[0]**2+x[1]**2)
        def flux_coordinates(self,x):
            return jnp.array([x[0]**2+x[1]**2,jnp.arctan2(x[1],x[0]),x[2]]),True
    from essos.dynamics import Tracing
    trace=Tracing(field=Cylinder(),particles=p,model='GuidingCenterAdaptative',maxtime=.001,timestep=.001,
                  times_to_trace=2,integrator='essos_dopri8',progress=False,max_returns=1,exterior_field=UniformField())
    assert np.isfinite(trace.trajectories).all()
    np.testing.assert_allclose(trace.trajectories[0,-1],jnp.array([.3,.4,.2003,.3]),atol=2e-14)


@pytest.mark.parametrize('method',['dopri5','dopri8'])
def test_independent_gpu_loop_on_cpu(method):
    from essos.rk import _prepare_gpu_solver as prepare_gpu_solver
    from essos.rk import _select_one
    def factory(refs,point,invariant):
        scalar=lambda y:_select_one(y,0,0).reshape(())
        return (lambda y:jnp.ones_like(y),lambda y,e,v:jnp.max(jnp.abs(e)),lambda y:.3-scalar(y))
    times=np.array([0.,.1,.2,.5])
    for adaptive in (False,True):
        solve=prepare_gpu_solver((),factory,1.,times,.2,1e-9,1e-9,16,method=method,
                                 adaptive=adaptive,interpret=True)
        run=jax.jit(solve)
        out=run(jnp.zeros(4),jnp.asarray(0.))
        assert out.status==rk.RKStatus.EVENT
        np.testing.assert_allclose(out.t,.3,atol=2e-14)
        np.testing.assert_allclose(out.ys[:3],np.broadcast_to(times[:3,None],(3,4)),atol=2e-14)
        assert np.isinf(out.ys[-1]).all()
        invalid=run(jnp.full(4,jnp.nan),jnp.asarray(0.))
        assert invalid.status==rk.RKStatus.NONFINITE and invalid.stats.steps==0
        initial=run(jnp.full(4,.3),jnp.asarray(0.))
        assert initial.status==rk.RKStatus.EVENT and np.isinf(initial.ys).all()
    base=dict(arrays=(),factory=factory,speed=1.,times=times,dt0=.2,rtol=1e-9,atol=1e-9,max_steps=16,interpret=True)
    for change in (dict(method='rk4'),dict(times=[.1,.2]),dict(dt0=-1),dict(max_steps=0),dict(interpret=False)):
        with pytest.raises(ValueError): prepare_gpu_solver(**(base|change))
    with pytest.raises(ValueError): solve(jnp.zeros(3),jnp.asarray(0.))
    limited=prepare_gpu_solver(**(base|dict(max_steps=1)))
    result=jax.jit(limited)(jnp.zeros(4),jnp.asarray(0.))
    assert result.status==rk.RKStatus.MAX_STEPS and result.stats.accepted==1


def test_rhs_preparation_and_gpu_helpers_preserve_values(monkeypatch):
    # Some supported JAX releases omit this deprecated facade export (CI).
    from jax.interpreters import partial_eval as facade
    monkeypatch.delattr(facade,'partial_eval_jaxpr_nounits',raising=False)
    from essos.rk import _select_one, _kernel_is_finite, _bind_primitive
    for dtype in (jnp.float32,jnp.float64):
        values=jnp.array([1.,np.inf,np.nan,-np.inf,-0.],dtype=dtype)
        np.testing.assert_array_equal(_kernel_is_finite(values),jnp.isfinite(values))
        assert np.signbit(_select_one(values,4,0).reshape(()))
    class Primitive:
        def get_bind_params(self,p): return ([],p)
        def bind(self,*args,**kwargs): return args[0]+kwargs['offset']
    assert _bind_primitive(Primitive(),[2.],dict(offset=3.))==5.
    def value(p):
        rhs=lambda t,y,args:jnp.sin(p)*y+t
        prepared=rk._prepare_rhs(rhs,jnp.asarray(0.),jnp.ones(2),None)
        return prepared(.2,jnp.array([2.,3.]),None).sum()
    np.testing.assert_allclose(jax.grad(value)(.3),5*np.cos(.3),atol=2e-14)
    with pytest.raises(TypeError): _kernel_is_finite(jnp.array([1],dtype=jnp.int32))
    import essos.rk as module
    from essos import boozer as boozer_module
    model=SimpleNamespace(modB_derivatives=lambda *a:(1.,0.,0.,0.))
    for f in (lambda field,y,*a:jnp.concatenate((y[:2],y[:2])),
              lambda field,y,*a:jnp.concatenate((jax.lax.dynamic_slice(y,(y[0].astype(jnp.int32),),(2,)),jnp.ones(2)))):
        monkeypatch.setattr(boozer_module,'guiding_center_rhs',f)
        with pytest.raises(ValueError,match='slice'):
            module._kernel_rhs(model,jnp.ones(4),.1,1.,1.)(jnp.ones(4))


def test_fixed_field_adapter_solvers_on_cpu():
    from essos.rk import prepare_solver
    from essos.rk import _prepare_boozer_gpu as prepare_orbit_solver
    from essos.rk import _boozer_batch as _native_orbits
    field=coil_field()
    args=dict(mass=1.,charge=1.,energy=.5,times=np.array([0.,.001]),dt0=.001,max_steps=8)
    with pytest.raises(ValueError): prepare_solver(UniformField(),execution='cuda',**args)
    from essos.rk import _prepare_coils_gpu as coil_solver
    with pytest.raises(ValueError): coil_solver(field,1.,1.,.5,1.,[0.,.001],.001,1e-9,1e-9,8,model='x',interpret=True)
    for model,y in (('gc',jnp.array([1.,.03,.01,.3])),
                    ('lorentz',jnp.array([1.,.03,.01,.2,.1,.3]))):
        args=dict(mass=1.,charge=1.,energy=.5,times=np.array([0.,.001]),dt0=.001,model=model,max_steps=8)
        reference=jax.jit(prepare_solver(field,**args))(y)
        gpu=jax.jit(prepare_solver(field,execution='cuda',interpret=True,**args))(y)
        assert gpu.status==0
        np.testing.assert_allclose(gpu.y,reference.y,atol=3e-13)
    boozer=tokamak(); y=jnp.array([[.5,0.,0.,.2]]); moments=jnp.array([.08])
    times=np.array([0.,.001])
    for method in ('dopri5','dopri8'):
        solve=prepare_orbit_solver(boozer,1.,1.,1.,times,.001,1e-9,1e-9,8,method=method,interpret=True)
        args=(boozer,y,moments,1.,1.,1.,jnp.asarray(times),.001,1e-9,1e-9,method,True,'i',.4,.3,8)
        expected=_native_orbits(*args)
        actual=_native_orbits(*args,orbit_solver=solve)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,atol=3e-13)
