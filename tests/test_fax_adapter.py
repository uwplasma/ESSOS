"""Portable FAX data preserve phase, units and equilibrium provenance."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from essos.fax_adapter import FaxMode


def fixture(tmp_path):
    path = tmp_path/"wout.nc"
    path.write_bytes(b"equilibrium used by producer")
    s = np.linspace(.1, .9, 8)
    harmonics = np.array([[3, 2], [4, 2]])
    structure = np.stack([np.sin(np.pi*(s-.1)/.8), .3j*np.sin(np.pi*(s-.1)/.8)], axis=1)
    problem = SimpleNamespace(s=s, harmonics=harmonics)
    result = SimpleNamespace(mode_structures=structure[None], frequencies_hz=np.array([1e5]),
                             omega2=np.array([(2*np.pi*1e5)**2]), residuals=np.array([1e-14]))
    return path, problem, result


def test_round_trip_preserves_relative_complex_phases_and_checks_geometry(tmp_path):
    equilibrium, problem, result = fixture(tmp_path)
    packet = FaxMode.from_result(problem, result, 0, equilibrium=equilibrium, interpretation="potential_shape")
    destination = tmp_path/"mode.npz"
    packet.save(destination)
    loaded = FaxMode.load(destination, equilibrium=equilibrium)
    np.testing.assert_array_equal(loaded.structure, result.mode_structures[0])
    assert np.max(abs(loaded.structure.imag)) > 0
    assert loaded.metadata["equilibrium_sha256"] == hashlib.sha256(equilibrium.read_bytes()).hexdigest()
    wave = loaded.as_perturbation(potential_volts=1000., frequency_sign=-1, taper_width=.1)
    assert float(wave.omega) == pytest.approx(-2*np.pi*1e5)
    assert wave.harmonics == ((3, 2), (4, 2))
    import jax.numpy as jnp
    from scipy.interpolate import CubicSpline
    from examples.benchmarks.alfven_gc_verification import equilibrium as background
    from essos.ae3d_adapter import AE3DMode
    field, point, t = background(), jnp.array([.4,.3,.7]), 1.3e-6
    for interpolation in ('linear','natural-cubic'):
        physical = loaded.as_physical_potential(potential_volts=1000.,frequency_sign=-1,interpolation=interpolation)
        grid = np.r_[0.,loaded.s]
        values = np.vstack([np.zeros((1,2)),loaded.structure/np.max(abs(loaded.structure))*1000.])
        if interpolation == 'natural-cubic':
            coefficient = CubicSpline(grid,values,axis=0,bc_type='natural')(.25)
        else:
            coefficient = np.array([np.interp(.25,grid,v.real)+1j*np.interp(.25,grid,v.imag) for v in values.T])
        theta = np.arctan2(.3,.4)
        phase = loaded.harmonics[:,0]*theta-loaded.harmonics[:,1]*.7+float(physical.omega)*t
        expected = np.real(coefficient*np.exp(1j*phase)).sum()
        assert float(physical.potentials(field,point,t)[1]) == pytest.approx(expected,rel=1e-12)
        assert np.isfinite(physical.derivatives(field,point,t)[1]).all()
    export = tmp_path/'ae3d.npy'
    rows = np.empty((1,3),object);rows[0] = [3,2,np.array([1.,-2.,3.,4.])]
    np.save(export,dict(eigenvalue=9.,s_coords=np.linspace(.1,.9,4),harmonics=rows))
    imported = AE3DMode.from_firm3d_numpy(export)
    assert imported.omega == 3000.
    np.testing.assert_array_equal(imported.potential_volts[:,0],[1.,-2.,3.,4.])
    equilibrium.write_bytes(b"different geometry")
    with pytest.raises(ValueError, match="hashes differ"):
        FaxMode.load(destination, equilibrium=equilibrium)


def test_unidentified_or_bad_eigenpairs_are_rejected(tmp_path):
    for change in ("interpretation", "omega2", "residual", "frequency"):
        equilibrium, problem, result = fixture(tmp_path)
        interpretation = "potential_shape"
        if change == "interpretation": interpretation = "displacement"
        if change == "omega2": result.omega2[0] *= -1
        if change == "residual": result.residuals[0] = .01
        if change == "frequency": result.frequencies_hz[0] /= 2*np.pi
        with pytest.raises(ValueError):
            FaxMode.from_result(problem, result, 0, equilibrium=equilibrium, interpretation=interpretation)


def test_live_fax_eigensolver_can_export_a_small_mode(tmp_path):
    fax = pytest.importorskip("fax.vorticity")
    from fax.units import MU0
    grid = np.linspace(.1,.9,5)
    theta,zeta = np.meshgrid(2*np.pi*np.arange(8)/8,2*np.pi*np.arange(8)/8,indexing='ij')
    basis = np.broadcast_to(np.diag([1.,1.,4.]),(5,64,3,3))
    problem = fax.physical_potential_problem_from_grid(grid,[[1,2],[3,1]],theta=theta,zeta=zeta,
        magnetic_field=np.full((5,64),2.),magnetic_field_angle_derivatives=np.zeros((5,64,2)),
        covariant_basis=basis,iota=0.,boozer_i=0.,boozer_g=8.,mass_density=1.7e-6)
    result = fax.solve_physical_potential_eigenmodes(problem)
    np.testing.assert_allclose(result.omega2,np.repeat([1.,4.],3)/(4*MU0*1.7e-6),rtol=3e-13)
    equilibrium = tmp_path/"analytic-equilibrium.txt"
    equilibrium.write_text("FAX uniform-plasma physical-potential fixture")
    packet = FaxMode.from_result(problem, result, 0, equilibrium=equilibrium, interpretation="potential_shape")
    assert packet.metadata["eigenpair_residual"] < 1e-12
    np.testing.assert_array_equal(packet.structure, result.mode_structures[0])
    # Exercise the same non-monoenergetic births used by the archive driver.
    from examples.benchmarks.alfven_gc_verification import equilibrium as background
    from examples.benchmarks.ae3d_firm3d_loss import native_trace
    field = background()
    from examples.benchmarks.fax_qa_loss_collisions import volume_births
    small,weights = volume_births(field,3,42)
    large,large_weights = volume_births(field,7,42)
    np.testing.assert_array_equal(small[:,:4],large[:3,:4])
    np.testing.assert_allclose(small[:,4],large[:3,4],rtol=1e-13)
    np.testing.assert_allclose(weights,large_weights[:3],rtol=1e-13)
    assert np.all(weights>0) and np.all((small[:,0]>=0)&(small[:,0]<1))
    b = np.array([field.modB(*p) for p in small[:,:3]])
    from examples.benchmarks.ae3d_firm3d_loss import ENERGY,MASS
    np.testing.assert_allclose(.5*MASS*(small[:,3]**2+2*small[:,4]*b),ENERGY,rtol=2e-14)
    wave = packet.as_physical_potential(potential_volts=10.,frequency_sign=1)
    data = np.array([[.3,.2,.3,6e6,1e12],[.4,.1,.2,-5e6,2e12]])
    trace = native_trace(field,data,wave,tmax=2e-8,timestep=1e-8,tolerance=1e-8,
                         n_save=3,collision_dtmax=1e-8,stable_keys=True)
    assert not trace.failed.any() and np.isfinite(trace.states).all()
    np.testing.assert_allclose(trace.states[:,0,3],data[:,3],rtol=1e-14)
    B = np.array([field.modB(*p) for p in data[:,:3]])
    np.testing.assert_allclose((trace.states[:,0,4]**2-data[:,3]**2)/(2*B),data[:,4],rtol=1e-13)
    baseline=native_trace(field,data,None,tmax=2e-8,timestep=1e-8,tolerance=1e-8,n_save=3,timing_repeats=1)
    assert not baseline.failed.any() and len(baseline.timings['warm_seconds'])==1
    np.testing.assert_allclose(baseline.states[:,0],trace.states[:,0],rtol=1e-13)


def test_archived_embedding_and_spatial_wave_collision_trace(tmp_path,monkeypatch):
    import jax
    import jax.numpy as jnp
    from netCDF4 import Dataset
    from examples.benchmarks.ae3d_firm3d_loss import spatial_geometry,native_trace
    from examples.benchmarks.alfven_gc_verification import equilibrium
    from essos.background_species import BackgroundSpecies
    from essos.mode_reconstruction import PhysicalPotentialMode
    path=tmp_path/'boozmn.nc';s=np.array([1/6,.5,5/6])
    with Dataset(path,'w') as nc:
        nc.createDimension('surface',3);nc.createDimension('mode',4)
        nc.createVariable('ns_b','i4',())[...]=4
        for name,values,dimension in (('jlist',[2,3,4],'surface'),('ixm_b',[0,1,2,3],'mode'),('ixn_b',[0,0,2,-1],'mode')):
            nc.createVariable(name,'i4',(dimension,))[:]=values
        for name,values in (('rmnc_b',np.column_stack((s*0+2,.2*np.sqrt(s),s*0+1e-7,2e-8*s**1.5))),
            ('zmns_b',np.column_stack((s*0,.2*np.sqrt(s),s*0,3e-8*s**1.5))),
            ('pmns_b',np.column_stack((s*0,.03*np.sqrt(s),s*0,-2e-8*s**1.5))),
            ('rmns_b',np.column_stack((s*0,s*0,1e-8*s,-2e-8*s**1.5)))):
            nc.createVariable(name,'f8',('surface','mode'))[:]=values
    full=spatial_geometry(path)
    assert len(full.xm)==4
    refs,evaluate=full._kernel_view()
    def expected(point):
        return jnp.r_[full(point),jax.jacfwd(full)(point).reshape(-1)]
    jet=lambda point:jnp.asarray(evaluate(refs,*point,jet=True))
    points=jnp.array([[.001,.4,.2],[.5,1.3,-.7],[.95,-.2,2.]])
    np.testing.assert_allclose(jax.jit(jax.vmap(jet))(points),jax.jit(jax.vmap(expected))(points),rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(jet))(points[1]),jax.jit(jax.jacfwd(expected))(points[1]),rtol=2e-11,atol=2e-11)
    geometry=spatial_geometry(path,mode_tolerance=1e-6)
    assert len(geometry.xm)==2
    q=jnp.array([.5,.4,.2]);R=2+.2*np.sqrt(q[0])*np.cos(q[1]);phi=q[2]+.03*np.sqrt(q[0])*np.sin(q[1])
    np.testing.assert_allclose(geometry(q),[R*np.cos(phi),R*np.sin(phi),.2*np.sqrt(q[0])*np.sin(q[1])],atol=2e-14)
    # The native value/Jacobian callback must invert the same Cartesian kick.
    geometry=geometry._replace(coefficients=jnp.pad(geometry.coefficients,((0,0),(0,128),(0,0),(0,0))),
        xm=jnp.pad(geometry.xm,(0,128)),xn=jnp.pad(geometry.xn,(0,128)))
    from jax.experimental import pallas as pl
    from essos.split import prepare_physics,_split_field_data,move_in_cartesian,CollisionStep
    initial=jnp.array([.4,.3,.2,6e6,1e-14]);displacement=jnp.array([1e-5,-2e-5,3e-5,0.,0.,0.,0.,0.])
    public=lambda y:jnp.array([y[0]**2+y[1]**2,jnp.arctan2(y[1],y[0]),y[2]])
    def build(view):
        kick=lambda t,y,h,dw,a:CollisionStep(move_in_cartesian(y,dw[:3],lambda y:view.to_xyz(public(y))),0.,0.)
        return lambda t,y,a:jnp.zeros(5),kick,lambda a,b,e:0.,lambda t,y,a:1.,lambda t,y,a:jnp.inf
    field=equilibrium()
    arrays,factory=prepare_physics(field,_split_field_data(field),build,initial,5,spatial_geometry=geometry)
    def kernel(yref,dref,*refs):
        *data,outref=refs
        outref[...]=factory(data,yref[...])[1](yref[...],jnp.asarray(1.),dref[...])[0]
    native=pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((8,),initial.dtype),grid=(),interpret=True)
    moved=jax.jit(native)(jnp.pad(initial,(0,3)),displacement,*arrays)[:5]
    np.testing.assert_allclose(geometry.regular_xyz(moved[:3]),
        geometry.regular_xyz(initial[:3])+displacement[:3],rtol=0,atol=2e-13)
    np.testing.assert_array_equal(moved[3:],initial[3:])
    bath=BackgroundSpecies(1,jnp.array([2.]),jnp.array([1.]),jnp.array([4.21e20]),jnp.array([17800.]))
    wave=PhysicalPotentialMode([0.,.5,1.],[[2,1]],np.array([[0.],[500j],[1000j]]),9e5)
    data=np.array([[.5,.4,.2,6e6,1e12]])
    traced=native_trace(equilibrium(),data,wave,tmax=1e-10,timestep=1e-10,tolerance=1e-7,
        species=bath,n_save=2,collision_dtmax=1e-10,spatial_diffusion=True,spatial_geometry=geometry)
    assert not traced.failed.any() and np.isfinite(traced.states).all()
    assert traced.timings['compile_seconds']>0 and traced.timings['first_execution_seconds']>0
    np.testing.assert_allclose(traced.states[0,0,:4],data[0,:4],rtol=1e-13)
    baseline=native_trace(field,data,None,tmax=1e-10,timestep=1e-10,tolerance=1e-7,
        species=bath,n_save=2,collision_dtmax=1e-10,spatial_diffusion=True,spatial_geometry=geometry)
    import essos.split as split
    original=split._prepare_split_kernel
    monkeypatch.setattr(split,'_prepare_split_kernel',lambda *a,**k:original(*a,**(k|dict(interpret=True))))
    monkeypatch.setattr(split,'_validate_independent',lambda *a,**k:None)
    native=native_trace(field,data,None,tmax=1e-10,timestep=1e-10,tolerance=1e-7,
        species=bath,n_save=2,collision_dtmax=1e-10,spatial_diffusion=True,spatial_geometry=geometry,execution='independent')
    assert not native.failed.any()
    np.testing.assert_allclose(native.states,baseline.states,rtol=3e-11,atol=3e-11)


def test_six_case_outputs_preserve_births_and_weighted_loss_curves(tmp_path,monkeypatch):
    from examples.benchmarks import fax_qa_loss_collisions as benchmark
    data=np.array([[.2,.1,.3,6e6,1e12],[.4,.2,.5,-5e6,2e12]])
    births=dict(initial_conditions=data,weights=np.array([1.,3.]),marker_indices=np.arange(2))
    args=SimpleNamespace(boozmn=tmp_path/'unused.nc',particles=2,tmax=.01,method='essos_dopri8',
        timestep=1e-8,n_save=2,tolerance=1e-9,seed=42,execution='independent',collision_dtmax=1e-6,
        birth_source='volume',collision_tolerance=.001,moment_tolerance=.01,timing_repeats=0,
        density=4.21e20,temperature=17800.,geometry_mode_tolerance=1e-6,batch_size=2,output=tmp_path)
    summary=dict(requested_cases=list(benchmark.CASES),cases={},device_kind='mock',mass_kg=benchmark.MASS,
        particles=2,duration_s=.01,frequency_hz=27800.,birth_source='volume')
    embedding=SimpleNamespace(xm=np.array([0,1]));wave=object();calls=[]
    monkeypatch.setattr(benchmark,'spatial_geometry',lambda path,tolerance:embedding)
    def trace(field,received,perturbation,**settings):
        assert received is data and settings['seed']==42
        assert settings['batch_size']==2
        assert settings['spatial_geometry'] is (embedding if settings['spatial_diffusion'] else None)
        settings['progress'](2,2,.005,.01,.2)
        import json
        checkpoint=json.loads((tmp_path/'summary.json').read_text())
        assert checkpoint['completed_particles']==2 and checkpoint['last_batch_simulated_time_s']==[.005,.01]
        calls.append((perturbation is wave,settings['species'] is not None,settings['spatial_diffusion']))
        return SimpleNamespace(times=np.array([0.,.01]),states=np.ones((2,2,5))*1e6,
            lost=np.array([False,True]),failed=np.zeros(2,bool),axis_stopped=np.zeros(2,bool),
            loss_times=np.array([-1.,.005]),thermalized_times=np.full(2,-1.),
            failed_times=np.full(2,-1.),axis_stop_times=np.full(2,-1.))
    monkeypatch.setattr(benchmark,'native_trace',trace)
    benchmark.run_cases(args,SimpleNamespace(xm=np.arange(25)),wave,births,summary)
    assert set(calls)=={(False,False,False),(True,False,False),(False,True,False),
                        (True,True,False),(False,True,True),(True,True,True)}
    benchmark.run_cases(args,SimpleNamespace(xm=np.arange(25)),wave,births,summary)
    assert len(calls)==6  # Resuming keeps completed trajectories unchanged.
    benchmark.plot_saved(tmp_path)
    curves=np.genfromtxt(tmp_path/'loss_fraction.csv',delimiter=',',names=True)
    assert len(curves.dtype.names)==7
    for case in benchmark.CASES:
        np.testing.assert_allclose(curves[case],[0.,.75])
        with np.load(tmp_path/f'{case}.npz') as saved:
            np.testing.assert_array_equal(saved['marker_indices'],births['marker_indices'])
            np.testing.assert_array_equal(saved['weights'],births['weights'])


def test_progress_batches_preserve_keys_order_and_tail():
    import jax
    import jax.numpy as jnp
    from examples.benchmarks.ae3d_firm3d_loss import _trace_batches
    points=jnp.arange(15.).reshape(5,3)
    keys=jax.random.split(jax.random.key(42),5)
    def solve(points,keys):
        return points+jax.vmap(jax.random.normal)(keys)[:,None],points[:,0]/1000
    progress=[]
    result,timing=_trace_batches(solve,(points,keys),batch_size=2,
        end_times=lambda out:out[1],progress=lambda *row:progress.append(row),repeats=1)
    expected=jax.jit(solve)(points,keys)
    for actual,reference in zip(result,expected):np.testing.assert_array_equal(actual,reference)
    assert [r[0] for r in progress]==[2,4,5]
    assert all(r[1]==5 for r in progress)
    np.testing.assert_allclose([(r[2],r[3]) for r in progress],[(0.,.003),(.006,.009),(.012,.012)],rtol=1e-15,atol=0.)
    assert timing['batch_size']==2 and timing['batches']==3 and len(timing['warm_seconds'])==1
    from examples.benchmarks.alfven_gc_verification import equilibrium
    from examples.benchmarks.ae3d_firm3d_loss import native_trace
    data=np.array([[.3,.2,.3,6e6,1e12],[.4,.1,.2,-5e6,2e12]])
    options=dict(tmax=2e-8,timestep=1e-8,tolerance=1e-8,n_save=3)
    full=native_trace(equilibrium(),data,None,**options)
    batched=native_trace(equilibrium(),data,None,batch_size=1,**options)
    np.testing.assert_allclose(batched.states,full.states,rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(batched.step_stats.accepted,full.step_stats.accepted)


def test_default_qa_mode_packet_is_available_without_local_artifacts():
    import json
    from examples.benchmarks.fax_qa_loss_collisions import REFERENCE_MODE
    summary=json.loads((REFERENCE_MODE/'wave_summary.json').read_text())
    assert hashlib.sha256((REFERENCE_MODE/'mode.npz').read_bytes()).hexdigest()==summary['mode_sha256']
    with np.load(REFERENCE_MODE/'mode.npz',allow_pickle=False) as packet:
        assert packet['harmonics'].shape==(35,2)
        assert float(packet['frequency_hz'])==summary['frequency_hz']
        assert json.loads(str(packet['metadata_json']))['equilibrium_sha256']==summary['equilibrium_sha256']
