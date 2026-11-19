"""Reproduce accepted-path gradient checks and blocked timing/memory measurements.

Run each field/model/stepping case in a fresh process for allocator peaks.
No particle chunking, field interpolation, fitted controller or GPU selection
is imposed. JAX uses the machine's selected backend.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.boozer import BoozerField, psi0_from_vmec
from essos.dynamics import Particles
from essos.background_species import BackgroundSpecies
from essos.split import trace_split, BoozerGeometry

ROOT=Path(__file__).resolve().parents[2]


def build(options):
    if options.kernel_audit:
        return kernel_audit(options)
    geometry=None; n=options.particles
    if options.field=='coils':
        base=Coils.from_json(str(ROOT/'examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json'))
        coil_dofs=jnp.asarray(base.dofs)
        points=jnp.column_stack((jnp.linspace(1.23,1.27,n),jnp.zeros((n,2))))
        field=None
    elif options.field=='vmec':
        field=Vmec(str(options.wout),ntheta=8,nphi=8)
        coil_dofs=jnp.empty(0)
    else:
        from booz_xform_jax import Booz_xform
        from netCDF4 import Dataset
        transform=Booz_xform(verbose=0,mboz=options.mboz,nboz=options.nboz)
        transform.read_wout(str(options.wout),flux=False);transform.run()
        with Dataset(options.wout) as data: psi=psi0_from_vmec(data['phi'][-1])
        field=BoozerField.from_booz(transform.s_b,transform.bmnc_b,transform.xm_b,transform.xn_b,
            transform.iota,transform.Boozer_G,transform.Boozer_I,psi,transform.nfp,
            mode_tolerance=options.mode_cut)
        geometry=BoozerGeometry.from_booz_xform(transform)
        coil_dofs=jnp.empty(0)
    if options.field!='coils':
        rng=np.random.default_rng(options.seed)
        points=jnp.asarray(np.column_stack((np.full(n,.3),rng.uniform(0,2*np.pi,n),
            rng.uniform(0,2*np.pi/field.nfp,n))))
    bath=BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),jnp.array([1e20]),jnp.array([1000.]))
    pitch=jnp.linspace(.35,.65,n)
    parameters=jnp.concatenate((coil_dofs,jnp.zeros(3)))
    gc=options.model=='gc';adaptive=options.stepping=='adaptive'
    dt=options.horizon/4 if adaptive else options.horizon/options.steps
    times=jnp.linspace(0.,options.horizon,options.saves)
    def run(p, *, adjoint='none', record=False, tapes=None):
        source=BiotSavart(base.with_dofs(p[:-3])) if options.field=='coils' else field
        shifted=points.at[:,0].add(p[-3])
        birth=Particles(shifted,mass=M,charge=Q,energy=options.energy_ev*ONE_EV*jnp.exp(p[-1]),
            initial_vparallel_over_v=pitch+p[-2],field=source if not gc else None,phase_angle_full_orbit=jnp.pi)
        birth.random_keys=jax.random.split(jax.random.key(options.seed),n)
        return trace_split(source,birth,bath,model='GuidingCenter' if gc else 'FullOrbit',
            tmax=options.horizon,timestep=dt,collision_dtmax=options.horizon/2 if adaptive else dt,
            times=times,orbit_dt0=dt,adaptive=adaptive,orbit_adaptive=adaptive,
            rtol=options.rtol,atol=options.atol,collision_tolerance=.001,
            moment_tolerance=options.moment_tolerance,spatial_geometry=geometry,
            spatial_diffusion=gc,execution='shared' if tapes is not None else options.execution,
            reverse_execution=options.reverse_execution,adjoint=adjoint,
            max_steps=options.max_steps,max_orbit_steps=options.max_orbit_steps,
            adjoint_max_accepted_steps=options.capacity,adjoint_max_orbit_steps=options.capacity,
            adjoint_checkpoint_steps=options.segment,adjoint_orbit_checkpoint_steps=options.segment,
            _record_ad=record,_replay_tapes=tapes)
    def metric(saved,final):
        # The same smooth scalar in every chart, in that chart's public units.
        # Position/pitch/energy derivatives are all exercised. This is an AD
        # diagnostic, not a proposed confinement objective.
        speed=np.sqrt(2*options.energy_ev*ONE_EV/M)
        return jnp.mean(jnp.sum(saved[...,:3]**2,axis=-1))+jnp.mean(final[...,3]**2)/speed**2
    return parameters,run,metric


def kernel_audit(options):
    """Small field/kernel regression fixtures, separate from physical statistics."""
    from types import SimpleNamespace
    from essos.coils import CreateEquallySpacedCurves
    gc=options.model=='gc';coordinate=options.field!='coils';geometry=None
    if options.field=='coils':
        base=Coils(CreateEquallySpacedCurves(1,1,1.,.4,7,1,False),[1e6])
        dofs=jnp.asarray(base.dofs)
    elif options.field=='vmec':
        field=Vmec(str(options.wout),ntheta=8,nphi=8)
    else:
        s=np.linspace(.005,.995,5)
        field=BoozerField.from_booz(s,np.stack((s*0+5.,-.2*np.sqrt(s))),[0,1],[0,0],
            s*0+.4,s*0+40.,s*0,40.,1,angle_grid=None)
        radial=np.array([.1,.3,.5]);r=.2*np.sqrt(radial)
        geometry=BoozerGeometry.from_booz_xform(SimpleNamespace(s_b=radial,xm_b=np.array([0,1]),xn_b=np.zeros(2),
            rmnc_b=np.stack((radial*0+2.,r)),zmns_b=np.stack((radial*0,r)),
            numns_b=np.stack((radial*0,.05*np.sqrt(radial)))))
    bath=BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),jnp.array([1e20]),jnp.array([1000.]))
    options.particles=1;options.horizon=2e-10 if coordinate else 2e-9;options.saves=3
    options.energy_ev=4000.;options.steps=2
    p=jnp.array([.4,1.,.3 if coordinate else 0.])
    def run(p, *, adjoint='none', record=False, tapes=None):
        source=field if coordinate else BiotSavart(base.with_dofs(dofs.at[0].add(p[2])))
        points=jnp.array([[p[2],.2,.1]]) if coordinate else jnp.array([[1.,.03,.01]])
        birth=Particles(points,mass=M,charge=Q,energy=4000*ONE_EV*p[1],initial_vparallel_over_v=p[:1],
            field=source if not gc else None)
        return trace_split(source,birth,bath,model='GuidingCenter' if gc else 'FullOrbit',
            tmax=options.horizon,timestep=options.horizon/2,n_save=3,
            execution='shared' if tapes is not None else options.execution,
            adjoint=adjoint,reverse_execution=options.reverse_execution,
            adaptive=options.stepping=='adaptive',orbit_adaptive=options.stepping=='adaptive',
            moment_tolerance=.01,max_steps=16,max_orbit_steps=16,
            adjoint_max_accepted_steps=16,adjoint_max_orbit_steps=32,
            adjoint_orbit_checkpoint_steps=2,spatial_geometry=geometry,_record_ad=record,_replay_tapes=tapes)
    speed=np.sqrt(8000*ONE_EV/M)
    def metric(saved,final):
        spatial=jnp.sum(saved[...,:3]**2)
        return (spatial+jnp.sum(final[...,3])/speed+jnp.sum(final[...,4])/(4000*ONE_EV) if coordinate else
            spatial+jnp.sum(final[...,3]**2)/1e12)
    return p,run,metric


def measure(function, parameters, repeats):
    start=perf_counter();exe=jax.jit(function).lower(parameters).compile()
    compile_s=perf_counter()-start
    start=perf_counter();result=jax.block_until_ready(exe(parameters))
    first_s=perf_counter()-start
    stats=jax.devices()[0].memory_stats()
    jax.block_until_ready(exe(parameters))
    samples=[]
    for _ in range(repeats):
        start=perf_counter();jax.block_until_ready(exe(parameters));samples.append(1000*(perf_counter()-start))
    memory=exe.memory_analysis()
    buffers=sum(getattr(memory,k,0) or 0 for k in
        ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))-(getattr(memory,'alias_size_in_bytes',0) or 0)
    return result,dict(compile_s=compile_s,first_s=first_s,warm_ms=float(np.median(samples)),
        samples_ms=samples,compiled_buffers_bytes=buffers,allocator_after_first=stats)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field',choices=('coils','vmec','boozer'),default='coils')
    parser.add_argument('--model',choices=('gc','lorentz'),default='gc')
    parser.add_argument('--stepping',choices=('fixed','adaptive'),default='adaptive')
    parser.add_argument('--execution',choices=('shared','independent'))
    parser.add_argument('--reverse-execution',choices=('auto','shared','independent'),default='auto')
    parser.add_argument('--particles',type=int,default=64);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--horizon',type=float,default=2e-6);parser.add_argument('--steps',type=int,default=28)
    parser.add_argument('--saves',type=int,default=8);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--energy-ev',type=float,default=4000.)
    parser.add_argument('--rtol',type=float,default=1e-9);parser.add_argument('--atol',type=float,default=1e-9)
    parser.add_argument('--moment-tolerance',type=float,default=.01)
    parser.add_argument('--segment',type=int,default=16);parser.add_argument('--capacity',type=int,default=4096)
    parser.add_argument('--max-steps',type=int,default=4096);parser.add_argument('--max-orbit-steps',type=int,default=256)
    parser.add_argument('--wout',type=Path,default=ROOT/'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
    parser.add_argument('--mboz',type=int,default=16);parser.add_argument('--nboz',type=int,default=16)
    parser.add_argument('--mode-cut',type=float,default=1e-4)
    parser.add_argument('--check-gradient',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--kernel-audit',action='store_true',help='One-particle derivative fixtures; manufactured Boozer spectrum/geometry')
    parser.add_argument('--forward-only',action='store_true',help='Fresh-process forward allocator peak without building AD')
    parser.add_argument('--cold-compile',action='store_true',help='Disable the persistent compilation cache for this process')
    parser.add_argument('--output',type=Path,default=Path('output/native_ad/comparison.json'))
    o=parser.parse_args()
    if o.cold_compile:jax.config.update('jax_enable_compilation_cache',False)
    if o.execution is None:o.execution='independent' if jax.default_backend()=='gpu' else 'shared'
    if o.model=='lorentz' and o.field!='coils':parser.error('Lorentz requires Cartesian coil fields')
    if min(o.particles,o.repeats,o.steps,o.saves-1,o.segment,o.capacity,o.max_steps,o.max_orbit_steps)<1:
        parser.error('particle/save/work limits must be positive')
    if not np.isfinite(o.horizon) or o.horizon<=0:parser.error('positive finite horizon required')
    print('Preparing',o.field,o.model,o.stepping,jax.devices()[0],flush=True)
    p,run,metric=build(o)
    if o.forward_only:
        out,forward=measure(run,p,o.repeats)
        np.testing.assert_array_equal(out.status,0)
        report=dict(field=o.field,model=o.model,stepping=o.stepping,particles=o.particles,
            horizon_s=o.horizon,forward=forward,particle_chunking=False,kernel_audit=o.kernel_audit,
            peak_scope='Fresh process: setup and first forward execution; no AD executable built',
            jax=jax.__version__,device=jax.devices()[0].device_kind,cold_compile=o.cold_compile)
        o.output.parent.mkdir(parents=True,exist_ok=True)
        o.output.write_text(json.dumps(report,indent=2)+'\n')
        print('Forward',forward['warm_ms'],'ms; saved',o.output,flush=True)
        return
    def objective(p):
        out=run(p,adjoint='accepted')
        return metric(out.ys,out.y),out
    # Compile and execute AD first: the allocator peak includes this case's
    # setup/AD compilation/execution, before other executables are built.
    ((value,out),gradient),ad=measure(jax.value_and_grad(objective,has_aux=True),p,o.repeats)
    print('Complete AD',ad['warm_ms'],'ms',flush=True)
    ordinary,forward=measure(run,p,o.repeats)
    ad['allocator_peak_scope']='Fresh case setup/AD compilation/first execution, before forward executable'
    forward['allocator_peak_scope']='Cumulative after AD; use --forward-only for an isolated forward peak'
    np.testing.assert_array_equal(out.status,0)
    np.testing.assert_allclose(out.ys,ordinary.ys,rtol=3e-12,atol=1e-15)
    np.testing.assert_array_equal(out.stats[:6],ordinary.stats[:6])
    if not np.isfinite(gradient).all():raise AssertionError('nonfinite gradient')
    report=dict(field=o.field,model=o.model,stepping=o.stepping,particles=o.particles,horizon_s=o.horizon,
        execution=o.execution,reverse_execution=o.reverse_execution,spatial_kicks=o.model=='gc',
        moment_tolerance=o.moment_tolerance,rtol=o.rtol,atol=o.atol,seed=o.seed,
        jax=jax.__version__,device=jax.devices()[0].device_kind,particle_chunking=False,
        dofs=('pitch, energy, first coil DOF' if o.field=='coils' else 'pitch, energy, birth radius') if o.kernel_audit else (
            'coil Fourier/current parameters and three birth controls' if o.field=='coils' else 'three birth controls; fixed equilibrium'),
        kernel_audit=o.kernel_audit,cold_compile=o.cold_compile,
        forward=forward,value_and_gradient=ad,total_over_forward=ad['warm_ms']/forward['warm_ms'],
        value=float(value),gradient_norm=float(jnp.linalg.norm(gradient)),
        collision_updates=np.asarray(out.stats.accepted).tolist(),
        accepted_orbit_steps=np.asarray(out.stats.accepted_orbit_steps).tolist())
    if o.check_gradient:
        recorded,tapes=jax.jit(lambda p:run(p,record=True))(p)
        def frozen(p):return metric(*run(p,tapes=tapes))
        direction=jnp.array([.2,.3,.1]) if o.kernel_audit else jnp.asarray(np.random.default_rng(24).normal(size=p.size))
        if not o.kernel_audit:direction/=jnp.linalg.norm(direction)
        _,expected=jax.jit(lambda p:jax.jvp(frozen,(p,),(direction,)))(p)
        actual=jnp.vdot(gradient,direction)
        np.testing.assert_allclose(actual,expected,rtol=5e-7,atol=1e-9)
        report.update(directional_reverse=float(actual),directional_frozen_jvp=float(expected))
    o.output.parent.mkdir(parents=True,exist_ok=True)
    o.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Forward',forward['warm_ms'],'ms; complete/forward',report['total_over_forward'],flush=True)
    print('Saved',o.output,flush=True)


if __name__=='__main__':main()
