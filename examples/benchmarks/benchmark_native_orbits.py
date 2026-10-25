"""Forward native/Diffrax comparison on bundled coil or VMEC/Boozer fields.

Each pair uses identical births, fields, save times, error blocks and stepping
mode. Fixed runs use the same dt. All executions block on device completion;
setup and reference integration are excluded. No collisions, AD or chunking.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.dynamics import Particles, GuidingCenter, Lorentz, _axis_regular, _to_axis_regular, _from_axis_regular
from essos.electric_field import Electric_field_zero
from essos.boozer import guiding_center_rhs
from essos.rk import prepare_solver
from essos.rk import _boozer_error_norm
from essos.rk import OrbitErrorNorm
from essos.rk import solve_rk
from .benchmark_native_boozer_alpha import PARAMETERS, setup, compiler_options, compiled_bytes

ROOT=Path(__file__).resolve().parents[2]


def case(options):
    rng=np.random.default_rng(options.seed)
    speed=float(np.sqrt(2*options.energy_ev*ONE_EV/M))
    if options.field=='coils':
        field=BiotSavart(Coils.from_json(str(ROOT/'examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json')))
        points=jnp.column_stack((jnp.linspace(1.23,1.27,options.particles),jnp.zeros((options.particles,2))))
        p=Particles(points,mass=M,charge=Q,energy=options.energy_ev*ONE_EV,
            initial_vparallel_over_v=jnp.linspace(.35,.65,options.particles),
            field=field if options.model=='lorentz' else None,phase_angle_full_orbit=jnp.pi)
        initial=(jnp.column_stack((points,p.initial_vparallel)) if options.model=='gc' else
                 jnp.column_stack((p.initial_xyz_fullorbit,p.initial_vxvyvz)))
        rhs=(lambda t,y,mu:GuidingCenter(t,y,(field,p,Electric_field_zero()))) if options.model=='gc' else (
             lambda t,y,mu:Lorentz(t,y,(field,p)))
        norm=OrbitErrorNorm(options.rtol,options.atol,speed,velocity_components=1 if options.model=='gc' else 3)
        event=None
        moment=jnp.zeros(options.particles)
    elif options.field=='vmec':
        field=Vmec(str(options.wout),ntheta=8,nphi=8)
        points=jnp.asarray(np.column_stack((np.full(options.particles,.3),rng.uniform(0,2*np.pi,options.particles),
                                           rng.uniform(0,2*np.pi/field.nfp,options.particles))))
        p=Particles(points,mass=M,charge=Q,energy=options.energy_ev*ONE_EV,
                    initial_vparallel_over_v=jnp.asarray(rng.uniform(-.7,.7,options.particles)))
        initial=jax.vmap(_to_axis_regular)(jnp.column_stack((points,p.initial_vparallel)))
        rhs=_axis_regular(lambda t,y,mu:GuidingCenter(t,y,(field,p,Electric_field_zero())))
        norm=OrbitErrorNorm(options.rtol,options.atol,speed,
                            position_map=lambda y:field.to_xyz(_from_axis_regular(y)[:3]))
        event=lambda t,y,args,**kwargs:1.-y[0]**2-y[1]**2
        moment=jnp.zeros(options.particles)
    else:
        settings=argparse.Namespace(**(PARAMETERS|vars(options)))
        settings.birth_s=.3
        field,initial,moment,speed,_,_=setup(settings)
        rhs=lambda t,y,mu:guiding_center_rhs(field,y,mu,M,Q)
        norm=lambda y0,y1,error:_boozer_error_norm(error,speed,options.rtol,options.atol)
        event=lambda t,y,args,**kwargs:1.-y[0]**2-y[1]**2
    return field,initial,moment,speed,rhs,norm,event


def functions(options,data,method,tolerance=None):
    field,initial,moment,speed,rhs,norm,event=data
    adaptive=options.stepping=='adaptive'
    times=np.linspace(0.,options.horizon,options.saves)
    dt=options.horizon/4 if adaptive else options.horizon/options.steps
    rtol,atol=(options.rtol,options.atol) if tolerance is None else (tolerance,tolerance)
    if tolerance is not None:
        # Scale the same normalized physical/chart error budget for the reference.
        original=norm
        norm=lambda y0,y1,e:original(y0,y1,e)*(options.rtol+options.atol)/(rtol+atol)
    if options.field=='boozer':
        if options.execution=='cuda' and tolerance is None:
            from essos.rk import _prepare_boozer_gpu as prepare_orbit_solver
            solve=prepare_orbit_solver(field,M,Q,speed,times,dt,rtol,atol,options.max_steps,
                                       method=method,adaptive=adaptive)
        else:
            solve=lambda y,mu:solve_rk(rhs,y,mu,t1=options.horizon,dt0=dt,ts=times,method=method,
                adaptive=adaptive if tolerance is None else True,error_norm=norm,event=event,
                rtol=rtol,atol=atol,max_steps=options.max_steps)
    else:
        prepared=prepare_solver(field,mass=M,charge=Q,energy=options.energy_ev*ONE_EV,times=times,
            dt0=dt,model=options.model,method=method,adaptive=adaptive,rtol=rtol,atol=atol,
            max_steps=options.max_steps,execution=options.execution)
        solve=lambda y,mu:prepared(y)
    def native_one(y,mu):
        out=solve(y,mu)
        return out.ys,out.y,out.t,out.status,jnp.stack(out.stats)

    # Feed the same scalar embedded block error to Diffrax's ordinary I controller.
    # Its default component RMS error would impose a different acceptance test.
    base=diffrax.Dopri5 if method=='dopri5' else diffrax.Dopri8
    class PhysicalSolver(base):
        physical_norm: object=eqx.field(static=True,default=None)
        def step(self,terms,t0,t1,y0,args,solver_state,made_jump):
            y1,error,dense,state,result=super().step(terms,t0,t1,y0,args,solver_state,made_jump)
            return y1,jnp.full_like(error,self.physical_norm(y0,y1,error)),dense,state,result
    solver=PhysicalSolver(physical_norm=norm) if adaptive or tolerance is not None else base()
    control=(diffrax.PIDController(rtol=0.,atol=1.,pcoeff=0.,icoeff=1.,dcoeff=0.,factormin=.2,factormax=5.)
             if adaptive or tolerance is not None else diffrax.ConstantStepSize())
    def diffrax_one(y,mu):
        out=diffrax.diffeqsolve(diffrax.ODETerm(rhs),solver,t0=0.,t1=options.horizon,dt0=dt,y0=y,args=mu,
            stepsize_controller=control,event=None if event is None else diffrax.Event(event,
                root_finder=__import__('optimistix').Bisection(rtol=1e-12,atol=1e-12)),
            saveat=diffrax.SaveAt(subs=[diffrax.SubSaveAt(ts=jnp.asarray(times)),diffrax.SubSaveAt(t1=True)]),
            max_steps=options.max_steps,throw=False)
        status=jnp.where(out.result==diffrax.RESULTS.successful,0,
                         jnp.where(out.result==diffrax.RESULTS.event_occurred,1,2))
        counts=jnp.stack([out.stats[k] for k in ('num_steps','num_accepted_steps','num_rejected_steps')])
        return out.ys[0],out.ys[1][-1],out.ts[1][-1],status,counts
    return jax.jit(jax.vmap(native_one),compiler_options=compiler_options()),jax.jit(jax.vmap(diffrax_one),compiler_options=compiler_options())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field',choices=['coils','vmec','boozer'],default='coils')
    parser.add_argument('--model',choices=['gc','lorentz'],default='gc')
    parser.add_argument('--stepping',choices=['fixed','adaptive'],default='adaptive')
    parser.add_argument('--execution',choices=['jax','cuda'],default='cuda')
    parser.add_argument('--cases',nargs='+',default=['native_dopri8','diffrax_dopri8'])
    for name,value in dict(particles=64,horizon=2e-5,saves=8,steps=160,repeats=3,seed=42,energy_ev=4000.,
                           rtol=1e-9,atol=1e-9,max_steps=4096,mboz=32,nboz=32,mode_cut=1e-4).items():
        parser.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    parser.add_argument('--wout',type=Path,default=PARAMETERS['wout'])
    parser.add_argument('--field-cache',type=Path)
    parser.add_argument('--output',type=Path,default=Path('output/native_forward_comparison'))
    options=parser.parse_args(argv)
    if options.model=='lorentz' and options.field!='coils': parser.error('Lorentz requires Cartesian coils')
    if options.field_cache is not None and not options.field_cache.is_file(): parser.error('Missing --field-cache')
    if min(options.particles,options.horizon,options.steps,options.repeats,options.atol)<=0 or options.saves<2 or options.rtol<0:
        parser.error('Require positive counts, horizon, steps, repeats and atol; saves>=2, rtol>=0')
    if (options.output/'comparison.json').exists(): parser.error('Choose a fresh output directory')
    options.output.mkdir(parents=True,exist_ok=True)
    print(f'{jax.devices()}; {options.field}/{options.model}; {options.stepping}; {options.execution}',flush=True)
    data=case(options)
    initial,moment=data[1:3]
    jax.block_until_ready(data[1:4])
    report=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        device=jax.devices()[0].device_kind,jax=jax.__version__,collisions=False,reverse_ad=False,
        birth_sha256=hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(),
        input_sha256=hashlib.sha256((ROOT/'examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json'
                                    if options.field=='coils' else options.wout).read_bytes()).hexdigest(),
        compiled_buffers_scope='argument + output + temporary - alias; excludes executable constants and resident setup data',
        rows={})
    outputs={}; executables={}
    for name in options.cases:
        backend,method=name.split('_',1)
        if backend not in ('native','diffrax') or method not in ('dopri5','dopri8'): parser.error('Invalid --cases')
        native,reference=functions(options,data,method)
        print('Compiling '+name,flush=True)
        start=perf_counter(); exe=(native if backend=='native' else reference).lower(initial,moment).compile()
        compile_s=perf_counter()-start
        output=jax.block_until_ready(exe(initial,moment))
        jax.block_until_ready(exe(initial,moment))
        outputs[name]=output;executables[name]=exe
        report['rows'][name]=dict(compile_s=compile_s,compiled_buffers_bytes=compiled_bytes(exe),warm_ms=[])
    for iteration in range(options.repeats):
        for name in list(executables) if iteration%2==0 else reversed(list(executables)):
            start=perf_counter(); jax.block_until_ready(executables[name](initial,moment))
            report['rows'][name]['warm_ms'].append(1000*(perf_counter()-start))
    _,reference=functions(options,data,'dopri8',tolerance=1e-12)
    truth=jax.block_until_ready(reference(initial,moment))[0]
    for name,row in report['rows'].items():
        states,_,_,status,counts=map(np.asarray,outputs[name]); target=np.asarray(truth)
        valid=np.isfinite(states).all(-1)&np.isfinite(target).all(-1)
        if options.field=='vmec':
            mapped=jax.jit(jax.vmap(jax.vmap(lambda y:data[0].to_xyz(_from_axis_regular(y)[:3]))))
            position=np.asarray(mapped(jnp.asarray(states)))-np.asarray(mapped(jnp.asarray(target)))
        else: position=states[:,:,:3]-target[:,:,:3]
        row.update(warm_median_ms=float(np.median(row['warm_ms'])),accepted=int(counts[:,1].sum()),
            rejected=int(counts[:,2].sum()),failed=int(np.count_nonzero(status>1)),lost=int(np.count_nonzero(status==1)),
            max_position_or_chart_error=float(np.max(np.linalg.norm(position[valid],axis=-1))),
            max_velocity_error_m_s=float(np.max(np.linalg.norm(
                (states[:,:,3:6 if options.model=='lorentz' else 4]-target[:,:,3:6 if options.model=='lorentz' else 4])[valid],axis=-1))),
            max_velocity_error_over_birth_speed=float(np.max(np.linalg.norm(
                (states[:,:,3:6 if options.model=='lorentz' else 4]-target[:,:,3:6 if options.model=='lorentz' else 4])[valid],axis=-1))/data[3]))
        print(f'{name}: {row["warm_median_ms"]:.3f} ms, {row["accepted"]} accepted, {row["rejected"]} rejected',flush=True)
    report['gpu_memory']=jax.devices()[0].memory_stats()
    report['peak_scope']='cumulative across setup, compilation, all cases and reference; not a per-solver peak'
    (options.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (options.output/'comparison.csv').open('w',newline='') as f:
        rows=[dict(case=name,**{k:v for k,v in row.items() if k!='warm_ms'}) for name,row in report['rows'].items()]
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


if __name__=='__main__': main()
