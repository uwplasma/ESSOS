"""Matched forward split timing and memory on bundled coil/VMEC/Boozer fields.

Each case runs in a fresh process when --case all is selected. Fixed cases
use identical orbit/collision intervals; adaptive native cases adapt both.
Diffrax uses the existing fixed SDE tracer. Boozer's existing collisional
baseline is its fixed Dopri8/Euler implementation, labelled separately.
Field construction is excluded from blocked warm timings and included in
allocator peaks. No custom reverse rule or particle chunking is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from essos.background_species import BackgroundSpecies
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV, SPEED_OF_LIGHT
from essos.dynamics import Particles, _to_axis_regular, _from_axis_regular, _axis_regular, _SDE_MODELS, _EVENT_ROOT_FINDER
from essos.electric_field import Electric_field_zero
from essos.split import trace_split
from .benchmark_native_orbits import case, functions
from .benchmark_native_boozer_alpha import PARAMETERS, compiler_options, compiled_bytes

MODULE = 'examples.benchmarks.benchmark_split_orbits'


def build(options):
    data = case(options); field, initial, moment, speed, *_ = data
    if options.field=='coils':
        # Materialize cached geometry during field setup, before the outer JIT.
        field.coils.gamma, field.coils.gamma_dash, field.coils.currents
    if options.field == 'vmec':
        physical = jax.vmap(_from_axis_regular)(initial); points = physical[:, :3]; pitches = physical[:, 3]/speed
    elif options.field == 'boozer':
        points = jnp.column_stack((initial[:,0]**2+initial[:,1]**2,jnp.arctan2(initial[:,1],initial[:,0]),initial[:,2]))
        pitches = initial[:,3]/speed
    else:
        points = initial[:,:3]; pitches = jnp.linspace(.35,.65,options.particles)
    birth = Particles(points,mass=M,charge=Q,energy=options.energy_ev*ONE_EV,initial_vparallel_over_v=pitches,
        initial_xyz_fullorbit=initial[:,:3] if options.model=='lorentz' else None,
        initial_vxvyvz=initial[:,3:] if options.model=='lorentz' else None)
    birth.random_keys = jax.random.split(jax.random.PRNGKey(options.seed),options.particles)
    background = BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),jnp.array([1e20]),jnp.array([1000.]))
    times = jnp.linspace(0.,options.horizon,options.saves)
    adaptive = options.stepping == 'adaptive'; dt = options.horizon/4 if adaptive else options.horizon/options.steps
    fingerprint = hashlib.sha256(np.asarray(jnp.column_stack((points,pitches))).tobytes()).hexdigest()
    if options.case.startswith('native_') and options.case != 'native_collisionless':
        def run():
            out = trace_split(field,birth,background,model='GuidingCenter' if options.model=='gc' else 'FullOrbit',
                tmax=options.horizon,timestep=dt,collision_dtmax=options.horizon/2 if adaptive else dt,
                orbit_dt0=dt,adaptive=adaptive,orbit_adaptive=adaptive,times=times,
                rtol=options.rtol,atol=options.atol,collision_tolerance=options.collision_tolerance,
                collision_controller=options.collision_controller,moment_tolerance=options.moment_tolerance,
                spatial_diffusion=options.spatial_diffusion,
                max_steps=options.max_steps,max_orbit_steps=options.max_orbit_steps,
                execution='independent' if options.case=='native_independent' else 'shared')
            return out.ys,out.y,out.t,out.status,jnp.stack(out.stats,axis=-1)
        label = 'Native Dopri8 / Milstein; '+(options.collision_controller if adaptive else 'fixed')
    elif options.case == 'native_collisionless':
        native,_ = functions(options,data,'dopri8')
        def run():
            ys,y,t,status,counts = native.__wrapped__(initial,moment)
            counts = jnp.pad(counts,((0,0),(0,4)))
            return ys,y,t,status,counts
        label = 'Native Dopri8; no collisions'
    elif options.field == 'boozer':
        from essos.boozer import _start, _advance
        def run():
            carry,first = _start(field,initial,moment)
            carry,states = _advance(field,dt,options.steps//(options.saves-1),M,Q,background,0.,
                carry,birth.random_keys,0,options.saves-1,'dopri8',1e-9)
            states = jnp.concatenate((first[:,None],states),axis=1)
            status = jnp.where(carry[6]>=0,4,jnp.where(carry[4]>=0,1,0))
            counts = jnp.stack((carry[10],carry[10],jnp.zeros_like(carry[10])),axis=-1)
            return states,states[:,-1],carry[2],status,jnp.pad(counts,((0,0),(0,4)))
        label = 'Existing Boozer fixed Dopri8 / Euler collisions (not Diffrax)'
    else:
        gc, vmec = options.model=='gc', options.field=='vmec'
        drift,noise,stepper,dimension = _SDE_MODELS['GuidingCenterCollisionsMuFixed' if gc else 'FullOrbitCollisions']
        scale = jnp.array([1.,1.,1.,SPEED_OF_LIGHT,SPEED_OF_LIGHT**2*M] if gc else [1.]*6)
        if gc:
            B = jax.vmap(field.AbsB)(points)
            y0 = jnp.column_stack((points,birth.initial_vparallel,M*birth.initial_vperpendicular**2/(2*B)))/scale
            if vmec: y0 = jax.vmap(_to_axis_regular)(y0)
        else: y0 = initial/scale
        args = (field,birth,Electric_field_zero(),background,1.) if gc else (field,birth,background)
        rhs = lambda t,y,a:drift(t,y,args)
        diffusion = lambda t,y,a:noise(t,y,args)
        if vmec: rhs,diffusion = _axis_regular(rhs),_axis_regular(diffusion)
        def one(y,key):
            bm = diffrax.VirtualBrownianTree(0.,options.horizon,tol=dt*.5,shape=(dimension,),key=key,
                levy_area=diffrax.SpaceTimeTimeLevyArea)
            terms = diffrax.MultiTerm(diffrax.ODETerm(rhs),diffrax.ControlTerm(diffusion,bm))
            event = diffrax.Event(lambda t,y,args,**k:1-y[0]**2-y[1]**2,root_finder=_EVENT_ROOT_FINDER) if vmec else None
            out = diffrax.diffeqsolve(terms,stepper(),t0=0.,t1=options.horizon,dt0=dt,y0=y,
                stepsize_controller=diffrax.ConstantStepSize(),event=event,
                saveat=diffrax.SaveAt(subs=[diffrax.SubSaveAt(ts=times),diffrax.SubSaveAt(t1=True)]),
                max_steps=options.max_steps,throw=False)
            saved,terminal = out.ys[0],out.ys[1][-1]
            saved = jnp.where(jnp.isfinite(saved).all(axis=1)[:,None],saved,terminal)
            if vmec: saved,terminal = jax.vmap(_from_axis_regular)(saved),_from_axis_regular(terminal)
            status = jnp.where(out.result==diffrax.RESULTS.successful,0,jnp.where(out.result==diffrax.RESULTS.event_occurred,1,4))
            counts = jnp.array([out.stats[k] for k in ('num_steps','num_accepted_steps','num_rejected_steps')]+[0]*4)
            return saved*scale,terminal*scale,out.ts[1][-1],status,counts
        run = lambda:jax.vmap(one)(y0,birth.random_keys)
        label = f'Existing Diffrax {stepper.__name__}, fixed SDE steps'
    return run,label,fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field',choices=('coils','vmec','boozer'),default='coils')
    parser.add_argument('--model',choices=('gc','lorentz'),default='gc')
    parser.add_argument('--stepping',choices=('fixed','adaptive'),default='fixed')
    parser.add_argument('--case',choices=('all','native_shared','native_independent','reference_fixed','native_collisionless'),default='all')
    parser.add_argument('--particles',type=int,default=64); parser.add_argument('--saves',type=int,default=8)
    parser.add_argument('--horizon',type=float); parser.add_argument('--steps',type=int)
    parser.add_argument('--repeats',type=int,default=3); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--energy-ev',type=float,default=4000.)
    parser.add_argument('--rtol',type=float,default=1e-9); parser.add_argument('--atol',type=float,default=1e-9)
    parser.add_argument('--collision-tolerance',type=float,default=.001)
    parser.add_argument('--collision-controller',choices=('moments','conservative'),default='moments')
    parser.add_argument('--moment-tolerance',type=float,default=.005)
    parser.add_argument('--spatial-diffusion',action=argparse.BooleanOptionalAction,default=None,
        help='GC position diffusion; enabled by default for native GC; legacy reference omits it')
    parser.add_argument('--max-steps',type=int,default=4096); parser.add_argument('--max-orbit-steps',type=int,default=256)
    parser.add_argument('--field-cache',type=Path); parser.add_argument('--output',type=Path,default=Path('output/split_forward'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cold-compile',action='store_true',help='Disable the persistent cache for measured solver compilation')
    options = parser.parse_args()
    if min(options.particles,options.repeats,options.max_steps,options.max_orbit_steps)<1 or options.saves<2:
        parser.error('require positive work limits and at least two saves')
    if not all(np.isfinite(x) and x>0 for x in (options.energy_ev,options.atol,options.collision_tolerance,options.moment_tolerance)) or options.rtol<0:
        parser.error('require finite positive energy, atol and collision tolerance; rtol>=0')
    if options.field_cache is not None and not options.field_cache.is_file(): parser.error('missing --field-cache')
    if options.model=='lorentz' and options.field!='coils': parser.error('Lorentz requires Cartesian coil fields')
    if options.case=='reference_fixed' and options.stepping!='fixed': parser.error('the reference is fixed-step only')
    if options.horizon is None: options.horizon=2e-6 if options.model=='lorentz' else 2e-5
    if options.steps is None: options.steps=1792 if options.model=='lorentz' else 224
    if options.steps<1 or not np.isfinite(options.horizon) or options.horizon<=0: parser.error('positive steps and horizon required')
    if options.steps%(options.saves-1): parser.error('fixed steps must be divisible by saves-1 for aligned clocks')
    options.output.mkdir(parents=True,exist_ok=True)
    if options.case=='all':
        rows={}; cases=['native_independent' if jax.default_backend()=='gpu' else 'native_shared','native_collisionless']
        if options.stepping=='fixed': cases.append('reference_fixed')
        for name in cases:
            saved=options.output/name/'comparison.json'
            if saved.is_file() and options.resume:
                previous=json.loads(saved.read_text())
                expected={k:getattr(options,k) for k in ('field','model','stepping','particles','saves','steps','seed','rtol','atol','collision_tolerance','collision_controller','moment_tolerance')}
                expected['horizon_s']=options.horizon
                expected['spatial_diffusion']=bool(options.model=='gc' and name in ('native_shared','native_independent') and options.spatial_diffusion is not False)
                if not all(previous.get(k)==v for k,v in expected.items()) or previous.get('energy_ev',4000.)!=options.energy_ev:
                    parser.error('saved case settings differ; choose a fresh output directory')
                rows[name]=previous
                continue
            command=[sys.executable,'-m',MODULE,*sys.argv[1:],'--case',name,'--output',str(options.output/name)]
            subprocess.run(command,check=True)
            rows[name]=json.loads((options.output/name/'comparison.json').read_text())
        (options.output/'comparison.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
        return
    # Reuse the orbit PR's field/birth preparation and collisionless baseline.
    for name,value in PARAMETERS.items():
        if not hasattr(options,name): setattr(options,name,value)
    options.execution='cuda' if jax.default_backend()=='gpu' else 'jax'
    if options.case=='native_independent' and options.execution!='cuda': parser.error('independent execution requires GPU')
    run,label,fingerprint=build(options)
    if options.cold_compile:
        from jax.experimental.compilation_cache import compilation_cache
        jax.config.update('jax_enable_compilation_cache',False)
        # Setup may already have initialized JAX's cached cache-use decision.
        compilation_cache.reset_cache()
    if options.cold_compile:print(f'{options.field}/{options.model}/{options.case}: lowering',flush=True)
    start=perf_counter(); lowered=jax.jit(run,compiler_options=compiler_options()).lower()
    lower_s=perf_counter()-start
    if options.cold_compile:print(f'Lowering {lower_s:.3f} s; compiling',flush=True)
    start=perf_counter();executable=lowered.compile();backend_compile_s=perf_counter()-start
    compile_s=lower_s+backend_compile_s
    if options.cold_compile:print(f'Compilation {backend_compile_s:.3f} s; first execution',flush=True)
    start=perf_counter();out=jax.block_until_ready(executable());first_execution_s=perf_counter()-start
    jax.block_until_ready(executable())
    samples=[]
    for _ in range(options.repeats):
        start=perf_counter(); out=jax.block_until_ready(executable()); samples.append(1000*(perf_counter()-start))
    states,terminal,end,status,counts=map(np.asarray,out)
    memory=jax.devices()[0].memory_stats() or {}
    report=dict(field=options.field,model=options.model,stepping=options.stepping,case=options.case,
        reference=label,backend=jax.default_backend(),device=jax.devices()[0].device_kind,jax=jax.__version__,
        particles=options.particles,energy_ev=options.energy_ev,horizon_s=options.horizon,saves=options.saves,steps=options.steps,
        seed=options.seed,birth_sha256=fingerprint,rtol=options.rtol,atol=options.atol,
        collision_tolerance=options.collision_tolerance,collision_controller=options.collision_controller,
        moment_tolerance=options.moment_tolerance,
        spatial_diffusion=bool(options.model=='gc' and options.case in ('native_shared','native_independent') and options.spatial_diffusion is not False),
        lower_s=lower_s,backend_compile_s=backend_compile_s,compile_s=compile_s,lower_and_compile_s=compile_s,
        first_execution_s=first_execution_s,startup_s=compile_s+first_execution_s,
        persistent_compilation_cache=bool(jax.config.jax_enable_compilation_cache),samples_ms=samples,
        warm_ms=float(np.median(samples)),compiled_buffers_bytes=compiled_bytes(executable),
        peak_allocator_bytes=memory.get('peak_bytes_in_use'),live_allocator_bytes=memory.get('bytes_in_use'),
        reserved_pool_bytes=memory.get('pool_bytes'),memory_scope='Fresh process; setup, compilation and warm executions; JAX allocator only',
        lost=int(np.sum(status==1)),failed=int(np.sum(status>1)),
        counts_total=counts[:,:5].sum(axis=0).tolist(),
        count_columns=(['collision_trials','collision_accepted','collision_rejected','orbit_trials','orbit_rejected']
            if options.case in ('native_shared','native_independent') else ['solver_steps','solver_accepted','solver_rejected','unused','unused']),
        peak_pending=int(counts[:,5].max()),max_accepted_collision_error=float(counts[:,6].max()),
        terminal_sha256=hashlib.sha256(terminal.tobytes()).hexdigest())
    if options.case in ('native_shared','native_independent'):
        report.update(collision_updates_total=int(counts[:,1].sum()),accepted_orbit_steps_total=int(counts[:,7].sum()),
            max_orbit_steps_per_collision=int(counts[:,8].max()))
    if report['failed']: raise RuntimeError(f"solver failed: {report['failed']} particles")
    np.savez_compressed(options.output/'states.npz',states=states,terminal=terminal,status=status,counts=counts)
    (options.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"{options.field}/{options.model}/{options.stepping}/{options.case}: {report['warm_ms']:.3f} ms; "
          f"buffers {report['compiled_buffers_bytes']/2**20:.3f} MiB; peak {report['peak_allocator_bytes']}",flush=True)


if __name__=='__main__': main()
