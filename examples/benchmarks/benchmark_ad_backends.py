"""Matched CPU timing/memory cases for native and Diffrax orbit/collision AD.

The parent starts one fresh worker per case, sequentially. Fused objective RAM
is captured before compiling the separate forward/reverse timing executables.
Production fields, collision kernels and per-particle Brownian generators are
used unchanged. Different stochastic schemes do not share a sample path.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[2]
CASES = ('native_orbit', 'diffrax_orbit', 'native_split_adaptive',
         'native_split_fixed', 'diffrax_sde_fixed')


def fingerprints():
    from .benchmark_split_ad import SOURCES
    names = tuple(dict.fromkeys((*SOURCES, 'essos/orbit_error.py',
        'essos/electric_field.py', 'examples/benchmarks/benchmark_ad_backends.py')))
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def build(options, model, case):
    import diffrax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.background_species import BackgroundSpecies
    from essos.coils import Coils
    from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV, SPEED_OF_LIGHT
    from essos.dynamics import Particles, GuidingCenter, Lorentz, _SDE_MODELS
    from essos.electric_field import Electric_field_zero
    from essos.fields import BiotSavart
    from essos.orbit_error import OrbitErrorNorm, orbit_block_norm
    from essos.particle_losses import mean_final_radius
    from essos.rk import solve_rk
    from essos.rk_ad import replay_accepted
    from essos.split import trace_split

    base = Coils.from_json(str(ROOT/'examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json'))
    theta = jnp.asarray(base.dofs)
    reference = BiotSavart(base)
    axis = float(reference.r_axis), float(reference.z_axis)
    points = jnp.column_stack((jnp.linspace(1.23, 1.27, options.particles),
                               jnp.zeros((options.particles, 2))))
    pitches = jnp.linspace(.35, .65, options.particles)
    energy = 4000*ONE_EV
    speed = float(np.sqrt(2*energy/M))
    horizon = options.gc_horizon if model == 'gc' else options.lorentz_horizon
    steps = options.fixed_gc_steps if model == 'gc' else options.fixed_lorentz_steps
    dt = horizon/steps
    times = jnp.linspace(0., horizon, options.saves)
    keys = jax.random.split(jax.random.key(42), options.particles)
    background = BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),jnp.array([1e20]),jnp.array([1000.]))
    electric = Electric_field_zero()
    velocity_components = 1 if model == 'gc' else 3
    norm = OrbitErrorNorm(options.tolerance,options.tolerance,speed,
                          velocity_components=velocity_components)
    scales = jnp.array([1.]*3+[speed]*velocity_components)
    controller = diffrax.PIDController(rtol=0.,atol=2*options.tolerance*scales,
        norm=lambda error:orbit_block_norm(error,velocity_components),pcoeff=.4,icoeff=.3,dcoeff=0.)
    sde_model = 'GuidingCenterCollisionsMuFixed' if model == 'gc' else 'FullOrbitCollisions'
    drift, noise, sde_solver, noise_dim = _SDE_MODELS[sde_model]
    unit = jnp.array([1.,1.,1.,SPEED_OF_LIGHT,SPEED_OF_LIGHT**2*M]) if model == 'gc' else jnp.ones(6)
    collisional = case not in ('native_orbit','diffrax_orbit')
    scale = jnp.array([1.,1.,1.,speed,energy] if model == 'gc' and collisional else
                       [1.]*3+[speed]*velocity_components)
    adaptive = case == 'native_split_adaptive'
    outer_capacity = options.accepted_capacity if adaptive else steps+2*options.saves

    def initialize(parameters):
        field = BiotSavart(base.with_dofs(parameters))
        birth = Particles(points,mass=M,charge=Q,energy=energy,
            initial_vparallel_over_v=pitches,field=field if model == 'lorentz' else None,
            phase_angle_full_orbit=jnp.pi)
        birth.random_keys = keys
        if model == 'gc':
            initial = jnp.column_stack((points,birth.initial_vparallel))
            rhs = lambda t,y,a:GuidingCenter(t,y,(field,birth,electric))
        else:
            initial = jnp.column_stack((birth.initial_xyz_fullorbit,birth.initial_vxvyvz))
            rhs = lambda t,y,a:Lorentz(t,y,(field,birth))
        return field,birth,initial,rhs

    def cost(parameters, adjoint='accepted', *, record=False, tapes=None):
        field,birth,initial,rhs = initialize(parameters)
        if case.startswith('native_split'):
            out = trace_split(field,birth,background,
                model='GuidingCenter' if model == 'gc' else 'FullOrbit',
                integrator='essos_dopri8',tmax=horizon,
                timestep=horizon/4 if adaptive else dt,
                collision_dtmax=horizon/2 if adaptive else dt,
                orbit_dt0=horizon/4 if adaptive else dt,
                adaptive=adaptive,orbit_adaptive=True,times=times,
                collision_tolerance=options.collision_tolerance,collision_controller='conservative',
                rtol=options.tolerance,atol=options.tolerance,max_steps=options.max_steps,
                max_orbit_steps=options.max_orbit_steps,adjoint=adjoint,
                adjoint_checkpoint_steps=options.checkpoint_steps,
                adjoint_orbit_checkpoint_steps=options.checkpoint_steps,
                adjoint_max_accepted_steps=outer_capacity,adjoint_max_orbit_steps=options.orbit_capacity,
                _record_ad=record,_replay_tapes=tapes)
            if record:
                return out
            if tapes is not None:
                return mean_final_radius(out[0],*axis),out
            return mean_final_radius(out.ys,*axis),(out.ys,jnp.stack(out.stats,axis=-1),out.status)

        if case == 'native_orbit':
            if tapes is not None:
                ys,final = jax.vmap(lambda y,tape:replay_accepted(rhs,y,None,tape,times,
                    method='dopri8'))(initial,tapes)
                return mean_final_radius(ys,*axis),(ys,final)
            def one(y):
                return solve_rk(rhs,y,t1=horizon,dt0=horizon/4,ts=times,method='dopri8',
                    rtol=options.tolerance,atol=options.tolerance,error_norm=norm,max_steps=options.max_steps,
                    adjoint=adjoint,adjoint_checkpoint_steps=options.checkpoint_steps,
                    adjoint_parameter_accumulation='segment',adjoint_max_accepted_steps=options.accepted_capacity,
                    _record_ad=record)
            out = jax.vmap(one)(initial)
            if record:
                return out
            return mean_final_radius(out.ys,*axis),(out.ys,jnp.stack(out.stats,axis=-1),out.status)

        if case == 'diffrax_orbit':
            def one(y):
                out = diffrax.diffeqsolve(diffrax.ODETerm(rhs),diffrax.Dopri8(),t0=0.,t1=horizon,
                    dt0=horizon/4,y0=y,saveat=diffrax.SaveAt(ts=times),stepsize_controller=controller,
                    adjoint=diffrax.RecursiveCheckpointAdjoint(),max_steps=options.max_steps,throw=False)
                return out.ys,jnp.array([out.stats[k] for k in
                    ('num_steps','num_accepted_steps','num_rejected_steps')]),jnp.where(
                        out.result == diffrax.RESULTS.successful,0,1)
        else:
            args = (field,birth,electric,background,1.) if model == 'gc' else (field,birth,background)
            if model == 'gc':
                B = jax.vmap(field.AbsB)(points)
                initial = jnp.column_stack((points,birth.initial_vparallel,M*birth.initial_vperpendicular**2/(2*B)))
            def one(y,key):
                brownian = diffrax.VirtualBrownianTree(0.,horizon,tol=dt*.5,shape=(noise_dim,),key=key,
                    levy_area=diffrax.SpaceTimeTimeLevyArea)
                terms = diffrax.MultiTerm(diffrax.ODETerm(lambda t,y,a:drift(t,y,args)),
                    diffrax.ControlTerm(lambda t,y,a:noise(t,y,args),brownian))
                out = diffrax.diffeqsolve(terms,sde_solver(),t0=0.,t1=horizon,dt0=dt,y0=y/unit,
                    saveat=diffrax.SaveAt(ts=times),stepsize_controller=diffrax.ConstantStepSize(),
                    adjoint=diffrax.RecursiveCheckpointAdjoint(),max_steps=options.max_steps,throw=False)
                return out.ys*unit,jnp.array([out.stats[k] for k in
                    ('num_steps','num_accepted_steps','num_rejected_steps')]),jnp.where(
                        out.result == diffrax.RESULTS.successful,0,1)
        ys,counts,status = jax.vmap(one)(initial) if case == 'diffrax_orbit' else jax.vmap(one)(initial,keys)
        return mean_final_radius(ys,*axis),(ys,counts,status)

    metadata = dict(horizon=horizon,fixed_dt=dt,requested_fixed_steps=steps,
        parameters='136 bundled QA coil Fourier/normalized-current DOFs only; fixed reference axis and births',
        parameter_labels=list(base.dof_names),birth_xyz=np.asarray(points).tolist(),birth_pitch=np.asarray(pitches).tolist(),
        energy_ev=4000.,seed=42,background='Hydrogen, 1e20 m^-3, 1 keV; production coefficient model',
        spatial_kicks=False,particle_chunking=False,
        solver=('Dopri8' if not collisional else ('native Dopri8 + production Milstein kick' if case.startswith('native') else sde_solver.__name__)),
        native_outer_capacity=outer_capacity if case.startswith('native_split') else None,
        count_columns=list(('steps','accepted','rejected','orbit_steps','orbit_rejected','peak_pending','max_accepted_error')
                           if case.startswith('native_split') else ('steps','accepted','rejected')),
        derivative_policy='accepted maps, frozen controls/noise' if case.startswith('native') else 'Diffrax RecursiveCheckpointAdjoint default checkpoints',
        error_control='fixed SDE step; no rtol/atol controller' if case == 'diffrax_sde_fixed' else 'matched position/speed blocks, 2*tolerance; native I, Diffrax PI (.4,.3)')
    return theta,cost,scale,metadata


def memory(executable):
    value = executable.memory_analysis()
    fields = {name:int(getattr(value,name)) for name in
        ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
    return dict(bytes=fields['argument_size_in_bytes']+fields['output_size_in_bytes']
                +fields['temp_size_in_bytes']-fields['alias_size_in_bytes'],components=fields)


def resident():
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value = line.partition(':')
        if key in ('VmRSS','VmHWM'):
            result[key] = int(value.split()[0])*1024
    return result


def worker(options):
    import resource
    import diffrax
    import jax
    import jax.numpy as jnp
    import numpy as np
    if sys.platform != 'linux' or jax.default_backend() != 'cpu':
        raise ValueError('Run workers on Linux/WSL CPU')
    model,case = options.worker_model,options.worker_case
    source_hashes = fingerprints()
    theta,cost,scale,metadata = build(options,model,case)
    metadata.update(backend='cpu',jax=jax.__version__,diffrax=diffrax.__version__,source_sha256=source_hashes,
        input_sha256=hashlib.sha256((ROOT/'examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json').read_bytes()).hexdigest(),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        memory_scope='Fused peak includes input setup, fused compilation, first and warm execution; measured before separate forward/reverse or FD validation executables exist. Warm RSS includes resident JAX/executables/allocator. Compiler buffers are separate.',
        timing_scope='Warm blocked calls, compilation excluded. Forward is ordinary production solve; linearization is public VJP of jitted cost; reverse reuses its residuals; fused is value_and_grad. No particle chunks.')
    result=dict(complete=False,case=case,model=model,metadata=metadata)
    destination=options.output/(model+'_'+case+'.json')
    def save():
        destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    peak=lambda:int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    result['setup_peak_rss_bytes']=peak()
    total=jax.jit(jax.value_and_grad(lambda p:cost(p,'accepted'),has_aux=True))
    start=perf_counter();executable=total.lower(theta).compile()
    result['fused_compile_seconds']=perf_counter()-start
    result['fused_compile_peak_rss_bytes']=peak()
    (value,aux),gradient=jax.block_until_ready(total(theta))
    result['first_fused_peak_rss_bytes']=peak()
    ys,counts,status=map(np.asarray,aux)
    result.update(value=float(value),counts_per_particle=counts.tolist(),status=status.tolist(),
        gradient_finite=bool(np.isfinite(gradient).all()),state_finite=bool(np.isfinite(ys).all()))
    if np.any(status) or not np.isfinite(ys).all() or not np.isfinite(gradient).all():
        raise RuntimeError('Nonfinite/incomplete primal or reverse; see worker report')
    result['gradient']=np.asarray(gradient).tolist()
    result['gradient_l2']=float(jnp.linalg.norm(gradient))
    result['saved_positions']=ys[:,:,:3].tolist()
    result['fused_memory']=memory(executable)
    Path('/proc/self/clear_refs').write_text('5\n')
    result['warm_fused_baseline_rss_bytes']=resident()['VmRSS']
    samples=[]
    for _ in range(options.repeats):
        start=perf_counter();jax.block_until_ready(total(theta))
        samples.append(1000*(perf_counter()-start))
    result['fused_samples_ms']=samples
    result['fused_median_ms']=float(np.median(samples))
    result['warm_fused_peak_rss_bytes']=resident()['VmHWM']
    result['fused_peak_rss_bytes']=max(result[k] for k in ('setup_peak_rss_bytes',
        'fused_compile_peak_rss_bytes','first_fused_peak_rss_bytes','warm_fused_peak_rss_bytes'))
    save()

    forward=jax.jit(lambda p:cost(p,'none'))
    linear=jax.jit(lambda p:cost(p,'accepted'))
    linear_value,pullback,linear_aux=jax.block_until_ready(jax.vjp(linear,theta,has_aux=True))
    seed=jnp.ones_like(value)
    reverse=jax.jit(lambda pb,seed:pb(seed)[0])
    start=perf_counter();reverse_executable=reverse.lower(pullback,seed).compile()
    result['reverse_compile_seconds']=perf_counter()-start
    reverse_gradient=jax.block_until_ready(reverse(pullback,seed))
    ordinary_value,ordinary_aux=jax.block_until_ready(forward(theta))
    np.testing.assert_allclose(reverse_gradient,gradient,rtol=2e-8,atol=1e-10)
    for other_value,other_aux in ((linear_value,linear_aux),(ordinary_value,ordinary_aux)):
        np.testing.assert_allclose(other_value,value,rtol=2e-12,atol=1e-25)
        np.testing.assert_allclose((other_aux[0]-aux[0])/scale,0.,rtol=0.,atol=2e-12)
        np.testing.assert_allclose(other_aux[1],aux[1],rtol=2e-12,atol=1e-25)
        np.testing.assert_array_equal(other_aux[2],aux[2])
    result['reverse_memory']=memory(reverse_executable)
    result['forward_memory']=memory(forward.lower(theta).compile())
    result['gradient_max_abs_reverse_fused_difference']=float(jnp.max(jnp.abs(reverse_gradient-gradient)))
    functions=dict(forward=lambda:forward(theta),reverse=lambda:reverse(pullback,seed),
                   linearization=lambda:jax.vjp(linear,theta,has_aux=True))
    random=np.random.default_rng(24)
    phase_samples=[]
    for _ in range(options.repeats):
        sample={}
        for name in random.permutation(list(functions)):
            start=perf_counter();jax.block_until_ready(functions[name]())
            sample[name]=1000*(perf_counter()-start)
        phase_samples.append(sample)
    result['phase_samples_ms']=phase_samples
    result['phase_medians_ms']={name:float(np.median([row[name] for row in phase_samples])) for name in functions}
    result['reverse_over_forward']=result['phase_medians_ms']['reverse']/result['phase_medians_ms']['forward']
    result['fused_over_forward']=result['fused_median_ms']/result['phase_medians_ms']['forward']
    save()

    # Native AD is defined on its recorded accepted grid. Diffrax fixed SDE
    # already has a parameter-independent time grid and retained random path.
    oracle=forward
    if case.startswith('native'):
        recorded,tapes=jax.block_until_ready(jax.jit(lambda p:cost(p,'none',record=True))(theta))
        oracle=jax.jit(lambda p:cost(p,'none',tapes=tapes))
        frozen_value,(frozen_ys,final)=jax.block_until_ready(oracle(theta))
        np.testing.assert_allclose((frozen_ys-aux[0])/scale,0.,rtol=0.,atol=2e-12)
        np.testing.assert_allclose(frozen_value,value,rtol=2e-12,atol=1e-25)
        result['gradient_oracle']='Independent frozen accepted-grid/production-noise replay'
    else:
        result['gradient_oracle']=('Fixed-grid, fixed-production-VBT path FD' if case == 'diffrax_sde_fixed'
                                   else 'Adaptive primal FD diagnostic; controller can change the accepted grid')
    direction=jnp.asarray(np.random.default_rng(12).normal(size=theta.size))
    direction/=jnp.linalg.norm(direction)
    expected=float(jnp.dot(gradient,direction))
    if case.startswith('native'):
        tangent=jax.block_until_ready(jax.jit(lambda p,d:jax.jvp(lambda p:oracle(p)[0],(p,),(d,))[1])(theta,direction))
        np.testing.assert_allclose(tangent,expected,rtol=2e-7,atol=1e-10)
        result['frozen_directional_jvp']=float(tangent)
    fd=[]
    for epsilon in (1e-5,3e-6,1e-6):
        finite=float((oracle(theta+epsilon*direction)[0]-oracle(theta-epsilon*direction)[0])/(2*epsilon))
        fd.append(dict(epsilon=epsilon,ad=expected,finite_difference=finite,
            absolute_error=abs(finite-expected),relative_error=abs(finite-expected)/max(abs(finite),abs(expected),1e-30)))
    result['directional_fd']=fd
    if case != 'diffrax_orbit' and min(item['relative_error'] for item in fd)>2e-5 and min(item['absolute_error'] for item in fd)>1e-9:
        raise RuntimeError('Fixed-path directional FD did not agree with the reverse gradient')
    if source_hashes != fingerprints():
        raise RuntimeError('Numerical sources changed during the worker')
    result['whole_worker_peak_rss_bytes']=max(peak(),resident()['VmHWM'],result['fused_peak_rss_bytes'])
    result['complete']=True
    save()
    print(json.dumps({k:result[k] for k in ('model','case','value','phase_medians_ms','fused_median_ms',
        'reverse_over_forward','fused_over_forward','fused_peak_rss_bytes')}),flush=True)


def parser():
    command=argparse.ArgumentParser(description=__doc__)
    command.add_argument('--output',type=Path,required=True,help='New results directory')
    command.add_argument('--models',nargs='+',choices=('gc','lorentz'),default=['gc','lorentz'])
    command.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    command.add_argument('--particles',type=int,default=2)
    command.add_argument('--repeats',type=int,default=21)
    command.add_argument('--saves',type=int,default=8)
    command.add_argument('--gc-horizon',type=float,default=2e-5)
    command.add_argument('--lorentz-horizon',type=float,default=2e-6)
    command.add_argument('--tolerance',type=float,default=1e-9)
    command.add_argument('--collision-tolerance',type=float,default=.003)
    command.add_argument('--fixed-gc-steps',type=int,default=224)
    command.add_argument('--fixed-lorentz-steps',type=int,default=1792)
    command.add_argument('--max-steps',type=int,default=4096)
    command.add_argument('--max-orbit-steps',type=int,default=128)
    command.add_argument('--accepted-capacity',type=int,default=128)
    command.add_argument('--orbit-capacity',type=int,default=2048)
    command.add_argument('--checkpoint-steps',type=int,default=16)
    command.add_argument('--worker-case',choices=CASES,help=argparse.SUPPRESS)
    command.add_argument('--worker-model',choices=('gc','lorentz'),help=argparse.SUPPRESS)
    return command


def parent(options):
    import math
    if options.output.exists():
        raise ValueError('Choose a new results directory')
    options.output.mkdir(parents=True)
    report=dict(complete=False,all_cases_passed=False,
        settings={key:str(value) if isinstance(value,Path) else value for key,value in vars(options).items()},
        scope='One fresh CPU worker per case, sequential; no chunking. Identical coil DOFs, births, fields, horizons, save times and fixed-axis mean-final-radius objective.',
        collision_comparison='Production native adaptive/fixed split maps versus production fixed Diffrax SDE. Different Brownian generators/noise bases and integrators; not a pathwise accuracy or gradient-equality comparison.',
        memory_scope='Per-worker fused-objective peak RSS measured before standalone F/R and validation executables are built; includes setup and compilation. Also retain complete-worker peak and compiler buffer estimates.',
        rows=[],comparisons=[],failures=[])
    def save():
        (options.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    environment=dict(os.environ,JAX_PLATFORMS='cpu',ESSOS_CPU_DEVICES='1',ESSOS_PROGRESS='0',ESSOS_XLA_CACHE='')
    settings=[]
    for name in ('particles','repeats','saves','gc_horizon','lorentz_horizon','tolerance',
                 'collision_tolerance','fixed_gc_steps','fixed_lorentz_steps','max_steps',
                 'max_orbit_steps','accepted_capacity','orbit_capacity','checkpoint_steps'):
        settings += ['--'+name.replace('_','-'),str(getattr(options,name))]
    expected_sources=None
    expected_input=None
    for model in options.models:
        for case in options.cases:
            name=model+'_'+case
            command=[sys.executable,'-m','examples.benchmarks.benchmark_ad_backends',
                '--output',str(options.output.resolve()),'--worker-case',case,'--worker-model',model,*settings]
            print('START '+name,flush=True)
            start=perf_counter()
            log_path=options.output/(name+'.log')
            with log_path.open('w') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                try:
                    code=child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait()
                    code=124
            path=options.output/(name+'.json')
            row=json.loads(path.read_text()) if path.exists() else dict(complete=False,model=model,case=case)
            row.update(worker_exit_code=code,worker_wall_seconds=perf_counter()-start)
            if code or not row['complete']:
                report['failures'].append(dict(model=model,case=case,exit_code=code,log=log_path.name,
                    error=row.get('error','worker incomplete; inspect log')))
            else:
                hashes=row['metadata']['source_sha256']
                if expected_sources is None:
                    expected_sources=hashes
                    expected_input=row['metadata']['input_sha256']
                if hashes != expected_sources or row['metadata']['input_sha256'] != expected_input:
                    raise RuntimeError('Numerical sources or coil inputs changed between cases')
                print(json.dumps(dict(model=model,case=case,forward_ms=row['phase_medians_ms']['forward'],
                    reverse_ms=row['phase_medians_ms']['reverse'],fused_ms=row['fused_median_ms'],
                    fused_buffers_MiB=row['fused_memory']['bytes']/2**20,
                    fused_peak_MiB=row['fused_peak_rss_bytes']/2**20)),flush=True)
            report['rows'].append(row);save()
    for name,digest in (expected_sources or {}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Numerical source changed during comparison: '+name)
    for model in options.models:
        rows={row['case']:row for row in report['rows'] if row['model']==model and row['complete']}
        for native,reference in [('native_orbit','diffrax_orbit'),
                                 ('native_split_adaptive','diffrax_sde_fixed'),('native_split_fixed','diffrax_sde_fixed')]:
            if native not in rows or reference not in rows:
                continue
            a,b=rows[native],rows[reference]
            difference=math.sqrt(sum((x-y)**2 for x,y in zip(a['gradient'],b['gradient'])))
            reference_norm=math.sqrt(sum(y*y for y in b['gradient']))
            report['comparisons'].append(dict(model=model,native=native,reference=reference,
                reverse_speedup=b['phase_medians_ms']['reverse']/a['phase_medians_ms']['reverse'],
                fused_speedup=b['fused_median_ms']/a['fused_median_ms'],
                compiler_buffer_ratio=a['fused_memory']['bytes']/b['fused_memory']['bytes'],
                peak_rss_ratio=a['fused_peak_rss_bytes']/b['fused_peak_rss_bytes'],
                value_difference=a['value']-b['value'],gradient_relative_l2=difference/max(reference_norm,1e-30),
                gradient_interpretation='Numerical ODE-gradient comparison' if native=='native_orbit' else
                    'Independent noise bases/realizations and SDE schemes; gradient equality is not expected'))
    report['complete']=True
    report['all_cases_passed']=not report['failures']
    save()
    fields=('model','case','value','forward_ms','linearization_ms','reverse_ms','fused_ms',
        'reverse_over_forward','fused_over_forward','forward_buffers_MiB','fused_buffers_MiB','reverse_buffers_MiB',
        'fused_peak_MiB','warm_fused_peak_MiB','whole_worker_peak_MiB')
    with (options.output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in report['rows']:
            if row['complete']:
                writer.writerow(dict(model=row['model'],case=row['case'],value=row['value'],
                    **{name+'_ms':row['phase_medians_ms'][name] for name in ('forward','linearization','reverse')},
                    fused_ms=row['fused_median_ms'],reverse_over_forward=row['reverse_over_forward'],
                    fused_over_forward=row['fused_over_forward'],forward_buffers_MiB=row['forward_memory']['bytes']/2**20,
                    fused_buffers_MiB=row['fused_memory']['bytes']/2**20,
                    reverse_buffers_MiB=row['reverse_memory']['bytes']/2**20,
                    fused_peak_MiB=row['fused_peak_rss_bytes']/2**20,
                    warm_fused_peak_MiB=row['warm_fused_peak_rss_bytes']/2**20,
                    whole_worker_peak_MiB=row['whole_worker_peak_rss_bytes']/2**20))
    if report['failures']:
        raise RuntimeError('Some cases failed: '+repr(report['failures']))


def main():
    command=parser()
    options=command.parse_args()
    import math
    if not 1<=options.particles<=8 or options.repeats<3 or options.saves<2:
        command.error('Use 1..8 particles, at least three repeats and two saves')
    if min(options.fixed_gc_steps,options.fixed_lorentz_steps,options.max_steps,
           options.max_orbit_steps,options.accepted_capacity,options.orbit_capacity,options.checkpoint_steps)<1:
        command.error('Positive work/storage limits required')
    if max(options.fixed_gc_steps,options.fixed_lorentz_steps)+2>=options.max_steps:
        command.error('Leave room above the fixed-step count in max_steps')
    if any(not math.isfinite(x) or x<=0 for x in (options.gc_horizon,options.lorentz_horizon,
                                                  options.tolerance,options.collision_tolerance)):
        command.error('Positive finite horizons and tolerances required')
    if options.worker_case:
        if options.worker_model is None:
            command.error('Worker model required')
        try:
            worker(options)
        except Exception as error:
            destination=options.output/(options.worker_model+'_'+options.worker_case+'.json')
            previous=json.loads(destination.read_text()) if destination.exists() else {}
            previous.update(complete=False,model=options.worker_model,case=options.worker_case,
                error=repr(error),traceback=traceback.format_exc())
            destination.write_text(json.dumps(previous,indent=2,allow_nan=False)+'\n')
            raise
    else:
        parent(options)


if __name__=='__main__':
    main()
