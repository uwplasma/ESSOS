"""CPU reverse AD for native accepted-step RK and Diffrax on QA-coil orbits.

One forward linearization is shared by independent scalar-objective VJPs.
Report forward-linearization and per-objective reverse times separately; a
pullback timing excludes the adaptive forward solve. Both use production RHS
and unchanged field classes. Finite differences for native AD replay its frozen
accepted schedule; ordinary adaptive finite differences are reported separately.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import GuidingCenter, Lorentz, Particles
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart
from essos.orbit_error import OrbitErrorNorm, orbit_block_norm
from essos.particle_losses import (mean_final_radius, legacy_rcross_cost,
    classifier_loss_fraction, classifier_loss_location, classifier_energy_cost)
from essos.rk import solve_rk
from essos.rk_ad import replay_accepted
from .validate_coil_orbit_loss import reference_wall

ROOT=Path(__file__).resolve().parents[2]
NAMES=("mean_final_minor_radius_m", "main_legacy_time_mean_radius_norm_m",
    "classifier_loss_fraction_jax9_10", "loss_location_distance_m_jax9_10",
    "heat_proxy_jax10", "perpendicular_energy_J_jax9", "perpendicular_energy_jax10")


def compiler_memory(executable):
    value=executable.memory_analysis()
    return {name:int(getattr(value,name)) for name in
        ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")}


def json_safe(value):
    """Keep failed/nonfinite cases in valid JSON with explicit status flags."""
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [json_safe(v) for v in value]
    return None if isinstance(value,float) and not np.isfinite(value) else value


def build_case(options, model, method, backend, boundary, *, objective_index=None):
    base=Coils.from_json(str(ROOT/'examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json'))
    # Original QA launches include phi=0; the benchmark classifier uses the
    # periodic cubic option so angular derivatives also match at the wrap.
    points=jnp.column_stack([jnp.linspace(1.23,1.27,options.particles),jnp.zeros((options.particles,2))])
    pitches=jnp.linspace(.35,.65,options.particles)
    family=options.parameter_family
    coil_parameters=jnp.asarray(base.dofs)
    # Start the full orbit on the opposite gyro phase so the loss surrogate
    # also probes evolved states, rather than every maximum being at birth.
    energy_phase=jnp.array([0.]) if model=='gc' else jnp.array([0.,jnp.pi])
    birth_parameters=jnp.concatenate([points.ravel(),pitches,energy_phase])
    if family == 'coils':
        parameters=coil_parameters
    elif family == 'birth':
        parameters=birth_parameters
    else:
        parameters=jnp.concatenate([coil_parameters,birth_parameters])
    horizon=options.gc_horizon if model == "gc" else options.lorentz_horizon
    dt0=horizon*options.initial_step_fraction
    times=jnp.linspace(0.,horizon,options.saves)
    speed=float(np.sqrt(2*4000*ONE_EV/M))
    norm=OrbitErrorNorm(options.tolerance,options.tolerance,speed,velocity_components=1 if model == "gc" else 3)
    scales=jnp.array([1.]*3+[speed]*(1 if model == "gc" else 3))
    controller=diffrax.PIDController(rtol=0.,atol=2*options.tolerance*scales,
        norm=lambda e:orbit_block_norm(e,1 if model == "gc" else 3),
        pcoeff=.4,icoeff=.3,dcoeff=0.)
    solver=diffrax.Dopri5() if method == "dopri5" else diffrax.Dopri8()
    physical_model="GuidingCenter" if model == "gc" else "FullOrbit"

    def initialize(theta):
        coils=theta[:coil_parameters.size] if family != 'birth' else coil_parameters
        if family=='coils':
            births=birth_parameters
        else:
            births=theta if family=='birth' else theta[coil_parameters.size:]
        locations=births[:3*options.particles].reshape((-1,3))
        pitch=births[3*options.particles:4*options.particles]
        energy=4000*ONE_EV*jnp.exp(births[4*options.particles])
        phase=births[-1] if model!='gc' else 0.
        field=BiotSavart(base.with_dofs(coils))
        birth=Particles(locations,mass=M,charge=Q,energy=energy,
            initial_vparallel_over_v=pitch,field=field if model != "gc" else None,
            phase_angle_full_orbit=phase)
        if model == "gc":
            initial=jnp.column_stack([locations,birth.initial_vparallel])
            rhs=lambda t,y,a:GuidingCenter(t,y,(field,birth,Electric_field_zero()))
        else:
            initial=jnp.column_stack([birth.initial_xyz_fullorbit,birth.initial_vxvyvz])
            rhs=lambda t,y,a:Lorentz(t,y,(field,birth))
        return field,birth,initial,rhs

    def objectives(ys,field,birth):
        common=(ys,field,M,birth.energy,physical_model,boundary)
        functions=(lambda: mean_final_radius(ys,field.r_axis,field.z_axis),
            lambda: legacy_rcross_cost(ys,field.r_axis,field.z_axis),
            lambda: classifier_loss_fraction(ys,boundary),
            lambda: classifier_loss_location(ys,jnp.array([1.1,.1,0.]),boundary),
            lambda: classifier_energy_cost(*common,kind="heat_jax10"),
            lambda: classifier_energy_cost(*common,kind="perpendicular_jax9"),
            lambda: classifier_energy_cost(*common,kind="perpendicular_jax10"))
        # Select before constructing the graph. Gathering one element of an
        # already stacked objective vector can retain unused AD residuals.
        return jnp.stack([function() for function in functions]) if objective_index is None else functions[objective_index]()

    def cost(theta):
        field,birth,initial,rhs=initialize(theta)
        def one(y):
            if backend == "native":
                out=solve_rk(rhs,y,t1=horizon,dt0=dt0,ts=times,method=method,
                    rtol=options.tolerance,atol=options.tolerance,error_norm=norm,
                    max_steps=options.max_steps,adjoint="accepted",
                    adjoint_checkpoint_steps=getattr(options,'checkpoint_steps',16),
                    adjoint_max_accepted_steps=getattr(options,'accepted_capacity',None))
                return out.ys,jnp.stack(out.stats),out.status
            out=diffrax.diffeqsolve(diffrax.ODETerm(rhs),solver,t0=0.,t1=horizon,dt0=dt0,
                y0=y,saveat=diffrax.SaveAt(ts=times),stepsize_controller=controller,
                max_steps=options.max_steps,throw=False)
            return out.ys,jnp.stack([out.stats[k] for k in
                ("num_steps","num_accepted_steps","num_rejected_steps")]),jnp.where(
                    out.result==diffrax.RESULTS.successful,0,1)
        ys,counts,status=jax.vmap(one)(initial)
        values=objectives(ys,field,birth)
        return values,(ys,counts,status)

    def record(theta):
        _,_,initial,rhs=initialize(theta)
        return jax.vmap(lambda y:solve_rk(rhs,y,t1=horizon,dt0=dt0,ts=times,
            method=method,rtol=options.tolerance,atol=options.tolerance,error_norm=norm,
            max_steps=options.max_steps,_record_ad=True))(initial)

    def frozen(theta,tapes):
        field,birth,initial,rhs=initialize(theta)
        ys,_=jax.vmap(lambda y,tape:replay_accepted(rhs,y,None,tape,times,method=method))(initial,tapes)
        return objectives(ys,field,birth),ys
    labels=[]
    if family!='birth':
        labels.extend(base.dof_names)
    if family!='coils':
        labels.extend(f'birth_{i}_{axis}_m' for i in range(options.particles) for axis in ('x','y','z'))
        labels.extend(f'birth_{i}_pitch_vparallel_over_v' for i in range(options.particles))
        labels.append('shared_log_energy_over_4000eV')
        if model!='gc':
            labels.append('shared_gyrophase_rad')
    return parameters,cost,record,frozen,horizon,labels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--models',nargs='+',choices=['gc','lorentz'],default=['gc','lorentz'])
    parser.add_argument('--methods',nargs='+',choices=['dopri5','dopri8'],default=['dopri5','dopri8'])
    parser.add_argument('--backends',nargs='+',choices=['native','diffrax'],default=['native','diffrax'])
    parser.add_argument('--particles',type=int,default=2)
    parser.add_argument('--saves',type=int,default=32)
    parser.add_argument('--max-steps',type=int,default=512)
    parser.add_argument('--checkpoint-steps',type=int,default=16,
        help='Accepted steps per native state/FSAL checkpoint; 1 stores every step')
    parser.add_argument('--accepted-capacity',type=int,
        help='Native accepted-step tape capacity, independent of the trial-work limit')
    parser.add_argument('--repeats',type=int,default=3,
        help='Warm timing repetitions (>=3), or 0 for gradient validation without timing repetitions')
    parser.add_argument('--tolerance',type=float,default=1e-9)
    parser.add_argument('--gc-horizon',type=float,default=2e-6)
    parser.add_argument('--lorentz-horizon',type=float,default=2e-7)
    parser.add_argument('--initial-step-fraction',type=float,default=.5)
    parser.add_argument('--fd-epsilons',type=float,nargs='+',default=[1e-5,3e-6,1e-6])
    parser.add_argument('--replay-state-tolerance',type=float,default=2e-12,
        help='Maximum scaled state difference for independently recomputed accepted-grid replay')
    parser.add_argument('--parameter-family',choices=['coils','birth','joint'],default='joint',
        help='Birth: independent Cartesian positions/pitches, shared log-energy and Lorentz gyrophase')
    options=parser.parse_args()
    if options.output.exists() or jax.default_backend() != 'cpu':
        parser.error('fresh output path and CPU required')
    if not 1<=options.particles<=8 or options.saves<2 or (options.repeats!=0 and options.repeats<3):
        parser.error('use 1..8 particles, >=2 saves, and 0 or >=3 repeats')
    if (not 0<options.initial_step_fraction<=1 or options.gc_horizon<=0 or options.lorentz_horizon<=0
            or options.tolerance<=0 or options.max_steps<1 or options.checkpoint_steps<1 or any(e<=0 for e in options.fd_epsilons)
            or (options.accepted_capacity is not None and options.accepted_capacity<1)
            or not np.isfinite(options.replay_state_tolerance) or options.replay_state_tolerance<=0):
        parser.error('positive horizons/tolerance/FD epsilons/max_steps and 0 < initial-step-fraction <= 1 required')
    print('Building fixed QA reference classifier (outside timed region)',flush=True)
    boundary=reference_wall(angular_interpolation='periodic_cubic').classifier
    sources=['essos/rk.py','essos/rk_ad.py','essos/dynamics.py','essos/particle_losses.py',
        'essos/surfaces.py','essos/fields.py','essos/coils.py','essos/orbit_error.py',
        'essos/rk_dense.py','essos/rk_dense_high.py','essos/objective_functions.py',
        'essos/electric_field.py','examples/benchmarks/validate_coil_orbit_loss.py',
        'examples/benchmarks/benchmark_native_ad.py']
    report=dict(complete=False,metadata=dict(backend=jax.default_backend(),jax=jax.__version__,
        diffrax=diffrax.__version__,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        objective_names=NAMES,parameters=options.parameter_family,
        parameter_policy='Coils: all bundled QA Fourier/normalized-current DOFs. Birth: independent XYZ [m] and pitch per particle; shared ln(E/4000eV), and shared gyrophase [rad] for Lorentz. Joint: both groups. Fixed reference wall and error scales. No VMEC/Boozer DOFs.',
        birth_setup=dict(x_m=np.linspace(1.23,1.27,options.particles).tolist(),y_m=0.,z_m=0.,
            pitch=np.linspace(.35,.65,options.particles).tolist(),energy_eV=4000.,lorentz_gyrophase_rad=float(np.pi),
            classifier='Periodic cubic in phi, linear in R/Z; original QA births include the angular seam'),
        forward_policy='Native I; ESSOS Diffrax PI (.4,.3); identical physical embedded-error budgets',
        reverse_policy=(f'Native accepted steps with frozen times, segment length {options.checkpoint_steps}, '
            f'accepted capacity {options.accepted_capacity}; '
            'Diffrax default RecursiveCheckpointAdjoint'),
        timing_scope=('No warm timing repetitions; all seven reverse gradients and validation checks are computed'
            if options.repeats==0 else
            'Public jax.vjp of jitted cost for forward linearization (includes host dispatch); per-objective jitted VJP reuses residuals and takes a dynamic one-hot cotangent'),
        memory_scope='Compiled pullback arguments + outputs + temporaries - aliases, not process RSS',
        nonfinite_policy='Preserve failed cases; nonfinite numbers are null and finiteness/status flags identify them',
        objective_sources=dict(main_base='42c70a1e3f6e2822b2a9958e7bace06ec19a80e9',
            jax_9='4b689780876a134076a4bf89de15f5dee0f18abb',
            jax_10='9258ce01821ca44d293ceb0857f343b8745ad960',
            jax_8_distance_shape='bb74c07895f2cd6dc805b03c02bb992bbe66bff8'),
        objective_adaptation='Collisionless SI GC/Lorentz reconstruction; branch formulas and 1e-12 normalization constants preserved; heat is a proxy, not area-resolved flux',
        input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json',
             'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc')},
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}),rows=[])
    options.output.parent.mkdir(parents=True,exist_ok=True)
    for model in options.models:
        for method in options.methods:
            for backend in options.backends:
                print(f'Compiling {model} {method} {backend}',flush=True)
                theta,cost,record,frozen,horizon,labels=build_case(options,model,method,backend,boundary)
                # Linearize the jitted numerical function with the standard
                # public API. Do not return a Python pullback through jit;
                # that path miscounts lifted constants in JAX 0.11.2.
                primal_cost=jax.jit(cost)
                started=perf_counter()
                forward_exe=primal_cost.lower(theta).compile()
                values,pullback,aux=jax.block_until_ready(jax.vjp(primal_cost,theta,has_aux=True))
                forward_compile=perf_counter()-started
                ys,counts,status=map(np.asarray,aux)
                common=dict(model=model,method=method,backend=backend,horizon=horizon,
                    parameter_count=int(theta.size),parameter_labels=labels,values=np.asarray(values).tolist(),
                    counts_per_particle=counts.tolist(),forward_status=status.tolist(),
                    forward_compile_seconds=forward_compile,forward_memory=compiler_memory(forward_exe))
                if np.any(status) or not np.isfinite(ys).all():
                    report['rows'].append(dict(common,outcome='forward_failed'))
                    options.output.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
                    print(json.dumps(dict(model=model,method=method,backend=backend,outcome='forward_failed',
                        counts_per_particle=counts.tolist(),status=status.tolist())),flush=True)
                    continue
                distances=np.asarray(jax.jit(jax.vmap(jax.vmap(boundary.evaluate_xyz)))(ys[...,:3]))
                peaks=np.argmin(distances,axis=1)
                reverse=jax.jit(lambda pullback,seed:pullback(seed)[0])
                seed=jnp.eye(len(NAMES),dtype=theta.dtype)
                started=perf_counter()
                reverse_exe=reverse.lower(pullback,seed[0]).compile()
                reverse_compile=perf_counter()-started
                # Use JAX's normal dispatcher for pullback PyTrees; it handles
                # lifted constants in the installed JAX version. The lowered
                # executable is retained only for its compiler memory report.
                gradients=np.stack([np.asarray(jax.block_until_ready(reverse(pullback,s))) for s in seed])
                gradient_finite=np.isfinite(gradients).all(axis=1)
                direction=jnp.asarray(np.random.default_rng(317).normal(size=theta.shape))
                direction=direction/jnp.linalg.norm(direction)
                epsilon=options.fd_epsilons[0]
                primal=jax.jit(lambda p:cost(p)[0])
                adaptive_fd=np.asarray((primal(theta+epsilon*direction)-primal(theta-epsilon*direction))/(2*epsilon))
                checks=dict(adaptive_directional_fd=adaptive_fd.tolist(),
                    ad_directional=np.asarray(gradients@np.asarray(direction)).tolist())
                if backend == 'native':
                    original,tapes=jax.jit(record)(theta)
                    values_replay,ys_replay=jax.jit(frozen)(theta,tapes)
                    state_scales=np.array([1.]*3+[np.sqrt(2*4000*ONE_EV/M)]*(1 if model=='gc' else 3))
                    replay_difference=float(np.max(abs(np.asarray(ys_replay)-ys)/state_scales))
                    if replay_difference>options.replay_state_tolerance:
                        raise RuntimeError(f'accepted replay disagrees with adaptive primal: {replay_difference}')
                    np.testing.assert_allclose(values_replay,values,atol=1e-25,rtol=2e-10)
                    replay=jax.jit(lambda p:frozen(p,tapes)[0])
                    exact=np.asarray(gradients@np.asarray(direction))
                    fd_samples=[]
                    for epsilon in options.fd_epsilons:
                        fd=np.asarray((replay(theta+epsilon*direction)-replay(theta-epsilon*direction))/(2*epsilon))
                        relative=np.abs(exact-fd)/np.maximum(np.maximum(abs(exact),abs(fd)),1e-25)
                        fd_samples.append(dict(epsilon=epsilon,fd=fd.tolist(),relative_error=relative.tolist()))
                    best=min(fd_samples,key=lambda sample:float(np.max(sample['relative_error']))
                        if np.isfinite(sample['relative_error']).all() else float('inf'))
                    fd=np.asarray(best['fd'])
                    relative=np.asarray(best['relative_error'])
                    checks.update(frozen_directional_fd=fd.tolist(),ad_directional=exact.tolist(),
                        frozen_fd_relative_error=relative.tolist(),frozen_fd_samples=fd_samples,
                        frozen_fd_selected_epsilon=best['epsilon'],
                        frozen_fd_passed=bool(np.isfinite(relative).all() and np.max(relative)<=2e-4),
                        max_replay_state_difference=float(np.max(abs(np.asarray(ys_replay)-ys))))
                    checks['max_scaled_replay_state_difference']=replay_difference
                    checks['scaled_replay_state_tolerance']=options.replay_state_tolerance
                timings=[]
                for repeat in range(options.repeats):
                    started=perf_counter()
                    jax.block_until_ready(jax.vjp(primal_cost,theta,has_aux=True))
                    forward_ms=1000*(perf_counter()-started)
                    reverse_ms=[None]*len(NAMES)
                    for index in np.random.default_rng(730+repeat).permutation(len(NAMES)):
                        started=perf_counter()
                        # Every repetition measures the same parameter point.
                        # Keep its original pullback: fresh Python pullback
                        # identities otherwise trigger recompilation, which
                        # would contaminate a purported warm reverse timing.
                        jax.block_until_ready(reverse(pullback,seed[index]))
                        reverse_ms[index]=1000*(perf_counter()-started)
                    timings.append(dict(forward_linearization_ms=forward_ms,reverse_ms=reverse_ms))
                row=dict(common,outcome='ok' if gradient_finite.all() and checks.get('frozen_fd_passed',True)
                    else 'nonfinite_gradient' if not gradient_finite.all() else 'gradient_check_failed',
                    gradient_finite_per_objective=gradient_finite.tolist(),
                    classifier_orbit_sensitive=bool(np.any(peaks>0)),
                    classifier_maximum_save_indices=peaks.tolist(),
                    gradients=gradients.tolist(),reverse_compile_seconds=reverse_compile,
                    reverse_memory=compiler_memory(reverse_exe),
                    validation=checks,warm_samples=timings)
                if timings:
                    row.update(forward_median_ms=float(np.median([t['forward_linearization_ms'] for t in timings])),
                        reverse_median_ms=np.median([t['reverse_ms'] for t in timings],axis=0).tolist())
                report['rows'].append(row)
                options.output.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
                print(json.dumps({k:row[k] for k in ('model','method','backend','outcome','counts_per_particle','forward_median_ms','reverse_median_ms') if k in row}),flush=True)
    report['complete']=True
    report['all_cases_passed']=all(row['outcome']=='ok' for row in report['rows'])
    options.output.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    table=[]
    for row in report['rows']:
        memory=row.get('reverse_memory')
        buffers=(memory['argument_size_in_bytes']+memory['output_size_in_bytes']+memory['temp_size_in_bytes']
            -memory['alias_size_in_bytes'])/2**20 if memory is not None else None
        for i,name in enumerate(NAMES):
            table.append(dict(model=row['model'],method=row['method'],backend=row['backend'],objective=name,
                outcome=row['outcome'],value=row['values'][i],
                gradient_finite=row.get('gradient_finite_per_objective',[False]*len(NAMES))[i],
                gradient_l2=float(np.linalg.norm(row['gradients'][i])) if 'gradients' in row else None,
                accepted_mean=float(np.mean(np.asarray(row['counts_per_particle'])[:,1])),
                rejected_mean=float(np.mean(np.asarray(row['counts_per_particle'])[:,2])),
                forward_linearization_ms=row.get('forward_median_ms'),
                reverse_ms=row.get('reverse_median_ms',[None]*len(NAMES))[i],
                separate_phase_sum_ms=row['forward_median_ms']+row['reverse_median_ms'][i] if 'reverse_median_ms' in row else None,
                reverse_compiler_buffer_MiB=buffers,
                frozen_fd_relative_error=row.get('validation',{}).get('frozen_fd_relative_error',[None]*len(NAMES))[i]))
    with options.output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(json_safe(table))
    print('Complete',flush=True)


if __name__ == '__main__':
    main()
