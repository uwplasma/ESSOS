"""Joint velocity and spatial statistics with separate orbit/collision clocks.

Uses production trace_split, Lorentz motion, SI Coulomb coefficients and
retained Brownian memory, for Lorentz and GC motion. Velocity moments and
distributions come from the same trajectories as spatial moments. Independent
Fokker-Planck references describe the transient beam, including GC's reflecting
speed floor. Lorentz spatial theory retains finite gyrofrequency; GC uses its
own spatial diffusion operator. JAX selects the machine backend. No source.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from essos.split import trace_split
from essos.background_species import BackgroundSpecies
from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from essos.dynamics import Particles
from scipy.stats import kstest, norm
from .validate_beam_relaxation import estimate, statistics
from .nonrelativistic_collision_reference import ProtonBackground
from .thermal_relaxation_reference import RelaxationReference, PhysicalRelaxationReference, impact_coefficients
from .lorentz_spatial_reference import guiding_center_msd

PARAMETERS = dict(particles=512, batch_size=64, seed=7706, horizon=.1, saves=5,
    omegas=[128., 512., 2048.], caps=[.0025, .00125], tolerance=.001,
    orbit_tolerance=1e-10, phase_step=.1, coulomb_log_model='impact', models=['lorentz','gc'])


class UniformMagneticField:
    def __init__(self, magnitude): self.magnitude = magnitude
    def B_contravariant(self, point): return jnp.array([0., 0., self.magnitude])
    B_covariant = B_contravariant
    def AbsB(self, point): return self.magnitude
    def dAbsB_by_dX(self, point): return jnp.zeros(3)
    curl_b = kappa = dAbsB_by_dX
    def sqrtg(self, point): return 1.


def solver(adaptive, times, options, model='lorentz'):
    physical = ProtonBackground(coulomb_log_model=options.coulomb_log_model)
    sigma, frequency = physical.sigma, physical.nu
    background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([physical.density]), jnp.array([physical.temperature_ev]),
        coulomb_log_model=options.coulomb_log_model)
    gc = model == 'gc'
    def batch(initial, keys, cap, omega):
        field = UniformMagneticField(omega*frequency*PROTON_MASS/ELEMENTARY_CHARGE)
        scale = (jnp.array([sigma/frequency]*3+[sigma,PROTON_MASS*sigma**2/field.magnitude])
                 if gc else jnp.array([sigma/frequency]*3+[sigma]*3))
        birth = Particles(initial[:, :3]*scale[:3], mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
            energy=5*physical.temperature_ev*ONE_EV, initial_vparallel_over_v=jnp.full(len(initial), .6),
            initial_xyz_fullorbit=initial[:, :3]*scale[:3], initial_vxvyvz=initial[:, 3:]*sigma)
        birth.random_keys = keys
        out = trace_split(field, birth, background, model='GuidingCenter' if gc else 'FullOrbit',
            tmax=options.horizon/frequency, timestep=cap/frequency,
            collision_dtmax=cap/frequency, orbit_dt0=jnp.minimum(cap, options.phase_step/omega)/frequency,
            times=jnp.asarray(times)/frequency, adaptive=adaptive, orbit_adaptive=adaptive,
            collision_controller=getattr(options,'collision_controller','conservative'), collision_tolerance=options.tolerance,
            moment_tolerance=getattr(options,'moment_tolerance',.005),
            spatial_diffusion=gc, speed_floor=.05*np.sqrt(2.)*sigma if gc else None,
            orbit_stage_execution='unrolled', rtol=options.orbit_tolerance, atol=options.orbit_tolerance,
            max_steps=10000, max_orbit_steps=2048)
        return out._replace(ys=out.ys/scale, y=out.y/scale, t=out.t*frequency, ts=jnp.asarray(times))
    return jax.jit(batch)


def observables(states, omega, model):
    """Read physical observables from the production solver's normalized output."""
    if model == 'gc':
        speed = np.sqrt(states[...,3]**2+2*states[...,4])
        pitch = states[...,3]/speed
        displacement = omega*(states[...,:2]-states[:,:1,:2])
    else:
        speed = np.linalg.norm(states[...,3:],axis=-1)
        pitch = states[...,5]/speed
        cross = np.stack((states[...,4],-states[...,3]),axis=-1)
        guiding = omega*states[...,:2]+cross
        displacement = guiding-guiding[:,:1]
    return speed,pitch,displacement,np.sum(displacement**2,axis=-1)


def velocity_reference(model,times,coefficient_model,*,need_pitch=True):
    lower = .05*np.sqrt(2.) if model=='gc' else 0.
    cls = RelaxationReference if coefficient_model=='impact' else PhysicalRelaxationReference
    kwargs = dict(lower=lower,**(dict(law=impact_coefficients) if coefficient_model=='impact' else {}))
    coarse,fine,upper = cls(nodes=384,**kwargs),cls(nodes=768,**kwargs),cls(nodes=960,upper=10.,**kwargs)
    a,b,c = [ref.moments(times) for ref in (coarse,fine,upper)]
    changes = {name:dict(mesh=float(np.max(abs(a[name]-b[name]))/max(1.,np.max(abs(b[name])))),
                        upper=float(np.max(abs(c[name]-b[name]))/max(1.,np.max(abs(b[name]))))) for name in b}
    if max(value[key] for value in changes.values() for key in ('mesh','upper'))>3e-4:
        raise RuntimeError(f'velocity reference refinement failed: {changes}')
    pitch_cdf,harmonic_change=None,None
    if need_pitch:
        pitch_grid = np.linspace(-1.,1.,801)
        pitch_low = fine.pitch_cdf(times[-1],harmonics=32)
        pitch_cdf = fine.pitch_cdf(times[-1],harmonics=64)
        harmonic_change = float(np.max(abs(pitch_low(pitch_grid)-pitch_cdf(pitch_grid))))
        if harmonic_change>1e-7:
            pitch_low,pitch_cdf = pitch_cdf,fine.pitch_cdf(times[-1],harmonics=96)
            harmonic_change = float(np.max(abs(pitch_low(pitch_grid)-pitch_cdf(pitch_grid))))
        if harmonic_change>1e-7: raise RuntimeError(f'pitch reference needs more harmonics: {harmonic_change}')
    speed_cdf = fine.speed_cdf(times[-1])
    # Only l=0,1,2 spectra are needed for spatial/moment references afterward.
    fine._spectra = {ell:value for ell,value in fine._spectra.items() if ell<=2}
    speed_edges,pitch_edges = np.linspace(0.,6.,61),np.linspace(-1.,1.,61)
    record = dict(speed_floor=lower,moments={name:value.tolist() for name,value in b.items()},
        refinement=changes,pitch_harmonic_change=harmonic_change,
        distribution=dict(speed_edges=speed_edges.tolist(),
            speed_density=(np.diff(speed_cdf(speed_edges))/np.diff(speed_edges)).tolist()))
    if need_pitch:record['distribution'].update(pitch_edges=pitch_edges.tolist(),
        pitch_density=(np.diff(pitch_cdf(pitch_edges))/np.diff(pitch_edges)).tolist())
    return coarse,fine,record,speed_cdf,pitch_cdf


def plot_spatial(report, folder, model):
    import matplotlib.pyplot as plt
    meta = report['metadata']; omegas = meta['omegas']; times = np.asarray(meta['times'])
    fig, axes = plt.subplots(2, len(omegas), figsize=(5*len(omegas), 7), constrained_layout=True, squeeze=False)
    for column, omega in enumerate(omegas):
        ax, work = axes[:, column]
        refs = report['gc_references'] if model=='gc' else report['references']
        ax.plot(times, refs[str(omega)]['msd'], 'k', lw=2,
                label='GC diffusion theory' if model=='gc' else 'Finite-Omega theory')
        selected = [r for r in report['rows'] if r['omega'] == omega and r.get('model','lorentz')==model]
        for index, row in enumerate(selected):
            color = 'C0' if row['method'] == 'fixed' else 'C1'
            fine = row['cap'] == min(meta['caps'])
            values = np.array([e['value'] for e in row['evolution']])
            half = np.array([e['half95'] for e in row['evolution']])
            label = f"{row['method']}, collision cap {row['cap']:g}"
            ax.plot(times, values, color=color, ls='-' if fine else ':', label=label)
            if fine: ax.fill_between(times, values-half, values+half, color=color, alpha=.12)
            work.bar(index, row['accepted_orbit_per_collision'], color=color, alpha=.8)
        ax.set(title=rf'$\Omega/\nu_0={omega:g}$', xlabel=r'$t\nu_0$', ylabel=r'$E[|\Delta G_\perp|^2]/\rho_\sigma^2$')
        work.axhline(1., color='k', ls='--', lw=1)
        work.set(xticks=range(len(selected)), xticklabels=[f"{r['method']}\n{r['cap']:g}" for r in selected],
                 ylabel='Accepted orbit steps / collision steps')
        ax.grid(alpha=.2); work.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Production split {model.upper()}/Coulomb statistics; N={meta['particles']}; {meta['backend']}\n"
                 "Initially 5T beam, pitch 0.6; no source; shading = marginal 95% sampling intervals")
    stem = 'multirate_gc' if model=='gc' else 'multirate'
    fig.savefig(folder/(stem+'.png'), dpi=160)
    fig.savefig(folder/(stem+'.svg'))
    plt.close(fig)


def plot_velocity(report,folder,model):
    import matplotlib.pyplot as plt
    meta=report['metadata'];times=np.asarray(meta['times']);reference=report['velocity_references'][model]
    selected=[r for r in report['rows'] if r['model']==model and r['cap']==min(meta['caps'])]
    names=('speed_mean','speed_variance','pitch_mean','pitch_variance','parallel_mean','parallel_variance')
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    hist,hist_axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name,ax in zip(names,axes.flat):
        ax.plot(times,reference['moments'][name],'k',lw=2,label='Independent transient theory')
        for row in selected:
            values=np.array([sample[name]['value'] for sample in row['velocity']])
            half=np.array([sample[name]['half95'] for sample in row['velocity']])
            color='C'+str(meta['omegas'].index(row['omega']))
            label=rf"{row['method']}, $\Omega/\nu_0={row['omega']:g}$"
            ax.plot(times,values,color=color,ls='--' if row['method']=='fixed' else '-',label=label)
            ax.fill_between(times,values-half,values+half,color=color,alpha=.07)
        ax.set(title=name.replace('_',' '),xlabel=r'$t\nu_0$',ylabel='Normalized velocity / pitch units');ax.grid(alpha=.2)
    for name,ax in zip(('speed','pitch'),hist_axes):
        edges=np.asarray(reference['distribution'][name+'_edges'])
        ax.stairs(reference['distribution'][name+'_density'],edges,color='k',lw=2,label='Independent transient theory')
        for row in selected:
            color='C'+str(meta['omegas'].index(row['omega']))
            ax.stairs(np.asarray(row['histograms'][name])/(meta['particles']*np.diff(edges)),edges,
                color=color,ls='--' if row['method']=='fixed' else '-',label=f"{row['method']}, Omega={row['omega']:g}")
        ax.set(title='Final '+name+' distribution',xlabel='v/sigma' if name=='speed' else 'pitch',ylabel='Probability density');ax.grid(alpha=.2)
    caption=f"{model.upper()}: same multirate trajectories; N={meta['particles']}; finest cap {min(meta['caps']):g}"
    fig.suptitle(caption+'\nTransient beam reference; shading = marginal 95% sampling intervals')
    hist.suptitle(caption+f"; final t*nu0={times[-1]:g}")
    axes[0,0].legend(fontsize=7);hist_axes[0].legend(fontsize=7)
    for suffix,figure in (('velocity',fig),('distributions',hist)):
        for extension in ('png','svg'):figure.savefig(folder/f'multirate_{model}_{suffix}.{extension}',dpi=160)
        plt.close(figure)


def plot(report,folder):
    for model in report['metadata'].get('models',['lorentz']):
        plot_spatial(report,folder,model)
        if 'velocity_references' in report:plot_velocity(report,folder,model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('particles', 'batch_size', 'seed', 'saves'):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=PARAMETERS[name])
    for name in ('horizon', 'tolerance', 'orbit_tolerance', 'phase_step'):
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=PARAMETERS[name])
    parser.add_argument('--omegas', type=float, nargs='+', default=PARAMETERS['omegas'])
    parser.add_argument('--caps', type=float, nargs='+', default=PARAMETERS['caps'])
    parser.add_argument('--coulomb-log-model', choices=('nrl', 'impact'), default=PARAMETERS['coulomb_log_model'])
    parser.add_argument('--models',choices=('lorentz','gc'),nargs='+',default=PARAMETERS['models'])
    parser.add_argument('--collision-controller',choices=('conservative','moments'),default='conservative')
    parser.add_argument('--moment-tolerance',type=float,default=.005)
    parser.add_argument('--output', type=Path, default=Path('output/split_multirate'))
    parser.add_argument('--plot-only', action='store_true')
    options = parser.parse_args(); folder = options.output; folder.mkdir(parents=True, exist_ok=True)
    if options.plot_only:
        plot(json.loads((folder/'comparison.json').read_text()), folder); return
    if (folder/'comparison.json').exists():parser.error('choose a fresh output directory')
    if options.particles < 2 or options.batch_size < 1 or options.particles % options.batch_size or options.saves<2:
        parser.error('particles must be >=2 and divisible by batch size')
    if not all(np.isfinite(x) and x>0 for x in options.omegas+options.caps+[options.horizon, options.tolerance, options.phase_step,options.orbit_tolerance]):
        parser.error('frequencies, intervals and tolerances must be positive')
    times = np.linspace(0., options.horizon, options.saves)
    meta = vars(options).copy(); meta['output'] = str(folder)
    meta.update(backend=jax.default_backend(), device=jax.devices()[0].device_kind, jax=jax.__version__,
        times=times.tolist(), orbit_control='fixed/fixed or adaptive/adaptive; independent intervals',
        framework='production trace_split with actual SI fields, births, Coulomb coefficients and Brownian stack',
        uncertainty='Independent particles; marginal 95% intervals; shared keys across settings do not imply shared paths')
    meta.update(velocity_scope='Transient speed/pitch/parallel moments and final CDFs from the same saved paths; not an equilibrium test',
        gc_orbit_scope='Uniform-field GC has no gyromotion; adaptive orbit control may need one step per collision. Fixed controls subcycle explicitly.')
    report = dict(complete=False, metadata=meta, references={}, gc_references={},velocity_references={},rows=[])
    root=Path(__file__).resolve().parents[2]
    report['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        'essos/split.py','essos/background_species.py','essos/rk.py',
        'examples/benchmarks/benchmark_split_multirate.py','examples/benchmarks/thermal_relaxation_reference.py',
        'examples/benchmarks/lorentz_spatial_reference.py','examples/benchmarks/nonrelativistic_collision_reference.py')}
    save = lambda: (folder/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    references={}
    for model in options.models:
        start=perf_counter()
        coarse,fine,record,speed_cdf,pitch_cdf=velocity_reference(model,times,options.coulomb_log_model)
        report['velocity_references'][model]=record;references[model]=(speed_cdf,pitch_cdf)
        for omega in options.omegas:
            a,b=(coarse.moments(times)['spatial_integral'],np.asarray(record['moments']['spatial_integral'])) if model=='gc' else (
                guiding_center_msd(coarse,times,omega),guiding_center_msd(fine,times,omega))
            change=float(np.max(abs(a-b))/max(b[-1],1e-30))
            if change>1e-3:raise RuntimeError(f'spatial reference refinement failed: {change}')
            target=report['gc_references'] if model=='gc' else report['references']
            target[str(omega)]=dict(msd=b.tolist(),radial_refinement=change)
        print(f'{model.upper()} references: {perf_counter()-start:.2f}s; pitch harmonic change {record["pitch_harmonic_change"]:.3g}',flush=True)
        save()
    keys = jax.random.split(jax.random.PRNGKey(options.seed), options.particles)
    initial = np.array([0., 0., 0., np.sqrt(10.)*.8, 0., np.sqrt(10.)*.6])
    births = jnp.broadcast_to(jnp.asarray(initial), (options.batch_size, 6))
    cdf_alpha=.01/(2*len(options.models)*2*len(options.omegas)*len(options.caps))
    cdf_limit=float(np.sqrt(np.log(2/cdf_alpha)/(2*options.particles)))
    critical=float(norm.ppf(1-.01/(2*len(options.models)*2*len(options.omegas)*len(options.caps)*(len(times)-1)*7)))
    gc_controls={}
    for model,adaptive in ((m,a) for m in options.models for a in (False,True)):
        function = solver(adaptive, times, options,model)
        for omega in options.omegas:
            for cap in options.caps:
                start = perf_counter(); outputs = []
                for offset in range(0, options.particles, options.batch_size):
                    outputs.append(jax.device_get(jax.block_until_ready(function(births,
                        keys[offset:offset+options.batch_size], jnp.asarray(cap), jnp.asarray(omega)))))
                out = jax.tree.map(lambda *a: np.concatenate(a, axis=0), *outputs)
                if np.any(out.status != 0): raise RuntimeError(f'solver failures: {np.unique(out.status, return_counts=True)}')
                speed,pitch,delta,msd = observables(out.ys,omega,model)
                expected = (report['gc_references'] if model=='gc' else report['references'])[str(omega)]['msd']
                velocity_expected=report['velocity_references'][model]['moments']
                inner = out.stats.orbit_steps-out.stats.orbit_rejected
                row = dict(model=model,method='adaptive' if adaptive else 'fixed', omega=omega, cap=cap,
                    elapsed_s=perf_counter()-start, collision_accepted=int(out.stats.accepted.sum()),
                    collision_rejected=int(out.stats.rejected.sum()), orbit_accepted=int(inner.sum()),
                    orbit_rejected=int(out.stats.orbit_rejected.sum()),
                    accepted_orbit_per_collision=float(inner.sum()/out.stats.accepted.sum()),
                    max_accepted_collision_error=float(out.stats.max_accepted_error.max()),
                    evolution=[estimate(msd[:, i], target) for i, target in enumerate(expected)],
                    velocity=[statistics(speed[:,i],pitch[:,i],{name:values[i] for name,values in velocity_expected.items()}) for i in range(len(times))],
                    cdf={},histograms={})
                if adaptive and row['max_accepted_collision_error']>1:raise RuntimeError('accepted collision estimate exceeds one')
                for name,values,cdf in (('speed',speed[:,-1],references[model][0]),('pitch',pitch[:,-1],references[model][1])):
                    result=kstest(values,cdf)
                    row['cdf'][name]=dict(distance=float(result.statistic),pvalue=float(result.pvalue),familywise_limit=cdf_limit,
                        passed=bool(result.statistic<=cdf_limit))
                    edges=report['velocity_references'][model]['distribution'][name+'_edges']
                    row['histograms'][name]=np.histogram(values,edges)[0].tolist()
                worst=max(abs(value['z']) for sample in row['velocity'][1:] for value in sample.values() if value['z'] is not None)
                row['velocity_screen']=dict(max_abs_z=worst,critical=critical,passed=bool(worst<=critical))
                if model=='gc':
                    row['spatial_components']=[dict(mean=[estimate(delta[:,i,j],0.) for j in range(2)],
                        variance=[estimate(delta[:,i,j],expected[i]/2,variance=True) for j in range(2)]) for i in range(len(times))]
                    control_key=(adaptive,cap)
                    if control_key in gc_controls:
                        a,b,c=gc_controls[control_key]
                        row['frequency_control']=dict(max_speed_difference=float(np.max(abs(speed-a))),
                            max_pitch_difference=float(np.max(abs(pitch-b))),max_scaled_msd_difference=float(np.max(abs(msd-c))))
                    else:gc_controls[control_key]=(speed,pitch,msd)
                report['rows'].append(row); save()
                print(f"{model}/{row['method']} Omega={omega:g} cap={cap:g}: orbit/collision={row['accepted_orbit_per_collision']:.2f}, "
                      f"MSD z={row['evolution'][-1]['z']:.2f}, velocity max|z|={worst:.2f}, CDF pass={all(v['passed'] for v in row['cdf'].values())}, {row['elapsed_s']:.2f}s", flush=True)
    report['complete'] = True; save(); plot(report, folder)
    fields = ['model','method','omega','cap','collision_accepted','collision_rejected','orbit_accepted',
              'orbit_rejected','accepted_orbit_per_collision','max_accepted_collision_error','elapsed_s']
    with (folder/'comparison.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(report['rows'])


if __name__ == '__main__':
    main()
