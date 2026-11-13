"""Global refinement on the SAME native Brownian increments as each coarse solve.

Record accepted production endpoints/increments without changing numerical
math. Refine each recorded interval with the production Brownian-memory
sampler, using an independent diagnostic namespace for conditional bridges.
Reference paths continue globally; they do NOT restart at coarse states.
"""
import inspect
import jax
import jax.numpy as jnp
import essos.split as split
from essos.rk import solve_rk,RKStatus

MAX_ROOTS=4096
def make_paired_solver(base,*,max_roots=MAX_ROOTS,refinements=(8,16,32)):
    if len(refinements)!=3 or tuple(sorted(set(refinements)))!=tuple(refinements) or any(n<2 or n>64 or n&(n-1) for n in refinements):
        raise ValueError('three increasing power-of-two refinements in 2..64 required')
    finest=refinements[-1]
    source=getattr(base,'_prototype_source',None) or inspect.getsource(base)
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    replace("    if record_steps:\n        s['history']", "    s.update(pair_t=jnp.zeros(MAX_ROOTS,y0.dtype),pair_dw=jnp.zeros((MAX_ROOTS,noise_dim),y0.dtype))\n    if record_steps:\n        s['history']")
    replace("            state = save(state)","""            if combined:
                state['pair_t']=lax.cond(accepted,lambda a:a.at[2*s['accepted']].set(middle).at[2*s['accepted']+1].set(end),lambda a:a,s['pair_t'])
                state['pair_dw']=lax.cond(accepted,lambda a:a.at[2*s['accepted']].set(left).at[2*s['accepted']+1].set(dw-left),lambda a:a,s['pair_dw'])
            else:
                state['pair_t']=lax.cond(accepted,lambda a:a.at[s['accepted']].set(end),lambda a:a,s['pair_t'])
                state['pair_dw']=lax.cond(accepted,lambda a:a.at[s['accepted']].set(dw),lambda a:a,s['pair_dw'])
            state = save(state)""")
    replace('    return solution\n',"    return solution._replace(step_ts=(s['pair_t'],s['pair_dw']))\n")
    namespace=dict(base.__globals__);namespace['MAX_ROOTS']=max_roots
    exec(compile(source,'<paired coarse noise recorder>','exec'),namespace)
    recorded=namespace['solve_split']

    def solve(rhs,collision,y0,args=None,**kwargs):
        coarse=recorded(rhs,collision,y0,args,**kwargs)
        roots,dws=coarse.step_ts
        count=coarse.stats.collision_updates
        key=kwargs['key'];ts=kwargs['ts'];t0=jnp.asarray(kwargs.get('t0',0.),y0.dtype)
        initial_ok=(coarse.status==0)&(count<=max_roots)
        tag=jax.random.fold_in(key,0x50414952)

        def noise(i,start,end,dw):
            memory=split.brownian_memory(jax.random.fold_in(tag,i),dw.size,capacity=64,dtype=y0.dtype,t0=start)
            memory=memory._replace(ends=memory.ends.at[0].set(end),increments=memory.increments.at[0].set(dw),size=jnp.array(1),peak=jnp.array(1))
            valid=jnp.array(True)
            for n in (2**i for i in range(1,finest.bit_length())):
                grid=(start+(end-start)*jnp.arange(1,n+1,dtype=y0.dtype)/n).at[-1].set(end)
                def draw(m,t):
                    m,w,full=split.brownian_sample(m,t)
                    return m,(w,full)
                memory,(cum,full)=jax.lax.scan(draw,memory,grid)
                valid &= ~jnp.any(full)
            return jnp.diff(jnp.concatenate((jnp.zeros((1,dw.size),y0.dtype),cum)),axis=0),valid

        def flow(y,start,end):
            out=solve_rk(rhs,y,args,t0=start,t1=end,dt0=end-start,ts=jnp.empty(0,y0.dtype),
                method=kwargs['method'],adaptive=True,error_norm=kwargs['error_norm'],
                rtol=kwargs['rtol'],atol=kwargs['atol'],stage_execution=kwargs['orbit_stage_execution'],
                max_steps=64)
            return out.y,out.status==RKStatus.SUCCESS

        answers=[];valids=[]
        for n in refinements:
            state=(jnp.array(0),t0,y0,jnp.broadcast_to(y0,(len(ts),len(y0))),jnp.array(1),initial_ok)
            def advance(state):
                i,start,y,saves,index,ok=state;end=roots[i]
                fine,valid=noise(i,start,end,dws[i])
                increments=fine.reshape(n,finest//n,-1).sum(axis=1)
                grid=(start+(end-start)*jnp.arange(1,n+1,dtype=y0.dtype)/n).at[-1].set(end)
                def substep(carry,data):
                    t,y,ok=carry;target,w=data;h=target-t
                    y,a=flow(y,t,target)
                    kick=collision(target,y,h,w,args)
                    y=kick.y
                    good=a&jnp.isfinite(y).all()&(kick.drift_error<=1)&(kick.diffusion_error<=1)
                    return (target,y,ok&good),None
                (_,y,good),_=jax.lax.scan(substep,(start,y,ok&valid),(grid,increments))
                def save(carry):
                    saves,index=carry
                    return saves.at[index].set(y),index+1
                saves,index=jax.lax.while_loop(lambda c:(c[1]<len(ts))&(ts[jnp.minimum(c[1],len(ts)-1)]<=end),save,(saves,index))
                return i+1,end,y,saves,index,good
            final=jax.lax.while_loop(lambda s:(s[0]<count)&s[-1],advance,state)
            answers.append(final[3]);valids.append(final[-1]&(final[4]==len(ts)))
        return coarse._replace(step_ts=(jnp.stack(answers),jnp.stack(valids),roots,dws))
    return solve


def main():
    import argparse,json
    from pathlib import Path
    from time import perf_counter
    import numpy as np
    from essos.constants import PROTON_MASS as M,ELEMENTARY_CHARGE as Q,ONE_EV
    from essos.dynamics import Particles
    from .benchmark_split_helix import HelicalField
    from .benchmark_split_mirror import MagneticMirror,TransitField,background
    from .nonrelativistic_collision_reference import ProtonBackground
    from essos.split import make_step_limit
    from .validate_beam_relaxation import estimate
    jax.config.update('jax_enable_compilation_cache',False)
    parser=argparse.ArgumentParser();parser.add_argument('--fields',nargs='+',choices=('mirror','helix','profiles'),default=['mirror','helix']);parser.add_argument('--particles',type=int,default=64)
    parser.add_argument('--tolerances',nargs='+',type=float,default=[1e-4,5e-5,1e-5])
    parser.add_argument('--control',choices=('combined','moments'),default='moments',help='Control under test; orbit controller remains I')
    parser.add_argument('--max-steps',type=int,default=4000)
    parser.add_argument('--max-roots',type=int,default=MAX_ROOTS)
    parser.add_argument('--refinements',nargs=3,type=int,default=[8,16,32])
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--reference',type=Path,help='Optional coupled reference generated by benchmark_split_coordinate_physics')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 2<=args.particles<=64 or any(not np.isfinite(t) or t<=0 for t in args.tolerances):parser.error('2..64 paths and positive finite tolerances required')
    if args.max_steps<1 or args.max_roots<1:parser.error('positive step and root capacities required')
    if args.refinements!=sorted(set(args.refinements)) or any(n<2 or n>64 or n&(n-1) for n in args.refinements):parser.error('three increasing power-of-two refinements in 2..64 required')
    original=split.solve_split;folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    from types import SimpleNamespace
    from .benchmark_combined_validation import references,add_speed_reference,physical_moments
    if args.reference is not None:
        data=json.loads(args.reference.read_text())
    else:
        settings=SimpleNamespace(fields=[f for f in args.fields if f!='profiles'],output=folder)
        data=references(settings);add_speed_reference(settings,data)
    report=dict(particles=args.particles,control=args.control,refinements=args.refinements,max_steps=args.max_steps,max_roots=args.max_roots,scope='Global coarse/native-refined trajectory errors and paired weak moment differences on the SAME Brownian path within each case. Combined recording preserves BOTH accepted half-step increments. Each tolerance has its own production path; refinements are conditional native Brownian bridges, no new coarse noise. No timing claims from this instrumented diagnostic.',rows=[])
    for fieldname in args.fields:
        p=ProtonBackground(coulomb_log_model='impact');sigma,nu=p.sigma,p.nu
        field=(MagneticMirror(2048*nu*M/Q,.03*sigma/nu) if fieldname=='mirror' else
            HelicalField(2e6*nu*M/Q,2e4*nu/sigma) if fieldname=='helix' else TransitField(2048*nu*M/Q))
        bath=background(p,varying=fieldname=='profiles')
        if fieldname=='profiles':
            expected=dict(position_variance=np.ones(6),pitch_variance=np.ones(6));speed_expected=dict(speed_variance=np.ones(6))
        else:
            _,covariance,_=physical_moments(data,fieldname)
            expected=dict(position_variance=covariance[:,0,0],pitch_variance=covariance[:,2,2])
            speed_expected=dict(speed_variance=covariance[:,1,1])
        keys=jax.random.split(jax.random.PRNGKey(8706 if fieldname=='mirror' else 9706),2000)[:args.particles]
        for composition in ('lie',):
            paired=make_paired_solver(original,max_roots=args.max_roots,refinements=tuple(args.refinements))
            def batch(keys,tolerance):
                birth=Particles(jnp.zeros((len(keys),3)),mass=M,charge=Q,energy=5*p.temperature_ev*ONE_EV,initial_vparallel_over_v=jnp.full(len(keys),.6))
                birth.random_keys=keys
                return split.trace_split(field,birth,bath,tmax=.025/nu,timestep=.0025/nu,collision_dtmax=.0025/nu,
                    times=jnp.linspace(0.,.025/nu,6),orbit_dt0=(.1/20000 if fieldname=='helix' else min(.0025,.1/2048))/nu,
                    orbit_stage_execution='unrolled',rtol=1e-10,atol=1e-10,collision_tolerance=.001,speed_floor=.05*np.sqrt(2.)*sigma,
                    spatial_diffusion=False,  # Retain the axial coupled-reference model and its two noise components.
                    step_limit=make_step_limit(field,bath,p,tolerance) if args.control=='moments' else None,
                    combined_tolerance=tolerance if args.control=='combined' else None,
                    combined_length_scale=.03*sigma/nu,max_steps=args.max_steps,max_orbit_steps=4096)
            fn=jax.jit(batch)
            for tolerance in args.tolerances:
                print('PAIRED START',fieldname,'budget',tolerance,'paths',args.particles,'refinements',args.refinements,flush=True)
                start=perf_counter();previous=split.solve_split;split.solve_split=paired
                try:out=jax.device_get(fn(keys,jnp.asarray(tolerance)))
                finally:split.solve_split=previous
                elapsed=perf_counter()-start
                refs,valid,roots,dws=out.step_ts
                assert np.all(valid) and np.all(out.status==0),(fieldname,composition,tolerance,valid,out.status)
                def phase(states):
                    points=states[...,:3]
                    b=np.asarray(jax.jit(jax.vmap(field.AbsB))(jnp.asarray(points.reshape(-1,3)))).reshape(points.shape[:-1])
                    speed=np.sqrt(states[...,3]**2+2*states[...,4]*b/M)/sigma
                    return states[...,:3]*nu/sigma,speed,states[...,3]/(speed*sigma)
                xc,vc,pc=phase(out.ys);xr,vr,pr=phase(refs)
                rms=[]
                for j in range(3):
                    rms.append(dict(refinement=args.refinements[j],position=(np.sqrt(np.mean(np.sum((xc-xr[:,j])**2,axis=-1),axis=0))/.03).tolist(),
                        speed=(np.sqrt(np.mean((vc-vr[:,j])**2,axis=0))/np.sqrt(10)).tolist(),pitch=np.sqrt(np.mean((pc-pr[:,j])**2,axis=0)).tolist()))
                weak=[]
                for i in range(1,6):
                    moments={}
                    axis=0 if fieldname=='profiles' else 2
                    for name,a,b,denominator in (('position_variance',xc[:,i,axis],xr[:,2,i,axis],expected['position_variance'][i]),
                        ('speed_variance',vc[:,i],vr[:,2,i],speed_expected['speed_variance'][i]),
                        ('pitch_variance',pc[:,i],pr[:,2,i],expected['pitch_variance'][i])):
                        if fieldname=='profiles':denominator=np.var(b,ddof=1)
                        influence=args.particles/(args.particles-1)*((a-a.mean())**2-(b-b.mean())**2)
                        value=estimate(influence,0.)
                        moments[name]=dict(percent=100*value['value']/denominator,half95=100*value['half95']/denominator)
                    weak.append(moments)
                row=dict(field=fieldname,composition=composition,tolerance=tolerance,seconds_including_compile=elapsed,
                    refinements=args.refinements,
                    relative_moment_scale='fine-sample variance' if fieldname=='profiles' else 'independent reference variance',
                    mean_collision_updates=float(out.stats.collision_updates.mean()),rms=rms,paired_weak=weak,all_reference_solves_passed=True)
                report['rows'].append(row)
                np.savez_compressed(folder/f'{fieldname}_{composition}_{tolerance:g}.npz',coarse=out.ys,reference=refs,roots=roots,noise=dws,counts=out.stats.accepted)
                (folder/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
                print('GLOBAL PAIRED',fieldname,composition,tolerance,'kicks',row['mean_collision_updates'],'RMS final',rms[-1]['position'][-1],rms[-1]['speed'][-1],rms[-1]['pitch'][-1],
                    'early weak',weak[0],'later weak',weak[3],'seconds',elapsed,flush=True)
    print('Saved',folder,flush=True)
    if not args.no_plots:plot(report,folder)


def plot(report,folder):
    """Plot paired numerical errors, separate from population/theory gaps."""
    import numpy as np
    import matplotlib.pyplot as plt
    fields=list(dict.fromkeys(r['field'] for r in report['rows']))
    fig,axes=plt.subplots(len(fields),3,figsize=(14,4*len(fields)),squeeze=False,constrained_layout=True)
    weak,panels=plt.subplots(len(fields),3,figsize=(14,4*len(fields)),squeeze=False,constrained_layout=True)
    for fieldname,row_axes,row_panels in zip(fields,axes,panels):
        for comp in ('lie',):
            rows=sorted((r for r in report['rows'] if r['field']==fieldname and r['composition']==comp),key=lambda r:r['tolerance'])
            if not rows:continue
            for ax,name in zip(row_axes,('position','speed','pitch')):
                ax.loglog([r['tolerance'] for r in rows],[r['rms'][-1][name][-1] for r in rows],'-o',label=comp+'; finest reference')
                ax.loglog([r['tolerance'] for r in rows],[r['rms'][-2][name][-1] for r in rows],':',alpha=.6,label=comp+'; preceding reference')
                ax.set(title=fieldname+': '+name,xlabel='Local tolerance',ylabel='Normalized global RMS error');ax.grid(alpha=.2,which='both')
        for row in (r for r in report['rows'] if r['field']==fieldname):
            for ax,name in zip(row_panels,('position_variance','speed_variance','pitch_variance')):
                points=[r[name] for r in row['paired_weak']]
                ax.errorbar(np.linspace(.005,.025,5),[p['percent'] for p in points],yerr=[p['half95'] for p in points],alpha=.8,
                    label=f"{row['composition']} {row['tolerance']:g}")
                ax.set(title=fieldname+': '+name.replace('_',' '),xlabel=r'$t\nu_0$',ylabel='Paired numerical difference [%]');ax.grid(alpha=.2)
        row_axes[0].legend(fontsize=7)
        row_panels[0].legend(fontsize=7)
        for ax in row_panels:ax.axhline(0,color='k',lw=1.5)
    fig.suptitle(f"Brownian-coupled numerical convergence; N={report['particles']}; final time t nu0=.025")
    weak.suptitle(f"Coarse minus refined native statistics on the SAME Brownian path; N={report['particles']}\nPaired 95% sampling intervals; zero is the reference")
    fig.savefig(folder/'local_rms_path_convergence.png',dpi=160)
    weak.savefig(folder/'local_rms_paired_moments.png',dpi=160)
    plt.close(fig);plt.close(weak)


if __name__=='__main__':main()
