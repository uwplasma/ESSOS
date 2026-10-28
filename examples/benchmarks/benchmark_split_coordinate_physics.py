"""Regenerate coupled mirror/helix statistics in Cartesian and flux charts.

These are coordinate transformations of prescribed fields, not VMEC MHD
equilibria or Boozer K=0 magnetic spectra. Real equilibrium adapter checks are
separate. All charts use the production collision maps, moment predictor,
retained Brownian driver and nonlinear Cartesian spatial-kick inversion.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from essos.background_species import BackgroundSpecies
from essos.constants import PROTON_MASS as M,ELEMENTARY_CHARGE as Q
from essos.dynamics import _guiding_center_velocity
from essos.electric_field import Electric_field_zero
from essos.rk import OrbitErrorNorm
from essos.split import (solve_split,collision_coefficients,guiding_center_collision_step,
    guiding_center_spatial_increment,particle_collision_step,native_step_limit,move_in_cartesian,BoozerGeometry)
from .benchmark_split_mirror import MagneticMirror
from .benchmark_split_helix import HelicalField
from .benchmark_split_multirate import UniformMagneticField
from .nonrelativistic_collision_reference import ProtonBackground
from .validate_beam_relaxation import estimate


def chart_maps(chart,scale):
    """Circular-tube embedding, regular axis variables and VMEC angle gauge.

    The Boozer-style chart also has a toroidal angle shift. This manufactured
    geometry checks coordinate covariance; it is not a magnetic equilibrium.
    """
    if chart=='cartesian':return lambda y:y,lambda y:y
    gauge=.2 if chart=='vmec' else 0.;shift=.1 if chart=='boozer' else 0.
    def cartesian(y):
        u,w=y[0],y[1]
        if chart=='vmec':u,w=u*jnp.cos(y[-1])-w*jnp.sin(y[-1]),w*jnp.cos(y[-1])+u*jnp.sin(y[-1])
        phi=y[2]-shift*w;R=scale*(3+u)
        return jnp.concatenate((jnp.array([R*jnp.cos(phi)-3.3*scale,R*jnp.sin(phi),scale*w]),y[3:5]))
    def regular(y):
        R=jnp.hypot(y[0]+3.3*scale,y[1]);u=R/scale-3;w=y[2]/scale
        zeta=jnp.arctan2(y[1],y[0]+3.3*scale)+shift*w
        point=jnp.array([u*jnp.cos(gauge)+w*jnp.sin(gauge),w*jnp.cos(gauge)-u*jnp.sin(gauge),zeta])
        return jnp.concatenate((point,y[3:5],jnp.array([gauge]) if chart=='vmec' else jnp.empty(0)))
    return cartesian,regular


def birth_state(field,sigma,model):
    """Same birth speed/pitch; Lorentz gyromotion is centred at the GC birth."""
    speed=sigma*np.sqrt(10.);point=jnp.zeros(3);B=field.AbsB(point)
    if model=='gc':return jnp.array([0.,0.,0.,.6*speed,M*(.8*speed)**2/(2*B)])
    unit=field.B_contravariant(point)/B
    perpendicular=jnp.cross(jnp.array([0.,1.,0.]),unit)
    perpendicular=perpendicular/jnp.linalg.norm(perpendicular)
    velocity=speed*(.6*unit+.8*perpendicular)
    point=-jnp.cross(velocity,unit)/(Q*B/M)
    return jnp.concatenate((point,velocity))


def solver(fieldname,chart,options,*,spatial=True,model='gc'):
    p=ProtonBackground(coulomb_log_model='impact');sigma,nu=p.sigma,p.nu
    omega=2e6 if fieldname=='helix' else 2048.
    field=(MagneticMirror(omega*nu*M/Q,.03*sigma/nu) if fieldname=='mirror' else
        HelicalField(omega*nu*M/Q,20000*nu/sigma) if fieldname=='helix' else
        UniformMagneticField(omega*nu*M/Q))
    bath=BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),jnp.array([p.density]),
        jnp.array([p.temperature_ev]),coulomb_log_model='impact')
    electric=Electric_field_zero();gc=model=='gc'
    to_cartesian,to_regular=chart_maps(chart,sigma/nu)
    y0=birth_state(field,sigma,model)
    def physical_rhs(t,y,args):
        if gc:
            dx,dv=_guiding_center_velocity(field,electric,Q,M,y[:3],y[3],mu=y[4])
            return jnp.concatenate((dx,jnp.array([dv,0.])))
        return jnp.concatenate((y[3:],Q/M*jnp.cross(y[3:],field.B_contravariant(y[:3]))))
    cap=getattr(options,'cap',.0025)
    bound=native_step_limit(field,bath,M,Q,y0,physical_rhs,lambda y:y,lambda y:y[:3],field.AbsB,
        gc=gc,electric_field=electric,tmax=options.horizon/nu,dtmax=cap/nu,tolerance=options.tolerance)
    initial=to_regular(y0)
    def rhs(t,y,args):
        if chart=='cartesian':return physical_rhs(t,y,args)
        physical=to_cartesian(y);flow=physical_rhs(t,physical,args)
        result=jax.jvp(to_regular,(physical,),(flow,))[1]
        if chart=='vmec':
            # Re-express the inverse with this state's gauge, not the birth gauge.
            angle=y[-1]-.2;c,s=jnp.cos(angle),jnp.sin(angle)
            du,dw=result[0]*c+result[1]*s,result[1]*c-result[0]*s
            radius=y[0]**2+y[1]**2
            rotation=(y[0]*dw-y[1]*du)/(radius+.01)
            result=result.at[0].set(du+y[1]*rotation).at[1].set(dw-y[0]*rotation).at[-1].set(rotation)
        return result
    def kick(t,y,h,dw,args):
        physical=to_cartesian(y);point=physical[:3];B=field.AbsB(point)
        speed=jnp.sqrt(physical[3]**2+2*physical[4]*B/M) if gc else jnp.linalg.norm(physical[3:])
        coefficients=collision_coefficients(bath,M,Q,speed,point,full_orbit=not gc)
        if not gc:
            step=particle_collision_step(physical[3:],h,dw,coefficients,tolerance=.001)
            return step._replace(y=y.at[3:].set(step.y))
        pitch=physical[3]/speed
        step=guiding_center_collision_step(speed,pitch,h,dw[-2:],coefficients,
            tolerance=.001,speed_floor=.05*np.sqrt(2)*sigma)
        if spatial:
            dx=guiding_center_spatial_increment(pitch,dw[:3],coefficients,field.B_contravariant(point)/B,Q*B/M)
            y=y.at[:3].add(dx) if chart=='cartesian' else move_in_cartesian(y,dx,lambda q:to_cartesian(q)[:3])
            B=field.AbsB(to_cartesian(y)[:3])
        v,xi=step.y
        return step._replace(y=y.at[3].set(v*xi).at[4].set(M*v*v*(1-xi*xi)/(2*B)))
    times=np.linspace(0.,options.horizon,6);si_times=times/nu;si_times[-1]=options.horizon/nu
    def one(key):
        out=solve_split(rhs,kick,initial,t1=options.horizon/nu,dt0=cap/nu,dtmax=cap/nu,
            ts=jnp.asarray(si_times),key=key,noise_dim=5 if gc and spatial else 2 if gc else 3,
            method='dopri8',rtol=1e-10,atol=1e-10,orbit_stage_execution='unrolled',
            error_norm=OrbitErrorNorm(1e-10,1e-10,float(sigma*np.sqrt(10)),velocity_components=1 if gc else 3,
                position_map=lambda y:to_cartesian(y)[:3]),
            step_limit=lambda t,y,a:bound(t,to_cartesian(y),a),align_orbit=True,
            collision_wiener_norm=(lambda w:jnp.linalg.norm(w[-2:])) if gc else lambda w:jnp.abs(w[2]),
            max_steps=10000,max_orbit_steps=4096)
        return out._replace(ys=jax.vmap(to_cartesian)(out.ys),y=to_cartesian(out.y))
    return jax.jit(jax.vmap(one)),p,field,times


def summarize(states,physical,field,model):
    xyz=states[...,:3]
    if model=='gc':
        B=np.asarray(jax.vmap(jax.vmap(field.AbsB))(jnp.asarray(xyz)))
        speed=np.sqrt(states[...,3]**2+2*states[...,4]*B/M);parallel=states[...,3]
    else:
        speed=np.linalg.norm(states[...,3:],axis=-1)
        unit=np.asarray(jax.vmap(jax.vmap(lambda x:field.B_contravariant(x)/field.AbsB(x)))(jnp.asarray(xyz)))
        parallel=np.sum(states[...,3:]*unit,axis=-1)
    return np.stack((xyz[...,2],speed,parallel/speed,parallel),axis=-1)


def apply_lorentz_reference(report,reference):
    """Correct spatial/pitch references; keep independent radial speed theory."""
    from scipy.stats import t as student_t
    critical=student_t.ppf(.975,min(report['particles'],reference['particles'])-1)
    for row in report['rows']:
        if row.get('model','gc')!='lorentz':continue
        target=reference['fields'][row['field']]['levels'][-1]['evolution']
        if len(target)!=len(row['evolution']):raise ValueError('Lorentz reference save times do not match the particle report')
        for measured,expected in zip(row['evolution'],target):
            for name,value in measured.items():
                if name.startswith('speed_'):continue
                other=expected[name];value.update(reference=other['value'],reference_se=other['se'],
                    reference_half95=other['half95'],difference=value['value']-other['value'])
                se=np.hypot(value['se'],other['se'])
                value.update(difference_half95=float(critical*se),z=float(value['difference']/se) if se>1e-14 else None)
        row['reference_scope']='Independent full-Lorentz numerical ensemble with refinement controls; speed uses independent radial theory.'
    report['lorentz_reference']=reference


def run(options):
    from .benchmark_combined_validation import references,add_speed_reference,physical_moments
    output=options.output;output.mkdir(parents=True,exist_ok=True)
    reference_path=options.reference or output/'reference.json'
    if reference_path.exists():reference=json.loads(reference_path.read_text())
    else:
        reference=references(SimpleNamespace(fields=options.fields,output=output))
        add_speed_reference(SimpleNamespace(output=output),reference)
    report=dict(moment_tolerance=options.tolerance,collision_tolerance=.001,orbit_tolerance=1e-10,seed=options.seed,cap=options.cap,
        particles=options.particles,backend=jax.default_backend(),device=jax.devices()[0].device_kind,
        scope='Same prescribed Cartesian physics transformed into regular flux charts; these are not VMEC equilibria or Boozer K=0 spectra.',rows=[])
    numerical=None
    if 'lorentz' in options.models:
        from .lorentz_collision_reference import generate
        path=options.lorentz_reference or output/'lorentz_reference.json'
        numerical=json.loads(path.read_text()) if path.exists() else generate(options.fields,options.particles,output=path)
    for name in options.fields:
        expected_mean,expected_cov,error=physical_moments(reference,name)
        configurations=[(model,chart) for model in options.models for chart in (options.charts if model=='gc' else ['cartesian'])]
        for model,chart in configurations:
            function,p,field,times=solver(name,chart,options,spatial=options.spatial_diffusion,model=model)
            seed=options.seed if options.seed is not None else 8706 if name=='mirror' else 9706
            keys=jax.random.split(jax.random.PRNGKey(seed),options.particles)
            print('Compiling',name,model,chart,'N=',options.particles,flush=True)
            start=perf_counter();executable=function.lower(keys).compile();compile_s=perf_counter()-start
            start=perf_counter();out=jax.device_get(jax.block_until_ready(executable(keys)));elapsed=perf_counter()-start
            if np.any(out.status):raise RuntimeError((name,chart,np.unique(out.status,return_counts=True)))
            values=summarize(out.ys,p,field,model);n=len(values)
            units=np.array([p.sigma/p.nu,p.sigma,1.,p.sigma]);mu=expected_mean*units;cov=expected_cov*units[None,:,None]*units[None,None,:]
            if model=='lorentz':
                from .benchmark_split_multirate import velocity_reference
                _,_,radial,_,_=velocity_reference(model,times,'impact',need_pitch=False)
                mu[:,1]=np.asarray(radial['moments']['speed_mean'])*p.sigma
                cov[:,1,1]=np.asarray(radial['moments']['speed_variance'])*p.sigma**2
            checks=[]
            for i in range(6):
                row={}
                for k,variable in enumerate(('position','speed','pitch','parallel')):
                    row[variable+'_mean']=estimate(values[:,i,k],mu[i,k])
                    row[variable+'_variance']=estimate(values[:,i,k],cov[i,k,k],variance=True)
                row['position_parallel_covariance']=estimate(n/(n-1)*(values[:,i,0]-values[:,i,0].mean())*(values[:,i,3]-values[:,i,3].mean()),cov[i,0,3])
                checks.append(row)
            counts=out.stats
            record=dict(field=name,model=model,chart=chart,spatial_diffusion=options.spatial_diffusion and model=='gc',compile_s=compile_s,execution_s=elapsed,
                reference_scope='Independent axial GC reference' if model=='gc' else 'Independent speed theory; GC-limit comparison only for position/pitch/parallel moments',
                collision_updates=int(counts.collision_updates.sum()),accepted_orbit_steps=int(counts.accepted_orbit_steps.sum()),
                max_orbit_steps_per_collision=int(counts.max_orbit_steps_per_collision.max()),evolution=checks)
            report['rows'].append(record)
            if numerical is not None:apply_lorentz_reference(report,numerical)
            suffix='' if model=='gc' else '_lorentz'
            np.savez_compressed(output/f'{name}_{chart}{suffix}.npz',moments=values,states=out.ys)
            (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,chart,'completed',elapsed,'s; orbit/collision',record['accepted_orbit_steps']/record['collision_updates'],flush=True)
    plot(report,output,times)


def plot(report,output,times):
    import matplotlib.pyplot as plt
    panels=[('position_mean','Mean z [m]'),('speed_mean','Mean speed [m/s]'),('pitch_mean','Mean pitch'),
        ('position_variance','Var z [m²]'),('speed_variance','Var speed [m²/s²]'),('pitch_variance','Var pitch'),
        ('parallel_mean','Mean parallel speed [m/s]'),('parallel_variance','Var parallel speed [m²/s²]'),
        ('position_parallel_covariance','Cov(z,vparallel) [m²/s]')]
    for field,model in dict.fromkeys((row['field'],row.get('model','gc')) for row in report['rows']):
        rows=[row for row in report['rows'] if row['field']==field and row.get('model','gc')==model]
        fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
        for ax,(name,label) in zip(axes.flat,panels):
            expected=[e[name]['reference'] for e in rows[0]['evolution']]
            exact_speed=name.startswith('speed_')
            ax.plot(times,expected,color='k',lw=2,
                label='Independent speed theory' if exact_speed else 'Independent axial GC reference' if model=='gc' else 'Independent full-Lorentz numerical reference')
            half=np.array([e[name].get('reference_half95',0.) for e in rows[0]['evolution']])
            if np.any(half):ax.fill_between(times,np.asarray(expected)-half,np.asarray(expected)+half,color='k',alpha=.12,label='Reference 95% sampling')
            for row in rows:
                values=np.array([e[name]['value'] for e in row['evolution']]);half=np.array([e[name]['half95'] for e in row['evolution']])
                curve_label=model.upper()+' / '+{'cartesian':'Cartesian','vmec':'VMEC regular chart','boozer':'Boozer regular chart'}[row['chart']]
                line,=ax.plot(times,values,label=curve_label);ax.fill_between(times,values-half,values+half,color=line.get_color(),alpha=.12)
            ax.set(xlabel=r'$t\nu_0$',ylabel=label);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useOffset=False)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f"{model.upper()} {field}: N={report['particles']}; moment budget {report['moment_tolerance']:g}\n"+
            ('Shading: pointwise 95% sampling intervals. Axial GC reference omits perpendicular kicks.' if model=='gc' else
             'Black: independent full-Lorentz numerical reference (speed: radial theory). Grey/blue: separate 95% sampling intervals.'))
        suffix='' if model=='gc' else '_lorentz'
        fig.savefig(output/f'{field}{suffix}_moments.png',dpi=160);plt.close(fig)


def diffusion(options):
    """Known transient velocity and perpendicular-diffusion references."""
    from .benchmark_split_multirate import velocity_reference,observables
    from .lorentz_spatial_reference import guiding_center_msd
    output=options.output;output.mkdir(parents=True,exist_ok=True)
    times=np.linspace(0.,options.horizon,6);rows=[];omega=2048.
    for model in options.models:
        coarse,fine,reference,_,_=velocity_reference(model,times,'impact',need_pitch=False)
        target=reference['moments']
        expected=np.asarray(target['spatial_integral']) if model=='gc' else guiding_center_msd(fine,times,omega)
        for chart in options.charts if model=='gc' else ['cartesian']:
            function,p,field,_=solver('uniform',chart,options,model=model)
            seed=7706 if options.seed is None else options.seed
            keys=jax.random.split(jax.random.PRNGKey(seed),options.particles)
            print('Compiling diffusion',model,chart,flush=True)
            out=jax.device_get(jax.block_until_ready(function(keys)))
            if np.any(out.status):raise RuntimeError((model,chart,np.unique(out.status,return_counts=True)))
            scale=np.array([p.sigma/p.nu]*3+([p.sigma,M*p.sigma**2/field.magnitude] if model=='gc' else [p.sigma]*3))
            speed,pitch,_,msd=observables(out.ys/scale,omega,model)
            values=(msd,speed,pitch,speed,pitch,speed*pitch)
            names=('spatial','speed_mean','pitch_mean','speed_variance','pitch_variance','parallel_variance')
            evolution=[]
            for i in range(6):
                evolution.append({name:estimate(value[:,i],expected[i] if name=='spatial' else target[name][i],
                    variance='variance' in name) for name,value in zip(names,values)})
            label=model+' / '+chart
            row=dict(model=model,chart=chart,evolution=evolution,collision_updates=int(out.stats.collision_updates.sum()),
                accepted_orbit_steps=int(out.stats.accepted_orbit_steps.sum()),
                max_orbit_steps_per_collision=int(out.stats.max_orbit_steps_per_collision.max()))
            rows.append(row)
            print('Completed diffusion',label,'orbit/collision',row['accepted_orbit_steps']/row['collision_updates'],flush=True)
            report=dict(particles=options.particles,moment_tolerance=options.tolerance,times=times.tolist(),seed=seed,cap=options.cap,
                omega=omega,scope='Uniform physical field; manufactured flux charts; normalized speed and rho_sigma displacement.',rows=rows)
            (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_diffusion(report,output)


def plot_diffusion(report,output):
    import matplotlib.pyplot as plt
    times=np.asarray(report.get('times',np.linspace(0.,.1,6)))
    panels=(('spatial',r'$E[|\Delta G_\perp|^2]/\rho_\sigma^2$','Spatial diffusion'),
        ('speed_mean',r'$E[v]/\sigma$','Mean speed'),('pitch_mean',r'$E[\xi]$','Mean pitch'),
        ('speed_variance',r'$\mathrm{Var}(v)/\sigma^2$','Speed variance'),
        ('pitch_variance',r'$\mathrm{Var}(\xi)$','Pitch variance'),
        ('parallel_variance',r'$\mathrm{Var}(v_\parallel)/\sigma^2$','Parallel-speed variance'))
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');seen=set()
    for row in report['rows']:
        for ax,(name,label,title) in zip(axes.flat,panels):
            if row['model'] not in seen:
                ax.plot(times,[e[name]['reference'] for e in row['evolution']],color='k',
                    ls='-' if row['model']=='gc' else '--',lw=2,label='Independent '+row['model']+' reference')
            values=np.array([e[name]['value'] for e in row['evolution']]);half=np.array([e[name]['half95'] for e in row['evolution']])
            line,=ax.plot(times,values,label=row['model']+' / '+row['chart'])
            ax.fill_between(times,values-half,values+half,color=line.get_color(),alpha=.1)
            ax.set(title=title,xlabel=r'$t\nu_0$',ylabel=label);ax.grid(alpha=.2)
        seen.add(row['model'])
    axes[0,0].legend(fontsize=6)
    fig.suptitle(f"Uniform-field coupled spatial/velocity statistics; N={report['particles']}; moment budget {report['moment_tolerance']:g}\n"
        'Black = independent transient generator; shading = pointwise 95% sampling intervals; GC spatial kicks enabled')
    fig.savefig(output/'diffusion_moments.png',dpi=160);plt.close(fig)


def adapters(options):
    """Actual QA field-table checks, distinct from manufactured-chart theory."""
    from netCDF4 import Dataset
    from booz_xform_jax import Booz_xform
    from essos.coils import Coils
    from essos.fields import BiotSavart,Vmec
    from essos.boozer import BoozerField
    from essos.dynamics import Particles
    from essos.split import trace_split
    root=Path(__file__).resolve().parents[2]/'examples/input_files'
    path=root/'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'
    transform=Booz_xform(verbose=0,mboz=options.mboz,nboz=options.nboz)
    transform.read_wout(str(path),flux=False);transform.run()
    with Dataset(path) as data:psi0=-float(data.variables['phi'][-1])/(2*np.pi)
    boozer=BoozerField.from_booz_xform(transform,psi0,angle_grid=None)
    geometry=BoozerGeometry.from_booz_xform(transform)
    coil=BiotSavart(Coils.from_json(str(root/'ESSOS_biot_savart_LandremanPaulQA.json')))
    coil.coils.gamma,coil.coils.gamma_dash,coil.coils.currents
    vmec=Vmec(str(path),ntheta=8,nphi=8)
    bath=BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),jnp.array([1e20]),jnp.array([1000.]),coulomb_log_model='impact')
    options.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,field,embedding,model in (('coils_gc',coil,None,'GuidingCenter'),
        ('coils_lorentz',coil,None,'FullOrbit'),('vmec_gc',vmec,None,'GuidingCenter'),('boozer_gc',boozer,geometry,'GuidingCenter')):
        if name not in options.adapter_fields:continue
        points=(jnp.column_stack((jnp.linspace(1.23,1.27,options.particles),jnp.zeros((options.particles,2))))
            if field is coil else jnp.column_stack((jnp.full(options.particles,.3),jnp.linspace(.2,.5,options.particles),jnp.full(options.particles,.1))))
        birth=Particles(points,mass=M,charge=Q,energy=5000*Q,
            initial_vparallel_over_v=jnp.linspace(.4,.6,options.particles),field=field if model=='FullOrbit' else None,
            phase_angle_full_orbit=jnp.pi)
        birth.random_keys=jax.random.split(jax.random.PRNGKey(42),options.particles)
        for clock in options.clocks:
            outputs={}
            for execution in ('shared','independent') if jax.default_backend()=='gpu' else ('shared',):
                def run():return trace_split(field,birth,bath,model=model,tmax=4e-10,timestep=1e-10,
                    collision_dtmax=2e-10,orbit_dt0=1e-10,n_save=3,adaptive=clock=='adaptive',orbit_adaptive=clock=='adaptive',
                    collision_controller='moments',moment_tolerance=options.tolerance,spatial_diffusion=model=='GuidingCenter',
                    spatial_geometry=embedding,max_steps=32,max_orbit_steps=64,execution=execution)
                print('Preparing actual QA adapter',name,clock,execution,flush=True)
                start=perf_counter();exe=jax.jit(run).lower().compile();compile_s=perf_counter()-start
                first=jax.device_get(jax.block_until_ready(exe()));second=jax.device_get(jax.block_until_ready(exe()))
                print('Results',name,clock,execution,'failed',int(np.count_nonzero(first.status)),
                    'repeatable',np.array_equal(first.ys,second.ys),flush=True)
                if np.any(first.status) or not np.array_equal(first.ys,second.ys):raise RuntimeError((name,clock,execution,'failure or nonrepeatable'))
                outputs[execution]=first
                rows.append(dict(field=name,clock=clock,execution=execution,compile_s=compile_s,failed=0,
                    collision_updates=int(first.stats.collision_updates.sum()),accepted_orbit_steps=int(first.stats.accepted_orbit_steps.sum())))
                (options.output/'comparison.json').write_text(json.dumps(dict(particles=options.particles,moment_tolerance=options.tolerance,
                    mboz=options.mboz,nboz=options.nboz,scope='Real QA field tables; repeatability and shared/GPU path agreement, not an independent statistical physics reference.',rows=rows),indent=2)+'\n')
            if 'independent' in outputs:
                np.testing.assert_allclose(outputs['independent'].ys,outputs['shared'].ys,rtol=2e-9,atol=3e-12)
            print('PASS actual QA adapter',name,clock,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles',type=int,default=2000)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--cap',type=float,default=.0025)
    parser.add_argument('--tolerance',type=float,default=.01)
    parser.add_argument('--horizon',type=float)
    parser.add_argument('--study',choices=('coupled','diffusion','adapters'),default='coupled')
    parser.add_argument('--models',nargs='+',choices=('gc','lorentz'),help='Coupled defaults to GC; select Lorentz explicitly for full-gyromotion runs')
    parser.add_argument('--fields',nargs='+',choices=('mirror','helix'),default=['mirror','helix'])
    parser.add_argument('--charts',nargs='+',choices=('cartesian','vmec','boozer'),default=['cartesian','vmec','boozer'])
    parser.add_argument('--spatial-diffusion',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--lorentz-reference',type=Path,help='Independent full-Lorentz ensemble reference; generated for coupled Lorentz runs if absent')
    parser.add_argument('--plot-only',action='store_true',help='Regenerate figures from an ignored generated report without tracing')
    parser.add_argument('--mboz',type=int,default=4);parser.add_argument('--nboz',type=int,default=4)
    parser.add_argument('--clocks',nargs='+',choices=('fixed','adaptive'),default=['fixed','adaptive'])
    parser.add_argument('--adapter-fields',nargs='+',choices=('coils_gc','coils_lorentz','vmec_gc','boozer_gc'),
        default=['coils_gc','coils_lorentz','vmec_gc','boozer_gc'])
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    if options.models is None:options.models=['gc'] if options.study=='coupled' else ['gc','lorentz']
    if options.horizon is None:options.horizon=.025 if options.study=='coupled' else .1
    if options.particles<2 or options.tolerance<=0 or options.cap<=0 or (options.study=='coupled' and options.horizon!=.025):
        parser.error('use >=2 particles, positive budget and the coupled reference horizon .025')
    if options.plot_only:
        report=json.loads((options.output/'comparison.json').read_text())
        if options.lorentz_reference is not None:
            apply_lorentz_reference(report,json.loads(options.lorentz_reference.read_text()))
            (options.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        elif any(row.get('model')=='lorentz' for row in report['rows']) and options.study=='coupled' and 'lorentz_reference' not in report:
            parser.error('old Lorentz reports require --lorentz-reference; GC curves are not Lorentz references')
        if options.study=='diffusion':plot_diffusion(report,options.output)
        elif options.study=='coupled':plot(report,options.output,np.linspace(0.,options.horizon,6))
        else:parser.error('adapter checks have no figure')
    else:{'coupled':run,'diffusion':diffusion,'adapters':adapters}[options.study](options)


if __name__=='__main__':main()
