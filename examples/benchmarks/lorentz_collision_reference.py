"""Independent full-Lorentz numerical ensemble reference, with refinement.

Dimensionless Cartesian Lorentz motion uses a centred exact-angle rotation. Coulomb
coefficients are independently evaluated from the proton impact-parameter law;
Cartesian Euler--Maruyama samples the corresponding Ito generator. This is a
reference calculation, not an alteration of the production splitting scheme.
Black reference curves require their own sampling and refinement uncertainties.
An orthogonal frame transported by the magnetic rotation couples refinement
levels without replacing their physical diffusion tensor or Ito drift.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf,gammainc
from essos.constants import ELEMENTARY_CHARGE as Q,PROTON_MASS as M,EPSILON_0,HBAR
from .nonrelativistic_collision_reference import ProtonBackground
from .validate_beam_relaxation import estimate


def magnetic(x,field):
    x=jnp.asarray(x);z=x[...,2]
    if field=='helix':return 2e6*jnp.stack((jnp.sqrt(.75)*jnp.cos(20000*z),jnp.sqrt(.75)*jnp.sin(20000*z),jnp.full_like(z,.5)),axis=-1)
    if field=='mirror':return 2048*jnp.stack((-x[...,0]*z/.03**2,-x[...,1]*z/.03**2,1+z*z/.03**2),axis=-1)
    if field!='uniform':raise ValueError('unknown reference magnetic field')
    return jnp.broadcast_to(jnp.array([0.,0.,2048.]),x.shape)


def coefficients(r):
    """Independent dimensionless parallel/perpendicular diffusion and Ito drift."""
    p=ProtonBackground(coulomb_log_model='impact');s=p.sigma
    relative=s*s*(r*r+3);classical=Q*Q/(4*np.pi*EPSILON_0*(M/2));quantum=HBAR/M
    debye=np.sqrt(EPSILON_0*p.temperature_ev/(p.density*Q))
    logarithm=jnp.log(debye)+jnp.log(relative)-.5*jnp.log(classical**2+quantum**2*relative)
    slope=2*r/(r*r+3)-quantum**2*s*s*r/(classical**2+quantum**2*relative)
    x=r/jnp.sqrt(2.);g=gammainc(1.5,x*x)/(2*x*x)
    dg=(2/jnp.sqrt(jnp.pi)*jnp.exp(-x*x)-2*g/x)/jnp.sqrt(2.)
    factor=Q**4*p.density/(4*np.pi*EPSILON_0**2*M**2*s**3*p.nu)
    parallel=factor*logarithm*g/r
    perpendicular=.5*factor*logarithm*(erf(x)-g)/r
    derivative=factor*(slope*g/r+logarithm*(dg/r-g/r**2))
    drift=-r*parallel+derivative+2*(parallel-perpendicular)/r
    return parallel,perpendicular,drift


def orbit(y,h,field,phase):
    """Centred drift/rotation/drift; magnetic-only update preserves |v|."""
    # A conservative bound for the tested mirror trajectories; checked below.
    frequency=2e6 if field=='helix' else 4*2048
    count=max(1,int(np.ceil(h*frequency/phase)));dt=h/count
    def step(_,state):
        x,v=state[...,:3],state[...,3:6];middle=x+.5*dt*v
        B=magnetic(middle,field);strength=jnp.linalg.norm(B,axis=-1,keepdims=True);unit=B/strength
        angle=dt*strength;sine=jnp.sin(angle);cosine_defect=2*jnp.sin(angle/2)**2
        cross=jnp.cross(v,unit);rotated=v+sine*cross+cosine_defect*jnp.cross(cross,unit)
        result=jnp.concatenate((middle+.5*dt*rotated,rotated),axis=-1)
        if state.shape[-1]>6:
            frame=state[...,6:].reshape(state.shape[:-1]+(3,3))
            cross=jnp.cross(frame,unit[...,None,:])
            frame=frame+sine[...,None]*cross+cosine_defect[...,None]*jnp.cross(cross,unit[...,None,:])
            result=jnp.concatenate((result,frame.reshape(state.shape[:-1]+(9,))),axis=-1)
        return result
    return jax.lax.fori_loop(0,count,step,y)


def collision(y,h,dw):
    v=y[...,3:6];r=jnp.linalg.norm(v,axis=-1);u=v/r[...,None]
    if y.shape[-1]>6:dw=jnp.einsum('...i,...ij->...j',dw,y[...,6:].reshape(y.shape[:-1]+(3,3)))
    parallel,perpendicular,drift=coefficients(r)
    root=jnp.sqrt(2*perpendicular);difference=jnp.sqrt(2*parallel)-root
    increment=h*drift[...,None]*u+root[...,None]*dw+difference[...,None]*u*jnp.sum(u*dw,axis=-1,keepdims=True)
    return y.at[...,3:6].add(increment)


def birth(field):
    b=np.array([np.sqrt(.75),0.,.5]) if field=='helix' else np.array([0.,0.,1.])
    e=np.cross([0.,1.,0.],b);e/=np.linalg.norm(e)
    v=np.sqrt(10.)*(.6*b+.8*e);omega=2e6 if field=='helix' else 2048.
    return jnp.asarray(np.r_[-np.cross(v,b)/omega,v])


def solve(field,particles,seed,*,steps=400,phase=.1,save_points=6):
    """Three coupled controls: base, finer orbit, finer orbit+collision time."""
    h=.025/((save_points-1)*steps);state=jnp.concatenate((birth(field),jnp.eye(3).ravel()))
    initial=jnp.broadcast_to(state,(particles,15));key=jax.random.PRNGKey(seed)
    # The same fine Cartesian Brownian increments drive all reference levels.
    def run():
        def segment(states,segment_index):
            def advance(i,states):
                k=jax.random.fold_in(key,segment_index*steps+i)
                left,right=jax.random.normal(k,(2,particles,3),dtype=jnp.float64)*jnp.sqrt(h/2)
                a=collision(orbit(states[0],h,field,phase),h,left+right)
                b=collision(orbit(states[1],h,field,phase/2),h,left+right)
                c=collision(orbit(states[2],h/2,field,phase/2),h/2,left)
                c=collision(orbit(c,h/2,field,phase/2),h/2,right)
                return jnp.stack((a,b,c))
            states=jax.lax.fori_loop(0,steps,advance,states)
            return states,states
        start=jnp.broadcast_to(initial,(3,particles,15))
        _,saved=jax.lax.scan(segment,start,jnp.arange(save_points-1))
        return jnp.concatenate((start[None],saved),axis=0)
    states=np.asarray(jax.jit(run)())
    if not np.isfinite(states).all():raise RuntimeError('nonfinite independent Lorentz reference')
    frame=states[...,6:].reshape(states.shape[:-1]+(3,3))
    if np.max(abs(frame@np.swapaxes(frame,-1,-2)-np.eye(3)))>1e-8:raise RuntimeError('reference noise frame lost orthogonality')
    if field=='mirror' and np.max(np.linalg.norm(np.asarray(magnetic(states[...,:3],field)),axis=-1))>4*2048:
        raise RuntimeError('reference mirror gyrofrequency exceeded its subdivision bound')
    return states[...,:6]


def moments(states,field):
    p=ProtonBackground(coulomb_log_model='impact');v=states[...,3:];speed=np.linalg.norm(v,axis=-1)
    b=np.array(magnetic(states[...,:3],field));b/=np.linalg.norm(b,axis=-1,keepdims=True)
    parallel=np.sum(v*b,axis=-1)
    return np.stack((states[...,2]*p.sigma/p.nu,speed*p.sigma,parallel/speed,parallel*p.sigma),axis=-1)


def statistics(values,other=None):
    rows=[];n=values.shape[1]
    for index,sample in enumerate(values):
        row={}
        for k,name in enumerate(('position','speed','pitch','parallel')):
            x=sample[:,k]
            row[name+'_mean']=estimate(x if other is None else x-other[index,:,k],0.)
            row[name+'_variance']=(estimate(x,0.,variance=True) if other is None else
                estimate(n/(n-1)*((x-x.mean())**2-(other[index,:,k]-other[index,:,k].mean())**2),0.))
        covariance=n/(n-1)*(sample[:,0]-sample[:,0].mean())*(sample[:,3]-sample[:,3].mean())
        if other is not None:
            b=other[index];covariance-=n/(n-1)*(b[:,0]-b[:,0].mean())*(b[:,3]-b[:,3].mean())
        row['position_parallel_covariance']=estimate(covariance,0.)
        rows.append(row)
    return rows


def generate(fields,particles,*,steps=400,phase=.1,seed=20261028,output=None,save_points=6):
    steps=max(1,int(np.ceil(steps*5/(save_points-1))))
    report=dict(particles=particles,times=np.linspace(0.,.025,save_points).tolist(),seed=seed,steps_per_save=steps,phase=phase,
        scope='Independent full-Lorentz numerical Monte Carlo reference; Cartesian Ito EM in a transported orthogonal noise frame and centred exact-angle magnetic rotation, with orbit/collision refinements. Not an analytic solution.',fields={})
    for field in fields:
        print('Independent Lorentz reference',field,'N=',particles,'steps/save=',steps,flush=True)
        start=perf_counter();states=solve(field,particles,seed+(field=='helix'),steps=steps,phase=phase,save_points=save_points)
        values=moments(states,field);levels=[]
        for i in range(3):levels.append(dict(collision_interval=.005/steps/(2 if i==2 else 1),
            phase=phase/(2 if i else 1),evolution=statistics(values[:,i])))
        report['fields'][field]=dict(elapsed_s=perf_counter()-start,levels=levels,
            paired_changes=dict(orbit=statistics(values[:,0],values[:,1]),collision=statistics(values[:,1],values[:,2])))
        if output is not None:Path(output).write_text(json.dumps(report,indent=2)+'\n')
        print('Completed reference',field,report['fields'][field]['elapsed_s'],'s',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields',nargs='+',choices=('mirror','helix'),default=['mirror','helix'])
    parser.add_argument('--particles',type=int,default=2000);parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--phase',type=float,default=.1);parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--save-points',type=int,default=6)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.particles<2 or args.steps<1 or args.phase<=0 or args.save_points<2:parser.error('positive resolution and at least two particles required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    generate(args.fields,args.particles,steps=args.steps,phase=args.phase,seed=args.seed,output=args.output,save_points=args.save_points)


if __name__=='__main__':main()
