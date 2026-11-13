"""Independent coupled axial GC moment references for retained benchmarks."""
import json
from time import perf_counter
import numpy as np
from .mirror_collision_reference import MirrorCollisionReference,interpolation_weights

VARIABLES=('position','speed','pitch','parallel')
PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


def observables(z,r,xi):
    p=r*xi
    return np.stack((np.ones_like(z),z,r,xi,p,z*z,r*r,xi*xi,p*p,z*r,z*xi,z*p,r*p,xi*p),axis=-1)

def moments(raw):
    raw=np.asarray(raw);mean=raw[...,1:5]
    covariance=np.zeros(mean.shape[:-1]+(4,4))
    for i in range(4):covariance[...,i,i]=raw[...,5+i]-mean[...,i]**2
    for (i,j),index in zip(PAIRS,(9,10,11,4,12,13)):
        covariance[...,i,j]=covariance[...,j,i]=raw[...,index]-mean[...,i]*mean[...,j]
    return dict(mean=mean.tolist(),covariance=covariance.tolist(),mass=raw[...,0].tolist())

def reference(field,times,grid):
    """Same independently coded generator, more joint observables."""
    if field=='mirror':
        nz,nr,nxi,steps=grid
        ref=MirrorCollisionReference(nz=nz,nr=nr,nxi=nxi)
        z,r,xi=np.meshgrid(ref.z,ref.r,ref.xi,indexing='ij')
        f=observables(z,r,xi)
    else:
        nr,nxi,steps=grid
        ref=MirrorCollisionReference(nz=8,nr=nr,nxi=nxi)
        r,xi=np.meshgrid(ref.r,ref.xi,indexing='ij')
        # Exact polynomial transport in z, retaining coefficients of z**0/1/2.
        f=np.zeros((3,nr,nxi,14));f[0]=observables(np.zeros_like(r),r,xi)
        f[1,...,1]=1.;f[2,...,5]=1.
        f[1,...,9]=r;f[1,...,10]=xi;f[1,...,11]=r*xi
        u=.5*r*xi/(1-.75*20000*r*xi/2000000)
        ir,wr=interpolation_weights(ref.r,np.asarray(np.sqrt(10.)))
        ix,wx=interpolation_weights(ref.xi,np.asarray(.6))
    h=times[-1]/((len(times)-1)*steps)
    half,whole=ref.collision_matrices(h)
    mapping=ref.characteristics(h) if field=='mirror' else None
    values=[observables(np.asarray(0.),np.asarray(np.sqrt(10.)),np.asarray(.6))]
    current=ref.collide(f,half)
    for k in range(1,(len(times)-1)*steps+1):
        if field=='mirror':current=ref.transport(current,mapping)
        else:
            old=current.copy();shift=(h*u)[...,None]
            current[0]=old[0]+shift*old[1]+shift*shift*old[2]
            current[1]=old[1]+2*shift*old[2]
        if k%steps==0:
            value=ref.collide(current,half)
            values.append(ref.evaluate(value) if field=='mirror' else np.einsum('i,j,ijo->o',wr,wx,value[0,ir[:,None],ix[None,:]]))
        if k<(len(times)-1)*steps:current=ref.collide(current,whole)
    answer=moments(values)
    assert np.max(abs(np.asarray(answer['mass'])-1))<1e-9
    assert np.min(np.diagonal(np.asarray(answer['covariance'])[1:],axis1=-2,axis2=-1))>-1e-8
    return answer

def references(options):
    report=dict(times=np.asarray(getattr(options,'times',np.linspace(0.,.025,6))).tolist(),variables=VARIABLES,
        scope='Independent backward Fokker-Planck generator for coupled z/speed/pitch motion, plus parallel velocity. Helix uses exact polynomial-in-z transport. No native tracer or stochastic trajectories.',fields={})
    for field in options.fields:
        grids=([(65,32,33,4),(97,48,49,8),(145,64,65,16),(97,48,49,16)] if field=='mirror' else
               [(128,12,16),(256,16,32),(384,24,64),(256,16,64)])
        grids=[(*grid[:-1],max(1,int(np.ceil(grid[-1]*5/(len(report['times'])-1))))) for grid in grids]
        grids[-1]=(*grids[-1][:-1],max(grids[-1][-1],2*grids[1][-1]))
        rows=[]
        for grid in grids:
            print('Coupled physical reference',field,grid,flush=True)
            start=perf_counter();value=reference(field,np.asarray(report['times']),grid)
            rows.append(dict(grid=grid,seconds=perf_counter()-start,**value))
            report['fields'][field]=dict(levels=rows)
            (options.output/'reference.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print('Reference finished',field,grid,rows[-1]['seconds'],flush=True)
    return report

def uncertainty(levels):
    fine,medium,temporal=levels[2],levels[1],levels[3]
    return {name:np.maximum(abs(np.asarray(fine[name])-medium[name]),abs(np.asarray(temporal[name])-medium[name])) for name in ('mean','covariance')}

def add_speed_reference(options,data):
    """Use the better-resolved independent radial marginal for speed moments.

    Six-point evaluation preserves the exact monoenergetic birth moments;
    two-point evaluation otherwise introduces a small initial mesh variance.
    Only smooth moments use these signed interpolation weights, not CDFs.
    """
    if 'speed_reference' in data:return
    from .thermal_relaxation_reference import RelaxationReference,impact_coefficients
    levels=[]
    for nodes in (768,1536):
        ref=RelaxationReference(lower=.05*np.sqrt(2.),nodes=nodes,law=impact_coefficients)
        index,weight=interpolation_weights(ref.grid,np.asarray(np.sqrt(10.)))
        ref.evaluate[:]=0.;ref.evaluate[index]=weight;ref._spectra={}
        birth=np.dot(ref.evaluate,ref.grid**2)-np.dot(ref.evaluate,ref.grid)**2
        assert abs(birth)<1e-12
        value=ref.moments(np.asarray(data['times']))
        levels.append(dict(nodes=nodes,initial_variance=float(birth),speed_mean=value['speed_mean'].tolist(),speed_variance=value['speed_variance'].tolist()))
    data['speed_reference']=dict(levels=levels,scope='Independent radial finite-volume generator; high-order evaluation of smooth moments at the exact birth. The magnetic field does no work in either benchmark.')
    (options.output/'reference.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def physical_moments(data,field):
    levels=data['fields'][field]['levels'];ref=levels[2]
    mu=np.asarray(ref['mean']).copy();cov=np.asarray(ref['covariance']).copy()
    error=uncertainty(levels)
    a,b=data['speed_reference']['levels']
    mu[:,1]=b['speed_mean'];cov[:,1,1]=b['speed_variance']
    error['mean'][:,1]=abs(np.asarray(b['speed_mean'])-a['speed_mean'])
    error['covariance'][:,1,1]=abs(np.asarray(b['speed_variance'])-a['speed_variance'])
    return mu,cov,error
