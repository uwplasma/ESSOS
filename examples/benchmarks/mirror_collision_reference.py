"""Independent radial/Legendre Coulomb generator and exact axial GC transport."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh_tridiagonal
from .thermal_relaxation_reference import RelaxationReference,impact_coefficients


def interpolation_weights(grid,points):
    """Six-point local polynomial interpolation on a nonuniform grid."""
    start=np.clip(np.searchsorted(grid,points)-3,0,len(grid)-6)
    indices=start[...,None]+np.arange(6)
    nodes=grid[indices];weights=np.ones_like(nodes)
    for i in range(6):
        for j in range(6):
            if i!=j:weights[...,i]*=(points-nodes[...,j])/(nodes[...,i]-nodes[...,j])
    return indices.astype(np.int32),weights

@dataclass
class MirrorCollisionReference:
    length: float=.03
    nz: int=81
    nr: int=48
    nxi: int=49
    zmax: float=.8
    upper: float=6.
    lower: float=.05*np.sqrt(2.)

    def __post_init__(self):
        if min(self.nz,self.nr,self.nxi)<8 or self.length<=0 or self.zmax<=0:raise ValueError('invalid reference grid')
        self.radial=RelaxationReference(lower=self.lower,upper=self.upper,nodes=self.nr,law=impact_coefficients)
        self.r=self.radial.grid
        self.z=self.length*np.sinh(np.linspace(-np.arcsinh(self.zmax/self.length),np.arcsinh(self.zmax/self.length),self.nz))
        self.xi,weights=np.polynomial.legendre.leggauss(self.nxi)
        ell=np.arange(self.nxi);self.angular_root=np.sqrt(weights)
        self.angular=np.polynomial.legendre.legvander(self.xi,self.nxi-1)*self.angular_root[:,None]*np.sqrt((2*ell+1)/2)
        self.killing=ell*(ell+1)/2

    def collision_matrices(self,h):
        _,_,nu=self.radial.law(self.r);root=np.sqrt(self.radial.mass)
        half,whole=[],[]
        for rate in self.killing:
            values,vectors=eigh_tridiagonal(self.radial.diagonal-rate*nu,self.radial.off)
            if rate==0:values[-1]=0.
            half.append(((vectors*np.exp(.5*h*values))@vectors.T)*root[None,:]/root[:,None])
            whole.append(((vectors*np.exp(h*values))@vectors.T)*root[None,:]/root[:,None])
        return np.asarray(half),np.asarray(whole)

    def collide(self,f,matrices):
        nz,nr,nxi,observables=f.shape
        weighted=f*self.angular_root[None,None,:,None]
        modes=(weighted.transpose(0,1,3,2).reshape(-1,nxi)@self.angular).reshape(nz,nr,observables,nxi)
        for ell in range(nxi):
            value=modes[:,:,:,ell].transpose(0,2,1).reshape(-1,nr)@matrices[ell].T
            modes[:,:,:,ell]=value.reshape(nz,observables,nr).transpose(0,2,1)
        result=(modes.reshape(-1,nxi)@self.angular.T).reshape(nz,nr,observables,nxi).transpose(0,1,3,2)
        return result/self.angular_root[None,None,:,None]

    def characteristics(self,h):
        z,r,xi=np.meshgrid(self.z,self.r,self.xi,indexing='ij')
        w=r*np.sqrt(1-xi*xi)/np.sqrt(self.length**2+z*z)
        sine=h*np.sinc(w*h/np.pi);cosine=np.cos(w*h)
        new_z=z*cosine+r*xi*sine
        new_xi=xi*cosine-z*w*w*sine/r
        # No birth can reach the remote z walls within the reporting horizon;
        # the grid edge treatment is checked independently by domain refinement.
        iz,wz=interpolation_weights(self.z,np.clip(new_z,self.z[0],self.z[-1]))
        ix,wx=interpolation_weights(self.xi,np.clip(new_xi,-1.,1.))
        return iz,wz,ix,wx

    def transport(self,f,mapping):
        iz,wz,ix,wx=mapping;radial=np.arange(self.nr)[None,:,None]
        out=np.zeros_like(f)
        for a in range(6):
            for b in range(6):out+=(wz[...,a]*wx[...,b])[...,None]*f[iz[...,a],radial,ix[...,b],:]
        return out

    def evaluate(self,f,z=0.,r=np.sqrt(10.),xi=.6):
        iz,wz=interpolation_weights(self.z,np.asarray(z));ir,wr=interpolation_weights(self.r,np.asarray(r));ix,wx=interpolation_weights(self.xi,np.asarray(xi))
        return np.einsum('i,j,k,ijko->o',wz,wr,wx,f[iz[:,None,None],ir[None,:,None],ix[None,None,:],:])

    def moments(self,times,*,steps_per_save=8,collisions=True,transport=True):
        times=np.asarray(times,float)
        if times[0]!=0 or len(times)<2 or not np.allclose(np.diff(times),times[-1]/(len(times)-1)):raise ValueError('uniform times starting at zero required')
        if self.zmax<=self.upper*times[-1]:raise ValueError('z domain can be reached from the birth; increase zmax')
        z,r,xi=np.meshgrid(self.z,self.r,self.xi,indexing='ij')
        f=np.stack((z,z*z,xi,xi*xi,z*xi),axis=-1)
        h=times[-1]/((len(times)-1)*steps_per_save)
        half,whole=self.collision_matrices(h) if collisions else (None,None)
        mapping=self.characteristics(h) if transport else None
        values=[np.array([0.,0.,.6,.36,0.])]
        current=self.collide(f,half) if collisions else f
        steps=(len(times)-1)*steps_per_save
        for step in range(1,steps+1):
            if transport:current=self.transport(current,mapping)
            if step%steps_per_save==0:
                value=self.collide(current,half) if collisions else current
                values.append(self.evaluate(value))
            if step<steps and collisions:current=self.collide(current,whole)
        z,z2,xi,xi2,zxi=np.asarray(values).T
        if min(np.min(z2-z*z),np.min(xi2-xi*xi))<-1e-3:raise RuntimeError('negative reference variance; refine grid/time')
        return dict(position_mean=z,position_variance=z2-z*z,pitch_mean=xi,
            pitch_variance=xi2-xi*xi,position_pitch_covariance=zxi-z*xi)
