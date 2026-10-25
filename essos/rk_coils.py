"""Private padded Biot-Savart view; public field storage and equations are unchanged."""
import jax.numpy as jnp
from .rk_gpu_utils import _select_one

def cross(a,b):
    return a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def pack(values):
    result = jnp.zeros((len(values),),values[0].dtype)
    index = jnp.arange(len(values))
    for i,value in enumerate(values):
        result = jnp.where(index==i,value,result)
    return result


def geometry(field):
    gamma,tangent,current = field.coils.gamma,field.coils.gamma_dash,field.coils.currents
    nc,nq,_ = gamma.shape
    pc,pq = (1<<(size-1).bit_length() for size in (nc,nq))
    weight = current*1e-7
    arrays = tuple(jnp.pad(value[:,:,i].T,((0,pq-nq),(0,pc-nc))).reshape(-1)
        for value in (gamma,tangent*weight[:,None,None]) for i in range(3))
    return arrays,nc,nq


def field_values(refs,point,nc,nq,jacobian):
    gamma,tangent = tuple(ref[...] for ref in refs[:3]),tuple(ref[...] for ref in refs[3:6])
    pc = 1<<(nc-1).bit_length()
    index = jnp.arange(gamma[0].size)
    mask = (index//pc<nq)&(index%pc<nc)
    r = tuple(point[i]-gamma[i] for i in range(3))
    radius2 = jnp.where(mask,dot(r,r),1.)
    inverse3 = 1./jnp.sqrt(radius2)**3
    contribution = tuple(jnp.where(mask,value*inverse3,0.) for value in cross(tangent,r))
    mean = lambda value:jnp.sum(value)/nq
    B = tuple(mean(value) for value in contribution)
    if not jacobian:
        return B,None
    tx,ty,tz = (mean(jnp.where(mask,value*inverse3,0.)) for value in tangent)
    zero = jnp.asarray(0.,point[0].dtype)
    skew = ((zero,-tz,ty),(tz,zero,-tx),(-ty,tx,zero))
    J = tuple(tuple(skew[i][j]-3*mean(contribution[i]*r[j]/radius2)
                    for j in range(3)) for i in range(3))
    return B,J


def rhs_values(refs,y,nc,nq,model,energy,mass,charge):
    scalar = lambda i:_select_one(y,i,0).reshape(())
    point = tuple(scalar(i) for i in range(3))
    B,J = field_values(refs,point,nc,nq,model=='gc')
    if model=='lorentz':
        velocity = tuple(scalar(i) for i in range(3,6))
        force = cross(velocity,B)
        return pack((*velocity,*(charge/mass*value for value in force),
                          jnp.asarray(0.,y.dtype),jnp.asarray(0.,y.dtype)))
    magnitude = jnp.sqrt(dot(B,B))
    grad = tuple(sum(J[i][j]*B[i]/magnitude for i in range(3)) for j in range(3))
    curl = (J[2][1]-J[1][2],J[0][2]-J[2][0],J[1][0]-J[0][1])
    bg = cross(B,grad)
    curl_b = tuple(curl[i]/magnitude+bg[i]/magnitude**2 for i in range(3))
    bx = cross(B,curl_b)
    curvature = tuple(-value/magnitude for value in bx)
    vpar = scalar(3)
    moment = (energy-mass*vpar**2/2)/magnitude
    star = tuple(B[i]+vpar*mass/charge*curl_b[i] for i in range(3))
    force = tuple(moment*grad[i]+mass*vpar**2*curvature[i] for i in range(3))
    denominator = dot(B,star)
    perpendicular = cross(B,force)
    velocity = tuple(vpar*B[i]/magnitude+perpendicular[i]/denominator/charge for i in range(3))
    acceleration = -dot(star,force)/denominator*magnitude/mass
    return pack((*velocity,acceleration))



def prepare_orbit_solver(field,mass,charge,energy,speed,times,dt0,rtol,atol,max_steps,*,
                         model="gc",method="dopri8",adaptive=True,event=None,
                         length_scale=1.,interpret=False):
    from .rk_gpu import prepare_gpu_solver
    if model not in ("gc","lorentz"):
        raise ValueError("model must be gc or lorentz")
    arrays,nc,nq=geometry(field)
    scalar=lambda y,i:_select_one(y,i,0).reshape(())
    def factory(refs,point,invariant):
        evaluate=lambda y:rhs_values(refs,y,nc,nq,model,energy,mass,charge)
        def estimate(endpoint,error,speed):
            position=jnp.sqrt(sum(scalar(error,i)**2 for i in range(3)))/length_scale
            velocity=(jnp.abs(scalar(error,3)) if model=="gc" else
                      jnp.sqrt(sum(scalar(error,i)**2 for i in range(3,6))))/speed
            return jnp.maximum(position,velocity)
        level=(lambda y:scalar(y,0)*0+1.) if event is None else event
        return evaluate,estimate,level
    return prepare_gpu_solver(arrays,factory,speed,times,dt0,rtol,atol,max_steps,
        state_size=4 if model=="gc" else 8,method=method,adaptive=adaptive,num_warps=2,
        interpret=interpret)
