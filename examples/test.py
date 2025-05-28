import jax
import jax.numpy as jnp
import diffrax as dfx
import matplotlib as plt

q = 1.6e-19 #Charge
m = 1.21e-21 #Mass
V0 = jnp.array([1.,1.,1.]) #Intial Velocity
r0 = jnp.array ([1.,1.,1.]) #Initial Position

@jit
def magnetic_field(r):
    x,y,z = r
    B0=1.
    Bx = 0.
    By = 0.
    Bz = jnp.sin(y)
    return B0*jnp.array([Bx, By, Bz])
#Mag Field Can be modified for any B as a function of position

@jit
def vector_field(t, y, args):
    q, m = args
    x,y,z,vx,vy,vz=y
    def magnetic_field(r):
        xx,yy,zz = r
        B0=1.
        Bx = 0.
        By = 0.
        Bz = jnp.sin(yy)
        return B0*jnp.array([Bx, By, Bz])
    points=jnp.array([x,y,z])
    vpoints=jnp.array([vx,vy,vz])
    B = magnetic_field(points)
    d_V = (jnp.cross(vpoints,B))*(q/m)
    d_r = vpoints
    return jnp.concatenate([d_r, d_V])

y0 = jnp.concatenate([r0, V0])

term = dfx.ODETerm(vector_field)
solver = dfx.Tsit5()
t0 = 0
t1 = 1 
dt0 = 0.1
y0 = y0

saveat = dfx.SaveAt(ts = jnp.linspace(t0, t1, 1000))
sol = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args = (q, m), saveat=saveat)





