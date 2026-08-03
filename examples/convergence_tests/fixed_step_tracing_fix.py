"""Fixed-step field-line tracing: compile-time fix.

The fixed-step path in Tracing (model='FieldLine') sets max_steps=10000000000
and does not pass an explicit stepsize controller, which makes XLA build an
enormous graph -- a 2-step warm-up projected ~43 h to compile.

Passing diffrax.ConstantStepSize() explicitly with a realistic max_steps
compiles in seconds instead:

    10 field lines, 18,000 fixed steps:  5.60 s compile, 5.17 s run

Run from the repository root:
    python examples/convergence_tests/fixed_step_tracing_fix.py
"""
"""Verify the fix at the scale that previously projected 43 h, and confirm
max_steps is really the culprit by scanning it."""
import time, jax, jax.numpy as jnp, diffrax, json
from diffrax import diffeqsolve, ODETerm, SaveAt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.dynamics import FieldLine
nfp=3
f0=near_axis(rc=jnp.array([1,0.045]),zs=jnp.array([0,-0.045]),etabar=-0.9,nfp=nfp)
cur=17e5*f0.B0/nfp/2
curves=CreateEquallySpacedCurves(n_curves=3,order=5,R=f0.R0[0],r=f0.R0[0]/2.,
    n_segments=50,nfp=nfp,stellsym=True)
bs=BiotSavart(Coils(curves=curves,currents=[cur]*3))
term=ODETerm(FieldLine)
res={}

def bench(nlines, nsteps, max_steps, tmax=1e-4, nsave=300):
    R0=jnp.linspace(f0.R0[0],1.04*f0.R0[0],nlines)
    y0=jnp.array([R0,jnp.zeros(nlines),jnp.zeros(nlines)]).T
    ts=jnp.linspace(0,tmax,nsave); dt=tmax/nsteps
    def one(ic):
        return diffeqsolve(term,t0=0.,t1=tmax,dt0=dt,y0=ic,solver=diffrax.Dopri5(),
            args=bs,saveat=SaveAt(ts=ts),throw=False,max_steps=max_steps,
            stepsize_controller=diffrax.ConstantStepSize()).ys
    fn=jax.jit(jax.vmap(one))
    t0=time.time(); o=fn(y0); o.block_until_ready(); tc=time.time()-t0
    t0=time.time(); o=fn(y0); o.block_until_ready(); tr=time.time()-t0
    return tc,tr,bool(jnp.all(jnp.isfinite(o)))

print("=== max_steps scan (10 lines, 2000 fixed steps) ===")
for ms in [2048, 4096, 16384, 65536]:
    tc,tr,ok=bench(10,2000,ms)
    print(f"  max_steps={ms:6d}   compile {tc:7.2f}s   run {tr:6.3f}s  ok={ok}",flush=True)
    res[f'ms{ms}']=dict(compile=tc,run=tr,ok=ok)

print("\n=== the scale that previously projected ~43 h: 18,000 fixed steps ===")
tc,tr,ok=bench(10,18000,20000,tmax=1e-4,nsave=300)
print(f"  10 lines, 18k steps, max_steps=20000:  compile {tc:.2f}s  run {tr:.3f}s  ok={ok}",flush=True)
res['18k_steps']=dict(compile=tc,run=tr,ok=ok)
json.dump(res,open('rk4_fix_results.json','w'),indent=1)
