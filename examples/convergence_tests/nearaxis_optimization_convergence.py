"""Convergence study for near-axis coil optimization.

Sweeps the function-evaluation budget for loss_coils_for_nearaxis and records
wall time, final loss, and the on-axis B / grad-B errors at each budget.

Run from the repository root:
    python examples/convergence_tests/nearaxis_optimization_convergence.py

Uses reduced parameters so each point completes in seconds.
"""
# Full-parameter near-axis coil optimization + convergence study
import matplotlib; matplotlib.use('Agg')
from time import time
import jax.numpy as jnp, matplotlib.pyplot as plt, json
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_coils_for_nearaxis, difference_B_gradB_onaxis

max_coil_length=4; max_coil_curvature=6; order=5
nseg=order*10; ncoils=3; nfp=3; tol=1e-8
field = near_axis(rc=jnp.array([1,0.045]), zs=jnp.array([0,-0.045]), etabar=-0.9, nfp=nfp)
cur = 17e5*field.B0/nfp/2

def make_coils():
    c = CreateEquallySpacedCurves(n_curves=ncoils, order=order, R=field.R0[0],
        r=field.R0[0]/2.0, n_segments=nseg, nfp=nfp, stellsym=True)
    return Coils(curves=c, currents=[cur]*ncoils)

def eval_loss(coils):
    return float(loss_coils_for_nearaxis(coils.x, field, coils.dofs_curves,
        coils.currents_scale, nfp, max_coil_length=max_coil_length,
        n_segments=nseg, stellsym=True, max_coil_curvature=max_coil_curvature))

results={}
for nfev in [10,20,50,100,200]:
    ci = make_coils(); lb = eval_loss(ci)
    t0=time()
    co = optimize_loss_function(loss_coils_for_nearaxis, initial_dofs=ci.x, coils=ci,
        tolerance_optimization=tol, maximum_function_evaluations=nfev, field_nearaxis=field,
        max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature)
    el=time()-t0; la=eval_loss(co)
    Bd,gBd = difference_B_gradB_onaxis(field, BiotSavart(co))
    results[nfev]=dict(time=el, loss_initial=lb, loss_final=la,
        reduction=lb/la, B_err=float(jnp.sum(jnp.abs(Bd))), gradB_err=float(jnp.sum(jnp.abs(gBd))))
    print(f'nfev={nfev:4d}  {el:7.2f}s  loss {lb:.4f} -> {la:.4f}  ({lb/la:.2f}x)  '
          f'B_err={results[nfev]["B_err"]:.4f}  gradB_err={results[nfev]["gradB_err"]:.4f}', flush=True)
    if nfev==200: ci_final, co_final = ci, co

json.dump(results, open('nearaxis_results.json','w'), indent=1)

fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
ks=sorted(results); a1.plot(ks,[results[k]['loss_final'] for k in ks],'o-')
a1.set_xlabel('function evaluations'); a1.set_ylabel('final loss'); a1.set_yscale('log')
a1.set_title('Near-axis coil optimization convergence'); a1.grid(alpha=.3)
a2.plot(ks,[results[k]['time'] for k in ks],'s-',color='darkred')
a2.set_xlabel('function evaluations'); a2.set_ylabel('wall time (s)')
a2.set_title('Optimization cost'); a2.grid(alpha=.3)
plt.tight_layout(); plt.savefig('nearaxis_convergence.png',dpi=200)

fig=plt.figure(figsize=(11,5))
b1=fig.add_subplot(121,projection='3d'); b2=fig.add_subplot(122,projection='3d')
ci_final.plot(ax=b1,show=False); field.plot(ax=b1,show=False,alpha=0.25)
co_final.plot(ax=b2,show=False); field.plot(ax=b2,show=False,alpha=0.25)
b1.set_title(f'Initial coils (loss={results[200]["loss_initial"]:.3g})')
b2.set_title(f'Optimized, 200 nfev (loss={results[200]["loss_final"]:.3g})')
plt.tight_layout(); plt.savefig('nearaxis_coils_200.png',dpi=200)
print('saved plots + nearaxis_results.json')
