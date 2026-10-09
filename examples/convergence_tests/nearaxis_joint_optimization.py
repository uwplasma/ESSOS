"""Joint coils + near-axis optimization (optimize_coils_and_nearaxis).

Two stages with reduced parameters: coils fitted to a fixed near-axis field,
then coils and near-axis optimized together.

Measured: stage 1 6.2 s, stage 2 6.8 s; iota 0.41831 -> 0.45294,
max elongation 2.413 -> 2.559, on-axis B error 1.354 -> 1.569.
"""
"""Joint coils + near-axis optimization (the second example), reduced params."""
import matplotlib; matplotlib.use('Agg')
from time import time
import jax.numpy as jnp, json
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.optimization import optimize_loss_function
from essos.objective_functions import (loss_coils_for_nearaxis, loss_coils_and_nearaxis,
                                       difference_B_gradB_onaxis)
nfp=3; MCL=4.; MCC=6.; order=5; nseg=order*10; ncoils=3; TOL=1e-8; NFEV=100
f0 = near_axis(rc=jnp.array([1,0.045]), zs=jnp.array([0,-0.045]), etabar=-0.9, nfp=nfp)
cur = 17e5*f0.B0/nfp/2
curves = CreateEquallySpacedCurves(n_curves=ncoils, order=order, R=f0.R0[0],
    r=f0.R0[0]/2.0, n_segments=nseg, nfp=nfp, stellsym=True)
ci = Coils(curves=curves, currents=[cur]*ncoils)

print('stage 1: coils only ...', flush=True)
t0=time()
c1 = optimize_loss_function(loss_coils_for_nearaxis, initial_dofs=ci.x, coils=ci,
    tolerance_optimization=TOL, maximum_function_evaluations=NFEV, field_nearaxis=f0,
    max_coil_length=MCL, max_coil_curvature=MCC)
t1=time()-t0
print(f'  stage 1 done {t1:.2f}s', flush=True)

print('stage 2: joint coils + near-axis ...', flush=True)
x0 = jnp.concatenate((c1.x, f0.x))
t0=time()
res = optimize_loss_function(loss_coils_and_nearaxis, initial_dofs=x0, coils=ci,
    tolerance_optimization=TOL, maximum_function_evaluations=NFEV, field_nearaxis=f0,
    max_coil_length=MCL, max_coil_curvature=MCC)
t2=time()-t0
c2, f1 = res
print(f'  stage 2 done {t2:.2f}s', flush=True)

Bd0,Gd0 = difference_B_gradB_onaxis(f0, BiotSavart(c1))
Bd1,Gd1 = difference_B_gradB_onaxis(f1, BiotSavart(c2))
out = dict(stage1_time=t1, stage2_time=t2,
  iota_initial=float(f0.iota), iota_optimized=float(f1.iota),
  elong_initial=float(max(f0.elongation)), elong_optimized=float(max(f1.elongation)),
  B_err_stage1=float(jnp.sum(jnp.abs(Bd0))), B_err_joint=float(jnp.sum(jnp.abs(Bd1))),
  gradB_err_stage1=float(jnp.sum(jnp.abs(Gd0))), gradB_err_joint=float(jnp.sum(jnp.abs(Gd1))))
print()
print(f"iota            {out['iota_initial']:.5f}  ->  {out['iota_optimized']:.5f}")
print(f"max elongation  {out['elong_initial']:.3f}  ->  {out['elong_optimized']:.3f}")
print(f"B error         {out['B_err_stage1']:.4f}  ->  {out['B_err_joint']:.4f}")
print(f"gradB error     {out['gradB_err_stage1']:.4f}  ->  {out['gradB_err_joint']:.4f}")
json.dump(out, open('joint_results.json','w'), indent=1)
