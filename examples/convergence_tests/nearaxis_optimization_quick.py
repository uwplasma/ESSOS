# Small-parameter near-axis coil optimization, per Rogerio's suggestion:
# "start with smaller max times, less num steps, and less particles"
import matplotlib
matplotlib.use('Agg')
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_coils_for_nearaxis

# ---- reduced parameters (upstream example uses 200 fn evals, order 5) ----
max_coil_length = 4
max_coil_curvature = 6
order_Fourier_series_coils = 3          # was 5
number_coil_points = order_Fourier_series_coils*10
maximum_function_evaluations = 20       # was 200
number_coils_per_half_field_period = 3
tolerance_optimization = 1e-6           # was 1e-8

rc=jnp.array([1, 0.045]); zs=jnp.array([0,-0.045]); etabar=-0.9; nfp=3
field = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=nfp)
print(f'near_axis built: iota={field.iota}, B0={field.B0}, R0[0]={field.R0[0]}')

current_on_each_coil = 17e5*field.B0/nfp/2
major_radius_coils = field.R0[0]
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=major_radius_coils/2.0,
                                   n_segments=number_coil_points,
                                   nfp=nfp, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

def eval_loss(coils):
    return loss_coils_for_nearaxis(
        coils.x, field, coils.dofs_curves, coils.currents_scale, nfp,
        max_coil_length=max_coil_length, n_segments=number_coil_points,
        stellsym=True, max_coil_curvature=max_coil_curvature)

loss_before = eval_loss(coils_initial)
print(f'initial loss = {loss_before}')

print(f'Optimizing with {maximum_function_evaluations} function evaluations...')
t0 = time()
coils_optimized = optimize_loss_function(
    loss_coils_for_nearaxis, initial_dofs=coils_initial.x, coils=coils_initial,
    tolerance_optimization=tolerance_optimization,
    maximum_function_evaluations=maximum_function_evaluations, field_nearaxis=field,
    max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature)
elapsed = time()-t0
print(f'OPTIMIZATION COMPLETED in {elapsed:.2f} s')

loss_after = eval_loss(coils_optimized)
print(f'final loss   = {loss_after}')
print(f'loss reduction factor = {loss_before/loss_after:.3f}x')

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121, projection='3d'); ax2 = fig.add_subplot(122, projection='3d')
coils_initial.plot(ax=ax1, show=False);   field.plot(ax=ax1, show=False, alpha=0.2)
coils_optimized.plot(ax=ax2, show=False); field.plot(ax=ax2, show=False, alpha=0.2)
ax1.set_title(f'Initial (loss={loss_before:.4g})'); ax2.set_title(f'Optimized (loss={loss_after:.4g})')
plt.tight_layout(); plt.savefig('nearaxis_opt.png', dpi=150)
print('saved /tmp/essos-work/nearaxis_opt.png')
