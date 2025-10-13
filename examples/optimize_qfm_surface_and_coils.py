import os
number_of_processors_to_use = 1 
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time
from jax import device_get

from essos.surfaces import BdotN_over_B, toroidal_flux
from essos.surfaces import SurfaceRZFourier
from essos.qfm import QfmSurface 
from essos.fields import Vmec, BiotSavart
from essos.qfm import QfmSurface_with_coils
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.dynamics import Particles
from essos.objective_functions import loss_particle_r_cross_max_constraint,loss_particle_gamma_c
from essos.objective_functions import loss_coil_curvature,loss_coil_length,loss_normB_axis_average,loss_Br,loss_iota
from functools import partial




# Optimization parameters
target_B_on_axis = 5.7
max_coil_length = 31
max_coil_curvature = 0.4
nparticles = number_of_processors_to_use*5
order_Fourier_series_coils = 2
number_coil_points = 60
maximum_function_evaluations = 1
maxtimes = [1.e-5]
num_steps=100
number_coils_per_half_field_period = 3
number_of_field_periods = 2
model = 'GuidingCenter'

# Initialize coils
current_on_each_coil = 1.84e7
major_radius_coils = 10.0
minor_radius_coils = 4.45
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)


len_dofs_curves = len(jnp.ravel(coils_initial.dofs_curves))
nfp = coils_initial.nfp
stellsym = coils_initial.stellsym
n_segments = coils_initial.n_segments
dofs_curves_shape = coils_initial.dofs_curves.shape
currents_scale = coils_initial.currents_scale


# Initialize particles
phi_array = jnp.linspace(0, 2*jnp.pi, nparticles)
initial_xyz=jnp.array([major_radius_coils*jnp.cos(phi_array), major_radius_coils*jnp.sin(phi_array), 0*phi_array]).T
particles = Particles(initial_xyz=initial_xyz)

t=maxtimes[0]


loss_partial = partial(loss_particle_gamma_c,particles=particles, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,maxtime=t,model=model,num_steps=num_steps)
curvature_partial=partial(loss_coil_curvature, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,max_coil_curvature=max_coil_curvature)
length_partial=partial(loss_coil_length, dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,max_coil_length=max_coil_length)
Baxis_average_partial=partial(loss_normB_axis_average,dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,npoints=15,target_B_on_axis=target_B_on_axis)
r_max_partial = partial(loss_particle_r_cross_max_constraint,target_r=0.4, particles=particles,dofs_curves=coils_initial.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym,maxtime=t,model=model,num_steps=num_steps)




# Load initial guess surface
ntheta=32
nphi=32
mpol=2
ntor=2
vmec = os.path.join('input_files','input.toroidal_surface')
surf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True)
surf.change_resolution(mpol,ntor)

initialsurf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True)
initialsurf.change_resolution(mpol,ntor)

# Load target VMEC surface
truevmec = Vmec(os.path.join(os.path.dirname(__name__), 'input_files', 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
                ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=1.0)




# QFM optimization setup
method = 'alm' # lbfgs, slsqp, alm
label = 'multi'  # 'area', 'volume', 'toroidal_flux'

if method == 'lbfgs':
    tol = 1e-4
elif method == 'slsqp':
    tol = 1e-6
elif method == 'alm':
    tol = 1e-6

maxiter = 20000
constraint_weight = 1e-3


initial_label_flux = toroidal_flux(surf,  BiotSavart(coils_initial))
targetlabel_flux = toroidal_flux(truevmec.surface,  BiotSavart(coils_initial))
targetlabel_flux_final = toroidal_flux(truevmec.surface,  BiotSavart(coils_initial),idx=-1)


initial_label_volume = surf.volume
targetlabel_volume = truevmec.surface.volume

initial_label_area = surf.area
targetlabel_area = truevmec.surface.area




BdotN_over_B_initial = BdotN_over_B(surf, BiotSavart(coils_initial))

# Initialize QFM optimizer
qfm = QfmSurface_with_coils(coils=coils_initial, surface=surf,  targetlabel_flux=targetlabel_flux,targetlabel_flux_final=targetlabel_flux_final,targetlabel_area=targetlabel_area,targetlabel_volume=targetlabel_volume,label=label,coil_loss=loss_partial,coil_constraint=[curvature_partial,length_partial,Baxis_average_partial])

print("Degrees of Freedom:", qfm.surface.x.shape[0]+qfm.coils.x.shape[0])
start_time = time() 
print('start')


result = qfm.run(
    tol=tol,
    maxiter=maxiter,
    method=method,
    constraint_weight=constraint_weight,
    log_every=10  
)

print('done')
end_time = time()

# Evaluate final objective and constraint
x_surf_opt = device_get(result["s"].x)
x_coils_opt = device_get(result["c"].x)
qfm_loss = float(jnp.asarray(qfm.objective(x_opt)))
c_loss = float(jnp.asarray(qfm.constraint_flux(x_opt))+jnp.asarray(qfm.constraint_area(x_opt))+jnp.asarray(qfm.constraint_volume(x_opt)))

BdotN_over_B_optimized = BdotN_over_B(result['s'], BiotSavart(result['c']))
print("Optimization method:", method)
print("Optimization label:", label)
print("Optimization success:", result['success'])
print(f"final qfm objective = {qfm_loss:.3e}, final constraint objective = {c_loss:.3e}")
print("Iterations:", result['iter'])
print(f"Optimization time: {end_time - start_time}")

print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")

initial_area = surf.area
initial_volume = surf.volume
initial_tf = toroidal_flux(surf, field)

final_area = result['s'].area
final_volume = result['s'].volume
final_tf = toroidal_flux(result['s'], field)

print(f"Initial labels -> area: {initial_area:.6e}, volume: {initial_volume:.6e}, toroidal_flux: {initial_tf:.6e}")
print(f"target label: {label}   target label value: {targetlabel_area:.6e}, {targetlabel_volume:.6e}, {targetlabel_flux:.6e}")
print(f"Final labels   -> area: {final_area:.6e}, volume: {final_volume:.6e}, toroidal_flux: {final_tf:.6e}")


# Plot surfaces
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# coils.plot(ax=ax1, show=False)
# coils.plot(ax=ax2, show=False)
# coils.plot(ax=ax3, show=False)


initialsurf.plot(ax=ax1, show=False)
#surf.plot(ax=ax2, show=False)
truevmec.surface.plot(ax=ax2, show=False)
result['s'].plot(ax=ax3, show=False)

ax1.set_title("Initial Surface")
ax2.set_title("True VMEC Surface")
ax3.set_title("Final Surface")

plt.tight_layout()
plt.savefig('optimize_qfm_surface.png', dpi=300)


# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing

tmax = 100000000000
nfieldlines_per_core = 13
nfieldlines = nfieldlines_per_core * number_of_processors_to_use
R0 = jnp.linspace(12.2, 13.5, nfieldlines)
trace_tolerance = 1e-7
num_steps = 60000

Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T

time0 = time()
tracing = block_until_ready(Tracing(
    field=field,
    model='FieldLineAdaptative',
    initial_conditions=initial_xyz,
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance
))
print(f"ESSOS tracing took {time() - time0:.2f} seconds")

trajectories = tracing.trajectories
traj = trajectories[0]
R, phi, Z = traj[:, 0], traj[:, 1], traj[:, 2]

phi_u = jnp.unwrap(phi)
phi0_cross = jnp.where((phi_u[:-1] < 0) & (phi_u[1:] >= 0))[0]
phi90_cross = jnp.where((phi_u[:-1] < jnp.pi / 2) & (phi_u[1:] >= jnp.pi / 2))[0]

theta = jnp.linspace(0, 2 * jnp.pi, 200)

def compute_rz_on_phi(surface, theta, phi=0.0):
    angles = jnp.outer(theta, surface.xm) - phi * surface.xn
    R = jnp.sum(surface.rmnc_interp * jnp.cos(angles), axis=1)
    Z = jnp.sum(surface.zmns_interp * jnp.sin(angles), axis=1)
    return R, Z

# # Contours from optimized surface
R0_opt, Z0_opt = compute_rz_on_phi(result['s'], theta, phi=0.0)
R90_opt, Z90_opt = compute_rz_on_phi(result['s'], theta, phi=jnp.pi/2)

# # Contours from true VMEC surface
R0_true, Z0_true = compute_rz_on_phi(truevmec.surface, theta, phi=0.0)
R90_true, Z90_true = compute_rz_on_phi(truevmec.surface, theta, phi=jnp.pi/2)

fig, ax = plt.subplots(figsize=(6, 6))

tracing.poincare_plot(ax=ax, show=False, shifts=[0, jnp.pi / 2])
ax.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 0$")
ax.plot(R90_opt, Z90_opt, color='black', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2$")
ax.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
ax.plot(R90_true, Z90_true, color='blue', linestyle='--', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")

ax.set_xlabel("R")
ax.set_ylabel("Z")
ax.set_title("Poincaré + Surfaces Comparison @ φ = 0 and π/2")
ax.legend()
ax.axis("equal")
plt.tight_layout()
plt.savefig('optimize_qfm_surface_poincare.png', dpi=300)