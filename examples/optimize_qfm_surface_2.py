import os
number_of_processors_to_use = 2 
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time
from jax import device_get

from essos.surfaces import BdotN_over_B, toroidal_flux,poloidal_flux
from essos.surfaces import SurfaceRZFourier
from essos.qfm import QfmSurface 
from essos.fields import Vmec, BiotSavart

# Load initial guess surface
ntheta=60
nphi=60
mpol=5
ntor=5

# # Load coils and field
# json_file = os.path.join(os.path.dirname(__name__), 'input_files', 'QH_simple_scaled.json')
# coils = Coils_from_simsopt(json_file,nfp=4)
# field = BiotSavart(coils)


# # Load coils and field
# wout_file = os.path.join(os.path.dirname(__name__), 'input_files','wout_QH_simple_scaled.nc')
# vmec = Vmec(wout_file)

#vmec = os.path.join('input_files','input.toroidal_surface_nfp4')
#surf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,rescaling_type='L_infty',rescaling_factor=1.2)
#surf.change_resolution(mpol,ntor)

#initialsurf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True)
#initialsurf.change_resolution(mpol,ntor)


filename='wout_LandremanPaul2021_QA_reactorScale_lowres.nc'
filename_vmec=os.path.join('input_files','input.toroidal_surface')
#filename='wout_QH_simple_scaled.nc'
filename_coils='QH_simple_scaled.json'
# Load target VMEC surface
#truevmec = Vmec(os.path.join(os.path.dirname(__name__), 'input_files', 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
#                ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=1.0)
truevmec = Vmec(os.path.join(os.path.dirname(__name__), 'input_files', filename),ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=1.0)
vmec_s1p0 = Vmec(os.path.join(os.path.dirname(__name__), 'input_files', filename),ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=1.0)

#surf_to_opt=Vmec(os.path.join(os.path.dirname(__name__), 'input_files', filename),ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=1.0)

surf = SurfaceRZFourier(vmec=vmec_s1p0,s=1.0, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,rescaling_type='L_infty',rescaling_factor=1.2)
surf.change_resolution(3,1)
surf.change_resolution(mpol,ntor)

initialsurf = SurfaceRZFourier(vmec=vmec_s1p0,s=1.0, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,rescaling_type='L_infty',rescaling_factor=1.2)
initialsurf.change_resolution(3,1)
initialsurf.change_resolution(mpol,ntor)

#surf = surf_to_opt.surface
#surf.change_resolution(3,3)
#surf.change_resolution(mpol,ntor)



#initial=Vmec(os.path.join(os.path.dirname(__name__), 'input_files', filename),
#                ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=1.0)
#initialsurf = initial.surface
#initialsurf.change_resolution(2,2)

##surf = SurfaceRZFourier(filename_vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,rescaling_type='L_infty',rescaling_factor=1.2)
##surf.change_resolution(mpol,ntor)

##initialsurf = SurfaceRZFourier(filename_vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True)
##initialsurf.change_resolution(mpol,ntor)

vmec_s0p98=Vmec(os.path.join(os.path.dirname(__name__), 'input_files', filename),ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=0.98)
truevmec_2 = Vmec(os.path.join(os.path.dirname(__name__), 'input_files', filename),ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=0.98)


surf_2 = SurfaceRZFourier(vmec=vmec_s0p98,s=0.98, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,rescaling_type='L_infty',rescaling_factor=1.2)
surf_2.change_resolution(3,1)
surf_2.change_resolution(mpol,ntor)

initialsurf_2 = SurfaceRZFourier(vmec=vmec_s0p98,s=0.98, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,rescaling_type='L_infty',rescaling_factor=1.2)
initialsurf_2.change_resolution(3,1)
initialsurf_2.change_resolution(mpol,ntor)
#surf_to_opt_2=Vmec(os.path.join(os.path.dirname(__name__), 'input_files', filename),ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=0.98)

#surf_2 = surf_to_opt_2.surface
#surf_2.change_resolution(2,2)
#surf_2.change_resolution(mpol,ntor)

#initial_2=Vmec(os.path.join(os.path.dirname(__name__), 'input_files', filename),
#                ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,s_vmec=0.98)
#initialsurf_2 = initial_2.surface
#initialsurf_2.change_resolution(2,2)

##surf_2 = SurfaceRZFourier(filename_vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True,rescaling_type='L_infty',rescaling_factor=1.2)
##surf_2.change_resolution(mpol,ntor)

##initialsurf_2 = SurfaceRZFourier(filename_vmec, ntheta=ntheta, nphi=nphi, range_torus='half period', close=True)
##initialsurf_2.change_resolution(mpol,ntor)


# Load coils and construct field
from essos.coils import Coils_from_json,Coils_from_simsopt
coils = Coils_from_json("input_files/stellarator_coils_normal.json") # from optimize_coils_vmec_surface.py
#json_file = os.path.join(os.path.dirname(__name__), 'input_files', filename_coils)
#coils = Coils_from_simsopt(json_file,nfp=4)
field = BiotSavart(coils)

# QFM optimization setup
method = 'alm' # lbfgs, slsqp, alm
label = 'multi'  # 'area', 'volume', 'toroidal_flux'

if method == 'lbfgs':
    tol = 1e-4
elif method == 'slsqp':
    tol = 1e-6
elif method == 'alm':
    tol = 1e-6

maxiter = 10000
constraint_weight = 1e-3
factor=1.

initial_label_flux = toroidal_flux(surf, field)
targetlabel_flux = toroidal_flux(truevmec.surface, field,idx=0)*factor
targetlabel_flux_final = toroidal_flux(truevmec.surface, field,idx=-1)*factor

initial_label_flux_poloidal = poloidal_flux(surf, field)
targetlabel_flux_poloidal = poloidal_flux(truevmec.surface, field,idx=0)*factor
targetlabel_flux_poloidal_final = poloidal_flux(truevmec.surface, field,idx=-1)*factor

initial_label_volume = surf.volume*factor
targetlabel_volume = truevmec.surface.volume

initial_label_area = surf.area
targetlabel_area = truevmec.surface.area*factor

#Second surface at s=0.98
initial_label_flux_2 = toroidal_flux(surf_2, field)
targetlabel_flux_2 = toroidal_flux(truevmec_2.surface, field,idx=0)*factor
targetlabel_flux_final_2 = toroidal_flux(truevmec_2.surface, field,idx=-1)*factor

initial_label_flux_poloidal_2 = poloidal_flux(surf_2, field)
targetlabel_flux_poloidal_2 = poloidal_flux(truevmec_2.surface, field,idx=0)*factor
targetlabel_flux_poloidal_final_2 = poloidal_flux(truevmec_2.surface, field,idx=-1)*factor

initial_label_volume_2 = surf_2.volume*factor
targetlabel_volume_2 = truevmec_2.surface.volume

initial_label_area_2 = surf_2.area
targetlabel_area_2 = truevmec_2.surface.area*factor



BdotN_over_B_initial = BdotN_over_B(surf, BiotSavart(coils))

BdotN_over_B_initial_2 = BdotN_over_B(surf_2, BiotSavart(coils))
# Initialize QFM optimizer
qfm = QfmSurface(field=field, surface=surf,  targetlabel_flux=targetlabel_flux,targetlabel_flux_final=targetlabel_flux_final,targetlabel_area=targetlabel_area,targetlabel_volume=targetlabel_volume,label=label)


qfm_2 = QfmSurface(field=field, surface=surf_2,  targetlabel_flux=targetlabel_flux_2,targetlabel_flux_final=targetlabel_flux_final_2,targetlabel_area=targetlabel_area_2,targetlabel_volume=targetlabel_volume_2,label=label)

print("Degrees of Freedom:", qfm.surface.x.shape[0])
start_time = time() 
print('start')


result = qfm.run(
    tol=tol,
    maxiter=maxiter,
    method=method,
    constraint_weight=constraint_weight,
    log_every=10  
)


result_2 = qfm_2.run(
    tol=tol,
    maxiter=maxiter,
    method=method,
    constraint_weight=constraint_weight,
    log_every=10  
)


print('done')
end_time = time()

# Evaluate final objective and constraint
x_opt = device_get(result["s"].x)
qfm_loss = float(jnp.asarray(qfm.objective(x_opt)))
c_loss = float(jnp.asarray(qfm.constraint_flux(x_opt))+jnp.asarray(qfm.constraint_area(x_opt))+jnp.asarray(qfm.constraint_volume(x_opt)))

BdotN_over_B_optimized = BdotN_over_B(result['s'], BiotSavart(coils))
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
initial_pf = poloidal_flux(surf, field)

final_area = result['s'].area
final_volume = result['s'].volume
final_tf = toroidal_flux(result['s'], field)
final_pf = poloidal_flux(result['s'], field)

print(f"Initial labels -> area: {initial_area:.6e}, volume: {initial_volume:.6e}, toroidal_flux: {initial_tf:.6e},poloidal_flux: {initial_pf:.6e}")
print(f"target label: {label}   target label value: {targetlabel_area:.6e}, {targetlabel_volume:.6e}, {targetlabel_flux:.6e}, {targetlabel_flux_poloidal:.6e}")
print(f"Final labels   -> area: {final_area:.6e}, volume: {final_volume:.6e}, toroidal_flux: {final_tf:.6e}, poloidal_flux: {final_pf:.6e}")


#Second surface at s=0.98
x_opt_2 = device_get(result_2["s"].x)
qfm_loss_2 = float(jnp.asarray(qfm_2.objective(x_opt_2)))
c_loss_2 = float(jnp.asarray(qfm_2.constraint_flux(x_opt_2))+jnp.asarray(qfm_2.constraint_area(x_opt_2))+jnp.asarray(qfm_2.constraint_volume(x_opt_2)))     
BdotN_over_B_optimized_2 = BdotN_over_B(result_2['s'], BiotSavart(coils))
print("Optimization method:", method)
print("Optimization label:", label)
print("Optimization success:", result_2['success'])
print(f"final qfm objective = {qfm_loss_2:.3e}, final constraint objective = {c_loss_2:.3e}")
print("Iterations:", result_2['iter'])
print(f"Optimization time: {end_time - start_time}")        
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial_2):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized_2):.2e}")
initial_area_2 = surf_2.area
initial_volume_2 = surf_2.volume
initial_tf_2 = toroidal_flux(surf_2, field)
initial_pf_2 = poloidal_flux(surf_2, field)
final_area_2 = result_2['s'].area
final_volume_2 = result_2['s'].volume
final_tf_2 = toroidal_flux(result_2['s'], field)
final_pf_2 = poloidal_flux(result_2['s'], field)
print(f"Initial labels -> area: {initial_area_2:.6e}, volume: {initial_volume_2:.6e}, toroidal_flux: {initial_tf_2:.6e},poloidal_flux: {initial_pf_2:.6e}")
print(f"target label: {label}   target label value: {targetlabel_area_2:.6e}, {targetlabel_volume_2:.6e}, {targetlabel_flux_2:.6e}, {targetlabel_flux_poloidal_2:.6e}")
print(f"Final labels   -> area: {final_area_2:.6e}, volume: {final_volume_2:.6e}, toroidal_flux: {final_tf_2:.6e}, poloidal_flux: {final_pf_2:.6e}")




# Plot surfaces
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

coils.plot(ax=ax1, show=False)
coils.plot(ax=ax2, show=False)
coils.plot(ax=ax3, show=False)


initialsurf.plot(ax=ax1, show=False)
#surf.plot(ax=ax2, show=False)
#truevmec.surface.plot(ax=ax1, show=False)
truevmec.surface.plot(ax=ax2, show=False)
result['s'].plot(ax=ax3, show=False)

ax1.set_title("Initial Surface")
ax2.set_title("True VMEC Surface")
ax3.set_title("Final Surface")

plt.tight_layout()
plt.savefig('optimize_qfm_surface.png', dpi=300)


#Plot surfaces 2 at s=0.98
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
coils.plot(ax=ax1, show=False)
coils.plot(ax=ax2, show=False)
coils.plot(ax=ax3, show=False)
initialsurf_2.plot(ax=ax1, show=False)
#surf_2.plot(ax=ax2, show=False)
#truevmec_2.surface.plot(ax=ax1, show=False)
truevmec_2.surface.plot(ax=ax2, show=False)
result_2['s'].plot(ax=ax3, show=False)
ax1.set_title("Initial Surface s=0.98")
ax2.set_title("True VMEC Surface s=0.98")
ax3.set_title("Final Surface s=0.98")
plt.tight_layout()
plt.savefig('optimize_qfm_surface_2.png', dpi=300)


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
# # Contours from optimized surface
R0_opt_2, Z0_opt_2 = compute_rz_on_phi(result_2['s'], theta, phi=0.0)
R90_opt_2, Z90_opt_2 = compute_rz_on_phi(result_2['s'], theta, phi=jnp.pi/2)
# # Contours from true VMEC surface
R0_true, Z0_true = compute_rz_on_phi(truevmec.surface, theta, phi=0.0)
R90_true, Z90_true = compute_rz_on_phi(truevmec.surface, theta, phi=jnp.pi/2)

fig, ax = plt.subplots(figsize=(6, 6))

tracing.poincare_plot(ax=ax, show=False, shifts=[0, jnp.pi / 2])
ax.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 0$")
ax.plot(R90_opt, Z90_opt, color='black', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2$")

ax.plot(R0_opt_2, Z0_opt_2, color='red', linewidth=1.5, label=r"Optimized @ $\phi = 0 s=0.98$")
ax.plot(R90_opt_2, Z90_opt_2, color='red', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2, s=0.98$")
#ax.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
#ax.plot(R90_true, Z90_true, color='blue', linestyle='--', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")

ax.set_xlabel("R")
ax.set_ylabel("Z")
ax.set_title("Poincaré + Surfaces Comparison @ φ = 0 and π/2")
ax.legend()
ax.axis("equal")
plt.tight_layout()
plt.savefig('optimize_qfm_surface_poincare.png', dpi=300)




from essos.surfaces import B_on_surface

B_final= B_on_surface(result['s'], field)
e_s=(result['s'].gamma-result_2['s'].gamma)/0.02  
jac=jnp.einsum('ijk,ijk->ij',e_s, jnp.cross(result['s'].gammadash_theta,result['s'].gammadash_phi,axis=-1))
jac_g = jac[:, :, jnp.newaxis]
jac_g = jnp.repeat(jac_g, 3, axis=2)
grad_alpha_final = -jnp.cross(B_final, result['s'].unitnormal, axis=-1)
grad_psi = jnp.cross(result['s'].gammadash_theta, result['s'].gammadash_phi, axis=-1)/jac_g
#grad_psi=jnp.true_divide(grad_psi,jnp.linalg.norm(grad_psi,axis=-1,keepdims=True))
grad_theta = -jnp.cross(e_s, result['s'].gammadash_phi, axis=-1)/jac_g
grad_phi = jnp.cross(e_s, result['s'].gammadash_theta, axis=-1)/jac_g


#grad_phi=jnp.true_divide(grad_phi,jnp.linalg.norm(grad_phi,axis=-1,keepdims=True))
e_phi=jnp.cross(grad_phi,grad_psi , axis=-1)
B_contravariant_psi=jnp.einsum('ijk,ijk->ij',B_final, grad_psi)*jac
B_contravariant_theta=jnp.einsum('ijk,ijk->ij',B_final, grad_theta)*jac
B_contravariant_phi=jnp.einsum('ijk,ijk->ij',B_final, grad_phi)*jac
iota=jnp.average(B_contravariant_theta)/jnp.average(B_contravariant_phi)#*result['s'].nfp
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(result['s'].phi_2d,result['s'].theta_2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB.png', dpi=300)

fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(result['s'].phi_2d,result['s'].theta_2d,B_contravariant_psi, levels=200, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('Bsup_psi.png', dpi=300)

fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(result['s'].phi_2d,result['s'].theta_2d,B_contravariant_theta, levels=200, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('Bsup_theta.png', dpi=300)

fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(result['s'].phi_2d,result['s'].theta_2d,B_contravariant_phi, levels=200, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('Bsup_phi.png', dpi=300)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(result['s'].phi_2d,result['s'].theta_2d, iota, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('iota.png', dpi=300)

poloidal_flux(truevmec.surface, field,idx=40)/toroidal_flux(truevmec.surface, field,idx=40)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(result['s'].phi_2d,result['s'].theta_2d,jacobian, levels=200, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('jacobian.png', dpi=300)

for i in range(truevmec.surface.nphi):
    for j in range(truevmec.surface.ntheta):
        pol_av=jnp.sum(poloidal_flux(truevmec.surface, field,idx=j)*jacobian[i,j])

for i in range(truevmec.surface.nphi):
    for j in range(truevmec.surface.ntheta):
        tol_av=jnp.sum(toroidal_flux(truevmec.surface, field,idx=i)*jacobian[i,j])


      