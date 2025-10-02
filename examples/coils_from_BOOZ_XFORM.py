#!/usr/bin/env python3.11
import os
number_of_processors_to_use = 6 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import numpy as np
from time import time
import booz_xform as bx
import plotly.graph_objects as go
from essos.dynamics import Tracing
from essos.fields import BiotSavart, Vmec as VmecESSOS
from simsopt.mhd import Vmec, Boozer
from jax import block_until_ready
import jax.numpy as jnp
from essos.coils import fit_dofs_from_coils, Curves, Coils, CreateEquallySpacedCurves
import matplotlib.pyplot as plt
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_BdotN
from essos.surfaces import BdotN_over_B
from plot_helpers import (
    TubeFactory, surface_trace_from_RZ_phi, tubes_mesh3d_from_gammas,
    add_tubes_from_columns, add_tubes_from_gamma, add_polyline_trajs, npf,
    plot_loss_logs
)

# file_to_use = 'LandremanPaul2021_QA_reactorScale_lowres'
file_to_use = 'LandremanPaul2021_QH_reactorScale_lowres'

ntheta = 41
ncoils = 5
tmax = 1100
nfieldlines_per_core=1
trace_tolerance = 1e-9
num_steps = 22000
order_Fourier_coils = 5
current_on_each_coil = 2e8
refine_nphi_for_surface_plot = max(4, number_of_processors_to_use)
n_segments_coils = 80

radial_extension_of_the_surface = 0.01
max_coil_length_amplification = 3.5
max_coil_curvature_amplification = 1.0/max_coil_length_amplification/3
min_distance_cc = 0.05
maximum_function_evaluations = 1000
tolerance_optimization = 1e-6
s_surface = 0.9

x_scale = False
use_circular_coils = False
plot_fieldlines = True

Poincare_plot_phi = jnp.array([0])
shift_surface_plot_for_phi = jnp.pi
plot_fieldlines_constant_phi = False
show_coils_fitted_to_Fourier = False

input_dir = os.path.join(os.path.dirname(__file__), 'input_files')
output_dir = os.path.join(os.path.dirname(__file__), 'output_files')
os.makedirs(output_dir, exist_ok=True)

wout_filename = os.path.join(input_dir, 'wout_'+file_to_use+'.nc')
boozmn_filename = os.path.join(output_dir, 'boozmn_'+file_to_use+'.nc')

print(f"Computing {boozmn_filename}")
vmec = Vmec(wout_filename, verbose=False)
b = Boozer(vmec, mpol=64, ntor=64, verbose=True)
time0 = time()
b.register([1])
b.run()
# b.bx.write_boozmn(boozmn_filename)
b = b.bx
print(f"Computing Boozer harmonics took {time()-time0:.2f} seconds")

current_on_each_coil = current_on_each_coil / ncoils*vmec.wout.Aminor_p**2/1.7**2
nfieldlines = number_of_processors_to_use*nfieldlines_per_core
nphi   = ncoils * 2 * b.nfp
vmec_ESSOS = VmecESSOS(wout_filename, ntheta=ntheta, nphi=nphi*refine_nphi_for_surface_plot, range_torus='half period', s=s_surface)

theta1D = np.linspace(0, 2 * np.pi, ntheta)
phi1D = jnp.linspace(2*jnp.pi/nphi/2, 2*jnp.pi + 2*jnp.pi/nphi/2, nphi, endpoint=False)
phi1D_surface = jnp.linspace(0, 2*jnp.pi, nphi*refine_nphi_for_surface_plot, endpoint=True)
varphi, theta = np.meshgrid(phi1D, theta1D)
varphi_surface, theta_surface = np.meshgrid(phi1D_surface, theta1D)

R = np.zeros_like(theta)
R_surface = np.zeros_like(theta_surface)
Z = np.zeros_like(theta)
Z_surface = np.zeros_like(theta_surface)
nu = np.zeros_like(theta)
d_R_d_theta = np.zeros_like(theta)
d_R_d_theta_surface = np.zeros_like(theta_surface)
d_Z_d_theta = np.zeros_like(theta)
d_Z_d_theta_surface = np.zeros_like(theta_surface)

phi1D_Boozerplot = np.linspace(0, 2 * np.pi / b.nfp / 2, nphi*refine_nphi_for_surface_plot)
phi_Boozerplot, theta_Boozerplot = np.meshgrid(phi1D_Boozerplot, theta1D)
modB_Boozerplot = np.zeros_like(theta_Boozerplot)

js = None
for jmn in range(b.mnboz):
    m = b.xm_b[jmn]
    n = b.xn_b[jmn]
    angle = m * theta - n * varphi
    angle_surface = m * theta_surface - n * varphi_surface
    sinangle = np.sin(angle)
    sinangle_surface = np.sin(angle_surface)
    cosangle = np.cos(angle)
    cosangle_surface = np.cos(angle_surface)
    R += b.rmnc_b[jmn, js] * cosangle
    R_surface += b.rmnc_b[jmn, js] * cosangle_surface
    Z += b.zmns_b[jmn, js] * sinangle
    Z_surface += b.zmns_b[jmn, js] * sinangle_surface
    nu += b.numns_b[jmn, js] * sinangle
    d_R_d_theta += -m * b.rmnc_b[jmn, js] * sinangle
    d_R_d_theta_surface += -m * b.rmnc_b[jmn, js] * sinangle_surface
    d_Z_d_theta += m * b.zmns_b[jmn, js] * cosangle
    d_Z_d_theta_surface += m * b.zmns_b[jmn, js] * cosangle_surface
    cosangle_Boozerplot = np.cos(m * theta_Boozerplot - n * phi_Boozerplot)
    modB_Boozerplot += b.bmnc_b[jmn, js] * np.cos(cosangle_Boozerplot)

denom = np.sqrt(d_R_d_theta * d_R_d_theta + d_Z_d_theta * d_Z_d_theta)
denom_surface = np.sqrt(d_R_d_theta_surface * d_R_d_theta_surface + d_Z_d_theta_surface * d_Z_d_theta_surface)
R = R - radial_extension_of_the_surface * (d_Z_d_theta / denom)
R_surface = R_surface - radial_extension_of_the_surface * (d_Z_d_theta_surface / denom_surface)
Z = Z + radial_extension_of_the_surface * (d_R_d_theta / denom)
Z_surface = Z_surface + radial_extension_of_the_surface * (d_R_d_theta_surface / denom_surface)

# Following the sign convention in the code, to convert from the
# Boozer toroidal angle to the standard toroidal angle, we
# *subtract* nu:
phi = varphi - nu
X = R * np.cos(phi)
Y = R * np.sin(phi)

coils_gamma = np.zeros((ncoils, ntheta, 3))
for i in range(ncoils):
    coils_gamma[i, :, 0] = X[:, i]
    coils_gamma[i, :, 1] = Y[:, i]
    coils_gamma[i, :, 2] = Z[:, i]
    
major_radius_coils = vmec_ESSOS.r_axis
# minor_radius_coils = vmec_ESSOS.r_axis/2.5
minor_radius_coils = (jnp.max(coils_gamma[:, :, 0])-vmec_ESSOS.r_axis)*1.2

def CreateEquallySpacedCurves_from_axis(n_curves: int, order: int, R: float, r: float, n_segments: int = 100,
                              nfp: int = 1, stellsym: bool = False, rc = None, zs = None) -> jnp.ndarray:
    angles = (jnp.arange(n_curves) + 0.5) * (2 * jnp.pi) / ((1 + int(stellsym)) * nfp * n_curves)
    curves = jnp.zeros((n_curves, 3, 1 + 2 * order))
    r_axis = jnp.array([sum([rc[j]*jnp.cos(j*b.nfp*angles[i]) for j in range(len(rc))]) for i in range(len(angles))])
    z_axis = jnp.array([sum([zs[j]*jnp.sin(j*b.nfp*angles[i]) for j in range(len(zs))]) for i in range(len(angles))])
    axis_gamma = jnp.array([r_axis*jnp.cos(angles), r_axis*jnp.sin(angles), z_axis]).T
    curves = curves.at[:, 0, 0].set(axis_gamma[:,0])  # x[0]
    curves = curves.at[:, 0, 2].set(jnp.cos(angles) * r)  # x[2]
    curves = curves.at[:, 1, 0].set(axis_gamma[:,1])  # y[0]
    curves = curves.at[:, 1, 2].set(jnp.sin(angles) * r)  # y[2]
    curves = curves.at[:, 2, 0].set(axis_gamma[:,2])  # z[0]
    curves = curves.at[:, 2, 1].set(-r)             # z[1]
    return Curves(curves, n_segments=n_segments, nfp=nfp, stellsym=stellsym)
# curves_circular = CreateEquallySpacedCurves(n_curves=ncoils, order=order_Fourier_coils,
#                                 R=major_radius_coils, r=minor_radius_coils,
#                                 n_segments=n_segments_coils, nfp=vmec_ESSOS.nfp, stellsym=True)
curves_circular = CreateEquallySpacedCurves_from_axis(n_curves=ncoils, order=order_Fourier_coils,
                                R=major_radius_coils, r=minor_radius_coils,
                                n_segments=n_segments_coils, nfp=vmec_ESSOS.nfp, stellsym=True,
                                rc=vmec_ESSOS.raxis_cc, zs=-vmec_ESSOS.zaxis_cs)


time0 = time()
dofs, gamma_uni = fit_dofs_from_coils(coils_gamma[:ncoils], order=order_Fourier_coils, n_segments=n_segments_coils, assume_uniform=True)
curves_from_BOOZ_XFORM = Curves(dofs=dofs, n_segments=n_segments_coils, nfp=b.nfp, stellsym=True)
print(f"Fitting coils took {time()-time0:.2f} seconds")

if use_circular_coils:
    X, Y, Z = curves_circular.gamma[:, :, 0].T, curves_circular.gamma[:, :, 1].T, curves_circular.gamma[:, :, 2].T
    curves = curves_circular
else:
    curves = curves_from_BOOZ_XFORM

coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*(ncoils))
field_coils_DOFS_initial = BiotSavart(coils_initial)

initial_length = BiotSavart(Coils(curves=curves_from_BOOZ_XFORM, currents=[current_on_each_coil]*(ncoils))).coils.length
initial_curvature = BiotSavart(Coils(curves=curves_from_BOOZ_XFORM, currents=[current_on_each_coil]*(ncoils))).coils.curvature

max_coil_length = jnp.sum(initial_length).copy()*max_coil_length_amplification
max_coil_curvature = float(jnp.max(initial_curvature)*max_coil_curvature_amplification)

print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
time0 = time()
coils_optimized = optimize_loss_function(loss_BdotN, initial_dofs=coils_initial.x, coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                  maximum_function_evaluations=maximum_function_evaluations, vmec=vmec_ESSOS,
                                  max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature, min_distance_cc=min_distance_cc,
                                  log_csv_path=f"opt_" + file_to_use + ("circular" if use_circular_coils else "booz") + ("_xscale" if x_scale else "") + ".csv", x_scale=x_scale)
print(f"Optimization took {time()-time0:.2f} seconds")
BdotN_over_B_initial = BdotN_over_B(vmec_ESSOS.surface, BiotSavart(coils_initial))
BdotN_over_B_optimized = BdotN_over_B(vmec_ESSOS.surface, BiotSavart(coils_optimized))
curvature=jnp.mean(BiotSavart(coils_optimized).coils.curvature, axis=1)
length=jnp.max(jnp.ravel(BiotSavart(coils_optimized).coils.length))
print(f"Maximum coil length before optimization: {jnp.max(initial_length):.2f} m")
print(f"Maximum coil length after optimization: {jnp.max(coils_optimized.length):.2f} m")
print(f"Mean coil curvature before optimization: {jnp.mean(initial_curvature):.2f} m^-1")
print(f"Mean coil curvature after optimization: {jnp.mean(coils_optimized.curvature):.2f} m^-1")
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")

R0 = jnp.linspace(sum(vmec.wout.rmnc)[0], sum(vmec.wout.rmnc)[-1], nfieldlines+1)[:-1]
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
# tube = TubeFactory(n_theta=12)

data=[]
# Magnetic surface
data.append(surface_trace_from_RZ_phi(R_surface, Z_surface, phi1D_surface, color="#C5B6A7", opacity=0.6))
# Original coils -> make tubes
decimate = 1
tube_radius = 0.15
ntheta_tube = 12
opacity_tube = 1.0
coils_orig = [np.column_stack([npf(xi), npf(yi), npf(zi)])[::decimate]
              for xi, yi, zi in zip(npf(X).T, npf(Y).T, npf(Z).T)]
mesh_orig = tubes_mesh3d_from_gammas(coils_orig, radius=tube_radius, n_theta=ntheta_tube, color="#BA4444", opacity=opacity_tube)
data.append(mesh_orig)
# Optimized coils -> make tubes
coils_opt = [npf(P)[::decimate] for P in npf(coils_optimized.gamma)]
mesh_opt = tubes_mesh3d_from_gammas(coils_opt, radius=tube_radius, n_theta=ntheta_tube, color="#CD9B3F", opacity=opacity_tube)
data.append(mesh_opt)
# Optional Fourier-fit as lines (cheap to render)
if show_coils_fitted_to_Fourier:
    gamma_coils = np.transpose(curves.gamma, (1, 0, 2))
    for i, j, k in zip(gamma_coils[:, :, 0].T, gamma_coils[:, :, 1].T, gamma_coils[:, :, 2].T):
        data.append(go.Scatter3d(x=npf(i), y=npf(j), z=npf(k), mode="lines",
                                 line=dict(color="blue", width=8),
                                 showlegend=False, name="Coils fitted to Fourier"))
# Fieldlines (heavily downsampled for interactivity)
if plot_fieldlines:
    time0 = time()
    tracing_coils_DOFS = block_until_ready(Tracing(field=BiotSavart(coils_optimized), model='FieldLineAdaptative', initial_conditions=initial_xyz,
                        maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
    print(f"ESSOS tracing coils_DOFS took {time()-time0:.2f} seconds")
    trajectories_coils_DOFS = tracing_coils_DOFS.trajectories
    add_polyline_trajs(data, trajectories_coils_DOFS, color="black", width=0.2, every=2,
                       name="Fieldline constant Boozer coils")
if plot_fieldlines_constant_phi:
    js_phi = b.compute_surfs[js]
    R_phi = np.zeros_like(theta)
    Z_phi = np.zeros_like(theta)
    phi1D_phi = jnp.linspace(2*jnp.pi/nphi/2, 2*jnp.pi + 2*jnp.pi/nphi/2, nphi, endpoint=False)
    phi_phi, _ = np.meshgrid(phi1D_phi, theta1D)
    for jmn in range(b.mnmax):
        angle = b.xm[jmn] * theta - b.xn[jmn] * phi_phi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        R_phi += b.rmnc[jmn, js_phi] * cosangle
        Z_phi += b.zmns[jmn, js_phi] * sinangle
    X_phi = R_phi * np.cos(phi_phi)
    Y_phi = R_phi * np.sin(phi_phi)
    coils_gamma_phi = np.zeros((ncoils, ntheta, 3))
    for i in range(ncoils):
        coils_gamma_phi[i, :, 0] = X_phi[:, i]
        coils_gamma_phi[i, :, 1] = Y_phi[:, i]
        coils_gamma_phi[i, :, 2] = Z_phi[:, i]
    time0 = time()
    dofs_phi, gamma_uni_phi = fit_dofs_from_coils(coils_gamma_phi[:ncoils], order=order_Fourier_coils, n_segments=ntheta, assume_uniform=True)
    curves_phi = Curves(dofs=dofs_phi, n_segments=n_segments_coils, nfp=b.nfp, stellsym=True)
    coils_phi = Coils(curves=curves_phi, currents=[-current_on_each_coil]*(ncoils))
    field_coils_phi = BiotSavart(coils_phi)
    print(f"Fitting coils took {time()-time0:.2f} seconds")
    time0 = time()
    tracing_coils_phi = block_until_ready(Tracing(field=field_coils_phi, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                    maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
    print(f"ESSOS tracing coils_phi took {time()-time0:.2f} seconds")
    trajectories_coils_phi = tracing_coils_phi.trajectories
    add_polyline_trajs(data, trajectories_coils_phi, color="blue", width=0.2, every=2,
                       name="Fieldline constant phi coils")

fig = go.Figure(data=data)
fig.update_traces(contours_x_highlight=False, contours_y_highlight=False, contours_z_highlight=False,
                  selector={"type": "surface"})
fig.update_layout(
    scene=dict(
        aspectmode="data",
        xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False,
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
    ),
    hovermode=False,
    margin=dict(l=0, r=0, t=25, b=0),
)
fig.write_image(os.path.join(output_dir, '3D_'+file_to_use+'_' + ("circular" if use_circular_coils else "booz") + ("_xscale" if x_scale else "") + '.png'), scale=4, width=800, height=600)
fig.write_html(os.path.join(output_dir, '3D_'+file_to_use+'_' + ("circular" if use_circular_coils else "booz") + ("_xscale" if x_scale else "") + '.html'))
fig.show()

# Now plot the 2D Poincare plot with Matplotlib (ax2 only)
fig2 = plt.figure(figsize=(6, 5))
ax2 = fig2.add_subplot(111)

if plot_fieldlines:
    tracing_coils_DOFS.poincare_plot(ax=ax2, show=False, shifts=Poincare_plot_phi/b.nfp/2, color='b', s=0.15)
if plot_fieldlines_constant_phi:
    tracing_coils_phi.poincare_plot(ax=ax2, show=False, shifts=Poincare_plot_phi/b.nfp/2, color='r', s=0.15)

Rsurf_phi0  = np.array([0.0]*ntheta)
Zsurf_phi0  = np.array([0.0]*ntheta)
for jmn in range(b.mnboz):
    Rsurf_phi0 += (b.rmnc_b[jmn, js] * np.cos(b.xm_b[jmn] * theta1D - b.xn_b[jmn] * shift_surface_plot_for_phi))[0]
    Zsurf_phi0 += (b.zmns_b[jmn, js] * np.sin(b.xm_b[jmn] * theta1D - b.xn_b[jmn] * shift_surface_plot_for_phi))[0]
ax2.plot(Rsurf_phi0,  Zsurf_phi0,  color='black', alpha=1.0, linewidth=2, label='Surface of Constant Boozer Angle')
ax2.set_xlabel('R (m)')
ax2.set_ylabel('Z (m)')
ax2.plot([], [], color='blue', label='Fieldlines')
if plot_fieldlines_constant_phi:
    ax2.plot([], [], color='red',  label='Fieldlines (constant phi)')

# Plot VMEC flux surfaces for reference
# Match the surfaces of VMEC closest to the radii of the fieldlines traced
s_fieldlines = (jnp.linspace(sum(vmec.wout.rmnc)[0], sum(vmec.wout.rmnc)[-1], nfieldlines+1)[:-1] - sum(vmec.wout.rmnc)[0])/ \
               (sum(vmec.wout.rmnc)[-1] - sum(vmec.wout.rmnc)[0])
s_vmec = jnp.sqrt(jnp.linspace(0, 1, vmec.wout.ns))
iradii = np.array([np.abs(s_vmec - s).argmin() for s in s_fieldlines])
for iradius in range(nfieldlines):
    R = [0]*ntheta
    Z = [0]*ntheta
    for imode, xnn in enumerate(vmec.wout.xn):
        angle = vmec.wout.xm[imode]*theta1D - xnn*shift_surface_plot_for_phi
        R += vmec.wout.rmnc[imode, iradii[iradius]]*np.cos(angle)
        Z += vmec.wout.zmns[imode, iradii[iradius]]*np.sin(angle)
    ax2.plot(R, Z, 'r--', linewidth=1.5, label='Surfaces of Constant Cylindrical Angle' if iradius ==0 else '_nolegend_')
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'poincare_'+file_to_use+'_' + ("circular" if use_circular_coils else "booz") + ("_xscale" if x_scale else "") + '.png'), dpi=300)

fig = plt.figure()
plt.contourf(phi_Boozerplot, theta_Boozerplot, modB_Boozerplot, levels=6)
plt.xlabel(r'Boozer toroidal angle $\varphi$')
plt.ylabel(r'Boozer poloidal angle $\theta$')
for i in range(ncoils):
    plt.axvline(x=phi1D[i], color='black', linewidth=2.5)
plt.colorbar(label='|B| (T)')
fig.savefig(os.path.join(output_dir, 'modB_Boozerplot_'+file_to_use+'_' + ("circular" if use_circular_coils else "booz") + ("_xscale" if x_scale else "") + '.png'), dpi=300)

all_files = [os.path.join(output_dir, f"opt_" + file_to_use + suffix + ".csv") for suffix in ["circular", "booz", "circular_xscale", "booz_xscale"]]
existing_files = []
labels = []
for fname in all_files:
    if os.path.exists(fname):
        existing_files.append(fname)
        if "circular" in fname and "xscale" in fname:
            labels.append("Circular (xscale)")
        elif "booz" in fname and "xscale" in fname:
            labels.append("Boozer (xscale)")
        elif "circular" in fname:
            labels.append("Circular")
        elif "booz" in fname:
            labels.append("Boozer")
if existing_files:
    plot_loss_logs(existing_files, out_path=os.path.join(output_dir, "loss_compare_"+file_to_use+".png"), ylog=True, labels=labels)

plt.show()