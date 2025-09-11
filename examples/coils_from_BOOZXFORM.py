#!/usr/bin/env python3.11
import os
number_of_processors_to_use = 6 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import numpy as np
from time import time
import booz_xform as bx
import plotly.graph_objects as go
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from simsopt.mhd import Vmec, Boozer
from jax import block_until_ready
import jax.numpy as jnp
from essos.coils import fit_dofs_from_coils, Curves, Coils
import matplotlib.pyplot as plt

input_dir = os.path.join(os.path.dirname(__file__), 'input_files')
output_dir = os.path.join(os.path.dirname(__file__), 'output_files')
os.makedirs(output_dir, exist_ok=True)

wout_filename = os.path.join(input_dir, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
boozmn_filename = os.path.join(output_dir, 'boozmn_LandremanPaul2021_QA_reactorScale_lowres.nc')

# if boozmn_filename.split('/')[-1] in os.listdir(output_dir):
#     print(f"File {boozmn_filename} already exists, skipping computation")
#     b = bx.Booz_xform()
#     b.read_boozmn(boozmn_filename)
# else:
print(f"Computing {boozmn_filename}")
vmec = Vmec(wout_filename, verbose=False)
b = Boozer(vmec, mpol=64, ntor=64, verbose=True)
b.register([1])
b.run()
b.bx.write_boozmn(boozmn_filename)
b = b.bx

# bx.wireplot(b1, refine=6, orig=True).show()

js = None
ntheta = 41
ncoils = 4
tmax = 2000
nfieldlines_per_core=1
nfieldlines = number_of_processors_to_use*nfieldlines_per_core
trace_tolerance = 1e-8
num_steps = 8*tmax
order = 4
current_on_each_coil = 2e8/ncoils
n_segments = ntheta
# nphi = 80
nphi   = ncoils * 2 * b.nfp
surf = True
orig = True
R0_initial = sum(vmec.wout.rmnc)[0]
r_max_poincare = 0.8

theta1D = np.linspace(0, 2 * np.pi, ntheta)
# phi1D = np.linspace(0, 2 * np.pi, nphi)
phi1D = jnp.linspace(2*jnp.pi/nphi/2, 2*jnp.pi + 2*jnp.pi/nphi/2, nphi, endpoint=False)
varphi, theta = np.meshgrid(phi1D, theta1D)

R = np.zeros_like(theta)
Z = np.zeros_like(theta)
nu = np.zeros_like(theta)

for jmn in range(b.mnboz):
    m = b.xm_b[jmn]
    n = b.xn_b[jmn]
    angle = m * theta - n * varphi
    sinangle = np.sin(angle)
    cosangle = np.cos(angle)
    R += b.rmnc_b[jmn, js] * cosangle
    Z += b.zmns_b[jmn, js] * sinangle
    nu += b.numns_b[jmn, js] * sinangle

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
    
time0 = time()
dofs, gamma_uni = fit_dofs_from_coils(coils_gamma[:ncoils], order=order, n_segments=n_segments, assume_uniform=True)
curves = Curves(dofs=dofs, n_segments=n_segments, nfp=b.nfp, stellsym=True)
coils = Coils(curves=curves, currents=[-current_on_each_coil]*(ncoils))
field_coils_DOFS = BiotSavart(coils)
print(f"Fitting coils took {time()-time0:.2f} seconds")

# color = '#FF9999'
# # Hack to get a uniform surface color:
# colorscale = [[0, color], [1, color]]

data=[]
# Rsurf = R
# Zsurf = Z
# Xsurf = Rsurf * np.cos(phi)
# Ysurf = Rsurf * np.sin(phi)
# data.append(go.Surface(x=Xsurf, y=Ysurf, z=Zsurf,
#                     colorscale=colorscale,
#                     opacity=0.5,
#                     showscale=False, # Turns off colorbar
#                     lighting={"specular": 0.3, "diffuse":0.9}))

line_width = 8
line_marker = dict(color='red', width=line_width)
index = 0
index = 0
for i, j, k in zip(X.T, Y.T, Z.T):
    index += 1
    showlegend = True
    if index > 1:
        showlegend = False
    data.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker, showlegend=showlegend, name=r'Constant $\varphi$ contours'))

line_marker = dict(color='blue', width=line_width)
gamma_coils = np.transpose(curves.gamma, (1, 0 , 2))
index = 0
index = 0
for i, j, k in zip(gamma_coils[:, :, 0].T, gamma_coils[:, :, 1].T, gamma_coils[:, :, 2].T):
    index += 1
    showlegend = True
    if index > 1:
        showlegend = False
    data.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker, showlegend=showlegend, name='Coils fitted to Fourier'))

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
line_marker = dict(color='green', width=line_width)
    
index = 0
for i, j, k in zip(X_phi.T, Y_phi.T, Z_phi.T):
    index += 1
    showlegend = True
    if index > 1:
        showlegend = False
    data.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker, showlegend=showlegend, name=r"Constant $\phi$ contours"))

coils_gamma_phi = np.zeros((ncoils, ntheta, 3))
for i in range(ncoils):
    coils_gamma_phi[i, :, 0] = X_phi[:, i]
    coils_gamma_phi[i, :, 1] = Y_phi[:, i]
    coils_gamma_phi[i, :, 2] = Z_phi[:, i]
    
time0 = time()
dofs_phi, gamma_uni_phi = fit_dofs_from_coils(coils_gamma_phi[:ncoils], order=order, n_segments=n_segments, assume_uniform=True)
curves_phi = Curves(dofs=dofs_phi, n_segments=n_segments, nfp=b.nfp, stellsym=True)
coils_phi = Coils(curves=curves_phi, currents=[-current_on_each_coil]*(ncoils))
field_coils_phi = BiotSavart(coils_phi)
print(f"Fitting coils took {time()-time0:.2f} seconds")

R0 = jnp.linspace(R0_initial, R0_initial+r_max_poincare, nfieldlines)
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
time0 = time()
tracing_coils_phi = block_until_ready(Tracing(field=field_coils_phi, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                  maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
print(f"ESSOS tracing coils_phi took {time()-time0:.2f} seconds")
trajectories_coils_phi = tracing_coils_phi.trajectories
time0 = time()
tracing_coils_DOFS = block_until_ready(Tracing(field=field_coils_DOFS, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                    maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
print(f"ESSOS tracing coils_DOFS took {time()-time0:.2f} seconds")
trajectories_coils_DOFS = tracing_coils_DOFS.trajectories

# Add fieldline traces from fitted coils
for traj in trajectories_coils_DOFS:
    data.append(go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines',
        line=dict(color='black', width=0.1),
        opacity=1.0,
        name='Fieldline constant Boozer coils',
        showlegend=False if traj is not trajectories_coils_DOFS[0] else True
    ))

# Add fieldlines from phi coils
for traj in trajectories_coils_phi:
    data.append(go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines',
        line=dict(color='blue', width=0.1),
        opacity=1.0,
        name='Fieldline constant phi coils',
        showlegend=False if traj is not trajectories_coils_phi[0] else True
    ))
        
fig = go.Figure(data=data)

# Turn off hover contours on the surface:
fig.update_traces(contours_x_highlight=False,
                contours_y_highlight=False,
                contours_z_highlight=False,
                selector={"type":"surface"})

# Make x, y, z coordinate scales equal, and turn off more hover stuff
fig.update_layout(scene={"aspectmode": "data",
                            "xaxis_showspikes": False,
                            "yaxis_showspikes": False,
                            "zaxis_showspikes": False,
                            "xaxis_visible": False,
                            "yaxis_visible": False,
                            "zaxis_visible": False},
                    hovermode=False,
                    margin={"l":0, "r":0, "t":25, "b":0},
                    # title="Curves of constant poloidal or toroidal angle"
                    )

fig.show()

# Now plot the 2D Poincare plot with Matplotlib (ax2 only)
fig2 = plt.figure(figsize=(6, 5))
ax2 = fig2.add_subplot(111)
shifts = jnp.array([0])
tracing_coils_phi.poincare_plot(ax=ax2, show=False, shifts=shifts/b.nfp/2, color='r', s=0.1)
tracing_coils_DOFS.poincare_plot(ax=ax2, show=False, shifts=shifts/b.nfp/2, color='b', s=0.1)

Rsurf_phi0 = np.array([0.0]*ntheta)
Zsurf_phi0 = np.array([0.0]*ntheta)
for jmn in range(b.mnboz):
    Rsurf_phi0 += (b.rmnc_b[jmn, js] * np.cos(b.xm_b[jmn] * theta1D))[0]
    Zsurf_phi0 += (b.zmns_b[jmn, js] * np.sin(b.xm_b[jmn] * theta1D))[0]
ax2.plot(Rsurf_phi0, Zsurf_phi0, color='grey', alpha=1.0, linewidth=4, label='Coil Surface')

# for coil_number in range(ncoils):
#     R_coils_gamma = jnp.sqrt(coils_gamma[coil_number,:,0]**2 + coils_gamma[coil_number,:,1]**2)
#     R_curve = jnp.sqrt(curves.gamma[coil_number,:,0]**2 + curves.gamma[coil_number,:,1]**2)

#     ax2.plot(R_coils_gamma, coils_gamma[coil_number,:,2],
#             color='black', linewidth=3.5, alpha=0.9,
#             label='Coils from Boozer Surface' if coil_number==0 else '_nolegend_')

#     ax2.plot(R_curve, curves.gamma[coil_number,:,2],
#             linestyle='--', color='tab:orange', linewidth=3.5, alpha=0.8,
#             label='Coil fitted to Fourier' if coil_number==0 else '_nolegend_')
ax2.plot([], [], color='red',  label='Fieldlines from coils of constant phi')
ax2.plot([], [], color='blue', label='Fieldlines from coils of constant Boozer')
ax2.legend()
plt.tight_layout()
plt.show()