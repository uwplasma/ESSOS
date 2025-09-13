import os
number_of_processors_to_use = 6 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import jax.numpy as jnp
from jax import block_until_ready, vmap
import matplotlib.pyplot as plt
from essos.fields import near_axis, BiotSavart_from_gamma, BiotSavart
import plotly.graph_objects as go
from essos.dynamics import Tracing
from essos.coils import fit_dofs_from_coils, Curves, Coils
from time import time

# Initialize Near-Axis field
rc = jnp.array([1,  0.045])
zs = jnp.array([0, -0.045])
etabar = -0.9
nfp = 3
nphi_internal_pyQSC = 51
r_coils = 0.4
r_surface = 0.2
ntheta = 41
ncoils = 4
tmax = 800
nfieldlines_per_core=1
trace_tolerance = 1e-8
num_steps = 22000
order = 4
current_on_each_coil = 2e8
plot_coils_without_Fourier_fit = False
plot_coils_on_2D = False
plot_difference_varphi_phi = False

field_nearaxis = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=nfp, nphi=nphi_internal_pyQSC)

nfieldlines = number_of_processors_to_use*nfieldlines_per_core
current_on_each_coil = current_on_each_coil / ncoils*r_surface**2/1.7**2
r_array = jnp.linspace(1e-5, r_surface, nfieldlines)
n_segments = ntheta

time0 = time()
r_array = jnp.linspace(1e-5, r_surface, nfieldlines)
results = [field_nearaxis.get_boundary(r=r, ntheta=ntheta, nphi=nphi_internal_pyQSC) for r in r_array]
x_2D_surface_array, y_2D_surface_array, z_2D_surface_array, R_2D_surface_array = map(lambda arr: jnp.stack(arr), zip(*results))
print(f"Creating surfaces of constant phi took {time()-time0:.2f} seconds")
time0 = time()
nphi   = ncoils * 2 * nfp
x_2D_coils, y_2D_coils, z_2D_coils, R_2D_coils = field_nearaxis.get_boundary(r=r_coils, ntheta=ntheta, nphi=nphi, phi_is_varphi=True, phi_offset = 2*jnp.pi/nphi/2)
print(f"Creating surfaces of constant varphi took {time()-time0:.2f} seconds")

time0 = time()
coils_gamma = jnp.zeros((ncoils * 2 * nfp, ntheta, 3))
coil_i = 0
for n in range(2*nfp):
    phi_vals = (jnp.arange(ncoils) + 0.5) * (2 * jnp.pi) / ((2) * nfp * ncoils) + 2*jnp.pi/(2*nfp)*n
    phi_idx = (phi_vals / (2*jnp.pi) * nphi).astype(int) % nphi
    for i in phi_idx:
        loop = jnp.stack([x_2D_coils[:, i], y_2D_coils[:, i], z_2D_coils[:, i]], axis=-1)
        coils_gamma = coils_gamma.at[coil_i].set(loop)
        coil_i += 1
print(f"Creating coils_gamma took {time()-time0:.2f} seconds for {ncoils*2*nfp} coils")


time0 = time()
dofs, gamma_uni = fit_dofs_from_coils(coils_gamma[:ncoils], order=order, n_segments=n_segments, assume_uniform=True)
curves = Curves(dofs=dofs, n_segments=n_segments, nfp=nfp, stellsym=True)
coils = Coils(curves=curves, currents=[-current_on_each_coil]*(ncoils))
field_coils_DOFS = BiotSavart(coils)
print(f"Fitting coils took {time()-time0:.2f} seconds")

R0 = R_2D_surface_array[:,0,0]
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
if plot_coils_without_Fourier_fit:
    time0 = time()
    field_coils_gamma = BiotSavart_from_gamma(coils_gamma, currents=current_on_each_coil*jnp.ones(len(coils_gamma)))
    tracing_coils_gamma = block_until_ready(Tracing(field=field_coils_gamma, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                      maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
    print(f"ESSOS tracing coils_gamma took {time()-time0:.2f} seconds")
    trajectories_coils_gamma = tracing_coils_gamma.trajectories
time0 = time()
tracing_coils_DOFS = block_until_ready(Tracing(field=field_coils_DOFS, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                    maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
print(f"ESSOS tracing coils_DOFS took {time()-time0:.2f} seconds")
trajectories_coils_DOFS = tracing_coils_DOFS.trajectories

fig_plotly = go.Figure()

color = "#C2AC95"
colorscale = [[0, color], [1, color]]
fig_plotly.add_surface(
    x=x_2D_surface_array[-1],
    y=y_2D_surface_array[-1],
    z=z_2D_surface_array[-1],
    opacity=0.3,
    colorscale=colorscale,
    showscale=False,
    name='Surface',
    lighting={"specular": 0.3, "diffuse":0.9},
    showlegend=False#True,
)

if plot_coils_without_Fourier_fit:
    for coil in coils_gamma:
        fig_plotly.add_trace(go.Scatter3d(
            x=coil[:, 0],
            y=coil[:, 1],
            z=coil[:, 2],
        mode='lines',
        line=dict(width=10, color='#b87333'),
        name='Coil (Near-Axis)',
        showlegend=False# if coil is not coils_gamma[0] else True
    ))


line_width = 12
line_marker = dict(color="#5B2222", width=line_width)
for i, curve_gamma in enumerate(curves.gamma):
    color = "#93785A"
    fig_plotly.add_trace(go.Scatter3d(
        x=curve_gamma[:, 0],
        y=curve_gamma[:, 1],
        z=curve_gamma[:, 2],
        mode='lines',
        line=line_marker,
        name='Coils',
        showlegend=False#(i==0)
    ))

if plot_coils_without_Fourier_fit:
    for traj in trajectories_coils_gamma:
        fig_plotly.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines',
            line=dict(color='black', width=2),
            name='Fieldline',
            showlegend=False# if traj is not trajectories_coils_gamma[0] else True
        ))

for traj in trajectories_coils_DOFS:
    fig_plotly.add_trace(go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines',
        line=dict(color='black', width=0.2),
        name='Fieldline (Fitted Coils)',
        showlegend=False# if traj is not trajectories_coils_DOFS[0] else True
    ))

# Turn off hover contours on the surface:
fig_plotly.update_traces(contours_x_highlight=False,
                contours_y_highlight=False,
                contours_z_highlight=False,
                selector={"type":"surface"})

# Make x, y, z coordinate scales equal, and turn off more hover stuff
fig_plotly.update_layout(scene={"aspectmode": "data",
                            "xaxis_showspikes": False,
                            "yaxis_showspikes": False,
                            "zaxis_showspikes": False,
                            "xaxis_visible": False,
                            "yaxis_visible": False,
                            "zaxis_visible": False},
                    hovermode=False,
                    margin={"l":0, "r":0, "t":25, "b":0},
                    )

fig_plotly.show()

# Now plot the 2D Poincare plot with Matplotlib
fig2 = plt.figure(figsize=(6, 5))
ax = fig2.add_subplot(111)
shifts = jnp.array([0])
if plot_coils_without_Fourier_fit:
    tracing_coils_gamma.poincare_plot(ax=ax, show=False, shifts=shifts/nfp/2, color='k', s=0.05)
tracing_coils_DOFS.poincare_plot(ax=ax, show=False, shifts=shifts/nfp/2, color='b', s=0.05)

for i in range(nfieldlines):
    ax.plot(R_2D_surface_array[i,:,0], z_2D_surface_array[i,:,0], 'r--', linewidth=1.5, label='Surfaces of Constant Cylindrical Angle' if i==0 else '_nolegend_')
_, _, z_2D_at_coils, R_2D_at_coils = field_nearaxis.get_boundary(r=r_coils, ntheta=ntheta, nphi=nphi)
ax.plot(R_2D_at_coils[:,0], z_2D_at_coils[:,0], 'r--', linewidth=1.5, label='_nolegend_')

x_2D_coil0, y_2D_coil0, z_2D_coil0, R_2D_coil0 = field_nearaxis.get_boundary(r=r_coils, ntheta=ntheta, nphi=nphi, phi_is_varphi=True)
ax.plot(R_2D_coil0[:,0],  z_2D_coil0[:,0],  color='black', alpha=1.0, linewidth=2, label='Surface of Constant Boozer Angle')
if plot_coils_on_2D:
    for coil_number in range(ncoils):
        if plot_coils_without_Fourier_fit:
            R_coils_gamma = jnp.sqrt(coils_gamma[coil_number,:,0]**2 + coils_gamma[coil_number,:,1]**2)
            ax.plot(R_coils_gamma, coils_gamma[coil_number,:,2], color='#b87333', linewidth=2, label='Coils from Near-Axis' if coil_number==0 else '_nolegend_')
        R_curve = jnp.sqrt(curves.gamma[coil_number,:,0]**2 + curves.gamma[coil_number,:,1]**2)
        ax.plot(R_curve, curves.gamma[coil_number,:,2], '-', color='blue', linewidth=2, label='Coils' if coil_number==0 else '_nolegend_')
if plot_coils_without_Fourier_fit:
    ax.plot([], [], color='k', label='Fieldlines from Coils from Near-Axis')
ax.plot([], [], color='b', label='Fieldlines')
ax.legend()
plt.tight_layout()

if plot_difference_varphi_phi:
    itheta = 0 # ntheta // 2
    theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
    varphi1D = jnp.linspace(0, 2 * jnp.pi / nfp, nphi)
    varphi2D, theta2D = jnp.meshgrid(varphi1D, theta1D, indexing='ij')
    phi2D = vmap(lambda theta_row, varphi_row: vmap(lambda theta, varphi: field_nearaxis.phi_of_theta_varphi(r_coils, theta, varphi))(theta_row, varphi_row))(theta2D, varphi2D)
    plt.figure(figsize=(8,6))
    plt.plot(varphi2D[:,itheta], label='varphi')
    plt.plot(phi2D[:,itheta], label='phi')
    plt.legend()
    plt.title(f'Conversion from varphi to phi at theta={itheta}')
    plt.grid()
    plt.tight_layout()

plt.show()