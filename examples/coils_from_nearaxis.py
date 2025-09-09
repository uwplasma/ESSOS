import os
number_of_processors_to_use = 1 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import jax.numpy as jnp
from essos.fields import near_axis, BiotSavart_from_gamma, BiotSavart
import plotly.graph_objects as go
from essos.dynamics import Tracing
from essos.coils import fit_dofs_from_coils, Curves, Coils
import matplotlib.pyplot as plt
from time import time
from jax import block_until_ready

# Initialize Near-Axis field
rc = jnp.array([1,  0.045])
zs = jnp.array([0, -0.045])
etabar = -0.9
nfp = 3
nphi = 51
r_surface = 0.1
r_max_poincare = 0.05
r_coils = 0.5
ntheta = 41
ncoils = 6
tmax = 5000
nfieldlines_per_core=1
nfieldlines = number_of_processors_to_use*nfieldlines_per_core
trace_tolerance = 1e-10
num_steps = 7*tmax
order = 8
current_on_each_coil = 1e5
n_segments = 61

field_nearaxis = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=nfp, nphi=nphi)

nphi   = 151
time0 = time()
x_2D_surface, y_2D_surface, z_2D_surface, R_2D_surface = field_nearaxis.get_boundary(r=r_surface, ntheta=ntheta, nphi=nphi)
x_2D_coils, y_2D_coils, z_2D_coils, R_2D_coils = field_nearaxis.get_boundary_varphi_theta(r=r_coils, ntheta=ntheta, nphi=nphi)
print(f"Creating surfaces took {time()-time0:.2f} seconds")

# import matplotlib.pyplot as plt
# from jax import vmap
# theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
# varphi1D = jnp.linspace(0, 2 * jnp.pi, nphi)
# varphi2D, theta2D = jnp.meshgrid(varphi1D, theta1D, indexing='ij')
# phi2D = vmap(lambda theta_row, varphi_row: vmap(lambda theta, varphi: field_nearaxis.phi_of_theta_varphi(r_coils, theta, varphi))(theta_row, varphi_row))(theta2D, varphi2D)
# plt.plot(varphi2D[:,ntheta // 2], label='varphi')
# plt.plot(phi2D[:,ntheta // 2], label='phi')
# plt.legend()
# plt.title('Conversion from varphi to phi at theta=0')
# plt.grid()
# plt.show()
# exit()

time0 = time()
coils_gamma = jnp.zeros((ncoils * 2 * nfp, ntheta, 3))
coil_i = 0
for n in range(2*nfp):
    phi_vals = (jnp.arange(ncoils) + 0.5) * (2 * jnp.pi) / ((2) * nfp * ncoils) + 2*jnp.pi/(2*nfp)*n
    # phi_vals = jnp.linspace(2*jnp.pi/(2*nfp)*n, 2*jnp.pi/(2*nfp)*(n+1), ncoils, endpoint=False)
    phi_idx = (phi_vals / (2*jnp.pi) * nphi).astype(int) % nphi
    for i in phi_idx:
        loop = jnp.stack([x_2D_coils[:, i], y_2D_coils[:, i], z_2D_coils[:, i]], axis=-1)  # (ntheta,3)
        coils_gamma = coils_gamma.at[coil_i].set(loop)
        coil_i += 1
print(f"Creating coils_gamma took {time()-time0:.2f} seconds for {ncoils*2*nfp} coils")
        
def d_dtheta_fft(f_theta):
    """
    f_theta: (..., ntheta) periodic samples over θ in [0, 2π)
    Returns ∂f/∂θ with same shape.
    """
    ntheta = f_theta.shape[-1]
    # k = 0, 1, ..., ntheta-1 mapped to integer Fourier modes with period 2π
    k = jnp.fft.fftfreq(ntheta, d=1.0/ntheta)  # integers (…, -2, -1, 0, 1, 2, …)
    Fk = jnp.fft.fft(f_theta, axis=-1)
    dF = (1j * k) * Fk  # for period 2π, ∂/∂θ multiplies by i*k
    return jnp.fft.ifft(dF, axis=-1).real * (2*jnp.pi)

# Apply along the θ axis to each Cartesian component
coils_gamma_dash = jnp.stack([
    d_dtheta_fft(coils_gamma[..., 0]),
    d_dtheta_fft(coils_gamma[..., 1]),
    d_dtheta_fft(coils_gamma[..., 2]),
], axis=-1)  # (Ncoils, ntheta, 3)
field_coils_gamma = BiotSavart_from_gamma(coils_gamma, coils_gamma_dash, currents=current_on_each_coil*jnp.ones(len(coils_gamma)))

time0 = time()
dofs, gamma_uni = fit_dofs_from_coils(coils_gamma[:ncoils], order=order, n_segments=n_segments, assume_uniform=True)
curves = Curves(dofs=dofs, n_segments=n_segments, nfp=nfp, stellsym=True)
coils = Coils(curves=curves, currents=[current_on_each_coil]*(ncoils))
field_coils_DOFS = BiotSavart(coils)
print(f"Fitting coils took {time()-time0:.2f} seconds")


R0 = jnp.linspace(rc[0]+rc[1], rc[0]+rc[1]+r_max_poincare, nfieldlines)
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
time0 = time()
tracing_coils_gamma = block_until_ready(Tracing(field=field_coils_gamma, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                  maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
print(f"ESSOS tracing coils_gamma took {time()-time0:.2f} seconds")
trajectories_coils_gamma = tracing_coils_gamma.trajectories
time0 = time()
tracing_coils_DOFS = block_until_ready(Tracing(field=field_coils_DOFS, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                    maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
print(f"ESSOS tracing coils_DOFS took {time()-time0:.2f} seconds")
trajectories_coils_DOFS = tracing_coils_DOFS.trajectories

# Plot 3D surface and coils with Plotly
fig_plotly = go.Figure()

# Add surface
fig_plotly.add_surface(
    x=x_2D_surface,
    y=y_2D_surface,
    z=z_2D_surface,
    opacity=0.5,
    colorscale='Greys',
    showscale=False,
    name='Surface'
)

# Add coils from near-axis
for coil in coils_gamma:
    fig_plotly.add_trace(go.Scatter3d(
        x=coil[:, 0],
        y=coil[:, 1],
        z=coil[:, 2],
        mode='lines',
        line=dict(color='#b87333', width=5),
        name='Coil (Near-Axis)'
    ))

# Add fitted curves
for curve_gamma in curves.gamma:
    fig_plotly.add_trace(go.Scatter3d(
        x=curve_gamma[:, 0],
        y=curve_gamma[:, 1],
        z=curve_gamma[:, 2],
        mode='lines',
        line=dict(color='blue', width=2, dash='dash'),
        name='Fitted Curve'
    ))

# Add fieldline traces
for traj in trajectories_coils_gamma:
    fig_plotly.add_trace(go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines',
        line=dict(color='black', width=2),
        name='Fieldline'
    ))

# Add fieldline traces from fitted coils
for traj in trajectories_coils_DOFS:
    fig_plotly.add_trace(go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines',
        line=dict(color='red', width=1),
        name='Fieldline (Fitted Coils)'
    ))

fig_plotly.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    title='3D Surface, Coils, and Fieldlines (Plotly)'
)
fig_plotly.show()

# Now plot the 2D Poincare plot with Matplotlib (ax2 only)
fig2 = plt.figure(figsize=(6, 5))
ax2 = fig2.add_subplot(111)
shifts = jnp.array([0, jnp.pi])
tracing_coils_gamma.poincare_plot(ax=ax2, show=False, shifts=shifts/nfp/2, color='k', s=0.05)
tracing_coils_DOFS.poincare_plot(ax=ax2, show=False, shifts=shifts/nfp/2, color='r', s=0.05)
for i, shift1 in enumerate(shifts):
    phi_idx = int(shift1/ (2*jnp.pi) * nphi) % nphi
    ax2.plot(R_2D_surface[:,phi_idx], z_2D_surface[:,phi_idx], color='grey', alpha=1.0, linewidth=2, label='Surfaces' if i==0 else '_nolegend_')
for coil_number in range(ncoils):
    R_coils_gamma = jnp.sqrt(coils_gamma[coil_number,:,0]**2 + coils_gamma[coil_number,:,1]**2)
    ax2.plot(R_coils_gamma, coils_gamma[coil_number,:,2], color='#b87333', linewidth=2, label='Coils from Near-Axis' if coil_number==0 else '_nolegend_')
    R_curve = jnp.sqrt(curves.gamma[coil_number,:,0]**2 + curves.gamma[coil_number,:,1]**2)
    ax2.plot(R_curve, curves.gamma[coil_number,:,2], '--', color='blue', linewidth=1, label='Coil fitted to Fourier' if coil_number==0 else '_nolegend_')
ax2.plot([], [], color='k', label='Fieldlines from Coils from Near-Axis')
ax2.plot([], [], color='r', label='Fieldlines from Coils from Fourier')
ax2.legend()
plt.tight_layout()
plt.show()

# fig = plt.figure(figsize=(9, 5))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122)
# ax1.plot_surface(
#     x_2D_surface, y_2D_surface, z_2D_surface,
#     color='grey', alpha=0.5, linewidth=0, antialiased=True, shade=True)
# for coil in coils_gamma:
#     ax1.plot(coil[:, 0], coil[:, 1], coil[:, 2], color='#b87333', linewidth=2)
# for curve_gamma in curves.gamma:
#     ax1.plot(curve_gamma[:, 0], curve_gamma[:, 1], curve_gamma[:, 2], '--', color='blue', linewidth=1)

# shifts = jnp.array([0, jnp.pi])#, jnp.pi/2, 3*jnp.pi/2])
# tracing_coils_gamma.plot(ax=ax1, show=False)
# tracing_coils_gamma.poincare_plot(ax=ax2, show=False, shifts=shifts/nfp/2, color='k', s=0.05)#, jnp.pi/2, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4])
# # tracing_coils_DOFS.plot(ax=ax1, show=False)
# # tracing_coils_DOFS.poincare_plot(ax=ax2, show=False, shifts=shifts/nfp/2, color='r')#, jnp.pi/2, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4])
# for i, shift1 in enumerate(shifts):
#     phi_idx = int(shift1/ (2*jnp.pi) * nphi) % nphi
#     ax2.plot(R_2D_surface[:,phi_idx], z_2D_surface[:,phi_idx], color='grey',      alpha=1.0, linewidth=2, label='Surfaces' if i==0 else '_nolegend_')
#     # ax2.plot(R_2D_coils[:,phi_idx],   z_2D_coils[:,phi_idx], '--',  color="#5633b8", alpha=1.0, linewidth=2, label='Coils' if i==0 else '_nolegend_')
# for coil_number in range(ncoils):
#     R_coils_gamma = jnp.sqrt(coils_gamma[coil_number,:,0]**2 + coils_gamma[coil_number,:,1]**2)
#     ax2.plot(R_coils_gamma, coils_gamma[coil_number,:,2], color='#b87333', linewidth=2, label='Coils from Near-Axis' if coil_number==0 else '_nolegend_')
#     # R_curve = jnp.sqrt(curves.gamma[coil_number,:,0]**2 + curves.gamma[coil_number,:,1]**2)
#     # ax2.plot(R_curve, curves.gamma[coil_number,:,2], '--', color='blue', linewidth=1, label='Coil fitted to Fourier' if coil_number==0 else '_nolegend_')
# ax2.plot([], [], color='k', label='Fieldlines from Coils from Near-Axis')
# # ax2.plot([], [], color='r', label='Fieldlines from Coils from Fourier')
# ax2.legend()
# plt.tight_layout()
# plt.show()