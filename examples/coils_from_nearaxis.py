import os
number_of_processors_to_use = 4 # Parallelization, this should divide nfieldlines
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
r_surface = 0.15
r_max_poincare = 0.2
r_coils = 0.45
ntheta = 41
ncoils = 6
tmax = 1000
nfieldlines_per_core=1
nfieldlines = number_of_processors_to_use*nfieldlines_per_core
trace_tolerance = 1e-10
num_steps = 3000
order = 3
current_on_each_coil = 1e5
n_segments = 60

field_nearaxis = near_axis(rc=rc, zs=zs, etabar=etabar, nfp=nfp)

nphi   = 151
time0 = time()
x_2D_surface, y_2D_surface, z_2D_surface, R_2D_surface = field_nearaxis.get_boundary(r=r_surface, ntheta=ntheta, nphi=nphi)
x_2D_coils, y_2D_coils, z_2D_coils, R_2D_coils = field_nearaxis.get_boundary(r=r_coils, ntheta=ntheta, nphi=nphi)
print(f"Creating surfaces took {time()-time0:.2f} seconds")

time0 = time()
coils_gamma = jnp.zeros((ncoils * 2 * nfp, ntheta, 3))
coil_i = 0
ntheta_Fourier = 20
R_2D, Z_2D, phi0_2D = field_nearaxis.Frenet_to_cylindrical(r=r_coils, ntheta=ntheta_Fourier)
x_2D = R_2D * jnp.cos(phi0_2D)
y_2D = R_2D * jnp.sin(phi0_2D)
x_2D_boozer = jnp.zeros((ntheta_Fourier, field_nearaxis.nphi))
y_2D_boozer = jnp.zeros((ntheta_Fourier, field_nearaxis.nphi))
z_2D_boozer = jnp.zeros((ntheta_Fourier, field_nearaxis.nphi))
for i in range(ntheta_Fourier):
    (self,array,point)
    x_spline_this_theta = field_nearaxis.interpolated_array_at_point(
                                 jnp.append(field_nearaxis.varphi, 2 * np.pi / field_nearaxis.nfp),
                                 jnp.append(x_2D[i,:], x_2D[i,0]), bc_type='periodic')
    y_spline_this_theta = field_nearaxis.interpolated_array_at_point(
                                 jnp.append(field_nearaxis.varphi, 2 * np.pi / field_nearaxis.nfp),
                                 jnp.append(y_2D[i,:], y_2D[i,0]), bc_type='periodic')
    z_spline_this_theta = field_nearaxis.interpolated_array_at_point(
                                 jnp.append(field_nearaxis.varphi, 2 * np.pi / field_nearaxis.nfp),
                                 jnp.append(z_2D[i,:], z_2D[i,0]), bc_type='periodic')
    x_2D_boozer.at[i].set(x_spline_this_theta)
    y_2D_boozer.at[i].set(y_spline_this_theta)
    z_2D_boozer.at[i].set(z_spline_this_theta)
for n in range(1):#2*nfp):
    phi_vals = jnp.linspace(2*jnp.pi/(2*nfp)*n, 2*jnp.pi/(2*nfp)*(n+1), ncoils, endpoint=False)
    phi_idx = (phi_vals / (2*jnp.pi) * nphi).astype(int) % nphi
    for i in phi_idx:
        loop = jnp.stack([x_2D_coils[:, i], y_2D_coils[:, i], z_2D_coils[:, i]], axis=-1)  # (ntheta,3)
        coils_gamma = coils_gamma.at[coil_i].set(loop)
        coil_i += 1
print(f"Creating coils_gamma took {time()-time0:.2f} seconds")
        
# def d_dtheta_fft(f_theta):
#     """
#     f_theta: (..., ntheta) periodic samples over θ in [0, 2π)
#     Returns ∂f/∂θ with same shape.
#     """
#     ntheta = f_theta.shape[-1]
#     # k = 0, 1, ..., ntheta-1 mapped to integer Fourier modes with period 2π
#     k = jnp.fft.fftfreq(ntheta, d=1.0/ntheta)  # integers (…, -2, -1, 0, 1, 2, …)
#     Fk = jnp.fft.fft(f_theta, axis=-1)
#     dF = (1j * k) * Fk  # for period 2π, ∂/∂θ multiplies by i*k
#     return jnp.fft.ifft(dF, axis=-1).real

# # Apply along the θ axis to each Cartesian component
# coils_gamma_dash = jnp.stack([
#     d_dtheta_fft(coils_gamma[..., 0]),
#     d_dtheta_fft(coils_gamma[..., 1]),
#     d_dtheta_fft(coils_gamma[..., 2]),
# ], axis=-1)  # (Ncoils, ntheta, 3)
# field_coils_gamma = BiotSavart_from_gamma(coils_gamma, coils_gamma_dash, currents=current_on_each_coil*jnp.ones(len(coils_gamma)))

time0 = time()
dofs, gamma_uni = fit_dofs_from_coils(coils_gamma[:ncoils+1], order=order, n_segments=n_segments, assume_uniform=True)
curves = Curves(dofs=dofs, n_segments=n_segments, nfp=nfp, stellsym=True)
coils = Coils(curves=curves, currents=[current_on_each_coil]*(ncoils+1))
field_coils_DOFS = BiotSavart(coils)
print(f"Fitting coils took {time()-time0:.2f} seconds")


R0 = jnp.linspace(rc[0]+rc[1], rc[0]+rc[1]+r_max_poincare, nfieldlines)
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
# time0 = time()
# tracing_coils_gamma = block_until_ready(Tracing(field=field_coils_gamma, model='FieldLineAdaptative', initial_conditions=initial_xyz,
#                   maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
# print(f"ESSOS tracing coils_gamma took {time()-time0:.2f} seconds")
# trajectories_coils_gamma = tracing_coils_gamma.trajectories
time0 = time()
tracing_coils_DOFS = block_until_ready(Tracing(field=field_coils_DOFS, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                    maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
print(f"ESSOS tracing coils_DOFS took {time()-time0:.2f} seconds")
trajectories_coils_DOFS = tracing_coils_DOFS.trajectories


fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
ax1.plot_surface(
    x_2D_surface, y_2D_surface, z_2D_surface,
    color='grey', alpha=0.5, linewidth=0, antialiased=True, shade=True)
for coil in coils_gamma:
    ax1.plot(coil[:, 0], coil[:, 1], coil[:, 2], color='#b87333', linewidth=2)
for curve_gamma in curves.gamma:
    ax1.plot(curve_gamma[:, 0], curve_gamma[:, 1], curve_gamma[:, 2], '--', color='blue', linewidth=1)

shifts = jnp.array([0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
# tracing_coils_gamma.plot(ax=ax1, show=False)
# tracing_coils_gamma.poincare_plot(ax=ax2, show=False, shifts=shifts/nfp/2, color='k')#, jnp.pi/2, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4])
tracing_coils_DOFS.plot(ax=ax1, show=False)
tracing_coils_DOFS.poincare_plot(ax=ax2, show=False, shifts=shifts/nfp/2, color='r')#, jnp.pi/2, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4])
for i, shift1 in enumerate(shifts):
    phi_idx = int(shift1/ (2*jnp.pi) * nphi) % nphi
    ax2.plot(R_2D_surface[:,phi_idx], z_2D_surface[:,phi_idx], color='grey',      alpha=1.0, linewidth=2, label='Surfaces' if i==0 else '_nolegend_')
    # ax2.plot(R_2D_coils[:,phi_idx],   z_2D_coils[:,phi_idx],   color='#b87333', alpha=1.0, linewidth=2, label='Coils' if i==0 else '_nolegend_')
for coil_number in range(ncoils+1):
    R_coils_gamma = jnp.sqrt(coils_gamma[coil_number,:,0]**2 + coils_gamma[coil_number,:,1]**2)
    ax2.plot(R_coils_gamma, coils_gamma[coil_number,:,2], color='#b87333', linewidth=2, label='Coils from Near-Axis' if coil_number==0 else '_nolegend_')
    R_curve = jnp.sqrt(curves.gamma[coil_number,:,0]**2 + curves.gamma[coil_number,:,1]**2)
    ax2.plot(R_curve, curves.gamma[coil_number,:,2], '--', color='blue', linewidth=1, label='Coil fitted to Fourier' if coil_number==0 else '_nolegend_')
# ax2.plot([], [], color='k', label='Fieldlines from Coils from Near-Axis')
ax2.plot([], [], color='r', label='Fieldlines from Coils from Fourier')
ax2.legend()
plt.tight_layout()
plt.show()