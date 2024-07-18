import os
import jax
import sys
from time import time
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
number_of_cores = 10
number_of_particles_per_core = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.insert(1, os.path.dirname(os.getcwd()))
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, remove_3D_axes
from MagneticField import B, B_norm

n_curves=2
nfp=4
order=2
r = 3
A = 2 # Aspect ratio
R = A*r
r_init = r/4
maxtime = 1.0e-4
timesteps_guiding_center=max(1000,int(maxtime/1.0e-8))
timesteps_lorentz=int(maxtime/1.0e-10)
nparticles = len(jax.devices())*number_of_particles_per_core
n_segments=100
coil_current = 7e6
n_points_plot_L =  5000
n_points_plot_gc = 1000

particles = Particles(nparticles)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

# If there is a previous optimization, use the following x to set the dofs and currents
x = [5.670857828727212, -0.15278687271697447, 3.0051577628994672, 0.058659716598112324, -0.06196801841987212, 1.483323046067383, 0.7322720450703063, 0.49065665854358725, 0.06360107573730789, 0.12439902281113421, -0.1677669987569341, -3.156931494231882, -0.015975143350275774, 0.14605238705444348, 0.06165122311747209, 5.2428748018381, -0.45306069690874273, 2.4824923204025886, 0.20010222030174762, 0.06637750119488427, 3.585276861404259, 0.5649805559211224, 1.3636712498413397, -0.21901960625652148, 0.11673438384067335, -0.06195902933943496, -2.9812284607777184, 0.010448180413384236, -0.18130634969893872, 0.03427506775496329]
len_dofs = len(jnp.ravel(stel.dofs))
dofs = jnp.reshape(jnp.array(x)[:len_dofs], shape=stel.dofs.shape)
stel.dofs = dofs
if len(x)>len_dofs:
    print("Setting currents")
    currents = jnp.array(x)[len_dofs:]
    stel.currents = currents

x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
v0 = jnp.zeros((3, nparticles))
normB0 = jnp.zeros((nparticles,))
for i in range(nparticles):
    B0 = B(jnp.array([x0[i],y0[i],z0[i]]), curve_segments=stel.gamma(n_segments), currents=stel.currents)
    b0 = B0/jnp.linalg.norm(B0)
    perp_vector_1 = jnp.array([0, b0[2], -b0[1]])
    perp_vector_1_normalized = perp_vector_1/jnp.linalg.norm(perp_vector_1)
    perp_vector_2 = jnp.cross(b0, perp_vector_1_normalized)
    v0 = v0.at[:,i].set(vpar0[i]*b0 + vperp0[i]*(perp_vector_1_normalized/jnp.sqrt(2)+perp_vector_2/jnp.sqrt(2)))
    normB0 = normB0.at[i].set(B_norm(jnp.array([x0[i],y0[i],z0[i]]), stel.gamma(n_segments), stel.currents))
μ = particles.mass*vperp0**2/(2*normB0)

time0 = time()
trajectories_lorentz = stel.trace_trajectories_lorentz(particles, initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime=maxtime, timesteps=timesteps_lorentz, n_segments=n_segments)
print(f"Time to trace trajectories Lorentz: {time()-time0:.2f} seconds")

time0 = time()
trajectories_guiding_center = stel.trace_trajectories(particles, initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime=maxtime, timesteps=timesteps_guiding_center, n_segments=n_segments)
print(f"Time to trace trajectories Guiding Center: {time()-time0:.2f} seconds")

# Define the plot setup function
def setup_plots():
    fig1 = plt.figure()
    ax_3D = plt.axes(projection='3d')
    ax_3D.set_xlabel("x [m]")
    ax_3D.set_ylabel("y [m]")
    ax_3D.set_zlabel("z [m]")
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    
    axs[0, 0].set_xlabel("time [s]")
    axs[0, 0].set_ylabel("relative error in energy")
    
    axs[0, 1].set_xlabel("time [s]")
    axs[0, 1].set_ylabel("v_par/v_th [m/s]")
    
    axs[1, 0].set_xlabel("R [m]")
    axs[1, 0].set_ylabel("Z [m]")
    
    axs[1, 1].set_xlabel("time [s]")
    axs[1, 1].set_ylabel("|B| [T]")
    
    return fig1, ax_3D, fig, axs

# Plot trajectories
def plot_trajectories(ax_3D, axs, times_gc, times_L, trajectories_lorentz, trajectories_guiding_center, colors):
    ax_E, ax_vpar, ax_rz, ax_B = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    
    N_points_L = len(times_L)
    N_points_gc = len(times_gc)
    step_L = int(jnp.ceil(N_points_L / n_points_plot_L))
    step_gc = int(jnp.ceil(N_points_gc / n_points_plot_gc))
    for i, color in enumerate(colors):
        # if i==0: continue
        x_L, y_L, z_L, vx_L, vy_L, vz_L = trajectories_lorentz[i, ::step_L, :].transpose()
        x_gc, y_gc, z_gc, vpar_gc = trajectories_guiding_center[i, ::step_gc, :].transpose()
        
        x_L = jnp.where(jnp.abs(x_L) > 1.5*(R+r), jnp.nan, x_L)
        y_L = jnp.where(jnp.abs(y_L) > 1.5*(R+r), jnp.nan, y_L)
        z_L = jnp.where(jnp.abs(z_L) > 1.5*r, jnp.nan, z_L)
        
        x_gc = jnp.where(jnp.abs(x_gc) > 1.5*(R+r), jnp.nan, x_gc)
        y_gc = jnp.where(jnp.abs(y_gc) > 1.5*(R+r), jnp.nan, y_gc)
        z_gc = jnp.where(jnp.abs(z_gc) > 1.5*r, jnp.nan, z_gc)

        # ax_3D.plot(x_L, y_L, z_L, color=color)
        ax_3D.plot(x_gc, y_gc, z_gc, color=color)
        
        r_gc = jnp.sqrt(x_gc**2 + y_gc**2)
        r_L = jnp.sqrt(x_L**2 + y_L**2)
        
        ax_rz.plot(r_gc, z_gc, ':', color=color)
        ax_rz.plot(r_L, z_L, '-', color=color)
        
        vth = jnp.sqrt(2*particles.energy/particles.mass)
        v02_L = vx_L[0]**2 + vy_L[0]**2 + vz_L[0]**2
        B_L = jnp.apply_along_axis(B, 1, jnp.array([x_L, y_L, z_L]).transpose(), stel.gamma(n_segments), stel.currents)
        vpar_L = jnp.sum(jnp.array([vx_L, vy_L, vz_L])*B_L.transpose(), axis=0)/jnp.linalg.norm(B_L, axis=1)
        normB_gc = jnp.apply_along_axis(B_norm, 1, jnp.array([x_gc, y_gc, z_gc]).transpose(), stel.gamma(n_segments), stel.currents)
        normB_L = jnp.apply_along_axis(B_norm, 1, jnp.array([x_L, y_L, z_L]).transpose(), stel.gamma(n_segments), stel.currents)
        
        ax_vpar.plot(times_gc[::step_gc], vpar_gc/vth, ':', color=color)
        ax_vpar.plot(times_L[::step_L], vpar_L/vth, '-', color=color)

        ax_E.plot(times_gc[::step_gc], (jnp.array(μ[i]*normB_gc + 0.5*particles.mass*vpar_gc**2)-particles.energy)/particles.energy, ':', color=color)
        ax_E.plot(times_L[::step_L], (vx_L**2 + vy_L**2 + vz_L**2 - v02_L) / v02_L, color=color)
        
        ax_B.plot(times_gc[::step_gc], normB_gc, ':', color=color)
        ax_B.plot(times_L[::step_L], normB_L, '-', color=color)

# Main plotting routine
time0 = time()
fig1, ax_3D, fig, axs = setup_plots()
colors = cm.viridis(jnp.linspace(0, 1, nparticles))

plot_trajectories(ax_3D, axs, jnp.linspace(0, maxtime, timesteps_guiding_center), jnp.linspace(0, maxtime, timesteps_lorentz), trajectories_lorentz, trajectories_guiding_center, colors)

gamma = stel.gamma(n_segments)
for i in range(n_curves * 2 * curves._nfp):
    color = "orangered" if i < n_curves else "lightgrey"
    ax_3D.plot(gamma[i, :, 0], gamma[i, :, 1], gamma[i, :, 2], color=color, zorder=10, linewidth=2)

set_axes_equal(ax_3D)
ax_3D.view_init(elev=20., azim=30)
ax_3D.set_box_aspect([1,1,1], zoom=2)
remove_3D_axes(ax_3D)
# plt.tight_layout()

print(f"Plotting trajectories: {time()-time0:.2f} seconds")

plt.show()
