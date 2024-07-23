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

os.mkdir("images") if not os.path.exists("images") else None

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
x = [5.8816768669949235, -0.0001091781773779678, 2.9426538084165696, -0.0006121931796838491, 0.0005488831652268538, 1.1700514303927547, -3.846204946440075e-05, 0.5854790792369962, 0.0007283374762158154, -0.00032084656581056047, 0.0001646041074676944, -3.0003310202122244, 0.0006047324370085148, -0.00042555013540153966, -0.0006396618232154825, 4.988660226798091, -6.1443615346557e-05, 2.4948909560523433, 6.478701619722714e-05, -0.0006480203964514419, 3.2928787356556644, -0.00011788627795246733, 1.6669849337476699, 7.960341750094052e-05, 0.0010048939352689755, 0.0014223007560808846, -2.9993493557197306, 0.0001873509988254814, -0.0006047079448731891, -0.0005231007977422194]
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

plt.savefig("images/compare_lorentz_gc.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300, transparent=True)
plt.show()
