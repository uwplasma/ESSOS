import os
import jax
import sys
from time import time
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
number_of_cores = 14
number_of_particles_per_core = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.append("..")
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, remove_3D_axes
from MagneticField import B, B_norm

n_curves=2
nfp=5
order=3
r = 2
A = 3. # Aspect ratio
R = A*r

r_init = r/4
maxtime = 1e-5
timesteps=1000
nparticles = len(jax.devices())*1
n_segments=100

particles = Particles(nparticles)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([1e7]*n_curves))

# If there is a previous optimization, use the following x to set the dofs and currents
x = [5.924702620314064, -0.0025519824337822145, 1.9745555877622545, 0.004077400730383376, 8.668008416872513e-05, 0.0034401995743134992, 1.796247508439101e-05, 4.298691119635643e-05, 0.0008401643249744287, 0.9379368992260929, 0.018098380157840842, 0.31300470559201754, -0.04087018312708382, -0.002655274674303366, -0.019306099340735728, -0.00015807353477626462, 0.0009538130813538795, 0.00016009458962658942, 0.004944042080719642, -1.9994608872184694, -0.00025544998157077327, 0.0016116019011788475, -0.0025502098991872437, 0.0015062491556655025, 0.0002353264645362859, 0.00107182365489318, 1.275056963582103e-05, 5.349208162900211, -0.004662648498102098, 1.7865107907581779, 0.017320250409466748, 0.002670438026381779, 0.0091904275163497, 0.0011876340035089595, -0.0003347171710933947, 0.00041658258327954937, 2.7227551260788028, 0.010135499377536931, 0.9017534355512505, -0.035742880945879574, -0.0019559059555094324, -0.016855551396964426, 9.2710296168846e-06, 0.0013500924496970927, 0.00036574620086413773, 0.001927819368657137, -1.9993801895347771, -0.00045623599427501804, -0.0006626854693094752, -0.0010561432434404775, -2.9632102413688948e-06, 0.0004554209377657044, 0.0008101938557124701, 0.00014615004859748506]
len_dofs = len(jnp.ravel(stel.dofs))
dofs = jnp.reshape(jnp.array(x)[:len_dofs], shape=stel.dofs.shape)
stel.dofs = dofs
if len(x)>len_dofs:
    print("Setting currents")
    currents = jnp.array(x)[len_dofs:]
    stel.currents = currents

times = jnp.linspace(0, maxtime, timesteps)
x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
v0 = jnp.zeros((3, nparticles))
for i in range(nparticles):
    B0 = B(jnp.array([x0[i],y0[i],z0[i]]), curve_segments=curves.gamma(n_segments), currents=stel.currents)
    b0 = B0/jnp.linalg.norm(B0)
    perp_vector_1 = jnp.array([0, b0[2], -b0[1]])
    perp_vector_1_normalized = perp_vector_1/jnp.linalg.norm(perp_vector_1)
    perp_vector_2 = jnp.cross(b0, perp_vector_1)
    v0 = v0.at[:,i].set(vpar0[i]*b0 + vperp0[i]*(perp_vector_1_normalized/jnp.sqrt(2)+perp_vector_2/jnp.sqrt(2)))
normB0 = jnp.apply_along_axis(B_norm, 0, jnp.array([x0, y0, z0]), stel.gamma(), stel.currents)
μ = particles.mass*vperp0**2/(2*normB0)

time0 = time()
trajectories_lorentz = stel.trace_trajectories_lorentz(particles, initial_values=jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
print(f"Time to trace trajectories Lorentz: {time()-time0:.2f} seconds")

time0 = time()
trajectories_guiding_center = stel.trace_trajectories(particles, initial_values=jnp.array([x0, y0, z0, vpar0, vperp0]), maxtime=maxtime, timesteps=timesteps, n_segments=n_segments)
print(f"Time to trace trajectories Guiding Center: {time()-time0:.2f} seconds")

# Define the plot setup function
def setup_plots():
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    
    # 3D subplot for x, y, z
    ax00 = fig.add_subplot(331, projection='3d')
    axs[0, 0].remove()  # Remove the existing 2D subplot
    axs[0, 0] = ax00  # Replace it with the 3D subplot
    axs[0, 0].set_xlabel("x [m]")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].set_zlabel("z [m]")
    
    # Other subplots
    axs[0, 1].set_xlabel("time [s]")
    axs[0, 1].set_ylabel("relative error in energy")
    
    axs[0, 2].set_xlabel("time [s]")
    axs[0, 2].set_ylabel("v_par/v_th [m/s]")
    
    axs[1, 0].set_xlabel("time [s]")
    axs[1, 0].set_ylabel("x, y [m]")
    
    axs[1, 1].set_xlabel("time [s]")
    axs[1, 1].set_ylabel("z [m]")
    
    axs[1, 2].set_xlabel("time [s]")
    axs[1, 2].set_ylabel("|B| [T]")
    
    return fig, axs

# Plot trajectories
def plot_trajectories(axs, times, trajectories_lorentz, trajectories_guiding_center, colors):
    ax_3D, ax_E, ax_vpar, ax_xy, ax_z, ax_B = axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]
    
    for i, color in enumerate(colors):
        # if i==0: continue
        x_L, y_L, z_L, vx_L, vy_L, vz_L = trajectories_lorentz[i, :, :].transpose()
        x_gc, y_gc, z_gc, vpar_gc = trajectories_guiding_center[i, :, :].transpose()
        
        # x_L = jnp.where(jnp.abs(x_L) > R, jnp.nan, x_L)
        # y_L = jnp.where(jnp.abs(y_L) > R, jnp.nan, y_L)
        # z_L = jnp.where(jnp.abs(z_L) > r, jnp.nan, z_L)
        
        # x_gc = jnp.where(jnp.abs(x_gc) > R, jnp.nan, x_gc)
        # y_gc = jnp.where(jnp.abs(y_gc) > R, jnp.nan, y_gc)
        # z_gc = jnp.where(jnp.abs(z_gc) > r, jnp.nan, z_gc)

        ax_xy.plot(times, x_gc, ':', color=color)
        ax_xy.plot(times, x_L, '-', color=color)
        ax_xy.plot(times, y_gc, ':', color=color)
        ax_xy.plot(times, y_L, '-', color=color)
        ax_z.plot(times, z_gc, ':', color=color)
        ax_z.plot(times, z_L, '-', color=color)
        
        ax_3D.plot(x_L, y_L, z_L, color=color)
        ax_3D.plot(x_gc, y_gc, z_gc, color=color)
        
        vth = jnp.sqrt(2*particles.energy/particles.mass)
        ax_vpar.plot(times, vpar_gc/vth, ':', color=color)
        B_L = jnp.apply_along_axis(B, 1, trajectories_lorentz[i, :, :3], stel.gamma(), stel.currents)
        vpar_L = jnp.sum(jnp.array([vx_L, vy_L, vz_L])*B_L.transpose(), axis=0)/jnp.linalg.norm(B_L, axis=1)
        ax_vpar.plot(times, vpar_L/vth, '-', color=color)
        
        v02_L = vx_L[0]**2 + vy_L[0]**2 + vz_L[0]**2
        ax_E.plot(times, (vx_L**2 + vy_L**2 + vz_L**2 - v02_L) / v02_L, color=color)
        normB_gc = jnp.apply_along_axis(B_norm, 1, trajectories_guiding_center[i, :, :3], stel.gamma(), stel.currents)
        ax_E.plot(times, (jnp.array(μ[i]*normB_gc + 0.5*particles.mass*trajectories_guiding_center[i, :, 3]**2)-particles.energy)/particles.energy, ':', color=color)
        
        ax_B.plot(times, normB_gc, ':', color=color)
        normB_L = jnp.apply_along_axis(B_norm, 1, trajectories_lorentz[i, :, :3], stel.gamma(), stel.currents)
        ax_B.plot(times, normB_L, '-', color=color)
        

# Main plotting routine
time0 = time()
fig, axs = setup_plots()
colors = cm.viridis(jnp.linspace(0, 1, nparticles))

plot_trajectories(axs, times, trajectories_lorentz, trajectories_guiding_center, colors)

gamma = stel.gamma()
for i in range(n_curves * 2 * curves._nfp):
    color = "orangered" if i < n_curves else "lightgrey"
    axs[0, 0].plot(gamma[i, :, 0], gamma[i, :, 1], gamma[i, :, 2], color=color, zorder=10)

set_axes_equal(axs[0, 0])
axs[0, 0].view_init(elev=20., azim=30)
axs[0, 0].set_box_aspect([1,1,1], zoom=2)
remove_3D_axes(axs[0,0])
# plt.tight_layout()

print(f"Plotting trajectories: {time()-time0:.2f} seconds")

plt.show()
