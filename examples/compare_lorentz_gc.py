import os
import jax
import sys
from time import time
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.append("..")
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal
from MagneticField import B, B_norm

n_curves=2
order=1
nfp=5
r = 1.7
A = 6. # Aspect ratio
R = A*r

r_init = r/5
maxtime = 1e-5
timesteps=1000
nparticles = len(jax.devices())*1
n_segments=100

particles = Particles(nparticles)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([3e6]*n_curves))

x = [10.079587427194943, 1.122863000267699, 2.907328295420131, 1.5998643461488353, 0.02626007651272216, 0.25905672210431535, 0.04564879798786802, -2.562297450094322, -0.01682551629916378, 9.294225769606797, 0.5836407986438034, 4.45371468509117, 4.893531853273665, 0.6751075586506592, 1.472411126324177, 0.30563312003758925, -1.9335920133261997, -0.012623988302193818]
dofs = jnp.reshape(jnp.array(x), shape=stel.dofs.shape)
stel.dofs = dofs

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
    ax11 = fig.add_subplot(331, projection='3d')
    axs[0, 0] = ax11
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
        x_L, y_L, z_L, vx_L, vy_L, vz_L = trajectories_lorentz[i, :, :].transpose()
        x_gc, y_gc, z_gc, vpar_gc = trajectories_guiding_center[i, :, :].transpose()
        
        x_L = jnp.where(jnp.abs(x_L) > R, jnp.nan, x_L)
        y_L = jnp.where(jnp.abs(y_L) > R, jnp.nan, y_L)
        z_L = jnp.where(jnp.abs(z_L) > r, jnp.nan, z_L)
        
        x_gc = jnp.where(jnp.abs(x_gc) > R, jnp.nan, x_gc)
        y_gc = jnp.where(jnp.abs(y_gc) > R, jnp.nan, y_gc)
        z_gc = jnp.where(jnp.abs(z_gc) > r, jnp.nan, z_gc)

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

gamma = curves.gamma()
ax11 = axs[0, 0]
for i in range(n_curves * 2 * curves._nfp):
    color = "orangered" if i < n_curves else "lightgrey"
    ax11.plot(gamma[i, :, 0], gamma[i, :, 1], gamma[i, :, 2], color=color, zorder=10)

set_axes_equal(ax11)
# plt.tight_layout()

print(f"Plotting trajectories: {time()-time0:.2f} seconds")

plt.show()
