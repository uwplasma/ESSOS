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

n_curves=3
nfp=4
order=4
r = 2
A = 3. # Aspect ratio
R = A*r
r_init = r/4
maxtime = 3.0e-5
timesteps=2000
nparticles = len(jax.devices())*number_of_particles_per_core
n_segments=100
coil_current = 1e7

maximum_integration_time_lorentz = 1e-5
maximum_time_index_lorentz = int(timesteps*maximum_integration_time_lorentz/maxtime)
particles = Particles(nparticles)

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))

# If there is a previous optimization, use the following x to set the dofs and currents
x = [5.761005018206357, -0.03217874387454478, 2.1523026880117087, -0.056394290858425, -0.02670514231469476, 0.015223740575546827, 0.016962232236878434, 0.011142162350601857, 0.03220289079562492, 1.0679336076895585, 0.3346908609647495, -0.05220035728901279, -0.0901512362478669, -0.18614321908151527, -0.024297725459829677, 0.027173903654063306, -0.01860025439798505, 0.03669747690774177, 0.1879019266794462, -2.091736886412198, -0.05392483502340483, 0.08912911428267228, -0.10111848598983585, 0.04629800524586173, 0.010678637591110996, 0.06204574085058528, 0.0044765973154110615, 5.74960658184721, -0.054461086556134575, 1.8727504160944906, -0.06794331153376382, 0.11602416578531412, 0.0038426400254395043, 0.06703670114570646, 0.03590987831645892, -0.001442262509199517, 2.242105275434312, 0.06102291717623049, 0.3333577991752048, -0.06426005433019982, 0.020420214260041033, 0.004396710432917432, 0.04773346847156976, 0.003605034838076655, 0.018344412877616446, 0.18006068264684005, -1.7205651824068793, -0.02865963818533225, -0.08910711847676438, -0.11783143645339124, -0.08206090969971559, -0.012962330585414388, 0.03714291494689949, 0.03280914892389674, 5.2190719441478794, -0.30416794091004506, 1.5010103887299697, -0.11391829312212226, 0.087148139491984, -0.038617224358240804, 0.010769947460525826, 0.02889572569755693, 0.03930168791321329, 3.6790278529819664, 0.39266418087374577, 1.615032428279713, 0.23486110051072231, 0.09094227716074844, 0.11533185302232794, -0.001191251536658603, -0.0026942433010710472, 0.06803584521313358, -0.02154419964486222, -2.2717082398949286, 0.08685187920776022, -0.09525254803135513, 0.19037270526124922, 0.01939872750376672, 0.08289485318647245, -0.017833863082218595, 0.03517967586686895]
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
def plot_trajectories(ax_3D, axs, times, trajectories_lorentz, trajectories_guiding_center, colors):
    ax_E, ax_vpar, ax_rz, ax_B = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    
    for i, color in enumerate(colors):
        # if i==0: continue
        x_L, y_L, z_L, vx_L, vy_L, vz_L = trajectories_lorentz[i, :maximum_time_index_lorentz, :].transpose()
        x_gc, y_gc, z_gc, vpar_gc = trajectories_guiding_center[i, :, :].transpose()
        
        # x_L = jnp.where(jnp.abs(x_L) > R, jnp.nan, x_L)
        # y_L = jnp.where(jnp.abs(y_L) > R, jnp.nan, y_L)
        # z_L = jnp.where(jnp.abs(z_L) > r, jnp.nan, z_L)
        
        # x_gc = jnp.where(jnp.abs(x_gc) > R, jnp.nan, x_gc)
        # y_gc = jnp.where(jnp.abs(y_gc) > R, jnp.nan, y_gc)
        # z_gc = jnp.where(jnp.abs(z_gc) > r, jnp.nan, z_gc)

        ax_3D.plot(x_L, y_L, z_L, color=color)
        ax_3D.plot(x_gc, y_gc, z_gc, color=color)
        
        r_gc = jnp.sqrt(x_gc**2 + y_gc**2)
        r_L = jnp.sqrt(x_L**2 + y_L**2)
        
        ax_rz.plot(r_gc, z_gc, ':', color=color)
        ax_rz.plot(r_L, z_L, '-', color=color)
        
        vth = jnp.sqrt(2*particles.energy/particles.mass)
        v02_L = vx_L[0]**2 + vy_L[0]**2 + vz_L[0]**2
        B_L = jnp.apply_along_axis(B, 1, trajectories_lorentz[i, :maximum_time_index_lorentz, :3], stel.gamma(), stel.currents)
        vpar_L = jnp.sum(jnp.array([vx_L, vy_L, vz_L])*B_L.transpose(), axis=0)/jnp.linalg.norm(B_L, axis=1)
        normB_gc = jnp.apply_along_axis(B_norm, 1, trajectories_guiding_center[i, :, :3], stel.gamma(), stel.currents)
        normB_L = jnp.apply_along_axis(B_norm, 1, trajectories_lorentz[i, :maximum_time_index_lorentz, :3], stel.gamma(), stel.currents)
        
        ax_vpar.plot(times, vpar_gc/vth, ':', color=color)
        ax_vpar.plot(times[:maximum_time_index_lorentz], vpar_L/vth, '-', color=color)
        
        ax_E.plot(times, (jnp.array(μ[i]*normB_gc + 0.5*particles.mass*trajectories_guiding_center[i, :, 3]**2)-particles.energy)/particles.energy, ':', color=color)
        ax_E.plot(times[:maximum_time_index_lorentz], (vx_L**2 + vy_L**2 + vz_L**2 - v02_L) / v02_L, color=color)
        
        ax_B.plot(times, normB_gc, ':', color=color)
        ax_B.plot(times[:maximum_time_index_lorentz], normB_L, '-', color=color)
    
# Main plotting routine
time0 = time()
fig1, ax_3D, fig, axs = setup_plots()
colors = cm.viridis(jnp.linspace(0, 1, nparticles))

plot_trajectories(ax_3D, axs, times, trajectories_lorentz, trajectories_guiding_center, colors)

gamma = stel.gamma()
for i in range(n_curves * 2 * curves._nfp):
    color = "orangered" if i < n_curves else "lightgrey"
    ax_3D.plot(gamma[i, :, 0], gamma[i, :, 1], gamma[i, :, 2], color=color, zorder=10)

set_axes_equal(ax_3D)
ax_3D.view_init(elev=20., azim=30)
ax_3D.set_box_aspect([1,1,1], zoom=2)
remove_3D_axes(ax_3D)
# plt.tight_layout()

print(f"Plotting trajectories: {time()-time0:.2f} seconds")

plt.show()
