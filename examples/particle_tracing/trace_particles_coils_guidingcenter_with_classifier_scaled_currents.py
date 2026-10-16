import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
import jax
print(jax.devices())
from time import time
from jax import block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart,SurfaceRZFourier
from essos.surfaces import SurfaceClassifier, B_on_surface
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,ONE_EV
from essos.dynamics import Tracing, Particles
from essos.objective_functions import normB_axis
import matplotlib.pyplot as plt

# Input parameters
tmax = 1.e-4
timestep=1.e-8
times_to_trace=1000
nparticles_per_core=500
nparticles = number_of_processors_to_use*nparticles_per_core
atol=1.e-7
rtol=1.e-7
energy=FUSION_ALPHA_PARTICLE_ENERGY


def compute_rz_on_phi(surface, theta, phi=0.0):
    angles = jnp.outer(theta, surface.xm) - phi * surface.xn
    R = jnp.sum(surface.rc * jnp.cos(angles), axis=1)
    Z = jnp.sum(surface.zs * jnp.sin(angles), axis=1)
    return R, Z

    
# Load coils and field
json_file = os.path.join(os.path.dirname(__name__), 'input_files', 'QI_opt_coils_essos_5.json')
coils = Coils.from_json(json_file)
field = BiotSavart(coils)

# Renormalize coils to have B_target=5.7 on axis
# Load surface first to extract magnetic axis
wout_file = os.path.join(os.path.dirname(__name__), 'input_files','wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc')
vmec = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=60, nphi=60, range_torus='full torus')

# Load the ACTUAL magnetic axis at s=0 (not from s=1 surface)
vmec_axis = SurfaceRZFourier.from_wout_file(wout_file, s=0, ntheta=60, nphi=60, range_torus='full period')

# Extract magnetic axis from s=0 surface (m=0 modes give the axis in cylindrical coordinates)
m0_mask_axis = vmec_axis.xm == 0
rc_axis = vmec_axis.rc[m0_mask_axis]
zs_axis = vmec_axis.zs[m0_mask_axis]
xn_axis = vmec_axis.xn[m0_mask_axis]

# Function to compute axis position at given phi
def compute_axis_point(phi):
    """Compute magnetic axis position at toroidal angle phi using s=0 surface"""
    angles = xn_axis * phi
    R_val = jnp.sum(rc_axis * jnp.cos(angles))
    Z = -jnp.sum(zs_axis * jnp.sin(angles))
    return R_val, Z

# Evaluate B on magnetic axis and scale
phi_array = jnp.linspace(0, 2 * jnp.pi, 200)
B_axis_old = jnp.array([field.AbsB(jnp.array([R_val * jnp.cos(phi), R_val * jnp.sin(phi), Z])) 
                         for phi in phi_array 
                         for R_val, Z in [compute_axis_point(phi)]])
B_target = 5.7
scale_factor = B_target / jnp.mean(B_axis_old)
#scale_factor = 1/(4.*jnp.pi*1.e-7) # No scaling for now, to compare with previous results
coils.dofs_currents = coils.dofs_currents * scale_factor
field_scaled = BiotSavart(coils)

timeI=time()
boundary=SurfaceClassifier(vmec,h=0.1)
print(f"ESSOS boundary took {time()-timeI:.2f} seconds")
# Initialize particles from surface axis
particles = Particles.InitializeParticlesAroundSurfaceAxis(
    surface=vmec_axis,                 # axis-defining surface
    boundary_surface=vmec,             # boundary/LCFS-like surface
    n_particles=nparticles,
    distance_from_axis=0.7,            # 60% of local axis->boundary distance
    distance_mode='fraction_to_boundary',
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    energy=energy,
)

print(f"Initialization performed")

# Diagnostic: Compare particle positions to magnetic axis at their phi angles
print("\n" + "="*70)
print("PARTICLE INITIALIZATION DIAGNOSTIC")
print("="*70)
print(f"{'Particle':<12} {'φ (rad)':<12} {'R_particle':<12} {'R_axis':<12} {'ΔR (m)':<12} {'Z_particle':<12} {'Z_axis':<12} {'ΔZ (m)':<12}")
print("-"*70)

for i in range(len(xyz_particles)):
    x, y = xyz_particles[i, 0], xyz_particles[i, 1]
    z = xyz_particles[i, 2]
    R_particle = jnp.sqrt(x**2 + y**2)
    phi_particle = jnp.arctan2(y, x)
    phi_particle = phi_particle if phi_particle >= 0 else phi_particle + 2*jnp.pi
    
    R_axis, Z_axis = compute_axis_point(phi_particle)
    
    delta_R = float(R_particle - R_axis)
    delta_Z = float(z - Z_axis)
    
    print(f"  {i+1:<11} {float(phi_particle):<12.6f} {float(R_particle):<12.6f} {float(R_axis):<12.6f} {delta_R:<12.6f} {float(z):<12.6f} {float(Z_axis):<12.6f} {delta_Z:<12.6f}")

print("="*70)
print("If ΔR and ΔZ are not near zero, particles are NOT initialized at the axis.")
print("="*70 + "\n")

# Plot initial particle positions relative to surface
fig_init = plt.figure(figsize=(16, 7))

# 3D plot: particles on axis
ax_3d = fig_init.add_subplot(121, projection='3d')
vmec.plot(ax=ax_3d, show=False, alpha=0.3, color='blue')
xyz_particles = particles.initial_xyz
ax_3d.scatter(xyz_particles[:, 0], xyz_particles[:, 1], xyz_particles[:, 2], 
              c='red', s=50, marker='o', label='Particles', alpha=0.7)
ax_3d.set_xlabel('X (m)')
ax_3d.set_ylabel('Y (m)')
ax_3d.set_zlabel('Z (m)')
ax_3d.set_title('Initial Particle Positions (distance_from_axis=0.0 m)')
#ax_3d.legend()
ax_3d.set_box_aspect([1,1,1])

# R-Z projection
ax_rz = fig_init.add_subplot(122)
R_particles = jnp.sqrt(xyz_particles[:, 0]**2 + xyz_particles[:, 1]**2)
Z_particles = xyz_particles[:, 2]

# Get phi angles for each particle's initialization
theta_init = jnp.linspace(0, 2 * jnp.pi, 200)
particle_phis_init = []
for i in range(len(xyz_particles)):
    x, y = xyz_particles[i, 0], xyz_particles[i, 1]
    phi = jnp.arctan2(y, x)
    phi = phi if phi >= 0 else phi + 2*jnp.pi
    particle_phis_init.append(float(phi))

# Plot cross sections at particle initialization phi angles
for i, phi_val in enumerate(particle_phis_init):
    R_sec, Z_sec = compute_rz_on_phi(vmec, theta_init, phi=phi_val)
    if i == 0:
        ax_rz.plot(R_sec, Z_sec, 'b-', alpha=0.4, linewidth=1.5, label='Cross sections @ particle φ')
    else:
        ax_rz.plot(R_sec, Z_sec, 'b-', alpha=0.4, linewidth=1.5)

# Plot magnetic axis points at particle initialization phi angles
axis_points_R = []
axis_points_Z = []
for phi_val in particle_phis_init:
    R_ax, Z_ax = compute_axis_point(phi_val)
    axis_points_R.append(R_ax)
    axis_points_Z.append(Z_ax)

# Plot particles colored by their index to match with cross sections
particle_colors_init = [plt.cm.tab10(i % 10) for i in range(len(xyz_particles))]
for i in range(len(xyz_particles)):
    ax_rz.scatter([R_particles[i]], [Z_particles[i]], c=[particle_colors_init[i]], s=100, marker='o', 
                 alpha=0.8, zorder=3, edgecolors='black', linewidths=1, label=f'Particle {i+1}')
    # Plot corresponding axis point with matching color
    ax_rz.scatter([axis_points_R[i]], [axis_points_Z[i]], c=[particle_colors_init[i]], s=80, marker='x', 
                 alpha=0.9, linewidths=2, zorder=4)

# Plot high-resolution magnetic axis curve for reference
axis_R_vals = []
axis_Z_vals = []
phi_axis_check = jnp.linspace(0, 2 * jnp.pi, 200)
for phi_val in phi_axis_check:
    R_ax, Z_ax = compute_axis_point(phi_val)
    axis_R_vals.append(R_ax)
    axis_Z_vals.append(Z_ax)
ax_rz.plot(axis_R_vals, axis_Z_vals, 'k--', linewidth=1.5, alpha=0.5, label='Magnetic Axis (continuous)', zorder=2)

ax_rz.set_xlabel('R (m)')
ax_rz.set_ylabel('Z (m)')
ax_rz.set_title('R-Z Projection: Initial Particle Positions (with Cross Sections)')
#ax_rz.legend()
ax_rz.grid(True, alpha=0.3)
ax_rz.set_aspect('equal')

plt.tight_layout()
plt.savefig('initial_particle_positions.png', dpi=150, bbox_inches='tight')
print("Initial particle positions plot saved to: initial_particle_positions.png")

# Trace in ESSOS
time0 = time()
particle_tracing = Tracing(field=field_scaled, model='GuidingCenterAdaptative', particles=particles,
                  maxtime=tmax, timestep=timestep,times_to_trace=times_to_trace, atol=atol,rtol=rtol,boundary=boundary)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
print(f"Final loss fraction: {particle_tracing.loss_fractions[-1]*100:.2f}%")
particle_trajectories = particle_tracing.trajectories
particle_colors = [plt.cm.tab10(i % 10) for i in range(len(particle_trajectories))]

# Plot trajectories, velocity parallel to the magnetic field, and energy error
fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

vmec.plot(ax=ax1, show=False, alpha=0.4)
coils.plot(ax=ax1, show=False)
particle_tracing.plot(ax=ax1, show=False, n_trajectories_plot=nparticles)

for i, trajectory in enumerate(particle_trajectories):
    ax2.plot(
        particle_tracing.times,
        jnp.abs(particle_tracing.energy()[i]-particles.energy)/particles.energy,
        label=f'Particle {i+1}',
        color=particle_colors[i],
    )
    ax3.plot(
        particle_tracing.times,
        trajectory[:, 3]/particles.total_speed,
        label=f'Particle {i+1}',
        color=particle_colors[i],
    )
    #ax4.plot(jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2), trajectory[:, 2], label=f'Particle {i+1}')
    ax4.plot(
        jnp.sqrt(trajectory[:,0]**2+trajectory[:,1]**2),
        trajectory[:, 2],
        label=f'Particle {i+1}',
        color=particle_colors[i],
    )

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Relative Energy Error')
ax3.set_ylabel(r'$v_{\parallel}/v$')
ax2.legend()
ax3.set_xlabel('Time (s)')
ax3.legend()
ax4.set_xlabel('R (m)')
ax4.set_ylabel('Z (m)')

# Add surface cross sections to ax4
theta = jnp.linspace(0, 2 * jnp.pi, 200)



# Compute cross sections at multiple phi angles
n_sections = 24
phi_angles = jnp.linspace(0, 2*jnp.pi, n_sections, endpoint=False)

# Plot cross sections at all angles
for i, phi_val in enumerate(phi_angles):
    R_section, Z_section = compute_rz_on_phi(vmec, theta, phi=phi_val)
    if i == 0:
        ax4.plot(R_section, Z_section, color='black', linewidth=1.5, linestyle='-', alpha=0.7, label='Surface cross sections')
    else:
        ax4.plot(R_section, Z_section, color='black', linewidth=1.5, linestyle='-', alpha=0.7)

ax4.legend(fontsize=8)
plt.tight_layout()
plt.savefig('tracing_results.png', dpi=300)




# Create animation of particle trajectories with cross sections
print("Creating particle trajectory animation...")
plot_all_cross_sections_in_animation = False
create_individual_particle_videos = True  # Create separate videos for each particle
cross_section_alpha_all = 0.75
cross_section_linewidth_all = 1.8
use_cross_section_based_limits = True
# Pre-compute all cross sections for efficiency
n_sections_movie = 1000
phi_movie_angles = jnp.linspace(0, 2*jnp.pi, n_sections_movie, endpoint=False)
cross_sections = []
for phi_val in phi_movie_angles:
    R_sec, Z_sec = compute_rz_on_phi(vmec, theta, phi=phi_val)
    if jnp.all(jnp.isfinite(R_sec)) and jnp.all(jnp.isfinite(Z_sec)):
        cross_sections.append((R_sec, Z_sec))

if cross_sections:
    all_R_sections = jnp.concatenate([cs[0] for cs in cross_sections])
    all_Z_sections = jnp.concatenate([cs[1] for cs in cross_sections])
    print(
        "Cross sections computed:",
        f"{len(cross_sections)}/{n_sections_movie}",
        f"R=[{float(jnp.min(all_R_sections)):.3f}, {float(jnp.max(all_R_sections)):.3f}]",
        f"Z=[{float(jnp.min(all_Z_sections)):.3f}, {float(jnp.max(all_Z_sections)):.3f}]",
    )
else:
    print("Warning: No finite cross sections computed; using fallback axis limits.")
    

# Create frames
frames = []
import io
for frame_num in range(n_frames):
    idx = frame_indices[frame_num]
    current_time = particle_tracing.times[idx]
    # Create figure for this frame
    fig_frame, ax_frame = plt.subplots(figsize=(10, 10))
    ax_frame.set_xlabel('R (m)', fontsize=12)
    ax_frame.set_ylabel('Z (m)', fontsize=12)
    ax_frame.set_aspect('equal')
    ax_frame.grid(True, alpha=0.3)
    ax_frame.set_xlim(R_min, R_max)
    ax_frame.set_ylim(Z_min, Z_max)
    # Determine which cross section to show based on particle position
    particle_phis = []
    for i, trajectory in enumerate(particle_trajectories):
        if idx < len(trajectory):
            x, y = trajectory[idx, 0], trajectory[idx, 1]
            phi = jnp.arctan2(y, x)
            # Normalize phi to [0, 2*pi]
            phi = phi if phi >= 0 else phi + 2*jnp.pi
            particle_phis.append(phi)
    # By default, keep all cross sections visible throughout the animation.
    if plot_all_cross_sections_in_animation:
        for R_sec, Z_sec in cross_sections:
            ax_frame.plot(
                R_sec,
                Z_sec,
                'k-',
                alpha=cross_section_alpha_all,
                linewidth=cross_section_linewidth_all,
                zorder=1,
            )
    else:
        # Dynamic mode: plot cross sections near particle toroidal locations.
        plotted_angles = set()
        if particle_phis:
            for particle_phi in particle_phis:
                # Find nearest cross section angle
                phi_diffs = jnp.abs(phi_movie_angles - particle_phi)
                # Also check wrapped around 2*pi
                phi_diffs_wrapped = jnp.abs(phi_movie_angles - (particle_phi + 2*jnp.pi))
                phi_diffs = jnp.minimum(phi_diffs, phi_diffs_wrapped)
                nearest_idx = int(jnp.argmin(phi_diffs))
                # Plot only the nearest cross section
                plot_idx = nearest_idx
                if plot_idx not in plotted_angles:
                    R_sec, Z_sec = cross_sections[plot_idx]
                    ax_frame.plot(R_sec, Z_sec, 'k-', alpha=0.6, linewidth=2)
                    plotted_angles.add(plot_idx)
        else:
            # Fallback: plot all cross sections if no particles
            for R_sec, Z_sec in cross_sections:
                ax_frame.plot(R_sec, Z_sec, 'k-', alpha=0.3, linewidth=1.5)
    # Plot particle trajectories and positions
    for i, trajectory in enumerate(particle_trajectories):
        if len(trajectory) > 0:
            # Use the minimum of idx and last available index
            traj_idx = min(idx, len(trajectory) - 1)
            # Plot trajectory up to current time
            R_traj = jnp.sqrt(trajectory[:traj_idx+1, 0]**2 + trajectory[:traj_idx+1, 1]**2)
            Z_traj = trajectory[:traj_idx+1, 2]
            ax_frame.plot(R_traj, Z_traj, color=particle_colors[i], alpha=0.6, linewidth=2)
            # Plot current position
            x, y, z = trajectory[traj_idx, 0], trajectory[traj_idx, 1], trajectory[traj_idx, 2]
            R = jnp.sqrt(x**2 + y**2)
            ax_frame.scatter(
                [R],
                [z],
                c=[particle_colors[i]],
                s=150,
                marker='o',
                zorder=3,
                edgecolors='black',
                linewidths=1.5,
            )
    # Add time text
    ax_frame.text(0.02, 0.98, f'Time: {current_time*1e6:.2f} μs', 
                  transform=ax_frame.transAxes, fontsize=14, 
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax_frame.set_title('Particle Trajectory with Cross Section', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Convert figure to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    # Try imageio first, fall back to PIL
    frame_data = None
    try:
        import imageio.v3 as iio
        frame_data = iio.imread(buf)
    except ImportError:
        try:
            from PIL import Image
            import numpy as np
            buf.seek(0)
            frame_data = np.array(Image.open(buf))
        except Exception as e_pil:
            if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
                print(f"    Warning (R-Z frame {frame_num + 1}): {e_pil}")
            frame_data = None
    except Exception as e_iio:
        if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
            print(f"    Warning (R-Z frame {frame_num + 1}): {e_iio}")
        frame_data = None
    if frame_data is not None:
        frames.append(frame_data)
    plt.close(fig_frame)
    buf.close()
    if (frame_num + 1) % 20 == 0:
        print(f"  Processed {frame_num + 1}/{n_frames} frames")

# Save animation
print(f"Combined R-Z animation: collected {len(frames)}/{n_frames} frames")
if frames:
    try:
        import imageio
        # Try to save as MP4 (requires imageio-ffmpeg which can be pip installed)
        print(f"  Attempting MP4 with {len(frames)} frames at 20 fps...")
        imageio.mimsave('particle_trajectory_animation.mp4', frames, fps=20, codec='libx264')
        print("✓ Animation saved as: particle_trajectory_animation.mp4")
    except Exception as e:
        print(f"  MP4 failed: {e}")
        print(f"  Trying GIF instead...")
        try:
            imageio.mimsave('particle_trajectory_animation.gif', frames, fps=20, loop=0)
            print("✓ Animation saved as: particle_trajectory_animation.gif")
        except Exception as e_gif:
            print(f"  GIF failed: {e_gif}")
            try:
                from PIL import Image
                pil_frames = [Image.fromarray(frame) for frame in frames]
                pil_frames[0].save('particle_trajectory_animation.gif', save_all=True, append_images=pil_frames[1:], duration=50, loop=0)
                print("✓ Animation saved as: particle_trajectory_animation.gif (PIL)")
            except Exception as e_pil:
                print(f"✗ Could not save animation: {e_pil}")


# Create individual particle videos for R-Z
if create_individual_particle_videos:
    print("\n" + "="*60)
    print(f"Creating individual R-Z videos for {len(particle_trajectories)} particles...")
    print(f"Each video will have {n_frames} frames at 20 fps")
    print("="*60)
    for particle_idx in range(len(particle_trajectories)):
        particle_name = f"Particle_{particle_idx + 1}"
        print(f"\nGenerating videos for {particle_name}...")
        
        # R-Z animation for this particle
        print(f"  Creating R-Z animation for {particle_name}...")
        frames_rz_single = []
        for frame_num in range(n_frames):
            idx = frame_indices[frame_num]
            current_time = particle_tracing.times[idx]
            fig_frame, ax_frame = plt.subplots(figsize=(10, 10))
            ax_frame.set_xlabel('R (m)', fontsize=12)
            ax_frame.set_ylabel('Z (m)', fontsize=12)
            ax_frame.set_aspect('equal')
            ax_frame.grid(True, alpha=0.3)
            ax_frame.set_xlim(R_min, R_max)
            ax_frame.set_ylim(Z_min, Z_max)
            # Plot cross section at particle's current position
            trajectory = particle_trajectories[particle_idx]
            if idx < len(trajectory):
                x, y = trajectory[idx, 0], trajectory[idx, 1]
                phi = jnp.arctan2(y, x)
                phi = phi if phi >= 0 else phi + 2*jnp.pi
                phi_diffs = jnp.abs(phi_movie_angles - phi)
                phi_diffs_wrapped = jnp.abs(phi_movie_angles - (phi + 2*jnp.pi))
                phi_diffs = jnp.minimum(phi_diffs, phi_diffs_wrapped)
                nearest_idx = int(jnp.argmin(phi_diffs))
                R_sec, Z_sec = cross_sections[nearest_idx]
                ax_frame.plot(R_sec, Z_sec, 'k-', alpha=0.6, linewidth=2, zorder=1)
            # Plot only this particle's trajectory
            if len(trajectory) > 0:
                # Use the minimum of idx and last available index
                traj_idx = min(idx, len(trajectory) - 1)
                R_traj = jnp.sqrt(trajectory[:traj_idx+1, 0]**2 + trajectory[:traj_idx+1, 1]**2)
                Z_traj = trajectory[:traj_idx+1, 2]
                ax_frame.plot(R_traj, Z_traj, color=particle_colors[particle_idx], alpha=0.7, linewidth=2.5, zorder=2)
                x, y, z = trajectory[traj_idx, 0], trajectory[traj_idx, 1], trajectory[traj_idx, 2]
                R = jnp.sqrt(x**2 + y**2)
                ax_frame.scatter([R], [z], c=[particle_colors[particle_idx]], s=150, marker='o', zorder=3, 
                                edgecolors='black', linewidths=1.5)
            ax_frame.text(0.02, 0.98, f'Time: {current_time*1e6:.2f} μs\n{particle_name}', 
                         transform=ax_frame.transAxes, fontsize=14, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            ax_frame.set_title(f'Particle Trajectory: R-Z Plane - {particle_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frame_data = None
            try:
                import imageio.v3 as iio
                frame_data = iio.imread(buf)
            except ImportError:
                try:
                    from PIL import Image
                    buf.seek(0)
                    frame_data = np.array(Image.open(buf))
                except Exception as e_pil:
                    if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
                        print(f"    Warning ({particle_name} frame {frame_num + 1}): {e_pil}")
                    frame_data = None
            except Exception as e_iio:
                if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
                    print(f"    Warning ({particle_name} frame {frame_num + 1}): {e_iio}")
                frame_data = None
            if frame_data is not None:
                frames_rz_single.append(frame_data)
            plt.close(fig_frame)
            buf.close()
            # Progress update for individual particle videos
            if (frame_num + 1) % 100 == 0:
                print(f"    Processed {frame_num + 1}/{n_frames} frames for {particle_name} (R-Z)")
        # Save individual particle R-Z animation
        print(f"  Saving R-Z animation for {particle_name}... ({len(frames_rz_single)}/{n_frames} frames)")
        if frames_rz_single:
            try:
                import imageio
                print(f"    Attempting MP4...")
                imageio.mimsave(f'particle_trajectory_animation_rz_{particle_name}.mp4', frames_rz_single, fps=20, codec='libx264')
                print(f"    ✓ R-Z animation saved as: particle_trajectory_animation_rz_{particle_name}.mp4")
            except Exception as e:
                print(f"    MP4 failed: {e}")
                try:
                    print(f"    Attempting GIF...")
                    imageio.mimsave(f'particle_trajectory_animation_rz_{particle_name}.gif', frames_rz_single, fps=20, loop=0)
                    print(f"    ✓ R-Z animation saved as: particle_trajectory_animation_rz_{particle_name}.gif")
                except Exception as e_gif:
                    print(f"    GIF failed: {e_gif}")
                    try:
                        from PIL import Image
                        pil_frames = [Image.fromarray(frame) for frame in frames_rz_single]
                        pil_frames[0].save(f'particle_trajectory_animation_rz_{particle_name}.gif', save_all=True, append_images=pil_frames[1:], duration=50, loop=0)
                        print(f"    ✓ R-Z animation saved as: particle_trajectory_animation_rz_{particle_name}.gif (PIL)")
                    except Exception as e_pil:
                        print(f"    ✗ Could not save R-Z animation for {particle_name}: {e_pil}")

    print("\n" + "="*60)
    print(f"Individual particle video generation complete!")
    print(f"Created {len(particle_trajectories)} individual R-Z videos:")
    for particle_idx in range(len(particle_trajectories)):
        particle_name = f"Particle_{particle_idx + 1}"
        print(f"  - particle_trajectory_animation_rz_{particle_name}.mp4 (or .gif)")
    print("="*60)


# Prepare full surface in Cartesian coordinates for X,Y, X,Z, Y,Z projections
print("\nCreating animations for other coordinate projections...")
print("Loading full torus surface for Cartesian projections...")
# Load a new surface with full torus (not just one field period)
vmec_full = SurfaceRZFourier.from_wout_file(wout_file, s=1, ntheta=60, nphi=120, range_torus='full torus')
# Get surface directly from gamma property (already in Cartesian X, Y, Z)
gamma = vmec_full.gamma  # Shape: (nphi, ntheta, 3): [X, Y, Z]
X_surf = gamma[:, :, 0]
Y_surf = gamma[:, :, 1]
Z_surf = gamma[:, :, 2]
print(f"Full torus surface shape: {gamma.shape}")

# Animation 1: X,Y plane (top-down view)
print("Creating X,Y plane animation...")
frames_xy = []
for frame_num in range(n_frames):
    idx = frame_indices[frame_num]
    current_time = particle_tracing.times[idx]
    fig_frame, ax_frame = plt.subplots(figsize=(10, 10))
    ax_frame.set_xlabel('X (m)', fontsize=12)
    ax_frame.set_ylabel('Y (m)', fontsize=12)
    ax_frame.set_aspect('equal')
    ax_frame.grid(True, alpha=0.3)
    # Plot surface as wireframe in X,Y plane
    # Plot lines along theta direction (constant phi)
    for j in range(X_surf.shape[0]):
        ax_frame.plot(X_surf[j, :], Y_surf[j, :], 'b-', alpha=0.3, linewidth=0.8, zorder=0)
    # Plot lines along phi direction (constant theta)
    for i in range(X_surf.shape[1]):
        ax_frame.plot(X_surf[:, i], Y_surf[:, i], 'b-', alpha=0.3, linewidth=0.8, zorder=0)
    # Plot particle trajectories and positions
    for i, trajectory in enumerate(particle_trajectories):
        if len(trajectory) > 0:
            # Use the minimum of idx and last available index
            traj_idx = min(idx, len(trajectory) - 1)
            x_traj = trajectory[:traj_idx+1, 0]
            y_traj = trajectory[:traj_idx+1, 1]
            ax_frame.plot(x_traj, y_traj, color=particle_colors[i], alpha=0.7, linewidth=2.5, zorder=2)
            x, y, z = trajectory[traj_idx, 0], trajectory[traj_idx, 1], trajectory[traj_idx, 2]
            ax_frame.scatter([x], [y], c=[particle_colors[i]], s=150, marker='o', zorder=3, edgecolors='black', linewidths=1.5)
    ax_frame.text(0.02, 0.98, f'Time: {current_time*1e6:.2f} μs', 
                  transform=ax_frame.transAxes, fontsize=14, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax_frame.set_title('Particle Trajectory: X-Y Plane (Top-Down)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    frame_data = None
    try:
        import imageio.v3 as iio
        frame_data = iio.imread(buf)
    except ImportError:
        try:
            from PIL import Image
            import numpy as np
            buf.seek(0)
            frame_data = np.array(Image.open(buf))
        except Exception as e_pil:
            if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
                print(f"    Warning (X-Y frame {frame_num + 1}): {e_pil}")
            frame_data = None
    except Exception as e_iio:
        if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
            print(f"    Warning (X-Y frame {frame_num + 1}): {e_iio}")
        frame_data = None
    if frame_data is not None:
        frames_xy.append(frame_data)
    plt.close(fig_frame)
    buf.close()
    if (frame_num + 1) % 20 == 0:
        print(f"  Processed {frame_num + 1}/{n_frames} frames (X-Y)")

print(f"X-Y animation: collected {len(frames_xy)}/{n_frames} frames")
if frames_xy:
    try:
        import imageio
        print(f"  Attempting MP4 with {len(frames_xy)} frames at 20 fps...")
        imageio.mimsave('particle_trajectory_animation_xy.mp4', frames_xy, fps=20, codec='libx264')
        print("✓ X-Y animation saved as: particle_trajectory_animation_xy.mp4")
    except Exception as e:
        print(f"  MP4 failed: {e}")
        print(f"  Trying GIF instead...")
        try:
            imageio.mimsave('particle_trajectory_animation_xy.gif', frames_xy, fps=20, loop=0)
            print("✓ X-Y animation saved as: particle_trajectory_animation_xy.gif")
        except Exception as e_gif:
            print(f"  GIF failed: {e_gif}")
            try:
                from PIL import Image
                pil_frames = [Image.fromarray(frame) for frame in frames_xy]
                pil_frames[0].save('particle_trajectory_animation_xy.gif', save_all=True, append_images=pil_frames[1:], duration=50, loop=0)
                print("✓ X-Y animation saved as: particle_trajectory_animation_xy.gif (PIL)")
            except Exception as e_pil:
                print(f"✗ Could not save X-Y animation: {e_pil}")


# Animation 2: X,Z plane (side view)
print("Creating X,Z plane animation...")
frames_xz = []
for frame_num in range(n_frames):
    idx = frame_indices[frame_num]
    current_time = particle_tracing.times[idx]
    fig_frame, ax_frame = plt.subplots(figsize=(10, 10))
    ax_frame.set_xlabel('X (m)', fontsize=12)
    ax_frame.set_ylabel('Z (m)', fontsize=12)
    ax_frame.set_aspect('equal')
    ax_frame.grid(True, alpha=0.3)
    # Plot surface as wireframe in X,Z plane
    # Plot lines along theta direction (constant phi)
    for j in range(X_surf.shape[0]):
        ax_frame.plot(X_surf[j, :], Z_surf[j, :], 'g-', alpha=0.3, linewidth=0.8, zorder=0)
    # Plot lines along phi direction (constant theta)
    for i in range(X_surf.shape[1]):
        ax_frame.plot(X_surf[:, i], Z_surf[:, i], 'g-', alpha=0.3, linewidth=0.8, zorder=0)
    # Plot particle trajectories and positions
    for i, trajectory in enumerate(particle_trajectories):
        if len(trajectory) > 0:
            # Use the minimum of idx and last available index
            traj_idx = min(idx, len(trajectory) - 1)
            x_traj = trajectory[:traj_idx+1, 0]
            z_traj = trajectory[:traj_idx+1, 2]
            ax_frame.plot(x_traj, z_traj, color=particle_colors[i], alpha=0.7, linewidth=2.5, zorder=2)
            x, y, z = trajectory[traj_idx, 0], trajectory[traj_idx, 1], trajectory[traj_idx, 2]
            ax_frame.scatter([x], [z], c=[particle_colors[i]], s=150, marker='o', zorder=3, edgecolors='black', linewidths=1.5)
    ax_frame.text(0.02, 0.98, f'Time: {current_time*1e6:.2f} μs', 
                  transform=ax_frame.transAxes, fontsize=14, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax_frame.set_title('Particle Trajectory: X-Z Plane (Side View)', fontsize=14, fontweight='bold')
    plt.tight_layout() 
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    frame_data = None
    try:
        import imageio.v3 as iio
        frame_data = iio.imread(buf)
    except ImportError:
        try:
            from PIL import Image
            import numpy as np
            buf.seek(0)
            frame_data = np.array(Image.open(buf))
        except Exception as e_pil:
            if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
                print(f"    Warning (X-Z frame {frame_num + 1}): {e_pil}")
            frame_data = None
    except Exception as e_iio:
        if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
            print(f"    Warning (X-Z frame {frame_num + 1}): {e_iio}")
        frame_data = None
    if frame_data is not None:
        frames_xz.append(frame_data)
    plt.close(fig_frame)
    buf.close()
    if (frame_num + 1) % 20 == 0:
        print(f"  Processed {frame_num + 1}/{n_frames} frames (X-Z)")

print(f"X-Z animation: collected {len(frames_xz)}/{n_frames} frames")
if frames_xz:
    try:
        import imageio
        print(f"  Attempting MP4 with {len(frames_xz)} frames at 20 fps...")
        imageio.mimsave('particle_trajectory_animation_xz.mp4', frames_xz, fps=20, codec='libx264')
        print("✓ X-Z animation saved as: particle_trajectory_animation_xz.mp4")
    except Exception as e:
        print(f"  MP4 failed: {e}")
        print(f"  Trying GIF instead...")
        try:
            imageio.mimsave('particle_trajectory_animation_xz.gif', frames_xz, fps=20, loop=0)
            print("✓ X-Z animation saved as: particle_trajectory_animation_xz.gif")
        except Exception as e_gif:
            print(f"  GIF failed: {e_gif}")
            try:
                from PIL import Image
                pil_frames = [Image.fromarray(frame) for frame in frames_xz]
                pil_frames[0].save('particle_trajectory_animation_xz.gif', save_all=True, append_images=pil_frames[1:], duration=50, loop=0)
                print("✓ X-Z animation saved as: particle_trajectory_animation_xz.gif (PIL)")
            except Exception as e_pil:
                print(f"✗ Could not save X-Z animation: {e_pil}")

# Animation 3: Y,Z plane (back view)
print("Creating Y,Z plane animation...")
frames_yz = []
for frame_num in range(n_frames):
    idx = frame_indices[frame_num]
    current_time = particle_tracing.times[idx]
    fig_frame, ax_frame = plt.subplots(figsize=(10, 10))
    ax_frame.set_xlabel('Y (m)', fontsize=12)
    ax_frame.set_ylabel('Z (m)', fontsize=12)
    ax_frame.set_aspect('equal')
    ax_frame.grid(True, alpha=0.3)
    # Plot surface as wireframe in Y,Z plane
    # Plot lines along theta direction (constant phi)
    for j in range(Y_surf.shape[0]):
        ax_frame.plot(Y_surf[j, :], Z_surf[j, :], 'r-', alpha=0.3, linewidth=0.8, zorder=0)
    # Plot lines along phi direction (constant theta)
    for i in range(Y_surf.shape[1]):
        ax_frame.plot(Y_surf[:, i], Z_surf[:, i], 'r-', alpha=0.3, linewidth=0.8, zorder=0)
    # Plot particle trajectories and positions
    for i, trajectory in enumerate(particle_trajectories):
        if len(trajectory) > 0:
            # Use the minimum of idx and last available index
            traj_idx = min(idx, len(trajectory) - 1)
            y_traj = trajectory[:traj_idx+1, 1]
            z_traj = trajectory[:traj_idx+1, 2]
            ax_frame.plot(y_traj, z_traj, color=particle_colors[i], alpha=0.7, linewidth=2.5, zorder=2)
            x, y, z = trajectory[traj_idx, 0], trajectory[traj_idx, 1], trajectory[traj_idx, 2]
            ax_frame.scatter([y], [z], c=[particle_colors[i]], s=150, marker='o', zorder=3, edgecolors='black', linewidths=1.5)
    ax_frame.text(0.02, 0.98, f'Time: {current_time*1e6:.2f} μs', 
                  transform=ax_frame.transAxes, fontsize=14, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax_frame.set_title('Particle Trajectory: Y-Z Plane (Back View)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    frame_data = None
    try:
        import imageio.v3 as iio
        frame_data = iio.imread(buf)
    except ImportError:
        try:
            from PIL import Image
            import numpy as np
            buf.seek(0)
            frame_data = np.array(Image.open(buf))
        except Exception as e_pil:
            if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
                print(f"    Warning (Y-Z frame {frame_num + 1}): {e_pil}")
            frame_data = None
    except Exception as e_iio:
        if (frame_num + 1) <= 5 or (frame_num + 1) % 100 == 0:
            print(f"    Warning (Y-Z frame {frame_num + 1}): {e_iio}")
        frame_data = None
    if frame_data is not None:
        frames_yz.append(frame_data)
    plt.close(fig_frame)
    buf.close()
    if (frame_num + 1) % 20 == 0:
        print(f"  Processed {frame_num + 1}/{n_frames} frames (Y-Z)")

print(f"Y-Z animation: collected {len(frames_yz)}/{n_frames} frames")
if frames_yz:
    try:
        import imageio
        print(f"  Attempting MP4 with {len(frames_yz)} frames at 20 fps...")
        imageio.mimsave('particle_trajectory_animation_yz.mp4', frames_yz, fps=20, codec='libx264')
        print("✓ Y-Z animation saved as: particle_trajectory_animation_yz.mp4")
    except Exception as e:
        print(f"  MP4 failed: {e}")
        print(f"  Trying GIF instead...")
        try:
            imageio.mimsave('particle_trajectory_animation_yz.gif', frames_yz, fps=20, loop=0)
            print("✓ Y-Z animation saved as: particle_trajectory_animation_yz.gif")
        except Exception as e_gif:
            print(f"  GIF failed: {e_gif}")
            try:
                from PIL import Image
                pil_frames = [Image.fromarray(frame) for frame in frames_yz]
                pil_frames[0].save('particle_trajectory_animation_yz.gif', save_all=True, append_images=pil_frames[1:], duration=50, loop=0)
                print("✓ Y-Z animation saved as: particle_trajectory_animation_yz.gif (PIL)")
            except Exception as e_pil:
                print(f"✗ Could not save Y-Z animation: {e_pil}")

print("\nAll animations created successfully!")


B_final= B_on_surface(vmec,field)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(vmec.phi2d,vmec.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB.png', dpi=300)

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')




# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing


phi_poincare=jnp.pi / 2
tmax = 50000000000
#tmax = 10000
nfieldlines_per_core = 40
nfieldlines = nfieldlines_per_core * number_of_processors_to_use

trace_tolerance = 1e-7
num_steps = 60000
R0=jnp.linspace(16.5, 17.4, nfieldlines) # Starting points for field line tracing, evenly spaced in R
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T
time0 = time()

fieldline_tracing = block_until_ready(Tracing(
    field=field,
    model='FieldLineAdaptative',
    initial_conditions=initial_xyz,
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance
))

print(f"ESSOS tracing took {time() - time0:.2f} seconds")

fieldline_trajectories = fieldline_tracing.trajectories
traj = fieldline_trajectories[0]
R, phi, Z = traj[:, 0], traj[:, 1], traj[:, 2]

phi_u = jnp.unwrap(phi)
phi0_cross = jnp.where((phi_u[:-1] < 0) & (phi_u[1:] >= 0))[0]
phi90_cross = jnp.where((phi_u[:-1] < phi_poincare) & (phi_u[1:] >= phi_poincare))[0]

theta = jnp.linspace(0, 2 * jnp.pi, 200)

def compute_rz_on_phi(surface, theta, phi=0.0):
    angles = jnp.outer(theta, surface.xm) - phi * surface.xn
    R = jnp.sum(surface.rc * jnp.cos(angles), axis=1)
    Z = jnp.sum(surface.zs * jnp.sin(angles), axis=1)
    return R, Z


# # Contours from true VMEC surface
phi_poincare_45 = jnp.pi / 4
phi_poincare_135 = 3 * jnp.pi / 4

R0_true, Z0_true = compute_rz_on_phi(vmec, theta, phi=0.0)
R45_true, Z45_true = compute_rz_on_phi(vmec, theta, phi=phi_poincare_45)
R90_true, Z90_true = compute_rz_on_phi(vmec, theta, phi=phi_poincare)
R135_true, Z135_true = compute_rz_on_phi(vmec, theta, phi=phi_poincare_135)

fig, ((ax0, ax45), (ax90, ax135)) = plt.subplots(2, 2, figsize=(14, 12))

# φ = 0 subplot
fieldline_tracing.poincare_plot(ax=ax0, show=False, shifts=[0])
ax0.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 0$")
ax0.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
ax0.set_xlabel("R")
ax0.set_ylabel("Z")
ax0.set_title("Poincaré + Surfaces @ φ = 0")
ax0.legend()
ax0.axis("equal")

# φ = π/4 subplot
fieldline_tracing.poincare_plot(ax=ax45, show=False, shifts=[phi_poincare_45])
ax45.plot(R45_opt, Z45_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = \pi/4$")
ax45.plot(R45_true, Z45_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/4$")
ax45.set_xlabel("R")
ax45.set_ylabel("Z")
ax45.set_title("Poincaré + Surfaces @ φ = π/4")
ax45.legend()
ax45.axis("equal")

# φ = π/2 subplot
fieldline_tracing.poincare_plot(ax=ax90, show=False, shifts=[phi_poincare])
ax90.plot(R90_opt, Z90_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2$")
ax90.plot(R90_true, Z90_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")
ax90.set_xlabel("R")
ax90.set_ylabel("Z")
ax90.set_title("Poincaré + Surfaces @ φ = π/2")
ax90.legend()
ax90.axis("equal")

# φ = 3π/4 subplot
fieldline_tracing.poincare_plot(ax=ax135, show=False, shifts=[phi_poincare_135])
ax135.plot(R135_opt, Z135_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 3\pi/4$")
ax135.plot(R135_true, Z135_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 3\pi/4$")
ax135.set_xlabel("R")
ax135.set_ylabel("Z")
ax135.set_title("Poincaré + Surfaces @ φ = 3π/4")
ax135.legend()
ax135.axis("equal")

plt.tight_layout()
plt.savefig('optimize_qfm_surface_poincare.png', dpi=300)