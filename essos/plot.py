
import numpy as np

def projection2D_small(R, r, Trajectories: jnp.ndarray, show=True, save_as=None, close=False):
    plt.figure()
    for i in range(10):
        X_particle = Trajectories[i, :, 0]
        Y_particle = Trajectories[i, :, 1]
        Z_particle = Trajectories[i, :, 2]
        R_particle = jnp.sqrt(X_particle**2 + Y_particle**2)
        plt.plot(R_particle, Z_particle)

    theta = jnp.linspace(0, 2*np.pi, 100)
    x = r*jnp.cos(theta)+R
    y = r*jnp.sin(theta)
    plt.plot(x, y, color="lightgrey")
    
    plt.xlim(R-1.2*r, R+1.2*r)
    plt.ylim(-1.2*r, 1.2*r)
    plt.gca().set_aspect('equal')

    plt.title("Projection of the Trajectories (poloidal view)")
    plt.xlabel("r [m]")
    plt.ylabel("z [m]")

    # Save the plot
    if save_as is not None:
        plt.savefig(save_as)

    # Show the plot
    if show:
        plt.show()
        
    if close:
        plt.close()
        
from matplotlib.collections import LineCollection

def projection2D(R, r, Trajectories: jnp.ndarray, show=True, save_as=None, close=False):
    fig, ax = plt.subplots()
    
    # Convert JAX arrays to NumPy (more compatible with Matplotlib)
    Trajectories = np.asarray(Trajectories)
    
    # Compute cylindrical radius R_particle and use LineCollection
    R_particle = np.sqrt(Trajectories[:, :, 0]**2 + Trajectories[:, :, 1]**2)
    Z_particle = Trajectories[:, :, 2]

    # Prepare line segments for faster plotting
    segments = [np.column_stack([R_particle[i], Z_particle[i]]) for i in range(len(Trajectories))]
    lc = LineCollection(segments, colors='b', linewidths=0.8)
    ax.add_collection(lc)

    # Plot the reference circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x = r * np.cos(theta) + R
    y = r * np.sin(theta)
    ax.plot(x, y, color="lightgrey")

    ax.set_xlim(R - 1.2 * r, R + 1.2 * r)
    ax.set_ylim(-1.2 * r, 1.2 * r)
    ax.set_aspect('equal')
    ax.set_title("Projection of the Trajectories (poloidal view)")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")

    if save_as is not None:
        plt.savefig(save_as)

    if show:
        plt.show()
    
    if close:
        plt.close()


def projection2D_top(R, r, Trajectories: jnp.ndarray, show=True, save_as=None, close=False):
    fig, ax = plt.subplots()

    Trajectories = np.asarray(Trajectories)

    # Precompute and plot the reference circles
    theta = np.linspace(0, 2*np.pi, 100)
    for radius in [(R - r), (R + r)]:
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        ax.plot(x, y, color="lightgrey")

    # Prepare line segments for faster plotting
    segments = [np.column_stack([Trajectories[i, :, 0], Trajectories[i, :, 1]]) for i in range(len(Trajectories))]
    lc = LineCollection(segments, colors='b', linewidths=0.8)
    ax.add_collection(lc)

    ax.set_xlim(-1.2 * (R + r), 1.2 * (R + r))
    ax.set_ylim(-1.2 * (R + r), 1.2 * (R + r))
    ax.set_aspect('equal')
    ax.set_title("Projection of the Trajectories (top view)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if save_as is not None:
        plt.savefig(save_as)

    if show:
        plt.show()

    if close:
        plt.close()

def Plot_3D_trajectories(R, r, Trajectories: jnp.ndarray, show=True, save_as=None, close=False):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import cnames
    from matplotlib import animation

    x_par=Trajectories[:,:,0]
    y_par=Trajectories[:,:,1]
    z_par=Trajectories[:,:,2]
    N_trajectories=x_par.shape[0]

    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # choose a different color for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

    # set up lines and points
    lines = sum([ax.plot([], [], [], '-', c=c)
                for c in colors], [])
    pts = sum([ax.plot([], [], [], 'o', c=c)
            for c in colors], [])

    # prepare the axes limits
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((0, 10))

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # initialization function: plot the background of each frame
    def init():
        for line, pt in zip(lines, pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])
        return lines + pts

    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        # we'll step two time-steps per frame.  This leads to nice results.
        i = (2 * i) % x_par.shape[1]

        for line, pt, xi, yi,zi in zip(lines, pts, x_par,y_par,z_par):
            x= xi[:i].T
            y= yi[:i].T
            z= zi[:i].T
            line.set_data(x, y)
            line.set_3d_properties(z)

            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])

        ax.view_init(30, 0.3 * i)
        fig.canvas.draw()
        return lines + pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=500, interval=30, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

    plt.show()


def plot(self, trajectories = None, show=False, title="", save_as=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n_coils = jnp.size(self.curves, 0)
    xlims = [jnp.min(self.gamma[0, :, 0]), jnp.max(self.gamma[0, :, 0])]
    ylims = [jnp.min(self.gamma[0, :, 1]), jnp.max(self.gamma[0, :, 1])]
    zlims = [jnp.min(self.gamma[0, :, 2]), jnp.max(self.gamma[0, :, 2])]
    for i in range(0, n_coils):
        color = "orangered" if i < n_coils/((1+int(self._stellsym))*self._nfp) else "darkgray"
        ax.plot3D(self.gamma[i, :, 0], self.gamma[i, :,  1], self.gamma[i, :, 2], color=color, zorder=10, linewidth=5)
        if i != 0:
            xlims = [min(xlims[0], jnp.min(self.gamma[i, :, 0])), max(xlims[1], jnp.max(self.gamma[i, :, 0]))]
            ylims = [min(ylims[0], jnp.min(self.gamma[i, :, 1])), max(ylims[1], jnp.max(self.gamma[i, :, 1]))]
            zlims = [min(zlims[0], jnp.min(self.gamma[i, :, 2])), max(zlims[1], jnp.max(self.gamma[i, :, 2]))]

    # Calculate zoomed limits
    zoom_factor=0.7
    x_center = (xlims[0] + xlims[1]) / 2
    y_center = (ylims[0] + ylims[1]) / 2
    z_center = (zlims[0] + zlims[1]) / 2
    x_range = (xlims[1] - xlims[0]) * zoom_factor / 2
    y_range = (ylims[1] - ylims[0]) * zoom_factor / 2
    z_range = (zlims[1] - zlims[0]) * zoom_factor / 2
    ax.set_xlim([x_center - x_range, x_center + x_range])
    ax.set_ylim([y_center - y_range, y_center + y_range])
    ax.set_zlim([z_center - z_range, z_center + z_range])

    if trajectories is not None:
        assert isinstance(trajectories, jnp.ndarray)
        for i in range(jnp.size(trajectories, 0)):
            ax.plot3D(trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2], zorder=0)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_xlim(xlims)
    # ax.set_ylim(ylims)
    # ax.set_zlim(zlims)

    ax.set_aspect('equal')
    ax.locator_params(axis='z', nbins=3)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.axis('off')
    ax.grid(False)

    plt.tight_layout()

    # Save the plot
    if save_as is not None:
        plt.savefig(save_as, transparent=True)
    
    # Show the plot
    if show:
        plt.show()
    plt.close()