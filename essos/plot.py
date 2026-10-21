import numpy as np
import matplotlib.pyplot as plt

def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_scalar_map_2d(surface, values, ax=None, label=None, title=None, symmetric=False, cmap=None, contours=0, show=False):
    """
    Plot a scalar on the surface grid as a 2D map in (phi, theta).

    surface   : SurfaceRZFourier / vmec.surface (uses phi2d, theta2d)
    values    : shape (nphi, ntheta) or flattened (nphi*ntheta,)
    symmetric : True -> color scale centered at 0 (for B.n, QS residual)
    contours  : number of contour lines (0 = none)
    Returns (fig, ax).
    """

    phi   = np.asarray(surface.phi2d)
    theta = np.asarray(surface.theta2d)
    vals  = np.asarray(values).reshape(phi.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    if symmetric:
        vmax = np.max(np.abs(vals)); vmin = -vmax
        cmap = cmap or "coolwarm"
    else:
        vmin, vmax = vals.min(), vals.max()
        cmap = cmap or "viridis"

    mesh = ax.pcolormesh(phi, theta, vals, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    if contours:
        ax.contour(phi, theta, vals, levels=contours, colors="k", linewidths=0.5, alpha=0.6)

    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_xlabel(r"$\phi$ [rad]")
    ax.set_ylabel(r"$\theta$ [rad]")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig, ax

def plot_scalar_on_surface_3d(surface, values, ax=None, coils=None, label=None, title=None, symmetric=False, cmap=None, show=False):
    """
    Color the 3D surface by a scalar (matplotlib). Optionally draw the coils.

    values    : shape (nphi, ntheta) or flattened (nphi*ntheta,)
    symmetric : True -> color scale centered at 0
    Returns (fig, ax).
    """

    xyz  = np.asarray(surface.gamma)
    vals = np.asarray(values).reshape(xyz.shape[:2])

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if symmetric:
        vmax = np.max(np.abs(vals)); vmin = -vmax
        cmap = cmap or "coolwarm"
    else:
        vmin, vmax = vals.min(), vals.max()
        cmap = cmap or "viridis"

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], facecolors=cm(norm(vals)),
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    if coils is not None:
        coils.plot(ax=ax, show=False, close=False, color="brown", linewidth=2)

    fix_matplotlib_3d(ax)

    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array(vals)
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, label=label)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig, ax


def plot_vector_field_on_surface_3d(surface, vectors, ax=None, coils=None, step=1,
                                    color=None, label=None, length=1.0, normalize=True,
                                    draw_surface=True, alpha=0.35, title=None, show=False):
    """
    Draw ONE vector field as arrows on the 3D surface (matplotlib).
    For several fields, call it again with ax=ax and draw_surface=False.

    vectors      : shape (nphi, ntheta, 3) or flattened (nphi*ntheta, 3)
    step         : plot one arrow every `step` grid points in phi and theta
    draw_surface : draw the surface with surface.plot (set False on extra calls)
    coils        : optional Coils object, drawn with coils.plot
    Returns (fig, ax).
    """

    xyz = np.asarray(surface.gamma)
    vec = np.asarray(vectors).reshape(xyz.shape)

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if draw_surface:
        surface.plot(ax=ax, show=False, axis_equal=True, alpha=alpha)

    if coils is not None:
        coils.plot(ax=ax, show=False, close=False, color="brown", linewidth=2)

    p = xyz[::step, ::step]
    v = vec[::step, ::step]
    ax.quiver(p[..., 0], p[..., 1], p[..., 2], v[..., 0], v[..., 1], v[..., 2],
              length=length, normalize=normalize, color=color, label=label)

    if label:
        ax.legend()
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig, ax

def plot_coil_currents_3d(coils, ax, n_arrows=1, length=None, color="blue", label=None):
    """
    Draw arrows showing the direction of the current in each coil (3D, matplotlib).
    The coils themselves are NOT drawn: call coils.plot(ax=ax, ...) first, then this function on the same axes.
    How the direction is obtained
    -----------------------------
    Each coil is a closed curve gamma(t). Its tangent gamma_dash = d(gamma)/dt points in the direction of increasing t. A positive current flows along
    gamma_dash, a negative current flows against it, so
        arrow direction = sign(I) * gamma_dash / |gamma_dash|

    Parameters
    ----------
    coils    : essos Coils object (uses coils.gamma, coils.gamma_dash, coils.currents)
    ax       : 3D matplotlib axes where the arrows are drawn
    n_arrows : number of arrows per coil, equally spaced along the curve
    length   : arrow length in the plot units (m). If None, it is set to
               0.3 x the mean coil radius, so it scales with the geometry
    color    : arrow color
    label    : optional legend label (e.g. "current direction")
    """
    gamma      = np.asarray(coils.gamma)        # (n_coils, n_segments, 3)
    gamma_dash = np.asarray(coils.gamma_dash)   # (n_coils, n_segments, 3)
    currents   = np.asarray(coils.currents)     # (n_coils,)

    # Indices of the points where the arrows start (equally spaced along each coil)
    n_segments = gamma.shape[1]
    idx = ((np.arange(n_arrows) + 0.5) * n_segments / n_arrows).astype(int)

    points     = gamma[:, idx, :].reshape(-1, 3)
    directions = (np.sign(currents)[:, None, None] * gamma_dash[:, idx, :]).reshape(-1, 3)

    # Default arrow length: 30% of the mean coil radius
    if length is None:
        centroids = gamma.mean(axis=1, keepdims=True)
        length = 0.3 * np.linalg.norm(gamma - centroids, axis=-1).mean()

    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
              directions[:, 0], directions[:, 1], directions[:, 2],
              length=length, normalize=True, color=color,
              arrow_length_ratio=0.3, linewidth=2, label=label)
    return ax