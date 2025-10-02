# plot_helpers.py
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------- small utils ----------------------------

def npf(a, dtype=np.float32):
    """Convert (JAX/NumPy) to plain NumPy with desired dtype (saves memory)."""
    return np.asarray(a, dtype=dtype)

def closed_loop(P):
    """Ensure (N,3) loop is closed (first point repeated at end if needed)."""
    P = npf(P)
    return P if np.allclose(P[0], P[-1]) else np.vstack([P, P[0]])

def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def frames_parallel_transport(P):
    """
    Parallel-transport frames; returns (T, N1, N2) with same length as P.
    Works for open or closed curves. P shape: (N,3).
    """
    P = npf(P)
    closed = np.allclose(P[0], P[-1])
    if closed:
        T = unit(np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0))
    else:
        Tmid = unit(P[2:] - P[:-2])
        T = np.vstack([Tmid[0], Tmid, Tmid[-1]])

    ref = np.array([0.0, 0.0, 1.0], dtype=P.dtype)
    if abs(np.dot(T[0], ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=P.dtype)

    N1 = unit(np.cross(T[0], ref))[None, :]
    N2 = unit(np.cross(T[0], N1[0]))[None, :]

    for i in range(1, len(T)):
        t = T[i]
        n1 = unit(N1[-1] - np.dot(N1[-1], t) * t)
        n2 = unit(np.cross(t, n1))
        N1 = np.vstack([N1, n1])
        N2 = np.vstack([N2, n2])
    return T, N1, N2

# ---------------------------- tube factory ----------------------------

class TubeFactory:
    """Precompute circle once; reuse for every tube to save time & memory."""
    def __init__(self, n_theta=24):
        self.set_n_theta(n_theta)

    def set_n_theta(self, n_theta):
        self.n_theta = int(n_theta)
        th = np.linspace(0, 2*np.pi, self.n_theta, endpoint=True, dtype=np.float32)
        self.c = np.cos(th)[None, :]  # (1, n_theta)
        self.s = np.sin(th)[None, :]  # (1, n_theta)

    def tube_surface(self, P, radius=0.01, closed=True):
        P = npf(P)
        if closed:
            P = closed_loop(P)
        _, N1, N2 = frames_parallel_transport(P)

        # broadcast once; everything float32 to cut memory ~50%
        X = P[:, [0]] + radius * (N1[:, [0]] * self.c + N2[:, [0]] * self.s)
        Y = P[:, [1]] + radius * (N1[:, [1]] * self.c + N2[:, [1]] * self.s)
        Z = P[:, [2]] + radius * (N1[:, [2]] * self.c + N2[:, [2]] * self.s)
        return X, Y, Z

    def tube_trace(self, P, color="#5B2222", radius=0.01, name=None):
        X, Y, Z = self.tube_surface(P, radius=radius, closed=True)
        # solid color via trivial colorscale
        return go.Surface(
            x=X, y=Y, z=Z,
            showscale=False,
            # REMOVE: surfacecolor=np.zeros_like(X)
            colorscale=[[0, color], [1, color]],
            cmin=0, cmax=1,
            opacity=1.0,
            lighting=dict(ambient=0.22, diffuse=0.75, specular=0.55, roughness=0.45, fresnel=0.2),
            lightposition=dict(x=80, y=160, z=240),
            name=name or "coil",
            hoverinfo="skip",
        )

# ---------------------------- helpers for batches ----------------------------

def add_tubes_from_columns(data, X, Y, Z, tube: TubeFactory, color, radius=0.012, name=None, every=1, max_pts=None):
    """
    Build tubes from column arrays; supports decimation for speed:
    - every: take every-k points along each centerline
    - max_pts: clip length per centerline
    """
    X, Y, Z = npf(X), npf(Y), npf(Z)
    for xi, yi, zi in zip(X.T, Y.T, Z.T):
        P = np.column_stack([xi, yi, zi])
        if every > 1:
            P = P[::every]
        if max_pts is not None and len(P) > max_pts:
            P = P[:max_pts]
        data.append(tube.tube_trace(P, color=color, radius=radius, name=name))

def add_tubes_from_gamma(data, gamma, tube: TubeFactory, color, radius=0.012, name=None, every=1, max_pts=None):
    """gamma: (n_coils, n_pts, 3)."""
    for P in npf(gamma):
        if every > 1:
            P = P[::every]
        if max_pts is not None and len(P) > max_pts:
            P = P[:max_pts]
        data.append(tube.tube_trace(P, color=color, radius=radius, name=name))

def surface_trace_from_RZ_phi(R, Z, phi1D, color="#C5B6A7", opacity=0.28):
    colorscale = [[0, color], [1, color]]
    X = npf(R) * np.cos(npf(phi1D))
    Y = npf(R) * np.sin(npf(phi1D))
    return go.Surface(
        x=X, y=Y, z=npf(Z),
        colorscale=colorscale, showscale=False,
        opacity=opacity,
        lighting={"specular": 0.3, "diffuse": 0.9},
        hoverinfo="skip"
    )

def add_polyline_trajs(data, trajectories, color="black", width=0.6, name=None, opacity=0.9, every=3, dtype=np.float32):
    """Downsample fieldlines heavily for interactivity."""
    for traj in trajectories:
        T = npf(traj, dtype=dtype)
        T = T[::every]
        data.append(go.Scatter3d(
            x=T[:, 0], y=T[:, 1], z=T[:, 2],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            name=name or "line",
            showlegend=False
        ))


def tubes_mesh3d_from_gammas(gammas, radius=0.015, n_theta=12, color="#5B2222", opacity=1.0):
    """
    gammas: iterable of (N,3) centerlines (NumPy/JAX ok).
    Returns a single go.Mesh3d trace for all coils.
    """
    gammas = [npf(g) for g in gammas]
    th = np.linspace(0, 2*np.pi, n_theta, endpoint=False, dtype=np.float32)
    c, s = np.cos(th), np.sin(th)

    verts = []
    faces_i = []
    faces_j = []
    faces_k = []
    base = 0

    for P in gammas:
        P = closed_loop(P)
        # parallel-transport frames
        _, N1, N2 = frames_parallel_transport(P)
        # ring vertices
        # V shape: (N, n_theta, 3)
        V = (P[:, None, :] +
             radius * (N1[:, None, :] * c[None, :, None] +
                       N2[:, None, :] * s[None, :, None])).astype(np.float32)
        n, m = V.shape[0], V.shape[1]
        verts.append(V.reshape(-1, 3))

        # quad → two triangles per strip cell
        for i in range(n - 1):
            for j in range(m):
                a = base + i * m + j
                b = base + i * m + (j + 1) % m
                c1 = base + (i + 1) * m + j
                d = base + (i + 1) * m + (j + 1) % m
                # tri1: a, b, c1
                faces_i.append(a); faces_j.append(b); faces_k.append(c1)
                # tri2: b, d, c1
                faces_i.append(b); faces_j.append(d); faces_k.append(c1)
        base += n * m

    verts = np.vstack(verts)
    return go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=np.asarray(faces_i, dtype=np.int32),
        j=np.asarray(faces_j, dtype=np.int32),
        k=np.asarray(faces_k, dtype=np.int32),
        color=color,
        opacity=opacity,
        flatshading=True,
        hoverinfo="skip",
    )

def plot_loss_logs(csv_paths, labels=None, out_path="loss_compare.png", ylog=True):
    if labels is None:
        labels = [Path(p).stem for p in csv_paths]

    plt.figure(figsize=(6, 4.2))
    for p, lab in zip(csv_paths, labels):
        df = pd.read_csv(p)
        plt.plot(df["iter"], df["loss"], label=lab)

    if ylog:
        plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss = 0.5 ||residual||^2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")