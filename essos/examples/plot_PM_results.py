import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax

from essos.fields import DipoleField
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  


# INPUT FILES — passed as command-line arguments
if len(sys.argv) != 4:
    sys.exit("Usage: python plot.py <surf_file> <mag_file> <coil_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])
COIL_FILE = Path(sys.argv[3])

SURFACE_RANGE = "half period"
SURFACE_NPHI   = 64
SURFACE_NTHETA = 64

def load_surface(surf_file, surface_range, nphi, ntheta):
    from simsopt.geo import SurfaceRZFourier
    try:
        surface = SurfaceRZFourier.from_vmec_input(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    except Exception:
        surface = SurfaceRZFourier.from_focus(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    return surface

def load_coils_essos(coil_file):

    from simsopt.field import Coil, Current
    from essos.coils import Coils_from_simsopt

    try:
        from simsopt.field.coil import load_coils_from_makegrid_file
        coils = load_coils_from_makegrid_file(str(COIL_FILE), order = 10)
        
    except:
        from simsopt.util.coil_optimization_helper_functions import read_focus_coils


        base_curves, base_currents0, ncoils = read_focus_coils(str(coil_file))
        total_current = float(np.sum([c.get_value() for c in base_currents0]))
        all_coils = [Coil(base_curves[i], Current(total_current / ncoils)) for i in range(ncoils)]
        coils = Coils_from_simsopt(all_coils, nfp=1, stellsym=False)
        
    return coils

def compute_Bn_fixed_essos(coils, surf_pts, surf_n):
    """Evaluate the essos-native BiotSavart field at each surface point and
    project onto the local normal. essos's BiotSavart.B() takes a single
    point, so we vmap over the surface points."""
    from essos.fields import BiotSavart

    field = BiotSavart(coils)
    surf_pts_jax = jnp.asarray(surf_pts, jnp.float64)
    B_at_pts = jax.vmap(field.B)(surf_pts_jax)
    Bn = jnp.sum(B_at_pts * jnp.asarray(surf_n, jnp.float64), axis=1)
    return np.asarray(Bn, np.float64)

def load_magnet_grid(mag_file):
    
    pos_list, mom_list, pho_list, ic_list = [], [], [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f.readlines()[3:]:
            tokens = line.replace(",", " ").split()
            
            x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
            ic      = float(tokens[6])
            m0      = float(tokens[7])
            pho     = float(tokens[8])
            az, pol = float(tokens[10]), float(tokens[11])
            pos_list.append((x, y, z))
            mom_list.append((m0*np.cos(az)*np.sin(pol), m0*np.sin(az)*np.sin(pol), m0*np.cos(pol)))
            pho_list.append(pho)
            ic_list.append(ic)
    return (
        np.asarray(pos_list, np.float64),
        np.asarray(mom_list, np.float64),
        np.asarray(pho_list, np.float64),
        np.asarray(ic_list,  np.float64),
    )



print(f"\n--- Loading surface: {SURF_FILE.name} ---")
surface = load_surface(SURF_FILE, SURFACE_RANGE, SURFACE_NPHI, SURFACE_NTHETA)


nfp      = int(surface.nfp)
stellsym = bool(surface.stellsym)
print(f"Surface nfp={nfp}  stellsym={stellsym}  (read from surface file)")

surf_xyz = np.asarray(surface.gamma(), np.float64)
surf_nrm = np.asarray(surface.unitnormal(), np.float64)
surf_pts = surf_xyz.reshape(-1, 3)
surf_n   = surf_nrm.reshape(-1, 3)

essos_coils = load_coils_essos(COIL_FILE)
Bn_fixed = compute_Bn_fixed_essos(essos_coils, surf_pts, surf_n)

magnet_positions, magnet_moments_raw, pho_loaded, ic_flags = load_magnet_grid(MAG_FILE)



t0 = time.time()
dipole_field = DipoleField(
    magnet_positions,
    magnet_moments_raw*pho_loaded[:,np.newaxis],
    jnp.zeros(len(magnet_positions)),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)



B_dipole = dipole_field.B(surf_pts)

nphi_s, ntheta_s = surf_xyz.shape[0], surf_xyz.shape[1]
phi_coords   = np.linspace(0, 2*np.pi/nfp, nphi_s)
theta_coords = np.linspace(0, 2*np.pi, ntheta_s)

Bn_before = Bn_fixed.reshape(nphi_s, ntheta_s)

Bn_pm     = (np.sum(B_dipole*surf_n,axis = -1)).reshape(nphi_s, ntheta_s)
Bn_after  = Bn_before + Bn_pm


abs_max_before = np.abs(Bn_before).max()
abs_max_after  = np.abs(Bn_after).max()
levels_before  = np.linspace(-abs_max_before, abs_max_before, 21)
levels_after   = np.linspace(-abs_max_after, abs_max_after, 21)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
im1 = axes[0].contourf(phi_coords, theta_coords, Bn_before.T, levels=levels_before, cmap="RdBu_r", extend="both")
axes[0].set_title("Coils only (before)", fontsize=12)
axes[0].set_xlabel("phi"); axes[0].set_ylabel("theta")
axes[0].set_xlim(phi_coords.min(), phi_coords.max())
axes[0].set_ylim(theta_coords.min(), theta_coords.max())
axes[0].set_aspect("auto")
plt.colorbar(im1, ax=axes[0], label="B\u00b7n [T]")
rms_before = float(np.sqrt(np.mean(Bn_before**2)))
axes[0].text(0.02, 0.97, f"RMS = {rms_before:.3e} T", transform=axes[0].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

im2 = axes[1].contourf(phi_coords, theta_coords, Bn_after.T, levels=levels_after, cmap="RdBu_r", extend="both")
axes[1].set_title("Coils + optimized PMs (after)", fontsize=12)
axes[1].set_xlabel("phi")
axes[1].set_xlim(phi_coords.min(), phi_coords.max())
axes[1].set_ylim(theta_coords.min(), theta_coords.max())
axes[1].set_aspect("auto")
plt.colorbar(im2, ax=axes[1], label="B\u00b7n [T]")
rms_after = float(np.sqrt(np.mean(Bn_after**2)))
axes[1].text(0.02, 0.97, f"RMS = {rms_after:.3e} T", transform=axes[1].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()






# surf has shape (N, M, 3)
try:
    x = surface.gamma()[:, :, 0]
    y = surface.gamma()[:, :, 1]
    z = surface.gamma()[:, :, 2]
except:
    x = surface_xyz[:, :, 0]
    y = surface_xyz[:, :, 1]
    z = surface_xyz[:, :, 2]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# --- coils ---
try:
    for c in essos_coils:
        g = c.gamma[0]
        ax.plot(g[:, 0], g[:, 1], g[:, 2],
                color='gold', linewidth=2)
except:
    print('No coils loaded')

# --- magnets (colored scatter) ---
arg = jnp.argwhere(pho_loaded != 0)[:, 0]
sc = ax.scatter(
    magnet_positions[:, 0][arg],
    magnet_positions[:, 1][arg],
    magnet_positions[:, 2][arg],
    c=pho_loaded[arg],
    cmap='RdBu_r',
    s=20,                # marker size (matplotlib uses points^2, tune as needed)
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('pho')

# --- surface ---
ax.plot_surface(
    x, y, z,
    rstride=1, cstride=1,
    color=None,
    cmap='RdBu_r',
    alpha=0.5,
    linewidth=0,
    antialiased=False,
    shade=False,
)

# --- styling to mimic plotly layout ---
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_axis_off()  # hides axes/panes/labels entirely (like visible=False on all axes)

# equal aspect ratio ('aspectmode=data' equivalent)
def set_axes_equal(ax):
    """Make 3D plot axes have equal scale (matplotlib has no native 'data' aspect)."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

set_axes_equal(ax)

# camera-ish view (elev/azim are the closest matplotlib analog to plotly's eye)
ax.view_init(elev=80, azim=45)

plt.tight_layout()
plt.show()
