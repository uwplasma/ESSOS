#!/usr/bin/env python3
import sys, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

ESSOS_ROOT  = Path(__file__).resolve().parents[2]
SIMSOPT_SRC = ESSOS_ROOT.parent / "simsopt" / "src"
for p in [str(ESSOS_ROOT), str(SIMSOPT_SRC)]:
    if p not in sys.path: sys.path.insert(0, p)

pho_file    = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pho_optimized.npy")
bundle_file = Path("/Users/joshuabourassa/Documents/New project/muse_opt_inputs_64x64.npz")
zot80_file  = Path("/Users/joshuabourassa/simsopt/tests/test_files/zot80.focus")

pho  = np.load(str(pho_file))
data = np.load(str(bundle_file))
area_w      = float(data["area_w"])
surface_xyz = np.asarray(data["surface_xyz"],    np.float64)
surface_nrm = np.asarray(data["surface_normal"], np.float64)
Bn_fixed    = np.asarray(data["Bn_fixed"],       np.float64)
positions   = np.asarray(data["positions"],      np.float64)
moments     = np.asarray(data["moments"],        np.float64)

port_mask = np.zeros(len(pho), dtype=bool)
with open(str(zot80_file)) as f:
    lines = f.readlines()
data_lines = [l.replace(",", " ").split() for l in lines[3:] if len(l.replace(",", " ").split()) >= 12]
if len(data_lines) == len(pho):
    port_mask = np.array([float(tok[6]) == 0.0 for tok in data_lines])

north_mask = pho >  0.5
south_mask = pho < -0.5
off_mask   = (np.abs(pho) <= 0.5) & ~port_mask
n_north, n_south = int(north_mask.sum()), int(south_mask.sum())
n_off, n_ports   = int(off_mask.sum()), int(port_mask.sum())

M0_SCALE     = float(np.linalg.norm(moments, axis=1).mean())
vol_per_cell = (M0_SCALE / (1.465 / (4*np.pi*1e-7))) * 1e6
fV_final     = vol_per_cell * float(np.sum(np.abs(pho)))
print(f"+1:{n_north}  -1:{n_south}  off:{n_off}  ports:{n_ports}  fV:{fV_final:.0f} cm3")

import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from essos.fields import DipoleField
from essos.optimization import compute_G_parallel

surf_pts = surface_xyz.reshape(-1, 3)
surf_n   = surface_nrm.reshape(-1, 3)
JAX_DTYPE = jnp.float32
dipole_field = DipoleField(
    jnp.asarray(positions, JAX_DTYPE), jnp.asarray(moments, JAX_DTYPE),
    jnp.zeros(len(positions), JAX_DTYPE), scale_factor=1.0)
G = np.asarray(compute_G_parallel(dipole_field,
    jnp.asarray(surf_pts, JAX_DTYPE), jnp.asarray(surf_n, JAX_DTYPE)), np.float64)
Bn_after = G @ pho + Bn_fixed
fB_final = float(0.5 * np.dot(Bn_after, Bn_after) * area_w)
print(f"fB={fB_final:.4e}  fV={fV_final:.0f} cm3")

nphi_s, ntheta_s = surface_xyz.shape[0], surface_xyz.shape[1]
R_axis = float(np.mean(np.sqrt(surface_xyz[:,:,0]**2 + surface_xyz[:,:,1]**2)))
Z_axis = float(np.mean(surface_xyz[:,:,2]))

def unroll(pos):
    phi   = np.arctan2(pos[:,1], pos[:,0]) / (np.pi/2)
    R_maj = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
    theta = (np.arctan2(pos[:,2] - Z_axis, R_maj - R_axis) % (2*np.pi)) / (2*np.pi)
    return phi, theta

phi_c, theta_c   = np.linspace(0,1,nphi_s), np.linspace(0,1,ntheta_s)
Bn_after_2d      = Bn_after.reshape(nphi_s, ntheta_s)
Bn_before_2d     = Bn_fixed.reshape(nphi_s, ntheta_s)
rms_a = float(np.sqrt(np.mean(Bn_after_2d**2)))
rms_b = float(np.sqrt(np.mean(Bn_before_2d**2)))

# Figure 1: 2D map + Bn after
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
ax = axes[0]
if port_mask.any():
    pp,pt = unroll(positions[port_mask])
    ax.scatter(pp,pt,c="lightgrey",s=1,alpha=0.5,linewidths=0,label=f"port ({n_ports})",zorder=0)
if off_mask.any():
    op,ot = unroll(positions[off_mask])
    ax.scatter(op,ot,c="#666666",s=3,alpha=0.6,linewidths=0,label=f"off ({n_off})",zorder=1)
if north_mask.any():
    np2,nt2 = unroll(positions[north_mask])
    ax.scatter(np2,nt2,c="red",s=8,alpha=0.9,linewidths=0,label=f"+1 ({n_north})",zorder=2)
if south_mask.any():
    sp,st = unroll(positions[south_mask])
    ax.scatter(sp,st,c="blue",s=8,alpha=0.9,linewidths=0,label=f"-1 ({n_south})",zorder=2)
ax.set_xlim(0.3, 0.6); ax.set_ylim(0.3, 0.7)
ax.set_xlabel("Toroidal angle — one field period",fontsize=12)
ax.set_ylabel("Poloidal angle theta",fontsize=12)
ax.set_title(f"+1:{n_north}  -1:{n_south}  off:{n_off}  ports:{n_ports}",fontsize=12)
ax.legend(loc="upper right",fontsize=9,markerscale=3)
ax2 = axes[1]
abs_max = np.abs(Bn_after_2d).max()
im = ax2.contourf(phi_c,theta_c,Bn_after_2d.T,levels=np.linspace(-abs_max,abs_max,31),cmap="RdBu_r",extend="both")
plt.colorbar(im,ax=ax2,label="B·n [T]")
ax2.set_xlabel("Toroidal angle phi",fontsize=12)
ax2.set_ylabel("Poloidal angle theta",fontsize=12)
ax2.set_title(f"Bn on plasma surface  RMS={rms_a:.3e} T",fontsize=12)
plt.suptitle(f"fB={fB_final:.2e}  fV={fV_final:.0f} cm³  +1:{n_north}  -1:{n_south}",fontsize=13)
plt.tight_layout()
plt.savefig("pho_magnet_map.png",dpi=200,bbox_inches="tight")
plt.show(); print("Saved pho_magnet_map.png")

# Figure 2: Bn before vs after
abs_max2 = max(np.abs(Bn_before_2d).max(), np.abs(Bn_after_2d).max())
levels2  = np.linspace(-abs_max2, abs_max2, 21)
fig2, ax2s = plt.subplots(1,2,figsize=(14,5))
im1 = ax2s[0].contourf(phi_c,theta_c,Bn_before_2d.T,levels=levels2,cmap="RdBu_r",extend="both")
plt.colorbar(im1,ax=ax2s[0],label="B·n [T]")
ax2s[0].set_title(f"Before  RMS={rms_b:.3e} T",fontsize=12)
ax2s[0].set_xlabel("phi"); ax2s[0].set_ylabel("theta")
im2 = ax2s[1].contourf(phi_c,theta_c,Bn_after_2d.T,levels=levels2,cmap="RdBu_r",extend="both")
plt.colorbar(im2,ax=ax2s[1],label="B·n [T]")
ax2s[1].set_title(f"After  RMS={rms_a:.3e} T",fontsize=12)
ax2s[1].set_xlabel("phi"); ax2s[1].set_ylabel("theta")
plt.suptitle(f"Bn reduction: {rms_b:.3e} -> {rms_a:.3e} T  ({rms_b/rms_a:.1f}x)",fontsize=13)
plt.tight_layout()
plt.savefig("pho_Bn_before_after.png",dpi=200,bbox_inches="tight")
plt.show(); print("Saved pho_Bn_before_after.png")

# Figure 3: 3D scatter
fig3 = plt.figure(figsize=(10,9))
ax3  = fig3.add_subplot(111,projection="3d")
if port_mask.any():
    ax3.scatter(positions[port_mask,0],positions[port_mask,1],positions[port_mask,2],
                c="lightgrey",s=0.3,alpha=0.15,depthshade=False)
if off_mask.any():
    ax3.scatter(positions[off_mask,0],positions[off_mask,1],positions[off_mask,2],
                c="#cccccc",s=0.3,alpha=0.1,depthshade=False)
if north_mask.any():
    ax3.scatter(positions[north_mask,0],positions[north_mask,1],positions[north_mask,2],
                c="red",s=3,alpha=0.9,depthshade=False,label=f"+1 ({n_north})")
if south_mask.any():
    ax3.scatter(positions[south_mask,0],positions[south_mask,1],positions[south_mask,2],
                c="blue",s=3,alpha=0.9,depthshade=False,label=f"-1 ({n_south})")
ax3.view_init(elev=25,azim=35)
ax3.set_axis_off()
ext = np.ptp(positions,axis=0).max()/2*0.9
ctr = np.mean(positions,axis=0)
ax3.set_xlim(ctr[0]-ext,ctr[0]+ext)
ax3.set_ylim(ctr[1]-ext,ctr[1]+ext)
ax3.set_zlim(ctr[2]-ext,ctr[2]+ext)
ax3.legend(loc="upper left",fontsize=10,markerscale=4)
ax3.set_title(f"MUSE PM — one field period\nfB={fB_final:.2e}  fV={fV_final:.0f} cm³",fontsize=12)
plt.tight_layout()
plt.savefig("pho_3d.png",dpi=200,bbox_inches="tight")
plt.show(); print("Saved pho_3d.png")
