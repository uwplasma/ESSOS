#!/usr/bin/env python3
"""
Full PM optimizer including essos.losses.custom_loss
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np

if len(sys.argv) != 4:
    sys.exit("Usage: python pm_opt_custom_loss.py <surf_file> <mag_file> <coil_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])
COIL_FILE = Path(sys.argv[3])

SURFACE_RANGE  = "half period"
SURFACE_NPHI   = 64
SURFACE_NTHETA = 64

FB_ONLY_STEPS       = 8000
FB_ONLY_LR_MAX      = 0.01
FB_ONLY_LR_MIN_FRAC = 0.1

FD_ANNEAL_STEPS       = 6000
FD_ANNEAL_LR_MAX      = 0.001
FD_ANNEAL_LR_MIN_FRAC = 0.001

MAX_WD        = 0.2
WD_RAMP_POWER = 4
LOG_INTERVAL  = 500

VOLUME_TARGET_CM3 = 811.25  
W_VOLUME_TARGET   = 1.0

B_MAX_T, MU0 = 1.465, 4 * np.pi * 1e-7

JAX_PLATFORM = "cuda"


OUTPUT_DIR = Path(__file__).resolve().parent / "pm_opt_custom_loss_output"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

if JAX_PLATFORM != "auto":
    os.environ["JAX_PLATFORMS"] = JAX_PLATFORM

backend_name = str(jax.default_backend()).lower()
print(f"JAX backend: {backend_name}  |  devices: {[str(d) for d in jax.devices()]}")   



from essos.fields import DipoleField
from essos.losses import custom_loss

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_surface(surf_file, surface_range, nphi, ntheta):
    from simsopt.geo import SurfaceRZFourier
    try:
        surface = SurfaceRZFourier.from_vmec_input(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    except Exception:
        surface = SurfaceRZFourier.from_focus(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    return surface


def load_coils_essos(coil_file):
    from simsopt.field import Coil, Current
    from simsopt.util.coil_optimization_helper_functions import read_focus_coils
    from essos.coils import Coils_from_simsopt

    base_curves, base_currents0, ncoils = read_focus_coils(str(coil_file))
    total_current = float(np.sum([c.get_value() for c in base_currents0]))
    all_coils = [Coil(base_curves[i], Current(total_current / ncoils)) for i in range(ncoils)]
    return Coils_from_simsopt(all_coils, nfp=1, stellsym=False)


def compute_Bn_fixed_essos(coils, surf_pts, surf_n):
    from essos.fields import BiotSavart
    field = BiotSavart(coils)
    surf_pts_jax = jnp.asarray(surf_pts)
    B_at_pts = jax.vmap(field.B)(surf_pts_jax)
    return np.asarray(jnp.sum(B_at_pts * jnp.asarray(surf_n), axis=1))


def load_magnet_grid(mag_file):
    pos_list, mom_list, pho_list, ic_list = [], [], [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.replace(",", " ").split()
            if len(tokens) < 12:
                continue
            try:
                x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
                ic      = float(tokens[6])
                m0      = float(tokens[7])
                pho     = float(tokens[8])
                az, pol = float(tokens[10]), float(tokens[11])
            except ValueError:
                continue
            pos_list.append((x, y, z))
            mom_list.append((m0*np.cos(az)*np.sin(pol), m0*np.sin(az)*np.sin(pol), m0*np.cos(pol)))
            pho_list.append(pho)
            ic_list.append(ic)
    return (
        np.asarray(pos_list),
        np.asarray(mom_list),
        np.asarray(pho_list),
        np.asarray(ic_list),
    )


def write_focus_file(path, positions, moments, pho, Ic_passed=None, momentq=1):
    """Write a FAMUS/FOCUS magnet file matching the source format exactly."""
    def fmt(x):
        return str(round(float(x), 6))

    n = len(positions)
    m0 = np.linalg.norm(moments, axis=1)
    m0_safe = np.where(m0 > 0, m0, 1.0)
    mhat = moments / m0_safe[:, None]
    pol = np.arccos(np.clip(mhat[:, 2], -1, 1))
    az = np.arctan2(mhat[:, 1], mhat[:, 0])

    # None -> all on; scalar -> same for all; array -> per-site
    if Ic_passed is None:
        Ic_arr = np.ones(n)
    else:
        Ic_arr = np.broadcast_to(np.asarray(Ic_passed, dtype=float), (n,))

    with open(path, "w") as f:
        f.write("# Total number of coils,  momentq \n")
        f.write(f"{n:>10}{momentq:>6}\n")
        f.write("# coiltype, symmetry,  coilname,  ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt \n")
        for i in range(n):
            name = f"pm{i:08d}"
            fields = [
                "2",
                f"{2:>11}",
                f"{name:>17}   ",
                f" {fmt(positions[i, 0])}",
                f" {fmt(positions[i, 1])}",
                f" {fmt(positions[i, 2])}",
                f" {fmt(Ic_arr[i])}",
                f" {fmt(m0[i])}",
                f" {fmt(pho[i])}",
                f" {fmt(0.0)}",
                f" {fmt(az[i])}",
                f" {fmt(pol[i])} ",
            ]
            f.write(",".join(fields) + "\n")
    print(f"Wrote {n} magnets to {path}")

T0_START = time.time()


print(f"--- Loading surface: {SURF_FILE.name} ({SURFACE_NPHI}x{SURFACE_NTHETA}) ---")
surface = load_surface(SURF_FILE, SURFACE_RANGE, SURFACE_NPHI, SURFACE_NTHETA)
nfp, stellsym = int(surface.nfp), bool(surface.stellsym)
print(f"nfp={nfp}  stellsym={stellsym}")

surf_pts = surface.gamma().reshape(-1, 3)
surf_n   = surface.unitnormal().reshape(-1, 3)
surf_n_raw = np.asarray(surface.normal()).reshape(-1, 3)
Nnorms = np.linalg.norm(surf_n_raw, axis=1)
n_pts  = len(surf_pts)
area_weight_vec = Nnorms / n_pts   

print(f"--- Loading coils: {COIL_FILE.name} ---")
essos_coils = load_coils_essos(COIL_FILE)
Bn_fixed = compute_Bn_fixed_essos(essos_coils, surf_pts, surf_n)

print(f"--- Loading magnet grid: {MAG_FILE.name} ---")
positions, moments_raw, pho_loaded, ic_flags = load_magnet_grid(MAG_FILE)
n_magnets = len(positions)

PORT_MASK = ic_flags < 0.5
n_excluded = int(np.sum(PORT_MASK))
print(f"Ic flag: {n_excluded} excluded sites (Ic=0)")

native_norms = np.linalg.norm(moments_raw, axis=1)
n_zero_moment = int(np.sum(native_norms == 0))
if n_zero_moment > 0:
    print(f"WARNING: {n_zero_moment} zero-moment magnets excluded like Ic=0 sites")
norms_safe = np.where(native_norms > 0, native_norms, 1.0)
M0_SCALE = float(np.mean(native_norms[native_norms > 0])) if np.any(native_norms > 0) else 0.0
orientations = np.where(native_norms[:, None] > 0, moments_raw / norms_safe[:, None], 0.0)
magnet_moments = orientations * M0_SCALE

ZERO_MOMENT_MASK = native_norms == 0
EXCLUDE_MASK = PORT_MASK | ZERO_MOMENT_MASK

M_MAX = B_MAX_T / MU0
volume_per_cell = (M0_SCALE / M_MAX) * 1e6
SYMMETRY_MULTIPLIER = nfp * (2 if stellsym else 1)
print(f"{n_magnets} magnets  |  V_cell={volume_per_cell:.4f} cm3  |  SYMMETRY_MULTIPLIER={SYMMETRY_MULTIPLIER}")

print("--- Build G matrix ---")
t0 = time.time()
dipole_field = DipoleField(
    jnp.asarray(positions),
    jnp.asarray(magnet_moments),
    jnp.zeros(n_magnets),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)
G = np.asarray(dipole_field.compute_interaction_matrix(
    jnp.asarray(surf_pts), jnp.asarray(surf_n)))
n_nan, n_inf = int(np.sum(np.isnan(G))), int(np.sum(np.isinf(G)))
if n_nan > 0 or n_inf > 0:
    raise RuntimeError(f"G has {n_nan} NaN / {n_inf} Inf values")
print(f"G: {G.shape}  {G.nbytes/1e9:.2f} GB  {time.time()-t0:.1f}s")

G_jax  = jnp.asarray(G)                              
Bn_jax = jnp.asarray(Bn_fixed)
aw_jax = jnp.asarray(area_weight_vec)
vc_jax = volume_per_cell
Vt_jax = VOLUME_TARGET_CM3
wVT_jax = W_VOLUME_TARGET
exclude_mask_jax = jnp.asarray(EXCLUDE_MASK)

fB_gen0 = float(0.5 * np.sum(area_weight_vec * (pho_loaded @ G.T + Bn_fixed) ** 2))
fB_ref = max(fB_gen0, 1e-20)
print(f"fB (loaded FAMUS solution, reference only -- we start from empty) = {fB_gen0:.4e}")


def combined_loss_fn(pho, wD, G, Bn_fix, area_w, fB_ref, vc, Vt, wVT):
    bn = pho @ G.T + Bn_fix
    fB_raw = 0.5 * jnp.sum(area_w * bn * bn)
    fB = fB_raw / fB_ref
    abs_p = jnp.sqrt(pho * pho + 1e-7)
    fD = jnp.sum(abs_p * (1.0 - abs_p))
    fV = vc * jnp.sum(abs_p)
    fVT = jnp.where(Vt > 0, jnp.maximum(fV - Vt, 0.0), 0.0)
    loss = fB + wD * fD + wVT * fVT

    return loss, (fB_raw, fV, fD)


total_loss = custom_loss(
    combined_loss_fn, "pho", "wD",
    G=G_jax, Bn_fix=Bn_jax, area_w=aw_jax, fB_ref=fB_ref,
    vc=vc_jax, Vt=Vt_jax, wVT=wVT_jax,
)

# combined_loss_fn returns (loss, aux) -- use total_loss.value_and_grad_pytree with has_aux=True 

def loss_and_metrics(pho, wD):
    (loss_val, aux), grad_dict = total_loss.value_and_grad_pytree(
        {"pho": pho, "wD": wD}, has_aux=True
    )
    return loss_val, grad_dict["pho"], aux

# pho = 0
pho = jnp.zeros(n_magnets)
print("*** STARTING FROM EMPTY GRID (pho=0) ***")


def cosine_lr(step, total, lr_max, lr_min):
    return lr_min + 0.5*(lr_max-lr_min)*(1+np.cos(np.pi*step/max(total, 1)))


def get_lr_and_wd(step):
    if step <= FB_ONLY_STEPS:
        lr_min = FB_ONLY_LR_MAX * FB_ONLY_LR_MIN_FRAC
        return cosine_lr(step-1, FB_ONLY_STEPS, FB_ONLY_LR_MAX, lr_min), 0.0
    s      = step - FB_ONLY_STEPS
    lr_min = FD_ANNEAL_LR_MAX * FD_ANNEAL_LR_MIN_FRAC
    lr     = cosine_lr(s-1, FD_ANNEAL_STEPS, FD_ANNEAL_LR_MAX, lr_min)
    frac   = (s-1) / max(FD_ANNEAL_STEPS-1, 1)
    wD     = (frac ** WD_RAMP_POWER) * MAX_WD
    return lr, wD


TOTAL_STEPS = FB_ONLY_STEPS + FD_ANNEAL_STEPS
print(f"\nStage 1: {FB_ONLY_STEPS} steps   Stage 2: {FD_ANNEAL_STEPS} steps")
print(f"Volume target: {VOLUME_TARGET_CM3:.1f} cm3 (unique domain)")

t_start = time.time()
mom = jnp.zeros(n_magnets)
var = jnp.zeros(n_magnets)
b1, b2, eps = 0.9, 0.999, 1e-8

hist_fB, hist_fV, hist_fD = [], [], []

def compute_metrics(pho):
    bn = pho @ G_jax.T + Bn_jax
    fB = float(0.5 * jnp.sum(aw_jax * bn * bn))
    abs_p = jnp.sqrt(pho * pho + 1e-7)
    fV = float(vc_jax * jnp.sum(abs_p))
    fD = float(jnp.sum(abs_p * (1.0 - abs_p)))
    return fB, fV, fD

print("Compiling first step...")
lr0, wd0 = get_lr_and_wd(1)
val, grad_pho, aux0 = loss_and_metrics(pho, wd0)
mom = b1*mom + (1-b1)*grad_pho
var = b2*var + (1-b2)*grad_pho*grad_pho
pho = jnp.clip(pho - lr0 * mom / (jnp.sqrt(var) + eps), -1, 1)
pho = jnp.where(exclude_mask_jax, 0.0, pho)
_ = pho.block_until_ready()
print("JIT compiled.")

for step in range(2, TOTAL_STEPS + 1):
    lr, wD = get_lr_and_wd(step)
    val, grad_pho, aux = loss_and_metrics(pho, wD)
    mom = b1*mom + (1-b1)*grad_pho
    var = b2*var + (1-b2)*grad_pho*grad_pho
    pho = jnp.clip(pho - lr * mom / (jnp.sqrt(var) + eps), -1, 1)
    pho = jnp.where(exclude_mask_jax, 0.0, pho)

    fB_now, fV_now, fD_now = float(aux[0]), float(aux[1]), float(aux[2])
    hist_fB.append(fB_now); hist_fV.append(fV_now); hist_fD.append(fD_now)

    if step % LOG_INTERVAL == 0 or step == TOTAL_STEPS:
        stage = "Stage1" if step <= FB_ONLY_STEPS else "Anneal"
        print(f"[{stage}] step {step:5d}/{TOTAL_STEPS}  fB={fB_now:.4e}  fV={fV_now:.1f}  fD={fD_now:.2f}  wD={wD:.4f}  lr={lr:.4f}")

p = np.asarray(pho)
p_d = np.sign(p) * (np.abs(p) > 0.5)
p_d = np.where(EXCLUDE_MASK, 0.0, p_d)

def fB64_of(pp):
    r = pp @ G.T + Bn_fixed
    return 0.5 * np.sum(area_weight_vec * r * r)

fB_fin = fB64_of(p)
fB_rnd = fB64_of(p_d)
fV_final = float(volume_per_cell * np.sum(np.abs(p_d)))
n_active = int(np.sum(np.abs(p_d) > 0.5))

print(f"\n{'='*70}")
print(f"Done in {time.time()-t_start:.0f}s")
print(f"  Final (smooth):   {fB_fin:.4e}")
print(f"  Hard-rounded:     {fB_rnd:.4e}  fV={fV_final:.1f} cm3 (unique)  = {fV_final*SYMMETRY_MULTIPLIER:.1f} cm3 (full)")
print(f"  Active magnets:   {n_active} / {n_magnets} ({100*n_active/n_magnets:.1f}%)")
print(f"{'='*70}")

np.save(OUTPUT_DIR / "pho_optimized.npy", p_d)
np.save(OUTPUT_DIR / "pho_continuous.npy", p)
final_moments = moments_raw # did not change, only pho did 

write_focus_file(OUTPUT_DIR / "pm_optimized.focus", positions, final_moments, p_d)

np.savez(OUTPUT_DIR / "convergence_history.npz",
         fB=np.array(hist_fB), fV=np.array(hist_fV), fD=np.array(hist_fD))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
steps_axis = np.arange(1, len(hist_fB)+1)
axes[0].plot(steps_axis, hist_fB, color="steelblue"); axes[0].set_yscale("log"); axes[0].set_ylabel("fB")
axes[1].plot(steps_axis, hist_fV, color="seagreen"); axes[1].set_ylabel("fV [cm3]")
axes[2].plot(steps_axis, hist_fD, color="purple"); axes[2].set_yscale("log"); axes[2].set_ylabel("fD"); axes[2].set_xlabel("iteration")
for ax in axes:
    ax.axvline(FB_ONLY_STEPS, color="gray", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "convergence_history.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR}/convergence_history.png")

T0_END = time.time()
print(f"Total Run time:{T0_END - T0_START}")