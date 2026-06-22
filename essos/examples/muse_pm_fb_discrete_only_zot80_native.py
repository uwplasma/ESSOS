


#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

JAX_PLATFORM    = "cpu" # "cpu", "gpu", or "auto"
CPU_THREADS     = 4
ENABLE_X64      = True

N_PARALLEL_STARTS = 2 # +1, -1, 0, and (N-3) random starts

FB_ONLY_STEPS    = 2000   # Stage 1: minimize fB only (continuous relaxation)
FD_ANNEAL_STEPS  = 2000   # Stage 2: anneal discreteness penalty wD from 0 → MAX_WD

FB_ONLY_LR_MAX      = 0.00001
FB_ONLY_LR_MIN_FRAC = 0.1

FD_ANNEAL_LR_MAX      = 0.003
FD_ANNEAL_LR_MIN_FRAC = 0.05

MAX_WD       = 1.0
ANNEAL_POWER = 5
LOG_INTERVAL = 500

VOLUME_TARGET_CM3 = 0.0
W_VOLUME_TARGET   = 0.0

REFERENCE_M0_SCALE = 0.074625
B_MAX_T = 1.465
MU0     = 4 * np.pi * 1e-7

SURFACE_RANGE  = "half period"
SURFACE_NPHI   = 64
SURFACE_NTHETA = 64

ESSOS_ROOT  = Path(__file__).resolve().parents[2]
SIMSOPT_SRC = ESSOS_ROOT.parent / "simsopt" / "src"

DEFAULT_SURF_FILE    = ESSOS_ROOT / "essos" / "input.muse"
DEFAULT_COIL_FILE    = SIMSOPT_SRC.parent / "tests" / "test_files" / "muse_tf_coils.focus"
DEFAULT_MAG_FILE     = SIMSOPT_SRC.parent / "tests" / "test_files" / "zot80.focus"
DEFAULT_INPUT_BUNDLE = ESSOS_ROOT / "essos" / "examples" / "input_files" / "muse_opt_inputs_64x64.npz"

if "jax" in sys.modules:
    print("WARNING: JAX already imported — device selection may not take effect.")

if JAX_PLATFORM != "auto":
    os.environ["JAX_PLATFORMS"] = JAX_PLATFORM

if JAX_PLATFORM == "cpu":
    xla_flag = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={CPU_THREADS}"
    existing = os.environ.get("XLA_FLAGS", "").strip()
    if xla_flag not in existing:
        os.environ["XLA_FLAGS"] = f"{existing} {xla_flag}".strip()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(CPU_THREADS))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", ENABLE_X64)

backend_name = str(jax.default_backend()).lower()
print(f"JAX backend: {backend_name}  |  devices: {[str(d) for d in jax.devices()]}")

for p in [str(ESSOS_ROOT), str(SIMSOPT_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from essos.fields import DipoleField


def load_from_bundle(path):
    data   = np.load(path)
    tf_pts = np.asarray(data["tf_coil_points"], np.float64) if "tf_coil_points" in data.files else None
    return (
        np.asarray(data["surface_xyz"],    np.float64),
        np.asarray(data["surface_normal"], np.float64),
        np.asarray(data["Bn_fixed"],       np.float64),
        float(data["area_w"]),
        np.asarray(data["positions"],      np.float64),
        np.asarray(data["moments"],        np.float64),
        tf_pts,
    )


def load_from_files(surf_file, coil_file, mag_file, surface_range, nphi, ntheta):
    from simsopt.field import BiotSavart, Coil, Current
    from simsopt.geo import SurfaceRZFourier
    from simsopt.util.permanent_magnet_helper_functions import read_focus_coils

    surface  = SurfaceRZFourier.from_focus(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    surf_xyz = np.asarray(surface.gamma(),      np.float64)
    surf_nrm = np.asarray(surface.unitnormal(), np.float64)
    surf_pts = surf_xyz.reshape(-1, 3)
    surf_n   = surf_nrm.reshape(-1, 3)
    area_w   = float(surface.area() / len(surf_pts))

    base_curves, base_currents0, ncoils = read_focus_coils(str(coil_file))
    total_current = float(np.sum([c.get_value() for c in base_currents0]))
    coils    = [Coil(base_curves[i], Current(total_current / ncoils)) for i in range(ncoils)]
    bs       = BiotSavart(coils)
    bs.set_points(surf_pts)
    Bn_fixed = np.sum(np.asarray(bs.B(), np.float64) * surf_n, axis=1)

    pos_list, mom_list = [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f.readlines()[3:]:
            tokens = line.replace(",", " ").split()
            if len(tokens) < 12:
                continue
            x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
            m0      = float(tokens[7])
            az, pol = float(tokens[10]), float(tokens[11])
            pos_list.append((x, y, z))
            mom_list.append((m0*np.cos(az)*np.sin(pol), m0*np.sin(az)*np.sin(pol), m0*np.cos(pol)))

    return (
        surf_xyz, surf_nrm, Bn_fixed, area_w,
        np.asarray(pos_list, np.float64),
        np.asarray(mom_list, np.float64),
        None,
    )


required = ["surface_xyz", "surface_normal", "Bn_fixed", "area_w", "positions", "moments"]
missing  = [v for v in required if v not in globals()]

if missing:
    if DEFAULT_INPUT_BUNDLE and os.path.exists(DEFAULT_INPUT_BUNDLE):
        print(f"Loading from bundle: {DEFAULT_INPUT_BUNDLE}")
        surface_xyz, surface_normal, Bn_fixed, area_w, positions, moments, _ = load_from_bundle(DEFAULT_INPUT_BUNDLE)
    else:
        print("Loading from surface/coil/magnet files...")
        surface_xyz, surface_normal, Bn_fixed, area_w, positions, moments, _ = load_from_files(
            DEFAULT_SURF_FILE, DEFAULT_COIL_FILE, DEFAULT_MAG_FILE,
            SURFACE_RANGE, SURFACE_NPHI, SURFACE_NTHETA,
        )

area_weight    = float(area_w)
surface_xyz    = np.asarray(surface_xyz,    np.float64)
surface_normal = np.asarray(surface_normal, np.float64)

zot80_norms = np.linalg.norm(np.asarray(moments), axis=1)
print(f"zot80 |m| mean: {zot80_norms.mean():.6f}  (reference: {REFERENCE_M0_SCALE})")
if abs(zot80_norms.mean() - REFERENCE_M0_SCALE) >= 1e-6:
    raise RuntimeError("Moment scale mismatch — check REFERENCE_M0_SCALE.")

M_MAX               = B_MAX_T / MU0
magnet_positions    = np.asarray(positions, np.float64)
native_norms        = np.linalg.norm(np.asarray(moments), axis=1)
magnet_orientations = np.asarray(moments, np.float64) / native_norms[:, None]
n_magnets           = len(magnet_positions)
M0_SCALE            = REFERENCE_M0_SCALE
volume_per_cell     = (M0_SCALE / M_MAX) * 1e6

print("=" * 70)
print("MUSE PM Optimization — fB + Discreteness (zot80 native lattice)")
print(f"Backend: {backend_name}  |  Starts: {N_PARALLEL_STARTS}")
print(f"Magnets: {n_magnets}  |  M0={M0_SCALE:.6f} A·m²  |  V_cell={volume_per_cell:.4f} cm³")
print("=" * 70)
# BUILD G MATRIX
print("\n--- Build G matrix ---")
magnet_moments   = magnet_orientations * M0_SCALE
JAX_DTYPE        = jnp.float32
surface_pts_flat = jnp.asarray(surface_xyz.reshape(-1, 3),    JAX_DTYPE)
surface_nrm_flat = jnp.asarray(surface_normal.reshape(-1, 3), JAX_DTYPE)

t0 = time.time()
dipole_field = DipoleField(
    jnp.asarray(magnet_positions, JAX_DTYPE),
    jnp.asarray(magnet_moments,   JAX_DTYPE),
    jnp.zeros(n_magnets, JAX_DTYPE),
    nfp=2, stellsym=True, scale_factor=1.0,
)
G_f32  = np.asarray(dipole_field.compute_interaction_matrix(surface_pts_flat, surface_nrm_flat), np.float32)
Bn_f32 = np.asarray(Bn_fixed, np.float32)
gc.collect()

fB_gen0 = float(0.5 * np.dot(Bn_f32.astype(np.float64), Bn_f32.astype(np.float64)) * area_weight)
print(f"G: {G_f32.shape}  {G_f32.nbytes/1e9:.2f} GB  {time.time()-t0:.1f}s")
print(f"fB(rho=0) = {fB_gen0:.4e}")

G_jax   = jnp.asarray(G_f32)
Bn_jax  = jnp.asarray(Bn_f32)
aw_jax  = jnp.float32(area_weight)
vc_jax  = jnp.float32(volume_per_cell)
fB_ref  = jnp.float32(max(fB_gen0, 1e-20))
Vt_jax  = jnp.float32(VOLUME_TARGET_CM3)
wVT_jax = jnp.float32(W_VOLUME_TARGET)
f32     = jnp.float32


@jax.jit
def compute_metrics(pho_batch):
    bn    = pho_batch @ G_jax.T + Bn_jax[None, :]
    fB    = f32(0.5) * jnp.sum(bn * bn, axis=1) * aw_jax
    abs_p = jnp.sqrt(pho_batch * pho_batch + f32(1e-7))
    fV    = vc_jax * jnp.sum(abs_p, axis=1)
    fD    = jnp.sum(abs_p * (f32(1) - abs_p), axis=1)
    return fB, fV, fD


@jax.jit
def adam_step(pho, m, v,t, lr, w_fB, w_fD):
    def loss(x):
        bn    = x @ G_jax.T + Bn_jax[None, :]
        fB    = f32(0.5) * jnp.sum(bn * bn, axis=1) * aw_jax
        abs_p = jnp.sqrt(x * x + f32(1e-7))
        fV    = vc_jax * jnp.sum(abs_p, axis=1)
        fD    = jnp.sum(abs_p * (f32(1) - abs_p), axis=1)
        Vt    = jnp.maximum(Vt_jax, f32(1e-6))
        fVT   = ((fV - Vt) / Vt) ** f32(2)
        vol_term = jnp.where(Vt_jax > f32(0), wVT_jax * fVT, f32(0))
        return jnp.sum(w_fB * fB / fB_ref + w_fD * fD + vol_term)
    b1, b2, eps = f32(0.9), f32(0.999), f32(1e-8)
    _, g = jax.value_and_grad(loss)(pho)
    m    = b1*m + (1-b1)*g
    v    = b2*v + (1-b2)*g*g
    mh   = m / (1 - b1**t)
    vh   = v / (1 - b2**t)
    pho  = jnp.clip(pho - lr * mh / (jnp.sqrt(vh) + eps), -1, 1)
    return pho, m, v


def cosine_lr(step, total, lr_max, lr_min):
    return lr_min + 0.5*(lr_max - lr_min)*(1 + np.cos(np.pi*step/max(total, 1)))


def get_lr_and_wd(step):
    if step <= FB_ONLY_STEPS:
        lr_min = FB_ONLY_LR_MAX * FB_ONLY_LR_MIN_FRAC
        return cosine_lr(step-1, FB_ONLY_STEPS, FB_ONLY_LR_MAX, lr_min), 0.0
    s      = step - FB_ONLY_STEPS
    lr_min = FD_ANNEAL_LR_MAX * FD_ANNEAL_LR_MIN_FRAC
    lr     = cosine_lr(s-1, FD_ANNEAL_STEPS, FD_ANNEAL_LR_MAX, lr_min)
    frac   = (s-1) / max(FD_ANNEAL_STEPS-1, 1)
    wD     = (frac ** ANNEAL_POWER) * MAX_WD
    return lr, wD

# OPTIMIZATION

TOTAL_STEPS = FB_ONLY_STEPS + FD_ANNEAL_STEPS

print(f"\n{'='*70}")
print(f"Stage 1 (fB only):   {FB_ONLY_STEPS} steps  LR {FB_ONLY_LR_MAX} → {FB_ONLY_LR_MAX*FB_ONLY_LR_MIN_FRAC:.4f}")
print(f"Stage 2 (fD anneal): {FD_ANNEAL_STEPS} steps  LR {FD_ANNEAL_LR_MAX} → {FD_ANNEAL_LR_MAX*FD_ANNEAL_LR_MIN_FRAC:.5f}  wD 0→{MAX_WD} (power={ANNEAL_POWER})")
print(f"Volume targeting: disabled")
print(f"{'='*70}")

t_start = time.time()
rng = np.random.default_rng(42)

start_list  = [np.zeros(n_magnets, np.float64), rng.uniform(-1.0, 1.0, n_magnets).astype(np.float64)]
start_names = ["zero", "rand0"]

pho = jnp.asarray(np.stack(start_list[:N_PARALLEL_STARTS], axis=0), jnp.float32)
mom = jnp.zeros_like(pho)
var = jnp.zeros_like(pho)

lr0, wd0 = get_lr_and_wd(1)
pho, mom, var = adam_step(pho, mom, var, f32(1), f32(lr0), f32(1.0), f32(wd0))
_ = pho.block_until_ready()
print("JIT compiled.")

for step in range(2, TOTAL_STEPS + 1):
    lr, wD = get_lr_and_wd(step)
    pho, mom, var = adam_step(pho, mom, var, f32(float(step)), f32(lr), f32(1.0), f32(wD))

    if step <= FB_ONLY_STEPS and (step % LOG_INTERVAL == 0 or step == FB_ONLY_STEPS):
        fB_all, fV_all, _ = compute_metrics(pho)
        best_i = int(jnp.argmin(fB_all))
        pho_np = np.asarray(pho[best_i])
        near0  = int(np.sum(np.abs(pho_np) < 0.05))
        near1  = int(np.sum(np.abs(pho_np) > 0.95))
        mid    = n_magnets - near0 - near1
        print(f"[Stage 1] step {step:5d}/{FB_ONLY_STEPS}  best fB={float(fB_all[best_i]):.4e}  fV={float(fV_all[best_i]):.1f}  lr={lr:.4f}  near0:{near0}  near1:{near1}  mid:{mid}")

    if step > FB_ONLY_STEPS and (step % LOG_INTERVAL == 0 or step == TOTAL_STEPS):
        fB_t, fV_t, fD_t = compute_metrics(pho)
        fB_np  = np.asarray(fB_t)
        best_i = int(np.argmin(fB_np))
        pho_np = np.asarray(pho[best_i])
        near0  = int(np.sum(np.abs(pho_np) < 0.05))
        near1  = int(np.sum(np.abs(pho_np) > 0.95))
        mid    = n_magnets - near0 - near1
        disc   = float(near0 + near1) / n_magnets * 100
        s      = step - FB_ONLY_STEPS
        print(f"[Stage 2] step {s:5d}/{FD_ANNEAL_STEPS}  fB={fB_np[best_i]:.4e}  "
              f"fV={float(fV_t[best_i]):.1f}  fD={float(fD_t[best_i]):.4e}  "
              f"wD={wD:.5f}  disc={disc:.0f}%  near0:{near0}  near1:{near1}  mid:{mid}")

fB_cont, fV_cont, fD_cont = [np.asarray(x) for x in compute_metrics(pho)]
pho_all = np.asarray(pho, np.float64)

print("\nFinal continuous results:")
for i, name in enumerate(start_names[:N_PARALLEL_STARTS]):
    print(f"  {name:>8s}  fB={fB_cont[i]:.4e}  fV={fV_cont[i]:.1f}  "
          f"{'below gen0' if fB_cont[i] < fB_gen0 else 'above gen0'}")

discrete_results = []
for i, name in enumerate(start_names[:N_PARALLEL_STARTS]):
    pho_d = np.zeros(n_magnets, np.float64)
    pho_d[pho_all[i] >  0.5] =  1.0
    pho_d[pho_all[i] < -0.5] = -1.0
    bn    = G_f32.astype(np.float64) @ pho_d + Bn_f32.astype(np.float64)
    fB_d  = float(0.5 * np.dot(bn, bn) * area_weight)
    fV_d  = float(volume_per_cell * np.sum(np.abs(pho_d)))
    n_act = int(np.sum(np.abs(pho_d) > 0.5))
    discrete_results.append({
        "name": name, "pho_continuous": pho_all[i], "pho_discrete": pho_d,
        "fB": fB_d, "fV": fV_d, "n_active": n_act,
        "n_positive": int(np.sum(pho_d >  0.5)),
        "n_negative": int(np.sum(pho_d < -0.5)),
        "n_off":      n_magnets - n_act,
    })

print("\nFinal discrete results:")
for r in discrete_results:
    print(f"  {r['name']:>8s}  fB={r['fB']:.4e}  fV={r['fV']:.1f}  active={r['n_active']}")

best           = min(discrete_results, key=lambda x: x["fB"])
pho_discrete   = best["pho_discrete"]
pho_continuous = best["pho_continuous"]
fB_final       = best["fB"]
fV_final       = best["fV"]
n_active       = best["n_active"]
fD_final       = float(np.sum(np.abs(pho_discrete) * (1.0 - np.abs(pho_discrete))))
total_time     = time.time() - t_start

print(f"\n{'='*70}")
print("FINAL RESULT")
print(f"{'='*70}")
print(f"Best start      = {best['name']}")
print(f"fB (continuous) = {fB_cont[start_names.index(best['name'])]:.4e}")
print(f"fB (discrete)   = {fB_final:.4e}")
print(f"fV              = {fV_final:.1f} cm³")
print(f"fD              = {fD_final:.4e}")
print(f"Active magnets  = {n_active} / {n_magnets} ({100*n_active/n_magnets:.1f}%)")
print(f"  North (+1)    = {best['n_positive']}")
print(f"  South (-1)    = {best['n_negative']}")
print(f"  Off   (0)     = {best['n_off']}")
print(f"Below gen0      = {fB_final < fB_gen0}")
print(f"Total time      = {total_time:.1f}s ({total_time/60:.1f} min)")

np.save("pho_optimized.npy",  pho_discrete)
np.save("pho_continuous.npy", pho_continuous)
np.save("grid_positions.npy", magnet_positions)
np.save("grid_moments.npy",   magnet_moments)
print("\nSaved: pho_optimized.npy, pho_continuous.npy, grid_positions.npy, grid_moments.npy")

del G_jax, Bn_jax
gc.collect()
jax.clear_caches()

# PLOTTING


nphi_s, ntheta_s = surface_xyz.shape[0], surface_xyz.shape[1]
phi_coords   = np.linspace(0, 1, nphi_s)
theta_coords = np.linspace(0, 1, ntheta_s)

Bn_before = Bn_f32.astype(np.float64).reshape(nphi_s, ntheta_s)
Bn_pm     = (G_f32.astype(np.float64) @ pho_discrete).reshape(nphi_s, ntheta_s)
Bn_after  = Bn_before + Bn_pm

abs_max = max(np.abs(Bn_before).max(), np.abs(Bn_after).max())
levels  = np.linspace(-abs_max, abs_max, 21)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].contourf(phi_coords, theta_coords, Bn_before.T, levels=levels, cmap="RdBu_r", extend="both")
axes[0].set_title("TF coils only (before)", fontsize=12)
axes[0].set_xlabel("phi"); axes[0].set_ylabel("theta")
plt.colorbar(im1, ax=axes[0], label="B·n [T]")
rms_before = float(np.sqrt(np.mean(Bn_before**2)))
axes[0].text(0.02, 0.97, f"RMS = {rms_before:.3e} T", transform=axes[0].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

im2 = axes[1].contourf(phi_coords, theta_coords, Bn_after.T, levels=levels, cmap="RdBu_r", extend="both")
axes[1].set_title("Coils + PMs (after)", fontsize=12)
axes[1].set_xlabel("phi")
plt.colorbar(im2, ax=axes[1], label="B·n [T]")
rms_after = float(np.sqrt(np.mean(Bn_after**2)))
axes[1].text(0.02, 0.97, f"RMS = {rms_after:.3e} T", transform=axes[1].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.suptitle(f"fB={fB_final:.2e}  fV={fV_final:.0f} cm³  RMS: {rms_before:.3e} → {rms_after:.3e} T  ({rms_before/rms_after:.1f}×)", fontsize=12)
plt.tight_layout()
plt.savefig("Bn_before_after.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved Bn_before_after.png")
