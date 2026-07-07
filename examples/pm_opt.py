#!/usr/bin/env python3
"""
PM dipole optimizer — fB minimization with discreteness annealing.

Starts at a known/loaded pho solution, relaxes it
continuously to minimize fB (Stage 1), then anneals a discreteness
penalty back up with a power-law schedule to recover a
+,-1/0 solution (Stage 2).

- nfp/stellsym are read from the SURFACE file so the
  optimizer can never accidentally run with the wrong symmetry.
- TF coil field uses essos Coils_from_simsopt + essos.fields.BiotSavart.
  simsopt is used only to parse the FOCUS coil-file format; all field math
  is essos's own. Verified against simsopt's BiotSavart
  when passing ALL physical coils with nfp=1, stellsym=False — MUSE's 16 TF
  coils are not copies of a smaller set, so do
  not pass a reduced set with nfp>1 here (it gives wrong results).
- PM dipole-grid G matrix uses DipoleField.compute_interaction_matrix,
  which correctly exploits the grid's real nfp/stellsym symmetry
- Swap the four INPUT FILES paths below to switch problems (MUSE, the
  rotating-ellipse example is included, commented out).
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np


# INPUT FILES — passed as command-line arguments
if len(sys.argv) != 4:
    sys.exit("Usage: python plot.py <surf_file> <mag_file> <coil_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])
COIL_FILE = Path(sys.argv[3])


SURFACE_RANGE = "half period"
SURFACE_NPHI   = 64
SURFACE_NTHETA = 64

OUTPUT_DIR = Path(__file__).resolve().parent / "pm_opt_output"


# OPTIMIZER CONFIG

JAX_PLATFORM = "cpu"
CPU_THREADS  = 4
ENABLE_X64   = True

FB_ONLY_STEPS       = 1000
FB_ONLY_LR_MAX      = 0.01
FB_ONLY_LR_MIN_FRAC = 0.1

FD_ANNEAL_STEPS       = 1000
FD_ANNEAL_LR_MAX      = 0.001
FD_ANNEAL_LR_MIN_FRAC = 0.001

MAX_WD        = 0.2
WD_RAMP_POWER = 4      # power-law annealing: wD = (progress)^WD_RAMP_POWER * MAX_WD
LOG_INTERVAL  = 500

# Volume targeting: ONE-SIDED penalty, fires only above target (never pulls volume up)
VOLUME_TARGET_CM3 = 0.0   # 0 disables
W_VOLUME_TARGET   = 1.0

B_MAX_T = 1.465
MU0     = 4 * np.pi * 1e-7


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

# for p in [str(ESSOS_ROOT), str(SIMSOPT_SRC)]:
#     if p not in sys.path:
#         sys.path.insert(0, p)

from essos.fields import DipoleField

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
    from essos.coils import Coils_from_simsopt

    try:
        from simsopt.field.coil import load_coils_from_makegrid_file
        coils = load_coils_from_makegrid_file(str(DEFAULT_COIL_FILE), order = 10)
        
    except:
        from simsopt.util.permanent_magnet_helper_functions import read_focus_coils


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
        np.asarray(pos_list, np.float64),
        np.asarray(mom_list, np.float64),
        np.asarray(pho_list, np.float64),
        np.asarray(ic_list,  np.float64),
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
    

print(f"\n--- Loading surface: {SURF_FILE.name} ---")
surface = load_surface(SURF_FILE, SURFACE_RANGE, SURFACE_NPHI, SURFACE_NTHETA)


nfp      = int(surface.nfp)
stellsym = bool(surface.stellsym)
print(f"Surface nfp={nfp}  stellsym={stellsym}  (read from surface file)")

surf_xyz = np.asarray(surface.gamma(), np.float64)
surf_nrm = np.asarray(surface.unitnormal(), np.float64)
surf_pts = surf_xyz.reshape(-1, 3)
surf_n   = surf_nrm.reshape(-1, 3)
area_weight = float(surface.area() / len(surf_pts))

print(f"\n--- Loading coils (essos-native): {COIL_FILE.name} ---")
essos_coils = load_coils_essos(COIL_FILE)
Bn_fixed = compute_Bn_fixed_essos(essos_coils, surf_pts, surf_n)

print(f"\n--- Loading magnet grid: {MAG_FILE.name} ---")
magnet_positions, magnet_moments_raw, pho_loaded, ic_flags = load_magnet_grid(MAG_FILE)


PORT_MASK = ic_flags < 0.5
n_excluded = int(np.sum(PORT_MASK))
if n_excluded > 0:
    print(f"Ic flag: {n_excluded} excluded sites found (Ic=0) — these magnet "
          f"sites will be excluded (forced pho=0) for the whole run")
    pho_loaded = np.where(PORT_MASK, 0.0, pho_loaded)
else:
    print("Ic flag: no excluded sites found (Ic=1 everywhere) — all magnet sites available")
n_magnets = len(magnet_positions)

native_norms = np.linalg.norm(magnet_moments_raw, axis=1)
n_zero_moment = int(np.sum(native_norms == 0))
if n_zero_moment > 0:
    print(f"WARNING: {n_zero_moment} zero-moment magnets - excluded like Ic=0 sites")
native_norms_safe = np.where(native_norms > 0, native_norms, 1.0)
M0_SCALE = float(np.mean(native_norms[native_norms > 0])) if np.any(native_norms > 0) else 0.0
magnet_orientations = magnet_moments_raw / native_norms_safe[:, None]
magnet_orientations = np.where(native_norms[:, None] > 0, magnet_orientations, 0.0)
magnet_moments = magnet_orientations * M0_SCALE
ZERO_MOMENT_MASK = native_norms == 0
EXCLUDE_MASK = PORT_MASK | ZERO_MOMENT_MASK
if n_zero_moment > 0:
    pho_loaded = np.where(ZERO_MOMENT_MASK, 0.0, pho_loaded)

M_MAX = B_MAX_T / MU0
volume_per_cell = (M0_SCALE / M_MAX) * 1e6
SYMMETRY_MULTIPLIER = nfp * (2 if stellsym else 1)
print(f"Symmetry: nfp={nfp}, stellsym={stellsym} -> x{SYMMETRY_MULTIPLIER} for full-device volume (fV prints are per-unique-domain)")


print(f"Magnets loaded: {n_magnets}  |  mean M0 (from file) = {M0_SCALE:.6f} A\u00b7m\u00b2  |  "
      f"V_cell={volume_per_cell:.4f} cm\u00b3")
print(f"  (V_cell assumes N52 NdFeB magnets, remnant field Br={B_MAX_T} T; "
      f"M0 is the MEAN |moment| across all loaded magnets, not a fixed nominal value)")
print(f"Loaded pho: {int(np.sum(np.abs(pho_loaded) > 0.5))} active magnets "
      f"({100*np.sum(np.abs(pho_loaded) > 0.5)/n_magnets:.1f}%)")

print("=" * 70)
print("PM Optimization — fB + Discreteness (start at loaded solution)")
print(f"Backend: {backend_name}")
print("=" * 70)


# BUILD G MATRIX via DipoleField.compute_interaction_matrix
print("\n--- Build G matrix ---")
JAX_DTYPE        = jnp.float32
surface_pts_flat = jnp.asarray(surf_pts, JAX_DTYPE)
surface_nrm_flat = jnp.asarray(surf_n,   JAX_DTYPE)

t0 = time.time()
dipole_field = DipoleField(
    jnp.asarray(magnet_positions, JAX_DTYPE),
    jnp.asarray(magnet_moments,   JAX_DTYPE),
    jnp.zeros(n_magnets, JAX_DTYPE),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)
G = np.asarray(dipole_field.compute_interaction_matrix(surface_pts_flat, surface_nrm_flat), np.float32)
n_nan, n_inf = int(np.sum(np.isnan(G))), int(np.sum(np.isinf(G)))
if n_nan > 0 or n_inf > 0:
    raise RuntimeError(f"G has {n_nan} NaN / {n_inf} Inf values - check for zero-moment or malformed rows in {MAG_FILE.name}")
gc.collect()
print(f"G: {G.shape}  {G.nbytes/1e9:.2f} GB  {time.time()-t0:.1f}s")

G_jax  = jnp.asarray(G)
Bn_jax = jnp.asarray(Bn_fixed, jnp.float32)
aw_jax = jnp.float32(area_weight)
vc_jax = jnp.float32(volume_per_cell)
f32    = jnp.float32

fB_gen0 = float(0.5 * np.dot(pho_loaded @ G.T.astype(np.float64) + Bn_fixed,
                             pho_loaded @ G.T.astype(np.float64) + Bn_fixed) * area_weight)
print(f"fB (loaded solution) = {fB_gen0:.4e}")

fB_ref  = jnp.float32(max(fB_gen0, 1e-20))
Vt_jax  = jnp.float32(VOLUME_TARGET_CM3)
wVT_jax = jnp.float32(W_VOLUME_TARGET)


# Define loss functions
@jax.jit
def compute_metrics(pho):
    bn    = pho @ G_jax.T + Bn_jax
    fB    = f32(0.5) * jnp.sum(bn * bn) * aw_jax
    abs_p = jnp.sqrt(pho * pho + f32(1e-7))
    fV    = vc_jax * jnp.sum(abs_p)
    fD    = jnp.sum(abs_p * (f32(1) - abs_p))
    return fB, fV, fD


@jax.jit
def adam_step(pho, m, v, lr, w_fD, port_mask):
    """One Adam step: L = fB/fB_ref + w_fD * fD + wVT * max(fV-Vt, 0).
    """
    def loss(x):
        bn    = x @ G_jax.T + Bn_jax
        fB    = f32(0.5) * jnp.sum(bn * bn) * aw_jax
        abs_p = jnp.sqrt(x * x + f32(1e-7))
        fV    = vc_jax * jnp.sum(abs_p)
        fD    = jnp.sum(abs_p * (f32(1) - abs_p))
        fVT_raw = jnp.maximum(fV - Vt_jax, f32(0))
        fVT     = jnp.where(Vt_jax > f32(0), fVT_raw, f32(0))
        return fB / fB_ref + w_fD * fD + fVT * wVT_jax, (fB, fV, fD)
    b1, b2, eps = f32(0.9), f32(0.999), f32(1e-8)
    (_, aux), g = jax.value_and_grad(loss, has_aux=True)(pho)
    m    = b1*m + (1-b1)*g
    v    = b2*v + (1-b2)*g*g
    pho  = jnp.clip(pho - lr * m / (jnp.sqrt(v) + eps), -1, 1)
    pho  = jnp.where(port_mask, f32(0), pho)
    return pho, m, v, aux


def cosine_lr(step, total, lr_max, lr_min):
    return lr_min + 0.5*(lr_max - lr_min)*(1 + np.cos(np.pi*step/max(total, 1)))


def get_lr_and_wd(step):
    if step <= FB_ONLY_STEPS:
        lr_min = FB_ONLY_LR_MAX * FB_ONLY_LR_MIN_FRAC
        return cosine_lr(step-1, FB_ONLY_STEPS, FB_ONLY_LR_MAX, lr_min), 0.0
    s      = step - FB_ONLY_STEPS
    lr_min = FD_ANNEAL_LR_MAX * FD_ANNEAL_LR_MIN_FRAC
    lr     = cosine_lr(s-1, FD_ANNEAL_STEPS, FD_ANNEAL_LR_MAX, lr_min)
    wD     = ((s-1) / max(FD_ANNEAL_STEPS-1, 1)) ** WD_RAMP_POWER * MAX_WD
    return lr, wD



# Optimization

TOTAL_STEPS = FB_ONLY_STEPS + FD_ANNEAL_STEPS

print(f"\n{'='*70}")
print(f"Stage 1 (fB only):   {FB_ONLY_STEPS} steps  LR {FB_ONLY_LR_MAX} \u2192 {FB_ONLY_LR_MAX*FB_ONLY_LR_MIN_FRAC:.4f}")
print(f"Stage 2 (fD anneal): {FD_ANNEAL_STEPS} steps  LR {FD_ANNEAL_LR_MAX} \u2192 {FD_ANNEAL_LR_MAX*FD_ANNEAL_LR_MIN_FRAC:.5f}  wD 0\u2192{MAX_WD} (power={WD_RAMP_POWER})")
print(f"Volume target: {'disabled' if VOLUME_TARGET_CM3 <= 0 else f'{VOLUME_TARGET_CM3:.0f} cm3, wVT={W_VOLUME_TARGET} (one-sided)'}")
print(f"{'='*70}")

G64  = G.astype(np.float64)
Bn64 = Bn_fixed.astype(np.float64)
fB_coils = float(0.5 * np.dot(Bn64, Bn64) * area_weight)
print(f"[ref] fB coils-only:   {fB_coils:.4e}")
print(f"[ref] fB loaded:       {fB_gen0:.4e}   fV: {volume_per_cell*np.abs(pho_loaded).sum():.1f} cm\u00b3")


def fB64_of(p):
    r = p @ G64.T + Bn64
    return 0.5 * np.sum(r*r) * area_weight


t_start = time.time()

pho = jnp.asarray(np.clip(pho_loaded, -1, 1), jnp.float32)
port_mask_jax = jnp.asarray(EXCLUDE_MASK)
mom = jnp.zeros_like(pho)
var = jnp.zeros_like(pho)

lr0, wd0 = get_lr_and_wd(1)
hist_fB, hist_fV, hist_fD = [], [], []
pho, mom, var, aux0 = adam_step(pho, mom, var, f32(lr0), f32(wd0), port_mask_jax)
hist_fB.append(aux0[0]); hist_fV.append(aux0[1]); hist_fD.append(aux0[2])
_ = pho.block_until_ready()
print("JIT compiled.")

fB_s1 = None
for step in range(2, TOTAL_STEPS + 1):
    lr, wD = get_lr_and_wd(step)
    pho, mom, var, aux = adam_step(pho, mom, var, f32(lr), f32(wD), port_mask_jax)
    hist_fB.append(aux[0]); hist_fV.append(aux[1]); hist_fD.append(aux[2])

    if step <= FB_ONLY_STEPS and (step % LOG_INTERVAL == 0 or step == FB_ONLY_STEPS):
        fB_t, fV_t, _ = compute_metrics(pho)
        clip = float(jnp.mean(jnp.abs(pho) > 0.99)) * 100
        print(f"[Stage 1] {step:5d}/{FB_ONLY_STEPS}  fB={float(fB_t):.4e}  "
              f"fV={float(fV_t):.1f}  clip={clip:.1f}%  lr={lr:.4f}")

    if step == FB_ONLY_STEPS:
        p1    = np.asarray(pho, np.float64)
        fB_s1 = fB64_of(p1)
        ap1   = np.abs(p1)
        lo, mid_lo, mid_hi = np.mean(ap1 < 0.05), np.mean((ap1 >= 0.05) & (ap1 < 0.5)), np.mean((ap1 >= 0.5) & (ap1 < 0.95))
        p_r   = np.sign(p1) * (ap1 > 0.5)
        print(f"[S1 done] fB={fB_s1:.4e}  ({fB_s1/fB_coils:.1e}x coils, {fB_s1/fB_gen0:.2f}x loaded)")
        print(f"[S1 dist] |p|<.05: {lo*100:.0f}%   .05-.5: {mid_lo*100:.0f}%   .5-.95: {mid_hi*100:.0f}%")
        print(f"[control] naive-round fB={fB64_of(p_r):.4e}  fV={volume_per_cell*np.abs(p_r).sum():.1f}")

    if step > FB_ONLY_STEPS and (step % LOG_INTERVAL == 0 or step == TOTAL_STEPS):
        fB_t, fV_t, fD_t = compute_metrics(pho)
        disc = float(jnp.mean((jnp.abs(pho) < 0.05) | (jnp.abs(pho) > 0.95))) * 100
        s    = step - FB_ONLY_STEPS
        print(f"[Anneal] {s:5d}/{FD_ANNEAL_STEPS}  fB={float(fB_t):.4e} ({float(fB_t)/fB_s1:.1f}x S1)  "
              f"fV={float(fV_t):.1f}  fD={float(fD_t):.1f}  wD={wD:.4f}  disc={disc:.0f}%")

p   = np.asarray(pho, np.float64)
p_d = np.sign(p) * (np.abs(p) > 0.5)
p_d = np.where(EXCLUDE_MASK, 0.0, p_d) 
fB_fin, fB_rnd = fB64_of(p), fB64_of(p_d)

print(f"\n{'='*70}")
print(f"Done in {time.time()-t_start:.0f}s")
print(f"  Stage 1 (continuous bound): {fB_s1:.4e}")
print(f"  annealed (still smooth):    {fB_fin:.4e}  ({fB_fin/fB_s1:.1f}x S1)")
print(f"  hard-rounded (buildable):   {fB_rnd:.4e}  ({fB_rnd/fB_s1:.1f}x S1)  fV={volume_per_cell*np.abs(p_d).sum():.1f}")
print(f"  vs loaded: {fB_rnd/fB_gen0:.2f}x   vs coils-only: {fB_rnd/fB_coils:.1e}x")

n_active = int(np.sum(np.abs(p_d) > 0.5))
print(f"  Active magnets: {n_active} / {n_magnets} ({100*n_active/n_magnets:.1f}%)")
print(f"    North (+1): {int(np.sum(p_d > 0.5))}   South (-1): {int(np.sum(p_d < -0.5))}   Off: {n_magnets - n_active}")
print(f"{'='*70}")



write_focus_file(OUTPUT_DIR / "pm_optimized_test.focus", magnet_positions, magnet_moments, p_d,Ic_passed = ic_flags)

# ============================================================
# CONVERGENCE HISTORY PLOT — fB, fV, fD per iteration
# ============================================================
import matplotlib.pyplot as plt
hist_fB_np = np.array([float(x) for x in hist_fB])
hist_fV_np = np.array([float(x) for x in hist_fV])
hist_fD_np = np.array([float(x) for x in hist_fD])
steps_axis = np.arange(1, len(hist_fB_np) + 1)
np.savez(OUTPUT_DIR / "convergence_history.npz",
         fB=hist_fB_np, fV=hist_fV_np, fD=hist_fD_np)

fig_h, axes_h = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes_h[0].plot(steps_axis, hist_fB_np, color="steelblue", linewidth=1)
axes_h[0].set_yscale("log")
axes_h[0].set_ylabel("fB")
axes_h[0].set_title("Convergence history")
axes_h[0].grid(alpha=0.3)

axes_h[1].plot(steps_axis, hist_fV_np, color="seagreen", linewidth=1)
axes_h[1].set_ylabel("fV [cm$^3$] (unique domain)")
axes_h[1].grid(alpha=0.3)

axes_h[2].plot(steps_axis, hist_fD_np, color="purple", linewidth=1)
axes_h[2].set_yscale("log")
axes_h[2].set_ylabel("fD")
axes_h[2].set_xlabel("iteration")
axes_h[2].grid(alpha=0.3)

for ax in axes_h:
    ax.axvline(FB_ONLY_STEPS, color="gray", linestyle="--", alpha=0.7)
axes_h[0].text(FB_ONLY_STEPS, axes_h[0].get_ylim()[1], "  Stage 2 (fD anneal) starts",
               va="top", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "convergence_history.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved {OUTPUT_DIR}/convergence_history.png and convergence_history.npz")
