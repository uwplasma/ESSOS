
#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from essos.fields import DipoleField
import numpy as np
import jax
import jax.numpy as jnp

JAX_PLATFORM    = "cpu"   
CPU_THREADS     = 4
ENABLE_X64      = True

N_PARALLEL_STARTS = 4     



VOLUME_TARGET_CM3 = 2000.0    # Target Volume
W_VOLUME_TARGET   = 1.0       # penalty weight

#REFERENCE_M0_SCALE = 0.074625   # zot80 dipole moment magnitude 
#B_MAX_T = 1.465                  
MU0     = 4 * np.pi * 1e-7

SURFACE_RANGE  = "half period"
SURFACE_NPHI   = 64
SURFACE_NTHETA = 64


FB_ONLY_STEPS    = 8000   # Stage 1: minimize fB only (continuous relaxation)
FB_ONLY_LR_MAX      = 0.01
FB_ONLY_LR_MIN_FRAC = 0.1

FD_ANNEAL_LR_MAX      = 0.001
FD_ANNEAL_LR_MIN_FRAC = 0.001
MAX_WD       = .2
FD_ANNEAL_STEPS  = 6000   # Stage 2: anneal discreteness penalty 
WD_RAMP_POWER         = 4    


LOG_INTERVAL = 500




ESSOS_ROOT  = Path('').resolve().parents[2]
SIMSOPT_SRC = ESSOS_ROOT.parent / "simsopt" / "src"

#DEFAULT_SURF_FILE = ESSOS_ROOT / "ESSOS" / "input.muse"
DEFAULT_SURF_FILE = ESSOS_ROOT / "ESSOS" / "essos" / "examples" / "input_rot_ellipse_nfp8_e_1p98"

#DEFAULT_COIL_FILE = SIMSOPT_SRC.parent / "tests" / "test_files" / "muse_tf_coils.focus"
DEFAULT_COIL_FILE = ESSOS_ROOT / "ESSOS" / "essos" / "examples" / "rot_ellipse_nfp8_e_198.focus"


#DEFAULT_MAG_FILE  = ESSOS_ROOT / "ESSOS" / "examples" / "input_files" / "zot80.focus"
DEFAULT_MAG_FILE  = ESSOS_ROOT / "ESSOS" / "essos" / "examples" / "dipole_grid_nfp8.focus"

DEFAULT_INPUT_BUNDLE = ESSOS_ROOT / "ESSOS" / "examples" / "input_files" / "muse_opt_inputs_64x64.npz"

TEST_DIR = (Path().parent / ".." / ".." / ".." / "simsopt" / "tests" / "test_files").resolve()



out_dir = Path("output_permanent_magnet_GPMO_MUSE")
out_dir.mkdir(parents=True, exist_ok=True)

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


jax.config.update("jax_enable_x64", ENABLE_X64)

backend_name = str(jax.default_backend()).lower()
print(f"JAX backend: {backend_name}  |  devices: {[str(d) for d in jax.devices()]}")



for p in [str(ESSOS_ROOT), str(SIMSOPT_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)



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
    from simsopt.util.permanent_magnet_helper_functions import initialize_coils_for_pm_optimization

    try:
        surface = SurfaceRZFourier.from_vmec_input(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)

    except:
        surface  = SurfaceRZFourier.from_focus(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)

        
    surf_xyz = np.asarray(surface.gamma(),      np.float64)
    surf_nrm = np.asarray(surface.unitnormal(), np.float64)
    surf_pts = surf_xyz.reshape(-1, 3)
    surf_n   = surf_nrm.reshape(-1, 3)
    area_w   = float(surface.area() / len(surf_pts))

    try:
        from simsopt.field.coil import load_coils_from_makegrid_file
        coils = load_coils_from_makegrid_file(str(DEFAULT_COIL_FILE), order = 10)
        
    except:
        print('error surface file not recognized')

    bs       = BiotSavart(coils)
    bs.set_points(surf_pts)
    Bn_fixed = np.sum(np.asarray(bs.B(), np.float64) * surf_n, axis=1)

    pos_list, mom_list, pho_list = [], [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f.readlines()[3:]:
            tokens = line.replace(",", " ").split()
            if len(tokens) < 12:
                continue
            x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
            m0      = float(tokens[7])
            pho =   float(tokens[8])
            az, pol = float(tokens[10]), float(tokens[11])
            pos_list.append((x, y, z))
            mom_list.append((m0*np.cos(az)*np.sin(pol), m0*np.sin(az)*np.sin(pol), m0*np.cos(pol)))
            pho_list.append(pho)

    return (
        surf_xyz, surf_nrm, Bn_fixed, area_w,
        np.asarray(pos_list, np.float64),
        np.asarray(mom_list, np.float64),
        np.asarray(pho_list),
    )


required = ["surface_xyz", "surface_normal", "Bn_fixed", "area_w", "positions", "moments"]
missing  = [v for v in required if v not in globals()]

if missing:
    if DEFAULT_INPUT_BUNDLE and os.path.exists(DEFAULT_INPUT_BUNDLE):
        print(f"Loading from bundle: {DEFAULT_INPUT_BUNDLE}")
        surface_xyz, surface_normal, Bn_fixed, area_w, positions, moments, _ = load_from_bundle(DEFAULT_INPUT_BUNDLE)
    else:
        print("Loading from surface/coil/magnet files...")
        surface_xyz, surface_normal, Bn_fixed, area_w, positions, moments, pho_loaded = load_from_files(
            DEFAULT_SURF_FILE, DEFAULT_COIL_FILE, DEFAULT_MAG_FILE,
            SURFACE_RANGE, SURFACE_NPHI, SURFACE_NTHETA,
        )

area_weight = float(area_w)/ (SURFACE_NPHI*SURFACE_NPHI)
surface_xyz    = np.asarray(surface_xyz,    np.float64)
surface_normal = np.asarray(surface_normal, np.float64)

loaded_norms = np.linalg.norm(np.asarray(moments), axis=1)
print(f"loaded |m| mean: {loaded_norms.mean():.6f}")



# Grid: Zot80

#M_MAX = B_MAX_T / MU0

magnet_positions    = np.asarray(positions, np.float64)
#native_norms        = np.linalg.norm(np.asarray(moments), axis=1)
magnet_orientations = np.asarray(moments, np.float64)  #/ native_norms[:, None]
n_magnets           = len(magnet_positions)
#M0_SCALE            = REFERENCE_M0_SCALE
volume_per_cell     =  6.4012e-08 * 1e6 # user input at the moment 

print("=" * 70)
print("MUSE PM Optimization — fB + Discreteness")
print(f"Backend: {backend_name}  |  Starts: {N_PARALLEL_STARTS}")
#print(f"Magnets: {n_magnets}  |  M0={M0_SCALE:.6f} A·m²  |  V_cell={volume_per_cell:.4f} cm³")
print("=" * 70)




from compute_G_symmetric import compute_G_symmetric

print("\n--- Build G matrix ---")
magnet_moments   = magnet_orientations #* M0_SCALE
JAX_DTYPE        = jnp.float32
surface_pts_flat = jnp.asarray(surface_xyz.reshape(-1, 3),    JAX_DTYPE)
surface_nrm_flat = jnp.asarray(surface_normal.reshape(-1, 3), JAX_DTYPE)

t0 = time.time()
G_f32 = np.asarray(compute_G_symmetric(
    jnp.asarray(magnet_positions, JAX_DTYPE),
    jnp.asarray(magnet_moments,   JAX_DTYPE),
    surface_pts_flat,
    surface_nrm_flat,
    nfp=8,
    stellsym=True,
), np.float32)
Bn_f32 = np.asarray(Bn_fixed, np.float32)
gc.collect()




G_jax  = jnp.asarray(G_f32)
Bn_jax = jnp.asarray(Bn_f32)
aw_jax = jnp.float32(area_weight)
vc_jax = jnp.float32(volume_per_cell)

fB_gen0 = float(0.5 * np.sum( (pho_loaded@ G_f32.T + Bn_jax) * ( pho_loaded@ G_jax.T + Bn_jax), axis = -1) * (np.pi/SURFACE_NTHETA) * (np.pi/SURFACE_NPHI/4)) 
print(f"G: {G_f32.shape}  {G_f32.nbytes/1e9:.2f} GB  {time.time()-t0:.1f}s")
print(f"fB = {fB_gen0:.4e}")

fB_ref  = jnp.float32(max(fB_gen0, 1e-20))
Vt_jax  = jnp.float32(VOLUME_TARGET_CM3)
wVT_jax = jnp.float32(W_VOLUME_TARGET)
f32     = jnp.float32



@jax.jit
def compute_metrics(pho_batch):
    """fB, fV, fD for a (K, n_magnets) batch."""
    bn    = pho_batch @ G_jax.T + Bn_jax
    fB    = f32(0.5) * jnp.sum(bn * bn, axis=1) *  (np.pi/SURFACE_NTHETA) * (np.pi/SURFACE_NPHI/4)
    abs_p = jnp.sqrt(pho_batch * pho_batch + f32(1e-7))
    fV    = vc_jax * jnp.sum(abs_p, axis=1)
    fD    = jnp.sum(abs_p * (f32(1) - abs_p), axis=1)
    return fB, fV, fD


@jax.jit
def adam_step(pho, m, v, t, lr, w_fB, w_fD):
    """One Adam step: L = w_fB * fB/fB_ref + w_fD * fD + wVT * ((fV-Vt)/Vt)^2."""
    def loss(x):
        bn    = x @ G_jax.T + Bn_jax
        fB    = f32(0.5) * jnp.sum(bn * bn, axis=-1) *  (np.pi/SURFACE_NTHETA) * (np.pi/SURFACE_NPHI/4)
        abs_p = jnp.sqrt(x * x + f32(1e-7))
        fV    = vc_jax * jnp.sum(abs_p, axis=1)
        fD    = jnp.sum(abs_p * (f32(1) - abs_p), axis=1)
        Vt    = jnp.maximum(Vt_jax, f32(1e-6))
        fVT   = jnp.maximum(fV - Vt_jax, f32(0))    # was: jnp.abs(fV - Vt)

        return jnp.sum(w_fB * fB / fB_ref + w_fD * fD + fVT*wVT_jax)

        
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
    #wD     = (s-1) / max(FD_ANNEAL_STEPS-1, 1) * MAX_WD
    wD     = ((s-1) / max(FD_ANNEAL_STEPS-1, 1)) ** WD_RAMP_POWER * MAX_WD
    return lr, wD




TOTAL_STEPS = FB_ONLY_STEPS + FD_ANNEAL_STEPS 


print(f"\n{'='*70}")
print(f"Stage 1 (fB only):   {FB_ONLY_STEPS} steps  LR {FB_ONLY_LR_MAX} → {FB_ONLY_LR_MAX*FB_ONLY_LR_MIN_FRAC:.4f}")
print(f"Stage 2 (fD anneal): {FD_ANNEAL_STEPS} steps  LR {FD_ANNEAL_LR_MAX} → {FD_ANNEAL_LR_MAX*FD_ANNEAL_LR_MIN_FRAC:.4f}  wD 0→{MAX_WD}")
if VOLUME_TARGET_CM3 > 0:
    print(f"Volume target: {VOLUME_TARGET_CM3:.0f} cm³  wVT={W_VOLUME_TARGET}")
else:
    print("Volume targeting: disabled")
print(f"{'='*70}")


# ---- references, printed once ----
qw  = (np.pi/SURFACE_NTHETA) * (np.pi/SURFACE_NPHI/4)
G64 = G_f32.astype(np.float64)
Bn64 = np.asarray(Bn_fixed, np.float64)
fB_coils = 0.5*np.sum(Bn64**2)*qw
print(f"[ref] fB coils-only: {fB_coils:.4e}")
print(f"[ref] fB loaded dipoles:      {fB_gen0:.4e}   fV: {volume_per_cell*np.abs(pho_loaded).sum():.1f} cm³")

def fB64_of(p):
    r = p @ G64.T + Bn64
    return 0.5*np.sum(r*r)*qw

t_start = time.time()
#pho = jnp.ones((1, n_magnets), jnp.float32) * 0 
pho = jnp.asarray(np.clip(pho_loaded, -1, 1), jnp.float32)[None, :]   # start AT the known solution

mom = jnp.zeros_like(pho)
var = jnp.zeros_like(pho)
lr0, wd0 = get_lr_and_wd(1)
pho, mom, var = adam_step(pho, mom, var, f32(1), f32(lr0), f32(1.0), f32(wd0))
_ = pho.block_until_ready()
print("JIT compiled.")

fB_s1 = None
for step in range(2, TOTAL_STEPS + 1):
    lr, wD = get_lr_and_wd(step)
    pho, mom, var = adam_step(pho, mom, var, f32(float(step)), f32(lr), f32(1.0), f32(wD))

    if step <= FB_ONLY_STEPS and (step % LOG_INTERVAL == 0 or step == FB_ONLY_STEPS):
        fB_t, fV_t, _ = compute_metrics(pho)
        ap   = jnp.abs(pho[0])
        clip = float(jnp.mean(ap > 0.99))*100
        print(f"[Stage 1] {step:5d}/{FB_ONLY_STEPS}  fB={float(fB_t[0]):.4e}  "
              f"fV={float(fV_t[0]):.1f}  clip={clip:.1f}%  lr={lr:.4f}")

    if step == FB_ONLY_STEPS:
        p1    = np.asarray(pho, np.float64)[0]
        fB_s1 = fB64_of(p1)
        ap1   = np.abs(p1)
        lo, mid_lo, mid_hi = np.mean(ap1<0.05), np.mean((ap1>=0.05)&(ap1<0.5)), np.mean((ap1>=0.5)&(ap1<0.95))
        p_r   = np.sign(p1)*(ap1 > 0.5)
        print(f"[S1 done] fB={fB_s1:.4e}  ({fB_s1/fB_coils:.1e}x coils, {fB_s1/fB_gen0:.2f}x loaded)")
        print(f"[S1 dist] |p|<.05: {lo*100:.0f}%   .05–.5: {mid_lo*100:.0f}%   .5–.95: {mid_hi*100:.0f}%")
        print(f"[control] naive-round fB={fB64_of(p_r):.4e}  fV={volume_per_cell*np.abs(p_r).sum():.1f}")

    if step > FB_ONLY_STEPS and (step % LOG_INTERVAL == 0 or step == TOTAL_STEPS):
        fB_t, fV_t, fD_t = compute_metrics(pho)
        disc = float(jnp.mean((jnp.abs(pho[0])<0.05)|(jnp.abs(pho[0])>0.95)))*100
        s    = step - FB_ONLY_STEPS
        tag, tot, ss = ("Anneal", FD_ANNEAL_STEPS, s) if s <= FD_ANNEAL_STEPS else ("Polish", POLISH_STEPS, s-FD_ANNEAL_STEPS)
        print(f"[{tag}] {ss:5d}/{tot}  fB={float(fB_t[0]):.4e} ({float(fB_t[0])/fB_s1:.1f}x S1)  "
              f"fV={float(fV_t[0]):.1f}  fD={float(fD_t[0]):.1f}  wD={wD:.3f}  disc={disc:.0f}%")

# ---- final report ----
pho_all = np.asarray(pho, np.float64)
p   = pho_all[0]
p_d = np.sign(p)*(np.abs(p) > 0.5)
fB_fin, fB_rnd = fB64_of(p), fB64_of(p_d)
print(f"\n{'='*70}\nDone in {time.time()-t_start:.0f}s")
print(f"  Stage 1 (continuous bound): {fB_s1:.4e}")
print(f"  annealed (still smooth):    {fB_fin:.4e}  ({fB_fin/fB_s1:.1f}x S1)")
print(f"  hard-rounded (buildable):   {fB_rnd:.4e}  ({fB_rnd/fB_s1:.1f}x S1)  fV={volume_per_cell*np.abs(p_d).sum():.1f}")
print(f"  vs loaded dipoles: {fB_rnd/fB_gen0:.2f}x   vs coils-only: {fB_rnd/fB_coils:.1e}x")

fB_cont, fV_cont, fD_cont = [np.asarray(x) for x in compute_metrics(pho)]
pho_all = np.asarray(pho, np.float64)



start_names = ['zero']
discrete_results = []
for i, name in enumerate(start_names[:N_PARALLEL_STARTS]):
    pho_d = np.zeros(n_magnets, np.float64)
    pho_d[pho_all[i] >  0.5] =  1.0
    pho_d[pho_all[i] < -0.5] = -1.0
    bn    = G_f32.astype(np.float64) @ pho_d + Bn_f32.astype(np.float64)
    fB_d  = float(f32(0.5) * jnp.sum(bn * bn) *  (np.pi/SURFACE_NTHETA) * (np.pi/SURFACE_NPHI/4))
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



nphi = SURFACE_NPHI
ntheta = SURFACE_NTHETA
import matplotlib.pyplot as plt

Bn_PM    = (G_f32 @ pho_discrete).reshape(nphi, ntheta)
Bn_coils = Bn_fixed.reshape(nphi, ntheta)
Bn_total = Bn_PM + Bn_coils

phi   = np.linspace(0, 2 * np.pi, nphi,   endpoint=False)
theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vmax = max(np.abs(Bn_coils).max(), np.abs(Bn_PM).max())
vmin = -vmax

cf1 = axes[0].contourf(phi, theta, Bn_coils.T, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
fig.colorbar(cf1, ax=axes[0], label='Bn [T]')
axes[0].set_title('Coils Bn')
axes[0].set_xlabel('phi')
axes[0].set_ylabel('theta')

cf2 = axes[1].contourf(phi, theta, Bn_PM.T, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
fig.colorbar(cf2, ax=axes[1], label='Bn [T]')
axes[1].set_title('PMs Bn')
axes[1].set_xlabel('phi')


cf2 = axes[2].contourf(phi, theta, Bn_total.T, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
fig.colorbar(cf2, ax=axes[2], label='Bn [T]')
axes[2].set_title('Total Bn')
axes[2].set_xlabel('phi')


plt.tight_layout()
plt.show()