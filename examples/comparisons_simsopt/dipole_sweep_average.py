#!/usr/bin/env python3
"""
Resolution-sweep benchmark: essos vs simsopt dipole field.

Times THREE evaluation paths at each surface resolution (PM dipole grid
only — no coils):

  1. simsopt DipoleField.B()      — reference C++ implementation
  2. essos   DipoleField.B()      — chunked evaluation (memory-safe)
  3. essos   G-path               — compute_interaction_matrix build
       (once) then Bn = G @ pho per evaluation.

TIMING METHODOLOGY (this is the part that was broken before):

  JAX dispatches asynchronously. `y = A @ x` returns a *future* the moment
  the op is queued — the value is not computed until something forces it.
  So timing `t0=time(); y = A@x; dt=time()-t0` on a JAX array measures the
  Python dispatch queue, NOT the GPU compute, and shows ~no speedup when
  you switch CPU->GPU because you were never timing the device.

  Fixes applied throughout:
    * Every timed JAX result is forced with .block_until_ready() INSIDE
      the timed region (via _sync / time_jax).
    * jnp.asarray() on an existing JAX array is lazy and does NOT sync,
      so builds are blocked explicitly too.
    * Small/fast ops are averaged over many repeats after warm-up, not
      measured single-shot, so launch latency doesn't dominate.
    * simsopt's C++ paths are synchronous, so they're timed the same way
      (warm-up + averaged repeats) for an apples-to-apples comparison.

Usage:
  python benchmark_dipole_sweep.py <surf_file> <mag_file>
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np

if len(sys.argv) < 3:
    sys.exit("Usage: python benchmark_dipole_sweep.py <surf_file> <mag_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])

SURFACE_RANGE = "half period"
RESOLUTIONS   = [8, 16, 32, 64]
CHUNK_SIZE    = 128

# Repeat counts for timing. Cheap ops get many repeats so launch/dispatch
# latency averages out; expensive builds get a few.
N_WARMUP_FAST, N_REPEAT_FAST   = 3, 50   # matvecs
N_WARMUP_BUILD, N_REPEAT_BUILD = 1, 3    # interaction-matrix builds

OUTPUT_DIR = Path(__file__).resolve().parent / "output_dipole_field_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ESSOS_ROOT  = Path(__file__).resolve().parents[2]
SIMSOPT_SRC = ESSOS_ROOT.parent / "simsopt" / "src"
for p in [str(ESSOS_ROOT), str(SIMSOPT_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

JAX_PLATFORM = "cuda"
if JAX_PLATFORM != "auto":
    os.environ["JAX_PLATFORMS"] = JAX_PLATFORM

backend_name = str(jax.default_backend()).lower()
print(f"JAX backend: {backend_name}  |  devices: {[str(d) for d in jax.devices()]}")


from essos.fields import DipoleField as EssosDipoleField
from simsopt.field import DipoleField as SimsoptDipoleField
from simsopt.geo import SurfaceRZFourier
import simsoptpp as sopp


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------
def _sync(x):
    """Block until a (possibly nested) JAX result is fully materialized.

    This is the JAX analogue of torch.cuda.synchronize(). Without it, the
    timer stops while the GPU is still working, so you measure dispatch
    overhead instead of compute.
    """
    if isinstance(x, (list, tuple)):
        for xi in x:
            _sync(xi)
    elif hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def time_jax(fn, n_warmup=N_WARMUP_FAST, n_repeat=N_REPEAT_FAST):
    """Time a JAX-producing callable. Returns (last_result, seconds_per_call).

    Warm-up runs absorb JIT compilation and first-touch allocations; the
    timed loop blocks on every result so the device actually finishes.
    """
    for _ in range(n_warmup):
        _sync(fn())
    t0 = time.perf_counter()
    res = None
    for _ in range(n_repeat):
        res = fn()
        _sync(res)
    dt = (time.perf_counter() - t0) / n_repeat
    return res, dt


def time_sync(fn, n_warmup=1, n_repeat=5):
    """Time a synchronous (numpy / C++) callable. Returns (result, s/call)."""
    for _ in range(n_warmup):
        res = fn()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        res = fn()
    dt = (time.perf_counter() - t0) / n_repeat
    return res, dt


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_surface(surf_file, surface_range, nphi, ntheta):
    try:
        surface = SurfaceRZFourier.from_vmec_input(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    except Exception:
        surface = SurfaceRZFourier.from_focus(str(surf_file), range=surface_range, nphi=nphi, ntheta=ntheta)
    return surface


def load_magnet_grid(mag_file):
    pos_list, mom_list, pho_list = [], [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.replace(",", " ").split()
            if len(tokens) < 12:
                continue
            try:
                x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
                m0      = float(tokens[7])
                pho     = float(tokens[8])
                az, pol = float(tokens[10]), float(tokens[11])
            except ValueError:
                continue
            pos_list.append((x, y, z))
            mom_list.append((m0*np.cos(az)*np.sin(pol), m0*np.sin(az)*np.sin(pol), m0*np.cos(pol)))
            pho_list.append(pho)
    return (
        jnp.asarray(pos_list, jnp.float64),
        jnp.asarray(mom_list, jnp.float64),
        jnp.asarray(pho_list, jnp.float64),
    )


def essos_B_chunked(df, pts, chunk_size):
    """Evaluate essos DipoleField.B in memory-bounded chunks of points.

    NOTE: JAX arrays are immutable, so the old `out[i:i+cs] = ...` on a
    jnp.empty buffer was illegal. Collect chunks and concatenate instead.
    """
    chunks = []
    for i in range(0, len(pts), chunk_size):
        chunk = jnp.asarray(pts[i:i+chunk_size], jnp.float64)
        chunks.append(jnp.asarray(df.B(chunk), jnp.float64))
    return jnp.concatenate(chunks, axis=0)


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
print(f"--- Loading magnet grid: {MAG_FILE.name} ---")
positions, moments_raw, pho = load_magnet_grid(MAG_FILE)
n_magnets = len(positions)
scaled_moments = moments_raw * pho[:, None]
print(f"{n_magnets} magnets  |  {int(np.sum(np.abs(np.asarray(pho)) > 0.5))} active")

surface0 = load_surface(SURF_FILE, SURFACE_RANGE, RESOLUTIONS[0], RESOLUTIONS[0])
nfp, stellsym = int(surface0.nfp), bool(surface0.stellsym)
print(f"Surface nfp={nfp}  stellsym={stellsym}")
print(f"essos B() chunk size: {CHUNK_SIZE} points\n")

df_essos = EssosDipoleField(
    jnp.asarray(positions,      jnp.float64),
    jnp.asarray(scaled_moments, jnp.float64),
    jnp.zeros(n_magnets, jnp.float64),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)
df_sims = SimsoptDipoleField(
    np.asarray(positions),
    np.asarray(scaled_moments).flatten(),
    nfp=nfp,
    coordinate_flag="cartesian",
    m_maxima=np.linalg.norm(np.asarray(moments_raw), axis=1),
)

native_norms = jnp.linalg.norm(moments_raw, axis=1)
norms_safe = jnp.where(native_norms > 0, native_norms, 1.0)
orientations = jnp.where(native_norms[:, None] > 0, moments_raw / norms_safe[:, None], 0.0)
M0_SCALE = float(jnp.mean(native_norms[native_norms > 0]))
unit_moments = orientations * M0_SCALE

df_essos_G = EssosDipoleField(
    jnp.asarray(positions,    jnp.float32),
    jnp.asarray(unit_moments, jnp.float32),
    jnp.zeros(n_magnets, jnp.float32),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)

# pho as a device array for the matvec (fp64, to match simsopt's A).
pho64 = jnp.asarray(pho, jnp.float64)

results = {"n_pts": [], "t_sims": [], "t_essos_B": [],
           "t_G_build": [], "t_G_matvec": [],
           "t_A_build": [], "t_A_matvec": [], "maxdiff": []}

# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
for res in RESOLUTIONS:
    surface = load_surface(SURF_FILE, SURFACE_RANGE, res, res)
    surf_pts = jnp.asarray(surface.gamma(), jnp.float64).reshape(-1, 3)
    surf_n   = jnp.asarray(surface.unitnormal(), jnp.float64).reshape(-1, 3)
    surf_pts_np = np.ascontiguousarray(np.asarray(surf_pts))
    surf_n_np   = np.ascontiguousarray(np.asarray(surf_n))
    n_pts = len(surf_pts)

    # 1. simsopt B (synchronous C++): warm-up + averaged repeats.
    def sims_B():
        df_sims.set_points(surf_pts_np)
        return np.asarray(df_sims.B(), np.float64)
    B_sims, t_sims = time_sync(sims_B, n_warmup=1, n_repeat=N_REPEAT_BUILD)

    # 2. essos B, chunked: warm-up + averaged repeats, each fully synced.
    B_essos, t_essos_B = time_jax(
        lambda: essos_B_chunked(df_essos, surf_pts, CHUNK_SIZE),
        n_warmup=1, n_repeat=N_REPEAT_BUILD)

    # 3a. essos G build (compute-bound — this is where the GPU should win).
    #     jnp.asarray on an existing JAX array is lazy, so time_jax blocks it.
    def build_G():
        return jnp.asarray(df_essos_G.compute_interaction_matrix(
            jnp.asarray(surf_pts, jnp.float32),
            jnp.asarray(surf_n,   jnp.float32)), jnp.float32)
    G, t_G_build = time_jax(build_G, n_warmup=N_WARMUP_BUILD, n_repeat=N_REPEAT_BUILD)

    # 3b. essos G matvec (fp64, to match simsopt's fp64 A matvec).
    G64 = _sync(G.astype(jnp.float64))
    _, t_G_matvec = time_jax(lambda: G64 @ pho64)

    # 4. simsopt A-matrix path (apples-to-apples with essos G path).
    #    A has 3x the columns of essos's G (full vector moments vs pho).
    b_dummy = np.zeros(n_pts)
    def build_A():
        return sopp.dipole_field_Bn(
            surf_pts_np,
            np.ascontiguousarray(np.asarray(positions)),
            surf_n_np,
            nfp, int(stellsym),
            np.ascontiguousarray(b_dummy),
            "cartesian", 1.0)
    A_raw, t_A_build = time_sync(build_A, n_warmup=N_WARMUP_BUILD, n_repeat=N_REPEAT_BUILD)
    A_obj = np.asarray(A_raw).reshape(n_pts, n_magnets * 3)

    m_vec = np.ascontiguousarray(np.asarray(scaled_moments).flatten())
    _, t_A_matvec = time_sync(lambda: A_obj @ m_vec, n_warmup=3, n_repeat=N_REPEAT_FAST)

    max_diff = float(jnp.linalg.norm(B_essos - jnp.asarray(B_sims), axis=1).max())

    results["n_pts"].append(n_pts)
    results["t_sims"].append(t_sims)
    results["t_essos_B"].append(t_essos_B)
    results["t_G_build"].append(t_G_build)
    results["t_G_matvec"].append(t_G_matvec)
    results["t_A_build"].append(t_A_build)
    results["t_A_matvec"].append(t_A_matvec)
    results["maxdiff"].append(max_diff)

    print(f"{res:3d}x{res:<3d} ({n_pts:5d} pts):  simsopt B {t_sims:7.3f}s   "
          f"essos B(chunked) {t_essos_B:7.3f}s   "
          f"G build {t_G_build:6.3f}s   G matvec {t_G_matvec*1e3:7.3f}ms   "
          f"A build {t_A_build:6.3f}s   A matvec {t_A_matvec*1e3:7.3f}ms   "
          f"max|dB| {max_diff:.2e} T")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt

n_pts_arr = np.array(results["n_pts"])

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Build time — the compute-bound comparison where the GPU should show up.
axes[0].plot(n_pts_arr, results["t_G_build"], "^-",  label="essos G build (GPU)",        color="seagreen", linewidth=2)
axes[0].plot(n_pts_arr, results["t_A_build"], "v--", label="simsopt A build (C++, 3x cols)", color="darkred",  linewidth=2)
axes[0].set_xlabel("number of evaluation points")
axes[0].set_ylabel("wall time [s]")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_title(f"Interaction-matrix build — {n_magnets} magnets (nfp={nfp}, stellsym={stellsym})")
axes[0].legend(fontsize=9); axes[0].grid(True, which="both", alpha=0.3)

# Matvec time — memory-bound; GPU win is much smaller here.
axes[1].plot(n_pts_arr, np.array(results["t_G_matvec"])*1e3, "^-",  label="essos G matvec (GPU, f64)", color="seagreen", linewidth=2)
axes[1].plot(n_pts_arr, np.array(results["t_A_matvec"])*1e3, "v--", label="simsopt A matvec (CPU, f64, 3x cols)", color="darkred", linewidth=2)
axes[1].set_xlabel("number of evaluation points")
axes[1].set_ylabel("wall time [ms]")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_title("Matvec (memory-bound)")
axes[1].legend(fontsize=9); axes[1].grid(True, which="both", alpha=0.3)

# Accuracy.
axes[2].plot(n_pts_arr, results["maxdiff"], "d-", color="purple", linewidth=2)
axes[2].set_xlabel("number of evaluation points")
axes[2].set_ylabel("max |B_essos - B_simsopt| [T]")
axes[2].set_xscale("log"); axes[2].set_yscale("log")
axes[2].set_title("Accuracy vs resolution")
axes[2].grid(True, which="both", alpha=0.3)

plt.suptitle(f"Dipole field benchmark ({backend_name})", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dipole_benchmark_sweep.png", dpi=200, bbox_inches="tight")
print(f"\nSaved {OUTPUT_DIR}/dipole_benchmark_sweep.png")