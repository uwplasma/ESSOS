#!/usr/bin/env python3
"""
Apples-to-apples per-iteration wall-clock benchmark: SIMSOPT vs ESSOS.

One "full iteration" = everything the optimizer must do per step to get
BOTH the objective value and its gradient w.r.t. the magnet variables.

  ESSOS : single JIT'd jax.value_and_grad pass over the scalar-density
          formulation (G is (N_surf, N_dip), rho is (N_dip,))

  SIMSOPT: A @ m  (forward residual)  THEN  A.T @ r  (gradient)
          -- two matvecs, because simsopt's A is a plain array with no
          autodiff. A is (N_surf, 3*N_dip): the vector-moment
          formulation carries 3 columns per dipole, so it is ~3x wider
          than ESSOS's scalar-density matrix at the same resolution.

Matrix BUILD time is measured separately and reported, since it is a
one-time cost amortized over thousands of iterations.

Usage:
    python benchmark_iteration_sweep.py <surf_file> <mag_file>

Example:
    python benchmark_iteration_sweep.py \
        /Users/joshuabourassa/essos_new/essos/input.muse \
        /Users/joshuabourassa/simsopt/tests/test_files/zot80.focus
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

if len(sys.argv) != 3:
    sys.exit("Usage: python benchmark_iteration_sweep.py <surf_file> <mag_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE = Path(sys.argv[2])

RESOLUTIONS = [8, 16, 32, 64]   # nphi = ntheta = R
N_TRIALS = 5                    # repeated timed trials per method
N_WARMUP = 2                    # untimed warmup iterations (JIT, cache)
SURFACE_RANGE = "half period"
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_iteration_output"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from essos.fields import DipoleField

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_surface(surf_file, nphi, ntheta):
    from simsopt.geo import SurfaceRZFourier
    try:
        return SurfaceRZFourier.from_vmec_input(
            str(surf_file), range=SURFACE_RANGE, nphi=nphi, ntheta=ntheta)
    except Exception:
        return SurfaceRZFourier.from_focus(
            str(surf_file), range=SURFACE_RANGE, nphi=nphi, ntheta=ntheta)


def load_magnet_grid(mag_file):
    """Parse the FOCUS-format PM grid: positions, unit orientations, m0."""
    pos, mom = [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f:
            t = line.replace(",", " ").split()
            if len(t) < 12:
                continue
            try:
                x, y, z = float(t[3]), float(t[4]), float(t[5])
                m0 = float(t[7])
                az, pol = float(t[10]), float(t[11])
            except ValueError:
                continue
            pos.append((x, y, z))
            mom.append((m0 * np.cos(az) * np.sin(pol),
                        m0 * np.sin(az) * np.sin(pol),
                        m0 * np.cos(pol)))
    return np.asarray(pos, np.float64), np.asarray(mom, np.float64)


def time_trials(fn, n_warmup, n_trials):
    """Warm up, then return (median, min, max) wall-clock seconds."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    a = np.asarray(times)
    return float(np.median(a)), float(a.min()), float(a.max())


# ----------------------------------------------------------------------
print(f"--- Loading magnet grid: {MAG_FILE.name} ---")
positions, moments = load_magnet_grid(MAG_FILE)
n_dip = len(positions)
norms = np.linalg.norm(moments, axis=1)
norms_safe = np.where(norms > 0, norms, 1.0)
orientations = np.where(norms[:, None] > 0, moments / norms_safe[:, None], 0.0)
M0 = float(np.mean(norms[norms > 0])) if np.any(norms > 0) else 1.0
magnet_moments = orientations * M0
print(f"{n_dip} dipoles")

rows = []

for R in RESOLUTIONS:
    print(f"\n{'=' * 72}\n  Resolution {R}x{R}\n{'=' * 72}")
    surface = load_surface(SURF_FILE, R, R)
    nfp, stellsym = int(surface.nfp), bool(surface.stellsym)
    surf_pts = surface.gamma().reshape(-1, 3)
    surf_n = surface.unitnormal().reshape(-1, 3)
    n_surf = len(surf_pts)
    print(f"  {n_surf} surface points  (nfp={nfp} stellsym={stellsym})")

    row = {"resolution": R, "n_surf": n_surf, "n_dip": n_dip}

    # ---------------- ESSOS: build G, then one value_and_grad pass -------
    G = None
    try:
        t0 = time.perf_counter()
        df = DipoleField(
            jnp.asarray(positions, jnp.float32),
            jnp.asarray(magnet_moments, jnp.float32),
            jnp.zeros(n_dip, jnp.float32),
            nfp=nfp, stellsym=stellsym, scale_factor=1.0,
        )
        G = df.compute_interaction_matrix(
            jnp.asarray(surf_pts, jnp.float32),
            jnp.asarray(surf_n, jnp.float32))
        G = jnp.asarray(G)
        G.block_until_ready()
        row["essos_build_s"] = time.perf_counter() - t0
        row["essos_gb"] = G.nbytes / 1e9
        print(f"  [essos ] G {tuple(G.shape)}  {row['essos_gb']:.2f} GB  "
              f"build {row['essos_build_s']:.2f}s")

        bn_fix = jnp.zeros(n_surf, jnp.float32)
        aw = jnp.full(n_surf, 1.0 / n_surf, jnp.float32)
        rho = jnp.zeros(n_dip, jnp.float32)

        G_local = G

        def essos_loss(r):
            bn = r @ G_local.T + bn_fix
            return 0.5 * jnp.sum(aw * bn * bn)

        essos_vg = jax.jit(jax.value_and_grad(essos_loss))

        def essos_iter():
            v, g = essos_vg(rho)
            g.block_until_ready()

        med, lo, hi = time_trials(essos_iter, N_WARMUP, N_TRIALS)
        row["essos_iter_s"] = med
        row["essos_iter_min"] = lo
        row["essos_iter_max"] = hi
        print(f"  [essos ] iteration (value+grad): {med * 1e3:.2f} ms "
              f"(min {lo * 1e3:.2f}, max {hi * 1e3:.2f})")
    except Exception as e:
        print(f"  [essos ] FAILED: {type(e).__name__}: {e}")
        row["essos_iter_s"] = np.nan
    finally:
        G = None
        gc.collect()

    # ---------------- SIMSOPT: build A, then forward + transpose ---------
    A = None
    pm = None
    try:
        from simsopt.geo import PermanentMagnetGrid
        t0 = time.perf_counter()
        Bn_zero = np.zeros((R, R))
        pm = PermanentMagnetGrid.geo_setup_from_famus(
            surface, Bn_zero, str(MAG_FILE))
        A = np.asarray(pm.A_obj, dtype=np.float64)
        b = np.asarray(pm.b_obj, dtype=np.float64).ravel()
        row["simsopt_build_s"] = time.perf_counter() - t0
        row["simsopt_gb"] = A.nbytes / 1e9
        print(f"  [simsopt] A {A.shape}  {row['simsopt_gb']:.2f} GB  "
              f"build {row['simsopt_build_s']:.2f}s")

        m = np.zeros(A.shape[1], dtype=np.float64)

        def simsopt_iter():
            # Forward: residual. Then transpose: gradient. Two passes,
            # because there is no autodiff -- this is what one optimizer
            # step actually costs.
            r = A @ m - b
            _ = A.T @ r

        med, lo, hi = time_trials(simsopt_iter, N_WARMUP, N_TRIALS)
        row["simsopt_iter_s"] = med
        row["simsopt_iter_min"] = lo
        row["simsopt_iter_max"] = hi
        print(f"  [simsopt] iteration (fwd+transpose): {med * 1e3:.2f} ms "
              f"(min {lo * 1e3:.2f}, max {hi * 1e3:.2f})")
    except Exception as e:
        print(f"  [simsopt] FAILED: {type(e).__name__}: {e}")
        row["simsopt_iter_s"] = np.nan
    finally:
        A = None
        pm = None
        gc.collect()

    if not np.isnan(row.get("essos_iter_s", np.nan)) and \
       not np.isnan(row.get("simsopt_iter_s", np.nan)):
        row["speedup"] = row["simsopt_iter_s"] / row["essos_iter_s"]
        print(f"  --> essos is {row['speedup']:.1f}x faster per iteration")

    rows.append(row)


# ---------------- Report + plot ----------------
import pandas as pd
df_out = pd.DataFrame(rows)
print(f"\n{'=' * 72}\n  SUMMARY\n{'=' * 72}")
print(df_out.to_string(index=False))
df_out.to_csv(OUTPUT_DIR / "iteration_benchmark.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
res = df_out["resolution"]

if df_out["essos_iter_s"].notna().any():
    ax.errorbar(
        res, df_out["essos_iter_s"] * 1e3,
        yerr=[(df_out["essos_iter_s"] - df_out["essos_iter_min"]) * 1e3,
              (df_out["essos_iter_max"] - df_out["essos_iter_s"]) * 1e3],
        fmt="o-", capsize=4, color="steelblue",
        label="ESSOS (value+grad, JIT)")
if df_out["simsopt_iter_s"].notna().any():
    ax.errorbar(
        res, df_out["simsopt_iter_s"] * 1e3,
        yerr=[(df_out["simsopt_iter_s"] - df_out["simsopt_iter_min"]) * 1e3,
              (df_out["simsopt_iter_max"] - df_out["simsopt_iter_s"]) * 1e3],
        fmt="s-", capsize=4, color="indianred",
        label="SIMSOPT (fwd + transpose)")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(res)
ax.set_xticklabels([f"{r}x{r}" for r in res])
ax.set_xlabel("surface resolution")
ax.set_ylabel("wall-clock per iteration [ms]")
ax.set_title("Per-iteration cost: SIMSOPT vs ESSOS")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "iteration_benchmark.png", dpi=200, bbox_inches="tight")
print(f"\nSaved {OUTPUT_DIR}/iteration_benchmark.png and iteration_benchmark.csv")
