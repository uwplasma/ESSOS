#!/usr/bin/env python3
"""
Time the three things the PM optimizer actually spends time on, so a
pre-merge / post-merge comparison measures the same quantities:

  1. G matrix build        (DipoleField.compute_interaction_matrix)
  2. optimizer inner step  (pho @ G.T  -> loss -> value_and_grad)
  3. direct field eval     (DipoleField.B)

Run it twice -- once with the current essos/fields.py, once with a
different version swapped in -- and compare. Keep the machine otherwise
idle, and run the cold version FIRST: a MacBook Air throttles noticeably
after a few minutes of sustained load, which will inflate whichever
version is measured second.

Usage:
    python time_pm_internals.py <surf_file> <mag_file>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

if len(sys.argv) != 3:
    sys.exit("Usage: python time_pm_internals.py <surf_file> <mag_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE = Path(sys.argv[2])

SURFACE_RANGE = "half period"
NPHI = NTHETA = 64
N_STEPS = 200          # timed optimizer steps
N_WARMUP = 3           # untimed, to absorb JIT

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from essos.fields import DipoleField

f32 = jnp.float32


def load_surface():
    from simsopt.geo import SurfaceRZFourier
    try:
        return SurfaceRZFourier.from_vmec_input(
            str(SURF_FILE), range=SURFACE_RANGE, nphi=NPHI, ntheta=NTHETA)
    except Exception:
        return SurfaceRZFourier.from_focus(
            str(SURF_FILE), range=SURFACE_RANGE, nphi=NPHI, ntheta=NTHETA)


def load_magnet_grid():
    pos, mom = [], []
    with open(str(MAG_FILE), encoding="utf-8") as fh:
        for line in fh:
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


surface = load_surface()
nfp, stellsym = int(surface.nfp), bool(surface.stellsym)
surf_pts = surface.gamma().reshape(-1, 3)
surf_n = surface.unitnormal().reshape(-1, 3)
n_surf = len(surf_pts)

positions, moments = load_magnet_grid()
n_dip = len(positions)
norms = np.linalg.norm(moments, axis=1)
safe = np.where(norms > 0, norms, 1.0)
orient = np.where(norms[:, None] > 0, moments / safe[:, None], 0.0)
M0 = float(np.mean(norms[norms > 0])) if np.any(norms > 0) else 1.0
magnet_moments = orient * M0

print(f"grid: {n_dip} dipoles, {n_surf} surface points "
      f"(nfp={nfp}, stellsym={stellsym})")

# ---- 1. G build -----------------------------------------------------
t0 = time.perf_counter()
field = DipoleField(
    jnp.asarray(positions, f32),
    jnp.asarray(magnet_moments, f32),
    jnp.zeros(n_dip, f32),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)
G = jnp.asarray(field.compute_interaction_matrix(
    jnp.asarray(surf_pts, f32), jnp.asarray(surf_n, f32)))
G.block_until_ready()
t_build = time.perf_counter() - t0
print(f"[1] G build            : {t_build:7.2f} s   "
      f"shape {tuple(G.shape)}  {G.nbytes/1e9:.2f} GB")

# ---- 2. optimizer inner step ----------------------------------------
bn_fix = jnp.zeros(n_surf, f32)
aw = jnp.full(n_surf, 1.0 / n_surf, f32)
pho = jnp.zeros(n_dip, f32)


def loss_fn(p):
    bn = p @ G.T + bn_fix
    fB = 0.5 * jnp.sum(aw * bn * bn)
    absp = jnp.sqrt(p * p + f32(1e-7))
    fD = jnp.sum(absp * (1.0 - absp))
    return fB + f32(0.1) * fD


vg = jax.jit(jax.value_and_grad(loss_fn))

for _ in range(N_WARMUP):
    v, g = vg(pho)
    g.block_until_ready()

t0 = time.perf_counter()
for _ in range(N_STEPS):
    v, g = vg(pho)
    g.block_until_ready()
t_steps = time.perf_counter() - t0
print(f"[2] optimizer step     : {t_steps/N_STEPS*1000:7.1f} ms/step "
      f"({N_STEPS} steps in {t_steps:.1f} s)")

# ---- 3. direct field eval -------------------------------------------
pts = jnp.asarray(surf_pts[:100], f32)
field.B(pts).block_until_ready()
t0 = time.perf_counter()
for _ in range(20):
    field.B(pts).block_until_ready()
t_B = (time.perf_counter() - t0) / 20
print(f"[3] DipoleField.B(100) : {t_B*1000:7.1f} ms")

print(f"\nextrapolated 14,000 steps: {t_steps/N_STEPS*14000/3600:.2f} hours "
      f"(excluding G build)")
