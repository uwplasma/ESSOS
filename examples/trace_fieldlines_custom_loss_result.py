#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if len(sys.argv) != 4:
    sys.exit("Usage: python trace_fieldlines_custom_loss_result.py <surf_file> <mag_file> <coil_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])
COIL_FILE = Path(sys.argv[3])

RESULTS_DIR = Path(__file__).resolve().parent / "pm_opt_custom_loss_output"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from essos.fields import DipoleField, BiotSavart
from essos.coils import Coils_from_simsopt
from essos.dynamics import Tracing
import matplotlib.pyplot as plt


class CombinedField:

    def __init__(self, coil_field, dipole_field):
        self.coil_field = coil_field
        self.dipole_field = dipole_field

    def B(self, points):
        return self.coil_field.B(points) + self.dipole_field.B(points)

    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points), axis=-1)

    def B_contravariant(self, points):
        return self.coil_field.B_contravariant(points) + self.dipole_field.B_contravariant(points)

    def to_xyz(self, points):
    
        return self.coil_field.to_xyz(points)


def load_surface(surf_file):
    from simsopt.geo import SurfaceRZFourier
    try:
        surface = SurfaceRZFourier.from_vmec_input(str(surf_file), range="full torus", nphi=64, ntheta=64)
    except Exception:
        surface = SurfaceRZFourier.from_focus(str(surf_file), range="full torus", nphi=64, ntheta=64)
    return surface


def load_coils_essos(coil_file):
    from simsopt.field import Coil, Current
    from simsopt.util.permanent_magnet_helper_functions import read_focus_coils

    base_curves, base_currents0, ncoils = read_focus_coils(str(coil_file))
    total_current = float(np.sum([c.get_value() for c in base_currents0]))
    all_coils = [Coil(base_curves[i], Current(total_current / ncoils)) for i in range(ncoils)]
    return Coils_from_simsopt(all_coils, nfp=1, stellsym=False)


def load_magnet_grid(mag_file):
    pos_list, mom_list = [], []
    with open(str(mag_file), encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.replace(",", " ").split()
            if len(tokens) < 12:
                continue
            try:
                x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
                m0      = float(tokens[7])
                az, pol = float(tokens[10]), float(tokens[11])
            except ValueError:
                continue
            pos_list.append((x, y, z))
            mom_list.append((m0*np.cos(az)*np.sin(pol), m0*np.sin(az)*np.sin(pol), m0*np.cos(pol)))
    return np.asarray(pos_list, np.float64), np.asarray(mom_list, np.float64)


print("--- Loading surface (for nfp/stellsym) ---")
surface = load_surface(SURF_FILE)
nfp, stellsym = int(surface.nfp), bool(surface.stellsym)
print(f"nfp={nfp}  stellsym={stellsym}")

print(f"--- Loading coils: {COIL_FILE.name} ---")
essos_coils = load_coils_essos(COIL_FILE)
coil_field = BiotSavart(essos_coils)

print(f"--- Loading magnet grid + optimized pho ---")
positions, moments_raw = load_magnet_grid(MAG_FILE)
n_magnets = len(positions)
native_norms = np.linalg.norm(moments_raw, axis=1)
norms_safe = np.where(native_norms > 0, native_norms, 1.0)
orientations = np.where(native_norms[:, None] > 0, moments_raw / norms_safe[:, None], 0.0)

pho_optimized = np.load(RESULTS_DIR / "pho_optimized.npy")
print(f"{n_magnets} magnet sites, {int(np.sum(np.abs(pho_optimized) > 0.5))} active")

scaled_moments = orientations * float(np.mean(native_norms[native_norms > 0])) * pho_optimized[:, None]
dipole_field = DipoleField(
    jnp.asarray(positions, jnp.float32),
    jnp.asarray(scaled_moments, jnp.float32),
    jnp.zeros(n_magnets, jnp.float32),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)

combined_field = CombinedField(coil_field, dipole_field)

print("Sanity check: evaluate combined field at a test point ")
test_pt = jnp.asarray(positions[0] * 1, jnp.float64)  
B_test = combined_field.B(test_pt)
absB_test = combined_field.AbsB(test_pt)
print(f"B at test point: {B_test}  shape={jnp.shape(B_test)}")
print(f"|B| at test point: {jnp.asarray(absB_test).ravel()}  shape={jnp.shape(absB_test)}")

print("\n Setting up field-line tracing ")
R0 = jnp.linspace(0, 10.0, 5) 
Z0 = jnp.zeros(len(R0))
phi0 = jnp.zeros(len(R0))
initial_xyz = jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T

tracing = Tracing(
    field=combined_field,
    initial_conditions=initial_xyz,
    model='FieldLine',
    maxtime=1e-6,
    timestep=1e-9,
)

print("Tracing field lines")
tracing.trace()
trajectories = tracing.trajectories
print(f"trajectories shape/type: {type(trajectories)}"
      f"{', shape=' + str(trajectories.shape) if hasattr(trajectories, 'shape') else ''}")

print("\n--- Poincare plot---")
fig_p, ax_p = plt.subplots(figsize=(8, 8))
tracing.poincare_plot(ax=ax_p, show=False)
ax_p.set_title("Poincare plot: TF coils + optimized PMs (custom_loss result)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "poincare_plot.png", dpi=200, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/poincare_plot.png")

print("\n--- 3D trajectory plot (essos built-in) ---")
fig_3d, ax_3d = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
tracing.plot(ax=ax_3d, show=False)
ax_3d.set_title("Field lines: TF coils + optimized PMs (custom_loss result)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fieldlines_3d.png", dpi=200, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/fieldlines_3d.png")
