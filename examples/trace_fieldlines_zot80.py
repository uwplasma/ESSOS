#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if len(sys.argv) != 4:
    sys.exit("Usage: python trace_fieldlines_zot80.py <surf_file> <mag_file> <coil_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])
COIL_FILE = Path(sys.argv[3])

RESULTS_DIR = Path(__file__).resolve().parent / "trace_zot80_output"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from essos.fields import DipoleField, BiotSavart
from essos.coils import Coils_from_simsopt
from essos.dynamics import Tracing
import matplotlib.pyplot as plt


class CombinedField:
    """Sums any number of Cartesian-native essos field objects into one
    field, implementing the interface essos.dynamics.Tracing expects."""
    def __init__(self, *fields):
        if len(fields) < 1:
            raise ValueError("CombinedField needs at least one field")
        self.fields = fields

    def B(self, points):
        return sum(f.B(points) for f in self.fields)

    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points), axis=-1)

    def B_contravariant(self, points):
        return sum(f.B_contravariant(points) for f in self.fields)

    def to_xyz(self, points):
        for f in self.fields:
            try:
                return f.to_xyz(points)
            except (NotImplementedError, AttributeError):
                continue
        raise NotImplementedError("no field implements to_xyz")


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


def load_magnet_grid_native_pho(mag_file):

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
        np.asarray(pos_list, np.float64),
        np.asarray(mom_list, np.float64),
        np.asarray(pho_list, np.float64),
    )


def cyl_to_xyz(r, phi, z):
    return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), z]).T


print("--- Loading surface (for nfp/stellsym) ---")
surface = load_surface(SURF_FILE)
nfp, stellsym = int(surface.nfp), bool(surface.stellsym)
print(f"nfp={nfp}  stellsym={stellsym}")

print(f"--- Loading coils: {COIL_FILE.name} ---")
essos_coils = load_coils_essos(COIL_FILE)
coil_field = BiotSavart(essos_coils)

print(f"--- Loading magnet grid (zot80 NATIVE pho): {MAG_FILE.name} ---")
positions, moments_raw, pho_native = load_magnet_grid_native_pho(MAG_FILE)
n_magnets = len(positions)
native_norms = np.linalg.norm(moments_raw, axis=1)
norms_safe = np.where(native_norms > 0, native_norms, 1.0)
orientations = np.where(native_norms[:, None] > 0, moments_raw / norms_safe[:, None], 0.0)

n_active = int(np.sum(np.abs(pho_native) > 0.5))
print(f"{n_magnets} magnet sites, {n_active} active (zot80's own FAMUS solution)")


active_filter = np.abs(pho_native) > 0.5
positions_active_only = positions[active_filter]
scaled_moments = orientations * float(np.mean(native_norms[native_norms > 0])) * pho_native[:, None]
scaled_moments_active_only = scaled_moments[active_filter]
n_magnets_active = int(active_filter.sum())
print(f"Filtered DipoleField to {n_magnets_active} active magnets for tracing")

dipole_field = DipoleField(
    jnp.asarray(positions_active_only, jnp.float32),
    jnp.asarray(scaled_moments_active_only, jnp.float32),
    jnp.zeros(n_magnets_active, jnp.float32),
    nfp=nfp, stellsym=stellsym, scale_factor=1.0,
)

combined_field = CombinedField(coil_field, dipole_field)

print("--- Sanity check: evaluate combined field at a test point ---")
test_pt = jnp.array([0.30, 0.0, 0.0], jnp.float64)
B_test = combined_field.B(test_pt)
print(f"B at (0.30,0,0): {B_test}")

print("\n--- Setting up field-line tracing (SAME parameters as optimized-result trace) ---")

R0   = jnp.linspace(0.34, 0.375, 3) 
phi0 = jnp.zeros(len(R0))
Z0   = jnp.zeros(len(R0))
initial_xyz = cyl_to_xyz(R0, phi0, Z0)
print(f"Starting positions (r,phi,z): {[(float(r),float(p),float(z)) for r,p,z in zip(R0,phi0,Z0)]}")

tracing = Tracing(
    field=combined_field,
    initial_conditions=initial_xyz,
    model='FieldLine',
    maxtime=1290,
    timestep=0.258,
    times_to_trace=30000,  
    rtol=1e-10,
    atol=1e-10,
)

print("Tracing field lines (zot80 native solution)...")
tracing.trace()
trajectories = tracing.trajectories
print(f"trajectories shape: {trajectories.shape}")

np.save(RESULTS_DIR / "trajectories.npy", np.asarray(trajectories))
np.save(RESULTS_DIR / "trace_times.npy", np.asarray(tracing.times))
np.save(RESULTS_DIR / "trace_R0.npy", np.asarray(R0))
print(f"Saved raw trajectories to {RESULTS_DIR}/trajectories.npy")

print("\n--- Poincare plot (zot80 native) ---")
fig_p, ax_p = plt.subplots(figsize=(8, 8))
tracing.poincare_plot(ax=ax_p, show=False)
ax_p.set_xlabel(r"$R$ [m]")
ax_p.set_ylabel(r"$Z$ [m]")
ax_p.set_title("Poincare plot: TF coils + zot80 NATIVE (FAMUS) solution")
ax_p.set_xlim(0.20, 0.45)
ax_p.set_ylim(-0.10, 0.10)
ax_p.set_aspect("equal")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "poincare_plot_zot80.png", dpi=200, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/poincare_plot_zot80.png")

print("\n--- 3D trajectory plot (zot80 native) ---")
fig_3d, ax_3d = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
tracing.plot(ax=ax_3d, show=False)
essos_coils.plot(ax=ax_3d, show=False, color="brown", linewidth=1.5, alpha=0.4, label="TF coils")
active_mask = np.abs(pho_native) > 0.5
pos_active = positions[active_mask]
pho_active = pho_native[active_mask]
sc = ax_3d.scatter(
    pos_active[:, 0], pos_active[:, 1], pos_active[:, 2],
    c=pho_active, cmap="RdBu_r", s=6, vmin=-1, vmax=1,
    label=f"PM magnets (zot80 native, {int(active_mask.sum())} active)",
)
fig_3d.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.1, label=r"$\rho$")
ax_3d.set_xlim(-0.45, 0.45)
ax_3d.set_ylim(-0.45, 0.45)
ax_3d.set_zlim(-0.2, 0.2)
ax_3d.set_xlabel(r"$x$ [m]")
ax_3d.set_ylabel(r"$y$ [m]")
ax_3d.set_zlabel(r"$z$ [m]")
ax_3d.set_title("Field lines + TF coils + zot80 NATIVE (FAMUS) solution")
ax_3d.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fieldlines_3d_zot80.png", dpi=200, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/fieldlines_3d_zot80.png")

print(f"\nAll outputs saved to {RESULTS_DIR}/")
