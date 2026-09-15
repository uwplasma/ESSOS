#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if len(sys.argv) != 4:
    sys.exit("Usage: python trace_particle_custom_loss_result.py <surf_file> <mag_file> <coil_file>")

SURF_FILE = Path(sys.argv[1])
MAG_FILE  = Path(sys.argv[2])
COIL_FILE = Path(sys.argv[3])

RESULTS_DIR = Path(__file__).resolve().parent / "pm_opt_custom_loss_output"
PLOT_SCRIPT = Path(__file__).resolve().parent / "plot_particle_poincare.py"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from essos.fields import DipoleField, BiotSavart
from essos.coils import Coils_from_simsopt
from essos.dynamics import Tracing, Particles
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, ONE_EV, ELECTRON_MASS, ELEMENTARY_CHARGE


class CombinedField:
    def __init__(self, *fields):
        if len(fields) < 1:
            raise ValueError("CombinedField needs at least one field")
        self.fields = fields

    def B(self, points):
        return sum(f.B(points) for f in self.fields)

    def B_covariant(self, points):
        return sum(f.B_covariant(points) for f in self.fields)

    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points), axis=-1)

    def B_contravariant(self, points):
        return sum(f.B_contravariant(points) for f in self.fields)

    def sqrtg(self, points):
   
        return 1.

    def dB_by_dX(self, points):
        return jax.jacfwd(self.B)(points)

    def dAbsB_by_dX(self, points):
        return jax.grad(self.AbsB)(points)

    def grad_B_covariant(self, points):
        return jax.jacfwd(self.B_covariant)(points)

    def curl_B(self, points):
        grad_B_cov = self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] - grad_B_cov[1][2],
                          grad_B_cov[0][2] - grad_B_cov[2][0],
                          grad_B_cov[1][0] - grad_B_cov[0][1]]) / self.sqrtg(points)

    def curl_b(self, points):
        return (self.curl_B(points) / self.AbsB(points)
                + jnp.cross(self.B_covariant(points), jnp.array(self.dAbsB_by_dX(points)))
                / self.AbsB(points)**2 / self.sqrtg(points))

    def kappa(self, points):

        return -jnp.cross(self.B_contravariant(points), self.curl_b(points)) * self.sqrtg(points) / self.AbsB(points)

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
scaled_moments = orientations * float(np.mean(native_norms[native_norms > 0])) * pho_optimized[:, None]

active_filter = np.abs(pho_optimized) > 0.5
positions_active_only = positions[active_filter]
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


R0_VALS = jnp.linspace(0.33, 0.348, 5)  # confirmed-safe window, no divergence
R0_VAL = R0_VALS[0]  
ENERGY_EV = 1000

test_pt = jnp.array([R0_VAL, 0.0, 0.0])
B_vec = combined_field.B(test_pt)
absB = float(jnp.linalg.norm(B_vec))
energy_J = ENERGY_EV * ONE_EV
v_perp_worst_case = np.sqrt(2 * energy_J / ELECTRON_MASS)
r_larmor_cm = (ELECTRON_MASS * v_perp_worst_case / (ELEMENTARY_CHARGE * absB)) * 100

print(f"\n--- Larmor radius check at R0={R0_VAL} ---")
print(f"|B| = {absB:.4f} T")
print(f"Energy = {ENERGY_EV} eV")
print(f"Worst-case r_Larmor (all energy perpendicular) = {r_larmor_cm:.3f} cm")
print(f"MUSE magnet shell spans R~0.18-0.43 m (~25 cm minor-radius scale)")
if r_larmor_cm > 5:
    print("WARNING: Consider lowering energy.")
else:
    print("Larmor radius is small")


initial_xyz = jnp.array([[r0, 0.0, 0.0] for r0 in R0_VALS])  

particles = Particles(
    initial_xyz=initial_xyz,
    initial_vparallel_over_v=jnp.array([0.3]*len(R0_VALS)), 
    charge=ELEMENTARY_CHARGE,
    mass=ELECTRON_MASS,
    energy=energy_J,
)

MAXTIME = 1e-5

print(f"\n--- Setting up Guiding Center trace ({len(R0_VALS)} particles) ---")
tracing = Tracing(
    field=combined_field,
    particles=particles,
    model='GuidingCenter',  
    maxtime=MAXTIME,
    timestep=5e-9,
    times_to_trace=20000,
    rtol=1e-7,
    atol=1e-7,
)

print("Tracing particle...")
trajectories = tracing.trajectories
print(f"trajectories shape: {trajectories.shape}")

np.save(RESULTS_DIR / "particle_trajectories.npy", np.asarray(trajectories))
np.save(RESULTS_DIR / "particle_times.npy", np.asarray(tracing.times))
print(f"Saved raw particle trajectory to {RESULTS_DIR}/particle_trajectories.npy")


import subprocess
print("\n--- Poincare plot (particle, Guiding Center) ---")
r0_vals_str = ", ".join(f"{v:.4f}" for v in np.asarray(R0_VALS))
result = subprocess.run(
    [sys.executable, str(PLOT_SCRIPT), str(RESULTS_DIR), str(MAXTIME), str(ENERGY_EV), r0_vals_str],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(f"Plot subprocess FAILED (exit code {result.returncode}):")
    print(result.stderr)
else:
    print("Plot subprocess completed successfully.")
