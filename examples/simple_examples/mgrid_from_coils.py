"""Export coils to MGRID and compare the reloaded field with Biot-Savart."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.coils import Coils
from essos.fields import BiotSavart
from essos.mgrid import MGrid


coil_file = Path(__file__).resolve().parents[1] / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
mgrid_file = Path("mgrid_landreman_paul_qa.nc")
plot_file = Path("mgrid_field_comparison.png")
nr, nphi, nz = 64, 32, 64

print(f"Loading Landreman-Paul QA coils from {coil_file}...", flush=True)
coils = Coils.from_json(str(coil_file))
direct_field = BiotSavart(coils)

print(f"Sampling Biot-Savart on a {nr} x {nphi} x {nz} grid and writing {mgrid_file}...", flush=True)
coils.to_mgrid(mgrid_file, nr=nr, nphi=nphi, nz=nz,
               rmin=0.5, rmax=2.0, zmin=-0.8, zmax=0.8)

print(f"Loading {mgrid_file} as a magnetic field...", flush=True)
grid_field = MGrid.from_file(mgrid_file)

# MGrid and BiotSavart both implement MagneticField.B at Cartesian (x, y, z) points.
r = jnp.array([0.9, 1.0, 1.1, 1.2])
phi = jnp.array([0.2, 0.6, 1.0, 1.4])
z = jnp.array([0.0, 0.05, -0.1, 0.15])
points = jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)
print(f"Evaluating both fields at {len(points)} off-grid 3D points...", flush=True)
direct = jax.vmap(direct_field.B)(points)
imported = grid_field.B(points)
relative_error = jnp.linalg.norm(imported - direct, axis=-1) / jnp.linalg.norm(direct, axis=-1)

print("\nPoint       |B_direct| [T]   |B_MGRID| [T]   relative difference")
for index, (original, restored, error) in enumerate(zip(direct, imported, relative_error), start=1):
    print(f"{index:>3}         {float(jnp.linalg.norm(original)):>12.6g}   "
          f"{float(jnp.linalg.norm(restored)):>12.6g}   {float(error):>8.3%}")
print(f"Maximum relative difference: {float(jnp.max(relative_error)):.3%}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, len(points) + 1), 100 * relative_error)
ax.set(xlabel="Off-grid point", ylabel="Relative field difference [%]",
       title="Direct Biot-Savart vs imported MGRID")
ax.set_xticks(range(1, len(points) + 1))
fig.tight_layout()
fig.savefig(plot_file, dpi=150)
plt.close(fig)
print(f"Saved comparison plot to {plot_file}")
