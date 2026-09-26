import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, SurfaceClassifier
from essos.dynamics import connection_length, trace_field_lines

# Connection length in the scrape-off layer of the Landreman-Paul QA coils.
# The wall is the plasma boundary offset outwards by a uniform clearance;
# seeds span the outboard midplane gap between boundary and wall at phi=0.
clearance, max_length, nseeds = 0.08, 300.0, 40
input_dir = os.path.join(os.path.dirname(__file__), '..', 'input_files')
field = BiotSavart(Coils.from_json(os.path.join(input_dir, 'ESSOS_biot_savart_LandremanPaulQA.json')))
wout = SurfaceRZFourier.from_wout_file(os.path.join(input_dir, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'))
scale = wout.rc[0]  # reactor-scale boundary rescaled to the 1 m coils
boundary = SurfaceRZFourier(wout.rc / scale, wout.zs / scale, wout.nfp, wout.mpol, wout.ntor, ntheta=64, nphi=64)
classifier = SurfaceClassifier(boundary, h=0.03, padding=2 * clearance)


def wall(xyz):  # positive inside the wall, zero on it
    return clearance + classifier.evaluate_xyz(xyz)


R_lcfs = jnp.sum(boundary.rc)  # outboard midplane at phi=0
R0 = jnp.linspace(R_lcfs, R_lcfs + clearance, nseeds + 2)[1:-1]
seeds = jnp.stack([R0, jnp.zeros_like(R0), jnp.zeros_like(R0)], axis=1)
time0 = time()
result = connection_length(field, seeds, wall, max_length=max_length)
Lc = result["connection_length"].block_until_ready()
print(f"Connection lengths of {nseeds} lines took {time() - time0:.1f} s; "
      f"{int(result['hit'].all(axis=1).sum())}/{nseeds} hit the wall both ways")

# Poincare section of the closed vacuum surfaces for context
poincare = trace_field_lines(field, jnp.stack([jnp.linspace(1.22, 1.42, 10), jnp.zeros(10), jnp.zeros(10)], 1),
                             length=1000.0, samples=100000, progress=False, label=None)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
plt.sca(ax1)
poincare.poincare_plot(ax=ax1, show=False, shifts=[0.0], color="0.65", s=0.2)
R, Z = jnp.meshgrid(jnp.linspace(0.9, 1.6, 200), jnp.linspace(-0.7, 0.7, 200))
ax1.contour(R, Z, jnp.vectorize(lambda r, z: wall(jnp.array([r, 0.0, z])))(R, Z), [0.0], colors="k", linewidths=2)
ax1.plot(jnp.hypot(boundary.gamma[0, :, 0], boundary.gamma[0, :, 1]), boundary.gamma[0, :, 2], "C0--", lw=1.5)
style = dict(c=jnp.log10(Lc), cmap="viridis", vmin=0, vmax=jnp.log10(2 * max_length), zorder=3)
ax1.scatter(R0, 0 * R0, s=10, **style)
ax1.set_xlabel("R [m]", fontsize=10)
ax1.set_ylabel("Z [m]", fontsize=10)
ax1.set(xlim=(0.85, 1.55), ylim=(-0.6, 0.6), aspect="equal", adjustable="box", title=r"$\phi=0$: boundary (dashed), wall (solid)")
ax1.grid(False)
ax2.semilogy(100 * (R0 - R_lcfs), Lc, "-", color="0.7", zorder=1)
points = ax2.scatter(100 * (R0 - R_lcfs), Lc, s=25, **style)
fig.colorbar(points, ax=ax2, label=r"$\log_{10}(L_c/\mathrm{m})$")
ax2.axhline(2 * max_length, color="0.5", ls=":", label="tracing cap")
ax2.set(xlabel="distance outside plasma boundary [cm]", ylabel=r"$L_c$ [m]", title="Connection length, outboard midplane")
ax2.legend(frameon=False, loc="center right")
ax2.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "connection_length_coils.png"), dpi=150)
plt.show()
