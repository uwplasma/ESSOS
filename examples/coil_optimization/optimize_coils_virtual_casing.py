#  Coils for a finite-beta VMEC/VMEX equilibrium, using virtual casing for the field the coils must produce.
#  The plasma carries current, so the coils must reproduce B_total - B_plasma on the boundary, not just
#  B.n = 0. virtual_casing_jax gives that field (all three components) from the wout boundary and total field.
#  Matching the full vector fixes the coil currents and the pressure balance a free-boundary solve needs.
#  Requires the optional packages vmex (wout reader) and virtual-casing-jax >= 0.0.10.
import os
from time import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from virtual_casing_jax import VirtualCasingJAX
from vmex.core.wout import read_wout
from vmex.core.virtual_casing import surface_field_data_from_wout

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier
from essos.losses import custom_loss
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance

input_filepath = os.path.join(os.path.dirname(__file__), "..", "input_files")
vmec_file = os.path.join(input_filepath, "wout_QH_simple_finite_scaled.nc")  # QH, beta = 5%, B = 5.7 T

""" Virtual casing parameters: a 32 x 32 grid per field period at 4 digits is converged to ~2e-6 |B|
    (digits 3: 2e-4; 16 x 16: 4e-4) and takes ~20 s and ~6 GB; memory grows steeply with the grid """
NPHI, NTHETA, VC_DIGITS = 32, 32, 4

""" Coil parameters """
N_COILS = 4; FOURIER_ORDER = 4; LARGE_R = 14.4; SMALL_R = 4.5; N_SEGMENTS = 60; STELLSYM = True
LENGTH_TARGET = 30.; CURVATURE_TARGET = 1.0; MIN_COIL_DISTANCE = 0.8; MIN_SURFACE_DISTANCE = 1.5  # m, 1/m, m, m
FIELD_WEIGHT = 1e3; LENGTH_WEIGHT = 1e1; CURVATURE_WEIGHT = 1e2; DISTANCE_WEIGHT = 1e2

""" Field the coils must produce on the boundary: B_external = B_total - B_plasma """
wout = read_wout(vmec_file)
NFP = int(wout.nfp)
t_start = time()
data = surface_field_data_from_wout(wout, nphi=NPHI, ntheta=NTHETA)  # one field period, (3, nphi, ntheta)
vc = VirtualCasingJAX()
vc.setup(VC_DIGITS, NFP, False, NPHI, NTHETA, data.gamma, NPHI, NTHETA, NPHI, NTHETA)
B_external = vc.compute_external_B(data.B_total.reshape(3, -1), digits=VC_DIGITS)
half = slice(0, NPHI // 2)  # stellarator symmetry: half a field period suffices
points = jnp.moveaxis(data.gamma, 0, -1)[half].reshape(-1, 3)
B_target = jnp.moveaxis(B_external.reshape(3, NPHI, NTHETA), 0, -1)[half].reshape(-1, 3)
normals = jnp.moveaxis(data.normal, 0, -1)[half].reshape(-1, 3)
B_norm = jnp.linalg.norm(jnp.moveaxis(data.B_total, 0, -1)[half].reshape(-1, 3), axis=1)
print(f"Virtual casing took {time() - t_start:.1f} s; plasma-current share of the boundary field: "
      f"{jnp.mean(jnp.linalg.norm(jnp.moveaxis(data.B_total, 0, -1)[half].reshape(-1, 3) - B_target, axis=1) / B_norm):.3f}")

""" Creating starting coils and surface (initial current fitted to the target) """
surface = SurfaceRZFourier.from_wout_file(vmec_file, s=1, ntheta=NTHETA, nphi=NPHI, range_torus="half period")
init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
B_unit = jax.vmap(BiotSavart(Coils(init_curves, [1.] * N_COILS)).B)(points)
COIL_CURRENT = float(jnp.vdot(B_unit, B_target) / jnp.vdot(B_unit, B_unit))
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT] * N_COILS)
init_field = BiotSavart(init_coils)

""" Creating the loss functions """
def loss_field(field):
    return jnp.mean(jnp.sum((jax.vmap(field.B)(points) - B_target)**2, axis=1) / B_norm**2)

def loss_length(field):
    return jnp.mean(jnp.maximum(0, field.coils.length - LENGTH_TARGET)**2)

def loss_curvature(field):
    return jnp.mean(jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET)**2)

def loss_distance(field):
    return (loss_coil_separation(field.coils, MIN_COIL_DISTANCE)
            + loss_coil_surface_distance(field.coils, surface, MIN_SURFACE_DISTANCE))

def report(label, field):
    dB = jax.vmap(field.B)(points) - B_target
    print(f"{label}: boundary |dB.n|/B mean {jnp.mean(jnp.abs(jnp.sum(dB * normals, 1)) / B_norm):.2e}, "
          f"|dB|/B mean {jnp.mean(jnp.linalg.norm(dB, axis=1) / B_norm):.2e} max {jnp.max(jnp.linalg.norm(dB, axis=1) / B_norm):.2e}, "
          f"lengths {jnp.round(field.coils.length[:N_COILS], 2)} m, max curvature {jnp.max(field.coils.curvature):.2f} 1/m")

""" Defining total loss + setting dependencies """
L_total = (FIELD_WEIGHT * custom_loss(loss_field, "field") + LENGTH_WEIGHT * custom_loss(loss_length, "field")
           + CURVATURE_WEIGHT * custom_loss(loss_curvature, "field") + DISTANCE_WEIGHT * custom_loss(loss_distance, "field"))
L_total.dependencies = {"field": init_field}

""" Optimizing the total loss """
report("Initial coils", init_field)
t_start = time()
res = least_squares(L_total, L_total.starting_dofs, L_total.grad, verbose=2, ftol=1e-10, gtol=1e-10, xtol=1e-14, max_nfev=500)
print(f"\nOptimization took {time() - t_start:.2f} seconds")
opt_field = L_total.dofs_to_pytree(res.x)["field"]
opt_coils = opt_field.coils
report("Optimized coils", opt_field)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection="3d")
init_coils.plot(ax=ax1, show=False)
surface.plot(ax=ax1, show=False)
ax2 = fig.add_subplot(122, projection="3d")
opt_coils.plot(ax=ax2, show=False)
surface.plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()

EXPORT = False
if EXPORT:
    output_filepath = os.path.join(os.path.dirname(__file__), "output")
    opt_coils.to_json(os.path.join(output_filepath, "opt_coils_virtual_casing.json"))
    opt_coils.to_vtk(os.path.join(output_filepath, "opt_coils_virtual_casing"))
    surface.to_vtk(os.path.join(output_filepath, "surface_virtual_casing"), field=opt_field)
