import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B
from essos.losses import custom_loss
from essos.objective_functions import (
    loss_coil_separation,
    loss_coil_surface_distance,
    loss_lorentz_force_coils,
    loss_linkingnumber,
)

#  In this exmple, `scipy.optimize.least_squares` is used, but any other optimizer, e.g. from
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

input_filepath = os.path.join(os.path.dirname(__file__), "..", "input_files")
vmec_input = os.path.join(input_filepath, "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")

""" Creating starting coils and surface """
N_COILS = 3; FOURIER_ORDER = 3; LARGE_R = 10; SMALL_R = 5.6; NFP = 2; N_SEGMENTS = 45; STELLSYM = True
COIL_CURRENT = 1.

init_curves = CreateEquallySpacedCurves(
    N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM
)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT] * N_COILS)
init_field = BiotSavart(init_coils)
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=30, nphi=30, range_torus="half period")

""" Setting the losses weights and targets """
NORMAL_FIELD_WEIGHT = 1.
SURFACE_DISTANCE_WEIGHT = 1.
COIL_DISTANCE_WEIGHT = 1.
FORCE_WEIGHT = 1.
LINKING_WEIGHT = 1.

MIN_SURFACE_DISTANCE = 0.2
MIN_COIL_DISTANCE = 0.2
FORCE_THRESHOLD = 0.5e6
FORCE_POWER = 1
BLOCK_SIZE = None

""" Creating the loss functions """
def loss(field, surface):
    return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))


def loss_surface_distance(field):
    return loss_coil_surface_distance(
        field.coils, surface, min_distance=MIN_SURFACE_DISTANCE, block_size=BLOCK_SIZE
    )


def loss_coil_distance(field):
    return loss_coil_separation(
        field.coils, min_separation=MIN_COIL_DISTANCE, block_size=BLOCK_SIZE
    )


def loss_force(field):
    return loss_lorentz_force_coils(
        field.coils, p=FORCE_POWER, threshold=FORCE_THRESHOLD, block_size=BLOCK_SIZE
    )


def loss_linking(field):
    return loss_linkingnumber(field.coils, block_size=BLOCK_SIZE)


def print_loss_costs(label, field):
    print(f"\n{label}")
    print("normal_field:", loss(field, surface))
    print("coil_surface_distance:", loss_surface_distance(field))
    print("coil_coil_distance:", loss_coil_distance(field))
    print("lorentz_force:", loss_force(field))
    print("linking_number:", loss_linking(field))
    print(
        "total_weighted:",
        NORMAL_FIELD_WEIGHT * loss(field, surface)
        + SURFACE_DISTANCE_WEIGHT * loss_surface_distance(field)
        + COIL_DISTANCE_WEIGHT * loss_coil_distance(field)
        + FORCE_WEIGHT * loss_force(field)
        + LINKING_WEIGHT * loss_linking(field),
    )


""" Defining custom losses """
L_normal_field = custom_loss(loss, "field", surface=surface)
L_surface_distance = custom_loss(loss_surface_distance, "field")
L_coil_distance = custom_loss(loss_coil_distance, "field")
L_force = custom_loss(loss_force, "field")
L_linking = custom_loss(loss_linking, "field")

""" Defining total loss + setting dependencies """
L_total = (
    NORMAL_FIELD_WEIGHT * L_normal_field
    + SURFACE_DISTANCE_WEIGHT * L_surface_distance
    + COIL_DISTANCE_WEIGHT * L_coil_distance
    + FORCE_WEIGHT * L_force
    + LINKING_WEIGHT * L_linking
)
L_total.dependencies = {"field": init_field}

""" Optimizing the total loss """
t_start = time()
res = least_squares(
    L_total,
    L_total.starting_dofs,
    L_total.grad,
    verbose=2,
    ftol=1e-5,
    gtol=1e-5,
    xtol=1e-14,
    max_nfev=200,
)
t_end = time()

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", L_total(L_total.starting_dofs))
print("Loss after optimization:", L_total(res.x))

opt_field = L_total.dofs_to_pytree(res.x)["field"]
opt_coils = opt_field.coils

print_loss_costs("Loss costs before optimization", init_field)
print_loss_costs("Loss costs after optimization", opt_field)

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

    """ Save the coils to a json file """
    init_coils.to_json(os.path.join(output_filepath, "init_coils_vmec_surface_distance_forces.json"))
    opt_coils.to_json(os.path.join(output_filepath, "opt_coils_vmec_surface_distance_forces.json"))

    """ Save results in vtk format to analyze in Paraview """
    surface.to_vtk(os.path.join(output_filepath, "init_surface_vmec_surface_distance_forces.json"), field=init_field)
    surface.to_vtk(os.path.join(output_filepath, "final_surface_vmec_surface_distance_forces.json"), field=opt_field)
    init_coils.to_vtk(os.path.join(output_filepath, "init_coils_vmec_surface_distance_forces.json"))
    opt_coils.to_vtk(os.path.join(output_filepath, "opt_coils_vmec_surface_distance_forces.json"))
