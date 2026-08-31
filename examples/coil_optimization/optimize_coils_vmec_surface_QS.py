import os
from time import time
import jax # gftd13@gmail.ocm
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B
from essos.losses import custom_loss
from essos.objective_functions import loss_QS, QS_check_on_surface


#  In this exmple, `scipy.optimize.least_squares` is used, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

# ====================================================================================
# ====================================================================================
""" Paths for the input files """
# ====================================================================================
# ====================================================================================

input_filepath = os.path.join(os.path.dirname(__file__), "..", "input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')

# ====================================================================================
# ====================================================================================
""" Creating starting coils and surface """
# ====================================================================================
# ====================================================================================

N_COILS = 3; FOURIER_ORDER = 3; LARGE_R = 10; SMALL_R = 5.6; NFP = 2; N_SEGMENTS = 45; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

# Initialize the coils and the field
init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)

# Initialize the surface from a VMEC output file.
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=30, nphi=30, range_torus='half period')

# ====================================================================================
# ====================================================================================
""" Setting the losses weights and targets """
# ====================================================================================
# ====================================================================================

LENGTH_WEIGHT = 1.; LENGTH_TARGET = 32.
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 0.1
NORMAL_FIELD_WEIGHT = 1.
QS_WEIGHT = 1.

# ====================================================================================
# ====================================================================================
""" Creating the loss functions """
# ====================================================================================
# ====================================================================================

def loss(field, surface):
    return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))

def loss_length(field):
    return jnp.mean(jnp.maximum(0, field.coils.length - LENGTH_TARGET))

def loss_curvature(field):
    return jnp.mean(jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET))

# ====================================================================================
# ====================================================================================
""" Defining custom losses """
# ====================================================================================
# ====================================================================================

L_normal_field = custom_loss(loss, "field", surface=surface)
L_length = custom_loss(loss_length, "field")
L_curvature = custom_loss(loss_curvature, "field")
L_QS = custom_loss(loss_QS, "field", surface=surface)

# ====================================================================================
# ====================================================================================
""" Defining total loss + setting dependencies """
# ====================================================================================
# ====================================================================================

L_total = NORMAL_FIELD_WEIGHT*L_normal_field + LENGTH_WEIGHT*L_length + CURVATURE_WEIGHT*L_curvature + QS_WEIGHT*L_QS
# L_total = NORMAL_FIELD_WEIGHT*L_normal_field + QS_WEIGHT*L_QS

L_total.dependencies = {"field": init_field}


QS_residual_xyz = QS_check_on_surface(init_field, surface)
print("mean abs residual (initial):", jnp.mean(jnp.abs(QS_residual_xyz)))
print("max abs residual (initial):", jnp.max(jnp.abs(QS_residual_xyz)))

print("Initial QS loss:", loss_QS(init_field, surface)) # gftd13@gmail.com test



# ====================================================================================
# ====================================================================================
""" Optimizing the total loss """
# ====================================================================================
# ====================================================================================

t_start = time()
res = least_squares(L_total, L_total.starting_dofs, L_total.grad, verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=200)
t_end = time()

# ====================================================================================
# ====================================================================================
""" Printing results """
# ====================================================================================
# ====================================================================================

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", L_total(L_total.starting_dofs))    
print("Loss after optimization:", L_total(res.x))

# ====================================================================================
# ====================================================================================
""" Extracting the different fields from the results array res """
# ====================================================================================
# ====================================================================================

# The optimized field is extracted from the results of the optimization. 
# The `dofs_to_pytree` method converts the flat array of degrees of freedom back into the structured format (pytree)
#  that includes the optimized coils and any other relevant parameters.
opt_field = L_total.dofs_to_pytree(res.x)["field"]
# opt_dict = L_total.dofs_to_pytree(res.x)
# opt_field = opt_dict["field"]

# Coils geometry is extracted from the optimized field.
opt_coils = opt_field.coils



# ====================================================================================
# ====================================================================================
""" Comparison of the initial QS loss and the optimized Qs loss """
# ====================================================================================
# ====================================================================================

print("Initial QS loss:", loss_QS(init_field, surface))
print("Optimized QS loss:", loss_QS(opt_field, surface))




# ====================================================================================
# ====================================================================================
""" Printing the initial and optimized coils """
# ====================================================================================
# ====================================================================================

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(121, projection='3d')
init_coils.plot(ax=ax1, show=False)
surface.plot(ax=ax1, show=False)
ax2 = fig.add_subplot(122, projection='3d')
opt_coils.plot(ax=ax2, show=False)
surface.plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()

""" Exporting results """

EXPORT = False
if EXPORT:
    output_filepath = os.path.join(os.path.dirname(__file__), "output")

    """ Save the coils to a json file """
    init_coils.to_json(os.path.join(output_filepath, "init_coils_vmec_surface.json"))
    opt_coils.to_json(os.path.join(output_filepath, "opt_coils_vmec_surface.json"))

    """ Save results in vtk format to analyze in Paraview """
    surface.to_vtk(os.path.join(output_filepath, "init_surface_vmec_surface.json"), field=init_field)
    surface.to_vtk(os.path.join(output_filepath, "final_surface_vmec_surface.json"), field=opt_field)
    init_coils.to_vtk(os.path.join(output_filepath, "init_coils_vmec_surface.json"))
    opt_coils.to_vtk(os.path.join(output_filepath, "opt_coils_vmec_surface.json"))