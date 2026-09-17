import os
from time import time
# import jax # gftd13@gmail.ocm
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B
from essos.losses import custom_loss
from essos.objective_functions import loss_BdotN_mean, loss_coil_curvature_from_field, loss_coil_length_max, loss_quasi_symmetry, quasi_symmetry_residual_on_surface


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
ntheta = 30; nphi = 30; Npoints = ntheta * nphi  # Surface parametersrs
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=30, nphi=30, range_torus='half period')


# Build and cache surface geometry before JAX starts the optimization.
surface.gamma.block_until_ready()
surface.unitnormal.block_until_ready()

# ====================================================================================
# ====================================================================================
""" Setting the losses weights and targets """
# ====================================================================================
# ====================================================================================

LENGTH_WEIGHT = 1.; LENGTH_TARGET = 32.
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 0.1
NORMAL_FIELD_WEIGHT = Npoints
QS_WEIGHT = 1.

# ====================================================================================
# ====================================================================================
""" Creating the loss functions """
# ====================================================================================
# ====================================================================================

# def loss_curvature(field):
#     return jnp.mean(jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET))

# ====================================================================================
# ====================================================================================
""" Defining custom losses """
# ====================================================================================
# ====================================================================================

L_normal_field = custom_loss( loss_BdotN_mean, "field", surface=surface)
L_length_max = custom_loss( loss_coil_length_max , "field", max_coil_length=LENGTH_TARGET )
L_curvature = custom_loss( loss_coil_curvature_from_field , "field" , max_coil_curvature=CURVATURE_TARGET )
L_QS = custom_loss(loss_quasi_symmetry, "field", surface=surface)

# ====================================================================================
# ====================================================================================
""" Defining total loss + setting dependencies """
# ====================================================================================
# ====================================================================================

L_total = NORMAL_FIELD_WEIGHT*L_normal_field + LENGTH_WEIGHT*L_length_max + CURVATURE_WEIGHT*L_curvature + QS_WEIGHT*L_QS
# L_total = NORMAL_FIELD_WEIGHT*L_normal_field + QS_WEIGHT*L_QS

L_total.dependencies = {"field": init_field}

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
""" Printing a comparison of the initial loss and the optimized loss """
# ====================================================================================
# ====================================================================================

print("\nNormal-field residuals and losses:")
B_dot_n_over_B_init = BdotN_over_B( surface , init_field )
B_dot_n_over_B_opt = BdotN_over_B( surface , opt_field )
print("mean abs residual (initial):", jnp.mean(jnp.abs(B_dot_n_over_B_init)) )
print("mean abs residual (optimized):",jnp.mean(jnp.abs(B_dot_n_over_B_opt)) )
print("max abs residual (initial):",jnp.max(jnp.abs(B_dot_n_over_B_init)) )
print("max abs residual (optimized):",jnp.max(jnp.abs(B_dot_n_over_B_opt)) )
print("Normal-field loss (initial):",loss_BdotN_mean(init_field, surface) )
print("Normal-field loss (optimized):",loss_BdotN_mean(opt_field, surface) )

print("\nCoil-length residuals and losses:")
coil_length_residual_init = jnp.maximum(0, init_field.coils.length - LENGTH_TARGET)
coil_length_residual_opt = jnp.maximum(0, opt_field.coils.length - LENGTH_TARGET)
print("Coil lengths (initial):", init_field.coils.length)
print("Coil lengths (optimized):", opt_field.coils.length)
print("mean excess length (initial):", jnp.mean(coil_length_residual_init))
print("mean excess length (optimized):", jnp.mean(coil_length_residual_opt))
print("max excess length (initial):", jnp.max(coil_length_residual_init))
print("max excess length (optimized):", jnp.max(coil_length_residual_opt))
print("Length loss (initial):", loss_coil_length_max(init_field, max_coil_length=LENGTH_TARGET))
print("Length loss (optimized):", loss_coil_length_max(opt_field, max_coil_length=LENGTH_TARGET))

print("\nCoil-curvature residuals and losses:")
coil_curvature_residual_init = jnp.maximum(0, init_field.coils.curvature - CURVATURE_TARGET)
coil_curvature_residual_opt = jnp.maximum(0, opt_field.coils.curvature - CURVATURE_TARGET)
print("mean curvature (initial):", jnp.mean(init_field.coils.curvature))
print("mean curvature (optimized):", jnp.mean(opt_field.coils.curvature))
print("max curvature (initial):", jnp.max(init_field.coils.curvature))
print("max curvature (optimized):", jnp.max(opt_field.coils.curvature))
print("mean excess curvature (initial):", jnp.mean(coil_curvature_residual_init))
print("mean excess curvature (optimized):", jnp.mean(coil_curvature_residual_opt))
print("max excess curvature (initial):", jnp.max(coil_curvature_residual_init))
print("max excess curvature (optimized):", jnp.max(coil_curvature_residual_opt))
print("Curvature loss (initial):", loss_coil_curvature_from_field(init_field, max_coil_curvature=CURVATURE_TARGET))
print("Curvature loss (optimized):", loss_coil_curvature_from_field(opt_field, max_coil_curvature=CURVATURE_TARGET))

print("\nQS residuals and losses:")
QS_residual_xyz_init = quasi_symmetry_residual_on_surface(init_field, surface)
QS_residual_xyz_opt = quasi_symmetry_residual_on_surface(opt_field, surface)
print("mean abs residual (initial):", jnp.mean(jnp.abs(QS_residual_xyz_init)))
print("mean abs residual (optimized):", jnp.mean(jnp.abs(QS_residual_xyz_opt)))
print("max abs residual (initial):", jnp.max(jnp.abs(QS_residual_xyz_init)))
print("max abs residual (optimized):", jnp.max(jnp.abs(QS_residual_xyz_opt)))
print("QS loss (initial):", loss_quasi_symmetry(init_field, surface))
print("QS loss (optimized):", loss_quasi_symmetry(opt_field, surface))


# ====================================================================================
# ====================================================================================
""" Plotting the initial and optimized coils """
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