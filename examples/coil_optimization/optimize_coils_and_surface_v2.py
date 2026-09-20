import os
from time import time
# import jax # gftd13@gmail.ocm
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B
from essos.losses import custom_loss
from essos.objective_functions import ( loss_BdotN_mean, loss_coil_curvature_from_field,
                                        loss_coil_length_max, loss_mean_cross_sectional_area,
                                        loss_quasi_symmetry, loss_surface_normal_displacement,
                                        quasi_symmetry_residual_on_surface )


#  In this exmple, `scipy.optimize.least_squares` is used, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

# There are two different controls:
# x_scale  → controls how easily variables move
# weights  → control what results the optimizer considers important

# ====================================================================================
# ====================================================================================
""" INPUT DATA """
# ====================================================================================
# ====================================================================================

# PATH TO THE VMEC INPUT FILE (wout file) FOR THE SURFACE ----------------------------
input_filepath = os.path.join(os.path.dirname(__file__), "..", "input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')

# SURFACE PARAMETERS -----------------------------------------------------------------
ntheta = 30; nphi = 30; Npoints = ntheta * nphi

#  CONTROL PARAMETERS FOR THE OPTIMIZATION -------------------------------------------
# There are two different controls:
# x_scale  → controls how easily variables move
SCALE_FIELD = 1.0
SCALE_SURFACE = 1.0

# Losses weights and targets  → control what results the optimizer considers important
# Field ~~~~~~~~~~~~~~~~~~~~
NORMAL_FIELD_WEIGHT = Npoints;
QS_WEIGHT = 1.
# Coils ~~~~~~~~~~~~~~~~~~~~
LENGTH_WEIGHT = 10.; LENGTH_TARGET = 32.;
CURVATURE_WEIGHT = 100.; CURVATURE_TARGET = 0.1
# Surface ~~~~~~~~~~~~~~~~~~~~
CROSS_SECTIONAL_AREA_WEIGHT = 1e4
NORMAL_DISPLACEMENT_WEIGHT = 1e4      # Importance of this constraint in the total loss

# ====================================================================================
# ====================================================================================
""" Initializing coils, field and surface """
# ====================================================================================
# ====================================================================================

N_COILS = 3; FOURIER_ORDER = 3; LARGE_R = 10; SMALL_R = 5.6; NFP = 2; N_SEGMENTS = 45; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

# Initialize the coils and their corresponding field
init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
field_init = BiotSavart(init_coils)

# Initialize the surface from a VMEC output file.
surface_init = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='half period')

# Initialize reference values for the surface geometry. These will be used to compute the surface normal displacement loss. 
# Compute and cache the initial surface coordinates and unit normals before JAX traces the optimization.
# Wait for both calculations to finish and store them as fixed reference arrays for the displacement loss.
surface_gamma_reference = surface_init.gamma.block_until_ready()
unitnormal_reference = surface_init.unitnormal.block_until_ready()

CROSS_SECTIONAL_AREA_TARGET = float(surface_init.area_section_by_phi())
LENGTH_SCALE_SURFACE = float(jnp.sqrt(CROSS_SECTIONAL_AREA_TARGET / jnp.pi))

# ====================================================================================
# ====================================================================================
""" Defining custom losses """
# ====================================================================================
# ====================================================================================

L_normal_field = custom_loss( loss_BdotN_mean, "field", "surface" )
L_length_max = custom_loss( loss_coil_length_max , "field", max_coil_length=LENGTH_TARGET )
L_curvature = custom_loss( loss_coil_curvature_from_field , "field" , max_coil_curvature=CURVATURE_TARGET )
L_quasi_symmetry = custom_loss(loss_quasi_symmetry, "field", "surface")
L_cross_sectional_area = custom_loss( loss_mean_cross_sectional_area, "surface", target_area=CROSS_SECTIONAL_AREA_TARGET )

L_surface_normal_displacement = custom_loss( loss_surface_normal_displacement , "surface" ,
                                            surface_gamma_reference=surface_gamma_reference , unitnormal_reference=unitnormal_reference,
                                            length_scale=LENGTH_SCALE_SURFACE )


# ====================================================================================
# ====================================================================================
""" Defining total loss + setting dependencies """
# ====================================================================================
# ====================================================================================

# The total loss is a weighted sum of the individual losses.
L_total = ( NORMAL_FIELD_WEIGHT*L_normal_field + 
           LENGTH_WEIGHT*L_length_max + 
           CURVATURE_WEIGHT*L_curvature + 
           CROSS_SECTIONAL_AREA_WEIGHT*L_cross_sectional_area +
           NORMAL_DISPLACEMENT_WEIGHT*L_surface_normal_displacement +
           QS_WEIGHT*L_quasi_symmetry )

# The dependencies of the total loss are set to the field and surface. Both will be mdified during the optimization.
L_total.dependencies = { "field": field_init, "surface": surface_init }


# We assign different scaling factors to the field and surface dofs to balance their characteristic step sizes.
NN_field_dofs = L_total.dependencies["field"].dofs.size
NN_surface_dofs = L_total.dependencies["surface"].dofs.size
x_scale_optimization = jnp.concatenate([
    SCALE_FIELD   * jnp.ones(NN_field_dofs),
    SCALE_SURFACE * jnp.ones(NN_surface_dofs),
])


# def grad(x, scale_grad=1e-1):
#     gradient = L_total.grad(x)
#     NN_surface_dofs = L_total.dependencies["surface"].dofs.size
#     NN_field_dofs = L_total.dependencies["field"].dofs.size
#     total_grad = jnp.concatenate([gradient[:NN_field_dofs], scale_grad*gradient[NN_field_dofs:]])
#     return total_grad

# ====================================================================================
# ====================================================================================
""" Optimizing the total loss """
# ====================================================================================
# ====================================================================================

t_start = time()
res = least_squares(L_total, L_total.starting_dofs, L_total.grad, x_scale=x_scale_optimization,
                     verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=200)
# res = least_squares(L_total, L_total.starting_dofs, lambda x: grad(x, scale_grad=1e-1), verbose=2, ftol=1e-7, gtol=1e-7, xtol=1e-14, max_nfev=200)
# res = least_squares(L_total, res.x, lambda x: grad(x, scale_grad=3e-1), verbose=2, ftol=1e-7, gtol=1e-7, xtol=1e-14, max_nfev=300)
# res = least_squares(L_total, res.x, lambda x: grad(x, scale_grad=6e-1), verbose=2, ftol=1e-7, gtol=1e-7, xtol=1e-14, max_nfev=300)
# res = least_squares(L_total, res.x, lambda x: grad(x, scale_grad=1), verbose=2, ftol=1e-7, gtol=1e-7, xtol=1e-14, max_nfev=300)
t_end = time()

# exit()

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
# The optimized field and surface are extracted from the results of the optimization.
opt_dict = L_total.dofs_to_pytree(res.x)

# The optimized field and surface are extracted from the results of the optimization.
field_opt = opt_dict["field"]
surface_opt = opt_dict["surface"]

# Coils geometry is extracted from the optimized field.
opt_coils = field_opt.coils


# Surface displacement between corresponding grid points
surface_displacement_xyz = surface_opt.gamma - surface_init.gamma
surface_displacement = jnp.linalg.norm(surface_displacement_xyz, axis=2)# Total displacement magnitude
surface_normal_displacement = jnp.sum( surface_displacement_xyz * surface_init.unitnormal, axis=2 ) # Displacement perpendicular to the initial surface



# ====================================================================================
# ====================================================================================
""" Printing a comparison of the initial loss and the optimized loss """
# ====================================================================================
# ====================================================================================


print("\n---------------------------------------------------------------------------")
print("\nSurface displacement:")
print("Mean total displacement:", jnp.mean(surface_displacement))
print("RMS total displacement:", jnp.sqrt(jnp.mean(surface_displacement**2)))
print("Maximum total displacement:", jnp.max(surface_displacement))
print("Mean absolute normal displacement:", jnp.mean(jnp.abs(surface_normal_displacement)))
print("RMS normal displacement:", jnp.sqrt(jnp.mean(surface_normal_displacement**2)))
print("Maximum absolute normal displacement:", jnp.max(jnp.abs(surface_normal_displacement)))

print("\n---------------------------------------------------------------------------")
print("\nNormal-field residuals and losses:")
B_dot_n_over_B_init = BdotN_over_B( surface_init , field_init )
B_dot_n_over_B_opt = BdotN_over_B( surface_opt , field_opt )
print("mean abs residual (initial):", jnp.mean(jnp.abs(B_dot_n_over_B_init)) )
print("mean abs residual (optimized):",jnp.mean(jnp.abs(B_dot_n_over_B_opt)) )
print("max abs residual (initial):",jnp.max(jnp.abs(B_dot_n_over_B_init)) )
print("max abs residual (optimized):",jnp.max(jnp.abs(B_dot_n_over_B_opt)) )
print("Normal-field loss (initial):",loss_BdotN_mean(field_init, surface_init) )
print("Normal-field loss (optimized):",loss_BdotN_mean(field_opt, surface_opt) )

print("\n---------------------------------------------------------------------------")
print("\nCoil-length residuals and losses:")
coil_length_residual_init = jnp.maximum(0, field_init.coils.length - LENGTH_TARGET)
coil_length_residual_opt = jnp.maximum(0, field_opt.coils.length - LENGTH_TARGET)
print("Coil lengths (initial):", field_init.coils.length)
print("Coil lengths (optimized):", field_opt.coils.length)
print("mean excess length (initial):", jnp.mean(coil_length_residual_init))
print("mean excess length (optimized):", jnp.mean(coil_length_residual_opt))
print("max excess length (initial):", jnp.max(coil_length_residual_init))
print("max excess length (optimized):", jnp.max(coil_length_residual_opt))
print("Length loss (initial):", loss_coil_length_max(field_init, max_coil_length=LENGTH_TARGET))
print("Length loss (optimized):", loss_coil_length_max(field_opt, max_coil_length=LENGTH_TARGET))

print("\n---------------------------------------------------------------------------")
print("\nCoil-curvature residuals and losses:")
coil_curvature_residual_init = jnp.maximum(0, field_init.coils.curvature - CURVATURE_TARGET)
coil_curvature_residual_opt = jnp.maximum(0, field_opt.coils.curvature - CURVATURE_TARGET)
print("mean curvature (initial):", jnp.mean(field_init.coils.curvature))
print("mean curvature (optimized):", jnp.mean(field_opt.coils.curvature))
print("max curvature (initial):", jnp.max(field_init.coils.curvature))
print("max curvature (optimized):", jnp.max(field_opt.coils.curvature))
print("mean excess curvature (initial):", jnp.mean(coil_curvature_residual_init))
print("mean excess curvature (optimized):", jnp.mean(coil_curvature_residual_opt))
print("max excess curvature (initial):", jnp.max(coil_curvature_residual_init))
print("max excess curvature (optimized):", jnp.max(coil_curvature_residual_opt))
print("Curvature loss (initial):", loss_coil_curvature_from_field(field_init, max_coil_curvature=CURVATURE_TARGET))
print("Curvature loss (optimized):", loss_coil_curvature_from_field(field_opt, max_coil_curvature=CURVATURE_TARGET))

print("\n---------------------------------------------------------------------------")
print("\nQuasi-symmetry residuals and losses:")
QS_residual_xyz_init = quasi_symmetry_residual_on_surface(field_init, surface_init)
QS_residual_xyz_opt = quasi_symmetry_residual_on_surface(field_opt, surface_opt)
print("mean abs residual (initial):", jnp.mean(jnp.abs(QS_residual_xyz_init)))
print("mean abs residual (optimized):", jnp.mean(jnp.abs(QS_residual_xyz_opt)))
print("max abs residual (initial):", jnp.max(jnp.abs(QS_residual_xyz_init)))
print("max abs residual (optimized):", jnp.max(jnp.abs(QS_residual_xyz_opt)))
print("QS loss (initial):", loss_quasi_symmetry(field_init, surface_init))
print("QS loss (optimized):", loss_quasi_symmetry(field_opt, surface_opt))

print("\n---------------------------------------------------------------------------")
print("\nMean cross-sectional area:")
cross_sectional_area_initial = surface_init.area_section_by_phi()
cross_sectional_area_optimized = surface_opt.area_section_by_phi()
relative_cross_sectional_area_change = ( cross_sectional_area_optimized - cross_sectional_area_initial ) / cross_sectional_area_initial

print("Initial area:", cross_sectional_area_initial)
print("Optimized area:", cross_sectional_area_optimized)
print("Absolute area change:", jnp.abs(cross_sectional_area_optimized - cross_sectional_area_initial))
print("Relative area change:", relative_cross_sectional_area_change)
print("Relative area change (%):", 100 * relative_cross_sectional_area_change)
print("Area loss:", loss_mean_cross_sectional_area( surface_opt , target_area=CROSS_SECTIONAL_AREA_TARGET ) )
print("Weighted area loss:", CROSS_SECTIONAL_AREA_WEIGHT * loss_mean_cross_sectional_area( surface_opt, target_area=CROSS_SECTIONAL_AREA_TARGET ) )


# ====================================================================================
# ====================================================================================
""" Plotting the initial and optimized coils """
# ====================================================================================
# ====================================================================================

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(121, projection='3d')
init_coils.plot(ax=ax1, show=False)
surface_init.plot(ax=ax1, show=False)

ax2 = fig.add_subplot(122, projection='3d')
opt_coils.plot(ax=ax2, show=False)
surface_opt.plot(ax=ax2, show=False)

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
    surface_init.to_vtk(os.path.join(output_filepath, "init_surface_vmec_surface.json"), field=field_init)
    surface_opt.to_vtk(os.path.join(output_filepath, "final_surface_vmec_surface.json"), field=field_opt)
    init_coils.to_vtk(os.path.join(output_filepath, "init_coils_vmec_surface.json"))
    opt_coils.to_vtk(os.path.join(output_filepath, "opt_coils_vmec_surface.json"))
