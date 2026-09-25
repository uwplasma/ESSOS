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
                                        loss_quasi_symmetry, loss_surface_curvature_section,
                                        loss_surface_normal_displacement,
                                        loss_surface_poloidal_derivative,
                                        quasi_symmetry_residual_on_surface )


#  In this exmple, `scipy.optimize.least_squares` is used, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares, minimize

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
# vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
vmec_input = os.path.join(input_filepath, 'input.toroidal_surface_02')


# SURFACE PARAMETERS -----------------------------------------------------------------
ntheta = 26; nphi = 26; Npoints = ntheta * nphi

#  CONTROL PARAMETERS FOR THE OPTIMIZATION -------------------------------------------
# There are two different controls:
# x_scale  → controls how easily variables move
SCALE_FIELD = 2.18
SCALE_SURFACE = 1.

# Losses weights and targets  → control what results the optimizer considers important

# Coils ~~~~~~~~~~~~~~~~~~~~
LENGTH_WEIGHT = 10.; LENGTH_TARGET = 40.;
CURVATURE_WEIGHT = 100.; CURVATURE_TARGET = 0.5

# Field ~~~~~~~~~~~~~~~~~~~~
NORMAL_FIELD_WEIGHT = 900.;
QS_WEIGHT = 30.

# Surface ~~~~~~~~~~~~~~~~~~~~
CROSS_SECTIONAL_AREA_WEIGHT = 1e3
NORMAL_DISPLACEMENT_WEIGHT = 0. # 1e4      # Importance of this constraint in the total loss

# Surface-parametrization control: prevent qq = ||dc_phi/dtheta|| from approaching zero.
QQ_WEIGHT = 0. # 1e5;
ALPHA_QQ = 1.

# Cross-sectional curvature control: limit kappa relative to the initial maximum.
KAPPA_WEIGHT = 0. # 1e5;
ALPHA_KAPPA = 1.


# ====================================================================================
# ====================================================================================
""" DIAGNOSTIC FUNCTIONS """
# ====================================================================================
# ====================================================================================

def diagnostic_gradient(loss_total, dofs, field, surface, scale_field, scale_surface, label):
    NN_field_dofs = field.dofs.size
    NN_surface_dofs = surface.dofs.size

    gradient = loss_total.grad(dofs)
    gradient_field = gradient[:NN_field_dofs]
    gradient_surface = gradient[NN_field_dofs:NN_field_dofs + NN_surface_dofs]

    gradient_rms_field = jnp.sqrt(jnp.mean(jnp.square(gradient_field)))
    gradient_rms_surface = jnp.sqrt(jnp.mean(jnp.square(gradient_surface)))

    gradient_rms_field_scaled = scale_field * gradient_rms_field
    gradient_rms_surface_scaled = scale_surface * gradient_rms_surface

    print("\n---------------------------------------------------------------------------")
    print(f"\n{label} gradient diagnostics:")
    print("Number of field dofs:", NN_field_dofs)
    print("Number of surface dofs:", NN_surface_dofs)
    print("Field-gradient L2 norm:", jnp.linalg.norm(gradient_field))
    print("Surface-gradient L2 norm:", jnp.linalg.norm(gradient_surface))
    print("Field-gradient RMS:", gradient_rms_field)
    print("Surface-gradient RMS:", gradient_rms_surface)
    print("Surface/field RMS ratio:", gradient_rms_surface / gradient_rms_field)
    print("Scaled field-gradient RMS:", gradient_rms_field_scaled)
    print("Scaled surface-gradient RMS:", gradient_rms_surface_scaled)
    print("Scaled surface/field RMS ratio:", gradient_rms_surface_scaled / gradient_rms_field_scaled)

# ====================================================================================
# ====================================================================================
""" Initializing coils, field and surface """
# ====================================================================================
# ====================================================================================

# ------------------------------------------------------------------------------------
# Define the initial coils and their corresponding field.
# The coils are initialized as equally spaced curves around a torus.
N_COILS = 3; FOURIER_ORDER = 3; LARGE_R = 10; SMALL_R = 5.6; NFP = 2; N_SEGMENTS = 40; STELLSYM = True
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
field_init = BiotSavart(init_coils)

# ------------------------------------------------------------------------------------
# Initialize the surface from a VMEC output file.
# surface_init = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='half period')
surface_init = SurfaceRZFourier.from_input_file( vmec_input, ntheta=ntheta, nphi=nphi, close=True, range_torus='half period')
# from_input_file(cls, file, ntheta=30, nphi=30, close=True, range_torus='full torus')

# ------------------------------------------------------------------------------------
# Initialize reference values for the surface geometry. These will be used to compute the surface normal displacement loss. 
# Compute and cache the initial surface coordinates and unit normals before JAX traces the optimization.
# Wait for both calculations to finish and store them as fixed reference arrays for the displacement loss.
surface_gamma_reference = surface_init.gamma.block_until_ready()
unitnormal_reference = surface_init.unitnormal.block_until_ready()
qq_reference = float( jnp.min(jnp.linalg.norm(surface_init.gammadash_theta, axis=2)) ) # Initial minimum = qq = ||∂c_phi/∂theta||
kappa_reference = float(jnp.max(surface_init.curvature_section_by_phi()))


CROSS_SECTIONAL_AREA_TARGET = float(surface_init.area_section_by_phi())
LENGTH_SCALE_SURFACE = float(jnp.sqrt(CROSS_SECTIONAL_AREA_TARGET / jnp.pi))
kappa_max = ALPHA_KAPPA * kappa_reference

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
L_surface_poloidal_derivative = custom_loss( loss_surface_poloidal_derivative, "surface", qq_reference=qq_reference, alpha_qq=ALPHA_QQ )
L_surface_curvature_section = custom_loss( loss_surface_curvature_section , "surface" , kappa_max=kappa_max )


# ====================================================================================
# ====================================================================================
""" Defining total loss + setting dependencies """
# ====================================================================================
# ====================================================================================

# The total loss is a weighted sum of the individual losses.
# L_total = ( NORMAL_FIELD_WEIGHT*L_normal_field + 
#            LENGTH_WEIGHT*L_length_max + 
#            CURVATURE_WEIGHT*L_curvature + 
#            CROSS_SECTIONAL_AREA_WEIGHT*L_cross_sectional_area +
#            NORMAL_DISPLACEMENT_WEIGHT*L_surface_normal_displacement +
#            QQ_WEIGHT*L_surface_poloidal_derivative +
#            KAPPA_WEIGHT*L_surface_curvature_section +
#            QS_WEIGHT*L_quasi_symmetry )


losses_weighted = [(NORMAL_FIELD_WEIGHT, L_normal_field),
                   (LENGTH_WEIGHT, L_length_max),
                   (CURVATURE_WEIGHT, L_curvature),
                   (CROSS_SECTIONAL_AREA_WEIGHT, L_cross_sectional_area),
                   (NORMAL_DISPLACEMENT_WEIGHT, L_surface_normal_displacement),
                   (QQ_WEIGHT, L_surface_poloidal_derivative),
                   (KAPPA_WEIGHT, L_surface_curvature_section),
                   (QS_WEIGHT, L_quasi_symmetry) ]

losses_active = [ weight * loss for weight, loss in losses_weighted if weight != 0.0 ]

if not losses_active:
    raise ValueError("At least one loss weight must be nonzero.")

L_total = losses_active[0]

for loss_active in losses_active[1:]:
    L_total = L_total + loss_active



# The dependencies of the total loss are set to the field and surface. Both will be modified during the optimization.
L_total.dependencies = { "field": field_init, "surface": surface_init }


# We assign different scaling factors to the field and surface dofs to balance their characteristic step sizes.
NN_field_dofs = L_total.dependencies["field"].dofs.size
NN_surface_dofs = L_total.dependencies["surface"].dofs.size


x_scale_optimization = jnp.concatenate([
    SCALE_FIELD   * jnp.ones(NN_field_dofs),
    SCALE_SURFACE * jnp.ones(NN_surface_dofs),
])

# Diagnostic of the initial gradient of the total loss with respect to the field and surface dofs.
diagnostic_gradient(L_total, L_total.starting_dofs, field_init, surface_init, SCALE_FIELD, SCALE_SURFACE, "Initial")


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
def loss_callback(intermediate_result): print(f"Current loss: {intermediate_result.fun}")
t_start = time()
res = minimize(L_total, L_total.starting_dofs, jac = L_total.grad, method = "L-BFGS-B", callback=loss_callback ,
                 options={'maxiter': 200, 'disp': True} )



# res = least_squares(L_total, L_total.starting_dofs, L_total.grad, x_scale=x_scale_optimization,
#                      verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=300)
# res = least_squares(L_total, L_total.starting_dofs, lambda x: grad(x, scale_grad=1e-1), verbose=2, ftol=1e-7, gtol=1e-7, xtol=1e-14, max_nfev=200)
# res = least_squares(L_total, res.x, lambda x: grad(x, scale_grad=3e-1), verbose=2, ftol=1e-7, gtol=1e-7, xtol=1e-14, max_nfev=300)
# res = least_squares(L_total, res.x, lambda x: grad(x, scale_grad=6e-1), verbose=2, ftol=1e-7, gtol=1e-7, xtol=1e-14, max_nfev=300)
# res = least_squares(L_total, res.x, lambda x: grad(x, scale_grad=1), verbose=2, ftol=1e-7, gtol=1e-7, xtol=1e-14, max_nfev=300)
t_end = time()

diagnostic_gradient(L_total, res.x, field_init, surface_init, SCALE_FIELD, SCALE_SURFACE, "Optimized")


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
""" Printing results """
# ====================================================================================
# ====================================================================================

print("\n===========================================================================")
print("\n===========================================================================")
print("\nGENERAL RESULTS:")
print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", L_total(L_total.starting_dofs))    
print("Loss after optimization:", L_total(res.x))

B_dot_n_over_B_init = BdotN_over_B( surface_init , field_init )
B_dot_n_over_B_opt = BdotN_over_B( surface_opt , field_opt )

print("max|B dot n| residual (initial):",jnp.max(jnp.abs(B_dot_n_over_B_init)) )
print("max|B dot n| residual (optimized):",jnp.max(jnp.abs(B_dot_n_over_B_opt)) )


print("\n===========================================================================")
print("\n===========================================================================")
print("\nWEIGHTED LOSSES:")

if NORMAL_FIELD_WEIGHT != 0.0:
    print("\n---------------------------------------------------------------------------")
    print("\nNormal-field weighted losses:")
    print("NORMAL-FIELD LOSS (INITIAL, WEIGHTED):", NORMAL_FIELD_WEIGHT * loss_BdotN_mean(field_init, surface_init) )
    print("NORMAL-FIELD LOSS (OPTIMIZED, WEIGHTED):",NORMAL_FIELD_WEIGHT * loss_BdotN_mean(field_opt, surface_opt) )

if QS_WEIGHT != 0.0:
    print("\n---------------------------------------------------------------------------")
    print("\nQuasi-symmetry weighted losses:")
    print("QS LOSS (INITIAL, WEIGHTED):", QS_WEIGHT * loss_quasi_symmetry(field_init, surface_init) )
    print("QS LOSS (OPTIMIZED, WEIGHTED):",QS_WEIGHT * loss_quasi_symmetry(field_opt, surface_opt) )


if CROSS_SECTIONAL_AREA_WEIGHT != 0.0:
    print("\n---------------------------------------------------------------------------")
    print("\nMean cross-sectional area weighted losses:")
    print("AREA LOSS (INITIAL, WEIGHTED):", CROSS_SECTIONAL_AREA_WEIGHT * loss_mean_cross_sectional_area( surface_init, target_area=CROSS_SECTIONAL_AREA_TARGET ) )
    print("AREA LOSS (OPTIMIZED, WEIGHTED):", CROSS_SECTIONAL_AREA_WEIGHT * loss_mean_cross_sectional_area( surface_opt, target_area=CROSS_SECTIONAL_AREA_TARGET ) )


if NORMAL_DISPLACEMENT_WEIGHT != 0.0:
    print("\n---------------------------------------------------------------------------")
    print("\nSurface normal-displacement weighted losses:")
    print("NORMAL-DISPLACEMENT LOSS (INITIAL, WEIGHTED):", NORMAL_DISPLACEMENT_WEIGHT * loss_surface_normal_displacement( surface_init, surface_gamma_reference=surface_gamma_reference, unitnormal_reference=unitnormal_reference, length_scale=LENGTH_SCALE_SURFACE ) )
    print("NORMAL-DISPLACEMENT LOSS (OPTIMIZED, WEIGHTED):", NORMAL_DISPLACEMENT_WEIGHT * loss_surface_normal_displacement( surface_opt, surface_gamma_reference=surface_gamma_reference, unitnormal_reference=unitnormal_reference, length_scale=LENGTH_SCALE_SURFACE ) )


if KAPPA_WEIGHT != 0.0:
    print("\n---------------------------------------------------------------------------")
    print("\nCross-sectional surface curvature:")
    loss_kappa_init = loss_surface_curvature_section(surface_init, kappa_max=kappa_max)
    loss_kappa_opt = loss_surface_curvature_section(surface_opt, kappa_max=kappa_max)
    print("CROSS-SECTIONAL SURFACE CURVATURE LOSS (INITIAL, WEIGHTED):", KAPPA_WEIGHT * loss_kappa_init)
    print("CROSS-SECTIONAL SURFACE CURVATURE LOSS (OPTIMIZED, WEIGHTED):", KAPPA_WEIGHT * loss_kappa_opt)

if QQ_WEIGHT != 0.0:
    print("\n---------------------------------------------------------------------------")
    print("\nNorm of the poloidal-parametrization derivative:")
    loss_qq_init = loss_surface_poloidal_derivative(surface_init, qq_reference=qq_reference, alpha_qq=ALPHA_QQ)
    loss_qq_opt = loss_surface_poloidal_derivative(surface_opt, qq_reference=qq_reference, alpha_qq=ALPHA_QQ)
    print("POLOIDAL-DERIVATIVE LOSS (INITIAL, WEIGHTED):", QQ_WEIGHT * loss_qq_init)
    print("POLOIDAL-DERIVATIVE LOSS (OPTIMIZED, WEIGHTED):", QQ_WEIGHT * loss_qq_opt)

if LENGTH_WEIGHT != 0.0:
    print("\n---------------------------------------------------------------------------")
    print("\nCoil-length residuals and losses:")
    print("LENGTH LOSS (INITIAL, WEIGHTED):", LENGTH_WEIGHT * loss_coil_length_max(field_init, max_coil_length=LENGTH_TARGET))
    print("LENGTH LOSS (OPTIMIZED, WEIGHTED):", LENGTH_WEIGHT * loss_coil_length_max(field_opt, max_coil_length=LENGTH_TARGET))

if CURVATURE_WEIGHT != 0.0:
    print("\n---------------------------------------------------------------------------")
    print("\nCoil-curvature residuals and losses:")
    print("COIL CURVATURE LOSS (INITIAL, WEIGHTED):", CURVATURE_WEIGHT * loss_coil_curvature_from_field(field_init, max_coil_curvature=CURVATURE_TARGET))
    print("COIL CURVATURE LOSS (OPTIMIZED, WEIGHTED):", CURVATURE_WEIGHT * loss_coil_curvature_from_field(field_opt, max_coil_curvature=CURVATURE_TARGET))




print("\n===========================================================================")
print("\n===========================================================================")
print("\nDIAGNOSTICS:")
print("\n---------------------------------------------------------------------------")
print("\nNormal-field:")
B_dot_n_over_B_init = BdotN_over_B( surface_init , field_init )
B_dot_n_over_B_opt = BdotN_over_B( surface_opt , field_opt )
print("mean abs residual (initial):", jnp.mean(jnp.abs(B_dot_n_over_B_init)) )
print("mean abs residual (optimized):",jnp.mean(jnp.abs(B_dot_n_over_B_opt)) )
print("max abs residual (initial):",jnp.max(jnp.abs(B_dot_n_over_B_init)) )
print("max abs residual (optimized):",jnp.max(jnp.abs(B_dot_n_over_B_opt)) )
if NORMAL_FIELD_WEIGHT != 0.0:
    print("Normal-field loss (initial):",loss_BdotN_mean(field_init, surface_init) )
    print("Normal-field loss (optimized):",loss_BdotN_mean(field_opt, surface_opt) )


print("\n---------------------------------------------------------------------------")
print("\nQuasi-symmetry residuals and losses:")
QS_residual_xyz_init = quasi_symmetry_residual_on_surface(field_init, surface_init)
QS_residual_xyz_opt = quasi_symmetry_residual_on_surface(field_opt, surface_opt)
print("mean abs residual (initial):", jnp.mean(jnp.abs(QS_residual_xyz_init)))
print("mean abs residual (optimized):", jnp.mean(jnp.abs(QS_residual_xyz_opt)))
print("max abs residual (initial):", jnp.max(jnp.abs(QS_residual_xyz_init)))
print("max abs residual (optimized):", jnp.max(jnp.abs(QS_residual_xyz_opt)))
if QS_WEIGHT != 0.0:
    print("QS loss (initial field, initial surface):", loss_quasi_symmetry(field_init, surface_init))
    print("QS loss (optimized field, initial surface):", loss_quasi_symmetry(field_opt, surface_init))
    print("QS loss (initial field, optimized surface):", loss_quasi_symmetry(field_init, surface_opt))
    print("QS loss (optimized field, optimized surface):", loss_quasi_symmetry(field_opt, surface_opt))

    coils_hybrid = Coils(curves=opt_coils.curves, currents=init_coils.dofs_currents_raw)
    field_hybrid = BiotSavart(coils_hybrid)
    print("QS loss (optimized coil shapes, initial currents, initial surface):", loss_quasi_symmetry(field_hybrid, surface_init))


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

if CROSS_SECTIONAL_AREA_WEIGHT != 0.0:
    print("Area loss:", loss_mean_cross_sectional_area( surface_opt , target_area=CROSS_SECTIONAL_AREA_TARGET ) )
    print("Weighted area loss:", CROSS_SECTIONAL_AREA_WEIGHT * loss_mean_cross_sectional_area( surface_opt, target_area=CROSS_SECTIONAL_AREA_TARGET ) )


print("\n---------------------------------------------------------------------------")
print("\nCross-sectional curvature:")

curvature_section_init = surface_init.curvature_section_by_phi()
curvature_section_opt = surface_opt.curvature_section_by_phi()

print("Curvature array shape:", curvature_section_init.shape)
print("Mean curvature (initial):", jnp.mean(curvature_section_init))
print("Mean curvature (optimized):", jnp.mean(curvature_section_opt))
print("Maximum curvature (initial):", jnp.max(curvature_section_init))
print("Maximum curvature (optimized):", jnp.max(curvature_section_opt))
print("95th percentile (initial):", jnp.percentile(curvature_section_init, 95))
print("95th percentile (optimized):", jnp.percentile(curvature_section_opt, 95))
print("99th percentile (initial):", jnp.percentile(curvature_section_init, 99))
print("99th percentile (optimized):", jnp.percentile(curvature_section_opt, 99))

index_curvature_max_opt = jnp.unravel_index(jnp.argmax(curvature_section_opt), curvature_section_opt.shape)
index_phi_max_opt = int(index_curvature_max_opt[0])
index_theta_max_opt = int(index_curvature_max_opt[1])

print("Index of optimized maximum (phi, theta):", index_phi_max_opt, index_theta_max_opt)
print("Phi at optimized maximum:", surface_opt.phi2d[index_phi_max_opt, index_theta_max_opt])
print("Theta at optimized maximum:", surface_opt.theta2d[index_phi_max_opt, index_theta_max_opt])

if KAPPA_WEIGHT != 0.0:
    loss_kappa_init = loss_surface_curvature_section(surface_init, kappa_max=kappa_max)
    loss_kappa_opt = loss_surface_curvature_section(surface_opt, kappa_max=kappa_max)
    print("Reference maximum curvature:", kappa_reference)
    print("Allowed maximum curvature:", kappa_max)
    print("Cross-sectional curvature loss (initial):", loss_kappa_init)
    print("Cross-sectional curvature loss (optimized):", loss_kappa_opt)
    print("Weighted cross-sectional curvature loss (optimized):", KAPPA_WEIGHT * loss_kappa_opt)


print("\n---------------------------------------------------------------------------")
print("\nNorm of the poloidal-parametrization derivative:")

qq_init = jnp.linalg.norm(surface_init.gammadash_theta, axis=2)
qq_opt = jnp.linalg.norm(surface_opt.gammadash_theta, axis=2)
qq_minimum_allowed = ALPHA_QQ * qq_reference

print("Minimum ||dc_phi/dtheta|| (initial):", jnp.min(qq_init))
print("Minimum ||dc_phi/dtheta|| (optimized):", jnp.min(qq_opt))
print("||dc_phi/dtheta|| at maximum optimized curvature:", qq_opt[index_phi_max_opt, index_theta_max_opt])

if QQ_WEIGHT != 0.0:
    loss_qq_init = loss_surface_poloidal_derivative(surface_init, qq_reference=qq_reference, alpha_qq=ALPHA_QQ)
    loss_qq_opt = loss_surface_poloidal_derivative(surface_opt, qq_reference=qq_reference, alpha_qq=ALPHA_QQ)
    print("Reference minimum ||dc_phi/dtheta||:", qq_reference)
    print("Allowed minimum ||dc_phi/dtheta||:", qq_minimum_allowed)
    print("Poloidal-derivative loss (initial):", loss_qq_init)
    print("Poloidal-derivative loss (optimized):", loss_qq_opt)
    print("Weighted poloidal-derivative loss (optimized):", QQ_WEIGHT * loss_qq_opt)


print("\n---------------------------------------------------------------------------")
print("\nCoils displacement:")
coil_displacement = jnp.linalg.norm(opt_coils.gamma - init_coils.gamma, axis=-1)
print("Mean coil-point displacement:", jnp.mean(coil_displacement))
print("Maximum coil-point displacement:", jnp.max(coil_displacement))

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

if LENGTH_WEIGHT != 0.0:
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


if CURVATURE_WEIGHT != 0.0:
    print("Curvature loss (initial):", loss_coil_curvature_from_field(field_init, max_coil_curvature=CURVATURE_TARGET))
    print("Curvature loss (optimized):", loss_coil_curvature_from_field(field_opt, max_coil_curvature=CURVATURE_TARGET))



print("\n---------------------------------------------------------------------------")
print("\nCoil currents:")
print("Base-coil currents (initial):", init_coils.dofs_currents_raw)
print("Base-coil currents (optimized):", opt_coils.dofs_currents_raw)



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
