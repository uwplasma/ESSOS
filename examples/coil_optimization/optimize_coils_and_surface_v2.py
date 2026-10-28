import os
from copy import copy
from time import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import netCDF4
import numpy as np

# ESSOS imports
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import ( SurfaceRZFourier, BdotN_over_B,
                             iota_approx_on_surface, toroidal_flux )
from essos.losses import custom_loss
from essos.objective_functions import ( loss_BdotN_mean_squared, loss_coil_curvature_from_field,
                                        loss_coil_length_max, loss_cross_sectional_area_mean,
                                        loss_quasi_symmetry, loss_iota_approx,
                                        quasi_symmetry_residual_on_surface )

from scipy.optimize import minimize

# Imports to run VMEX after the optimization is obtained.
from dataclasses import replace
import vmex as vj

import datetime

# ====================================================================================
# ====================================================================================
""" INPUT DATA """
# ====================================================================================
# ====================================================================================

# Exporting results and the folder name
EXPORT = True
output_filepath = os.path.join(os.path.dirname(__file__), "output_coils_surface_opt/mode1")
os.makedirs(output_filepath, exist_ok=True) # Make the path

# True -> run a fixed-boundary VMEX equilibrium on the exported surface
RUN_VMEX = True

# True -> skip optimization, load saved dofs from output_filepath
LOAD_RESULT = False

# CONTINUATION -----------------------------------------------------------------------
# False -> start from the initial torus (vmec_input) and equally spaced coils.
# True -> start from an already optimized surface and coils, at a higher surface resolution (select the folder).
RESTART_FROM = False
restart_filepath = os.path.join(os.path.dirname(__file__), "output_coils_surface_opt/test_QA_iota0p4_it00")
SURFACE_MPOL = 1; SURFACE_NTOR = 1   # maximum boundary m and |n|
VMEC_MPOL = max(SURFACE_MPOL + 2, 5); VMEC_NTOR = max(SURFACE_NTOR + 2, 5)
# Boundary modes + 2 (minimum 5) is a starting resolution; refine VMEX to check QA/iota.
# Continuation: set RESTART_FROM, restart_filepath, output_filepath, and surface modes.
# Short continuation: mode 2 uses IOTA_MODES=8; mode 3 uses 12 and a 64 x 65 grid.
# For mode 3, start with CONSTRAINT_WEIGHT=3e7, QS_WEIGHT=1e5 and scales 0.05/0.02.
# IOTA_MODES can be raised independently when checking the fitted angle.
# Coil order follows the surface limits; check final VMEX resolution independently.

# PATH TO THE VMEC INPUT FILE (wout file) FOR THE SURFACE ----------------------------
input_filepath = os.path.join(os.path.dirname(__file__), "..", "input_files")
# vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
vmec_input = os.path.join(input_filepath, 'input.toroidal_surface_02')

# SURFACE PARAMETERS -----------------------------------------------------------------
ntheta = 32; nphi = 33

# COILS PARAMETERS -------------------------------------------------------------------
# The coils are initialized as equally spaced curves around a torus.
N_COILS = 4; FOURIER_ORDER = 2 * max(SURFACE_MPOL, SURFACE_NTOR) + 2
LARGE_R = 10; SMALL_R = 5.6; NFP = 2;
N_SEGMENTS = 64; STELLSYM = True
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

#  CONTROL PARAMETERS FOR THE OPTIMIZATION -------------------------------------------
# There are two different controls (A and B):

# A) Losses weights and targets  → control what results the optimizer considers important

# Six physical objectives: normal field, QS, iota, section area, length, curvature.
CONSTRAINT_WEIGHT = 3e6  # Common weight for B.n/B, target iota, and section area.
QS_WEIGHT = 1e4          # Globally normalized helicity-free three-term product.
LENGTH_WEIGHT = 30.; LENGTH_TARGET = 50.
CURVATURE_WEIGHT = 1500.; CURVATURE_TARGET = 2.
CURVATURE_SEGMENTS = 256  # Resolve curvature independently of field quadrature.
IOTA_APPROX_TARGET = 0.41
IOTA_MODES = 6           # Resolve the straight-field-angle correction.

# B) x_scale  → controls how easily variables move
SCALE_FIELD = 2.18
SCALE_SURFACE = 1.

# Numerical method parameters --------------------------------------------------------
MAXITER=80


# ====================================================================================
# ====================================================================================
""" Initializing coils, field and surface """
# ====================================================================================
# ====================================================================================

# ------------------------------------------------------------------------------------
# Define the initial surface, coils, and their corresponding field.

if LOAD_RESULT:
    init_coils = Coils.from_json(os.path.join(output_filepath, "init_coils_coils_surface.json"))
    surface_file = os.path.join(output_filepath, "input.initial_surface")
elif RESTART_FROM:
    init_coils = Coils.from_json(os.path.join(restart_filepath, "opt_coils_coils_surface.json"))
    surface_file = os.path.join(restart_filepath, "input.coils_surface")
else:
    init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
    init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
    surface_file = vmec_input
init_coils.n_segments = N_SEGMENTS
if not LOAD_RESULT:
    init_coils.curves.order = FOURIER_ORDER
surface_init = SurfaceRZFourier.from_input_file(surface_file, ntheta=ntheta, nphi=nphi, close=False,
                                               range_torus='half period', mpol=SURFACE_MPOL, ntor=SURFACE_NTOR)
if surface_init.nfp != init_coils.nfp:
    raise ValueError("Boundary and coils must have the same NFP")


field_init = BiotSavart(init_coils)

# Report where the initial coils and surface come from.
if LOAD_RESULT:
    print(f"\nLOADED RESULT: coils and surface from '{os.path.basename(output_filepath)}'\n")
elif RESTART_FROM:
    print(f"\nRESTART: initial coils and surface from '{os.path.basename(restart_filepath)}', "
          f"surface resolution raised to mpol = {SURFACE_MPOL}, ntor = {SURFACE_NTOR}\n")
else:
    print(f"\nNEW RUN: initial surface from '{os.path.basename(vmec_input)}' and circular coils\n")

# Retain the original section-area target when increasing boundary resolution.
surface_reference = SurfaceRZFourier.from_input_file(vmec_input, ntheta=ntheta, nphi=nphi,
                                                    close=False, range_torus='half period')
CROSS_SECTIONAL_AREA_TARGET = float(jnp.mean(surface_reference.area_section_by_phi()))


# ====================================================================================
# ====================================================================================
""" Defining custom losses """
# ====================================================================================
# ====================================================================================

L_normal_field = custom_loss( loss_BdotN_mean_squared, "field", "surface" )
L_length_max = custom_loss( loss_coil_length_max , "field", max_coil_length=LENGTH_TARGET, squared=True )
L_curvature = custom_loss( loss_coil_curvature_from_field , "field" , max_coil_curvature=CURVATURE_TARGET, n_segments=CURVATURE_SEGMENTS )
L_quasi_symmetry = custom_loss(loss_quasi_symmetry, "field", "surface")
L_cross_sectional_area = custom_loss( loss_cross_sectional_area_mean, "surface", target_area=CROSS_SECTIONAL_AREA_TARGET )
L_iota_approx = custom_loss( loss_iota_approx, "field", "surface", iota_target=IOTA_APPROX_TARGET, lambda_mpol=IOTA_MODES, lambda_ntor=IOTA_MODES )


# ====================================================================================
# ====================================================================================
""" Defining total loss + setting dependencies """
# ====================================================================================
# ====================================================================================

L_total = (CONSTRAINT_WEIGHT * L_normal_field
           + LENGTH_WEIGHT * L_length_max
           + CURVATURE_WEIGHT * L_curvature
           + CONSTRAINT_WEIGHT * L_cross_sectional_area
           + CONSTRAINT_WEIGHT * L_iota_approx
           + QS_WEIGHT * L_quasi_symmetry)

# The dependencies of the total loss are set to the field and surface. Both will be modified during the optimization.
L_total.dependencies = { "field": field_init, "surface": surface_init }


# We assign different scaling factors to the field and surface dofs to balance their characteristic step sizes.
NN_field_dofs = L_total.dependencies["field"].dofs.size
NN_surface_dofs = L_total.dependencies["surface"].dofs.size
print(f"Surface dofs: {NN_surface_dofs}")


x_scale_optimization = jnp.concatenate([
    SCALE_FIELD   * jnp.ones(NN_field_dofs),
    SCALE_SURFACE * jnp.ones(NN_surface_dofs),
])

# ====================================================================================
# ====================================================================================
""" Optimizing the total loss """
# ====================================================================================
# ====================================================================================

t_start = time()


if LOAD_RESULT:
    from types import SimpleNamespace
    x_opt = jnp.load(os.path.join( output_filepath , "res_x.npy"))
    
    # The saved dofs only fit the problem they were optimized in (same coils and surface resolution).
    if x_opt.size != L_total.starting_dofs.size:
        raise ValueError(f"res_x.npy has {x_opt.size} dofs, but this setup has {L_total.starting_dofs.size}. "
                         f"Use the coil and surface settings in {os.path.join(output_filepath, 'essos_surf_coils.in')}.")
    
    res = SimpleNamespace(x=x_opt, nfev=0)
else:
    value_and_grad = jax.jit(jax.value_and_grad(L_total))
    def scaled_value_and_grad(x):
        value, gradient = value_and_grad(jnp.asarray(x) * x_scale_optimization)
        return float(value), np.asarray(gradient * x_scale_optimization)
    x_initial = np.asarray(L_total.starting_dofs / x_scale_optimization)
    bounds = [(None, None)] * x_initial.size
    # R00 is the first surface coefficient; keep the major radius fixed.
    bounds[NN_field_dofs] = (x_initial[NN_field_dofs], x_initial[NN_field_dofs])
    res = minimize(scaled_value_and_grad, x_initial,
                   jac=True, method="L-BFGS-B", bounds=bounds,
                   callback=lambda intermediate_result: print(f"Cost: {intermediate_result.fun:.6e}"),
                   options={'maxiter': MAXITER, 'maxcor': 128, 'maxls': 40, 'ftol': 1e-12, 'gtol': 1e-8})
    res.x = np.asarray(res.x * x_scale_optimization)
    jnp.save(os.path.join(output_filepath, "res_x.npy"), res.x)

t_end = time()


# ====================================================================================
# ====================================================================================
""" Extracting the different fields from the results array res """
# ====================================================================================
# ====================================================================================
# The optimized field and surface are extracted from the results of the optimization.
opt_dict = L_total.dofs_to_pytree(res.x)

field_opt = opt_dict["field"]
surface_opt = opt_dict["surface"]

# Coils geometry is extracted from the optimized field.
opt_coils = field_opt.coils


# ====================================================================================
# ====================================================================================
""" Printing results """
# ====================================================================================
# ====================================================================================

print("\n===========================================================================")
print("===========================================================================")
print("GENERAL RESULTS:")
print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", L_total(L_total.starting_dofs))    
print("Loss after optimization:", L_total(res.x))
print("Total function evaluations:", res.nfev)
print("Optimizer status:", getattr(res, 'message', 'loaded result'))

print("\nLOSSES (initial, optimized; raw and weighted):")
for name, weight, loss in (
    ("Normal field", CONSTRAINT_WEIGHT, L_normal_field),
    ("Quasisymmetry", QS_WEIGHT, L_quasi_symmetry),
    ("Iota", CONSTRAINT_WEIGHT, L_iota_approx),
    ("Section area", CONSTRAINT_WEIGHT, L_cross_sectional_area),
    ("Coil length", LENGTH_WEIGHT, L_length_max),
    ("Coil curvature", CURVATURE_WEIGHT, L_curvature),
):
    initial = float(loss.call_pytree({"field": field_init, "surface": surface_init}))
    optimized = float(loss.call_pytree(opt_dict))
    print(f"{name}: raw {initial:.6e}, {optimized:.6e}; "
          f"weighted {weight * initial:.6e}, {weight * optimized:.6e}")

print("\n===========================================================================")
print("DIAGNOSTICS:")
print("\n---------------------------------------------------------------------------")
print("Normal-field:")
B_dot_n_over_B_init = BdotN_over_B( surface_init , field_init )
B_dot_n_over_B_opt = BdotN_over_B( surface_opt , field_opt )
print("mean abs residual (initial):", jnp.mean(jnp.abs(B_dot_n_over_B_init)) )
print("mean abs residual (optimized):",jnp.mean(jnp.abs(B_dot_n_over_B_opt)) )
print("max abs residual (initial):",jnp.max(jnp.abs(B_dot_n_over_B_init)) )
print("max abs residual (optimized):",jnp.max(jnp.abs(B_dot_n_over_B_opt)) )
print("\n---------------------------------------------------------------------------")
print("Quasi-symmetry residuals and losses:")
QS_residual_xyz_init = quasi_symmetry_residual_on_surface(field_init, surface_init)
QS_residual_xyz_opt = quasi_symmetry_residual_on_surface(field_opt, surface_opt)
print("mean abs residual (initial):", jnp.mean(jnp.abs(QS_residual_xyz_init)))
print("mean abs residual (optimized):", jnp.mean(jnp.abs(QS_residual_xyz_opt)))
print("max abs residual (initial):", jnp.max(jnp.abs(QS_residual_xyz_init)))
print("max abs residual (optimized):", jnp.max(jnp.abs(QS_residual_xyz_opt)))
print("\n---------------------------------------------------------------------------")
print("Rotational transform:")
print("iota (target):", IOTA_APPROX_TARGET)
print("iota (initial):", iota_approx_on_surface(surface_init, field_init, lambda_mpol=IOTA_MODES, lambda_ntor=IOTA_MODES))
fit = iota_approx_on_surface(surface_opt, field_opt, lambda_mpol=IOTA_MODES, lambda_ntor=IOTA_MODES, return_diagnostics=True)
print("iota (optimized):", fit["iota"])
print("Straight-angle fit RMS and minimum Jacobian:", fit["rms_residual"], fit["minimum_angle_jacobian"])
print("RMS B.n/B:", jnp.sqrt(jnp.mean(B_dot_n_over_B_opt**2)))

print("\n---------------------------------------------------------------------------")
print("Nonaxisymmetric coefficient amplitude:")
print("3D coefficient amplitude (initial):", surface_init.non_axisymmetric_amplitude())
print("3D coefficient amplitude (optimized):", surface_opt.non_axisymmetric_amplitude())

print("\n---------------------------------------------------------------------------")
print("Mean cross-sectional area:")
cross_sectional_area_initial = surface_init.area_section_by_phi()
cross_sectional_area_optimized = surface_opt.area_section_by_phi()
relative_cross_sectional_area_change = ( cross_sectional_area_optimized - cross_sectional_area_initial ) / cross_sectional_area_initial

print("Relative area change (%):", 100 * relative_cross_sectional_area_change)

print("\n---------------------------------------------------------------------------")
print("Cross-sectional curvature:")

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


print("\n---------------------------------------------------------------------------")
print("Norm of the poloidal-parametrization derivative:")

qq_init = jnp.linalg.norm(surface_init.gammadash_theta, axis=2)
qq_opt = jnp.linalg.norm(surface_opt.gammadash_theta, axis=2)

print("Minimum ||dc_phi/dtheta|| (initial):", jnp.min(qq_init))
print("Minimum ||dc_phi/dtheta|| (optimized):", jnp.min(qq_opt))
print("||dc_phi/dtheta|| at maximum optimized curvature:", qq_opt[index_phi_max_opt, index_theta_max_opt])


print("\n---------------------------------------------------------------------------")
print("Coils displacement:")
coil_displacement = jnp.linalg.norm(opt_coils.gamma - init_coils.gamma, axis=-1)
print("Mean coil-point displacement:", jnp.mean(coil_displacement))
print("Maximum coil-point displacement:", jnp.max(coil_displacement))

print("\n---------------------------------------------------------------------------")
print("Coil-length residuals and losses:")
coil_length_residual_init = jnp.maximum(0, field_init.coils.length - LENGTH_TARGET)
coil_length_residual_opt = jnp.maximum(0, field_opt.coils.length - LENGTH_TARGET)
print("Coil lengths (initial):", field_init.coils.length)
print("Coil lengths (optimized):", field_opt.coils.length)
print("mean excess length (initial):", jnp.mean(coil_length_residual_init))
print("mean excess length (optimized):", jnp.mean(coil_length_residual_opt))
print("max excess length (initial):", jnp.max(coil_length_residual_init))
print("max excess length (optimized):", jnp.max(coil_length_residual_opt))

print("\n---------------------------------------------------------------------------")
print("Coil-curvature residuals and losses:")
curves_init, curves_opt = init_coils.curves.copy(), opt_coils.curves.copy()
curves_init.n_segments = curves_opt.n_segments = CURVATURE_SEGMENTS
coil_curvature_residual_init = jnp.maximum(0, curves_init.curvature - CURVATURE_TARGET)
coil_curvature_residual_opt = jnp.maximum(0, curves_opt.curvature - CURVATURE_TARGET)
print("mean curvature (initial):", jnp.mean(curves_init.curvature))
print("mean curvature (optimized):", jnp.mean(curves_opt.curvature))
print("max curvature (initial):", jnp.max(curves_init.curvature))
print("max curvature (optimized):", jnp.max(curves_opt.curvature))
print("mean excess curvature (initial):", jnp.mean(coil_curvature_residual_init))
print("mean excess curvature (optimized):", jnp.mean(coil_curvature_residual_opt))
print("max excess curvature (initial):", jnp.max(coil_curvature_residual_init))
print("max excess curvature (optimized):", jnp.max(coil_curvature_residual_opt))


print("\n---------------------------------------------------------------------------")
print("Coil currents:")
print("Base-coil currents (initial):", init_coils.dofs_currents_raw)
print("Base-coil currents (optimized):", opt_coils.dofs_currents_raw)


# ====================================================================================
# ====================================================================================
""" Record of the run parameters """
# The input parameters are grouped as in the INPUT DATA section and written as "NAME = value" lines,
# so the file can later be read back to set up a run.
RUN_PARAMETERS = {
    "Run control":          ["EXPORT", "RUN_VMEX", "VMEC_MPOL", "VMEC_NTOR", "LOAD_RESULT", "MAXITER"],
    "Continuation":         ["RESTART_FROM", "restart_filepath", "SURFACE_MPOL", "SURFACE_NTOR"],
    "Initial surface":      ["ntheta", "nphi"],
    "Coils":                ["N_COILS", "FOURIER_ORDER", "LARGE_R", "SMALL_R", "NFP", "N_SEGMENTS",
                             "STELLSYM", "COIL_CURRENT"],
    "Loss weights":         ["CONSTRAINT_WEIGHT", "QS_WEIGHT", "LENGTH_WEIGHT", "CURVATURE_WEIGHT"],
    "Coil limits":          ["LENGTH_TARGET", "CURVATURE_TARGET", "CURVATURE_SEGMENTS"],
    "Rotational transform": ["IOTA_APPROX_TARGET", "IOTA_MODES"],
    "x_scale":              ["SCALE_FIELD", "SCALE_SURFACE"],
}

if not LOAD_RESULT:
    with open(os.path.join(output_filepath, "essos_surf_coils.in"), "w") as f:
        f.write(f"# ESSOS single-stage run parameters ({datetime.datetime.now():%Y-%m-%d %H:%M})\n")
        f.write(f"VMEC_INPUT = {os.path.basename(vmec_input)!r}\n")
        for section, names in RUN_PARAMETERS.items():
            f.write(f"\n# {section}\n")
            for name in names:
                f.write(f"{name} = {globals()[name]!r}\n")

""" Plotting the initial and optimized coils """
# ====================================================================================
# ====================================================================================

essos_plots_filepath = os.path.join(output_filepath, "ESSOS_plots")
os.makedirs(essos_plots_filepath, exist_ok=True)

surface_plots = [copy(surface_init), copy(surface_opt)]
for surface_plot in surface_plots:
    surface_plot.range_torus = "full torus"
    surface_plot.reset_cache()

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(121, projection='3d')
init_coils.plot(ax=ax1, show=False)
surface_plots[0].plot(ax=ax1, show=False)

ax2 = fig.add_subplot(122, projection='3d')
opt_coils.plot(ax=ax2, show=False)
surface_plots[1].plot(ax=ax2, show=False)

plt.tight_layout()
plt.savefig(os.path.join(essos_plots_filepath, "coils_surface_init_opt.png"), dpi=200)
plt.show()


""" Exporting results """

if EXPORT:

    """ Save the coils to a json file """
    if not LOAD_RESULT:
        init_coils.to_json(os.path.join(output_filepath, "init_coils_coils_surface.json"))
        surface_init.to_vmec(os.path.join(output_filepath, "input.initial_surface"))
        opt_coils.to_json(os.path.join(output_filepath, "opt_coils_coils_surface.json"))

    """ Save results in vtk format to analyze in Paraview """
    surface_plots[0].to_vtk(os.path.join(output_filepath, "init_surface_coils_surface"), field=field_init)
    surface_plots[1].to_vtk(os.path.join(output_filepath, "opt_surface_coils_surface"), field=field_opt)
    init_coils.to_vtk(os.path.join(output_filepath, "init_coils_coils_surface"))
    opt_coils.to_vtk(os.path.join(output_filepath, "opt_coils_coils_surface"))

    """ Save the optimized boundary as a VMEC input file for VMEX """
    # Match the vacuum equilibrium flux to the optimized coil field:
    # NCURR = 1 holds the toroidal current at zero, so iota is computed (default 0 prescribes iota = 0).
    vmec_input_file = os.path.join(output_filepath, "input.coils_surface")
    surface_opt.to_vmec(vmec_input_file, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
    flux_surface = SurfaceRZFourier.from_input_file(vmec_input_file, ntheta=128, nphi=64,
                                                   close=False, range_torus="full torus")
    signed_area = jnp.sum(jnp.linalg.norm(flux_surface.gamma[..., :2], axis=-1)
                          * flux_surface.gammadash_theta[..., 2], axis=1)
    fluxes = jnp.array([-jnp.sign(signed_area[k]) * toroidal_flux(flux_surface, field_opt, k)
                       for k in range(flux_surface.nphi)])
    phiedge = float(jnp.mean(fluxes))
    if not np.isfinite(phiedge) or phiedge == 0 or bool(jnp.any(signed_area == 0)):
        raise ValueError("VMEX requires finite nonzero toroidal flux and valid boundary sections")
    print("Toroidal flux (Wb), relative spread:", phiedge, float(jnp.ptp(fluxes) / abs(phiedge)))
    vmec_inp = replace(vj.VmecInput.from_file(vmec_input_file), ncurr=1, phiedge=phiedge)
    vmec_inp.to_indata(vmec_input_file)

    """ Fixed-boundary VMEX equilibrium of the optimized surface """
    if RUN_VMEX:
        # All VMEX results (wout, boozmn and figures) go to their own subfolder inside the run folder.
        vmex_output_filepath = os.path.join(output_filepath, "vmex_output")
        os.makedirs(vmex_output_filepath, exist_ok=True)

        vmec_result = vj.solve_multigrid(vmec_inp, verbose=False)
        if not bool(vmec_result.converged):
            raise RuntimeError("VMEX did not converge; increase solver resolution or iteration limit")

        # The equilibrium is saved as a wout file, and VMEX's diagnostic figures are written next to it.
        wout = vj.wout_from_result(vmec_inp, vmec_result)
        wout_file = vj.write_wout(os.path.join(vmex_output_filepath, "wout_coils_surface.nc"), wout)
        vj.plot_wout(wout_file, vmex_output_filepath)

        # The Boozer transform shows how quasisymmetric |B| is.
        boozmn_file = vj.run_booz_xform(wout_file, outdir=vmex_output_filepath)
        vj.plot_boozmn(boozmn_file, vmex_output_filepath)

        print("\nVMEX (fixed boundary):")
        print("  converged:", bool(vmec_result.converged), "after", int(vmec_result.iterations), "iterations")
        print("  iota (axis, edge):", float(wout.iotaf[0]), float(wout.iotaf[-1]))
        print("  aspect ratio:", float(wout.aspect))

        # Quasi-axisymmetry check from the Boozer spectrum: every mode with n != 0 breaks QA.
        # The breaking is measured as the norm of those modes relative to the average field B00, on three surfaces.
        with netCDF4.Dataset(boozmn_file) as booz:
            bmnc = np.asarray(booz["bmnc_b"][:]); xm_b = np.asarray(booz["ixm_b"][:]); xn_b = np.asarray(booz["ixn_b"][:])
            s_b = (np.asarray(booz["jlist"][:]) - 1.5) / (int(wout.ns) - 1)
        B00 = bmnc[:, (xm_b == 0) & (xn_b == 0)][:, 0]
        symmetry_breaking = np.sqrt(np.sum(bmnc[:, xn_b != 0]**2, axis=1)) / np.abs(B00)
        print("  QA symmetry breaking (n != 0 modes / B00):")
        for s_index in [bmnc.shape[0] // 4, bmnc.shape[0] // 2, bmnc.shape[0] - 1]:
            print(f"    s = {s_b[s_index]:.2f}: {100 * symmetry_breaking[s_index]:.2f} %")
