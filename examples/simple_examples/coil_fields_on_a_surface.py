# coil_fields_on_a_surface.py
# Questions to: gonzalo.ftd@wisc.edu, gftd13@gmail.com
# ============================================================================================
# Importing libraries
# ============================================================================================

# JAX library ------------------------------------------------ 
# Library for high-performance numerical computing and automatic differentiation.
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ESSOS libraries --------------------------------------------
from essos.fields import BiotSavart, Vmec
from essos.objective_functions import quasi_symmetry_residual_on_surface
from essos.plot import plot_scalar_map_2d, plot_scalar_on_surface_3d, plot_vector_field_on_surface_3d
from essos.coils import Coils, CreateCurvesAlongAxis
from essos.surfaces import ( SurfaceRZFourier , BdotN_over_B )

# Plotting libraries ----------------------------------------
import matplotlib.pyplot as plt

# from dataclasses import dataclass

# Other libraries ------------------------------------------
import os

# ============================================================================================
# INPUT DATA =================================================================================
# ============================================================================================

# -----------------------------------------------------------------------------
# Flags for plotting and debugging
# -----------------------------------------------------------------------------
# We have two examples: one with a simple torus surface and another with a VMEC surface.
# flags: vmec, torus_simple
flag_surface = "vmec"
# flag_surface = "torus_simple"

# -----------------------------------------------------------------------------
# General input
# -----------------------------------------------------------------------------

# Discretization
ntheta = 50
nphi = 64
number_of_coils = 2 # base coils per half field period. Total coils = 2 * nfp * number_of_coils

# Current running through the coils
Ifactor = 1.e7 

# -----------------------------------------------------------------------------
# Simple torus input
# -----------------------------------------------------------------------------
Rmajor = 10.0
rminor = 2.0

# -----------------------------------------------------------------------------
# VMEC file input
# -----------------------------------------------------------------------------
wout_file = os.path.join( os.path.dirname(__file__) , ".." , "input_files", "wout_LandremanPaul2021_QA_reactorScale_lowres.nc" )
Rmajor = 10.0 # only valid for wout_LandremanPaul2021_QA_reactorScale_low
rminor = 2.0 # only valid for wout_LandremanPaul2021_QA_reactorScale_low

# -----------------------------------------------------------------------------
# PLOTTING input
# -----------------------------------------------------------------------------
# step_sample chooses how many surface points you skip when sampling.
# step_sample = 4 means: take one point every 4 points in both directions (phi and theta).
step_sample = 4
output_folder = os.path.join( os.path.dirname(__file__) , "output_coil_fields_on_a_surface" )

# ============================================================================================
# ============================================================================================
# ============================================================================================
# MAIN =======================================================================================
# ============================================================================================
# ============================================================================================
# ============================================================================================

os.makedirs(output_folder, exist_ok=True)

# ============================================================================================
# The surface is defined by the Fourier coefficients of R and Z as functions of theta and phi.
# ============================================================================================
# Simple torus geometry
# R(theta) = R0 + a cos(theta)
# Z(theta) = a sin(theta)

if flag_surface == "torus_simple":

    rc = jnp.array([Rmajor, rminor])
    zs = jnp.array([0.0, rminor])

    surface = SurfaceRZFourier( rc=rc, zs=zs, nfp=3, mpol=1, ntor=0, ntheta=ntheta, nphi=nphi, close=True, range_torus="full torus" )

elif flag_surface == "vmec":

    # Equilibrium is loaded from a VMEC output file.
    vmec = Vmec(wout_file, ntheta=ntheta, nphi=nphi, range_torus="full torus")
    surface = vmec.surface # We extract the outermost surface from the VMEC equilibrium

else:
    raise ValueError(f"Unknown flag_surface: {flag_surface}") 

# ============================================================================================
# Calculus of the magnetic axis
# ============================================================================================

# This creates the toroidal-angle grid where we will evaluate the magnetic axis.
phi_R1 = jnp.linspace(0, 2 * jnp.pi , surface.nphi, endpoint=False)

if flag_surface == "torus_simple":

    RR_axis = jnp.full_like(phi_R1, Rmajor)
    ZZ_axis = jnp.zeros_like(phi_R1)
    maxis_xyz = jnp.stack( [ RR_axis * jnp.cos(phi_R1) , RR_axis * jnp.sin(phi_R1) , ZZ_axis ] , axis=1 )

elif flag_surface == "vmec":
    maxis_xyz = vmec.maxis_xyz

# ============================================================================================
# Storing the surface points and surface-related quantities.
# ============================================================================================

# gamma has shape roughly like:(nphi, ntheta, 3). The last index 3 means:x y z
# gamma is a function that gives the 3D coordinates of the surface points for each (phi, theta) pair.
surf_pt_full = surface.gamma
# Converting the 2D grid of sampled points into one long list of 3D points. After this, each row is one point:[x, y, z]
surf_xyz_full = surf_pt_full.reshape(-1, 3)

# Unit normal vectors at the surface points. unitnormal has shape roughly like:(nphi, ntheta, 3).
unitnormal_pt_full = surface.unitnormal

# ============================================================================================
# Coil definitions: circular coils centered on the magnetic axis, normal = axis tangent
# ============================================================================================

if flag_surface == "torus_simple":
    rminor_coils = 2.0 * rminor
elif flag_surface == "vmec":
    rminor_coils = 2.3 * rminor

# Only the base coils are created; Curves replicates them using the surface symmetries (nfp, stellsym).
coil_curves = CreateCurvesAlongAxis(maxis_xyz, n_curves=number_of_coils, r=rminor_coils,
                                    n_segments=80, nfp=surface.nfp, stellsym=True)
coils = Coils(curves=coil_curves, currents=-Ifactor * jnp.ones(number_of_coils))

centers = coil_curves.curves[:, :, 0]   # coil centers (all coils), used for phi_coils

# ============================================================================================
# Magnetic field from the coils
# ============================================================================================
# Magnetic field from the coils using Biot-Savart law
BB_coils = BiotSavart(coils)
B_vector_xyz_full = jax.vmap( BB_coils.B )( surf_xyz_full ) # Magnetic-field vector

# Normal magnetic field, normalized by |B|, on the full surface grid.
B_dot_n_pt_full = BdotN_over_B( surface, BB_coils )

# ============================================================================================
# Calculus of the scale length
# ============================================================================================
L_B_xyz_full = jax.vmap(BB_coils.L_gradB)(surf_xyz_full)

# Reshape the scale length back to the surface grid shape for plotting.
L_B_pt_full = L_B_xyz_full.reshape(surf_pt_full.shape[:2])

# ============================================================================================
# Residual of the Quasi-Symmetry condition
# ============================================================================================
# (n × grad(B)) · grad(B · grad(B)) = 0?
QS_condition_xyz_full = quasi_symmetry_residual_on_surface( BB_coils , surface )

# ============================================================================================
# ============================================================================================
# PLOTTING SECTION ===========================================================================
# ============================================================================================
# ============================================================================================

# ========================================
# Plotting surface + coils + unit-normal vectors + B vectors
# ========================================

fig2, ax2 = plot_vector_field_on_surface_3d(surface, -unitnormal_pt_full, coils=coils, step=step_sample,
                                            color="darkgreen", label="unit normal")
plot_vector_field_on_surface_3d(surface, B_vector_xyz_full, ax=ax2, step=step_sample, draw_surface=False,
                                color="royalblue", label="B", title="Surface with unit-normal vectors and B vectors")

fig2.savefig(os.path.join(output_folder, "01_surface_normals_B_vectors.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig2)

# ========================================
# Scalar fields on the surface: one 3D figure + one 2D (phi, theta) map each
# ========================================

# Toroidal position of each coil center (dashed lines on the 2D maps)
phi_coils = jnp.mod(jnp.arctan2(centers[:, 1], centers[:, 0]), 2 * jnp.pi)

# ----------------------------------------
# B · n / |B|
# ----------------------------------------
fig3, ax3 = plot_scalar_on_surface_3d(surface, B_dot_n_pt_full, coils=coils,
                                      label=r"$B\cdot n/|B|$", title=r"$B\cdot n/|B|$ on the surface", symmetric=True)
fig3.savefig(os.path.join(output_folder, "02_B_dot_n_3d.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig3)

fig3m, ax3m = plot_scalar_map_2d(surface, B_dot_n_pt_full,
                                 label=r"$B\cdot n/|B|$", title=r"$B\cdot n/|B|$ in $(\phi,\theta)$", symmetric=True, contours=10)
for phi_c in phi_coils:
    ax3m.axvline(float(phi_c), color="brown", linestyle="--", linewidth=1)
fig3m.savefig(os.path.join(output_folder, "02_B_dot_n_map.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig3m)

# ----------------------------------------
# Magnetic-gradient scale length L_B
# ----------------------------------------
fig_LB, ax_LB = plot_scalar_on_surface_3d(surface, L_B_pt_full, coils=coils,
                                          label=r"$L_{\nabla B} = \sqrt{2}\,|B| / \|\nabla \mathbf{B}\|_F$",
                                          title="Magnetic-gradient scale length on the surface")
fig_LB.savefig(os.path.join(output_folder, "03_magnetic_scale_length_3d.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig_LB)

fig_LBm, ax_LBm = plot_scalar_map_2d(surface, L_B_pt_full,
                                     label=r"$L_{\nabla B}$", title=r"Magnetic-gradient scale length in $(\phi,\theta)$",
                                     contours=10)
for phi_c in phi_coils:
    ax_LBm.axvline(float(phi_c), color="brown", linestyle="--", linewidth=1)
fig_LBm.savefig(os.path.join(output_folder, "03_magnetic_scale_length_map.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig_LBm)

# ----------------------------------------
# Quasi-symmetry residual
# ----------------------------------------
fig4, ax4 = plot_scalar_on_surface_3d(surface, QS_condition_xyz_full, coils=coils,
                                      label="QS residual", title="Quasi-symmetry residual on the surface", symmetric=True)
fig4.savefig(os.path.join(output_folder, "04_QS_condition_3d.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig4)

fig4m, ax4m = plot_scalar_map_2d(surface, QS_condition_xyz_full,
                                 label="QS residual", title=r"Quasi-symmetry residual in $(\phi,\theta)$", symmetric=True, contours=10)
for phi_c in phi_coils:
    ax4m.axvline(float(phi_c), color="brown", linestyle="--", linewidth=1)
fig4m.savefig(os.path.join(output_folder, "04_QS_condition_map.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig4m)
