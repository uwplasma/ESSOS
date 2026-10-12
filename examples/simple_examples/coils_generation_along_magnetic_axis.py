# coils_generation_along_magnetic_axis.py
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
from essos.fields import Vmec
from essos.plot import plot_coil_currents_3d
from essos.coils import Coils, CreateCurvesAlongAxis

# Plotting libraries ----------------------------------------
import matplotlib.pyplot as plt

# Other libraries ------------------------------------------
import os

# ============================================================================================
# INPUT DATA =================================================================================
# ============================================================================================

# -----------------------------------------------------------------------------
# General input
# -----------------------------------------------------------------------------

# Discretization
ntheta = 50
nphi = 64
number_of_coils = 2 # base coils per half field period. Total coils = 2 * nfp * number_of_coils

# Current running through the coils
Ifactor = 1.e7 

# wout_file_name = "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
wout_file_name = "wout_QH_simple_scaled.nc"
# wout_file_name = "wout_n3are_R7.75B5.7.nc"

# -----------------------------------------------------------------------------
# VMEC file input
# -----------------------------------------------------------------------------
wout_file = os.path.join( os.path.dirname(__file__) , ".." , "input_files", wout_file_name )
rminor = 2.0 # only valid for wout_LandremanPaul2021_QA_reactorScale_low

# -----------------------------------------------------------------------------
# PLOTTING input
# -----------------------------------------------------------------------------
output_folder = os.path.join( os.path.dirname(__file__) , "output_coils_generation_along_magnetic_axis" )


# ============================================================================================
# ============================================================================================
# MAIN =======================================================================================
# ============================================================================================
# ============================================================================================

os.makedirs(output_folder, exist_ok=True)

# ============================================================================================
# The surface is defined by the Fourier coefficients of R and Z as functions of theta and phi.
# ============================================================================================

# Equilibrium is loaded from a VMEC output file.
vmec = Vmec(wout_file, ntheta=ntheta, nphi=nphi, range_torus="full torus")
surface = vmec.surface # We extract the outermost surface from the VMEC equilibrium

# ============================================================================================
# Calculus of the magnetic axis
# ============================================================================================

maxis_xyz = vmec.maxis_xyz

# ============================================================================================
# Coil definitions: circular coils centered on the magnetic axis, normal = axis tangent
# ============================================================================================

rminor_coils = 2.3 * rminor

# Only the base coils are created; Curves replicates them using the VMEC symmetries (nfp, stellsym).
coil_curves = CreateCurvesAlongAxis(maxis_xyz, n_curves=number_of_coils, r=rminor_coils,
                                    n_segments=80, nfp=vmec.nfp, stellsym=True)
coils = Coils(curves=coil_curves, currents=-Ifactor * jnp.ones(number_of_coils))

# ============================================================================================
# ============================================================================================
# PLOTTING SECTION ===========================================================================
# ============================================================================================
# ============================================================================================

# ========================================
# Plotting surface + coils + magnetic axis
# ========================================

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot the magnetic axis
ax.plot( maxis_xyz[:, 0], maxis_xyz[:, 1], maxis_xyz[:, 2], color="black", linewidth=2.5, label="magnetic axis" )

# Plotting the surface
surface.plot(ax=ax, show=False, axis_equal=True, alpha=0.25)

# Plotting the coils and the current direction on each coil. The current direction follows the coil tangent. If the current is negative, we reverse that tangent.
coils.plot(ax=ax, show=False, close=False, color="brown", linewidth=2)
plot_coil_currents_3d(coils, ax, n_arrows=2, length=3.0, label="current direction")

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Surface and coils")

fig.savefig( os.path.join(output_folder, "01_surface_coils_axis.png"), dpi=300 , bbox_inches="tight" )
plt.show()
plt.close(fig)
