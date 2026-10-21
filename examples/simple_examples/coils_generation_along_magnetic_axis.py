# coils_generation_along_magnetic_axis.py
# Questions to: gonzalo.ftd@wisc.edu, gftd13@gmail.com
# ============================================================================================
# Importing libraries
# ============================================================================================

# JAX library ------------------------------------------------ 
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ESSOS libraries --------------------------------------------
from essos.fields import Vmec
from essos.plot import plot_coil_currents_3d
from essos.coils import Curves, Coils, circular_coil_dofs, CreateCurvesAlongAxis

# Plotting libraries ----------------------------------------
import matplotlib.pyplot as plt

# Other libraries ------------------------------------------
import os
import numpy as np

# ============================================================================================
# INPUT DATA =================================================================================
# ============================================================================================

# General input ------------------------------------------------------------------------------
# Output folder input
output_folder = os.path.join( os.path.dirname(__file__) , "output_coils_generation_along_magnetic_axis" )

# Discretization 
ntheta = 50
nphi = 64
number_of_coils = 2 # base coils per half field period. Total coils = 2 * nfp * number_of_coils

# Current running through the coils
Ifactor = 1.e7 

# Surface example
# wout_file_name = "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
wout_file_name = "wout_QH_simple_scaled.nc"
# wout_file_name = "wout_n3are_R7.75B5.7.nc"

# Estimated minor radius
rminor = 2.0

# Simple circular coils (example of circular_coil_dofs). One row per coil --------------------
# Center of each coil [x, y, z]
simple_coils_centers  = jnp.array([[ 10.0,   0.0,   0.0],
                                   [  0.0,  10.0,   0.0],
                                   [-10.0,   0.0,   0.0],
                                   [  0.0, -10.0,   0.0],
                                   [  0.0,   0.0,  10.0],
                                   [  0.0,   0.0, -10.0]])
                                                                      
                                                                      
# Normal of each coil plane [nx, ny, nz] (here the toroidal direction phi_hat). It does not need to be unitary.
simple_coils_normals  = jnp.array([[ 0.0,  1.0,  0.0],
                                   [-1.0,  0.0,  0.0],
                                   [ 0.0, -1.0,  0.0],
                                   [ 1.0,  0.0,  0.0],
                                   [ 0.0,  0.0,  1.0],
                                   [ 0.0,  0.0, -1.0]])
# Radius of each coil
simple_coils_radii    = jnp.array([2.0, 4.0, 2.0, 4.0, 8.0, 8.0])
# Current of each coil (sign gives the direction: positive -> B along +normal at the center)
simple_coils_currents = -Ifactor * jnp.ones( len(simple_coils_radii))

# ============================================================================================
# ============================================================================================
# MAIN =======================================================================================
# ============================================================================================
# ============================================================================================

# ============================================================================================
# Folder managing
wout_file = os.path.join( os.path.dirname(__file__) , ".." , "input_files", wout_file_name )
os.makedirs(output_folder, exist_ok=True)

# ============================================================================================
# The surface is obtained by loading a VMEC output file.
vmec = Vmec(wout_file, ntheta=ntheta, nphi=nphi, range_torus="full torus")
surface = vmec.surface # We extract the outermost surface from the VMEC equilibrium

# ============================================================================================
# Calculus of the magnetic axis
maxis_xyz = vmec.maxis_xyz

# ============================================================================================
# Coil definitions: circular coils centered on the magnetic axis, normal = axis tangent
rminor_coils = 2.3 * rminor

# ============================================================================================
# Coils created along the magnetic axis in one instruction
# Only the base coils are created; Curves replicates them using the VMEC symmetries (nfp, stellsym).
coil_curves = CreateCurvesAlongAxis(maxis_xyz, n_curves=number_of_coils, r=rminor_coils,
                                    n_segments=80, nfp=vmec.nfp, stellsym=True)
coils = Coils(curves=coil_curves, currents=-Ifactor * jnp.ones(number_of_coils))

# ============================================================================================
# Simple circular coils: one coil per row of the input arrays (example of circular_coil_dofs)
simple_coil_dofs = jax.vmap(circular_coil_dofs)(simple_coils_centers, simple_coils_radii, simple_coils_normals)
simple_coil_curves = Curves(simple_coil_dofs, n_segments=80, nfp=1, stellsym=False)
simple_coils = Coils(curves=simple_coil_curves, currents=simple_coils_currents)

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

# ========================================
# Plotting surface + simple circular coils
# ========================================

fig2 = plt.figure(figsize=(7, 6))
ax2 = fig2.add_subplot(111, projection="3d")

# Plotting the simple coils and the current direction on each coil
simple_coils.plot(ax=ax2, show=False, close=False, color="brown", linewidth=2)
plot_coil_currents_3d(simple_coils, ax2, n_arrows=2, length=3.0, label="current direction")

ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
ax2.set_title("Surface and simple circular coils")

fig2.savefig( os.path.join(output_folder, "02_surface_simple_coils.png"), dpi=300 , bbox_inches="tight" )
plt.show()
plt.close(fig2)

# ============================================================================================
# ============================================================================================
# VTK EXPORT (to analyze in Paraview) ========================================================
# ============================================================================================
# ============================================================================================

""" Save results in vtk format to analyze in Paraview """
surface.to_vtk(os.path.join(output_folder, "surface"))                 # surface.vts (includes |B| from VMEC)
coils.to_vtk(os.path.join(output_folder, "coils_along_axis"))          # coils_along_axis.vtu
simple_coils.to_vtk(os.path.join(output_folder, "simple_coils"))       # simple_coils.vtu

# Magnetic axis as a closed polyline (first point repeated at the end)
from pyevtk.hl import polyLinesToVTK
maxis_closed = np.concatenate([np.asarray(maxis_xyz), np.asarray(maxis_xyz[:1])], axis=0)
polyLinesToVTK(os.path.join(output_folder, "magnetic_axis"),
               maxis_closed[:, 0].copy(), maxis_closed[:, 1].copy(), maxis_closed[:, 2].copy(),
               pointsPerLine=np.array([maxis_closed.shape[0]]))  # magnetic_axis.vtu
