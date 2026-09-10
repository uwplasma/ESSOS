# QS_illustrative_test.py
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
from essos.objective_functions import QS_check_on_surface
from essos.plot import fix_matplotlib_3d
from essos.coils import Curves, Coils
from essos.surfaces import ( SurfaceRZFourier , BdotN_over_B )

# Plotting libraries ----------------------------------------
import matplotlib.pyplot as plt

# from dataclasses import dataclass

# Other libraries ------------------------------------------
import os
import sys
import time

# ============================================================================================
# INPUT DATA =================================================================================
# ============================================================================================

# -----------------------------------------------------------------------------
# Flags for plotting and debugging
# -----------------------------------------------------------------------------
# We have two examples: one with a simple torus surface and another with a VMEC surface.
# flags: vmec, torus_simple
# flag_surface = "vmec"
flag_surface = "torus_simple"

# -----------------------------------------------------------------------------
# General input
# -----------------------------------------------------------------------------
# Current running through the coils
Ifactor = 1.e7 

# flag for the scale length definition
# flag_LB = 1 # L_B = |B| / ||grad(|B|)||
flag_LB = 2 # L_B = sqrt(2) * |B| / ||grad(B)||_F

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
step_sample = 5
output_folder = os.path.join( os.path.dirname(__file__) , "output_QS_test" )


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

    surface = SurfaceRZFourier(
        rc=rc,
        zs=zs,
        nfp=3,
        mpol=1,
        ntor=0,
        ntheta=50,
        nphi=60,
        close=True,
        range_torus="full torus",
    )

elif flag_surface == "vmec":

    # Equilibrium is loaded from a VMEC output file.
    # `Vmec` class reads the file and constructs the surface representation based on the Fourier coefficients.
    # Parameters `ntheta` and `nphi` specify the number of grid points
    # `range_torus` indicates that we want to consider the full torus for our calculations.
    vmec = Vmec(wout_file, ntheta=50, nphi=60, range_torus="full torus")

    # We extract the outermost surface from the VMEC equilibrium
    surface = vmec.surface

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
# Storing the surface points.
# ============================================================================================

# gamma has shape roughly like:(nphi, ntheta, 3). The last index 3 means:x y z
# gamma is a function that gives the 3D coordinates of the surface points for each (phi, theta) pair.
# The first index is phi, the second index is theta, and the last index is the 3D coordinates.
surf_pt_full = surface.gamma
# Converting the 2D grid of sampled points into one long list of 3D points.
# The -1 tells Python to infer the number of rows automatically. After this, each row is one point:[x, y, z]
surf_xyz_full = surf_pt_full.reshape(-1, 3)

# Unit normal vectors at the surface points. unitnormal has shape roughly like:(nphi, ntheta, 3).
# Unit-Normal vectors at the (phi, theta) grid points on the surface. The last index 3 gives the direction components.
unitnormal_pt_full = surface.unitnormal
unitnormal_xyz_full = unitnormal_pt_full.reshape(-1, 3)

# Normal vectors at the (phi, theta) grid points on the surface.
normal_pt_full = surface.normal
normal_xyz_full = normal_pt_full.reshape(-1, 3)


# ============================================================================================
# Test points on the surface
# ============================================================================================

# These are the same surface points where the arrows are sampled. ::step means “take every step-th value”.
surf_pt_sampled = surf_pt_full[::step_sample, ::step_sample, :]
surf_xyz_sampled = surf_pt_sampled.reshape(-1, 3)

# Same sampling for the unit normal vectors.
unitnormal_pt_sampled = unitnormal_pt_full[::step_sample, ::step_sample, :]
unitnormal_xyz_sampled = unitnormal_pt_sampled.reshape(-1, 3)


# ============================================================================================
# Coil definitions
# ============================================================================================
# Circular coils are defined by their Fourier coefficients.

number_of_coils = 4
order = 1  # (2*order + 1) = Fourier coefficients for each of the x, y, z coordinates

if flag_surface == "torus_simple":
    # Coils in the x-z plane: coil 0 and coil 1
    Rmajor01_coils = 1.0 * Rmajor
    rminor01_coils = 2.0 * rminor

    # Coils in the y-z plane: coil 2 and coil 3
    Rmajor02_coils = 1.0 * Rmajor
    rminor02_coils = 2.0 * rminor

    Icoils_direction = jnp.array([-1.0, 1.0, -1.0, 1.0])

elif flag_surface == "vmec":
    # Coils in the x-z plane: coil 0 and coil 1
    Rmajor01_coils = 1.0 * Rmajor
    rminor01_coils = 2.3 * rminor

    # Coils in the y-z plane: coil 2 and coil 3
    Rmajor02_coils = 0.7 * Rmajor
    rminor02_coils = 2.3 * rminor
    
    Icoils_direction = jnp.array([-1.0, 1.0, -1.0, 1.0])

else:
    raise ValueError(f"Unknown flag_surface: {flag_surface}")


Icoils = Ifactor * Icoils_direction

"""
# Attributes:
#         dofs (jnp.ndarray - shape (n_indcurves, 3, 2*order+1)): Fourier Coefficients of the independent curves
#         n_segments (int): Number of segments to discretize the curves
#         nfp (int): Number of field periods
#         stellsym (bool): Stellarator symmetry
#         order (int): Order of the Fourier series
#         curves jnp.ndarray - shape (n_indcurves*nfp*(1+stellsym), 3, 2*order+1)): Curves obtained by applying rotations and flipping corresponding to nfp fold rotational symmetry and optionally stellarator symmetry
#         gamma (jnp.array - shape (n_coils, n_segments, 3)): Discretized curves
#         gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Discretized curves derivatives
"""

# Shape = (number_of_coils, 3, 2*order + 1)
# Here order = 1, so the last dimension has size 3:
#   index 0 -> constant term
#   index 1 -> sine term
#   index 2 -> cosine term
# \[ # x(\theta) = a_0 + a_1 \sin(\theta) + a_2 \cos(\theta) \]
# \[ y(\theta) = b_0 + b_1 \sin(\theta) + b_2 \cos(\theta) \]
# \[ z(\theta) = c_0 + c_1 \sin(\theta) + c_2 \cos(\theta) \]

coil_dofs = jnp.zeros(( number_of_coils, 3 , 2*order+1 ))

# -----------------------------------------------------------------------------
# Coil 0: centered at (Rmajor, 0, 0), in the x-z plane
# -----------------------------------------------------------------------------
coil_dofs = coil_dofs.at[0, 0, 0].set(Rmajor01_coils)    # x constant
coil_dofs = coil_dofs.at[0, 0, 2].set(rminor01_coils)    # x cosine term
coil_dofs = coil_dofs.at[0, 2, 1].set(-rminor01_coils)   # z sine term

# -----------------------------------------------------------------------------
# Coil 1: centered at (-Rmajor, 0, 0), in the x-z plane
# -----------------------------------------------------------------------------
coil_dofs = coil_dofs.at[1, 0, 0].set(-Rmajor01_coils)   # x constant
coil_dofs = coil_dofs.at[1, 0, 2].set(rminor01_coils)    # x cosine term
coil_dofs = coil_dofs.at[1, 2, 1].set(-rminor01_coils)   # z sine term

# -----------------------------------------------------------------------------
# Coil 2: centered at (0, Rmajor, 0), in the y-z plane
# -----------------------------------------------------------------------------
coil_dofs = coil_dofs.at[2, 1, 0].set(Rmajor02_coils)    # y constant
coil_dofs = coil_dofs.at[2, 1, 2].set(rminor02_coils)    # y cosine term
coil_dofs = coil_dofs.at[2, 2, 1].set(-rminor02_coils)   # z sine term

# -----------------------------------------------------------------------------
# Coil 3: centered at (0, -Rmajor, 0), in the y-z plane
# -----------------------------------------------------------------------------
coil_dofs = coil_dofs.at[3, 1, 0].set(-Rmajor02_coils)   # y constant
coil_dofs = coil_dofs.at[3, 1, 2].set(rminor02_coils)    # y cosine term
coil_dofs = coil_dofs.at[3, 2, 1].set(-rminor02_coils)   # z sine term

coil_curves = Curves( coil_dofs , n_segments=80 , nfp=1 , stellsym=False )
coils = Coils( curves=coil_curves , currents=Icoils )

# ============================================================================================
# Magnetic field from the coils
# ============================================================================================
# Magnetic field from the coils using Biot-Savart law
BB_coils = BiotSavart(coils)

# -----------------------------------------------------------------------------
# Magnetic field data at every full surface grid point

# Normal magnetic field, normalized by |B|, on the full surface grid.
B_dot_n_pt_full = BdotN_over_B( surface, BB_coils )

# Other interesting data 
# B_vector_xyz_full = jax.vmap( BB_coils.B )( surf_xyz_full ) # Magnetic-field vector
# B_modulus_xyz_full = jax.vmap( BB_coils.AbsB )( surf_xyz_full ) # Magnetic-field modulus
# Bmod_grad_xyz_full = jax.vmap( BB_coils.dAbsB_by_dX )( surf_xyz_full ) # Gradient of |B|
# # Other usefull operations on the full surface grid
# B_dot_n_xyz_full = jnp.sum( B_vector_xyz_full * unitnormal_xyz_full, axis=1)

# ------------------------------------------------------------------------------
# Magnetic field data at every sampled surface grid point

B_vector_xyz_sampled = jax.vmap( BB_coils.B )( surf_xyz_sampled ) # Magnetic-field vector

# Other interesting data 
# B_modulus_xyz_sampled = jax.vmap( BB_coils.AbsB )( surf_xyz_sampled ) # Magnetic-field modulus
# Bmod_grad_xyz_sampled = jax.vmap( BB_coils.dAbsB_by_dX )( surf_xyz_sampled ) # Gradient of |B|
# # Other usefull operations on the sampled surface grid
# B_dot_n_xyz_sampled = jnp.sum( B_vector_xyz_sampled * unitnormal_xyz_sampled, axis=1)

# ============================================================================================
# Calculus of the scale length
# ============================================================================================

if flag_LB == 1: # L_B = |B| / ||grad(|B|)|| --------------------------------
    L_B_xyz_full = jax.vmap(BB_coils.L_gradB_type1)(surf_xyz_full)

elif flag_LB == 2: # L_B = sqrt(2) * |B| / ||grad(B)||_F --------------------
    L_B_xyz_full = jax.vmap( BB_coils.L_gradB_type2)(surf_xyz_full)

else:
    raise ValueError(f"Unknown flag_LB: {flag_LB}")

# Reshape the scale length back to the surface grid shape for plotting.
L_B_pt_full = L_B_xyz_full.reshape(surf_pt_full.shape[:2])

print("L_B minimum =", jnp.min(L_B_xyz_full))
print("L_B maximum =", jnp.max(L_B_xyz_full))
print("L_B average =", jnp.mean(L_B_xyz_full))


# ============================================================================================
# Residual of the Quasi-Symmetry condition
# ============================================================================================
# ( \nabla \psi \times \nabla |B| ) \cdot \nabla ( \mathbf{B} \cdot \nabla |B| )
# For optimization purposes, \nabla \psi is replaced by the unit normal vector to the surface. 
# (n × grad(B)) · grad(B · grad(B)) = 0?

# -----------------------------------------------------------------------------
# Full surface grid
QS_condition_xyz_full = QS_check_on_surface( BB_coils , surface )

print("QS_condition_xyz_full.shape =", QS_condition_xyz_full.shape)
print("QS_condition_xyz_full min =", jnp.min(QS_condition_xyz_full))
print("QS_condition_xyz_full max =", jnp.max(QS_condition_xyz_full))

# ==============================================================================
# Sampled surface grid

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
ax.plot( maxis_xyz[:, 0], maxis_xyz[:, 1], maxis_xyz[:, 2],
    color="black", linewidth=2.5,
    label="magnetic axis",
)

# Plotting the surface
surface.plot(ax=ax, show=False, axis_equal=True, alpha=0.25)

# Plotting the coils
coils.plot(ax=ax, show=False, close=False, color="brown", linewidth=2)

# Show the current direction on each coil. The current direction follows the coil tangent. If the current is negative, we reverse that tangent.

for coil_index, (gamma_curve, gamma_dash_curve) in enumerate(zip(coils.gamma, coils.gamma_dash)):
    current_sign = 1.0 if float(coils.currents[coil_index]) >= 0.0 else -1.0

    # Pick one representative point on the coil, here the midpoint
    midpoint_index = gamma_curve.shape[0] // 2
    point = gamma_curve[midpoint_index]

    # Tangent direction of the curve, corrected by the current sign
    tangent = current_sign * gamma_dash_curve[midpoint_index]

    # Normalize the arrow direction so the length is controlled by `length`
    tangent_norm = jnp.linalg.norm(tangent)
    if tangent_norm > 0:
        tangent = tangent / tangent_norm

    ax.quiver(
        point[0], point[1], point[2],
        tangent[0], tangent[1], tangent[2],
        length=4.,
        normalize=True,
        color="blue",
        arrow_length_ratio=0.25,
        linewidth=2,
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Surface and coils")


fig.savefig( os.path.join(output_folder, "01_surface_coils_axis.png"), dpi=300 , bbox_inches="tight" )
plt.show()
plt.close(fig)

# ========================================
# Plotting surface + coils + unit-normal vectors + B vectors
# ========================================

fig2 = plt.figure(figsize=(7, 6))
ax2 = fig2.add_subplot(111, projection="3d")

# Plot the surface again
surface.plot(ax=ax2, show=False, axis_equal=True, alpha=0.35)

# Plot the coils too
coils.plot(
    ax=ax2,
    show=False,
    close=False,
    color="brown",
    linewidth=2,
    label="coils",
)

ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

# surf_pt_sampled[..., 0] means: “take all the first two indices” and then take component 0 of the last axis
x = surf_pt_sampled[..., 0]
y = surf_pt_sampled[..., 1]
z = surf_pt_sampled[..., 2]

# Unit-normal vectors at the sampled points
u = (-unitnormal_pt_sampled)[..., 0]
v = (-unitnormal_pt_sampled)[..., 1]
w = (-unitnormal_pt_sampled)[..., 2]

# Magnetic-field vectors at the sampled points
B_vectors_pt_sampled = B_vector_xyz_sampled.reshape(surf_pt_sampled.shape)
bx = B_vectors_pt_sampled[..., 0]
by = B_vectors_pt_sampled[..., 1]
bz = B_vectors_pt_sampled[..., 2]

ax2.quiver(
    x, y, z,
    u, v, w,
    length=1.,
    normalize=True,
    color="darkgreen",
    label="unit normal",
)

ax2.quiver(
    x, y, z,
    bx, by, bz,
    length=1.,
    normalize=True,
    color="royalblue",
    label="B",
)

ax2.set_title("Surface with unit-normal vectors and B vectors")
ax2.legend()

fig2.savefig( os.path.join(output_folder, "02_surface_normals_B_vectors.png"), dpi=300, bbox_inches="tight" )
plt.show()
plt.close(fig2)


# ========================================
# Plotting the surface colored by B · n
# ========================================

fig3 = plt.figure(figsize=(7, 6))
ax3 = fig3.add_subplot(111, projection="3d")

# Coordinates of the full surface
x = surf_pt_full[..., 0]
y = surf_pt_full[..., 1]
z = surf_pt_full[..., 2]

# Symmetric color scale around zero
abs_max = jnp.max(jnp.abs(B_dot_n_pt_full))
norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

# Plot the smooth colored surface
ax3.plot_surface(
    x, y, z,
    facecolors=plt.cm.coolwarm(norm(B_dot_n_pt_full)),
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True,
    shade=False,
)

# Plot the coils too
coils.plot(
    ax=ax3,
    show=False,
    close=False,
    color="brown",
    linewidth=2,
    label="coils",
)

fix_matplotlib_3d(ax3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array(B_dot_n_pt_full)
fig3.colorbar(sm, ax=ax3, shrink=0.7, pad=0.1, label="B · n")

ax3.set_title("Surface colored by B · n")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")

fig3.savefig( os.path.join(output_folder, "03_B_dot_n.png") , dpi=300 , bbox_inches="tight" )
plt.show()
plt.close(fig3)

# ========================================
# Plotting the magnetic-gradient scale length L_B
# ========================================

x = surf_pt_full[..., 0]
y = surf_pt_full[..., 1]
z = surf_pt_full[..., 2]

fig_LB = plt.figure(figsize=(7, 6))
ax_LB = fig_LB.add_subplot(111, projection="3d")

# Color normalization based on the range of L_B
L_B_min = float(jnp.min(L_B_pt_full))
L_B_max = float(jnp.max(L_B_pt_full))
color_norm = plt.Normalize(vmin=L_B_min, vmax=L_B_max)

surface_colors = plt.cm.coolwarm( color_norm(L_B_pt_full) )

ax_LB.plot_surface(
    x,
    y,
    z,
    facecolors=surface_colors,
    linewidth=0,
    antialiased=True,
    alpha=0.9,
)

fix_matplotlib_3d(ax_LB)

# Colorbar
colorbar_map = plt.cm.ScalarMappable(norm=color_norm, cmap="coolwarm")
colorbar_map.set_array(L_B_pt_full)

fig_LB.colorbar(
    colorbar_map,
    ax=ax_LB,
    shrink=0.7,
    label=r"$L_B = |B| / ||\nabla |B|||$",
)

ax_LB.set_xlabel("x")
ax_LB.set_ylabel("y")
ax_LB.set_zlabel("z")
ax_LB.set_title("Magnetic-gradient scale length on the surface")

fig_LB.savefig( os.path.join(output_folder, "04_magnetic_scale_length.png") , dpi=300 , bbox_inches="tight" )
plt.show()
plt.close(fig_LB)



# ========================================
# Plotting the surface colored by QS_condition_xyz_full
# ========================================

fig4 = plt.figure(figsize=(7, 6))
ax4 = fig4.add_subplot(111, projection="3d")

# Reshape the scalar residual back to the full surface grid
QS_condition_pt_full = QS_condition_xyz_full.reshape(surf_pt_full.shape[:2])

# Coordinates of the full surface
x = surf_pt_full[..., 0]
y = surf_pt_full[..., 1]
z = surf_pt_full[..., 2]

# Symmetric color scale around zero
abs_max = jnp.max(jnp.abs(QS_condition_pt_full))
norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

# Plot the colored surface
ax4.plot_surface(
    x, y, z,
    facecolors=plt.cm.coolwarm(norm(QS_condition_pt_full)),
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True,
    shade=False,
)

# Plot the coils too
coils.plot(
    ax=ax4,
    show=False,
    close=False,
    color="brown",
    linewidth=2,
    label="coils",
)

fix_matplotlib_3d(ax4)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array(QS_condition_pt_full)
fig4.colorbar(sm, ax=ax4, shrink=0.7, pad=0.1, label="QS condition")

ax4.set_title("Surface colored by QS condition")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_zlabel("z")

fig4.savefig( os.path.join(output_folder, "05_QS_condition.png") , dpi=300 , bbox_inches="tight" )
plt.show()
plt.close(fig4)

