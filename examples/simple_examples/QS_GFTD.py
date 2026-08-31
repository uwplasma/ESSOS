
# ============================================================================================
# Importing libraries
# ============================================================================================

# JAX library ------------------------------------------------ 
# Library for high-performance numerical computing and automatic differentiation.
import jax

jax.config.update("jax_enable_x64", True)
from jax import grad
import jax.numpy as jnp


# ESSOS libraries --------------------------------------------
from essos.fields import BiotSavart, Vmec
from essos.plot import fix_matplotlib_3d
from essos.coils import Curves, Coils
from essos.surfaces import SurfaceRZFourier

# Plotting libraries ----------------------------------------
import matplotlib.pyplot as plt

# from dataclasses import dataclass

# Other libraries ------------------------------------------
import os
import sys
import time

# ============================================================================================
# We create the class for storing magnetic field data.
# ============================================================================================

class MagneticFieldData:
    def __init__(self, points, Bvec, Bmod, Bmodgrad):
        self.points = points
        self.Bvec = Bvec
        self.Bmod = Bmod
        self.Bmodgrad = Bmodgrad


# ============================================================================================
# Flags for plotting and debugging
# ============================================================================================
# flags: vmec, torus_simple
flag_surface_case = "vmec"
# flag_surface_case = "torus_simple"

# ============================================================================================
# The surface is defined by the Fourier coefficients of R and Z as functions of theta and phi.
# ============================================================================================
# Simple torus geometry
# R(theta) = R0 + a cos(theta)
# Z(theta) = a sin(theta)


if flag_surface_case == "torus_simple":

    Rmajor = 10.0
    rminor = 2.0
    epsilon = 0.03


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

elif flag_surface_case == "vmec":

    Rmajor = 12.0
    rminor = 3.0
    epsilon = 0.03


    wout_file = os.path.join(
        os.path.dirname(__file__),"..",
        "input_files",
        "wout_LandremanPaul2021_QA_reactorScale_lowres.nc",
    )

    # Equilibrium is loaded from a VMEC output file.
    # `Vmec` class reads the file and constructs the surface representation based on the Fourier coefficients.
    # Parameters `ntheta` and `nphi` specify the number of grid points
    # `range_torus` indicates that we want to consider the full torus for our calculations.
    vmec = Vmec(wout_file, ntheta=50, nphi=60, range_torus="full torus")

    # We extract the outermost surface from the VMEC equilibrium
    surface = vmec.surface

    
else:
    raise ValueError(f"Unknown flag_surface_case: {flag_surface_case}") 

# ============================================================================================
# Calculus of the magnetic axis

# This creates the toroidal-angle grid where we will evaluate the magnetic axis.
# phi_R1 = jnp.linspace(0, 2 * jnp.pi / vmec.nfp, surface.nphi, endpoint=False)
phi_R1 = jnp.linspace(0, 2 * jnp.pi , surface.nphi, endpoint=False)


if flag_surface_case == "torus_simple":

    RR_axis = jnp.full_like(phi_R1, Rmajor)
    ZZ_axis = jnp.zeros_like(phi_R1)


elif flag_surface_case == "vmec":

    # print("rmnc at s=0:", vmec.rmnc[0, :])
    # print("zmns at s=0:", vmec.zmns[0, :])

    # vmec.xm contains the poloidal mode number m for every Fourier mode.
    # If m=0, the theta dependence dissapears.
    # \[ R_{\text{axis}}(\phi) =  \sum_n R_{0n}\cos(n\phi),\]
    # \[ Z_{\text{axis}}(\phi) = -\sum_n Z_{0n}\sin(n\phi).\]
    # Creating a mask that is True for modes with \(m=0\).
    m_eq_0_bool = (vmec.xm == 0)
    # Now we take the n's associated to a m=0
    xn_axis = vmec.xn[m_eq_0_bool]
    
    # Now we take the R_{mn} and Z_{mn} that multiplies cos/sin where m=0
    rmnc_axis = vmec.rmnc[0, m_eq_0_bool]
    zmns_axis = vmec.zmns[0, m_eq_0_bool]



    # This is a matrix formed by (xn_phi_R2)_{i,j} = xn_axis[i] * phi_R1[j]
    xn_phi_R2 = jnp.outer(xn_axis, phi_R1)

    # Finnally, we evaluate the Fourier series for R and Z at each phi_R1[j].
    RR_axis = jnp.sum(rmnc_axis[:, None] * jnp.cos(xn_phi_R2), axis=0)
    ZZ_axis = -jnp.sum(zmns_axis[:, None] * jnp.sin(xn_phi_R2), axis=0)
    
else:
    raise ValueError(f"Unknown flag_surface_case: {flag_surface_case}") 

axis_xyz = jnp.stack([
    RR_axis * jnp.cos(phi_R1),
    RR_axis * jnp.sin(phi_R1),
    ZZ_axis,
    ], axis=1)


print("axis_xyz.shape =", axis_xyz.shape)




# ============================================================================================
# Storing all the surface points. gamma has shape roughly like:(nphi, ntheta, 3). The last index 3 means:x y z
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
# Printing some information about the surface 
print("Volume of surf_pt_full:", surface.volume)
print("Area of surf_pt_full:", surface.area)

rminor_test = 2. * surface.volume / surface.area
print("rminor = 2*volume/surface:", rminor_test)


print("\nFirst few points along the first phi row:")
for k in range(5):
    print(f" (nphi,ntheta) = { (0, k) } --> (x,y,z) = {surf_pt_full[0, k, :]}")

print("\nFirst few points along the first theta column:")
for k in range(5):
    print(f"(nphi,ntheta) = { (k, 0) } --> (x,y,z) =  {surf_pt_full[k, 0, :]}")

# print("unitnormal_full.shape =", unitnormal_full.shape)
# for k in range(5):
#     print(f"(nphi,ntheta) = { (0, k) } --> {unitnormal_full[0, k, :]}")

# print("normal_full.shape =", normal_full.shape)
# for k in range(5):
#     print(f"(nphi,ntheta) = { (0, k) } --> {normal_full[0, k, :]}")


# ============================================================================================
# Test points on the surface
# ============================================================================================

# step chooses how many surface points you skip when sampling.
# step = 4 means: take one point every 4 points in both directions.
step_sample = 5

# These are the same surface points where the arrows are sampled.
# ::step means “take every step-th value”.
surf_pt_sampled = surf_pt_full[::step_sample, ::step_sample, :]
surf_xyz_sampled = surf_pt_sampled.reshape(-1, 3)

# Same sampling for the unit normal vectors.
unitnormal_pt_sampled = unitnormal_pt_full[::step_sample, ::step_sample, :]
unitnormal_xyz_sampled = unitnormal_pt_sampled.reshape(-1, 3)

# ============================================================================================
print("============================================================================================")
print("Quick check: Surface sample points storage shape.")
print("surf_pt_sampled.shape =", surf_pt_sampled.shape)
print("surf_xyz_sampled.shape =", surf_xyz_sampled.shape)


# ============================================================================================
# Circular coils are defined by their Fourier coefficients.
# ============================================================================================

number_of_coils = 4
order = 1  # (2*order + 1) = Fourier coefficients for each of the x, y, z coordinates

if flag_surface_case == "torus_simple":
    # Coils in the x-z plane: coil 0 and coil 1
    Rmajor01_coils = 1.0 * Rmajor
    rminor01_coils = 2.0 * rminor

    # Coils in the y-z plane: coil 2 and coil 3
    Rmajor02_coils = 1.0 * Rmajor
    rminor02_coils = 2.0 * rminor

    Ifactor = 1.e7
    Icoils_direction = jnp.array([-1.0, 1.0, -1.0, 1.0])

elif flag_surface_case == "vmec":
    # Coils in the x-z plane: coil 0 and coil 1
    Rmajor01_coils = 1.0 * Rmajor
    rminor01_coils = 2.3 * rminor

    # Coils in the y-z plane: coil 2 and coil 3
    Rmajor02_coils = 0.7 * Rmajor
    rminor02_coils = 2.3 * rminor

    Ifactor = 1.e7
    Icoils_direction = jnp.array([-1.0, 1.0, -1.0, 1.0])

else:
    raise ValueError(f"Unknown flag_surface_case: {flag_surface_case}")


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


coil_curves = Curves(
    coil_dofs,
    n_segments=80,
    nfp=1,
    stellsym=False,
)

coils = Coils(
    curves=coil_curves,
    currents=Icoils,
)




# ============================================================================================
# Magnetic field from the coils from Biot-Savart law
# ============================================================================================

# Magnetic field from the coils using Biot-Savart law
BB_coils = BiotSavart(coils)

# ==============================================================================
# Full surface grid

# BB_coils.B(point) returns a 3-component vector:Bx, By, Bz
B_vector_xyz_full = jnp.array([BB_coils.B(point) for point in surf_xyz_full])
# B modulus for each sampled point
B_modulus_xyz_full = jnp.array([BB_coils.AbsB(point) for point in surf_xyz_full])
# \nabla |B| ( Gradient of |B| ) at each test point
Bmod_grad_xyz_full = jnp.array([BB_coils.dAbsB_by_dX(point) for point in surf_xyz_full])

# You create a class instance to store the magnetic field data at the test points.
BB_xyz_full = MagneticFieldData(
    points=surf_xyz_full,
    Bvec=B_vector_xyz_full,
    Bmod=B_modulus_xyz_full,
    Bmodgrad=Bmod_grad_xyz_full,
)

# Other usefull operations on the full surface grid
B_dot_n_xyz_full = jnp.sum(BB_xyz_full.Bvec * unitnormal_xyz_full, axis=1)




# ==============================================================================
# Sampled surface grid

B_vector_xyz_sampled = jnp.array([BB_coils.B(point) for point in surf_xyz_sampled])
B_modulus_xyz_sampled = jnp.array([BB_coils.AbsB(point) for point in surf_xyz_sampled])
Bmod_grad_xyz_sampled = jnp.array([BB_coils.dAbsB_by_dX(point) for point in surf_xyz_sampled])


# You create a class instance to store the magnetic field data at the test points.
BB_xyz_sampled = MagneticFieldData(
    points=surf_xyz_sampled,
    Bvec=B_vector_xyz_sampled,
    Bmod=B_modulus_xyz_sampled,
    Bmodgrad=Bmod_grad_xyz_sampled,
)

# Other usefull operations on the sample surface grid
B_dot_n_xyz_sampled = jnp.sum(BB_xyz_sampled.Bvec * unitnormal_xyz_sampled, axis=1)



# ============================================================================================
# Calculus of the scale length for the magnetic axis


# L_B = |B| / ||grad(|B|)||. =============================================

norm_gradient_Bmod_xyz_full = jnp.linalg.norm( Bmod_grad_xyz_full, axis=1 )

epsilon = 1e-14
L_B_xyz_full = ( B_modulus_xyz_full / (norm_gradient_Bmod_xyz_full + epsilon) )

print("L_B_xyz_full.shape =", L_B_xyz_full.shape)
print("L_B minimum =", jnp.min(L_B_xyz_full))
print("L_B maximum =", jnp.max(L_B_xyz_full))
print("L_B average =", jnp.mean(L_B_xyz_full))

sys.exit()

# ============================================================================================
# Residual of the Quasi-Symmetry condition: ( \nabla \psi \times \nabla |B| ) \cdot \nabla ( \mathbf{B} \cdot \nabla |B| )
# For the optimization purposes, \nabla \psi is replaced by the unit normal vector to the surface. 
# (n × grad(B)) · grad(B · grad(B)) = 0?
# ============================================================================================


# ==============================================================================
# Full surface grid

# \nabla ( \mathbf{B} \cdot \nabla B ) section ----------------------------

def B_dot_gradB_of_xyz(xyz):
    BB_vec = BB_coils.B(xyz)
    grad_Bmod = BB_coils.dAbsB_by_dX(xyz)
    return jnp.dot(BB_vec, grad_Bmod)

grad_B_dot_gradB_of_xyz = grad(B_dot_gradB_of_xyz)

grad_B_dot_gradB_xyz_full = jnp.array([ grad_B_dot_gradB_of_xyz(point) for point in surf_xyz_full ])
print("grad_B_dot_gradB_xyz_full.shape =", grad_B_dot_gradB_xyz_full.shape)

# \mathbf{n} \times \nabla B. That is, n × grad(B) -----------------------------

n_cross_gradB_xyz_full = jnp.cross(unitnormal_xyz_full, Bmod_grad_xyz_full)
print("n_cross_gradB_xyz_full.shape =", n_cross_gradB_xyz_full.shape)

# ( \mathbf{n} \times \nabla B ) \cdot \nabla ( \mathbf{B} \cdot \nabla B ) -----------------------------
Norm_factor = rminor**2 / B_modulus_xyz_full**4
QS_condition_xyz_full = Norm_factor * jnp.sum( n_cross_gradB_xyz_full * grad_B_dot_gradB_xyz_full, axis=1 )


print("QS_condition_xyz_full.shape =", QS_condition_xyz_full.shape)
print("QS_condition_xyz_full min =", jnp.min(QS_condition_xyz_full))
print("QS_condition_xyz_full max =", jnp.max(QS_condition_xyz_full))

# ==============================================================================
# Sampled surface grid

# \nabla ( \mathbf{B} \cdot \nabla B ) section ----------------------------
grad_B_dot_gradB_xyz_sampled = jnp.array([ grad_B_dot_gradB_of_xyz(point) for point in surf_xyz_sampled ])
print("grad_B_dot_gradB_xyz_sampled.shape =", grad_B_dot_gradB_xyz_sampled.shape)

# \mathbf{n} \times \nabla B. That is, n × grad(B) -----------------------------
n_cross_gradB_xyz_sampled = jnp.cross(unitnormal_xyz_sampled, Bmod_grad_xyz_sampled)
print("n_cross_gradB_xyz_sampled.shape =", n_cross_gradB_xyz_sampled.shape)

# ( \mathbf{n} \times \nabla B ) \cdot \nabla ( \mathbf{B} \cdot \nabla B ) -----------------------------
QS_condition_xyz_sampled = jnp.sum( n_cross_gradB_xyz_sampled * grad_B_dot_gradB_xyz_sampled, axis=1 )

print("QS_condition_xyz_sampled.shape =", QS_condition_xyz_sampled.shape)
print("QS_condition_xyz_sampled min =", jnp.min(QS_condition_xyz_sampled))
print("QS_condition_xyz_sampled max =", jnp.max(QS_condition_xyz_sampled))


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
ax.plot( axis_xyz[:, 0], axis_xyz[:, 1], axis_xyz[:, 2],
    color="black", linewidth=2.5,
    label="magnetic axis",
)

# Plotting the surface
surface.plot(ax=ax, show=False, axis_equal=True, alpha=0.25)

# Plotting the coils
coils.plot(ax=ax, show=False, close=False, color="brown", linewidth=2)



# Show the current direction on each coil
# The current direction follows the coil tangent.
# If the current is negative, we reverse that tangent.

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

plt.show()


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

plt.show()


# ========================================
# Plotting the surface colored by B · n
# ========================================

fig3 = plt.figure(figsize=(7, 6))
ax3 = fig3.add_subplot(111, projection="3d")

# Reshape back to surface grid shape
Norm_factor = B_modulus_xyz_full**(-1)
B_dot_n_pt_full = (Norm_factor * B_dot_n_xyz_full).reshape(surf_pt_full.shape[:2])

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
    facecolors=plt.cm.viridis(norm(B_dot_n_pt_full)),
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
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array(B_dot_n_pt_full)
fig3.colorbar(sm, ax=ax3, shrink=0.7, pad=0.1, label="B · n")

ax3.set_title("Surface colored by B · n")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")
plt.show()

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
plt.show()






# ============================================================================================
# ============================================================================================
# FINAL WRAPPERS THAT WILL GO TO THE CODE ====================================================
# ============================================================================================
# ============================================================================================




########################### QUASI-SYMMETRY LOSS ###########################


# def QS_check_on_surface_original(BBfield, surface):
#     """
#     Return the quasi-symmetry residual on the surface, point by point.
#     The output is a 1D array with one value per surface point.
#     """
#     # Surface points and unit normals, reshaped to (npoints, 3)
#     surf_xyz = surface.gamma.reshape(-1, 3)
#     unitnormal_xyz = surface.unitnormal.reshape(-1, 3)

#     # grad |B| at each surface point
#     grad_Bmod = jnp.array([BBfield.dAbsB_by_dX(point) for point in surf_xyz])

#     # B · grad(|B|) at each point
#     def B_dot_gradB_of_xyz(point):
#         return jnp.dot(BBfield.B(point), BBfield.dAbsB_by_dX(point))

#     # grad( ( B · grad(|B|) )
#     grad_B_dot_gradB = jnp.array([grad(B_dot_gradB_of_xyz)(point) for point in surf_xyz])

#     # QS condition: (n x grad|B|) · grad(B · grad|B|)
#     QS_residual_xyz = jnp.sum( jnp.cross( unitnormal_xyz, grad_Bmod ) * grad_B_dot_gradB , axis=1)

#     return QS_residual_xyz



# print("Computed QS_check_on_surface by the original version")
# QS_check_xyz_full = QS_check_on_surface_original(BB_coils, surface)
# print("QS_check_xyz_full.shape =", QS_check_xyz_full.shape)
# print("QS_check_xyz_full min =", jnp.min(QS_check_xyz_full))
# print("QS_check_xyz_full max =", jnp.max(QS_check_xyz_full))

# def QS_check_on_surface_wrapped(BBfield, surface):
#     """
#     Return a pointwise quasi-symmetry residual on the surface.
#     Shape is usually (nphi, ntheta) or flattened to (npoints,).
#     """
#     # 1. get surface points
#     surf_xyz = surface.gamma.reshape(-1, 3)

#     # 2. field and gradient of |B|
#     BBvec_xyx = jax.vmap(BBfield.B)(surf_xyz)
#     gradB = jax.vmap(BBfield.dAbsB_by_dX)(surf_xyz)

#     # 3. surface normal
#     unitnormal_xyz = surface.unitnormal.reshape(-1, 3)

#     # 4. quasi-symmetry condition
#     #    (n x grad|B|) · grad(B · grad|B|)
#     B_dot_gradB = jnp.sum(BBvec_xyx * gradB, axis=1)
#     grad_B_dot_gradB = jax.vmap(jax.grad(lambda x: jnp.dot(BBfield.B(x), BBfield.dAbsB_by_dX(x))))(surf_xyz)
#     QS_residual_xyz = jnp.sum(jnp.cross(unitnormal_xyz, gradB) * grad_B_dot_gradB, axis=1)

#     return QS_residual_xyz

# print("Computed QS_check_on_surface by the wrapped version")
# QS_check_xyz_full = QS_check_on_surface_wrapped(BB_coils, surface)
# print("QS_check_xyz_full.shape =", QS_check_xyz_full.shape)
# print("QS_check_xyz_full min =", jnp.min(QS_check_xyz_full))
# print("QS_check_xyz_full max =", jnp.max(QS_check_xyz_full))

# def loss_QS(field, surface):
#     """
#     Scalar objective: smaller means closer to quasi-symmetry.
#     """
#     QS_residual_xyz = QS_check_on_surface(field, surface)
#     QS_residual_sqr = jnp.mean(jnp.square(QS_residual_xyz))
#     return QS_residual_sqr
