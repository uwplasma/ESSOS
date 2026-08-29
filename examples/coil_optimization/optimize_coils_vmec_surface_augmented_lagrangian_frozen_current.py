"""VMEC-surface augmented-Lagrangian optimization with one frozen current.

This is the ALM portion of the existing VMEC-surface comparison example. The
only freezing-specific code is the named current selection on
``C_total_constraint``.
"""
import os
from time import time

import jax.numpy as jnp
import matplotlib.pyplot as plt

import essos.augmented_lagrangian as alm
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B


maximum_function_evaluations = 100
input_filepath = os.path.join(os.path.dirname(__file__), "..", "input_files")
vmec_input = os.path.join(input_filepath, "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")

N_COILS = 4; FOURIER_ORDER = 3; LARGE_R = 10.; SMALL_R = 5.7; NFP = 2; N_SEGMENTS = 30; STELLSYM = True
COIL_CURRENT = 1.
FROZEN_CURRENT_INDEX = 0

LENGTH_TARGET = 40.
CURVATURE_TARGET = 0.5
BdotN_TARGET_TOL = 1.e-6

curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R,
                                   n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
init_coils = Coils(curves=curves, currents=[COIL_CURRENT] * N_COILS)
init_field = BiotSavart(init_coils)
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=32, nphi=32, range_torus="half period")


def bdotn_constraint(field, surface, target_tol=BdotN_TARGET_TOL):
    bdotn_over_b = BdotN_over_B(surface, field)
    return jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(bdotn_over_b) - target_tol, 0.0)))


def length_constraint(field):
    return jnp.maximum(0, field.coils.length - LENGTH_TARGET)


def curvature_constraint(field):
    return jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET)


model_lagrangian = "Standard"
normal_field = alm.eq(bdotn_constraint, model_lagrangian=model_lagrangian)
length = alm.eq(length_constraint, model_lagrangian=model_lagrangian)
curvature = alm.eq(curvature_constraint, model_lagrangian=model_lagrangian)

C_normal_field = alm.SelectiveConstraint(normal_field, "field", surface=surface)
C_length = alm.SelectiveConstraint(length, "field")
C_curvature = alm.SelectiveConstraint(curvature, "field")
C_total_constraint = alm.combine(C_normal_field, C_length, C_curvature)
C_total_constraint.dependencies = {"field": init_field}

C_total_constraint.freeze_current("field", coil=FROZEN_CURRENT_INDEX)

ALM = alm.ALM_model_jaxopt_lbfgsb(
    constraints=C_total_constraint,
    model_lagrangian=model_lagrangian,
    beta=2., mu_max=1.e4, alpha=0.99, gamma=1.e-2, epsilon=1.e-8,
    eta_tol=1.e-7, omega_tol=1.e-7,
)

lagrange_params = C_total_constraint.init(init_field.dofs)
params = (init_field.dofs, lagrange_params)
lag_state, gradient, info = ALM.init(params)

t_start = time()
i = 0
while i <= maximum_function_evaluations and (
    jnp.linalg.norm(gradient[0]) > 1.e-7 or alm.norm_constraints(info[2]) > 1.e-7
):
    params, lag_state, gradient, info = ALM.update(params, lag_state, gradient, info)
    print(f"i: {i}, loss f: {info[0]:g}, loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}")
    i += 1

opt_field, = C_total_constraint.dofs_to_pytree(params[0])
opt_coils = opt_field.coils
print(f"\nOptimization took {time() - t_start:.2f} seconds")
print("Initial normalized currents:", init_coils.dofs_currents)
print("Optimized normalized currents:", opt_coils.dofs_currents)
print("Frozen physical current:", opt_coils.dofs_currents_raw[FROZEN_CURRENT_INDEX])

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection="3d")
init_coils.plot(ax=ax1, show=False)
surface.plot(ax=ax1, show=False)
ax2 = fig.add_subplot(122, projection="3d")
opt_coils.plot(ax=ax2, show=False)
surface.plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()
