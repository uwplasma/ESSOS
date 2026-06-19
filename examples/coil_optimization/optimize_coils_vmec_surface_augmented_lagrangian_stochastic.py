import os
from time import time

number_of_processors_to_use = 8  # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={number_of_processors_to_use}"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import essos.augmented_lagrangian as alm
from essos.coil_perturbation import (
    GaussianSampler,
    perturb_curves_statistic,
    perturb_curves_systematic,
)
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart, Vmec
from essos.losses import custom_loss
from essos.surfaces import BdotN_over_B


""" Creating stochastic field losses """
def copy_coils_from_field(field):
    return field.coils.copy()


def perturbed_field_from_field(field, key, sampler):
    coils = copy_coils_from_field(field)
    split_keys = jax.random.split(key, 2)
    perturb_curves_systematic(coils, sampler, key=split_keys[0])
    coils = perturb_curves_statistic(coils, sampler, key=split_keys[1])
    return BiotSavart(coils)


def loss_bdotn_stochastic(field, surface, sampler, keys):
    def perturbed_loss(key):
        perturbed_field = perturbed_field_from_field(field, key, sampler)
        bdotn_over_b = BdotN_over_B(surface, perturbed_field)
        return jnp.sum(jnp.abs(bdotn_over_b))

    return jnp.mean(jax.vmap(perturbed_loss)(keys))


def constraint_bdotn_stochastic(field, surface, sampler, keys, target_tol=1.0e-6):
    def perturbed_square(key):
        perturbed_field = perturbed_field_from_field(field, key, sampler)
        return jnp.square(BdotN_over_B(surface, perturbed_field))

    expected_square = jnp.mean(jax.vmap(perturbed_square)(keys), axis=0)
    return jnp.sqrt(jnp.sum(jnp.maximum(expected_square - target_tol, 0.0)))


def loss_length_constraint(field, max_coil_length):
    return jnp.maximum(0.0, field.coils.length - max_coil_length)


def loss_curvature_constraint(field, max_coil_curvature):
    return jnp.maximum(0.0, field.coils.curvature - max_coil_curvature)


# Optimization parameters
maximum_function_evaluations = 10

MAX_COIL_LENGTH = 40.0
MAX_COIL_CURVATURE = 0.5
BDOTN_TARGET_TOL = 1.0e-6
FOURIER_ORDER = 6
N_SEGMENTS = FOURIER_ORDER * 10
N_COILS = 4
NTHETA = 32
NPHI = 32

input_filepath = os.path.join(os.path.dirname(__file__), "input_files")
vmec_input = os.path.join(input_filepath, "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")

""" Creating starting coils and surface """
vmec = Vmec(vmec_input, ntheta=NTHETA, nphi=NPHI, range_torus="full torus")
surface = vmec.surface

COIL_CURRENT = 1.0
number_of_field_periods = vmec.nfp
major_radius_coils = vmec.r_axis
minor_radius_coils = vmec.r_axis / 1.5
curves = CreateEquallySpacedCurves(n_curves=N_COILS,order=FOURIER_ORDER, R=major_radius_coils,
                                   r=minor_radius_coils,n_segments=N_SEGMENTS,  nfp=number_of_field_periods,stellsym=True)
coils_initial = Coils(curves=curves,currents=[COIL_CURRENT] * N_COILS)
field_initial = BiotSavart(coils_initial)

""" Setting the stochastic sampling parameters """
SIGMA = 0.01
LENGTH_SCALE = 0.4 * jnp.pi
N_DERIVS = 2
N_samples = 10
sampler = GaussianSampler(coils_initial.curves.quadpoints, sigma=SIGMA, length_scale=LENGTH_SCALE, n_derivs=N_DERIVS)
stochastic_keys = jax.random.split(jax.random.PRNGKey(0), N_samples)

""" Defining custom losses """
L_normal_field = custom_loss(loss_bdotn_stochastic, "field", surface=surface, sampler=sampler, keys=stochastic_keys)
L_normal_field_constraint = custom_loss(constraint_bdotn_stochastic, "field", surface=surface, sampler=sampler, keys=stochastic_keys, target_tol=BDOTN_TARGET_TOL)
L_length_constraint = custom_loss(loss_length_constraint, "field", max_coil_length=MAX_COIL_LENGTH)
L_curvature_constraint = custom_loss(loss_curvature_constraint, "field", max_coil_curvature=MAX_COIL_CURVATURE)

""" Defining total loss + setting dependencies """
L_normal_field.dependencies = {"field": field_initial}
L_normal_field_constraint.dependencies = {"field": field_initial}
L_length_constraint.dependencies = {"field": field_initial}
L_curvature_constraint.dependencies = {"field": field_initial}

""" Creating the constraints """
penalty = 0.1
multiplier = 0.5
sq_grad = 0.0
model_lagrangian = "Standard"

beta = 2.0
mu_max = 1.0e4
alpha = 0.99
gamma = 1.0e-2
epsilon = 1.0e-8
omega_tol = 1.0e-7
eta_tol = 1.0e-7

normal_field_constraint = alm.eq(constraint_bdotn_stochastic, model_lagrangian=model_lagrangian, multiplier=multiplier, penalty=penalty, sq_grad=sq_grad)
length_constraint = alm.eq(loss_length_constraint, model_lagrangian=model_lagrangian, multiplier=multiplier, penalty=penalty, sq_grad=sq_grad)
curvature_constraint = alm.eq(loss_curvature_constraint, model_lagrangian=model_lagrangian, multiplier=multiplier, penalty=penalty, sq_grad=sq_grad)

C_normal_field_constraint = alm.SelectiveConstraint(normal_field_constraint, "field", surface=surface, sampler=sampler, keys=stochastic_keys, target_tol=BDOTN_TARGET_TOL)
C_length_constraint = alm.SelectiveConstraint(length_constraint, "field", max_coil_length=MAX_COIL_LENGTH)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field", max_coil_curvature=MAX_COIL_CURVATURE)
C_total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint)

C_normal_field_constraint.dependencies = {"field": field_initial}
C_length_constraint.dependencies = {"field": field_initial}
C_curvature_constraint.dependencies = {"field": field_initial}
C_total_constraint.dependencies = {"field": field_initial}

ALM = alm.ALM_model_jaxopt_lbfgsb(constraints=C_total_constraint, model_lagrangian=model_lagrangian, beta=beta, mu_max=mu_max, alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol)

""" Optimizing with alm """
lagrange_params = C_total_constraint.init(field_initial.dofs)
params = field_initial.dofs, lagrange_params
lag_state, grad, info = ALM.init(params)

# Initializing first tolerances for the inner minimisation loop iteration
mu_average = alm.penalty_average(lagrange_params)
omega = 1.0 / mu_average
eta = 1.0 / mu_average**0.1

print(f"Optimizing coils with {maximum_function_evaluations} function evaluations using stochastic ALM.")
time0 = time()

i = 0
while i <= maximum_function_evaluations and (jnp.linalg.norm(grad[0]) > omega_tol or alm.norm_constraints(info[2]) > eta_tol):
    params, lag_state, grad, info, eta, omega = ALM.update(params, lag_state, grad, info, eta, omega)
    print(f"i: {i}, loss f: {info[0]:g}, loss L: {info[1]:g}, " f"infeasibility: {alm.total_infeasibility(info[2]):g}")
    i += 1

field_optimized = C_normal_field_constraint.dofs_to_pytree(params[0])[0]
coils_optimized = field_optimized.coils

print(f"Stochastic optimization with ALM took {time() - time0:.2f} seconds")

BdotN_over_B_initial = BdotN_over_B(surface, field_initial)
BdotN_over_B_optimized = BdotN_over_B(surface, field_optimized)
curvature = jnp.mean(field_optimized.coils.curvature, axis=1)
length = jnp.max(jnp.ravel(field_optimized.coils.length))
stochastic_loss_initial = L_normal_field(L_normal_field.starting_dofs)
stochastic_loss_final = L_normal_field(params[0])

print("Mean curvature:", curvature)
print("Length:", length)
print(f"Stochastic |BdotN/B| loss before optimization: {stochastic_loss_initial:.2e}")
print(f"Stochastic |BdotN/B| loss after optimization: {stochastic_loss_final:.2e}")
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")
print(f"Average BdotN/B before optimization: {jnp.average(jnp.abs(BdotN_over_B_initial)):.2e}")
print(f"Average BdotN/B after optimization: {jnp.average(jnp.abs(BdotN_over_B_optimized)):.2e}")

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")
coils_initial.plot(ax=ax1, show=False)
surface.plot(ax=ax1, show=False)
coils_optimized.plot(ax=ax2, show=False)
surface.plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()
