"""Optimize a winding surface with a singular-value proxy for the coil-to-plasma operator.

The normal field on the plasma from a dipole layer on the winding surface is a linear operator.
Its singular values measure how much field each current mode can produce, and a flat spectrum
(maximal entropy) means many modes are usable. The winding surface starts as a normal offset of
the plasma and is moved to flatten the spectrum while keeping its volume, plasma clearance and
smoothness. The operator includes the surface Jacobians, so it is a quadrature of the integral
operator and does not depend on the grid. Run time: about 10 s on a laptop CPU.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from essos.constants import mu_0
from essos.current_potential import normal_offset_dofs
from essos.surfaces import SurfaceRZFourier

# Parameters
SVD_WEIGHT = 1.0
STRENGTH_WEIGHT = 1.0       # keeps the mean singular value from dropping while the entropy is raised
VOLUME_WEIGHT = 0.02
SPECTRAL_WEIGHT = 0.02
DISTANCE_WEIGHT = 10.0
MINIMUM_DISTANCE = 0.06
DISTANCE_WALL_SCALE = 0.01
SHARPNESS = 300.0
MAXITER = 100
ACTIVE_MPOL = 2
ACTIVE_NTOR = 2
COEFFICIENT_STEP_BOUND = 0.04
ntheta = 24
nphi = 24

wout = os.path.join(os.path.dirname(__file__), "input_files", "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
plasma = SurfaceRZFourier.from_wout_file(wout, s=1, ntheta=ntheta, nphi=nphi, close=False)
winding = SurfaceRZFourier.from_wout_file(wout, s=1, ntheta=ntheta, nphi=nphi, close=False)
minor_radius = jnp.sqrt(plasma.mean_cross_sectional_area() / jnp.pi)
winding.dofs = normal_offset_dofs(winding, -minor_radius)  # ESSOS normals point inward


def quadrature_weights(surface):
    """Area weights of the periodic trapezoid rule on the full torus."""
    return surface.area_element.reshape(-1) * (2 * jnp.pi) ** 2 / (surface.ntheta * surface.nphi)


def weighted_dipole_operator(winding):
    """Normal field on the plasma per unit dipole density, scaled by sqrt of the quadrature weights."""
    difference = winding.gamma.reshape(1, -1, 3) - plasma.gamma.reshape(-1, 1, 3)
    distance_squared = jnp.sum(difference ** 2, axis=2)
    dipole_normal = winding.unitnormal.reshape(1, -1, 3)
    plasma_normal = plasma.unitnormal.reshape(-1, 1, 3)
    kernel = mu_0 / (4 * jnp.pi) * (
        3 * jnp.sum(difference * dipole_normal, axis=2) * jnp.sum(difference * plasma_normal, axis=2)
        - distance_squared * jnp.sum(dipole_normal * plasma_normal, axis=2)) / distance_squared ** 2.5
    return jnp.sqrt(quadrature_weights(plasma))[:, None] * kernel * jnp.sqrt(quadrature_weights(winding))


def smooth_minimum_distance(surface1, surface2):
    distance = jnp.linalg.norm(surface1.gamma.reshape(-1, 1, 3) - surface2.gamma.reshape(1, -1, 3), axis=2).reshape(-1)
    return jnp.vdot(jax.nn.softmax(-SHARPNESS * distance), distance)


def objectives(winding):
    singular_values = jnp.linalg.svd(weighted_dipole_operator(winding), compute_uv=False)
    probabilities = singular_values / jnp.sum(singular_values)
    entropy = -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-300)))
    spectral = jnp.sum((winding.xm * winding.rc) ** 2) + jnp.sum((winding.xm * winding.zs) ** 2)
    distance = smooth_minimum_distance(plasma, winding)
    return jnp.stack((1 / entropy, jnp.sum(singular_values), winding.volume, spectral, distance,
                      jnp.min(winding.area_element)))


initial = objectives(winding)
dofs0 = winding.dofs
nmodes = winding.xm.size
active_modes = jnp.where((winding.xm <= ACTIVE_MPOL) & (jnp.abs(winding.xn / winding.nfp) <= ACTIVE_NTOR))[0]
active = jnp.concatenate((active_modes[active_modes != 0], active_modes + nmodes))  # rc(0, 0) is fixed
x0 = dofs0[active]


def loss(x):
    winding.dofs = dofs0.at[active].set(x)
    values = objectives(winding)
    inverse_entropy, strength, volume, spectral = (values / initial)[:4]
    distance_wall = 1 + jnp.tanh((MINIMUM_DISTANCE - values[4]) / DISTANCE_WALL_SCALE)
    collapse = 1e12 * jnp.maximum(1e-6 - values[5], 0) ** 2  # keeps the Jacobian positive
    return (SVD_WEIGHT * inverse_entropy
            + STRENGTH_WEIGHT * jnp.maximum(1 - strength, 0) ** 2  # operator must not weaken
            - VOLUME_WEIGHT * volume + SPECTRAL_WEIGHT * spectral
            + DISTANCE_WEIGHT * distance_wall + 100 * collapse)


value_and_grad = jax.jit(jax.value_and_grad(loss))
result = minimize(lambda x: (lambda v, g: (float(v), np.asarray(g)))(*value_and_grad(x)), x0, method="L-BFGS-B", jac=True,
                  bounds=[(float(v - COEFFICIENT_STEP_BOUND), float(v + COEFFICIENT_STEP_BOUND)) for v in x0],
                  options={"maxiter": MAXITER})
winding.dofs = dofs0.at[active].set(result.x)
final = objectives(winding)
print(f"L-BFGS-B: {result.nit} iterations, {result.nfev} evaluations")
for name, a, b in zip(("1/entropy", "sum of singular values", "volume [m^3]", "spectral", "plasma distance [m]"), initial, final):
    print(f"{name}: {float(a):.6g} -> {float(b):.6g}")
