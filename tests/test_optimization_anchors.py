"""Known-answer tests for the coil-optimization side.

1. Problem 71 of Hock and Schittkowski (Test Examples for Nonlinear Programming
   Codes, Lecture Notes in Economics and Mathematical Systems 187, Springer,
   1981): min x1 x4 (x1 + x2 + x3) + x3 subject to x1 x2 x3 x4 >= 25,
   x1^2 + x2^2 + x3^2 + x4^2 = 40 and 1 <= xi <= 5, with the published solution
   x* = (1, 4.7429994, 3.8211503, 1.3794082), f* = 17.0140173.
   The book lists no multipliers. They follow from stationarity at x*: x1 sits
   on its lower bound, so components 2 to 4 of grad f = l1 grad c1 + l2 grad c2
   fix l1 = 0.55229366, l2 = -0.16146857 (checked below to 1e-6 from the
   published x*). ESSOS writes the Lagrangian as
   f - l c + mu c^2 / 2 for c = 0 and for c >= 0, so its multipliers carry the
   same signs as l1, l2 above.
2. The squared flux of the Landreman-Paul QA coils on the QA boundary and its
   derivative with respect to the coil degrees of freedom against SIMSOPT on the
   same degrees of freedom. The numbers come from tests/references/simsopt_squared_flux.py.
"""
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.augmented_lagrangian import ALM_model_lbfgs, combine, eq, ineq
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.surfaces import SquaredFlux, SurfaceRZFourier

jax.config.update("jax_enable_x64", True)

HS71_X = np.array([1.0, 4.7429994, 3.8211503, 1.3794082])
HS71_F = 17.0140173
HS71_MULTIPLIERS = np.array([0.55229366, -0.16146857])


def _hs71_objective(x):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


def _hs71_solve(start, outer=100, **options):
    constraints = combine(ineq(lambda x: x[0] * x[1] * x[2] * x[3] - 25.0),
                          eq(lambda x: jnp.sum(x**2) - 40.0))
    x0 = jnp.asarray(start)
    params = x0, constraints.init(x0)
    alm = ALM_model_lbfgs(constraints, loss=_hs71_objective, bounds=(1.0, 5.0), inner_maxiter=500, **options)
    state, grad, info = alm.init(params)
    for _ in range(outer):
        params, state, grad, info = alm.update(params, state, grad, info)
    x, multipliers = params
    return np.asarray(x), np.array([float(m['lambda'].value) for m in multipliers])


def test_hs71_multipliers_follow_from_stationarity():
    grad_f = np.asarray(jax.grad(_hs71_objective)(jnp.asarray(HS71_X)))
    grad_c1 = np.prod(HS71_X) / HS71_X
    grad_c2 = 2 * HS71_X
    lsq = np.linalg.lstsq(np.stack([grad_c1, grad_c2], 1)[1:], grad_f[1:], rcond=None)[0]
    np.testing.assert_allclose(lsq, HS71_MULTIPLIERS, atol=1e-6)


@pytest.mark.parametrize("model_mu", ["Mu_Tolerance", "Mu_Adaptative_1"])
def test_hs71_augmented_lagrangian_known_point(model_mu):
    x, multipliers = _hs71_solve([1.5, 4.5, 3.5, 1.5], model_mu=model_mu, eta_tol=1e-10, omega_tol=1e-12)
    np.testing.assert_allclose(x, HS71_X, atol=1e-6)
    assert abs(float(_hs71_objective(x)) - HS71_F) < 1e-6
    np.testing.assert_allclose(multipliers, HS71_MULTIPLIERS, atol=1e-6)


@pytest.mark.xfail(strict=False, reason="the tanh map of the box has zero slope on the bound: a start with "
                   "x2 = x3 = 5 (the standard HS71 start (1, 5, 5, 1)) never moves")
def test_hs71_from_the_standard_start_on_the_bound():
    x, _ = _hs71_solve([1.0, 5.0, 5.0, 1.0], eta_tol=1e-10, omega_tol=1e-12)
    np.testing.assert_allclose(x, HS71_X, atol=1e-6)


INPUTS = Path(__file__).resolve().parents[1] / "examples" / "input_files"
# SIMSOPT 1.10.7.dev433+g24fa9b4b8: J, dJ.v for v_k = sin(1 + k) over the base-curve
# coefficients, dJ/dI for the four base currents, dJ for the first three curve coefficients.
SIMSOPT = {
    "quadratic flux": (0.0015371102024469593, 0.025811703742100622,
                       [7.033698235984576e-09, 7.905091637519413e-09, 8.05956941991598e-09, 7.74384475763028e-09],
                       [0.014905299530073012, 0.0007880556289943546, -0.01159790034524593]),
    "normalized": (0.0018731053549893475, 0.031188615579313406,
                   [-1.7573492886413003e-09, -1.6618587392015578e-10, 8.431671898501097e-10, 1.0803679728743138e-09],
                   [0.0178805393325929, 0.0009361300783638725, -0.013893719972785306]),
    "local": (0.015014254138214476, 0.2541949351457553,
              [-3.922300928810166e-08, -1.280132289116158e-08, 1.5622662526605715e-08, 3.6401669656417646e-08],
              [0.14040513999761287, 0.006853630040430432, -0.10711978406610913]),
}
# The two unnormalized definitions of ESSOS are (2 pi)^2 smaller than SIMSOPT's, exactly: ESSOS
# takes the mean over the angles in radians without the d(theta) d(phi) measure, while SIMSOPT
# integrates over [0, 1)^2. ESSOS's own area() and volume() do carry the (2 pi)^2, so the squared
# flux is not the surface integral in the units of the rest of the code. The factor is applied
# here so that the comparison is of the same quantity.
ANGLE_FACTOR = {"quadratic flux": (2 * math.pi)**2, "normalized": 1.0, "local": (2 * math.pi)**2}


@pytest.mark.parametrize("definition", list(SIMSOPT))
def test_squared_flux_matches_simsopt(definition):
    coils = Coils.from_json(INPUTS / "ESSOS_biot_savart_LandremanPaulQA.json")
    surface = SurfaceRZFourier.from_input_file(INPUTS / "input.LandremanPaul2021_QA_beta0p5_bootstrap",
                                               ntheta=32, nphi=32, close=False)

    def flux(dofs):
        return ANGLE_FACTOR[definition] * SquaredFlux(surface, BiotSavart(coils.with_dofs(dofs)), definition)

    value, grad = jax.value_and_grad(flux)(coils.dofs)
    n = coils.dofs_curves.size
    direction = np.sin(1.0 + np.arange(n))
    j, dj_dir, dj_currents, dj_first = SIMSOPT[definition]
    np.testing.assert_allclose(float(value), j, rtol=1e-10)
    np.testing.assert_allclose(float(grad[:n] @ direction), dj_dir, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(grad[:3]), dj_first, rtol=1e-10)
    # ESSOS optimizes the currents divided by currents_scale; SIMSOPT the currents in A.
    np.testing.assert_allclose(np.asarray(grad[n:]) / coils.currents_scale, dj_currents,
                               rtol=1e-9, atol=1e-10 * np.max(np.abs(dj_currents)))
