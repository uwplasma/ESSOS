"""Finite-beta near-axis coil targets and the fixed-coil pressure response of the axis."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from pyqsc_jax.near_axis import near_axis
from scipy.integrate import solve_ivp

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import near_axis_coil_residuals, near_axis_coil_targets, pressure_axis_response

QA = dict(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95)
QH = dict(rc=[1, 0.17, 0.01804, 0.001409, 5.877e-5], zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-5], nfp=4, etabar=1.569)


def vacuum(geometry=QA, nphi=51, **kwargs):
    return near_axis(**geometry, nphi=nphi, order="r1", B0=1, p2=0, I2=0, **kwargs).solution


@pytest.mark.parametrize("sG,spsi", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
@pytest.mark.parametrize("geometry", [QA, QH])
def test_response_orientations_scaling_and_length(sG, spsi, geometry):
    solution = vacuum(geometry, sG=sG, spsi=spsi)
    result = pressure_axis_response(solution, 0.02, -6e5)
    assert result["physical_relative_difference"] < 2e-7  # Frenet operator = closed form
    np.testing.assert_allclose(result["length_slope_over_L"], result["length_formula_over_L"], rtol=2e-7)
    assert result["length_slope_over_L"] > 0 and abs(result["delta_Z"][0]) < 1e-10
    doubled = pressure_axis_response(solution, 0.04, -6e5)
    np.testing.assert_allclose(doubled["delta_R"], 4 * result["delta_R"], rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(pressure_axis_response(solution, 0.02, 0)["delta_R"], 0, atol=1e-12)
    with pytest.raises(ValueError):
        pressure_axis_response(near_axis(**geometry, nphi=51, order="r1", I2=0.1).solution, 0.02, -6e5)


def test_supplied_gradient_changes_physical_observables_only():
    solution = vacuum(nphi=101)
    ideal = pressure_axis_response(solution, 0.03, -6e5)
    n, b = np.asarray(solution.geometry.normal_cartesian), np.asarray(solution.geometry.binormal_cartesian)
    gradient = np.asarray(solution.grad_B_axis) + 0.3 * (np.einsum("ni,nj->nij", n, n) - np.einsum("ni,nj->nij", b, b))
    result = pressure_axis_response(solution, 0.03, -6e5, gradient=gradient)
    for key in ("u", "delta_R", "length_slope_over_L", "rms_displacement"):
        np.testing.assert_array_equal(result[key], ideal[key])
    assert abs(result["physical_length_slope_over_L"] / ideal["length_slope_over_L"] - 1) > 1e-2
    # Independent first variation: displace the whole curve in fixed planes and difference its length.
    xi, D = result["physical_xi_lab"], np.asarray(solution.geometry.d_d_phi)
    phi, R0, Z0 = np.asarray(solution.phi), np.asarray(solution.R0), np.asarray(solution.Z0)
    dR = xi[:, 0] * np.cos(phi) + xi[:, 1] * np.sin(phi)
    length = lambda h: np.mean(np.sqrt((D @ (R0 + h * dR)) ** 2 + (R0 + h * dR) ** 2 + (D @ (Z0 + h * xi[:, 2])) ** 2))
    np.testing.assert_allclose((length(1e-4) - length(-1e-4)) / (2e-4 * length(0)),
                               result["physical_length_slope_over_L"], rtol=1e-6)


def test_closed_form_matches_periodic_shooting_in_boozer_angle():
    solution = vacuum()
    result = pressure_axis_response(solution, 0.02, -6e5)
    phi, varphi = np.asarray(solution.phi), np.asarray(solution.varphi)
    period, nu = 2 * np.pi / int(solution.inputs.axis.nfp), float(solution.iotaN)
    x = float(solution.inputs.etabar) / np.asarray(solution.geometry.curvature)
    forcing_samples = 1 - 1 / (1 + x**2 + 1j * np.asarray(solution.sigma))
    # Forcing as a periodic spline in the Boozer angle, then the ODE d zeta/d varphi = i nu zeta - i Cp F.
    grid = np.r_[varphi, varphi[0] + period]
    F = lambda s: np.interp(np.mod(s - varphi[0], period) + varphi[0], grid, np.r_[forcing_samples, forcing_samples[:1]])
    rhs = lambda s, z: 1j * nu * z - 1j * result["Cp"] * F(s)
    kw = dict(method="DOP853", rtol=1e-11, atol=1e-14, max_step=period / 400)
    drift = solve_ivp(rhs, (varphi[0], varphi[0] + period), [0j], **kw).y[0, -1]
    shot = solve_ivp(rhs, (varphi[0], varphi[0] + period), [drift / (1 - np.exp(1j * nu * period))],
                     dense_output=True, **kw)
    np.testing.assert_allclose(shot.sol(varphi)[0], result["zeta"], rtol=0, atol=2e-3 * np.max(abs(result["zeta"])))


def test_coil_targets_subtract_only_the_plasma_field():
    kw = dict(**QA, nphi=31, order="r3", B0=1, B2c=-0.7, I2=0.0)
    vac, beta = near_axis(**kw, p2=0.0).solution, near_axis(**kw, p2=-6e5).solution
    for total, external in zip(near_axis_coil_targets(vac, 0.03, False), near_axis_coil_targets(vac, 0.03, True)):
        np.testing.assert_allclose(external, total, atol=1e-12)
    _, B, _, _ = near_axis_coil_targets(beta, 0.03, True)
    assert 1e-6 < float(jnp.max(jnp.abs(B - beta.B_axis))) < 1e-2  # the plasma field is small but finite
    curves = CreateEquallySpacedCurves(n_curves=4, order=4, R=1.0, r=0.5, n_segments=40, nfp=2, stellsym=True)
    field = BiotSavart(Coils(curves=curves, currents=jnp.full(4, 1.0e5)))
    residual = near_axis_coil_residuals(field, beta, 0.03, hessian_weight=0.0)
    assert residual.shape == (31 * (3 + 9 + 27),) and np.all(np.isfinite(residual))
    np.testing.assert_allclose(residual[31 * 12:], 0)
