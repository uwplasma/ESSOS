"""Finite-beta near-axis helpers on a synthetic circular axis, without pyQSC_JAX."""
import sys
from types import ModuleType, SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from essos.coils import Coils, CreateEquallySpacedCurves  # noqa: E402
from essos.fields import BiotSavart  # noqa: E402
from essos.objective_functions import (frenet_axis_response, near_axis_coil_residuals,  # noqa: E402
                                       near_axis_coil_targets, pressure_axis_response)


def _spectral_derivative(n):
    """Periodic spectral d/dphi on n (odd) equally spaced points over 2 pi."""
    k = np.fft.fftfreq(n, d=1 / n)
    return np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0))


def circular_axis(n=21, iota=0.3, I2=0.0, coupling=0.4):
    """A unit circle with constant curvature and no torsion, written as a pyQSC_JAX-like solution."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    eR = np.stack((np.cos(phi), np.sin(phi), 0 * phi), -1)
    ephi = np.stack((-np.sin(phi), np.cos(phi), 0 * phi), -1)
    ez = np.tile([0.0, 0.0, 1.0], (n, 1))
    # A gradient coupling the normal and binormal directions keeps the Frenet operator invertible.
    gradient = coupling * (np.einsum("ni,nj->nij", -eR, -eR) - np.einsum("ni,nj->nij", ez, ez))
    geometry = SimpleNamespace(d_l_d_phi=np.ones(n), curvature=np.ones(n), torsion=np.zeros(n),
                               d_d_phi=_spectral_derivative(n), normal_cartesian=-eR, binormal_cartesian=ez,
                               tangent_cartesian=ephi, position_cartesian=jnp.asarray(eR))
    inputs = SimpleNamespace(B0=1.0, etabar=0.5, sG=1, spsi=1, I2=I2, p2=-1e5, axis=SimpleNamespace(nfp=1))
    return SimpleNamespace(inputs=inputs, geometry=geometry, phi=phi, varphi=phi, sigma=np.zeros(n),
                           axis_length=2 * np.pi, iotaN=iota, iota=iota, R0=np.ones(n), grad_B_axis=gradient)


def test_pressure_axis_response_scales_and_matches_its_length_formula():
    solution = circular_axis()
    result = pressure_axis_response(solution, 0.02)
    assert np.isfinite(result["physical_delta_R"]).all() and np.isfinite(result["delta_Z"]).all()
    np.testing.assert_allclose(result["length_slope_over_L"], result["length_formula_over_L"], rtol=1e-8)
    doubled = pressure_axis_response(solution, 0.04)
    np.testing.assert_allclose(doubled["delta_R"], 4 * result["delta_R"], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(doubled["physical_delta_R"], 4 * result["physical_delta_R"], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(pressure_axis_response(solution, 0.02, p2=0.0)["rms_displacement"], 0, atol=1e-14)
    assert result["helicity"] == 0 and result["iotaN"] == 0.3


@pytest.mark.parametrize("kwargs,radius", [(dict(I2=0.1), 0.02), (dict(n=20), 0.02), ({}, 0.0), (dict(iota=1.0), 0.02)])
def test_pressure_axis_response_rejects_invalid_or_resonant_axes(kwargs, radius):
    with pytest.raises(ValueError):
        pressure_axis_response(circular_axis(**kwargs), radius)


def test_frenet_axis_response_solves_the_periodic_operator():
    solution = circular_axis()
    n, N, Bn = len(solution.phi), solution.geometry.normal_cartesian, solution.geometry.binormal_cartesian
    perturbation = 1e-3 * (np.cos(solution.phi)[:, None] * N + np.sin(2 * solution.phi)[:, None] * Bn)
    u, v = frenet_axis_response(solution, perturbation)
    # Residual of (D - A) [u, v] = forcing for this axis: A_NN = c, A_BB = -c, no torsion.
    D, c = solution.geometry.d_d_phi, 0.4
    np.testing.assert_allclose(D @ u - c * u, np.sum(perturbation * N, 1), atol=1e-12)
    np.testing.assert_allclose(D @ v + c * v, np.sum(perturbation * Bn, 1), atol=1e-12)
    u2, _ = frenet_axis_response(solution, perturbation, tangent_field=np.full(n, 2.0))
    assert not np.allclose(u2, u)


def _coil_field():
    curves = CreateEquallySpacedCurves(n_curves=2, order=2, R=1.0, r=0.4, n_segments=30, nfp=2, stellsym=True)
    return BiotSavart(Coils(curves=curves, currents=jnp.full(2, 1.0e5)))


def _with_field_jet(solution, field):
    points = solution.geometry.position_cartesian
    solution.B_axis = jax.vmap(field.B)(points)
    solution.grad_B_axis = jax.vmap(field.dB_by_dX)(points)
    solution.grad_grad_B_axis = jax.vmap(jax.jacfwd(field.dB_by_dX))(points)
    return solution


def test_coil_residuals_vanish_for_the_coils_own_jet():
    field = _coil_field()
    solution = _with_field_jet(circular_axis(), field)
    targets = near_axis_coil_targets(solution, 0.03, subtract_plasma_field=False)
    np.testing.assert_array_equal(targets[1], solution.B_axis)
    residual = near_axis_coil_residuals(field, solution, 0.03, subtract_plasma_field=False)
    assert residual.shape == (21 * (3 + 9 + 27),)
    np.testing.assert_allclose(residual, 0, atol=1e-12)


def test_coil_targets_use_the_external_part_of_the_plasma_jet(monkeypatch):
    field = _coil_field()
    solution = _with_field_jet(circular_axis(), field)
    calls = []

    def plasma_hessian_on_axis(solution, formal_radius):
        calls.append(formal_radius)
        external = SimpleNamespace(external_field=solution.B_axis + 1.0, external_gradient=solution.grad_B_axis)
        return SimpleNamespace(field=external, external_hessian=solution.grad_grad_B_axis)

    plasma = ModuleType("pyqsc_jax.plasma")
    plasma.plasma_hessian_on_axis = plasma_hessian_on_axis
    monkeypatch.setitem(sys.modules, "pyqsc_jax", ModuleType("pyqsc_jax"))
    monkeypatch.setitem(sys.modules, "pyqsc_jax.plasma", plasma)
    _, B, _, _ = near_axis_coil_targets(solution, 0.03)
    np.testing.assert_allclose(B, solution.B_axis + 1.0)
    residual = near_axis_coil_residuals(field, solution, 0.03)
    np.testing.assert_allclose(residual[:21 * 3], -np.repeat(np.sqrt(1 / 21), 63) * 1.0, rtol=1e-12)
    np.testing.assert_allclose(residual[21 * 3:], 0, atol=1e-12)
    assert calls == [0.03, 0.03]
