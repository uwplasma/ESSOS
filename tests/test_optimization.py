import sys
import types

import jax.numpy as jnp
import numpy as np
import pytest

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.optimization import new_nearaxis_from_x_and_old_nearaxis, optimize_loss_function
from essos.surfaces import SurfaceRZFourier


def _coils():
    return Coils(CreateEquallySpacedCurves(2, 1, 1.0, 0.3, 16, 1, False), jnp.array([1e5, 2e5]))


def _residual(target, n_extra=0):
    """Linear residual x - target: its least-squares solution is exactly ``target``."""
    def fun(x, dofs_curves, currents_scale, nfp, n_segments, stellsym, **kwargs):
        return x - target
    return fun


def test_optimize_loss_function_solves_a_linear_least_squares_problem_for_every_dof_layout(monkeypatch):
    """The residual x - t is linear, so one finite-difference Newton step lands on t for each dof layout.

    Starts are shifted away from zero: the optimiser differentiates with a step relative to |x|, so
    entries that are exactly (or nearly) zero do not move."""
    coils = _coils()
    ncv, cs = coils.dofs_curves.size, float(coils.currents_scale)
    kw = dict(tolerance_optimization=1e-12, maximum_function_evaluations=10)

    def check_coils(new, t):
        np.testing.assert_allclose(new.dofs_curves.ravel(), t[:ncv], atol=1e-6)
        np.testing.assert_allclose(new.dofs_currents_raw, t[ncv:ncv + 2] * cs, rtol=1e-6)

    x0 = coils.x + 0.3
    t = x0 + 0.05
    check_coils(optimize_loss_function(_residual(t), x0, coils, **kw), t)

    # coils + surface: the trailing dofs are written back into a fresh surface
    surface = SurfaceRZFourier(jnp.array([1.0, 0.2]), jnp.array([0.0, 0.2]), nfp=1, mpol=1, ntor=0, ntheta=8, nphi=8)
    x0 = jnp.concatenate([coils.x, surface.x]) + 0.3
    t = x0 + 0.05
    new_coils, new_surface = optimize_loss_function(_residual(t), x0, coils, surface_all=surface, **kw)
    check_coils(new_coils, t)
    np.testing.assert_allclose(new_surface.dofs, t[ncv + 2:], atol=1e-6)

    # near-axis field: the last dofs are rc, zs, etabar handed to near_axis with the old scalar settings
    seen = {}
    fake = types.SimpleNamespace(near_axis=lambda **kw_: seen.update(kw_) or types.SimpleNamespace(**kw_))
    monkeypatch.setitem(sys.modules, "pyqsc_jax", types.ModuleType("pyqsc_jax"))
    monkeypatch.setitem(sys.modules, "pyqsc_jax.near_axis", fake)
    axis = types.SimpleNamespace(rc=jnp.array([1.0, 0.1]), zs=jnp.array([0.0, -0.1]),
                                 x=jnp.array([1.0, 0.1, 0.0, -0.1, -0.9]),
                                 B0=1.0, sigma0=0.0, I2=0.0, nphi=31, spsi=1, sG=1, nfp=3)
    x0 = jnp.concatenate([coils.x, axis.x]) + 0.3
    t = x0 + 0.05
    new_coils, new_axis = optimize_loss_function(_residual(t), x0, coils, field_nearaxis=axis, **kw)
    check_coils(new_coils, t)
    np.testing.assert_allclose(new_axis.rc, t[ncv + 2:ncv + 4], atol=1e-6)
    np.testing.assert_allclose(new_axis.zs, t[ncv + 4:ncv + 6], atol=1e-6)
    assert seen["nphi"] == 31 and seen["nfp"] == 3 and abs(float(seen["etabar"]) - float(t[-1])) < 1e-6

    # surface and near-axis together
    x0 = jnp.concatenate([coils.x, surface.x, axis.x]) + 0.3
    t = x0 + 0.05
    new_coils, new_surface, new_axis = optimize_loss_function(
        _residual(t), x0, coils, surface_all=surface, field_nearaxis=axis, **kw)
    check_coils(new_coils, t)
    ns = surface.x.size
    np.testing.assert_allclose(new_surface.dofs, t[ncv + 2:ncv + 2 + ns], atol=1e-6)
    np.testing.assert_allclose(new_axis.rc, t[ncv + 2 + ns:ncv + 4 + ns], atol=1e-6)

    # a dof vector matching no layout is reported and returns None
    assert optimize_loss_function(_residual(jnp.zeros(ncv + 5)), jnp.ones(ncv + 5), coils,
                                  maximum_function_evaluations=2) is None


def test_near_axis_dependency_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyqsc_jax", None)  # import raises ImportError
    with pytest.raises(ImportError, match="pyQSC_JAX"):
        new_nearaxis_from_x_and_old_nearaxis(jnp.zeros(3), types.SimpleNamespace(rc=[0, 0], zs=[0]))
