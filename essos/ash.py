"""Coefficient-driven thermal-helium core residence, not pumped exhaust.

The trace, local closure is Gamma=-D*dn/dr+V*n. Cell volumes and W=dV/dr
must describe the same physical radial coordinate. Profiles, electric field,
collision model and boundary assumptions belong to the coefficient provider.
This module neither infers them from a WOUT nor imports a kinetic solver.
All numerical functions are JIT/AD compatible; invalid values return a false
``valid`` flag and NaN physical outputs. Static shape mistakes raise ValueError.
Use :func:`check_helium_residence` at a host/reporting boundary.
"""
from typing import NamedTuple

import jax.numpy as jnp


class HeliumResidence(NamedTuple):
    density_m3: object
    occupation_s: object
    source_rate_s: object
    inventory: object
    cost: object
    mean_occupation_s: object
    mean_residence_s: object
    boundary_particle_rate_s: object
    forward_residual: object
    backward_residual: object
    duality_relative_error: object
    balance_relative_error: object
    minimum_relative_pivot: object
    maximum_peclet: object
    valid: object


def _bernoulli(x):
    """Stable x/expm1(x); safe inactive branches also have finite derivatives."""
    small = jnp.abs(x) < 1.e-4
    # Evaluate only a bounded nonzero argument in the ordinary branch.
    z = jnp.where(small, 1., jnp.clip(x, -50., 50.))
    ordinary = z / jnp.expm1(z)
    positive = jnp.maximum(x, 1.)
    negative = jnp.minimum(x, -1.)
    high = positive * jnp.exp(-positive) / (-jnp.expm1(-positive))
    low = -negative / (-jnp.expm1(negative))
    zsmall = jnp.where(small, x, 0.)
    series = 1 - zsmall/2 + zsmall*zsmall/12 - zsmall**4/720 + zsmall**6/30240
    return jnp.where(small, series,
                     jnp.where(x > 50., high, jnp.where(x < -50., low, ordinary)))


def evaluate_helium_residence(
    radial_edges_m, cell_volume_m3, volume_derivative_m2,
    diffusion_m2_s, convection_m_s, source_m3_s, *,
    boundary_kappa_m_s, core_cost=None,
):
    """Solve K*n=M*S and K.T*T=M*c with checked SOLVAX tridiagonal solves.

    Edges, W, D and V have shape (n+1,), volumes, S and dimensionless c (n,).
    At least two cells start at r=0; the axis reflects. ``boundary_kappa_m_s``
    is explicit: +inf absorbs; finite positive kappa gives Gamma=kappa*n at
    the outer face, including half-cell transport resistance. Zero kappa has
    no finite steady residence. Positive diffusivity is required without floors.

    ``occupation_s`` and ``mean_occupation_s`` use c; ``mean_residence_s`` is
    unweighted inventory/source. Validity qualifies this discrete solve only:
    pivot, RHS residual, positivity, duality and balance checks do not bound
    continuum error or certify a kinetic closure or its geometry derivatives.
    Gradients include every supplied coefficient/volume/source/cost dependence;
    host loading and an unprovided electric-field/root response are outside it.
    """
    try:
        from solvax import tridiagonal_factor, tridiagonal_solve_factored_checked
    except ImportError as error:
        raise ImportError("Helium residence requires ESSOS's optional ash extra: pip install 'essos[ash]'") from error

    e, m, w, d, v, s = (jnp.asarray(x, dtype=jnp.result_type(float, jnp.asarray(x))) for x in
                        (radial_edges_m, cell_volume_m3, volume_derivative_m2,
                         diffusion_m2_s, convection_m_s, source_m3_s))
    if e.ndim != 1 or e.size < 3:
        raise ValueError("Require at least two radial cells")
    n = e.size - 1
    c = jnp.ones_like(m) if core_cost is None else jnp.asarray(core_cost)
    kappa = jnp.asarray(boundary_kappa_m_s)
    if any(x.shape != (n+1,) for x in (w, d, v)) or any(x.shape != (n,) for x in (m, s, c)) or kappa.ndim != 0:
        raise ValueError("Edges/faces must have n+1 entries, cells n, and kappa be scalar")
    input_ok = ((e[0] == 0) & jnp.all(jnp.diff(e) > 0) &
                jnp.all(m > 0) & (w[0] >= 0) & jnp.all(w[1:] > 0) &
                jnp.all(d > 0) & jnp.all(s >= 0) & jnp.all(c >= 0) &
                (kappa > 0) & ~jnp.isnan(kappa))
    for x in (e, m, w, d, v, s, c):
        input_ok &= jnp.all(jnp.isfinite(x))
    centers = (e[1:] + e[:-1])/2
    h = jnp.diff(centers)
    pe = v[1:-1]*h/d[1:-1]
    g = w[1:-1]*d[1:-1]/h
    out, inc = g*_bernoulli(-pe), g*_bernoulli(pe)
    lo = jnp.concatenate((jnp.zeros(1), -out))
    up = jnp.concatenate((-inc, jnp.zeros(1)))
    diag = jnp.concatenate((out, jnp.zeros(1))) + jnp.concatenate((jnp.zeros(1), inc))
    hb = e[-1] - centers[-1]
    pb = v[-1]*hb/d[-1]
    gb = w[-1]*d[-1]/hb
    finite_kappa = jnp.where(jnp.isposinf(kappa), 1., kappa)
    inverse_kappa = jnp.where(jnp.isposinf(kappa), 0., 1/finite_kappa)
    escape = gb*_bernoulli(-pb)/(1 + gb*_bernoulli(pb)*inverse_kappa/w[-1])
    diag = diag.at[-1].add(escape)
    b, weight = m*s, m*c
    rate = jnp.sum(b)
    input_ok &= (rate > 0) & jnp.isfinite(escape) & (escape > 0)
    f = tridiagonal_factor(lo, diag, up)
    forward = tridiagonal_solve_factored_checked(f, b, fallback="nan")
    lot = jnp.concatenate((jnp.zeros(1), up[:-1]))
    upt = jnp.concatenate((lo[1:], jnp.zeros(1)))
    ft = tridiagonal_factor(lot, diag, upt)
    backward = tridiagonal_solve_factored_checked(ft, weight, fallback="nan")
    density, occupation = forward.solution, backward.solution

    def residual(lower, upper, x, rhs):
        applied = diag*x
        applied = applied.at[1:].add(lower[1:]*x[:-1])
        applied = applied.at[:-1].add(upper[:-1]*x[1:])
        scale = jnp.max(jnp.abs(rhs))
        return jnp.max(jnp.abs(applied-rhs))/jnp.where(scale > 0, scale, 1.)

    rf = residual(lo, up, density, b)
    rb = residual(lot, upt, occupation, weight)
    inventory = m @ density
    cost, dual = weight @ density, b @ occupation
    flux = escape*density[-1]
    duality = jnp.abs(cost-dual)/jnp.maximum(jnp.abs(cost), jnp.finfo(diag.dtype).tiny)
    balance = jnp.abs(flux-rate)/rate
    mean_occupation, mean_residence = cost/rate, inventory/rate
    physical_scalars = jnp.stack((rate, inventory, cost, mean_occupation, mean_residence, flux))
    tolerance = 2.e-8
    valid = (input_ok & jnp.all(jnp.isfinite(physical_scalars)) &
             jnp.all(jnp.isfinite(density)) & jnp.all(jnp.isfinite(occupation)) &
             jnp.all(density >= 0) & jnp.all(occupation >= 0) &
             (rf <= tolerance) & (rb <= tolerance) &
             (duality <= tolerance) & (balance <= tolerance))
    mask = lambda x: jnp.where(valid, x, jnp.nan)
    return HeliumResidence(
        mask(density), mask(occupation), mask(rate), mask(inventory), mask(cost),
        mask(mean_occupation), mask(mean_residence), mask(flux), rf, rb, duality, balance,
        jnp.minimum(forward.diagnostics.minimum_relative_pivot,
                    backward.diagnostics.minimum_relative_pivot),
        jnp.maximum(jnp.max(jnp.abs(pe)), jnp.abs(pb)), valid)


def check_helium_residence(result):
    """Host-only failure boundary; return a valid result or raise ArithmeticError."""
    if not bool(result.valid):
        raise ArithmeticError("Invalid helium inputs or unqualified radial solve; inspect residuals, balance and pivots")
    return result


class RestrictedResidence(NamedTuple):
    restricted_mean_s: object
    escaped_fraction: object
    failed_fraction: object
    standard_error_s: object
    effective_sample_size: object
    no_censoring_in_sample: object
    valid: object


def restricted_residence(exit_times_s, horizon_s, *, weights=None, failed=None):
    """Common-horizon E[min(T,H)], with +inf for physical survivors.

    Numerical failures invalidate the estimate, even at zero marker weight.
    This is not a full mean or a fitted tail. Standard error assumes independent
    identically distributed markers with fixed weights, using the weighted sample variance; it does not
    account for uncertain coefficients or correlated/common-noise cohorts.
    """
    t = jnp.asarray(exit_times_s, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("Require a nonempty vector of first-exit times")
    h = jnp.asarray(horizon_s)
    w = jnp.ones_like(t) if weights is None else jnp.asarray(weights, dtype=float)
    f = jnp.zeros_like(t, dtype=bool) if failed is None else jnp.asarray(failed, dtype=bool)
    if w.shape != t.shape or f.shape != t.shape or h.ndim != 0:
        raise ValueError("Times, weights and failures must match; horizon must be scalar")
    # Scaling before summation preserves normalized weights even when their
    # finite raw sum overflows. Invalid weights are still rejected below.
    largest_weight = jnp.max(w)
    safe_largest = jnp.where(jnp.isfinite(largest_weight) & (largest_weight > 0), largest_weight, 1.)
    # Use a power-of-two scale: direct division by a huge maximum may be
    # lowered to a reciprocal that underflows/flushes to zero under XLA.
    exponent = jnp.frexp(safe_largest)[1]
    scaled_weights = jnp.ldexp(w, -exponent)
    total = jnp.sum(scaled_weights)
    valid = (jnp.isfinite(h) & (h > 0) & jnp.all(t >= 0) &
             jnp.all(~jnp.isnan(t)) & jnp.all((t <= h) | jnp.isposinf(t)) &
             jnp.all(jnp.isfinite(w)) & jnp.all(w >= 0) & jnp.isfinite(total) & (total > 0) & ~jnp.any(f))
    w = scaled_weights/jnp.where(jnp.isfinite(total) & (total > 0), total, 1.)
    x = jnp.minimum(t, h)
    mean = w @ x
    sumsq = w @ w
    se = jnp.sqrt(sumsq*jnp.sum(w*(x-mean)**2)/jnp.where(sumsq < 1, 1-sumsq, jnp.nan))
    mask = lambda x: jnp.where(valid, x, jnp.nan)
    return RestrictedResidence(mask(mean), mask(w @ jnp.isfinite(t)), w @ f,
                               mask(se), 1/sumsq, valid & jnp.all(jnp.isfinite(t)), valid)


class TraceResponse(NamedTuple):
    diffusion_m2_s: object
    convection_m_s: object
    fit_defect_m_s: object
    valid: object


def trace_response(log_gradients_m_inv, flux_over_density_m_s, *, relative_tolerance=1.e-7):
    """Fit Gamma/n=V-D*d(log n)/dr; >=3 distinct probes test affine response.

    Providers must match all other profiles, kinetic models and Er across probes.
    This diagnostic alone cannot qualify trace abundance or full-FP physics.
    """
    x, y = map(jnp.asarray, (log_gradients_m_inv, flux_over_density_m_s))
    if x.ndim != 1 or x.size < 3 or y.shape != x.shape:
        raise ValueError("Require at least three paired gradient/flux probes")
    xc, yc = x-jnp.mean(x), y-jnp.mean(y)
    slope = (xc @ yc)/(xc @ xc)
    drift = jnp.mean(y)-slope*jnp.mean(x)
    defect = jnp.max(jnp.abs(y-(drift+slope*x)))
    distinct = 1 + jnp.sum(jnp.diff(jnp.sort(x)) > 0)
    valid = (jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(y)) & (distinct >= 3) &
             (jnp.isfinite(relative_tolerance)) & (relative_tolerance >= 0) & (slope < 0) & jnp.isfinite(slope) &
             (defect <= relative_tolerance*jnp.maximum(jnp.max(jnp.abs(y)), 1.e-12)))
    return TraceResponse(jnp.where(valid, -slope, jnp.nan),
                         jnp.where(valid, drift, jnp.nan), defect, valid)


def convection_from_marker_drift(drift_m_s, diffusion_m2_s,
                                 diffusion_gradient_m_s, log_volume_gradient_m_inv):
    """V=A-D'-D*d(log W)/dr away from the axis, in the same physical r.

    W'/W is singular at a regular axis; use conservative fluxes there.
    """
    return (jnp.asarray(drift_m_s) - jnp.asarray(diffusion_gradient_m_s) -
            jnp.asarray(diffusion_m2_s)*jnp.asarray(log_volume_gradient_m_inv))
