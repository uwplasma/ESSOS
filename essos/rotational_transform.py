"""Rotational transform of coil fields from weighted Birkhoff averages.

A field line is traced in the cylindrical angle, ``dR/dphi = R B_R/B_phi`` and
``dZ/dphi = R B_Z/B_phi``, with fixed-step RK4 alongside the magnetic axis.
Its poloidal angle about the axis advances by ``dtheta_n`` over field period
``n``; on a flux surface ``dtheta_n`` is a smooth function sampled along an
irrational rotation, so the weighted Birkhoff average

    iota = nfp / (2 pi) * sum_n w_n dtheta_n,   w_n ~ exp(-1 / (t (1 - t)))

converges faster than any power of the number of periods (Das et al. 2017,
Ruth & Bindel 2024), versus ``1/N`` for the plain average used in Poincare
winding-number estimates. The axis is the fixed point of the field-period map,
found by Newton iterations wrapped in :func:`solvax.root_solve`, so gradients
with respect to the coils use the implicit function theorem instead of
unrolling Newton. Everything is fixed-shape, jit-able and differentiable.
"""
import jax
import jax.numpy as jnp
from jax import lax
import solvax as sx

__all__ = ["field_period_map", "magnetic_axis", "rotational_transform"]


def _rhs(field, phi, RZ):
    """``d(R, Z)/dphi`` for any number of field lines, ``RZ`` of shape (..., 2)."""
    R, Z = RZ[..., 0], RZ[..., 1]
    c, s = jnp.cos(phi), jnp.sin(phi)
    xyz = jnp.stack([R * c, R * s, Z], axis=-1)
    B = jax.vmap(field.B)(xyz.reshape(-1, 3)).reshape(xyz.shape)
    BR = B[..., 0] * c + B[..., 1] * s
    Bphi = B[..., 1] * c - B[..., 0] * s
    return jnp.stack([R * BR / Bphi, R * B[..., 2] / Bphi], axis=-1)


def _rk4(field, phi, RZ, h):
    k1 = _rhs(field, phi, RZ)
    k2 = _rhs(field, phi + h / 2, RZ + h / 2 * k1)
    k3 = _rhs(field, phi + h / 2, RZ + h / 2 * k2)
    k4 = _rhs(field, phi + h, RZ + h * k3)
    return RZ + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def field_period_map(field, RZ, nfp, steps=64, phi0=0.0):
    """Map points ``RZ`` (..., 2) at ``phi0`` along field lines by ``2 pi / nfp``."""
    h = 2 * jnp.pi / nfp / steps
    step = lambda y, k: (_rk4(field, phi0 + k * h, y, h), None)
    return lax.scan(step, jnp.asarray(RZ, float), jnp.arange(steps))[0]


def magnetic_axis(field, RZ0, nfp, steps=64, newton_iterations=8, centroid_rounds=4, phi0=0.0):
    """Axis ``(R, Z)`` at ``phi0``: the fixed point of :func:`field_period_map`.

    The guess ``RZ0`` is first refined by replacing it ``centroid_rounds`` times
    with the centroid of its 24 Poincare iterates (robust far from the axis,
    where Newton on the strongly twisting map diverges), then polished by
    Newton. Differentiable via the implicit function theorem
    (:func:`solvax.root_solve`); the refinement is never differentiated.
    """
    period_map = lambda x: field_period_map(field, x, nfp, steps, phi0)
    residual = lambda x: period_map(x) - x

    def solver(f, x):
        centroid = lambda _, x: lax.scan(lambda y, _: (period_map(y), y), x, None, length=24)[1].mean(0)
        x = lax.fori_loop(0, centroid_rounds, centroid, x)
        return lax.fori_loop(0, newton_iterations,
                             lambda _, x: x - jnp.linalg.solve(jax.jacfwd(f)(x), f(x)), x)

    return sx.root_solve(residual, jnp.asarray(RZ0, float), solver)


def _birkhoff_weights(n):
    t = (jnp.arange(n) + 0.5) / n
    w = jnp.exp(-1.0 / (t * (1.0 - t)))
    return w / jnp.sum(w)


def rotational_transform(field, R0, Z0, nfp, axis=None, n_periods=64, steps=32,
                         weighted=True, phi0=0.0):
    """Rotational transform of the surfaces through the points ``(R0, Z0)`` at ``phi0``.

    Args:
        field: magnetic field with ``B(xyz)`` (e.g. :class:`essos.fields.BiotSavart`).
        R0, Z0: scalars or arrays of starting points, one field line each.
        nfp: number of field periods (toroidal symmetry used for the map).
        axis: ``(R, Z)`` of the magnetic axis at ``phi0``; if omitted it is
            found with :func:`magnetic_axis` from ``(field.r_axis, 0)``.
        n_periods: number of field periods to trace; iota converges
            super-polynomially in it when ``weighted`` is true.
        steps: RK4 steps per field period (4th-order error in ``1/steps``).
        weighted: use the smooth Birkhoff weights (else a plain average).

    Returns:
        iota with the shape of ``broadcast(R0, Z0)``.
    """
    if axis is None:
        axis = magnetic_axis(field, jnp.array([field.r_axis, 0.0]), nfp, steps, phi0=phi0)
    lines = jnp.stack(jnp.broadcast_arrays(jnp.asarray(R0, float), jnp.asarray(Z0, float)), -1)
    y0 = jnp.concatenate([jnp.asarray(axis, float)[None], lines.reshape(-1, 2)])
    h = 2 * jnp.pi / nfp / steps
    rel = lambda y: (y[1:, 0] - y[0, 0]) + 1j * (y[1:, 1] - y[0, 1])

    def rk_step(y, k):
        y_new = _rk4(field, phi0 + k * h, y, h)
        return y_new, jnp.angle(rel(y_new) / rel(y))  # small, unambiguous increments

    def period(y, n):
        y, dtheta = lax.scan(rk_step, y, n * steps + jnp.arange(steps))
        return y, jnp.sum(dtheta, axis=0)

    _, dtheta = lax.scan(period, y0, jnp.arange(n_periods))  # (n_periods, nlines)
    w = _birkhoff_weights(n_periods) if weighted else jnp.full(n_periods, 1.0 / n_periods)
    return (nfp / (2 * jnp.pi) * w @ dtheta).reshape(lines.shape[:-1])
