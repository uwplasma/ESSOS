"""Coil designs and coil constraint rows for constrained coil and single-stage optimization.

:class:`CoilDesign` maps one design vector to coils and their differentiable ``BiotSavart`` field.
:class:`CoilLimits` turns coil lengths, curvatures and distances (:func:`~essos.objective_functions.coil_coil_distances`
and its siblings) into scaled inequality rows, and :func:`coil_constraint` makes such rows a SciPy
``NonlinearConstraint`` for constrained optimizers such as SLSQP.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from essos.coils import Coils, Curves
from essos.fields import BiotSavart
from essos.objective_functions import (coil_coil_distances, coil_mean_squared_curvature, coil_self_distances,
                                       coil_surface_distances)


class CoilDesign:
    """Design vector -> coils.

    ``x`` holds, in order, relative changes of the ``current_dofs`` base-coil currents
    (``current[i] = nominal[i] * (1 + x[k])``) and additive changes of every base coil's unscaled Cartesian
    Fourier coefficients in metres; ``x0 = 0`` is the nominal design.  Calling the design gives the
    ``BiotSavart`` field of :meth:`coils_from_x`.  ``scales`` (one per coordinate) default to 0.06 per relative
    current, 0.002 m per constant Fourier coefficient and ``0.002 / k**2`` m per coefficient of order ``k``.
    """

    def __init__(self, coils, *, current_dofs=(), scales=None):
        self.coefficients = np.asarray(coils.dofs_curves) / np.asarray(coils.curves.scaling)[None, None, :]
        self.currents = np.asarray(coils.dofs_currents_raw, dtype=float)
        # a fixed current normalization: the default (the mean |current|) of traced currents would be a tracer
        # in the coils' pytree metadata, which jit cannot hash
        self.currents_scale = float(np.mean(np.abs(self.currents)))
        self.current_dofs = tuple(int(i) for i in current_dofs)
        if len(set(self.current_dofs)) != len(self.current_dofs) or any(
                not 0 <= i < len(self.currents) or self.currents[i] == 0 for i in self.current_dofs):
            raise ValueError("current_dofs must be unique indices of base coils with nonzero currents")
        self.nfp, self.stellsym, self.n_segments = int(coils.nfp), bool(coils.stellsym), int(coils.n_segments)
        self.size = len(self.current_dofs) + self.coefficients.size
        self.x0 = np.zeros(self.size)
        order = (self.coefficients.shape[2] - 1) // 2
        modes = ["constant"] + [f"{kind}({k})" for k in range(1, order + 1) for kind in ("sin", "cos")]
        self.dof_names = tuple([f"current[{i}]/nominal" for i in self.current_dofs]
                               + [f"coil[{i}].{axis}.{mode}" for i in range(len(self.currents))
                                  for axis in "xyz" for mode in modes])
        if scales is None:
            mode_scales = [0.002] + [0.002 / k**2 for k in range(1, order + 1) for _ in range(2)]
            scales = np.r_[np.full(len(self.current_dofs), 0.06), np.tile(mode_scales, 3 * len(self.currents))]
        self.scales = np.asarray(scales, dtype=float)
        if self.scales.shape != self.x0.shape or not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("one positive finite scale per coordinate required")

    def base_currents_at(self, x):
        """Base-coil currents [A] at ``x``."""
        x = jnp.asarray(x)
        currents = jnp.asarray(self.currents)
        for local, base in enumerate(self.current_dofs):
            currents = currents.at[base].add(x[local] * self.currents[base])
        return currents

    def coils_from_x(self, x):
        """Coils at ``x``, without changing the nominal coils."""
        x = jnp.asarray(x)
        raw = jnp.asarray(self.coefficients) + x[len(self.current_dofs):].reshape(self.coefficients.shape)
        return Coils(Curves(raw, self.n_segments, self.nfp, self.stellsym), self.base_currents_at(x),
                     currents_scale=self.currents_scale)

    def __call__(self, x):
        """The differentiable filament field (``BiotSavart``) of :meth:`coils_from_x`."""
        return BiotSavart(self.coils_from_x(x))


def resampled(coils, points: int, *, symmetric: bool = True):
    """Curves of ``coils`` on ``points`` quadrature points; only the base curves unless ``symmetric``."""
    curves = coils.curves.copy()
    curves.n_segments = points
    if not symmetric:
        curves.nfp, curves.stellsym = 1, False
    return curves


@dataclass(frozen=True)
class CoilLimits:
    """Engineering limits of a coil set, as scaled inequality rows (feasible where every row is >= 0).

    Each row is ``(limit - margin - value) / limit`` (or ``(value - limit - margin) / limit`` for a
    distance): the margins keep a design that SLSQP holds only to its tolerance strictly inside the
    limits.  Curvature, its arc-length mean square and the length are sampled on ``curvature_points``
    points of each base coil, distances on ``distance_points`` points of every coil (the segment-exact
    :func:`~essos.objective_functions.coil_coil_distances`, ``coil_self_distances`` and
    ``coil_surface_distances``).  ``self_clearance`` (m) guards non-intersection, ``min_speed`` (m per unit
    angle) a degenerate parametrization.
    """

    length: float
    curvature: float
    mean_squared_curvature: float
    coil_distance: float
    length_margin: float = 1e-5
    curvature_margin: float = 0.10
    mean_squared_curvature_margin: float = 0.02
    distance_margin: float = 0.001
    self_clearance: float = 1e-6
    min_speed: float = 1e-4
    curvature_points: int = 512
    distance_points: int = 128

    def rows(self, coils, *, per_point: bool = False):
        """The coil rows: per base coil its length, peak curvature, mean squared curvature and minimum speed,
        and the coil-coil and self distances.  ``per_point`` gives the curvature and distances one row per
        sample point instead of their extremes: smooth where the extreme switches points, the smallest of
        them equal to the extreme row."""
        base = resampled(coils, self.curvature_points, symmetric=False)
        speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
        curves = resampled(coils, self.distance_points)
        coil, own = coil_coil_distances(curves), coil_self_distances(curves)
        curvature = base.curvature if per_point else jnp.max(base.curvature, axis=1)
        if not per_point:
            coil, own = jnp.min(coil), jnp.min(own)
        return jnp.concatenate((
            (self.length - self.length_margin - base.length) / self.length,
            jnp.ravel(self.curvature - self.curvature_margin - curvature) / self.curvature,
            (self.mean_squared_curvature - self.mean_squared_curvature_margin - coil_mean_squared_curvature(base))
            / self.mean_squared_curvature,
            jnp.ravel(coil - self.coil_distance - self.distance_margin) / self.coil_distance,
            jnp.ravel(own - self.self_clearance) / self.coil_distance,
            (jnp.min(speed, axis=1) - self.min_speed) / self.length))

    def surface_clearance(self, coils, surface):
        """Smallest distance [m] from the coils (every symmetry copy) to ``surface.gamma``."""
        return jnp.min(coil_surface_distances(resampled(coils, self.distance_points), surface))


def chunked_jacfwd(fun, chunk: int = 32):
    """Forward-mode Jacobian of ``fun`` pushing ``chunk`` input directions at a time: memory of many-row
    constraints (per-point coil distances) grows with the directions pushed at once."""

    def jac(x):
        n = x.size
        basis = jnp.eye(-(-n // chunk) * chunk, n, dtype=x.dtype).reshape(-1, chunk, n)
        columns = jax.lax.map(jax.vmap(lambda v: jax.jvp(fun, (x,), (v,))[1]), basis)
        return columns.reshape(-1, columns.shape[-1])[:n].T

    return jac


def coil_constraint(rows_from_x, *, forward_chunk: int = None):
    """SciPy ``NonlinearConstraint`` (rows >= 0) of pure-coil rows ``rows_from_x(x)``.

    The Jacobian is reverse mode, or with ``forward_chunk`` forward mode in chunks of that many directions
    (:func:`chunked_jacfwd`), for thousands of per-point rows.
    """
    from scipy.optimize import NonlinearConstraint

    fun = jax.jit(rows_from_x)
    jac = jax.jit(jax.jacrev(fun) if forward_chunk is None else chunked_jacfwd(fun, forward_chunk))
    return NonlinearConstraint(lambda x: np.asarray(fun(jnp.asarray(x))), 0, np.inf,
                               jac=lambda x: np.asarray(jac(jnp.asarray(x))))
