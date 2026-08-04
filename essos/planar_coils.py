"""Planar Fourier curve coordinates compatible with :mod:`essos.coils`.

``PlanarXYCurves`` stores each independent curve in a local XY plane.  The
plane origin is a Cartesian center and its orientation is a scalar-first
quaternion ``[w, x, y, z]`` that rotates local vectors into Cartesian space.
The local Fourier coefficients omit the DC term because translation is
represented only by ``centers``.

The coefficient ordering follows :class:`essos.coils.Curves` after its DC
entry: ``[sin(1), cos(1), sin(2), cos(2), ...]``.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import tree_util

from essos.coils import Coils, Coils_from_json, Curves, apply_symmetries_to_currents

__all__ = [
    "PlanarCoils",
    "PlanarXYCurves",
    "load_coils_json",
    "load_simsopt_coils_json",
]


def _positive_integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        comparison = "greater than 2" if minimum == 3 else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return result


def _real_array(value: Any, *, name: str) -> jnp.ndarray:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.bool_):
        raise TypeError(f"{name} must be real-valued, not boolean")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be real-valued")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.asarray(0.0).dtype)
    return array


def _simsopt_planar_dof_names(order: int) -> tuple[str, ...]:
    return tuple(
        [*(f"rc({mode})" for mode in range(order + 1))]
        + [*(f"rs({mode})" for mode in range(1, order + 1))]
        + ["q0", "qi", "qj", "qk", "X", "Y", "Z"]
    )


def _simsopt_xyz_dof_names(order: int) -> tuple[str, ...]:
    names = []
    for axis in "xyz":
        names.append(f"{axis}c(0)")
        for mode in range(1, order + 1):
            names.extend((f"{axis}s({mode})", f"{axis}c({mode})"))
    return tuple(names)


def _normalize_quaternions(quaternions: jnp.ndarray) -> jnp.ndarray:
    """Normalize scalar-first quaternions without zero/overflow singularities."""

    scale = jnp.max(jnp.abs(quaternions), axis=-1, keepdims=True)
    valid_scale = jnp.isfinite(scale) & (scale > 0)
    scaled = quaternions / jnp.where(valid_scale, scale, 1)
    scaled_norm = jnp.linalg.norm(scaled, axis=-1, keepdims=True)
    valid = valid_scale & jnp.isfinite(scaled_norm) & (scaled_norm > 0)
    normalized = scaled / jnp.where(valid, scaled_norm, 1)
    identity = jnp.zeros_like(normalized).at[..., 0].set(1)
    return jnp.where(valid, normalized, identity)


def _quaternion_rotation_matrices(quaternions: jnp.ndarray) -> jnp.ndarray:
    """Return active local-to-global matrices for scalar-first quaternions."""

    # JAX reconstructs custom pytrees through ``tree_unflatten`` without
    # calling ``__init__``. Normalize at physical use so traced leaves retain
    # the same scale-invariant orientation semantics as directly constructed
    # ``PlanarXYCurves`` objects.
    quaternions = _normalize_quaternions(quaternions)
    w, x, y, z = jnp.moveaxis(quaternions, -1, 0)
    return jnp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


@tree_util.register_pytree_node_class
class PlanarXYCurves:
    """Independent planar curves represented by local X/Y Fourier series.

    Parameters
    ----------
    centers : array-like, shape (n_curves, 3)
        Cartesian DC positions of the independent curves.
    quaternions : array-like, shape (n_curves, 4)
        Scalar-first quaternions ``[w, x, y, z]`` rotating the local XY plane
        into Cartesian space. Nonzero finite values are normalized. A zero or
        nonfinite quaternion uses the identity orientation without producing
        NaNs.
    xy_dofs : array-like, shape (n_curves, 2, 2*order)
        Local X/Y Fourier coefficients with ordering
        ``[sin(1), cos(1), ..., sin(order), cos(order)]``. There is no local DC
        entry; use ``centers`` for translation.
    n_segments : int
        Number of points used by the delegated :class:`essos.coils.Curves`.
    nfp : int
        Number of rotational field periods.
    stellsym : bool
        Whether to add the stellarator-symmetric copies used by ESSOS.
    """

    def __init__(
        self,
        centers: Any,
        quaternions: Any,
        xy_dofs: Any,
        n_segments: int = 100,
        nfp: int = 1,
        stellsym: bool = True,
    ) -> None:
        centers = _real_array(centers, name="centers")
        quaternions = _real_array(quaternions, name="quaternions")
        xy_dofs = _real_array(xy_dofs, name="xy_dofs")

        if centers.ndim != 2 or centers.shape[1:] != (3,):
            raise ValueError(
                f"centers must have shape (n_curves, 3), got {centers.shape}"
            )
        if quaternions.ndim != 2 or quaternions.shape[1:] != (4,):
            raise ValueError(
                f"quaternions must have shape (n_curves, 4), got {quaternions.shape}"
            )
        if xy_dofs.ndim != 3 or xy_dofs.shape[1] != 2:
            raise ValueError(
                f"xy_dofs must have shape (n_curves, 2, 2*order), got {xy_dofs.shape}"
            )
        if xy_dofs.shape[2] < 2 or xy_dofs.shape[2] % 2:
            raise ValueError(
                "xy_dofs must contain an even, nonzero number of Fourier coefficients"
            )
        if not (centers.shape[0] == quaternions.shape[0] == xy_dofs.shape[0]):
            raise ValueError(
                "centers, quaternions, and xy_dofs must have the same number of curves"
            )
        if centers.shape[0] == 0:
            raise ValueError("at least one planar curve is required")
        if not isinstance(stellsym, bool):
            raise TypeError("stellsym must be a boolean")

        dtype = jnp.result_type(centers.dtype, quaternions.dtype, xy_dofs.dtype)
        self._centers = centers.astype(dtype)
        self._quaternions = _normalize_quaternions(quaternions.astype(dtype))
        self._xy_dofs = xy_dofs.astype(dtype)
        self._n_segments = _positive_integer(n_segments, name="n_segments", minimum=3)
        self._nfp = _positive_integer(nfp, name="nfp", minimum=1)
        self._stellsym = stellsym

    @property
    def centers(self) -> jnp.ndarray:
        """Cartesian DC positions of the independent curves."""

        return self._centers

    @property
    def quaternions(self) -> jnp.ndarray:
        """Scalar-first local-to-global quaternion coordinates.

        Directly constructed objects store normalized values. Physical rotation
        matrices normalize again because JAX reconstructs traced pytree leaves
        without calling the constructor.
        """

        return self._quaternions

    @property
    def xy_dofs(self) -> jnp.ndarray:
        """Local non-DC X/Y Fourier coefficients."""

        return self._xy_dofs

    @property
    def n_segments(self) -> int:
        return self._n_segments

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def stellsym(self) -> bool:
        return self._stellsym

    @property
    def order(self) -> int:
        return self.xy_dofs.shape[2] // 2

    @property
    def n_base_curves(self) -> int:
        return self.centers.shape[0]

    @property
    def rotation_matrices(self) -> jnp.ndarray:
        """Active matrices mapping local vectors to Cartesian vectors."""

        return _quaternion_rotation_matrices(self.quaternions)

    @property
    def normals(self) -> jnp.ndarray:
        """Cartesian unit normals of the independent curve planes."""

        return self.rotation_matrices[..., 2]

    @property
    def frames(self) -> jnp.ndarray:
        """Orthonormal local XY frame vectors in Cartesian coordinates."""

        return self.rotation_matrices[..., :2]

    def as_xyz_dofs(self) -> jnp.ndarray:
        """Return independent-curve coefficients accepted by ``Curves``.

        Returns
        -------
        jax.Array, shape (n_curves, 3, 2*order+1)
            Cartesian coefficients ordered as
            ``[DC, sin(1), cos(1), ..., sin(order), cos(order)]``.
        """

        local_dofs = jnp.concatenate(
            (
                self.xy_dofs,
                jnp.zeros(
                    (self.n_base_curves, 1, 2 * self.order),
                    dtype=self.xy_dofs.dtype,
                ),
            ),
            axis=1,
        )
        xyz_non_dc = jnp.einsum("nij,njk->nik", self.rotation_matrices, local_dofs)
        return jnp.concatenate((self.centers[..., None], xyz_non_dc), axis=-1)

    def as_curves(self) -> Curves:
        """Return the equivalent ESSOS ``Curves`` object."""

        return Curves(
            self.as_xyz_dofs(),
            n_segments=self.n_segments,
            nfp=self.nfp,
            stellsym=self.stellsym,
        )

    @property
    def gamma(self) -> jnp.ndarray:
        """Symmetry-expanded sampled curve positions."""

        return self.as_curves().gamma

    @property
    def tangent(self) -> jnp.ndarray:
        """Symmetry-expanded derivatives with respect to ESSOS quadpoints."""

        return self.as_curves().gamma_dash

    @property
    def gamma_dash(self) -> jnp.ndarray:
        """Alias for :attr:`tangent`, matching ``Curves``."""

        return self.tangent

    @property
    def gamma_dashdash(self) -> jnp.ndarray:
        """Symmetry-expanded second curve derivatives."""

        return self.as_curves().gamma_dashdash

    @property
    def curvature(self) -> jnp.ndarray:
        """Symmetry-expanded pointwise curvature delegated to ``Curves``."""

        return self.as_curves().curvature

    @property
    def length(self) -> jnp.ndarray:
        """Symmetry-expanded curve lengths delegated to ``Curves``."""

        return self.as_curves().length

    def to_coils(self, currents: Any) -> Coils:
        """Attach physical base-coil currents using the existing ``Coils`` API."""

        return Coils(self.as_curves(), currents)

    def to_simsopt_xyz(self) -> list[Any]:
        """Export independent curves as SIMSOPT ``CurveXYZFourier`` objects.

        The returned curves remain geometrically planar, but SIMSOPT receives
        the unrestricted XYZ representation. Conversion to SIMSOPT's radial
        ``CurvePlanarFourier`` requires a separate polar reparameterization and
        is intentionally not performed silently. Symmetry copies are not
        generated, so callers can apply SIMSOPT symmetry exactly once.
        """

        from simsopt.geo import CurveXYZFourier

        curves = []
        for dofs in np.asarray(self.as_xyz_dofs()):
            curve = CurveXYZFourier(self.n_segments, self.order)
            curve.x = dofs.reshape(curve.x.shape)
            curves.append(curve)
        return curves

    @classmethod
    def from_polar_radius(
        cls,
        centers: Any,
        quaternions: Any,
        radius_dofs: Any,
        n_segments: int = 100,
        nfp: int = 1,
        stellsym: bool = True,
    ) -> "PlanarXYCurves":
        """Create planar XY curves from a polar-radius Fourier series.

        ``radius_dofs`` has shape ``(n_curves, 2*radius_order+1)`` and uses the
        ESSOS ordering ``[DC, sin(1), cos(1), ...]``. Multiplication by
        ``cos(theta)`` and ``sin(theta)`` increases the XY order by one. Any
        resulting local XY DC is rotated into ``centers``, so the returned
        object still has exactly one source of translation.
        """

        centers_array = _real_array(centers, name="centers")
        quaternions_array = _real_array(quaternions, name="quaternions")
        radius_dofs = _real_array(radius_dofs, name="radius_dofs")
        if centers_array.ndim != 2 or centers_array.shape[1:] != (3,):
            raise ValueError(
                f"centers must have shape (n_curves, 3), got {centers_array.shape}"
            )
        if quaternions_array.ndim != 2 or quaternions_array.shape[1:] != (4,):
            raise ValueError(
                "quaternions must have shape (n_curves, 4), "
                f"got {quaternions_array.shape}"
            )
        if radius_dofs.ndim != 2 or radius_dofs.shape[1] % 2 != 1:
            raise ValueError(
                "radius_dofs must have shape (n_curves, 2*order+1), "
                f"got {radius_dofs.shape}"
            )
        if not (
            centers_array.shape[0] == quaternions_array.shape[0] == radius_dofs.shape[0]
        ):
            raise ValueError(
                "centers, quaternions, and radius_dofs must have the same "
                "number of curves"
            )

        dtype = jnp.result_type(
            centers_array.dtype, quaternions_array.dtype, radius_dofs.dtype
        )
        centers_array = centers_array.astype(dtype)
        quaternions_array = _normalize_quaternions(quaternions_array.astype(dtype))
        radius_dofs = radius_dofs.astype(dtype)

        radius_order = radius_dofs.shape[1] // 2
        xy_order = radius_order + 1
        sample_count = 2 * xy_order + 1
        theta = 2 * jnp.pi * jnp.arange(sample_count, dtype=dtype) / sample_count

        radius = jnp.broadcast_to(
            radius_dofs[:, :1], (radius_dofs.shape[0], sample_count)
        )
        if radius_order:
            radius_modes = jnp.arange(1, radius_order + 1, dtype=dtype)
            phase = radius_modes[:, None] * theta[None, :]
            radius = radius + jnp.einsum(
                "nk,ks->ns", radius_dofs[:, 1::2], jnp.sin(phase)
            )
            radius = radius + jnp.einsum(
                "nk,ks->ns", radius_dofs[:, 2::2], jnp.cos(phase)
            )

        local_xy = jnp.stack((radius * jnp.cos(theta), radius * jnp.sin(theta)), axis=1)
        local_dc = jnp.mean(local_xy, axis=-1)
        xy_modes = jnp.arange(1, xy_order + 1, dtype=dtype)
        xy_phase = xy_modes[:, None] * theta[None, :]
        sin_dofs = (2 / sample_count) * jnp.einsum(
            "ncs,ks->nck", local_xy, jnp.sin(xy_phase)
        )
        cos_dofs = (2 / sample_count) * jnp.einsum(
            "ncs,ks->nck", local_xy, jnp.cos(xy_phase)
        )
        xy_dofs = jnp.zeros((radius_dofs.shape[0], 2, 2 * xy_order), dtype=dtype)
        xy_dofs = xy_dofs.at[..., 0::2].set(sin_dofs)
        xy_dofs = xy_dofs.at[..., 1::2].set(cos_dofs)

        rotations = _quaternion_rotation_matrices(quaternions_array)
        local_shift = jnp.concatenate(
            (
                local_dc,
                jnp.zeros((local_dc.shape[0], 1), dtype=local_dc.dtype),
            ),
            axis=1,
        )
        shifted_centers = centers_array + jnp.einsum(
            "nij,nj->ni", rotations, local_shift
        )
        return cls(
            shifted_centers,
            quaternions_array,
            xy_dofs,
            n_segments=n_segments,
            nfp=nfp,
            stellsym=stellsym,
        )

    @classmethod
    def from_simsopt_planar(
        cls,
        simsopt_curves: Any,
        *,
        n_segments: int | None = None,
        nfp: int = 1,
        stellsym: bool = False,
    ) -> "PlanarXYCurves":
        """Import independent SIMSOPT planar curves without a hard dependency.

        Both ``CurvePlanarFourier`` and ``JaxCurvePlanarFourier`` expose their
        complete coordinates through ``local_full_x``. ``curve.x`` is never
        used because it omits fixed degrees of freedom. Positional parsing is
        guarded by ``local_full_dof_names`` so an upstream layout change fails
        loudly instead of corrupting geometry. Mixed radial orders are padded
        exactly to the largest order. With ``n_segments=None``, every source
        curve must already use ESSOS's canonical uniform quadrature; passing
        ``n_segments`` explicitly samples the Fourier geometry on that grid.

        SIMSOPT objects do not encode field-period or stellarator-symmetry
        metadata, so imports default to one period without stellarator
        symmetry. Nonzero quaternion norms below ``1e-8`` are rejected because
        SIMSOPT's JAX planar curve does not apply a unit-quaternion rotation in
        that regime.
        """

        try:
            curves = tuple(simsopt_curves)
        except TypeError as exc:
            raise TypeError(
                "simsopt_curves must be an iterable of base curves"
            ) from exc
        if not curves:
            raise ValueError("at least one SIMSOPT planar curve is required")

        orders = []
        full_dofs = []
        for index, curve in enumerate(curves):
            if not hasattr(curve, "order"):
                raise TypeError(f"SIMSOPT curve {index} has no order attribute")
            order = operator.index(curve.order)
            if order < 0:
                raise ValueError(f"SIMSOPT curve {index} has invalid order {order}")
            expected_names = _simsopt_planar_dof_names(order)
            if not hasattr(curve, "local_full_dof_names"):
                raise TypeError(
                    f"SIMSOPT curve {index} has no local_full_dof_names contract"
                )
            names = tuple(curve.local_full_dof_names)
            if names != expected_names:
                raise ValueError(
                    "unsupported SIMSOPT planar DOF layout for curve "
                    f"{index}: expected {expected_names}, got {names}"
                )
            if not hasattr(curve, "local_full_x"):
                raise TypeError(f"SIMSOPT curve {index} has no local_full_x contract")
            dofs = np.asarray(curve.local_full_x)
            if dofs.shape != (2 * order + 8,):
                raise ValueError(
                    f"SIMSOPT curve {index} has {dofs.shape}, expected "
                    f"{(2 * order + 8,)}"
                )
            if not np.issubdtype(dofs.dtype, np.floating) or not np.all(
                np.isfinite(dofs)
            ):
                raise ValueError(
                    f"SIMSOPT curve {index} full DOFs must be finite real floats"
                )
            orders.append(order)
            full_dofs.append(dofs)
        common_order = max(orders)
        common_dtype = np.result_type(*(dofs.dtype for dofs in full_dofs))
        radius_dofs = np.zeros((len(curves), 2 * common_order + 1), dtype=common_dtype)
        quaternions = np.zeros((len(curves), 4), dtype=common_dtype)
        centers = np.zeros((len(curves), 3), dtype=common_dtype)
        for index, (order, dofs) in enumerate(zip(orders, full_dofs, strict=True)):
            radius_dofs[index, 0] = dofs[0]
            for mode in range(1, order + 1):
                radius_dofs[index, 2 * mode] = dofs[mode]
                radius_dofs[index, 2 * mode - 1] = dofs[order + mode]
            quaternion_start = 2 * order + 1
            quaternions[index] = dofs[quaternion_start : quaternion_start + 4]
            centers[index] = dofs[quaternion_start + 4 : quaternion_start + 7]

        quaternion_scales = np.max(np.abs(quaternions), axis=1)
        quaternion_norms = np.zeros_like(quaternion_scales)
        nonzero = quaternion_scales > 0
        quaternion_norms[nonzero] = quaternion_scales[nonzero] * np.linalg.norm(
            quaternions[nonzero] / quaternion_scales[nonzero, None], axis=1
        )
        tiny_nonzero = nonzero & (quaternion_norms < 1.0e-8)
        if np.any(tiny_nonzero):
            indices = tuple(np.flatnonzero(tiny_nonzero).tolist())
            raise ValueError(
                "SIMSOPT planar curves with nonzero quaternion norm below "
                f"1e-8 cannot be imported exactly; offending curves: {indices}"
            )

        if n_segments is None:
            if not hasattr(curves[0], "quadpoints"):
                raise TypeError(
                    "n_segments is required when SIMSOPT curves expose no quadpoints"
                )
            segments = len(curves[0].quadpoints)
            if any(
                not hasattr(curve, "quadpoints") or len(curve.quadpoints) != segments
                for curve in curves
            ):
                raise ValueError("all SIMSOPT curves must share one quadrature size")
            for index, curve in enumerate(curves):
                quadpoints = np.asarray(curve.quadpoints)
                canonical = np.linspace(0.0, 1.0, segments, endpoint=False)
                if (
                    quadpoints.ndim != 1
                    or not np.issubdtype(quadpoints.dtype, np.floating)
                    or not np.all(np.isfinite(quadpoints))
                    or not np.allclose(quadpoints, canonical, rtol=0.0, atol=1.0e-14)
                ):
                    raise ValueError(
                        "SIMSOPT curve "
                        f"{index} uses noncanonical quadpoints; pass n_segments "
                        "explicitly to sample its Fourier geometry on the ESSOS grid"
                    )
        else:
            segments = n_segments
        return cls.from_polar_radius(
            centers,
            quaternions,
            radius_dofs,
            n_segments=segments,
            nfp=nfp,
            stellsym=stellsym,
        )

    def tree_flatten(self):
        children = (self.centers, self.quaternions, self.xy_dofs)
        metadata = (self.n_segments, self.nfp, self.stellsym)
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        obj = object.__new__(cls)
        obj._centers, obj._quaternions, obj._xy_dofs = children
        obj._n_segments, obj._nfp, obj._stellsym = metadata
        return obj


@tree_util.register_pytree_node_class
class PlanarCoils:
    """Planar curve coordinates bundled with independent physical currents.

    Use :meth:`as_coils` for legacy APIs that require an actual ``Coils``
    instance, or pass this object to sampled-geometry consumers such as
    ``FilamentaryBiotSavart.from_coils``.
    """

    schema_version = 2
    curve_type = "planar_xy_fourier"

    def __init__(self, curves: PlanarXYCurves, currents: Any) -> None:
        if not isinstance(curves, PlanarXYCurves):
            raise TypeError(
                f"curves must be a PlanarXYCurves instance, got {type(curves)}"
            )
        base_currents = _real_array(currents, name="currents")
        if base_currents.ndim != 1 or base_currents.shape != (curves.n_base_curves,):
            raise ValueError(
                "currents must have one value per independent planar curve, got "
                f"{base_currents.shape} and {curves.n_base_curves} curves"
            )
        self._planar_curves = curves
        self._base_currents = base_currents

    @property
    def planar_curves(self) -> PlanarXYCurves:
        return self._planar_curves

    @property
    def centers(self) -> jnp.ndarray:
        return self.planar_curves.centers

    @property
    def quaternions(self) -> jnp.ndarray:
        return self.planar_curves.quaternions

    @property
    def frames(self) -> jnp.ndarray:
        return self.planar_curves.frames

    @property
    def normals(self) -> jnp.ndarray:
        return self.planar_curves.normals

    @property
    def xy_dofs(self) -> jnp.ndarray:
        return self.planar_curves.xy_dofs

    @property
    def dofs_curves(self) -> jnp.ndarray:
        return self.planar_curves.as_xyz_dofs()

    @property
    def base_currents(self) -> jnp.ndarray:
        return self._base_currents

    @property
    def currents_scale(self) -> jnp.ndarray:
        mean_absolute = jnp.mean(jnp.abs(self.base_currents))
        return jnp.where(mean_absolute > 0, mean_absolute, 1)

    @property
    def dofs_currents(self) -> jnp.ndarray:
        return self.base_currents / self.currents_scale

    @property
    def currents(self) -> jnp.ndarray:
        return apply_symmetries_to_currents(self.base_currents, self.nfp, self.stellsym)

    @property
    def n_segments(self) -> int:
        return self.planar_curves.n_segments

    @property
    def nfp(self) -> int:
        return self.planar_curves.nfp

    @property
    def stellsym(self) -> bool:
        return self.planar_curves.stellsym

    @property
    def order(self) -> int:
        return self.planar_curves.order

    @property
    def gamma(self) -> jnp.ndarray:
        return self.planar_curves.gamma

    @property
    def gamma_dash(self) -> jnp.ndarray:
        return self.planar_curves.gamma_dash

    @property
    def gamma_dashdash(self) -> jnp.ndarray:
        return self.planar_curves.gamma_dashdash

    @property
    def length(self) -> jnp.ndarray:
        return self.planar_curves.length

    @property
    def curvature(self) -> jnp.ndarray:
        return self.planar_curves.curvature

    @property
    def x(self) -> jnp.ndarray:
        raise AttributeError(
            "PlanarCoils does not expose the unrestricted Coils.x vector; use "
            "make_planar_coil_design_field_builder so optimization remains planar"
        )

    def __len__(self) -> int:
        return int(self.gamma.shape[0])

    def as_coils(self) -> Coils:
        """Return an ordinary ESSOS coil object with identical geometry/field."""

        return Coils(self.planar_curves.as_curves(), self.base_currents)

    def to_field(self):
        """Return a differentiable sampled filamentary field."""

        from essos.fields import FilamentaryBiotSavart

        return FilamentaryBiotSavart.from_coils(self)

    def to_json(self, filename: str | Path) -> None:
        """Write a versioned representation that retains the planar manifold."""

        data = {
            "schema_version": self.schema_version,
            "curve_type": self.curve_type,
            "nfp": self.nfp,
            "stellsym": self.stellsym,
            "order": self.order,
            "n_segments": self.n_segments,
            "centers": self.centers.tolist(),
            "quaternions": self.quaternions.tolist(),
            "xy_dofs": self.xy_dofs.tolist(),
            "dofs_currents": self.dofs_currents.tolist(),
            "currents_scale": float(self.currents_scale),
            "base_currents": self.base_currents.tolist(),
        }
        with Path(filename).open("w") as stream:
            json.dump(data, stream)

    @classmethod
    def from_json(cls, filename: str | Path) -> "PlanarCoils":
        """Load the versioned planar JSON representation."""

        with Path(filename).open("r") as stream:
            data = json.load(stream)
        if data.get("schema_version") != cls.schema_version:
            raise ValueError(
                "unsupported planar coil schema_version: "
                f"{data.get('schema_version')!r}"
            )
        if data.get("curve_type") != cls.curve_type:
            raise ValueError(
                f"expected curve_type {cls.curve_type!r}, got "
                f"{data.get('curve_type')!r}"
            )
        curves = PlanarXYCurves(
            data["centers"],
            data["quaternions"],
            data["xy_dofs"],
            n_segments=data["n_segments"],
            nfp=data["nfp"],
            stellsym=data["stellsym"],
        )
        if curves.order != data.get("order"):
            raise ValueError(
                "planar coil JSON order is inconsistent with xy_dofs: "
                f"{data.get('order')!r} versus {curves.order}"
            )
        base_currents = np.asarray(data["base_currents"])
        if "dofs_currents" in data and "currents_scale" in data:
            reconstructed = np.asarray(data["dofs_currents"]) * float(
                data["currents_scale"]
            )
            if reconstructed.shape != base_currents.shape or not np.allclose(
                reconstructed,
                base_currents,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "inconsistent current representations in planar coil JSON"
                )
        return cls(curves, base_currents)

    @classmethod
    def from_simsopt_planar(
        cls,
        simsopt_coils: Any,
        *,
        n_segments: int | None = None,
        nfp: int = 1,
        stellsym: bool = False,
    ) -> "PlanarCoils":
        """Import independent SIMSOPT planar coils and scalar current values.

        Shared, summed, or scaled SIMSOPT current dependency graphs are reduced
        to the physical values returned by ``current.get_value()``.
        """

        try:
            coils = tuple(simsopt_coils)
        except TypeError as exc:
            raise TypeError("simsopt_coils must be an iterable of base coils") from exc
        if not coils:
            raise ValueError("at least one SIMSOPT planar coil is required")
        curves = PlanarXYCurves.from_simsopt_planar(
            [coil.curve for coil in coils],
            n_segments=n_segments,
            nfp=nfp,
            stellsym=stellsym,
        )
        currents = []
        for index, coil in enumerate(coils):
            if not hasattr(coil, "current") or not hasattr(coil.current, "get_value"):
                raise TypeError(
                    f"SIMSOPT coil {index} has no current.get_value() contract"
                )
            currents.append(coil.current.get_value())
        return cls(curves, jnp.asarray(currents))

    @classmethod
    def from_simsopt_json(
        cls,
        filename: str | Path,
        *,
        nfp: int,
        stellsym: bool,
        n_segments: int | None = None,
        source_is_expanded: bool = True,
    ) -> "PlanarCoils":
        """Load planar coils from a trusted SIMSOPT JSON graph.

        ``nfp`` and ``stellsym`` are explicit because ordinary SIMSOPT coil
        lists do not retain enough metadata to infer the intended ESSOS
        symmetry. Set ``source_is_expanded=False`` when the JSON root contains
        independent base coils instead of ``coils_via_symmetries`` output.
        """

        coils = load_simsopt_coils_json(
            filename,
            nfp=nfp,
            stellsym=stellsym,
            n_segments=n_segments,
            source_is_expanded=source_is_expanded,
        )
        if not isinstance(coils, cls):
            raise TypeError("SIMSOPT JSON contains non-planar base curves")
        return coils

    def to_simsopt_xyz(self) -> list[Any]:
        """Export independent physical coils with unrestricted XYZ curves.

        The scalar current values are preserved in independent SIMSOPT
        ``Current`` objects. Symmetry copies are not generated, so callers can
        apply SIMSOPT symmetry exactly once.
        """

        from simsopt.field import Coil, Current

        curves = self.planar_curves.to_simsopt_xyz()
        return [
            Coil(curve, Current(float(current)))
            for curve, current in zip(
                curves, np.asarray(self.base_currents), strict=True
            )
        ]

    def tree_flatten(self):
        children = (self.planar_curves, self.base_currents)
        return children, None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        del metadata
        obj = object.__new__(cls)
        obj._planar_curves, obj._base_currents = children
        return obj


def load_coils_json(filename: str | Path) -> Coils | PlanarCoils:
    """Load tagged planar JSON or any existing legacy ESSOS coil JSON."""

    with Path(filename).open("r") as stream:
        data = json.load(stream)
    curve_type = data.get("curve_type")
    if curve_type == PlanarCoils.curve_type:
        return PlanarCoils.from_json(filename)
    if curve_type in (None, "xyz_fourier"):
        return Coils_from_json(str(filename))
    raise ValueError(f"unsupported coil curve_type {curve_type!r}")


def _simsopt_coil_sequence(loaded: Any) -> tuple[tuple[Any, ...], bool]:
    """Return a SIMSOPT coil sequence and whether it is already base-only."""

    if hasattr(loaded, "base_coils"):
        candidate = loaded.base_coils
        base_only = True
    elif hasattr(loaded, "coils"):
        candidate = loaded.coils
        base_only = False
    elif hasattr(loaded, "curve") and hasattr(loaded, "current"):
        candidate = (loaded,)
        base_only = True
    else:
        try:
            candidate = tuple(loaded)
        except TypeError as exc:
            raise TypeError(
                "SIMSOPT JSON must contain a BiotSavart/CoilSet, one Coil, "
                "or a top-level coil sequence"
            ) from exc
        base_only = False

    try:
        coils = tuple(candidate)
    except TypeError as exc:
        raise TypeError("loaded SIMSOPT coils must be iterable") from exc
    if not coils:
        raise ValueError("SIMSOPT JSON contains no coils")
    for index, coil in enumerate(coils):
        if not hasattr(coil, "curve"):
            raise TypeError(f"SIMSOPT coil {index} has no curve attribute")
        if not hasattr(coil, "current") or not hasattr(coil.current, "get_value"):
            raise TypeError(f"SIMSOPT coil {index} has no current.get_value() contract")
    return coils, base_only


def _simsopt_base_coils(
    coils: tuple[Any, ...],
    *,
    nfp: int,
    stellsym: bool,
    source_is_expanded: bool,
) -> tuple[Any, ...]:
    """Validate and collapse standard SIMSOPT symmetry expansion."""

    if not source_is_expanded:
        return coils
    factor = nfp * (1 + int(stellsym))
    if len(coils) % factor:
        raise ValueError(
            "SIMSOPT coil count is not divisible by the requested symmetry "
            f"factor {factor}: got {len(coils)} coils"
        )
    base_count = len(coils) // factor
    base_coils = coils[:base_count]
    base_currents = np.asarray(
        [coil.current.get_value() for coil in base_coils], dtype=float
    )
    identity = np.eye(3)
    block = 0
    for period in range(nfp):
        for flip in [False, True] if stellsym else [False]:
            for base_index, base_coil in enumerate(base_coils):
                coil = coils[block * base_count + base_index]
                expected_current = (
                    -base_currents[base_index] if flip else base_currents[base_index]
                )
                if not np.isclose(
                    float(coil.current.get_value()),
                    expected_current,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        "SIMSOPT current signs do not match the requested "
                        "coils_via_symmetries ordering"
                    )
                if period == 0 and not flip:
                    if coil.curve is not base_coil.curve:
                        raise ValueError(
                            "SIMSOPT first symmetry block does not contain the "
                            "independent base curves"
                        )
                    continue
                curve = coil.curve
                if type(curve).__name__ != "RotatedCurve" or not hasattr(
                    curve, "curve"
                ):
                    raise ValueError(
                        "SIMSOPT expanded coils are not in standard "
                        "coils_via_symmetries form"
                    )
                if curve.curve is not base_coil.curve:
                    raise ValueError(
                        "SIMSOPT rotated curve does not reference the expected "
                        "independent base curve"
                    )
                angle = 2 * np.pi * period / nfp
                rotation = np.asarray(
                    (
                        (np.cos(angle), -np.sin(angle), 0.0),
                        (np.sin(angle), np.cos(angle), 0.0),
                        (0.0, 0.0, 1.0),
                    )
                ).T
                if flip:
                    rotation = rotation @ np.diag((1.0, -1.0, -1.0))
                actual_rotation = np.asarray(getattr(curve, "rotmat", identity))
                if actual_rotation.shape != (3, 3) or not np.allclose(
                    actual_rotation,
                    rotation,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        "SIMSOPT rotated curve transform does not match the "
                        "requested nfp/stellsym expansion"
                    )
            block += 1
    return base_coils


def _simsopt_xyz_coils(
    coils: tuple[Any, ...],
    *,
    n_segments: int | None,
    nfp: int,
    stellsym: bool,
) -> Coils:
    """Convert independent SIMSOPT XYZ Fourier coils using complete DOFs."""

    orders = []
    full_dofs = []
    for index, coil in enumerate(coils):
        curve = coil.curve
        if not hasattr(curve, "order"):
            raise TypeError(f"SIMSOPT curve {index} has no Fourier order")
        try:
            order = operator.index(curve.order)
        except TypeError as exc:
            raise TypeError(f"SIMSOPT curve {index} order must be an integer") from exc
        expected_names = _simsopt_xyz_dof_names(order)
        if (
            not hasattr(curve, "local_full_dof_names")
            or tuple(curve.local_full_dof_names) != expected_names
        ):
            raise TypeError(
                f"SIMSOPT curve {index} is not a supported XYZ Fourier curve"
            )
        dofs = np.asarray(curve.local_full_x)
        if dofs.shape != (3 * (2 * order + 1),):
            raise ValueError(
                f"SIMSOPT XYZ curve {index} has invalid full DOF shape {dofs.shape}"
            )
        if not np.issubdtype(dofs.dtype, np.floating) or not np.all(np.isfinite(dofs)):
            raise ValueError(
                f"SIMSOPT XYZ curve {index} full DOFs must be finite floats"
            )
        orders.append(order)
        full_dofs.append(dofs.reshape((3, 2 * order + 1)))

    common_order = max(orders)
    dtype = np.result_type(*(dofs.dtype for dofs in full_dofs))
    dofs_curves = np.zeros((len(coils), 3, 2 * common_order + 1), dtype=dtype)
    for index, (order, dofs) in enumerate(zip(orders, full_dofs, strict=True)):
        dofs_curves[index, :, : 2 * order + 1] = dofs

    if n_segments is None:
        if not hasattr(coils[0].curve, "quadpoints"):
            raise TypeError(
                "n_segments is required when SIMSOPT curves expose no quadpoints"
            )
        segments = len(coils[0].curve.quadpoints)
        canonical = np.linspace(0.0, 1.0, segments, endpoint=False)
        for index, coil in enumerate(coils):
            quadpoints = np.asarray(getattr(coil.curve, "quadpoints", ()))
            if quadpoints.shape != canonical.shape or not np.allclose(
                quadpoints, canonical, rtol=0.0, atol=1.0e-14
            ):
                raise ValueError(
                    f"SIMSOPT XYZ curve {index} uses noncanonical quadpoints; "
                    "pass n_segments explicitly to resample"
                )
    else:
        segments = _positive_integer(n_segments, name="n_segments", minimum=3)
    currents = jnp.asarray([coil.current.get_value() for coil in coils])
    return Coils(
        Curves(
            jnp.asarray(dofs_curves),
            n_segments=segments,
            nfp=nfp,
            stellsym=stellsym,
        ),
        currents,
    )


def load_simsopt_coils_json(
    filename: str | Path,
    *,
    nfp: int,
    stellsym: bool,
    n_segments: int | None = None,
    source_is_expanded: bool = True,
) -> Coils | PlanarCoils:
    """Initialize ESSOS coils from a trusted SIMSOPT JSON graph.

    SIMSOPT is imported lazily and remains an optional dependency. The JSON
    root may be a BiotSavart/CoilSet, one Coil, or a top-level coil sequence.
    Standard symmetry-expanded lists are validated before their independent
    base block is selected. All-planar inputs preserve the native
    :class:`PlanarCoils` representation; XYZ Fourier inputs return
    :class:`Coils`. Mixed curve representations are rejected.

    SIMSOPT's decoder dynamically imports the classes named in its graph, so
    only load JSON files from trusted sources. Current values are preserved,
    while SIMSOPT current dependency graphs are flattened.
    """

    checked_nfp = _positive_integer(nfp, name="nfp", minimum=1)
    if not isinstance(stellsym, bool):
        raise TypeError("stellsym must be a boolean")
    if not isinstance(source_is_expanded, bool):
        raise TypeError("source_is_expanded must be a boolean")
    try:
        from simsopt import load as simsopt_load
    except ImportError as exc:
        raise ImportError(
            "loading SIMSOPT JSON requires SIMSOPT in the active environment"
        ) from exc

    loaded = simsopt_load(str(filename))
    coils, base_only = _simsopt_coil_sequence(loaded)
    base_coils = _simsopt_base_coils(
        coils,
        nfp=checked_nfp,
        stellsym=stellsym,
        source_is_expanded=source_is_expanded and not base_only,
    )
    planar = []
    for coil in base_coils:
        curve = coil.curve
        try:
            order = operator.index(curve.order)
            names = tuple(curve.local_full_dof_names)
        except (AttributeError, TypeError):
            planar.append(False)
        else:
            planar.append(names == _simsopt_planar_dof_names(order))
    if all(planar):
        return PlanarCoils.from_simsopt_planar(
            base_coils,
            n_segments=n_segments,
            nfp=checked_nfp,
            stellsym=stellsym,
        )
    if any(planar):
        raise TypeError("SIMSOPT JSON mixes planar and non-planar independent curves")
    return _simsopt_xyz_coils(
        base_coils,
        n_segments=n_segments,
        nfp=checked_nfp,
        stellsym=stellsym,
    )
