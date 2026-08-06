"""Differentiable maps from coil design coordinates to filamentary fields.

The design vector is ordered as fractional independent-coil current changes,
followed by additive Fourier-coefficient shape coordinates. The returned field
contains only physical JAX arrays and implements both Cartesian and cylindrical
Biot--Savart evaluation.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from essos.coils import Coils, Curves, apply_symmetries_to_currents
from essos.fields import FilamentaryBiotSavart

__all__ = [
    "CoilDesignFieldBuilder",
    "PlanarCoilDesignFieldBuilder",
    "ShapeDeformationMetrics",
    "make_coil_design_field_builder",
    "make_fractional_current_field_builder",
    "make_planar_coil_design_field_builder",
    "make_shape_field_builder",
    "shape_deformation_metrics",
]


def _host_real_floating_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _trace_parameter_vector(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> Any:
    parameters = jnp.asarray(value)
    if parameters.shape != shape:
        raise ValueError(
            f"design parameters must have shape {shape}, got {parameters.shape}"
        )
    if jnp.issubdtype(parameters.dtype, jnp.complexfloating):
        raise ValueError("design parameters must be real-valued")
    if not jnp.issubdtype(parameters.dtype, jnp.floating):
        raise TypeError("design parameters must have a floating dtype")
    return parameters.astype(dtype)


def _normalize_groups(groups: Any, base_current_count: int) -> tuple[int, ...]:
    if isinstance(groups, (bool, np.bool_)):
        raise TypeError("current group indices must be integers, not booleans")
    if isinstance(groups, (int, np.integer)):
        requested = (groups,)
    else:
        try:
            requested = tuple(groups)
        except TypeError as exc:
            raise TypeError(
                "current_groups must be an integer or iterable of integers"
            ) from exc

    normalized = []
    for group in requested:
        if isinstance(group, (bool, np.bool_)):
            raise TypeError("current group indices must be integers, not booleans")
        try:
            index = operator.index(group)
        except TypeError as exc:
            raise TypeError(f"current group index {group!r} is not an integer") from exc
        if not 0 <= index < base_current_count:
            raise IndexError(
                f"current group {index} outside base-current length {base_current_count}"
            )
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"current_groups must be unique, got {tuple(normalized)}")
    return tuple(normalized)


def _normalize_shape_dofs(
    shape_dofs: Any,
    dofs_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    if shape_dofs is None:
        return ()
    try:
        requested = tuple(shape_dofs)
    except TypeError as exc:
        raise TypeError(
            "shape_dofs must be a coordinate or iterable of coordinates"
        ) from exc
    if len(requested) == 3 and all(
        isinstance(item, (int, np.integer)) for item in requested
    ):
        requested = (requested,)

    normalized = []
    for coordinate in requested:
        try:
            entries = tuple(coordinate)
        except TypeError as exc:
            raise TypeError(
                "each shape coordinate must contain three integer indices"
            ) from exc
        if len(entries) != 3:
            raise ValueError(f"each shape coordinate must have length 3, got {entries}")
        indices = []
        for entry in entries:
            if isinstance(entry, (bool, np.bool_)):
                raise TypeError(
                    "shape coordinate indices must be integers, not booleans"
                )
            try:
                indices.append(operator.index(entry))
            except TypeError as exc:
                raise TypeError(
                    f"shape coordinate index {entry!r} is not an integer"
                ) from exc
        item = tuple(indices)
        for index, extent in zip(item, dofs_shape, strict=True):
            if not 0 <= index < extent:
                raise IndexError(
                    f"shape coordinate {item} outside curve shape {dofs_shape}"
                )
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"shape_dofs must be unique, got {tuple(normalized)}")
    return tuple(normalized)


def _normalize_coordinate_pairs(
    coordinates: Any,
    *,
    name: str,
    extents: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    if coordinates is None:
        return ()
    try:
        requested = tuple(coordinates)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a coordinate or iterable of coordinates"
        ) from exc
    if len(requested) == 2 and all(
        isinstance(item, (int, np.integer)) for item in requested
    ):
        requested = (requested,)

    normalized = []
    for coordinate in requested:
        try:
            entries = tuple(coordinate)
        except TypeError as exc:
            raise TypeError(
                f"each {name} coordinate must contain two integer indices"
            ) from exc
        if len(entries) != 2:
            raise ValueError(
                f"each {name} coordinate must have length 2, got {entries}"
            )
        indices = []
        for entry in entries:
            if isinstance(entry, (bool, np.bool_)):
                raise TypeError(
                    f"{name} coordinate indices must be integers, not booleans"
                )
            try:
                indices.append(operator.index(entry))
            except TypeError as exc:
                raise TypeError(
                    f"{name} coordinate index {entry!r} is not an integer"
                ) from exc
        item = tuple(indices)
        for index, extent in zip(item, extents, strict=True):
            if not 0 <= index < extent:
                raise IndexError(f"{name} coordinate {item} outside shape {extents}")
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} coordinates must be unique, got {tuple(normalized)}")
    return tuple(normalized)


def _nonnegative_finite_tolerance(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _rotation_matrix_from_vector(rotation_vector: Any) -> Any:
    """Return an exact active rotation with a finite derivative at zero."""

    omega = jnp.asarray(rotation_vector)
    squared_angle = jnp.dot(omega, omega)
    safe_squared_angle = jnp.where(squared_angle > 0, squared_angle, 1)
    safe_angle = jnp.sqrt(safe_squared_angle)
    exact_a = jnp.sin(safe_angle) / safe_angle
    exact_b = (1 - jnp.cos(safe_angle)) / safe_squared_angle
    s = squared_angle
    series_a = 1 - s / 6 + s**2 / 120 - s**3 / 5040
    series_b = 0.5 - s / 24 + s**2 / 720 - s**3 / 40320
    use_series = squared_angle < jnp.asarray(1.0e-8, dtype=omega.dtype)
    coefficient_a = jnp.where(use_series, series_a, exact_a)
    coefficient_b = jnp.where(use_series, series_b, exact_b)
    x, y, z = omega
    cross_matrix = jnp.asarray(
        ((0, -z, y), (z, 0, -x), (-y, x, 0)),
        dtype=omega.dtype,
    )
    identity = jnp.eye(3, dtype=omega.dtype)
    return (
        identity
        + coefficient_a * cross_matrix
        + coefficient_b * (cross_matrix @ cross_matrix)
    )


def _quaternion_from_rotation_vector(rotation_vector: Any) -> Any:
    """Return a scalar-first rotation quaternion with a regular zero limit."""

    omega = jnp.asarray(rotation_vector)
    squared_angle = jnp.dot(omega, omega)
    safe_squared_angle = jnp.where(squared_angle > 0, squared_angle, 1)
    safe_angle = jnp.sqrt(safe_squared_angle)
    exact_vector_scale = jnp.sin(0.5 * safe_angle) / safe_angle
    s = squared_angle
    series_vector_scale = 0.5 - s / 48 + s**2 / 3840 - s**3 / 645120
    use_series = squared_angle < jnp.asarray(1.0e-8, dtype=omega.dtype)
    vector_scale = jnp.where(use_series, series_vector_scale, exact_vector_scale)
    scalar = jnp.where(
        use_series,
        1 - s / 8 + s**2 / 384 - s**3 / 46080,
        jnp.cos(0.5 * safe_angle),
    )
    return jnp.concatenate((jnp.reshape(scalar, (1,)), vector_scale * omega))


def _multiply_quaternions(left: Any, right: Any) -> Any:
    """Compose scalar-first active rotation quaternions as ``left @ right``."""

    left = jnp.asarray(left)
    right = jnp.asarray(right)
    left_scalar, left_vector = left[0], left[1:]
    right_scalar, right_vector = right[0], right[1:]
    scalar = left_scalar * right_scalar - jnp.dot(left_vector, right_vector)
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + jnp.cross(left_vector, right_vector)
    )
    return jnp.concatenate((jnp.reshape(scalar, (1,)), vector))


def _quaternions_from_frames(frames: np.ndarray) -> np.ndarray:
    """Convert host-side orthonormal XY frames to scalar-first quaternions."""

    from scipy.spatial.transform import Rotation

    normals = np.cross(frames[..., 0], frames[..., 1])
    rotation_matrices = np.stack(
        (frames[..., 0], frames[..., 1], normals),
        axis=-1,
    )
    xyzw = Rotation.from_matrix(rotation_matrices).as_quat()
    return np.concatenate((xyzw[..., 3:4], xyzw[..., :3]), axis=-1)


def _static_coil_metadata(coils: Any, n_segments: Any) -> tuple[int, int, bool]:
    segments_raw = coils.n_segments if n_segments is None else n_segments
    if isinstance(segments_raw, (bool, np.bool_)):
        raise TypeError("n_segments must be an integer, not a boolean")
    try:
        segments = operator.index(segments_raw)
    except TypeError as exc:
        raise TypeError("n_segments must be an integer") from exc
    if segments <= 2:
        raise ValueError("n_segments must be greater than 2")

    nfp_raw = coils.nfp
    if isinstance(nfp_raw, (bool, np.bool_)):
        raise TypeError("coils.nfp must be an integer, not a boolean")
    try:
        nfp = operator.index(nfp_raw)
    except TypeError as exc:
        raise TypeError("coils.nfp must be an integer") from exc
    if nfp <= 0:
        raise ValueError("coils.nfp must be positive")

    stellsym_raw = coils.stellsym
    if not isinstance(stellsym_raw, (bool, np.bool_)):
        raise TypeError("coils.stellsym must be a boolean")
    return segments, nfp, bool(stellsym_raw)


@dataclass(frozen=True, eq=False)
class CoilDesignFieldBuilder:
    """Immutable JAX-callable current/shape map for one ESSOS coil set.

    Each entry in ``current_groups`` selects one independent base coil and
    therefore its complete symmetry-expanded physical coil group. A
    ``shape_dofs`` coordinate is ``(base_coil, xyz_component, Fourier_index)``;
    its parameter is an additive Fourier coefficient in metres.
    """

    nominal_dofs_curves: Any
    nominal_base_currents: Any
    nominal_expanded_currents: Any
    fractional_base_directions: Any
    fractional_expanded_directions: Any
    shape_directions: Any
    current_groups: tuple[int, ...]
    shape_dofs: tuple[tuple[int, int, int], ...]
    n_segments: int
    nfp: int
    stellsym: bool

    @property
    def current_parameter_count(self) -> int:
        return len(self.current_groups)

    @property
    def shape_parameter_count(self) -> int:
        return int(self.shape_directions.shape[0])

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return (self.current_parameter_count + self.shape_parameter_count,)

    def _parameters(self, parameters: Any) -> Any:
        return _trace_parameter_vector(
            parameters,
            shape=self.parameter_shape,
            dtype=self.nominal_dofs_curves.dtype,
        )

    def _split(self, parameters: Any) -> tuple[Any, Any]:
        values = self._parameters(parameters)
        split = self.current_parameter_count
        return values[:split], values[split:]

    def curve_dofs_at(self, parameters: Any) -> Any:
        """Return independent-coil Fourier coefficients at ``parameters``."""

        _current_parameters, shape_parameters = self._split(parameters)
        if self.shape_parameter_count == 0:
            return self.nominal_dofs_curves
        return self.nominal_dofs_curves + jnp.einsum(
            "s,sijk->ijk",
            shape_parameters,
            self.shape_directions,
        )

    def base_currents_at(self, parameters: Any) -> Any:
        """Return physical independent-coil currents at ``parameters``."""

        current_parameters, _shape_parameters = self._split(parameters)
        if self.current_parameter_count == 0:
            return self.nominal_base_currents
        return self.nominal_base_currents + jnp.einsum(
            "g,gc->c",
            current_parameters,
            self.fractional_base_directions,
        )

    def expanded_currents_at(self, parameters: Any) -> Any:
        """Return symmetry-expanded physical currents at ``parameters``."""

        current_parameters, _shape_parameters = self._split(parameters)
        if self.current_parameter_count == 0:
            return self.nominal_expanded_currents
        return self.nominal_expanded_currents + jnp.einsum(
            "g,gc->c",
            current_parameters,
            self.fractional_expanded_directions,
        )

    def rebuild_curves(self, parameters: Any) -> Curves:
        """Rebuild the traced curve geometry at ``parameters``."""

        return Curves(
            self.curve_dofs_at(parameters),
            self.n_segments,
            self.nfp,
            self.stellsym,
        )

    def rebuild_coils(self, parameters: Any) -> Coils:
        """Rebuild an ESSOS coil collection at ``parameters``."""

        return Coils(
            self.rebuild_curves(parameters),
            self.base_currents_at(parameters),
        )

    def __call__(self, parameters: Any) -> FilamentaryBiotSavart:
        """Return the filamentary field at ``parameters``."""

        curves = self.rebuild_curves(parameters)
        return FilamentaryBiotSavart(
            gamma=curves.gamma,
            gamma_dash=curves.gamma_dash,
            currents=self.expanded_currents_at(parameters),
        )

    def field_from_scalar_current(
        self, fractional_change: Any
    ) -> FilamentaryBiotSavart:
        """Adapt a one-current, no-shape builder to a scalar parameter API."""

        if self.current_parameter_count != 1 or self.shape_parameter_count != 0:
            raise ValueError(
                "scalar current adaptation requires one current group and no shape coordinates"
            )
        value = jnp.asarray(fractional_change)
        if value.shape != ():
            raise ValueError(
                f"fractional current change must be scalar, got {value.shape}"
            )
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise ValueError("fractional current change must be real-valued")
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("fractional current change must have a floating dtype")
        return self(jnp.reshape(value, (1,)))


@dataclass(frozen=True, eq=False)
class PlanarCoilDesignFieldBuilder:
    """JAX-callable design chart that preserves exact coil planarity.

    Parameters are ordered as fractional current changes, local XY Fourier
    changes in metres, Cartesian center changes in metres, and global
    rotation-vector increments in radians. Shape coordinates use the native
    non-DC ordering ``[sin(1), cos(1), sin(2), cos(2), ...]``. The builder
    lowers every point in this reduced chart to ordinary :class:`Curves`,
    :class:`Coils`, and :class:`FilamentaryBiotSavart` objects.
    """

    nominal_centers: Any
    nominal_frames: Any
    nominal_quaternions: Any
    nominal_local_dofs: Any
    nominal_base_currents: Any
    nominal_expanded_currents: Any
    fractional_base_directions: Any
    fractional_expanded_directions: Any
    local_shape_directions: Any
    center_directions: Any
    orientation_directions: Any
    current_groups: tuple[int, ...]
    shape_dofs: tuple[tuple[int, int, int], ...]
    center_dofs: tuple[tuple[int, int], ...]
    orientation_dofs: tuple[tuple[int, int], ...]
    n_segments: int
    nfp: int
    stellsym: bool

    @property
    def current_parameter_count(self) -> int:
        return len(self.current_groups)

    @property
    def shape_parameter_count(self) -> int:
        return int(self.local_shape_directions.shape[0])

    @property
    def center_parameter_count(self) -> int:
        return int(self.center_directions.shape[0])

    @property
    def orientation_parameter_count(self) -> int:
        return int(self.orientation_directions.shape[0])

    @property
    def current_parameter_slice(self) -> slice:
        return slice(0, self.current_parameter_count)

    @property
    def shape_parameter_slice(self) -> slice:
        start = self.current_parameter_slice.stop
        return slice(start, start + self.shape_parameter_count)

    @property
    def center_parameter_slice(self) -> slice:
        start = self.shape_parameter_slice.stop
        return slice(start, start + self.center_parameter_count)

    @property
    def orientation_parameter_slice(self) -> slice:
        start = self.center_parameter_slice.stop
        return slice(start, start + self.orientation_parameter_count)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return (self.orientation_parameter_slice.stop,)

    def _parameters(self, parameters: Any) -> Any:
        return _trace_parameter_vector(
            parameters,
            shape=self.parameter_shape,
            dtype=self.nominal_local_dofs.dtype,
        )

    def _split(self, parameters: Any) -> tuple[Any, Any, Any, Any]:
        values = self._parameters(parameters)
        return (
            values[self.current_parameter_slice],
            values[self.shape_parameter_slice],
            values[self.center_parameter_slice],
            values[self.orientation_parameter_slice],
        )

    def local_dofs_at(self, parameters: Any) -> Any:
        """Return local XY Fourier coefficients in ESSOS ordering."""

        _current, shape, _center, _orientation = self._split(parameters)
        if self.shape_parameter_count == 0:
            return self.nominal_local_dofs
        return self.nominal_local_dofs + jnp.einsum(
            "s,snal->nal", shape, self.local_shape_directions
        )

    def centers_at(self, parameters: Any) -> Any:
        """Return Cartesian centers of the independent planar coils."""

        _current, _shape, center, _orientation = self._split(parameters)
        if self.center_parameter_count == 0:
            return self.nominal_centers
        return self.nominal_centers + jnp.einsum(
            "s,snc->nc", center, self.center_directions
        )

    def orientation_vectors_at(self, parameters: Any) -> Any:
        """Return global rotation-vector increments for each base coil."""

        _current, _shape, _center, orientation = self._split(parameters)
        if self.orientation_parameter_count == 0:
            return jnp.zeros_like(self.nominal_centers)
        return jnp.einsum("s,snc->nc", orientation, self.orientation_directions)

    def frames_at(self, parameters: Any) -> Any:
        """Return exactly rotated orthonormal local XY frames."""

        rotation_matrices = jax.vmap(_rotation_matrix_from_vector)(
            self.orientation_vectors_at(parameters)
        )
        return jnp.einsum("nij,nja->nia", rotation_matrices, self.nominal_frames)

    def quaternions_at(self, parameters: Any) -> Any:
        """Return scalar-first local-to-global orientation quaternions."""

        increments = jax.vmap(_quaternion_from_rotation_vector)(
            self.orientation_vectors_at(parameters)
        )
        return jax.vmap(_multiply_quaternions)(
            increments,
            self.nominal_quaternions,
        )

    def normals_at(self, parameters: Any) -> Any:
        """Return unit plane normals of the independent coils."""

        frames = self.frames_at(parameters)
        return jnp.cross(frames[..., 0], frames[..., 1])

    def curve_dofs_at(self, parameters: Any) -> Any:
        """Return full Cartesian Fourier coefficients for ordinary ``Curves``."""

        relative = jnp.einsum(
            "nca,nal->ncl", self.frames_at(parameters), self.local_dofs_at(parameters)
        )
        return relative.at[:, :, 0].add(self.centers_at(parameters))

    def base_currents_at(self, parameters: Any) -> Any:
        """Return physical independent-coil currents at ``parameters``."""

        current, _shape, _center, _orientation = self._split(parameters)
        if self.current_parameter_count == 0:
            return self.nominal_base_currents
        return self.nominal_base_currents + jnp.einsum(
            "g,gc->c", current, self.fractional_base_directions
        )

    def expanded_currents_at(self, parameters: Any) -> Any:
        """Return symmetry-expanded physical currents at ``parameters``."""

        current, _shape, _center, _orientation = self._split(parameters)
        if self.current_parameter_count == 0:
            return self.nominal_expanded_currents
        return self.nominal_expanded_currents + jnp.einsum(
            "g,gc->c", current, self.fractional_expanded_directions
        )

    def rebuild_curves(self, parameters: Any) -> Curves:
        """Rebuild ordinary ESSOS curves at ``parameters``."""

        return Curves(
            self.curve_dofs_at(parameters),
            self.n_segments,
            self.nfp,
            self.stellsym,
        )

    def rebuild_coils(self, parameters: Any) -> Coils:
        """Rebuild ordinary ESSOS coils at ``parameters``."""

        return Coils(self.rebuild_curves(parameters), self.base_currents_at(parameters))

    def rebuild_planar_coils(self, parameters: Any):
        """Rebuild native planar coils without lowering to unrestricted XYZ DOFs."""

        from essos.planar_coils import PlanarCoils, PlanarXYCurves

        curves = PlanarXYCurves(
            centers=self.centers_at(parameters),
            quaternions=self.quaternions_at(parameters),
            xy_dofs=self.local_dofs_at(parameters)[..., 1:],
            n_segments=self.n_segments,
            nfp=self.nfp,
            stellsym=self.stellsym,
        )
        return PlanarCoils(curves, self.base_currents_at(parameters))

    def __call__(self, parameters: Any) -> FilamentaryBiotSavart:
        """Return the filamentary field while retaining all design derivatives."""

        curves = self.rebuild_curves(parameters)
        return FilamentaryBiotSavart(
            gamma=curves.gamma,
            gamma_dash=curves.gamma_dash,
            currents=self.expanded_currents_at(parameters),
        )


@dataclass(frozen=True)
class ShapeDeformationMetrics:
    """Rigid-motion-invariant diagnostics for one sampled coil tangent."""

    coil_index: int
    sampled_point_count: int
    sampled_pair_count: int
    pair_distance_derivative_rms: float
    rigid_motion_fit_residual_rms: float
    shape_velocity_rms: float
    rigid_motion_fit_residual_ratio: float
    sampled_length: float
    length_derivative: float


def shape_deformation_metrics(
    gamma: Any,
    gamma_tangent: Any,
    *,
    coil_index: Any = 0,
) -> ShapeDeformationMetrics:
    """Measure on the host whether a sampled coil tangent deforms its shape.

    The returned metrics are descriptive rather than pass/fail thresholds:
    callers can set tolerances appropriate to their geometry and parameter
    units. Exact translations and infinitesimal rotations have zero pairwise
    distance and length derivatives and zero residual after the rigid fit.
    This reporting helper converts its inputs to NumPy and is intentionally not
    JIT-transformable; the field builder and its returned arrays remain JAX
    traceable.
    """

    positions = _host_real_floating_array(gamma, name="gamma", ndim=3)
    velocities = _host_real_floating_array(
        gamma_tangent,
        name="gamma_tangent",
        ndim=3,
    )
    if positions.shape != velocities.shape:
        raise ValueError(
            "gamma and gamma_tangent must have the same shape, got "
            f"{positions.shape} and {velocities.shape}"
        )
    if positions.shape[-1] != 3:
        raise ValueError(f"gamma must end in xyz, got shape {positions.shape}")
    if positions.shape[1] < 4:
        raise ValueError("at least four periodic samples per coil are required")
    if isinstance(coil_index, (bool, np.bool_)):
        raise TypeError("coil_index must be an integer, not a boolean")
    try:
        selected_index = operator.index(coil_index)
    except TypeError as exc:
        raise TypeError("coil_index must be an integer") from exc
    if not 0 <= selected_index < positions.shape[0]:
        raise IndexError(
            f"coil_index {selected_index} outside sampled coil count {positions.shape[0]}"
        )

    selected = positions[selected_index]
    selected_velocity = velocities[selected_index]
    point_count = selected.shape[0]
    first, second = np.triu_indices(point_count, k=1)
    separation = selected[first] - selected[second]
    separation_tangent = selected_velocity[first] - selected_velocity[second]
    distances = np.linalg.norm(separation, axis=1)
    if np.any(distances <= np.finfo(float).eps):
        raise ValueError("selected coil contains coincident sampled points")
    pair_derivatives = (
        np.einsum(
            "ij,ij->i",
            separation,
            separation_tangent,
        )
        / distances
    )
    pair_rms = float(np.sqrt(np.mean(pair_derivatives**2)))

    centered = selected - np.mean(selected, axis=0, keepdims=True)
    rigid_basis = np.zeros((3 * point_count, 6), dtype=float)
    for point_index, (x, y, z) in enumerate(centered):
        row = 3 * point_index
        rigid_basis[row : row + 3, :3] = np.eye(3)
        rigid_basis[row : row + 3, 3:] = np.asarray(
            (
                (0.0, z, -y),
                (-z, 0.0, x),
                (y, -x, 0.0),
            )
        )
    fitted, *_ = np.linalg.lstsq(
        rigid_basis,
        selected_velocity.reshape(-1),
        rcond=None,
    )
    residual = selected_velocity.reshape(-1) - rigid_basis @ fitted
    residual_rms = float(np.sqrt(np.mean(residual.reshape(point_count, 3) ** 2)))
    velocity_rms = float(np.sqrt(np.mean(selected_velocity**2)))
    residual_ratio = residual_rms / max(velocity_rms, np.finfo(float).tiny)

    edges = np.roll(selected, -1, axis=0) - selected
    edge_tangents = np.roll(selected_velocity, -1, axis=0) - selected_velocity
    edge_lengths = np.linalg.norm(edges, axis=1)
    if np.any(edge_lengths <= np.finfo(float).eps):
        raise ValueError("selected coil contains a zero-length sampled edge")
    sampled_length = float(np.sum(edge_lengths))
    length_derivative = float(
        np.sum(np.einsum("ij,ij->i", edges, edge_tangents) / edge_lengths)
    )

    return ShapeDeformationMetrics(
        coil_index=selected_index,
        sampled_point_count=point_count,
        sampled_pair_count=len(first),
        pair_distance_derivative_rms=pair_rms,
        rigid_motion_fit_residual_rms=residual_rms,
        shape_velocity_rms=velocity_rms,
        rigid_motion_fit_residual_ratio=residual_ratio,
        sampled_length=sampled_length,
        length_derivative=length_derivative,
    )


def make_coil_design_field_builder(
    coils: Coils,
    *,
    current_groups: Any = (),
    shape_dofs: Any = (),
    shape_directions: Any | None = None,
    n_segments: Any | None = None,
) -> CoilDesignFieldBuilder:
    """Build a generic current/shape parameterization from ESSOS coils.

    Current coordinates are fractional changes of selected independent-coil
    currents; each selected base-coil index controls its symmetry-expanded
    physical group. Shape coordinates are additive displacements in the
    supplied Fourier direction stack, in metres. A ``shape_dofs`` entry is
    ``(base_coil, xyz_component, Fourier_index)``. If ``shape_directions`` is
    omitted, one-hot directions are generated from ``shape_dofs``. Factory
    construction validates static inputs on the host; the returned builder is
    JAX traceable in its parameter vector.
    """

    # ``PlanarCoils`` deliberately exposes an XYZ lowering for compatibility,
    # but consuming it here would silently replace the reduced planar chart by
    # unrestricted Cartesian Fourier directions.
    from essos.planar_coils import PlanarCoils

    if isinstance(coils, PlanarCoils):
        raise TypeError(
            "make_coil_design_field_builder does not accept PlanarCoils because "
            "unrestricted XYZ shape coordinates can leave the coil plane; use "
            "make_planar_coil_design_field_builder(...) to preserve planarity, "
            "or pass coils.as_coils() explicitly for an unrestricted XYZ chart"
        )

    dofs = _host_real_floating_array(
        coils.dofs_curves,
        name="coils.dofs_curves",
        ndim=3,
    )
    if dofs.shape[1] != 3 or dofs.shape[2] % 2 != 1:
        raise ValueError(
            "coils.dofs_curves must have shape (n_base_coils, 3, 2 * order + 1), "
            f"got {dofs.shape}"
        )
    base_currents = _host_real_floating_array(
        coils.base_currents,
        name="coils.base_currents",
        ndim=1,
    )
    if base_currents.shape != (dofs.shape[0],):
        raise ValueError(
            "coils.base_currents must match the independent coil count, got "
            f"{base_currents.shape} and {dofs.shape[0]}"
        )
    segments, nfp, stellsym = _static_coil_metadata(coils, n_segments)
    groups = _normalize_groups(current_groups, len(base_currents))

    coordinates = _normalize_shape_dofs(shape_dofs, dofs.shape)
    if shape_directions is not None and coordinates:
        raise ValueError("provide either shape_dofs or shape_directions, not both")
    if shape_directions is None:
        directions = np.zeros((len(coordinates), *dofs.shape), dtype=dofs.dtype)
        for direction, coordinate in zip(directions, coordinates, strict=True):
            direction[coordinate] = 1.0
    else:
        directions = _host_real_floating_array(
            shape_directions,
            name="shape_directions",
            ndim=4,
        )
        if directions.shape[1:] != dofs.shape:
            raise ValueError(
                "shape_directions must have shape (n_shape, *coils.dofs_curves.shape), "
                f"got {directions.shape} and {dofs.shape}"
            )

    fractional_base_directions = np.zeros(
        (len(groups), len(base_currents)), dtype=base_currents.dtype
    )
    for local_index, group in enumerate(groups):
        fractional_base_directions[local_index, group] = base_currents[group]

    nominal_expanded = apply_symmetries_to_currents(
        jnp.asarray(base_currents),
        nfp,
        stellsym,
    )
    if groups:
        expanded_directions = jnp.stack(
            [
                apply_symmetries_to_currents(jnp.asarray(direction), nfp, stellsym)
                for direction in fractional_base_directions
            ]
        )
    else:
        expanded_directions = jnp.zeros(
            (0, nominal_expanded.shape[0]),
            dtype=nominal_expanded.dtype,
        )

    return CoilDesignFieldBuilder(
        nominal_dofs_curves=jnp.asarray(dofs),
        nominal_base_currents=jnp.asarray(base_currents),
        nominal_expanded_currents=nominal_expanded,
        fractional_base_directions=jnp.asarray(fractional_base_directions),
        fractional_expanded_directions=expanded_directions,
        shape_directions=jnp.asarray(directions),
        current_groups=groups,
        shape_dofs=coordinates,
        n_segments=segments,
        nfp=nfp,
        stellsym=stellsym,
    )


def make_planar_coil_design_field_builder(
    coils: Any,
    *,
    plane_frames: Any | None = None,
    current_groups: Any = (),
    shape_dofs: Any = (),
    center_dofs: Any = (),
    orientation_dofs: Any = (),
    n_segments: Any | None = None,
    planarity_atol: Any = 1.0e-12,
    frame_atol: Any = 1.0e-12,
) -> PlanarCoilDesignFieldBuilder:
    """Build a reduced current/shape chart that preserves planar coils.

    A :class:`essos.planar_coils.PlanarCoils` input supplies its own frames.
    Ordinary :class:`Coils` inputs require ``plane_frames`` with shape
    ``(n_base_coils, 3, 2)``. Its last axis stores orthonormal local X and Y
    vectors in global Cartesian coordinates. The flat design vector is ordered
    as current fractions, local XY Fourier changes, Cartesian center changes,
    and global rotation-vector increments.

    ``shape_dofs`` entries use ``(base_coil, local_axis, xy_index)`` with the
    native planar ordering ``[sin(1), cos(1), sin(2), cos(2), ...]`` and no DC
    entry. Translation belongs to ``center_dofs``. ``center_dofs`` and
    ``orientation_dofs`` use ``(base_coil, xyz_component)``.
    """

    nominal_quaternions = None
    if not isinstance(coils, Coils):
        from essos.planar_coils import PlanarCoils

        if not isinstance(coils, PlanarCoils):
            raise TypeError(
                f"coils must be a Coils or PlanarCoils instance, got {type(coils)}"
            )
        if plane_frames is None:
            plane_frames = coils.frames
            nominal_quaternions = np.asarray(coils.quaternions)
        coils = coils.as_coils()
    elif plane_frames is None:
        raise ValueError("plane_frames is required for an ordinary Coils input")
    dofs = _host_real_floating_array(
        coils.dofs_curves,
        name="coils.dofs_curves",
        ndim=3,
    )
    if dofs.shape[1] != 3 or dofs.shape[2] % 2 != 1:
        raise ValueError(
            "coils.dofs_curves must have shape (n_base_coils, 3, 2 * order + 1), "
            f"got {dofs.shape}"
        )
    base_currents = _host_real_floating_array(
        coils.base_currents,
        name="coils.base_currents",
        ndim=1,
    )
    if base_currents.shape != (dofs.shape[0],):
        raise ValueError(
            "coils.base_currents must match the independent coil count, got "
            f"{base_currents.shape} and {dofs.shape[0]}"
        )
    frames = _host_real_floating_array(
        plane_frames,
        name="plane_frames",
        ndim=3,
    )
    expected_frame_shape = (dofs.shape[0], 3, 2)
    if frames.shape != expected_frame_shape:
        raise ValueError(
            f"plane_frames must have shape {expected_frame_shape}, got {frames.shape}"
        )
    checked_frame_atol = _nonnegative_finite_tolerance(frame_atol, name="frame_atol")
    checked_planarity_atol = _nonnegative_finite_tolerance(
        planarity_atol, name="planarity_atol"
    )
    frame_gram = np.einsum("nca,ncb->nab", frames, frames)
    frame_error = float(
        np.max(np.abs(frame_gram - np.eye(2, dtype=frame_gram.dtype)[None, ...]))
    )
    if frame_error > checked_frame_atol:
        raise ValueError(
            "plane_frames must be orthonormal; maximum Gram-matrix error is "
            f"{frame_error:.6e}, tolerance {checked_frame_atol:.6e}"
        )
    if nominal_quaternions is None:
        nominal_quaternions = _quaternions_from_frames(frames)

    centers = dofs[:, :, 0].copy()
    relative_dofs = dofs.copy()
    relative_dofs[:, :, 0] = 0
    local_dofs = np.einsum("nca,ncl->nal", frames, relative_dofs)
    reconstructed = np.einsum("nca,nal->ncl", frames, local_dofs)
    planarity_error = float(np.max(np.abs(relative_dofs - reconstructed)))
    if planarity_error > checked_planarity_atol:
        raise ValueError(
            "coils are not planar in the supplied frames; maximum coefficient "
            f"residual is {planarity_error:.6e}, tolerance "
            f"{checked_planarity_atol:.6e}"
        )

    segments, nfp, stellsym = _static_coil_metadata(coils, n_segments)
    groups = _normalize_groups(current_groups, len(base_currents))
    planar_shape = (
        local_dofs.shape[0],
        local_dofs.shape[1],
        local_dofs.shape[2] - 1,
    )
    shape_coordinates = _normalize_shape_dofs(shape_dofs, planar_shape)
    center_coordinates = _normalize_coordinate_pairs(
        center_dofs,
        name="center_dofs",
        extents=centers.shape,
    )
    orientation_coordinates = _normalize_coordinate_pairs(
        orientation_dofs,
        name="orientation_dofs",
        extents=centers.shape,
    )

    local_shape_directions = np.zeros(
        (len(shape_coordinates), *local_dofs.shape), dtype=dofs.dtype
    )
    for direction, coordinate in zip(
        local_shape_directions, shape_coordinates, strict=True
    ):
        base_coil, local_axis, xy_index = coordinate
        direction[base_coil, local_axis, xy_index + 1] = 1
    center_directions = np.zeros(
        (len(center_coordinates), *centers.shape), dtype=dofs.dtype
    )
    for direction, coordinate in zip(
        center_directions, center_coordinates, strict=True
    ):
        direction[coordinate] = 1
    orientation_directions = np.zeros(
        (len(orientation_coordinates), *centers.shape), dtype=dofs.dtype
    )
    for direction, coordinate in zip(
        orientation_directions, orientation_coordinates, strict=True
    ):
        direction[coordinate] = 1

    fractional_base_directions = np.zeros(
        (len(groups), len(base_currents)), dtype=base_currents.dtype
    )
    for local_index, group in enumerate(groups):
        fractional_base_directions[local_index, group] = base_currents[group]
    nominal_expanded = apply_symmetries_to_currents(
        jnp.asarray(base_currents), nfp, stellsym
    )
    if groups:
        expanded_directions = jnp.stack(
            [
                apply_symmetries_to_currents(jnp.asarray(direction), nfp, stellsym)
                for direction in fractional_base_directions
            ]
        )
    else:
        expanded_directions = jnp.zeros(
            (0, nominal_expanded.shape[0]), dtype=nominal_expanded.dtype
        )

    return PlanarCoilDesignFieldBuilder(
        nominal_centers=jnp.asarray(centers),
        nominal_frames=jnp.asarray(frames),
        nominal_quaternions=jnp.asarray(nominal_quaternions),
        nominal_local_dofs=jnp.asarray(local_dofs),
        nominal_base_currents=jnp.asarray(base_currents),
        nominal_expanded_currents=nominal_expanded,
        fractional_base_directions=jnp.asarray(fractional_base_directions),
        fractional_expanded_directions=expanded_directions,
        local_shape_directions=jnp.asarray(local_shape_directions),
        center_directions=jnp.asarray(center_directions),
        orientation_directions=jnp.asarray(orientation_directions),
        current_groups=groups,
        shape_dofs=shape_coordinates,
        center_dofs=center_coordinates,
        orientation_dofs=orientation_coordinates,
        n_segments=segments,
        nfp=nfp,
        stellsym=stellsym,
    )


def make_fractional_current_field_builder(
    coils: Coils,
    groups: Any,
    *,
    n_segments: Any | None = None,
) -> CoilDesignFieldBuilder:
    """Build a current-only fractional parameterization."""

    return make_coil_design_field_builder(
        coils,
        current_groups=groups,
        n_segments=n_segments,
    )


def make_shape_field_builder(
    coils: Coils,
    *,
    shape_dofs: Any = (),
    shape_directions: Any | None = None,
    n_segments: Any | None = None,
) -> CoilDesignFieldBuilder:
    """Build a shape-only additive Fourier parameterization."""

    return make_coil_design_field_builder(
        coils,
        shape_dofs=shape_dofs,
        shape_directions=shape_directions,
        n_segments=n_segments,
    )
