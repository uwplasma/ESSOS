"""User-facing selectors for freezing ESSOS optimization degrees of freedom."""
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class FrozenDOFs:
    """Mixin for losses and constraints whose DOFs are built from dependencies.

    Subclasses provide ``dependencies`` and ``_dependency_dof_names`` in the
    same order used to construct ``starting_dofs``.
    """
    def _init_frozen_dofs(self):
        self._frozen_dofs_mask = None
        self._frozen_dofs_values = None

    def freeze_dofs(self, mask):
        """Freeze a flat boolean mask, or a matching pytree, at initial values."""
        if not hasattr(mask, "shape"):
            mask, _ = ravel_pytree(mask)
        mask = jnp.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != self.starting_dofs.shape:
            raise ValueError(
                f"freeze mask has shape {mask.shape}; expected {self.starting_dofs.shape}."
            )
        self._frozen_dofs_mask = mask
        self._frozen_dofs_values = self.starting_dofs
        return self

    def unfreeze_dofs(self):
        """Remove every frozen-DOF selection from this loss or constraint."""
        self._init_frozen_dofs()
        return self

    def _project_dofs(self, dofs):
        if self._frozen_dofs_mask is None:
            return dofs
        return jnp.where(self._frozen_dofs_mask, self._frozen_dofs_values, dofs)

    def _mask_gradient(self, gradient):
        if self._frozen_dofs_mask is None:
            return gradient
        return jnp.where(self._frozen_dofs_mask, 0, gradient)

    def _dependency_offset(self, dependency):
        names = tuple(self._dependency_dof_names)
        if dependency not in names:
            raise KeyError(
                f"{dependency!r} is not an optimizable dependency. Available: {names}."
            )
        preceding = tuple(self.dependencies[name] for name in names[:names.index(dependency)])
        return 0 if not preceding else ravel_pytree(preceding)[0].size

    def _add_frozen_indices(self, indices):
        mask = (jnp.zeros_like(self.starting_dofs, dtype=bool)
                if self._frozen_dofs_mask is None else self._frozen_dofs_mask)
        indices = jnp.asarray(indices, dtype=int).reshape(-1)
        if jnp.any(indices < 0) or jnp.any(indices >= mask.size):
            raise IndexError("selected DOF index is outside starting_dofs.")
        return self.freeze_dofs(mask.at[indices].set(True))

    @staticmethod
    def _indices(values, size, label):
        values = [values] if isinstance(values, int) else list(values)
        if any(value < 0 or value >= size for value in values):
            raise IndexError(f"{label} must be in [0, {size - 1}].")
        return values

    def freeze_current(self, dependency="field", coil=0):
        """Freeze normalized current DOFs for one or more base coils."""
        field = self.dependencies[dependency]
        coils = field.coils
        current_indices = self._indices(coil, coils.dofs_currents.size, "coil index")
        start = self._dependency_offset(dependency) + coils.dofs_curves.size
        return self._add_frozen_indices([start + index for index in current_indices])

    def freeze_coil(self, dependency="field", coil=0):
        """Freeze every curve coefficient of one or more base coils."""
        field = self.dependencies[dependency]
        shape = field.coils.dofs_curves.shape
        coil_indices = self._indices(coil, shape[0], "coil index")
        per_coil = shape[1] * shape[2]
        start = self._dependency_offset(dependency)
        return self._add_frozen_indices(
            [start + coil_index * per_coil + local for coil_index in coil_indices for local in range(per_coil)]
        )

    def freeze_curve_dofs(self, dependency="field", coil=0, coordinates=None, modes=None):
        """Freeze selected stored curve coefficients of one or more base coils.

        ``coordinates`` accepts ``"x"``, ``"y"``, ``"z"`` (or 0, 1, 2),
        and ``modes`` are indices into ``Curves.dofs[..., mode]``.
        """
        field = self.dependencies[dependency]
        shape = field.coils.dofs_curves.shape
        coil_indices = self._indices(coil, shape[0], "coil index")
        if coordinates is None:
            coordinates = range(shape[1])
        if modes is None:
            modes = range(shape[2])
        coordinate_map = {"x": 0, "y": 1, "z": 2}
        if isinstance(coordinates, (int, str)):
            coordinates = [coordinates]
        coordinates = [coordinate_map.get(value, value) for value in coordinates]
        coordinates = self._indices(coordinates, shape[1], "coordinate index")
        modes = self._indices(modes, shape[2], "mode index")
        start = self._dependency_offset(dependency)
        per_coil = shape[1] * shape[2]
        return self._add_frozen_indices([
            start + coil_index * per_coil + coordinate * shape[2] + mode
            for coil_index in coil_indices for coordinate in coordinates for mode in modes
        ])

    def freeze_surface_modes(self, dependency="surface", rc=(), zs=()):
        """Freeze ``SurfaceRZFourier`` coefficients by physical ``(m, n)`` mode."""
        surface = self.dependencies[dependency]
        start = self._dependency_offset(dependency)

        def mode_indices(modes, coefficients, component):
            out = []
            for m, n in modes:
                matches = jnp.where((surface.xm == m) & (surface.xn == n))[0]
                if matches.size != 1:
                    raise KeyError(f"{component}({n},{m}) is not present on this surface.")
                out.append(start + coefficients + int(matches[0]))
            return out

        indices = mode_indices(rc, 0, "RBC")
        indices += mode_indices(zs, surface.rc.size, "ZBS")
        if not indices:
            raise ValueError("select at least one rc or zs surface mode.")
        return self._add_frozen_indices(indices)
