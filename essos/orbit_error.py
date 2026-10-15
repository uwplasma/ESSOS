"""Physical error budgets for collisionless particle orbits, without Diffrax.

Both tolerances act on nondimensional physical blocks: position / L and
velocity / V. L is a reference length (one metre by default), and V is the
particle's characteristic speed, including at parallel-velocity zero crossings.
The larger Euclidean block error controls acceptance, so neither block can be
diluted by extra coordinates or diagnostic state.
"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def orbit_block_norm(error, velocity_components):
    """Maximum of position and velocity Euclidean norms after scaling."""
    return jnp.maximum(jnp.linalg.norm(error[:3]), jnp.linalg.norm(error[3:3 + velocity_components]))


@dataclass(frozen=True, eq=False)
class OrbitErrorNorm:
    """Embedded orbit error in physical position and velocity blocks.

    ``position_map``, when supplied, maps an entire integration state to its
    Cartesian position in metres. Comparing the embedded endpoints handles
    periodic coordinates and regular axis charts without dividing by radius.
    Its result is a local embedded estimate, not a global trajectory bound.

    ``velocity_components`` is one for guiding centers and three for full
    orbits; any remaining state (constant mu, failure flag) is excluded.
    A chart angle that affects position must be included by ``position_map``.
    """

    rtol: object
    atol: object
    speed_scale: object
    length_scale: object = 1.
    velocity_components: int = 1
    position_map: Callable | None = None

    def __post_init__(self):
        if self.velocity_components not in (1, 3):
            raise ValueError("velocity_components must be 1 or 3")
        for name in ("rtol", "atol", "speed_scale", "length_scale"):
            value = getattr(self, name)
            if np.ndim(value) != 0:
                raise ValueError(f"orbit {name} must be scalar")
            if not isinstance(value, jax.core.Tracer):
                value = np.asarray(value)
                valid = np.isfinite(value) & (value >= 0 if name == "rtol" else value > 0)
                if not bool(valid):
                    raise ValueError(f"orbit {name} must be finite and {'nonnegative' if name == 'rtol' else 'positive'}")

    def __call__(self, y0, y1, error):
        del y0  # Reference scales do not depend on the coordinate origin.
        if self.position_map is None:
            position_error = error[:3]
        else:
            position_error = self.position_map(y1) - self.position_map(y1 - error)
        velocity_error = error[3:3 + self.velocity_components]
        budget = jnp.asarray(self.atol) + jnp.asarray(self.rtol)
        normalized = jnp.concatenate([position_error / (budget * self.length_scale),
                                      velocity_error / (budget * self.speed_scale)])
        value = orbit_block_norm(normalized, self.velocity_components)
        valid = ((self.length_scale > 0) & jnp.isfinite(self.length_scale) &
                 (self.speed_scale > 0) & jnp.isfinite(self.speed_scale) &
                 (self.rtol >= 0) & (self.atol > 0) & jnp.isfinite(budget) &
                 jnp.isfinite(y1).all() & jnp.isfinite(error).all() & jnp.isfinite(value))
        return jnp.where(valid, value, jnp.inf)
