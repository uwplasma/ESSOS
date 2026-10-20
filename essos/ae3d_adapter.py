"""AE3D potential tables with the paper-pinned FIRM3D conventions.

This format explicitly contains the harmonics used as Phi by FIRM3D.
Signed amplitudes in volts and full toroidal numbers are preserved.
Natural cubic radial interpolation is the default; choose linear explicitly
to reproduce FIRM3D's archived piecewise-linear profiles.
"""

from dataclasses import dataclass
import hashlib
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from essos.perturbations import BoozerPerturbation


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Read the upstream object-array format without importing arbitrary globals."""

    def find_class(self, module, name):
        core = np._core if hasattr(np, "_core") else np.core
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(core.multiarray, name)
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return core.numeric._frombuffer
        raise ValueError(f"Unsupported global in AE3D array: {module}.{name}")


def _read_dictionary(path):
    with open(path, "rb") as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError("Unsupported upstream NumPy file version")
        shape, _, dtype = reader(stream)
        if shape != () or not dtype.hasobject:
            raise ValueError("Expected FIRM3D's scalar object-array dictionary")
        value = _NumpyOnlyUnpickler(stream).load()
    if not isinstance(value, np.ndarray) or value.shape != () or not isinstance(value.item(), dict):
        raise ValueError("Invalid upstream AE3D dictionary")
    return value.item()


@dataclass(frozen=True)
class AE3DMode:
    s: np.ndarray
    harmonics: np.ndarray
    potential_volts: np.ndarray
    omega: float
    source_sha256: str

    @classmethod
    def from_firm3d_numpy(cls, path):
        """Read the pinned AE3DEigenvector export; eigenvalue is (krad/s)^2.

        FIRM3D bc65bab's from_ae3d uses omega=sqrt(eigenvalue)*1000,
        without an extra 2*pi or amplitude normalization. No input scripts
        are executed. The NumPy-only unpickler supports the upstream format.
        """
        from pathlib import Path

        data = _read_dictionary(path)
        if set(data) != {"eigenvalue", "s_coords", "harmonics"}:
            raise ValueError("Unexpected upstream AE3D dictionary keys")
        eigenvalue = float(data["eigenvalue"])
        if not np.isfinite(eigenvalue) or eigenvalue <= 0:
            raise ValueError("Positive finite AE3D eigenvalue required")
        rows = data["harmonics"]
        labels = np.array([[row[0], row[1]] for row in rows])
        profiles = np.column_stack([np.asarray(row[2], float) for row in rows])
        mode = cls(np.asarray(data["s_coords"], float), labels, profiles,
                   np.sqrt(eigenvalue)*1000, hashlib.sha256(Path(path).read_bytes()).hexdigest())
        # Run the same table validation used when constructing the JAX model.
        mode.as_perturbation()
        return mode

    def as_perturbation(self, *, interpolation="natural-cubic"):
        return TabulatedAlfvenMode(self.s, self.harmonics, self.potential_volts, self.omega,
                                   interpolation=interpolation)


class TabulatedAlfvenMode(BoozerPerturbation):
    """AE3D Phi(s) tables with the FIRM3D sine angular/time convention.

    The inherited RHS/derivative interface is unchanged. Natural cubic
    interpolation in s is the default. Explicit linear interpolation retains
    FIRM3D's radial derivative jumps at table knots. The cubic option differs
    from the archived model and ASCOT's interpolation in its rho coordinate.
    For m!=0, the upstream constructor adds Phi(0)=0 if the first sample is
    off axis; m=0 instead receives the first value. Values clamp beyond the
    last sample. Clamping can retain endpoint derivative jumps in either mode.
    Neither choice rescales amplitudes or applies a radial taper.
    """

    s: jax.Array
    phi: jax.Array
    m: jax.Array
    n: jax.Array
    omega: jax.Array
    interpolation: str = eqx.field(static=True)
    radial_coefficients: jax.Array | None

    def __init__(self, s, harmonics, potential_volts, omega, *, interpolation="natural-cubic"):
        if interpolation not in ("linear", "natural-cubic"):
            raise ValueError("interpolation must be 'linear' or 'natural-cubic'")
        grid, modes, values = np.asarray(s, float), np.asarray(harmonics), np.asarray(potential_volts, float)
        if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
                or np.any(np.diff(grid) <= 0) or grid[0] < 0 or grid[-1] > 1):
            raise ValueError("Need increasing finite radial samples in [0,1]")
        if (modes.ndim != 2 or modes.shape[1] != 2 or len(modes) == 0
                or not np.isfinite(modes).all() or np.any(modes != np.round(modes))
                or len(set(map(tuple, modes))) != len(modes)):
            raise ValueError("Need unique integer (m,n) harmonics")
        if values.shape != (len(grid), len(modes)) or not np.isfinite(values).all():
            raise ValueError("Potential table must be finite, with shape (surface,harmonic)")
        if not np.isfinite(omega) or omega == 0:
            raise ValueError("Need a finite nonzero angular frequency")
        if grid[0] > 0:
            values = np.vstack([np.where(modes[:, 0] == 0, values[0], 0.), values])
            grid = np.r_[0., grid]
        elif np.any(values[0, modes[:, 0] != 0] != 0):
            raise ValueError("Nonzero-m potential must vanish on axis for a finite GC derivative")
        self.s, self.phi = jnp.asarray(grid), jnp.asarray(values)
        self.m, self.n = jnp.asarray(modes[:, 0], int), jnp.asarray(modes[:, 1], int)
        self.omega = jnp.asarray(omega)
        self.interpolation = interpolation
        self.radial_coefficients = None
        if interpolation == "natural-cubic":
            from scipy.interpolate import CubicSpline
            self.radial_coefficients = jnp.asarray(CubicSpline(grid, values, axis=0, bc_type="natural").c)

    def potentials(self, field, point, t):
        u, w, zeta = point
        s = u*u+w*w
        # Phi_nonzero_m=O(s) has a unique C1 extension at u=w=0.
        # Avoid differentiating atan2(0,0); this branch changes only that point.
        theta = jnp.arctan2(w, jnp.where(s > 0, u, 1.))
        sample = jnp.clip(s, self.s[0], self.s[-1])
        index = jnp.clip(jnp.searchsorted(self.s, sample, side="right")-1, 0, len(self.s)-2)
        if self.interpolation == "linear":
            weight = (sample-self.s[index])/(self.s[index+1]-self.s[index])
            phi = self.phi[index]+weight*(self.phi[index+1]-self.phi[index])
        else:
            d, c = sample-self.s[index], self.radial_coefficients[:, index]
            phi = ((c[0]*d+c[1])*d+c[2])*d+c[3]
        harmonic = phi*jnp.sin(self.m*theta-self.n*zeta+self.omega*t)
        (iota, G, I), _ = field.profiles(s)
        alpha = (self.n-self.m*iota)*harmonic/(self.omega*(G+iota*I))
        return jnp.stack([jnp.sum(alpha), jnp.sum(harmonic)])
