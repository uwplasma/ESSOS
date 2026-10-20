"""Optional FAX mode interchange; importing this module does not import FAX.

FAX's current reduced eigenvector is not a calibrated electromagnetic field.
The only supported closure explicitly interprets its scalar envelope as a
potential shape. No displacement-to-potential identification is implicit.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FaxMode:
    s: np.ndarray
    harmonics: np.ndarray
    structure: np.ndarray
    frequency_hz: float
    metadata: dict

    def __post_init__(self):
        s, modes, values = np.array(self.s, float), np.array(self.harmonics), np.array(self.structure, complex)
        if (s.ndim != 1 or len(s) < 4 or not np.isfinite(s).all() or np.any(np.diff(s) <= 0)
                or s[0] < 0 or s[-1] > 1):
            raise ValueError("Invalid mode radial grid")
        if (modes.ndim != 2 or modes.shape[1] != 2 or not len(modes)
                or not np.isfinite(modes).all() or np.any(modes != np.round(modes))):
            raise ValueError("Invalid integer harmonic labels")
        if len(set(map(tuple, modes))) != len(modes):
            raise ValueError("Duplicate harmonic")
        if values.shape != (len(s), len(modes)) or not np.isfinite(values).all() or not np.any(abs(values) > 0):
            raise ValueError("Invalid complex mode structure")
        if np.ndim(self.frequency_hz) or not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("Mode frequency must be positive and in Hz")
        if self.metadata.get("interpretation") != "potential_shape":
            raise ValueError("Explicit interpretation='potential_shape' is required for reduced FAX envelopes")
        if self.metadata.get("coordinates") != "essos_boozer":
            raise ValueError("Mode coordinates must explicitly be essos_boozer")
        digest = self.metadata.get("equilibrium_sha256", "")
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError("An equilibrium SHA256 is required")
        for value in (s, modes, values):
            value.setflags(write=False)
        object.__setattr__(self, "s", s)
        labels = modes.astype(int)
        labels.setflags(write=False)
        object.__setattr__(self, "harmonics", labels)
        object.__setattr__(self, "structure", values)
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata)))

    @classmethod
    def from_result(cls, problem, result, mode_index, *, equilibrium, interpretation, source="fax"):
        """Capture an in-memory FAX result calculated on the stated equilibrium.

        Caller must use the same Boozer angles/flux convention as ESSOS.
        The producer is responsible for that contract; hashing the wout alone
        cannot certify a coordinate convention or an upstream calculation.
        """
        if isinstance(mode_index, bool) or not isinstance(mode_index, (int, np.integer)):
            raise ValueError("mode_index must be an integer")
        if not 0 <= mode_index < len(result.frequencies_hz):
            raise ValueError("mode_index outside calculated spectrum")
        residual = float(np.asarray(result.residuals)[mode_index])
        omega2 = float(np.asarray(result.omega2)[mode_index])
        if not np.isfinite(residual) or residual > 1e-7 or not np.isfinite(omega2) or omega2 <= 0:
            raise ValueError("Selected mode fails the positive-eigenvalue/residual gate")
        frequency = float(np.asarray(result.frequencies_hz)[mode_index])
        if not np.isclose(omega2, (2*np.pi*frequency)**2, rtol=1e-10):
            raise ValueError("FAX frequency/omega2 units disagree")
        return cls(np.asarray(problem.s), np.asarray(problem.harmonics),
                   np.asarray(result.mode_structures)[mode_index], frequency,
                   dict(interpretation=interpretation, coordinates="essos_boozer", source=source,
                        equilibrium_sha256=hashlib.sha256(Path(equilibrium).read_bytes()).hexdigest(),
                        mode_index=int(mode_index), eigenpair_residual=residual,
                        operator_metadata=list(getattr(problem, "metadata", ())),
                        polarization="scalar envelope assigned to Phi; alpha from reduced ideal relation"))

    def save(self, path):
        """Portable versioned NPZ, with JSON metadata and no pickle objects."""
        np.savez_compressed(path, s=self.s, harmonics=self.harmonics, structure=self.structure,
                            frequency_hz=self.frequency_hz,
                            metadata_json=json.dumps(dict(self.metadata, schema_version=1)))

    @classmethod
    def load(cls, path, *, equilibrium):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.pop("schema_version", None) != 1:
                raise ValueError("Unsupported FAX mode schema")
            mode = cls(data["s"], data["harmonics"], data["structure"], float(data["frequency_hz"]), metadata)
        if hashlib.sha256(Path(equilibrium).read_bytes()).hexdigest() != mode.metadata["equilibrium_sha256"]:
            raise ValueError("Mode and tracing equilibrium hashes differ")
        return mode

    def as_perturbation(self, *, potential_volts, frequency_sign, taper_width):
        """Assign a peak Fourier-envelope scale and explicit propagation sign.

        Preserve all relative complex phases. Annular modes use a C2 taper
        of the specified width before zero extension. The amplitude in volts
        is an input; FAX does not predict the saturated amplitude.
        """
        from essos.mode_reconstruction import SpectralAlfvenMode

        if frequency_sign not in (-1, 1) or isinstance(frequency_sign, bool):
            raise ValueError("frequency_sign must explicitly be -1 or +1")
        if self.s[0] <= 0:
            raise ValueError("Current FAX scalar-envelope conversion needs strictly off-axis samples")
        if not np.isfinite(potential_volts):
            raise ValueError("potential_volts must be finite")
        coefficients = self.structure/np.max(abs(self.structure))
        coefficients = coefficients/self.s[:, None]**(abs(self.harmonics[:, 0])/2)
        return SpectralAlfvenMode(self.s, self.harmonics, coefficients,
            frequency_sign*2*np.pi*self.frequency_hz, amplitude=potential_volts, taper_width=taper_width)
