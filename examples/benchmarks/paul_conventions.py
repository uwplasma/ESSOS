"""Explicit convention conversion for benchmark inputs, outside the GC RHS.

For a stellarator-symmetric Boozer |B| spectrum, historical positive-psi0
equations map to ESSOS's negative-psi0 equations through
(theta,zeta,v_parallel) -> (-theta,-zeta,-v_parallel). The same map requires
(omega,Phi_hat,phase) -> (-omega,-Phi_hat,-phase) for the analytical harmonic.
The public March 2023 SIMSOPT adapter used the positive flux convention.
That observation does not establish the exact runtime used by Paul (2023).
"""

import json
from pathlib import Path
import tomllib

import numpy as np


CONVENTIONS = ("essos", "simsopt_2023")


def particle_parameters(model, *, measured_mass, energy_joule):
    """Explicit benchmark mass at fixed energy; no changes to ESSOS constants."""
    _validate(model)
    if model == "simsopt_2023":
        reference = Path(__file__).with_name("references")/"paul2023_particle.toml"
        historical = tomllib.loads(reference.read_text())["historical_particle"]
        mass, source = historical["mass_kg"], historical["source"]
    else:
        mass, source = measured_mass, "essos.constants.ALPHA_PARTICLE_MASS"
    return dict(particle_model=model, mass_kg=float(mass), energy_joule=float(energy_joule),
        birth_speed_m_s=float(np.sqrt(2*energy_joule/mass)), particle_model_source=source)


def _validate(source):
    if source not in CONVENTIONS:
        raise ValueError(f"Unknown source convention {source!r}; expected {CONVENTIONS}")


def convert_births(s, theta, zeta, pitch, *, source):
    """Return ESSOS-coordinate births; preserve the caller's arrays."""
    _validate(source)
    arrays = [np.array(value, dtype=float, copy=True) for value in (s, theta, zeta, pitch)]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("Birth coordinates and pitch must have identical shapes")
    if source == "simsopt_2023":
        for value in arrays[1:]:
            value *= -1
    return tuple(arrays)


def load_essos_births(path, particles, *, source):
    """Read already-converted births, rejecting a mismatched wave convention.

    Older archives without convention metadata used the ESSOS convention.
    Loading an archive must never apply the reflection a second time.
    """
    _validate(source)
    with np.load(path) as saved:
        metadata = json.loads(str(saved["metadata_json"])) if "metadata_json" in saved else {}
        archive_source = metadata.get("source_convention", "essos")
        if archive_source != source:
            raise ValueError(f"Birth archive source convention {archive_source!r} differs from {source!r}")
        if metadata.get("birth_coordinate_convention", "essos") != "essos":
            raise ValueError("Saved births must already use ESSOS coordinates")
        births = tuple(np.array(saved[f"birth_{key}"][:particles], copy=True)
                       for key in ("s", "theta", "zeta", "pitch"))
    if any(value.shape != (particles,) for value in births):
        raise ValueError("Birth archive has fewer markers than requested or invalid dimensions")
    return births


def convert_mode(mode, *, source):
    """Convert an analytical harmonic while retaining m, n and its envelope."""
    _validate(source)
    if source == "essos":
        return mode
    import equinox as eqx
    from essos.perturbations import AnalyticalAlfvenMode

    if not isinstance(mode, AnalyticalAlfvenMode):
        raise TypeError("The benchmark conversion requires an AnalyticalAlfvenMode")
    return eqx.tree_at(lambda wave: (wave.omega, wave.amplitude, wave.phase), mode,
                       (-mode.omega, -mode.amplitude, -mode.phase))
