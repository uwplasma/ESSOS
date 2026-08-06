"""Canonical coil-file loading entry points.

Use these functions when a file may contain either ordinary XYZ Fourier coils
or a constrained planar-coil representation.  The older ``Coils_from_json``
class remains available for legacy XYZ-only workflows.
"""

from essos.planar_coils import load_coils_json, load_simsopt_coils_json

__all__ = ["load_coils_json", "load_simsopt_coils_json"]
