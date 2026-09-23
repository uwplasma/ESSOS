"""Relative error of the matched on-axis plasma field against the unexpanded volume integral.

python volume_table.py -> volume_table.json. Uses the reference quadrature of pyQSC_JAX's
tests/physics/test_plasma_volume_integral.py (300 toroidal nodes per side, 24 radial, 48
poloidal), whose source current is rebuilt from the equilibrium coefficients.
"""
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

import pyqsc_jax as qsc

sys.path.insert(0, sys.argv[1])  # pyQSC_JAX tests/physics
import test_plasma_volume_integral as v

CASES = {"pressure-only": ({**v.QA, "I2": 0.0, "p2": -6.0e5}, v.SECTION),
         "current-and-pressure": ({**v.QA, "I2": 0.6, "p2": -6.0e5}, v.SECTION),
         "axisymmetric": ({**v.AXISYMMETRIC, "I2": 0.9, "p2": -2.0e5}, 3)}
rows = {}
for name, (parameters, section) in CASES.items():
    solution = v._solve(**parameters)
    rows[name] = {}
    for radius in (0.08, 0.04, 0.02, 0.01):
        matched = np.asarray(qsc.plasma_field_on_axis(solution, formal_radius=radius).field)[section]
        direct = v.volume_field(solution, radius, section)[1]
        rows[name][str(radius)] = float(np.linalg.norm(matched - direct) / np.linalg.norm(direct))
        print(name, radius, rows[name][str(radius)], flush=True)
Path("volume_table.json").write_text(json.dumps(dict(nphi=61, nodes=[300, 24, 48], errors=rows), indent=1) + "\n")
