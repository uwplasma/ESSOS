"""Measured relative errors of the fixed-Cartesian gradient/Hessian extraction (all cases, radii).

python cartesian_table.py PYQSC_TESTS_PHYSICS -> cartesian_table.json
"""
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, sys.argv[1])
import test_plasma_cartesian_derivatives as t

out = {}
for key, (_, _, radii) in t.CASES.items():
    out[key] = {}
    for radius in radii:
        result = t.measure(key, radius)
        gradient, hessian = t._errors(result)
        out[key][str(radius)] = dict(gradient=gradient, hessian=hessian,
                                     antisymmetric_gradient_norm=float(np.linalg.norm(
                                         (result["predicted_gradient"] - result["predicted_gradient"].T) / 2)
                                         / np.linalg.norm(result["predicted_gradient"])),
                                     inverse_residual=float(result["inverse_residual"]))
        print(key, radius, gradient, hessian, flush=True)
Path("cartesian_table.json").write_text(json.dumps(out, indent=1) + "\n")
