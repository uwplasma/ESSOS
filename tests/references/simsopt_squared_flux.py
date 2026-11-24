"""SIMSOPT reference for tests/test_optimization_anchors.py::test_squared_flux_matches_simsopt.

Squared flux of the Landreman-Paul QA coils (examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json)
on the boundary of examples/input_files/input.LandremanPaul2021_QA_beta0p5_bootstrap, full torus,
32 x 32 points, for the three definitions, with the derivative with respect to the base-curve Fourier
coefficients (SIMSOPT CurveXYZFourier order: xc0, xs1, xc1, ... then y, then z) and the base currents.
Run with SIMSOPT installed (not a test dependency); prints the numbers pasted into the test.
"""
import json
from pathlib import Path
import numpy as np
import simsopt
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier
from simsopt.objectives import SquaredFlux

INPUTS = Path(__file__).resolve().parents[2] / "examples" / "input_files"
NPHI = NTHETA = 32

data = json.loads((INPUTS / "ESSOS_biot_savart_LandremanPaulQA.json").read_text())
curves, currents = [], []
for dofs, current in zip(data["dofs_curves"], data["dofs_currents"]):
    curve = CurveXYZFourier(data["n_segments"], data["order"])
    curve.x = np.ravel(dofs)
    curves.append(curve)
    currents.append(Current(current))
field = BiotSavart(coils_via_symmetries(curves, currents, data["nfp"], data["stellsym"]))
surface = SurfaceRZFourier.from_vmec_input(str(INPUTS / "input.LandremanPaul2021_QA_beta0p5_bootstrap"),
                                           range="full torus", nphi=NPHI, ntheta=NTHETA)
field.set_points(surface.gamma().reshape((-1, 3)))
direction = np.sin(1.0 + np.arange(sum(c.x.size for c in curves)))
print("simsopt", simsopt.__version__)
for definition in ("quadratic flux", "normalized", "local"):
    objective = SquaredFlux(surface, field, definition=definition)
    derivative = objective.dJ(partials=True)
    grad_curves = np.concatenate([derivative(c) for c in curves])
    grad_currents = np.array([derivative(c)[0] for c in currents])
    print(repr(definition), repr(objective.J()), repr(float(grad_curves @ direction)),
          [float(g) for g in grad_currents], [float(g) for g in grad_curves[:3]])
