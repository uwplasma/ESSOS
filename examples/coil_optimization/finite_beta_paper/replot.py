"""Redraw the coil, axis-field and optimization figures of one archived run, without optimizing.

python replot.py RUN_DIR OUT_DIR [poincare]

RUN_DIR holds a run of optimize_coils_and_nearaxis_finite_beta.py (summary.json, coils_*.json and
axis_targets_*.npz). The near-axis states are rebuilt from the saved axis and the case inputs, the
coils are read back, and the targets and boundary mismatch recomputed, all at the example's
diagnostic resolution. With "poincare" the coil field lines are traced as well (vacuum runs only).
Figures are sized 6.6 in wide, without suptitles, like run.py.
"""
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nearaxis_finite_beta_helpers as helpers
from essos.coils import Coils
from essos.fields import BiotSavart
from pyqsc_jax.near_axis import near_axis

NPHI_DIAGNOSTIC = 151
FLUX_LEVELS = (0.0625, 0.25, 0.5625, 1.0)
WIDTH = 6.6

run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
_subplots = plt.subplots


def _journal_subplots(*args, **kwargs):
    if "figsize" in kwargs:
        width, height = kwargs["figsize"]
        kwargs["figsize"] = (WIDTH, height * WIDTH / width)
    return _subplots(*args, **kwargs)


plt.subplots = _journal_subplots
matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

summary = json.loads((run / "summary.json").read_text())
inputs, radius = summary["inputs"], summary["inputs"]["a"]
states = {}
for name in ("initial", "optimized"):
    saved = np.load(run / f"axis_targets_{name}.npz")
    near = near_axis(rc=jnp.asarray(saved["rc"]), zs=jnp.asarray(saved["zs"]), etabar=float(saved["etabar"]),
                     nfp=inputs["nfp"], nphi=NPHI_DIAGNOSTIC, order="r3", B0=inputs["B0"], I2=inputs["I2"],
                     p2=inputs["p2"], B2c=inputs["B2c"])
    solution = near.solution
    field = BiotSavart(Coils.from_json(str(run / f"coils_{name}.json")))
    targets = helpers.coil_targets(solution, radius)
    surface, normal = helpers.normal_field_error(solution, targets, field, radius)
    states[name] = dict(field=field, near=near, solution=solution, targets=targets, surface=surface, normal=normal,
                        match=helpers.axis_match(solution, targets, field))
    archived = summary["states"][name]["boundary"]
    print(f"{name}: B.n/|B| max {100 * normal['normal_error_max']:.4g} % (archived "
          f"{100 * archived['normal_error_max']:.4g} %), RMS {100 * normal['normal_error_rms']:.4g} % "
          f"(archived {100 * archived['normal_error_rms']:.4g} %)")

optimized = states["optimized"]
helpers.plot_coils_and_normal_error(states, {n: s["surface"] for n, s in states.items()},
                                    out / "coils_and_normal_field.png", "")
helpers.plot_axis_profiles(optimized["solution"], optimized["targets"], optimized["field"], out / "axis_fields.png", "")
helpers.plot_optimization(summary["cost_history"], {n: s["match"] for n, s in states.items()},
                          out / "optimization.png", "")
if "poincare" in sys.argv[3:]:
    sections, rows = helpers.trace_poincare(optimized["field"], optimized["solution"], radius, FLUX_LEVELS)
    for row in rows:
        print(f"   s = {row['s']:<7g} phi = {row['plane']:g} period: RMS distance / flux radius = "
              f"{row['rms_over_flux_radius']:.3e}")
    helpers.plot_poincare(optimized["solution"], sections, radius, FLUX_LEVELS, out / "poincare.png", "")
print(sorted(p.name for p in out.glob("*.png")))
