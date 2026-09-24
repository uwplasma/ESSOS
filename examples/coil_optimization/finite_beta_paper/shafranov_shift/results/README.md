# Result bundle

This is the compact, tracked subset of the fixed-coil runs. The binary WOUT
files and full VMEX logs are retained in the local ignored run folders; each
manifest and pressure-point record stores the corresponding input and WOUT
SHA-256 so an independently reproduced file can be checked.

- `runs/qa12`, `runs/qa15`, and `runs/qa18` hold the three accepted five-point
  pressure families, their signed-flux vacuum input, measured surface
  certificate, theory response arrays, and full axis/profile results.
- `runs/qa18_ns129` holds the selected cold-from-vacuum higher-radial-resolution
  point and its matching vacuum gate.
- `summary.json`, `pressure_points.csv`, `response_profiles.csv`, and
  `fit_residuals.csv` are written by `../analyze_results.py`.
- `resolution_check.csv` quantifies the 65/129 radial sensitivity;
  `quadrature_check.csv` compares direct-field preflights with 240, 480, and
  960 coil segments; `error_budget.csv` keeps measured and unresolved terms
  separate.
- `figures/` contains actual-simulation diagnostic PDFs, SVGs, and 400 dpi
  PNGs. They have no uncertainty bars and are not manuscript-validated.

No raw binary simulation archive is committed here. The full handoff text and
the local raw-run paths are recorded in the parent README and the PR #70
implementation update.
