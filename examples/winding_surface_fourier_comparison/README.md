# Fourier-coefficient winding-surface comparison

Compares a new SVD/entropy surrogate against the methods already tracked in
`examples/winding_surface_comparison_8_coils/`. Instead of representing the
winding surface's current potential Phi as dipole strengths at discrete grid
points (`induction_singular_values`, the existing `"ESSOS entropy"` method),
this represents Phi as a truncated Fourier series
(`sin(m*theta - n*nfp*phi)`, `mpol=ntor=6`, matching the REGCOIL validation
basis) and builds the induction matrix from those coefficients directly. See
`examples/winding_surface_opt_fourier.py` for the original small-scale
prototype and its own correctness benchmarks (flux conservation, an
independent Biot-Savart cross-check).

Not integrated into `winding_surface_comparison_8_coils/winding_surface_comparison.py`'s
`SURFACE_METHODS` (that file's `SURFACE_METHODS`-keyed color maps and
bar-chart centering offsets are a known footgun documented in
`examples/WINDING_SURFACE_AGENT_NOTES.md`; adding a method there is a
separate, larger change). Instead, `winding_surface_fourier_comparison.py`
is a self-contained script that imports that file's own helper functions
(`make_surface`, `baseline_surface`, `geometry_metrics`, `dipole_kernel`,
`quadrature_weights`, `build_operators`, and `worker()` itself for the 48x48
step) so every metric is defined identically to the tracked study.

## What's here

- `winding_surface_fourier_comparison.py` (in `examples/`, not this folder):
  optimizes the Fourier-coefficient surface, validates it with REGCOIL at
  48x48 (`comparison_metrics`-style), 48/56/64 (`sheet_resolution_convergence`-style),
  and 96x96 (`surface_validation_96`-style, including coil-cutting and an
  independent filament Biot-Savart evaluation), for all three PR cases
  (Landreman-Paul QA, Landreman-Paul QH, W7-X). Every REGCOIL-touching step
  runs as a watchdog-supervised subprocess (RSS-based polling, not
  `RLIMIT_AS` -- see the script's own comments for why) since this machine
  has a hard 12 GB memory ceiling.
- `winding_surface_fourier_comparison_plots.py` (in `examples/`): renders
  the figures in `figures/` from the CSVs below, styled to match
  `validate_surfaces()`/`sheet_resolution_study()`'s own bar/line charts.
- `data/*_combined.csv`: this run's output. Rows tagged `"(this machine)"`
  were optimized and validated fresh, on this machine, this session.
  `"ESSOS Pareto"` and `"REGCOIL adjoint"` rows are reused from the tracked,
  cross-machine `winding_surface_comparison_8_coils/data/*.csv` (rerunning
  those needs the ESSOS 32x32 Pareto current-solve and the legacy REGCOIL
  adjoint Fortran binary respectively; out of scope for this addition) and
  are shown hatched in the figures with that caveat.
- `figures/*.png`: the rendered comparison.

## Result (96x96, sheet fB, same machine)

| Case | normal offset | ESSOS entropy (dipole) | ESSOS Fourier entropy (new) |
|---|---:|---:|---:|
| Landreman-Paul QA | 0.000375 | 0.000677 | 0.000502 |
| Landreman-Paul QH | 25.20 | 17.48 | 11.90 |
| W7-X | 4.105 | 2.499 | 2.347 |

The Fourier method beats the dipole entropy method it's meant to replace on
all three cases (26% / 32% / 6% lower sheet fB), at 2-9x the optimization
wall-time (still under 20s absolute). See `figures/fourier_comparison_validation_96.png`
for the full metric set (max|Bn|/B, filament metrics, achieved Kmax) and
`figures/fourier_comparison_resolution_convergence.png` for 48/56/64
stability (both methods are resolution-stable; W7-X is missing the Fourier
method's resolution-64 point, aborted by the watchdog at 1.48 GiB available
against a 1.5 GiB floor -- a genuine, close-call memory constraint on this
machine, not a bug).

## Important: the tracked cross-machine data has a staleness gap for W7-X

While reproducing these numbers, `"normal offset"`, `"ESSOS entropy"`, and
(very likely) `"REGCOIL adjoint"`'s **W7-X** rows in the tracked
`winding_surface_comparison_8_coils/data/*.csv` and
`winding_surface_comparison/data/*.csv` were found to disagree substantially
(2-4x) with a fresh, same-machine rerun of the *current* code. This is not a
cross-machine numerical difference: the fresh rerun matches, to machine
precision, the untracked `winding_surface_comparison/output/` cache (dated
Sep 4), which postdates the tracked `data/` promotion (dated Sep 3) and was
never re-promoted -- exactly the gap `WINDING_SURFACE_AGENT_NOTES.md`
already flagged ("`output/` currently holds a completed run that has not
been promoted"). `"ESSOS Pareto"`'s W7-X row happens to be unaffected
(nearly identical between the stale and current data), which is why only
three of the four tracked methods show this gap.

**Practical effect**: any comparison against W7-X's currently-promoted
`"normal offset"` or `"ESSOS entropy"` numbers is comparing against stale
data. This affects this comparison too -- e.g. an earlier draft of this
analysis (before this was found) reported the Fourier method beating dipole
entropy on W7-X by 74%; the real, same-machine figure is 6%. The table above
already uses the corrected, freshly-rerun numbers. Promoting
`winding_surface_comparison_8_coils/data/*.csv` (and the primary
`winding_surface_comparison/data/*.csv`) from the current `output/` cache is
a separate, independent fix this addition does not make.

## Reproduce

```bash
cd examples
python3 winding_surface_fourier_comparison.py       # ~1-3 min, watchdog-protected
python3 winding_surface_fourier_comparison_plots.py  # regenerates figures/ from data/
```
