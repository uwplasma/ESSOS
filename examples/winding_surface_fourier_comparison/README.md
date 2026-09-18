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
  the "vs other methods" figures in `figures/` from the CSVs below, styled to
  match `validate_surfaces()`/`sheet_resolution_study()`'s own bar/line
  charts.
- `winding_surface_fourier_comparison_v1_vs_v2_plots.py` (in `examples/`):
  renders the two `fourier_v1_vs_v2_*.png` figures below from a snapshot of
  the pre-optimization run (see "Performance" below).
- `data/*_combined.csv`: this run's output. Rows tagged `"(this machine)"`
  were optimized and validated fresh, on this machine, this session.
  `"ESSOS Pareto"` and `"REGCOIL adjoint"` rows are reused from the tracked,
  cross-machine `winding_surface_comparison_8_coils/data/*.csv` (rerunning
  those needs the ESSOS 32x32 Pareto current-solve and the legacy REGCOIL
  adjoint Fortran binary respectively; out of scope for this addition) and
  are shown hatched in the figures with that caveat.
- `figures/*.png`: the rendered comparisons.

## Result (96x96, sheet fB, same machine)

| Case | normal offset | ESSOS entropy (dipole) | ESSOS Fourier entropy (new) |
|---|---:|---:|---:|
| Landreman-Paul QA | 0.000375 | 0.000677 | 0.000502 |
| Landreman-Paul QH | 25.20 | 17.48 | 11.90 |
| W7-X | 4.105 | 2.499 | 2.347 |

The Fourier method beats the dipole entropy method it's meant to replace on
all three cases (26% / 32% / 6% lower sheet fB). See
`figures/fourier_comparison_validation_96.png` for the full metric set
(max|Bn|/B, filament metrics, achieved Kmax) and
`figures/fourier_comparison_resolution_convergence.png` for 48/56/64
stability (all three methods are resolution-stable; W7-X's `"normal offset"`
row is missing its resolution-64 point, watchdog-aborted at the edge of
available memory consistently across repeated retries -- see "Memory
watchdog" below, not a bug).

## Performance

The first working version of `fourier_induction_matrix` took noticeably
longer to optimize than the dipole method it's meant to replace (2-9x its
wall-time). A follow-up investigation profiled it, found the per-iteration
induction-matrix construction was the dominant cost (not one-time JIT
compilation), and applied two resolution-independent, machine-precision-exact
optimizations to the same function:

1. **nfp-fold periodicity reduction.** The Fourier basis functions
   `sin(m*theta - n*nfp*phi)` are *exactly* periodic with period `2*pi/nfp`
   in `phi` (`xn` is always an integer multiple of `nfp` -- see
   `potential_modes`), so the field they produce is itself exactly
   nfp-periodic. Only one field period of plasma points is needed, and the
   kernel contribution from all `nfp` copies of each winding point can be
   summed *before* contracting with the potential basis, rather than after --
   an `O(nfp^2) -> O(nfp)` reduction in the dominant matrix-construction
   cost. Simpler than the dipole method's `induction_singular_values`, which
   needs a complex per-mode DFT because raw grid DOFs have no built-in
   periodicity; the Fourier basis functions already do, so a plain
   real-valued sum over rotations suffices.
2. **Matmul-restructured dipole kernel.** `ws.dipole_kernel` materializes a
   `(P,Q,3)` pairwise-difference tensor; the new `dipole_kernel_matmul` is
   algebraically identical but uses the `|a-b|^2 = |a|^2 - 2 a.b + |b|^2`
   identity to replace it with four `(P,3)@(3,Q)` matrix multiplications
   (BLAS GEMM) plus small vector reductions -- no `(P,Q,3)` intermediate.

Both were verified (see `figures/fourier_v1_vs_v2_fidelity.png`) to
reproduce the pre-optimization version's physical results to 13-14
significant figures (sheet fB at 48x48 and 96x96), with **identical L-BFGS-B
iteration and evaluation counts** in every case -- the optimizer takes the
exact same path, just faster:

| Case | v1 runtime | v2 runtime | Speedup |
|---|---:|---:|---:|
| Landreman-Paul QA | 2.37s | 1.32s | 1.80x |
| Landreman-Paul QH | 8.04s | 1.87s | 4.30x |
| W7-X | 19.37s | 3.63s | 5.33x |

See `figures/fourier_v1_vs_v2_speedup.png` for the same numbers as a chart.
A `float32`-precision variant and swapping the SVD for `eigvalsh` on the
smaller Gram matrix were also tried; both were rejected (`eigvalsh` produces
NaN gradients from the near-degenerate tail of the singular-value spectrum,
and `float32` corrupts that same tail by ~9 orders of magnitude while only
matching the full gradient to ~1e-5 relative -- both against this codebase's
own documented SVD-gradient fragility near degenerate spectra, for less
benefit than the two changes above).

## Memory watchdog

`build_operators`' `(n_plasma_period, n_winding_full, 3)`-scale intermediate
arrays (used by the REGCOIL-validation steps, not the induction-matrix
optimizations above) make W7-X's resolution-64 resconv step the single
heaviest computation in this study, independent of surface method: repeated
attempts across two sessions (5 total) peaked between 5.3 and 7.5 GB RSS for
what should be "the same" computation, presumably reflecting BLAS/allocator
run-to-run variance. Four of five method/case combinations eventually
succeeded after freeing background memory and retrying; `"normal offset"`
consistently needed more (6.5-7.5 GB across 4 attempts) than the other two
methods and never completed within this machine's available headroom. This
is a pre-existing cost in shared, unmodified code
(`winding_surface_comparison_8_coils.py`'s own `build_operators`), not
something introduced by this addition.

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
python3 winding_surface_fourier_comparison.py            # ~1-3 min, watchdog-protected
python3 winding_surface_fourier_comparison_plots.py       # vs-other-methods figures
python3 winding_surface_fourier_comparison_v1_vs_v2_plots.py  # needs a v1 snapshot at
                                                                # output/winding_surface_fourier_comparison_v1_snapshot/
                                                                # (not included; see "Performance")
```
