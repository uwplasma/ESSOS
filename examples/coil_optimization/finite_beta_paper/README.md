# Finite-beta plasma-field paper: validation archive and report

## Round 3 handoff (paused 2026-09-23)

Requested in this round:
1. investigate the vacuum iota gap;
2. add a QH (I2 = 0, finite beta) and a stellarator-tokamak hybrid (finite I2) coil-only example;
3. add a single-stage finite-beta (3 %) optimization;
4. scan the boundary radius (PHIEDGE) for surfaces beyond the design radius;
5. replace the matplotlib 3D figures with pyvista;
6. make all field and error plots relative to B0.

**Done**
- **Vacuum iota gap (resolved).** Scripts and data are in `investigation/`.
  - VMEX surfaces are coil-field surfaces: |B.n|/|B| < 0.5 % at 8 modes.
  - The coil field on those surfaces, <sqrt(g) B^u>/<sqrt(g) B^v>, and lines traced from them give -0.211 to -0.197. That is consistent with the near-axis -0.2125.
  - VMEX gives -0.188 to -0.177 on the same surfaces.
  - VMEC2000 on the identical runtime deck and MAKEGRID file reproduces VMEX: -0.196 and -0.188 on axis and at s = 1/2, against VMEX's -0.197 and -0.188. The gap is common to VMEC-type solvers.
  - It is not converged in Fourier resolution. At 12 modes VMEX meets FTOL, but its surfaces degrade (0.9 % B.n) and the axis iota is -0.089. At a_b = a it gives -0.165 with an 8.9 % shape error.
  - Conclusion: the VMEC vacuum iota is unreliable in this zero-pressure, zero-current, aspect-50 case. The coils and the near-axis design agree. The finite-beta iota agrees to 1 % and converges.
  - The mechanism is not identified. The manuscript text is updated.
- **Vacuum last closed surface** (Giuliani-style trace, `vacuum_surface_scan.py`). Lines launched on the outboard midplane stay confined up to a flux radius of 0.91 a; lines just outside escape within 4 turns. Across this range iota goes from -0.204 to -0.124. There are no vacuum surfaces beyond the design radius for these coils.
- **Flagship boundary scan** (`scan_qa.jobs`; p2 fixed, so the central pressure grows as a_b^2):

  | a_b | Result |
  |---|---|
  | 1.0 a | converged (reference) |
  | 1.25 a | converged: iota -0.2139 vs -0.2125, axis 0.2 mm, shape 3.7 %, beta 1.06e-3 |
  | 1.5 a | converged, but a different equilibrium: axis 10.6 mm (24 %), iota -0.185 |
  | 1.75 a, 2.0 a | FTOL not reached |
  | 2.5 a | Jacobian sign failure |
  | 3.0 a | FTOL not reached |

  Not yet in the manuscript.
- **Plots.** pyvista coil renderer (`render_coils_and_surface`, log |B.n|/|B| with viridis, never white), and every field and error plot is relative to B0. Figures are in `figures/qa`, `figures/vacuum` (from `replot.py`) and `figures/paper`.
- **New scripts:**
  - `optimize_coils_nearaxis_qh_finite_beta.py` and `optimize_coils_nearaxis_hybrid_finite_current.py`, which fit coils only to a fixed equilibrium (`helpers.fit_coils`, axis-centred initial coils `helpers.axis_centered_curves`, resumable segments);
  - `optimize_single_stage_nearaxis_finite_beta.py`.

**Incomplete (resume here)**
- **QH.** a = 0.035 m, p2 = -1.76e6 (central pressure 2160 Pa, beta 0.27 %), 4 coils/half-period, order 10, 120 segments. The checkpoint is at 450 of 2000 evaluations; resume with `segments2.sh`.
  - The previous run (a = 0.06, 60 segments, `figures/qh/*_a0.06_60seg.png`) reached axis mismatch 3.1e-4 B0, 10x below the plasma field.
  - Its boundary B.n/|B| scaled as r^3: 0.06 / 0.51 / 1.8 / 4.9 % at a/4 to a. That is truncation of the quadratic target, hence the smaller a.
  - Its curvature was 406 1/m at 240 points (a kink unseen at 60 points), hence 120 segments.
- **Hybrid.** Axis rc = [1, 0.045], nfp = 3, etabar 0.9, I2 = 0.4 (iota 0.42 -> 0.71), p2 = -6e5, a = 0.04 m, 4 coils/half-period, order 10, 120 segments. The checkpoint is at 600 of 2000 evaluations.
  - The previous 60-segment run: axis mismatch 7.4e-4 B0 (plasma field 4.0e-3), B.n max 1.16 %, curvature 21 1/m at 240 points.
  - The target is a vacuum field to 1e-5.
- **Next for QH and hybrid.** VMEX at a_b = a, plus boundary scans. Add the jobs like `scan_qa.jobs` with `EXAMPLE=<script>`. For the coil-only scripts, `run.py` takes `EXAMPLE` and inserts '+stmt' lines after `OUTPUT_DIR.mkdir`.
- **Single stage.** Seed: vacuum QA (Landreman-Sengupta 5.1); a = 0.1 m; staged: near-axis 400, coils 300, single-stage 500, coil limits 400 evaluations.
  - The optimization finished (`investigation/single_stage_summary.json`, `figures/single_stage`). The script is copied from the stopped worktree branch `work/single-stage-finite-beta`, untested there after the stop.
  - VMEX free boundary at NS = 17 converged: beta 2.89 % (target 3 %), iota 0.556, aspect 9.6.
  - The NS = 33 continuation (restart from the NS = 17 wout) had not converged. Resume the VMEX stage and the QS/coil report.
- **Manuscript.** The vacuum section is rewritten, and it is compiled locally. Still to add: the scans, QH, hybrid and single stage, and relative-unit tables (`make_tables.py` still reads SI keys).


These scripts and machine-readable results back every number in the plasma-field
manuscript (from the example `optimize_coils_and_nearaxis_finite_beta.py`). The report
below answers the external reviewer's validation handoff. It covers defects fixed,
independent validation that passed, what is incomplete, and physical discrepancies
that remain.

## Pinned stack

`versions.json`: vmec_jax/VMEX `2feba0d1c` (all free-boundary runs), SOLVAX `7b8ca553f`
(0.25.0), booz_xform_jax `cd250844` (0.4.0), virtual_casing_jax `c0bf626b` (0.0.8),
pyQSC_JAX PR #2 `6c7c7ea`, this ESSOS branch. Python 3.11.14, JAX 0.9.2, SciPy 1.17.1,
CPU, float64. Before running, put the pinned VMEX checkout and SOLVAX on `PYTHONPATH`.

## Files

| File | Purpose |
|---|---|
| `run.py OUT 'NAME = value' '+stmt'` | Runs the example with overrides (never edits the docstring) and journal-width figures. |
| `segments.sh NAME ARGS` | Optimization in 60-evaluation segments, each checkpointed, so no process exceeds 10 min. |
| `jobs.sh JOBFILE N` | Free-boundary jobs, N at a time, capped at `CAP` seconds. |
| `*.jobs` | The exact job lists that were run. |
| `adopt.py SRC RUN ARGS` | Adopts an earlier optimization of the same problem only if the current code recomputes its cost exactly. |
| `make_tables.py` | `results.json` and `tables.tex` from `runs/*/summary.json` only. |
| `make_figures.py` | Contour figures from the archived axes and wout files only. |
| `volume_table.py`, `cartesian_table.py` | Volume-integral and fixed-Cartesian derivative tables (`*_table.json`). |

The full archive holds 44 run folders with checkpoints, runtime decks, MAKEGRID files,
wouts, summaries and logs. It is `plasma_field_paper_archive_2026-09-23.tar.gz`
(201 MB, sha256 `13cf73e061a3328e...`). It is not in git and is to be deposited with
the paper.

## Code defects fixed

1. pyQSC_JAX `to_vmec` wrote `MPOL = mpol` while exporting `m <= mpol`, so VMEC dropped the
   top row. It now writes `mpol + 1`, with a round trip through the VMEX parser. The lost
   m = 8 amplitude in the flagship was 1.2e-7 m, so earlier results stand.
2. `estimated_field_remainder` was zero at `I2 = 0` with pressure. It now scales the
   retained field and is labelled an indicator. The `I2 == 0` gradient branch jump is
   documented and pinned by a test, not smoothed.
3. The finite-current global axis integral was limited to second order, for two reasons:
   - the trapezoidal Boozer angle inside the subtracted model;
   - the |s| kink of the bounded integrand at the coincident node.

   It now uses a spectral angle and removes the kink error. Against an independent
   Gauss-Legendre reference, the error at 41 points per period falls from 5e-4 to
   1.7e-6 m^-1, and the order rises from 1.7 to 3.9. The effect is on I2 != 0 only, so none of
   the optimizations (all I2 = 0, or a circle) change.
4. The benchmark now does the following:
   - saves the runtime free-boundary deck;
   - solves from a MAKEGRID NetCDF file written, re-read and loaded through the deck's
     EXTCUR (the round trip is exact);
   - makes the field NaN off the table and reports the grid margin;
   - reports interface diagnostics and signed laboratory transforms;
   - stores the full problem definition and hash in checkpoints, refusing mismatched reuse;
   - records optimizer status, nfev, cost, optimality and times;
   - labels the history as a running minimum.
5. Driver bug: an override of `SUBTRACT_PLASMA_FIELD` hit a line in the module docstring.
   That bug, not the old control run, made the control briefly look irreproducible. The
   old control reproduces exactly.

## Independent validation that passed

- **Adopted optimizations.** qa, no-Hessian, total-field control, vacuum and
  axisymmetric recompute to their recorded cost (relative difference at most 1e-16).
- **Resolution ladder on the flagship.** The refined controls are radial surfaces
  33/65/129, modes 8/10, toroidal points 32/64, FTOL 1e-9 to 3e-11, coil points 240/480,
  and field tables 65²×32, 97²×64 and 129²×64, plus a doubled extent. Across these:
  - translated edge shape 2.16-2.32 % of a;
  - axis offset 4.2-5.3 % of a;
  - |iota| 0.2055-0.2132, tending towards the near-axis 0.2125.
- **Direct coils vs field file.** The two routes agree to at most 6.5e-4 of the flux radius.
- **Signed transform.** A traced vacuum field line gives -0.2076, the same sign as VMEX
  and near-axis in the lab convention.
- **Enclosed toroidal flux (vacuum).** Second-order surfaces miss it by -2.3e-2 at
  r = a, scaling as r^2. With the r3 correction the miss is 1.9e-4.
- **Fixed-Cartesian volume derivatives.** This is an independent test with no predicted
  Hessian in the extraction and an inverse-map residual below 1e-15:
  - all 9 + 18 entries are compared, over 7 cases (4 orientation signs, pressure-only,
    I2 sign, circle);
  - the tensors converge as a^2 (ratio 3.6-4.1 per halving);
  - refining the quadrature or swapping the coincidence scheme changes them by less
    than 1 % of the error.
- **Volume field table.** Regenerated: 1.2e-3 / 4.0e-5 / 5.3e-5 at a = 0.01 m.
- **Hessian ablation at free boundary.** Same 4 coils, only the Hessian residual removed:
  - shape discrepancy 2.2 % -> 4.8 %;
  - axis offset 5.0 % -> 6.1 %;
  - tangential jump 0.96 % -> 4.9 %.

## Incomplete

- **Fixed-axis subtracted design.** Its free-boundary solve stagnates at 1e-8 to 1e-6
  (NS 33 and 65).
- **Fixed-axis total-field design.** It fails with a Jacobian sign change at NS = 17. Its
  axis field mismatch is 1.7e-2 T, against 3.1e-4 T for the subtracted design. So the
  fixed-target comparison quantifies the target mismatch, not an equilibrium.
- **a = 0.015 m.** No converged free-boundary solve. The radius sequence therefore uses
  NS = 33 and FTOL 1e-9, with 0.01 / 0.02 / 0.03 m and the flagship recomputed at those
  settings. 0.01 and 0.02 m do not converge at NS = 65.
- **Total-field control.** At FTOL 1e-10 its residual oscillates and it converges only
  at 1e-9 (axis 55 %, shape 6.8 %). A segmented re-optimization of the same problem, at
  the same cost, converges at 1e-10 (59 %, 8.5 %). An earlier solver revision gave 66 %.
  The 62 % in an earlier description was a second-order design.
- **Not done:**
  - a volume source rebuilt from a converged VMEX equilibrium;
  - source-divergence and boundary-normal current checks to retained order;
  - a uniform order-a^2 finite-current gradient (a derivation task).

## Physical and numerical discrepancies that remain

- **Axis displacement.** The flagship offset (5 % of a, 1.5 mm) is larger than the
  translated shape discrepancy (2.2 %). With FTOL between 1e-9 and 1e-10 it carries about
  0.7 % of a of numerical uncertainty. It grows in metres with a but is not proportional
  to a.
- **Vacuum transform.** The traced vacuum transform (-0.2076 at r = 7.5 mm) and the VMEX
  vacuum transform disagree by 5-10 %: -0.195 at that flux label at FTOL 1e-7, and
  -0.188 on axis at FTOL 1e-10. Tolerance and radial refinement do not close the gap.
  Finite-pressure transforms agree to about 1 %.
- **Plasma-vacuum interface.** The jump is not zero. It is 0.96 % for the flagship and
  0.69 % at NS = 129, partly radial extrapolation. For the no-Hessian case it is 4.9 %.
- **Ordering parameter.** eps2 = max a|x2|/|x1| is 0.33 for the optimized flagship and
  0.63 for its initial equilibrium at a = 0.03 m. These are not small numbers.
