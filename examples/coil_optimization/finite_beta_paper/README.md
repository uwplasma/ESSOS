# Finite-beta plasma-field paper: validation archive and report

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
