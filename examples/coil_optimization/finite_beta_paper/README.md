# Finite-beta plasma-field paper: validation runs and handoff

Scripts that produced (and will finish producing) the numbers in the plasma-field
manuscript, from the example `optimize_coils_and_nearaxis_finite_beta.py`. This folder
records the state of the external reviewer's validation handoff when work paused on
2026-09-22.

## Pinned stack

`versions.json`: vmec_jax/VMEX `07a47279d`, SOLVAX `7b8ca553f` (0.25.0), booz_xform_jax
`cd250844` (0.4.0), virtual_casing_jax `ef39ce14`, ESSOS this branch, pyQSC_JAX PR #2
(`de9fe5b`). Python 3.11, JAX 0.9.2, SciPy 1.17.1, CPU float64. Put the pinned VMEX
checkout and SOLVAX on `PYTHONPATH` before running.

## Scripts

| File | Purpose |
|---|---|
| `run.py OUT 'NAME = value' '+stmt'` | Runs the example with overrides (never edits the docstring) and journal-width figures. |
| `segments.sh NAME ARGS` | Optimization in 40-evaluation segments of at most 10 min, each checkpointed. |
| `jobs.sh JOBFILE N` | VMEX-stage jobs, N at a time, capped at `CAP` seconds (default 600). |
| `adopt.py SRC RUN ARGS` | Adopts an optimization made earlier with the same objective, but only if the current code recomputes the stored cost exactly. |
| `make_tables.py` | Builds `results.json` and `tables.tex` from `runs/*/summary.json` only. |

Split routes: `NAME` holds the direct Biot-Savart solve and `NAME_mgrid` the MAKEGRID-file solve.
`vacuum_trace` holds the vacuum field-line tracing.

## Fixed in this round

- pyQSC_JAX `to_vmec` wrote `MPOL = mpol` while exporting `m = 0..mpol`, so VMEC dropped
  the top row. It now writes `MPOL = mpol + 1`, and a round-trip test goes through the VMEX
  parser. In the flagship case the lost m = 8 amplitude is 1.2e-7 m, so earlier results stand.
- `estimated_field_remainder` was zero at `I2 = 0` even with pressure. It now scales
  the retained field and is labelled an indicator, not a bound. The `I2 == 0` gradient
  branch jump is documented and pinned by a test, not smoothed.
- The example now saves the runtime free-boundary deck. It also runs a real MAKEGRID
  NetCDF write/read/solve. Write/read table difference 0, reloaded vs in-memory 0, and
  interpolation error 8.4e-4 T max on the boundary. The field is NaN off the table, and
  the grid margin is reported (0.117 m).
- New diagnostics: an interface check (tangential jump, pressure balance), transforms in
  a signed laboratory convention, a traced vacuum transform, and the r2/r3 enclosed flux.
- Checkpoints carry config and hash, so incompatible reuse is refused. The optimizer's
  status, nfev, cost, optimality and times are recorded. The history plot is labelled a
  running minimum.
- A driver bug: overriding `SUBTRACT_PLASMA_FIELD` replaced a docstring line. See the control below.

## Results so far (`results.json`)

Finite-pressure flagship (`qa`, adopted, cost reproduces to 1e-16, 1000 evaluations):

- Axis RMS mismatch: 1.2e-4 T field, 8.9e-4 T/m gradient and 3.1e-2 T/m^2 Hessian. The
  plasma parts are 2.6e-3 T, 9.5e-3 T/m and 13.9 T/m^2.
- Normal target mismatch: 0.62 % max and 0.059 % RMS (28.6 % initially).
- VMEX, direct and file-mgrid routes: 427 iterations and beta 6.75e-4.
- Lab-frame iota: -0.2099 from VMEX and -0.2125 near axis. The raw values have opposite
  signs because the poloidal angles turn opposite ways.
- Axis offset: 4.96 % of a (1.49 mm).
- LCFS shape: 2.2 % after translation and 4.1 % raw.
- Interface: tangential jump 0.96 % of |B| max and 0.17 % RMS; pressure balance 0.18 % max.

The earlier **total-field control does not reproduce** with the current objective
(recomputed cost 0.186 vs recorded 0.642). Its numbers (66 % axis offset, and the older
62 %) are withdrawn. A fresh control is being run.

No-Hessian ablation (same 4 coils, adopted, reproduces exactly): the direct VMEX solve
converged at NS = 65, iteration 7417 of 8000. Its force residual oscillates between
1e-10 and 4e-7 and only touches FTOL. The job was killed while writing output, so there
are no numbers yet.

## Incomplete when paused (resume in this order)

1. Finish the segmented optimizations. `control`, `fixed_sub` and `fixed_nosub` stand at
   320/400/400 of 1000 evaluations. The radius runs `a010`, `a015` and `a020` stand at 80.
2. Run the VMEX jobs for `nohess`, `vacuum` (a_b = 0.7 a), `axisym`, their `_mgrid`
   splits and `vacuum_trace`. Then run the VMEX stage for the six runs above.
3. Resolution ladder on the `qa` coils: NS to 129, modes 10, nzeta 64, a tighter FTOL,
   coil quadrature 480, and grids 65/129 plus a wider margin. See `ladder.jobs` in the run
   log of this handoff.
4. pyQSC_JAX WIP branches, not verified or reviewed:
   `wip/cartesian-volume-derivatives` (fixed-Cartesian gradient/Hessian extraction
   from the volume integral) and `wip/boozer-angle-axis-integral` (spectral vs
   trapezoidal Boozer angle and full-axis integral convergence; touches `plasma.py`
   and `spectral.py`). Run, check against their stated physics, then merge into PR #2.
5. Not started: a source current rebuilt from a VMEX equilibrium, source divergence and
   normal-current checks, and a uniform order-a^2 finite-current gradient (a derivation task).
6. Regenerate the tables with `make_tables.py`, and update the unified `main.tex` only from
   `results.json`: control, ablation, radius sequence, ladder, interface, signed iota, the
   flux check and the MPOL note. Then assemble the portable archive (wouts, runtime decks,
   mgrid files, checkpoints, logs) and the final report.

Machine load was about 80 on 14 cores from other jobs. Solves that take 1-2 min on an idle
machine then took more than 10 min, which is why VMEX jobs were given `CAP=1500`.
