# Fixed-coil pressure response

This folder contains the pressure-response operator, the fixed-coil vacuum gate
and VMEX continuation driver, its focused tests, compact run summaries, and
diagnostic figures for ESSOS PR #70. The full user-provided implementation
handoff is preserved verbatim at
[`reference/ESSOS_PR70_Shafranov_Handoff.md`](reference/ESSOS_PR70_Shafranov_Handoff.md).

## How the supplied documents were used

The ZIP's handoff document was treated as the implementation plan. The
`Agent_Validation_Prompt.md` inside that ZIP is historical review context; it is
not a separate instruction from the requester. The plan has been revised in
response to the measured-flux mismatch and pressure-resolution sensitivity
recorded below. No manuscript claim is made from the exploratory VMEX pressure
figures.

## Plan and implementation

1. Reproduce the archived QA vacuum reference with a direct Biot-Savart
   Poincare return map, independent nearby-orbit winding, field and gradient
   comparisons, traced seed-surface certification, and enclosed-flux
   integration.
2. When the archived coils fail field fidelity, fit a fixed-coefficient
   current-free near-axis target and keep the fit labelled as a candidate.
3. For each radius, trace the actual vacuum surface from the near-axis
   outboard point. Use its measured signed flux as VMEX `PHIEDGE` and its
   effective flux radius in `p0_star = -p2_star*a_flux**2`. Gate pressure runs
   on the free-boundary vacuum residual, active exterior solve, direct Floquet
   transform, field and gradient agreement, surface certificate, and interface
   metrics.
4. Freeze the coil curves and current, signed edge flux at each radius, and
   pressure profile. Continue upward from vacuum with `AC=0` and `CURTOR=0`.
   Run five positive pressure levels at 12, 15, and 18 mm.
5. Compare same-resolution VMEX axes in fixed laboratory planes with the
   untuned first-order response. Fit each observable to
   `alpha*slope + alpha**2*curvature` with a fixed zero intercept. Check a
   second radial resolution at the largest 18 mm pressure before treating the
   signal as numerically resolved.

The first version used the nominal near-axis radius to set `PHIEDGE`. Direct
Poincare integration found that the actual enclosed flux was lower, so the
plan was revised to use the measured signed flux and effective radius. The
corrected-flux vacuum gates now pass at all three radii. The pressure
equilibria converge, but their shift does not agree with the analytic
prediction and is sensitive to VMEX radial resolution. The plan is therefore
revised again: retain the actual-result figures as diagnostics, do not tune the
analytic amplitude to the data, and do not promote these numbers into the
manuscript until pressure-step and equilibrium errors are separated.

## Equation-to-code map

[`../../shafranov_shift.py`](../../shafranov_shift.py) implements the native
geometric-grid periodic scalar solve, a separate FFT route with a spectral
Boozer-angle map, the Frenet-frame physical block solve using the actual or
target field gradient, signed orientation handling, first variation of axis
length, and conversion from the Frenet normal gauge to fixed cylindrical
planes.

| Manuscript label | Implementation |
|---|---|
| `eq:shiftforcing` | `pressure_axis_response`: pressure source and scalar `Cp` forcing |
| `eq:shiftformula` | Complex periodic solve and independent Fourier route |
| `eq:shiftsymmetry` | QA/QH orientation-sign tests and laboratory `delta_R`, `delta_Z` |
| `eq:shiftlength` | Independent displacement integral and positive analytic length identity |
| `eq:shiftlab` | Tangent-removal gauge and cylindrical-plane displacement |

The geometric toroidal angle is `phi`; `varphi` is the Boozer angle. The VMEX
axis transform is converted to the laboratory sign with the measured surface
orientation. A zero current profile is checked over the full radial interval,
not inferred from `I2=0` or zero edge current.

## Reproduction

The validated local dependency revisions are VMEX
`926892ab7131a6bc0c5b61218d1f75e7b77bc401` and pyQSC_JAX
`6c7c7ea8d14932d3400c5daa61432a6b93a426a9`. The runs used Python 3.13.7,
JAX 0.11.0, NumPy 2.5.1, and SciPy 1.18.0. Point `PYTHONPATH` at those checkouts
when reproducing:

```sh
PYTHONPATH=/path/to/pyqsc_jax/src:/path/to/vmex \
  pytest -q tests/test_shafranov_shift.py

PYTHONPATH=/path/to/pyqsc_jax/src:/path/to/vmex \
  python examples/coil_optimization/finite_beta_paper/shafranov_shift/scan.py \
  --reference examples/coil_optimization/finite_beta_paper/shafranov_shift/reference/vacuum_fitted_reference.json \
  --output /path/to/run/qa18 --radius 0.018 --segments 480 \
  --mpol 10 --ntor 10 --run-vmex --pressure-scan

python examples/coil_optimization/finite_beta_paper/shafranov_shift/analyze_results.py
```

The tracked `results/runs/` folder contains the compact manifests, input
boundary, pressure points, and theory arrays needed to rebuild the tables and
figures. Each pressure-point record carries runtime-input and WOUT SHA-256
hashes. Full WOUT files and solver logs remain in the local ignored run
directories; no large binary simulation archive is included in the source
branch.

## Vacuum and pressure findings

The archived QA coil set is not a verified reference. At 18 mm it has 0.260%
axis-field RMS error, a 0.423-radius target-axis offset, 0.436% peak seed
surface `|B.n|/|B|`, and about 21% disagreement between direct winding and the
near-axis transform. A 600-evaluation fixed-target fit reduces the weighted
field-fit cost from 0.8534 to `3.75e-6`, but reaches its evaluation limit
without optimizer convergence. The fit remains a candidate.

At 480 coil segments, the candidate agrees with the target-axis field to
`2.43e-5 B0` RMS and the target gradient to `6.68e-4 B0/R0` RMS at 18 mm. The
independent coil-field return map has determinant 1.00000008 and Floquet
transform `-0.212340`; finite-orbit winding at `0.25 a` differs by 0.65%.
Direct vacuum surfaces at 12, 15, and 18 mm remain bounded for 128 field
periods. Twelve-harmonic Poincare fits have 0.044%, 0.081%, and 0.099% RMS
residuals. Their enclosed fluxes are 1.55%, 1.75%, and 2.08% below the nominal
`pi B0 a**2`, giving effective flux radii 11.9066, 14.8678, and 17.8115 mm.

The 18 mm direct-field preflight was repeated at 240, 480, and 960 coil
segments with the same coil hash. The 240-to-960 relative ranges are
`1.4e-12` for target-field RMS, `7.3e-14` for target-gradient RMS, `5.1e-11`
for the return-map Floquet transform, and `1.6e-11` for effective flux radius.
Coil quadrature is therefore negligible against the observed VMEX
pressure-response sensitivity.

With corrected flux, the `m=n=10`, `NS=17/33/65`, `NZETA=64` VMEX vacua pass the
2% transform gate. Their laboratory transforms are `-0.212954`, `-0.212425`,
and `-0.211923`, compared with direct Floquet `-0.212340`; the largest gap is
0.289%. All 15 positive-pressure equilibria converge with active exterior
fields. The complete input current profile is zero, `CURTOR=0`, edge-flux
round-trip errors are below `1.8e-15`, and maximum tangential interface jumps
are 0.139%, 0.176%, and 0.237% of `B` at 12, 15, and 18 mm. The largest
axis-beta values are `5.60e-5`, `6.99e-5`, and `8.38e-5`.

Those solver checks do not establish theory agreement. Fixed-zero quadratic
fits to the five pressure points give symmetry-plane `delta_R(0)` slopes that
are **inward**, while the first-order theory predicts outward shifts. Their
ratios to theory are `-4.95%`, `-6.01%`, and `-7.70%` at 12, 15, and 18 mm.
Fitted axis-length slopes are also negative, at `-0.94%`, `-1.21%`, and
`-1.64%` of the positive first-order prediction. The finite-pressure
symmetry-plane `delta_Z(0)` remains zero by stellarator symmetry. These are
diagnostic fit values without an uncertainty estimate.

At 18 mm, raising the radial grid from 65 to 129 surfaces changes the measured
response substantially: the fitted-pressure-point `delta_R` profile RMS per
unit `alpha` drops from `65.9` to `31.9` micrometres, and the axis-length
response changes sign. The 129-surface vacuum still passes the transform gate,
but its transform gap is 0.61%. A 257-surface selected-point attempt did not
reach the first vacuum-iteration output after 15 minutes in the VMEX/JAX
steady-vacuum execution, so it was stopped and no result was retained in the
tracked bundle. Until a finer radial check and additional pressure-step tests
are complete, radial and pressure-step errors remain unresolved. No physical
amplitude has been fitted to hide the discrepancy.

## Current limits and revised next steps

- The pressure response is not numerically converged in radial resolution;
  its signal is much smaller than the first-order prediction and changes
  materially with `NS`. The attempted 257-surface check exceeded the available
  runtime without producing a vacuum result.
- The current bundle has no verified second QH/QA vacuum reference. The
  archived QH coil fit is incomplete, so the independent-geometry objective
  remains blocked by available reference data.
- No MAKEGRID-versus-direct-field comparison or independent VMEC2000/DESC
  pressure solve has been run. Direct Biot-Savart is the primary external
  field.
- The circular fixed-boundary tokamak coefficients are covered by a separate
  algebraic reference identity test; they are not passed through the
  current-free stellarator response operator.
- The figures in `results/figures/` are actual-simulation diagnostics. They
  have no numerical error bars because the response is not yet resolved; do
  not cite them as validated manuscript results.

The next useful work is to finish the selected-point radial ladder, add smaller
positive pressure steps and an independent cold-start check, then investigate
the VMEX/near-axis discrepancy before extending the result family. The
archived manuscript values use older pins and remain untouched.
