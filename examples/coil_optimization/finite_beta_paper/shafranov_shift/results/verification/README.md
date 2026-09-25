# Fixed-coil pressure response: verification campaign (September 2026)

This folder is the record for the pressure-response part of the Physical
Review Research draft *Plasma self-fields and pressure response in
stellarator coil design*. It follows `handoff/plan.md` of the
`PRR_Plasma_Field_and_Bootstrap_Package` (Phases 0, 1, 3 and the
diagnostic parts of Phases 2 and 5). The three figures in `figures/` are
drop-in candidates for the four principal numerical figures the plan asks
for. The fourth, the matched design and performance comparison, was not
executed (see "Not done").

## Context and question

The earlier record (`../README.md`, Table `tab:pressurescan` of the draft)
found fixed-coil VMEX pressure slopes of −4.95%, −6.01% and −7.70% of the
untuned first-order prediction at 12, 15 and 18 mm. Going from 65 to 129
radial surfaces changed them materially. The question was whether the
source (the near-axis pressure current), the closed-axis response operator,
the observable, or the finite-pressure equilibrium solve is responsible.

All calculations use the 18 mm QA candidate: the fitted coils
`reference/vacuum_fitted_coils.json` (SHA-256 `41d43bcc…`), flux radius
a = 17.8115 mm, p = α p0 (1 − s) with p0 = 190.3 Pa, zero current profile,
fixed signed PHIEDGE. The displacement screen gives α_max = 0.17497
(p(0) = 33.31 Pa, volume beta 4.20e-5, axis beta 8.38e-5). The untuned
theory gives δR(φ=0) = 2.9917 mm per unit α, i.e. 0.5235 mm at α_max.

## Outcome in one paragraph

The analytic chain is verified. The first-order source equals the
Biot–Savart field of its own current distribution to 6.8e-4 after
Richardson extrapolation. The closed-axis response operator equals direct
nonlinear coil-field tracing to 1.8e-8, and the near-axis Frenet operator
agrees with it to 0.22%, set by the 82 µm ideal-to-traced axis offset.
The VMEC-type free-boundary solves do **not** resolve the derivative, and
this has now been shown directly rather than inferred:

- VMEX warm and cold starts give opposite signs (−2.9% to −9.8% vs +1.8% to
  +2.7%).
- VMEC2000, an independent code, gives +4.2% to +11.5% at α_max, rising
  with NS without converging.
- VMEC2000's response has a non-analytic threshold in α, with iteration
  counts growing with α.
- At the force floor the VMEC2000 vacuum axis wanders by 200 µm as NITER
  varies, and the response ratio swings between 0.10 and 0.40.
- Every converged free-boundary vacuum axis sits 474–688 µm from the
  directly traced coil axis. That systematic vacuum error is as large as
  the entire predicted shift at α_max.

The plan's acceptance condition (axis uncertainty ≪ smallest pressure-step
signal) therefore fails by more than an order of magnitude. Neither the
earlier negative slopes nor any positive VMEC-type slope is a physical
response coefficient. No evidence contradicts the first-order theory, but
no full-equilibrium calculation here confirms it either.

## What was done, with the main numbers

### Phase 0: independent checks (reproduced)

All five package scripts pass under float64 and one BLAS thread
(`phase0/*.log`). Output differs from the shipped JSON only at round-off,
e.g. the length slope 0.014273914 vs 0.014273914. The Hessian identity
`dist(H, STF)^2 = (2/3)|Sym C|^2 + (4/5)|Skew C|^2` is now a pyQSC_JAX test
against its own projector (`tests/unit/test_hessian_incompatibility.py`):
the maximum relative error over 200 random divergence-free tensors is below
1e-12, the value is exact for a J_z = j1 x cylinder, and a vacuum Hessian
sits at distance zero.

### Phase 1: observable repair (ESSOS `22438b3`)

`pressure_axis_response` returned the *ideal* length slope even when given
the actual coil gradient, so the scan stored the ideal value as
`actual_gradient_length_slope_over_L`. It stored the ideal lab displacement
as `delta_R_actual_gradient` for the same reason. The function now returns
distinct `physical_*` observables and exposes `frenet_axis_response`. A
regression test perturbs the gradient: the physical length and RMS change
by more than 1%, the ideal observables stay bit-identical, and the length
agrees with a centred finite difference of the displaced complete curve to
1e-6. With the fix, the actual-gradient length slope at 18 mm is 0.00342345
against an ideal 0.00342334. The bug did not cause the discrepancy.

### Phase 3a: closed-axis operator (`axis_operator_check.json`)

A uniform vertical field εB0 (curl- and divergence-free) was added to the
coils.

| quantity | value |
|---|---|
| traced vacuum axis closure | 6.8e-16 m |
| ideal-to-traced axis offset (max) | 82.4 µm |
| monodromy transform | 0.212340 (direct Floquet −0.212340) |
| nonlinear / exact linear, ε = 1e-6, 2e-6, 4e-6 | 7.9e-5, 1.6e-4, 3.1e-4 (first order in ε) |
| Richardson (ε = 1e-6, 2e-6) vs exact linear | 1.8e-8 |
| near-axis Frenet operator vs traced, ideal gradient | 0.223% |
| same, actual coil gradient on the ideal curve | 0.222% |
| pressure forcing through the exact traced operator: δR(0) per α | 2.9979 mm (ideal 2.9917 mm; 0.21%) |

### Phase 3b: source (`source_field_check.json`)

Biot–Savart of the pyQSC_JAX positive-volume current on the first-order
surfaces against `plasma_field_on_axis`: relative differences 1.43, 0.450,
0.117 and 0.0293 at nφ = 101, 201, 401 and 801 (∝ nφ⁻²). The Richardson
value is 6.8e-4. On axis at φ = 0 the vertical plasma field is
9.264e-4 T per unit α.

Field of the equilibria's own current (Ampère's law on the WOUT covariant
field, differenced against the same-resolution vacuum), vertical on-axis
component per unit α at α_max:

| equilibrium | vertical B (1e-4 T per α) |
|---|---|
| near-axis closed form | +9.26 |
| VMEC2000 free, NS 33 | −0.10 |
| VMEC2000 free, NS 65 | −0.97 |
| VMEC2000 fixed boundary, NS 33 | −0.38 |
| VMEX free, NS 65 (warm) | −1.41 |

This extraction has a large differencing floor: the vacuum solution alone
produces about 0.03 T at the axis. It is corroborating evidence only. It
fails entirely when the axis moves by millimetres (the 8× pair is
excluded). Radial force balance holds on average (mean F_s ≤ 0.45 Pa
against p′(s) = −33.3 Pa). The angle-resolved m = 1 residual at s = 1/2 is
5.4–14.9 Pa, already present in the vacuum, against an expected
Pfirsch–Schlüter force of about 3.3 Pa at α_max.

### Phase 2 diagnostics: the equilibrium derivative (`pressure_runs.json`)

Response ratio δR(0)/δR_theory(0) at α_max (cold starts unless noted):

| solver | NS 33 | NS 65 | NS 129 | NS 257 |
|---|---|---|---|---|
| VMEC2000 free | 0.042 | 0.068 | 0.071 | 0.115 |
| VMEX free, warm continuation (recorded) | — | −0.029 | −0.005 | — |
| VMEX free, cold | — | 0.020 | 0.023 | — |

VMEC2000 controls at NS 65: MPOL 16 / NTOR 12 gives 0.064; a MAKEGRID with
four times finer (R, Z) spacing gives 0.068. FTOL below about 1e-13 is not
reachable: the residual stalls and grows.

VMEC2000 at NS 65 beyond the linearity screen:

| α/α_max | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| ratio | 0.068 | 0.078 | 0.188 | 0.330 | 0.491 | 0.598 | 0.735 | 0.890 | 0.987 | 1.327 |
| final-stage iterations | 1171 | 1312 | 1914 | 2562 | 3219 | 3602 | 4103 | 4738 | 5165 | 8485 |

The vacuum needs 1141 iterations. At NS 129 the ratios at 4, 8 and 12 are
0.364, 0.534 and 0.887. At 8× the MPOL 16 value is 0.905. The high-α curve
is therefore not resolution-converged either.

VMEX cold starts at 2, 4, 6, 8 and 12 × α_max: 0.0197, 0.0192, 0.0187,
0.0184 and 0.0178. These are linear, at 433–442 iterations, with the axis
pinned near its seed.

Iteration cap (VMEC2000, NS 17, FTOL 1e-30, all ier = 0):

| NITER | 1000 | 4000 | 16000 | 64000 |
|---|---|---|---|---|
| vacuum R_axis − traced (µm) | 449 | 252 | 380 | 263 |
| ratio at α_max | 0.104 | 0.401 | 0.124 | 0.343 |

Vacuum axis against the directly traced coil axis at φ = 0:

| run | offset (µm) |
|---|---|
| VMEC2000 free, NS 33, 65, 129, 257 | 536, 476, 479, 475 |
| VMEC2000 free, NS 65, MPOL 16 / fine MAKEGRID | 477 / 474 |
| VMEX free, NS 65 / 129 | 688 / 658 (520 warm ladder) |
| VMEC2000 fixed boundary, NS 33 | −33 |

Finite-amplitude trace (`finite_amplitude_trace.json`): coils plus the
frozen first-order current, closed axis traced directly. The shift is
0.533, 1.074, 2.104, 2.793 and 3.633 mm at 1, 2, 4, 6 and 8 × α_max,
against 0.523, 1.047, 2.094, 3.141 and 4.188 mm linear. The vacuum
field-line map is linear to 2% up to 4 α_max and sub-linear beyond.

### Interpretation

The axis displacement is a very soft mode of the free-boundary energy. A
descent solver started from the near-axis seed axis stops at FTOL long
before that mode relaxes. The final axis is then set by the iteration
count and the seed, not by force balance. That fits every observation: the
sign flips between warm and cold starts, VMEX's α-independent 433
iterations and fixed 2% fraction, VMEC2000's iteration count and response
growing together, the wander under an iteration cap, and the
NS-independent 475 µm vacuum-axis bias. A local force floor that
FSQR/FSQZ do not reveal also sits above the PS force. None of this is
specific to the near-axis source. At a = 18 mm and β ~ 1e-4, a pressure
derivative from these solvers requires a coupled tangent at an accepted
state, or a solver whose vacuum axis matches direct tracing to ≪ 50 µm.

## Figures (`figures/`, PDF for the paper, PNG previews)

- `source_and_operator_verification`: (a) near-axis current Biot–Savart vs
  closed form against nφ, with its Richardson value; (b) closed-axis
  operator, nonlinear tracing vs exact linear, and the Frenet near-axis
  operator error.
- `fixed_coil_pressure_response`: (a) axis shift against α/α_max with the
  untuned theory, the frozen-current trace, VMEC2000 NS 65/129 and VMEX
  cold; (b) response ratio at α_max against NS and restart policy;
  (c) vacuum-axis offset from the traced coil axis against NS, with the
  NITER wander and the fixed-boundary value.
- `equilibrium_current_diagnostic`: (a) vertical on-axis plasma field per
  unit α, near-axis vs the four equilibria's own currents; (b) vacuum m = 1
  radial-force residual against the expected PS force.

Suggested captions and a replacement for Sec. `sec:scanstatus` are in
`pressure_scan_status_revised.tex`.

## Reproduction

Dependency revisions: ESSOS `plasma-coil` + local commits (see
`git log`), pyQSC_JAX `6c7c7ea` + `6707209` (test only), VMEX `926892ab`
(v0.11.2), VMEC2000 from STELLOPT `ee175502` (binary SHA-256 `8ba11884…`).
Python 3.13.7, JAX 0.11.0, NumPy 2.5.1 and SciPy 1.18.0 on an Apple M4.
The office Xeon W-2295 runs (VMEX cold starts) used Python 3.12.13 with the
same JAX/NumPy/SciPy.

```sh
export PYTHONPATH=/path/ESSOS:/path/pyQSC_JAX/src:/path/VMEX JAX_ENABLE_X64=1
D=examples/coil_optimization/finite_beta_paper/shafranov_shift
python $D/axis_operator_check.py --output runs/axis_operator_check.json
python $D/scan.py --reference $D/reference/vacuum_fitted_reference.json \
  --output runs/qa18 --radius 0.018 --segments 480 --mpol 10 --ntor 10 \
  --run-vmex --pressure-scan [--cold-start] [--alpha-max-multiple 12 \
  --pressure-fractions 0.1666667 0.3333333 0.5 0.6666667 1]
python $D/vmec2000_check.py --run runs/qa18 --output runs/v2k --xvmec /path/xvmec2000
python $D/source_field_check.py --output runs/source_field_check.json \
  --pair NAME VACUUM_WOUT PRESSURE_WOUT ALPHA
python $D/finite_amplitude_trace.py --output runs/finite_amp.json
python $D/collect_verification.py --vmec2000 runs/v2k_* --vmex ... \
  --theory runs/qa18/manifest.json --traced-axis .../axis_operator_check.npz \
  --output $D/results/verification/pressure_runs.json
python $D/make_verification_figures.py
```

The VMEC2000 NS, FTOL, NITER, MPOL, MAKEGRID and pressure-multiple
variants were produced by editing only the named keys of the
`vmec2000_check.py` decks. Every deck's hash is in `pressure_runs.json`.
Native run directories (≈660 MB locally, 85 MB on office, 47 local WOUT files) are outside Git; each accepted case records
its WOUT SHA-256. The VMEX re-run of the recorded 18 mm scan reproduces
`results/runs/qa18` bit for bit: 642 s wall and 5.2 GB peak for vacuum
plus five points on the M4. VMEC2000 takes 230 s (NS 65) to 508 s (NS 257)
per point.

## Checks run

Passed: ESSOS `tests/test_shafranov_shift.py` (15, locally and on office);
pyQSC_JAX `tests/unit/test_hessian_incompatibility.py` (2); ESSOS CI flake8
gate (`E9,F63,F7,F82`, 0); ruff on the pyQSC_JAX test file. Not run: the full
ESSOS, pyQSC_JAX and VMEX suites.

## Not done, and why

- **Coupled free-boundary tangent** `F_U U_α = −F_α` (Phase 2): not
  attempted. It needs a qualified VMEX coupled-root contract, and it would
  still have to be evaluated at an accepted state whose vacuum axis is
  correct. Given the 475–690 µm vacuum-axis bias, this comes first.
- **Independent pressure-current reconstruction on the traced vacuum
  surfaces** (Phase 3, magnetic differential equation on the real coil
  field): not done. The near-axis current was checked only against its own
  closed form.
- **Matched QA design comparison and end-to-end timing** (Phase 4, fourth
  figure): not done. The plan conditions it on a qualified pressure chain,
  which this campaign shows is not yet available from VMEC-type solvers.
- **12 and 15 mm radii**: not re-run. The new diagnostics use 18 mm only.
- **Fractional bootstrap workstream**: only the supplied independent
  script was reproduced (Phase 0). There is no JAX port or theory
  extension.
