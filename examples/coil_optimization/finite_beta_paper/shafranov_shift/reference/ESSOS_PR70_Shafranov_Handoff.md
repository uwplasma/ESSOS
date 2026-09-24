# ESSOS PR #70: fixed-coil pressure response of the magnetic axis

## Objective and working rules

Implement and independently verify the new manuscript section, **Pressure-induced displacement of the magnetic axis**, using a vacuum reference and free-boundary VMEX equilibria with increasing pressure. Deliver reproducible, manuscript-ready figures and machine-readable uncertainty estimates. The primary result is the **difference between finite-pressure and vacuum axes in the same coils**, not the difference between a near-axis target and one reconstructed equilibrium.

Work on the existing ESSOS PR #70 branch `plasma-coil`. Preserve its active work, checkpoints, and unrelated modifications. Do not merge the PR, force-push, or rewrite existing commits. Read the local status and the current handoff before editing. If the local branch has advanced, record the actual revision and inspect the relevant diff; the pins below are the review baseline, not permission to overwrite newer work.

Use the terminal's authenticated **rogeriojorge** account for all new commits and pushes. Verify `gh auth status` and `gh api user --jq .login`, and inspect `git config --show-origin --get-regexp 'user\.(name|email)'`. Configure the repository-local author as `rogeriojorge` with an email already verified for that account; do not invent an address. The reviewed VMEX commit uses `rogerio.jorge@wisc.edu`, but verify the local account association before adopting it. Set `user.useConfigOnly=true`. Check the actual author and committer of each new commit. Do not add Claude, Codex, bot, or AI co-author trailers. Do not alter older commits' authorship. Push only the intended PR branch and report the resulting SHA.

Keep the implementation small. Prefer one pressure-scan/analysis driver, a compact reusable response function beside the existing near-axis helpers, and a focused test module. Reuse existing surface, field, checkpoint, and wout utilities. Do not refactor the entire PR or add a new equilibrium framework.

## Inspected baseline and important blocker

Reviewed on 2026-09-24:

| Repository | Revision | Relevant inspected source |
|---|---|---|
| ESSOS PR #70 | `7713e07a77f08c15e70a34526dc9edf5b00da44d` | `examples/coil_optimization/nearaxis_finite_beta_helpers.py`; `examples/coil_optimization/finite_beta_paper/README.md` |
| VMEX main, release 0.11.2 | `926892ab7131a6bc0c5b61218d1f75e7b77bc401` | `vmex/core/multigrid.py`; `vmex/core/input.py`; `docs/howto/profiles.md` |
| pyQSC_JAX PR #2 | `6c7c7ea8d14932d3400c5daa61432a6b93a426a9` | `src/pyqsc_jax/models.py`; `src/pyqsc_jax/near_axis.py`; public calls used by ESSOS |

The ESSOS validation README reports two unresolved vacuum issues. First, for the archived coils, traced vacuum surfaces survive only to about **0.91 of the design flux radius**; the finite-pressure design boundary is therefore not automatically an admissible vacuum reference. Second, in the high-aspect-ratio current-free vacuum case, VMEX/VMEC transform estimates disagree with direct coil-field winding even when FTOL is reached, and increasing the Fourier resolution did not give simple improvement. These are repository reports, not runs reproduced during preparation of this handoff.

**Gate 0 is a verified vacuum reference, not a pressure run.** Do not conceal this blocker by forcing the old design boundary, using a target axis as the vacuum solution, or accepting a vacuum wout solely on FTOL. Either select a smaller independently verified vacuum flux surface in the archived coils, or use the existing vacuum optimization path to obtain a better current-free reference. Keep the selected coils fixed throughout each resulting pressure/radius experiment. A second, well-behaved QA or QH reference is more valuable than forcing one problematic archive to agree.

The archived numerical results in the supplied manuscript use older pins. Its empirical tables have intentionally not been updated from newer README claims. Keep the new pressure-response results in a separate run family and provenance record until the full numerical manuscript is reconciled with its archive.

## Physical experiment

The reference is a vacuum stellarator with constant on-axis field to the retained order, first-order quasisymmetric geometry, nonzero curvature and nonresonant transform. Let `alpha` multiply the pressure, with

\[
p_\alpha(s_v)=\alpha p_{0*}(1-s_v),\quad
p_{0*}=-p_{2*}a^2>0,\quad
|\Phi_{\rm edge}|=\pi B_0a^2,\quad I_{\rm tor}(s_v)=0.
\]

Here `s_v` is normalized toroidal flux, not arclength; `p2_star` is in Pa/m², pressure in Pa, and flux in Wb. Keep the signed exported PHIEDGE convention. Reference `B0` and `a` are vacuum normalization quantities. The entire enclosed-current profile is zero; `I2=0` alone is not the experiment. Use no bootstrap closure. Preserve zero edge pressure, no changing conducting-wall current, and the no-physical-sheet-current source assumption.

Freeze coil Fourier coefficients, physical currents, field periods, symmetry, the pressure-profile shape, and enclosed toroidal flux at a given radius. Do not re-optimize coils as alpha changes. Do not prescribe iota: it is an output. Do not add the analytical plasma field to the external field supplied to VMEX; VMEX calculates the plasma response. Use one current scaling only for tabulated fields.

Choose the pressure range from the predicted displacement, not a desired large beta: initially require `max(alpha * |xi|) / a <= 0.05`, and include several smaller levels. A reasonable initial sequence is `alpha_max * [0, 1/16, 1/8, 1/4, 1/2, 1]`, followed by additional smaller positive steps if numerical accuracy permits. Negative-pressure equilibria are not needed. Continue from vacuum upward and repeat selected points downward or cold-started to test branch selection.

At fixed coils and `p2_star`, use at least three radii inside the verified nested-surface domain. Shrinking `a` changes both `PHIEDGE` and `p0_star=-p2_star*a²`, but keeps the formal local pressure coefficient fixed. A radius scan at fixed central pressure is a different ordering and must not be presented as this convergence test. Radius changes may also change the selected finite vacuum surface; verify it independently each time.

## Equations to implement

Use the manuscript's angle convention: **phi is geometrical cylindrical angle; varphi is Boozer angle**. In pyQSC_JAX `solution.phi` is the geometrical sampling grid. Do not FFT first-order arrays on that grid and call the result a Boozer spectrum.

Take

\[
\nu=\iota_N=\iota_0-N,\quad \ell=L/(2\pi),\quad
x=\bar\eta/\kappa,\quad \chi=s_Gs_\psi,\quad
D=(1+x^2)^2+\sigma^2,\quad c_{p*}=\mu_0p_{2*}/B_0.
\]

The pressure derivative of the plasma field at the vacuum axis is

\[
\mathbf b_1=c_{p*}a^2\left[s_G\mathbf t+
 {2\ell\bar\eta x\over\nu D}
 \{s_G\sigma\mathbf n-s_\psi(1+x^2)\mathbf b\}\right].
\]

This is the source derivative per unit alpha. All coefficients are vacuum quantities. It is local at the retained radius order because the baseline current is zero; neither second-order shaping nor the nonlocal finite-current integral enters this leading forcing. This property does not justify dropping second-order quantities in the manuscript's plasma-gradient calculation.

The displacement in normal gauge is `xi = u*n + v*b`. With physical output-first `G_ij = dB_i/dx_j`,

\[
 {d\over ds}\begin{pmatrix}u\\v\end{pmatrix}
 =A\begin{pmatrix}u\\v\end{pmatrix}+f,
\quad A=\begin{pmatrix}
 G_{nn}/(s_GB_0)&G_{nb}/(s_GB_0)+\tau\\
 G_{bn}/(s_GB_0)-\tau&G_{bb}/(s_GB_0)
 \end{pmatrix},\quad
 f={1\over s_GB_0}\begin{pmatrix}b_{1n}\\b_{1b}\end{pmatrix}.
\]

The solution is periodic. This is not `-inverse(G_perp) @ b_perp`; the derivative and frame rotation matter. For an actual-coil reference with small B0 variation, use the actual signed tangent field in this general response and document that the explicit QS forcing remains an approximation. Do not call that mixed diagnostic an exact non-QS theory.

For the ideal first-order reference,

\[
 E=\begin{pmatrix}x&0\\\chi\sigma/x&\chi/x\end{pmatrix},\quad
 (u,v)^T=E(q_1,q_2)^T,\quad
 \zeta=q_1+iq_2,
\]
\[
 F=1-{1\over1+x^2+i\sigma},\quad
 C_p={2\mu_0p_{2*}a^2\ell^2\bar\eta\over B_0^2\nu},\quad
 \zeta'-i\nu\zeta=-iC_pF,
\]
\[
 \zeta(\varphi)=C_p\sum_k {F_k\over\nu-k}e^{ik\varphi}.
\]

The prime is `d/dvarphi`. Reconstruct

\[
 u=x\Re\zeta,\quad v={\chi\over x}(\sigma\Re\zeta+\Im\zeta).
\]

For one field period sampled uniformly in Boozer angle, a NumPy implementation of the transform itself is

```python
m = np.fft.fftfreq(nphi, d=1.0 / nphi)
k = nfp * m                         # FULL-TURN integer harmonics
F = 1.0 - 1.0 / (1.0 + x*x + 1j*sigma)
zeta = np.fft.ifft(Cp * np.fft.fft(F) / (nu - k))
u = x * zeta.real
v = chi * (sigma * zeta.real + zeta.imag) / x
```

Validate finite inputs and positive radius/B0, nonzero x and curvature, and the relevant spectral gap **before** division. Reject an exactly resonant forced harmonic. Report rather than silently clip small denominators. The full-turn axis nondegeneracy condition is `nu not in integers`; restricting the forcing to a field-period symmetry class changes its allowed lattice but does not prove uniqueness against all full-turn perturbations. Recover `N` consistently from `iota-iotaN`; do not mistake `helicity` or `nfp` for N without checking the library convention.

On the native geometrical grid, avoid interpolation altogether for one independent check:

\[
D_\varphi=\operatorname{diag}(\ell/|\partial_\phi r_v|)D_\phi,
\qquad(D_\varphi-i\nu I)\zeta=-iC_pF.
\]

Use odd point counts initially. A separate FFT route should construct a high-accuracy monotone Boozer-angle map by spectrally integrating the metric, invert it, and periodically interpolate the data. Verify convergence of that coordinate conversion separately. The old trapezoidal-angle accuracy issue in this PR is not permission to claim spectral convergence for a low-order angle map.

Also implement the physical-frame block collocation equation, or a two-dimensional fundamental-matrix shooting solve. For the latter, integrate `U_s=A@U`, `U(0)=I`, and the inhomogeneous solution, then enforce `(I-U(L)) w0 = U(L) integral(U^-1 f ds)`. Compare it with the reduced solve without sharing its transformed operator. Record the matrix condition number or singular gap. The bundled `checks/check_shafranov_shift.py` gives an independent NumPy/SymPy reference, not a replacement for tests against native code.

## Observables and non-negotiable coordinate conversion

To compare at the same laboratory toroidal plane, convert the normal-gauge displacement:

\[
\xi_{\rm lab}=\xi-\mathbf t
 {\mathbf e_\phi\cdot\xi\over\mathbf e_\phi\cdot\mathbf t},\quad
\delta R=\mathbf e_R\cdot\xi_{\rm lab},\quad
\delta Z=\mathbf e_Z\cdot\xi_{\rm lab}.
\]

Check that the toroidal tangent does not vanish. Then compare VMEX axes at identical `phi` to `alpha * (delta_R, delta_Z)`, subtracting the **vacuum VMEX axis at the same numerical resolution and external-field representation**. For a stellarator-symmetric axis, `sigma(0)=0`, `v(0)=0`, the Frenet normal is radial, and `delta_Z(0)=0`. The normal sign is determined from the actual frame, not assumed. The value of `delta_R(0)` still depends on every allowed harmonic.

Do not independently translate away the axis displacement before measuring this observable. The old helper `compare_to_near_axis` measures target/reconstruction offsets and may still be used for a separate diagnostic, but it is not the Shafranov-shift result.

Compute axis length from the cylindrical Fourier axis at common resolution:

\[
 L=\int_0^{2\pi}\sqrt{R_\phi^2+R^2+Z_\phi^2}\,d\phi.
\]

Use high-order periodic quadrature and Fourier derivatives. Never use polygon length at a fixed coarse grid as a pressure signal. The first-order prediction is

\[
 {L_1\over L}=\beta_*{(\ell\bar\eta)^2\over\nu^2}{\cal A},\quad
 \beta_*={-2\mu_0p_{2*}a^2\over B_0^2},\quad
 {\cal A}=\left\langle {x^2(1+x^2)+\sigma^2\over D}\right\rangle_{\varphi}.
\]

This coefficient is positive for the specified branch. Test both `-integral(kappa*u ds)/L` and the formula, and compare `(L_alpha-L_0)/alpha/L_0` from VMEX. The positivity is not a prediction that R increases at every plane. For geometrical-grid averages use the arclength weights, not a flat average.

The `k=-N` harmonic has coefficient proportional to `1/[(iota0-N)*iota0]`, whereas the mean length response contains `1/nu²`. Record these separately for QA and QH; a larger `|nu|` alone does not exclude a nearly resonant nonconstant response.

## Existing interfaces to reuse

In `nearaxis_finite_beta_helpers.py`:

- `coil_targets(solution, radius)` uses `plasma_hessian_on_axis(..., formal_radius=radius)`, with sample-first, field-output-first arrays. Its nested fields include `data.field.field.field`, `data.field.gradient`, and `data.hessian`; do not infer array order from a name.
- `evaluate_field`, `refine_coils`, `flux_surface`, and `toroidal_flux` provide field sampling, quadrature refinement without changing coil curves, cylindrical-plane surfaces, and an independent flux integral.
- `solve_free_boundary(solution, external_field, radius, directory, name, settings, mgrid=None)` exports a boundary, sets LFREEB, saves the runtime input, invokes VMEX, writes a wout, and reports vacuum activation and interface checks. Its current interface has no pressure-scan/restart override, so make a small deliberate extension or use a short scan driver around the same public VMEX calls. Do not repeatedly export a changing fixed-axis finite-pressure target and confuse that with freezing the vacuum experiment.
- `write_coil_mgrid` writes one scaled group including the physical coil currents. Use EXTCUR=1, read the file back, and retain `GuardedMgridField` or an equivalent bounds check. The default unguarded interpolation can clamp out-of-range points.
- `interface_check` extrapolates from the final two plasma half-meshes. Its current no-sheet-current diagnostic needs radial/extrapolation convergence; it is not exact interface evidence.

VMEX's inspected public call is

```python
result = vj.solve_free_boundary_multigrid(
    inp,
    external_field=coil_field,
    ns_array=ns,
    ftol_array=ftol,
    niter_array=niter,
    restart_from=previous_wout_or_result,  # or initial_state, never both
    verbose=True,
    emit=emit_to_console_and_file,
    raise_on_max_iterations=False,
)
```

`restart_from` accepts a wout path, `WoutData`, `SolveResult`, or `SpectralState`; its documented ladder skips rungs below the restart resolution. Preserve the free edge rather than clamping it to the seed boundary. `initial_state` is an alternative with reset-file activation semantics. The free-boundary entry point does **not** accept every fixed-boundary polishing option: inspect its signature before adding controls. Final exterior data are in `result.vacuum`.

Follow the existing helper for serialization:

```python
wout = vj.wout_from_state(
    inp=inp, state=result.state,
    fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
    niter=int(result.iterations), converged=bool(result.converged),
    vacuum_output=result.vacuum,
)
vj.write_wout(path, wout)
R_axis, Z_axis = surface_rz(
    wout, s_index=0, theta=np.zeros(1), phi=common_geometric_phi,
)
R_axis, Z_axis = R_axis[0], Z_axis[0]
```

For input generation, export **one reference boundary** through `pyqsc_jax.vmec.to_vmec`, parse with `vj.VmecInput.from_file`, and use `dataclasses.replace` for the pressure family. Set and round-trip-check the actual fields corresponding to

```text
LFREEB = T
MGRID_FILE = 'nonempty label or actual MAKEGRID filename'
NCURR = 1
CURTOR = 0
PCURR_TYPE = 'power_series'
AC = 0
PMASS_TYPE = 'power_series'
AM = 1 -1
PRES_SCALE = alpha * p0_star
GAMMA = 0
SPRES_PED = 1
BLOAT = 1
PHIEDGE = fixed signed reference flux
```

The inspected VMEX example decks use `AC=0` with zero CURTOR. Verify that the parsed and evaluated current profile is zero at every radial sample; zero edge current alone is not enough. Do not change both AM and PRES_SCALE by alpha. Input pressure is in Pa; VMEX handles its internal magnetic units. The helper uses a nonempty `mgrid_file` label even for direct Biot--Savart, because an empty value can cause the input reader to demote the free-boundary request. Check the parsed `lfreeb`, `ncurr`, pressure, flux, and external-current scaling before running. In exported Fourier boundary inputs MPOL counts retained modes from zero: retaining m=8 requires MPOL=9.

pyQSC_JAX canonical fields inspected in `models.py` include `solution.sigma`, `solution.iotaN`, `solution.iota`, `solution.G0`, `solution.X1c`, `solution.Y1s`, `solution.Y1c`, `solution.inputs.etabar/B0/sG/spsi/I2/p2`, and `solution.geometry`. Use `solution.geometry`'s Cartesian frames and metric fields, and verify `ell=abs(G0)/B0`. The optional mutable facade in `near_axis.py` is an adapter; do not transpose canonical arrays into its legacy convention accidentally.

## Verification sequence and error separation

### 1. Vacuum existence and field fidelity

Locate the actual coil-field periodic axis by a Poincare return-map root in `(R,Z)` over one field period, initialized near the target. Use direct Biot--Savart and refined coil quadrature. Verify closure by a separate integration. The field-line equations in geometrical phi are `dR/dphi = R B_R/B_phi`, `dZ/dphi = R B_Z/B_phi`; reject vanishing B_phi. Obtain the variational return map independently from the coil derivative or converged finite differences. Compare its Floquet phase with signed field-line winding and the near-axis transform, including the field-period/helicity convention.

Trace nearby vacuum surfaces and measure enclosed flux. Establish an admissible radius below any detected island, escape, or stochastic region. Solve the vacuum free-boundary VMEX problem there and compare its axis, surfaces, total-field tangency, and transform to direct coil tracing. Investigate the documented vacuum-iota discrepancy rather than carrying it into a pressure derivative. A vacuum solver can introduce an effective interface response if the seed is not a true coil-field surface; check both field sides.

Compare the actual coil field and gradient on the vacuum axis with the first-order target, and evaluate the residual of the target axis tangency and linear response operator. The ideal-QS spectral predictor and an actual-coil-gradient response are useful separate diagnostics. Any persistent difference is a coil-fit/model error, not a radius-truncation error. Tighten the vacuum coil fit or reconstruct the reference geometry as needed; do not fit a pressure-dependent amplitude to absorb the difference.

### 2. Algebra and linear response

Run the bundled symbolic checks. Add native tests for both orientation signs, QA and QH helicity, a manufactured Fourier forcing, symmetry parity, and periodic closure. Compare the scalar complex solve against the physical-frame block solve and, for one case, a shooting solve. Verify the zero-pressure and a² scaling of the *prediction*. Verify the invariant axis-length first variation using a centered geometric perturbation of the reference curve; this uses no negative-pressure physical equilibrium.

Recover the fixed-boundary circular-tokamak formula separately:

\[
Y_{2s}=X_{2c},\quad X_{20}=(\beta_p+3/4)/R,\quad
B_{2c}=-B_0(\beta_p+1/4)/(2R^2),\quad
\Delta_{R,\rm ax}=a^2(\beta_p+1/4)/(2R).
\]

Changing pressure at fixed B2c is not the shifted-circle family. This test has finite current and a fixed boundary; it is not a current-free vacuum stellarator test. Do not force the zero-current spectral formula into the singular circular-vacuum limit.

### 3. Pressure and numerical ladders

Use a manageable pilot resolution first, then a resolution study with independently varied controls. Suggested starting ladders, not automatic accuracy certificates: NS=33/65/129, highest m,n=8/10/12, NZETA=32/64/96 per field period, FTOL=1e-9/1e-10/1e-11, and coil quadrature=240/480/960. Adapt these to the actual geometry and measured error; increasing modes is not a cure for a wrong branch. Preserve all physical parameters across a resolution comparison. Use common laboratory planes and common flux labels, not nearest unequal radial mesh indices.

At every accepted point save runtime deck, field/coil hash, wout, solver log, residuals, vacuum activation, interface jumps, pressure balance, axis Fourier data, measured current profile, volume-averaged beta, target beta, signed transform, and coordinate-conversion diagnostics. Refine the exterior vacuum solve and plasma side separately where possible. A no-sheet-current assumption requires the relevant jump to converge below the pressure-response error budget.

Fit `axis_alpha - axis_0 = alpha*slope + alpha²*quadratic` with a **fixed zero intercept**, then compare against smaller-amplitude one-sided slope fits. A useful extrapolation is `slope_est(h)=2*D(h/2)-D(h)`, where `D(h)=(axis(h)-axis(0))/h`. Its leading pressure-step error is O(h²) when the branch is smooth. The signal must remain larger than numerical noise. Repeat the same analysis for length. Do not enforce a theoretical sign on the data.

At each finite radius the extrapolated slope can still differ from the leading source theory. Keep four contributions distinct: pressure-step error, equilibrium/field discretization error, coil-reference/model mismatch, and radius truncation. Under the paper's smooth fixed-coefficient expansion the *leading expected* relative radius correction is O(a² log(R/a)); fit its trend only over a regime where the other errors are smaller. Do not insist on an exact slope two over a short radius range. Report an upper bound or an unresolved plateau when the data cannot distinguish terms.

A direct-field versus mgrid comparison is a tabulation check, not independent MHD physics validation. Use the direct field for the primary result. Refine a reloaded MAKEGRID table for selected points with a fixed safe domain and unit group scaling. Compare its induced axis shift to the direct-field numerical uncertainty. Neither route should sample outside the table.

### 4. Independence and failure handling

At least one central QA case should be verified against direct tracing at vacuum and independently converged VMEX pressure continuation. A second geometry, preferably QH if a good current-free vacuum baseline exists, tests toroidally varying and helical response. An additional VMEC2000 or DESC solve at selected pressure values would strengthen independence but is not a substitute for the vacuum-reference gate. Do not count direct and mgrid VMEX routes as two equilibrium codes.

An unaccepted run stays in the archive with its reason. Failure to converge, a vacuum surface outside the nested region, current/flux drift, a conditioning failure, or a pressure signal below the numerical floor cannot enter agreement plots as valid data. Do not extrapolate a missing wout, copy another checkpoint's results, or tune tolerances selectively to improve agreement.

## Figures and data products

Produce separate single-axis figures suitable for LaTeX inclusion, without titles or prose paragraphs. Use consistent physical conventions, readable journal-size labels, and legends naming the theory and computation. Put assumptions and resolution in captions, not across the image. Use curves and symbols as well as color; retain signed displacements.

Required primary graphics:

1. **Symmetry-plane shift versus pressure**: delta_R(0)/a against alpha or measured reference-normalized beta. Show the untuned linear prediction and converged VMEX points with numerical uncertainty. Show delta_Z(0) as a separately documented symmetry residual, not an arbitrary tiny axis range that exaggerates it.
2. **Toroidal displacement profile**: two separate graphics for delta_R(phi)/alpha and delta_Z(phi)/alpha on common laboratory planes, with predictions and selected pressure-extrapolated data. Include the normal-to-laboratory conversion. A single plane is insufficient to test the periodic response.
3. **Axis-length change versus pressure**: `(L_alpha-L_0)/L_0` with the positive predicted slope. Include the independently measured length quadrature error.
4. **Convergence evidence**: a separate slope-error-versus-radius plot and a compact numerical error-budget table. Where pressure nonlinearity is visible, a separate residual-versus-alpha plot distinguishes O(alpha²) behavior from the finite-radius floor.

These are actual simulation results, not synthetic demonstrations. The simple manufactured checks should be labeled as tests and kept out of the physical-result figures. Save PDF and SVG plus 300--450 dpi PNG. Render the PDFs independently and inspect at intended print width; check text bounds, legends, signs, common axis conventions, and missing points. No large suptitles, decorative boxes, or unlabeled gray lines. Store an adjacent CSV/NPZ/JSON of every plotted quantity so figures can be rebuilt without running equilibria.

Provide a short results section in LaTeX, captions, and an equation-to-code map keyed to the manuscript labels (`eq:shiftforcing`, `eq:shiftformula`, `eq:shiftsymmetry`, `eq:shiftlength`, `eq:shiftlab`). Do not insert numeric claims into the manuscript until their run hashes and uncertainty estimates exist. The current compiled paper deliberately contains no new VMEX pressure-scan results.

## Definition of done

The branch contains a reproducible small driver, the theory and coordinate-convention documentation, focused independent tests, immutable run manifests, and the figures above. The analytic implementation passes both physical-frame and scalar-response comparisons. The vacuum reference is independently verified, the complete toroidal-current profile is zero, and coils/flux remain fixed. At least a well-resolved primary pressure family and an independently checked second reference are evaluated, or a specific documented physical/numerical blocker explains why the second reference cannot be used.

For the primary result, pressure slopes stabilize under step reduction; numerical errors are measured separately; the full toroidal profile and symmetry-plane result are compared; and the length identity is tested without fitting its amplitude. Agreement is assessed against the combined error budget. A practical target is a few-percent or better leading-order agreement at the smallest usable radii, with numerical uncertainty substantially smaller, but this is not permission to hide a reproducible discrepancy. A demonstrated discrepancy with isolated causes is an honest outcome and must remain visible.

Before pushing, rerun the relevant ESSOS tests, the new tests, and at least one clean driver replay from saved inputs. Verify no large simulation archive is accidentally committed; store compact results in git and link the full immutable archive with a checksum. Report changed files, test commands, revisions, runtime requirements, validated cases, unresolved limitations, figure paths, and manuscript-ready conclusions in PR #70. Check all new commit authors and trailers as specified above.

## Literature and exact source links

- Shafranov (1960), *Equilibrium of a Plasma Toroid in a Magnetic Field*, Soviet Physics JETP 10, 775--779: https://jetp.ras.ru/cgi-bin/dn/e_010_04_0775.pdf
- Jorge, Sengupta & Landreman (2020), direct expansion, sections 4.3--4.4: https://arxiv.org/abs/1911.02659 ; doi:10.1017/S0022377820000033.
- Rodriguez & Plunk (2025), section 7, fixed-axis sensitivity and first-order dependence: https://arxiv.org/abs/2505.02465.
- Hudson, Guinchard & Sengupta (2025), axis sensitivity: doi:10.1063/5.0241455.
- Hudson et al. (2025), fixed-coil vacuum/pressure code verification: doi:10.1063/5.0253843.
- Helander & Nikulsin (2026), fixed-boundary flux-weighted shift: https://arxiv.org/abs/2605.22105.
- ESSOS helper: https://github.com/uwplasma/ESSOS/blob/7713e07a77f08c15e70a34526dc9edf5b00da44d/examples/coil_optimization/nearaxis_finite_beta_helpers.py
- Existing blockers/archive: https://github.com/uwplasma/ESSOS/blob/7713e07a77f08c15e70a34526dc9edf5b00da44d/examples/coil_optimization/finite_beta_paper/README.md
- VMEX multigrid/restart contract: https://github.com/uwplasma/vmex/blob/926892ab7131a6bc0c5b61218d1f75e7b77bc401/vmex/core/multigrid.py
- VMEX profile semantics: https://github.com/uwplasma/vmex/blob/926892ab7131a6bc0c5b61218d1f75e7b77bc401/docs/howto/profiles.md
