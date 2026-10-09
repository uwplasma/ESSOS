"""Compare the Fourier-coefficient-basis SVD/entropy winding surface against
tracked PR baselines (normal offset, ESSOS entropy, ESSOS Pareto, REGCOIL
adjoint), reusing winding_surface_comparison_8_coils's own helper functions
and methodology so the numbers are directly comparable.

Not integrated into the tracked comparison study; nothing here is committed.
The optimizer runs in-process; every REGCOIL-touching step runs as a
watchdog-supervised subprocess (memory-limited, killed if system memory runs
low) since this machine has a hard 12 GB WSL ceiling.

Run directly: python3 winding_surface_fourier_comparison.py
Internal subprocess entry points (do not call directly):
  --task validate96 <spec.json>
  --task resconv <spec.json>
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from winding_surface_comparison_8_coils import winding_surface_comparison as ws

FOURIER_METHOD_NAME = "ESSOS Fourier entropy"
DIPOLE_METHOD_NAME = "ESSOS entropy (this machine)"
# Tracked cross-machine baselines to reuse as-is (not recomputed). Note
# "ESSOS entropy" (no suffix) is still pulled in here too, so the combined
# CSVs let you compare the tracked cross-machine dipole numbers directly
# against DIPOLE_METHOD_NAME's freshly-rerun, same-machine numbers.
BASELINE_METHODS = ("normal offset", "ESSOS entropy", "ESSOS Pareto", "REGCOIL adjoint")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, "output", "winding_surface_fourier_comparison")
BASELINE_DATA = os.path.join(
    ROOT, "winding_surface_comparison_8_coils", "data")

# Quadrature grid for fourier_induction_matrix's integral inside the
# optimization loop. Calibrated empirically (see scratch calibration run):
# at mpol=ntor=6 (ws.POTENTIAL_MPOL/NTOR), this resolution gives Biot-Savart
# cross-check agreement of ~1e-6 to ~1e-8 relative error across all three
# cases (QA, QH, W7-X) -- far tighter than the ~1e-3 level that would already
# be adequate for an optimization proxy.
QUAD_NTHETA = 64
QUAD_NPHI_PER_PERIOD = 96

# Watchdog settings. WSL has a hard 12 GB ceiling (.wslconfig: memory=12GB).
# Both limits are enforced via psutil RSS/available-memory polling, not
# resource.setrlimit(RLIMIT_AS, ...): that was tried first and rejected after
# it killed a JAX/XLA worker using only ~1.9 GB RSS -- XLA's CPU backend
# reserves large virtual address ranges independent of actual physical usage,
# so RLIMIT_AS (a virtual-memory cap) fires on JAX processes long before any
# real memory pressure exists. RSS-based polling reflects real usage instead.
MEMORY_FLOOR_GIB = 1.5
PROCESS_RSS_CAP_GIB = 8.0
POLL_INTERVAL_S = 2.0

CASES = None  # populated by build_cases(), needs `regcoil` for W7-X paths


def build_cases():
    import regcoil
    input_files = os.path.join(ROOT, "input_files")
    return {
        "Landreman-Paul QA": {
            "wout": os.path.join(
                input_files, "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")},
        "Landreman-Paul QH": {
            "wout": os.path.join(
                input_files, "wout_LandremanPaul2021_QH_reactorScale_lowres.nc")},
        "W7-X": {"wout": str(regcoil.examples("W7-X").wout),
                 "bnormal": str(regcoil.examples("W7-X").vcasing)},
    }


# ---------------------------------------------------------------------------
# Fourier induction matrix (production settings: ws.POTENTIAL_MPOL/NTOR = 6).
#
# Two wall-time optimizations over the first working version (both verified,
# in a separate investigation pass, to reproduce the original matrix/entropy/
# gradient to machine precision -- see the v1-vs-v2 comparison figures and
# examples/winding_surface_fourier_comparison/README.md):
#
# 1. nfp-fold periodicity reduction. The Fourier basis functions
#    sin(m*theta - n*nfp*phi) are *exactly* periodic with period 2*pi/nfp in
#    phi (xn is always an integer multiple of nfp -- see potential_modes), so
#    the field they produce is itself exactly nfp-periodic. Only one field
#    period of plasma points is needed, and the kernel contribution from all
#    nfp copies of each winding point can be summed *before* contracting with
#    the potential basis, rather than after -- an O(nfp^2) -> O(nfp)
#    reduction in the dominant matrix-construction cost. This is simpler than
#    the dipole method's induction_singular_values, which needs a complex
#    per-mode DFT because raw grid DOFs have no built-in periodicity; the
#    Fourier basis functions already do, so a plain real-valued sum over
#    rotations suffices. Uses the same one_period/full_integral quadrature
#    weight convention build_operators already establishes for this exact
#    kind of reduction.
# 2. Matmul-restructured dipole kernel. ws.dipole_kernel materializes a
#    (P,Q,3) pairwise-difference tensor; dipole_kernel_matmul is algebraically
#    identical but uses the |a-b|^2 = |a|^2 - 2 a.b + |b|^2 identity to
#    replace it with four (P,3)@(3,Q) matrix multiplications (BLAS GEMM) plus
#    small vector reductions -- no (P,Q,3) intermediate.
# ---------------------------------------------------------------------------

def dipole_kernel_matmul(plasma_points, plasma_normals, winding_points, winding_normals):
    """Algebraically identical to ws.dipole_kernel; avoids materializing a
    (P,Q,3) difference tensor by using (P,3)@(3,Q) matrix multiplications."""
    plasma_sq = jnp.sum(plasma_points ** 2, axis=1)
    winding_sq = jnp.sum(winding_points ** 2, axis=1)
    distance_squared = (plasma_sq[:, None] - 2 * (plasma_points @ winding_points.T)
                       + winding_sq[None, :])

    plasma_dot_normal = jnp.sum(plasma_points * plasma_normals, axis=1)
    winding_dot_normal = jnp.sum(winding_points * winding_normals, axis=1)
    diff_dot_plasma_normal = (plasma_dot_normal[:, None]
                              - plasma_normals @ winding_points.T)
    diff_dot_winding_normal = (plasma_points @ winding_normals.T
                               - winding_dot_normal[None, :])
    normals_dot = plasma_normals @ winding_normals.T

    return ws.MU0 / (4 * jnp.pi) * (
        normals_dot
        - 3 * diff_dot_plasma_normal * diff_dot_winding_normal / distance_squared
    ) / distance_squared ** 1.5


def fourier_induction_matrix(plasma, winding, xm, xn):
    """Row-weighted map from Phi's Fourier coefficients to B_normal.

    Exploits the winding/plasma surfaces' exact nfp-fold discrete rotational
    symmetry (see module-level comment above) and dipole_kernel_matmul's
    matmul-restructured kernel; uses the same one_period/full_integral
    quadrature-weight convention build_operators already establishes for
    this kind of reduction, so the sign/normalization convention is
    guaranteed identical to the rest of this study.
    """
    nphi_period_plasma = plasma.nphi // plasma.nfp
    plasma_points = plasma.gamma[:nphi_period_plasma].reshape(-1, 3)
    plasma_normals = plasma.unitnormal[:nphi_period_plasma].reshape(-1, 3)

    nphi_period_winding = winding.nphi // winding.nfp
    theta0 = winding.theta2d[:nphi_period_winding].reshape(-1, 1)
    phi0 = winding.phi2d[:nphi_period_winding].reshape(-1, 1)
    potential_basis0 = jnp.sin(xm[None, :] * theta0 - xn[None, :] * phi0)

    winding_weights = ws.quadrature_weights(winding).reshape(winding.nphi, winding.ntheta)
    winding_weights0 = winding_weights[:nphi_period_winding].reshape(-1)

    summed_kernel = 0.0
    for period in range(winding.nfp):
        section = slice(period * nphi_period_winding, (period + 1) * nphi_period_winding)
        kernel = dipole_kernel_matmul(
            plasma_points, plasma_normals,
            winding.gamma[section].reshape(-1, 3), winding.unitnormal[section].reshape(-1, 3))
        summed_kernel = summed_kernel + kernel

    physical_field0 = summed_kernel @ (winding_weights0[:, None] * potential_basis0)
    plasma_weights0 = ws.quadrature_weights(plasma, one_period=True, full_integral=True)
    return jnp.sqrt(plasma_weights0)[:, None] * physical_field0


def optimize_fourier_surface(wout):
    """Mirrors ws.optimize_surface(wout, "entropy", ...) exactly, except the
    induction matrix comes from fourier_induction_matrix (evaluated on a
    separate, finer quadrature surface) instead of
    ws.induction_singular_values (a per-grid-point dipole matrix).
    Same outer grid, same geometry regularizers/weights, same optimizer
    settings, same initialization -- so any difference in outcome is
    attributable to the induction-matrix representation alone.
    """
    plasma = ws.make_surface(wout, ws.OPT_NTHETA, ws.OPT_NPHI_PER_PERIOD)
    winding = ws.make_surface(wout, ws.OPT_NTHETA, ws.OPT_NPHI_PER_PERIOD)
    winding.dofs = ws.baseline_surface(wout)
    quadrature_winding = ws.make_surface(
        wout, QUAD_NTHETA, QUAD_NPHI_PER_PERIOD, winding.dofs)
    xm, xn = ws.potential_modes(winding.nfp)

    def values(surface):
        geometry = ws.geometry_metrics(plasma, surface)
        quadrature_winding.dofs = surface.dofs
        induction_matrix = fourier_induction_matrix(
            plasma, quadrature_winding, xm, xn)
        singular_values = jnp.linalg.svd(induction_matrix, compute_uv=False)
        probabilities = singular_values / jnp.sum(singular_values)
        entropy = -jnp.sum(probabilities * jnp.log(
            jnp.maximum(probabilities, 1e-300)))
        return (1 / entropy, jnp.sum(singular_values)) + geometry

    initial = values(winding)
    jax.block_until_ready(initial)
    initial_dofs = winding.dofs
    active = ws.active_surface_dofs(winding)
    x0 = initial_dofs[active]
    step_bound = min(ws.COEFFICIENT_STEP_BOUND,
                     0.02 * float(ws.mean_minor_radius(plasma)))
    bounds = [(float(x - step_bound), float(x + step_bound)) for x in x0]

    def objective_function(active_dofs):
        winding.dofs = initial_dofs.at[active].set(active_dofs)
        current = values(winding)
        physics = (current[0] / initial[0]
                  + jnp.maximum(1 - current[1] / initial[1], 0) ** 2)
        geometry, reference = current[2:], initial[2:]
        volume_term = -0.02 * geometry[0] / reference[0]
        return (physics + volume_term
                + ws.SPECTRAL_WEIGHT * geometry[1] / reference[1]
                + ws.DISTANCE_WEIGHT * jnp.maximum(
                    1 - geometry[2] / (0.9 * reference[2]), 0) ** 2
                + 100 * geometry[5]
                + ws.SELF_INTERSECTION_WEIGHT * geometry[6])

    value_and_grad = jax.jit(jax.value_and_grad(objective_function))
    evaluations = 0

    def fun(x):
        nonlocal evaluations
        value, gradient = value_and_grad(jnp.asarray(x))
        evaluations += 1
        if evaluations == 1 or evaluations % 10 == 0:
            print(f"{os.path.basename(wout)} fourier_entropy evaluation "
                 f"{evaluations:03d}: {float(value):.8e}")
        return float(value), np.asarray(gradient)

    from scipy.optimize import minimize
    start = time.perf_counter()
    result = minimize(fun, np.asarray(x0), method="L-BFGS-B", jac=True,
                      bounds=bounds, options={"maxiter": ws.MAXITER})
    elapsed = time.perf_counter() - start
    winding.dofs = initial_dofs.at[active].set(jnp.asarray(result.x))
    final_geometry = ws.geometry_metrics(plasma, winding)
    print(f"fourier_entropy: success={result.success}, iterations={result.nit}, "
         f"runtime={elapsed:.2f} s, distance={float(final_geometry[2]):.4f} m")
    return np.asarray(winding.dofs), elapsed, int(result.nit), bool(result.success)


def optimize_dipole_surface(wout):
    """The dipole/grid method ("ESSOS entropy"), run fresh on this machine.

    Thin wrapper around ws.optimize_surface -- the exact same call
    run_comparison() makes for the "ESSOS entropy" baseline -- so this
    reruns the tracked cross-machine baseline's own code path unmodified,
    just on this machine's JAX/BLAS/library versions and under this
    script's watchdog for the downstream REGCOIL steps.
    """
    return ws.optimize_surface(wout, "entropy", None)


# ---------------------------------------------------------------------------
# Validation workers (subprocess entry points). Each replicates the relevant
# per-(case, surface_method) block from ws.validate_surfaces /
# ws.sheet_resolution_study, generalized to an arbitrary set of dofs instead
# of iterating ws.SURFACE_METHODS, and writing to this script's own output
# directory instead of ws.OUTPUT.
# ---------------------------------------------------------------------------

def task_validate96(spec_file):
    """96x96 headline validation: one independent REGCOIL solve + coil cut,
    mirroring ws.validate_surfaces's per-(case, surface_method) block."""
    import regcoil
    with open(spec_file) as stream:
        spec = json.load(stream)
    wout, bnormal, dofs = spec["wout"], spec.get("bnormal"), np.asarray(spec["dofs"])
    resolution = 96
    surface = ws.make_surface(wout, resolution, resolution, dofs)
    plasma = regcoil.PlasmaSurface.from_wout(wout, ntheta=resolution, nzeta=resolution)
    if bnormal:
        plasma.set_bnormal_from_virtual_casing(bnormal)
    coil = regcoil.CoilSurface(
        np.asarray(surface.xm, int), np.asarray(surface.xn, int),
        np.asarray(surface.rc), np.asarray(surface.zs), nfp=surface.nfp,
        ntheta=resolution, nzeta=resolution)
    problem = regcoil.Regcoil(plasma, coil, mpol_potential=ws.POTENTIAL_MPOL,
                              ntor_potential=ws.POTENTIAL_NTOR)
    start = time.perf_counter()
    feasible = True
    try:
        solution = problem.solve_for_target("max_K", ws.KMAX)
    except ValueError:
        solution = problem.solve(lam=0)
        if solution.max_K > ws.KMAX:
            feasible = False
            solution = problem.solve(lam=1e20)
    if feasible:
        theta, phi, potential = ws.potential_grid(
            solution.solution, surface.nfp, ws.net_poloidal_current(wout),
            ws.CUT_RESOLUTION)
        curves, levels = ws.cut_coils(surface, potential)
        plasma_eval = ws.make_surface(wout, ws.FIELD_RESOLUTION, ws.FIELD_RESOLUTION)
        nphi_eval = plasma_eval.nphi // plasma_eval.nfp
        points = np.asarray(plasma_eval.gamma[:nphi_eval]).reshape(-1, 3)
        normals = np.asarray(plasma_eval.unitnormal[:nphi_eval]).reshape(-1, 3)
        field = ws.filament_field(
            points, curves, ws.net_poloidal_current(wout) / len(curves))
        filament_bnormal = np.sum(field * normals, axis=1)
        plasma_bnormal = ws.load_plasma_bnormal(
            wout, bnormal, ws.FIELD_RESOLUTION, ws.FIELD_RESOLUTION)
        if plasma_bnormal is not None:
            filament_bnormal = filament_bnormal + plasma_bnormal.reshape(-1)
        weights = np.asarray(ws.quadrature_weights(
            plasma_eval, one_period=True, full_integral=True))
        mod_b = ws.vmec_mod_b(
            wout, plasma_eval.theta2d[:nphi_eval], plasma_eval.phi2d[:nphi_eval])
        filament_f_b = float(np.sum(weights * filament_bnormal ** 2))
        filament_max_ratio = float(np.max(np.abs(filament_bnormal) / np.abs(mod_b)))
    else:
        filament_f_b = filament_max_ratio = float("nan")
    row = dict(
        configuration=spec["configuration"], surface_method=spec["surface_method"],
        resolution=resolution, feasible_at_Kmax=feasible,
        sheet_f_B_T2_m2=solution.f_B if feasible else float("nan"),
        sheet_max_abs_Bn_over_B=(solution.max_Bnormal_over_B if feasible else float("nan")),
        filament_f_B_T2_m2=filament_f_b,
        filament_max_abs_Bn_over_B=filament_max_ratio,
        sheet_f_K_A2=solution.f_K if feasible else float("nan"),
        sheet_rms_K_A_per_m=solution.rms_K if feasible else float("nan"),
        achieved_or_minimum_Kmax_A_per_m=solution.max_K,
        surface_runtime_s=spec["surface_runtime_s"],
        validation_runtime_s=time.perf_counter() - start)
    with open(spec["result_file"], "w") as stream:
        json.dump(row, stream)


def task_resconv(spec_file):
    """Re-evaluate the fixed 48x48-solved coefficients at one resolution,
    mirroring ws.sheet_resolution_study's per-row block (REGCOIL only, as
    entropy-family methods use throughout this study). One resolution per
    process invocation (not all three in one process): build_operators's
    intermediate (n_plasma_period x n_winding_full x 3) arrays scale with
    nfp, and at resolution=64 on the nfp=4/5 cases (QH/W7-X) computing all
    three resolutions in a single process pushed system memory low enough
    to trip the watchdog even though each resolution alone is comfortably
    within budget -- confirmed empirically before this split."""
    with open(spec_file) as stream:
        spec = json.load(stream)
    wout, bnormal = spec["wout"], spec.get("bnormal")
    dofs = np.asarray(spec["dofs"])
    coefficients = jnp.asarray(spec["coefficients"])
    resolution = spec["resolution"]
    surface = ws.make_surface(wout, resolution, resolution, dofs)
    plasma = ws.make_surface(wout, resolution, resolution)
    operators = ws.build_operators(
        plasma, surface, ws.net_poloidal_current(wout),
        ws.load_plasma_bnormal(wout, bnormal, resolution, resolution))
    weighted_bnormal = np.asarray(operators[0] @ coefficients + operators[1])
    current = np.asarray(
        operators[4] @ coefficients + operators[5]).reshape(-1, 3)
    row = dict(
        configuration=spec["configuration"], surface_method=spec["surface_method"],
        method="REGCOIL", resolution=resolution,
        sheet_f_B_T2_m2=float(np.vdot(weighted_bnormal, weighted_bnormal)),
        achieved_Kmax_A_per_m=float(np.max(np.linalg.norm(current, axis=1))))
    with open(spec["result_file"], "w") as stream:
        json.dump(row, stream)


# ---------------------------------------------------------------------------
# Watchdog: run one subprocess with a hard memory cap and a polling monitor.
# ---------------------------------------------------------------------------

def run_with_watchdog(args, label):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(f"WATCHDOG: launching [{label}]: {' '.join(args)}")
    process = subprocess.Popen(args, env=env)
    peak_rss_mb = 0.0
    killed = False
    kill_reason = None
    while True:
        if process.poll() is not None:
            break
        try:
            proc_info = psutil.Process(process.pid)
            rss = proc_info.memory_info().rss
            for child in proc_info.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak_rss_mb = max(peak_rss_mb, rss / 1024 ** 2)
        except psutil.NoSuchProcess:
            pass
        available_gib = psutil.virtual_memory().available / 1024 ** 3
        if available_gib < MEMORY_FLOOR_GIB:
            kill_reason = (f"available memory {available_gib:.2f} GiB < floor "
                           f"{MEMORY_FLOOR_GIB} GiB")
        elif peak_rss_mb / 1024 > PROCESS_RSS_CAP_GIB:
            kill_reason = (f"process RSS {peak_rss_mb / 1024:.2f} GiB > cap "
                           f"{PROCESS_RSS_CAP_GIB} GiB")
        if kill_reason:
            print(f"WATCHDOG: {kill_reason} -- terminating [{label}]")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            killed = True
            break
        time.sleep(POLL_INTERVAL_S)
    return_code = process.wait()
    print(f"WATCHDOG: [{label}] exited code={return_code} "
         f"peak_rss={peak_rss_mb:.1f} MB killed={killed}")
    return return_code, peak_rss_mb, killed


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run_case(case, data, method_name, optimizer_fn, file_tag):
    os.makedirs(OUTPUT, exist_ok=True)
    stem = case.replace(" ", "_").replace("-", "_")
    wout, bnormal = data["wout"], data.get("bnormal")

    surface_file = os.path.join(OUTPUT, f"surface_{stem}_{file_tag}.npz")
    if os.path.exists(surface_file):
        print(f"[{case}] reusing cached optimized surface at {surface_file}")
        cached = np.load(surface_file)
        dofs, elapsed, iterations, success = (
            cached["dofs"], float(cached["runtime"]),
            int(cached["iterations"]), bool(cached["success"]))
    else:
        dofs, elapsed, iterations, success = optimizer_fn(wout)
        dofs = np.asarray(dofs)
        np.savez(surface_file, dofs=dofs, runtime=elapsed,
                iterations=iterations, success=success)

    # 48x48 solver-comparison row: reuse ws.worker() UNMODIFIED via subprocess,
    # for byte-for-bit identical metric definitions to the baselines.
    k48_result_file = os.path.join(OUTPUT, f"k48_{stem}_{file_tag}_REGCOIL.npz")
    if not os.path.exists(k48_result_file):
        k48_spec_file = os.path.join(OUTPUT, f"k48_{stem}_{file_tag}_REGCOIL.json")
        spec = dict(configuration=case, surface_method=method_name, method="REGCOIL",
                   wout=wout, bnormal=bnormal, surface_runtime_s=float(elapsed),
                   surface_iterations=int(iterations), surface_converged=bool(success),
                   surface_file=surface_file, result_file=k48_result_file)
        with open(k48_spec_file, "w") as stream:
            json.dump(spec, stream)
        return_code, peak_mb, killed = run_with_watchdog(
            [sys.executable, ws.__file__, "--worker", k48_spec_file],
            f"{case} [{method_name}]: 48x48 solver comparison")
        if return_code != 0 or killed:
            print(f"[{case}] [{method_name}] 48x48 solver-comparison step "
                 "failed/aborted; skipping downstream steps for this case.")
            return None
    k48_result = np.load(k48_result_file)
    k48_metrics = json.loads(str(k48_result["metrics"]))
    coefficients = k48_result["coefficients"]

    # 96x96 headline validation.
    v96_result_file = os.path.join(OUTPUT, f"v96_{stem}_{file_tag}.json")
    if not os.path.exists(v96_result_file):
        v96_spec_file = os.path.join(OUTPUT, f"v96_{stem}_{file_tag}_spec.json")
        with open(v96_spec_file, "w") as stream:
            json.dump(dict(configuration=case, surface_method=method_name,
                           wout=wout, bnormal=bnormal, dofs=dofs.tolist(),
                           surface_runtime_s=float(elapsed),
                           result_file=v96_result_file), stream)
        return_code, peak_mb, killed = run_with_watchdog(
            [sys.executable, os.path.abspath(__file__), "--task", "validate96",
            v96_spec_file], f"{case} [{method_name}]: 96x96 validation")
        if return_code != 0 or killed:
            print(f"[{case}] [{method_name}] 96x96 validation step failed/aborted.")
            v96_row = None
        else:
            with open(v96_result_file) as stream:
                v96_row = json.load(stream)
    else:
        with open(v96_result_file) as stream:
            v96_row = json.load(stream)

    # 48/56/64 resolution convergence: one subprocess per resolution, for
    # real memory isolation (see task_resconv's docstring).
    resconv_rows = []
    for resolution in (48, 56, 64):
        resconv_result_file = os.path.join(
            OUTPUT, f"resconv_{stem}_{file_tag}_{resolution}.json")
        if os.path.exists(resconv_result_file):
            with open(resconv_result_file) as stream:
                resconv_rows.append(json.load(stream))
            continue
        resconv_spec_file = os.path.join(
            OUTPUT, f"resconv_{stem}_{file_tag}_{resolution}_spec.json")
        with open(resconv_spec_file, "w") as stream:
            json.dump(dict(configuration=case, surface_method=method_name,
                           wout=wout, bnormal=bnormal, dofs=dofs.tolist(),
                           coefficients=coefficients.tolist(),
                           resolution=resolution, result_file=resconv_result_file), stream)
        return_code, peak_mb, killed = run_with_watchdog(
            [sys.executable, os.path.abspath(__file__), "--task", "resconv",
            resconv_spec_file],
            f"{case} [{method_name}]: resolution convergence @ {resolution}")
        if return_code != 0 or killed:
            print(f"[{case}] [{method_name}] resolution-convergence @ {resolution} "
                 "failed/aborted.")
            continue
        with open(resconv_result_file) as stream:
            resconv_rows.append(json.load(stream))

    return dict(k48_metrics=k48_metrics, v96_row=v96_row, resconv_rows=resconv_rows)


def load_baseline_rows(csv_name, case_filter=None):
    path = os.path.join(BASELINE_DATA, csv_name)
    with open(path, newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
               if row["surface_method"] in BASELINE_METHODS]
    return rows


def optimize_normal_offset_surface(wout):
    """The trivial "normal offset" baseline, matching run_comparison()'s own
    lambda exactly: no optimization, just the fitted offset surface."""
    return ws.baseline_surface(wout), 0.0, 0, True


NORMAL_OFFSET_METHOD_NAME = "normal offset (this machine)"

METHODS_TO_RUN = (
    (NORMAL_OFFSET_METHOD_NAME, optimize_normal_offset_surface, "normal_offset"),
    (DIPOLE_METHOD_NAME, optimize_dipole_surface, "dipole_entropy"),
    (FOURIER_METHOD_NAME, optimize_fourier_surface, "fourier_entropy"),
)


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    cases = build_cases()
    results = {}
    for case, data in cases.items():
        for method_name, optimizer_fn, file_tag in METHODS_TO_RUN:
            print(f"\n=== {case} [{method_name}] ===")
            results[(case, method_name)] = run_case(
                case, data, method_name, optimizer_fn, file_tag)

    combined_96 = load_baseline_rows("surface_validation_96.csv")
    combined_48 = load_baseline_rows("comparison_metrics.csv")
    combined_resconv = load_baseline_rows("sheet_resolution_convergence.csv")

    for (case, method_name), result in results.items():
        if result is None:
            continue
        if result["v96_row"] is not None:
            combined_96.append(result["v96_row"])
        combined_48.append(result["k48_metrics"])
        combined_resconv.extend(result["resconv_rows"])

    def write_csv(rows, filename):
        if not rows:
            return
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with open(os.path.join(OUTPUT, filename), "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(combined_96, "surface_validation_96_combined.csv")
    write_csv(combined_48, "comparison_metrics_combined.csv")
    write_csv(combined_resconv, "sheet_resolution_convergence_combined.csv")
    print(f"\nSaved combined comparison CSVs to {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["validate96", "resconv"])
    parser.add_argument("spec_file", nargs="?")
    arguments = parser.parse_args()
    if arguments.task == "validate96":
        task_validate96(arguments.spec_file)
    elif arguments.task == "resconv":
        task_resconv(arguments.spec_file)
    else:
        main()
