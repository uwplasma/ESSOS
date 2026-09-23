"""Coils and near-axis equilibrium optimized together at finite plasma pressure.

At finite pressure the coils must not reproduce the total equilibrium field on the
magnetic axis, but the total field minus the field of the plasma current,

    B_coils = B_total - B_plasma,   and likewise for its gradient and its Hessian.

The Hessian matters most: the plasma part of it is order one rather than order a^2, so
coils fitted to the total field reproduce the wrong plasma shaping. Run with
SUBTRACT_PLASMA_FIELD = False to see that control case fail.

pyQSC_JAX evaluates the plasma contribution analytically from the near-axis solution,
for a plasma of boundary radius a = sqrt(PHIEDGE / (pi B0)), so no plasma surface or
virtual-casing integral is needed. The coils and the axis are then optimized together,
and the result is benchmarked with VMEX: a free-boundary equilibrium is computed in the
optimized coils and its flux surfaces are compared with the near-axis ones.

Choose the configuration with CASE. "qa_vacuum" and "axisymmetric" verify the method in
limits with known answers: no plasma field at all, and the vertical field of a tokamak.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import hashlib
import json
from pathlib import Path
from time import time
import jax
jax.config.update("jax_enable_x64", True)  # The plasma field is ~1e-4 of B0.
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import jacfwd, jit, vmap
from jax.flatten_util import ravel_pytree
from scipy.optimize import least_squares
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from pyqsc_jax.near_axis import near_axis
from pyqsc_jax.plasma import plasma_hessian_on_axis
import nearaxis_finite_beta_helpers as helpers


""" Input parameters """
CASE = "qa_finite_beta"   # "qa_finite_beta", "qa_vacuum" or "axisymmetric"
CASES = {
    # Landreman & Sengupta (2019) section 5.3 axis, re-solved with zero on-axis current.
    "qa_finite_beta": dict(rc=[1.0, 0.09], zs=[0.0, -0.09], nfp=2, etabar=0.95, B2c=-0.7, p2=-6.0e5, I2=0.0),
    # Same axis without pressure: the plasma field vanishes and the vacuum example is recovered.
    # The coils are fitted on the axis, so their closed surfaces do not reach a; the benchmark
    # therefore uses an inner surface, which costs nothing here because the pressure is zero anyway.
    # Without pressure or current the plasma has no response to drive the force residual down, so
    # the solve stalls near 1e-7 however long it runs; that is its floor, not a failure to converge.
    "qa_vacuum":      dict(rc=[1.0, 0.09], zs=[0.0, -0.09], nfp=2, etabar=0.95, B2c=-0.7, p2=0.0, I2=0.0,
                           vmex_radius_fraction=0.7, vmex=dict(ftol=(1e-7, 1e-7, 1e-7))),
    # Circular tokamak: the transform comes from I2, and the coils must supply the vertical field.
    "axisymmetric":   dict(rc=[1.0], zs=[0.0], nfp=4, etabar=1.0, B2c=0.0, p2=-2.0e5, I2=0.4),
}
case = CASES[CASE]
B0 = 1.0                              # Field on the magnetic axis [T]
# Boundary flux radius a [m], with PHIEDGE = pi B0 a^2. The plasma field grows as a^2 while the coil-fit
# error does not: at a = 0.03 the plasma field is ~30x and its gradient ~3x the remaining coil mismatch,
# whereas at a = 0.01 the plasma gradient is below that mismatch and its removal cannot be resolved.
PLASMA_RADIUS = 0.03
SUBTRACT_PLASMA_FIELD = True          # False: fit the coils to the total field instead, as a control
# False keeps the axis and etabar at their initial values and optimizes the coils only. With
# SUBTRACT_PLASMA_FIELD on and off it compares the two targets at one fixed equilibrium, which the
# joint control cannot do, since there the axis and the transform move with the coils.
OPTIMIZE_AXIS = True
OPTIMIZE = True                       # False: reuse OUTPUT_DIR/optimized_dofs.npz
# Evaluations per call of the optimizer, or None for all at once. With a limit, a run that has not
# used MAX_FUNCTION_EVALUATIONS yet continues from its own checkpoint, so a long optimization can be
# done in several shorter runs. Each restart resets the trust region, so the path differs slightly.
EVALUATIONS_PER_RUN = None
RUN_VMEX = True                       # Free-boundary benchmark of the coils
TRACE_FIELD_LINES = True              # Vacuum only: Poincare sections, traced transform and enclosed flux
SHOW_PLOTS = True
RUN_NAME = f"{CASE}{'' if SUBTRACT_PLASMA_FIELD else '_no_subtraction'}{'' if OPTIMIZE_AXIS else '_fixed_axis'}"
OUTPUT_DIR = Path(__file__).resolve().parent / f"output_finite_beta_{RUN_NAME}"

# Coils
# Matching the Hessian as well as the field and its gradient needs this much coil freedom: with 3 coils of
# order 6 the three targets compete, and the field match degrades below the plasma contribution.
N_COILS = 4; FOURIER_ORDER = 8; N_SEGMENTS = 60
N_SEGMENTS_BENCHMARK = 240            # Finer Biot-Savart quadrature for every diagnostic

# Optimization
NPHI = 41                             # Axis points per field period in the objective
NPHI_DIAGNOSTIC = 151                 # ... and in the diagnostics
# The whole script takes about nine minutes at these settings. Raising the budget to 2000 with NPHI = 61
# takes three times as long and buys little: the boundary error falls from 0.62 % to 0.55 % of |B|.
MAX_FUNCTION_EVALUATIONS = 1000
LENGTH_TARGET = 5.0; CURVATURE_TARGET = 6.0
IOTA_WEIGHT = 10.0; R0_WEIGHT = 10.0
# The field gradient fixes the elliptical cross-section, and the Hessian the triangularity and the
# second-order axis shift. Without it the coils reproduce the ellipse but not the shaping of the plasma.
# A weight near 0.01 brings B.n/|B| on the boundary below 1 %; a weight of 1 sacrifices the field match.
HESSIAN_WEIGHT = 0.01                 # 0: match only the field and its gradient
AXIS_SHAPE_BOUND = 0.02; ETABAR_RELATIVE_BOUND = 0.2

# VMEX free-boundary benchmark
VMEX_STATES = ("optimized",)          # The initial planar coils have no rotational transform to hold a plasma
VMEX_ROUTES = ("direct", "mgrid")     # Coil field evaluated directly, and read from a MAKEGRID file
# Optional stress test: a plasma of this fraction of r_singularity in the same coils, e.g. 0.8. The coils were
# optimized for the fitted plasma only, and VMEX rarely converges for it.
LARGER_PLASMA_FRACTION = None
# Fraction of PLASMA_RADIUS used as the VMEX plasma boundary. One solves exactly the plasma the coils
# were designed for. A smaller value solves an inner surface instead, which is what a vacuum case needs:
# with zero pressure it is the same physical field, but with pressure it is a different, smaller plasma.
VMEX_RADIUS_FRACTION = case.get("vmex_radius_fraction", 1.0)
FLUX_LEVELS = (0.0625, 0.25, 0.5625, 1.0)
VMEX = dict(mpol=8, ntor=8, nzeta=32, ntheta_boundary=64, delt=0.5, ns=(17, 33, 65),
            ftol=(1e-8, 1e-9, 1e-10), niter=(2000, 4000, 8000), flux_levels=FLUX_LEVELS)
VMEX.update(case.get("vmex", {}))
# The NESTOR vacuum solve limits the force residual near 2e-11 at NS = 65, so a tighter FTOL never converges.
MGRID_SHAPE = (97, 97, 64)            # (R, Z, phi); the phi count must be a multiple of nzeta
MGRID_MARGIN = None                   # Grid extent beyond the plasma [m]; None: max(8 % of R, 4 a)

RC, ZS, NFP = np.array(case["rc"]), np.array(case["zs"]), case["nfp"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not SHOW_PLOTS:
    plt.switch_backend("Agg")


""" Near-axis equilibrium, initial coils and degrees of freedom """
def make_near_axis(rc, zs, etabar, nphi=NPHI):
    # B0, p2, I2 and B2c are fixed inputs, so the optimizer cannot lower the pressure.
    # Third order adds no field-jet terms, so the coil targets are unchanged, but it relabels
    # the radius so that each surface encloses exactly the flux pi r^2 B0 through O(r^2).
    return near_axis(rc=jnp.asarray(rc), zs=jnp.asarray(zs), etabar=etabar, nfp=NFP, nphi=nphi, order="r3",
                     B0=B0, I2=case["I2"], p2=case["p2"], B2c=case["B2c"])

near_axis_initial = make_near_axis(RC, ZS, case["etabar"])
R0 = float(near_axis_initial.R0[0])
current_on_each_coil = 2 * np.pi * R0 * B0 / (4e-7 * np.pi * 2 * NFP * N_COILS)
curves = CreateEquallySpacedCurves(n_curves=N_COILS, order=FOURIER_ORDER, R=R0, r=R0 / 2,
                                   n_segments=N_SEGMENTS, nfp=NFP, stellsym=True)
field_initial = BiotSavart(Coils(curves=curves, currents=jnp.full(N_COILS, current_on_each_coil)))

coil_dofs, unravel_coils = ravel_pytree(field_initial)
n_coil_dofs, n_modes = coil_dofs.size, RC.size - 1
initial_shape = jnp.concatenate((jnp.asarray(RC[1:]), jnp.asarray(ZS[1:]), jnp.array([case["etabar"]])))
initial_dofs = jnp.concatenate((coil_dofs, initial_shape)) if OPTIMIZE_AXIS else coil_dofs

def dofs_to_fields(dofs, nphi=NPHI):
    shape = dofs[n_coil_dofs:] if OPTIMIZE_AXIS else initial_shape
    rc = jnp.concatenate((jnp.asarray(RC[:1]), shape[:n_modes]))
    zs = jnp.concatenate((jnp.asarray(ZS[:1]), shape[n_modes:2 * n_modes]))
    return unravel_coils(dofs[:n_coil_dofs]), make_near_axis(rc, zs, shape[-1], nphi)


""" Objective: match the coil field, its gradient and its Hessian to the external-field target """
IOTA_TARGET = float(near_axis_initial.iota)

def residuals(dofs):
    field, near = dofs_to_fields(dofs)
    solution = near.solution
    target = plasma_hessian_on_axis(solution, formal_radius=PLASMA_RADIUS)
    B_target, gradB_target, hessB_target = (
        (target.field.external_field, target.field.external_gradient, target.external_hessian) if SUBTRACT_PLASMA_FIELD
        else (solution.B_axis, solution.grad_B_axis, solution.grad_grad_B_axis))
    points = solution.geometry.position_cartesian
    weights = jnp.sqrt(solution.geometry.d_l_d_phi / jnp.sum(solution.geometry.d_l_d_phi))
    B_residual = weights[:, None] * (vmap(field.B)(points) - B_target) / B0
    gradB_residual = weights[:, None, None] * (vmap(field.dB_by_dX)(points) - gradB_target) * R0 / B0
    hessB_residual = jnp.sqrt(HESSIAN_WEIGHT) * weights[:, None, None, None] * (
        vmap(jacfwd(jacfwd(field.B)))(points) - hessB_target) * R0**2 / B0
    length = jnp.maximum(0.0, field.coils.length / LENGTH_TARGET - 1)
    curvature = jnp.maximum(0.0, field.coils.curvature / CURVATURE_TARGET - 1)
    residual = jnp.concatenate((
        B_residual.ravel(), gradB_residual.ravel(), hessB_residual.ravel(),
        length.ravel() / jnp.sqrt(length.size), curvature.ravel() / jnp.sqrt(curvature.size),
        jnp.atleast_1d(jnp.sqrt(IOTA_WEIGHT) * (near.iota - IOTA_TARGET)),
        jnp.atleast_1d(jnp.sqrt(R0_WEIGHT) * (near.R0[0] - R0) / R0)))
    converged = solution.root_report.converged & solution.second_order.linear_report.converged
    return jnp.where(converged, residual, jnp.nan)


""" Everything that defines the optimization problem. A checkpoint is reused only if this matches. """
CONFIG = dict(case=CASE, inputs=dict(case, B0=B0), plasma_radius=PLASMA_RADIUS, order="r3",
              subtract_plasma_field=SUBTRACT_PLASMA_FIELD, optimize_axis=OPTIMIZE_AXIS, n_coils=N_COILS,
              fourier_order=FOURIER_ORDER, n_segments=N_SEGMENTS, nphi=NPHI, max_nfev=MAX_FUNCTION_EVALUATIONS,
              length_target=LENGTH_TARGET, curvature_target=CURVATURE_TARGET, iota_weight=IOTA_WEIGHT,
              r0_weight=R0_WEIGHT, hessian_weight=HESSIAN_WEIGHT, axis_shape_bound=AXIS_SHAPE_BOUND,
              etabar_relative_bound=ETABAR_RELATIVE_BOUND, tolerances=dict(ftol=1e-8, gtol=1e-8, xtol=1e-10),
              x_scale="jac", method="trf")
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, default=float).encode()).hexdigest()[:16]


""" Optimization """
print(f"Case '{CASE}' ({'plasma field subtracted' if SUBTRACT_PLASMA_FIELD else 'CONTROL: coils fit to the total field'}): nfp={NFP}, a={PLASMA_RADIUS} m, p2={case['p2']:g} Pa/m^2, I2={case['I2']:g} T/m, "
      f"axis pressure {-case['p2'] * PLASMA_RADIUS**2:.3g} Pa")
checkpoint = OUTPUT_DIR / "optimized_dofs.npz"
cost_history, segments, start_dofs = [], [], np.asarray(initial_dofs)
if OPTIMIZE and EVALUATIONS_PER_RUN and checkpoint.exists():
    with np.load(checkpoint) as saved:
        if "config_hash" in saved and str(saved["config_hash"]) == CONFIG_HASH:
            start_dofs, cost_history = saved["optimized"], list(saved["cost_history"])
            segments = json.loads(str(saved["optimization"])).get("segments", [])
    if len(cost_history) >= MAX_FUNCTION_EVALUATIONS or (segments and segments[-1]["status"] != 0):
        OPTIMIZE = False  # Finished: the budget is spent or the optimizer stopped on its own.
    else:
        print(f"Continuing from evaluation {len(cost_history)} of {MAX_FUNCTION_EVALUATIONS}")
if OPTIMIZE:
    residuals_jit, jacobian_jit = jit(residuals), jit(jacfwd(residuals))
    time0 = time()
    jax.block_until_ready(jacobian_jit(initial_dofs))
    jax.block_until_ready(residuals_jit(initial_dofs))
    compile_seconds = time() - time0
    def residuals_recorded(dofs):
        value = np.asarray(residuals_jit(jnp.asarray(dofs)))
        cost_history.append(0.5 * float(np.sum(value**2)))
        return value
    width = np.r_[np.full(2 * n_modes, AXIS_SHAPE_BOUND), ETABAR_RELATIVE_BOUND * abs(case["etabar"])]
    lower = np.r_[np.full(n_coil_dofs, -np.inf), (np.asarray(initial_shape) - width) if OPTIMIZE_AXIS else []]
    upper = np.r_[np.full(n_coil_dofs, np.inf), (np.asarray(initial_shape) + width) if OPTIMIZE_AXIS else []]
    time0 = time()
    budget = MAX_FUNCTION_EVALUATIONS - len(cost_history)
    result = least_squares(residuals_recorded, start_dofs, jac=lambda x: np.asarray(jacobian_jit(jnp.asarray(x))),
                           bounds=(lower, upper), x_scale="jac", verbose=2, ftol=1e-8, gtol=1e-8, xtol=1e-10,
                           max_nfev=min(budget, EVALUATIONS_PER_RUN or budget))
    segments.append(dict(status=int(result.status), message=str(result.message), nfev=int(result.nfev),
                         njev=int(result.njev), cost=float(result.cost), optimality=float(result.optimality),
                         seconds=time() - time0, compile_seconds=compile_seconds))
    optimization = dict(segments[-1], success=bool(result.success), nfev=len(cost_history), segments=segments,
                        seconds=sum(g["seconds"] for g in segments))
    print(f"Optimization took {optimization['seconds']:.1f} seconds after {compile_seconds:.1f} s of compilation: "
          f"{result.message}")
    optimized_dofs = result.x
    np.savez(checkpoint, initial=np.asarray(initial_dofs), optimized=optimized_dofs, cost_history=cost_history,
             config=json.dumps(CONFIG, default=float), config_hash=CONFIG_HASH, optimization=json.dumps(optimization))
else:
    with np.load(checkpoint) as saved:
        if "config_hash" not in saved or str(saved["config_hash"]) != CONFIG_HASH:
            stored = json.loads(str(saved["config"])) if "config" in saved else {}
            changed = sorted(k for k in set(stored) | set(CONFIG) if json.dumps(stored.get(k), default=float)
                             != json.dumps(CONFIG.get(k), default=float))
            raise ValueError(f"{checkpoint} was made for a different problem (differs in {changed or 'unrecorded inputs'})."
                             " Set OPTIMIZE = True.")
        if not np.allclose(saved["initial"], initial_dofs, rtol=1e-12):
            raise ValueError(f"{checkpoint} starts from different coils. Set OPTIMIZE = True.")
        optimized_dofs, cost_history = saved["optimized"], list(saved["cost_history"])
        optimization = json.loads(str(saved["optimization"]))
    print(f"Reusing {checkpoint} (configuration {CONFIG_HASH})")


""" Diagnostics of the initial and optimized states, at higher axis and coil resolution """
states = {}
for name, dofs in (("initial", initial_dofs), ("optimized", jnp.asarray(optimized_dofs))):
    field, near = dofs_to_fields(dofs, NPHI_DIAGNOSTIC)
    solution = near.solution
    if not bool(solution.root_report.converged & solution.second_order.linear_report.converged):
        raise RuntimeError(f"The {name} near-axis solution did not converge.")
    field = helpers.refine_coils(field, N_SEGMENTS_BENCHMARK)
    targets = helpers.coil_targets(solution, PLASMA_RADIUS)
    surface, normal = helpers.normal_field_error(solution, targets, field, PLASMA_RADIUS)
    states[name] = dict(field=field, near=near, solution=solution, targets=targets, surface=surface, normal=normal,
                        match=helpers.axis_match(solution, targets, field))
    field.coils.to_json(str(OUTPUT_DIR / f"coils_{name}.json"))
    np.savez(OUTPUT_DIR / f"axis_targets_{name}.npz", rc=near.rc, zs=near.zs, etabar=near.etabar,
             a=PLASMA_RADIUS, **{k: v for k, v in targets.items() if isinstance(v, np.ndarray)})


""" Results """
print("\n" + "#" * 78)
for name, state in states.items():
    near, solution, match, targets = state["near"], state["solution"], state["match"], state["targets"]
    print(f"{name.capitalize()}: iota = {float(near.iota):.5f}, etabar = {float(near.etabar):.4f}, "
          f"max elongation = {float(jnp.max(near.elongation)):.3f}, r_singularity = {float(solution.r_singularity):.4f} m")
    print(f"   plasma field on axis       : RMS {match['plasma_field_rms_T']:.3e} T   "
          f"gradient RMS {match['plasma_gradient_rms_T_per_m']:.3e} T/m")
    print(f"   coils - target on the axis : RMS {match['field_rms_T']:.3e} T   "
          f"gradient RMS {match['gradient_rms_T_per_m']:.3e} T/m")
    print(f"   Hessian: coils - target RMS {match['hessian_rms_T_per_m2']:.3e} T/m^2 of {match['target_hessian_rms_T_per_m2']:.3e} "
          f"(plasma part {match['plasma_hessian_rms_T_per_m2']:.3e})")
    print(f"   B.n/|B| on the a = {PLASMA_RADIUS} m boundary: max {100 * state['normal']['normal_error_max']:.3f} %, "
          f"RMS {100 * state['normal']['normal_error_rms']:.3f} %  ({'below' if state['normal']['normal_error_max'] < 0.01 else 'ABOVE'} the 1 % target)")
    print(f"   target is a vacuum field   : gradient asymmetry {targets['gradient_asymmetry']:.1e} T/m, "
          f"Hessian asymmetry {targets['hessian_asymmetry']:.1e} T/m^2")
    print(f"   coil length max {float(jnp.max(state['field'].coils.length)):.3f} m, "
          f"coil curvature max {float(jnp.max(state['field'].coils.curvature)):.3f} 1/m")
match = states["optimized"]["match"]
print(f"The plasma field is {match['plasma_field_rms_T'] / match['field_rms_T']:.2f} times the remaining coil mismatch "
      f"on the axis, the plasma gradient {match['plasma_gradient_rms_T_per_m'] / match['gradient_rms_T_per_m']:.2f} times.")
summary = dict(case=CASE, inputs=dict(case, B0=B0, a=PLASMA_RADIUS), config=CONFIG, config_hash=CONFIG_HASH,
               optimization=optimization, vmex_settings=dict(VMEX, mgrid_shape=MGRID_SHAPE, mgrid_margin=MGRID_MARGIN),
               cost_history=cost_history,
               states={n: dict(iota=float(s["near"].iota), etabar=float(s["near"].etabar),
                               r_singularity=float(s["solution"].r_singularity), axis_match=s["match"],
                               boundary=s["normal"]) for n, s in states.items()}, vmex={})


""" VMEX free-boundary benchmark: the coils are fixed and VMEX finds the plasma they hold """
equilibria = {}
if RUN_VMEX:
    radii = {"fitted": VMEX_RADIUS_FRACTION * PLASMA_RADIUS}
    if LARGER_PLASMA_FRACTION:  # Same p2 in the same coils, which were optimized for the fitted plasma only.
        radii["larger"] = LARGER_PLASMA_FRACTION * min(float(s["solution"].r_singularity) for s in states.values())
    for kind, radius in radii.items():
        equilibria[kind], summary["vmex"][kind] = {}, {}
        for name in VMEX_STATES:
            state, directory = states[name], OUTPUT_DIR / f"vmex_{kind}_{name}"
            equilibria[kind][name], reports = {}, {}
            for route in VMEX_ROUTES:
                print(f"\n{'=' * 78}\nVMEX free boundary: {kind} plasma (a = {radius:.4g} m), {name} coils, {route} coil field")
                mgrid = None
                if route == "mgrid":
                    boundary = helpers.flux_surface(state["solution"], radius)
                    directory.mkdir(parents=True, exist_ok=True)
                    mgrid = helpers.write_coil_mgrid(state["field"], boundary, radius, MGRID_SHAPE, NFP,
                                                     directory / "mgrid_coils.nc", margin=MGRID_MARGIN)
                    reports["mgrid_table"] = mgrid[1]
                wout, reports[route] = helpers.solve_free_boundary(state["solution"], state["field"], radius, directory,
                                                                   route, VMEX, mgrid=mgrid)
                equilibria[kind][name][route] = wout
            solved = [w for w in equilibria[kind][name].values() if w is not None]
            if len(solved) == 2:
                reports["mgrid_vs_direct"] = helpers.compare_equilibria(*solved, radius, FLUX_LEVELS, NFP)
            summary["vmex"][kind][name] = reports
            helpers.save_json(OUTPUT_DIR / "summary.json", summary)

    print("\n" + "#" * 78 + "\nVMEX free-boundary equilibria compared with the near-axis expansion")
    # Transforms are converted to one laboratory convention: counterclockwise rotation in (R, Z) per
    # turn in +phi. The raw values differ in sign because the two poloidal angles turn opposite ways.
    print(f"{'plasma':8s} {'coils':10s} {'field':7s} {'iters':>6s} {'iota lab':>10s} {'(near-axis)':>12s} "
          f"{'axis shift/a_b':>15s} {'LCFS shape/a_b':>15s} {'beta':>9s}")
    for kind, cases in summary["vmex"].items():
        for name, reports in cases.items():
            for route in VMEX_ROUTES:
                report = reports[route]
                if not report["converged"]:
                    print(f"{kind:8s} {name:10s} {route:7s}   not converged: {report.get('error')}")
                    continue
                near = report["near_axis"]
                print(f"{kind:8s} {name:10s} {route:7s} {report['iterations']:6d} "
                      f"{report['signed']['vmex_iota_axis_lab']:10.5f} {report['signed']['near_axis_iota_lab']:12.5f} "
                      f"{near['axis_shift_over_benchmark_radius']:15.3e} "
                      f"{near['surfaces'][-1]['shape_rms_over_flux_radius']:15.3e} {report['betatotal']:9.2e}")

""" Field-line tracing: without plasma the coil field is the total field, so its Poincare section is exact """
poincare = None
if TRACE_FIELD_LINES and case["p2"] == 0 and case["I2"] == 0:
    time0 = time()
    poincare, summary["poincare"] = helpers.trace_poincare(states["optimized"]["field"], states["optimized"]["solution"],
                                                           PLASMA_RADIUS, FLUX_LEVELS)
    print(f"\nField lines of the optimized coils traced in {time() - time0:.1f} seconds. Distance from the near-axis surfaces:")
    for row in summary["poincare"]:
        print(f"   s = {row['s']:<7g} phi = {row['plane']:g} period: {row['punctures']:4d} punctures, "
              f"RMS distance / flux radius = {row['rms_over_flux_radius']:.3e}")
    # Two independent checks that need the total field, so only here: the handedness of the transform
    # from a traced line, and the toroidal flux enclosed by the r2 and r3 surfaces of the same label.
    optimized = states["optimized"]
    traced, turns = helpers.traced_lab_iota(optimized["field"], optimized["solution"], 0.25 * PLASMA_RADIUS)
    summary["signed_iota_traced"] = dict(radius_m=0.25 * PLASMA_RADIUS, toroidal_turns=turns, iota_lab=traced,
                                         near_axis_iota_lab=helpers.near_axis_lab_iota(optimized["solution"],
                                                                                       PLASMA_RADIUS)[0])
    print(f"Traced laboratory transform {traced:.5f} over {turns:.1f} turns; near axis "
          f"{summary['signed_iota_traced']['near_axis_iota_lab']:.5f}")
    r2 = near_axis(rc=optimized["near"].rc, zs=optimized["near"].zs, etabar=optimized["near"].etabar, nfp=NFP,
                   nphi=NPHI_DIAGNOSTIC, order="r2", B0=B0, I2=case["I2"], p2=case["p2"], B2c=case["B2c"]).solution
    summary["flux_check"] = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        r = fraction * PLASMA_RADIUS
        row = dict(radius_m=r, **{order: helpers.toroidal_flux(optimized["field"], sol, r) / (np.pi * B0 * r**2) - 1
                                  for order, sol in (("r2", r2), ("r3", optimized["solution"]))})
        summary["flux_check"].append(row)
        print(f"   r = {r:.4f} m: enclosed flux / (pi B0 r^2) - 1 = {row['r2']:+.3e} (r2), {row['r3']:+.3e} (r3)")
helpers.save_json(OUTPUT_DIR / "summary.json", summary)


""" Plots """
label = f"{CASE}:  a = {PLASMA_RADIUS} m,  p$_2$ = {case['p2']:g} Pa/m$^2$,  I$_2$ = {case['I2']:g} T/m"
helpers.plot_optimization(cost_history, {n: s["match"] for n, s in states.items()}, OUTPUT_DIR / "optimization.png", label)
helpers.plot_axis_profiles(states["optimized"]["solution"], states["optimized"]["targets"], states["optimized"]["field"],
                           OUTPUT_DIR / "axis_fields.png", label)
helpers.plot_coils_and_normal_error(states, {n: s["surface"] for n, s in states.items()},
                                    OUTPUT_DIR / "coils_and_normal_field.png", label)
for kind, solved in equilibria.items():
    helpers.plot_cross_sections({n: states[n] for n in VMEX_STATES}, solved, radii[kind], FLUX_LEVELS,
                                OUTPUT_DIR / f"cross_sections_{kind}.png", f"{label}  ({kind} plasma)")
if poincare is not None:
    helpers.plot_poincare(states["optimized"]["solution"], poincare, PLASMA_RADIUS, FLUX_LEVELS, OUTPUT_DIR / "poincare.png", label)
if RUN_VMEX:
    helpers.plot_benchmark_summary(summary["vmex"], OUTPUT_DIR / "vmex_benchmark.png", label)
print(f"\nResults, coils, wout files and figures saved in {OUTPUT_DIR}")
if SHOW_PLOTS:
    plt.show()
