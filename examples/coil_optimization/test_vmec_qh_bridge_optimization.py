import json
import sys
from pathlib import Path
from time import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VMEC_JAX_ROOT = ROOT.parent / "vmec_jax"
if str(VMEC_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(VMEC_JAX_ROOT))

from essos.losses import custom_loss
from essos.mhd import VmecJAXBoundary
from vmec_jax.boundary import boundary_from_indata, boundary_input_from_indata
from vmec_jax.config import config_from_indata
from vmec_jax.namelist import read_indata
from vmec_jax.optimization import extend_boundary_for_max_mode
from vmec_jax.plotting import (
    plot_3d_boundary_comparison,
    plot_bmag_contours,
    plot_objective_history,
)
from vmec_jax.static import build_static


VMEC_JAX_DATA = ROOT.parent / "vmec_jax" / "examples" / "data"
INPUT_FILE = VMEC_JAX_DATA / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "vmec_qh_bridge_test"

MAX_MODE = 3
MAX_NFEV = 30
FTOL = 1.0e-3
GTOL = 1.0e-3
XTOL = 1.0e-3
VMEC_SCALING_FACTOR = 1.2

TARGET_ASPECT = 5.0
# Best available native vmec_jax QH ESS continuation reference value found locally.
# Update this if you want to mirror a different recorded native run.
TARGET_MEAN_IOTA = -1.2524066267038414
HELICITY_M = 1
HELICITY_N = -1
SURFACES = jnp.arange(0.0, 1.01, 0.1)
QS_NTHETA = 63
QS_NPHI = 64
ASPECT_WEIGHT = 1.0
IOTA_TARGET_WEIGHT = 40000.0
QS_WEIGHT = 1.0


def _build_vmec():
    indata = read_indata(INPUT_FILE)
    cfg = config_from_indata(indata)
    static = build_static(cfg)
    base_boundary = boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
    indata, static, base_boundary = extend_boundary_for_max_mode(indata, static, base_boundary, MAX_MODE)
    base_boundary_input = boundary_input_from_indata(indata, static.modes)

    vmec = VmecJAXBoundary(
        static=static,
        indata=indata,
        base_boundary_input=base_boundary_input,
        input_path=INPUT_FILE,
        solver="lbfgs",
        max_iter=25,
        vmec_project=True,
        verbose=False,
        scaling_type=jnp.inf,
        scaling_factor=VMEC_SCALING_FACTOR,
    )
    vmec.with_mode_selection(
        max_mode=MAX_MODE,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    return vmec


def _residual_vector_from_vmec(vmec):
    state = vmec.state
    parts = [
        vmec.aspect_ratio_residual_from_state(
            state,
            TARGET_ASPECT,
            weight=ASPECT_WEIGHT,
        ),
        vmec.mean_iota_target_residual_from_state(
            state,
            TARGET_MEAN_IOTA,
            weight=IOTA_TARGET_WEIGHT,
        ),
        vmec.qs_residual_block_from_state(
            state,
            SURFACES,
            HELICITY_M,
            HELICITY_N,
            ntheta=QS_NTHETA,
            nphi=QS_NPHI,
            weight=QS_WEIGHT,
        ),
    ]
    return jnp.concatenate(parts)


def _objective_from_vmec(vmec):
    residual = _residual_vector_from_vmec(vmec)
    return jnp.dot(residual, residual)


def main():
    vmec = _build_vmec()
    L_total = custom_loss(_objective_from_vmec, "vmec")
    L_total.dependencies = {"vmec": vmec}

    def _residuals_from_state(state):
        parts = [
            vmec.aspect_ratio_residual_from_state(
                state,
                TARGET_ASPECT,
                weight=ASPECT_WEIGHT,
            ),
            vmec.mean_iota_target_residual_from_state(
                state,
                TARGET_MEAN_IOTA,
                weight=IOTA_TARGET_WEIGHT,
            ),
            vmec.qs_residual_block_from_state(
                state,
                SURFACES,
                HELICITY_M,
                HELICITY_N,
                ntheta=QS_NTHETA,
                nphi=QS_NPHI,
                weight=QS_WEIGHT,
            ),
        ]
        return jnp.concatenate(parts)

    bridge = vmec.build_exact_residual_bridge(_residuals_from_state)
    starting_dofs = jnp.asarray(bridge.initial_dofs, dtype=jnp.float64)

    print(f"Using VMEC input: {INPUT_FILE}")
    print(f"Using stage VMEC resolution: mpol={vmec.static.cfg.mpol}, ntor={vmec.static.cfg.ntor}")
    print(f"QH bridge test: max_mode={MAX_MODE}, max_nfev={MAX_NFEV}")
    print(f"Target mean iota: {TARGET_MEAN_IOTA}")
    print(
        "QS residual block:",
        f"surfaces={np.array2string(np.asarray(SURFACES), precision=2)},",
        f"ntheta={QS_NTHETA}, nphi={QS_NPHI}",
    )
    print("Active optimization parameters:")
    print(vmec.parameter_summary())

    eval_history = []
    cache = {"x": None, "residual": None, "jac": None, "state": None}

    def _record_evaluation(x, state, residual):
        eval_history.append(
            {
                "objective": float(np.dot(residual, residual)),
                "qs_objective": float(
                    vmec.qs_sumsq_from_state(
                        state,
                        SURFACES,
                        HELICITY_M,
                        HELICITY_N,
                        ntheta=QS_NTHETA,
                        nphi=QS_NPHI,
                    )
                ),
                "aspect": float(vmec.aspect_ratio_from_state(state)),
                "iota": float(vmec.iota_from_state(state)),
                "mean_abs_iota": float(vmec.mean_abs_iota_from_state(state)),
                "dof_norm": float(np.linalg.norm(np.asarray(x, dtype=float))),
                "x": np.asarray(x, dtype=float).tolist(),
                "solve_counter": vmec.get_solve_counter(),
            }
        )

    def _residual_and_jacobian(dofs):
        x = np.asarray(dofs, dtype=float)
        cached_x = cache["x"]
        if cached_x is not None and np.array_equal(x, cached_x):
            return cache["residual"], cache["jac"]

        residual, jac, _objective, _gradient, state = bridge.residual_jacobian_objective_gradient_state(x)
        residual = np.asarray(residual, dtype=float)
        jac = np.asarray(jac, dtype=float)
        cache["x"] = x.copy()
        cache["residual"] = residual
        cache["jac"] = jac
        cache["state"] = state
        _record_evaluation(x, state, residual)
        return residual, jac

    def _lsq_residual(dofs):
        return _residual_and_jacobian(dofs)[0]

    def _lsq_jacobian(dofs):
        return _residual_and_jacobian(dofs)[1]

    vmec.reset_solve_counter()
    t0 = time()
    r0_vec, J0 = _residual_and_jacobian(starting_dofs)
    _r0_cached, _J0_cached, r0, g0, _state0 = bridge.residual_jacobian_objective_gradient_state(starting_dofs)
    solves_after_initial = vmec.get_solve_counter()

    print("Initial residual size:", int(r0_vec.size))
    print("Initial gradient norm:", float(np.linalg.norm(g0)))
    print("Initial loss:", r0)

    res = least_squares(
        _lsq_residual,
        np.asarray(starting_dofs, dtype=float),
        jac=_lsq_jacobian,
        verbose=2,
        ftol=FTOL,
        gtol=GTOL,
        xtol=XTOL,
        max_nfev=int(MAX_NFEV),
    )
    t1 = time()
    solves_after_opt = vmec.get_solve_counter()

    opt_x = jnp.asarray(res.x, dtype=jnp.float64)
    _r1_cached, _J1_cached, r1, _g1_cached, opt_state = bridge.residual_jacobian_objective_gradient_state(opt_x)
    opt_vmec = _build_vmec()
    opt_vmec.dofs = opt_x
    solves_after_final = vmec.get_solve_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge.save_wout(OUTPUT_DIR / "wout_initial.nc", starting_dofs)
    bridge.save_wout(OUTPUT_DIR / "wout_final.nc", opt_x)

    history = {
        "input_file": str(INPUT_FILE),
        "max_mode": int(MAX_MODE),
        "max_nfev": int(MAX_NFEV),
        "ftol": float(FTOL),
        "gtol": float(GTOL),
        "xtol": float(XTOL),
        "target_aspect": float(TARGET_ASPECT),
        "target_mean_iota": float(TARGET_MEAN_IOTA),
        "label": "ESSOS QH bridge test",
        "surfaces": np.asarray(SURFACES, dtype=float).tolist(),
        "helicity_m": int(HELICITY_M),
        "helicity_n": int(HELICITY_N),
        "qs_ntheta": int(QS_NTHETA),
        "qs_nphi": int(QS_NPHI),
        "initial_loss": r0,
        "final_loss": r1,
        "initial_qs": float(vmec.qs_sumsq(SURFACES, HELICITY_M, HELICITY_N, ntheta=QS_NTHETA, nphi=QS_NPHI)),
        "final_qs": float(opt_vmec.qs_sumsq_from_state(opt_state, SURFACES, HELICITY_M, HELICITY_N, ntheta=QS_NTHETA, nphi=QS_NPHI)),
        "initial_aspect": float(vmec.aspect_ratio()),
        "final_aspect": float(opt_vmec.aspect_ratio_from_state(opt_state)),
        "initial_mean_iota": float(vmec.iota()),
        "final_mean_iota": float(opt_vmec.iota_from_state(opt_state)),
        "initial_mean_abs_iota": float(vmec.mean_abs_iota()),
        "final_mean_abs_iota": float(opt_vmec.mean_abs_iota_from_state(opt_state)),
        "initial_dof_norm": float(jnp.linalg.norm(vmec.dofs)),
        "final_dof_norm": float(jnp.linalg.norm(opt_vmec.dofs)),
        "dof_change_norm": float(jnp.linalg.norm(opt_vmec.dofs - vmec.dofs)),
        "optimizer_success": bool(res.success),
        "optimizer_status": int(res.status),
        "optimizer_message": str(res.message),
        "nfev": None if res.nfev is None else int(res.nfev),
        "njev": None if res.njev is None else int(res.njev),
        "total_wall_time_s": float(t1 - t0),
        "vmec_solves_after_initial_value_grad": int(solves_after_initial),
        "vmec_solves_during_optimizer_call": int(solves_after_opt - solves_after_initial),
        "vmec_solves_for_final_loss_eval": int(solves_after_final - solves_after_opt),
        "vmec_solves_total": int(solves_after_final),
        "bridge_profile": bridge.profile_dump(),
        "history": eval_history,
        "evaluation_history": eval_history,
    }
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2))

    plot_paths = {}
    plot_errors = {}
    for name, factory in (
        (
            "boundary_comparison",
            lambda: plot_3d_boundary_comparison(
                OUTPUT_DIR / "wout_initial.nc",
                OUTPUT_DIR / "wout_final.nc",
                outdir=OUTPUT_DIR,
            ),
        ),
        (
            "bmag_contours",
            lambda: plot_bmag_contours(
                OUTPUT_DIR / "wout_initial.nc",
                OUTPUT_DIR / "wout_final.nc",
                outdir=OUTPUT_DIR,
            ),
        ),
        (
            "objective_history",
            lambda: plot_objective_history(
                OUTPUT_DIR / "history.json",
                outdir=OUTPUT_DIR,
            ),
        ),
    ):
        try:
            plot_paths[name] = factory()
        except Exception as exc:
            plot_errors[name] = str(exc)

    print(f"\nOptimization took {t1 - t0:.2f} seconds")
    print("Final loss:", r1)
    print("VMEC solves after initial value+grad:", solves_after_initial)
    print("VMEC solves during optimizer call:", solves_after_opt - solves_after_initial)
    print("VMEC solves for final loss eval:", solves_after_final - solves_after_opt)
    print("Total VMEC solves so far:", solves_after_final)
    print("Initial QS sumsq:", history["initial_qs"])
    print("Final QS sumsq:", history["final_qs"])
    print("Initial aspect:", history["initial_aspect"])
    print("Final aspect:", history["final_aspect"])
    print("Initial mean iota:", history["initial_mean_iota"])
    print("Final mean iota:", history["final_mean_iota"])
    print("Initial mean |iota|:", history["initial_mean_abs_iota"])
    print("Final mean |iota|:", history["final_mean_abs_iota"])
    print("Saved diagnostics to:", OUTPUT_DIR / "history.json")
    print("Bridge profile summary:")
    for name, rec in sorted(history["bridge_profile"].items()):
        print(
            f"  {name}: "
            f"count={rec['count']} "
            f"wall={rec['wall_time_s']:.3f}s "
            f"mean={rec['mean_wall_time_s']:.3f}s"
        )
    print("Saved plots:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")
    if plot_errors:
        print("Plotting warnings:")
        for name, msg in plot_errors.items():
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
