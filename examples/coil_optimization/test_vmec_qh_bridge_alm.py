import json
import sys
from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VMEC_JAX_ROOT = ROOT.parent / "vmec_jax"
if str(VMEC_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(VMEC_JAX_ROOT))

from essos.losses import custom_loss
from essos.mhd import VmecJAXBoundary

try:
    import essos.augmented_lagrangian as alm
except ModuleNotFoundError as exc:
    if exc.name == "jaxopt":
        raise ModuleNotFoundError(
            "This ALM example requires `jaxopt` in the active Python environment "
            "because `essos.augmented_lagrangian` uses jaxopt.LBFGSB."
        ) from exc
    raise

from vmec_jax.boundary import boundary_from_indata, boundary_input_from_indata
from vmec_jax.config import config_from_indata
from vmec_jax.namelist import read_indata
from vmec_jax.optimization import extend_boundary_for_max_mode, smooth_min_abs_iota_residual
from vmec_jax.plotting import (
    plot_3d_boundary_comparison,
    plot_bmag_contours,
    plot_objective_history,
)
from vmec_jax.static import build_static


VMEC_JAX_DATA = ROOT.parent / "vmec_jax" / "examples" / "data"
INPUT_FILE = VMEC_JAX_DATA / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "vmec_qh_bridge_alm"

MAX_MODE = 3
MAX_ALM_STEPS = 30
VMEC_SCALING_FACTOR = 1.2

TARGET_ASPECT = 5.0
TARGET_ABS_IOTA_MIN = 0.41
HELICITY_M = 1
HELICITY_N = -1
SURFACES = jnp.arange(0.0, 1.01, 0.1)
QS_NTHETA = 17
QS_NPHI = 18

MODEL_LAGRANGIAN = "Squared"
MODEL_MU = "Mu_Tolerance"
BETA = 2.0
MU_MAX = 1.0e4
ALPHA = 0.99
GAMMA = 1.0e-2
EPSILON = 1.0e-8
ETA_TOL = 1.0e-4
OMEGA_TOL = 1.0e-4


def _build_vmec():
    indata = read_indata(INPUT_FILE)
    cfg = config_from_indata(indata)
    static = build_static(cfg)
    base_boundary = boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
    indata, static, _base_boundary = extend_boundary_for_max_mode(indata, static, base_boundary, MAX_MODE)
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


def _qs_loss_from_vmec(vmec):
    return vmec.qs_sumsq(
        SURFACES,
        HELICITY_M,
        HELICITY_N,
        ntheta=QS_NTHETA,
        nphi=QS_NPHI,
    )


def _aspect_constraint_from_vmec(vmec):
    return vmec.aspect_ratio() - TARGET_ASPECT


def _iota_floor_constraint_from_vmec(vmec):
    return smooth_min_abs_iota_residual(
        vmec.mean_abs_iota(),
        TARGET_ABS_IOTA_MIN,
        softness=1.0e-3,
    )


def main():
    vmec = _build_vmec()

    qs_loss = custom_loss(_qs_loss_from_vmec, "vmec")
    aspect_residual = custom_loss(_aspect_constraint_from_vmec, "vmec")
    iota_floor_residual = custom_loss(_iota_floor_constraint_from_vmec, "vmec")
    for loss_obj in (qs_loss, aspect_residual, iota_floor_residual):
        loss_obj.dependencies = {"vmec": vmec}

    constraints = alm.combine(
        alm.eq(lambda dofs: aspect_residual(dofs), model_lagrangian=MODEL_LAGRANGIAN),
        alm.eq(lambda dofs: iota_floor_residual(dofs), model_lagrangian=MODEL_LAGRANGIAN),
    )
    objective_fun = lambda dofs: qs_loss(dofs)
    alm_solver = alm.ALM_model_jaxopt_lbfgsb(
        constraints=constraints,
        loss=objective_fun,
        model_lagrangian=MODEL_LAGRANGIAN,
        model_mu=MODEL_MU,
        beta=BETA,
        mu_max=MU_MAX,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        eta_tol=ETA_TOL,
        omega_tol=OMEGA_TOL,
    )

    starting_dofs = jnp.asarray(qs_loss.starting_dofs, dtype=jnp.float64)
    lagrange_params = constraints.init(starting_dofs)
    params = (starting_dofs, lagrange_params)

    print(f"Using VMEC input: {INPUT_FILE}")
    print(f"Using stage VMEC resolution: mpol={vmec.static.cfg.mpol}, ntor={vmec.static.cfg.ntor}")
    print(f"QH bridge ALM test: max_mode={MAX_MODE}, max_steps={MAX_ALM_STEPS}")
    print(f"Target aspect: {TARGET_ASPECT}")
    print(f"Target minimum mean |iota|: {TARGET_ABS_IOTA_MIN}")
    print(
        "QS scalar objective:",
        f"surfaces={np.array2string(np.asarray(SURFACES), precision=2)},",
        f"ntheta={QS_NTHETA}, nphi={QS_NPHI}",
    )
    print("Active optimization parameters:")
    print(vmec.parameter_summary())

    vmec.reset_solve_counter()
    t0 = time()
    with jax.disable_jit():
        lag_state, grad, info = alm_solver.init(params)
    solves_after_initial = vmec.get_solve_counter()

    print("Initial QS loss:", float(info[0]))
    print("Initial ALM loss:", float(info[1]))
    print("Initial primal gradient norm:", float(jnp.linalg.norm(grad[0])))
    print("Initial constraint norm:", float(alm.norm_constraints(info[2])))

    history_rows = []

    def _record(tag, main_params, grad_info, info_tuple):
        vmec_local = qs_loss.dofs_to_pytree(jnp.asarray(main_params, dtype=jnp.float64))["vmec"]
        history_rows.append(
            {
                "tag": tag,
                "objective": float(info_tuple[0]),
                "alm_objective": float(info_tuple[1]),
                "constraint_norm": float(alm.norm_constraints(info_tuple[2])),
                "aspect": float(vmec_local.aspect_ratio()),
                "iota": float(vmec_local.iota()),
                "mean_abs_iota": float(vmec_local.mean_abs_iota()),
                "qs_objective": float(
                    vmec_local.qs_sumsq(
                        SURFACES,
                        HELICITY_M,
                        HELICITY_N,
                        ntheta=QS_NTHETA,
                        nphi=QS_NPHI,
                    )
                ),
                "grad_norm": float(jnp.linalg.norm(grad_info[0])),
                "dof_norm": float(jnp.linalg.norm(jnp.asarray(main_params, dtype=jnp.float64))),
                "x": np.asarray(main_params, dtype=float).tolist(),
                "solve_counter": vmec.get_solve_counter(),
            }
        )

    _record("initial", params[0], grad, info)

    step = 0
    while step < MAX_ALM_STEPS and (
        float(jnp.linalg.norm(grad[0])) > OMEGA_TOL
        or float(alm.norm_constraints(info[2])) > ETA_TOL
    ):
        with jax.disable_jit():
            params, lag_state, grad, info = alm_solver.update(params, lag_state, grad, info)
        step += 1
        print(
            f"step={step:02d} "
            f"qs={float(info[0]):.6e} "
            f"alm={float(info[1]):.6e} "
            f"constraint_norm={float(alm.norm_constraints(info[2])):.6e} "
            f"grad_norm={float(jnp.linalg.norm(grad[0])):.6e}"
        )
        _record(f"iter_{step}", params[0], grad, info)

    t1 = time()
    solves_after_opt = vmec.get_solve_counter()

    opt_x = jnp.asarray(params[0], dtype=jnp.float64)
    final_vmec = qs_loss.dofs_to_pytree(opt_x)["vmec"]
    final_loss = float(qs_loss(opt_x))
    final_aspect = float(final_vmec.aspect_ratio())
    final_iota = float(final_vmec.iota())
    final_mean_abs_iota = float(final_vmec.mean_abs_iota())
    final_constraint_norm = float(alm.norm_constraints(info[2]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vmec.write_wout(OUTPUT_DIR / "wout_initial.nc")
    final_vmec.write_wout(OUTPUT_DIR / "wout_final.nc")
    solves_after_final = vmec.get_solve_counter()

    history = {
        "input_file": str(INPUT_FILE),
        "optimizer": "ALM+jaxopt.LBFGSB",
        "max_mode": int(MAX_MODE),
        "max_steps": int(MAX_ALM_STEPS),
        "target_aspect": float(TARGET_ASPECT),
        "target_abs_iota_min": float(TARGET_ABS_IOTA_MIN),
        "iota_abs_min": float(TARGET_ABS_IOTA_MIN),
        "label": "ESSOS QH bridge ALM test",
        "surfaces": np.asarray(SURFACES, dtype=float).tolist(),
        "helicity_m": int(HELICITY_M),
        "helicity_n": int(HELICITY_N),
        "qs_ntheta": int(QS_NTHETA),
        "qs_nphi": int(QS_NPHI),
        "initial_qs": float(qs_loss(starting_dofs)),
        "final_qs": final_loss,
        "initial_aspect": float(vmec.aspect_ratio()),
        "final_aspect": final_aspect,
        "initial_mean_iota": float(vmec.iota()),
        "final_mean_iota": final_iota,
        "initial_mean_abs_iota": float(vmec.mean_abs_iota()),
        "final_mean_abs_iota": final_mean_abs_iota,
        "initial_constraint_norm": float(history_rows[0]["constraint_norm"]),
        "final_constraint_norm": final_constraint_norm,
        "initial_grad_norm": float(history_rows[0]["grad_norm"]),
        "final_grad_norm": float(jnp.linalg.norm(grad[0])),
        "initial_dof_norm": float(jnp.linalg.norm(starting_dofs)),
        "final_dof_norm": float(jnp.linalg.norm(opt_x)),
        "dof_change_norm": float(jnp.linalg.norm(opt_x - starting_dofs)),
        "steps_taken": int(step),
        "stopped_by_tolerances": bool(
            float(jnp.linalg.norm(grad[0])) <= OMEGA_TOL and final_constraint_norm <= ETA_TOL
        ),
        "total_wall_time_s": float(t1 - t0),
        "vmec_solves_after_initial_value_grad": int(solves_after_initial),
        "vmec_solves_during_optimizer_call": int(solves_after_opt - solves_after_initial),
        "vmec_solves_for_final_loss_eval": int(solves_after_final - solves_after_opt),
        "vmec_solves_total": int(solves_after_final),
        "history": history_rows,
        "evaluation_history": history_rows,
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
    print("Final QS loss:", history["final_qs"])
    print("Final aspect:", history["final_aspect"])
    print("Final mean iota:", history["final_mean_iota"])
    print("Final mean |iota|:", history["final_mean_abs_iota"])
    print("Final constraint norm:", history["final_constraint_norm"])
    print("VMEC solves after initial value+grad:", solves_after_initial)
    print("VMEC solves during optimizer call:", solves_after_opt - solves_after_initial)
    print("VMEC solves for final loss eval:", solves_after_final - solves_after_opt)
    print("Total VMEC solves so far:", solves_after_final)
    print("Saved diagnostics to:", OUTPUT_DIR / "history.json")
    print("Saved plots:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")
    if plot_errors:
        print("Plotting warnings:")
        for name, msg in plot_errors.items():
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
