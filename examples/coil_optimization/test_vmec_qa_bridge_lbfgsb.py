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
INPUT_FILE = VMEC_JAX_DATA / "input.nfp2_QA_omnigenity"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "vmec_qa_bridge_lbfgsb"

MAX_MODE = 3
MAX_NFEV = 60
GTOL = 1.0e-4
VMEC_SCALING_FACTOR = 1.2

TARGET_ASPECT = 5.0
TARGET_MEAN_IOTA = 0.42
HELICITY_M = 1
HELICITY_N = 0
SURFACES = jnp.arange(0.0, 1.01, 0.1)
QS_NTHETA = 63
QS_NPHI = 64
ASPECT_WEIGHT = 1.0
IOTA_TARGET_WEIGHT = 10000.0
QS_WEIGHT = 1.0
BOUND_MAGNITUDE = 100.0


def _require_jaxopt():
    try:
        import jaxopt  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "jaxopt is not installed in this Python environment. "
            "This test is a real jaxopt.LBFGSB path, so please install `jaxopt` "
            "in the same environment you use to run ESSOS."
        ) from exc
    return jaxopt


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


def _qs_surface_average_from_state(vmec, state):
    return jnp.mean(
        vmec.qs_surface_sumsq_from_state(
            state,
            SURFACES,
            HELICITY_M,
            HELICITY_N,
            ntheta=QS_NTHETA,
            nphi=QS_NPHI,
        )
    )


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


def _scalar_objective_from_vmec(vmec):
    state = vmec.state
    qs = _qs_surface_average_from_state(vmec, state)
    aspect = vmec.aspect_ratio_from_state(state)
    mean_iota = vmec.iota_from_state(state)
    aspect_term = ASPECT_WEIGHT * (aspect - TARGET_ASPECT) ** 2
    iota_term = IOTA_TARGET_WEIGHT * (mean_iota - TARGET_MEAN_IOTA) ** 2
    return QS_WEIGHT * qs + aspect_term + iota_term


def main():
    jaxopt = _require_jaxopt()

    vmec = _build_vmec()
    objective = custom_loss(_scalar_objective_from_vmec, "vmec")
    objective.dependencies = {"vmec": vmec}
    starting_dofs = jnp.asarray(objective.starting_dofs, dtype=jnp.float64)
    bounds = (
        -BOUND_MAGNITUDE * jnp.ones_like(starting_dofs),
        BOUND_MAGNITUDE * jnp.ones_like(starting_dofs),
    )

    print(f"Using VMEC input: {INPUT_FILE}")
    print(f"Using stage VMEC resolution: mpol={vmec.static.cfg.mpol}, ntor={vmec.static.cfg.ntor}")
    print(f"QA bridge jaxopt.LBFGSB test: max_mode={MAX_MODE}, max_nfev={MAX_NFEV}")
    print(f"Target aspect: {TARGET_ASPECT}")
    print(f"Target mean iota: {TARGET_MEAN_IOTA}")
    print(
        "QS surface-averaged objective:",
        f"surfaces={np.array2string(np.asarray(SURFACES), precision=2)},",
        f"ntheta={QS_NTHETA}, nphi={QS_NPHI}",
    )
    print("Active optimization parameters:")
    print(vmec.parameter_summary())

    vmec.reset_solve_counter()
    t0 = time()
    r0, g0 = objective.value_and_grad(starting_dofs)
    solves_after_initial = vmec.get_solve_counter()

    print("Initial gradient norm:", float(np.linalg.norm(np.asarray(g0, dtype=float))))
    print("Initial loss:", float(r0))

    solver = jaxopt.LBFGSB(
        fun=objective.value_and_grad,
        value_and_grad=True,
        has_aux=False,
        tol=GTOL,
        maxiter=int(MAX_NFEV),
    )
    with jax.disable_jit():
        opt_step = solver.run(starting_dofs, bounds=bounds)
    opt_x = jnp.asarray(opt_step.params, dtype=jnp.float64)
    solves_after_opt = vmec.get_solve_counter()

    r1, g1 = objective.value_and_grad(opt_x)
    solves_after_final = vmec.get_solve_counter()
    t1 = time()

    initial_state = vmec.state
    initial_qs_surface_mean = float(_qs_surface_average_from_state(vmec, initial_state))
    initial_qs = float(
        vmec.qs_sumsq_from_state(
            initial_state,
            SURFACES,
            HELICITY_M,
            HELICITY_N,
            ntheta=QS_NTHETA,
            nphi=QS_NPHI,
        )
    )
    initial_aspect = float(vmec.aspect_ratio_from_state(initial_state))
    initial_mean_iota = float(vmec.iota_from_state(initial_state))
    initial_mean_abs_iota = float(vmec.mean_abs_iota_from_state(initial_state))

    opt_vmec = _build_vmec()
    opt_vmec.dofs = opt_x
    opt_state = opt_vmec.state
    final_qs_surface_mean = float(_qs_surface_average_from_state(opt_vmec, opt_state))
    final_qs = float(
        opt_vmec.qs_sumsq_from_state(
            opt_state,
            SURFACES,
            HELICITY_M,
            HELICITY_N,
            ntheta=QS_NTHETA,
            nphi=QS_NPHI,
        )
    )
    final_aspect = float(opt_vmec.aspect_ratio_from_state(opt_state))
    final_mean_iota = float(opt_vmec.iota_from_state(opt_state))
    final_mean_abs_iota = float(opt_vmec.mean_abs_iota_from_state(opt_state))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vmec.write_wout(OUTPUT_DIR / "wout_initial.nc")
    opt_vmec.write_wout(OUTPUT_DIR / "wout_final.nc")

    state_info = getattr(opt_step, "state", None)
    history_rows = [
        {
            "objective": float(r0),
            "qs_surface_mean_objective": initial_qs_surface_mean,
            "qs_objective": initial_qs,
            "aspect": initial_aspect,
            "iota": initial_mean_iota,
            "mean_abs_iota": initial_mean_abs_iota,
            "dof_norm": float(np.linalg.norm(np.asarray(starting_dofs, dtype=float))),
            "x": np.asarray(starting_dofs, dtype=float).tolist(),
        },
        {
            "objective": float(r1),
            "qs_surface_mean_objective": final_qs_surface_mean,
            "qs_objective": final_qs,
            "aspect": final_aspect,
            "iota": final_mean_iota,
            "mean_abs_iota": final_mean_abs_iota,
            "dof_norm": float(np.linalg.norm(np.asarray(opt_x, dtype=float))),
            "x": np.asarray(opt_x, dtype=float).tolist(),
        },
    ]

    history = {
        "input_file": str(INPUT_FILE),
        "optimizer": "jaxopt.LBFGSB",
        "max_mode": int(MAX_MODE),
        "max_nfev": int(MAX_NFEV),
        "gtol": float(GTOL),
        "target_aspect": float(TARGET_ASPECT),
        "target_mean_iota": float(TARGET_MEAN_IOTA),
        "label": "ESSOS QA bridge jaxopt.LBFGSB test",
        "surfaces": np.asarray(SURFACES, dtype=float).tolist(),
        "helicity_m": int(HELICITY_M),
        "helicity_n": int(HELICITY_N),
        "qs_ntheta": int(QS_NTHETA),
        "qs_nphi": int(QS_NPHI),
        "qs_aggregation": "mean_surface_sumsq",
        "initial_loss": float(r0),
        "final_loss": float(r1),
        "initial_qs_surface_mean": initial_qs_surface_mean,
        "final_qs_surface_mean": final_qs_surface_mean,
        "initial_qs": initial_qs,
        "final_qs": final_qs,
        "initial_aspect": initial_aspect,
        "final_aspect": final_aspect,
        "initial_mean_iota": initial_mean_iota,
        "final_mean_iota": final_mean_iota,
        "initial_mean_abs_iota": initial_mean_abs_iota,
        "final_mean_abs_iota": final_mean_abs_iota,
        "initial_grad_norm": float(np.linalg.norm(np.asarray(g0, dtype=float))),
        "final_grad_norm": float(np.linalg.norm(np.asarray(g1, dtype=float))),
        "initial_dof_norm": float(np.linalg.norm(np.asarray(starting_dofs, dtype=float))),
        "final_dof_norm": float(np.linalg.norm(np.asarray(opt_x, dtype=float))),
        "dof_change_norm": float(np.linalg.norm(np.asarray(opt_x - starting_dofs, dtype=float))),
        "solver_state_iter_num": None if state_info is None or not hasattr(state_info, "iter_num") else int(state_info.iter_num),
        "solver_state_error": None if state_info is None or not hasattr(state_info, "error") else float(state_info.error),
        "solver_state_value": None if state_info is None or not hasattr(state_info, "value") else float(state_info.value),
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
                OUTPUT_DIR,
            ),
        ),
        (
            "bmag_contours",
            lambda: plot_bmag_contours(
                OUTPUT_DIR / "wout_initial.nc",
                OUTPUT_DIR / "wout_final.nc",
                OUTPUT_DIR,
            ),
        ),
        (
            "objective_history",
            lambda: plot_objective_history(
                OUTPUT_DIR / "history.json",
                OUTPUT_DIR,
            ),
        ),
    ):
        try:
            plot_paths[name] = str(factory())
        except Exception as exc:  # pragma: no cover - plotting is best effort
            plot_errors[name] = repr(exc)

    print()
    print(f"Optimization took {t1 - t0:.2f} seconds")
    print("Final loss:", float(r1))
    print("VMEC solves after initial value+grad:", int(solves_after_initial))
    print("VMEC solves during optimizer call:", int(solves_after_opt - solves_after_initial))
    print("VMEC solves for final loss eval:", int(solves_after_final - solves_after_opt))
    print("Total VMEC solves so far:", int(solves_after_final))
    print("Initial QS surface-mean sumsq:", initial_qs_surface_mean)
    print("Final QS surface-mean sumsq:", final_qs_surface_mean)
    print("Initial QS sumsq:", initial_qs)
    print("Final QS sumsq:", final_qs)
    print("Initial aspect:", initial_aspect)
    print("Final aspect:", final_aspect)
    print("Initial mean iota:", initial_mean_iota)
    print("Final mean iota:", final_mean_iota)
    print("Saved diagnostics to:", OUTPUT_DIR / "history.json")
    if plot_paths:
        print("Saved plots:")
        for key, value in plot_paths.items():
            print(f"  {key}: {value}")
    if plot_errors:
        print("Plot warnings:")
        for key, value in plot_errors.items():
            print(f"  {key}: {value}")
    if state_info is not None:
        if hasattr(state_info, "iter_num"):
            print("jaxopt iterations:", int(state_info.iter_num))
        if hasattr(state_info, "error"):
            print("jaxopt reported error:", float(state_info.error))


if __name__ == "__main__":
    main()
