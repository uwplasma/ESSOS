import os
import sys
from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from jax.flatten_util import ravel_pytree

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
from vmec_jax.optimization import extend_boundary_for_max_mode, smooth_min_abs_iota_residual
from vmec_jax.plotting import bmag_from_state_physical, surface_rz_from_state_physical
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.static import build_static


VMEC_JAX_DATA = ROOT.parent / "vmec_jax" / "examples" / "data"
INPUT_FILE = VMEC_JAX_DATA / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "vmec_qh_boundary"

# Boundary parameterization
MAX_MODE = 3
# Optimization settings
MAX_NFEV = jnp.inf
FTOL = 1.0e-8
GTOL = 1.0e-6
XTOL = 1.0e-14
VMEC_SCALING_FACTOR = 1.2

# VMEC / target settings
TARGET_ASPECT = 7.0
TARGET_IOTA = None
TARGET_ABS_MEAN_IOTA_MIN = 0.40
USE_IOTA_FLOOR = True
HELICITY_M = 1
HELICITY_N = -1
OPT_SURFACES = jnp.arange(0.0, 1.01, 0.1)
OPT_QS_NTHETA = 63
OPT_QS_NPHI = 64
PLOT_NTHETA = 160
PLOT_NPHI_FULL = 240
PLOT_NPHI_PERIOD = 120

ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 100.0
QS_WEIGHT = 1.0


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


def loss_aspect_ratio(vmec, target_aspect_ratio=TARGET_ASPECT):
    aspect = jnp.nan_to_num(vmec.aspect_ratio(), nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
    return ASPECT_WEIGHT * (aspect - target_aspect_ratio) ** 2


def loss_abs_mean_iota_floor(vmec, target_abs_mean_iota_min=TARGET_ABS_MEAN_IOTA_MIN):
    mean_abs_iota = jnp.nan_to_num(vmec.mean_abs_iota(), nan=0.0)
    residual = smooth_min_abs_iota_residual(mean_abs_iota, target_abs_mean_iota_min, softness=1.0e-3)
    return IOTA_WEIGHT * residual**2


def loss_qh(vmec, surfaces=OPT_SURFACES, M=HELICITY_M, N=HELICITY_N):
    qs_sumsq = vmec.qs_sumsq(
        surfaces=surfaces,
        helicity_m=M,
        helicity_n=N,
        ntheta=OPT_QS_NTHETA,
        nphi=OPT_QS_NPHI,
    )
    qs_sumsq = jnp.nan_to_num(
        qs_sumsq,
        nan=1.0e12,
        posinf=1.0e12,
        neginf=1.0e12,
    )
    return QS_WEIGHT * qs_sumsq


def loss_total(vmec, surfaces=OPT_SURFACES, M=HELICITY_M, N=HELICITY_N):
    state = vmec.state
    aspect = jnp.nan_to_num(
        vmec.aspect_ratio_from_state(state),
        nan=1.0e6,
        posinf=1.0e6,
        neginf=-1.0e6,
    )
    qs_sumsq = jnp.nan_to_num(
        vmec.qs_sumsq_from_state(
            state,
            surfaces=surfaces,
            helicity_m=M,
            helicity_n=N,
            ntheta=OPT_QS_NTHETA,
            nphi=OPT_QS_NPHI,
        ),
        nan=1.0e12,
        posinf=1.0e12,
        neginf=1.0e12,
    )
    total = ASPECT_WEIGHT * (aspect - TARGET_ASPECT) ** 2 + QS_WEIGHT * qs_sumsq
    if USE_IOTA_FLOOR:
        mean_abs_iota = jnp.nan_to_num(vmec.mean_abs_iota_from_state(state), nan=0.0)
        residual = smooth_min_abs_iota_residual(
            mean_abs_iota,
            TARGET_ABS_MEAN_IOTA_MIN,
            softness=1.0e-3,
        )
        total = total + IOTA_WEIGHT * residual**2
    return total


def _lcfs_data(vmec, *, full_torus: bool):
    theta = np.linspace(0.0, 2.0 * np.pi, PLOT_NTHETA, endpoint=False)
    phi_max = 2.0 * np.pi if full_torus else (2.0 * np.pi / float(vmec.static.cfg.nfp))
    nphi = PLOT_NPHI_FULL if full_torus else PLOT_NPHI_PERIOD
    phi = np.linspace(0.0, phi_max, nphi, endpoint=False)

    state = vmec.state
    s_index = -1
    R, Z = surface_rz_from_state_physical(
        state,
        vmec.static.modes,
        theta=theta,
        phi=phi,
        s_index=s_index,
        nfp=int(vmec.static.cfg.nfp),
    )
    Bmag = np.asarray(
        bmag_from_state_physical(
            state,
            vmec.static,
            vmec.indata,
            theta=theta,
            phi=phi,
            s_index=s_index,
            signgs=int(vmec.signgs),
            phipf=vmec.flux.phipf,
            chipf=vmec.flux.chipf,
            lamscale=vmec.flux.lamscale,
            flux_is_internal=True,
        )
    )
    X = R * np.cos(phi)[None, :]
    Y = R * np.sin(phi)[None, :]
    return {"X": X, "Y": Y, "Z": Z, "Bmag": Bmag, "theta": theta, "phi": phi}


def _set_equal_3d(ax, X, Y, Z):
    x_min, x_max = float(np.min(X)), float(np.max(X))
    y_min, y_max = float(np.min(Y)), float(np.max(Y))
    z_min, z_max = float(np.min(Z)), float(np.max(Z))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)
    ax.set_box_aspect((1, 1, 1))


def save_summary_plots(initial_vmec, final_vmec, output_dir: Path):
    initial_full = _lcfs_data(initial_vmec, full_torus=True)
    final_full = _lcfs_data(final_vmec, full_torus=True)
    initial_period = _lcfs_data(initial_vmec, full_torus=False)
    final_period = _lcfs_data(final_vmec, full_torus=False)

    surf_vmin = min(float(initial_full["Bmag"].min()), float(final_full["Bmag"].min()))
    surf_vmax = max(float(initial_full["Bmag"].max()), float(final_full["Bmag"].max()))
    surf_norm = colors.Normalize(vmin=surf_vmin, vmax=surf_vmax)
    contour_levels = np.linspace(surf_vmin, surf_vmax, 24)
    cmap = cm.viridis

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.06), hspace=0.28, wspace=0.18)

    ax3d_initial = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d_final = fig.add_subplot(gs[0, 1], projection="3d")
    cax3d = fig.add_subplot(gs[0, 2])

    for ax, data, title in (
        (ax3d_initial, initial_full, "Initial boundary"),
        (ax3d_final, final_full, "Optimized boundary"),
    ):
        ax.plot_surface(
            data["X"],
            data["Y"],
            data["Z"],
            facecolors=cmap(surf_norm(data["Bmag"])),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        ax.set_title(title)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.view_init(elev=24, azim=-42)
        _set_equal_3d(ax, data["X"], data["Y"], data["Z"])

    mappable3d = cm.ScalarMappable(norm=surf_norm, cmap=cmap)
    mappable3d.set_array([])
    fig.colorbar(mappable3d, cax=cax3d, label="|B| on LCFS (full torus)")

    ax2d_initial = fig.add_subplot(gs[1, 0])
    ax2d_final = fig.add_subplot(gs[1, 1])
    cax2d = fig.add_subplot(gs[1, 2])

    contour_initial = ax2d_initial.contour(
        initial_period["phi"],
        initial_period["theta"],
        initial_period["Bmag"],
        levels=contour_levels,
        cmap=cmap,
    )
    contour_final = ax2d_final.contour(
        final_period["phi"],
        final_period["theta"],
        final_period["Bmag"],
        levels=contour_levels,
        cmap=cmap,
    )
    fig.colorbar(contour_final, cax=cax2d, label="|B| on LCFS (one field period)")

    for ax, title in (
        (ax2d_initial, "|B| on LCFS - initial"),
        (ax2d_final, "|B| on LCFS - optimized"),
    ):
        ax.set_title(title)
        ax.set_xlabel(r"$\phi$ [rad]")
        ax.set_ylabel(r"$\theta$ [rad]")
        ax.set_xlim(0.0, 2.0 * np.pi / float(initial_vmec.static.cfg.nfp))
        ax.set_ylim(0.0, 2.0 * np.pi)

    fig.suptitle(f"LCFS coloured by |B|, nfp={initial_vmec.static.cfg.nfp}", y=0.98)
    fig.subplots_adjust(top=0.94, hspace=0.28, wspace=0.18)
    fig.savefig(output_dir / "qh_boundary_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def qs_objective_sumsq(vmec):
    return vmec.qs_sumsq(
        surfaces=OPT_SURFACES,
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        ntheta=OPT_QS_NTHETA,
        nphi=OPT_QS_NPHI,
    )

L_total = custom_loss(loss_total, "vmec")
L_total.dependencies = {"vmec": vmec}
starting_dofs, _dofs_to_tree = ravel_pytree({"vmec": vmec})


print(f"Using VMEC input: {INPUT_FILE}")
print(f"Using stage VMEC resolution: mpol={static.cfg.mpol}, ntor={static.cfg.ntor}")
print(f"Target signed iota: {TARGET_IOTA}")
print(f"Target minimum mean |iota|: {TARGET_ABS_MEAN_IOTA_MIN}")
print(f"Use iota floor in optimization: {USE_IOTA_FLOOR}")
print(
    "Optimization QS proxy:",
    f"surfaces={np.array2string(np.asarray(OPT_SURFACES), precision=2)},",
    f"ntheta={OPT_QS_NTHETA}, nphi={OPT_QS_NPHI},",
    f"inner_max_iter={vmec.max_iter}",
)
print("Active optimization parameters:")
print(vmec.parameter_summary())

vmec.reset_solve_counter()
t_start = time()
def _residual_vector_from_vmec(vmec_obj, surfaces=OPT_SURFACES, M=HELICITY_M, N=HELICITY_N):
    state = vmec_obj.state
    parts = []

    aspect = jnp.nan_to_num(
        vmec_obj.aspect_ratio_from_state(state),
        nan=1.0e6,
        posinf=1.0e6,
        neginf=-1.0e6,
    )
    parts.append(jnp.asarray([jnp.sqrt(ASPECT_WEIGHT) * (aspect - TARGET_ASPECT)], dtype=jnp.float64))

    qs = quasisymmetry_ratio_residual_from_state(
        state=state,
        static=vmec_obj.static,
        indata=vmec_obj.indata,
        signgs=int(vmec_obj.signgs),
        surfaces=surfaces,
        helicity_m=int(M),
        helicity_n=int(N),
        ntheta=int(OPT_QS_NTHETA),
        nphi=int(OPT_QS_NPHI),
        flux_local=vmec_obj.flux,
        pressure_local=vmec_obj.pressure,
    )
    qs_residual = jnp.nan_to_num(
        jnp.asarray(qs["residuals1d"], dtype=jnp.float64),
        nan=1.0e6,
        posinf=1.0e6,
        neginf=1.0e6,
    )
    nsurf = int(np.asarray(surfaces).size)
    qs_residual_2d = jnp.reshape(qs_residual, (nsurf, int(OPT_QS_NTHETA) * int(OPT_QS_NPHI)))
    qs_surface_residual = jnp.sqrt(jnp.sum(qs_residual_2d**2, axis=1) + 1.0e-32)
    parts.append(jnp.sqrt(QS_WEIGHT) * qs_surface_residual)

    if USE_IOTA_FLOOR:
        mean_abs_iota = jnp.nan_to_num(vmec_obj.mean_abs_iota_from_state(state), nan=0.0)
        iota_residual = smooth_min_abs_iota_residual(
            mean_abs_iota,
            TARGET_ABS_MEAN_IOTA_MIN,
            softness=1.0e-3,
        )
        parts.append(jnp.asarray([jnp.sqrt(IOTA_WEIGHT) * iota_residual], dtype=jnp.float64))

    return jnp.concatenate(parts)


def _residual_vector(dofs):
    vmec_obj = _dofs_to_tree(jnp.asarray(dofs, dtype=jnp.float64))["vmec"]
    return _residual_vector_from_vmec(vmec_obj)


_lsq_cache = {"x": None, "residual": None, "jac": None}


def _residual_and_jacobian(dofs):
    x = np.asarray(dofs, dtype=float)
    cached_x = _lsq_cache["x"]
    if cached_x is not None and np.array_equal(x, cached_x):
        return _lsq_cache["residual"], _lsq_cache["jac"]

    x_jax = jnp.asarray(x, dtype=jnp.float64)
    residual = np.asarray(_residual_vector(x_jax), dtype=float)
    jac = np.asarray(jax.jacrev(_residual_vector)(x_jax), dtype=float)
    _lsq_cache["x"] = x.copy()
    _lsq_cache["residual"] = residual
    _lsq_cache["jac"] = jac
    return residual, jac


def _lsq_residual(dofs):
    return _residual_and_jacobian(dofs)[0]


def _lsq_jacobian(dofs):
    return _residual_and_jacobian(dofs)[1]


r0_vec, J0 = _residual_and_jacobian(starting_dofs)
r0 = float(L_total(starting_dofs))
g0 = J0.T @ r0_vec
solves_after_initial_value_grad = vmec.get_solve_counter()
print("Initial gradient norm:", float(np.linalg.norm(g0)))
print("Initial gradient:", g0)

res = least_squares(
    _lsq_residual,
    np.asarray(starting_dofs, dtype=float),
    jac=_lsq_jacobian,
    verbose=2,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    max_nfev=None if jnp.isinf(MAX_NFEV) else int(MAX_NFEV),
)
t_end = time()
total_solves_after_opt = vmec.get_solve_counter()

opt_x = jnp.asarray(res.x)
opt_vmec = _dofs_to_tree(opt_x)["vmec"]
r1 = float(L_total(opt_x))
total_solves_after_final_eval = vmec.get_solve_counter()

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", r0)
print("Final loss:", r1)
print("VMEC solves after initial value+grad:", solves_after_initial_value_grad)
print("VMEC solves during optimizer call:", total_solves_after_opt - solves_after_initial_value_grad)
print("VMEC solves for final loss eval:", total_solves_after_final_eval - total_solves_after_opt)
print("Total VMEC solves so far:", total_solves_after_final_eval)
print("Optimizer success:", bool(res.success))
print("Optimizer status:", res.status)
print("Optimizer message:", res.message)
print("Initial QS sumsq:", float(qs_objective_sumsq(vmec)))
print("Final QS sumsq:", float(qs_objective_sumsq(opt_vmec)))
print("Initial aspect:", float(vmec.aspect_ratio()))
print("Final aspect:", float(opt_vmec.aspect_ratio()))
print("Initial mean |iota|:", float(vmec.mean_abs_iota()))
print("Final mean |iota|:", float(opt_vmec.mean_abs_iota()))
print("Initial dof norm:", float(jnp.linalg.norm(vmec.dofs)))
print("Final dof norm:", float(jnp.linalg.norm(opt_vmec.dofs)))
print("Dof change norm:", float(jnp.linalg.norm(opt_vmec.dofs - vmec.dofs)))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
vmec.write_wout(OUTPUT_DIR / "wout_initial_qh_boundary.nc", include_fsq=False, fast_bcovar=True)
try:
    opt_vmec.write_wout(OUTPUT_DIR / "wout_final_qh_boundary.nc", include_fsq=False, fast_bcovar=True)
    save_summary_plots(vmec, opt_vmec, OUTPUT_DIR)
    print(f"Saved summary plot to: {OUTPUT_DIR / 'qh_boundary_summary.png'}")
except Exception as exc:
    print(f"Skipping final wout/plot export because the optimized state could not be re-solved: {exc}")
