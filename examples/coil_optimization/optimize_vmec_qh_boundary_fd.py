from pathlib import Path
from time import time

import jax.numpy as jnp
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from essos.losses import custom_loss
from essos.mhd import VmecJAXBoundary
from vmec_jax.boundary import boundary_input_from_indata
from vmec_jax.config import config_from_indata
from vmec_jax.namelist import read_indata
from vmec_jax.optimization import rebuild_indata_with_resolution, smooth_min_abs_iota_residual
from vmec_jax.static import build_static


ROOT = Path(__file__).resolve().parents[2]
VMEC_JAX_DATA = ROOT.parent / "vmec_jax" / "examples" / "data"
INPUT_FILE = VMEC_JAX_DATA / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "vmec_qh_boundary_fd"

# Boundary parameterization
MAX_MODE = 1
VMEC_MPOL = 5
VMEC_NTOR = 5

# Optimization settings
MAX_NFEV = 40
FTOL = 1.0e-5
GTOL = 1.0e-5
XTOL = 1.0e-12

# VMEC / target settings
TARGET_ASPECT = 7.0
TARGET_IOTA = None
TARGET_ABS_MEAN_IOTA_MIN = 0.40
HELICITY_M = 1
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
QS_NTHETA = 64
QS_NPHI = 64

ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 100.0
QS_WEIGHT = 1.0


indata = read_indata(INPUT_FILE)
indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
cfg = config_from_indata(indata)
static = build_static(cfg)
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
)
vmec.with_mode_selection(
    max_mode=MAX_MODE,
    include=("rc", "zs"),
    fix=("rc00",),
)


def loss_aspect_ratio(vmec, target_aspect_ratio=TARGET_ASPECT):
    aspect = jnp.nan_to_num(vmec.aspect_ratio(), nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
    return ASPECT_WEIGHT * (aspect - target_aspect_ratio)


def loss_abs_mean_iota_floor(vmec, target_abs_mean_iota_min=TARGET_ABS_MEAN_IOTA_MIN):
    mean_abs_iota = jnp.nan_to_num(vmec.mean_abs_iota(), nan=0.0)
    return IOTA_WEIGHT * smooth_min_abs_iota_residual(
        mean_abs_iota,
        target_abs_mean_iota_min,
        softness=1.0e-3,
    )


def loss_qh_fd(vmec, surfaces=SURFACES, M=HELICITY_M, N=HELICITY_N):
    # This uses VmecJAXBoundary.qs_sumsq(), whose custom gradient currently
    # comes from a finite-difference rule in DOF space.
    qs_sumsq = vmec.qs_sumsq(
        surfaces=surfaces,
        helicity_m=M,
        helicity_n=N,
        ntheta=QS_NTHETA,
        nphi=QS_NPHI,
    )
    qs_sumsq = jnp.nan_to_num(qs_sumsq, nan=1.0e12, posinf=1.0e12, neginf=1.0e12)
    return QS_WEIGHT * jnp.sqrt(qs_sumsq + 1.0e-16)


def qs_objective_sumsq(vmec):
    return vmec.qs_sumsq(
        surfaces=SURFACES,
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        ntheta=QS_NTHETA,
        nphi=QS_NPHI,
    )


def _lcfs_data(vmec):
    geom = vmec.get_geom()
    B = np.asarray(vmec.B_on_surface(s_index=-1))
    Bmag = np.linalg.norm(B, axis=-1)
    R = np.asarray(geom.R[-1])
    Z = np.asarray(geom.Z[-1])
    zeta = np.asarray(vmec.static.grid.zeta)
    X = R * np.cos(zeta)[None, :]
    Y = R * np.sin(zeta)[None, :]
    return {"X": X, "Y": Y, "Z": Z, "Bmag": Bmag}


def save_summary_plots(initial_vmec, final_vmec, output_dir: Path):
    initial = _lcfs_data(initial_vmec)
    final = _lcfs_data(final_vmec)

    surf_vmin = min(float(initial["Bmag"].min()), float(final["Bmag"].min()))
    surf_vmax = max(float(initial["Bmag"].max()), float(final["Bmag"].max()))
    surf_norm = colors.Normalize(vmin=surf_vmin, vmax=surf_vmax)
    contour_levels = np.linspace(surf_vmin, surf_vmax, 24)
    cmap = cm.viridis

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.06), hspace=0.28, wspace=0.18)

    ax3d_initial = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d_final = fig.add_subplot(gs[0, 1], projection="3d")
    cax3d = fig.add_subplot(gs[0, 2])

    for ax, data, title in (
        (ax3d_initial, initial, "Initial boundary"),
        (ax3d_final, final, "Optimized boundary"),
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
        ax.view_init(elev=25, azim=-60)
        ax.set_box_aspect((1, 1, 1))

    mappable3d = cm.ScalarMappable(norm=surf_norm, cmap=cmap)
    mappable3d.set_array([])
    fig.colorbar(mappable3d, cax=cax3d, label="|B| on LCFS")

    ax2d_initial = fig.add_subplot(gs[1, 0])
    ax2d_final = fig.add_subplot(gs[1, 1])
    cax2d = fig.add_subplot(gs[1, 2])

    contour_initial = ax2d_initial.contour(initial["Bmag"], levels=contour_levels, cmap=cmap)
    contour_final = ax2d_final.contour(final["Bmag"], levels=contour_levels, cmap=cmap)
    fig.colorbar(contour_final, cax=cax2d, label="|B| on LCFS")

    for ax, title in (
        (ax2d_initial, "|B| on LCFS - initial"),
        (ax2d_final, "|B| on LCFS - optimized"),
    ):
        ax.set_title(title)
        ax.set_xlabel("nfp")
        ax.set_ylabel("Poloidal angle index")

    fig.suptitle("LCFS coloured by |B|, nfp=4", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_dir / "qh_boundary_summary_fd.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


L_aspect_ratio = custom_loss(loss_aspect_ratio, "vmec")
L_iota = custom_loss(loss_abs_mean_iota_floor, "vmec")
L_qh = custom_loss(loss_qh_fd, "vmec")

L_total = L_qh + L_iota + L_aspect_ratio
L_total.dependencies = {"vmec": vmec}


print(f"Using VMEC input: {INPUT_FILE}")
print(f"Using rebuilt VMEC resolution: mpol={VMEC_MPOL}, ntor={VMEC_NTOR}")
print(f"Target signed iota: {TARGET_IOTA}")
print(f"Target minimum mean |iota|: {TARGET_ABS_MEAN_IOTA_MIN}")
print("QS gradient mode: finite-difference custom rule via VmecJAXBoundary.qs_sumsq()")
print("Active optimization parameters:")
print(vmec.parameter_summary())

t_start = time()
res = least_squares(
    L_total,
    L_total.starting_dofs,
    L_total.grad,
    verbose=2,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    max_nfev=MAX_NFEV,
)
t_end = time()

opt_vmec = L_total.dofs_to_pytree(res.x)["vmec"]

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", float(L_total(L_total.starting_dofs)))
print("Final loss:", float(L_total(res.x)))
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
vmec.write_wout(OUTPUT_DIR / "wout_initial_qh_boundary_fd.nc", include_fsq=False, fast_bcovar=True)
opt_vmec.write_wout(OUTPUT_DIR / "wout_final_qh_boundary_fd.nc", include_fsq=False, fast_bcovar=True)
save_summary_plots(vmec, opt_vmec, OUTPUT_DIR)
print(f"Saved summary plot to: {OUTPUT_DIR / 'qh_boundary_summary_fd.png'}")
