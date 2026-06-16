import sys
from pathlib import Path

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
from vmec_jax.static import build_static


VMEC_JAX_DATA = ROOT.parent / "vmec_jax" / "examples" / "data"
INPUT_FILE = VMEC_JAX_DATA / "input.nfp4_QH_warm_start"

MAX_MODE = 1
VMEC_SCALING_FACTOR = 0.5
QS_SURFACES = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
QS_M = 1
QS_N = -1
QS_NTHETA = 9
QS_NPHI = 10
EPS = 1.0e-5
REL_TOL = 2.0e-3
ABS_TOL = 2.0e-5


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


def _make_projection_weights(size: int) -> jnp.ndarray:
    if size <= 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    weights = jnp.linspace(0.5, 1.5, size, dtype=jnp.float64)
    return weights / jnp.linalg.norm(weights)


def _direction(vmec: VmecJAXBoundary) -> jnp.ndarray:
    x0 = np.asarray(vmec.dofs, dtype=float)
    direction = np.linspace(1.0, 2.0, x0.size, dtype=float)
    direction /= max(np.linalg.norm(direction), 1.0e-15)
    return jnp.asarray(direction, dtype=jnp.float64)


def _scalarized_metric_specs(vmec: VmecJAXBoundary):
    iota_profile_size = int(np.asarray(vmec.iota(profile=True)).size)
    qs_surface_size = int(np.asarray(vmec.qs_surface_sumsq(QS_SURFACES, QS_M, QS_N, ntheta=QS_NTHETA, nphi=QS_NPHI)).size)
    triple_product_size = int(
        np.asarray(
            vmec.triple_product_metric(
                QS_SURFACES,
                QS_M,
                QS_N,
                ntheta=QS_NTHETA,
                nphi=QS_NPHI,
            )
        ).size
    )
    volume_profile_size = int(np.asarray(vmec.volume()).size)

    iota_profile_weights = _make_projection_weights(iota_profile_size)
    qs_surface_weights = _make_projection_weights(qs_surface_size)
    triple_product_weights = _make_projection_weights(triple_product_size)
    volume_profile_weights = _make_projection_weights(volume_profile_size)

    return [
        ("aspect_ratio", lambda vm: vm.aspect_ratio()),
        ("mean_iota", lambda vm: vm.iota()),
        ("mean_abs_iota", lambda vm: vm.mean_abs_iota()),
        (
            "iota_profile_projection",
            lambda vm: jnp.vdot(iota_profile_weights, vm.iota(profile=True)),
        ),
        (
            "qs_sumsq",
            lambda vm: vm.qs_sumsq(
                QS_SURFACES,
                QS_M,
                QS_N,
                ntheta=QS_NTHETA,
                nphi=QS_NPHI,
            ),
        ),
        (
            "qs_surface_sumsq_projection",
            lambda vm: jnp.vdot(
                qs_surface_weights,
                vm.qs_surface_sumsq(
                    QS_SURFACES,
                    QS_M,
                    QS_N,
                    ntheta=QS_NTHETA,
                    nphi=QS_NPHI,
                ),
            ),
        ),
        (
            "triple_product_projection",
            lambda vm: jnp.vdot(
                triple_product_weights,
                vm.triple_product_metric(
                    QS_SURFACES,
                    QS_M,
                    QS_N,
                    ntheta=QS_NTHETA,
                    nphi=QS_NPHI,
                ),
            ),
        ),
        (
            "volume_profile_projection",
            lambda vm: jnp.vdot(volume_profile_weights, vm.volume()),
        ),
        ("volume_edge", lambda vm: vm.volume(s_index=-1)),
        ("volume_averaged_B", lambda vm: vm.volume_averaged_B()),
        ("volume_averaged_beta", lambda vm: vm.volume_averaged_beta()),
        ("vacuum_well", lambda vm: vm.vacuum_well()),
    ]


def _validate_metric(name: str, fun, vmec: VmecJAXBoundary, direction: jnp.ndarray):
    loss = custom_loss(fun, "vmec")
    loss.dependencies = {"vmec": vmec}
    x0 = jnp.asarray(loss.starting_dofs, dtype=jnp.float64)
    grad = np.asarray(loss.grad(x0), dtype=float)
    ad_dir = float(np.dot(grad, np.asarray(direction, dtype=float)))
    f_plus = float(loss(x0 + EPS * direction))
    f_minus = float(loss(x0 - EPS * direction))
    fd_dir = (f_plus - f_minus) / (2.0 * EPS)
    abs_err = abs(ad_dir - fd_dir)
    rel_err = abs_err / max(abs(fd_dir), abs(ad_dir), 1.0)
    ok = bool(abs_err <= ABS_TOL or rel_err <= REL_TOL)
    return {
        "name": name,
        "value": float(loss(x0)),
        "ad_dir": ad_dir,
        "fd_dir": fd_dir,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "ok": ok,
    }


def _selected_metric_names():
    if len(sys.argv) <= 1:
        return None
    return {arg.strip() for arg in sys.argv[1:] if arg.strip()}


def main():
    vmec = _build_vmec()
    if len(vmec.dofs) > 0:
        seed = np.linspace(-3.0e-3, 3.0e-3, len(vmec.dofs), dtype=float)
        vmec.dofs = jnp.asarray(seed, dtype=jnp.float64)

    direction = _direction(vmec)
    selected = _selected_metric_names()
    specs = _scalarized_metric_specs(vmec)
    if selected is not None:
        specs = [(name, fun) for name, fun in specs if name in selected]
        missing = sorted(selected.difference({name for name, _ in specs}))
        if missing:
            raise SystemExit(f"Unknown metric name(s): {', '.join(missing)}")
    results = [
        _validate_metric(name, fun, vmec, direction)
        for name, fun in specs
    ]

    print(
        f"{'metric':<32} {'value':>12} {'ad_dir':>12} {'fd_dir':>12} {'abs_err':>12} {'rel_err':>12} {'ok':>6}"
    )
    print("-" * 106)
    failed = []
    for row in results:
        print(
            f"{row['name']:<32} "
            f"{row['value']:>12.4e} "
            f"{row['ad_dir']:>12.4e} "
            f"{row['fd_dir']:>12.4e} "
            f"{row['abs_err']:>12.4e} "
            f"{row['rel_err']:>12.4e} "
            f"{str(row['ok']):>6}"
        )
        if not row["ok"]:
            failed.append(row["name"])

    if failed:
        print("\nFAILED:", ", ".join(failed))
        raise SystemExit(1)
    print("\nAll metric derivative checks passed.")


if __name__ == "__main__":
    main()
