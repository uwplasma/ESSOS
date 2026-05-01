"""ESSOS adapter for virtual-casing VMEC exterior fields."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def load_vmec_state_static_from_files(*, input_path=None, wout_path=None, src_nphi=None, src_ntheta=None):
    """Load ``vmec_jax`` state/static/indata/wout from file paths.

    Passing both ``input_path`` and ``wout_path`` is the preferred path because
    the VMEC input defines grid and profile conventions. If only ``wout_path``
    is supplied, this constructs a minimal static config from ``wout`` metadata;
    that path is intended for quick diagnostics and should be validated against
    an input-backed run before benchmark use.
    """
    if wout_path is None:
        raise ValueError("wout_path is required")
    try:
        import vmec_jax
        from vmec_jax.wout import read_wout, state_from_wout
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("vmec_jax is required for VMEC extender file loading") from exc

    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    if input_path is not None:
        cfg, indata = vmec_jax.load_config(input_path)
    else:
        from vmec_jax.config import VMECConfig
        from vmec_jax.modes import default_grid_sizes

        ntheta = int(src_ntheta) if src_ntheta is not None else 0
        nzeta = int(src_nphi) if src_nphi is not None else 0
        if ntheta <= 0 or nzeta <= 0:
            ntheta, nzeta = default_grid_sizes(
                mpol=int(wout.mpol),
                ntor=int(wout.ntor),
                ntheta=ntheta,
                nzeta=nzeta,
            )
        cfg = VMECConfig(
            mpol=int(wout.mpol),
            ntor=int(wout.ntor),
            ns=int(wout.ns),
            nfp=int(wout.nfp),
            lasym=bool(wout.lasym),
            lthreed=bool(int(wout.ntor) > 0),
            lconm1=True,
            ntheta=int(ntheta),
            nzeta=int(nzeta),
        )
        indata = None

    static = vmec_jax.build_static(cfg)
    return state, static, indata, wout


def _wrap_field_method(method):
    """Wrap an ESSOS single-point field method for batched/SoA use."""

    def wrapped(points):
        pts = jnp.asarray(points)
        if pts.ndim == 1:
            return method(pts)
        if pts.shape[-1] == 3:
            shape = pts.shape
            out = jax.vmap(method)(pts.reshape((-1, 3)))
            return out.reshape(shape)
        if pts.shape[0] == 3:
            trailing = pts.shape[1:]
            out = jax.vmap(method)(pts.reshape((3, -1)).T)
            return out.T.reshape((3,) + trailing)
        raise ValueError(f"points must have shape (..., 3) or (3, ...), got {pts.shape}")

    return wrapped


def _wrap_grad_method(method):
    """Wrap an ESSOS single-point gradient method for batched/SoA use."""

    def wrapped(points):
        pts = jnp.asarray(points)
        if pts.ndim == 1:
            return method(pts)
        if pts.shape[-1] == 3:
            leading = pts.shape[:-1]
            out = jax.vmap(method)(pts.reshape((-1, 3)))
            return out.reshape(leading + (3, 3))
        if pts.shape[0] == 3:
            trailing = pts.shape[1:]
            out = jax.vmap(method)(pts.reshape((3, -1)).T)
            return out.reshape((-1, 3, 3)).transpose((1, 2, 0)).reshape((3, 3) + trailing)
        raise ValueError(f"points must have shape (..., 3) or (3, ...), got {pts.shape}")

    return wrapped


class VmecExtendedField:
    """ESSOS-compatible wrapper around ``VirtualCasingExteriorField``."""

    def __init__(self, vc_field):
        self.vc_field = vc_field

    def sqrtg(self, points):
        return 1.0

    def B(self, points):
        return self.vc_field.B_xyz(points)

    def B_covariant(self, points):
        return self.B(points)

    def B_contravariant(self, points):
        return self.B(points)

    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points), axis=-1 if jnp.asarray(points).shape[-1] == 3 else 0)

    def dB_by_dX(self, points):
        if hasattr(self.vc_field, "gradB_xyz"):
            return self.vc_field.gradB_xyz(points)
        return jax.jacfwd(self.B)(points)

    def grad_B_covariant(self, points):
        return self.dB_by_dX(points)

    def curl_B(self, points):
        grad_B = self.grad_B_covariant(points)
        return jnp.array(
            [
                grad_B[2, 1] - grad_B[1, 2],
                grad_B[0, 2] - grad_B[2, 0],
                grad_B[1, 0] - grad_B[0, 1],
            ]
        ) / self.sqrtg(points)

    def to_xyz(self, points):
        return points


def build_vmec_extended_field(
    *,
    vmec_state=None,
    vmec_static=None,
    indata=None,
    wout=None,
    vmec_input=None,
    wout_path=None,
    coils=None,
    coil_field=None,
    config=None,
):
    """Return an ESSOS-compatible VMEC exterior field.

    The differentiable path is ``vmec_state`` + ``vmec_static`` + ``indata``.
    A standalone ``wout`` path is intentionally not hidden here yet because
    constructing a matching ``vmec_jax`` static grid from only ``wout`` still
    needs a convention-tested public helper upstream.
    """
    if (vmec_state is None or vmec_static is None) and wout_path is not None:
        vmec_state, vmec_static, indata_loaded, wout_loaded = load_vmec_state_static_from_files(
            input_path=vmec_input,
            wout_path=wout_path,
            src_nphi=getattr(config, "src_nphi", None),
            src_ntheta=getattr(config, "src_ntheta", None),
        )
        if indata is None:
            indata = indata_loaded
        if wout is None:
            wout = wout_loaded

    if vmec_state is None or vmec_static is None:
        if wout is not None:
            raise NotImplementedError(
                "wout object construction requires vmec_state/vmec_static. "
                "Pass wout_path instead for file-based loading."
            )
        raise ValueError("vmec_state and vmec_static are required")

    if coil_field is None and coils is not None:
        from essos.fields import BiotSavart

        coil_field = BiotSavart(coils)

    external_B_fn = None
    external_gradB_fn = None
    if coil_field is not None:
        external_B_fn = _wrap_field_method(coil_field.B)
        if hasattr(coil_field, "dB_by_dX"):
            external_gradB_fn = _wrap_grad_method(coil_field.dB_by_dX)

    from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, surface_field_from_vmec_jax

    surface_data = surface_field_from_vmec_jax(vmec_state, vmec_static, indata=indata, wout=wout)
    vc_field = VirtualCasingExteriorField(
        surface_data,
        config or ExteriorFieldConfig(),
        external_B_fn=external_B_fn,
        external_gradB_fn=external_gradB_fn,
    )
    return VmecExtendedField(vc_field)
