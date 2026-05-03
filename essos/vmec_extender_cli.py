"""Command line workflows for VMEC exterior fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import jax.numpy as jnp

from .vmec_extender import build_vmec_extended_field


def parse_range(spec: str) -> np.ndarray:
    """Parse ``start:stop:n`` or comma-separated numeric values."""
    spec = str(spec).strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("ranges must be start:stop:n")
        start, stop = float(parts[0]), float(parts[1])
        n = int(parts[2])
        if n <= 0:
            raise argparse.ArgumentTypeError("range count must be positive")
        return np.linspace(start, stop, n)
    values = [float(x) for x in spec.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected numeric values")
    return np.asarray(values, dtype=float)


def _config_from_args(args):
    from virtual_casing_jax import ExteriorFieldConfig

    return ExteriorFieldConfig(
        digits=int(args.digits),
        src_nphi=int(args.src_nphi),
        src_ntheta=int(args.src_ntheta),
        levels=((int(args.src_nphi), int(args.src_ntheta)),),
        chunk_size=_parse_chunk_arg(args.chunk_size),
        target_chunk_size=_parse_chunk_arg(args.target_chunk_size),
        dtype=args.dtype,
    )


def _parse_chunk_arg(value):
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    if value is None:
        return "auto"
    return int(value)


def _levels_for_json(levels):
    return [[int(nt), int(np)] for nt, np in tuple(levels)]


def _load_coil_field(path):
    if path is None:
        return None
    from essos.coils import Coils_from_json
    from essos.fields import BiotSavart

    return BiotSavart(Coils_from_json(str(path)))


def load_seed_points(path) -> np.ndarray:
    """Load Cartesian or cylindrical seed points from JSON."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if "xyz" in data:
            pts = np.asarray(data["xyz"], dtype=float)
        elif "R_phi_Z" in data:
            from virtual_casing_jax import cyl_to_xyz

            pts = np.asarray(cyl_to_xyz(jnp.asarray(data["R_phi_Z"], dtype=float)), dtype=float)
        else:
            raise ValueError("seed JSON dict must contain 'xyz' or 'R_phi_Z'")
    else:
        pts = np.asarray(data, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"seed points must have shape (n, 3), got {pts.shape}")
    return pts


def trajectories_xyz_to_rphiz(trajectories_xyz) -> np.ndarray:
    """Convert Cartesian trajectories to ``(R, unwrapped phi, Z)`` arrays."""
    xyz = np.asarray(trajectories_xyz, dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"trajectories_xyz must have shape (nlines, nsteps, 3), got {xyz.shape}")
    R = np.hypot(xyz[..., 0], xyz[..., 1])
    phi = np.unwrap(np.arctan2(xyz[..., 1], xyz[..., 0]), axis=1)
    return np.stack((R, phi, xyz[..., 2]), axis=-1)


def _arc_lengths(trajectories_xyz) -> np.ndarray:
    xyz = np.asarray(trajectories_xyz, dtype=float)
    if xyz.shape[1] < 2:
        return np.zeros(xyz.shape[0], dtype=float)
    return np.sum(np.linalg.norm(np.diff(xyz, axis=1), axis=-1), axis=1)


def _flatten_section_major(samples: np.ndarray) -> np.ndarray:
    return np.transpose(samples, (1, 0, 2)).reshape((-1, 3))


def fieldline_samples_from_xyz_trajectories(
    trajectories_xyz,
    *,
    sample_stride: int | None = None,
    sample_phi_period: float | None = None,
    sample_nsections: int | None = None,
) -> dict[str, np.ndarray]:
    """Return comparator-ready field-line samples from traced Cartesian paths.

    The flattened point order matches STELLOPT/FIELDLINES HDF5 loading in
    ``virtual_casing_jax``: section index varies slowest and line index varies
    fastest.
    """
    rphiz = trajectories_xyz_to_rphiz(trajectories_xyz)
    nlines, nsteps, _ = rphiz.shape

    if sample_phi_period is not None:
        period = float(sample_phi_period)
        if period <= 0.0:
            raise ValueError("sample_phi_period must be positive")
        phi = rphiz[..., 1]
        lower = max(float(np.min(line_phi)) for line_phi in phi)
        upper = min(float(np.max(line_phi)) for line_phi in phi)
        start = np.ceil(lower / period) * period
        if sample_nsections is None:
            if start > upper:
                raise ValueError("no common phi sections are covered by all field lines")
            targets = start + period * np.arange(int(np.floor((upper - start) / period)) + 1)
        else:
            if sample_nsections <= 0:
                raise ValueError("sample_nsections must be positive")
            targets = start + period * np.arange(int(sample_nsections))
            if targets[-1] > upper + 1e-12:
                raise ValueError("requested phi sections exceed the common traced phi interval")
        samples = np.empty((nlines, len(targets), 3), dtype=float)
        for i in range(nlines):
            order = np.argsort(phi[i])
            phi_i = phi[i, order]
            R_i = rphiz[i, order, 0]
            Z_i = rphiz[i, order, 2]
            samples[i, :, 0] = np.interp(targets, phi_i, R_i)
            samples[i, :, 1] = targets
            samples[i, :, 2] = np.interp(targets, phi_i, Z_i)
        section_phi = np.broadcast_to(targets[:, None], (len(targets), nlines)).reshape(-1)
    else:
        stride = 1 if sample_stride is None else int(sample_stride)
        if stride <= 0:
            raise ValueError("sample_stride must be positive")
        indices = np.arange(0, nsteps, stride, dtype=int)
        samples = rphiz[:, indices, :]
        section_phi = np.transpose(samples[:, :, 1], (1, 0)).reshape(-1)

    nsections = samples.shape[1]
    return {
        "poincare_rphiz": _flatten_section_major(samples),
        "line_id": np.broadcast_to(np.arange(nlines, dtype=float), (nsections, nlines)).reshape(-1),
        "section_phi": section_phi,
        "connection_lengths": _arc_lengths(trajectories_xyz),
    }


def write_fieldline_samples_npz(
    path,
    trajectories_xyz,
    *,
    sample_stride: int | None = None,
    sample_phi_period: float | None = None,
    sample_nsections: int | None = None,
    metadata: dict | None = None,
) -> dict[str, np.ndarray]:
    """Write field-line samples in the external benchmark comparator schema."""
    samples = fieldline_samples_from_xyz_trajectories(
        trajectories_xyz,
        sample_stride=sample_stride,
        sample_phi_period=sample_phi_period,
        sample_nsections=sample_nsections,
    )
    payload = dict(samples)
    if metadata is not None:
        payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return samples


def _build_field_from_args(args):
    return build_vmec_extended_field(
        vmec_input=args.input,
        wout_path=args.wout,
        coil_field=_load_coil_field(args.coils),
        config=_config_from_args(args),
    )


def _surface_metrics(surface_data):
    Bn = jnp.sum(surface_data.B_total * surface_data.normal, axis=0)
    absB = jnp.linalg.norm(surface_data.B_total, axis=0)
    scale = jnp.maximum(jnp.sqrt(jnp.mean(absB * absB)), jnp.asarray(1e-300, dtype=absB.dtype))
    return {
        "B_dot_n_mean": float(jnp.mean(Bn)),
        "B_dot_n_rms_normalized": float(jnp.sqrt(jnp.mean(Bn * Bn)) / scale),
        "B_dot_n_max_normalized": float(jnp.max(jnp.abs(Bn)) / scale),
    }


def _branch_identity_metrics(vc_field):
    surface = vc_field.surface_data
    nphi = int(surface.gamma.shape[1])
    ntheta = int(surface.gamma.shape[2])
    nfp = int(surface.nfp)
    min_nt = max(nfp * nphi, 13)
    min_np = max(ntheta, 13)
    quad_nt = ((min_nt + nfp * nphi - 1) // (nfp * nphi)) * (nfp * nphi)
    quad_np = ((min_np + ntheta - 1) // ntheta) * ntheta
    try:
        Bint = vc_field._vc.compute_internal_B(
            vc_field.B_total,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=int(vc_field.config.digits),
            chunk_size=vc_field.config.chunk_size,
            target_chunk_size=vc_field.config.target_chunk_size,
        )
        Bext = vc_field._vc.compute_external_B(
            vc_field.B_total,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=int(vc_field.config.digits),
            chunk_size=vc_field.config.chunk_size,
            target_chunk_size=vc_field.config.target_chunk_size,
        )
        residual = Bint + Bext - surface.B_total
        denom = jnp.maximum(jnp.linalg.norm(surface.B_total), jnp.asarray(1e-300, dtype=surface.B_total.dtype))
        metrics = {
            "branch_identity_relative_l2": float(jnp.linalg.norm(residual) / denom),
            "branch_identity_max_abs": float(jnp.max(jnp.abs(residual))),
        }
        if vc_field.external_B_fn is not None:
            Bcoils = vc_field.external_B_fn(surface.gamma)
            Bint_n = jnp.sum(Bint * surface.normal, axis=0)
            Bext_n = jnp.sum(Bext * surface.normal, axis=0)
            Bcoils_n = jnp.sum(Bcoils * surface.normal, axis=0)
            total_n = Bcoils_n + Bint_n
            total_scale = jnp.maximum(
                jnp.sqrt(jnp.mean(jnp.linalg.norm(Bcoils + Bint, axis=0) ** 2)),
                jnp.asarray(1e-300, dtype=surface.B_total.dtype),
            )
            coil_normal_scale = jnp.maximum(
                jnp.sqrt(jnp.mean(Bcoils_n * Bcoils_n)),
                jnp.asarray(1e-300, dtype=surface.B_total.dtype),
            )
            metrics.update(
                {
                    "coil_plus_internal_Bn_rms_normalized": float(jnp.sqrt(jnp.mean(total_n * total_n)) / total_scale),
                    "coil_plus_internal_Bn_max_normalized": float(jnp.max(jnp.abs(total_n)) / total_scale),
                    "external_branch_vs_coil_Bn_relative_l2": float(jnp.linalg.norm(Bext_n - Bcoils_n) / coil_normal_scale),
                }
            )
        return metrics
    except Exception as exc:
        return {"branch_identity_error": str(exc)}


def cmd_validate(args):
    t0 = time.perf_counter()
    field = _build_field_from_args(args)
    vc_field = field.vc_field
    surface = vc_field.surface_data
    metrics = {
        "status": "ok",
        "input": str(args.input) if args.input is not None else None,
        "wout": str(args.wout),
        "coils": str(args.coils) if args.coils is not None else None,
        "nfp": int(surface.nfp),
        "stellsym": bool(surface.stellsym),
        "signgs": int(surface.signgs),
        "source_nphi": int(surface.gamma.shape[1]),
        "source_ntheta": int(surface.gamma.shape[2]),
        "branch": str(vc_field.config.branch),
        "vcp_levels_requested": _levels_for_json(vc_field.config.levels),
        "vcp_levels_effective": _levels_for_json(
            getattr(vc_field, "schedule_levels", vc_field.config.levels)
        ),
        "surface_orientation": "outward_enforced_by_bridge",
    }
    metrics.update(_surface_metrics(surface))
    metrics.update(_branch_identity_metrics(vc_field))
    metrics["runtime_seconds"] = time.perf_counter() - t0

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def cmd_grid(args):
    from virtual_casing_jax.grid_export import write_extended_field_netcdf

    t0 = time.perf_counter()
    field = _build_field_from_args(args)
    grid = field.vc_field.export_rphiz_grid(
        parse_range(args.R),
        parse_range(args.phi),
        parse_range(args.Z),
        chunk_size=_parse_chunk_arg(args.grid_chunk_size),
    )
    metadata = {
        "nfp": int(field.vc_field.surface_data.nfp),
        "stellsym": bool(field.vc_field.surface_data.stellsym),
        "source_wout": str(args.wout),
        "source_input": str(args.input) if args.input is not None else "",
        "source_nphi": int(field.vc_field.surface_data.gamma.shape[1]),
        "source_ntheta": int(field.vc_field.surface_data.gamma.shape[2]),
        "vcp_digits": int(field.vc_field.config.digits),
        "vcp_levels_requested": tuple(tuple(x) for x in field.vc_field.config.levels),
        "vcp_levels_effective": tuple(
            tuple(x) for x in getattr(field.vc_field, "schedule_levels", field.vc_field.config.levels)
        ),
        "coil_source": str(args.coils) if args.coils is not None else "",
        "units": "SI-like VMEC/coil units",
        "coordinate_convention": "R, physical phi, Z",
        "sign_convention": "B_total_out = B_coils + B_internal^VC",
        "runtime_seconds": time.perf_counter() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_extended_field_netcdf(args.out, grid, metadata)
    print(json.dumps({"status": "ok", "out": str(args.out), "shape": list(np.asarray(grid["BR"]).shape)}, indent=2))
    return 0


def cmd_trace(args):
    from essos.dynamics import Tracing

    t0 = time.perf_counter()
    field = _build_field_from_args(args)
    initial_xyz = jnp.asarray(load_seed_points(args.seeds))
    maxtime = float(args.maxtime) if args.maxtime is not None else float(args.nturns)
    tracing = Tracing(
        field=field,
        model="FieldLineAdaptative",
        initial_conditions=initial_xyz,
        maxtime=maxtime,
        times_to_trace=int(args.times_to_trace),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    trajectories = np.asarray(tracing.trajectories)
    trajectories_xyz = np.asarray(tracing.trajectories_xyz)
    times = np.asarray(tracing.times)

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out,
            trajectories=trajectories,
            trajectories_xyz=trajectories_xyz,
            times=times,
            initial_xyz=np.asarray(initial_xyz),
        )
    samples_shape = None
    if args.samples_out is not None:
        samples = write_fieldline_samples_npz(
            args.samples_out,
            trajectories_xyz,
            sample_stride=args.sample_stride,
            sample_phi_period=args.sample_phi_period,
            sample_nsections=args.sample_nsections,
            metadata={
                "source": "ESSOS vmec_extender trace",
                "sample_stride": args.sample_stride,
                "sample_phi_period": args.sample_phi_period,
                "sample_nsections": args.sample_nsections,
                "maxtime": maxtime,
                "times_to_trace": int(args.times_to_trace),
            },
        )
        samples_shape = list(samples["poincare_rphiz"].shape)
    if args.plot is not None:
        import matplotlib.pyplot as plt

        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        shifts = [float(x) for x in str(args.phis).split(",") if x.strip()]
        tracing.poincare_plot(ax=ax, show=False, shifts=shifts)
        fig.tight_layout()
        fig.savefig(args.plot)
        plt.close(fig)

    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(args.out) if args.out is not None else None,
                "samples_out": str(args.samples_out) if args.samples_out is not None else None,
                "plot": str(args.plot) if args.plot is not None else None,
                "shape": list(trajectories.shape),
                "samples_shape": samples_shape,
                "runtime_seconds": time.perf_counter() - t0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def add_common_args(parser):
    parser.add_argument("--input", type=Path, default=None, help="VMEC input file; recommended with --wout")
    parser.add_argument("--wout", type=Path, required=True, help="VMEC wout NetCDF file")
    parser.add_argument("--coils", type=Path, default=None, help="Optional ESSOS coil JSON")
    parser.add_argument("--src-nphi", type=int, default=32)
    parser.add_argument("--src-ntheta", type=int, default=32)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--chunk-size", default="auto")
    parser.add_argument("--target-chunk-size", default="auto")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))


def build_parser():
    parser = argparse.ArgumentParser(prog="essos-vmec-extender")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate VMEC exterior-field signs and surface metrics")
    add_common_args(validate)
    validate.add_argument("--out", type=Path, default=None, help="Optional validation JSON output")
    validate.set_defaults(func=cmd_validate)

    grid = sub.add_parser("grid", help="Export an extended field on an R,phi,Z grid")
    add_common_args(grid)
    grid.add_argument("--R", required=True, help="R values as start:stop:n or comma list")
    grid.add_argument("--phi", required=True, help="physical phi values as start:stop:n or comma list")
    grid.add_argument("--Z", required=True, help="Z values as start:stop:n or comma list")
    grid.add_argument("--out", type=Path, required=True)
    grid.add_argument("--grid-chunk-size", default="auto")
    grid.set_defaults(func=cmd_grid)

    trace = sub.add_parser("trace", help="Trace field lines with the VMEC exterior field")
    add_common_args(trace)
    trace.add_argument("--seeds", type=Path, required=True, help="JSON seed points: [[x,y,z], ...] or {'R_phi_Z': ...}")
    trace.add_argument("--nturns", type=float, default=200.0, help="Integration length used when --maxtime is omitted")
    trace.add_argument("--maxtime", type=float, default=None, help="Explicit ESSOS field-line integration time")
    trace.add_argument("--times-to-trace", type=int, default=6000)
    trace.add_argument("--phis", default="0.0", help="Comma-separated Poincare toroidal sections for --plot")
    trace.add_argument("--atol", type=float, default=1e-7)
    trace.add_argument("--rtol", type=float, default=1e-7)
    trace.add_argument("--out", type=Path, default=None, help="Optional .npz trajectory output")
    trace.add_argument("--samples-out", type=Path, default=None, help="Optional benchmark-compatible .npz field-line samples")
    trace.add_argument("--sample-stride", type=int, default=None, help="Sample every N saved trace points for --samples-out")
    trace.add_argument("--sample-phi-period", type=float, default=None, help="Sample common unwrapped toroidal-phi sections")
    trace.add_argument("--sample-nsections", type=int, default=None, help="Number of phi sections for --sample-phi-period")
    trace.add_argument("--plot", type=Path, default=None, help="Optional Poincare plot output")
    trace.set_defaults(func=cmd_trace)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
