"""Small ESSOS VMEC-extender CLI benchmark using bundled example inputs."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import time

import numpy as np

from essos import vmec_extender_cli as cli


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "examples" / "input_files"


def _run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return cli.main(args)


def _base_args():
    return [
        "--input",
        str(INPUT_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres"),
        "--wout",
        str(INPUT_DIR / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"),
        "--coils",
        str(INPUT_DIR / "ESSOS_biot_savart_LandremanPaulQA.json"),
        "--src-nphi",
        "13",
        "--src-ntheta",
        "13",
        "--digits",
        "3",
    ]


def run_benchmark(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_json = output_dir / "validation.json"
    t0 = time.perf_counter()
    _run_cli(["validate", *_base_args(), "--out", str(validate_json)])
    validate_seconds = time.perf_counter() - t0
    validation = json.loads(validate_json.read_text())

    grid_path = output_dir / "field_grid.nc"
    t0 = time.perf_counter()
    _run_cli(
        [
            "grid",
            *_base_args(),
            "--R",
            "1.0:1.2:4",
            "--phi",
            "0:1.57079632679:4",
            "--Z=-0.1:0.1:4",
            "--out",
            str(grid_path),
        ]
    )
    grid_seconds = time.perf_counter() - t0

    seeds = output_dir / "seeds.json"
    seeds.write_text(json.dumps([[1.1, 0.0, 0.0], [1.15, 0.0, 0.01]]))
    trace_path = output_dir / "trace.npz"
    t0 = time.perf_counter()
    _run_cli(
        [
            "trace",
            *_base_args(),
            "--seeds",
            str(seeds),
            "--maxtime",
            "0.1",
            "--times-to-trace",
            "4",
            "--out",
            str(trace_path),
        ]
    )
    trace_seconds = time.perf_counter() - t0
    trace = np.load(trace_path)

    return {
        "case": "LandremanPaul2021_QA_reactorScale_lowres",
        "validate_seconds": float(validate_seconds),
        "grid_seconds": float(grid_seconds),
        "grid_shape": [4, 4, 4],
        "trace_seconds": float(trace_seconds),
        "trace_shape": list(trace["trajectories"].shape),
        "B_dot_n_rms_normalized": validation["B_dot_n_rms_normalized"],
        "branch_identity_relative_l2": validation["branch_identity_relative_l2"],
        "coil_plus_internal_Bn_rms_normalized": validation.get("coil_plus_internal_Bn_rms_normalized"),
        "external_branch_vs_coil_Bn_relative_l2": validation.get("external_branch_vs_coil_Bn_relative_l2"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output" / "vmec_extender_benchmark")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_benchmark(args.output_dir)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
