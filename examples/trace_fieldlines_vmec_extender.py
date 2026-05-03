"""Trace a tiny VMEC-extender field-line smoke case."""

from __future__ import annotations

import json
from pathlib import Path

from essos.vmec_extender_cli import main


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_files"


if __name__ == "__main__":
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = output_dir / "vmec_extender_seeds.json"
    seeds.write_text(json.dumps([[1.1, 0.0, 0.0], [1.15, 0.0, 0.01]]))
    raise SystemExit(
        main(
            [
                "trace",
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
                "--seeds",
                str(seeds),
                "--maxtime",
                "0.1",
                "--times-to-trace",
                "4",
                "--out",
                str(output_dir / "vmec_extender_trace.npz"),
                "--samples-out",
                str(output_dir / "vmec_extender_trace_samples.npz"),
                "--sample-stride",
                "1",
            ]
        )
    )
