"""Validate a VMEC-extender field using bundled ESSOS example inputs."""

from __future__ import annotations

from pathlib import Path

from essos.vmec_extender_cli import main


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_files"


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "validate",
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
        )
    )
