"""Export a tiny VMEC-extender field grid using bundled example inputs."""

from __future__ import annotations

from pathlib import Path

from essos.vmec_extender_cli import main


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_files"


if __name__ == "__main__":
    out = ROOT / "output" / "vmec_extender_grid.nc"
    raise SystemExit(
        main(
            [
                "grid",
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
                "--R",
                "1.0:1.2:4",
                "--phi",
                "0:1.57079632679:4",
                "--Z=-0.1:0.1:4",
                "--out",
                str(out),
            ]
        )
    )
