import json
from pathlib import Path

import pytest

from essos import vmec_extender_cli as cli


def test_validate_cli_real_bundled_vmec_case(tmp_path):
    pytest.importorskip("virtual_casing_jax")
    pytest.importorskip("vmec_jax")
    root = Path(__file__).resolve().parents[1]
    input_dir = root / "examples" / "input_files"
    out = tmp_path / "validation.json"

    code = cli.main(
        [
            "validate",
            "--input",
            str(input_dir / "input.LandremanPaul2021_QA_reactorScale_lowres"),
            "--wout",
            str(input_dir / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"),
            "--coils",
            str(input_dir / "ESSOS_biot_savart_LandremanPaulQA.json"),
            "--src-nphi",
            "13",
            "--src-ntheta",
            "13",
            "--digits",
            "3",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["nfp"] == 2
    assert data["source_nphi"] == 14
    assert data["source_ntheta"] == 16
    assert data["B_dot_n_rms_normalized"] < 1e-12
    assert data["branch_identity_relative_l2"] < 1e-12
    assert "coil_plus_internal_Bn_rms_normalized" in data
