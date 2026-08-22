from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PYQSC_SOURCE = REPOSITORY.parent / "pyQSC_JAX" / "src"


@pytest.mark.parametrize(
    "script, expected_text",
    [
        (
            "optimize_coils_for_external_nearaxis.py",
            "final normalized objective:",
        ),
        (
            "optimize_coils_and_external_nearaxis.py",
            "final near-axis variables",
        ),
    ],
)
def test_external_field_jet_examples_run(tmp_path, script, expected_text):
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "true"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(PYQSC_SOURCE),
            str(REPOSITORY),
            environment.get("PYTHONPATH", ""),
        )
    )
    script_path = REPOSITORY / "examples" / "coil_optimization" / script

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected_text in result.stdout
