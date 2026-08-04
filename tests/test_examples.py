from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_differentiate_coil_design_example_covers_the_public_api():
    source = (REPO / "examples" / "differentiate_coil_design.py").read_text()
    public_symbols = {
        "CoilDesignFieldBuilder",
        "FilamentaryBiotSavart",
        "ShapeDeformationMetrics",
        "make_coil_design_field_builder",
        "make_fractional_current_field_builder",
        "make_shape_field_builder",
        "shape_deformation_metrics",
    }
    public_calls = {
        "make_coil_design_field_builder",
        "make_fractional_current_field_builder",
        "make_shape_field_builder",
        "shape_deformation_metrics",
    }

    assert all(symbol in source for symbol in public_symbols)
    assert all(f"{function}(" in source for function in public_calls)
    assert "shape_directions=shape_directions" in source
    for method in (
        "from_coils",
        "B",
        "b_cyl",
        "field_from_scalar_current",
        "curve_dofs_at",
        "base_currents_at",
        "expanded_currents_at",
        "rebuild_curves",
        "rebuild_coils",
    ):
        assert f".{method}(" in source


def test_differentiate_coil_design_example(tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(REPO / "examples" / "differentiate_coil_design.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "independent physical currents [A]:" in result.stdout
    assert "direct Cartesian B [T]:" in result.stdout
    assert "direct cylindrical B [T]:" in result.stdout
    assert "joint builder: 2 current + 1 shape parameters" in result.stdout
    assert "shape deformation:" in result.stdout
    assert (
        "arbitrary multi-shape chart: 2 current + 2 shape parameters" in result.stdout
    )
    assert "arbitrary chart parameter order:" in result.stdout
    assert "[0] fractional current group 0:" in result.stdout
    assert "[1] fractional current group 2:" in result.stdout
    assert "[2] shape direction 0 [m]:" in result.stdout
    assert "[3] shape direction 1 [m]:" in result.stdout
    assert "arbitrary chart field B [T]:" in result.stdout
    assert "arbitrary chart field JVP norm [T]:" in result.stdout
    assert "rebuilt physical-coil length range [m]:" in result.stdout


def test_differentiate_planar_coils_example_covers_the_public_api():
    source = (REPO / "examples" / "differentiate_planar_coils.py").read_text()
    for symbol in (
        "PlanarCoils",
        "PlanarXYCurves",
        "PlanarCoilDesignFieldBuilder",
        "make_planar_coil_design_field_builder",
    ):
        assert symbol in source
    for method in (
        "from_polar_radius",
        "rebuild_curves",
        "centers_at",
        "normals_at",
    ):
        assert f".{method}(" in source


def test_differentiate_planar_coils_example(tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(REPO / "examples" / "differentiate_planar_coils.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "planar builder blocks: 1 current + 2 local shape + 2 center + "
        "2 orientation" in result.stdout
    )
    assert "planar parameter order:" in result.stdout
    assert "[0] fractional current group 0:" in result.stdout
    assert "[5] coil 0 rotation-Y [rad]:" in result.stdout
    assert "planar field B [T]:" in result.stdout
    assert "planar field JVP [T]:" in result.stdout
    assert "max planarity residual [m]:" in result.stdout
    assert "polar compatibility XY order: 1" in result.stdout
