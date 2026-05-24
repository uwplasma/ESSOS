from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from essos.coils import Coils_from_json, Coils_from_simsopt, CreateEquallySpacedCurves
from essos.mgrid import MGrid, coils_to_mgrid


def test_mgrid_write_read_roundtrip(tmp_path):
    mgrid = MGrid(nr=3, nz=4, nphi=2, nfp=1, rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5)
    br = np.ones((2, 4, 3))
    bp = 2.0 * br
    bz = 3.0 * br
    mgrid.add_field_cylindrical(br, bp, bz, name="test_coil")

    filename = tmp_path / "mgrid.test.nc"
    mgrid.write(filename)
    loaded = MGrid.from_file(filename)

    assert loaded.n_ext_cur == 1
    assert loaded.nr == 3
    assert loaded.nz == 4
    assert loaded.nphi == 2
    assert loaded.nfp == 1
    assert loaded.coil_names[0] == "__________test_coil___________"
    np.testing.assert_allclose(loaded.br_arr[0], br)
    np.testing.assert_allclose(loaded.bp_arr[0], bp)
    np.testing.assert_allclose(loaded.bz_arr[0], bz)


def test_coils_to_mgrid_writes_expected_shape_and_finite_values(tmp_path):
    coils = CreateEquallySpacedCurves(1, order=1, R=1.0, r=0.2, n_segments=32, nfp=1, stellsym=False)
    from essos.coils import Coils
    import jax.numpy as jnp

    coil_set = Coils(coils, jnp.asarray([2.0]))
    filename = tmp_path / "mgrid.coils.nc"
    mgrid = coils_to_mgrid(
        coil_set,
        filename,
        nr=4,
        nphi=3,
        nz=5,
        rmin=0.4,
        rmax=1.8,
        zmin=-0.7,
        zmax=0.7,
        nfp=1,
    )
    loaded = MGrid.from_file(filename)

    assert mgrid.n_ext_cur == 1
    assert loaded.br_arr[0].shape == (3, 5, 4)
    assert loaded.bp_arr[0].shape == (3, 5, 4)
    assert loaded.bz_arr[0].shape == (3, 5, 4)
    assert np.all(np.isfinite(loaded.br_arr[0]))
    assert np.all(np.isfinite(loaded.bp_arr[0]))
    assert np.all(np.isfinite(loaded.bz_arr[0]))


def test_landreman_paul_qa_essos_json_can_write_mgrid(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
    coils = Coils_from_json(str(path))
    filename = tmp_path / "mgrid.lp_qa.nc"

    coils.to_mgrid(filename, nr=4, nphi=3, nz=5, rmin=0.5, rmax=2.0, zmin=-0.8, zmax=0.8)
    loaded = MGrid.from_file(filename)

    assert loaded.nfp == 2
    assert loaded.br_arr[0].shape == (3, 5, 4)
    assert np.max(np.abs(loaded.br_arr[0])) > 0.0
    assert np.max(np.abs(loaded.bz_arr[0])) > 0.0


def test_simsopt_to_mgrid_parity_when_simsopt_is_available(tmp_path):
    simsopt = pytest.importorskip("simsopt")
    from simsopt import load
    from simsopt.field import MGrid as SimsoptMGrid

    del simsopt
    json_file = Path(__file__).resolve().parents[1] / "examples" / "input_files" / "SIMSOPT_biot_savart_LandremanPaulQA.json"
    essos_coils = Coils_from_simsopt(str(json_file), nfp=2, stellsym=True)
    simsopt_field = load(str(json_file))

    kwargs = dict(nr=4, nphi=3, nz=5, rmin=0.5, rmax=2.0, zmin=-0.8, zmax=0.8, nfp=2)
    essos_file = tmp_path / "mgrid.essos.nc"
    simsopt_file = tmp_path / "mgrid.simsopt.nc"
    essos_coils.to_mgrid(essos_file, **kwargs)
    simsopt_field.to_mgrid(simsopt_file, **kwargs)

    essos_grid = MGrid.from_file(essos_file)
    simsopt_grid = SimsoptMGrid.from_file(simsopt_file)

    np.testing.assert_allclose(essos_grid.br_arr[0], simsopt_grid.br_arr[0], rtol=5.0e-12, atol=1.0e-16)
    np.testing.assert_allclose(essos_grid.bp_arr[0], simsopt_grid.bp_arr[0], rtol=5.0e-12, atol=1.0e-16)
    np.testing.assert_allclose(essos_grid.bz_arr[0], simsopt_grid.bz_arr[0], rtol=5.0e-12, atol=1.0e-16)
