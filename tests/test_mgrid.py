from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from netCDF4 import Dataset

import essos.mgrid as mgrid_module
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.mgrid import MGrid, coils_to_mgrid


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning")
def test_mgrid_write_read_roundtrip(tmp_path):
    mgrid = MGrid(nr=3, nz=4, nphi=2, nfp=1, rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5)
    br = np.ones((2, 4, 3))
    bp = 2.0 * br
    bz = 3.0 * br
    mgrid.add_field_cylindrical(br, bp, bz, name="test_coil")
    mgrid.raw_coil_current = [2.5]

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
    assert loaded.mode == "N"
    np.testing.assert_allclose(loaded.raw_coil_current, [2.5])
    np.testing.assert_allclose(loaded.br, br)

    # NetCDF4 files use the same MGRID variables as the NetCDF3 writer.
    netcdf4_file = tmp_path / "mgrid.netcdf4.nc"
    with Dataset(filename) as source, Dataset(netcdf4_file, "w") as target:
        source.set_auto_mask(False)
        for name, dimension in source.dimensions.items():
            target.createDimension(name, len(dimension))
        for name, variable in source.variables.items():
            target.createVariable(name, variable.dtype, variable.dimensions)[...] = variable[...]
    np.testing.assert_allclose(MGrid.from_file(netcdf4_file).B([1.5, 0.0, 0.0]), [1.0, 2.0, 3.0])


def test_coils_to_mgrid_writes_expected_shape_and_finite_values(tmp_path, monkeypatch):
    # Keep JIT compilation independent of the cache populated by earlier tests.
    jax.clear_caches()
    coils = CreateEquallySpacedCurves(1, order=1, R=1.0, r=0.2, n_segments=32, nfp=1, stellsym=False)
    coil_set = Coils(coils, jnp.asarray([2.0]))
    monkeypatch.setattr(mgrid_module, "_BIOT_SAVART_PAIR_BUDGET", 32 * 31)
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

    # The file reproduces the source BiotSavart field exactly at grid nodes.
    phi = 2.0 * np.pi * np.arange(3) / (3 * coil_set.nfp)
    point = np.array([0.4 * np.cos(phi[1]), 0.4 * np.sin(phi[1]), -0.7])
    np.testing.assert_allclose(loaded.B(point), BiotSavart(coil_set).B(point), rtol=2e-13, atol=1e-15)
    with pytest.raises(ValueError, match="nfp must divide"):
        coil_set.to_mgrid(tmp_path / "invalid.nc", nfp=2)


def test_mgrid_field_is_periodic_batched_jittable_and_differentiable(tmp_path):
    mgrid = MGrid(nr=2, nz=2, nphi=2, nfp=2, rmin=1.0, rmax=2.0, zmin=-1.0, zmax=1.0)
    rr, zz = np.meshgrid([1.0, 2.0], [-1.0, 1.0], indexing="xy")
    base = np.broadcast_to(1.0 + rr + 2.0 * zz, (2, 2, 2))
    bp = np.broadcast_to(2.0 * rr - zz, (2, 2, 2))
    bz = np.broadcast_to(3.0 + zz, (2, 2, 2))
    br_planes = np.stack((base[0], base[1] + 4.0))
    bp_planes = np.stack((bp[0], bp[1] + 2.0))
    bz_planes = np.stack((bz[0], bz[1] + 6.0))
    mgrid.add_field_cylindrical(br_planes, bp_planes, bz_planes)
    mgrid.add_field_cylindrical(2.0 * br_planes, 2.0 * bp_planes, 2.0 * bz_planes)
    mgrid.mode = "S"
    mgrid.raw_coil_current = [2.0, 3.0]
    filename = tmp_path / "mgrid.scaled.nc"
    mgrid.write(filename)
    field = MGrid.from_file(filename)

    phi = np.pi / 8
    r, z = 1.5, 0.25
    expected_cyl = np.array([32.0, 26.0, 38.0])
    np.testing.assert_allclose(field.b_cyl(r, phi, z), expected_cyl)
    np.testing.assert_allclose(field.b_cyl(r, 7.0 * np.pi / 8, z), expected_cyl)
    np.testing.assert_allclose(field.b_cyl(r, phi + np.pi, z), expected_cyl)
    np.testing.assert_allclose(field.b_cyl(r, phi - np.pi, z), expected_cyl)
    np.testing.assert_allclose(field.b_cyl(0.5, phi, z), field.b_cyl(1.0, phi, z))
    np.testing.assert_allclose(field.b_cyl(r, phi, 2.0), field.b_cyl(r, phi, 1.0))

    points = np.array([[r * np.cos(phi), r * np.sin(phi), z],
                       [r * np.cos(phi + np.pi), r * np.sin(phi + np.pi), z]])
    values = jax.jit(jax.vmap(field.B))(points)
    expected = np.array([[32.0 * np.cos(p) - 26.0 * np.sin(p),
                          32.0 * np.sin(p) + 26.0 * np.cos(p), 38.0]
                         for p in (phi, phi + np.pi)])
    np.testing.assert_allclose(values, expected, atol=1e-14)
    np.testing.assert_allclose(field.AbsB(points), np.linalg.norm(expected, axis=-1))
    assert np.all(np.isfinite(jax.grad(lambda x: field.AbsB(x))(points[0])))
    with pytest.raises(ValueError, match="trailing dimension of 3"):
        field.B(np.ones((2, 2)))


def test_landreman_paul_qa_essos_json_can_write_mgrid(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
    coils = Coils.from_json(str(path))
    filename = tmp_path / "mgrid.lp_qa.nc"

    coils.to_mgrid(filename, nr=64, nphi=32, nz=64, rmin=0.5, rmax=2.0, zmin=-0.8, zmax=0.8)
    loaded = MGrid.from_file(filename)

    assert loaded.nfp == 2
    assert loaded.br_arr[0].shape == (32, 64, 64)
    assert np.max(np.abs(loaded.br_arr[0])) > 0.0
    assert np.max(np.abs(loaded.bz_arr[0])) > 0.0

    r = jnp.array([0.9, 1.0, 1.1, 1.2])
    phi = jnp.array([0.2, 0.6, 1.0, 1.4])
    z = jnp.array([0.0, 0.05, -0.1, 0.15])
    points = jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)
    direct = jax.vmap(BiotSavart(coils).B)(points)
    error = jnp.linalg.norm(loaded.B(points) - direct, axis=-1) / jnp.linalg.norm(direct, axis=-1)
    assert float(jnp.max(error)) < 0.003


@pytest.mark.parametrize("mode", ["R", "N"])
def test_single_plane_raw_modes_and_pytree_roundtrip(tmp_path, mode):
    grid = MGrid(nr=2, nz=2, nphi=1, nfp=3, rmin=1.0, rmax=2.0, zmin=-1.0, zmax=1.0)
    ones = np.ones((1, 2, 2))
    grid.add_field_cylindrical(ones, 2 * ones, 3 * ones)
    grid.add_field_cylindrical(4 * ones, 5 * ones, 6 * ones)
    grid.mode = mode
    grid.raw_coil_current = [10.0, 20.0]  # Raw-mode tables already include these currents.
    filename = tmp_path / "mgrid.raw.nc"
    grid.write(filename)
    loaded = MGrid.from_file(filename)

    phi = np.pi / 4
    np.testing.assert_allclose(loaded.b_cyl(1.5, phi, 0.0), [5.0, 7.0, 9.0])
    np.testing.assert_allclose(loaded.b_cyl(0.1, phi + 2 * np.pi / 3, 2.0), [5.0, 7.0, 9.0])
    point = [1.5 * np.cos(phi), 1.5 * np.sin(phi), 0.0]
    np.testing.assert_allclose(loaded.B(point),
                               [5 * np.cos(phi) - 7 * np.sin(phi),
                                5 * np.sin(phi) + 7 * np.cos(phi), 9])

    leaves, structure = jax.tree_util.tree_flatten(loaded)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    restored.add_field_cylindrical(np.zeros_like(ones), np.zeros_like(ones), np.zeros_like(ones))
    assert restored.n_ext_cur == 3
    assert loaded.n_ext_cur == 2


@pytest.mark.parametrize("kwargs", [
    {"nr": 1}, {"nphi": 0}, {"nr": 2.5}, {"nfp": 1.5},
    {"rmin": -0.1}, {"rmax": np.inf},
])
def test_mgrid_rejects_invalid_grid(kwargs):
    with pytest.raises(ValueError):
        MGrid(**kwargs)


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning")
def test_mgrid_rejects_invalid_fields_and_currents(tmp_path):
    grid = MGrid(nr=2, nz=2, nphi=1, nfp=1)
    ones = np.ones((1, 2, 2))
    with pytest.raises(ValueError, match="one name and field"):
        grid.write(tmp_path / "empty.nc")
    with pytest.raises(ValueError, match="shape"):
        grid.add_field_cylindrical(ones[:, :, :1], ones, ones)
    with pytest.raises(ValueError, match="finite"):
        grid.add_field_cylindrical(np.full_like(ones, np.nan), ones, ones)
    with pytest.raises(ValueError, match="30 characters"):
        grid.add_field_cylindrical(ones, ones, ones, name="x" * 31)

    grid.add_field_cylindrical(ones, ones, ones)
    grid.raw_coil_current = [np.nan]
    with pytest.raises(ValueError, match="finite"):
        grid.write(tmp_path / "invalid.nc")
    grid.raw_coil_current = [1.0]
    filename = tmp_path / "valid.nc"
    grid.write(filename)
    with Dataset(filename, "a") as ds:
        ds.variables["br_001"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        MGrid.from_file(filename)
    with Dataset(filename, "a") as ds:
        ds.variables["br_001"][0, 0, 0] = 1.0
        ds.variables["ir"].assignValue(3)
    with pytest.raises(ValueError, match="shape"):
        MGrid.from_file(filename)


def test_simsopt_to_mgrid_parity_when_simsopt_is_available(tmp_path):
    simsopt = pytest.importorskip("simsopt")
    from simsopt import load
    from simsopt.field import MGrid as SimsoptMGrid

    del simsopt
    json_file = Path(__file__).resolve().parents[1] / "examples" / "input_files" / "SIMSOPT_biot_savart_LandremanPaulQA.json"
    essos_coils = Coils.from_simsopt(str(json_file), nfp=2, stellsym=True)
    simsopt_field = load(str(json_file))

    kwargs = dict(nr=4, nphi=3, nz=5, rmin=0.5, rmax=2.0, zmin=-0.8, zmax=0.8, nfp=2)
    essos_file = tmp_path / "mgrid.essos.nc"
    simsopt_file = tmp_path / "mgrid.simsopt.nc"
    essos_coils.to_mgrid(essos_file, **kwargs)
    simsopt_field.to_mgrid(simsopt_file, **kwargs)

    essos_grid = MGrid.from_file(essos_file)
    imported_simsopt_grid = MGrid.from_file(simsopt_file)
    simsopt_grid = SimsoptMGrid.from_file(simsopt_file)

    np.testing.assert_allclose(essos_grid.br_arr[0], simsopt_grid.br_arr[0], rtol=5.0e-12, atol=1.0e-16)
    np.testing.assert_allclose(essos_grid.bp_arr[0], simsopt_grid.bp_arr[0], rtol=5.0e-12, atol=1.0e-16)
    np.testing.assert_allclose(essos_grid.bz_arr[0], simsopt_grid.bz_arr[0], rtol=5.0e-12, atol=1.0e-16)
    np.testing.assert_allclose(imported_simsopt_grid.bvec, essos_grid.bvec, rtol=5.0e-12, atol=1.0e-16)
