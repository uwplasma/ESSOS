"""VMEC mgrid read/write helpers for ESSOS coil fields.

The writer mirrors the SIMSOPT ``MagneticField.to_mgrid`` convention:
fields are evaluated on a cylindrical tensor grid with layout
``(nphi, nz, nr)`` and written to VMEC/MAKEGRID-compatible NetCDF files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file


def _pad_string(string: str) -> str:
    return "{:^30}".format(str(string)).replace(" ", "_")


def _unpack(binary_array: Any) -> str:
    return "".join(np.char.decode(binary_array)).strip()


@dataclass
class MGrid:
    """Container for VMEC mgrid field data."""

    nr: int = 51
    nz: int = 51
    nphi: int = 24
    nfp: int = 2
    rmin: float = 0.20
    rmax: float = 0.40
    zmin: float = -0.10
    zmax: float = 0.10
    br_arr: list[Any] = field(default_factory=list)
    bp_arr: list[Any] = field(default_factory=list)
    bz_arr: list[Any] = field(default_factory=list)
    coil_names: list[str] = field(default_factory=list)

    @property
    def n_ext_cur(self) -> int:
        return len(self.br_arr)

    def add_field_cylindrical(self, br: Any, bp: Any, bz: Any, name: str | None = None) -> None:
        """Append one external-current group in cylindrical components."""

        expected = (int(self.nphi), int(self.nz), int(self.nr))
        br_arr = np.asarray(br, dtype=float)
        bp_arr = np.asarray(bp, dtype=float)
        bz_arr = np.asarray(bz, dtype=float)
        if br_arr.shape != expected or bp_arr.shape != expected or bz_arr.shape != expected:
            raise ValueError(f"mgrid fields must have shape {expected}")
        self.br_arr.append(br_arr)
        self.bp_arr.append(bp_arr)
        self.bz_arr.append(bz_arr)
        self.coil_names.append(_pad_string(name or f"magnet_{self.n_ext_cur}"))

    def write(self, filename: str | Path) -> None:
        """Write the mgrid to a VMEC-compatible NetCDF file."""

        filename = str(filename)
        with netcdf_file(filename, "w", mmap=False, version=2) as ds:
            ds.createDimension("stringsize", 30)
            ds.createDimension("dim_00001", 1)
            ds.createDimension("external_coil_groups", self.n_ext_cur)
            ds.createDimension("external_coils", self.n_ext_cur)
            ds.createDimension("rad", int(self.nr))
            ds.createDimension("zee", int(self.nz))
            ds.createDimension("phi", int(self.nphi))

            ds.createVariable("ir", "i4", tuple()).data[()] = int(self.nr)
            ds.createVariable("jz", "i4", tuple()).data[()] = int(self.nz)
            ds.createVariable("kp", "i4", tuple()).data[()] = int(self.nphi)
            ds.createVariable("nfp", "i4", tuple()).data[()] = int(self.nfp)
            ds.createVariable("nextcur", "i4", tuple()).data[()] = int(self.n_ext_cur)
            ds.createVariable("rmin", "f8", tuple()).data[()] = float(self.rmin)
            ds.createVariable("zmin", "f8", tuple()).data[()] = float(self.zmin)
            ds.createVariable("rmax", "f8", tuple()).data[()] = float(self.rmax)
            ds.createVariable("zmax", "f8", tuple()).data[()] = float(self.zmax)

            if self.n_ext_cur == 1:
                coil_group = ds.createVariable("coil_group", "c", ("stringsize",))
                coil_group[:] = self.coil_names[0]
            else:
                coil_group = ds.createVariable("coil_group", "c", ("external_coil_groups", "stringsize"))
                coil_group[:] = self.coil_names
            mode = ds.createVariable("mgrid_mode", "c", ("dim_00001",))
            mode[:] = "N"
            raw_current = ds.createVariable("raw_coil_cur", "f8", ("external_coils",))
            raw_current[:] = np.ones(self.n_ext_cur)

            for idx in range(self.n_ext_cur):
                tag = f"_{idx + 1:03d}"
                ds.createVariable("br" + tag, "f8", ("phi", "zee", "rad"))[:, :, :] = self.br_arr[idx]
                ds.createVariable("bp" + tag, "f8", ("phi", "zee", "rad"))[:, :, :] = self.bp_arr[idx]
                ds.createVariable("bz" + tag, "f8", ("phi", "zee", "rad"))[:, :, :] = self.bz_arr[idx]

    @classmethod
    def from_file(cls, filename: str | Path) -> "MGrid":
        """Read an mgrid NetCDF file."""

        with netcdf_file(str(filename), "r", mmap=False, version=2) as ds:
            mgrid = cls(
                nr=int(ds.variables["ir"].getValue()),
                nz=int(ds.variables["jz"].getValue()),
                nphi=int(ds.variables["kp"].getValue()),
                nfp=int(ds.variables["nfp"].getValue()),
                rmin=float(ds.variables["rmin"].getValue()),
                rmax=float(ds.variables["rmax"].getValue()),
                zmin=float(ds.variables["zmin"].getValue()),
                zmax=float(ds.variables["zmax"].getValue()),
            )
            nextcur = int(ds.variables["nextcur"].getValue())
            coil_data = ds.variables["coil_group"][:]
            if len(ds.variables["coil_group"].dimensions) == 2:
                mgrid.coil_names = [_unpack(coil_data[j]) for j in range(nextcur)]
            else:
                mgrid.coil_names = [_unpack(coil_data)]
            mgrid.mode = ds.variables["mgrid_mode"][:][0].decode()
            mgrid.raw_coil_current = np.asarray(ds.variables["raw_coil_cur"][:], dtype=float)
            for idx in range(nextcur):
                tag = f"_{idx + 1:03d}"
                mgrid.br_arr.append(np.asarray(ds.variables["br" + tag][:], dtype=float))
                mgrid.bp_arr.append(np.asarray(ds.variables["bp" + tag][:], dtype=float))
                mgrid.bz_arr.append(np.asarray(ds.variables["bz" + tag][:], dtype=float))
            mgrid.br = mgrid.br_arr[0] if nextcur == 1 else np.sum(mgrid.br_arr, axis=0)
            mgrid.bp = mgrid.bp_arr[0] if nextcur == 1 else np.sum(mgrid.bp_arr, axis=0)
            mgrid.bz = mgrid.bz_arr[0] if nextcur == 1 else np.sum(mgrid.bz_arr, axis=0)
            mgrid.bvec = np.transpose([mgrid.br, mgrid.bp, mgrid.bz])
            return mgrid


def coils_to_mgrid(
    coils: Any,
    filename: str | Path,
    *,
    nr: int = 10,
    nphi: int = 4,
    nz: int = 12,
    rmin: float = 1.0,
    rmax: float = 2.0,
    zmin: float = -0.5,
    zmax: float = 0.5,
    nfp: int | None = None,
    name: str = "essos_coils",
) -> MGrid:
    """Evaluate ESSOS coils on a cylindrical grid and write an mgrid file."""

    import jax
    import jax.numpy as jnp

    from .fields import BiotSavart

    nfp_eff = int(coils.nfp if nfp is None else nfp)
    rs = np.linspace(float(rmin), float(rmax), int(nr), endpoint=True)
    phis = np.linspace(0.0, 2.0 * np.pi / nfp_eff, int(nphi), endpoint=False)
    zs = np.linspace(float(zmin), float(zmax), int(nz), endpoint=True)
    Phi, Z, R = np.meshgrid(phis, zs, rs, indexing="ij")
    xyz = np.stack(
        [
            R.reshape(-1) * np.cos(Phi.reshape(-1)),
            R.reshape(-1) * np.sin(Phi.reshape(-1)),
            Z.reshape(-1),
        ],
        axis=1,
    )

    field = BiotSavart(coils)
    b_xyz = np.asarray(jax.vmap(field.B)(jnp.asarray(xyz)), dtype=float)
    phi_flat = Phi.reshape(-1)
    bx = b_xyz[:, 0]
    by = b_xyz[:, 1]
    br = bx * np.cos(phi_flat) + by * np.sin(phi_flat)
    bp = -bx * np.sin(phi_flat) + by * np.cos(phi_flat)
    bz = b_xyz[:, 2]

    mgrid = MGrid(nfp=nfp_eff, nr=int(nr), nz=int(nz), nphi=int(nphi), rmin=rmin, rmax=rmax, zmin=zmin, zmax=zmax)
    mgrid.add_field_cylindrical(
        br.reshape((int(nphi), int(nz), int(nr))),
        bp.reshape((int(nphi), int(nz), int(nr))),
        bz.reshape((int(nphi), int(nz), int(nr))),
        name=name,
    )
    mgrid.write(filename)
    return mgrid
