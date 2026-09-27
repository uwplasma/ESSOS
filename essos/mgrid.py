"""VMEC mgrid read/write helpers for ESSOS coil fields.

The writer mirrors the SIMSOPT ``MagneticField.to_mgrid`` convention:
fields are evaluated on a cylindrical tensor grid with layout
``(nphi, nz, nr)`` and written to VMEC/MAKEGRID-compatible NetCDF files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset
from scipy.io import netcdf_file

from .fields import BiotSavart, MagneticField

_BIOT_SAVART_PAIR_BUDGET = 4_000_000


def _pad_string(string: str) -> str:
    return "{:^30}".format(str(string)).replace(" ", "_")


def _unpack(binary_array: Any) -> str:
    return "".join(np.char.decode(binary_array)).strip("\x00 ")


def _interpolate_component(fields, weights, r, phi, z, *, rmin, rmax, zmin, zmax, nfp):
    """Trilinearly interpolate coil-group tables, periodically in phi."""
    _, nphi, nz, nr = fields.shape
    r, phi, z = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
    out_shape = r.shape
    rf = jnp.clip(r.reshape((-1,)), rmin, rmax)
    zf = jnp.clip(z.reshape((-1,)), zmin, zmax)

    fr = (rf - rmin) * ((nr - 1) / (rmax - rmin))
    fz = (zf - zmin) * ((nz - 1) / (zmax - zmin))
    ir0 = jnp.clip(jnp.floor(fr).astype(jnp.int32), 0, nr - 2)
    iz0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, nz - 2)
    wr, wz = fr - ir0, fz - iz0
    period = 2.0 * jnp.pi / nfp
    fp = jnp.mod(phi.reshape((-1,)), period) * (nphi / period)
    ip0_float = jnp.floor(fp)
    ip0 = ip0_float.astype(jnp.int32) % nphi
    ip1 = (ip0 + 1) % nphi
    wp = fp - ip0_float

    def sample(field):
        v000 = field[:, ip0, iz0, ir0]
        v001 = field[:, ip0, iz0, ir0 + 1]
        v010 = field[:, ip0, iz0 + 1, ir0]
        v011 = field[:, ip0, iz0 + 1, ir0 + 1]
        v100 = field[:, ip1, iz0, ir0]
        v101 = field[:, ip1, iz0, ir0 + 1]
        v110 = field[:, ip1, iz0 + 1, ir0]
        v111 = field[:, ip1, iz0 + 1, ir0 + 1]
        c00 = v000 * (1.0 - wr) + v001 * wr
        c01 = v010 * (1.0 - wr) + v011 * wr
        c10 = v100 * (1.0 - wr) + v101 * wr
        c11 = v110 * (1.0 - wr) + v111 * wr
        c0 = c00 * (1.0 - wz) + c01 * wz
        c1 = c10 * (1.0 - wz) + c11 * wz
        values = c0 * (1.0 - wp) + c1 * wp
        return jnp.sum(weights[:, None] * values, axis=0).reshape(out_shape)

    return sample(fields)


@dataclass(eq=False)
class MGrid(MagneticField):
    """VMEC mgrid data and a JAX-compatible interpolated magnetic field.

    ``B(points)`` returns Cartesian field vectors. R and Z are clamped to the
    tabulated bounds; phi is periodic with period ``2*pi/nfp``. The field is
    trilinearly interpolated between grid nodes.
    """

    nr: int = 51
    nz: int = 51
    nphi: int = 24
    nfp: int = 2
    rmin: float = 0.20
    rmax: float = 0.40
    zmin: float = -0.10
    zmax: float = 0.10
    br_arr: Any = field(init=False, repr=False)
    bp_arr: Any = field(init=False, repr=False)
    bz_arr: Any = field(init=False, repr=False)
    coil_names: list[str] = field(init=False, default_factory=list)
    mode: str = "N"
    raw_coil_current: Any = field(init=False, repr=False)

    def __post_init__(self):
        dimensions = (self.nr, self.nz, self.nphi, self.nfp)
        if any(not isinstance(n, (int, np.integer)) for n in dimensions):
            raise ValueError("mgrid dimensions must be integers")
        if self.nr < 2 or self.nz < 2 or self.nphi < 1 or self.nfp < 1:
            raise ValueError("mgrid requires nr,nz >= 2 and nphi,nfp >= 1")
        if (not np.all(np.isfinite((self.rmin, self.rmax, self.zmin, self.zmax)))
                or self.rmin < 0 or self.rmax <= self.rmin or self.zmax <= self.zmin):
            raise ValueError("mgrid requires finite bounds, 0 <= rmin < rmax, and zmin < zmax")
        self.br_arr = self.bp_arr = self.bz_arr = jnp.empty((0, self.nphi, self.nz, self.nr))
        self.raw_coil_current = jnp.empty((0,))

    @property
    def n_ext_cur(self) -> int:
        return len(self.br_arr)

    def add_field_cylindrical(self, br: Any, bp: Any, bz: Any, name: str | None = None) -> None:
        """Append one external-current group in cylindrical components."""

        expected = (self.nphi, self.nz, self.nr)
        br_arr = np.asarray(br, dtype=float)
        bp_arr = np.asarray(bp, dtype=float)
        bz_arr = np.asarray(bz, dtype=float)
        if br_arr.shape != expected or bp_arr.shape != expected or bz_arr.shape != expected:
            raise ValueError(f"mgrid fields must have shape {expected}")
        if not all(np.all(np.isfinite(arr)) for arr in (br_arr, bp_arr, bz_arr)):
            raise ValueError("mgrid fields must contain only finite values")
        label = _pad_string(name or f"magnet_{self.n_ext_cur}")
        if len(label) > 30:
            raise ValueError("mgrid coil-group names must fit in 30 characters")
        self.coil_names.append(label)
        for component, values in (("br_arr", br_arr), ("bp_arr", bp_arr), ("bz_arr", bz_arr)):
            setattr(self, component, jnp.concatenate((getattr(self, component), jnp.asarray(values)[None])))
        self.raw_coil_current = jnp.append(jnp.asarray(self.raw_coil_current), 1.0)

    @property
    def _field_weights(self):
        if self.mode.upper() == "S":
            return jnp.asarray(self.raw_coil_current, dtype=float)
        return jnp.ones((self.n_ext_cur,), dtype=float)

    @property
    def br(self):
        return np.sum(np.asarray(self.br_arr) * np.asarray(self._field_weights)[:, None, None, None], axis=0)

    @property
    def bp(self):
        return np.sum(np.asarray(self.bp_arr) * np.asarray(self._field_weights)[:, None, None, None], axis=0)

    @property
    def bz(self):
        return np.sum(np.asarray(self.bz_arr) * np.asarray(self._field_weights)[:, None, None, None], axis=0)

    @property
    def bvec(self):
        return np.stack((self.br, self.bp, self.bz), axis=-1)

    @jax.jit
    def b_cyl(self, r, phi, z):
        """Return ``(B_R, B_phi, B_Z)`` at broadcastable cylindrical points."""
        weights = self._field_weights
        kwargs = dict(rmin=self.rmin, rmax=self.rmax, zmin=self.zmin,
                      zmax=self.zmax, nfp=self.nfp)
        return (
            _interpolate_component(jnp.asarray(self.br_arr), weights, r, phi, z, **kwargs),
            _interpolate_component(jnp.asarray(self.bp_arr), weights, r, phi, z, **kwargs),
            _interpolate_component(jnp.asarray(self.bz_arr), weights, r, phi, z, **kwargs),
        )

    @jax.jit
    def B(self, points):
        """Return the interpolated field in Cartesian coordinates."""
        points = jnp.asarray(points)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("points must have a trailing dimension of 3")
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        r = jnp.hypot(x, y)
        phi = jnp.arctan2(y, x)
        br, bp, bz = self.b_cyl(r, phi, z)
        return jnp.stack((br * jnp.cos(phi) - bp * jnp.sin(phi),
                          br * jnp.sin(phi) + bp * jnp.cos(phi), bz), axis=-1)

    @jax.jit
    def AbsB(self, points):
        """Return the field magnitude for each Cartesian point."""
        return jnp.linalg.norm(self.B(points), axis=-1)

    @jax.jit
    def sqrtg(self, points):
        del points
        return 1.0

    @jax.jit
    def to_xyz(self, points):
        return jnp.asarray(points)

    def _tree_flatten(self):
        children = (self.br_arr, self.bp_arr, self.bz_arr,
                    jnp.asarray(self.raw_coil_current, dtype=float))
        aux = (self.nr, self.nz, self.nphi, self.nfp, self.rmin, self.rmax,
               self.zmin, self.zmax, tuple(self.coil_names), self.mode)
        return children, aux

    @classmethod
    def _tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (obj.nr, obj.nz, obj.nphi, obj.nfp, obj.rmin, obj.rmax,
         obj.zmin, obj.zmax, coil_names, obj.mode) = aux
        obj.coil_names = list(coil_names)
        obj.br_arr, obj.bp_arr, obj.bz_arr, obj.raw_coil_current = children
        return obj


    def write(self, filename: str | Path) -> None:
        """Write the mgrid to a VMEC-compatible NetCDF file."""

        if self.n_ext_cur < 1 or len(self.coil_names) != self.n_ext_cur:
            raise ValueError("mgrid must contain one name and field per external-current group")
        if len(self.raw_coil_current) != self.n_ext_cur:
            raise ValueError("raw_coil_cur length does not match nextcur")
        if self.mode.upper() not in {"N", "R", "S"}:
            raise ValueError("mgrid_mode must be one of 'N', 'R', or 'S'")
        if not np.all(np.isfinite(self.raw_coil_current)):
            raise ValueError("raw_coil_cur must contain only finite values")
        with netcdf_file(str(filename), "w", mmap=False, version=2) as ds:
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
            mode[:] = self.mode.upper()
            raw_current = ds.createVariable("raw_coil_cur", "f8", ("external_coils",))
            raw_current[:] = np.asarray(self.raw_coil_current, dtype=float)

            for idx in range(self.n_ext_cur):
                tag = f"_{idx + 1:03d}"
                ds.createVariable("br" + tag, "f8", ("phi", "zee", "rad"))[:, :, :] = self.br_arr[idx]
                ds.createVariable("bp" + tag, "f8", ("phi", "zee", "rad"))[:, :, :] = self.bp_arr[idx]
                ds.createVariable("bz" + tag, "f8", ("phi", "zee", "rad"))[:, :, :] = self.bz_arr[idx]

    @classmethod
    def from_file(cls, filename: str | Path) -> "MGrid":
        """Read an mgrid NetCDF file."""

        with Dataset(str(filename), "r") as ds:
            ds.set_auto_mask(False)
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
            if nextcur < 1:
                raise ValueError("mgrid file contains no external coil groups")
            coil_data = ds.variables["coil_group"][:]
            if len(ds.variables["coil_group"].dimensions) == 2:
                mgrid.coil_names = [_unpack(coil_data[j]) for j in range(nextcur)]
            else:
                mgrid.coil_names = [_unpack(coil_data)]
            if len(mgrid.coil_names) != nextcur:
                raise ValueError("coil_group count does not match nextcur")
            mode = ds.variables.get("mgrid_mode")
            mgrid.mode = (mode[:][0].decode() if mode is not None else "N").upper()
            if mgrid.mode not in {"N", "R", "S"}:
                raise ValueError(f"unsupported mgrid_mode {mgrid.mode!r}")
            currents = ds.variables.get("raw_coil_cur")
            mgrid.raw_coil_current = jnp.asarray(
                currents[:] if currents is not None else np.ones(nextcur), dtype=float).reshape(-1)
            if len(mgrid.raw_coil_current) != nextcur:
                raise ValueError("raw_coil_cur length does not match nextcur")
            if not np.all(np.isfinite(mgrid.raw_coil_current)):
                raise ValueError("raw_coil_cur must contain only finite values")
            components = ([], [], [])
            for idx in range(nextcur):
                tag = f"_{idx + 1:03d}"
                fields = tuple(np.asarray(ds.variables[name + tag][:], dtype=float)
                               for name in ("br", "bp", "bz"))
                expected = (mgrid.nphi, mgrid.nz, mgrid.nr)
                if any(values.shape != expected for values in fields):
                    raise ValueError(f"mgrid group {idx + 1} fields must have shape {expected}")
                if not all(np.all(np.isfinite(values)) for values in fields):
                    raise ValueError(f"mgrid group {idx + 1} fields must contain only finite values")
                for target, values in zip(components, fields):
                    target.append(values)
            mgrid.br_arr, mgrid.bp_arr, mgrid.bz_arr = (jnp.asarray(values) for values in components)
            return mgrid


jax.tree_util.register_pytree_node(MGrid, MGrid._tree_flatten, MGrid._tree_unflatten)


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
    """Sample a periodic coil field and write a VMEC-compatible mgrid file."""

    mgrid = MGrid(nr=nr, nphi=nphi, nz=nz, nfp=coils.nfp if nfp is None else nfp,
                  rmin=rmin, rmax=rmax, zmin=zmin, zmax=zmax)
    if coils.nfp % mgrid.nfp:
        raise ValueError("mgrid nfp must divide the coil field's nfp")
    rs = np.linspace(rmin, rmax, nr)
    phis = np.linspace(0.0, 2.0 * np.pi / mgrid.nfp, nphi, endpoint=False)
    zs = np.linspace(zmin, zmax, nz)
    Phi, Z, R = np.meshgrid(phis, zs, rs, indexing="ij")
    phi_flat = Phi.ravel()
    cos_phi, sin_phi = np.cos(phi_flat), np.sin(phi_flat)
    xyz = np.column_stack((R.ravel() * cos_phi, R.ravel() * sin_phi, Z.ravel()))

    n_segments = max(1, coils.gamma.size // 3)
    chunk_size = max(1, _BIOT_SAVART_PAIR_BUDGET // n_segments)
    n_chunks = (len(xyz) + chunk_size - 1) // chunk_size
    batched_field = jax.vmap(BiotSavart(coils).B)
    b_xyz = np.concatenate([
        np.asarray(batched_field(jnp.asarray(points)), dtype=float)
        for points in np.array_split(xyz, n_chunks)
    ])
    bx, by, bz = b_xyz.T
    br = bx * cos_phi + by * sin_phi
    bp = -bx * sin_phi + by * cos_phi

    mgrid.add_field_cylindrical(
        br.reshape((nphi, nz, nr)),
        bp.reshape((nphi, nz, nr)),
        bz.reshape((nphi, nz, nr)),
        name=name,
    )
    mgrid.write(filename)
    return mgrid
