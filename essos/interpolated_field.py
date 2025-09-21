"""
InterpolatedField (JAX)
-----------------------

A fast, jittable 3‑D piecewise‑polynomial interpolator on a regular grid, modeled
on the SIMSOPT C++ InterpolatedField/RegularGridInterpolant3D classes.

Features
- Uniform or Chebyshev nodes per cell (Lagrange polynomials of degree d per axis)
- Optional domain mask (skip function) to avoid filling out‑of‑plasma regions
- Cylindrical (r,phi,z) grid with nfp periodicity and optional stellarator symmetry (z<0)
- Vector‑valued interpolation (value_size = 3 by default for B or ∇|B|)
- Batch evaluation and batched coefficient building
- Fully JIT‑able with Equinox; uses pure JAX (no Python loops at runtime)
- Exposes convenient wrappers to evaluate Cartesian B from an underlying field
  that returns B(x,y,z) in Cartesian.

Dependencies: jax, equinox (eqx)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax

# --------------------------------------------------------------------------------------
# Utility: Lagrange basis with precomputed nodes and scalings (barycentric‑like product)
# --------------------------------------------------------------------------------------


class InterpolationRule(eqx.Module):
    degree: int
    nodes: jnp.ndarray  # (d+1,)
    scalings: jnp.ndarray  # (d+1,)

    def basis(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return p_i(x) for i=0..d as shape (d+1, *x.shape). Vectorized over x.
        p_i(x) = (∏_{k!=i} (x - nodes[k])) * scalings[i]
        Note: x is in the *cell local* [0,1] coordinate.
        """
        d = self.degree
        # Evaluate all (x - nodes[k]) for broadcasting: (d+1, *xshape)
        diffs = x[None, ...] - self.nodes[:, None]
        # For each i, product over k!=i. We compute total product then divide by (x - nodes[i]).
        prod_all = jnp.prod(diffs, axis=0)  # (*xshape,)
        # Guard division by zero when x equals a node: use polynomial limit via L'Hôpital with one‑hot mask
        def single_pi(i):
            di = diffs[i]
            # Where di==0, p_i(x) should be 1 and others 0. Implement stable selection:
            # base formula (prod_all / di) * scalings[i]
            base = (prod_all / di) * self.scalings[i]
            # exact node selection
            at_node = (di == 0)
            return jnp.where(at_node, jnp.ones_like(base), base)

        pis = jax.vmap(single_pi)(jnp.arange(d + 1))  # (d+1,*xshape)
        # When x equals nodes[j], *only* j‑th basis should be 1; others 0.
        # Enforce explicitly:
        # Find any node hit
        hits = (diffs == 0)
        any_hit = jnp.any(hits, axis=0)
        if pis.ndim == 1:
            # scalar x
            if any_hit:  # type: ignore
                j = jnp.argmax(hits)
                pis = jax.nn.one_hot(j, d + 1)
        else:
            # broadcasted x
            j = jnp.argmax(jnp.where(hits, 1, 0), axis=0)
            pis = jnp.where(
                any_hit[None, ...],
                jax.nn.one_hot(j, d + 1)[...].swapaxes(0, -1).reshape((d + 1,) + x.shape),
                pis,
            )
        return pis


class UniformInterpolationRule(InterpolationRule):
    def __init__(self, degree: int):
        nodes = jnp.linspace(0.0, 1.0, degree + 1)
        # barycentric‑like scalings: ∏_{k≠i} 1/(x_i - x_k)
        diffs = nodes[:, None] - nodes[None, :]
        scalings = jnp.prod(jnp.where(jnp.eye(degree + 1, dtype=bool), 1.0, 1.0 / diffs), axis=1)
        super().__init__(degree=degree, nodes=nodes, scalings=scalings)


class ChebyshevInterpolationRule(InterpolationRule):
    def __init__(self, degree: int):
        # map Chebyshev nodes from [-1,1] to [0,1]
        k = jnp.arange(degree + 1)
        nodes = 0.5 * (1.0 - jnp.cos(math.pi * k / degree)) if degree > 0 else jnp.array([0.0])
        diffs = nodes[:, None] - nodes[None, :]
        scalings = jnp.prod(jnp.where(jnp.eye(degree + 1, dtype=bool), 1.0, 1.0 / diffs), axis=1)
        super().__init__(degree=degree, nodes=nodes, scalings=scalings)


# --------------------------------------------------------------------------------------
# RegularGridInterpolant3D: vector‑valued values at (d+1)^3 dofs per cell
# --------------------------------------------------------------------------------------


@dataclass
class GridSpec:
    r_range: Tuple[float, float, int]  # (rmin, rmax, nr_cells)
    phi_range: Tuple[float, float, int]  # (phimin, phimax, nphi_cells)
    z_range: Tuple[float, float, int]  # (zmin, zmax, nz_cells)
    value_size: int = 3


class RegularGridInterpolant3D(eqx.Module):
    rule: InterpolationRule
    grid: GridSpec
    extrapolate: bool

    # Precomputed mesh params (static)
    rmin: float = eqx.static_field()
    rmax: float = eqx.static_field()
    phimin: float = eqx.static_field()
    phimax: float = eqx.static_field()
    zmin: float = eqx.static_field()
    zmax: float = eqx.static_field()
    nr: int = eqx.static_field()
    nphi: int = eqx.static_field()
    nz: int = eqx.static_field()
    hr: float = eqx.static_field()
    hphi: float = eqx.static_field()
    hz: float = eqx.static_field()

    # Reduced DOFs (kept points) in tensor grid ordering; domain masking handled via skip_mask
    r_dofs: jnp.ndarray  # (Nd,)
    phi_dofs: jnp.ndarray  # (Nd,)
    z_dofs: jnp.ndarray  # (Nd,)
    dof_is_kept: jnp.ndarray  # (Nfull,) boolean mask for tensor grid dofs
    dof_full2reduced: jnp.ndarray  # (Nfull,) int
    dof_reduced2full: jnp.ndarray  # (Nd,) int

    # For fast local assembly: for every cell, the flattened indices (in reduced‑dof space)
    # of its (d+1)^3 corner DOFs, with -1 for skipped DOFs (not used but kept for shape)
    cell_dof_idx: jnp.ndarray  # (nr*nphi*nz, (d+1)^3)
    skip_cell: jnp.ndarray  # (nr*nphi*nz,) bool

    # Interpolated values at reduced DOFs
    vals: jnp.ndarray  # (Nd, value_size)

    def __init__(
        self,
        rule: InterpolationRule,
        grid: GridSpec,
        extrapolate: bool,
        skip_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    ):
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "extrapolate", extrapolate)

        rmin, rmax, nr = grid.r_range
        phimin, phimax, nphi = grid.phi_range
        zmin, zmax, nz = grid.z_range
        hr = (rmax - rmin) / nr
        hphi = (phimax - phimin) / nphi
        hz = (zmax - zmin) / nz
        object.__setattr__(self, "rmin", rmin)
        object.__setattr__(self, "rmax", rmax)
        object.__setattr__(self, "phimin", phimin)
        object.__setattr__(self, "phimax", phimax)
        object.__setattr__(self, "zmin", zmin)
        object.__setattr__(self, "zmax", zmax)
        object.__setattr__(self, "nr", nr)
        object.__setattr__(self, "nphi", nphi)
        object.__setattr__(self, "nz", nz)
        object.__setattr__(self, "hr", hr)
        object.__setattr__(self, "hphi", hphi)
        object.__setattr__(self, "hz", hz)

        d = rule.degree
        # 1D DOF locations for each axis: (n_cells*d + 1)
        r_dof_1d = jnp.concatenate(
            [rmin + (i * hr + rule.nodes * hr) for i in range(nr)] + [jnp.array([rmax])]
        )
        phi_dof_1d = jnp.concatenate(
            [phimin + (j * hphi + rule.nodes * hphi) for j in range(nphi)] + [jnp.array([phimax])]
        )
        z_dof_1d = jnp.concatenate(
            [zmin + (k * hz + rule.nodes * hz) for k in range(nz)] + [jnp.array([zmax])]
        )
        # Full tensor grid of DOFs
        R, P, Z = jnp.meshgrid(r_dof_1d, phi_dof_1d, z_dof_1d, indexing="ij")
        Rf = R.reshape(-1)
        Pf = P.reshape(-1)
        Zf = Z.reshape(-1)
        Nfull = Rf.size

        # Domain mask from skip_fn evaluated on mesh nodes; keep dof if any adjacent cell may use it
        if skip_fn is None:
            keep_dof = jnp.ones((Nfull,), dtype=bool)
        else:
            keep_dof = ~skip_fn(Rf, Pf, Zf)

        dof_full2reduced = jnp.cumsum(keep_dof.astype(jnp.int32)) - 1
        Nd = int(keep_dof.sum())
        dof_reduced2full = jnp.nonzero(keep_dof, size=Nd, fill_value=0)[0]
        r_dofs = Rf[keep_dof]
        phi_dofs = Pf[keep_dof]
        z_dofs = Zf[keep_dof]

        # Build per‑cell mapping to its local (d+1)^3 DOFs. Also mark entirely skipped cells
        def cell_map(i, j, k):
            # local 1D indices in the full dof grid
            ii = jnp.arange(i * d, i * d + d + 1)
            jj = jnp.arange(j * d, j * d + d + 1)
            kk = jnp.arange(k * d, k * d + d + 1)
            # convert 1D indices to full tensor DOF index
            def idx_full(a, b, c):
                return (
                    a * (phi_dof_1d.size) * (z_dof_1d.size)
                    + b * (z_dof_1d.size)
                    + c
                )

            A, B, C = jnp.meshgrid(ii, jj, kk, indexing="ij")
            full_idx = idx_full(A, B, C).reshape(-1)  # ((d+1)^3,)
            kept = keep_dof[full_idx]
            # A cell is skipped iff *all* its corner DOFs are skipped
            skip_this = ~jnp.any(kept)
            # Map to reduced indices; put -1 for skipped DOFs (unused in eval)
            red = jnp.where(kept, dof_full2reduced[full_idx], -1)
            return red, skip_this

        cell_idx_list = []
        cell_skip_list = []
        for i in range(nr):
            for j in range(nphi):
                for k in range(nz):
                    red, sk = cell_map(i, j, k)
                    cell_idx_list.append(red)
                    cell_skip_list.append(sk)
        cell_dof_idx = jnp.stack(cell_idx_list, axis=0)  # (ncells, (d+1)^3)
        skip_cell = jnp.stack(cell_skip_list, axis=0)

        vals = jnp.zeros((Nd, grid.value_size))

        object.__setattr__(self, "r_dofs", r_dofs)
        object.__setattr__(self, "phi_dofs", phi_dofs)
        object.__setattr__(self, "z_dofs", z_dofs)
        object.__setattr__(self, "dof_is_kept", keep_dof)
        object.__setattr__(self, "dof_full2reduced", dof_full2reduced)
        object.__setattr__(self, "dof_reduced2full", dof_reduced2full)
        object.__setattr__(self, "cell_dof_idx", cell_dof_idx)
        object.__setattr__(self, "skip_cell", skip_cell)
        object.__setattr__(self, "vals", vals)

    # --------------------------- build (interpolation) ---------------------------

    def build(self, fbatch: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        """Fill coefficients by evaluating fbatch at reduced DOFs.
        fbatch must map (r_vec, phi_vec, z_vec) -> (Nd, value_size).
        """
        vals = fbatch(self.r_dofs, self.phi_dofs, self.z_dofs)
        return eqx.tree_at(lambda s: s.vals, self, vals)

    # --------------------------- evaluation helpers -----------------------------

    def _locate_cell_and_local(self, r, phi, z):
        # clamp (or leave) to domain
        if self.extrapolate:
            rc = r
            pc = phi
            zc = z
        else:
            rc = jnp.clip(r, self.rmin, self.rmax - 1e-15)
            pc = jnp.clip(phi, self.phimin, self.phimax - 1e-15)
            zc = jnp.clip(z, self.zmin, self.zmax - 1e-15)
        # integer cell indices
        ir = jnp.floor((rc - self.rmin) / self.hr).astype(jnp.int32)
        ip = jnp.floor((pc - self.phimin) / self.hphi).astype(jnp.int32)
        iz = jnp.floor((zc - self.zmin) / self.hz).astype(jnp.int32)
        ir = jnp.clip(ir, 0, self.nr - 1)
        ip = jnp.clip(ip, 0, self.nphi - 1)
        iz = jnp.clip(iz, 0, self.nz - 1)
        # local coords in [0,1]
        xr = (rc - (self.rmin + ir * self.hr)) / self.hr
        xp = (pc - (self.phimin + ip * self.hphi)) / self.hphi
        xz = (zc - (self.zmin + iz * self.hz)) / self.hz
        cell_idx = (ir * self.nphi * self.nz) + (ip * self.nz) + iz
        return cell_idx, xr, xp, xz

    def _eval_in_cell(self, cell_idx: jnp.ndarray, xr, xp, xz) -> jnp.ndarray:
        # Fetch local coefficient block (flattened) for this cell
        d = self.rule.degree
        local_idx = self.cell_dof_idx[cell_idx]  # ((d+1)^3,)
        # Gather coefficients: ( (d+1)^3, value_size )
        local_vals = jnp.where(
            (local_idx[:, None] >= 0),
            self.vals[jnp.maximum(local_idx, 0)],
            0.0,
        )
        # Basis on each axis: (d+1,)
        br = self.rule.basis(xr)
        bp = self.rule.basis(xp)
        bz = self.rule.basis(xz)
        # Tensor multiply: sum_{a,b,c} br[a]*bp[b]*bz[c]*V[a,b,c,:]
        # Reshape to (d+1,d+1,d+1,val)
        V = local_vals.reshape((d + 1, d + 1, d + 1, self.grid.value_size))
        tmp = jnp.tensordot(br, V, axes=[[0], [0]])  # (d+1, d+1, val)
        tmp = jnp.tensordot(bp, tmp, axes=[[0], [0]])  # (d+1, val)
        out = jnp.tensordot(bz, tmp, axes=[[0], [0]])  # (val,)
        return out

    def evaluate_batch(self, rphiz: jnp.ndarray) -> jnp.ndarray:
        """Evaluate interpolant at a batch of N points (r,phi,z).
        rphiz: (N,3) -> returns (N, value_size)
        """
        def one(p):
            r, phi, z = p
            cell_idx, xr, xp, xz = self._locate_cell_and_local(r, phi, z)
            return self._eval_in_cell(cell_idx, xr, xp, xz)

        return jax.vmap(one)(rphiz)


# --------------------------------------------------------------------------------------
# Cylindrical symmetry helpers and top‑level InterpolatedField API
# --------------------------------------------------------------------------------------


def _reduce_by_symmetry(rphiz: jnp.ndarray, nfp: int, stellsym: bool):
    """Map points into fundamental domain; remember flips for later component fixes.
    Returns (rphiz_sym, flags) where flags=bool array whether z<0 reflection was used.
    """
    r = rphiz[:, 0]
    phi = rphiz[:, 1]
    z = rphiz[:, 2]

    period = (2.0 * jnp.pi) / nfp
    # mod phi to [0,period)
    k = jnp.floor(phi / period)
    phi_mod = phi - k * period

    if stellsym:
        reflect = z < 0.0
        z_mod = jnp.where(reflect, -z, z)
        phi_mod = jnp.where(reflect, 2 * jnp.pi - phi_mod, phi_mod)
        # re‑mod to [0,period)
        k2 = jnp.floor(phi_mod / period)
        phi_mod = phi_mod - k2 * period
    else:
        reflect = jnp.zeros_like(z, dtype=bool)
        z_mod = z

    r_sym = jnp.stack([r, phi_mod, z_mod], axis=1)
    return r_sym, reflect


def _apply_symmetry_to_B_cyl(Bcyl: jnp.ndarray, reflect: jnp.ndarray) -> jnp.ndarray:
    # If reflected (z<0), flip radial component (matches C++ apply_symmetries_to_B_cyl)
    Br, Bp, Bz = Bcyl.T
    Br = jnp.where(reflect, -Br, Br)
    return jnp.stack([Br, Bp, Bz], axis=1)


def _apply_symmetry_to_GradAbsB_cyl(grad: jnp.ndarray, reflect: jnp.ndarray) -> jnp.ndarray:
    # If reflected, flip phi and z components (matches C++ apply_symmetries_to_GradAbsB_cyl)
    Gr, Gp, Gz = grad.T
    Gp = jnp.where(reflect, -Gp, Gp)
    Gz = jnp.where(reflect, -Gz, Gz)
    return jnp.stack([Gr, Gp, Gz], axis=1)


def _cyl_to_cart_vectors(phi: jnp.ndarray, vec_cyl: jnp.ndarray) -> jnp.ndarray:
    """Rotate cylindrical vector to Cartesian at given phi for each point.
    vec_cyl: (N,3)
    returns (N,3)
    """
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    Br, Bp, Bz = vec_cyl.T
    Bx = c * Br - s * Bp
    By = s * Br + c * Bp
    return jnp.stack([Bx, By, Bz], axis=1)


def _cart_to_cyl_vectors(phi: jnp.ndarray, vec_xyz: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    Bx, By, Bz = vec_xyz.T
    Br = c * Bx + s * By
    Bp = -s * Bx + c * By
    return jnp.stack([Br, Bp, Bz], axis=1)


class InterpolatedField(eqx.Module):
    # configuration
    nfp: int
    stellsym: bool

    # underlying field callable: given (x,y,z) -> (3,) Cartesian
    base_field_cart: Callable[[jnp.ndarray], jnp.ndarray] = eqx.static_field()

    # interpolants in cylindrical space
    interp_B: RegularGridInterpolant3D
    interp_GradAbsB: Optional[RegularGridInterpolant3D]

    # Which parts have been built
    has_B: bool
    has_GradAbsB: bool

    def __init__(
        self,
        base_field_cart: Callable[[jnp.ndarray], jnp.ndarray],
        degree: int,
        rrange: Tuple[float, float, int],
        phirange: Tuple[float, float, int],
        zrange: Tuple[float, float, int],
        extrapolate: bool = True,
        nfp: int = 1,
        stellsym: bool = False,
        skip_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
        use_chebyshev: bool = False,
        build_gradabsb: bool = False,
    ):
        rule = ChebyshevInterpolationRule(degree) if use_chebyshev else UniformInterpolationRule(degree)
        grid = GridSpec(rrange, phirange, zrange, value_size=3)
        interp_B = RegularGridInterpolant3D(rule, grid, extrapolate, skip_fn)
        interp_G = RegularGridInterpolant3D(rule, grid, extrapolate, skip_fn) if build_gradabsb else None
        object.__setattr__(self, "nfp", nfp)
        object.__setattr__(self, "stellsym", stellsym)
        object.__setattr__(self, "base_field_cart", base_field_cart)
        object.__setattr__(self, "interp_B", interp_B)
        object.__setattr__(self, "interp_GradAbsB", interp_G)
        object.__setattr__(self, "has_B", False)
        object.__setattr__(self, "has_GradAbsB", False)

    # --------------------- builders: fill coefficient arrays ---------------------

    def _fbatch_B(self, r: jnp.ndarray, phi: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        # Convert to xyz, call base field, project to cylindrical
        x = r * jnp.cos(phi)
        y = r * jnp.sin(phi)
        pts = jnp.stack([x, y, z], axis=1)
        Bxyz = jax.vmap(self.base_field_cart)(pts)
        Bcyl = _cart_to_cyl_vectors(phi, Bxyz)
        return Bcyl

    def _fbatch_GradAbsB(self, r: jnp.ndarray, phi: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        def absB(pt):
            return jnp.linalg.norm(self.base_field_cart(pt))

        grad_abs = jax.vmap(jax.grad(absB))(jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=1))
        # grad in Cartesian -> convert to cylindrical components
        return _cart_to_cyl_vectors(phi, grad_abs)

    def build_B(self):
        interp = self.interp_B.build(self._fbatch_B)
        return eqx.tree_at(lambda s: (s.interp_B, s.has_B), self, (interp, True))

    def build_GradAbsB(self):
        assert self.interp_GradAbsB is not None, "build_gradabsb=False in constructor"
        interp = self.interp_GradAbsB.build(self._fbatch_GradAbsB)  # type: ignore
        return eqx.tree_at(lambda s: (s.interp_GradAbsB, s.has_GradAbsB), self, (interp, True))

    # --------------------- evaluation on batches of points -----------------------

    @eqx.filter_jit
    def B_cyl(self, rphiz: jnp.ndarray) -> jnp.ndarray:
        """Evaluate B in cylindrical components at (r,phi,z) batch.
        rphiz shape (N,3).
        """
        assert self.has_B, "Coefficients not built; call build_B() first"
        rphiz_sym, reflect = _reduce_by_symmetry(rphiz, self.nfp, self.stellsym)
        Bcyl = self.interp_B.evaluate_batch(rphiz_sym)
        Bcyl = _apply_symmetry_to_B_cyl(Bcyl, reflect)
        return Bcyl

    @eqx.filter_jit
    def GradAbsB_cyl(self, rphiz: jnp.ndarray) -> jnp.ndarray:
        assert self.has_GradAbsB and (self.interp_GradAbsB is not None), "Coefficients not built; call build_GradAbsB() first"
        rphiz_sym, reflect = _reduce_by_symmetry(rphiz, self.nfp, self.stellsym)
        G = self.interp_GradAbsB.evaluate_batch(rphiz_sym)  # type: ignore
        G = _apply_symmetry_to_GradAbsB_cyl(G, reflect)
        return G

    @eqx.filter_jit
    def B_xyz(self, xyz: jnp.ndarray) -> jnp.ndarray:
        """Convenience: evaluate B on Cartesian input batch xyz (N,3).
        Internally convert to cylindrical, call B_cyl, rotate back to Cartesian.
        """
        x, y, z = xyz.T
        r = jnp.sqrt(x * x + y * y)
        phi = jnp.arctan2(y, x)
        rphiz = jnp.stack([r, phi, z], axis=1)
        Bcyl = self.B_cyl(rphiz)
        return _cyl_to_cart_vectors(phi, Bcyl)

    # --------------------- error estimate (RMS, max) ----------------------------

    def estimate_error_B(self, key: jax.Array, nsamples: int = 10_000) -> Tuple[float, float]:
        assert self.has_B, "Coefficients not built; call build_B() first"
        rmin, rmax, _ = self.interp_B.grid.r_range
        pmin, pmax, _ = self.interp_B.grid.phi_range
        zmin, zmax, _ = self.interp_B.grid.z_range
        u = jax.random.uniform(key, (nsamples, 3))
        rphiz = jnp.stack([
            rmin + (rmax - rmin) * u[:, 0],
            pmin + (pmax - pmin) * u[:, 1],
            zmin + (zmax - zmin) * u[:, 2],
        ], axis=1)
        x = rphiz[:, 0] * jnp.cos(rphiz[:, 1])
        y = rphiz[:, 0] * jnp.sin(rphiz[:, 1])
        xyz = jnp.stack([x, y, rphiz[:, 2]], axis=1)
        B_true = jax.vmap(self.base_field_cart)(xyz)
        B_pred = self.B_xyz(xyz)
        diff = jnp.linalg.norm(B_true - B_pred, axis=1)
        rms = jnp.sqrt(jnp.mean(diff**2))
        mx = jnp.max(diff)
        return float(rms), float(mx)
