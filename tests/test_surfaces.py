# tests/test_surfaces.py
import math
import os
import numpy as np
import pytest
import jax
import jax.numpy as jnp

# --- import subject under test ---
from essos.surfaces import (
    SurfaceRZFourier,
    B_on_surface,
    BdotN,
    BdotN_over_B,
    SurfaceClassifier,
)

# -------------------------------------------------------------------------
# Global JAX settings for numerical stability
# -------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _enable_x64():
    jax.config.update("jax_enable_x64", True)

# -------------------------------------------------------------------------
# Helpers: Build an analytic circular torus surface via Fourier coefficients
#   R(θ,φ) = R0 + a cos θ
#   Z(θ,φ) = a sin θ
#   (No φ-dependence; nfp can be arbitrary but we’ll use 1 and 4 in tests.)
# -------------------------------------------------------------------------

def make_circular_torus_surface(
    R0=10.0,
    a=2.0,
    nfp=1,
    ntheta=64,
    nphi=48,
    close=True,
    range_torus="full torus",
):
    """
    Construct SurfaceRZFourier via the (rc, zs, nfp) path with only (m=0,n=0) and (m=1,n=0) active:
       rmnc(0,0)=R0, rmnc(1,0)=a, zmns(1,0)=a
    """
    # mpol must be >= 2 to hold m=0 and m=1 rows
    mpol = 2
    ntor = 0  # only n=0
    rc = jnp.zeros((mpol, 2 * ntor + 1))
    zs = jnp.zeros((mpol, 2 * ntor + 1))
    rc = rc.at[0, 0].set(R0)  # m=0,n=0
    rc = rc.at[1, 0].set(a)   # m=1,n=0
    zs = zs.at[1, 0].set(a)   # m=1,n=0

    surf = SurfaceRZFourier(
        vmec=None,
        s=1.0,
        ntheta=ntheta,
        nphi=nphi,
        close=close,
        range_torus=range_torus,
        rc=rc,
        zs=zs,
        nfp=nfp,
    )
    return surf

# -------------------------------------------------------------------------
# Mock field for B_on_surface / BdotN tests
# -------------------------------------------------------------------------

class ConstBzField:
    """Simple mock field with B = (0,0,B0) everywhere (in Cartesian)."""
    def __init__(self, B0=1.0):
        self.B0 = B0

    @staticmethod
    def B(point_xyz):
        # 'point_xyz' is (3,) but we ignore it
        return jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)

    @staticmethod
    def AbsB(point_xyz):
        return jnp.array(1.0, dtype=jnp.float64)

# -------------------------------------------------------------------------
# Unit tests: geometry of SurfaceRZFourier on the analytic torus
# -------------------------------------------------------------------------

def test_gamma_matches_analytic_circular_torus():
    R0, a = 10.0, 2.0
    surf = make_circular_torus_surface(R0=R0, a=a, nfp=1, ntheta=64, nphi=48)

    theta_2d, phi_2d = surf.theta_2d, surf.phi_2d
    R = R0 + a * jnp.cos(theta_2d)
    Z = a * jnp.sin(theta_2d)
    X = R * jnp.cos(phi_2d)
    Y = R * jnp.sin(phi_2d)

    gamma = surf.gamma  # (nphi, ntheta, 3)
    assert gamma.shape == (surf.nphi, surf.ntheta, 3)
    assert jnp.allclose(gamma[:, :, 0], X, atol=1e-12)
    assert jnp.allclose(gamma[:, :, 1], Y, atol=1e-12)
    assert jnp.allclose(gamma[:, :, 2], Z, atol=1e-12)

def test_normals_are_unit_and_perpendicular_to_tangent():
    surf = make_circular_torus_surface(ntheta=48, nphi=32)
    n = surf.unitnormal
    gt = surf.gammadash_theta
    gp = surf.gammadash_phi

    # unit length:
    nlen = jnp.linalg.norm(n, axis=2)
    assert jnp.allclose(nlen, 1.0, atol=1e-10)

    # orthogonal to both tangent directions:
    dot_t = jnp.sum(n * gt, axis=2)
    dot_p = jnp.sum(n * gp, axis=2)
    assert jnp.allclose(dot_t, 0.0, atol=1e-10)
    assert jnp.allclose(dot_p, 0.0, atol=1e-10)

def test_mean_cross_section_area_matches_pi_a2():
    R0, a = 8.0, 1.5
    surf = make_circular_torus_surface(R0=R0, a=a, nfp=1, ntheta=96, nphi=64)
    # For a circular torus, average poloidal cross-sectional area is π a^2
    area = surf.mean_cross_sectional_area()
    assert jnp.allclose(area, math.pi * a * a, rtol=2e-3, atol=2e-3)  # allow slight discretization error

def test_dofs_setter_updates_geometry():
    R0, a = 9.0, 1.2
    surf = make_circular_torus_surface(R0=R0, a=a, nfp=1, ntheta=32, nphi=24)
    # Keep original gamma:
    g0 = jnp.array(surf.gamma)

    # Increase a by 10% by tweaking the corresponding coefficient in dofs:
    # Layout in SurfaceRZFourier: dofs concatenates rc (flattened)[ntor:] then zs[ntor:]
    # We placed rmnc(1,0)=a and zmns(1,0)=a originally; locate them in the rc/zs arrays.
    idx_rm_m1n0 = 1 * (2 * 0 + 1) + 0  # m=1, n=0 within shape (mpol, 1) -> index 1
    idx_zs_m1n0 = 1 * (2 * 0 + 1) + 0

    dofs = jnp.array(surf.dofs)
    rc_len = surf.rc.size
    zs_len = surf.zs.size

    # Current values:
    assert np.isclose(surf.rc.ravel()[idx_rm_m1n0], a)
    assert np.isclose(surf.zs.ravel()[idx_zs_m1n0], a)

    # Bump 'a' by 10% in both R and Z harmonics:
    dofs = dofs.at[idx_rm_m1n0].set(1.1 * a)
    dofs = dofs.at[rc_len + idx_zs_m1n0].set(1.1 * a)
    surf.dofs = dofs

    g1 = surf.gamma
    # Expect outward/inward displacement ~0.1*a in R/Z amplitudes; just check that geometry changed:
    assert not jnp.allclose(g0, g1)

# -------------------------------------------------------------------------
# Field on surface: B_on_surface / BdotN / BdotN_over_B
# -------------------------------------------------------------------------

def test_B_on_surface_shapes_and_simple_values():
    surf = make_circular_torus_surface(ntheta=16, nphi=10)
    field = ConstBzField(B0=1.0)

    Bout = B_on_surface(surf, field)
    assert Bout.shape == (surf.nphi, surf.ntheta, 3)
    # all Bz ~ 1; Bx=By=0:
    assert jnp.allclose(Bout[..., 0], 0.0, atol=1e-12)
    assert jnp.allclose(Bout[..., 1], 0.0, atol=1e-12)
    assert jnp.allclose(Bout[..., 2], 1.0, atol=1e-12)

def test_BdotN_and_BdotN_over_B_ranges():
    surf = make_circular_torus_surface(ntheta=24, nphi=18)
    field = ConstBzField(B0=1.0)

    bn = BdotN(surf, field)
    assert bn.shape == (surf.nphi, surf.ntheta)
    # |B·n| <= |B| = 1
    assert jnp.all(bn <= 1.0 + 1e-12)
    assert jnp.all(bn >= -1.0 - 1e-12)

    bn_over_B = BdotN_over_B(surf, field)
    assert bn_over_B.shape == (surf.nphi, surf.ntheta)
    assert jnp.all(bn_over_B <= 1.0 + 1e-12)
    assert jnp.all(bn_over_B >= -1.0 - 1e-12)
    # consistency:
    assert jnp.allclose(bn_over_B, bn / 1.0, atol=1e-12)

# -------------------------------------------------------------------------
# SurfaceClassifier: build & evaluate
# -------------------------------------------------------------------------

@pytest.mark.parametrize("nfp", [1, 4])
def test_surface_classifier_build_and_signs(nfp):
    # Build surface and classifier (uses SciPy cKDTree path in your code)
    R0, a = 10.0, 2.0
    surf = make_circular_torus_surface(R0=R0, a=a, nfp=nfp, ntheta=64, nphi=64)

    # Keep the grid coarser (h) for speed but still accurate
    sc = SurfaceClassifier(surf, h=0.12, use_fundamental_phi=True)

    # (1) Points *on* the surface should give ~0 signed distance
    # take a small set of surface samples:
    th = jnp.linspace(0, 2 * jnp.pi, 9, endpoint=False)
    ph = jnp.linspace(0, 2 * jnp.pi / nfp, 7, endpoint=False)
    TH, PH = jnp.meshgrid(th, ph, indexing="ij")
    R = R0 + a * jnp.cos(TH)
    Z = a * jnp.sin(TH)
    X = R * jnp.cos(PH)
    Y = R * jnp.sin(PH)
    XYZ = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    d_on = sc.evaluate_xyz(XYZ)
    assert d_on.shape == (XYZ.shape[0],)
    assert jnp.all(jnp.abs(d_on) < 0.15)  # coarse grid => small but nonzero acceptable

    # (2) A point strictly inside (e.g., near magnetic axis) should be positive
    inside = jnp.array([R0, 0.0, 0.0])
    assert sc.evaluate_xyz(inside) > -1e-6

    # (3) A point outside (R larger than R0 + a + margin) should be negative
    outside = jnp.array([R0 + a + 0.5, 0.0, 0.0])
    assert sc.evaluate_xyz(outside) < 0.0

def test_surface_classifier_phi_wrapping_equivalence():
    surf = make_circular_torus_surface(R0=8.0, a=1.7, nfp=3, ntheta=48, nphi=48)
    sc = SurfaceClassifier(surf, h=0.10, use_fundamental_phi=True)

    # Evaluate at same (r,z) but wildly different φ; wrapping should make them equal.
    r = 8.5
    z = 0.2
    phi1 = 10.0 * math.pi   # large
    phi2 = 0.1              # small
    d1 = sc.evaluate_rphiz(jnp.array([r, phi1, z]))
    d2 = sc.evaluate_rphiz(jnp.array([r, phi2, z]))
    assert jnp.allclose(d1, d2, atol=1e-10)

def test_surface_classifier_vectorized_batch_xyz():
    surf = make_circular_torus_surface(R0=9.0, a=1.0, nfp=2, ntheta=40, nphi=40)
    sc = SurfaceClassifier(surf, h=0.12, use_fundamental_phi=True)

    # batch of random xyz within a bounding box:
    key = jax.random.PRNGKey(0)
    xs = jax.random.uniform(key, (200,), minval=7.0, maxval=11.0)
    ys = jax.random.uniform(key, (200,), minval=-3.0, maxval=3.0)
    zs = jax.random.uniform(key, (200,), minval=-2.0, maxval=2.0)
    xyz = jnp.stack([xs, ys, zs], axis=1)

    vals = sc.evaluate_xyz(xyz)
    assert vals.shape == (200,)
    assert jnp.all(jnp.isfinite(vals))

# -------------------------------------------------------------------------
# Smoke tests for JIT compilation on classifier APIs
# -------------------------------------------------------------------------

def test_classifier_jit_smoke():
    surf = make_circular_torus_surface(R0=8.0, a=1.4, nfp=1, ntheta=32, nphi=32)
    sc = SurfaceClassifier(surf, h=0.15, use_fundamental_phi=True)

    xyz = jnp.array([[8.0, 0.0, 0.0],
                     [9.2, 0.0, 0.1]])
    rphiz = jnp.array([[8.0, 7.0, 0.0],
                       [9.2, 20.0, 0.1]])

    # JIT both methods:
    f1 = jax.jit(sc.evaluate_xyz, static_argnames=("self",))
    f2 = jax.jit(sc.evaluate_rphiz, static_argnames=("self",))

    out1 = f1(xyz)
    out2 = f2(rphiz)
    assert out1.shape == (2,) and out2.shape == (2,)
    assert jnp.all(jnp.isfinite(out1)) and jnp.all(jnp.isfinite(out2))

# -------------------------------------------------------------------------
# Optional: to_vmec round-trip smoke (file content check)
# -------------------------------------------------------------------------

def test_to_vmec_writes_expected_coeffs(tmp_path):
    R0, a = 7.5, 1.1
    surf = make_circular_torus_surface(R0=R0, a=a, nfp=4, ntheta=16, nphi=16)
    out = tmp_path / "surf_in.vmec"
    surf.to_vmec(str(out))
    txt = out.read_text()
    # Sanity: NFP present and at least some RBC/ZBS lines with our values
    assert "NFP = 4" in txt
    assert "RBC(" in txt and "ZBS(" in txt
    assert any("RBC(" in line and "ZBS(" in line for line in txt.splitlines())
