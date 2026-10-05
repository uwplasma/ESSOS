# tests/test_surfaces.py
import math
import pytest
import jax
import jax.numpy as jnp

# --- import subject under test ---
from essos.surfaces import (
    SurfaceRZFourier,
    B_on_surface,
    BdotN,
    BdotN_over_B,
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

def make_circular_torus_surface(R0=10.0, a=2.0, nfp=1, ntheta=64, nphi=48):
    """Circular torus from the (m=0,n=0) and (m=1,n=0) modes:
    rc = [R0, a], zs = [0, a] with mpol=1, ntor=0. An open (close=False) grid
    keeps the trapezoidal averages spectrally accurate.
    """
    rc = jnp.array([R0, a])
    zs = jnp.array([0.0, a])
    return SurfaceRZFourier(rc, zs, nfp=nfp, mpol=1, ntor=0,
                            ntheta=ntheta, nphi=nphi, close=False)

# -------------------------------------------------------------------------
# Mock field for B_on_surface / BdotN tests
# -------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class ConstBzField:
    """Simple mock field with B = (0,0,B0) everywhere (in Cartesian).
    Registered as a (leafless) pytree because BdotN/B_on_surface are jitted."""

    def tree_flatten(self):
        return (), self.B0

    @classmethod
    def tree_unflatten(cls, B0, children):
        return cls(B0)

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

    theta_2d, phi_2d = surf.theta2d, surf.phi2d
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

@pytest.mark.xfail(strict=True, reason=(
    "SurfaceRZFourier.mean_cross_sectional_area uses the simsopt formula, which assumes "
    "derivatives w.r.t. quadrature points in [0, 1]; ESSOS gammadash_* are per radian, "
    "so the result is low by (2*pi)**2."))
def test_mean_cross_section_area_matches_pi_a2():
    R0, a = 8.0, 1.5
    surf = make_circular_torus_surface(R0=R0, a=a, nfp=1, ntheta=96, nphi=64)
    # For a circular torus, average poloidal cross-sectional area is π a^2
    area = surf.mean_cross_sectional_area()
    assert jnp.allclose(area, math.pi * a * a, rtol=2e-3, atol=2e-3)  # allow slight discretization error

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
