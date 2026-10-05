import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from essos.surfaces import SurfaceRZFourier


def _polygon_area(R, Z):
    return 0.5 * abs(np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z))


@pytest.mark.parametrize("close", [False, True])
def test_mean_cross_sectional_area_elliptic_torus(close):
    R0, a, b = 3.0, 0.7, 0.4
    surface = SurfaceRZFourier(jnp.array([R0, a]), jnp.array([0.0, b]), nfp=1, mpol=1, ntor=0,
                               ntheta=40, nphi=30, close=close)
    np.testing.assert_allclose(surface.mean_cross_sectional_area(), np.pi * a * b, rtol=1e-12)


@pytest.mark.parametrize("close", [False, True])
def test_mean_cross_sectional_area_matches_polygon_cross_sections(close):
    # Shaped, toroidally varying boundary: compare with the shoelace area of
    # finely sampled constant-phi cross sections averaged over the same phi grid.
    rc = jnp.array([3., .12, .03, .6, .07, .02, -.08, .04])
    zs = jnp.array([0., .09, -.02, .65, .06, .01, .07, -.03])
    nfp, nphi = 2, 9
    surface = SurfaceRZFourier(rc, zs, nfp=nfp, mpol=2, ntor=1, ntheta=64, nphi=nphi, close=close)
    phis = np.asarray(surface.phi2d[:, 0])
    if close:
        phis = phis[:-1]
    theta = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    xm, xn = np.asarray(surface.xm), np.asarray(surface.xn)
    areas = []
    for phi in phis:
        angle = xm[:, None] * theta[None, :] - xn[:, None] * phi
        R = np.asarray(rc) @ np.cos(angle)
        Z = np.asarray(zs) @ np.sin(angle)
        areas.append(_polygon_area(R, Z))
    np.testing.assert_allclose(surface.mean_cross_sectional_area(), np.mean(areas), rtol=1e-6)
