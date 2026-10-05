import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from essos.surfaces import SurfaceRZFourier

RC = jnp.array([1., .05, .01, .02, .03, .3, .04, -.02])
ZS = jnp.array([0., .04, .01, -.01, .02, .3, .03, .01])


@pytest.mark.parametrize("close", [False, True])
def test_circular_torus_volume_and_area(close):
    R0, a = 1.0, 0.3
    s = SurfaceRZFourier(jnp.array([R0, a]), jnp.array([0.0, a]), nfp=1, mpol=1, ntor=0,
                         ntheta=32, nphi=32, close=close)
    np.testing.assert_allclose(s.volume, 2 * np.pi**2 * R0 * a**2, rtol=1e-12)
    np.testing.assert_allclose(s.area, 4 * np.pi**2 * R0 * a, rtol=1e-12)


@pytest.mark.parametrize("range_torus,close", [("full torus", False), ("full torus", True),
                                               ("half period", True)])
def test_shaped_surface_integrals_do_not_depend_on_the_grid(range_torus, close):
    # Volume from fine constant-phi sections integrated over phi.
    nfp = 3
    s = SurfaceRZFourier(RC, ZS, nfp=nfp, mpol=1, ntor=2, ntheta=32, nphi=32, close=close,
                         range_torus=range_torus)
    xm, xn = np.asarray(s.xm), np.asarray(s.xn)
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    phis = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    volume = 0.0
    for phi in phis:
        angle = xm[:, None] * theta - xn[:, None] * phi
        R = np.asarray(RC) @ np.cos(angle)
        dZ_dtheta = (np.asarray(ZS) * xm) @ np.cos(angle)
        # Per section: dV = (integral R dA) dphi = (1/2) oint R^2 dZ dphi
        volume += abs(np.mean(0.5 * R**2 * dZ_dtheta)) * 2 * np.pi * (2 * np.pi / phis.size)
    np.testing.assert_allclose(s.volume, volume, rtol=1e-8)
    fine = SurfaceRZFourier(RC, ZS, nfp=nfp, mpol=1, ntor=2, ntheta=128, nphi=256, close=False)
    np.testing.assert_allclose(s.area, fine.area, rtol=1e-10)
