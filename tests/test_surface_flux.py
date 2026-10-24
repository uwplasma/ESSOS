"""Signed toroidal flux for VMEC export and its coil vector potential."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.surfaces import SurfaceRZFourier, toroidal_flux


class ToroidalField(eqx.Module):
    strength: float = 2.0

    def A(self, xyz):
        # curl(-B R e_z) = B e_phi.
        return jnp.array([0., 0., -self.strength * jnp.hypot(xyz[0], xyz[1])])


def ellipse(a=.7, b=.4, count=12, close=False, range_torus='full torus'):
    return SurfaceRZFourier(jnp.array([3., a]), jnp.array([0., b]),
                            nfp=3, mpol=1, ntor=0, ntheta=count, nphi=count,
                            close=close, range_torus=range_torus)


@pytest.mark.parametrize('close', [False, True])
@pytest.mark.parametrize('count', [5, 9, 17])
@pytest.mark.parametrize('range_torus', ['full torus', 'half period'])
def test_toroidal_flux_analytic_area_and_sign(close, count, range_torus):
    surface = ellipse(count=count, close=close, range_torus=range_torus)
    for idx in (0, 2):
        np.testing.assert_allclose(toroidal_flux(surface, ToroidalField(), idx),
                                   -2 * np.pi * .7 * .4, atol=2e-14)


@pytest.mark.parametrize('close,count', [(False, 1), (True, 2)])
def test_toroidal_flux_requires_distinct_points(close, count):
    with pytest.raises(ValueError, match='distinct theta'):
        toroidal_flux(ellipse(count=count, close=close), ToroidalField())


@pytest.mark.parametrize('close', [False, True])
def test_toroidal_flux_shape_gradient(close):
    def flux(a):
        return toroidal_flux(ellipse(a=a, close=close), ToroidalField())
    derivative = jax.jit(jax.grad(flux))(.7)
    finite_difference = (flux(.7 + 1e-5) - flux(.7 - 1e-5)) / 2e-5
    np.testing.assert_allclose(derivative, -2 * np.pi * .4, rtol=1e-12)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-9)


def test_biot_savart_vector_potential_curl_matches_field():
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    curves = CreateEquallySpacedCurves(2, 2, 3., 1., n_segments=40, nfp=2, stellsym=True)
    field = BiotSavart(Coils(curves, jnp.array([8e4, -12e4])))
    for xyz in (jnp.array([.6, .2, -.3]), jnp.array([2., .4, .1])):
        jacobian = jax.jacfwd(field.A)(xyz)
        curl = jnp.array([jacobian[2, 1] - jacobian[1, 2],
                          jacobian[0, 2] - jacobian[2, 0],
                          jacobian[1, 0] - jacobian[0, 1]])
        np.testing.assert_allclose(curl, field.B(xyz), rtol=2e-13, atol=2e-15)


def test_coil_toroidal_flux_matches_independent_disk_integral():
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart
    # A vertical circular coil surrounds the small elliptical poloidal loop.
    coefficients = jnp.zeros((1, 3, 3)).at[0, 0, 0].set(3.)
    coefficients = coefficients.at[0, 0, 2].set(1.).at[0, 2, 1].set(1.)
    field = BiotSavart(Coils(Curves(coefficients, n_segments=64, stellsym=False), jnp.array([1e5])))
    surface = ellipse(a=.3, b=.2, count=32)
    nodes, weights = np.polynomial.legendre.leggauss(20)
    rho = jnp.asarray((nodes + 1) / 2)[:, None]
    theta = jnp.linspace(0, 2 * jnp.pi, 48, endpoint=False)[None, :]
    x, z = 3 + .3 * rho * jnp.cos(theta), .2 * rho * jnp.sin(theta)
    points = jnp.stack((x, jnp.zeros_like(x), z), axis=-1)
    by = jax.vmap(field.B)(points.reshape(-1, 3))[:, 1].reshape(x.shape)
    # Increasing theta gives a disk normal of -e_phi at phi=0.
    disk_flux = jnp.sum(-by * (.3 * .2 * rho) * jnp.asarray(weights / 2)[:, None]) * (2 * jnp.pi / 48)
    np.testing.assert_allclose(toroidal_flux(surface, field), disk_flux, rtol=1e-12, atol=1e-14)
