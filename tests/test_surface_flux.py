"""Flux line integrals use surface tangents and periodic angular quadrature."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.surfaces import SurfaceRZFourier, poloidal_flux, toroidal_flux


class ToroidalField(eqx.Module):
    strength: float = 2.0

    def A(self, xyz):
        # curl(-B R e_z) = B e_phi.
        return jnp.array([0., 0., -self.strength * jnp.hypot(xyz[0], xyz[1])])


class VerticalField(eqx.Module):
    strength: float = 2.0

    def A(self, xyz):
        return self.strength / 2 * jnp.array([-xyz[1], xyz[0], 0.])


def ellipse(a=.7, b=.4, count=12, close=False, range_torus='full torus'):
    return SurfaceRZFourier(jnp.array([3., a]), jnp.array([0., b]),
                            nfp=3, mpol=1, ntor=0, ntheta=count, nphi=count,
                            close=close, range_torus=range_torus)


@pytest.mark.parametrize('close', [False, True])
@pytest.mark.parametrize('count', [5, 9, 17])
@pytest.mark.parametrize('range_torus', ['full torus', 'half period'])
def test_toroidal_flux_analytic_area_and_sign(close, count, range_torus):
    surface = ellipse(count=count, close=close, range_torus=range_torus)
    field = ToroidalField()
    for idx in (0, 2):
        np.testing.assert_allclose(toroidal_flux(surface, field, idx),
                                   -field.strength * np.pi * .7 * .4, atol=2e-14)
    # Vertical field is tangent to this poloidal plane and contributes no flux.
    np.testing.assert_allclose(toroidal_flux(surface, VerticalField()), 0., atol=2e-14)


@pytest.mark.parametrize('close', [False, True])
@pytest.mark.parametrize('count', [5, 9, 17])
def test_poloidal_flux_uniform_vertical_field(close, count):
    surface = ellipse(count=count, close=close)
    field = VerticalField()
    for idx in (0, 2):
        radius = 3 + .7 * np.cos(float(surface.theta2d[0, idx]))
        np.testing.assert_allclose(poloidal_flux(surface, field, idx),
                                   field.strength * np.pi * radius**2, atol=5e-14)
    np.testing.assert_allclose(poloidal_flux(surface, ToroidalField()), 0., atol=2e-14)


@pytest.mark.parametrize('close', [False, True])
def test_poloidal_flux_nonaxisymmetric_full_loop(close):
    # R = 3 + delta cos(3 phi) + a cos(theta), Z = b sin(theta).
    # The flux is B pi <R**2>_phi, including the delta**2 / 2 term.
    surface = SurfaceRZFourier(jnp.array([3., .2, 0., .7, 0.]),
                               jnp.array([0., 0., 0., .4, 0.]),
                               nfp=3, mpol=1, ntor=1, ntheta=12, nphi=18, close=close)
    np.testing.assert_allclose(poloidal_flux(surface, VerticalField()),
                               2 * np.pi * ((3 + .7)**2 + .2**2 / 2), atol=5e-14)


def test_poloidal_flux_rejects_open_arc():
    with pytest.raises(ValueError, match='full torus'):
        poloidal_flux(ellipse(range_torus='half period'), VerticalField())


@pytest.mark.parametrize('close,count', [(False, 1), (True, 2)])
def test_flux_rejects_insufficient_quadrature(close, count):
    surface = ellipse(count=count, close=close)
    with pytest.raises(ValueError, match='distinct theta'):
        toroidal_flux(surface, ToroidalField())
    with pytest.raises(ValueError, match='distinct phi'):
        poloidal_flux(surface, VerticalField())


@pytest.mark.parametrize('close', [False, True])
def test_flux_shape_gradients_match_analytic_and_finite_differences(close):
    def toroidal(a):
        return toroidal_flux(ellipse(a=a, close=close), ToroidalField())

    def poloidal(a):
        return poloidal_flux(ellipse(a=a, close=close), VerticalField())

    for flux, expected in ((toroidal, -2 * np.pi * .4), (poloidal, 4 * np.pi * (3 + .7))):
        derivative = jax.jit(jax.grad(flux))(.7)
        finite_difference = (flux(.7 + 1e-5) - flux(.7 - 1e-5)) / 2e-5
        np.testing.assert_allclose(derivative, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-9, atol=1e-9)


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


@pytest.mark.parametrize('close', [False, True])
def test_coil_flux_matches_independent_disk_integral_and_current_gradient(close):
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart
    # Circular horizontal coil of radius 2.5 m, surrounding the flux disk.
    coefficients = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(2.5).at[0, 1, 1].set(2.5)
    curves = Curves(coefficients, n_segments=64, stellsym=False)
    field = BiotSavart(Coils(curves, jnp.array([1e5]), currents_scale=1e5))
    surface = SurfaceRZFourier(jnp.array([.8, .2]), jnp.array([0., .1]),
                               nfp=1, mpol=1, ntor=0, ntheta=20, nphi=20, close=close)
    # At theta=0 this is the unit disk in z=0. Integrate Bz over its interior.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    radius = jnp.asarray((nodes + 1) / 2)
    points = jnp.stack((radius, jnp.zeros_like(radius), jnp.zeros_like(radius)), axis=-1)
    disk_flux = 2 * np.pi * jnp.sum(jnp.asarray(weights / 2) * radius * jax.vmap(field.B)(points)[:, 2])
    line_flux = poloidal_flux(surface, field)
    np.testing.assert_allclose(line_flux, disk_flux, rtol=1e-12, atol=1e-14)

    def flux(current):
        return poloidal_flux(surface, BiotSavart(Coils(curves, jnp.atleast_1d(current), currents_scale=1e5)))

    derivative = jax.jit(jax.grad(flux))(1e5)
    finite_difference = (flux(1e5 + 1.) - flux(1e5 - 1.)) / 2
    np.testing.assert_allclose(derivative, line_flux / 1e5, rtol=1e-12)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-9)
