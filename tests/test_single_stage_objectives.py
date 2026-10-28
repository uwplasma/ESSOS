"""Manufactured fields and derivatives for the active single-stage objectives."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coils import Coils, Curves
from essos.fields import BiotSavart, MagneticField
from essos.losses import custom_loss
from essos.objective_functions import (
    loss_BdotN_mean_squared, loss_coil_curvature_from_field, loss_coil_length_max,
    loss_cross_sectional_area_mean, loss_iota_approx, loss_quasi_symmetry,
    quasi_symmetry_residual_on_surface,
)
from essos.surfaces import (BdotN_over_B, B_contravariant_theta_phi_on_surface,
                            SurfaceRZFourier, iota_approx_on_surface)


def torus(a=.4, *, ntheta=24, nphi=12, close=False):
    rc, zs = jnp.array([2., a]), jnp.array([0., a])
    return SurfaceRZFourier(rc, zs, 1, 1, 0, ntheta=ntheta, nphi=nphi, close=close)


class PolynomialField(MagneticField, eqx.Module):
    """Curl-free, divergence-free Cartesian polynomial field."""
    coefficients: jax.Array

    def __init__(self, coefficients):
        self.coefficients = coefficients

    def B(self, x):
        a, b, c = self.coefficients
        return jnp.array([1 + a*x[0] + b*x[1],
                          1 + b*x[0] - a*x[1] + c*x[2], 2 + c*x[1]])


class ToroidalField(MagneticField, eqx.Module):
    amplitude: jax.Array

    def __init__(self, amplitude):
        self.amplitude = amplitude

    def B(self, x):
        return self.amplitude * jnp.array([-x[1], x[0], 0.]) / (x[0]**2+x[1]**2)


class StraightAngleField(eqx.Module):
    iota: jax.Array
    correction: jax.Array
    harmonic: int = eqx.field(static=True, default=2)
    amplitude: float = 1.

    def dAbsB_by_dX(self, x):
        return jax.grad(lambda p: jnp.linalg.norm(self.B(p)))(x)

    def B(self, x):
        R = jnp.hypot(x[0], x[1])
        er = jnp.array([x[0]/R, x[1]/R, 0.])
        etheta = -x[2]*er + jnp.array([0., 0., R-2.])
        ephi = jnp.array([-x[1], x[0], 0.])
        theta = jnp.arctan2(x[2], R-2.)
        dlam = jnp.cos(self.harmonic*theta) * self.harmonic * self.correction
        return self.amplitude * (ephi + self.iota/(1+dlam)*etheta)


class NormalContaminatedStraightAngleField(StraightAngleField):
    """Add a small normal field without changing the tangential flow."""
    normal_strength: jax.Array = 1e-3

    def B(self, x):
        R = jnp.hypot(x[0], x[1])
        normal = jnp.array([(R-2.)*x[0]/R, (R-2.)*x[1]/R, x[2]])
        normal /= jnp.linalg.norm(normal)
        return super().B(x) + self.amplitude*self.normal_strength*normal


def directional_check(fun, x, *, rtol=2e-5, atol=1e-8, step=2e-5):
    direction = jnp.cos(jnp.arange(x.size)+.31).reshape(x.shape)
    direction /= jnp.linalg.norm(direction)
    value, gradient = jax.jit(jax.value_and_grad(fun))(x)
    exact = jnp.vdot(gradient, direction)
    finite = (fun(x+step*direction)-fun(x-step*direction))/(2*step)
    assert value.shape == () and gradient.shape == x.shape
    assert np.isfinite(value) and np.isfinite(gradient).all()
    np.testing.assert_allclose(exact, finite, rtol=rtol, atol=atol)


def test_qs_matches_direct_triple_product_and_field_scaling():
    surface = torus(ntheta=9, nphi=7)
    coefficients = jnp.array([.13, .21, -.17])
    field = PolynomialField(coefficients)
    residual = quasi_symmetry_residual_on_surface(field, surface)
    points = surface.gamma.reshape(-1, 3)
    gradb = jax.vmap(jax.grad(field.AbsB))(points)
    gradparallel = jax.vmap(jax.grad(lambda x: field.B(x) @ jax.grad(field.AbsB)(x)))(points)
    triple = jnp.sum(jnp.cross(surface.unitnormal.reshape(-1, 3), gradb)*gradparallel, axis=-1)
    mean_squared_field = jnp.mean(jax.vmap(lambda x: field.B(x) @ field.B(x))(points))
    expected = surface.rc[0]**3 * triple / mean_squared_field**1.5
    np.testing.assert_allclose(residual.ravel(), expected, rtol=1e-12, atol=1e-14)
    # Scaling all currents by a negative factor reverses the signed residual;
    # the objective is unchanged, also at very small absolute field strength.
    class Scaled(PolynomialField):
        scale: float
        def __init__(self, coefficients, scale):
            self.coefficients = coefficients
            self.scale = scale
        def B(self, x):
            return self.scale * super().B(x)
    for scale in (3., -2., 1e-18):
        actual = quasi_symmetry_residual_on_surface(Scaled(coefficients, scale), surface)
        np.testing.assert_allclose(actual, np.sign(scale)*residual, rtol=2e-12, atol=1e-13)
    directional_check(lambda c: loss_quasi_symmetry(PolynomialField(c), surface), coefficients)


def test_qs_is_independent_of_length_units():
    surface = torus(ntheta=9, nphi=7)
    coefficients = jnp.array([.13, .21, -.17])
    reference = quasi_symmetry_residual_on_surface(PolynomialField(coefficients), surface)

    class SpatiallyScaledField(PolynomialField):
        length_scale: float

        def __init__(self, coefficients, length_scale):
            self.coefficients = coefficients
            self.length_scale = length_scale

        def B(self, x):
            return super().B(x/self.length_scale)

    for scale in (.1, 10.):
        scaled_surface = SurfaceRZFourier(surface.rc*scale, surface.zs*scale,
                                          1, 1, 0, ntheta=9, nphi=7, close=False)
        actual = quasi_symmetry_residual_on_surface(
            SpatiallyScaledField(coefficients, scale), scaled_surface)
        np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=1e-13)


def test_qs_axisymmetry_and_uniform_field_have_finite_gradients():
    surface = torus(ntheta=9, nphi=7)
    assert float(loss_quasi_symmetry(ToroidalField(jnp.array(1.)), surface)) < 1e-25
    zero = jnp.zeros(3)
    assert float(loss_quasi_symmetry(PolynomialField(zero), surface)) == 0
    gradient = jax.grad(lambda c: loss_quasi_symmetry(PolynomialField(c), surface))(zero)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient, 0.)


@pytest.mark.parametrize('objective', [loss_quasi_symmetry, loss_BdotN_mean_squared])
def test_grid_averaging_drops_duplicated_endpoints(objective):
    field = PolynomialField(jnp.array([.13, .21, -.17]))
    values = [objective(field, torus(ntheta=nt, nphi=np_, close=close))
              for nt, np_, close in [(24, 24, False), (25, 25, True), (48, 48, False)]]
    np.testing.assert_allclose(values, values[0], rtol=1e-6)


def test_iota_needs_coordinate_resolution_independent_of_shape():
    surface = torus(ntheta=64, nphi=8)
    field = StraightAngleField(jnp.array(.4), jnp.array(.3))
    # theta* = theta + .3 sin(2 theta), whose Jacobian stays positive.
    # The shape has only m=1. Its default lambda basis gives 0.5, not 0.4.
    np.testing.assert_allclose(iota_approx_on_surface(surface, field), .5, rtol=1e-8)
    for m in (2, 3, 4):
        diagnostics = iota_approx_on_surface(surface, field, lambda_mpol=m, lambda_ntor=0,
                                              return_diagnostics=True)
        np.testing.assert_allclose(diagnostics['iota'], .4, atol=1e-9)
        assert float(diagnostics['rms_residual']) < 1e-9
        np.testing.assert_allclose(diagnostics['minimum_angle_jacobian'], .4, atol=1e-8)
    directional_check(lambda x: iota_approx_on_surface(
        surface, StraightAngleField(x[0], x[1]), lambda_mpol=3, lambda_ntor=0), jnp.array([.4, .3]))
    assert float(loss_iota_approx(field, surface, .4, lambda_mpol=3, lambda_ntor=0)) < 1e-18


def test_iota_is_independent_of_field_sign_and_closed_grid():
    for close in (False, True):
        surface = torus(ntheta=65 if close else 64, nphi=9 if close else 8, close=close)
        for amplitude in (1., -3., 1e-12):
            field = StraightAngleField(jnp.array(.4), jnp.array(.3), amplitude=amplitude)
            np.testing.assert_allclose(iota_approx_on_surface(surface, field, lambda_mpol=3, lambda_ntor=0),
                                       .4, atol=1e-9)


def test_iota_remains_accurate_under_small_normal_contamination():
    surface = torus(ntheta=64, nphi=8)
    tangent = StraightAngleField(jnp.array(.4), jnp.array(.3))
    reference_flow = B_contravariant_theta_phi_on_surface(surface, tangent)
    estimates = []
    for amplitude in (1., -3., 1e-12):
        field = NormalContaminatedStraightAngleField(
            jnp.array(.4), jnp.array(.3), amplitude=amplitude,
            normal_strength=jnp.array(1e-3))
        # The exact projected rotation remains .4: the perturbation is normal.
        np.testing.assert_allclose(B_contravariant_theta_phi_on_surface(surface, field)/amplitude,
                                   reference_flow, rtol=2e-13, atol=2e-14)
        normal_rms = jnp.sqrt(jnp.mean(BdotN_over_B(surface, field)**2))
        assert 1e-4 < float(normal_rms) < 1e-3
        fit = iota_approx_on_surface(surface, field, lambda_mpol=3, lambda_ntor=0,
                                      return_diagnostics=True)
        # Regularization may introduce bias even though the true flow is fixed.
        assert abs(float(fit['iota'])-.4) < 1e-5
        assert float(fit['minimum_angle_jacobian']) > .39
        estimates.append(fit['iota'])
    np.testing.assert_allclose(estimates, estimates[0], rtol=2e-12, atol=1e-13)

    def objective(x):
        field = NormalContaminatedStraightAngleField(x[0], x[1], normal_strength=x[2])
        return loss_iota_approx(field, surface, .35, lambda_mpol=3, lambda_ntor=0)

    # Include the contamination parameter so AD must differentiate the adaptive
    # estimator as well as the projected field. Keep FD away from its ridge floor.
    directional_check(objective, jnp.array([.4, .3, 1e-3]), rtol=3e-5, step=1e-6)


def test_actual_biot_savart_coil_and_surface_directional_derivatives():
    # Poloidal loops give a toroidal field with nonzero B^phi everywhere on
    # this grid, as required by the straight-field-angle fit. A lone loop
    # normal to the radial direction instead has B^phi zeros and reversals.
    curve_dofs = jnp.array([[[2., 0., .8], [0., 0., 0.], [0., .8, 0.]]])
    curves = Curves(curve_dofs, n_segments=12, nfp=4, stellsym=False)
    field = BiotSavart(Coils(curves, jnp.array([1e5])))
    surface = torus(ntheta=6, nphi=5)
    bphi = B_contravariant_theta_phi_on_surface(surface, field)[..., 1]
    assert jnp.all(bphi < 0.)
    assert jnp.min(jnp.abs(bphi)) > .05*jnp.max(jnp.abs(bphi))
    qs = custom_loss(loss_quasi_symmetry, 'field', 'surface')
    geometry = custom_loss(lambda f: loss_coil_length_max(f, 3.)
                           + loss_coil_curvature_from_field(f, 1.), 'field')
    normal = custom_loss(loss_BdotN_mean_squared, 'field', 'surface')
    iota = custom_loss(loss_iota_approx, 'field', 'surface', iota_target=.4,
                       lambda_mpol=2, lambda_ntor=1)
    area = custom_loss(loss_cross_sectional_area_mean, 'surface', target_area=.7)
    combined = qs + .03*geometry + .02*normal + .001*iota + .1*area
    combined.dependencies = {'field': field, 'surface': surface}
    x = combined.starting_dofs
    directional_check(combined, x, rtol=2e-4, step=1e-6)
    np.testing.assert_allclose(combined.grad(x), jax.grad(combined)(x), rtol=2e-10, atol=1e-10)
    # Independently verify B.n/|B| current scaling, including sign reversal.
    reference = BdotN_over_B(surface, field)
    for scale in (-3., 1e-12):
        scaled = BiotSavart(Coils(curves, jnp.array([1e5*scale])))
        np.testing.assert_allclose(BdotN_over_B(surface, scaled), np.sign(scale)*reference, rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(loss_BdotN_mean_squared(scaled, surface),
                                   loss_BdotN_mean_squared(field, surface), rtol=2e-12)


@pytest.mark.parametrize('close', [False, True])
def test_cross_section_area_value_and_derivative(close):
    target_radius = .45
    def objective(x):
        surface = torus(x[0], ntheta=33 if close else 32, close=close)
        return loss_cross_sectional_area_mean(surface, target_area=np.pi*target_radius**2)
    radius = jnp.array([.4])
    value, gradient = jax.value_and_grad(objective)(radius)
    relative_change = (.4/target_radius)**2-1
    np.testing.assert_allclose(value, relative_change**2, rtol=1e-12)
    np.testing.assert_allclose(gradient[0], 4*.4/target_radius**2*relative_change, rtol=1e-12)
    directional_check(objective, radius)


@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('squared', [False, True])
def test_individual_coil_penalties_and_gradients_on_circle(active, squared):
    def circle_field(radius):
        coefficients = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(radius).at[0, 1, 1].set(radius)
        return BiotSavart(Coils(Curves(coefficients, n_segments=16, stellsym=False), jnp.array([1e5])))
    radius = .7
    length_target, curvature_target = (3., 1.) if active else (10., 10.)
    def penalties(x):
        field = circle_field(x[0])
        return jnp.stack((loss_coil_length_max(field, length_target, squared=squared),
                          loss_coil_curvature_from_field(field, curvature_target)))
    expected = np.array([max(2*np.pi*radius-length_target, 0.)**(2 if squared else 1),
                          max(1/radius-curvature_target, 0.)**2*2*np.pi*radius])
    np.testing.assert_allclose(penalties(jnp.array([radius])), expected, rtol=1e-12)
    jacobian = jax.jacrev(penalties)(jnp.array([radius]))[:, 0]
    finite = (penalties(jnp.array([radius+1e-6]))-penalties(jnp.array([radius-1e-6])))/(2e-6)
    np.testing.assert_allclose(jacobian, finite, rtol=1e-7, atol=1e-9)
    if not active:
        np.testing.assert_allclose(jacobian, 0.)


def test_curvature_sampling_is_independent_of_field_quadrature():
    coefficients = jnp.zeros((1, 3, 5)).at[0, 0, 2].set(2.).at[0, 1, 1].set(.7)
    coefficients = coefficients.at[0, 2, 4].set(.2)

    def penalty(c, field_segments, curvature_segments=None):
        field = BiotSavart(Coils(Curves(c, n_segments=field_segments, nfp=2,
                                       stellsym=True), jnp.array([1.]), currents_scale=1.))
        value = loss_coil_curvature_from_field(field, 1., n_segments=curvature_segments)
        assert field.coils.n_segments == field_segments
        return value

    resolved = lambda c: penalty(c, 8, 256)
    reference = lambda c: penalty(c, 256)
    for a, b in zip(jax.value_and_grad(resolved)(coefficients),
                    jax.value_and_grad(reference)(coefficients)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    assert not np.isclose(float(resolved(coefficients)), float(penalty(coefficients, 8)))
    directional_check(resolved, coefficients)
