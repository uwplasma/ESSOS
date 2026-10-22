"""Physical limiting cases for the joint QS/QI example."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from examples.coil_optimization.optimize_coils_surface_augmented_lagrangian import (
    bounce_integrals, build_problem, initial_configuration, qi_residuals,
    qs_measure, qs_reference, quasisymmetry, toroidal_contours, toroidal_flux, trace_surface,
)


def test_traces_keep_signed_slope_and_unwrapped_theta():
    grid = jnp.stack((jnp.full((8, 8), -.37), jnp.ones((8, 8)), jnp.full((8, 8), 2.)))
    phi, theta, b, dl = jax.jit(lambda g: trace_surface(g, 2, nphi=65, nalpha=4, periods=8))(grid)
    np.testing.assert_allclose((theta[:, -1] - theta[:, 0]) / phi[-1], -.37, atol=1e-12)
    np.testing.assert_allclose(b, 2.)
    np.testing.assert_allclose(dl, 1.)

    # A one-period grid and its full-torus repetition must produce the same
    # nonconstant traces, including across the period boundary.
    angle = jnp.linspace(0., jnp.pi, 32, endpoint=False)[:, None]
    slope = jnp.broadcast_to(.3 + .1 * jnp.cos(2 * angle), (32, 8))
    grid = jnp.stack((slope, jnp.ones_like(slope), 2 + slope))
    one_period = trace_surface(grid, 2, nphi=161, nalpha=3, periods=5, grid_period=jnp.pi)
    full_torus = trace_surface(jnp.tile(grid, (1, 2, 1)), 2, nphi=161, nalpha=3, periods=5)
    for actual, expected in zip(one_period, full_torus):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(one_period[1][:, -1] - one_period[1][:, 0], .3 * 5 * jnp.pi, atol=1e-12)


def test_qi_bounce_quadrature_scaling_and_single_well_topology():
    phi = jnp.linspace(0., 2 * jnp.pi, 2049)
    b = (1 + .2 * jnp.cos(phi))[None]
    levels = jnp.array([.9, 1., 1.1])
    action = bounce_integrals(phi, b, jnp.ones_like(b), levels)
    expected = [quad(lambda p: np.sqrt(max(0., 1 - (1 + .2 * np.cos(p)) / level)),
                     0, 2*np.pi, points=[np.arccos((level - 1) / .2),
                                          2*np.pi - np.arccos((level - 1) / .2)])[0]
                for level in levels]
    np.testing.assert_allclose(action[0], expected, rtol=1e-4)
    np.testing.assert_allclose(bounce_integrals(phi, b * 5, jnp.ones_like(b), levels * 5), action)

    phi = jnp.linspace(0., 2 * jnp.pi, 257)
    b = jnp.tile(1 + .2 * jnp.cos(phi), (4, 1))
    residual, extrema, monotonic = qi_residuals(phi, b, jnp.ones_like(b))
    np.testing.assert_allclose(residual, 0., atol=1e-12)
    np.testing.assert_allclose(extrema, 0., atol=1e-12)
    assert jnp.min(monotonic) >= 0
    # Identical J on every alpha does not make a two-well field QI here.
    two_wells = jnp.tile(1 + .2 * jnp.cos(2 * phi), (4, 1))
    assert jnp.min(qi_residuals(phi, two_wells, jnp.ones_like(b))[2]) < 0
    perturbed = b.at[0].add(.03)
    assert jnp.linalg.norm(qi_residuals(phi, perturbed, jnp.ones_like(b))[0]) > .01
    gradient = jax.grad(lambda values: jnp.sum(qi_residuals(phi, values, jnp.ones_like(b))[0]**2))(perturbed)
    assert jnp.all(jnp.isfinite(gradient))


def test_axisymmetric_qs_and_biot_savart_vector_potential():
    _, surface, _ = initial_configuration('qs', smoke=True)

    class ToroidalField:
        def B(self, x):
            return jnp.array([-x[1], x[0], 0.]) / jnp.sum(x[:2]**2)

        def AbsB(self, x):
            return jnp.linalg.norm(self.B(x))

        def dAbsB_by_dX(self, x):
            return jax.grad(self.AbsB)(x)

    np.testing.assert_allclose(quasisymmetry(ToroidalField(), surface), 0., atol=1e-12)

    field, _, _ = initial_configuration('qs', smoke=True)
    x = jnp.array([10., .4, .1])
    jacobian = jax.jacfwd(field.A)(x)
    curl = jnp.array([jacobian[2, 1] - jacobian[1, 2], jacobian[0, 2] - jacobian[2, 0],
                       jacobian[1, 0] - jacobian[0, 1]])
    np.testing.assert_allclose(curl, field.B(x), rtol=1e-12, atol=1e-12)


def test_qs_qi_reference_inputs_flux_and_joint_geometry_gradients():
    for case in ('qs', 'qi'):
        field, surface, _ = initial_configuration(case, smoke=True)
        x, constraints, objective, _, names = build_problem(field, surface, case,
            ntrace=9, nalpha=2, npitch=2, iota_periods=1, smoke=True)
        p = constraints.init(x)
        value, gradient = jax.jit(jax.value_and_grad(
            lambda dofs: objective(dofs) + constraints.loss(p, dofs)[0]))(x)
        assert jnp.isfinite(value)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.linalg.norm(gradient[:field.dofs.size]) > 0
        assert jnp.linalg.norm(gradient[field.dofs.size:]) > 0
        assert ('QI single well' in names) == (case == 'qi')

    for case in ('qs', 'qi'):
        field, surface, target = initial_configuration(case, resolution=8, segments=16)
        assert surface.nfp == 2
        assert surface.area > 0 and surface.volume > 0
        assert (target > 0) == (case == 'qs')
        flux = toroidal_flux(field, surface)
        assert jnp.isfinite(flux) and jnp.abs(flux) > 1e-6

    # The full QS case must start from the source torus and circular coils,
    # while its targets come from the independent QA reference.
    field, surface, iota = initial_configuration('qs', resolution=9, segments=16)
    assert field.coils.curves.n_base_curves == 4 and field.coils.curves.order == 6
    assert surface.mpol == surface.ntor == 2 and surface.dofs.size == 26
    assert surface.nphi == surface.ntheta == 9 and surface.range_torus == 'full period'
    assert not surface.close
    np.testing.assert_allclose(surface.phi2d[:, 0], np.arange(9) * np.pi / 9)
    assert field.coils.curves.scaling_factor == surface.scaling_factor == 1.2
    np.testing.assert_allclose(field.coils.dofs_currents_raw, jnp.ones(4))
    gamma = field.coils.gamma
    centers = jnp.mean(gamma, axis=1)
    np.testing.assert_allclose(jnp.linalg.norm(centers[:, :2], axis=-1), 10., atol=1e-12)
    np.testing.assert_allclose(jnp.linalg.norm(gamma - centers[:, None], axis=-1), 5.7, atol=1e-12)
    points = surface.gamma
    np.testing.assert_allclose(jnp.sqrt((jnp.linalg.norm(points[..., :2], axis=-1) - 10.)**2
                                         + points[..., 2]**2), 4., atol=1e-12)
    reference = qs_reference(9)
    ref_field, ref_surface, _ = reference
    assert ref_surface.nphi == ref_surface.ntheta == 9 and ref_surface.range_torus == 'full period'
    # Check the public defaults without evaluating the high-resolution field.
    default_field, default_surface, _ = initial_configuration('qs', reference=reference)
    assert default_surface.nphi == default_surface.ntheta == 60
    assert default_surface.range_torus == 'full period' and default_field.coils.n_segments == 60
    full_surface = jax.tree_util.tree_map(lambda x: x, surface)
    full_surface.nphi, full_surface.range_torus = 18, 'full torus'
    for close in (False, True):
        one_period = jax.tree_util.tree_map(lambda x: x, ref_surface)
        whole_torus = jax.tree_util.tree_map(lambda x: x, ref_surface)
        whole_torus.range_torus = 'full torus'
        one_period.close = whole_torus.close = close
        one_period.nphi, whole_torus.nphi = (9, 17) if close else (9, 18)
        np.testing.assert_allclose(one_period.area, whole_torus.area, rtol=1e-12)
        np.testing.assert_allclose(one_period.volume, whole_torus.volume, rtol=1e-12)
    np.testing.assert_allclose(qs_measure(field, surface), qs_measure(field, full_surface), rtol=1e-12)
    np.testing.assert_allclose(toroidal_contours(field, surface), toroidal_contours(field, full_surface), rtol=1e-12)
    assert ref_field.coils.curves.n_base_curves == 3 and ref_field.coils.curves.order == 3
    assert ref_field.coils.n_segments == 45
    assert jnp.all((ref_field.coils.dofs_currents_raw > .9) & (ref_field.coils.dofs_currents_raw < 1.1))
    x, constraints, objective, _, names = build_problem(field, surface, 'qs', reference=reference,
        ntrace=9, nalpha=2, iota_periods=1)
    assert x.size == 186
    assert names == ['normal field', 'length', 'curvature', 'major radius', 'area',
        'volume', 'toroidal flux', 'metric aspect bound', 'iota', 'quasisymmetry',
        'toroidal contours (geometric)']
    p = constraints.init(x)
    residual = dict(zip(names, constraints.loss(p, x, _raw=True)[1]))
    np.testing.assert_allclose(residual['area'], surface.area - ref_surface.area)
    np.testing.assert_allclose(residual['volume'], surface.volume - ref_surface.volume)
    np.testing.assert_allclose(residual['toroidal flux'],
        toroidal_flux(field, surface) - toroidal_flux(ref_field, ref_surface))
    assert abs(residual['area']) > 1 and abs(residual['volume']) > 1
    np.testing.assert_allclose(residual['quasisymmetry'], qs_measure(field, surface))
    np.testing.assert_allclose(p[0]['eta_tol'], 1e-8)
    np.testing.assert_allclose(p[1]['eta_tol'], 4e-4)
    np.testing.assert_allclose(p[2]['eta_tol'], 5e-6)
    assert objective(x) == 0 and '_inequality' not in p[names.index('quasisymmetry')]
    value, gradient = jax.jit(jax.value_and_grad(lambda dofs: constraints.loss(p, dofs)[0]))(x)
    assert jnp.isfinite(value) and jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient[:field.dofs.size]) > 0
    assert jnp.linalg.norm(gradient[field.dofs.size:]) > 0
