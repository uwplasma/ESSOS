"""Physical limiting cases for the joint QS/QI example."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from examples.coil_optimization.optimize_coils_surface_augmented_lagrangian import (
    bounce_integrals, build_problem, initial_configuration, qi_residuals,
    quasisymmetry, toroidal_flux, trace_surface,
)


def test_traces_keep_signed_slope_and_unwrapped_theta():
    grid = jnp.stack((jnp.full((8, 8), -.37), jnp.ones((8, 8)), jnp.full((8, 8), 2.)))
    phi, theta, b, dl = jax.jit(lambda g: trace_surface(g, 2, nphi=65, nalpha=4, periods=8))(grid)
    np.testing.assert_allclose((theta[:, -1] - theta[:, 0]) / phi[-1], -.37, atol=1e-12)
    np.testing.assert_allclose(b, 2.)
    np.testing.assert_allclose(dl, 1.)


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
            ntrace=9, nalpha=2, npitch=2, iota_periods=1)
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
