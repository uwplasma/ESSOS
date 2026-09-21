import pytest
from pathlib import Path
from essos.coils import Coils, Curves
from essos.fields import BiotSavart
import jax
import jax.numpy as jnp
from jax import random

class MockCoils:
    def __init__(self):
        self.currents = jnp.array([1.0, 2.0, 3.0])
        self.gamma = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.gamma_dash = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.gamma_dashdash = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.dofs_curves = random.uniform(random.PRNGKey(0), (3, 3, 3))

def test_biot_savart_initialization():
    coils = MockCoils()
    biot_savart = BiotSavart(coils)
    assert biot_savart.coils == coils
    assert jnp.allclose(biot_savart.coils.currents, coils.currents)
    assert jnp.allclose(biot_savart.coils.gamma, coils.gamma)
    assert jnp.allclose(biot_savart.coils.gamma_dash, coils.gamma_dash)


def test_biot_savart_cylindrical_interface_matches_cartesian_and_differentiates():
    dofs = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(1.0).at[0, 1, 1].set(1.0)
    field = BiotSavart(Coils(Curves(dofs, n_segments=32, stellsym=False),
                              jnp.array([1.0e5])))
    R = jnp.array([[0.7, 0.8], [0.9, 1.0]])
    phi = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    Z = jnp.array([[-0.2, -0.1], [0.1, 0.2]])
    br, bp, bz = field.b_cyl(R, phi, Z)
    xyz = jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1)
    B = jax.vmap(field.B)(xyz.reshape((-1, 3))).reshape(xyz.shape)
    expected = jnp.stack((B[..., 0] * jnp.cos(phi) + B[..., 1] * jnp.sin(phi),
                          -B[..., 0] * jnp.sin(phi) + B[..., 1] * jnp.cos(phi),
                          B[..., 2]))
    assert jnp.allclose(jnp.stack((br, bp, bz)), expected)
    derivative = jax.grad(lambda radius: jnp.sum(field.b_cyl(radius, phi, Z)[0]))(R)
    assert jnp.all(jnp.isfinite(derivative))

# def test_biot_savart_B():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B = biot_savart.B(points)
#     assert jnp.allclose(B, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_B_covariant():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B_covariant = biot_savart.B_covariant(points)
#     assert jnp.allclose(B_covariant, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_B_contravariant():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B_contravariant = biot_savart.B_contravariant(points)
#     assert jnp.allclose(B_contravariant, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_AbsB():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     AbsB = biot_savart.AbsB(points)
#     assert jnp.allclose(AbsB, 4.42495529e-06)

# def test_biot_savart_dB_by_dX():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     dB_by_dX = biot_savart.dB_by_dX(points)
#     assert jnp.allclose(dB_by_dX[0], jnp.array([6.80204469e-05, 2.29490027e-05, 7.88513155e-05]))

# def test_biot_savart_dAbsB_by_dX():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     dAbsB_by_dX = biot_savart.dAbsB_by_dX(points)
#     assert jnp.allclose(dAbsB_by_dX, jnp.array([7.16688661e-05, 3.82872752e-05, 1.01490560e-04]))

if __name__ == "__main__":
    pytest.main()


def test_combined_field_sums_correctly():
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import CombinedField

    curves = CreateEquallySpacedCurves(n_curves=2, order=1, R=1.0, r=0.3,
                                       n_segments=20, nfp=2, stellsym=True)
    field = BiotSavart(Coils(curves=curves, currents=[1e5] * 2))
    points = jnp.array([0.5, 0.5, 0.5])

    assert jnp.allclose(CombinedField(field, field).B(points), 2 * field.B(points))
    assert jnp.allclose(CombinedField(field, field, field).B(points), 3 * field.B(points))


def test_combined_field_requires_at_least_one_field():
    from essos.fields import CombinedField

    with pytest.raises(ValueError):
        CombinedField()



WOUT_QA = str(Path(__file__).resolve().parents[1] / "examples" / "input_files"
              / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")


def test_vmec_fourier_modes_are_regular_on_the_axis():
    """Odd-m modes go as sqrt(s) near the axis; linear interpolation in s left
    them finite there, so |B| depended on theta on the axis itself."""
    from essos.fields import Vmec

    vmec = Vmec(WOUT_QA, ntheta=8, nphi=8)
    theta = jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False)

    def on_circle(fun, s):
        points = jnp.stack([jnp.full_like(theta, s), theta, jnp.full_like(theta, 0.3)], 1)
        return jax.vmap(fun)(points)

    d_theta = [jnp.abs(on_circle(vmec.dAbsB_by_dX, s)[:, 1]).max() / jnp.sqrt(s) for s in (1e-8, 1e-4)]
    assert d_theta[0] < 1.1 * d_theta[1]
    axis = vmec.to_xyz(jnp.array([0.0, 0.0, 0.3]))
    radius = [jnp.linalg.norm(on_circle(vmec.to_xyz, s) - axis, axis=1).max() for s in (1e-6, 1e-4)]
    assert radius[0] / radius[1] == pytest.approx(0.1, rel=1e-2)
    AbsB = on_circle(vmec.AbsB, 1e-6)
    assert jnp.abs(jnp.linalg.norm(on_circle(vmec.B, 1e-6), axis=1) / AbsB - 1).max() < 0.05
    # sqrt(g) J^phi = d_s B_theta - d_theta B_s: its m = 1 part vanishes on the axis instead of
    # tending to a constant (linear interpolation) or growing as 1/sqrt(s) (inconsistent B_s).
    current = [jnp.abs(on_circle(lambda p: vmec.sqrtg(p) * vmec.curl_B(p)[2], s)).max() for s in (1e-8, 1e-4)]
    assert current[0] < 0.5 * current[1]


def test_vmec_radial_interpolation_keeps_grid_values_and_m0_modes():
    from essos.fields import Vmec, _radial_interp

    vmec = Vmec(WOUT_QA, ntheta=8, nphi=8)
    k = 7
    assert jnp.allclose(_radial_interp(vmec.s_half_grid[k], vmec.s_half_grid, vmec.bmnc, vmec.xm_nyq, half_grid=True),
                        vmec.bmnc[k + 1])
    assert jnp.allclose(_radial_interp(vmec.s_full_grid[k], vmec.s_full_grid, vmec.rmnc, vmec.xm),
                        vmec.rmnc[k])
    s = 0.3141
    m0 = vmec.xm == 0
    expected = jax.vmap(lambda row: jnp.interp(s, vmec.s_full_grid, row))(vmec.rmnc[:, m0].T)
    assert jnp.allclose(_radial_interp(s, vmec.s_full_grid, vmec.rmnc, vmec.xm)[m0], expected)


def test_vmec_analytic_derivatives_match_automatic_differentiation():
    from essos.fields import Vmec

    vmec = Vmec(WOUT_QA, ntheta=8, nphi=8)
    for point in (jnp.array([0.3, 0.4, 0.5]), jnp.array([1e-3, 2.0, 1.0]), jnp.array([0.97, 5.0, 3.0])):
        assert jnp.allclose(vmec.dAbsB_by_dX(point), jax.grad(vmec.AbsB)(point), rtol=1e-12, atol=1e-12)
        assert jnp.allclose(vmec.grad_B_covariant(point), jax.jacfwd(vmec.B_covariant)(point), rtol=1e-12, atol=1e-12)


def test_vmec_mode_tolerance_keeps_the_field():
    from essos.fields import Vmec

    full = Vmec(WOUT_QA, ntheta=8, nphi=8)
    truncated = Vmec(WOUT_QA, ntheta=8, nphi=8, mode_tolerance=1e-3)
    assert truncated.len_xm_nyq < full.len_xm_nyq and len(truncated.xm) < len(full.xm)
    assert truncated.bmnc.shape == (full.bmnc.shape[0], truncated.len_xm_nyq)
    points = jnp.array([[0.2, 0.1, 0.3], [0.7, 2.0, 1.0]])
    for name in ("AbsB", "B_contravariant", "to_xyz"):
        a, b = jax.vmap(getattr(full, name))(points), jax.vmap(getattr(truncated, name))(points)
        assert jnp.abs(a - b).max() < 5e-3 * jnp.abs(a).max()


def test_fused_guiding_center_quantities_match_the_separate_methods():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart, MagneticField

    curves = CreateEquallySpacedCurves(n_curves=2, order=2, R=1.7, r=0.6, nfp=2, stellsym=True)
    field = BiotSavart(Coils(curves=curves, currents=[1.1e6, 0.9e6]))
    rng = np.random.default_rng(4)
    points = jnp.asarray(rng.normal(scale=0.2, size=(6, 3)) + np.array([1.7, 0.1, 0.05]))
    fused = jax.vmap(field.gc_quantities)(points)
    generic = jax.vmap(lambda p: MagneticField.gc_quantities(field, p))(points)
    for a, b in zip(fused, generic):
        np.testing.assert_allclose(np.broadcast_to(a, np.shape(b)), b, rtol=1e-9, atol=1e-12)
def _circular_loop(radius=1.3, current=2.5e5, n_segments=64):
    """One counterclockwise circular filament in the xy plane."""
    dofs = jnp.zeros((1, 3, 3))
    dofs = dofs.at[0, 0, 2].set(radius)  # x = R cos(theta)
    dofs = dofs.at[0, 1, 1].set(radius)  # y = R sin(theta)
    return Coils(
        Curves(dofs, n_segments=n_segments, nfp=1, stellsym=False),
        jnp.array([current]),
    )


def _axis_field_derivatives(radius, current, z):
    """Bz and its first three z derivatives for an ideal circular loop."""
    # mu_0 / 2 in SI units; this is independent of ESSOS's quadrature kernel.
    coefficient = 2.0 * jnp.pi * 1.0e-7 * current * radius**2
    q = radius**2 + z**2
    field = coefficient * q**(-1.5)
    first = -3.0 * coefficient * z * q**(-2.5)
    second = -3.0 * coefficient * (radius**2 - 4.0 * z**2) * q**(-3.5)
    third = 15.0 * coefficient * z * (3.0 * radius**2 - 4.0 * z**2) * q**(-4.5)
    return field, first, second, third


def _axis_cartesian_tensors(radius, current, z):
    """Full Cartesian B derivative tensors through third order on the axis.

    The entries follow from the source-free axisymmetric expansion

      Bz = F - (x**2 + y**2) F'' / 4 + ...
      (Bx, By) = -(x, y) F' / 2
                 + (x, y) (x**2 + y**2) F''' / 16 + ... .
    """
    field, first, second, third = _axis_field_derivatives(radius, current, z)
    tensors = [jnp.array([0.0, 0.0, field])]

    d1 = jnp.zeros((3, 3))
    d1 = d1.at[0, 0].set(-first / 2).at[1, 1].set(-first / 2)
    d1 = d1.at[2, 2].set(first)
    tensors.append(d1)

    d2 = jnp.zeros((3, 3, 3))
    for output, radial in ((0, 0), (1, 1)):
        d2 = d2.at[output, radial, 2].set(-second / 2)
        d2 = d2.at[output, 2, radial].set(-second / 2)
    d2 = d2.at[2, 0, 0].set(-second / 2)
    d2 = d2.at[2, 1, 1].set(-second / 2)
    d2 = d2.at[2, 2, 2].set(second)
    tensors.append(d2)

    d3 = jnp.zeros((3, 3, 3, 3))
    for output, radial, transverse in ((0, 0, 1), (1, 1, 0)):
        for indices in ((radial, 2, 2), (2, radial, 2), (2, 2, radial)):
            d3 = d3.at[(output,) + indices].set(-third / 2)
        d3 = d3.at[output, radial, radial, radial].set(3 * third / 8)
        for indices in ((radial, transverse, transverse),
                        (transverse, radial, transverse),
                        (transverse, transverse, radial)):
            d3 = d3.at[(output,) + indices].set(third / 8)
    d3 = d3.at[2, 2, 2, 2].set(third)
    for radial in (0, 1):
        for indices in ((radial, radial, 2), (radial, 2, radial),
                        (2, radial, radial)):
            d3 = d3.at[(2,) + indices].set(-third / 2)
    tensors.append(d3)
    return tensors


def test_biot_savart_circular_loop_axis_derivative_tensors():
    radius, current, z = 1.3, 2.5e5, 0.4
    field = BiotSavart(_circular_loop(radius, current))
    point = jnp.array([0.0, 0.0, z])

    actual = [field.B(point)]
    derivative = field.B
    for _ in range(3):
        derivative = jax.jacfwd(derivative)
        actual.append(derivative(point))

    expected = _axis_cartesian_tensors(radius, current, z)
    for order, (result, reference) in enumerate(zip(actual, expected)):
        assert result.shape == (3,) * (order + 1)
        atol = 2e-12 * jnp.max(jnp.abs(reference))
        assert jnp.allclose(result, reference, rtol=2e-11, atol=atol), order


def test_biot_savart_circular_loop_current_and_radius_sensitivities():
    radius, current, z = 1.3, 2.5e5, 0.4
    coils = _circular_loop(radius, current)
    point = jnp.array([0.0, 0.0, z])

    def field_from_dofs(dofs):
        return BiotSavart(coils.with_dofs(dofs)).B(point)

    # The public current DOF is normalized; this tangent represents +1 ampere.
    current_index = coils.dof_names.index("coil[0].current")
    current_direction = jnp.zeros_like(coils.dofs).at[current_index].set(
        1.0 / coils.currents_scale)
    radius_direction = jnp.zeros_like(coils.dofs)
    for name in ("coil[0].xc(1)", "coil[0].ys(1)"):
        radius_direction = radius_direction.at[coils.dof_names.index(name)].set(1.0)
    _, current_tangent = jax.jvp(field_from_dofs, (coils.dofs,), (current_direction,))
    _, radius_tangent = jax.jvp(field_from_dofs, (coils.dofs,), (radius_direction,))

    axis_field = _axis_field_derivatives(radius, current, z)[0]
    expected_current = jnp.array([0.0, 0.0, axis_field / current])
    q = radius**2 + z**2
    expected_radius_z = (2.0 * jnp.pi * 1.0e-7 * current * radius
                         * (2.0 * z**2 - radius**2) * q**(-2.5))
    expected_radius = jnp.array([0.0, 0.0, expected_radius_z])
    current_atol = 2e-12 * jnp.max(jnp.abs(expected_current))
    radius_atol = 2e-12 * jnp.max(jnp.abs(expected_radius))
    assert jnp.allclose(
        current_tangent, expected_current, rtol=2e-11, atol=current_atol)
    assert jnp.allclose(
        radius_tangent, expected_radius, rtol=2e-11, atol=radius_atol)
